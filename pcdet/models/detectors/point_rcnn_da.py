from .detector3d_template import Detector3DTemplate

from pcdet.models.roi_heads import (
    PointRCNNHeadCTS,
)


import torch
from ..model_utils import model_nms_utils


class PointRCNNDA(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.contrastive_learning = isinstance(
            self.roi_head,
            (
                PointRCNNHeadCTS,
            ),
        )
        self.adversarial_learning = (
            self.model_cfg.get("DOMAIN_ADAPTER", None) is not None
        )
        self.point_head_st = self.model_cfg.ROI_HEAD.get("POINT_HEAD_ST", False)
        self.ignore_thresh = self.model_cfg.ROI_HEAD.get("IGNORE_THRESH", 0.1)
        self.fg_thresh = torch.sigmoid(
            torch.ones(1) * self.model_cfg.ROI_HEAD.FG_THRESH
        ).item()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {"loss": loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        domain_label = batch_dict["domain_label"]
        disp_dict, tb_dict = {}, {}
        if domain_label == 0:  # source domain
            loss_point, tb_dict = self.point_head.get_loss()
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            # loss_da = self.da.get_loss()
            disp_dict["loss_det"] = loss_point.item() + loss_rcnn.item()
            loss = loss_point + loss_rcnn
        else:
            loss, tb_dict = 0, {}
            if self.adversarial_learning:
                loss_al = self.da.get_loss()
                disp_dict["loss_al"] = loss_al.item()
                loss += loss_al

            if self.contrastive_learning:
                # stage 2 loss
                cl_loss, cl_tb_dict = self.roi_head.get_aug_box_loss()
                loss = loss + cl_loss
                disp_dict["loss_cl"] = cl_loss.item()
                tb_dict.update(cl_tb_dict)
                if self.point_head_st:
                    # stage 1
                    batch_dict.pop("gt_boxes")
                    pred_dicts = self.post_processing_for_st(batch_dict)
                    max_gt = max([len(x["pred_boxes"]) for x in pred_dicts])
                    batch_gt_boxes3d = torch.zeros(
                        (
                            len(pred_dicts),
                            max_gt,
                            pred_dicts[0]["pred_boxes"].shape[-1] + 1,
                        ),
                        dtype=pred_dicts[0]["pred_boxes"].dtype,
                        device=pred_dicts[0]["pred_boxes"].device,
                    )
                    gt_scores = torch.zeros_like(batch_gt_boxes3d[..., 0])
                    for k, pred_dict in enumerate(pred_dicts):
                        box = torch.cat(
                            [
                                pred_dict["pred_boxes"],
                                pred_dict["pred_labels"].view(-1, 1),
                            ],
                            dim=1,
                        )
                        batch_gt_boxes3d[k, : box.__len__(), :] = box
                        gt_scores[k, : box.__len__()] = pred_dict["pred_scores"]
                    batch_dict["gt_boxes"] = batch_gt_boxes3d
                    batch_dict["gt_scores"] = gt_scores
                    batch_dict["thresh_bound"] = (self.ignore_thresh, self.fg_thresh)
                    targets_dict = self.point_head.assign_targets(batch_dict)
                    self.point_head.forward_ret_dict.update(
                        {
                            "point_cls_labels": targets_dict["point_cls_labels"],
                            "point_box_labels": targets_dict["point_box_labels"],
                        }
                    )
                    loss_point, point_st_tb_dict = self.point_head.get_loss()
                    loss = loss + loss_point
                    disp_dict["loss_phst"] = loss_point.item()
                    tb_dict.update(
                        {k + "_phst": v for k, v in point_st_tb_dict.items()}
                    )

        return loss, tb_dict, disp_dict

    def post_processing_for_st(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict["batch_size"]
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get("batch_index", None) is not None:
                assert batch_dict["batch_box_preds"].shape.__len__() == 2
                batch_mask = batch_dict["batch_index"] == index
            else:
                assert batch_dict["batch_box_preds"].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict["batch_box_preds"][batch_mask]
            # uncertainty
            uncer_flag = False
            if batch_dict.get("batch_au_preds", None) is not None:
                uncer_flag = True
                au_preds = batch_dict["batch_au_preds"][batch_mask]

            if not isinstance(batch_dict["batch_cls_preds"], list):
                cls_preds = batch_dict["batch_cls_preds"][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict["cls_preds_normalized"]:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict["batch_cls_preds"]]
                src_cls_preds = cls_preds
                if not batch_dict["cls_preds_normalized"]:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if uncer_flag:
                    raise NotImplementedError
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [
                        torch.arange(1, self.num_class, device=cls_preds[0].device)
                    ]
                else:
                    multihead_label_mapping = batch_dict["multihead_label_mapping"]

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(
                    cls_preds, multihead_label_mapping
                ):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[
                        cur_start_idx : cur_start_idx + cur_cls_preds.shape[0]
                    ]
                    (
                        cur_pred_scores,
                        cur_pred_labels,
                        cur_pred_boxes,
                    ) = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds,
                        box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=self.ignore_thresh,
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get("has_class_labels", False):
                    label_key = (
                        "roi_labels"
                        if "roi_labels" in batch_dict
                        else "batch_pred_labels"
                    )
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds,
                    box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=self.ignore_thresh,
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
                if uncer_flag:
                    final_uncers = au_preds[selected]

            record_dict = {
                "pred_boxes": final_boxes,
                "pred_scores": final_scores,
                "pred_labels": final_labels,
                # "pred_uncer":
            }
            if uncer_flag:
                record_dict.update({"pred_uncer": final_uncers})
            pred_dicts.append(record_dict)

        return pred_dicts

    # modify to get assign uncertainity to results
    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict["batch_size"]
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get("batch_index", None) is not None:
                assert batch_dict["batch_box_preds"].shape.__len__() == 2
                batch_mask = batch_dict["batch_index"] == index
            else:
                assert batch_dict["batch_box_preds"].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict["batch_box_preds"][batch_mask]
            rois_3D = batch_dict["rois"][batch_mask]
            src_box_preds = box_preds
            use_au = False
            if batch_dict.get("batch_au_preds", None) is not None:
                use_au = True
                au_preds = batch_dict["batch_au_preds"][batch_mask]

            if not isinstance(batch_dict["batch_cls_preds"], list):
                cls_preds = batch_dict["batch_cls_preds"][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict["cls_preds_normalized"]:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict["batch_cls_preds"]]
                src_cls_preds = cls_preds
                if not batch_dict["cls_preds_normalized"]:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError("au not implemented for multi-class-nms")
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [
                        torch.arange(1, self.num_class, device=cls_preds[0].device)
                    ]
                else:
                    multihead_label_mapping = batch_dict["multihead_label_mapping"]

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(
                    cls_preds, multihead_label_mapping
                ):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[
                        cur_start_idx : cur_start_idx + cur_cls_preds.shape[0]
                    ]

                    (
                        cur_pred_scores,
                        cur_pred_labels,
                        cur_pred_boxes,
                    ) = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds,
                        box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH,
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get("has_class_labels", False):
                    label_key = (
                        "roi_labels"
                        if "roi_labels" in batch_dict
                        else "batch_pred_labels"
                    )
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds,
                    box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH,
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
                rois_3D = rois_3D[selected]
                if use_au:
                    final_aus = au_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if "rois" not in batch_dict else src_box_preds,
                recall_dict=recall_dict,
                batch_index=index,
                data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST,
            )

            record_dict = {
                "pred_boxes": final_boxes,
                "pred_scores": final_scores,
                "pred_labels": final_labels,
                "rois_3D": rois_3D,
            }
            if use_au:
                record_dict.update({"pred_uncer": final_aus})
                au_dict = self.generate_au_record(
                    final_boxes, final_aus, batch_dict["gt_boxes"][index], 0.5, 0.05
                )
                recall_dict.update(au_dict)
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict
