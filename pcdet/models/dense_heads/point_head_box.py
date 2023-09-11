import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
from ...utils import common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class PointHeadBox(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """

    def __init__(
        self,
        num_class,
        input_channels,
        model_cfg,
        predict_boxes_when_training=False,
        **kwargs
    ):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class,
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size,
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict["point_coords"]
        gt_boxes = input_dict["gt_boxes"]
        assert gt_boxes.shape.__len__() == 3, "gt_boxes.shape=%s" % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], "points.shape=%s" % str(
            point_coords.shape
        )

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]),
            extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH,
        ).view(batch_size, -1, gt_boxes.shape[-1])
        if "gt_scores" in input_dict:
            targets_dict = self.assign_stack_pseudo_targets(
                points=point_coords,
                gt_boxes=gt_boxes,
                gt_scores=input_dict["gt_scores"],
                thresh_bound=input_dict["thresh_bound"],
                extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True,
                use_ball_constraint=False,
                ret_part_labels=False,
                ret_box_labels=True,
                self_training_mode=input_dict.get("self_training_mode", False),
            )
        else:
            targets_dict = self.assign_stack_targets(
                points=point_coords,
                gt_boxes=gt_boxes,
                extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True,
                use_ball_constraint=False,
                ret_part_labels=False,
                ret_box_labels=True,
                self_training_mode=input_dict.get("self_training_mode", False),
            )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        if (
            "point_box_labels" in self.forward_ret_dict
            and not self.forward_ret_dict.get("ignore_box_loss", False)
        ):
            point_loss_box, tb_dict_2 = self.get_box_layer_loss()
        else:
            point_loss_box, tb_dict_2 = 0, {}

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1), the maximum class score
                batch_cls_preds: (N1 + N2 + N3 + ..., num_classes)
                batch_box_preds: (N1 + N2 + N3 + ..., 7), decode pred box

                Optional[point_part_offset: (N1 + N2 + N3 + ..., 3)]

        Cache:
            ret_dict:
                point_cls_preds: (N1 + N2 + N3 + ..., num_classes)
                point_box_preds: (N1 + N2 + N3 + ..., 8), encode box pred
                point_cls_labels: (N1 + N2 + N3 + ..., )
                point_box_labels: (N1 + N2 + N3 + ..., 8), encode box label
        """
        if self.model_cfg.get("USE_POINT_FEATURES_BEFORE_FUSION", False):
            point_features = batch_dict["point_features_before_fusion"]
        else:
            point_features = batch_dict["point_features"]
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        point_box_preds = self.box_layers(
            point_features
        )  # (total_points, box_code_size)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict["point_cls_scores"] = torch.sigmoid(point_cls_preds_max)

        ret_dict = {
            "point_cls_preds": point_cls_preds,
            "point_box_preds": point_box_preds,
        }
        if self.training:
            if "gt_boxes" in batch_dict:
                targets_dict = self.assign_targets(batch_dict)
                ret_dict["point_cls_labels"] = targets_dict["point_cls_labels"]
                ret_dict["point_box_labels"] = targets_dict["point_box_labels"]
            else:
                ret_dict["point_cls_labels"] = batch_dict["pts_fake_labels"]
            # # TODO remove debug
            # batch_dict['point_cls_labels'] = ret_dict['point_cls_labels']
            # debug_utils.save_image_with_instances(data_batch=batch_dict)

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict["point_coords"][:, 1:4],
                point_cls_preds=point_cls_preds,
                point_box_preds=point_box_preds,
            )
            batch_dict["batch_cls_preds"] = point_cls_preds
            batch_dict["batch_box_preds"] = point_box_preds
            batch_dict["batch_index"] = batch_dict["point_coords"][:, 0]
            batch_dict["cls_preds_normalized"] = False
        ret_dict["ignore_box_loss"] = batch_dict.get("ignore_box_loss", False)
        self.forward_ret_dict = ret_dict

        return batch_dict

    def assign_stack_pseudo_targets(
        self,
        points,
        gt_boxes,
        gt_scores,
        thresh_bound,
        extend_gt_boxes=None,
        ret_box_labels=False,
        ret_part_labels=False,
        set_ignore_flag=True,
        use_ball_constraint=False,
        central_radius=2.0,
        self_training_mode=False,
    ):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, "points.shape=%s" % str(
            points.shape
        )
        assert (
            len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8
        ), "gt_boxes.shape=%s" % str(gt_boxes.shape)
        assert (
            extend_gt_boxes is None
            or len(extend_gt_boxes.shape) == 3
            and extend_gt_boxes.shape[2] == 8
        ), "extend_gt_boxes.shape=%s" % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, "Choose one only!"
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = (
            gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        )
        point_part_labels = (
            gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        )
        for k in range(batch_size):
            bs_mask = bs_idx == k
            pseudo_score = gt_scores[k]
            pseudo_label = gt_boxes[k, :, -1].long()
            pseudo_ignore_mask = (thresh_bound[0] <= pseudo_score) & (
                pseudo_score <= thresh_bound[1]
            )
            pseudo_label[pseudo_ignore_mask] = -1
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = (
                roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0),
                    gt_boxes[k : k + 1, :, 0:7].contiguous(),
                )
                .long()
                .squeeze(dim=0)
            )
            box_fg_flag = box_idxs_of_pts >= 0
            if set_ignore_flag:
                extend_box_idxs_of_pts = (
                    roiaware_pool3d_utils.points_in_boxes_gpu(
                        points_single.unsqueeze(dim=0),
                        extend_gt_boxes[k : k + 1, :, 0:7].contiguous(),
                    )
                    .long()
                    .squeeze(dim=0)
                )
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = (box_centers - points_single).norm(dim=1) < central_radius
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = pseudo_label[box_idxs_of_pts[fg_flag]]
            # if self_training_mode or self.num_class > 1:
            #     point_cls_labels_single[fg_flag] = gt_box_of_fg_points[:,
            #                                        -1].long()
            # else:
            #     point_cls_labels_single[fg_flag] = 1
            point_cls_labels[bs_mask] = point_cls_labels_single

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1],
                    points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long(),
                )
                if self_training_mode:
                    fg_point_box_labels[gt_box_of_fg_points[:, -1] == -1.0] = 0
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros(
                    (bs_mask.sum(), 3)
                )
                transformed_points = (
                    points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                )
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = (
                    torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                )
                point_part_labels_single[fg_flag] = (
                    transformed_points / gt_box_of_fg_points[:, 3:6]
                ) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            "point_cls_labels": point_cls_labels,
            "point_box_labels": point_box_labels,
            "point_part_labels": point_part_labels,
        }
        return targets_dict
