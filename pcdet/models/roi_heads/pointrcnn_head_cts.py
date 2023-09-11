import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import box_coder_utils, common_utils, loss_utils
from .roi_head_template import RoIHeadTemplate


class PredictBranch(nn.Module):
    def __init__(self, input_channels, use_bn, model_cfg, num_class, code_size):
        super(PredictBranch, self).__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth
        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER
        shared_mlps = []
        for k in range(len(xyz_mlps) - 1):
            shared_mlps.append(
                nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn)
            )
            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())
        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]
        self.use_bbfeat = self.model_cfg.USE_BBFEAT
        c_in = c_out * 2 if self.use_bbfeat else c_out
        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = (
                self.model_cfg.SA_CONFIG.NPOINTS[k]
                if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1
                else None
            )
            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn,
                )
            )
            channel_in = mlps[-1]

        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.num_class,
            fc_list=self.model_cfg.CLS_FC,
        )
        self.abs_reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC,
        )
        uncer_type = self.model_cfg.LOSS_CONFIG.get("UNCERTAINTY_TYPE", "box")
        if uncer_type == "box":
            uncer_code_size = code_size
        elif uncer_type == "corner":
            uncer_code_size = 8
        elif uncer_type == "corner_xyz":
            uncer_code_size = 24
        else:
            raise RuntimeError

        if self.model_cfg.get("UNCERTAIN_FC", None) is not None:
            uncertain_fc = self.model_cfg.UNCERTAIN_FC
        else:
            uncertain_fc = self.model_cfg.REG_FC

        self.au_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=uncer_code_size * self.num_class,
            fc_list=uncertain_fc,
        )
        self.init_weights(weight_init="xavier")

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend(
                [
                    nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(fc_list[k]),
                    nn.ReLU(),
                ]
            )
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(
            nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True)
        )
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def init_weights(self, weight_init="xavier"):
        if weight_init == "kaiming":
            init_func = nn.init.kaiming_normal_
        elif weight_init == "xavier":
            init_func = nn.init.xavier_normal_
        elif weight_init == "normal":
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == "normal":
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.abs_reg_layers[-1].weight, mean=0, std=0.001)

    def forward(self, pooled_features):
        xyz_input = (
            pooled_features[..., 0 : self.num_prefix_channels]
            .transpose(1, 2)
            .unsqueeze(dim=3)
            .contiguous()
        )
        xyz_features = self.xyz_up_layer(xyz_input)
        if self.use_bbfeat:
            point_features = (
                pooled_features[..., self.num_prefix_channels :]
                .transpose(1, 2)
                .unsqueeze(dim=3)
            )
            merged_features = torch.cat((xyz_features, point_features), dim=1)
        else:
            merged_features = xyz_features
        merged_features = self.merge_down_layer(merged_features)

        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [
            merged_features.squeeze(dim=3).contiguous()
        ]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, _ = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        shared_features = l_features[-1]  # (total_rois, num_features, 1)
        rcnn_cls = (
            self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
        )  # (B, 1 or 2)
        rcnn_abs_reg = (
            self.abs_reg_layers(shared_features)
            .transpose(1, 2)
            .contiguous()
            .squeeze(dim=1)
        )  # (B, C)

        rcnn_au = (
            self.au_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
        )  # (B, C)
        rcnn_au = F.elu(rcnn_au) + 1.0 + 1e-7

        return rcnn_cls, rcnn_abs_reg, rcnn_au, l_features


class PointRCNNHeadCTS(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        use_bn = self.model_cfg.USE_BN
        self.aug_source = model_cfg.AUG_SOURCE
        self.abs_decode = model_cfg.ABS_DECODE
        self.roi_aug_cfg = model_cfg.ROI_AUGMENT
        self.pred_box = model_cfg.get("POINT_HEAD_ST", False)
        self.student_input_aug = model_cfg.STUDENT_INPUT_AUG
        self.fg_thresh = model_cfg.FG_THRESH
        self.mt_beta = model_cfg.BETA
        self.abs_box_coder = getattr(
            box_coder_utils, self.model_cfg.ABS_BOX_CODER.BOX_CODER
        )(mean_size=self.model_cfg.ABS_BOX_CODER.MEAN_SIZE)
        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH,
        )

        self.predict_branch = PredictBranch(
            input_channels, use_bn, model_cfg, num_class, self.box_coder.code_size
        )
        self.uncer_type = self.model_cfg.LOSS_CONFIG.get("UNCERTAINTY_TYPE", "box")

    def init_mt_branch(self):
        mt_predict_branch = copy.deepcopy(self.predict_branch)
        for params in mt_predict_branch.parameters():
            params.requires_grad_(False)
        mt_predict_branch.eval()
        return mt_predict_branch

    @torch.no_grad()
    def update_mt_branch(self, mt_predict_bracn):
        student_state_dict = self.predict_branch.state_dict()
        for k, v in mt_predict_bracn.state_dict().items():
            buf = student_state_dict[k]
            if buf.dtype.is_floating_point:
                v.data[:] = v.data * self.mt_beta + buf.data * (1.0 - self.mt_beta)

    def roi_augmentation(self, pooled_features, batch_dict):
        """
            pooled_features: # (total_rois, num_sampled_points, x, y, z, score, depth + C)
            pooled_features: # (total_rois, num_sampled_points, x, y, z, score, depth + C)
            batch_dict:

        Returns:

        pooled_features: # (total_rois, num_sampled_points, x, y, z, score, depth + C)
            batch_dict:

        Returns:

        """

        aug_features = pooled_features.clone().detach()  # copy and detach
        non_zero = (torch.any(aug_features > 0, dim=-1, keepdim=True)).float()
        roi_num = aug_features.shape[0]
        B, N = batch_dict["rois"].shape[:2]
        device = aug_features.device
        dtype = aug_features.dtype
        # random rotation
        alpha = (
            torch.rand(roi_num, device=device, dtype=dtype)
            * (self.roi_aug_cfg.ROT[1] - self.roi_aug_cfg.ROT[0])
            + self.roi_aug_cfg.ROT[0]
        )
        alpha = torch.zeros(roi_num, device=device, dtype=dtype)
        aug_features[:, :, 0:3] = common_utils.rotate_points_along_z(
            aug_features[:, :, 0:3], alpha
        )

        # random scale
        scale = (
            torch.rand(roi_num, 3, device=device, dtype=dtype)
            * (self.roi_aug_cfg.WHL_RATIO[1] - self.roi_aug_cfg.WHL_RATIO[0])
            + self.roi_aug_cfg.WHL_RATIO[0]
        )
        scale = scale.unsqueeze(1)
        aug_features[:, :, 0:3] *= scale

        # random translation
        delta_xyz = (
            torch.zeros(roi_num, 3, device=device, dtype=dtype)
            * (self.roi_aug_cfg.TRANSITION[1] - self.roi_aug_cfg.TRANSITION[0])
            + self.roi_aug_cfg.TRANSITION[0]
        )
        delta_xyz = torch.zeros(roi_num, 3, device=device, dtype=dtype)
        delta_xyz = delta_xyz.unsqueeze(1)
        aug_features[:, :, 0:3] += delta_xyz

        # random flip
        flip = torch.rand(1) < 0.0
        if flip:
            aug_features[:, :, 1] = -aug_features[:, :, 1]

        # correct depth
        rois = batch_dict["rois"]
        xyz = common_utils.rotate_points_along_z(
            aug_features[:, :, 0:3], rois[:, :, 6].view(roi_num)
        ) + rois[:, :, :3].view(roi_num, 1, 3)
        aug_features[:, :, 4] = (
            torch.norm(xyz, dim=-1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER
            - 0.5
        )
        aug_features *= non_zero
        self.roi_aug_dict = {
            "alpha": alpha.view(B, N),
            "scale": scale.view(B, N, 3),
            "delta_xyz": delta_xyz.view(B, N, 3),
            "flip": flip,
        }

        return aug_features

    def apply_augment(self, boxes, clip_head, alpha, scale, delta_xyz, flip):
        """
        Args:
            boxes: B x N x 7
            alpha: Bx N
            scale: B x N x 3
            delta_xyz: B x N x 3
            flip: bool
            clip_head: bool

        Returns:

        """
        boxes = boxes.clone()
        roi_num = boxes.shape[0] * boxes.shape[1]
        B, N = boxes.shape[0], boxes.shape[1]
        # random rotation
        boxes[:, :, 6] += alpha
        boxes[:, :, :3] = common_utils.rotate_points_along_z(
            boxes[:, :, :3].view(roi_num, 1, 3), alpha.view(-1)
        ).view(B, N, 3)

        # random scaling
        boxes[:, :, :3] *= scale
        boxes[:, :, 3:6] *= scale

        # random translation
        boxes[:, :, 0:3] += delta_xyz

        # random flip
        if flip:
            boxes[:, :, 6] = -boxes[:, :, 6]
            boxes[:, :, 1] = -boxes[:, :, 1]

        # TODO whether need to flip orientation if rois have opposite orientation ?
        if clip_head:
            heading_label = boxes[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (
                heading_label < np.pi * 1.5
            )
            heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (
                2 * np.pi
            )  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

            boxes[:, :, 6] = heading_label
        return boxes

    def backout_augment(self, aug_boxes, clip_head, alpha, scale, delta_xyz, flip):
        """

        Args:
            aug_boxes: B x N x 7
            opposite_flag: optional[torch.Tensor], B x N
            alpha: B x N
            scale: B x N x 3
            delta_xyz: B x N x 3
            flip: bool

        Returns:

        """
        boxes = aug_boxes.clone()

        # decode flip
        if flip:
            boxes[:, :, 1] = -boxes[:, :, 1]
            boxes[:, :, 6] = -boxes[:, :, 6]

        # decode translation
        boxes[:, :, :3] -= delta_xyz

        # decode scaling
        boxes[:, :, :3] /= scale
        boxes[:, :, 3:6] /= scale

        # decode rotation
        roi_num = boxes.shape[0] * boxes.shape[1]
        B, N = boxes.shape[0], boxes.shape[1]
        boxes[:, :, 6] -= alpha
        boxes[:, :, :3] = common_utils.rotate_points_along_z(
            boxes[:, :, :3].view(roi_num, 1, 3), -alpha.view(-1)
        ).view(B, N, 3)

        if clip_head:
            heading_label = boxes[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (
                heading_label < np.pi * 1.5
            )
            heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (
                2 * np.pi
            )  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

            boxes[:, :, 6] = heading_label
        return boxes

    def roipool3d_gpu(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
            pooled_features (B, num_rois, 512, C), num_rois = 128 if train else 512
            # pooled_features.shape[2] is points number sampled for each iou.

        """
        batch_size = batch_dict["batch_size"]
        batch_idx = batch_dict["point_coords"][:, 0]
        point_coords = batch_dict["point_coords"][:, 1:4]
        point_features = batch_dict["point_features"]
        rois = batch_dict["rois"]  # (B, num_rois, 7 + C)
        batch_cnt = point_coords.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert batch_cnt.min() == batch_cnt.max()

        point_scores = batch_dict["point_cls_scores"].clone().detach()
        point_depths = (
            point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER
            - 0.5
        )
        point_features_list = [
            point_scores[:, None],
            point_depths[:, None],
            point_features,
        ]
        point_features_all = torch.cat(point_features_list, dim=1)
        batch_points = point_coords.view(batch_size, -1, 3)
        batch_point_features = point_features_all.view(
            batch_size, -1, point_features_all.shape[-1]
        )

        with torch.no_grad():
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )  # pooled_features: (B, num_rois, num_sampled_points, 3 + C), pooled_empty_flag: (B, num_rois)

            # canonical transformation
            roi_center = rois[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)

            pooled_features = pooled_features.view(
                -1, pooled_features.shape[-2], pooled_features.shape[-1]
            )
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                ------------------------------------> used in proposal layer

                gt_boxes: (B, N, 7 + C + 1) ->
                ------------------------------------> use in proposal target layer

                point_features: (num_points, C)
                ------------------------------------> used in roi_pool3d
            nms_config:

            batch_size:
        rois: (B, num_rois, 7 + C)
        point_coords: (num_points, 4)  [bs_idx, x, y, z]
        point_features: (num_points, C)
        point_cls_scores: (N1 + N2 + N3 + ..., 1)
        point_part_offset: (N1 + N2 + N3 + ..., 3)


        Returns:

        """
        # nms on rois, choose proposal from first stage, in lidar
        targets_dict = self.proposal_layer(
            batch_dict,
            nms_config=self.model_cfg.NMS_CONFIG["TRAIN" if self.training else "TEST"],
        )

        if self.training:
            # get gt_of_rois in ccs
            # gt_of_rois_src in lidar
            targets_dict = self.assign_targets(batch_dict)
            batch_dict["rois"] = targets_dict["rois"]
            batch_dict["roi_labels"] = targets_dict["roi_labels"]
            # batch_dict['gt_of_rois'] = targets_dict['gt_of_rois']
            # batch_dict['gt_of_rois_src'] = targets_dict['gt_of_rois_src']

        # NOTE get roi features  roi_nums * point_num * c
        pooled_features = self.roipool3d_gpu(
            batch_dict
        )  # (total_rois, num_sampled_points, 3 + C)

        domain_label = batch_dict.get("domain_label", 0)
        predict_branch = self.predict_branch
        if self.training and domain_label == 1 and self.student_input_aug:
            # use teacher branch
            predict_branch = batch_dict["mt_predict_branch"]
        rcnn_cls, rcnn_abs_reg, rcnn_au, l_features = predict_branch(pooled_features)

        if self.training and (self.aug_source or domain_label == 1):
            # ---------- augmentation ---------
            # NOTE

            # during training process, target data domain with augmentation
            aug_predict_branch = self.predict_branch
            if domain_label == 1 and not self.student_input_aug:
                # NOTE if target domain + student_aug
                # use teacher branch
                aug_predict_branch = batch_dict["mt_predict_branch"]
            aug_features = self.roi_augmentation(
                pooled_features=pooled_features, batch_dict=batch_dict
            )
            aug_rcnn_cls, aug_rcnn_abs_reg, aug_rcnn_au, _ = aug_predict_branch(
                aug_features
            )
            self.aug_forward_ret_dict = {
                "rcnn_cls": aug_rcnn_cls,
                "rcnn_abs_reg": aug_rcnn_abs_reg,
                "rcnn_au": aug_rcnn_au,
            }
            self.aug_forward_ret_dict.update(targets_dict)

        if not self.training or self.pred_box:
            roi_boxes3d = batch_dict["rois"]
            pred_classes = torch.argmax(rcnn_cls, dim=1)
            anchors = roi_boxes3d.clone()
            whl = self.abs_box_coder.mean_size[pred_classes]
            # anchor head
            anchors[:, :, 3:6] = whl.view(roi_boxes3d.shape[0], roi_boxes3d.shape[1], 3)

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict["batch_size"],
                rois=anchors,
                cls_preds=rcnn_cls.detach(),
                box_preds=rcnn_abs_reg.detach(),
            )
            if self.uncer_type == "box":
                rcnn_au = box_coder_utils.box_uncertainty_decode(
                    rcnn_au, anchors.view(-1, rcnn_au.shape[-1]), rcnn_abs_reg
                )

            batch_au_preds = rcnn_au.view([*batch_box_preds.shape[:-1], -1])
            batch_dict["batch_cls_preds"] = batch_cls_preds
            batch_dict["batch_box_preds"] = batch_box_preds
            batch_dict["batch_au_preds"] = batch_au_preds
            batch_dict["cls_preds_normalized"] = False

        if self.training:
            targets_dict["rcnn_cls"] = rcnn_cls
            targets_dict["rcnn_abs_reg"] = rcnn_abs_reg
            targets_dict["rcnn_au"] = rcnn_au
            targets_dict["domain_label"] = batch_dict.get("domain_label", 0)

            batch_dict["rcnn_cls_score"] = rcnn_cls
            batch_dict["roi_features_all"] = l_features

            self.forward_ret_dict = targets_dict
        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_abs_loss_reg, reg_tb_dict = self.get_abs_box_reg_layer_loss(
            self.forward_ret_dict
        )
        rcnn_loss += rcnn_abs_loss_reg
        tb_dict.update(reg_tb_dict)

        aug_loss, aug_tb_dict = self.get_aug_box_loss()
        rcnn_loss += aug_loss
        tb_dict.update(aug_tb_dict)
        tb_dict["rcnn_loss"] = rcnn_loss.item()

        return rcnn_loss, tb_dict

    def get_soft_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict["rcnn_cls"]
        rcnn_cls_labels = forward_ret_dict["rcnn_cls_labels"].view(-1)
        if loss_cfgs.CLS_LOSS == "BinaryCrossEntropy":
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(
                torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction="none"
            )
            cls_valid_mask = rcnn_cls_labels
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(
                cls_valid_mask.sum(), min=1.0
            )
        elif loss_cfgs.CLS_LOSS == "CrossEntropy":
            batch_loss_cls = F.cross_entropy(
                rcnn_cls, rcnn_cls_labels, reduction="none", ignore_index=-1
            )
            cls_valid_mask = rcnn_cls_labels
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(
                cls_valid_mask.sum(), min=1.0
            )
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS["rcnn_cls_weight"]
        tb_dict = {"rcnn_loss_cls": rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_aug_box_loss(self):
        # on source domain
        if self.forward_ret_dict["domain_label"] == 0 and not self.aug_source:
            return 0, {}
        aug_forward_ret_dict = self.aug_forward_ret_dict
        if self.forward_ret_dict["domain_label"] == 0:
            if not self.aug_source:
                return 0, {}

            aug_forward_ret_dict.update(
                {
                    "rcnn_cls_labels": self.forward_ret_dict["rcnn_cls_labels"],
                    "reg_valid_mask": self.forward_ret_dict["reg_valid_mask"],
                    "gt_of_rois": self.apply_augment(
                        self.forward_ret_dict["gt_of_rois"], True, **self.roi_aug_dict
                    ),
                }
            )
            rcnn_loss, tb_dict = 0, {}
            # classification loss
            rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(
                aug_forward_ret_dict
            )
            rcnn_loss += rcnn_loss_cls
            tb_dict.update({"aug_" + k: v for k, v in cls_tb_dict.items()})

            # regression loss
            rcnn_loss_reg, reg_tb_dict = self.get_abs_box_reg_layer_loss(
                aug_forward_ret_dict, tag="aug_abs_reg_"
            )
            rcnn_loss += rcnn_loss_reg
            tb_dict.update(reg_tb_dict)
            return rcnn_loss, tb_dict

        rois_lidar = self.forward_ret_dict["rois"]
        rcnn_cls = self.forward_ret_dict["rcnn_cls"]
        rcnn_uncer = self.forward_ret_dict["rcnn_au"]
        rcnn_abs_reg = self.forward_ret_dict["rcnn_abs_reg"]

        aug_rcnn_cls = self.aug_forward_ret_dict["rcnn_cls"]
        aug_rcnn_uncer = self.aug_forward_ret_dict["rcnn_au"]
        aug_rcnn_abs_reg = self.aug_forward_ret_dict["rcnn_abs_reg"]

        rcnn_loss, tb_dict = 0, {}

        if self.student_input_aug:
            rcnn_cls_labels = rcnn_cls.clone().detach()
            reg_valid_mask = (rcnn_cls_labels > self.fg_thresh).float()
            pred_score, pred_classes = torch.max(rcnn_cls, dim=1)
            anchors = torch.zeros_like(self.forward_ret_dict["rois"])
            B, N = anchors.shape[0], anchors.shape[1]
            anchors[:, :, 3:6] = self.abs_box_coder.mean_size[pred_classes].view(
                B, N, 3
            )
            decode_box = self.abs_box_coder.decode_torch(
                rcnn_abs_reg.view(B, N, -1).clone().detach(), anchors
            )

            aug_forward_ret_dict = {
                "rois": rois_lidar,
                "rcnn_cls": aug_rcnn_cls,
                "rcnn_abs_reg": aug_rcnn_abs_reg,
                "rcnn_au": aug_rcnn_uncer,
                "rcnn_cls_labels": torch.sigmoid(rcnn_cls_labels),
                "reg_valid_mask": reg_valid_mask,
                "gt_of_rois": self.apply_augment(decode_box, True, **self.roi_aug_dict),
                "gt_of_rois_uncer": rcnn_uncer,
            }

            rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(
                aug_forward_ret_dict
            )
            rcnn_loss += rcnn_loss_cls
            tb_dict.update({"cl_aug_" + k: v for k, v in cls_tb_dict.items()})

            rcnn_loss_reg, reg_tb_dict = self.get_abs_box_reg_layer_loss(
                aug_forward_ret_dict, tag="cl_aug_abs_reg_"
            )
            rcnn_loss += rcnn_loss_reg
            tb_dict.update(reg_tb_dict)
        else:
            aug_rcnn_cls_labels = aug_rcnn_cls.clone().detach()
            reg_valid_mask = (aug_rcnn_cls_labels > self.fg_thresh).float()
            aug_pred_score, aug_pred_classes = torch.max(aug_rcnn_cls, dim=1)
            aug_anchors = torch.zeros_like(self.forward_ret_dict["rois"])
            B, N = aug_anchors.shape[0], aug_anchors.shape[1]
            aug_anchors[:, :, 3:6] = self.abs_box_coder.mean_size[
                aug_pred_classes
            ].view(B, N, 3)

            decode_aug_box = self.abs_box_coder.decode_torch(
                aug_rcnn_abs_reg.view(B, N, -1).clone().detach(), aug_anchors
            )
            forward_ret_dict = {
                "rois": rois_lidar,
                "rcnn_cls": rcnn_cls,
                "rcnn_abs_reg": rcnn_abs_reg,
                "rcnn_au": rcnn_uncer,
                "rcnn_cls_labels": torch.sigmoid(aug_rcnn_cls_labels),
                "reg_valid_mask": reg_valid_mask,
                "gt_of_rois": self.backout_augment(
                    decode_aug_box, True, **self.roi_aug_dict
                ),
                "gt_of_rois_uncer": aug_rcnn_uncer,
            }
            rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(forward_ret_dict)
            rcnn_loss += rcnn_loss_cls
            tb_dict.update({"cl_" + k: v for k, v in cls_tb_dict.items()})

            rcnn_loss_reg, reg_tb_dict = self.get_abs_box_reg_layer_loss(
                forward_ret_dict, tag="cl_abs_reg_"
            )
            rcnn_loss += rcnn_loss_reg
            tb_dict.update(reg_tb_dict)
        return rcnn_loss, tb_dict

    def build_losses(self, losses_cfg):
        if losses_cfg.REG_LOSS == "smooth-l1":
            self.add_module(
                "reg_loss_func",
                loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS["code_weights"]
                ),
            )
        else:
            raise NotImplementedError

        if losses_cfg.get("UNCERTAINTY_TYPE", "box") == "box":
            nll_func = loss_utils.AULoss(
                code_weights=losses_cfg.LOSS_WEIGHTS["code_weights"],
                distribution=losses_cfg.get("AU_LOSS_DIST", "gaussion"),
            )
            self.add_module("nll_loss_func", nll_func)

    def get_abs_box_reg_layer_loss(self, forward_ret_dict, tag="abs_reg_"):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.abs_box_coder.code_size
        rcnn_cls = forward_ret_dict["rcnn_cls"].detach()
        reg_valid_mask = forward_ret_dict["reg_valid_mask"].view(-1)
        gt_boxes3d_ct = forward_ret_dict["gt_of_rois"][..., 0:code_size]
        rcnn_reg = forward_ret_dict["rcnn_abs_reg"]  # (rcnn_batch_size, C)
        samples_in_all_batches = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = reg_valid_mask > 0
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}
        tb_dict["rcnn_fg_sum"] = fg_sum

        rcnn_loss_reg = 0.0
        if not fg_sum > 0:
            return 0.0, {}

        pred_classes = torch.argmax(rcnn_cls, dim=1)
        anchors = torch.zeros_like(rcnn_reg)
        anchors[:, 3:6] = self.abs_box_coder.mean_size[pred_classes]
        reg_targets = self.box_coder.encode_torch(
            gt_boxes3d_ct.view(samples_in_all_batches, code_size), anchors
        )

        uncer_type = loss_cfgs.get("UNCERTAINTY_TYPE", "box")

        fg_label_weight = fg_mask[fg_mask].float().unsqueeze(-1)
        if (
            loss_cfgs.get("REG_UNCER_WEIGHTED", False)
            and tag.startswith("cl_")  # only enable idn self training
            and forward_ret_dict.get("gt_of_rois_uncer", None) is not None
        ):
            gt_of_rois_uncer = forward_ret_dict["gt_of_rois_uncer"].detach()
            reduce_method = loss_cfgs.get("INSTANCE_UNCER_REDUCE", "mean")
            if reduce_method == "mean":
                gt_of_rois_uncer = gt_of_rois_uncer.mean(dim=-1, keepdim=True)
            elif reduce_method == "max":
                gt_of_rois_uncer = gt_of_rois_uncer.amax(dim=-1, keepdim=True)
            elif reduce_method == "none":
                pass
            else:
                raise NotImplementedError

            weight_sum = torch.sum(1 / gt_of_rois_uncer[fg_mask])
            uncer_weight = (1 / gt_of_rois_uncer[fg_mask]) / weight_sum * fg_sum
            fg_label_weight = fg_label_weight * uncer_weight

        if loss_cfgs.REG_LOSS == "smooth-l1":
            smoothl1_loss = self.reg_loss_func(
                rcnn_reg.view(samples_in_all_batches, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            smoothl1_loss = smoothl1_loss.view(samples_in_all_batches, -1)[fg_mask]
            smoothl1_loss = (smoothl1_loss * fg_label_weight).mean()
            loss_weight = (
                loss_cfgs.LOSS_WEIGHTS["rcnn_reg_weight"]
                if "rcnn_reg_weight" in loss_cfgs.LOSS_WEIGHTS.keys()
                else 1.0
            )
            smoothl1_loss = smoothl1_loss * loss_weight
            tb_dict[tag + "rcnn_reg_loss"] = smoothl1_loss.item()
            rcnn_loss_reg += smoothl1_loss
            if uncer_type == "box":
                rcnn_au = forward_ret_dict["rcnn_au"]
                box_nll_loss = self.nll_loss_func(
                    torch.cat(
                        [
                            rcnn_reg.view(samples_in_all_batches, -1).unsqueeze(dim=0),
                            rcnn_au.view(samples_in_all_batches, -1).unsqueeze(dim=0),
                        ],
                        dim=-1,
                    ),
                    reg_targets.unsqueeze(dim=0),
                )
                box_nll_loss = box_nll_loss.view(samples_in_all_batches, -1)[fg_mask]
                box_nll_loss = (box_nll_loss * fg_label_weight).mean()
                loss_weight = (
                    loss_cfgs.LOSS_WEIGHTS["rcnn_nll_weight"]
                    if "rcnn_nll_weight" in loss_cfgs.LOSS_WEIGHTS.keys()
                    else 1.0
                )
                box_nll_loss = box_nll_loss * loss_weight
                tb_dict[tag + "rcnn_nll"] = box_nll_loss.item()

                rcnn_loss_reg += box_nll_loss

        else:
            raise NotImplementedError

        if uncer_type == "corner" or loss_cfgs.CORNER_LOSS_REGULARIZATION:
            gt_of_rois = forward_ret_dict["gt_of_rois"][..., 0:code_size].view(
                -1, code_size
            )
            roi_boxes3d = forward_ret_dict["rois"]

            fg_rcnn_reg = rcnn_reg.view(samples_in_all_batches, -1)[fg_mask]
            fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]
            fg_rcnn_cls = torch.argmax(rcnn_cls[fg_mask], dim=1)
            fg_gt_of_rois = gt_of_rois[fg_mask].detach()

            roi_ry = fg_roi_boxes3d[:, 6].view(-1)
            roi_xyz = fg_roi_boxes3d[:, 0:3].view(-1, 3)

            fg_anchors = fg_roi_boxes3d
            fg_anchors[:, 0:3] = 0
            fg_anchors[:, 3:6] = self.abs_box_coder.mean_size[fg_rcnn_cls]

            fg_rcnn_boxes3d_ccs = self.box_coder.decode_torch(
                fg_rcnn_reg.unsqueeze(0),
                fg_anchors.unsqueeze(0),
            ).view(-1, code_size)

            fg_rcnn_boxes3d_src = common_utils.rotate_points_along_z(
                fg_rcnn_boxes3d_ccs.unsqueeze(dim=1), roi_ry
            ).squeeze(dim=1)
            fg_rcnn_boxes3d_src[:, 0:3] += roi_xyz

            fg_gt_of_rois_src = common_utils.rotate_points_along_z(
                fg_gt_of_rois.unsqueeze(dim=1), roi_ry
            ).squeeze(dim=1)
            fg_gt_of_rois_src[:, 0:3] += roi_xyz
            fg_gt_of_rois_src[:, 6] += fg_anchors.view(-1, code_size)[:, 6]

            loss_corner = 0
            if uncer_type == "corner":
                # corner nll loss
                fg_rcnn_cnau = forward_ret_dict["rcnn_au"].view(
                    samples_in_all_batches, -1
                )[fg_mask]
                loss_corner_nll = loss_utils.get_corner_loss_lidar_uncer(
                    fg_rcnn_boxes3d_src[:, 0:7],
                    fg_gt_of_rois_src[:, 0:7],
                    fg_rcnn_cnau,
                )  # [B, M, 8]
                loss_corner_nll = (loss_corner_nll * fg_label_weight.squeeze(1)).mean()
                loss_weight = (
                    loss_cfgs.LOSS_WEIGHTS["rcnn_nll_weight"]
                    if "rcnn_nll_weight" in loss_cfgs.LOSS_WEIGHTS.keys()
                    else 1.0
                )
                loss_corner_nll = loss_corner_nll * loss_weight
                loss_corner += loss_corner_nll
                tb_dict[tag + "rcnn_nll"] = loss_corner_nll.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION:
                # corner l1 loss
                loss_corner_l1 = loss_utils.get_corner_loss_lidar(
                    fg_rcnn_boxes3d_src[:, 0:7],
                    fg_gt_of_rois_src[:, 0:7],
                )
                loss_corner_l1 = (loss_corner_l1 * fg_label_weight.squeeze(1)).mean()
                loss_weight = (
                    loss_cfgs.LOSS_WEIGHTS["rcnn_corner_weight"]
                    if "rcnn_corner_weight" in loss_cfgs.LOSS_WEIGHTS.keys()
                    else 1.0
                )
                loss_corner_l1 = loss_corner_l1 * loss_weight
                loss_corner += loss_corner_l1
                tb_dict[tag + "rcnn_corner"] = loss_corner_l1.item()

            rcnn_loss_reg += loss_corner
            # tb_dict[tag + "rcnn_loss_corner_all"] = loss_corner.item()

        return rcnn_loss_reg, tb_dict


if __name__ == "__main__":
    pass
