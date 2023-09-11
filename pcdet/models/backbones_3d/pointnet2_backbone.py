import torch
import torch.nn as nn
from torch.nn.functional import grid_sample

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import (
    pointnet2_modules as pointnet2_modules_stack,
)
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack


class Identity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *args):
        return args[-1]


class PCT(nn.Module):
    def __init__(self, model_cfg, img_channels, pts_channels):
        super(PCT, self).__init__()

        self.model_cfg = model_cfg
        in_channels = img_channels + pts_channels
        reduction = model_cfg.REDUCTION

        self.q_conv = nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(pts_channels, pts_channels, 1)
        self.trans_conv = nn.Conv1d(pts_channels, pts_channels, 1)
        self.after_norm = nn.BatchNorm1d(pts_channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pts_img, img_feats, pts_feats):
        pts_grid = pts_img.unsqueeze(1)  # (B, 1, N, 2)
        pts_wise_img_feats = grid_sample(img_feats, pts_grid)  # (B, C_I, 1, N)

        pts_wise_img_feats = pts_wise_img_feats.squeeze(2)  # (B, C_I, 1, N)
        fusion_feats = torch.cat([pts_wise_img_feats, pts_feats], dim=1)

        # b, n, c
        x_q = self.q_conv(fusion_feats).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(fusion_feats)
        x_v = self.v_conv(pts_feats)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(pts_feats - x_r)))
        x = pts_feats + x_r
        return x


class PointWiseGate(nn.Module):
    def __init__(self, model_cfg, img_channels, pts_channels):
        super(PointWiseGate, self).__init__()
        self.model_cfg = model_cfg
        in_channels = img_channels + pts_channels
        reduction = model_cfg.REDUCTION
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(in_channels, pts_channels, kernel_size=1),
            nn.BatchNorm1d(
                pts_channels,
            ),
            nn.ReLU(),
        )

        self.channel_att = nn.Sequential(
            nn.Conv1d(pts_channels, pts_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(pts_channels // reduction, pts_channels, kernel_size=1),
        )

        self.spatial_att = nn.Conv1d(
            pts_channels, pts_channels // reduction, kernel_size=1
        )

    def forward(self, pts_img, img_feats, pts_feats):
        """

        Args:
            pts_img: B x N x 2
            img_feats: B x C_I x H x W
            pts_feats: B x C_P x N

        Returns: B x C x C_P

        """
        pts_grid = pts_img.unsqueeze(1)  # (B, 1, N, 2)
        pts_wise_img_feats = grid_sample(img_feats, pts_grid)  # (B, C_I, 1, N)

        pts_wise_img_feats = pts_wise_img_feats.squeeze(2)  # (B, C_I, 1, N)
        fusion_feats = torch.cat([pts_wise_img_feats, pts_feats], dim=1)

        fusion_feats = self.fusion_mlp(fusion_feats)

        # spatial attention
        spatial_att, _ = torch.max(self.spatial_att(fusion_feats), dim=1, keepdim=True)
        spatial_att = torch.sigmoid(spatial_att)
        fusion_feats = spatial_att * fusion_feats

        # channel attention
        channel_att = torch.max_pool1d(
            self.channel_att(fusion_feats), kernel_size=fusion_feats.shape[-1]
        )  # B, C
        channel_att = torch.sigmoid(channel_att)  # B, C, 1
        fusion_feats = channel_att * fusion_feats

        return fusion_feats


class PointWiseGate7(nn.Module):
    def __init__(self, model_cfg, img_channels, pts_channels):
        super(PointWiseGate7, self).__init__()
        self.model_cfg = model_cfg
        reduction = model_cfg.REDUCTION

        self.spatial_att = nn.Sequential(
            nn.Conv1d(img_channels, img_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(img_channels // reduction, img_channels, kernel_size=1),
        )

    def forward(self, pts_img, img_feats, pts_feats):
        """

        Args:
            pts_img: B x N x 2
            img_feats: B x C_I x H x W
            pts_feats: B x C_P x N

        Returns: B x C x C_P

        """
        pts_grid = pts_img.unsqueeze(1)  # (B, 1, N, 2)
        pts_wise_img_feats = grid_sample(img_feats, pts_grid)  # (B, C_I, 1, N)

        pts_wise_img_feats = pts_wise_img_feats.squeeze(2)  # (B, C_I, 1, N)

        # spatial attention
        spatial_att, _ = torch.max(
            self.spatial_att(pts_wise_img_feats), dim=1, keepdim=True
        )
        spatial_att = torch.sigmoid(spatial_att)
        # fusion_feats = spatial_att * pts_feats

        return spatial_att * pts_feats


class PointWiseGate12(nn.Module):
    def __init__(self, model_cfg, img_channels, pts_channels):
        super(PointWiseGate12, self).__init__()
        self.model_cfg = model_cfg
        reduction = model_cfg.REDUCTION
        input_channels = img_channels + pts_channels
        self.spatial_att = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=1),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(),
            nn.Conv1d(input_channels, input_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(input_channels // reduction, input_channels, kernel_size=1),
        )

    def forward(self, pts_img, img_feats, pts_feats):
        """

        Args:
            pts_img: B x N x 2
            img_feats: B x C_I x H x W
            pts_feats: B x C_P x N

        Returns: B x C x C_P

        """
        pts_grid = pts_img.unsqueeze(1)  # (B, 1, N, 2)
        pts_wise_img_feats = grid_sample(img_feats, pts_grid)  # (B, C_I, 1, N)

        pts_wise_img_feats = pts_wise_img_feats.squeeze(2)  # (B, C_I, 1, N)
        fusion_feats = torch.cat([pts_wise_img_feats, pts_feats], dim=1)

        # spatial attention
        spatial_att, _ = torch.max(self.spatial_att(fusion_feats), dim=1, keepdim=True)
        spatial_att = torch.tanh(spatial_att)
        # fusion_feats = spatial_att * pts_feats

        return (1 + spatial_att) * pts_feats


class PointWiseGate8(nn.Module):
    def __init__(self, model_cfg, img_channels, pts_channels):
        super(PointWiseGate8, self).__init__()
        self.model_cfg = model_cfg
        reduction = model_cfg.REDUCTION
        input_channels = img_channels + pts_channels
        self.spatial_att = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=1),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(),
            nn.Conv1d(input_channels, input_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(input_channels // reduction, input_channels, kernel_size=1),
        )

    def forward(self, pts_img, img_feats, pts_feats):
        """

        Args:
            pts_img: B x N x 2
            img_feats: B x C_I x H x W
            pts_feats: B x C_P x N

        Returns: B x C x C_P

        """
        pts_grid = pts_img.unsqueeze(1)  # (B, 1, N, 2)
        pts_wise_img_feats = grid_sample(img_feats, pts_grid)  # (B, C_I, 1, N)

        pts_wise_img_feats = pts_wise_img_feats.squeeze(2)  # (B, C_I, 1, N)
        fusion_feats = torch.cat([pts_wise_img_feats, pts_feats], dim=1)

        # spatial attention
        spatial_att, _ = torch.max(self.spatial_att(fusion_feats), dim=1, keepdim=True)
        spatial_att = torch.sigmoid(spatial_att)
        # fusion_feats = spatial_att * pts_feats

        return spatial_att * pts_feats


class PBMA(nn.Module):
    def __init__(self, in_channels, reduction):
        super(PBMA, self).__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1),
        )

        self.channel_att = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1),
        )

    def forward(self, feats):
        """

        Args:
            feats: B x C x N

        Returns: B x C x N

        """
        spatial_att, _ = torch.max(self.spatial_att(feats), dim=1, keepdim=True)
        spatial_att = torch.sigmoid(spatial_att)
        feats = spatial_att * feats

        # channel attention
        channel_att = torch.max_pool1d(
            self.channel_att(feats), kernel_size=feats.shape[-1]
        )  # B, C
        channel_att = torch.sigmoid(channel_att)  # B, C, 1
        feats = channel_att * feats

        return feats


class PointWiseGate6(nn.Module):
    def __init__(self, model_cfg, img_channels, pts_channels):
        super(PointWiseGate6, self).__init__()
        self.model_cfg = model_cfg
        in_channels = img_channels + pts_channels
        reduction = model_cfg.REDUCTION
        self.align_mlp = nn.Sequential(
            nn.Conv1d(img_channels, pts_channels, kernel_size=1),
            nn.BatchNorm1d(
                pts_channels,
            ),
            nn.ReLU(),
        )

        self.pts_pbma = PBMA(pts_channels, reduction)
        self.img_pbma = PBMA(img_channels, reduction)
        self.weight = nn.Parameter(torch.zeros((1, pts_channels, 1)))

    def forward(self, pts_img, img_feats, pts_feats):
        """

        Args:
            pts_img: B x N x 2
            img_feats: B x C_I x H x W
            pts_feats: B x C_P x N

        Returns: B x C x C_P

        """
        pts_grid = pts_img.unsqueeze(1)  # (B, 1, N, 2)
        pts_wise_img_feats = grid_sample(img_feats, pts_grid)  # (B, C_I, 1, N)

        pts_wise_img_feats = pts_wise_img_feats.squeeze(2)  # (B, C_I, 1, N)

        pts_feats = self.pts_pbma(pts_feats)
        pts_wise_img_feats = self.img_pbma(pts_wise_img_feats)
        pts_wise_img_feats = self.align_mlp(pts_wise_img_feats)

        w = torch.sigmoid(self.weight)

        return w * pts_feats + (1.0 - w) * pts_wise_img_feats


class PointWiseGate5(nn.Module):
    def __init__(self, model_cfg, img_channels, pts_channels):
        super(PointWiseGate5, self).__init__()
        self.model_cfg = model_cfg
        offset_x, offset_y = torch.tensor([-1, 0, 1]), torch.tensor([-1, 0, 1])
        offset_x, offset_y = torch.meshgrid(offset_x, offset_y)
        offset = torch.cat([offset_x.reshape(-1, 1), offset_y.reshape(-1, 1)], dim=1)
        offset = offset.view(1, 9, 1, 2)
        self.register_buffer("offset", offset)
        in_channels = img_channels * 9 + pts_channels
        reduction = model_cfg.REDUCTION
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(in_channels, pts_channels, kernel_size=1),
            nn.BatchNorm1d(
                pts_channels,
            ),
            nn.ReLU(),
        )

        self.channel_att = nn.Sequential(
            nn.Conv1d(pts_channels, pts_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(pts_channels // reduction, pts_channels, kernel_size=1),
        )

        self.spatial_att = nn.Conv1d(
            pts_channels, pts_channels // reduction, kernel_size=1
        )

    def forward(self, pts_img, img_feats, pts_feats):
        """

        Args:
            pts_img: B x N x 2
            img_feats: B x C_I x H x W
            pts_feats: B x C_P x N

        Returns: B x C x C_P

        """
        h, w = img_feats.shape[2:]
        delta = torch.tensor([2 / w, 2 / h], device=self.offset.device).view(1, 1, 1, 2)
        offset = self.offset * delta

        pts_grid = pts_img.unsqueeze(1)  # (B, 1, N, 2)
        pts_grid = pts_grid + offset  # (B, 9, N, 2)
        pts_wise_img_feats = grid_sample(
            img_feats, pts_grid, padding_mode="zeros"
        )  # (B, C_I, 9, N)
        B, _, _, N = pts_wise_img_feats.shape
        pts_wise_img_feats = pts_wise_img_feats.view(B, -1, N)  # (B, C_I, 1, N)
        fusion_feats = torch.cat([pts_wise_img_feats, pts_feats], dim=1)

        fusion_feats = self.fusion_mlp(fusion_feats)

        # spatial attention
        spatial_att, _ = torch.max(self.spatial_att(fusion_feats), dim=1, keepdim=True)
        spatial_att = torch.sigmoid(spatial_att)
        fusion_feats = spatial_att * fusion_feats

        # channel attention
        channel_att = torch.max_pool1d(
            self.channel_att(fusion_feats), kernel_size=fusion_feats.shape[-1]
        )  # B, C
        channel_att = torch.sigmoid(channel_att)  # B, C, 1
        fusion_feats = channel_att * fusion_feats

        return fusion_feats


class PointWiseGate3(nn.Module):
    def __init__(self, model_cfg, img_channels, pts_channels):
        super(PointWiseGate3, self).__init__()
        self.model_cfg = model_cfg
        in_channels = img_channels + pts_channels
        reduction = model_cfg.REDUCTION
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(pts_channels, pts_channels, kernel_size=1),
            nn.BatchNorm1d(
                pts_channels,
            ),
            nn.ReLU(),
        )

        self.channel_att = nn.Sequential(
            nn.Conv1d(pts_channels, pts_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(pts_channels // reduction, pts_channels, kernel_size=1),
        )

        self.spatial_att = nn.Conv1d(
            pts_channels, pts_channels // reduction, kernel_size=1
        )

    def forward(self, pts_img, img_feats, pts_feats):
        """

        Args:
            pts_img: B x N x 2
            img_feats: B x C_I x H x W
            pts_feats: B x C_P x N

        Returns: B x C x C_P

        """
        # pts_grid = pts_img.unsqueeze(1)  # (B, 1, N, 2)
        # pts_wise_img_feats = grid_sample(img_feats, pts_grid)  # (B, C_I, 1, N)
        #
        # pts_wise_img_feats = pts_wise_img_feats.squeeze(2)  # (B, C_I, 1, N)
        # fusion_feats = torch.cat([pts_wise_img_feats, pts_feats], dim=1)

        fusion_feats = self.fusion_mlp(pts_feats)

        # spatial attention
        spatial_att, _ = torch.max(self.spatial_att(fusion_feats), dim=1, keepdim=True)
        spatial_att = torch.sigmoid(spatial_att)
        fusion_feats = spatial_att * fusion_feats

        # channel attention
        channel_att = torch.max_pool1d(
            self.channel_att(fusion_feats), kernel_size=fusion_feats.shape[-1]
        )  # B, C
        channel_att = torch.sigmoid(channel_att)  # B, C, 1
        fusion_feats = channel_att * fusion_feats

        return fusion_feats


class PointWiseGate2(nn.Module):
    def __init__(self, model_cfg, img_channels, pts_channels):
        super(PointWiseGate2, self).__init__()
        self.model_cfg = model_cfg
        in_channels = img_channels + pts_channels
        reduction = model_cfg.REDUCTION
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(in_channels, pts_channels, kernel_size=1),
            nn.BatchNorm1d(
                pts_channels,
            ),
            nn.ReLU(),
        )

        self.channel_att = nn.Sequential(
            nn.Conv1d(pts_channels, pts_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(pts_channels // reduction, pts_channels, kernel_size=1),
        )

        self.spatial_att = nn.Conv1d(
            pts_channels, pts_channels // reduction, kernel_size=1
        )

    def forward(self, pts_img, img_feats, pts_feats):
        """

        Args:
            pts_img: B x N x 2
            img_feats: B x C_I x H x W
            pts_feats: B x C_P x N

        Returns: B x C x C_P

        """
        pts_grid = pts_img.unsqueeze(1)  # (B, 1, N, 2)
        pts_wise_img_feats = grid_sample(img_feats, pts_grid)  # (B, C_I, 1, N)

        pts_wise_img_feats = pts_wise_img_feats.squeeze(2)  # (B, C_I, 1, N)
        fusion_feats = torch.cat([pts_wise_img_feats, pts_feats], dim=1)

        fusion_feats = self.fusion_mlp(fusion_feats)

        # spatial attention
        spatial_att, _ = torch.max(self.spatial_att(fusion_feats), dim=1, keepdim=True)
        spatial_att = torch.sigmoid(spatial_att)
        fusion_feats = spatial_att * fusion_feats

        # channel attention
        channel_att = torch.max_pool1d(
            self.channel_att(fusion_feats), kernel_size=fusion_feats.shape[-1]
        )  # B, C
        channel_att = torch.sigmoid(channel_att)  # B, C, 1
        fusion_feats = channel_att * fusion_feats

        return fusion_feats


class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        self.pwg_bot_up_modules = nn.ModuleList()

        pwg_config = model_cfg.PWG_CONFIG
        # pwg_module = eval(pwg_config.NAME)
        pwg_bu_modules = [eval(name) for name in pwg_config.SPECIFIED_ATT_BU]
        pwg_td_modules = [eval(name) for name in pwg_config.SPECIFIED_ATT_TD]

        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get("USE_XYZ", True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out
            self.pwg_bot_up_modules.append(
                pwg_bu_modules[k](
                    model_cfg.PWG_CONFIG, pwg_config.IMAGE_CHANNELS_BU[k], channel_in
                )
            )

        self.FP_modules = nn.ModuleList()
        self.pwg_top_down_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = (
                self.model_cfg.FP_MLPS[k + 1][-1]
                if k + 1 < len(self.model_cfg.FP_MLPS)
                else channel_out
            )
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )
            self.pwg_top_down_modules.append(
                pwg_td_modules[k](
                    model_cfg.PWG_CONFIG,
                    pwg_config.IMAGE_CHANNELS_TD[k],
                    self.model_cfg.FP_MLPS[k][-1],
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:pointnet2_backbone.py
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                point_features: (N, C)
                point_coords: (N, 4) -> (N, [batch_idx, coords])
        """
        batch_size = batch_dict["batch_size"]
        points = batch_dict["points"]
        batch_idx, xyz, features = self.break_up_pc(points)
        pts_img = batch_dict["pts_img"][:, 1:]
        img_feats_bot_up = [
            batch_dict["bottom_up_features"][k]
            for k in self.model_cfg.PWG_CONFIG.IMAGE_FEATURES_BU
        ]
        img_feats_top_down = [
            batch_dict["top_down_features"][k]
            for k in self.model_cfg.PWG_CONFIG.IMAGE_FEATURES_TD
        ]

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        pts_img = pts_img.view(batch_size, -1, 2)
        h, w = batch_dict["images"].image_sizes[0]
        pts_img[..., 0] = 2.0 * (pts_img[..., 0] / w) - 1.0
        pts_img[..., 1] = 2.0 * (pts_img[..., 1] / h) - 1.0
        assert torch.all(pts_img.min() >= -1) and torch.all(pts_img.max() <= 1)

        features = (
            features.view(batch_size, -1, features.shape[-1])
            .permute(0, 2, 1)
            .contiguous()
            if features is not None
            else None
        )

        l_xyz, l_features, li_idx, selected_pts = [xyz], [features], [], [pts_img]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, pts_idx = self.SA_modules[i](l_xyz[i], l_features[i])

            pts = torch.gather(
                selected_pts[-1], 1, pts_idx.long().unsqueeze(-1).repeat(1, 1, 2)
            )
            li_features = self.pwg_bot_up_modules[i](
                pts, img_feats_bot_up[i], li_features
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            li_idx.append(pts_idx)
            selected_pts.append(pts)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            pts_feats = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)
            pts_feats = self.pwg_top_down_modules[i](
                selected_pts[i - 1], img_feats_top_down[i], pts_feats
            )
            l_features[i - 1] = pts_feats
        point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict["point_features"] = point_features.view(-1, point_features.shape[-1])
        batch_dict["point_coords"] = torch.cat(
            (batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1
        )
        return batch_dict


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        assert (
            False
        ), "DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723"
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules_stack.StackSAModuleMSG(
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get("USE_XYZ", True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = (
                self.model_cfg.FP_MLPS[k + 1][-1]
                if k + 1 < len(self.model_cfg.FP_MLPS)
                else channel_out
            )
            self.FP_modules.append(
                pointnet2_modules_stack.StackPointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict["batch_size"]
        points = batch_dict["points"]
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points : (k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.furthest_point_sample(
                    cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]
                ).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)

            new_xyz_batch_cnt = (
                xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            )
            li_xyz, li_features = self.SA_modules[i](
                xyz=l_xyz[i],
                features=l_features[i],
                xyz_batch_cnt=l_batch_cnt[i],
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)

        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                unknown=l_xyz[i - 1],
                unknown_batch_cnt=l_batch_cnt[i - 1],
                known=l_xyz[i],
                known_batch_cnt=l_batch_cnt[i],
                unknown_feats=l_features[i - 1],
                known_feats=l_features[i],
            )

        batch_dict["point_features"] = l_features[0]
        batch_dict["point_coords"] = torch.cat(
            (batch_idx[:, None].float(), l_xyz[0]), dim=1
        )
        return batch_dict
