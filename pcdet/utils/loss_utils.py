from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils


class BCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        return F.binary_cross_entropy(
            input=F.sigmoid(input),
            target=target,
            weight=weights,
            reduction=self.reduction,
        )


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0)
            - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or (
            weights.shape.__len__() == 1 and target.shape.__len__() == 2
        ):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)

        return loss

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None
    ):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert (
                weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            )
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None
    ):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert (
                weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            )
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction="none") * weights
        return loss


class AULoss(nn.Module):
    def __init__(
        self,
        distribution: Literal["gaussion", "laplace"] = "gaussion",
        code_weights: list = None,
    ):
        super().__init__()
        self.distribution = distribution
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(
        self,
        input: torch.tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input: (B, #anchors, #codes + #codes) float tensor.
                Predicted locations of objects, with its predicated uncertainity (variance)
            target: (B, #anchors, #codes) float tensor.
                Regression targets.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        assert input.shape[:2] == target.shape[:2], f"{input.shape} vs {target.shape}"
        codes_size = int(input.shape[-1] / 2)
        # assert(codes_size == target.shape[-1], "codes size mismatch")
        result = input[..., :codes_size]
        au = input[..., codes_size:]

        # ignore nan targets
        target = torch.where(torch.isnan(target), result, target)  
        au = torch.where(torch.isnan(target), torch.zeros_like(target), au)

        loss = self.nll(result, target, uncertainty=au, distribution=self.distribution)
        if self.code_weights is not None:
            loss = loss * self.code_weights
        if weights is not None:
            loss = loss * weights.unsqueeze(-1)
        return loss

    @staticmethod
    def nll(result, target, uncertainty, distribution="gaussion"):
        if distribution == "gaussion":
            # l2 when var is constant
            var = uncertainty
            loss = torch.square(result - target) / (2 * var) + torch.log(var) / 2
        elif distribution == "laplace":
            # l1 when var is constant
            b = uncertainty
            loss = torch.abs(result - target) / b + torch.log(2 * b)
        else:
            raise NotImplementedError(f"{distribution} nll not implemented")
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi

    gt_bbox3d[:, 6] %= np.pi * 2
    gt_bbox3d_flip[:, 6] %= np.pi * 2
    
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(
        torch.norm(pred_box_corners - gt_box_corners, dim=2),
        torch.norm(pred_box_corners - gt_box_corners_flip, dim=2),
    )
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def get_corner_loss_lidar_uncer(
    pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor, pred_cn_au: torch.Tensor
):
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0] == pred_cn_au.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    
    gt_bbox3d[:, 6] %= np.pi * 2
    gt_bbox3d_flip[:, 6] %= np.pi * 2

    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    # corner_dist = torch.min(
    #     torch.norm(pred_box_corners - gt_box_corners, dim=2), # distance
    #     torch.norm(pred_box_corners - gt_box_corners_flip, dim=2),
    # )
    # n*8 -> n*8*3 
    # uncertainty towards xyz is consided the same in same corner
    if pred_cn_au.shape[-1] == 24:
        # xyz 方向独立
        pred_box_corners_au = pred_cn_au.reshape(gt_box_corners.shape)
    else:
        pred_box_corners_au = pred_cn_au.unsqueeze(-1).expand_as(gt_box_corners)
    corner_loss_au = torch.min(
        AULoss.nll(pred_box_corners, gt_box_corners, pred_box_corners_au),
        AULoss.nll(pred_box_corners, gt_box_corners_flip, pred_box_corners_au),
    )
    return corner_loss_au.mean(dim=1).mean(dim=1)
