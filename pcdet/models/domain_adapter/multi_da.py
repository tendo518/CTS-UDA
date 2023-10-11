import torch
import torch.nn as nn
from ...utils import loss_utils
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class WeightedReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha, ctx_list):
        ctx.alpha = alpha
        ctx_list.append(ctx)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha * ctx.weight
        # print('grad shape, weight shape:', grad_output.shape, ctx.weight.shape)
        return output, None, None


weighted_reverseL = WeightedReverseLayerF.apply


class MultiDA(nn.Module):
    def __init__(self, model_cfg):
        super(MultiDA, self).__init__()
        self.model_cfg = model_cfg
        self.weight = getattr(model_cfg, "WEIGHT", True)
        self.global_features = model_cfg.GLOBAL_FEATURES
        self.global_da = nn.ModuleList()
        self.global_losses = []
        self.forward_dict = [{}, {}]

        assert len(model_cfg.GLOBAL_FEATURES) == len(model_cfg.GLOBAL_LOSSES)
        for i, idx in enumerate(self.global_features):
            assert idx > 0, "feature 0 is None"

            if model_cfg.GLOBAL_LOSSES[i] == "CE":
                self.global_losses.append(loss_utils.BCELoss(reduction="none"))
            elif model_cfg.GLOBAL_LOSSES[i] == "FL":
                self.global_losses.append(loss_utils.SigmoidFocalClassificationLoss())
            else:
                raise NotImplementedError

            self.global_da.append(
                nn.Sequential(
                    nn.Linear(model_cfg.GLOBAL_FEATURES_DIM[idx], 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(True),
                    nn.Linear(128, 1),
                )
            )

        self.inst_features = model_cfg.INST_FEATURES
        self.inst_da = nn.ModuleList()
        self.inst_losses = []
        assert len(model_cfg.INST_FEATURES) == len(model_cfg.INST_LOSSES)
        for i, idx in enumerate(self.inst_features):
            assert idx > 0, "feature 0 is None"
            if model_cfg.INST_LOSSES[i] == "CE":
                self.inst_losses.append(loss_utils.BCELoss(reduction="none"))
            elif model_cfg.INST_LOSSES[i] == "FL":
                self.inst_losses.append(loss_utils.SigmoidFocalClassificationLoss())
            else:
                raise NotImplementedError

            self.inst_da.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.modules.AdaptiveAvgPool1d(1),
                            nn.modules.Flatten(start_dim=1),
                        ),
                        nn.Sequential(
                            nn.Linear(model_cfg.INST_FEATURES_DIM[idx], 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(True),
                            nn.Linear(128, 1),
                        ),
                    ]
                )
            )

    def forward(self, batch_dict):
        if not self.training:
            return batch_dict

        alpha = batch_dict["alpha"]

        # global
        global_out = []
        global_features = batch_dict["point_features_all"]
        for idx, model in zip(self.global_features, self.global_da):
            idx, x = global_features[idx]
            x = ReverseLayerF.apply(x, alpha)
            x = x.transpose(1, 2).contiguous().view(-1, x.shape[1])
            out = model(x)
            global_out.append((idx, out))

        inst_features = batch_dict["roi_features_all"]
        inst_out = []
        inst_pooled_feats = []
        for idx, model in zip(self.inst_features, self.inst_da):
            x = inst_features[idx]
            x = ReverseLayerF.apply(x, alpha)
            x = model[0](x)
            inst_pooled_feats.append(x.detach())
            out = model[1](x)
            inst_out.append(out)
        self.forward_dict[batch_dict["domain_label"]] = {
            "global": global_out,
            "point_cls_prob": batch_dict["point_cls_scores"].detach(),
            "inst_feats": inst_pooled_feats,
            "inst": inst_out,
            "domain_label": batch_dict["domain_label"],
            "batch_size": batch_dict["batch_size"],
            "roi_prob": torch.sigmoid(batch_dict["rcnn_cls_score"].detach()).squeeze(),
        }
        return batch_dict

    def get_loss(self):
        loss = 0
        for domain_lable in range(2):
            forward_dict = self.forward_dict[domain_lable]
            batch_size = forward_dict["batch_size"]
            point_cls_prob = forward_dict["point_cls_prob"].view(batch_size, -1)
            roi_prob = forward_dict["roi_prob"]
            for loss_fun, (idx, y) in zip(self.global_losses, forward_dict["global"]):
                y = y.view(-1)
                target = torch.empty_like(y).fill_(forward_dict["domain_label"])
                if not self.weight:
                    weight = torch.ones_like(y)
                elif idx is None:
                    weight = point_cls_prob.view(-1)
                else:
                    weight = torch.gather(point_cls_prob, dim=1, index=idx.long()).view(
                        -1
                    )
                    weight = (weight > weight.mean()) * weight
                    weight = 1 + weight
                loss = loss + loss_fun(y, target, weight).sum() / weight.sum()

            for loss_fun, y in zip(self.inst_losses, forward_dict["inst"]):
                y = y.view(-1)
                target = torch.empty_like(y).fill_(forward_dict["domain_label"])
                if not self.weight:
                    weight = torch.ones_like(y)
                else:
                    weight = roi_prob
                    weight = (weight > weight.mean()) * weight
                    weight = 1 + weight
                loss = loss + loss_fun(y, target, weight).sum() / weight.sum()
        return loss


def cos_similarity(feat1, feat2):
    """

    Args:
        feat1: N x C
        feat2: M x C
    """
    feat1 = feat1 / (torch.norm(feat1, dim=1, p=2, keepdim=True) + 1e-8)
    feat2 = feat2 / (torch.norm(feat2, dim=1, p=2, keepdim=True) + 1e-8)
    sim = feat1 @ feat2.transpose(0, 1)
    return sim


class HEMMultiDA(MultiDA):
    def __init__(self, model_cfg):
        super(HEMMultiDA, self).__init__(model_cfg)
        self.ctx_list = [[], []]
        self.eps = 1e-8
        self.register_buffer(
            "bank_idx_0", torch.zeros(len(self.inst_features), dtype=torch.int)
        )
        self.register_buffer(
            "bank_idx_1", torch.zeros(len(self.inst_features), dtype=torch.int)
        )
        self.bank_len = self.model_cfg.BANK_LEN
        for i, idx in enumerate(self.inst_features):
            feat_bank = torch.zeros(
                self.model_cfg.BANK_LEN, self.model_cfg.INST_FEATURES_DIM[idx]
            )
            self.register_buffer("feat_bank_0_" + str(i), feat_bank)  # source domain
            self.register_buffer(
                "feat_bank_1_" + str(i), feat_bank.clone()
            )  # target domain

    def update_feat_bank(self, i, feats: torch.Tensor, domain: int):
        """
        Args:
            i: [0, len(self.inst_features))
            feats: N x K
            domain: 0 source, 1 target
        Returns:
        """
        feat_bank = getattr(self, "feat_bank_%d_%d" % (domain, i))
        bank_indices = getattr(self, "bank_idx_%d" % domain)
        bank_idx = bank_indices[i]
        N = feats.shape[0]
        if bank_idx + N <= self.bank_len:
            feat_bank[bank_idx : bank_idx + N, :] = feats[:, :]
            bank_indices[i] = bank_idx + N
        else:
            n = self.bank_len - bank_idx
            feat_bank[bank_idx : bank_idx + n, :] = feats[:n, :]
            n2 = N - n
            feat_bank[:n2, :] = feats[n:, :]
            bank_indices[i] = n2

    def get_similarity(self, i, feats: torch.Tensor, domain: int):
        """
        Args:
            i: [0, len(self.inst_features))
            feats: N x K
            domain: 0 source, 1 target
        Returns:
        """

        feat_bank = getattr(self, "feat_bank_%d_%d" % (abs(1 - domain), i))
        similarity = cos_similarity(feats, feat_bank)
        w = (
            similarity.max(1)[0] * 2 - 1
        )  # since any(feats >= 0) is true, the similarity is between 0 and 1
        return w

    def forward(self, batch_dict):
        if not self.training:
            return batch_dict

        alpha = batch_dict["alpha"]
        domain_label = batch_dict["domain_label"]

        # global
        global_out = []
        global_features = batch_dict["point_features_all"]
        for idx, model in zip(self.global_features, self.global_da):
            idx, x = global_features[idx]
            x = ReverseLayerF.apply(x, alpha)
            x = x.transpose(1, 2).contiguous().view(-1, x.shape[1])
            out = model(x)
            global_out.append((idx, out))

        inst_features = batch_dict["roi_features_all"]
        inst_out = []
        inst_pooled_feats = []
        for idx, model in zip(self.inst_features, self.inst_da):
            x = inst_features[idx]
            x = WeightedReverseLayerF.apply(x, alpha, self.ctx_list[domain_label])
            x = model[0](x)
            inst_pooled_feats.append(x.detach())
            out = model[1](x)
            inst_out.append(out)
        self.forward_dict[batch_dict["domain_label"]] = {
            "global": global_out,
            "point_cls_prob": batch_dict["point_cls_scores"].detach(),
            "inst_feats": inst_pooled_feats,
            "inst": inst_out,
            "domain_label": batch_dict["domain_label"],
            "batch_size": batch_dict["batch_size"],
            "roi_prob": torch.sigmoid(batch_dict["rcnn_cls_score"].detach()).squeeze(),
        }
        return batch_dict

    def get_loss(self):
        loss = 0
        for domain_lable in range(2):
            forward_dict = self.forward_dict[domain_lable]
            batch_size = forward_dict["batch_size"]
            point_cls_prob = forward_dict["point_cls_prob"].view(batch_size, -1)
            roi_prob = forward_dict["roi_prob"]
            for loss_fun, (idx, y) in zip(self.global_losses, forward_dict["global"]):
                y = y.view(-1)
                target = torch.empty_like(y).fill_(forward_dict["domain_label"])
                if not self.weight:
                    weight = torch.ones_like(y)
                elif idx is None:
                    weight = point_cls_prob.view(-1)
                else:
                    weight = torch.gather(point_cls_prob, dim=1, index=idx.long()).view(
                        -1
                    )
                    weight = (weight > weight.mean()) * weight
                    weight = 1 + weight
                loss = loss + loss_fun(y, target, weight).sum() / weight.sum()
        pos1 = (self.forward_dict[0]["roi_prob"] > 0.5).view(-1)
        pos2 = (self.forward_dict[1]["roi_prob"] > 0.5).view(-1)
        for i, (loss_fun, x1, y1, ctx1, x2, y2, ctx2) in enumerate(
            zip(
                self.inst_losses,
                self.forward_dict[0]["inst_feats"],
                self.forward_dict[0]["inst"],
                self.ctx_list[0],
                self.forward_dict[1]["inst_feats"],
                self.forward_dict[1]["inst"],
                self.ctx_list[1],
            )
        ):
            y1 = y1.view(-1)
            y2 = y2.view(-1)
            target1 = torch.empty_like(y1).fill_(self.forward_dict[0]["domain_label"])
            target2 = torch.empty_like(y2).fill_(self.forward_dict[1]["domain_label"])

            w1 = torch.ones_like(y1)
            w2 = torch.ones_like(y2)
            x1_pos = x1[pos1]
            x2_pos = x2[pos2]

            self.update_feat_bank(i, x1_pos, self.forward_dict[0]["domain_label"])
            self.update_feat_bank(i, x2_pos, self.forward_dict[1]["domain_label"])
            if torch.any(pos1):
                w1[pos1] = self.get_similarity(
                    i, x1_pos, self.forward_dict[0]["domain_label"]
                )
            if torch.any(pos2):
                w2[pos2] = self.get_similarity(
                    i, x2_pos, self.forward_dict[1]["domain_label"]
                )

            loss = loss + loss_fun(y1, target1, w1).mean()
            loss = loss + loss_fun(y2, target2, w2).mean()

            wctx1 = 2.0 - w1
            wctx2 = 2.0 - w2
            wctx1 /= w1 + self.eps
            wctx2 /= w2 + self.eps
            ctx1.weight = wctx1.view(-1, 1, 1)
            ctx2.weight = wctx2.view(-1, 1, 1)

        self.ctx_list = [[], []]
        return loss
