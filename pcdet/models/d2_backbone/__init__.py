# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .fpn import build_resnet_fpn_backbone

__all__ = {"build_resnet_fpn_backbone": build_resnet_fpn_backbone}
# TODO can expose more resnet blocks after careful consideration
