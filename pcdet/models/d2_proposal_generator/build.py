# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.registry import Registry

PROPOSAL_GENERATOR_REGISTRY = Registry("PROPOSAL_GENERATOR")
PROPOSAL_GENERATOR_REGISTRY.__doc__ = """
Registry for proposal generator, which produces object proposals from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""

from . import rpn  # noqa F401 isort:skip


def build_proposal_generator(cfg, input_shape):
    """
    Build a proposal generator from `cfg.MODEL.PROPOSAL_GENERATOR.NAME`.
    The name can be "PrecomputedProposals" to use no proposal generator.
    """
    name = cfg.IMAGE_PROPOSAL_GENERATOR.NAME
    if name == "PrecomputedProposals":
        return None

    return PROPOSAL_GENERATOR_REGISTRY.get(name)(
        cfg.IMAGE_PROPOSAL_GENERATOR, input_shape
    )
