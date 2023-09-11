# from .partA2_head import PartA2FCHead
from .fusionrcnn_head import FusionRCNNHead
from .pointrcnn_head import PointRCNNHead
from .pointrcnn_head_cts import PointRCNNHeadCTS
# from .pvrcnn_head import PVRCNNHead
from .roi_head_template import RoIHeadTemplate

__all__ = {
    "RoIHeadTemplate": RoIHeadTemplate,
    #    'PartA2FCHead': PartA2FCHead,
    #   'PVRCNNHead': PVRCNNHead,
    "PointRCNNHead": PointRCNNHead,
    "FusionRCNNHead": FusionRCNNHead,
    "PointRCNNHeadCTS": PointRCNNHeadCTS,
}
