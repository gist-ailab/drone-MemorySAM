# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Detection and segmentation head subpackage."""
from __future__ import annotations  # vendored: bengio env is py3.8

from rfdetr.models.heads.keypoints import ConditionalQueryInitializer
from rfdetr.models.heads.segmentation import DepthwiseConvBlock, MLPBlock, SegmentationHead

__all__ = [
    "SegmentationHead",
    "DepthwiseConvBlock",
    "MLPBlock",
    "ConditionalQueryInitializer",
]
