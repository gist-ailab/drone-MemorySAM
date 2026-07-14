# Vendored subset of RF-DETR (roboflow/rf-detr, Apache-2.0). Backbone/CLI removed;
# only the NMS-free decoder + matcher/criterion/postprocess are kept.
# Package init intentionally empty: upstream __init__ eagerly imports the DINOv2 backbone.

from __future__ import annotations  # vendored: bengio env is py3.8