"""Model adapter for the public robustness-benchmark scripts.

The benchmark scripts call ``model(images)`` and expect a logits tensor at the input
resolution. Our model is built and loaded exactly as in ``val.load_model`` (the same
path used for every legal evaluation) and wrapped so that only logits are returned.
Nothing about preprocessing, corruption, scoring or E(p) lives here.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model(images, multimask_output=True)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        h, w = images[0].shape[-2:]
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        return logits


def build_ours_model(cfg, model_path):
    import val as valmod
    model = valmod.load_model(cfg, Path(model_path), torch.device('cpu'))
    model.eval()
    return LogitsOnly(model), f'built with val.load_model from {model_path}'
