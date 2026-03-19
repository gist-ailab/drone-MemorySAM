"""
FCOS (Fully Convolutional One-Stage) Detection Head.

MemorySAM의 FPN 출력을 받아 anchor-free detection 수행.
FPN 스펙 (SAM2 Hiera B+):
  - fpn[0]: (B, 32, 64, 64)   — stride 16
  - fpn[1]: (B, 64, 32, 32)   — stride 32
  - fpn[2]: (B, 256, 16, 16)  — stride 64

각 FPN level에서 per-pixel로:
  - classification: (B, n_classes, H, W)
  - regression: (B, 4, H, W)  — (l, t, r, b) distances
  - centerness: (B, 1, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional


class Scale(nn.Module):
    """Learnable scalar multiplier (per FPN level)."""
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    """
    FCOS anchor-free detection head for multi-scale FPN features.

    Args:
        fpn_channels: List of channel dims per FPN level. e.g. [32, 64, 256]
        n_classes: Number of object categories.
        n_convs: Number of conv layers in cls/reg towers.
        hidden_dim: Hidden channel dim for towers (all levels projected to this).
        fpn_strides: Stride of each FPN level relative to input image.
        regress_ranges: (min, max) regression range per level for target assignment.
    """

    def __init__(
        self,
        fpn_channels: List[int] = [32, 64, 256],
        n_classes: int = 2,
        n_convs: int = 4,
        hidden_dim: int = 256,
        fpn_strides: List[int] = [16, 32, 64],
        regress_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_levels = len(fpn_channels)
        self.fpn_strides = fpn_strides
        self.regress_ranges = regress_ranges or [
            (-1, 64), (64, 128), (128, 1e8)
        ]

        # Per-level input projection to unified hidden_dim
        self.input_projs = nn.ModuleList([
            nn.Conv2d(ch, hidden_dim, 1) for ch in fpn_channels
        ])

        # Shared classification tower
        cls_tower = []
        for _ in range(n_convs):
            cls_tower.append(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False))
            cls_tower.append(nn.GroupNorm(32, hidden_dim))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Shared regression tower
        reg_tower = []
        for _ in range(n_convs):
            reg_tower.append(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False))
            reg_tower.append(nn.GroupNorm(32, hidden_dim))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)

        # Prediction heads
        self.cls_logits = nn.Conv2d(hidden_dim, n_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(hidden_dim, 4, 3, padding=1)
        self.centerness = nn.Conv2d(hidden_dim, 1, 3, padding=1)

        # Per-level learnable scale for regression
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(self.n_levels)])

        self._init_weights()

    def _init_weights(self):
        for module in [self.cls_tower, self.reg_tower]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Classification bias: prior probability 0.01
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, bias_value)

        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.zeros_(self.bbox_pred.bias)

        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.zeros_(self.centerness.bias)

    def forward(
        self,
        fpn_features: List[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            fpn_features: List of (B, C_level, H_level, W_level) per FPN level.

        Returns:
            dict with keys:
                'cls_logits': List[(B, n_classes, H, W)] per level
                'bbox_pred':  List[(B, 4, H, W)] per level — (l, t, r, b) * stride
                'centerness': List[(B, 1, H, W)] per level
        """
        all_cls = []
        all_reg = []
        all_ctr = []

        for level, feat in enumerate(fpn_features):
            # Project to unified dim
            feat = self.input_projs[level](feat)

            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_out = self.cls_logits(cls_feat)            # (B, n_classes, H, W)
            reg_out = self.scales[level](self.bbox_pred(reg_feat))  # (B, 4, H, W)
            reg_out = F.relu(reg_out) * self.fpn_strides[level]    # positive distances, scaled by stride
            ctr_out = self.centerness(reg_feat)            # (B, 1, H, W)

            all_cls.append(cls_out)
            all_reg.append(reg_out)
            all_ctr.append(ctr_out)

        return {
            'cls_logits': all_cls,
            'bbox_pred': all_reg,
            'centerness': all_ctr,
        }

    def get_locations(
        self,
        fpn_features: List[torch.Tensor],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Compute grid locations (cx, cy) for each FPN level.

        Returns:
            List of (H*W, 2) tensors — pixel centers in input image coordinates.
        """
        locations = []
        for level, feat in enumerate(fpn_features):
            h, w = feat.shape[-2:]
            stride = self.fpn_strides[level]
            # Grid centers: (stride/2, stride/2), (stride/2 + stride, stride/2), ...
            shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device) + stride // 2
            shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device) + stride // 2
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            locations.append(torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1))
        return locations

    def decode_predictions(
        self,
        cls_logits: List[torch.Tensor],
        bbox_pred: List[torch.Tensor],
        centerness: List[torch.Tensor],
        locations: List[torch.Tensor],
        score_thresh: float = 0.05,
        topk_per_level: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode FCOS predictions to (x1, y1, x2, y2) boxes.

        Returns per image:
            boxes: (K, 4)
            scores: (K,)
            class_ids: (K,)
        """
        B = cls_logits[0].shape[0]
        all_boxes = [[] for _ in range(B)]
        all_scores = [[] for _ in range(B)]
        all_classes = [[] for _ in range(B)]

        for level in range(self.n_levels):
            cls = cls_logits[level]     # (B, C, H, W)
            reg = bbox_pred[level]      # (B, 4, H, W)
            ctr = centerness[level]     # (B, 1, H, W)
            locs = locations[level]     # (H*W, 2)

            B, C, H, W = cls.shape
            cls = cls.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)
            reg = reg.permute(0, 2, 3, 1).reshape(B, H * W, 4)  # (B, HW, 4)
            ctr = ctr.permute(0, 2, 3, 1).reshape(B, H * W, 1)  # (B, HW, 1)

            # Score = sqrt(sigmoid(cls) * sigmoid(centerness))
            cls_scores = cls.sigmoid() * ctr.sigmoid()  # (B, HW, C)

            for b in range(B):
                scores_b = cls_scores[b]  # (HW, C)
                max_scores, max_cls = scores_b.max(dim=1)  # (HW,), (HW,)

                # Filter by score threshold
                keep = max_scores > score_thresh
                if keep.sum() == 0:
                    continue

                scores_k = max_scores[keep]
                cls_k = max_cls[keep]
                reg_k = reg[b][keep]    # (K, 4) — (l, t, r, b)
                locs_k = locs[keep]     # (K, 2) — (cx, cy)

                # Top-k per level
                if scores_k.shape[0] > topk_per_level:
                    topk_idx = scores_k.topk(topk_per_level).indices
                    scores_k = scores_k[topk_idx]
                    cls_k = cls_k[topk_idx]
                    reg_k = reg_k[topk_idx]
                    locs_k = locs_k[topk_idx]

                # Decode ltrb → x1y1x2y2
                x1 = locs_k[:, 0] - reg_k[:, 0]
                y1 = locs_k[:, 1] - reg_k[:, 1]
                x2 = locs_k[:, 0] + reg_k[:, 2]
                y2 = locs_k[:, 1] + reg_k[:, 3]
                boxes_k = torch.stack([x1, y1, x2, y2], dim=1)

                all_boxes[b].append(boxes_k)
                all_scores[b].append(scores_k)
                all_classes[b].append(cls_k)

        # Concatenate across levels per image
        results = []
        for b in range(B):
            if all_boxes[b]:
                results.append((
                    torch.cat(all_boxes[b]),
                    torch.cat(all_scores[b]),
                    torch.cat(all_classes[b]),
                ))
            else:
                results.append((
                    torch.zeros((0, 4), device=cls_logits[0].device),
                    torch.zeros((0,), device=cls_logits[0].device),
                    torch.zeros((0,), dtype=torch.int64, device=cls_logits[0].device),
                ))
        return results
