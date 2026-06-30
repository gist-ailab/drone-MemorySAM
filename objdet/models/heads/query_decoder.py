"""
Object-Query Detection Decoder + Hungarian set loss (P30-Det).

DETR-style set-prediction head: a fixed set of N object queries cross-attend the
fused, memory-conditioned cross-modal feature (`mem`, 256ch, carries all modalities
+ RBMA bias) produced by the P30 backbone. This ports P30 seg's **class-token decoder
(기구 ①)** from per-class masks to per-object (box + class): the detector actively
*queries* the memory feature instead of relying solely on dense FCOS per-pixel
prediction. Trained with a Hungarian-matched set loss (CE + L1 + GIoU).

Positional encoding for the memory tokens is sine/parameter-free so it is robust to
any feature-map size (mem at 1024 input = 64x64; no learned-table size cap).

Boxes are handled in normalized cxcywh ∈ [0,1] internally; the detector converts to
pixel xyxy for output / metric computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - greedy fallback if scipy missing
    _HAS_SCIPY = False


# ──────────────────────────── box utils ────────────────────────────
def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU between two sets of xyxy boxes → (N, M)."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    lti = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    whi = (rbi - lti).clamp(min=0)
    area_c = whi[..., 0] * whi[..., 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-6)


def sine_pos_embed(h: int, w: int, dim: int, device, temperature: float = 10000.0) -> torch.Tensor:
    """Parameter-free 2D sine positional embedding → (h*w, dim)."""
    d = dim // 2                      # channels per spatial axis
    half = d // 2
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    omega = torch.arange(half, device=device, dtype=torch.float32)
    omega = 1.0 / (temperature ** (omega / max(half, 1)))
    ox = xx.flatten()[:, None] * omega[None]      # (hw, half)
    oy = yy.flatten()[:, None] * omega[None]
    pe = torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], dim=1)  # (hw, 4*half)
    if pe.shape[1] < dim:             # pad if dim not divisible by 4
        pe = F.pad(pe, (0, dim - pe.shape[1]))
    return pe[:, :dim]


# ──────────────────────────── decoder ────────────────────────────
class _DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, dim))
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)

    def forward(self, q, q_pos, mem, mem_pos):
        qk = q + q_pos
        q = self.n1(q + self.self_attn(qk, qk, q)[0])
        q = self.n2(q + self.cross_attn(q + q_pos, mem + mem_pos, mem)[0])
        q = self.n3(q + self.ffn(q))
        return q


class ObjectQueryDecoder(nn.Module):
    """N object queries → (cls logits, box cxcywh) by cross-attending a fused feature map.

    Args:
        in_ch: channels of the fused memory feature (P30-Det = 256, the `mem` level).
        n_classes: object categories (no-object is an extra class = index n_classes).
        num_queries: number of object slots.
        dim/heads/ffn/n_layers: transformer-decoder hyper-params.
    """

    def __init__(self, in_ch: int, n_classes: int, num_queries: int = 100,
                 dim: int = 256, heads: int = 8, ffn: int = 1024, n_layers: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.num_queries = num_queries
        self.dim = dim
        self.proj = nn.Conv2d(in_ch, dim, 1)
        self.query_embed = nn.Embedding(num_queries, dim)   # content init
        self.query_pos = nn.Embedding(num_queries, dim)     # learned query positional
        self.layers = nn.ModuleList([_DecoderLayer(dim, heads, ffn) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.cls_head = nn.Linear(dim, n_classes + 1)       # +1 = no-object (last index)
        self.box_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, 4),
        )
        # bias box head toward center/medium so early queries are valid
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """feat: (B, in_ch, h, w) → {'pred_logits': (B,N,C+1), 'pred_boxes': (B,N,4) cxcywh}."""
        B, _, h, w = feat.shape
        f = self.proj(feat)                                  # (B, dim, h, w)
        mem = f.flatten(2).permute(0, 2, 1)                  # (B, hw, dim)
        mem_pos = sine_pos_embed(h, w, self.dim, feat.device).unsqueeze(0).expand(B, -1, -1)
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)    # (B, N, dim)
        q_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, q_pos, mem, mem_pos)
        q = self.norm(q)
        return {
            'pred_logits': self.cls_head(q),                 # (B, N, C+1)
            'pred_boxes': self.box_head(q).sigmoid(),        # (B, N, 4) cxcywh ∈ [0,1]
        }


# ──────────────────────────── matcher + loss ────────────────────────────
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        B, N = outputs['pred_logits'].shape[:2]
        indices = []
        for b in range(B):
            tgt_ids = targets[b]['labels']
            tgt_box = targets[b]['boxes']               # cxcywh normalized
            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue
            prob = outputs['pred_logits'][b].softmax(-1)     # (N, C+1)
            pred_box = outputs['pred_boxes'][b]              # (N, 4)
            cost_class = -prob[:, tgt_ids]                   # (N, n_tgt)
            cost_bbox = torch.cdist(pred_box, tgt_box, p=1)  # (N, n_tgt)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(pred_box), box_cxcywh_to_xyxy(tgt_box))
            C = (self.cost_class * cost_class + self.cost_bbox * cost_bbox
                 + self.cost_giou * cost_giou)              # (N, n_tgt)
            C = torch.nan_to_num(C, nan=1e4, posinf=1e4, neginf=-1e4)
            if _HAS_SCIPY:
                ri, ci = linear_sum_assignment(C.cpu().numpy())
                indices.append((torch.as_tensor(ri, dtype=torch.int64),
                                torch.as_tensor(ci, dtype=torch.int64)))
            else:
                # greedy fallback: assign each gt to its lowest-cost free query
                ri, ci, used = [], [], set()
                order = list(range(tgt_ids.numel()))
                for t in order:
                    col = C[:, t].clone()
                    for u in used:
                        col[u] = 1e9
                    q = int(col.argmin().item())
                    used.add(q); ri.append(q); ci.append(t)
                indices.append((torch.as_tensor(ri, dtype=torch.int64),
                                torch.as_tensor(ci, dtype=torch.int64)))
        return indices


class SetCriterion(nn.Module):
    """DETR-style set loss: CE (with no-object) + L1 + GIoU on Hungarian-matched pairs."""

    def __init__(self, n_classes: int, matcher: HungarianMatcher,
                 eos_coef: float = 0.1, w_class: float = 1.0, w_bbox: float = 5.0, w_giou: float = 2.0):
        super().__init__()
        self.n_classes = n_classes
        self.matcher = matcher
        self.w_class = w_class
        self.w_bbox = w_bbox
        self.w_giou = w_giou
        empty_weight = torch.ones(n_classes + 1)
        empty_weight[-1] = eos_coef                  # down-weight no-object class
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        pred_logits = outputs['pred_logits']         # (B,N,C+1)
        pred_boxes = outputs['pred_boxes']           # (B,N,4)
        B, N = pred_logits.shape[:2]
        device = pred_logits.device

        # ── classification (all queries) ──
        target_classes = torch.full((B, N), self.n_classes, dtype=torch.int64, device=device)
        for b, (qi, ti) in enumerate(indices):
            if qi.numel() > 0:
                target_classes[b, qi.to(device)] = targets[b]['labels'][ti.to(device)]
        loss_cls = F.cross_entropy(
            pred_logits.transpose(1, 2), target_classes, self.empty_weight)

        # ── box (matched only) ──
        num_boxes = max(sum(qi.numel() for qi, _ in indices), 1)
        loss_l1 = pred_boxes.new_zeros(())
        loss_giou = pred_boxes.new_zeros(())
        for b, (qi, ti) in enumerate(indices):
            if qi.numel() == 0:
                continue
            qi = qi.to(device); ti = ti.to(device)
            pb = pred_boxes[b, qi]                    # (n,4) cxcywh
            tb = targets[b]['boxes'][ti]              # (n,4) cxcywh
            loss_l1 = loss_l1 + F.l1_loss(pb, tb, reduction='sum')
            giou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(tb)))
            loss_giou = loss_giou + (1.0 - giou).sum()
        loss_l1 = loss_l1 / num_boxes
        loss_giou = loss_giou / num_boxes

        loss_total = self.w_class * loss_cls + self.w_bbox * loss_l1 + self.w_giou * loss_giou
        return {
            'loss_query_cls': self.w_class * loss_cls,
            'loss_query_bbox': self.w_bbox * loss_l1,
            'loss_query_giou': self.w_giou * loss_giou,
            'loss_query_total': loss_total,
        }


@torch.no_grad()
def decode_queries(outputs: Dict[str, torch.Tensor], img_w: int, img_h: int,
                   score_thresh: float = 0.05, max_det: int = 100):
    """Decode query outputs → per-image (boxes xyxy px, scores, class_ids)."""
    prob = outputs['pred_logits'].softmax(-1)[..., :-1]      # drop no-object → (B,N,C)
    boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])        # (B,N,4) normalized
    scale = boxes.new_tensor([img_w, img_h, img_w, img_h])
    boxes = boxes * scale
    B = prob.shape[0]
    results = []
    for b in range(B):
        scores_b, labels_b = prob[b].max(-1)                 # (N,)
        keep = scores_b > score_thresh
        s, l, bx = scores_b[keep], labels_b[keep], boxes[b][keep]
        if s.numel() > max_det:
            idx = s.topk(max_det).indices
            s, l, bx = s[idx], l[idx], bx[idx]
        results.append({'boxes': bx, 'scores': s, 'class_ids': l})
    return results
