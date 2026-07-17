"""P38 — MaskQueryLiteHead: Mask2Former-lite panoptic-capable query head.

Why (P38 rationale): DGFusion/CAFuser sit on OneFormer (mask-classification)
heads — they emit semantic+instance+panoptic from one model, so their MUSES
tables are PQ. Our FPNSegHead is per-pixel (semantic-only, PQ structurally
impossible) AND mask-cls heads are +1~3 mIoU on thin/rare classes (exactly our
Wall/Water/RailTrack weak spot). P38 upgrades the head to the same paradigm
class so (a) MUSES PQ becomes producible and (b) the head confound vs
DGFusion is removed, while keeping the frozen-DINOv3 + per-modal LoRA +
reliability-routing story untouched.

Design ("lite", house rules kept):
  - N_q LEARNED queries (default 100) over the GATED fused stride-16 map,
    6 pre-norm layers of masked cross-attn + self-attn + FFN (reuses P37b's
    `_TokenDecoderLayer`). Layer 1 unmasked, layers 2+ masked by the current
    per-query mask prediction (Mask2Former convention, NaN guard included).
  - Shared class head (K+1, incl. no-object) and 3-layer mask-embed MLP
    applied after EVERY layer -> deep supervision.
  - Masks = mask_embed · FPN stride-4 feature (same seam as P37b).
  - Hungarian matching (scipy) + CE(no-object weight 0.1) + point-sampled
    BCE + dice (uniform points, default 12544) — M2F weights 2/5/5.
  - Semantic scores = softmax(cls)[...,:K] ⊗ sigmoid(masks); the model merges
    them collapse-safe: final = conv_head + beta·sem_q (+ router residual)
    with beta ZERO-INIT. NOTE the P30/P37b "no Hungarian" lesson is
    deliberately overridden here — instances require matching; the zero-init
    residual + the always-on conv path is the collapse containment.
  - `panoptic_inference()` implements the standard M2F post-processing for
    MUSES PQ later; DELIVER training never calls it.

Params @ dim_t=256, 6 layers, mlp 2.0, ViT-L map: ~5.5M.
Losses are computed INSIDE the model (it already receives gt_mask when
training) and returned pre-scaled as aux['m2f_loss'] — trainer adds one term.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:                                        # pragma: no cover
    HAS_SCIPY = False

from .classtoken import _TokenDecoderLayer


def _dice_loss(pred_logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """pred_logits/tgt: (M, P). Returns mean dice loss over M matched masks."""
    x = pred_logits.sigmoid()
    num = 2.0 * (x * tgt).sum(-1)
    den = x.sum(-1) + tgt.sum(-1)
    return (1.0 - (num + 1.0) / (den + 1.0)).mean()


class MaskQueryLiteHead(nn.Module):
    """[P38] forward(fused, feat_s4) -> dict(cls, membed, masks, aux_states).

    fused   : (B, dim, h, w)          gated fused stride-16 map
    feat_s4 : (B, fpn_dim, H/4, W/4)  FPNSegHead pre-classifier feature
    """

    def __init__(self, dim: int, fpn_dim: int, num_classes: int,
                 num_queries: int = 100, num_layers: int = 6, dim_t: int = 256,
                 num_heads: int = 8, mlp_ratio: float = 2.0,
                 beta_init: float = 0.0, w_cls: float = 2.0, w_bce: float = 5.0,
                 w_dice: float = 5.0, no_obj_w: float = 0.1,
                 num_points: int = 12544, deep_supervision: bool = True):
        super().__init__()
        if not HAS_SCIPY:
            raise ImportError("[P38] MaskQueryLiteHead requires scipy "
                              "(Hungarian matching)")
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.w_cls, self.w_bce, self.w_dice = w_cls, w_bce, w_dice
        self.num_points = num_points
        self.deep_supervision = deep_supervision

        self.query = nn.Parameter(torch.empty(num_queries, dim_t))
        nn.init.normal_(self.query, std=0.02)
        self.in_proj = nn.Linear(dim, dim_t)
        self.layers = nn.ModuleList(
            _TokenDecoderLayer(dim_t, num_heads, mlp_ratio) for _ in range(num_layers))
        self.norm_out = nn.LayerNorm(dim_t)
        self.cls_head = nn.Linear(dim_t, num_classes + 1)  # +1 = no-object
        self.mask_mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.GELU(),
            nn.Linear(dim_t, dim_t), nn.GELU(),
            nn.Linear(dim_t, fpn_dim))
        # zero-init residual scale -> byte-identical to m2f-off dense path at
        # start; the query branch still trains at full strength through its
        # own Hungarian losses (unlike beta, they don't route through here).
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        ew = torch.ones(num_classes + 1)
        ew[num_classes] = no_obj_w
        self.register_buffer('empty_weight', ew)

    # ── decoder ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _attn_bias(self, q: torch.Tensor, feat_s4: torch.Tensor,
                   hw) -> torch.Tensor:
        """Mask2Former convention: the attention mask for layer l comes from
        the CURRENT (layer l-1) mask prediction through the SHARED heads —
        which deep supervision trains — resized from stride 4 to the stride-16
        attention grid. (An earlier separate `mask_proj` predictor received no
        gradient — thresholding is non-differentiable — and would have stayed
        at random init; same latent flaw exists in P37b classtoken.py.)
        Returns (B,1,Q,N) additive 0/-inf bias."""
        membed = self.mask_mlp(self.norm_out(q))               # (B,Q,fpn_dim)
        mlog = torch.einsum('bqc,bchw->bqhw', membed, feat_s4)
        mlog = F.interpolate(mlog, size=hw, mode='bilinear', align_corners=False)
        masked = (mlog.sigmoid() < 0.5).flatten(2)             # (B,Q,N)
        masked = masked & ~masked.all(dim=-1, keepdim=True)    # NaN guard
        bias = torch.zeros(masked.shape, dtype=q.dtype, device=q.device)
        bias = bias.masked_fill(masked, float('-inf'))
        return bias.unsqueeze(1)

    def forward(self, fused: torch.Tensor, feat_s4: torch.Tensor) -> Dict:
        B = fused.shape[0]
        hw = fused.shape[-2:]
        src = self.in_proj(fused.flatten(2).transpose(1, 2))   # (B,N,dim_t)
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        aux_states: List[torch.Tensor] = []
        for li, layer in enumerate(self.layers):
            bias = self._attn_bias(q, feat_s4, hw) if li > 0 else None
            q = layer(q, src, bias)
            if self.training and self.deep_supervision and li < len(self.layers) - 1:
                aux_states.append(q)
        qn = self.norm_out(q)
        membed = self.mask_mlp(qn)                             # (B,Q,fpn_dim)
        return {
            'cls': self.cls_head(qn),                          # (B,Q,K+1)
            'membed': membed,
            'masks': torch.einsum('bqc,bchw->bqhw', membed, feat_s4),
            'state': q,                                        # pre-norm final
            'aux_states': aux_states,
        }

    def semantic_scores(self, out: Dict) -> torch.Tensor:
        """Assembled per-class semantic scores (B,K,H/4,W/4), fp32."""
        prob = F.softmax(out['cls'].float(), dim=-1)[..., :self.num_classes]
        return torch.einsum('bqk,bqhw->bkhw', prob, out['masks'].float().sigmoid())

    # ── training losses (Hungarian, point-sampled, deep-supervised) ─────────
    @torch.no_grad()
    def _match(self, cls_logits: torch.Tensor, pred_pts: torch.Tensor,
               tgt_cls: torch.Tensor, tgt_pts: torch.Tensor):
        """Per-image Hungarian. cls_logits (Q,K+1), pred_pts (Q,P),
        tgt_cls (T,), tgt_pts (T,P) -> (query_idx, target_idx)."""
        prob = F.softmax(cls_logits, dim=-1)
        cost_cls = -prob[:, tgt_cls]                            # (Q,T)
        pos = F.binary_cross_entropy_with_logits(
            pred_pts, torch.ones_like(pred_pts), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            pred_pts, torch.zeros_like(pred_pts), reduction='none')
        cost_bce = (pos @ tgt_pts.T + neg @ (1.0 - tgt_pts).T) / pred_pts.shape[-1]
        x = pred_pts.sigmoid()
        num = 2.0 * (x @ tgt_pts.T)
        den = x.sum(-1)[:, None] + tgt_pts.sum(-1)[None, :]
        cost_dice = 1.0 - (num + 1.0) / (den + 1.0)
        C = (self.w_cls * cost_cls + self.w_bce * cost_bce
             + self.w_dice * cost_dice).cpu().numpy()
        qi, ti = linear_sum_assignment(C)
        return (torch.as_tensor(qi, device=cls_logits.device, dtype=torch.long),
                torch.as_tensor(ti, device=cls_logits.device, dtype=torch.long))

    def _layer_loss(self, state: torch.Tensor, feat_pts: torch.Tensor,
                    tgt_cls: List[torch.Tensor], tgt_pts: List[torch.Tensor]):
        """One decoder state -> matched CE + BCE + dice at sampled points.
        state (B,Q,dim_t); feat_pts (B,C,P) fp32."""
        qn = self.norm_out(state)
        cls_logits = self.cls_head(qn).float()                  # (B,Q,K+1)
        membed = self.mask_mlp(qn).float()                      # (B,Q,C)
        B = state.shape[0]
        ce_tgt = cls_logits.new_full((B, self.num_queries), self.num_classes,
                                     dtype=torch.long)
        bce = dice = cls_logits.sum() * 0.0
        n_masks = 0
        for b in range(B):
            if tgt_cls[b].numel() == 0:
                continue
            pred_pts = membed[b] @ feat_pts[b]                  # (Q,P)
            qi, ti = self._match(cls_logits[b].detach(), pred_pts.detach(),
                                 tgt_cls[b], tgt_pts[b])
            ce_tgt[b, qi] = tgt_cls[b][ti]
            mp, mt = pred_pts[qi], tgt_pts[b][ti]
            bce = bce + F.binary_cross_entropy_with_logits(mp, mt, reduction='mean') \
                * qi.numel()
            dice = dice + _dice_loss(mp, mt) * qi.numel()
            n_masks += qi.numel()
        loss_ce = F.cross_entropy(cls_logits.flatten(0, 1), ce_tgt.flatten(),
                                  weight=self.empty_weight)
        denom = max(n_masks, 1)
        return self.w_cls * loss_ce + self.w_bce * bce / denom \
            + self.w_dice * dice / denom

    def losses(self, out: Dict, feat_s4: torch.Tensor,
               gt_mask: torch.Tensor) -> torch.Tensor:
        """Total mask-cls loss (final layer + deep supervision), fp32 scalar."""
        B, C, h, w = feat_s4.shape
        with torch.no_grad():
            gt_s4 = F.interpolate(gt_mask.unsqueeze(1).float(), size=(h, w),
                                  mode='nearest').squeeze(1).long()   # (B,h,w)
        N = h * w
        P = min(self.num_points, N)
        idx = torch.randint(0, N, (B, P), device=feat_s4.device)
        feat_flat = feat_s4.flatten(2).float()                  # (B,C,N)
        feat_pts = torch.gather(feat_flat, 2,
                                idx.unsqueeze(1).expand(-1, C, -1))   # (B,C,P)
        gt_flat = gt_s4.flatten(1)                              # (B,N)
        gt_pts = torch.gather(gt_flat, 1, idx)                  # (B,P)
        tgt_cls, tgt_pts = [], []
        for b in range(B):
            cs = torch.unique(gt_s4[b])
            cs = cs[cs != 255]
            tgt_cls.append(cs)
            tgt_pts.append((gt_pts[b].unsqueeze(0) == cs.unsqueeze(1)).float())
        # one code path for final + deep-supervised layers (heads are shared;
        # re-applying them to the stored states is gradient-equivalent).
        states = out['aux_states'] + [out['state']]
        total = feat_s4.new_zeros((), dtype=torch.float32)
        for state in states:
            total = total + self._layer_loss(state, feat_pts, tgt_cls, tgt_pts)
        return total / len(states)

    # ── MUSES panoptic post-processing (unused on DELIVER) ──────────────────
    @torch.no_grad()
    def panoptic_inference(self, out: Dict, thing_ids: List[int],
                           obj_thresh: float = 0.8, overlap_thresh: float = 0.8):
        """Standard Mask2Former panoptic post-processing, per image.
        Returns list of (panoptic_seg (H/4,W/4) int32 segment ids,
        segments_info [{id, category_id, isthing}])."""
        results = []
        prob = F.softmax(out['cls'].float(), dim=-1)
        masks = out['masks'].float().sigmoid()
        for b in range(prob.shape[0]):
            scores, labels = prob[b].max(-1)
            keep = (labels != self.num_classes) & (scores > obj_thresh)
            cur_scores, cur_labels = scores[keep], labels[keep]
            cur_masks = masks[b][keep]                          # (M,h,w)
            h, w = cur_masks.shape[-2:]
            pan = torch.zeros((h, w), dtype=torch.int32, device=masks.device)
            segments: List[Dict] = []
            if cur_masks.numel() == 0:
                results.append((pan, segments))
                continue
            cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
            mask_ids = cur_prob_masks.argmax(0)                 # (h,w)
            seg_id = 0
            stuff_seg: Dict[int, int] = {}
            for k in range(cur_masks.shape[0]):
                cat = int(cur_labels[k])
                orig = (cur_masks[k] >= 0.5)
                area_orig = int(orig.sum())
                m = (mask_ids == k) & orig
                area = int(m.sum())
                if area == 0 or area_orig == 0 or area / area_orig < overlap_thresh:
                    continue
                isthing = cat in thing_ids
                if not isthing and cat in stuff_seg:            # merge stuff
                    pan[m] = stuff_seg[cat]
                    continue
                seg_id += 1
                pan[m] = seg_id
                if not isthing:
                    stuff_seg[cat] = seg_id
                segments.append({'id': seg_id, 'category_id': cat,
                                 'isthing': isthing})
            results.append((pan, segments))
        return results
