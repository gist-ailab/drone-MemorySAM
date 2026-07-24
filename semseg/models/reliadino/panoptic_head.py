"""P43 — PanopticDual: an INDEPENDENT Mask2Former mask-classification head.

Why a new module instead of extending P38's `m2f_head.MaskQueryLiteHead`:
P38 wired its query branch into the pixel head as `logits + beta*sem_q` with a
ZERO-INIT beta. That is failure-key 1 (`.claude_logs/experiments/analysis/
2026-07-20-failure-keys-...md`): the dense path already drives the loss to zero,
the residual never earns gradient, beta stalls at 0.13 and the module is an
inference no-op (off-Δ +0.04~+0.12). P30 tried the opposite extreme — replacing
the conv head with a query decoder — and small objects collapsed. P43 takes
neither: the mask-cls head is a **second primary loss** on the SAME SimpleFPN
trunk, with NO logit-level coupling to the pixel head whatsoever.

    L = L_pixel(CE + deep-sup, untouched) + lambda(t) * L_mask(Hungarian)

Nothing here adds to, gates, blends, or replaces the per-pixel logits during
training. `semantic_scores()` exists for EVAL-time analysis only (and for
MODEL.P43.SEM_SOURCE, which is documented as eval-only). The two heads meet
only where they are supposed to compete: the shared stride-4 SimpleFPN feature.

Architecture (Mask2Former 2112.01527, PMT 2603.25398 recipe):
  - N learned queries over the SimpleFPN levels {1/32, 1/16, 1/8} in round-robin
    (layer i attends level i % 3, coarse first), each layer masked by the
    CURRENT mask prediction from the SHARED cls/mask heads (so the mask that
    drives the attention is the one deep supervision trains — the P37b
    `mask_proj` "predictor that never receives gradient" bug cannot recur).
  - Masks = mask_embed . mask_features, where mask_features = 1x1 conv on the
    stride-4 SimpleFPN level (NOT the pixel head's internal feature — the heads
    share the trunk, not each other's decoders).
  - Loss = Hungarian match + CE(no-object 0.1) + point-sampled BCE + dice
    (2/5/5), PointRend sampling: uniform points for the matcher, uncertainty
    importance sampling (oversample 3x, 75% uncertain) for the loss.
  - Targets are built from the SEMANTIC GT in MaskFormer's "semantic
    segmentation as mask classification" mode (2107.06278 §3.3): one binary
    target mask per class PRESENT in the image. This trains today, with the
    semantic-only labels we actually have; a real panoptic loader would only
    change the target construction, not this module.

`panoptic_inference()` is the standard M2F post-processing and is what makes PQ
producible at all (the per-pixel head structurally cannot emit segments).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:                                        # pragma: no cover
    HAS_SCIPY = False

from .classtoken import _TokenDecoderLayer


def _point_sample(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """PointRend point_sample. feat (N,C,h,w), coords (N,P,2) in [0,1] -> (N,C,P).

    `align_corners=False` matches detectron2's point_rend; `padding_mode` is
    irrelevant because coords never leave [0,1].
    """
    grid = 2.0 * coords - 1.0                            # -> [-1,1]
    out = F.grid_sample(feat, grid.unsqueeze(2), mode='bilinear',
                        align_corners=False)
    return out.squeeze(-1)


def _dice_loss(pred_logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """pred_logits/tgt: (M,P). Mean dice loss over the M matched masks."""
    x = pred_logits.sigmoid()
    num = 2.0 * (x * tgt).sum(-1)
    den = x.sum(-1) + tgt.sum(-1)
    return (1.0 - (num + 1.0) / (den + 1.0)).mean()


class MaskClsHead(nn.Module):
    """[P43] Independent mask-classification head. forward(levels, feat_s4)."""

    def __init__(self, fpn_dim: int, num_classes: int,
                 num_queries: int = 100, dec_layers: int = 6, dim_t: int = 256,
                 num_heads: int = 8, mlp_ratio: float = 2.0, num_levels: int = 3,
                 w_cls: float = 2.0, w_bce: float = 5.0, w_dice: float = 5.0,
                 no_obj_w: float = 0.1, num_points: int = 12544,
                 oversample: float = 3.0, importance: float = 0.75,
                 deep_supervision: bool = True, ignore_index: int = 255):
        super().__init__()
        if not HAS_SCIPY:
            raise ImportError("[P43] MaskClsHead requires scipy (Hungarian matching)")
        if dec_layers < num_levels:
            # every level's in_proj/level_embed must see gradient every step,
            # otherwise DDP reports unused parameters (and one pyramid level
            # would silently never be read).
            raise ValueError(f"[P43] DEC_LAYERS ({dec_layers}) must be >= "
                             f"NUM_LEVELS ({num_levels}) — round-robin would "
                             f"leave a level unused")
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_levels = num_levels
        self.w_cls, self.w_bce, self.w_dice = w_cls, w_bce, w_dice
        self.num_points = int(num_points)
        self.oversample = float(oversample)
        self.importance = float(importance)
        self.deep_supervision = deep_supervision
        self.ignore_index = int(ignore_index)

        self.query = nn.Parameter(torch.empty(num_queries, dim_t))
        nn.init.normal_(self.query, std=0.02)
        self.level_embed = nn.Parameter(torch.empty(num_levels, dim_t))
        nn.init.normal_(self.level_embed, std=0.02)
        self.in_proj = nn.ModuleList(nn.Linear(fpn_dim, dim_t)
                                     for _ in range(num_levels))
        self.layers = nn.ModuleList(_TokenDecoderLayer(dim_t, num_heads, mlp_ratio)
                                    for _ in range(dec_layers))
        self.norm_out = nn.LayerNorm(dim_t)
        self.cls_head = nn.Linear(dim_t, num_classes + 1)      # +1 = no-object
        self.mask_mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.GELU(),
            nn.Linear(dim_t, dim_t), nn.GELU(),
            nn.Linear(dim_t, dim_t))
        # own pixel-decoder seam: the head reads the stride-4 SimpleFPN level
        # directly, never the pixel head's internal `fuse` feature.
        self.mask_feat_proj = nn.Conv2d(fpn_dim, dim_t, 1)
        ew = torch.ones(num_classes + 1)
        ew[num_classes] = no_obj_w
        self.register_buffer('empty_weight', ew)

    # ── decoder ─────────────────────────────────────────────────────────────
    def _heads(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qn = self.norm_out(q)
        return self.cls_head(qn), self.mask_mlp(qn)

    @torch.no_grad()
    def _attn_bias(self, membed: torch.Tensor, mf: torch.Tensor) -> torch.Tensor:
        """Additive 0/-inf mask (B,1,Q,N) for the next cross-attention.

        Mask2Former resizes the stride-4 mask LOGITS down to the attended
        level; because bilinear resampling is linear in the spatial index and
        the mask logit is linear in the channel index, the two commute — so
        `mf` is the mask FEATURE pre-resized to that level once per forward
        (far cheaper than resizing (B,Q,h4,w4) logits at every layer).
        """
        ml = torch.einsum('bqc,bchw->bqhw', membed, mf)
        masked = (ml.sigmoid() < 0.5).flatten(2)             # (B,Q,N)
        masked = masked & ~masked.all(dim=-1, keepdim=True)  # NaN guard
        bias = torch.zeros(masked.shape, dtype=membed.dtype, device=membed.device)
        return bias.masked_fill(masked, float('-inf')).unsqueeze(1)

    def forward(self, levels: Sequence[torch.Tensor],
                feat_s4: torch.Tensor) -> Dict:
        """levels: 3 SimpleFPN maps, COARSE FIRST ({1/32, 1/16, 1/8}).
        feat_s4: the stride-4 SimpleFPN level (B, fpn_dim, H/4, W/4)."""
        assert len(levels) == self.num_levels, \
            f"[P43] expected {self.num_levels} levels, got {len(levels)}"
        B = feat_s4.shape[0]
        mask_features = self.mask_feat_proj(feat_s4)          # (B,dim_t,h4,w4)
        srcs = [self.in_proj[i](l.flatten(2).transpose(1, 2)) + self.level_embed[i]
                for i, l in enumerate(levels)]
        with torch.no_grad():
            mf_down = [F.interpolate(mask_features, size=l.shape[-2:],
                                     mode='bilinear', align_corners=False)
                       for l in levels]
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        cls_l, me_l = self._heads(q)
        preds: List[Tuple[torch.Tensor, torch.Tensor]] = [(cls_l, me_l)]
        for li, layer in enumerate(self.layers):
            lvl = li % self.num_levels
            bias = self._attn_bias(me_l, mf_down[lvl])
            q = layer(q, srcs[lvl], bias)
            cls_l, me_l = self._heads(q)
            preds.append((cls_l, me_l))
        return {
            'cls': cls_l,                                     # (B,Q,K+1)
            'membed': me_l,                                   # (B,Q,dim_t)
            'mask_features': mask_features,
            'preds': preds if (self.training and self.deep_supervision)
                     else [preds[-1]],
        }

    def mask_logits(self, out: Dict) -> torch.Tensor:
        """(B,Q,H/4,W/4) high-res per-query mask logits."""
        return torch.einsum('bqc,bchw->bqhw', out['membed'], out['mask_features'])

    def semantic_scores(self, out: Dict) -> torch.Tensor:
        """Per-class semantic scores from the queries (B,K,H/4,W/4), fp32.

        EVAL-ONLY analysis path — never mixed into the training pixel logits.
        """
        prob = F.softmax(out['cls'].float(), dim=-1)[..., :self.num_classes]
        return torch.einsum('bqk,bqhw->bkhw', prob,
                            self.mask_logits(out).float().sigmoid())

    # ── training losses (Hungarian, PointRend-sampled, deep-supervised) ─────
    @torch.no_grad()
    def _match(self, cls_logits: torch.Tensor, pred_pts: torch.Tensor,
               tgt_cls: torch.Tensor, tgt_pts: torch.Tensor):
        """cls_logits (Q,K+1), pred_pts (Q,P), tgt_cls (T,), tgt_pts (T,P)."""
        prob = F.softmax(cls_logits, dim=-1)
        cost_cls = -prob[:, tgt_cls]                                    # (Q,T)
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
             + self.w_dice * cost_dice)
        C = torch.nan_to_num(C, nan=0.0, posinf=1e4, neginf=-1e4).cpu().numpy()
        qi, ti = linear_sum_assignment(C)
        return (torch.as_tensor(qi, device=cls_logits.device, dtype=torch.long),
                torch.as_tensor(ti, device=cls_logits.device, dtype=torch.long))

    @torch.no_grad()
    def _uncertain_coords(self, logits: torch.Tensor) -> torch.Tensor:
        """PointRend importance sampling. logits (M,h,w) -> coords (M,P,2)."""
        M = logits.shape[0]
        dev = logits.device
        P = self.num_points
        n_over = max(int(P * self.oversample), P)
        coords = torch.rand(M, n_over, 2, device=dev, dtype=logits.dtype)
        pts = _point_sample(logits.unsqueeze(1), coords)[:, 0]           # (M,n)
        k = min(int(P * self.importance), n_over)
        idx = (-pts.abs()).topk(k, dim=1).indices                       # (M,k)
        sel = torch.gather(coords, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
        rest = P - k
        if rest > 0:
            sel = torch.cat(
                [sel, torch.rand(M, rest, 2, device=dev, dtype=logits.dtype)], 1)
        return sel

    def _layer_loss(self, cls_logits: torch.Tensor, membed: torch.Tensor,
                    mask_features: torch.Tensor, targets) -> torch.Tensor:
        cls_logits = cls_logits.float()
        membed = membed.float()
        mf = mask_features.float()
        B, Q = cls_logits.shape[:2]
        h, w = mf.shape[-2:]
        ce_tgt = cls_logits.new_full((B, Q), self.num_classes, dtype=torch.long)
        # keeps mask_feat_proj AND mask_mlp in the autograd graph even when a
        # whole batch is all-ignore (otherwise DDP reports unused parameters).
        bce = dice = mf.sum() * 0.0 + membed.sum() * 0.0
        n_masks = 0
        for b in range(B):
            tgt_cls, tgt_mask = targets[b]
            if tgt_cls.numel() == 0:
                continue
            with torch.no_grad():
                # matcher: ONE uniform point set shared by all (query,target)
                # pairs, per Mask2Former's HungarianMatcher.
                u = torch.rand(1, self.num_points, 2, device=mf.device,
                               dtype=mf.dtype)
                feat_u = _point_sample(mf[b:b + 1], u)[0]                # (C,P)
                pred_u = membed[b] @ feat_u                              # (Q,P)
                tgt_u = _point_sample(tgt_mask.unsqueeze(0), u)[0]       # (T,P)
                qi, ti = self._match(cls_logits[b], pred_u, tgt_cls, tgt_u)
            ce_tgt[b, qi] = tgt_cls[ti]
            # full-res logits for the MATCHED queries only (M is the number of
            # present classes, ~10-19 — the (B,Q,h,w) tensor is never built).
            pred_m = (membed[b, qi] @ mf[b].flatten(1)).view(-1, h, w)   # (M,h,w)
            tgt_m = tgt_mask[ti]                                         # (M,h,w)
            coords = self._uncertain_coords(pred_m.detach())
            p_pts = _point_sample(pred_m.unsqueeze(1), coords)[:, 0]     # (M,P)
            t_pts = _point_sample(tgt_m.unsqueeze(1), coords)[:, 0]      # (M,P)
            m = qi.numel()
            bce = bce + F.binary_cross_entropy_with_logits(
                p_pts, t_pts, reduction='mean') * m
            dice = dice + _dice_loss(p_pts, t_pts) * m
            n_masks += m
        loss_ce = F.cross_entropy(cls_logits.flatten(0, 1), ce_tgt.flatten(),
                                  weight=self.empty_weight)
        denom = max(n_masks, 1)
        return (self.w_cls * loss_ce + self.w_bce * bce / denom
                + self.w_dice * dice / denom)

    def losses(self, out: Dict, gt_mask: torch.Tensor) -> torch.Tensor:
        """Total mask-cls loss (final + deep-supervised layers), fp32 scalar.

        Targets follow MaskFormer's semantic mode: one binary mask per class
        present in the image (ignore pixels belong to no target mask).
        """
        mf = out['mask_features']
        h, w = mf.shape[-2:]
        with torch.autocast(device_type=mf.device.type, enabled=False):
            with torch.no_grad():
                gt_s4 = F.interpolate(gt_mask.unsqueeze(1).float(), size=(h, w),
                                      mode='nearest').squeeze(1).long()
                targets = []
                for b in range(gt_s4.shape[0]):
                    cs = torch.unique(gt_s4[b])
                    cs = cs[(cs != self.ignore_index) & (cs < self.num_classes)]
                    targets.append(
                        (cs, (gt_s4[b].unsqueeze(0) == cs.view(-1, 1, 1)).float()))
            total = mf.new_zeros((), dtype=torch.float32)
            for cls_l, me_l in out['preds']:
                total = total + self._layer_loss(cls_l, me_l, mf, targets)
            return total / len(out['preds'])

    # ── inference ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def panoptic_inference(self, out: Dict, thing_ids: Sequence[int] = (),
                           obj_thresh: float = 0.8, overlap_thresh: float = 0.8,
                           size: Optional[Tuple[int, int]] = None):
        """Standard Mask2Former panoptic post-processing, per image.

        Returns a list of (panoptic_seg (h,w) int32 segment ids,
        segments_info [{id, category_id, isthing}]). `size` upsamples the mask
        logits first (evaluate PQ at label resolution).
        """
        prob = F.softmax(out['cls'].float(), dim=-1)
        ml = self.mask_logits(out).float()
        if size is not None:
            ml = F.interpolate(ml, size=size, mode='bilinear', align_corners=False)
        masks = ml.sigmoid()
        thing = set(int(t) for t in thing_ids)
        results = []
        for b in range(prob.shape[0]):
            scores, labels = prob[b].max(-1)
            keep = (labels != self.num_classes) & (scores > obj_thresh)
            cur_scores, cur_labels = scores[keep], labels[keep]
            cur_masks = masks[b][keep]                       # (M,h,w)
            h, w = masks.shape[-2:]
            pan = torch.zeros((h, w), dtype=torch.int32, device=masks.device)
            segments: List[Dict] = []
            if cur_masks.shape[0] == 0:
                results.append((pan, segments))
                continue
            mask_ids = (cur_scores.view(-1, 1, 1) * cur_masks).argmax(0)
            seg_id = 0
            stuff_seg: Dict[int, int] = {}
            for k in range(cur_masks.shape[0]):
                cat = int(cur_labels[k])
                orig = cur_masks[k] >= 0.5
                area_orig = int(orig.sum())
                m = (mask_ids == k) & orig
                # the standard M2F overlap numerator is the whole argmax-won
                # region, not its intersection with the >=0.5 mask (using the
                # intersection over-rejects thin/low-confidence segments and
                # biases PQ down) — same audit fix as m2f_head.py 2026-07-21.
                area_win = int((mask_ids == k).sum())
                if int(m.sum()) == 0 or area_orig == 0 \
                        or area_win / area_orig < overlap_thresh:
                    continue
                isthing = cat in thing
                if not isthing and cat in stuff_seg:         # merge stuff
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
