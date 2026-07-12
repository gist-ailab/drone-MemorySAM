"""P34-ReliaDINO fusion — ReliabilityGatedFusion.

Memory-attention generalization (card A): each modality's tokens cross-attend
over the concatenation of the OTHER modalities' tokens (SAM2 memory attention
with "frames" = modalities, freed from the SAM2 dependency), with an RBMA-v2
pre-softmax additive bias, followed by a competence gate on the output fusion.

Signals — empirical constraints honored (P32/P33 analysis, do not "fix" these):
  1. The FUSION GATE uses calibrated self-entropy reliability (rel_cal), NEVER
     corr_veto: corr_veto mis-ranks dead modalities high (it rewards agreeing
     with consensus in easy regions — P33 docstring, doc 24/25).
  2. corroboration/consistency is a SECONDARY additive attention-bias term only
     (λ2·B_cons). P32 lesson: a soft pre-softmax bias alone does not change
     decisions — the gate is where signals must act; the bias is kept as the
     RBMA mechanism heritage + paper ablation axis.
  3. The veto floor (training-free) uses corr_veto only to CAP the gate of
     modalities that are both self-unconfident and consensus-contradicted
     (P33-v2 M3: "training-free hard floor ... 게이트 밖에 유지").

Ported math (kept numerically identical to the validated SAM2-side code):
  - calibrated self-entropy rel  <- LoRA_Sam_P33._stash_comp_rel
  - Bhattacharyya leave-one-out corroboration + veto blend
                                 <- LoRA_Sam_P32._compute_bias_source
  - correctness-contrastive calibration loss (+ per-modal AUROC stash)
                                 <- LoRA_Sam_P31._calibration_loss
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class AuxDecoder(nn.Module):
    """Light per-modality conv decoder at token resolution (stride 16).

    Provides (a) the per-modal posterior for all reliability signals and
    (b) an auxiliary CE path that keeps each modality's branch honest
    (prerequisite for calibration — P31 Seg-A lineage).
    """

    def __init__(self, dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(32, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class CrossModalAttentionLayer(nn.Module):
    """One pre-norm cross-attention + MLP block. Weights are shared across
    modalities (like SAM2 memory attention across frames); modality identity
    enters through the per-modality LoRA'd features and the additive bias."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, kv: torch.Tensor,
                key_bias: Optional[torch.Tensor]) -> torch.Tensor:
        """x: (B, Nq, C) queries; kv: (B, Nk, C); key_bias: (B, Nk) additive
        pre-softmax logit bias, broadcast over heads and queries (RBMA form)."""
        B, Nq, C = x.shape
        h = self.num_heads
        q = self.q(self.norm_q(x)).reshape(B, Nq, h, C // h).transpose(1, 2)
        kvn = self.norm_kv(kv)
        k = self.k(kvn).reshape(B, -1, h, C // h).transpose(1, 2)
        v = self.v(kvn).reshape(B, -1, h, C // h).transpose(1, 2)
        attn_mask = None
        if key_bias is not None:
            # (B, Nk) -> (B, 1, 1, Nk): same bias for every head and query —
            # exactly the RBMA "per-memory-token additive logit bias" shape.
            attn_mask = key_bias[:, None, None, :].to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Nq, C)
        x = x + self.proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class ReliabilityGatedFusion(nn.Module):
    """Cross-modal fusion with RBMA-v2 attention bias + competence output gate.

    forward(feats, gt_mask=None) with feats = list of m (B, C, h, w) maps
    returns (fused (B, C, h, w), aux dict). aux keys (training w/ gt):
      'rbma_cal_loss'  correctness-contrastive calibration loss (P31 port)
      'aux_ce'         mean per-modal auxiliary CE (at 1/4 label res)
      'gate_entropy'   hinge-entropy gate regularizer (0 unless configured)
    Always stashed for logging: self._last_rel_auroc / _last_rel_stats /
    self._last_gate_mean (per-modality mean gate weight).
    """

    RBMA_EPS = 1e-8

    def __init__(self,
                 dim: int,
                 num_classes: int,
                 num_modalities: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 aux_hidden: int = 256,
                 # attention bias (RBMA v2): softmax(QK^T/√d + λ1·B_cal + λ2·B_cons)
                 attn_bias: bool = True,
                 lambda1_init: float = 1.0,
                 consistency_bias: bool = True,
                 lambda2_init: float = 0.5,
                 # output competence gate (P33-v2 M3)
                 gate_enable: bool = True,
                 gate_tau: float = 0.25,
                 gate_entropy_reg: float = 0.0,
                 gate_entropy_floor: float = 0.5,
                 veto_floor: bool = True,
                 veto_thresh: float = 0.10,
                 veto_cap: float = 0.05,
                 # calibration (P31/P33 M3)
                 calibrate: bool = True):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.attn_bias = attn_bias
        self.consistency_bias = consistency_bias
        self.gate_enable = gate_enable
        self.gate_tau = gate_tau
        self.gate_entropy_reg = gate_entropy_reg
        self.gate_entropy_floor = gate_entropy_floor
        self.veto_floor = veto_floor
        self.veto_thresh = veto_thresh
        self.veto_cap = veto_cap
        self.calibrate = calibrate

        self.layers = nn.ModuleList(
            CrossModalAttentionLayer(dim, num_heads, mlp_ratio) for _ in range(num_layers))
        self.aux_decoders = nn.ModuleList(
            AuxDecoder(dim, num_classes, aux_hidden) for _ in range(num_modalities))
        if attn_bias:
            self.lambda1 = nn.Parameter(torch.tensor(float(lambda1_init)))
            if consistency_bias:
                self.lambda2 = nn.Parameter(torch.tensor(float(lambda2_init)))
        if calibrate:
            self.rbma_log_temp = nn.Parameter(torch.zeros(num_modalities))

        self._last_rel_auroc = None
        self._last_rel_stats = None
        self._last_gate_mean = None
        # [analysis] eval 전용 스태시 (tools/seg_analysis 계열이 소비; 학습 영향 0)
        self._last_aux_logits = None      # m x (B,C,h,w) per-modal aux decoder logits
        self._last_rel_spatial = None     # (m,B,1,h,w) calibrated reliability
        self._last_gate_spatial = None    # (m,B,1,h,w) competence gate weights

    # ── signals ──────────────────────────────────────────────────────────────
    def _temps(self, grad_ok: bool) -> Optional[torch.Tensor]:
        if not self.calibrate:
            return None
        T = self.rbma_log_temp.exp().clamp(min=0.05, max=20.0)
        return T if grad_ok else T.detach()

    def _compute_signals(self, aux_logits: List[torch.Tensor]):
        """From per-modal aux logits (B, C, h, w) compute, all at token res:
          rel_cal   (m,B,1,h,w)  calibrated self-entropy reliability (gate + B_cal)
          corr_veto (m,B,1,h,w)  P32 veto-blended corroboration      (veto floor)
          b_cons    (m,B,1,h,w)  centered leave-one-out Bhattacharyya (B_cons)
        Math is the P32._compute_bias_source / P33._stash_comp_rel port; grads
        flow only through T (and only when gate_entropy_reg > 0, P33 convention).
        """
        m = len(aux_logits)
        grad_T = bool(self.training and self.gate_entropy_reg > 0)
        T = self._temps(grad_ok=grad_T)
        C = aux_logits[0].shape[1]
        logC = math.log(C)

        probs, rel_cal, selfent = [], [], []
        for i in range(m):
            lg = aux_logits[i].float().detach()          # training-free signal
            lg_cal = lg / T[i] if T is not None else lg
            p_cal = F.softmax(lg_cal, dim=1)
            ent = -(p_cal * torch.log(p_cal + self.RBMA_EPS)).sum(dim=1, keepdim=True)
            rel_cal.append(1.0 - ent / logC)             # calibrated self-entropy
            with torch.no_grad():
                # corroboration path is temperature-free — matches the validated
                # P32 diagnostic exactly (tools/eval_reliability_auroc.py).
                p = F.softmax(lg, dim=1)
                p = p / p.sum(dim=1, keepdim=True).clamp(min=EPS)
                e = -(p * torch.log(p + self.RBMA_EPS)).sum(dim=1, keepdim=True)
                probs.append(p)
                selfent.append(1.0 - e / logC)
        rel_cal = torch.stack(rel_cal, dim=0)            # (m,B,1,h,w)

        with torch.no_grad():
            p_sum = torch.stack(probs, dim=0).sum(dim=0)
            corr_list, veto_list = [], []
            for i in range(m):
                cons = ((p_sum - probs[i]) / max(m - 1, 1)).clamp_min(0)
                corr = (probs[i] * cons).clamp_min(0).sqrt().sum(dim=1, keepdim=True)
                others_max = torch.stack(
                    [selfent[j] for j in range(m) if j != i], dim=0).amax(dim=0)
                g = (selfent[i] - others_max).clamp(0, 1)      # unique-confidence veto gate
                veto_list.append((g * selfent[i] + (1.0 - g) * corr).clamp(0, 1))
                corr_list.append(corr)
            corr_veto = torch.stack(veto_list, dim=0)          # (m,B,1,h,w)
            corr_stack = torch.stack(corr_list, dim=0)
            b_cons = corr_stack - corr_stack.mean(dim=0, keepdim=True)
        return rel_cal, corr_veto, b_cons

    # ── gate (P33-v2 M3: soft + hinge-entropy + training-free veto floor) ────
    def _gate(self, rel_cal: torch.Tensor, corr_veto: torch.Tensor):
        m = rel_cal.shape[0]
        w = F.softmax(rel_cal / self.gate_tau, dim=0)          # (m,B,1,h,w)
        if self.veto_floor:
            with torch.no_grad():
                low = (corr_veto < self.veto_thresh).float()   # extreme-low corr_veto
                cap = low * self.veto_cap + (1.0 - low)        # cap only flagged modals
            w = torch.minimum(w, cap)
            w = w / w.sum(dim=0, keepdim=True).clamp(min=EPS)
        gate_ent = None
        if self.training and self.gate_entropy_reg > 0:
            # hinge: penalize only when mixing entropy drops BELOW the floor
            # (AECF collapse warning — do not push toward uniform, only prevent
            # single-modality collapse). Grads reach T via rel_cal.
            H = -(w * (w + 1e-8).log()).sum(dim=0).mean()
            gate_ent = self.gate_entropy_reg * F.relu(self.gate_entropy_floor - H)
        self._last_gate_mean = w.detach().mean(dim=(1, 2, 3, 4))
        if not self.training:
            self._last_gate_spatial = w.detach()
        return w, gate_ent

    # ── calibration loss: exact P31._calibration_loss port ──────────────────
    def _calibration_loss(self, aux_logits_list, gt_mask):
        gt = gt_mask.long()
        Ht, Wt = max(1, gt.shape[-2] // 4), max(1, gt.shape[-1] // 4)
        gt_ds = F.interpolate(gt.unsqueeze(1).float(), size=(Ht, Wt),
                              mode='nearest').squeeze(1).long()
        valid = gt_ds != 255
        T = self.rbma_log_temp.exp().clamp(min=0.05, max=20.0)
        total = None
        aurocs, rel_mu, rel_sd = [], [], []
        for i, logits in enumerate(aux_logits_list):
            C = logits.shape[1]
            lg = F.interpolate(logits.float(), size=(Ht, Wt),
                               mode='bilinear', align_corners=False) / T[i]
            p = F.softmax(lg, dim=1)
            ent = -(p * (p + 1e-8).log()).sum(dim=1) / math.log(C)
            pred = lg.argmax(dim=1)
            wrong = (pred != gt_ds) & valid
            correct = (pred == gt_ds) & valid
            l_wrong = ((1.0 - ent) * wrong.float()).sum() / wrong.float().sum().clamp(min=1.0)
            l_correct = (ent * correct.float()).sum() / correct.float().sum().clamp(min=1.0)
            li = l_wrong + l_correct
            total = li if total is None else total + li
            with torch.no_grad():
                score = (1.0 - ent).detach()[valid].float()
                lab = correct[valid]
                n1 = lab.sum().float(); n0 = lab.numel() - lab.sum().float()
                if n1 > 0 and n0 > 0:
                    order = score.argsort()
                    ranks = torch.zeros_like(score)
                    ranks[order] = torch.arange(1, score.numel() + 1,
                                                device=score.device, dtype=score.dtype)
                    auroc = ((ranks[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()
                else:
                    auroc = float('nan')
                aurocs.append(auroc)
                rel_mu.append(score.mean().item())
                rel_sd.append(score.std().item())
        self._last_rel_auroc = aurocs
        self._last_rel_stats = (rel_mu, rel_sd)
        return total / len(aux_logits_list)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, feats: List[torch.Tensor],
                gt_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, dict]:
        m = len(feats)
        assert m == self.num_modalities, f"got {m} modalities, expected {self.num_modalities}"
        B, C, h, w = feats[0].shape
        aux: dict = {}

        # 1) per-modal aux decode (grad-attached: trains decoders + T)
        aux_logits = [self.aux_decoders[i](feats[i]) for i in range(m)]

        # 2) reliability signals
        rel_cal, corr_veto, b_cons = self._compute_signals(aux_logits)
        b_cal = rel_cal.detach() - rel_cal.detach().mean(dim=0, keepdim=True)
        if not self.training:
            self._last_aux_logits = [al.detach() for al in aux_logits]
            self._last_rel_spatial = rel_cal.detach()

        # 3) cross-modal attention with per-key additive bias.
        #    Key set for modality i = concat of the other modalities' tokens;
        #    a key token at location s of modality j carries bias
        #    λ1·B_cal_j(s) + λ2·B_cons_j(s)  — added pre-softmax (RBMA form).
        tokens = [f.flatten(2).transpose(1, 2) for f in feats]      # m x (B, N, C)
        if self.attn_bias:
            bias_maps = self.lambda1 * b_cal                        # grads reach λ1
            if self.consistency_bias:
                bias_maps = bias_maps + self.lambda2 * b_cons       # secondary term
            bias_flat = [bias_maps[j].flatten(1) for j in range(m)]  # m x (B, N)
        fused_tokens = []
        for i in range(m):
            kv = torch.cat([tokens[j] for j in range(m) if j != i], dim=1)
            key_bias = None
            if self.attn_bias:
                key_bias = torch.cat([bias_flat[j] for j in range(m) if j != i], dim=1)
            x = tokens[i]
            for layer in self.layers:
                x = layer(x, kv, key_bias)
            fused_tokens.append(x.transpose(1, 2).reshape(B, C, h, w))

        # 4) output fusion: competence gate (calibrated self-entropy, veto floor)
        if self.gate_enable and m >= 2:
            wgt, gate_ent = self._gate(rel_cal, corr_veto)
            fused = sum(wgt[i] * fused_tokens[i] for i in range(m))
            if gate_ent is not None:
                aux['gate_entropy'] = gate_ent
        else:
            fused = sum(fused_tokens) / m
            self._last_gate_mean = None

        # 5) training losses
        if self.training and gt_mask is not None:
            Ht, Wt = max(1, gt_mask.shape[-2] // 4), max(1, gt_mask.shape[-1] // 4)
            gt_ds = F.interpolate(gt_mask.unsqueeze(1).float(), size=(Ht, Wt),
                                  mode='nearest').squeeze(1).long()
            ce = [F.cross_entropy(
                F.interpolate(al.float(), size=(Ht, Wt), mode='bilinear',
                              align_corners=False),
                gt_ds, ignore_index=255) for al in aux_logits]
            aux['aux_ce'] = sum(ce) / len(ce)
            if self.calibrate:
                aux['rbma_cal_loss'] = self._calibration_loss(aux_logits, gt_mask)
        return fused, aux
