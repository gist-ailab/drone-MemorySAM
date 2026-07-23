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


class PerClassRouter(nn.Module):
    """[P36] Per-class reliability-anchored modality router — port of the P31
    ReliabilityAnchoredRouter mechanism (sam_lola_utils.py), the only large-
    contribution module in the SAM2 lineage. No SAM2 imports.

        w_i = softmax_over_modalities( zero_init_head(feat_i) + λ_anchor · rel_cal_i )

    producing per-class per-pixel weights (m, B, num_classes, h, w). The zero-init
    last conv makes routing purely reliability-driven at start (collapse-safe —
    the documented P10–P27 'gate 상수수렴' fix), then learns per-class ratios
    end-to-end. Motivation: the scalar competence gate anti-selects per class
    (measured night RoadLine: competence img .798 / depth .001, gate depth .432);
    a per-CLASS router lets each class pick the modality that sees it.

    reg_mode 'decisive' (P31): reward = batch-marginal mixing entropy − per-pixel
    mixing entropy → commit locally, stay diverse globally. Returned as a REWARD
    (caller negates into a loss). 'diversity' = per-pixel entropy (P30 legacy).
    Stashes `self._last_w_mean` (per-modality mean weight, (m,) cpu tensor)."""

    def __init__(self, dim: int, num_classes: int, num_modalities: int,
                 anchor_lambda: float = 1.0, hidden: int = 64,
                 reg_mode: str = 'decisive'):
        super().__init__()
        self.m = num_modalities
        self.num_classes = num_classes
        self.anchor_lambda = anchor_lambda
        self.reg_mode = reg_mode
        self._last_w_mean = None
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.ReLU(inplace=True),
                          nn.Conv2d(hidden, num_classes, 1))
            for _ in range(num_modalities)])
        for h in self.heads:            # zero-init last conv → reliability-driven start
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    def forward(self, feats: List[torch.Tensor], rel_cal: torch.Tensor):
        """feats: m x (B, dim, h, w) PRE-fusion features; rel_cal: (m, B, 1, h, w)
        calibrated self-entropy reliability (detached by the caller).
        Returns (w: (m, B, num_classes, h, w), reward: scalar)."""
        logits = torch.stack(
            [self.heads[i](feats[i]).float() for i in range(self.m)], dim=0)
        logits = logits + self.anchor_lambda * rel_cal      # broadcast over classes
        w = F.softmax(logits, dim=0)                        # over modalities
        ent_pix = -(w * (w + 1e-8).log()).sum(dim=0).mean()
        if self.reg_mode == 'decisive':
            w_bar = w.mean(dim=(1, 3, 4))                   # (m, K) marginal
            ent_bar = -(w_bar * (w_bar + 1e-8).log()).sum(dim=0).mean()
            reward = ent_bar - ent_pix
        else:                                               # 'diversity' (legacy)
            reward = ent_pix
        self._last_w_mean = w.detach().float().mean(dim=(1, 2, 3, 4)).cpu()
        return w, reward


class CEFRHead(nn.Module):
    """[P37a] CEFR-Head + CA2-NoText anchor — class-expected fusion routing
    over POST-attention fused tokens (NOT the pre-fusion feats PerClassRouter
    uses), producing a second fused map that is blended with the P36
    gate_fused at feature level (the cheap equivalent of a second head pass):

        A_{i,k}(s)  = λ1·rel_cal_i(s) + λ2(t)·log p̂_i(k|s)     (CA2, no CLIP)
        w_{i,k}(s)  = softmax_over_modalities( head_i(fused_i)_k(s) + A_{i,k}(s) )
        w̄_i(s)     = Σ_k q_k(s)·w_{i,k}(s)                    (class-expected)
        fused'(s)   = Σ_i w̄_i(s)·fused_i(s)
        fused_final = (1−σ(a))·gate_fused + σ(a)·fused'        (a init −4)

    q = pass-1 posterior softmax(logits1), DETACHED and avg-pooled to the
    stride-16 grid (the model computes it — fusion never sees the seg head).
    p̂_i = calibrated per-modal aux posterior (detached, log clamped ≥ −14).
    λ2(t) warms up linearly 0→lambda2_target over lambda2_warmup_ep epochs;
    the epoch arrives via model._current_epoch — the exact mechanism the
    trainer already uses for modal_dropout (train_reliadino.py sets
    `_core._current_epoch = epoch` every epoch, no trainer change needed).

    Anti-collapse (all mandatory, P10–P27 'gate 상수수렴' lineage):
      - zero-init head last conv → anchor-driven routing at start
      - q detached → no gradient into pass-1 logits
      - decisive-entropy reg on w̄ (P31 form, negated reward)
      - AECF-style hinge floor on the BATCH-MARGINAL mixing entropy of w̄
        (penalize only when it drops below entropy_floor — never push uniform)
      - σ(a init −4) ≈ 0.018 → byte-near P36 output at start
    λ1 is a learnable scalar (grads via the softmax); λ2 is scheduled, not
    learned. Stashes `_last_w_mean` ((m,) cpu) and `_last_sigma_a` for the
    trainer's p37/cefr_w_* / p37/cefr_sigma_a logging."""

    def __init__(self, dim: int, num_classes: int, num_modalities: int,
                 hidden: int = 64, morph_init: float = -4.0,
                 anchor_posterior: bool = True, lambda1: float = 1.0,
                 lambda2_target: float = 0.5, lambda2_warmup_ep: int = 10,
                 reg_lambda: float = 0.01, entropy_floor: float = 0.5,
                 hinge_reg: float = 1.0):
        super().__init__()
        self.m = num_modalities
        self.num_classes = num_classes
        self.anchor_posterior = anchor_posterior
        self.lambda2_target = float(lambda2_target)
        self.lambda2_warmup_ep = int(lambda2_warmup_ep)
        self.reg_lambda = reg_lambda
        self.entropy_floor = entropy_floor
        self.hinge_reg = hinge_reg
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.ReLU(inplace=True),
                          nn.Conv2d(hidden, num_classes, 1))
            for _ in range(num_modalities)])
        for h in self.heads:            # zero-init last conv → anchor-driven start
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)
        self.lam1 = nn.Parameter(torch.tensor(float(lambda1)))
        self.a = nn.Parameter(torch.tensor(float(morph_init)))   # blend morph scalar
        self._last_w_mean = None
        self._last_sigma_a = None
        # [analysis] eval 전용: per-class 라우팅 원본 w (m,B,K,h,w) — 분화 분석용
        self._last_cefr_w = None
        self._last_cefr_wbar = None

    def lambda2(self, epoch: int) -> float:
        """λ2(t): linear warmup 0→target over warmup_ep epochs (modal_dropout
        pattern: p * min(1, epoch/warmup_ep))."""
        if not self.anchor_posterior:
            return 0.0
        if self.lambda2_warmup_ep > 0:
            return self.lambda2_target * min(
                1.0, float(epoch) / self.lambda2_warmup_ep)
        return self.lambda2_target

    def forward(self, fused_tokens: List[torch.Tensor], rel_cal: torch.Tensor,
                log_post: Optional[torch.Tensor], q: torch.Tensor, epoch: int):
        """fused_tokens: m x (B, dim, h, w) POST-attention maps;
        rel_cal: (m,B,1,h,w) DETACHED calibrated reliability;
        log_post: (m,B,K,h,w) DETACHED clamped log p̂_i (None if anchor off);
        q: (B,K,h,w) DETACHED pass-1 posterior at stride 16.
        Returns (fused_p (B,dim,h,w), w_bar (m,B,1,h,w), reg loss or None)."""
        m = self.m
        z = torch.stack([self.heads[i](fused_tokens[i]).float()
                         for i in range(m)], dim=0)          # (m,B,K,h,w)
        A = self.lam1 * rel_cal                              # broadcast over K
        lam2 = self.lambda2(epoch)
        if log_post is not None and lam2 > 0:
            A = A + lam2 * log_post
        w = F.softmax(z + A, dim=0)                          # over modalities
        w_bar = (w * q.unsqueeze(0)).sum(dim=2, keepdim=True)  # (m,B,1,h,w)
        if not self.training:
            self._last_cefr_w = w.detach()
            self._last_cefr_wbar = w_bar.detach()
        fused_p = sum(w_bar[i] * fused_tokens[i] for i in range(m))
        self._last_w_mean = w_bar.detach().float().mean(dim=(1, 2, 3, 4)).cpu()
        self._last_sigma_a = float(torch.sigmoid(self.a.detach()))
        reg = None
        if self.training:
            ent_pix = -(w_bar * (w_bar + 1e-8).log()).sum(dim=0).mean()
            marg = w_bar.mean(dim=(1, 2, 3, 4))              # (m,) batch marginal
            ent_bar = -(marg * (marg + 1e-8).log()).sum()
            # decisive (P31): reward = ent_bar − ent_pix → negate into a loss
            reg = self.reg_lambda * (ent_pix - ent_bar)
            # AECF hinge floor: penalize ONLY below-floor batch-marginal entropy
            reg = reg + self.hinge_reg * F.relu(self.entropy_floor - ent_bar)
        return fused_p, w_bar, reg


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
      'router_reg'     [P36] decisive-router regularizer (only if ROUTER on)
    With ROUTER on, aux also carries 'routed_logits' (B,C,h,w) in train AND
    eval — the model consumes (pops) it as a residual on the head output.
    Always stashed for logging: self._last_rel_auroc / _last_rel_stats /
    self._last_gate_mean (per-modality mean gate weight) / _last_router_mean.
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
                 calibrate: bool = True,
                 # [P36] per-class reliability-anchored router (P31 port).
                 # Routed per-class aux logits enter the FINAL prediction as a
                 # residual: final = head_logits + router_alpha · up(routed);
                 # router_alpha zero-init → byte-identical to P35 at start.
                 router_enable: bool = False,
                 router_anchor_lambda: float = 1.0,
                 router_reg_mode: str = 'decisive',
                 router_reg_lambda: float = 0.01,
                 router_alpha_init: float = 0.0,
                 router_hidden: int = 64,
                 # [P37a] CEFR-Head + CA2-NoText anchor. Fusion only HOSTS the
                 # module and exports aux['cefr_ctx']; the model drives the
                 # two-pass flow (pass-1 q → cefr → σ(a) blend → pass 2).
                 # Default OFF → byte-identical P36.
                 cefr_enable: bool = False,
                 cefr_hidden: int = 64,
                 cefr_morph_init: float = -4.0,
                 cefr_anchor_posterior: bool = True,
                 cefr_lambda1: float = 1.0,
                 cefr_lambda2_target: float = 0.5,
                 cefr_lambda2_warmup_ep: int = 10,
                 cefr_reg_lambda: float = 0.01,
                 cefr_entropy_floor: float = 0.5,
                 cefr_hinge_reg: float = 1.0):
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
        self.router_enable = router_enable
        self.router_reg_lambda = router_reg_lambda
        if router_enable:
            self.router = PerClassRouter(
                dim, num_classes, num_modalities,
                anchor_lambda=router_anchor_lambda, hidden=router_hidden,
                reg_mode=router_reg_mode)
            self.router_alpha = nn.Parameter(torch.tensor(float(router_alpha_init)))
        self.cefr_enable = cefr_enable
        if cefr_enable:
            self.cefr = CEFRHead(
                dim, num_classes, num_modalities, hidden=cefr_hidden,
                morph_init=cefr_morph_init,
                anchor_posterior=cefr_anchor_posterior,
                lambda1=cefr_lambda1, lambda2_target=cefr_lambda2_target,
                lambda2_warmup_ep=cefr_lambda2_warmup_ep,
                reg_lambda=cefr_reg_lambda, entropy_floor=cefr_entropy_floor,
                hinge_reg=cefr_hinge_reg)

        self._last_rel_auroc = None
        self._last_rel_stats = None
        self._last_gate_mean = None
        self._last_router_mean = None
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
                gt_mask: Optional[torch.Tensor] = None,
                img_mask: Optional[torch.Tensor] = None,   # [P42-M1/발견C] masked img 샘플 (B,)
                img_idx: int = -1                          # img 모달 인덱스
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

        # 4b) [P36] per-class reliability-anchored router (train AND eval — the
        #     routed residual is part of the prediction). Anchor = detached
        #     rel_cal (training-free signal, P31 convention); heads see the
        #     PRE-fusion feats; routed logits reweight the per-modal aux logits.
        #     The model adds them to the head output scaled by router_alpha.
        if self.router_enable and m >= 2:
            w_route, route_reward = self.router(feats, rel_cal.detach())
            aux['routed_logits'] = sum(
                w_route[i] * aux_logits[i].float() for i in range(m))
            self._last_router_mean = self.router._last_w_mean
            if self.training and self.router_reg_lambda > 0:
                # 'decisive' returns a REWARD → negate into an added loss
                aux['router_reg'] = self.router_reg_lambda * (-route_reward)

        # 4c) [P37a] CEFR context export — the model (which owns FPN+head) runs
        #     pass 1 on `fused` to get q = softmax(logits1).detach() at stride
        #     16, then calls self.cefr and blends via σ(a). All anchor signals
        #     are DETACHED here (training-free, P31 convention); the calibrated
        #     per-modal posterior reuses the aux decoders + temperatures T_i.
        if self.cefr_enable and m >= 2:
            log_post = None
            if self.cefr.anchor_posterior:
                with torch.no_grad():
                    T = self._temps(grad_ok=False)
                    log_post = torch.stack([
                        F.log_softmax(
                            aux_logits[i].float().detach()
                            / (T[i] if T is not None else 1.0), dim=1)
                        for i in range(m)], dim=0).clamp_(min=-14.0)
            aux['cefr_ctx'] = {'fused_tokens': fused_tokens,
                               'rel_cal': rel_cal.detach(),
                               'log_post': log_post}

        # 5) training losses
        if self.training and gt_mask is not None:
            Ht, Wt = max(1, gt_mask.shape[-2] // 4), max(1, gt_mask.shape[-1] // 4)
            gt_ds = F.interpolate(gt_mask.unsqueeze(1).float(), size=(Ht, Wt),
                                  mode='nearest').squeeze(1).long()
            ce = []
            for i, al in enumerate(aux_logits):
                lg = F.interpolate(al.float(), size=(Ht, Wt), mode='bilinear',
                                   align_corners=False)
                if img_mask is not None and i == img_idx:
                    # [발견C] masked img 샘플 제외 — 0-입력에서 GT를 예측하도록 img 브랜치를
                    # 학습시키는 오염(위치기반 장면 prior 환각) 방지. 비마스킹 샘플만 CE.
                    keep = (img_mask < 0.5).nonzero(as_tuple=True)[0]
                    ce.append(F.cross_entropy(lg[keep], gt_ds[keep], ignore_index=255)
                              if keep.numel() > 0 else lg.new_zeros(()))
                else:
                    ce.append(F.cross_entropy(lg, gt_ds, ignore_index=255))
            aux['aux_ce'] = sum(ce) / len(ce)
            if self.calibrate:
                aux['rbma_cal_loss'] = self._calibration_loss(aux_logits, gt_mask)
        return fused, aux
