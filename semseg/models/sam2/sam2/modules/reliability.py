"""Reliability/confidence/quality 모듈류 (구 sam_lola_utils.py에서 verbatim 이동)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ConfidenceHeadV2', 'InputQualityEstimator', 'ConfidenceHead', 'SelfDerivedCondition', 'ReliabilityAnchoredRouter', 'SpatialQualityGating']

class ConfidenceHeadV2(nn.Module):
    """
    A deeper Confidence Head using CNN layers to capture spatial features
    before global pooling. This helps in detecting local noise (like rain drops)
    better than a simple MLP on avg-pooled features.
    """
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        return self.net(x)


class InputQualityEstimator(nn.Module):
    """
    Input-level Modality Quality Estimator (P11).

    핵심 인사이트:
      SAM2의 강력한 pretrained encoder는 feature space에서 modality 품질 차이를
      정규화해버려, feature-level quality assessment (P10 ModalAuxHead)가
      near-uniform oracle을 생성한다.

    해결:
      Encoder 이전의 raw input에서 직접 품질을 추정하는 lightweight CNN.
      Per-modality 독립 네트워크로, 각 센서 타입(RGB/Thermal/LiDAR)이
      자신만의 quality metric을 학습한다.

      - RGB: 야간 brightness 저하, noise 증가 등을 학습
      - Thermal: thermal contrast 패턴, saturation 등을 학습
      - LiDAR: point density, spatial coverage 등을 학습

    구조 (per modality, ~15K params):
      Conv(3→16, 7×7, stride=8) → ReLU → Conv(16→32, 5×5, stride=4)
      → ReLU → GAP → Linear(32→1)

    설계 원칙:
      - Fully learnable: handcrafted feature 없이 end-to-end 학습
      - Per-modality: 센서별 독립 네트워크 (RGB/Thermal/LiDAR 특성 차이 반영)
      - Lightweight: aggressive stride로 1024→128→32→1 빠른 축소
      - Zero-init output: 학습 초기 uniform weight 보장
    """

    def __init__(self, num_modalities=3, in_channels=3, hidden_dim=32):
        super().__init__()

        # Per-modality lightweight CNN: raw input → quality score
        self.quality_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 16, 7, stride=8, padding=3),   # 1024→128
                nn.ReLU(),
                nn.Conv2d(16, hidden_dim, 5, stride=4, padding=2),    # 128→32
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),                              # 32→1
                nn.Flatten(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_modalities)
        ])

        # Zero-init output layer: 초기에 모든 quality score = 0 → uniform gating
        for net in self.quality_nets:
            nn.init.constant_(net[-1].weight, 0)
            nn.init.constant_(net[-1].bias, 0)

    def forward(self, raw_inputs):
        """
        Args:
            raw_inputs: List of (B, C, H, W) — 각 modality의 raw input
        Returns:
            quality_scores: (B, num_modalities) — per-modality quality score (logit)
        """
        scores = []
        for i, x in enumerate(raw_inputs):
            score = self.quality_nets[i](x)  # (B, 1)
            scores.append(score)
        return torch.cat(scores, dim=1)  # (B, m)


class ConfidenceHead(nn.Module):
    """
    Lightweight module to estimate the confidence of a modality based on its features.
    Architecture: GlobalAvgPool -> MLP -> Linear (Logits for Softmax)
    """
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class SelfDerivedCondition(nn.Module):
    """[P29] Self-Derived Condition (SDC): label-free, image-derived condition latent.

    early feature map → (channel-wise spatial mean ⊕ std) descriptor → MLP → z
    → cosine soft-assign to a learned **prototype bank** (K) → prototype-refined latent z_c.
    Returns (z_c, clustering_loss). No labels / no text / no extra sensor:
    prototypes are discovered by an entropy clustering objective
    (confident per-sample assignment + diverse batch usage). This is the differentiator
    vs CAFuser (CLIP-text condition) / DGFusion (depth + depth-GT reliability).

    Args:
        in_channels: channels of the early feature fed in.
        latent_dim:  condition latent dim z_c (concatenated with modal_embed for the gate).
        K:           number of condition prototypes.
        tau:         softmax temperature for cosine assignment.
    """
    def __init__(self, in_channels, latent_dim=32, K=6, tau=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.K = K
        self.tau = tau
        self.proj = nn.Sequential(
            nn.Linear(2 * in_channels, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(K, latent_dim) * 0.02)

    def forward(self, feat):
        """feat: (B, C, H, W) early feature map. Returns (z_c: (B, latent_dim), loss: scalar)."""
        if feat.dim() == 4:
            mu = feat.mean(dim=(2, 3))                 # (B, C)
            sd = feat.var(dim=(2, 3), unbiased=False).clamp_min(1e-6).sqrt()
        elif feat.dim() == 3:                          # (B, N, C)
            mu = feat.mean(dim=1)
            sd = feat.var(dim=1, unbiased=False).clamp_min(1e-6).sqrt()
        else:
            raise ValueError(f"SDC expects 4D/3D feature, got {feat.shape}")
        desc = torch.cat([mu, sd], dim=-1)             # (B, 2C)
        z = self.proj(desc)                            # (B, latent_dim)
        zc = F.normalize(z, dim=-1)
        pc = F.normalize(self.prototypes, dim=-1)
        sim = zc @ pc.t()                              # (B, K)
        a = F.softmax(sim / self.tau, dim=-1)          # (B, K) soft assignment
        z_refined = a @ self.prototypes                # (B, latent_dim) prototype-refined

        # label-free clustering loss: confident per-sample + diverse batch usage
        eps = 1e-8
        sample_ent = -(a * (a + eps).log()).sum(dim=-1).mean()    # ↓ confident
        batch_mean = a.mean(dim=0)                                # (K,)
        batch_ent = -(batch_mean * (batch_mean + eps).log()).sum()  # ↑ diverse
        loss = sample_ent - batch_ent
        return z_refined, loss


class ReliabilityAnchoredRouter(nn.Module):
    """[P30] Learned modality-fusion router, ANCHORED by RBMA reliability so it cannot
    collapse to a constant ratio (the documented P10–P27 'gate 상수수렴', ISSUE-002/015).

        w = softmax_modality( learned_logits(feat_i) + anchor_lambda * reliability_i )

    The learned conv head is zero-init → at start w is purely reliability-driven (no
    collapse), then learns the ratios end-to-end. per_class=True → per-class weights
    (K=num_classes) so a class can route to the modality that sees it (revive event/LiDAR);
    else scalar (K=1). Returns (w: (m,B,K,h,w), reg: reward term the trainer SUBTRACTS).

    reg_mode (P31): the measured P28/P29 fusion weights were near-uniform
    ([.27,.28,.23,.23] vs true drop-Δ contribution [8.4,23.5,0.02,0.01] — doc 16 §7,
    "router does not adaptively select modalities"), and the original 'diversity' reward
    (maximize per-pixel mixing entropy) actively pushes TOWARD uniform.
      - 'diversity' (P30 default, unchanged): reg = per-pixel modality-mixing entropy.
      - 'decisive' (P31): reg = batch-marginal entropy − per-pixel entropy → the router is
        rewarded for committing per pixel/class (low local entropy = adaptive selection)
        while keeping all modalities used on average (high marginal entropy = no global
        single-modality collapse). Same confident+diverse pairing as the SDC loss.
    Stashes `self._last_w_mean` (per-modality mean weight, (m,)) for monitoring."""

    def __init__(self, in_ch, num_modalities, num_classes=1, per_class=False,
                 anchor_lambda=1.0, hidden=64, reg_mode='diversity'):
        super().__init__()
        self.m = num_modalities
        self.per_class = per_class
        self.K = num_classes if per_class else 1
        self.anchor_lambda = anchor_lambda
        self.reg_mode = reg_mode
        self._last_w_mean = None
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, hidden, 1), nn.ReLU(inplace=True),
                          nn.Conv2d(hidden, self.K, 1))
            for _ in range(num_modalities)])
        for h in self.heads:                 # zero-init last conv → start reliability-driven
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    def forward(self, feats, reliability=None):
        # feats: list of m (B, in_ch, h, w); reliability: (m, B, 1, h, w) or None
        logits = torch.stack([self.heads[i](feats[i]) for i in range(self.m)], dim=0)  # (m,B,K,h,w)
        if reliability is not None:
            rb = reliability if reliability.shape[2] == self.K else reliability.expand(-1, -1, self.K, -1, -1)
            logits = logits + self.anchor_lambda * rb
        w = F.softmax(logits, dim=0)                         # over modality
        ent_pix = -(w * (w + 1e-8).log()).sum(dim=0).mean()  # per-pixel modality-mixing entropy
        if self.reg_mode == 'decisive':
            # [P31] reward = marginal entropy − pixel entropy: commit locally, stay
            # diverse globally (fixes the near-uniform routing measured in doc 16 §7).
            w_bar = w.mean(dim=(1, 3, 4))                    # (m, K) batch+space marginal
            ent_bar = -(w_bar * (w_bar + 1e-8).log()).sum(dim=0).mean()
            reg = ent_bar - ent_pix
        else:
            reg = ent_pix                                    # 'diversity' (P30 default)
        self._last_w_mean = w.detach().float().mean(dim=(1, 2, 3, 4)).cpu()  # (m,)
        return w, reg


class SpatialQualityGating(nn.Module):
    """
    P24: Spatial Quality Gating Network.

    Predicts a per-pixel quality map from encoded modality features.
    During training, supervised by teacher quality maps derived from
    per-modality SAM2 decoder CE loss against GT: target = exp(-BCE).
    During inference, runs standalone (no teacher needed).

    Input:  backbone FPN[0] features (B, C, H, W) — lowest-resolution level
    Output (forward): raw logits (B, 1, H, W) — use logits_to_quality() for [min_quality, 1.0]

    Loss: F.binary_cross_entropy_with_logits(logits, target)
    Design: Lightweight conv head (no cross-attention — keeps it simple and spatial).
    """

    def __init__(self, in_channels: int = 256, hidden_dim: int = 64,
                 min_quality: float = 0.1):
        """
        Args:
            in_channels: input feature channels (FPN[0] dim, typically 256)
            hidden_dim: intermediate conv channels
            min_quality: minimum quality value to prevent total memory zeroing
        """
        super().__init__()
        self.min_quality = min_quality

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Last conv bias init to +1.0 so sigmoid starts near 0.73
        # → quality starts high (optimistic), learns to lower for bad regions
        self.head[-1].bias.data.fill_(1.0)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, C, H, W) backbone feature map
        Returns:
            raw logits: (B, 1, H, W) — NOT sigmoid-ed.
            Use logits_to_quality() to convert to [min_quality, 1.0] for inference/memory modulation.
        """
        return self.head(feat)                         # (B, 1, H, W) raw logits

    def logits_to_quality(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to quality map in [min_quality, 1.0]."""
        quality = torch.sigmoid(logits)
        return quality * (1.0 - self.min_quality) + self.min_quality
