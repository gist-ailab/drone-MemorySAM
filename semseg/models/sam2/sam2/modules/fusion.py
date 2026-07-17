"""Cross-modal fusion head류 (구 sam_lola_utils.py에서 verbatim 이동)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CrossModalFusionHead', 'SpatialCrossModalFusionHead', 'CrossModalFusionHeadV2', 'ModalAuxHead', 'CrossModalFusionHeadV3']

class CrossModalFusionHead(nn.Module):
    """
    Cross-Modal Fusion Head (P9/P12): 모든 모달리티 feature를 동시에 비교하여
    상대적 융합 가중치를 산출.

    기존 ConfidenceHeadV2는 각 모달리티를 독립 평가(sigmoid → 포화 → 균등화)하지만,
    이 모듈은 모든 모달리티를 동시에 보고 상대 중요도를 비교한다.

    구조: 공유 Compress(GAP+Linear) → Concat → Compare MLP → Softmax

    [P12] cond_dim > 0 이면 Input-Conditioned Scoring 활성화:
    raw input statistics를 추가 경로로 받아 logit에 bias를 더함.
    이를 통해 frozen encoder가 지운 quality 정보를 보충.
    """
    def __init__(self, in_channels, num_modalities=3, hidden_dim=64, temperature=1.0, cond_dim=0):
        super().__init__()
        self.num_modalities = num_modalities
        self.temperature = temperature

        # 각 모달리티 feature를 동일한 compact space로 압축 (가중치 공유)
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU()
        )

        # concat된 feature로 모달리티 간 상대 비교
        self.compare = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities)
        )

        # Zero-init: 초기에 균등 (softmax([0,0,0])=[1/3,1/3,1/3]) → 학습 진행 시 차별화
        nn.init.constant_(self.compare[-1].weight, 0)
        nn.init.constant_(self.compare[-1].bias, 0)

        # [P12] Input condition path: raw stats → logit bias
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_compare = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_modalities)
            )
            # Zero-init: 초기에는 condition 영향 없음
            nn.init.zeros_(self.cond_compare[-1].weight)
            nn.init.zeros_(self.cond_compare[-1].bias)

    def forward(self, features_list, condition=None):
        """
        Args:
            features_list: List of (B, C, H, W) — 각 모달리티의 backbone feature
            condition: (B, cond_dim) or None — [P12] raw input statistics
        Returns:
            weights: (B, num_modalities) softmax 정규화된 가중치
            logits:  (B, num_modalities) raw logits (시각화/디버깅용)
        """
        compressed = [self.compress(f) for f in features_list]
        concat = torch.cat(compressed, dim=1)       # (B, hidden_dim * m)
        logits = self.compare(concat)                # (B, m)

        # [P12] Add condition bias to logits
        if condition is not None and self.cond_dim > 0:
            logits = logits + self.cond_compare(condition)

        weights = F.softmax(logits / self.temperature, dim=1)  # (B, m)
        return weights, logits


class SpatialCrossModalFusionHead(nn.Module):
    """P19: Learned spatial cross-modal fusion from multi-scale FPN features.

    P9 CrossModalFusionHead의 공간정보 소실 문제(GAP→1×1) 해결.
    3개 FPN 레벨에서 multi-scale feature 추출 → 위치별 모달리티 가중치 학습.

    Architecture:
      Phase A: Multi-scale FPN projection (weight-shared across modalities)
        fpn[0](32ch) → Conv1×1→D, fpn[1](64ch) → Conv1×1→D → ×2↑,
        fpn[2](256ch) → Conv1×1→D → ×4↑ → concat → (B, 3D, 256, 256)

      Phase B: Per-modality spatial context extraction (weight-shared)
        DWConv 3×3(3D) → BN → ReLU → Conv1×1(3D→D) → BN → ReLU
        → local spatial context (LiDAR density, Thermal edge, RGB illumination)

      Phase C: Cross-modal spatial comparison
        concat m modalities (B, m*D, H, W)
        → Conv1×1(m*D→hidden) → BN → ReLU
        → DWConv 3×3(hidden) → BN → ReLU  (spatial coherence)
        → Conv1×1(hidden→m) [zero-init]
        → softmax → (B, m, H, W)

    Output: (B, m, H, W) spatial weights at fpn[0] resolution (256×256).
    ~23K params (D=32, hidden=64, m=3).
    """
    def __init__(self, fpn_channels=(32, 64, 256), num_modalities=3,
                 proj_dim=32, hidden_dim=64, temperature=1.0):
        super().__init__()
        self.num_modalities = num_modalities
        self.temperature = temperature

        # Phase A: multi-scale projection (shared across modalities)
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, proj_dim, 1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for ch in fpn_channels
        ])

        fused_dim = proj_dim * len(fpn_channels)  # 96

        # Phase B: per-modality spatial context (shared across modalities)
        # DWConv 3×3: local spatial context (LiDAR point density, Thermal padding boundary)
        # Conv1×1: channel mixing + dimensionality reduction
        self.spatial_context = nn.Sequential(
            nn.Conv2d(fused_dim, fused_dim, 3, padding=1,
                      groups=fused_dim, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_dim, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Phase C: cross-modal comparison with spatial coherence
        compare_in = proj_dim * num_modalities  # 96
        self.compare = nn.Sequential(
            nn.Conv2d(compare_in, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_modalities, 1),
        )

        # Zero-init last Conv → softmax starts uniform [1/m, ..., 1/m]
        nn.init.constant_(self.compare[-1].weight, 0)
        nn.init.constant_(self.compare[-1].bias, 0)

    def forward(self, all_fpn_feats):
        """
        Args:
            all_fpn_feats: list of m lists, each with 3 FPN features
                all_fpn_feats[modal_idx][fpn_level] = (B, C, H, W)
        Returns:
            weights: (B, m, H, W) softmax-normalized spatial fusion weights
            logits:  (B, m, H, W) raw logits before softmax
        """
        m = self.num_modalities
        target_size = all_fpn_feats[0][0].shape[2:]  # fpn[0] resolution

        per_modal_feats = []
        for i in range(m):
            # Phase A: project + upsample + concat
            projected = []
            for j, feat in enumerate(all_fpn_feats[i]):
                p = self.proj_layers[j](feat)
                if p.shape[2:] != target_size:
                    p = F.interpolate(p, size=target_size,
                                      mode='bilinear', align_corners=False)
                projected.append(p)
            fused = torch.cat(projected, dim=1)  # (B, 3*proj_dim, H, W)

            # Phase B: spatial context
            spatial = self.spatial_context(fused)  # (B, proj_dim, H, W)
            per_modal_feats.append(spatial)

        # Phase C: cross-modal comparison
        concat = torch.cat(per_modal_feats, dim=1)  # (B, m*proj_dim, H, W)
        logits = self.compare(concat)  # (B, m, H, W)
        weights = F.softmax(logits / self.temperature, dim=1)
        return weights, logits


class CrossModalFusionHeadV2(nn.Module):
    """
    Cross-Modal Fusion Head V2 (P10): Quality-Aware Adaptive Gating

    P9(V1) 문제:
      - 공유 GAP+Linear가 semantic feature만 압축 → 이미지 품질 정보 소실
      - 모든 장면에서 동일한 압축 → gating이 상수로 수렴 (thermal≈1.0, lidar≈0.96, img≈0.74 고정)

    P10(V2) 개선:
      1. Multi-pool: GAP + GMP + Channel Std → 품질 proxy 정보 포함
         - Std: 텍스처/노이즈 정도 (값이 클수록 정보량 많음)
         - GMP: 가장 강한 활성 (peak signal 강도)
      2. Per-modality 별도 compress: 모달리티별 특성(온도/반사율/색상) 독립 학습
    """

    def __init__(self, in_channels, num_modalities=3, hidden_dim=64, temperature=1.0):
        super().__init__()
        self.num_modalities = num_modalities
        self.temperature = temperature

        # Per-modality 별도 compress: GAP+GMP+Std → (B, C*3) → (B, hidden_dim)
        self.compresses = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels * 3, hidden_dim),
                nn.ReLU()
            )
            for _ in range(num_modalities)
        ])

        # concat된 feature로 모달리티 간 상대 비교 (P9와 동일 구조)
        self.compare = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities)
        )

        # Zero-init: 초기에 균등 (softmax([0,0,0])=[1/3,1/3,1/3])
        nn.init.constant_(self.compare[-1].weight, 0)
        nn.init.constant_(self.compare[-1].bias, 0)

    def _multi_pool(self, x):
        """(B, C, H, W) → (B, C*3): GAP + GMP + Channel Std"""
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, C) — 평균 신호 강도
        gmp = F.adaptive_max_pool2d(x, 1).flatten(1)   # (B, C) — 최대 활성 (salient)
        std = x.flatten(2).std(dim=2)                   # (B, C) — 텍스처/노이즈 정도
        return torch.cat([gap, gmp, std], dim=1)        # (B, C*3)

    def forward(self, features_list):
        """
        Args:
            features_list: List of (B, C, H, W)
        Returns:
            weights: (B, num_modalities) softmax 정규화된 가중치
            logits:  (B, num_modalities) raw logits
        """
        pooled = [self._multi_pool(f) for f in features_list]              # List of (B, C*3)
        compressed = [self.compresses[i](p) for i, p in enumerate(pooled)] # List of (B, hidden)
        concat = torch.cat(compressed, dim=1)                               # (B, hidden*m)
        logits = self.compare(concat)                                       # (B, m)
        weights = F.softmax(logits / self.temperature, dim=1)
        return weights, logits


class ModalAuxHead(nn.Module):
    """
    Per-modality lightweight auxiliary segmentation head (P10).

    목적: 각 모달리티의 독립 예측 품질을 측정 → gating oracle 생성
    구조: backbone_fpn feature → (B, num_classes, H_feat, W_feat)
    학습 신호: per-modality aux loss → oracle weight → KL로 gating 지도
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        mid = max(in_channels // 4, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, 1)
        )

    def forward(self, x):
        return self.head(x)   # (B, num_classes, H_feat, W_feat)


class CrossModalFusionHeadV3(nn.Module):
    """
    Cross-Modal Fusion Head V3 (P11): Dual-Level Quality-Aware Gating

    P10(V2) 문제:
      - backbone feature(deep)에서만 품질을 평가 → SAM2 encoder가 품질 차이를 정규화
      - AMF weights가 near-uniform으로 수렴 (gating 무력화)

    P11(V3) 개선:
      1. Input-level quality (IQE): encoder 이전 raw input에서 직접 품질 측정
      2. Feature-level quality (V2 계승): GAP+GMP+Std multi-pool
      3. Dual-level fusion: input quality + feature quality를 결합하여 gating 결정
         - Input quality가 strong prior 제공 (야간: RGB 어두움 → 낮은 weight)
         - Feature quality가 fine-grained 조정 (encoding 후 실제 feature 품질)

    UAMM 개선: max-normalization → softmax with temperature
      - P10: max modality = 1.0 고정 → 특정 modality 억제 불가
      - P11: softmax → 모든 modality가 독립적으로 조절 가능
    """

    def __init__(self, in_channels, num_modalities=3, hidden_dim=64, temperature=1.0):
        super().__init__()
        self.num_modalities = num_modalities
        self.temperature = temperature

        # Feature-level path (P10 V2 계승): multi-pool → per-modality compress
        self.compresses = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels * 3, hidden_dim),
                nn.ReLU()
            )
            for _ in range(num_modalities)
        ])

        # Dual-level fusion: feature_hidden + input_quality_score → final gating
        # feature path: (B, hidden*m) → (B, hidden)
        # input quality: (B, m) → (B, m) (직접 logit에 더함)
        self.feature_compare = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities)
        )

        # Input quality → gating logit에 대한 residual scale (learnable)
        self.iq_scale = nn.Parameter(torch.tensor(1.0))

        # Zero-init feature path: 초기에는 input quality만 gating 결정
        nn.init.constant_(self.feature_compare[-1].weight, 0)
        nn.init.constant_(self.feature_compare[-1].bias, 0)

    def _multi_pool(self, x):
        """(B, C, H, W) → (B, C*3): GAP + GMP + Channel Std"""
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(x, 1).flatten(1)
        std = x.flatten(2).std(dim=2)
        return torch.cat([gap, gmp, std], dim=1)

    def forward(self, features_list, iq_scores=None):
        """
        Args:
            features_list: List of (B, C, H, W) backbone features
            iq_scores: (B, m) input quality scores from IQE (optional)
        Returns:
            weights: (B, m) softmax gating weights
            logits:  (B, m) raw logits
        """
        # Feature-level path
        pooled = [self._multi_pool(f) for f in features_list]
        compressed = [self.compresses[i](p) for i, p in enumerate(pooled)]
        concat = torch.cat(compressed, dim=1)
        feat_logits = self.feature_compare(concat)  # (B, m)

        # Dual-level fusion
        if iq_scores is not None:
            logits = feat_logits + self.iq_scale * iq_scores
        else:
            logits = feat_logits

        weights = F.softmax(logits / self.temperature, dim=1)
        return weights, logits
