"""
LoRA/LoLA 공통 유틸: 실험 모델(LoRA_Sam_P1~P9)이 아닌 보조 클래스·함수.
- MLP_my, _LoRA_qkv, ConfidenceHead, ConfidenceHeadV2, CrossModalFusionHead
- MoE_LoRA_Layer, _MoE_LoRA_qkv
- MoE_DeBA_BB, _MoE_DeBA_BB_qkv (P23)
- random_element_swap
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_my(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


def random_element_swap(tensor_list):
    if len(tensor_list) != 2:
        raise ValueError("列表必须包含两个张量")

    tensor1, tensor2 = tensor_list

    if tensor1.size() != tensor2.size():
        raise ValueError("两个张量的大小必须相同")

    swap_mask = torch.rand(tensor1.size()) > 0.5
    temp = tensor1.clone()
    tensor1[swap_mask] = tensor2[swap_mask]
    tensor2[swap_mask] = temp[swap_mask]

    return [tensor1, tensor2]


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


class MoE_LoRA_Layer(nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) LoRA layer.
    Instead of one pair of A/B matrices, it holds 'num_experts' pairs.
    A gating network selects top-k experts for each token.
    """
    def __init__(self, in_features, rank, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank
        self.in_features = in_features

        self.gate = nn.Linear(in_features, num_experts)
        self.experts_a = nn.ModuleList([
            nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)
        ])
        self.experts_b = nn.ModuleList([
            nn.Linear(rank, in_features, bias=False) for _ in range(num_experts)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.experts_a[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_b[i].weight)

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)

        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        weights, indices = torch.topk(gate_probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        final_output = torch.zeros_like(x_flat)
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(1, indices, 1.0)
        masked_probs = gate_probs * mask
        masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

        for i in range(self.num_experts):
            expert_weight = masked_probs[:, i].unsqueeze(-1)
            if expert_weight.sum() == 0:
                continue
            expert_out = self.experts_b[i](self.experts_a[i](x_flat))
            final_output += expert_weight * expert_out

        return final_output.view(original_shape)


class _MoE_LoRA_qkv(nn.Module):
    """
    QKV layer wrapper that replaces standard LoRA with MoE-LoRA.
    """
    def __init__(
            self,
            qkv: nn.Module,
            moe_layer_q: MoE_LoRA_Layer,
            moe_layer_v: MoE_LoRA_Layer,
    ):
        super().__init__()
        self.qkv = qkv
        self.moe_layer_q = moe_layer_q
        self.moe_layer_v = moe_layer_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.moe_layer_q(x)
        new_v = self.moe_layer_v(x)
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
        
class SoftMoE_LoRA_Layer(nn.Module):
    """
    Soft-MoE (Soft Mixture-of-Experts) LoRA Layer.

    기존의 Top-k 방식(Hard Routing) 대신, 입력에 따라 모든 전문가(Expert)의 출력을
    Softmax 가중치로 섞어서 사용합니다.

    장점:
    1. 전문가 수가 적을 때(예: 4개) Top-k보다 정보 손실이 적음.
    2. 모든 전문가에게 Gradient가 흘러 Dead Expert 문제가 완화됨.
    3. 미분 가능성이 완벽하게 보장됨.

    [P12] cond_dim > 0 이면 Input-Conditioned Gating 활성화:
    외부에서 set_condition()으로 설정한 (B, cond_dim) 벡터를 gate logit에 bias로 더함.
    Zero-init → 초기에는 기존과 동일하게 동작, 학습 진행 시 condition 활용.
    """
    def __init__(self, in_features, rank, num_experts=4, cond_dim=0):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank
        self.in_features = in_features

        # Gating Network: 입력 토큰별로 전문가 가중치를 계산
        self.gate = nn.Linear(in_features, num_experts)

        # [P12] Input condition → gate bias projection
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, num_experts)
        self._condition = None  # set externally via set_condition()

        # Experts: LoRA 어댑터들
        self.experts_a = nn.ModuleList([
            nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)
        ])
        self.experts_b = nn.ModuleList([
            nn.Linear(rank, in_features, bias=False) for _ in range(num_experts)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        # Gate 초기화: 초기에는 모든 전문가를 균등하게 사용하도록 0 근처로 설정
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)

        # [P12] Condition projection zero-init: 초기에는 condition 영향 없음
        if self.cond_dim > 0:
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)

        # Expert 초기화
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.experts_a[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_b[i].weight)

    def set_condition(self, cond):
        """Set input condition for gating. cond: (B, cond_dim) or None."""
        self._condition = cond

    def forward(self, x):
        # x shape: (B, N, C) 또는 (B, H, W, C) — Hiera 백본은 4D (B, H, W, C) 사용
        gate_logits = self.gate(x)  # (..., num_experts)

        # [P12] Add condition bias to gate logits
        if self._condition is not None and self.cond_dim > 0:
            cond = self._condition  # (B, cond_dim)
            B = cond.shape[0]
            x_B = x.shape[0]  # could be B * num_windows (windowed attention)

            if x_B > B:
                # Windowed attention: expand condition to match B*nw
                nw = x_B // B
                cond = cond.repeat_interleave(nw, dim=0)  # (B*nw, cond_dim)

            cond_bias = self.cond_proj(cond)  # (x_B, num_experts)
            # Broadcast over spatial dims: add 1-dims for H, W (or N)
            for _ in range(x.dim() - 2):
                cond_bias = cond_bias.unsqueeze(1)
            gate_logits = gate_logits + cond_bias

        gate_weights = F.softmax(gate_logits, dim=-1)  # (..., num_experts)

        # For visualization: store spatial-mean gate weights (B, num_experts)
        if hasattr(self, '_gate_callback') and self._gate_callback is not None:
            gw_mean = gate_weights.mean(dim=tuple(range(gate_weights.dim()-1))).detach().cpu().numpy()
            self._gate_callback(gw_mean)

        # [P11] MI loss용: gradient 유지한 채 spatial mean gate distribution 수집
        if hasattr(self, '_grad_gate_collector') and self._grad_gate_collector is not None:
            # Hiera windowed attention에서 batch dim에 num_windows가 곱해질 수 있음
            # → 모든 공간/윈도우 차원을 평균하여 (E,) 스칼라 벡터로 통일
            gate_mean = gate_weights.mean(dim=tuple(range(gate_weights.dim() - 1)))  # (E,)
            self._grad_gate_collector.append(gate_mean)

        final_output = 0
        for i in range(self.num_experts):
            expert_out = self.experts_b[i](self.experts_a[i](x))
            # Expert 인덱스는 항상 마지막 차원: gate_weights[..., i] -> (..., 1)
            weight = gate_weights[..., i].unsqueeze(-1)
            final_output = final_output + weight * expert_out

        return final_output


class SharedGateMLP(nn.Module):
    """
    Shared 2-layer MLP gate for SoftMoE routing (P20).

    같은 차원의 모든 MoE 레이어(Q/V, 여러 블록)가 하나의 gate를 공유하여
    파라미터를 절약하고, 안정적인 routing 학습을 도모.

    구조: Linear(in_features → hidden) → ReLU → Linear(hidden → num_experts)
    hidden_ratio=4 → hidden = in_features // 4

    Ref: LD-MoLE (shared gate 방식), DynMoLE (MLP gate)
    """
    def __init__(self, in_features, num_experts, hidden_ratio=4):
        super().__init__()
        hidden = max(in_features // hidden_ratio, 16)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts),
        )
        self.reset_parameters()

    def reset_parameters(self):
        # First layer: kaiming for ReLU
        nn.init.kaiming_uniform_(self.net[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.net[0].bias)
        # Last layer: near-zero → initial softmax ≈ uniform (1/E)
        nn.init.normal_(self.net[2].weight, std=0.01)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x)


class SoftMoE_LoRA_Layer_V2(nn.Module):
    """
    SoftMoE LoRA Layer V2 — per-layer independent 2-layer MLP gate (P20).

    V1(SoftMoE_LoRA_Layer)과의 차이:
    - gate가 2-layer MLP: Linear(C→C//gate_hidden_ratio) → ReLU → Linear(→num_experts)
    - 각 레이어가 독립적인 MLP gate를 보유 (공유 없음)
    - gate 파라미터가 state_dict에 자동 포함

    V2 초기 버전은 SharedGateMLP을 외부에서 공유했으나, 38개 레이어가 1개 gate를
    공유하면 uniform compromise에 빠지는 문제가 확인되어 per-layer 독립으로 변경.
    """
    def __init__(self, in_features, rank, num_experts=4, gate_hidden_ratio=4):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank
        self.in_features = in_features

        # Per-layer independent 2-layer MLP gate
        hidden = max(in_features // gate_hidden_ratio, 16)
        self.gate = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts),
        )

        # Experts: LoRA adapters
        self.experts_a = nn.ModuleList([
            nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)
        ])
        self.experts_b = nn.ModuleList([
            nn.Linear(rank, in_features, bias=False) for _ in range(num_experts)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        # Gate: kaiming for ReLU first layer, near-zero last layer (initial softmax ≈ uniform)
        nn.init.kaiming_uniform_(self.gate[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.gate[0].bias)
        nn.init.normal_(self.gate[2].weight, std=0.01)
        nn.init.zeros_(self.gate[2].bias)
        # Experts
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.experts_a[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_b[i].weight)

    def forward(self, x):
        gate_logits = self.gate(x)  # (..., num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (..., num_experts)

        # For visualization: store spatial-mean gate weights
        if hasattr(self, '_gate_callback') and self._gate_callback is not None:
            gw_mean = gate_weights.mean(dim=tuple(range(gate_weights.dim()-1))).detach().cpu().numpy()
            self._gate_callback(gw_mean)

        final_output = 0
        for i in range(self.num_experts):
            expert_out = self.experts_b[i](self.experts_a[i](x))
            weight = gate_weights[..., i].unsqueeze(-1)
            final_output = final_output + weight * expert_out

        return final_output


class _SoftMoE_LoRA_qkv(nn.Module):
    """
    QKV layer wrapper that replaces standard LoRA with Soft-MoE LoRA.
    """
    def __init__(
            self,
            qkv: nn.Module,
            moe_layer_q: SoftMoE_LoRA_Layer,
            moe_layer_v: SoftMoE_LoRA_Layer,
    ):
        super().__init__()
        self.qkv = qkv
        self.moe_layer_q = moe_layer_q
        self.moe_layer_v = moe_layer_v
        self.dim = qkv.in_features
        
    def forward(self, x):
        # Original QKV
        qkv = self.qkv(x)
        
        # Soft-MoE LoRA Update for Query
        new_q = self.moe_layer_q(x)
        
        # Soft-MoE LoRA Update for Value
        new_v = self.moe_layer_v(x)
        
        # Add residual connection
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v

        return qkv


# ─────────────────────────────────────────────────────────────────────
# DeBA-FP: Deformable Bottleneck Adapter for Feature Pyramid (P21)
# Ref: "Rethinking Deformable Convolution as an Adapter with Cross-layer
#       Weight Sharing for Robust Semantic Segmentation in the Wild" (CVPR 2026)
# ─────────────────────────────────────────────────────────────────────


class DeBAFP(nn.Module):
    """
    Deformable Bottleneck Adapter for Feature Pyramid (DeBA-FP).

    Applies a shared deformable convolution bottleneck to FPN features.
    Cross-modal weight sharing: DCM, norm, W_d, W_u are shared across modalities.
    Only per-modality learnable scaling factor α is independent.

    Structure per application:
        feat' = feat + α_m * W_u(GELU(LN(DCM(W_d(feat)))))

    where DCM = offset_prediction → deform_conv2d (DCNv2).
    """

    def __init__(
        self,
        in_channels: int = 256,
        bottleneck_dim: int = 64,
        kernel_size: int = 3,
        num_modalities: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # ── Shared layers (cross-modal weight sharing) ──

        # Down projection: (B, C, H, W) → (B, d_b, H, W)
        self.W_d = nn.Conv2d(in_channels, bottleneck_dim, 1, bias=True)

        # DCM: offset + modulation mask prediction
        # 2*K*K for (dx, dy) offsets + K*K for modulation masks
        k2 = kernel_size * kernel_size
        self.offset_mask_conv = nn.Conv2d(
            bottleneck_dim, 3 * k2, kernel_size, padding=padding, bias=True,
        )

        # DCM: deformable conv weight (no bias, applied via deform_conv2d)
        self.dcm_weight = nn.Parameter(
            torch.empty(bottleneck_dim, bottleneck_dim, kernel_size, kernel_size)
        )

        # LayerNorm (shared θ_norm)
        self.norm = nn.LayerNorm(bottleneck_dim)

        # Up projection: (B, d_b, H, W) → (B, C, H, W)
        self.W_u = nn.Conv2d(bottleneck_dim, in_channels, 1, bias=True)

        # ── Per-modality scaling (init=0 → identity at start) ──
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_modalities)]
        )

        self._padding = padding
        self._init_weights()

    def _init_weights(self):
        # W_d, W_u: kaiming
        nn.init.kaiming_uniform_(self.W_d.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_d.bias)
        nn.init.kaiming_uniform_(self.W_u.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_u.bias)
        # DCM weight: kaiming
        nn.init.kaiming_uniform_(self.dcm_weight, a=math.sqrt(5))
        # Offset/mask: zero-init → starts as regular conv (no deformation)
        nn.init.zeros_(self.offset_mask_conv.weight)
        nn.init.zeros_(self.offset_mask_conv.bias)

    def forward(self, x: torch.Tensor, modality_idx: int = 0) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) FPN feature for one modality
            modality_idx: index for per-modality α selection
        Returns:
            (B, C, H, W) refined feature
        """
        from torchvision.ops import deform_conv2d

        residual = x

        # Down project
        h = self.W_d(x)  # (B, d_b, H, W)

        # DCM: predict offsets and modulation masks
        om = self.offset_mask_conv(h)  # (B, 3*K*K, H, W)
        k2 = self.kernel_size * self.kernel_size
        offset = om[:, :2 * k2]         # (B, 2*K*K, H, W)
        mask = om[:, 2 * k2:].sigmoid()  # (B, K*K, H, W)

        # DCM: deformable conv (DCNv2)
        h = deform_conv2d(
            h, offset, self.dcm_weight,
            mask=mask, padding=self._padding,
        )  # (B, d_b, H, W)

        # LayerNorm + GELU
        h = h.permute(0, 2, 3, 1)   # (B, H, W, d_b)
        h = self.norm(h)
        h = F.gelu(h)
        h = h.permute(0, 3, 1, 2)   # (B, d_b, H, W)

        # Up project
        h = self.W_u(h)  # (B, C, H, W)

        # Residual with per-modality scaling
        return residual + self.alpha[modality_idx] * h


class DeBAFP_MultiScale(nn.Module):
    """
    Multi-Scale Deformable Bottleneck Adapter for Feature Pyramid (DeBA-FP).

    Applies DeBA-FP to ALL FPN levels with cross-layer weight sharing
    (following CVPR 2026 DeBA paper).

    Cross-layer sharing structure:
      - Shared across levels + modalities: DCM (offset + deform conv), LayerNorm
      - Per-level: W_d, W_u (different in_channels per FPN level)
      - Per-modality: α scaling (shared across levels)

    feat' = feat + α_m * W_u_l(GELU(LN(DCM(W_d_l(feat)))))
    """

    def __init__(
        self,
        fpn_channels: list = None,
        bottleneck_dim: int = 64,
        kernel_size: int = 3,
        num_modalities: int = 3,
    ):
        super().__init__()
        if fpn_channels is None:
            fpn_channels = [32, 64, 256]
        self.fpn_channels = fpn_channels
        self.bottleneck_dim = bottleneck_dim
        self.kernel_size = kernel_size
        self.num_levels = len(fpn_channels)
        padding = kernel_size // 2

        # ── Per-level W_d, W_u (cross-layer: different in_channels) ──
        self.W_d_list = nn.ModuleList([
            nn.Conv2d(ch, bottleneck_dim, 1, bias=True)
            for ch in fpn_channels
        ])
        self.W_u_list = nn.ModuleList([
            nn.Conv2d(bottleneck_dim, ch, 1, bias=True)
            for ch in fpn_channels
        ])

        # ── Shared DCM (cross-layer + cross-modal weight sharing) ──
        k2 = kernel_size * kernel_size
        self.offset_mask_conv = nn.Conv2d(
            bottleneck_dim, 3 * k2, kernel_size, padding=padding, bias=True,
        )
        self.dcm_weight = nn.Parameter(
            torch.empty(bottleneck_dim, bottleneck_dim, kernel_size, kernel_size)
        )
        self.norm = nn.LayerNorm(bottleneck_dim)

        # ── Per-modality α (init=0 → identity at start) ──
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_modalities)]
        )

        self._padding = padding
        self._init_weights()

    def _init_weights(self):
        for W_d in self.W_d_list:
            nn.init.kaiming_uniform_(W_d.weight, a=math.sqrt(5))
            nn.init.zeros_(W_d.bias)
        for W_u in self.W_u_list:
            nn.init.kaiming_uniform_(W_u.weight, a=math.sqrt(5))
            nn.init.zeros_(W_u.bias)
        nn.init.kaiming_uniform_(self.dcm_weight, a=math.sqrt(5))
        nn.init.zeros_(self.offset_mask_conv.weight)
        nn.init.zeros_(self.offset_mask_conv.bias)

    def forward(self, x: torch.Tensor, modality_idx: int = 0,
                level_idx: int = 0) -> torch.Tensor:
        """
        Args:
            x: (B, C_l, H, W) FPN feature for one modality at one level
            modality_idx: index for per-modality α selection
            level_idx: FPN level index for per-level W_d/W_u selection
        Returns:
            (B, C_l, H, W) refined feature
        """
        from torchvision.ops import deform_conv2d

        residual = x

        # Per-level down projection
        h = self.W_d_list[level_idx](x)  # (B, d_b, H, W)

        # Shared DCM: predict offsets and modulation masks
        om = self.offset_mask_conv(h)  # (B, 3*K*K, H, W)
        k2 = self.kernel_size * self.kernel_size
        offset = om[:, :2 * k2]         # (B, 2*K*K, H, W)
        mask = om[:, 2 * k2:].sigmoid()  # (B, K*K, H, W)

        # Shared DCM: deformable conv (DCNv2)
        h = deform_conv2d(
            h, offset, self.dcm_weight,
            mask=mask, padding=self._padding,
        )  # (B, d_b, H, W)

        # Shared LayerNorm + GELU
        h = h.permute(0, 2, 3, 1)   # (B, H, W, d_b)
        h = self.norm(h)
        h = F.gelu(h)
        h = h.permute(0, 3, 1, 2)   # (B, d_b, H, W)

        # Per-level up projection
        h = self.W_u_list[level_idx](h)  # (B, C_l, H, W)

        # Per-modality scaling
        return residual + self.alpha[modality_idx] * h


class MoE_DeBA_BB(nn.Module):
    """
    Mixture-of-Experts Deformable Bottleneck Adapter for Backbone (MoE DeBA-BB).

    Replaces SoftMoE_LoRA_Layer with deformable conv bottleneck adapters as MoE experts,
    using GAP (Global Average Pooling) gating for per-image routing.

    Design Refs:
      - DeBA (CVPR 2026): Deformable bottleneck adapter with cross-layer weight sharing
      - ConvLoRA (ICLR 2024): Multi-scale MoE with GAP gating

    Cross-layer weight sharing:
      - Shared across layers + experts: LayerNorm
      - Shared across layers, per-expert: DCM (offset_mask_conv + dcm_weight)
      - Per-stage, shared across experts: W_d, W_u, GAP gate
      - Per-modality, shared across layers + experts: α scaling

    Multi-scale expert differentiation:
      - Expert 0: ×1 scale (original resolution)
      - Expert 1: ×2 scale (upsample → DCM → downsample)

    Forward (per backbone block):
      x → GAP gate → weights (B, E)
      x → W_d_s(shared) → Σ w_i × DCM_i(scale_i) → LN(shared) → GELU → W_u_s(shared) → α_m × out
    """

    def __init__(
        self,
        stage_dims: list,
        bottleneck_dim: int = 64,
        kernel_size: int = 3,
        num_experts: int = 2,
        num_modalities: int = 3,
        scales: list = None,
        gate_noise_std: float = 0.1,
    ):
        """
        Args:
            stage_dims: input dims per Hiera stage, e.g. [112, 224, 448, 896]
            bottleneck_dim: DCM bottleneck dimension (d_b)
            kernel_size: DCM kernel size
            num_experts: number of MoE experts (default 2: ×1, ×2)
            num_modalities: for per-modality α
            scales: multi-scale factors per expert (default [1, 2])
            gate_noise_std: additive Gaussian noise for gate exploration during training
        """
        super().__init__()
        self.stage_dims = stage_dims
        self.bottleneck_dim = bottleneck_dim
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        self.num_modalities = num_modalities
        self.scales = scales or [1, 2]
        self.gate_noise_std = gate_noise_std
        assert len(self.scales) == num_experts
        padding = kernel_size // 2
        self._padding = padding
        k2 = kernel_size * kernel_size

        # ── Per-expert DCMs (shared across all backbone layers) ──
        self.offset_mask_convs = nn.ModuleList()
        self.dcm_weights = nn.ParameterList()
        for _ in range(num_experts):
            self.offset_mask_convs.append(
                nn.Conv2d(bottleneck_dim, 3 * k2, kernel_size, padding=padding, bias=True)
            )
            self.dcm_weights.append(
                nn.Parameter(torch.empty(bottleneck_dim, bottleneck_dim, kernel_size, kernel_size))
            )

        # ── Shared LayerNorm (cross-layer + cross-expert) ──
        self.norm = nn.LayerNorm(bottleneck_dim)

        # ── Per-stage W_d, W_u (shared across experts) ──
        self.W_d_list = nn.ModuleList([
            nn.Linear(dim, bottleneck_dim, bias=True) for dim in stage_dims
        ])
        self.W_u_list = nn.ModuleList([
            nn.Linear(bottleneck_dim, dim, bias=True) for dim in stage_dims
        ])

        # ── Per-stage GAP gate ──
        self.gates = nn.ModuleList([
            nn.Linear(dim, num_experts) for dim in stage_dims
        ])

        # ── Per-modality α (init=0 → identity at start) ──
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_modalities)]
        )

        # Current modality index (set externally before forward_image)
        self._modality_idx = 0

        # Gate callback for visualization (same interface as SoftMoE_LoRA_Layer)
        self._gate_callback = None

        self._init_weights()

    def _init_weights(self):
        # W_d, W_u: kaiming
        for W_d in self.W_d_list:
            nn.init.kaiming_uniform_(W_d.weight, a=math.sqrt(5))
            nn.init.zeros_(W_d.bias)
        for W_u in self.W_u_list:
            # Zero-init W_u so adapter starts as identity (output = 0)
            nn.init.zeros_(W_u.weight)
            nn.init.zeros_(W_u.bias)
        # DCM weights: kaiming
        for dcm_w in self.dcm_weights:
            nn.init.kaiming_uniform_(dcm_w, a=math.sqrt(5))
        # Offset/mask conv: zero-init → starts as regular conv (no deformation)
        for omc in self.offset_mask_convs:
            nn.init.zeros_(omc.weight)
            nn.init.zeros_(omc.bias)
        # Gate: near-uniform initial routing
        for gate in self.gates:
            nn.init.normal_(gate.weight, std=0.01)
            nn.init.zeros_(gate.bias)

    def set_modality(self, idx: int):
        """Set current modality index for per-modality α scaling."""
        self._modality_idx = idx

    def _apply_dcm(self, h: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        Apply DCM (Deformable Conv Modulation) for one expert.
        Args:
            h: (B, d_b, H, W) bottleneck features
            expert_idx: which expert's DCM to use
        Returns:
            (B, d_b, H, W) deformed features
        """
        from torchvision.ops import deform_conv2d

        om = self.offset_mask_convs[expert_idx](h)  # (B, 3*K², H, W)
        k2 = self.kernel_size * self.kernel_size
        offset = om[:, :2 * k2]          # (B, 2*K², H, W)
        mask = om[:, 2 * k2:].sigmoid()  # (B, K², H, W)

        return deform_conv2d(
            h, offset, self.dcm_weights[expert_idx],
            mask=mask, padding=self._padding,
        )

    def forward(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) — Hiera format (may be windowed: B includes num_windows)
            stage_idx: which stage's W_d/W_u/gate to use
        Returns:
            (B, H, W, C) adapter output delta (to be added to Q and V)
        """
        B, H, W, C = x.shape

        # ── GAP gate: per-image (or per-window) routing ──
        gap = x.mean(dim=[1, 2])  # (B, C)
        gate_logits = self.gates[stage_idx](gap)  # (B, E)
        if self.training and self.gate_noise_std > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.gate_noise_std
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, E)

        # Visualization callback
        if self._gate_callback is not None:
            gw_mean = gate_weights.mean(dim=0).detach().cpu().numpy()
            self._gate_callback(gw_mean)

        # ── Shared down-projection ──
        h = self.W_d_list[stage_idx](x)  # (B, H, W, d_b)
        h_4d = h.permute(0, 3, 1, 2)     # (B, d_b, H, W)

        # ── Per-expert DCM with multi-scale ──
        combined = torch.zeros_like(h_4d)
        for i in range(self.num_experts):
            scale = self.scales[i]
            w_i = gate_weights[:, i].view(B, 1, 1, 1)  # (B, 1, 1, 1)
            if scale == 1:
                ei = self._apply_dcm(h_4d, i)
            else:
                ei = F.interpolate(
                    h_4d, scale_factor=float(scale),
                    mode='bilinear', align_corners=False,
                )
                ei = self._apply_dcm(ei, i)
                ei = F.interpolate(
                    ei, size=(H, W),
                    mode='bilinear', align_corners=False,
                )
            combined = combined + w_i * ei

        # ── Shared: norm → GELU → up-project ──
        combined = combined.permute(0, 2, 3, 1)  # (B, H, W, d_b)
        combined = self.norm(combined)
        combined = F.gelu(combined)
        out = self.W_u_list[stage_idx](combined)  # (B, H, W, C)

        # ── Per-modality α scaling ──
        return self.alpha[self._modality_idx] * out


class _MoE_DeBA_BB_qkv(nn.Module):
    """
    QKV layer wrapper that replaces SoftMoE-LoRA with MoE-DeBA-BB.

    A single DeBA-BB adapter output delta is added to both Q and V,
    following the DeBA paper's concept of refining features before
    they are used in attention.

    The shared MoE_DeBA_BB module is referenced (not owned) — cross-layer
    weight sharing is achieved by passing the same module to all blocks.
    """

    def __init__(
        self,
        qkv: nn.Module,
        shared_deba_bb: MoE_DeBA_BB,
        stage_idx: int,
    ):
        super().__init__()
        self.qkv = qkv
        self.shared_deba_bb = shared_deba_bb
        self.stage_idx = stage_idx
        self.dim = qkv.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) — Hiera block input (may be windowed)
        Returns:
            (B, H*W, 3, nHead, C_head) or (B, H, W, 3*dim_out) — QKV output
        """
        qkv = self.qkv(x)

        # Single adapter delta, shared for Q and V
        delta = self.shared_deba_bb(x, self.stage_idx)  # (B, H, W, C)

        # Add delta to Q (first dim channels) and V (last dim channels)
        qkv[:, :, :, :self.dim] += delta
        qkv[:, :, :, -self.dim:] += delta

        return qkv


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
