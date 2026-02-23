"""
LoRA/LoLA 공통 유틸: 실험 모델(LoRA_Sam_P1~P9)이 아닌 보조 클래스·함수.
- MLP_my, _LoRA_qkv, ConfidenceHead, ConfidenceHeadV2, CrossModalFusionHead
- MoE_LoRA_Layer, _MoE_LoRA_qkv
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
    Cross-Modal Fusion Head (P9): 모든 모달리티 feature를 동시에 비교하여
    상대적 융합 가중치를 산출.

    기존 ConfidenceHeadV2는 각 모달리티를 독립 평가(sigmoid → 포화 → 균등화)하지만,
    이 모듈은 모든 모달리티를 동시에 보고 상대 중요도를 비교한다.

    구조: 공유 Compress(GAP+Linear) → Concat → Compare MLP → Softmax
    """
    def __init__(self, in_channels, num_modalities=3, hidden_dim=64, temperature=1.0):
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

    def forward(self, features_list):
        """
        Args:
            features_list: List of (B, C, H, W) — 각 모달리티의 backbone feature
        Returns:
            weights: (B, num_modalities) softmax 정규화된 가중치
            logits:  (B, num_modalities) raw logits (시각화/디버깅용)
        """
        compressed = [self.compress(f) for f in features_list]
        concat = torch.cat(compressed, dim=1)       # (B, hidden_dim * m)
        logits = self.compare(concat)                # (B, m)
        weights = F.softmax(logits / self.temperature, dim=1)  # (B, m)
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
    """
    def __init__(self, in_features, rank, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank
        self.in_features = in_features
        
        # Gating Network: 입력 토큰별로 전문가 가중치를 계산
        self.gate = nn.Linear(in_features, num_experts)
        
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
        
        # Expert 초기화
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.experts_a[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_b[i].weight)

    def forward(self, x):
        # x shape: (B, N, C) 또는 (B, H, W, C) — Hiera 백본은 4D (B, H, W, C) 사용
        gate_logits = self.gate(x)  # (..., num_experts)
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
