"""Aux head 5종 + energy/entropy confidence 함수 (구 sam_lora_image_encoder_seg.py에서 verbatim 이동)."""
import copy
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as tv_models
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
from icecream import ic

from ..modeling.sam2_base import SAM2Base
from ..modules import *  # noqa: F401,F403
from ..modules.moe import _LoRA_qkv, _MoE_LoRA_qkv, _SoftMoE_LoRA_qkv, _MoE_DeBA_BB_qkv  # noqa: F401


class ConfidenceAuxHead(nn.Module):
    """공유 auxiliary segmentation head.
    모든 모달리티가 동일한 head를 사용하여 파라미터 최소화.
    Energy score 계산을 위한 raw logit을 출력한다 (softmax 적용 전).
    """

    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        mid_channels = max(in_channels // 4, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, feat):
        """
        Args:
            feat: backbone feature (B, C, H, W)
        Returns:
            logits: (B, num_classes, H, W) — raw logits
        """
        return self.head(feat)


class ModalAuxDecoder(nn.Module):
    """P14 전용: 모달리티별 독립 auxiliary segmentation decoder.

    P13의 공유 ConfidenceAuxHead와 달리 각 모달리티가 고유 파라미터를 가짐.
    첫 conv를 3×3으로 변경 → 모달리티별 공간 패턴에 특화:
      - RGB: 텍스처 경계 / 색상 gradient
      - LiDAR: 점군 집적 패턴 / 깊이 불연속
      - Thermal: 온도 gradient / 열 분포 형태
    Energy score 계산을 위한 raw logit 출력 (softmax 미적용).
    """

    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        mid_channels = max(in_channels // 4, 32)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),  # 3×3 spatial context
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, feat):
        """
        Args:
            feat: backbone feature (B, C, H, W)
        Returns:
            logits: (B, num_classes, H, W) — raw logits
        """
        return self.decoder(feat)


class MultiScaleModalAuxDecoder(nn.Module):
    """P17: Multi-scale FPN feature를 활용한 aux segmentation decoder.

    기존 ModalAuxDecoder는 backbone_fpn[0](32ch)만 사용.
    이 decoder는 3개 FPN 레벨 모두 활용:
      - fpn[0]: 256×256, 32ch  (high-res spatial detail)
      - fpn[1]: 128×128, 64ch  (mid-level features)
      - fpn[2]:  64×64, 256ch  (semantic context)

    전략: 각 레벨을 proj_dim으로 project → fpn[0] 해상도로 upsample → concat → decode
    """

    def __init__(self, fpn_channels=(32, 64, 256), proj_dim=32, num_classes=4):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, proj_dim, 1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for ch in fpn_channels
        ])

        fused_dim = proj_dim * len(fpn_channels)  # 32 * 3 = 96
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_dim, fused_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_dim // 2, num_classes, 1),
        )

    def forward(self, fpn_feats):
        """
        Args:
            fpn_feats: list of 3 tensors [fpn0, fpn1, fpn2]
                fpn0: (B, 32, H0, W0)   — highest resolution
                fpn1: (B, 64, H1, W1)   — mid resolution
                fpn2: (B, 256, H2, W2)  — lowest resolution
        Returns:
            logits: (B, num_classes, H0, W0) — raw logits at fpn[0] resolution
        """
        target_size = fpn_feats[0].shape[2:]

        projected = []
        for i, feat in enumerate(fpn_feats):
            p = self.proj_layers[i](feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear',
                                  align_corners=False)
            projected.append(p)

        fused = torch.cat(projected, dim=1)
        return self.decoder(fused)


class ResNetAuxBackbone(nn.Module):
    """P18: Trainable ResNet-18 aux backbone with per-modality input stems.

    Frozen SAM2 FPN feature의 정보량 한계(ISSUE-008)를 극복하기 위해
    ImageNet pretrained ResNet-18을 trainable aux backbone으로 사용.

    구조: 3개 독립 stem(모달리티별 low-level 특화) + 1개 공유 body(layer1~3)
    Output: layer2(128ch, H/8) + layer3(256ch, H/16)
    Params: ~11.2M (shared body) + ~28K (3 stems)
    """

    def __init__(self, num_modalities=3, pretrained=True):
        super().__init__()

        resnet = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Per-modality input stems (conv1+bn1 replacement)
        # 모든 모달리티는 dataset에서 3ch로 repeat → 입력 3ch 통일
        self.stems = nn.ModuleList()
        for _ in range(num_modalities):
            stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            if pretrained:
                stem[0].weight.data.copy_(resnet.conv1.weight.data)
                stem[1].weight.data.copy_(resnet.bn1.weight.data)
                stem[1].bias.data.copy_(resnet.bn1.bias.data)
                stem[1].running_mean.copy_(resnet.bn1.running_mean)
                stem[1].running_var.copy_(resnet.bn1.running_var)
            self.stems.append(stem)

        # Shared body: maxpool + layer1~3
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # 64ch, H/4
        self.layer2 = resnet.layer2   # 128ch, H/8
        self.layer3 = resnet.layer3   # 256ch, H/16

    def forward(self, x, modal_idx):
        """Extract multi-scale features for a single modality.

        Args:
            x: (B, 3, H, W) input tensor
            modal_idx: modality index (0=RGB, 1=LiDAR, 2=Thermal)
        Returns:
            (layer2_feat, layer3_feat): tuple
                layer2_feat: (B, 128, H/8, W/8)
                layer3_feat: (B, 256, H/16, W/16)
        """
        x = self.stems[modal_idx](x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layer2_feat = self.layer2(x)
        layer3_feat = self.layer3(layer2_feat)
        return layer2_feat, layer3_feat


class ResNetAuxDecoder(nn.Module):
    """P18: ResNet layer2(128ch) + layer3(256ch) → aux segmentation logits.

    MultiScaleModalAuxDecoder와 유사하나 ResNet 2-level feature 사용.
    Output resolution: layer2와 동일 (H/8 × W/8 = 128×128 for 1024 input)
    Params: ~53K per modality
    """

    def __init__(self, resnet_channels=(128, 256), proj_dim=32, num_classes=4):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, proj_dim, 1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for ch in resnet_channels
        ])

        fused_dim = proj_dim * len(resnet_channels)  # 32 * 2 = 64
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_dim, fused_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_dim, num_classes, 1),
        )

    def forward(self, resnet_feats):
        """
        Args:
            resnet_feats: (layer2_feat, layer3_feat)
                layer2_feat: (B, 128, H2, W2)
                layer3_feat: (B, 256, H3, W3) where H3=H2/2
        Returns:
            logits: (B, num_classes, H2, W2) at layer2 resolution
        """
        layer2_feat, layer3_feat = resnet_feats
        target_size = layer2_feat.shape[2:]

        projected = []
        for i, feat in enumerate([layer2_feat, layer3_feat]):
            p = self.proj_layers[i](feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear',
                                  align_corners=False)
            projected.append(p)

        fused = torch.cat(projected, dim=1)
        return self.decoder(fused)


def compute_energy_confidence(aux_logits_list, temperature=1.0):
    """Energy score 기반 modality confidence 계산.

    학습 가능 파라미터 없음 — computed signal이므로 상수로 수렴 불가.
    학습/추론 동일 메커니즘 — oracle-at-train / guess-at-test 불일치 없음.

    Args:
        aux_logits_list: List[Tensor], 길이 = num_modalities
            각 원소: (B, num_classes, H, W) raw logits
        temperature: energy score temperature (default 1.0)
    Returns:
        weights: (B, num_modalities) — softmax normalized fusion weights
    """
    confidences = []
    for z in aux_logits_list:
        # Energy score: E(x) = -T * log(sum(exp(z_k / T)))
        energy = -temperature * torch.logsumexp(z / temperature, dim=1)  # (B, H, W)
        # 높은 confidence = 낮은 energy (더 음수) → -energy가 클수록 confident
        conf = -energy.mean(dim=[1, 2])  # (B,)
        confidences.append(conf)

    confidences = torch.stack(confidences, dim=1)  # (B, num_modalities)
    weights = F.softmax(confidences / temperature, dim=1)  # (B, num_modalities)
    return weights


def compute_spatial_energy_confidence(aux_logits_list, temperature=1.0):
    """P15 전용: spatial-wise modality confidence map 계산.

    compute_energy_confidence()에서 spatial mean(.mean(dim=[1,2]))을 제거해
    위치별 서로 다른 모달리티 가중치 map을 반환한다.

    Args:
        aux_logits_list: List[Tensor], 길이 = num_modalities
            각 원소: (B, num_classes, H_feat, W_feat) raw logits
        temperature: energy score temperature (default 1.0)
    Returns:
        weights: (B, num_modalities, H_feat, W_feat) — 위치별 softmax 가중치
    """
    conf_maps = []
    for z in aux_logits_list:
        energy = -temperature * torch.logsumexp(z / temperature, dim=1)
        conf_map = -energy
        conf_maps.append(conf_map)

    stacked = torch.stack(conf_maps, dim=1)
    weights = F.softmax(stacked / temperature, dim=1)
    return weights


def compute_spatial_entropy_confidence(aux_logits_list, temperature=1.0, num_classes=4):
    """P15 전용: Calibrated Entropy 기반 spatial confidence map.

    Energy Score 대신 예측 분포의 entropy를 사용.
    - Energy: logit magnitude 기반 → "자신있게 틀리면" 높은 점수 (dangerous)
    - Entropy: 확률 분포 균등도 → 4클래스에 골고루 분산 = 낮은 confidence (safe)

    예: LiDAR가 Sky에서 Water로 확신있게 오예측
       → Energy: 높음 (logit 크니까) → 나쁨
       → Entropy: 낮음 (한 클래스에 집중) → confidence 높음... BUT aux head가
         부정확하면 분산된 예측 → 높은 entropy → 낮은 confidence → 안전

    Args:
        aux_logits_list: List[Tensor], 길이 = num_modalities
            각 원소: (B, num_classes, H_feat, W_feat) raw logits (반드시 .detach()된 것)
        temperature: softmax temperature (default 1.0)
        num_classes: 클래스 수 (entropy 정규화용, default 4)
    Returns:
        weights: (B, num_modalities, H_feat, W_feat) — 위치별 softmax 가중치
    """
    conf_maps = []
    max_entropy = math.log(num_classes)

    for z in aux_logits_list:
        # Temperature-scaled softmax → calibrated probability
        probs = F.softmax(z / temperature, dim=1)           # (B, C, H, W)
        log_probs = F.log_softmax(z / temperature, dim=1)   # (B, C, H, W)
        entropy = -(probs * log_probs).sum(dim=1)           # (B, H, W)
        # Normalize: 0 (완전 확신) ~ 1 (완전 균등), confidence = 1 - normalized_entropy
        confidence = 1.0 - entropy / max_entropy            # (B, H, W)
        conf_maps.append(confidence)

    stacked = torch.stack(conf_maps, dim=1)                 # (B, m, H, W)
    weights = F.softmax(stacked / temperature, dim=1)       # (B, m, H, W)
    return weights
