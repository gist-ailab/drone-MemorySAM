"""P34-ReliaDINO — DINOv3-RBMA multimodal semantic segmentation (card A).

ReliaDINO = FrozenViTEncoder (shared frozen DINOv3/DINOv2 ViT, per-modality
LoRA) -> ReliabilityGatedFusion (cross-modal memory-style attention, RBMA-v2
bias + competence gate) -> SimpleFPN on the FUSED stride-16 map -> light
query-free conv head to num_classes at stride 4 (Mask2Former-lite is a staged
TODO — speed first; the head is deliberately dumb so backbone/fusion effects
are readable).

Trainer contract (compatible with the P24+ gate_loss_data pattern):
  training : forward(inputs, multimask_output, gt_mask) ->
             (logits(B,K,H,W), m_feat(B,C,H/4,W/4), gate_loss_data: dict)
  eval     : forward(inputs, multimask_output) -> (logits, m_feat)
so `output, _ = model(images, True)` (val_mm_sam.evaluate) works unchanged.
No SAM2 imports anywhere in this package.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import FrozenViTEncoder, SimpleFPN, LayerNorm2d
from .fusion import ReliabilityGatedFusion


class FPNSegHead(nn.Module):
    """Light query-free head: upsample every pyramid level to stride 4, sum,
    two 3x3 conv blocks, 1x1 classifier. (GOOSE/SemanticFPN-style.)"""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
        )
        self.cls = nn.Conv2d(dim, num_classes, 1)

    def forward(self, pyramid: List[torch.Tensor]):
        tgt = pyramid[0].shape[-2:]                       # stride-4 level
        x = pyramid[0]
        for p in pyramid[1:]:
            x = x + F.interpolate(p, size=tgt, mode='bilinear', align_corners=False)
        feat = self.fuse(x)                               # (B, dim, H/4, W/4)
        return self.cls(feat), feat


class ReliaDINO(nn.Module):

    def __init__(self,
                 num_classes: int = 25,
                 modalities: Sequence[str] = ('img', 'depth', 'event', 'lidar'),
                 backbone: str = 'vit_large_patch16_dinov3',
                 backbone_fallback: str = 'vit_large_patch14_reg4_dinov2',
                 pretrained: bool = True,
                 img_size: int = 1024,
                 lora_r: int = 8,
                 lora_alpha: Optional[float] = None,
                 fpn_dim: int = 256,
                 fusion_layers: int = 2,
                 fusion_heads: int = 8,
                 fusion_mlp_ratio: float = 4.0,
                 aux_hidden: int = 256,
                 attn_bias: bool = True,
                 lambda1_init: float = 1.0,
                 consistency_bias: bool = True,
                 lambda2_init: float = 0.5,
                 gate_enable: bool = True,
                 gate_tau: float = 0.25,
                 gate_entropy_reg: float = 0.0,
                 gate_entropy_floor: float = 0.5,
                 veto_floor: bool = True,
                 veto_thresh: float = 0.10,
                 veto_cap: float = 0.05,
                 calibrate: bool = True,
                 modal_dropout: bool = False,
                 modal_dropout_p: float = 0.3,
                 modal_dropout_targets: Sequence[int] = (0, 1),
                 modal_dropout_warmup_ep: int = 20):
        super().__init__()
        self.modalities = list(modalities)
        self.num_modalities = len(self.modalities)
        self.num_classes = num_classes

        self.encoder = FrozenViTEncoder(
            backbone=backbone, fallback=backbone_fallback, pretrained=pretrained,
            img_size=img_size, num_modalities=self.num_modalities,
            lora_r=lora_r, lora_alpha=lora_alpha)
        dim = self.encoder.embed_dim
        self.fusion = ReliabilityGatedFusion(
            dim=dim, num_classes=num_classes, num_modalities=self.num_modalities,
            num_layers=fusion_layers, num_heads=fusion_heads,
            mlp_ratio=fusion_mlp_ratio, aux_hidden=aux_hidden,
            attn_bias=attn_bias, lambda1_init=lambda1_init,
            consistency_bias=consistency_bias, lambda2_init=lambda2_init,
            gate_enable=gate_enable, gate_tau=gate_tau,
            gate_entropy_reg=gate_entropy_reg, gate_entropy_floor=gate_entropy_floor,
            veto_floor=veto_floor, veto_thresh=veto_thresh, veto_cap=veto_cap,
            calibrate=calibrate)
        self.fpn = SimpleFPN(dim, fpn_dim)
        self.head = FPNSegHead(fpn_dim, num_classes)

        # M2 seam (asymmetric modality dropout) — default OFF: it helped nothing
        # at mid-run so far (P33 empirical constraint 4); seam kept for P34.2.
        self.modal_dropout = modal_dropout
        self.modal_dropout_p = modal_dropout_p
        self.modal_dropout_targets = tuple(int(t) for t in modal_dropout_targets)
        self.modal_dropout_warmup_ep = int(modal_dropout_warmup_ep)
        self._current_epoch = 0          # trainer sets this each epoch
        self._last_dropped_modality = None
        # [analysis] 표준 분석항목 1/2용 eval 전용 스태시 (tools/seg_analysis_pipeline이
        # capability probe로 감지 — 학습 동작에는 영향 0)
        self._last_per_modal_feats = None

    # ── M2: P33._maybe_drop_modality port (zero-input replacement, train only) ─
    def _maybe_drop_modality(self, batched_input):
        if not (self.modal_dropout and self.training):
            return batched_input
        m = len(batched_input)
        if m < 2:
            return batched_input
        p = self.modal_dropout_p
        if self.modal_dropout_warmup_ep > 0:
            p = p * min(1.0, float(self._current_epoch) / self.modal_dropout_warmup_ep)
        if float(torch.rand(1).item()) >= p:
            return batched_input
        targets = [t for t in self.modal_dropout_targets if 0 <= t < m]
        if not targets:
            return batched_input
        j = targets[int(torch.randint(len(targets), (1,)).item())]
        self._last_dropped_modality = j
        new_input = list(batched_input)
        new_input[j] = torch.zeros_like(new_input[j])
        return new_input

    def set_grad_checkpointing(self, enable: bool = True):
        self.encoder.set_grad_checkpointing(enable)

    def forward(self, batched_input: List[torch.Tensor], multimask_output: bool = True,
                gt_mask: Optional[torch.Tensor] = None):
        # `multimask_output` kept for call-site compatibility with the SAM2 fleet.
        self._last_dropped_modality = None
        x = self._maybe_drop_modality(batched_input)
        H, W = x[0].shape[-2:]
        feats = [self.encoder(x[i], i) for i in range(self.num_modalities)]
        if not self.training:
            self._last_per_modal_feats = [f.detach() for f in feats]
        fused, aux = self.fusion(feats, gt_mask if self.training else None)
        pyramid = self.fpn(fused)
        logits, m_feat = self.head(pyramid)
        logits = F.interpolate(logits.float(), size=(H, W),
                               mode='bilinear', align_corners=False)
        if self.training:
            return logits, m_feat, aux
        return logits, m_feat


def build_reliadino(cfg: dict, num_classes: int) -> ReliaDINO:
    """Map a training-config dict (configs/*_P34_reliadino.yaml) to ReliaDINO."""
    mc = cfg['MODEL']
    fus = mc.get('FUSION', {}) or {}
    ab = fus.get('ATTN_BIAS', {}) or {}
    gate = mc.get('GATE', {}) or {}
    veto = gate.get('VETO_FLOOR', {}) or {}
    cal = mc.get('CALIBRATION', {}) or {}
    cons = mc.get('CONSISTENCY', {}) or {}
    mdrop = mc.get('MODAL_DROPOUT', {}) or {}
    modals = cfg['DATASET']['MODALS']
    raw_targets = mdrop.get('TARGETS', ['img', 'depth'])
    tgt_idx = [modals.index(t) if isinstance(t, str) else int(t)
               for t in raw_targets if (not isinstance(t, str)) or (t in modals)]
    img_size = cfg['TRAIN']['IMAGE_SIZE'][0]
    return ReliaDINO(
        num_classes=num_classes,
        modalities=modals,
        backbone=mc.get('BACKBONE_TIMM', 'vit_large_patch16_dinov3'),
        backbone_fallback=mc.get('BACKBONE_FALLBACK', 'vit_large_patch14_reg4_dinov2'),
        pretrained=mc.get('PRETRAINED_BACKBONE', True),
        img_size=img_size,
        lora_r=mc.get('LORA_R', 8),
        lora_alpha=mc.get('LORA_ALPHA', None),
        fpn_dim=mc.get('FPN_DIM', 256),
        fusion_layers=fus.get('NUM_LAYERS', 2),
        fusion_heads=fus.get('NUM_HEADS', 8),
        fusion_mlp_ratio=fus.get('MLP_RATIO', 4.0),
        aux_hidden=fus.get('AUX_HIDDEN', 256),
        attn_bias=ab.get('ENABLE', True),
        lambda1_init=ab.get('LAMBDA1_INIT', 1.0),
        consistency_bias=cons.get('ENABLE', True),
        lambda2_init=cons.get('LAMBDA2_INIT', 0.5),
        gate_enable=gate.get('ENABLE', True),
        gate_tau=gate.get('TAU', 0.25),
        gate_entropy_reg=gate.get('ENTROPY_REG', 0.0),
        gate_entropy_floor=gate.get('ENTROPY_FLOOR', 0.5),
        veto_floor=veto.get('ENABLE', True),
        veto_thresh=veto.get('THRESH', 0.10),
        veto_cap=veto.get('CAP', 0.05),
        calibrate=cal.get('ENABLE', True),
        modal_dropout=mdrop.get('ENABLE', False),
        modal_dropout_p=mdrop.get('P', 0.3),
        modal_dropout_targets=tuple(tgt_idx) if tgt_idx else (0, 1),
        modal_dropout_warmup_ep=mdrop.get('WARMUP_EP', 20),
    )
