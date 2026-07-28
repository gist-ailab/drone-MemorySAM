"""P34-ReliaDINO encoder — frozen timm ViT (DINOv3 / DINOv2 fallback) with
per-modality LoRA on the fused qkv projection + ViTDet-style SimpleFPN.

Design source: .claude_logs/research_vault/material/brainstorm_next_arch_20260708.md
(카드 A) — "DINOv3 ViT-L/16 frozen + per-modality LoRA(Q/V, r8~16, 모달별 독립) +
ViTDet simple-FPN(stride {4,8,16,32})".

Why a new LoRA wrapper instead of reusing sam_lora_image_encoder_seg._LoRA_qkv:
that wrapper is written against SAM2 Hiera's MultiScaleAttention (separate q/v
linears inside a Hiera block, SoftMoE expert mixing) and drags the whole SAM2
package in as an import. Card A's point is to remove the SAM2 dependency, so we
re-implement the *pattern* (additive low-rank delta on Q and V, one independent
adapter pair per modality) minimally for timm ViTs.

timm compatibility notes (verified against timm 1.0.24):
  - `vit_*_dinov3`  -> timm.models.eva.Eva; blocks[k].attn = EvaAttention with a
    fused `qkv` nn.Linear(bias=False) plus separate q_bias/k_bias/v_bias params.
    EvaAttention.forward may bypass the qkv Module via
    `F.linear(x, self.qkv.weight, qkv_bias)` unless `qkv_bias_separate=True`;
    we force that flag so our wrapper's forward actually runs.
  - `vit_*_dinov2` / plain `vit_*` -> VisionTransformer; blocks[k].attn.qkv is a
    fused nn.Linear and is always invoked as a Module. Same wrapper applies.
"""
from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalLoRAQKV(nn.Module):
    """Wraps a fused qkv nn.Linear with per-modality LoRA adapters on Q and V.

    y = qkv(x) ; y[..., :D]  += s * x @ A_q[m].T @ B_q[m].T   (query slice)
                 y[..., 2D:] += s * x @ A_v[m].T @ B_v[m].T   (value slice)

    K is left untouched (card A: "per-modality LoRA(Q/V)"; also matches the SAM2
    lineage where only q/v carry adapters). The active modality m is set by
    `FrozenViTEncoder.set_modality()` before each per-modality forward.
    """

    def __init__(self, base: nn.Linear, num_modalities: int, r: int, alpha: float = None):
        super().__init__()
        assert base.out_features % 3 == 0, "expected fused qkv linear (out = 3*attn_dim)"
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.attn_dim = base.out_features // 3
        self.num_modalities = num_modalities
        self.r = r
        self.scale = (alpha if alpha is not None else float(r)) / float(r)
        # (M, r, in) down-projections and (M, attn_dim, r) up-projections.
        self.a_q = nn.Parameter(torch.empty(num_modalities, r, self.in_features))
        self.b_q = nn.Parameter(torch.zeros(num_modalities, self.attn_dim, r))
        self.a_v = nn.Parameter(torch.empty(num_modalities, r, self.in_features))
        self.b_v = nn.Parameter(torch.zeros(num_modalities, self.attn_dim, r))
        for m in range(num_modalities):
            nn.init.kaiming_uniform_(self.a_q[m], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.a_v[m], a=math.sqrt(5))
        self.active_modality = 0

    @property
    def weight(self):  # safety for code paths that touch qkv.weight directly
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        m = self.active_modality
        aq, bq = self.a_q[m], self.b_q[m]
        av, bv = self.a_v[m], self.b_v[m]
        dq = F.linear(F.linear(x, aq), bq) * self.scale
        dv = F.linear(F.linear(x, av), bv) * self.scale
        d = self.attn_dim
        y = torch.cat([y[..., :d] + dq, y[..., d:2 * d], y[..., 2 * d:] + dv], dim=-1)
        return y


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm on (B, C, H, W) — ViTDet convention."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SimpleFPN(nn.Module):
    """ViTDet-style simple feature pyramid from a single stride-16 map.

    Produces 4 levels at relative scales {4x up? no: x4, x2, x1, x0.5} of the
    input map — i.e. strides {4, 8, 16, 32} when the input map is stride 16
    (patch16 backbone). For the patch14 fallback the absolute strides become
    {3.5, 7, 14, 28}; the decoder only relies on the relative 2x ladder, so this
    is transparent.
    """

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim // 2, 2, stride=2),
            LayerNorm2d(in_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, 2, stride=2),
        )
        self.up8 = nn.ConvTranspose2d(in_dim, in_dim // 2, 2, stride=2)
        self.down32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lateral = nn.ModuleList()
        for dim in (in_dim // 4, in_dim // 2, in_dim, in_dim):
            self.lateral.append(nn.Sequential(
                nn.Conv2d(dim, out_dim, 1, bias=False),
                LayerNorm2d(out_dim),
                nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
                LayerNorm2d(out_dim),
            ))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.up4(x), self.up8(x), x, self.down32(x)]
        return [lat(f) for lat, f in zip(self.lateral, feats)]


class FrozenViTEncoder(nn.Module):
    """Frozen timm ViT shared across modalities; per-modality LoRA on qkv.

    forward(x, modality_idx) -> (B, C, h, w) token map at stride = patch size.
    The SimpleFPN is intentionally NOT applied here: P34 fuses modalities at the
    token level first and runs a single shared FPN on the fused map (ViTDet
    generates the whole pyramid from one stride-16 map anyway; running 4 FPNs
    pre-fusion would 4x the cost for no design gain — kept as an ablation seam
    in the design doc, not in code).
    """

    def __init__(self,
                 backbone: str = 'vit_large_patch16_dinov3',
                 fallback: str = 'vit_large_patch14_reg4_dinov2',
                 pretrained: bool = True,
                 img_size: int = 1024,
                 num_modalities: int = 4,
                 lora_r: int = 8,
                 lora_alpha: Optional[float] = None,
                 tap_layers: Optional[Sequence[int]] = None,
                 num_taps: int = 0):
        super().__init__()
        import timm  # local import: keeps semseg.models importable without timm

        self.backbone_name, self.backbone = self._create(
            timm, backbone, fallback, pretrained, img_size)
        self.patch = self.backbone.patch_embed.patch_size[0]
        self.embed_dim = self.backbone.embed_dim
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 0)
        self.num_modalities = num_modalities

        # Freeze everything, then inject trainable per-modality LoRA.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lora_layers: List[MultiModalLoRAQKV] = []
        for blk in self.backbone.blocks:
            attn = blk.attn
            assert getattr(attn, 'qkv', None) is not None, \
                f"{self.backbone_name}: block attn has no fused qkv — unsupported layout"
            # Eva path: force module-call so the wrapper forward is not bypassed
            # by F.linear(x, self.qkv.weight, bias) (see module docstring).
            if hasattr(attn, 'qkv_bias_separate') and getattr(attn, 'q_bias', None) is not None:
                attn.qkv_bias_separate = True
            wrapper = MultiModalLoRAQKV(attn.qkv, num_modalities, lora_r, lora_alpha)
            attn.qkv = wrapper
            self.lora_layers.append(wrapper)
        # register so .parameters()/.state_dict() see the adapters exactly once
        # (they already live under backbone.blocks[k].attn.qkv — nothing extra needed;
        # self.lora_layers is a plain python list on purpose).

        # [P43-T2] PMT-style multi-depth lateral taps. Forward HOOKS (not a
        # re-implemented block loop) so `forward_features` — with its
        # backbone-specific pos-embed/RoPE/norm handling — stays the single
        # source of truth and the no-tap path is bit-identical. Hooks are
        # registered only when taps are requested.
        self.tap_layers: List[int] = []
        self.last_taps: Optional[List[torch.Tensor]] = None
        self.collect_taps = True     # model flips this off when P43.LATERAL is ablated
        self._tap_buf: dict = {}
        n = len(self.backbone.blocks)
        if tap_layers:
            self.tap_layers = sorted({int(t) % n for t in tap_layers})
        elif num_taps > 0:
            # evenly spaced over the depth, excluding the last block (whose
            # output forward_features already returns): 24 blocks, 3 taps ->
            # [5, 11, 17].
            self.tap_layers = sorted({
                max(0, min(n - 2, round((i + 1) * n / (num_taps + 1)) - 1))
                for i in range(num_taps)})
        for j, li in enumerate(self.tap_layers):
            self.backbone.blocks[li].register_forward_hook(self._tap_hook(j))

    def _tap_hook(self, slot: int):
        def hook(_module, _inp, out):
            self._tap_buf[slot] = out[0] if isinstance(out, tuple) else out
        return hook

    @staticmethod
    def _create(timm, name, fallback, pretrained, img_size):
        # [HF-offline workaround 2026-07-28] RELIADINO_LOCAL_BACKBONE env가 있으면
        # 로컬 weights 파일에서 백본 로드(HF Hub 우회). hpca100 HF가 offline=RANDOM INIT
        # /online=hang으로 양쪽 깨져 있어, timm pretrained_cfg_overlay(file=)로 정규
        # 로딩 로직을 로컬 파일에 적용한다. primary backbone에만 적용.
        import os
        local_weights = os.environ.get('RELIADINO_LOCAL_BACKBONE') or None
        def _try(n, pre):
            kwargs = dict(pretrained=pre, num_classes=0, img_size=img_size)
            if pre and local_weights and n == name:
                kwargs['pretrained_cfg_overlay'] = dict(file=local_weights)
            try:
                return timm.create_model(n, dynamic_img_size=True, **kwargs)
            except TypeError:
                return timm.create_model(n, **kwargs)

        candidates = [name] + ([fallback] if fallback and fallback != name else [])
        last_err = None
        for cand in candidates:
            try:
                return cand, _try(cand, pretrained)
            except Exception as e:  # gated HF weights / offline / unknown name
                last_err = e
                warnings.warn(f"[ReliaDINO] create_model('{cand}', pretrained={pretrained}) "
                              f"failed: {e}")
        if pretrained:
            warnings.warn(f"[ReliaDINO] all pretrained loads failed — falling back to "
                          f"RANDOM INIT '{name}'. Do NOT train a real run like this.")
            return name, FrozenViTEncoder._create(timm, name, fallback, False, img_size)[1]
        raise RuntimeError(f"could not build backbone '{name}' (last error: {last_err})")

    def set_modality(self, idx: int):
        for w in self.lora_layers:
            w.active_modality = idx

    def set_grad_checkpointing(self, enable: bool = True):
        # ISSUE-027 가드: timm의 non-reentrant checkpoint는 backward 재계산
        # 시점의 self.active_modality(=마지막 모달)를 읽어, 비최종 모달들의
        # LoRA gradient를 오염된 활성화로 계산한다(무경고·무에러 — bengio
        # 실증, jarvis/hpca100 configs의 "절대 true 금지" 주석). 멀티모달
        # 구성에서는 요청을 거부하고 강제 off한다. 메모리 부족 시 배치를
        # 줄일 것. (근본 수정 = 모달 인덱스를 checkpoint 함수 인자로 결박)
        if enable and getattr(self, 'num_modalities', 1) > 1:
            print("[FrozenViTEncoder] 🔴 GRADIENT_CHECKPOINT=true 거부 → off "
                  "(ISSUE-027: stale active_modality가 LoRA grad 오염)")
            enable = False
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)

    def forward(self, x: torch.Tensor, modality_idx: int) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, embed_dim, H//patch, W//patch)."""
        self.set_modality(modality_idx)
        H, W = x.shape[-2:]
        Hp = (H // self.patch) * self.patch
        Wp = (W // self.patch) * self.patch
        if (Hp, Wp) != (H, W):  # patch14 fallback: round to patch multiple
            x = F.interpolate(x, size=(Hp, Wp), mode='bilinear', align_corners=False)
        self.last_taps = None
        self._tap_buf.clear()
        tokens = self.backbone.forward_features(x)
        h, w = Hp // self.patch, Wp // self.patch
        if self.tap_layers and self.collect_taps:
            self.last_taps = [self._to_map(self._tap_buf[j], h, w)
                              for j in range(len(self.tap_layers))]
        self._tap_buf.clear()
        if tokens.dim() == 4:  # some timm models return NHWC maps with dynamic_img_size
            return tokens.permute(0, 3, 1, 2).contiguous()
        tokens = tokens[:, self.num_prefix_tokens:]
        B, N, C = tokens.shape
        assert N == h * w, f"token count {N} != grid {h}x{w}"
        return tokens.transpose(1, 2).reshape(B, C, h, w)

    def _to_map(self, t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Block output (B,N,C) or (B,h,w,C) -> (B,C,h,w), prefix tokens dropped."""
        if t.dim() == 4:
            return t.permute(0, 3, 1, 2).contiguous()
        t = t[:, t.shape[1] - h * w:]      # drops prefix (cls/reg) tokens, if any
        return t.transpose(1, 2).reshape(t.shape[0], -1, h, w)
