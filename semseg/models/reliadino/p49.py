"""[P49-AIR] Asymmetric Injection with RGB-primary.

설계 정본 = `.claude_logs/decisions/2026-08-10-p49-air-asymmetric-injection-proposal.md`
(§2 변경 목록 A1~A5, §4 게이트). 이 파일은 그 A1~A4의 구현이다.

한 줄
  대칭 융합 트렁크(P34~P47 계보)를 버리고 **온전한 RGB 주경로(DINOv3-L 부분
  fine-tune) + 인코더-내부 비대칭 보조 주입(zero-init γ)** 으로 간다. 보조 모달은
  RGB 표현을 대체하지 않고 **편향만** 주입한다.

구조 (A1~A4)
  A1 RGB 주경로 : timm DINOv3 ViT-L/16. `P49.RGB_FT=true` 면 백본 전체가
                  trainable(LoRA 폐지, layer-wise LR decay는 train_reliadino.py가
                  건다). false면 기존 frozen + per-modality LoRA 경로 그대로.
  A2 보조 인코더: 모달별 **독립** ConvNeXt-S(timm, ImageNet pretrained) →
                  4-스테이지 멀티스케일(stride 4/8/16/32). `P49.AUX_CNN=false`
                  면 ablation 팔(`AUX_ENCODER: vit_lora | stem`).
  A3 비대칭 주입: ViT 24층을 4블록(6층)으로 나눠 각 경계에서
                  Injector(ViT=Q, 보조=K/V) + Extractor(반대 방향 + FFN).
                  Injector 출력은 **zero-init 스칼라 γ[block, modal]** 을 곱해
                  잔차 가산 → init 시 RGB 트렁크가 정확히 identity.
  A4 헤드       : ViT 최종 feature로 SimpleFPN 4-스케일 피라미드를 만들고, 최종
                  Extractor 상태를 **zero-init γ_pyr[modal, level]** 로 가산.
                  P39 dual-path(query 경쟁 + path dropout)는 쓰지 않는다 — 단일
                  head(`HEAD_MODE`)이고 arbiter/β 잔차 경로가 없다.

🔴 왜 γ_pyr 까지 zero-init 인가 (ViT-Adapter/MM-SA와 다른 점)
  ViT-Adapter는 injector만 zero-init 하고 spatial branch는 디코더에 직결한다 →
  init 출력이 보조 입력에 의존한다. 우리는 §2 A3의 계약("init 시 RGB 경로가 정확히
  identity")을 **모델 출력 수준**까지 밀어 올린다: 보조 모달이 출력에 닿는 경로가
  γ(주입) 와 γ_pyr(헤드 공급) 둘뿐이고 둘 다 0에서 시작한다. 그래서
  `tools/smoke_p49.py` C(보조 입력을 바꿔도 init 출력 불변)가 성립한다.

  대가: γ=0 인 **step 0 한정**으로 보조 인코더(ConvNeXt)에는 task gradient가 0이다
  (γ 자신은 ∂L/∂γ = ⟨∂L/∂out, Δ⟩ ≠ 0 이라 즉시 움직인다 — 키1의 "흡수"와 다르다.
  흡수는 *대체 경로가 있어서* γ가 안 움직이는 현상인데, 여기엔 대체 경로가 없다).
  그래도 첫 스텝부터 보조 인코더를 세우고 싶으면 **VICReg(A5)** 를 켜라 — 보조
  feature에 직접 걸리는 손실이라 γ와 무관하게 gradient 출구가 된다. 기본 config는
  켜 둔다.

🔴 deformable attention
  `P49.DEFORM: true` 는 **미구현**이고 명시적으로 죽는다(조용한 fallback 금지).
  MSDeformAttn은 CUDA 확장 컴파일 의존이 있어 서버별 기동 실패 리스크가 크다.
  기본은 vanilla multi-head cross-attn이고, K/V 토큰 수는 레벨별 그리드 상한
  (`KV_GRID`, 레벨마다 절반씩, 하한 `KV_GRID_FLOOR`)으로 잡는다. attention 행렬을
  실체화하지 않도록 `F.scaled_dot_product_attention`을 쓴다 — 1024² · ViT grid
  64×64(4096 쿼리) 기준 이게 없으면 메모리가 터진다.

🔴 위치 대응
  vanilla cross-attn은 deformable의 reference point가 없으므로, ViT 쿼리와 보조
  키에 **같은 정규화 좌표계**의 2D sin-cos PE를 (Q/K 투영 입력에만, DETR 관례)
  더한다. 해상도가 달라도 [0,64) 로 정규화하므로 레벨 간·모달 간 좌표가 맞는다.

🔴 gradient checkpointing
  `INJECT=true` 에서는 거부한다. 주입은 timm block 의 **forward hook** 으로 붙는데
  (encoder.py의 P43 tap과 같은 이유 — forward_features 가 pos-embed/RoPE 처리의
  단일 출처로 남아야 한다), 블록이 checkpoint 로 감싸이면 backward 재계산 때 훅이
  다시 돌아 보조 상태 텐서가 재계산 그래프에서 만들어진다. ISSUE-027과 같은 계열의
  조용한 오염이라 아예 막는다. 메모리는 BS1 + grad-accum 으로 맞춘다.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import p46 as P46
from .encoder import FrozenViTEncoder, MultiModalLoRAQKV, SimpleFPN
from .m2f_head import MaskQueryLiteHead
from .model import FPNSegHead


# ─────────────────────────────────────────────────────────────────────────────
# 공용 소품
# ─────────────────────────────────────────────────────────────────────────────

def _gn(dim: int) -> nn.Module:
    """GroupNorm(32, ·) 관례 (리포의 AuxDecoder/FPNSegHead/UniModalHead와 동일).

    encoder.LayerNorm2d 를 쓰지 않는 이유는 p47.py `_norm2d` docstring 참조 —
    파이썬으로 편 LN2d는 (B,C,H,W) 중간 텐서 3장을 autograd 그래프에 남긴다.
    """
    for g in (32, 16, 8, 4, 2, 1):
        if dim % g == 0:
            return nn.GroupNorm(g, dim)
    return nn.GroupNorm(1, dim)                      # 도달 불가 (g=1이 항상 나눔)


def sincos_2d(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """(h*w, dim) 2D sin-cos 위치 인코딩. 좌표를 [0,64)로 **정규화**한다.

    정규화가 핵심이다: ViT 그리드(예: 64×64)와 보조 레벨 그리드(32×32, 16×16 …)가
    해상도가 달라도 같은 좌표계 위에 놓여야 cross-attention이 "여기 근처"를 찾을 수
    있다. 절대 인덱스를 쓰면 레벨마다 다른 좌표계가 되어 대응이 깨진다.
    """
    if dim % 4 != 0:
        raise ValueError(f"sincos_2d needs dim % 4 == 0, got {dim}")
    d4 = dim // 4
    omega = torch.arange(d4, device=device, dtype=torch.float32) / float(d4)
    omega = 1.0 / (10000.0 ** omega)
    y = torch.arange(h, device=device, dtype=torch.float32) / max(h - 1, 1) * 64.0
    x = torch.arange(w, device=device, dtype=torch.float32) / max(w - 1, 1) * 64.0
    oy = y[:, None] * omega[None, :]                                  # (h, d4)
    ox = x[:, None] * omega[None, :]                                  # (w, d4)
    pe_y = torch.cat([oy.sin(), oy.cos()], dim=1)                     # (h, 2*d4)
    pe_x = torch.cat([ox.sin(), ox.cos()], dim=1)                     # (w, 2*d4)
    pe = torch.cat([pe_y[:, None, :].expand(h, w, 2 * d4),
                    pe_x[None, :, :].expand(h, w, 2 * d4)], dim=-1)   # (h, w, dim)
    return pe.reshape(h * w, dim).to(dtype)


class _CrossAttn(nn.Module):
    """multi-head cross-attention. pos는 Q/K 투영 입력에만 더한다(DETR 관례).

    `F.scaled_dot_product_attention`을 쓴다 — (4096 쿼리 × 수천 키)에서 attention
    행렬을 실체화하면 1024² 학습이 성립하지 않는다.
    """

    def __init__(self, q_dim: int, kv_dim: int, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"P49 attn dim {dim} % heads {num_heads} != 0")
        self.h = int(num_heads)
        self.dh = dim // int(num_heads)
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.to_q = nn.Linear(q_dim, dim, bias=False)
        self.to_k = nn.Linear(kv_dim, dim, bias=False)
        self.to_v = nn.Linear(kv_dim, dim, bias=False)
        self.proj = nn.Linear(dim, q_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                q_pos: Optional[torch.Tensor] = None,
                k_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Nq, _ = q.shape
        Nk = kv.shape[1]
        qn = self.norm_q(q)
        kn = self.norm_kv(kv)
        qh = qn if q_pos is None else qn + q_pos.to(qn.dtype)
        kh = kn if k_pos is None else kn + k_pos.to(kn.dtype)
        qq = self.to_q(qh).view(B, Nq, self.h, self.dh).transpose(1, 2)
        kk = self.to_k(kh).view(B, Nk, self.h, self.dh).transpose(1, 2)
        vv = self.to_v(kn).view(B, Nk, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(qq, kk, vv)
        o = o.transpose(1, 2).reshape(B, Nq, self.h * self.dh)
        return self.proj(o)


class Injector(nn.Module):
    """[A3] 보조 → ViT. 출력은 **게이트 없는 raw Δ** 다 (γ는 모델이 곱한다).

    γ를 여기 두지 않는 이유: γ는 블록·모달별 스칼라 하나이고 ep30 게이트의 관측
    대상이라, 모델이 (nb, M) 파라미터 한 장으로 들고 있어야 로깅/판정이 단순하다.
    """

    def __init__(self, vit_dim: int, aux_dim: int, dim: int, num_heads: int):
        super().__init__()
        self.attn = _CrossAttn(vit_dim, aux_dim, dim, num_heads)

    def forward(self, vit_tok, aux_tok, vit_pos=None, aux_pos=None):
        return self.attn(vit_tok, aux_tok, q_pos=vit_pos, k_pos=aux_pos)


class Extractor(nn.Module):
    """[A3] ViT → 보조. cross-attn + FFN, 둘 다 일반 잔차(게이트 없음).

    여기 zero-init을 걸지 않는 이유: extractor의 출력은 보조 branch 안에만 머물고,
    모델 출력에 닿을 때 γ_pyr 게이트를 한 번 더 통과한다. 게이트를 두 번 겹치면
    ∂L/∂γ 자체가 0이 되어 **γ가 못 움직인다**(진짜 키1 재발).
    """

    def __init__(self, aux_dim: int, vit_dim: int, dim: int, num_heads: int,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = _CrossAttn(aux_dim, vit_dim, dim, num_heads)
        self.norm = nn.LayerNorm(aux_dim)
        hid = max(1, int(aux_dim * mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(aux_dim, hid), nn.GELU(),
                                 nn.Linear(hid, aux_dim))

    def forward(self, aux_tok, vit_tok, aux_pos=None, vit_pos=None):
        a = aux_tok + self.attn(aux_tok, vit_tok, q_pos=aux_pos, k_pos=vit_pos)
        return a + self.mlp(self.norm(a))


# ─────────────────────────────────────────────────────────────────────────────
# A1 · RGB 주경로 인코더
# ─────────────────────────────────────────────────────────────────────────────

class P49ViTEncoder(nn.Module):
    """DINOv3 ViT-L. `rgb_ft` 면 백본 전체 trainable(LoRA 없음), 아니면 frozen+LoRA.

    블록 경계 주입은 **forward hook**으로 붙인다 — `forward_features` 가
    pos-embed/RoPE/prefix-token 처리의 단일 출처로 남아야 하기 때문(encoder.py의
    P43 tap과 같은 논거). 훅은 `boundary_fn` 이 주어진 forward 에서만 동작하고,
    없으면 출력을 그대로 통과시킨다(= 주입 off 경로가 바이트 동일).

    layer-wise LR decay 계약: 백본 파라미터 이름이 `backbone.blocks.<i>.…` 로
    드러나므로 train_reliadino.py 의 `_p49_llrd_groups` 가 `<i>` 를 읽어
    깊이별 그룹을 만든다. 이 이름 규약을 바꾸지 말 것.
    """

    def __init__(self,
                 backbone: str = 'vit_large_patch16_dinov3',
                 fallback: str = 'vit_large_patch14_reg4_dinov2',
                 pretrained: bool = True,
                 img_size: int = 1024,
                 rgb_ft: bool = True,
                 lora_r: int = 8,
                 lora_alpha: Optional[float] = None,
                 num_boundaries: int = 4):
        super().__init__()
        import timm  # local import: keeps semseg.models importable without timm

        self.backbone_name, self.backbone = FrozenViTEncoder._create(
            timm, backbone, fallback, pretrained, img_size)
        self.patch = self.backbone.patch_embed.patch_size[0]
        self.embed_dim = self.backbone.embed_dim
        self.num_prefix_tokens = getattr(self.backbone, 'num_prefix_tokens', 0)
        self.depth = len(self.backbone.blocks)
        self.rgb_ft = bool(rgb_ft)

        self.lora_layers: List[MultiModalLoRAQKV] = []
        if self.rgb_ft:
            # [A1] 백본 전체 학습 가능. MM-SA Tab.4 — frozen+LoRA 49.77 은 RGB-only
            # 53.32 보다도 낮은 **최약 구성**이고 full FT 는 57.14 다.
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            # ablation 팔: 기존 P34~P47 계보와 같은 frozen + LoRA (모달 1개 = RGB).
            for p in self.backbone.parameters():
                p.requires_grad = False
            for blk in self.backbone.blocks:
                attn = blk.attn
                assert getattr(attn, 'qkv', None) is not None, \
                    f"{self.backbone_name}: block attn has no fused qkv — unsupported"
                if hasattr(attn, 'qkv_bias_separate') and getattr(attn, 'q_bias', None) is not None:
                    attn.qkv_bias_separate = True
                wrapper = MultiModalLoRAQKV(attn.qkv, 1, lora_r, lora_alpha)
                attn.qkv = wrapper
                self.lora_layers.append(wrapper)

        # 블록 경계 = 깊이를 num_boundaries 등분한 마지막 블록들 (24층/4 -> 5,11,17,23)
        nb = max(1, int(num_boundaries))
        self.boundaries: List[int] = sorted({
            max(0, min(self.depth - 1, (i + 1) * self.depth // nb - 1))
            for i in range(nb)})
        self._boundary_fn = None
        for bi, li in enumerate(self.boundaries):
            self.backbone.blocks[li].register_forward_hook(self._mk_hook(bi))

    def _mk_hook(self, bi: int):
        def hook(_module, _inp, out):
            fn = self._boundary_fn
            if fn is None:
                return None                       # 주입 off: 출력 그대로
            t = out[0] if isinstance(out, tuple) else out
            new = fn(bi, t)
            if new is None:
                return None
            return ((new,) + tuple(out[1:])) if isinstance(out, tuple) else new
        return hook

    def grid(self, H: int, W: int) -> Tuple[int, int, int, int]:
        """입력 (H,W) -> (Hp, Wp, h, w). patch 배수로 내림한 뒤의 토큰 그리드."""
        Hp = (H // self.patch) * self.patch
        Wp = (W // self.patch) * self.patch
        return Hp, Wp, Hp // self.patch, Wp // self.patch

    def set_grad_checkpointing(self, enable: bool = True):
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)

    def forward(self, x: torch.Tensor, boundary_fn=None) -> torch.Tensor:
        """x: (B,3,H,W) -> (B, embed_dim, h, w)."""
        if not self.rgb_ft and self.lora_layers:
            for wgt in self.lora_layers:
                wgt.active_modality = 0
        H, W = x.shape[-2:]
        Hp, Wp, h, w = self.grid(H, W)
        if (Hp, Wp) != (H, W):                    # patch14 fallback 대비
            x = F.interpolate(x, size=(Hp, Wp), mode='bilinear', align_corners=False)
        self._boundary_fn = boundary_fn
        try:
            tokens = self.backbone.forward_features(x)
        finally:
            self._boundary_fn = None
        if tokens.dim() == 4:                     # NHWC map (dynamic_img_size 경로)
            return tokens.permute(0, 3, 1, 2).contiguous()
        tokens = tokens[:, tokens.shape[1] - h * w:]
        B, N, C = tokens.shape
        assert N == h * w, f"token count {N} != grid {h}x{w}"
        return tokens.transpose(1, 2).reshape(B, C, h, w)


# ─────────────────────────────────────────────────────────────────────────────
# A2 · 보조 모달 인코더 (모달별 독립)
# ─────────────────────────────────────────────────────────────────────────────

def _create_timm(name: str, pretrained: bool, **kw):
    """timm create_model + pretrained 실패 시 경고 후 random init (encoder._create 관례)."""
    import timm
    try:
        return timm.create_model(name, pretrained=pretrained, **kw)
    except Exception as e:
        if not pretrained:
            raise
        warnings.warn(f"[P49] create_model('{name}', pretrained=True) 실패: {e} — "
                      f"RANDOM INIT 로 폴백한다. 본학습을 이 상태로 돌리지 말 것.")
        return timm.create_model(name, pretrained=False, **kw)


class AuxCNNEncoder(nn.Module):
    """[A2] 모달 1개용 ConvNeXt-S. 4 스테이지(stride 4/8/16/32) → 공통 dim 투영."""

    def __init__(self, dim: int, backbone: str = 'convnext_small',
                 pretrained: bool = True):
        super().__init__()
        self.net = _create_timm(backbone, pretrained, features_only=True,
                                out_indices=(0, 1, 2, 3))
        chs = list(self.net.feature_info.channels())
        self.strides = list(self.net.feature_info.reduction())
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, dim, 1, bias=False), _gn(dim)) for c in chs])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [p(f) for p, f in zip(self.proj, self.net(x))]


class AuxStemEncoder(nn.Module):
    """[A2 ablation] scratch conv prior (ViT-Adapter SPM 계열). pretrained 없음."""

    def __init__(self, dim: int, width: int = 64):
        super().__init__()
        def blk(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                _gn(cout), nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                _gn(cout), nn.GELU())
        self.s4 = nn.Sequential(blk(3, width, 2), blk(width, width, 2))
        self.s8 = blk(width, width * 2, 2)
        self.s16 = blk(width * 2, width * 4, 2)
        self.s32 = blk(width * 4, width * 4, 2)
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, dim, 1, bias=False), _gn(dim))
            for c in (width, width * 2, width * 4, width * 4)])
        self.strides = [4, 8, 16, 32]

    def forward(self, x):
        c1 = self.s4(x)
        c2 = self.s8(c1)
        c3 = self.s16(c2)
        c4 = self.s32(c3)
        return [p(f) for p, f in zip(self.proj, (c1, c2, c3, c4))]


class AuxViTLoRAEncoder(nn.Module):
    """[A2 ablation] 기존 계보와 동일한 frozen ViT + per-modality LoRA.

    보조 모달 **전체가 공유**하는 별도 ViT 한 벌이다(주경로 백본과 분리 — 주경로는
    RGB_FT 로 풀려 있어 같은 인스턴스를 공유하면 LoRA/full-FT 가 섞인다).
    stride-16 단일 맵을 리샘플해 4 스케일로 편다.
    """

    def __init__(self, dim: int, num_modalities: int,
                 backbone: str = 'vit_small_patch16_dinov3',
                 fallback: str = 'vit_small_patch16_224',
                 pretrained: bool = True, img_size: int = 1024,
                 lora_r: int = 8, lora_alpha: Optional[float] = None):
        super().__init__()
        import timm
        self.backbone_name, self.backbone = FrozenViTEncoder._create(
            timm, backbone, fallback, pretrained, img_size)
        self.patch = self.backbone.patch_embed.patch_size[0]
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lora_layers: List[MultiModalLoRAQKV] = []
        for blk in self.backbone.blocks:
            attn = blk.attn
            if hasattr(attn, 'qkv_bias_separate') and getattr(attn, 'q_bias', None) is not None:
                attn.qkv_bias_separate = True
            wrapper = MultiModalLoRAQKV(attn.qkv, num_modalities, lora_r, lora_alpha)
            attn.qkv = wrapper
            self.lora_layers.append(wrapper)
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(self.backbone.embed_dim, dim, 1, bias=False), _gn(dim))
            for _ in range(4)])
        self.strides = [4, 8, 16, 32]
        self.modality_idx = 0

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        for wgt in self.lora_layers:
            wgt.active_modality = self.modality_idx
        H, W = x.shape[-2:]
        Hp, Wp = (H // self.patch) * self.patch, (W // self.patch) * self.patch
        if (Hp, Wp) != (H, W):
            x = F.interpolate(x, size=(Hp, Wp), mode='bilinear', align_corners=False)
        t = self.backbone.forward_features(x)
        h, w = Hp // self.patch, Wp // self.patch
        if t.dim() == 4:
            m = t.permute(0, 3, 1, 2).contiguous()
        else:
            t = t[:, t.shape[1] - h * w:]
            m = t.transpose(1, 2).reshape(t.shape[0], -1, h, w)
        out = []
        for i, s in enumerate(self.strides):
            size = (max(1, Hp // s), max(1, Wp // s))
            f = m if size == m.shape[-2:] else F.interpolate(
                m, size=size, mode='bilinear', align_corners=False)
            out.append(self.proj[i](f))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# A5 · 보조 feature VICReg (γ=0 구간의 gradient 출구)
# ─────────────────────────────────────────────────────────────────────────────

def vicreg_var_cov(feats: Sequence[torch.Tensor], weights: Sequence[float],
                   l_var: float = 0.1, l_cov: float = 0.01,
                   tokens: int = 2048) -> torch.Tensor:
    """[P39.1-R2 계승] per-modal 토큰의 variance+covariance 항. λ 적용 후 반환.

    model.ReliaDINO._vicreg_loss 와 같은 식이다(거기 것은 ReliaDINO 인스턴스
    상태에 묶인 메서드라 재사용 불가 — 수식만 옮기고 상태 의존을 끊었다).
    autocast(bf16) 아래에서 covariance 행렬곱이 bf16이 되지 않도록 fp32 강제.
    """
    total = feats[0].new_zeros((), dtype=torch.float32)
    with torch.autocast(device_type=feats[0].device.type, enabled=False):
        for f, w in zip(feats, weights):
            if w <= 0:
                continue
            z = f.flatten(2).transpose(1, 2).reshape(-1, f.shape[1]).float()
            M = z.shape[0]
            k = min(int(tokens), M)
            if k < M:
                z = z[torch.randint(0, M, (k,), device=z.device)]
            z = z - z.mean(0, keepdim=True)
            lv = F.relu(1.0 - torch.sqrt(z.var(0) + 1e-4)).mean()
            C = (z.T @ z) / max(z.shape[0] - 1, 1)
            lc = (C.pow(2).sum() - C.diagonal().pow(2).sum()) / C.shape[0]
            total = total + w * (l_var * lv + l_cov * lc)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# P49-AIR
# ─────────────────────────────────────────────────────────────────────────────

class P49AIR(nn.Module):
    """Asymmetric Injection, RGB-primary.

    trainer 계약은 ReliaDINO 와 동일하다:
      training : forward(inputs, multimask_output, gt_mask) -> (logits, m_feat, aux)
      eval     : forward(inputs, multimask_output)          -> (logits, m_feat)
    aux 항은 전부 **pre-scaled** (리포 관례).
    """

    def __init__(self,
                 num_classes: int = 25,
                 modalities: Sequence[str] = ('img', 'depth', 'event', 'lidar'),
                 backbone: str = 'vit_large_patch16_dinov3',
                 backbone_fallback: str = 'vit_large_patch14_reg4_dinov2',
                 pretrained: bool = True,
                 img_size: int = 1024,
                 # A1
                 rgb_ft: bool = True,
                 lora_r: int = 8,
                 lora_alpha: Optional[float] = None,
                 # A2
                 aux_encoder: str = 'convnext',
                 aux_backbone: str = 'convnext_small',
                 aux_pretrained: bool = True,
                 # A3
                 inject: bool = True,
                 deform: bool = False,
                 num_blocks: int = 4,
                 inj_dim: int = 256,
                 inj_attn_dim: int = 256,
                 inj_heads: int = 8,
                 inj_mlp_ratio: float = 4.0,
                 kv_grid: int = 32,
                 kv_grid_floor: int = 8,
                 # A4
                 ms_head: bool = True,
                 head_mode: str = 'pixel',
                 fpn_dim: int = 256,
                 m2f_enable: bool = True,
                 m2f_num_queries: int = 100,
                 m2f_num_layers: int = 6,
                 m2f_dim: int = 256,
                 m2f_num_heads: int = 8,
                 m2f_mlp_ratio: float = 2.0,
                 m2f_w_cls: float = 2.0,
                 m2f_w_bce: float = 5.0,
                 m2f_w_dice: float = 5.0,
                 m2f_no_obj_w: float = 0.1,
                 m2f_points: int = 12544,
                 m2f_deep_supervision: bool = True,
                 m2f_loss_w: float = 0.5,
                 m2f_anchored: bool = True,
                 m2f_point_quota: int = 256,
                 # A5
                 vicreg: bool = True,
                 vicreg_lvar: float = 0.1,
                 vicreg_lcov: float = 0.01,
                 vicreg_tokens: int = 2048,
                 vicreg_level: int = 2,
                 vicreg_lidar_w: float = 1.0,
                 vicreg_other_w: float = 0.25,
                 proto: bool = False,
                 proto_lambda: float = 0.2,
                 proto_ema: float = 0.999,
                 proto_temp: float = 0.1,
                 proto_pixels: int = 4096,
                 proto_warmup_ep: int = 5,
                 ignore_label: int = 255):
        super().__init__()
        if deform:
            raise NotImplementedError(
                "[P49] MODEL.P49.DEFORM=true 는 미구현이다. deformable attention은 "
                "MSDeformAttn CUDA 확장 컴파일에 의존해 서버별 기동 실패 위험이 크다. "
                "vanilla cross-attn(기본)으로 돌리거나, 확장을 붙인 뒤 이 자리에 "
                "구현하라 — 조용히 vanilla로 폴백하지 않는다.")
        self.modalities = list(modalities)
        self.num_modalities = len(self.modalities)
        self._img_idx = (self.modalities.index('img') if 'img' in self.modalities else 0)
        self.aux_idx = [i for i in range(self.num_modalities) if i != self._img_idx]
        self.aux_names = [self.modalities[i] for i in self.aux_idx]
        self.inject = bool(inject) and len(self.aux_idx) > 0
        self.ms_head = bool(ms_head)
        self.head_mode = str(head_mode).lower()
        if self.head_mode not in ('pixel', 'query'):
            raise ValueError(f"P49.HEAD_MODE must be pixel|query, got {head_mode!r}")
        if self.head_mode == 'query' and not m2f_enable:
            # 조용히 pixel 헤드로 떨어지면 "query 헤드를 평가했다"는 오판을 부른다.
            raise ValueError("P49.HEAD_MODE=query 는 P49.M2F.ENABLE 없이는 쓸 수 없다")
        self.ignore_label = int(ignore_label)
        self.inj_dim = int(inj_dim)
        self.kv_grid = int(kv_grid)
        self.kv_grid_floor = int(kv_grid_floor)

        # ── A1 RGB 주경로 ────────────────────────────────────────────────────
        self.encoder = P49ViTEncoder(
            backbone=backbone, fallback=backbone_fallback, pretrained=pretrained,
            img_size=img_size, rgb_ft=rgb_ft, lora_r=lora_r, lora_alpha=lora_alpha,
            num_boundaries=(num_blocks if self.inject else 1))
        D = self.encoder.embed_dim
        self.num_blocks = len(self.encoder.boundaries) if self.inject else 0

        # ── A2/A3 보조 인코더 + injector/extractor ───────────────────────────
        M = len(self.aux_idx)
        self.aux_enc: Optional[nn.ModuleList] = None
        self.injectors: Optional[nn.ModuleList] = None
        self.extractors: Optional[nn.ModuleList] = None
        self.gamma_inj: Optional[nn.Parameter] = None
        self.gamma_pyr: Optional[nn.Parameter] = None
        self.pyr_levels: List[int] = []
        if self.inject:
            aux_encoder = str(aux_encoder).lower()
            if aux_encoder == 'convnext':
                self.aux_enc = nn.ModuleList([
                    AuxCNNEncoder(self.inj_dim, aux_backbone, aux_pretrained)
                    for _ in range(M)])
            elif aux_encoder == 'stem':
                self.aux_enc = nn.ModuleList([
                    AuxStemEncoder(self.inj_dim) for _ in range(M)])
            elif aux_encoder == 'vit_lora':
                shared = AuxViTLoRAEncoder(self.inj_dim, M, pretrained=aux_pretrained,
                                           img_size=img_size, lora_r=lora_r,
                                           lora_alpha=lora_alpha)
                # 한 벌을 공유하되 모달 인덱스만 갈아끼운다 (LoRA가 모달별로 독립).
                self.aux_shared = shared
                self.aux_enc = None
            else:
                raise ValueError("P49.AUX_ENCODER must be convnext|stem|vit_lora, "
                                 f"got {aux_encoder!r}")
            self.aux_encoder_kind = aux_encoder
            self.injectors = nn.ModuleList([
                nn.ModuleList([Injector(D, self.inj_dim, inj_attn_dim, inj_heads)
                               for _ in range(M)])
                for _ in range(self.num_blocks)])
            self.extractors = nn.ModuleList([
                nn.ModuleList([Extractor(self.inj_dim, D, inj_attn_dim, inj_heads,
                                         inj_mlp_ratio)
                               for _ in range(M)])
                for _ in range(self.num_blocks)])
            # 🔴 zero-init 게이트 2종. 이 둘이 보조 모달이 출력에 닿는 **유일한** 경로다.
            self.gamma_inj = nn.Parameter(torch.zeros(self.num_blocks, M))
            self.pyr_levels = [0, 1, 2, 3] if self.ms_head else [2]
            self.gamma_pyr = nn.Parameter(torch.zeros(M, len(self.pyr_levels)))
            self.level_embed = nn.Parameter(torch.zeros(4, self.inj_dim))
            nn.init.trunc_normal_(self.level_embed, std=0.02)
            self.pyr_proj = nn.ModuleList([
                nn.ModuleList([nn.Conv2d(self.inj_dim, fpn_dim, 1)
                               for _ in self.pyr_levels])
                for _ in range(M)])
        else:
            self.aux_encoder_kind = 'none'

        # ── A4 피라미드 + 헤드 ───────────────────────────────────────────────
        self.fpn = SimpleFPN(D, fpn_dim)
        self.head = FPNSegHead(fpn_dim, num_classes)
        self.m2f = None
        if m2f_enable:
            self.m2f = MaskQueryLiteHead(
                dim=fpn_dim, fpn_dim=fpn_dim, num_classes=num_classes,
                num_queries=m2f_num_queries, num_layers=m2f_num_layers,
                dim_t=m2f_dim, num_heads=m2f_num_heads, mlp_ratio=m2f_mlp_ratio,
                beta_init=0.0, w_cls=m2f_w_cls, w_bce=m2f_w_bce, w_dice=m2f_w_dice,
                no_obj_w=m2f_no_obj_w, num_points=m2f_points,
                deep_supervision=m2f_deep_supervision,
                # [A4] 멀티스케일 공급: 4개 피라미드 레벨을 "모달"처럼 토큰 소스로
                # 준다. m2f_head의 attn-bias 타일링이 소스마다 **같은 토큰 수**를
                # 가정하므로 레벨들은 stride-16 그리드로 맞춰서 넣는다(아래 forward).
                use_modal_src=self.ms_head, num_modalities=4,
                anchored=m2f_anchored, point_quota=m2f_point_quota)
        self.m2f_loss_w = float(m2f_loss_w)

        # ── A5 손실 토글 ─────────────────────────────────────────────────────
        self.vicreg = bool(vicreg) and self.inject
        self.vicreg_lvar, self.vicreg_lcov = float(vicreg_lvar), float(vicreg_lcov)
        self.vicreg_tokens = int(vicreg_tokens)
        self.vicreg_level = int(vicreg_level)
        self.vicreg_w = [float(vicreg_lidar_w) if n == 'lidar' else float(vicreg_other_w)
                         for n in self.aux_names]
        self.p46_proto = None
        self.p46_proto_lambda = float(proto_lambda)
        self.p46_proto_warmup_ep = int(proto_warmup_ep)
        if proto:
            self.p46_proto = P46.PrototypeBank(
                num_classes, fpn_dim, momentum=proto_ema, temperature=proto_temp,
                pixels=proto_pixels, ignore_label=self.ignore_label)

        # ── trainer 호환 스텁 (ReliaDINO 전용 토글들 — P49에는 없다) ─────────
        # train_reliadino.py 가 getattr 로 읽는 플래그들. False/None 로 고정해
        # 그 로깅·분기 전체가 조용히 꺼지게 한다. `fusion` 은 **일부러 정의하지
        # 않는다** — 트레이너가 `getattr(_core, 'fusion', None)` 로 유무를 본다.
        self.modal_dropout = False
        self.p42_mask_img = False
        self.p44_local_mask = False
        self.p44_hard_pixel_aux = False
        self.rca_enable = False
        self.p45_fogstyle = False
        self.p391_vicreg = False
        self.p41_fcr = False
        self.p47_2 = None
        self.classtoken = None
        self.arb_lambda = None
        self.trunk_gamma = None
        self.p43 = None
        self._current_epoch = 0
        self._p46_replay_path = False
        self._last_p42_mask = None
        self._last_p44_mask = None
        self._rca_pick = None
        self._last_per_modal_feats = None
        self._last_fused_prehead = None

        # 런타임 캐시 (파라미터 아님)
        self._pe_cache: Dict = {}
        self._grid: Tuple[int, int] = (0, 0)
        self._aux_seq: List = []
        self._aux_shapes: List[Tuple[int, int]] = []
        self._aux_pos: Optional[torch.Tensor] = None
        self._aux_raw: List[List[torch.Tensor]] = []

    # ── 위치 인코딩 캐시 ─────────────────────────────────────────────────────
    def _pe(self, h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
        key = (h, w, dim, str(device), str(dtype))
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = sincos_2d(h, w, dim, device, dtype)
            self._pe_cache[key] = pe
        return pe

    # ── 보조 토큰 pack/unpack ────────────────────────────────────────────────
    def _kv_cap(self, level: int) -> int:
        return max(self.kv_grid_floor, self.kv_grid >> level)

    def _pool_levels(self, maps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """레벨별 그리드 상한. vanilla attention의 K/V 예산을 여기서 잡는다."""
        out = []
        for i, f in enumerate(maps):
            cap = self._kv_cap(i)
            h, w = f.shape[-2:]
            th, tw = min(h, cap), min(w, cap)
            out.append(f if (th, tw) == (h, w) else F.adaptive_avg_pool2d(f, (th, tw)))
        return out

    def _pack(self, pooled: Sequence[torch.Tensor]) -> torch.Tensor:
        toks = [f.flatten(2).transpose(1, 2) + self.level_embed[i].to(f.dtype)
                for i, f in enumerate(pooled)]
        return torch.cat(toks, dim=1)

    def _unpack(self, tok: torch.Tensor) -> List[torch.Tensor]:
        out, s = [], 0
        for (h, w) in self._aux_shapes:
            n = h * w
            out.append(tok[:, s:s + n].transpose(1, 2).reshape(tok.shape[0], -1, h, w))
            s += n
        return out

    def _aux_pos_emb(self, device, dtype) -> torch.Tensor:
        return torch.cat([self._pe(h, w, self.inj_dim, device, dtype)
                          for (h, w) in self._aux_shapes], dim=0)

    # ── A3 블록 경계 훅 ──────────────────────────────────────────────────────
    def _boundary_fn(self, bi: int, tokens: torch.Tensor) -> torch.Tensor:
        h, w = self._grid
        n = h * w
        npre = tokens.shape[1] - n
        pre, sp = tokens[:, :npre], tokens[:, npre:]
        vpos = self._pe(h, w, self.encoder.embed_dim, sp.device, sp.dtype)
        apos = self._aux_pos
        a_prev = self._aux_seq[bi]

        delta = None
        for m in range(len(a_prev)):
            d = self.injectors[bi][m](sp, a_prev[m], vit_pos=vpos, aux_pos=apos)
            # γ 는 fp32 파라미터다. `γ * d` 로 두면 타입 승격으로 트렁크 전체가
            # fp32로 새어나가 AMP가 무력화된다 → d 의 dtype 으로 캐스팅해 곱한다.
            d = d * self.gamma_inj[bi, m].to(d.dtype)
            delta = d if delta is None else delta + d
        new_sp = sp + delta

        # extractor 는 **갱신된** ViT 토큰을 읽는다(주입 결과를 보조가 다시 본다).
        # `_aux_seq[bi+1]` 는 인덱스 대입이라 checkpoint 재계산에도 멱등하다.
        self._aux_seq[bi + 1] = [
            self.extractors[bi][m](a_prev[m], new_sp, aux_pos=apos, vit_pos=vpos)
            for m in range(len(a_prev))]
        return torch.cat([pre, new_sp], dim=1) if npre > 0 else new_sp

    # ── 진단 ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def gamma_log(self) -> Dict[str, float]:
        """ep30 게이트①용 — γ 노름. 키는 `p49/gamma_<block>_<modal>`."""
        out: Dict[str, float] = {}
        if not self.inject:
            return out
        g = self.gamma_inj.detach().abs().float().cpu()
        for b in range(g.shape[0]):
            for m, name in enumerate(self.aux_names):
                out[f'p49/gamma_b{b}_{name}'] = float(g[b, m])
        gp = self.gamma_pyr.detach().abs().float().cpu()
        for m, name in enumerate(self.aux_names):
            out[f'p49/gammapyr_{name}'] = float(gp[m].mean())
        out['p49/gamma_mean'] = float(g.mean())
        return out

    def set_grad_checkpointing(self, enable: bool = True):
        if enable and self.inject:
            print("[P49-AIR] 🔴 GRADIENT_CHECKPOINT=true 거부 → off "
                  "(블록 경계 주입 훅의 보조 상태가 backward 재계산과 충돌 — "
                  "ISSUE-027 계열). 메모리는 BATCH_SIZE/accum 으로 맞춰라.")
            enable = False
        self.encoder.set_grad_checkpointing(enable)

    # ── forward ─────────────────────────────────────────────────────────────
    def _encode_aux(self, x: Sequence[torch.Tensor]) -> None:
        """보조 모달 인코딩 → pooled 멀티스케일 토큰(모달별)으로 pack."""
        self._aux_raw = []
        packed = []
        self._aux_shapes = []
        for j, mi in enumerate(self.aux_idx):
            if self.aux_enc is not None:
                maps = self.aux_enc[j](x[mi])
            else:
                self.aux_shared.modality_idx = j
                maps = self.aux_shared(x[mi])
            self._aux_raw.append(maps)
            pooled = self._pool_levels(maps)
            if not self._aux_shapes:
                self._aux_shapes = [tuple(f.shape[-2:]) for f in pooled]
            packed.append(self._pack(pooled))
        dev, dt = packed[0].device, packed[0].dtype
        self._aux_pos = self._aux_pos_emb(dev, dt)
        self._aux_seq = [packed] + [None] * self.num_blocks

    def _build_pyramid(self, vit_map: torch.Tensor) -> List[torch.Tensor]:
        pyr = list(self.fpn(vit_map))            # [s4, s8, s16, s32] @ fpn_dim
        if not self.inject:
            return pyr
        final = self._aux_seq[self.num_blocks]
        for m in range(len(final)):
            maps = self._unpack(final[m])
            for k, lvl in enumerate(self.pyr_levels):
                c = self.pyr_proj[m][k](maps[lvl])
                if c.shape[-2:] != pyr[lvl].shape[-2:]:
                    c = F.interpolate(c, size=pyr[lvl].shape[-2:], mode='bilinear',
                                      align_corners=False)
                pyr[lvl] = pyr[lvl] + c * self.gamma_pyr[m, k].to(c.dtype)
        return pyr

    def forward(self, batched_input: List[torch.Tensor],
                multimask_output: bool = True,
                gt_mask: Optional[torch.Tensor] = None):
        # `multimask_output` 은 SAM2 계보 호출부 호환용 (미사용).
        x = batched_input
        img = x[self._img_idx]
        H, W = img.shape[-2:]
        _, _, h, w = self.encoder.grid(H, W)
        self._grid = (h, w)

        self._aux_raw = []
        if self.inject:
            self._encode_aux(x)
        vit_map = self.encoder(img, self._boundary_fn if self.inject else None)
        pyr = self._build_pyramid(vit_map)

        logits_s4, m_feat = self.head(pyr)        # (B,K,H/4,W/4), (B,fpn_dim,H/4,W/4)

        aux: Dict[str, torch.Tensor] = {}
        m2f_out = None
        run_m2f = self.m2f is not None and (self.training or self.head_mode == 'query')
        if run_m2f:
            modal_feats = None
            if self.ms_head:
                # 레벨을 stride-16 그리드로 맞춰 4개 토큰 소스로 준다
                # (m2f_head._attn_bias 타일링이 소스별 동일 토큰 수를 요구).
                tgt = pyr[2].shape[-2:]
                modal_feats = [p if p.shape[-2:] == tgt else F.interpolate(
                    p, size=tgt, mode='bilinear', align_corners=False) for p in pyr]
            m2f_out = self.m2f(pyr[2], m_feat, modal_feats=modal_feats)
            if self.head_mode == 'query':
                logits_s4 = self.m2f.semantic_scores(m2f_out)
            if self.training and gt_mask is not None:
                # 🔴 독립 주손실이다 — logits 에 더하지 않는다. P39 arbiter/path
                # dropout(경쟁 경로)은 P49에서 쓰지 않는다(H9: 순기여 −0.09).
                aux['m2f_loss'] = self.m2f_loss_w * self.m2f.losses(
                    m2f_out, m_feat, gt_mask)

        if self.training and self.inject and self.vicreg:
            lvl = max(0, min(3, self.vicreg_level))
            aux['vicreg'] = vicreg_var_cov(
                [mm[lvl] for mm in self._aux_raw], self.vicreg_w,
                self.vicreg_lvar, self.vicreg_lcov, self.vicreg_tokens)
        if (self.p46_proto is not None and self.training and gt_mask is not None
                and self._current_epoch >= self.p46_proto_warmup_ep):
            aux['p46_proto'] = self.p46_proto_lambda * self.p46_proto(
                m_feat, gt_mask, update=(not self._p46_replay_path))

        if not self.training:
            self._last_fused_prehead = pyr[2].detach()
            # 분석 훅(train_reliadino.py 의 eff.rank 모니터 / tools/*)은 이 리스트를
            # **DATASET.MODALS 순서**로 읽는다 → img 자리엔 ViT 토큰맵을, 보조
            # 자리엔 그 모달 인코더의 stride-16 레벨을 넣어 인덱스를 맞춘다.
            pmf = [None] * self.num_modalities
            pmf[self._img_idx] = vit_map.detach()
            for j, mi in enumerate(self.aux_idx):
                if j < len(self._aux_raw):
                    pmf[mi] = self._aux_raw[j][2].detach()
            self._last_per_modal_feats = ([f for f in pmf if f is not None]
                                          if self._aux_raw else [vit_map.detach()])

        logits = F.interpolate(logits_s4.float(), size=(H, W), mode='bilinear',
                               align_corners=False)
        # 그래프를 붙들지 않도록 스텝 캐시를 끊는다 (model.forward 말미 관례 —
        # 파이썬 지역/인스턴스 참조가 살아 있으면 다음 스텝 peak 위에 얹힌다).
        self._aux_seq = []
        self._aux_pos = None
        self._aux_raw = []
        if self.training:
            return logits, m_feat, aux
        return logits, m_feat


# ─────────────────────────────────────────────────────────────────────────────
# config -> 모델
# ─────────────────────────────────────────────────────────────────────────────

def build_p49(cfg: dict, num_classes: int) -> P49AIR:
    """`MODEL.P49.ENABLE: true` 인 config를 P49AIR 로 매핑.

    `model.build_reliadino` 가 이 함수로 분기한다 (그 분기는 P49 키가 없으면
    존재하지 않는 것과 같아, 기존 모델 경로는 완전 무변경이다).
    """
    mc = cfg['MODEL']
    p49 = mc.get('P49', {}) or {}
    m2f = p49.get('M2F', {}) or {}
    vic = p49.get('VICREG', {}) or {}
    # C-3 prototype 은 기존 P46 키를 그대로 재사용한다 (토글 호환 계약).
    c3 = ((mc.get('P46', {}) or {}).get('C3_PROTO', {}) or {})
    modals = cfg['DATASET']['MODALS']
    img_size = cfg['TRAIN']['IMAGE_SIZE'][0]
    ignore_label = (cfg.get('DATASET', {}) or {}).get('IGNORE_LABEL', 255)

    aux_kind = p49.get('AUX_ENCODER', None)
    if aux_kind is None:
        aux_kind = 'convnext' if p49.get('AUX_CNN', True) else 'vit_lora'

    if c3.get('ENABLE', False) and str(c3.get('FEATURE', 'mfeat')).lower() != 'mfeat':
        raise ValueError("[P49] P46.C3_PROTO.FEATURE 는 P49에서 'mfeat'만 지원한다 "
                         "(P49에는 ReliaDINO의 'fused' 텐서가 없다).")

    return P49AIR(
        num_classes=num_classes,
        modalities=modals,
        backbone=mc.get('BACKBONE_TIMM', 'vit_large_patch16_dinov3'),
        backbone_fallback=mc.get('BACKBONE_FALLBACK', 'vit_large_patch14_reg4_dinov2'),
        pretrained=mc.get('PRETRAINED_BACKBONE', True),
        img_size=img_size,
        rgb_ft=p49.get('RGB_FT', True),
        lora_r=mc.get('LORA_R', 8),
        lora_alpha=mc.get('LORA_ALPHA', None),
        aux_encoder=aux_kind,
        aux_backbone=p49.get('AUX_BACKBONE', 'convnext_small'),
        aux_pretrained=p49.get('AUX_PRETRAINED', mc.get('PRETRAINED_BACKBONE', True)),
        inject=p49.get('INJECT', True),
        deform=p49.get('DEFORM', False),
        num_blocks=p49.get('NUM_BLOCKS', 4),
        inj_dim=p49.get('DIM', 256),
        inj_attn_dim=p49.get('ATTN_DIM', p49.get('DIM', 256)),
        inj_heads=p49.get('NUM_HEADS', 8),
        inj_mlp_ratio=p49.get('MLP_RATIO', 4.0),
        kv_grid=p49.get('KV_GRID', 32),
        kv_grid_floor=p49.get('KV_GRID_FLOOR', 8),
        ms_head=p49.get('MS_HEAD', True),
        head_mode=p49.get('HEAD_MODE', 'pixel'),
        fpn_dim=mc.get('FPN_DIM', 256),
        m2f_enable=m2f.get('ENABLE', True),
        m2f_num_queries=m2f.get('NUM_QUERIES', 100),
        m2f_num_layers=m2f.get('NUM_LAYERS', 6),
        m2f_dim=m2f.get('DIM', 256),
        m2f_num_heads=m2f.get('NUM_HEADS', 8),
        m2f_mlp_ratio=m2f.get('MLP_RATIO', 2.0),
        m2f_w_cls=m2f.get('W_CLS', 2.0),
        m2f_w_bce=m2f.get('W_BCE', 5.0),
        m2f_w_dice=m2f.get('W_DICE', 5.0),
        m2f_no_obj_w=m2f.get('NO_OBJ_W', 0.1),
        m2f_points=m2f.get('POINTS', 12544),
        m2f_deep_supervision=m2f.get('DEEP_SUPERVISION', True),
        m2f_loss_w=m2f.get('LOSS_W', 0.5),
        m2f_anchored=m2f.get('ANCHORED', True),
        m2f_point_quota=m2f.get('POINT_QUOTA', 256),
        vicreg=vic.get('ENABLE', True),
        vicreg_lvar=vic.get('LVAR', 0.1),
        vicreg_lcov=vic.get('LCOV', 0.01),
        vicreg_tokens=vic.get('TOKENS', 2048),
        vicreg_level=vic.get('LEVEL', 2),
        vicreg_lidar_w=vic.get('LIDAR_W', 1.0),
        vicreg_other_w=vic.get('OTHER_W', 0.25),
        proto=c3.get('ENABLE', False),
        proto_lambda=c3.get('LAMBDA', 0.2),
        proto_ema=c3.get('EMA', 0.999),
        proto_temp=c3.get('TEMPERATURE', 0.1),
        proto_pixels=c3.get('PIXELS', 4096),
        proto_warmup_ep=c3.get('WARMUP_EP', 5),
        ignore_label=ignore_label,
    )
