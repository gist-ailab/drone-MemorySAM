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

    K is left untouched by default (card A: "per-modality LoRA(Q/V)"; also matches
    the SAM2 lineage where only q/v carry adapters). The active modality m is set
    by `FrozenViTEncoder.set_modality()` before each per-modality forward.

    [E2/S2] `include_k=True` adds a matching per-modality adapter on the key slice
    (target `qkv_full` = Q/K/V). K params (a_k/b_k) are created *only* when
    requested, so the default `include_k=False` path is byte-identical to the
    prior Q/V-only wrapper — same state_dict keys, same RNG consumption order
    (the K init loop below is skipped entirely).
    """

    def __init__(self, base: nn.Linear, num_modalities: int, r: int, alpha: float = None,
                 include_k: bool = False):
        super().__init__()
        assert base.out_features % 3 == 0, "expected fused qkv linear (out = 3*attn_dim)"
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.attn_dim = base.out_features // 3
        self.num_modalities = num_modalities
        self.r = r
        self.include_k = include_k
        self.scale = (alpha if alpha is not None else float(r)) / float(r)
        # (M, r, in) down-projections and (M, attn_dim, r) up-projections.
        self.a_q = nn.Parameter(torch.empty(num_modalities, r, self.in_features))
        self.b_q = nn.Parameter(torch.zeros(num_modalities, self.attn_dim, r))
        self.a_v = nn.Parameter(torch.empty(num_modalities, r, self.in_features))
        self.b_v = nn.Parameter(torch.zeros(num_modalities, self.attn_dim, r))
        for m in range(num_modalities):
            nn.init.kaiming_uniform_(self.a_q[m], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.a_v[m], a=math.sqrt(5))
        if include_k:  # [E2] key-slice adapter — params + RNG only when requested
            self.a_k = nn.Parameter(torch.empty(num_modalities, r, self.in_features))
            self.b_k = nn.Parameter(torch.zeros(num_modalities, self.attn_dim, r))
            for m in range(num_modalities):
                nn.init.kaiming_uniform_(self.a_k[m], a=math.sqrt(5))
        self.active_modality = 0
        # [P51] 배치-모달 경로용 per-element 모달 인덱스(길이 B의 LongTensor).
        # None 이면 기존 스칼라 active_modality 경로가 그대로 돈다(하위호환).
        # forward_coupled 가 심고 forward 후 반드시 None 으로 되돌린다.
        self.modality_ids: Optional[torch.Tensor] = None
        # [E0] LoRA delta on/off 토글. 기본 True 에서 forward 는 기존과 byte-동일하다
        # (아래 forward 첫 줄의 `if not self.enabled` 는 True 일 때 건너뛰므로 실행
        # 경로가 그대로다). False 면 어댑터 delta 를 전혀 더하지 않아 frozen 원
        # 백본 출력만 나온다 — 특징 정보 프로브(tools/probe_feature_info.py)의
        # `raw_taps` 세트를 얻는 데 쓴다. 파라미터가 아니라 state_dict 에 영향 없다.
        self.enabled: bool = True

    @property
    def weight(self):  # safety for code paths that touch qkv.weight directly
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            # [E0] 어댑터 비활성 — delta 없이 frozen 백본 qkv 출력을 그대로 반환.
            return y
        # [P51] modality_ids 가 있으면 배치 원소별로 다른 모달 LoRA 를 적용:
        # (M, r, in) 파라미터를 ids 로 gather 해 (B, r, in) 배치 weight 를 만들고
        # einsum 이 배치 행렬곱으로 돌린다(F.linear 는 2D weight 만 받는다) —
        # 스칼라 경로와 같은 선형 변환을 원소별로 수행한다(스모크 4가 수치
        # 일치를 검사한다).
        if self.modality_ids is not None:
            ids = self.modality_ids
            aq, bq = self.a_q[ids], self.b_q[ids]      # (B, r, in), (B, d, r)
            av, bv = self.a_v[ids], self.b_v[ids]
            dq = torch.einsum('bnr,bdr->bnd',
                              torch.einsum('bni,bri->bnr', x, aq), bq) * self.scale
            dv = torch.einsum('bnr,bdr->bnd',
                              torch.einsum('bni,bri->bnr', x, av), bv) * self.scale
            if self.include_k:  # [E2] key slice — same batch-modal gather as q/v
                ak, bk = self.a_k[ids], self.b_k[ids]
                dk = torch.einsum('bnr,bdr->bnd',
                                  torch.einsum('bni,bri->bnr', x, ak), bk) * self.scale
        else:
            m = self.active_modality
            aq, bq = self.a_q[m], self.b_q[m]
            av, bv = self.a_v[m], self.b_v[m]
            dq = F.linear(F.linear(x, aq), bq) * self.scale
            dv = F.linear(F.linear(x, av), bv) * self.scale
            if self.include_k:  # [E2] key slice
                dk = F.linear(F.linear(x, self.a_k[m]), self.b_k[m]) * self.scale
        d = self.attn_dim
        kmid = y[..., d:2 * d] + dk if self.include_k else y[..., d:2 * d]
        y = torch.cat([y[..., :d] + dq, kmid, y[..., 2 * d:] + dv], dim=-1)
        return y


class MultiModalLoRALinear(nn.Module):
    """[E2/S2] 임의 nn.Linear 에 붙는 센서별 LoRA 어댑터.

    attention 출력 projection(O = `attn.proj`) 과 MLP(`mlp.fc1`/`mlp.fc2`) 를
    LoRA 로 확장할 때 쓴다. `MultiModalLoRAQKV` 와 동일 규약
    (`active_modality`/`modality_ids`(P51 배치-모달)/`enabled`(E0 토글)/`scale`)
    을 따르되, qkv 처럼 슬라이스를 나누지 않고 출력 전체에 additive delta 를 더한다:

        y = base(x) + s · (x @ A[m].T) @ B[m].T
        A: (M, r, in),  B: (M, out, r),  B zero-init → 초기 delta=0 (base 와 동일).

    근거: SoMA(2412.04077) 표 8 — attn-only < MLP-only < 둘 다(+1.5);
    "LoRA Learns Less and Forgets Less"(2405.09673) — MLP 가 학습 주 loci, α=2r.
    """

    def __init__(self, base: nn.Linear, num_modalities: int, r: int, alpha: float = None):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.num_modalities = num_modalities
        self.r = r
        self.scale = (alpha if alpha is not None else float(r)) / float(r)
        # a_lin/b_lin: qkv 래퍼의 a_q/b_q 와 이름을 달리해, train_reliadino.py 의
        # LORA_NORM_CAP 수집 접미사(.b_lin)로 명시적으로 잡히게 한다.
        self.a_lin = nn.Parameter(torch.empty(num_modalities, r, self.in_features))
        self.b_lin = nn.Parameter(torch.zeros(num_modalities, self.out_features, r))
        for m in range(num_modalities):
            nn.init.kaiming_uniform_(self.a_lin[m], a=math.sqrt(5))
        self.active_modality = 0
        # [P51] 배치-모달 경로. forward_coupled 가 심고 forward 후 None 으로 되돌린다.
        self.modality_ids: Optional[torch.Tensor] = None
        # [E0] LoRA delta on/off 토글 — 기본 True, byte-동일.
        self.enabled: bool = True

    @property
    def weight(self):  # safety for code paths that touch .weight directly
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            # [E0] 어댑터 비활성 — delta 없이 frozen base 출력 그대로.
            return y
        if self.modality_ids is not None:
            ids = self.modality_ids
            a, b = self.a_lin[ids], self.b_lin[ids]     # (B, r, in), (B, out, r)
            d = torch.einsum('bnr,bor->bno',
                             torch.einsum('bni,bri->bnr', x, a), b) * self.scale
        else:
            m = self.active_modality
            d = F.linear(F.linear(x, self.a_lin[m]), self.b_lin[m]) * self.scale
        return y + d


class SharedLoRAQKV(nn.Module):
    """[E-LORA] LoRA 구조 ablation의 arm B/C용 qkv 래퍼.

    `MultiModalLoRAQKV`(arm A, per-modal)는 그대로 두고 옆에 새로 둔다 — arm A
    경로·기존 체크포인트는 이 클래스가 존재해도 전혀 영향을 받지 않는다.

    두 모드를 하나의 클래스가 담당한다(shared 는 residual_r=0 인 특수형):
      · shared          : y += s_s · x @ A_s.T @ B_s.T          (전 모달 공유, 단일 어댑터)
      · shared_residual : y += s_s · shared_delta(x)
                              + s_r · x @ A_r[m].T @ B_r[m].T    (공유 + 모달별 잔차)
    두 delta 모두 Q/V 슬라이스에만 additive 하며 K 는 건드리지 않는다(arm A 와
    동일 규약). shared delta 는 모달 무관이므로 스칼라 `active_modality` 경로와
    배치별 `modality_ids`(P51) 경로 모두에서 그대로 F.linear 로 계산된다. 잔차
    delta 만 modality_ids 를 gather 한다(arm A 의 배치-모달 패턴 그대로 재사용).
    """

    def __init__(self, base: nn.Linear, num_modalities: int, shared_r: int,
                 residual_r: int = 0, alpha: float = None):
        super().__init__()
        assert base.out_features % 3 == 0, "expected fused qkv linear (out = 3*attn_dim)"
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.attn_dim = base.out_features // 3
        self.num_modalities = num_modalities
        self.shared_r = shared_r
        self.residual_r = residual_r
        d = self.attn_dim
        # 공유 어댑터(모달 무관) — Q/V 각각. b_*_s zero-init 이라 init 시 delta=0.
        self.scale_s = (alpha if alpha is not None else float(shared_r)) / float(shared_r)
        self.a_q_s = nn.Parameter(torch.empty(shared_r, self.in_features))
        self.b_q_s = nn.Parameter(torch.zeros(d, shared_r))
        self.a_v_s = nn.Parameter(torch.empty(shared_r, self.in_features))
        self.b_v_s = nn.Parameter(torch.zeros(d, shared_r))
        nn.init.kaiming_uniform_(self.a_q_s, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.a_v_s, a=math.sqrt(5))
        # 모달별 잔차 어댑터(residual_r>0 일 때만) — arm A 와 같은 (M,r,in)/(M,d,r) 배열.
        if residual_r > 0:
            self.scale_r = (alpha if alpha is not None else float(residual_r)) / float(residual_r)
            self.a_q_r = nn.Parameter(torch.empty(num_modalities, residual_r, self.in_features))
            self.b_q_r = nn.Parameter(torch.zeros(num_modalities, d, residual_r))
            self.a_v_r = nn.Parameter(torch.empty(num_modalities, residual_r, self.in_features))
            self.b_v_r = nn.Parameter(torch.zeros(num_modalities, d, residual_r))
            for m in range(num_modalities):
                nn.init.kaiming_uniform_(self.a_q_r[m], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.a_v_r[m], a=math.sqrt(5))
        self.active_modality = 0
        # [P51] 배치-모달 경로용 per-element 모달 인덱스. 잔차 항만 사용한다.
        self.modality_ids: Optional[torch.Tensor] = None
        # [E0] LoRA delta on/off 토글 — MultiModalLoRAQKV 와 동일 규약. 기본 True.
        self.enabled: bool = True

    @property
    def weight(self):  # safety for code paths that touch qkv.weight directly
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled:
            # [E0] 어댑터 비활성 — 공유·잔차 delta 없이 frozen 백본 출력 그대로.
            return y
        # 공유 delta — 모달 무관이라 스칼라/배치 경로 공통(2D weight → F.linear).
        dq = F.linear(F.linear(x, self.a_q_s), self.b_q_s) * self.scale_s
        dv = F.linear(F.linear(x, self.a_v_s), self.b_v_s) * self.scale_s
        if self.residual_r > 0:
            if self.modality_ids is not None:
                ids = self.modality_ids
                aq, bq = self.a_q_r[ids], self.b_q_r[ids]      # (B, r, in), (B, d, r)
                av, bv = self.a_v_r[ids], self.b_v_r[ids]
                dq = dq + torch.einsum('bnr,bdr->bnd',
                                       torch.einsum('bni,bri->bnr', x, aq), bq) * self.scale_r
                dv = dv + torch.einsum('bnr,bdr->bnd',
                                       torch.einsum('bni,bri->bnr', x, av), bv) * self.scale_r
            else:
                m = self.active_modality
                dq = dq + F.linear(F.linear(x, self.a_q_r[m]), self.b_q_r[m]) * self.scale_r
                dv = dv + F.linear(F.linear(x, self.a_v_r[m]), self.b_v_r[m]) * self.scale_r
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
                 lora_mode: str = 'per_modal',       # [E-LORA] per_modal | shared | shared_residual
                 lora_shared_r: int = 8,             # [E-LORA] shared_residual 공유항 rank
                 lora_residual_r: int = 8,           # [E-LORA] shared_residual 모달별 잔차항 rank
                 lora_targets: Optional[Sequence[str]] = None,  # [E2] None → ['qkv'] (byte-동일)
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

        # [E-LORA] LoRA 구조 모드. per_modal(기본) 은 arm A = 기존 MultiModalLoRAQKV
        # 와 byte-동일해야 하므로, 이 분기 밖에서는 어떤 RNG 도 소비하지 않는다.
        self.lora_mode = lora_mode
        if lora_mode not in ('per_modal', 'shared', 'shared_residual'):
            raise ValueError(f"[E-LORA] unknown LORA_MODE '{lora_mode}' "
                             "(per_modal | shared | shared_residual)")

        # [E2/S2] LoRA 타깃 파싱. None → ['qkv'] = 현행(Q/V), byte-동일.
        #   qkv      = fused qkv 의 Q/V (현행)            qkv_full = Q/K/V
        #   proj     = attention 출력 O (attn.proj)
        #   fc1/fc2  = MLP 두 선형층 (mlp.fc1 / mlp.fc2)
        # 각 항은 이름 문자열이거나 [이름, rank] 2원소 시퀀스다. rank 를 생략하면
        # LORA_R 을 쓴다(E2b 는 fc1/fc2 만 낮은 rank 로 두어 24GB 여유를 확보한다).
        # LoRA scale(=α/r)은 전 타깃 균일하게 유지한다 — 타깃별 α = scale·r_t 로
        # 정해, rank 가 달라도 α=2r 규약이 그대로 지켜진다.
        _raw = list(lora_targets) if lora_targets else ['qkv']
        _allowed = {'qkv', 'qkv_full', 'proj', 'fc1', 'fc2'}
        target_names: List[str] = []
        target_r: dict = {}
        for item in _raw:
            if isinstance(item, str):
                nm, rk = item, lora_r
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                nm, rk = str(item[0]), int(item[1])
            else:
                raise ValueError(f"[E2] LORA_TARGETS 항 형식 오류 {item!r} — "
                                 "'name' 또는 [name, rank] 형식이어야 한다.")
            if nm not in _allowed:
                raise ValueError(f"[E2] 알 수 없는 LoRA 타깃 '{nm}' "
                                 f"(허용: {sorted(_allowed)})")
            if nm in target_r:
                raise ValueError(f"[E2] LORA_TARGETS 에 '{nm}' 이 중복된다.")
            target_names.append(nm)
            target_r[nm] = rk
        if 'qkv' in target_r and 'qkv_full' in target_r:
            raise ValueError("[E2] LORA_TARGETS 에 'qkv' 와 'qkv_full' 을 동시에 "
                             "줄 수 없다 — 둘 다 fused qkv 를 대상으로 한다.")
        self.lora_targets = target_names
        # 전역 scale = α/r (LORA_ALPHA=null → scale 1.0). 타깃별 α = scale·r_t.
        _scale = ((float(lora_alpha) if lora_alpha is not None else float(lora_r))
                  / float(lora_r))

        def _alpha_for(rk: int) -> float:
            return _scale * float(rk)

        qkv_target = ('qkv_full' if 'qkv_full' in target_r
                      else ('qkv' if 'qkv' in target_r else None))
        include_k = (qkv_target == 'qkv_full')
        _ext = [t for t in ('proj', 'fc1', 'fc2') if t in target_r]
        is_extended = include_k or bool(_ext)
        # shared/shared_residual arm 은 아직 qkv(Q/V) 확장을 구현하지 않았다 —
        # 확장 타깃과의 조합은 조용히 무시하지 말고 명시적으로 막는다.
        if is_extended and lora_mode != 'per_modal':
            raise NotImplementedError(
                f"[E2] LORA_MODE='{lora_mode}' 는 확장 타깃"
                f"({[t for t in target_names if t != 'qkv']}) 과 조합할 수 없다 — "
                "per_modal 만 qkv_full/proj/fc1/fc2 를 지원한다. shared 계열은 "
                "LORA_TARGETS: [qkv] 단독으로만 쓸 것.")

        def _make_lora_qkv(qkv: nn.Linear):
            rk = target_r[qkv_target]
            if lora_mode == 'per_modal':
                return MultiModalLoRAQKV(qkv, num_modalities, rk, _alpha_for(rk),
                                         include_k=include_k)
            if lora_mode == 'shared':
                # arm B: 전 모달이 하나의 공유 어댑터(rank=lora_r). 잔차 없음.
                return SharedLoRAQKV(qkv, num_modalities, lora_r, 0, lora_alpha)
            # arm C: 공유(rank=lora_shared_r) + 모달별 잔차(rank=lora_residual_r).
            return SharedLoRAQKV(qkv, num_modalities, lora_shared_r,
                                 lora_residual_r, lora_alpha)

        def _tag(w, t):  # 파라미터 수 로그(타깃별 집계)에 쓰는 꼬리표
            w._e2_target = t
            return w

        def _make_lin(lin: nn.Linear, name: str):
            rk = target_r[name]
            return _tag(MultiModalLoRALinear(lin, num_modalities, rk, _alpha_for(rk)), name)

        # Freeze everything, then inject trainable per-modality LoRA.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lora_layers: List[nn.Module] = []
        for blk in self.backbone.blocks:
            attn = blk.attn
            if qkv_target is not None:
                assert getattr(attn, 'qkv', None) is not None, \
                    f"{self.backbone_name}: block attn has no fused qkv — unsupported layout"
                # Eva path: force module-call so the wrapper forward is not bypassed
                # by F.linear(x, self.qkv.weight, bias) (see module docstring).
                if hasattr(attn, 'qkv_bias_separate') and getattr(attn, 'q_bias', None) is not None:
                    attn.qkv_bias_separate = True
                wrapper = _tag(_make_lora_qkv(attn.qkv), qkv_target)
                attn.qkv = wrapper
                self.lora_layers.append(wrapper)
            if 'proj' in target_r:
                proj = getattr(attn, 'proj', None)
                if not isinstance(proj, nn.Linear):
                    raise RuntimeError(
                        f"{self.backbone_name}: attn.proj 가 nn.Linear 가 아니다 "
                        f"({type(proj).__name__}) — 'proj' 타깃을 뺄 것.")
                wrapper = _make_lin(proj, 'proj')
                attn.proj = wrapper
                self.lora_layers.append(wrapper)
            mlp = getattr(blk, 'mlp', None)
            for name in ('fc1', 'fc2'):
                if name not in target_r:
                    continue
                lin = getattr(mlp, name, None) if mlp is not None else None
                if not isinstance(lin, nn.Linear):
                    raise RuntimeError(
                        f"{self.backbone_name}: mlp.{name} 이 nn.Linear 가 아니다 "
                        f"({type(lin).__name__}) — SwiGLU 등 fused-gate MLP 는 "
                        f"미지원. '{name}' 타깃을 뺄 것.")
                wrapper = _make_lin(lin, name)
                setattr(mlp, name, wrapper)
                self.lora_layers.append(wrapper)
        # register so .parameters()/.state_dict() see the adapters exactly once
        # (they already live under backbone.blocks[k].{attn.qkv,attn.proj,mlp.fcN} —
        # nothing extra needed; self.lora_layers is a plain python list on purpose).
        self._log_lora_param_counts()

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
            # [E1] 1-indexed 블록 번호를 resolve_taps 규약으로 해석: [1, n] clamp
            # 후 -1 (24블록에서 24=마지막 블록 23, 6→5). tools/probe_feature_info.py
            # 의 resolve_taps 와 동일하다. (구 `% n` 은 24→0 으로 마지막 블록을
            # 첫 블록으로 접어 규약과 어긋났고, 프로덕션에서 tap_layers 를 넘긴
            # 호출부가 없어 이 변경은 P43(num_taps 경로)에 영향이 없다.)
            self.tap_layers = sorted({min(max(int(t), 1), n) - 1 for t in tap_layers})
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

    def set_lora_enabled(self, flag: bool):
        """[E0] 전 블록의 LoRA 어댑터 delta 를 켜거나 끈다. flag=False 면 frozen
        원 백본(어댑터 이전) 특징이 나온다. 기본 상태(True)로 되돌리지 않으면
        이후 forward 가 오염되므로, 호출한 쪽이 반드시 원상복구해야 한다
        (tools/probe_feature_info.py 는 try/finally 로 감싼다)."""
        for w in self.lora_layers:
            w.enabled = flag

    def _log_lora_param_counts(self):
        """[E2] 기동 시 타깃별 LoRA 학습 파라미터 수를 rank0 에서 한 줄로 출력.
        model.py 의 `[E-LORA] ... lora_trainable=` 로그(qkv 만 셈)와 나란히 놓여,
        확장 타깃(proj/fc1/fc2)까지 포함한 정확한 분해를 보여준다."""
        rank0 = (not torch.distributed.is_initialized()
                 or torch.distributed.get_rank() == 0)
        if not rank0:
            return
        counts: dict = {}
        for w in self.lora_layers:
            t = getattr(w, '_e2_target', 'qkv')
            n = sum(p.numel() for name, p in w.named_parameters()
                    if not name.startswith('base.'))
            counts[t] = counts.get(t, 0) + n
        total = sum(counts.values())
        by = ', '.join(f"{k}={v:,}" for k, v in counts.items())
        print(f"[E2] lora_targets={self.lora_targets} lora_mode={self.lora_mode} "
              f"| by_target[{by}] | lora_params_total={total:,}")

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

    # ── [P51] CMLC: 배치-모달 단일 forward + block 경계 cross-modal 결합 ──────
    def _cmlc_hook(self, cmlc, point_idx: int, M: int, B: int):
        """couple block 출력 직후: (M*B,N,D) → (M,B,N,D) → cmlc → (M*B,N,D).

        forward hook 이 값을 반환하면 그 값이 block 출력을 대체한다 → 결합된
        토큰이 **다음 block 의 입력**이 된다(결합이 다운스트림 인코딩에 영향).
        """
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() != 3:
                raise RuntimeError(
                    f"[CMLC] couple block 출력이 (B,N,C) 3차원이어야 하는데 "
                    f"{tuple(t.shape)} — 결합을 적용할 수 없다 (조용한 skip 금지).")
            coupled = cmlc(t.view(M, B, t.shape[1], t.shape[2]),
                           point_idx=point_idx).view_as(t)
            return (coupled,) if isinstance(out, tuple) else coupled
        return hook

    def forward_coupled(self, x_stack: torch.Tensor, cmlc,
                        couple_block_idx: Sequence[int]) -> List[torch.Tensor]:
        """[P51] x_stack (M,B,C,H,W) → per-modal (B,C,h,w) 리스트.

        모달을 (M*B,C,H,W)로 쌓아 forward_features 를 **한 번만** 돌리고(LoRA 는
        modality_ids 로 배치 원소별 적용), couple_block_idx 각 block 출력 직후
        _cmlc_hook 이 저랭크 결합을 끼워 넣는다 — 순차 forward 대비 모달별
        forward 를 배치로 합쳐 결합 seam 을 제공하는 것이 전부라, 결합이 없으면
        (cmlc γ=0) 순차 forward 와 수치 등가여야 한다(스모크 4).
        last_taps 는 tap 버퍼의 모달 평균으로 채운다 — 순차 경로의
        Σ_m taps/M 와 같은 결정론적 reduction 이라 P43 lateral 계약이 유지된다.
        """
        M, B = x_stack.shape[0], x_stack.shape[1]
        H, W = x_stack.shape[-2:]
        Hp, Wp = (H // self.patch) * self.patch, (W // self.patch) * self.patch
        if (Hp, Wp) != (H, W):  # patch14 fallback: 순차 forward 와 같은 절차로 보간
            x_stack = F.interpolate(
                x_stack.reshape(M * B, *x_stack.shape[2:]), size=(Hp, Wp),
                mode='bilinear', align_corners=False).view(M, B, *x_stack.shape[2:])
        self.last_taps = None
        self._tap_buf.clear()
        ids = torch.arange(M, device=x_stack.device).repeat_interleave(B)
        for w in self.lora_layers:
            w.modality_ids = ids
        n = len(self.backbone.blocks)
        handles = [self.backbone.blocks[int(i) % n].register_forward_hook(
                       self._cmlc_hook(cmlc, j, M, B))
                   for j, i in enumerate(couple_block_idx)]
        try:
            tokens = self.backbone.forward_features(x_stack.reshape(M * B, *x_stack.shape[2:]))
        finally:
            for h_ in handles:
                h_.remove()
            for w in self.lora_layers:  # 순차 forward 오염 방지: 반드시 해제
                w.modality_ids = None
        h, w = Hp // self.patch, Wp // self.patch
        if self.tap_layers and self.collect_taps:
            self.last_taps = [
                self._to_map(
                    self._tap_buf[j].view(M, B, *self._tap_buf[j].shape[1:]).mean(0),
                    h, w)
                for j in range(len(self.tap_layers))]
        self._tap_buf.clear()
        if tokens.dim() == 4:  # some timm models return NHWC maps (same branch as forward)
            maps = tokens.permute(0, 3, 1, 2).contiguous()
            return [t.contiguous() for t in maps.view(M, B, *maps.shape[1:]).unbind(0)]
        tokens = tokens[:, self.num_prefix_tokens:]
        NMB, N, C = tokens.shape
        assert NMB == M * B and N == h * w, \
            f"token count ({NMB},{N}) != ({M*B},{h}x{w})"
        tokens = tokens.view(M, B, N, C)
        return [tokens[m].transpose(1, 2).reshape(B, C, h, w) for m in range(M)]
