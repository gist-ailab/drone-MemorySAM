"""[HIRES_ADAPTER] 고해상 경로 — ReliaDINO 얇은 객체 세부 복원.

진단: 백본은 stride-16 토큰만 내보내고, stride-4 특징은 전부 16px 토큰의 보간이라
16px 미만 정보가 경로에 없다(얇은 물체를 못 그린다). 평가 입력만 키워도 해결 방향으로
움직였다. E17(detail_branch)은 작은 conv 스템을 tanh 게이트 잔차로 얹었으나 얇은 객체
개선이 없었다(반증된 축).

이 모듈은 DINOv3 공식 분할의 DINOv3_Adapter 구조를 따른다 — SpatialPriorModule(conv
스템으로 s4/s8/s16/s32 prior) + InteractionBlock 의 **extractor 만**(injector 없음,
백본 무수정). conv 쿼리(c2·c3·c4)가 각 탭 토큰 맵을 MSDeformAttn 으로 샘플링해 ViT 의
의미를 끌어온 뒤 ConvFFN 을 거친다. E17 과의 핵심 차이: (a) extractor 로 ViT 의미를 끌어온
conv 특징, (b) 게이트 없는 주 경로(일반 초기화), (c) 넓은 폭(DIM 512 기본).

CUDA 확장은 쓰지 않는다(서버별 컴파일 실패 전례). MSDeformAttn 은 순수 PyTorch
(F.grid_sample)로 구현한다(Deformable-DETR 와 같은 파라미터화).

기본 off: build_reliadino 가 `MODEL.HIRES_ADAPTER.ENABLE=false` 면 이 모듈을 아예 만들지
않으므로 forward·state_dict 가 baseline 과 byte-동일하다(off 계약).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detail_branch import _norm2d
from .encoder import LayerNorm2d


# ── MSDeformAttn (순수 PyTorch) ──────────────────────────────────────────────
def _ms_deform_attn_core_pytorch(value: torch.Tensor,
                                 value_spatial_shapes: Sequence[Tuple[int, int]],
                                 sampling_locations: torch.Tensor,
                                 attention_weights: torch.Tensor) -> torch.Tensor:
    """Deformable-DETR 표준 레이아웃의 CPU/GPU 공용 core.

    value: (N, Len_v, n_heads, head_dim), sampling_locations: (N, Len_q, n_heads,
    n_levels, n_points, 2) ∈ [0,1], attention_weights: (N, Len_q, n_heads,
    n_levels, n_points). → (N, Len_q, n_heads*head_dim)."""
    N_, _, M_, D_ = value.shape
    _, Lq_, _, L_, P_, _ = sampling_locations.shape
    split = [int(h) * int(w) for h, w in value_spatial_shapes]
    value_list = value.split(split, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid, (h, w) in enumerate(value_spatial_shapes):
        # (N, h*w, M, D) -> (N*M, D, h, w)
        value_l = value_list[lid].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, int(h), int(w))
        # (N, Lq, M, P, 2) -> (N*M, Lq, P, 2)
        grid_l = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        sampling_value_l = F.grid_sample(
            value_l, grid_l, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l)                    # (N*M, D, Lq, P)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    out = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
    out = out.sum(-1).view(N_, M_ * D_, Lq_)
    return out.transpose(1, 2).contiguous()


class MSDeformAttnTorch(nn.Module):
    """순수 PyTorch multi-scale deformable attention(Deformable-DETR 파라미터화).

    value 차원은 `ratio`(공식 deform_ratio)로 축소한다: 내부 value 폭 = int(d_model·ratio).
    초기화는 Deformable-DETR `_reset_parameters` 관례(sampling_offsets bias 는 머리별
    방사 그리드, 나머지는 0/xavier)."""

    def __init__(self, d_model: int = 256, n_levels: int = 1, n_heads: int = 8,
                 n_points: int = 4, ratio: float = 1.0):
        super().__init__()
        self.d_model = int(d_model)
        self.n_levels = int(n_levels)
        self.n_heads = int(n_heads)
        self.n_points = int(n_points)
        self.ratio = float(ratio)
        self.value_dim = int(d_model * ratio)
        if self.value_dim % n_heads != 0:
            raise ValueError(f"[HIRES] value_dim(int(d_model·ratio)={self.value_dim}) 가 "
                             f"n_heads({n_heads})로 나뉘지 않는다.")
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, self.value_dim)
        self.output_proj = nn.Linear(self.value_dim, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor,
                value_flatten: torch.Tensor,
                value_spatial_shapes: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """query: (N, Lq, d_model), reference_points: (N, Lq, n_levels, 2) ∈ [0,1],
        value_flatten: (N, Lv, d_model), value_spatial_shapes: [(h,w), ...]."""
        N, Lq, _ = query.shape
        _, Lv, _ = value_flatten.shape
        value = self.value_proj(value_flatten).view(N, Lv, self.n_heads, self.value_dim // self.n_heads)
        offsets = self.sampling_offsets(query).view(
            N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        attn = self.attention_weights(query).view(N, Lq, self.n_heads, self.n_levels * self.n_points)
        attn = F.softmax(attn, -1).view(N, Lq, self.n_heads, self.n_levels, self.n_points)
        # reference_points (N,Lq,n_levels,2) → sampling_locations 는 격자 크기로 정규화
        offset_normalizer = torch.as_tensor(
            [[w, h] for h, w in value_spatial_shapes],
            dtype=query.dtype, device=query.device)                        # (n_levels, 2)=(w,h)
        sampling_locations = (reference_points[:, :, None, :, None, :]
                              + offsets / offset_normalizer[None, None, None, :, None, :])
        out = _ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attn)
        return self.output_proj(out)


# ── Spatial Prior Module ─────────────────────────────────────────────────────
class _SPMStem(nn.Module):
    """공식 SPM 스템. stem(conv3×3 s2 → conv3×3 → conv3×3 → maxpool s2) = s4,
    conv s2 → s8, conv s2 → s16, conv s2 → s32. 정규화는 GroupNorm(BS1 안전)."""

    def __init__(self, stem_dim: int, norm: str):
        super().__init__()
        c = int(stem_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1, bias=False), _norm2d(norm, c), nn.ReLU(inplace=False),
            nn.Conv2d(c, c, 3, 1, 1, bias=False), _norm2d(norm, c), nn.ReLU(inplace=False),
            nn.Conv2d(c, c, 3, 1, 1, bias=False), _norm2d(norm, c), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))                 # → s4, c
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, 2 * c, 3, 2, 1, bias=False), _norm2d(norm, 2 * c), nn.ReLU(inplace=False))  # s8
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, 3, 2, 1, bias=False), _norm2d(norm, 4 * c), nn.ReLU(inplace=False))  # s16
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * c, 4 * c, 3, 2, 1, bias=False), _norm2d(norm, 4 * c), nn.ReLU(inplace=False))  # s32

    def forward(self, x: torch.Tensor):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c1, c2, c3, c4


class SpatialPrior(nn.Module):
    """센서별 전처리 이미지 → DIM 레벨 prior (c1 s4, c2 s8, c3 s16, c4 s32), 모달 합산.

    MODE shared_stem: 스템 가중치는 모달 공유, 레벨별 1×1 투영만 센서별.
         per_modal   : 스템도 센서별. 센서별 레벨 특징은 TAPS per_modal 과 같은 방식으로 합산."""

    def __init__(self, dim: int, stem_dim: int = 64, mode: str = 'shared_stem',
                 num_modalities: int = 4, norm: str = 'gn'):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in ('shared_stem', 'per_modal'):
            raise ValueError(f"[HIRES] MODE 는 shared_stem|per_modal 이어야 한다 (got {mode!r}).")
        self.num_modalities = int(num_modalities)
        c = int(stem_dim)
        self._level_in = (c, 2 * c, 4 * c, 4 * c)

        def _make_fc():
            return nn.ModuleList(nn.Conv2d(self._level_in[k], dim, 1) for k in range(4))

        if self.mode == 'per_modal':
            self.stems = nn.ModuleList(_SPMStem(stem_dim, norm) for _ in range(self.num_modalities))
            self.stem = None
        else:
            self.stem = _SPMStem(stem_dim, norm)
            self.stems = None
        # 레벨별 1×1 DIM 투영은 두 모드 공통으로 센서별(센서 특성은 투영에서).
        self.fcs = nn.ModuleList(_make_fc() for _ in range(self.num_modalities))

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(x_list) != self.num_modalities:
            raise ValueError(f"[HIRES] SPM 모달 수 불일치: 입력 {len(x_list)} vs "
                             f"num_modalities {self.num_modalities}.")
        acc: List = [None, None, None, None]
        for m in range(self.num_modalities):
            stem = self.stems[m] if self.mode == 'per_modal' else self.stem
            raws = stem(x_list[m])
            for k in range(4):
                pj = self.fcs[m][k](raws[k])
                acc[k] = pj if acc[k] is None else acc[k] + pj
        return acc


# ── ConvFFN / Extractor ──────────────────────────────────────────────────────
class _DWConv(nn.Module):
    """공식 ConvFFN 의 DWConv — 쿼리를 세 레벨(c2/c3/c4)로 되돌려 각 레벨에 DWConv 를
    건다. 레벨 격자 크기를 명시로 받아(16n/4n/n 가정 없이) 32 비배수 입력에서도 안전."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, shapes: Sequence[Tuple[int, int]]) -> torch.Tensor:
        B, N, C = x.shape
        sizes = [int(h) * int(w) for h, w in shapes]
        parts = x.split(sizes, dim=1)
        outs = []
        for p, (h, w) in zip(parts, shapes):
            t = p.transpose(1, 2).view(B, C, int(h), int(w)).contiguous()
            outs.append(self.dwconv(t).flatten(2).transpose(1, 2))
        return torch.cat(outs, dim=1)


class ConvFFN(nn.Module):
    """공식 ConvFFN: fc1 → (레벨별 DWConv) → act → fc2."""

    def __init__(self, in_features: int, hidden_features: int, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = _DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, shapes: Sequence[Tuple[int, int]]) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, shapes)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Extractor(nn.Module):
    """공식 Extractor: query_norm/feat_norm + MSDeformAttn + ConvFFN. drop_path 0.

    쿼리 c=[c2;c3;c4] 가 단일 레벨(탭 토큰 맵)을 샘플링해 갱신한다(injector 없음)."""

    def __init__(self, dim: int, n_heads: int = 8, n_points: int = 4,
                 deform_ratio: float = 0.5, cffn_ratio: float = 0.25):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.feat_norm = nn.LayerNorm(dim)
        self.attn = MSDeformAttnTorch(
            d_model=dim, n_levels=1, n_heads=n_heads, n_points=n_points, ratio=deform_ratio)
        self.ffn = ConvFFN(in_features=dim, hidden_features=max(1, int(dim * cffn_ratio)))
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, query, reference_points, feat, value_spatial_shapes, query_shapes):
        attn = self.attn(self.query_norm(query), reference_points,
                         self.feat_norm(feat), value_spatial_shapes)
        query = query + attn
        query = query + self.ffn(self.ffn_norm(query), query_shapes)
        return query


class InteractionBlock(nn.Module):
    """블록 하나 = extractor 1개(+ 마지막 블록은 extra extractor 몇 개). 모두 같은 탭을 샘플링."""

    def __init__(self, dim: int, n_heads: int = 8, n_points: int = 4,
                 deform_ratio: float = 0.5, cffn_ratio: float = 0.25, extra: int = 0):
        super().__init__()
        self.extractor = Extractor(dim, n_heads, n_points, deform_ratio, cffn_ratio)
        self.extra_extractors = (nn.ModuleList(
            Extractor(dim, n_heads, n_points, deform_ratio, cffn_ratio) for _ in range(extra))
            if extra > 0 else None)

    def forward(self, c, reference_points, feat, value_spatial_shapes, query_shapes):
        c = self.extractor(c, reference_points, feat, value_spatial_shapes, query_shapes)
        if self.extra_extractors is not None:
            for ex in self.extra_extractors:
                c = ex(c, reference_points, feat, value_spatial_shapes, query_shapes)
        return c


# ── HiRes Adapter 본체 ───────────────────────────────────────────────────────
def _get_reference_points(spatial_shapes: Sequence[Tuple[int, int]],
                          device, dtype) -> torch.Tensor:
    """쿼리 격자(c2/c3/c4)의 정규화 좌표 (1, ΣHW, 1, 2). 공식 get_reference_points 규약."""
    ref_list = []
    for (h, w) in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, int(h), device=device, dtype=dtype),
            torch.linspace(0.5, w - 0.5, int(w), device=device, dtype=dtype),
            indexing='ij')
        ref_y = ref_y.reshape(-1)[None] / h
        ref_x = ref_x.reshape(-1)[None] / w
        ref_list.append(torch.stack((ref_x, ref_y), -1))                 # (1, h*w, 2)
    ref = torch.cat(ref_list, 1)
    return ref[:, :, None]                                               # (1, ΣHW, 1, 2)


class HiResAdapter(nn.Module):
    """ReliaDINO 고해상 경로.

    입력: 센서별 전처리 이미지 x(각 (B,3,H,W))와, _encode_all 에서 **센서 합산·DIM 투영
    까지 끝낸** 탭 4개(각 (B,DIM,h,w)). 블록 i 는 탭 i 를 key/value(단일 레벨)로, 쿼리
    c=[c2;c3;c4] 를 갱신한다. 마지막 블록은 extractor 를 extra_extractors 만큼 더 쌓는다.
    출력: f1=up(c2)+c1, f2..f4=c2..c4, 각 GroupNorm 후 1×1 로 fpn_dim → [s4,s8,s16,s32].

    탭 투영(tap_proj)은 기존 taps_proj_permodal 과 **별개 파라미터**다(기존 경로 무간섭).
    """

    def __init__(self, fpn_dim: int, num_taps: int = 4, num_modalities: int = 4,
                 tap_in_dim: int = 1024, dim: int = 512, stem_dim: int = 64,
                 mode: str = 'shared_stem', n_heads: int = 8, n_points: int = 4,
                 deform_ratio: float = 0.5, cffn_ratio: float = 0.25,
                 extra_extractors: int = 2, norm: str = 'gn'):
        super().__init__()
        if num_taps != 4:
            raise ValueError(f"[HIRES] 4 상호작용 블록 설계이므로 탭 4개가 필요하다 "
                             f"(got {num_taps}). MODEL.TAPS.LAYERS 를 4개로 두라.")
        self.dim = int(dim)
        self.num_taps = int(num_taps)
        self.num_modalities = int(num_modalities)

        self.spm = SpatialPrior(dim, stem_dim=stem_dim, mode=mode,
                                num_modalities=num_modalities, norm=norm)
        # 탭 투영: 센서별·슬롯별 LN2d + 1×1 (tap_in_dim → dim). 기존 경로와 별개.
        self.tap_proj = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(LayerNorm2d(tap_in_dim), nn.Conv2d(tap_in_dim, dim, 1))
                for _ in range(num_taps))
            for _ in range(num_modalities))
        # 쿼리 3레벨(c2/c3/c4) level embed (공식 zeros init).
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        # 상호작용 블록: 마지막 블록만 extra extractor.
        self.interactions = nn.ModuleList(
            InteractionBlock(dim, n_heads=n_heads, n_points=n_points,
                             deform_ratio=deform_ratio, cffn_ratio=cffn_ratio,
                             extra=(extra_extractors if i == num_taps - 1 else 0))
            for i in range(num_taps))
        self.up = nn.ConvTranspose2d(dim, dim, 2, 2)
        # 출력: 레벨별 GroupNorm + 1×1 fpn_dim 투영 ([s4,s8,s16,s32]).
        self.out_norm = nn.ModuleList(_norm2d(norm, dim) for _ in range(4))
        self.out_proj = nn.ModuleList(nn.Conv2d(dim, fpn_dim, 1) for _ in range(4))

    def project_sensor_taps(self, modal_idx: int,
                            taps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """센서 하나의 raw 탭(각 (B,1024,h,w))을 즉시 DIM 투영한다(_encode_all 루프에서
        센서별 호출 → 모델이 슬롯별 합산을 누적; raw 1024ch 를 들고 있지 않기 위함)."""
        if len(taps) != self.num_taps:
            raise ValueError(f"[HIRES] 탭 수 불일치: 입력 {len(taps)} vs {self.num_taps} "
                             f"(MODEL.TAPS.LAYERS 가 4개여야 한다).")
        pj = self.tap_proj[modal_idx]
        return [pj[j](taps[j]) for j in range(self.num_taps)]

    @staticmethod
    def _add_aligned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """b 를 a 격자에 bilinear 정렬해 더한다(32 비배수 입력의 ±1 불일치 흡수)."""
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        return a + b

    def forward(self, x_list: List[torch.Tensor],
                taps: List[torch.Tensor]) -> List[torch.Tensor]:
        """x_list: 센서별 (B,3,H,W). taps: 센서 합산·DIM 투영된 4개 (B,DIM,h,w).
        → fpn_dim 레벨 리스트 [s4,s8,s16,s32]."""
        if taps is None or len(taps) != self.num_taps:
            raise ValueError("[HIRES] 어댑터가 켜졌는데 탭이 수집되지 않았다 "
                             "(MODEL.TAPS 를 켜고 LAYERS 를 4개로 두라).")
        c1, c2, c3, c4 = self.spm(x_list)
        B = c2.shape[0]
        shapes = [tuple(c.shape[-2:]) for c in (c2, c3, c4)]             # 쿼리 3레벨
        splits = [int(h) * int(w) for h, w in shapes]
        # 쿼리 플래튼 + level embed
        q2 = c2.flatten(2).transpose(1, 2) + self.level_embed[0]
        q3 = c3.flatten(2).transpose(1, 2) + self.level_embed[1]
        q4 = c4.flatten(2).transpose(1, 2) + self.level_embed[2]
        c = torch.cat([q2, q3, q4], dim=1)                              # (B, ΣHW, dim)
        ref = _get_reference_points(shapes, c.device, c.dtype).expand(B, -1, -1, -1)
        # 블록 i: 탭 i 를 단일 레벨 value 로
        for i, block in enumerate(self.interactions):
            tap = taps[i]
            th, tw = int(tap.shape[-2]), int(tap.shape[-1])
            feat = tap.flatten(2).transpose(1, 2)                       # (B, th*tw, dim)
            c = block(c, ref, feat, [(th, tw)], shapes)
        # 쿼리 → 레벨 맵 복원
        o2, o3, o4 = c.split(splits, dim=1)
        o2 = o2.transpose(1, 2).view(B, self.dim, shapes[0][0], shapes[0][1]).contiguous()
        o3 = o3.transpose(1, 2).view(B, self.dim, shapes[1][0], shapes[1][1]).contiguous()
        o4 = o4.transpose(1, 2).view(B, self.dim, shapes[2][0], shapes[2][1]).contiguous()
        f1 = self._add_aligned(c1, self.up(o2))                         # s4 = up(c2)+c1
        levels = [f1, o2, o3, o4]
        return [self.out_proj[k](self.out_norm[k](levels[k])) for k in range(4)]
