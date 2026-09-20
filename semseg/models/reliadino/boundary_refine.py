"""[BOUNDARY_REFINE / R1] depth 에지-prior 로 SimpleFPN stride-4(/8) 정제.

진단(제안서 2026-09-20 §6 "Q3+R", 판정 대장 "원거리 손해 원인 규명"): 우리 모델은
얇은 객체를 **찾긴 하는데 윤곽을 못 그린다**(찾고도 못 그린 비율 0.204/0.429 대
기준선 0.085/0.253, 경계 F 원거리 TrafficLight −0.167). 결함은 같은 크기여도 거리가
멀수록 커지고 손실은 **경계**에서 난다. 검출·클래스 지식은 기준선과 같다.

이 모듈은 depth 모달을 특징으로 **융합하지 않는다**(그 축 = 반증됨). depth 의 **불연속
(에지) 위치 정보만** 경계 prior 로 삼아, SimpleFPN 의 stride-4(선택적으로 stride-8)
특징을 에지 인지 방식으로 정제한다.

동작:
  · 에지 맵 = depth 의 Sobel 그래디언트 크기(채널 평균, 정규화된 depth 텐서 기준)
    → 레벨 해상도로 avg-pool → 이미지별 표준화(zero-mean/unit-std, 그래서 τ init 0 =
    "평균") → `e = sigmoid(k·(g − τ))` (k, τ 는 학습 가능한 스칼라, init k=10, τ=0=평균).
  · 정제: `s' = s + tanh(γ)·Conv3x3([s ; s⊙e ; e])` (γ init 0.1 — zero-init 금지,
    계보 규칙). 채널 = fpn_dim. 학습 가능 파라미터 = k, τ, 레벨별 Conv, 레벨별 γ 뿐.

DELIVER depth 채널 구성(로더 확인, semseg/datasets/deliver.py:141·183-190): depth 는
`/hha` 경로의 **HHA 3채널** 이미지다(`_open_img` 가 1채널이면 3채널 반복, 4채널이면
앞 3채널). 이 파이프라인은 전 모달을 공유 백본(3채널 patch_embed)에 넣으므로 forward 에
들어오는 depth 텐서는 (B,3,H,W) 정규화 텐서다. 채널 평균 후 Sobel 을 건다.

기본 off: 모델이 `MODEL.BOUNDARY_REFINE.ENABLE=false` 면 이 모듈을 아예 만들지 않아
forward·state_dict 가 baseline 과 byte-동일하다(off 계약). 추론 시에도 동일 경로를 탄다
(추론 전용 후처리가 아니다).
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryRefine(nn.Module):
    """depth 에지-prior 로 FPN pyramid 의 지정 stride 레벨을 정제한다.

    LEVELS: 적용할 stride 리스트(기본 [4], 옵션 [4, 8]). SimpleFPN 은 [s4, s8, s16, s32]
    순서라 stride s → pyramid index = log2(s/4) (stride4→0, stride8→1).
    """

    def __init__(self,
                 fpn_dim: int,
                 levels: Sequence[int] = (4,),
                 k_init: float = 10.0,
                 gamma_init: float = 0.1):
        super().__init__()
        lv = [int(s) for s in levels]
        for s in lv:
            if s not in (4, 8):
                raise ValueError(f"[BREFINE] LEVELS 는 4 또는 8 만 지원한다 "
                                 f"(SimpleFPN stride-4/8 정제, got {levels}).")
        self.strides: List[int] = sorted(set(lv))
        self.pyr_index = [int(round(math.log2(s / 4))) for s in self.strides]

        # 에지 게이트 스칼라(학습 가능): k=날카로움, τ=문턱. g 를 이미지별 표준화하므로
        # τ init 0 이 "에지 맵 평균"에 해당한다(제안서: τ init=평균).
        self.k = nn.Parameter(torch.tensor(float(k_init)))
        self.tau = nn.Parameter(torch.tensor(0.0))

        # 레벨별 정제 conv: [s ; s⊙e ; e] → fpn_dim (입력 채널 = 2·fpn_dim + 1).
        # zero-init 금지(계보 규칙): Conv2d 기본 kaiming, 대신 γ init 0.1 로 게이트한다.
        self.refine = nn.ModuleList(
            nn.Conv2d(2 * fpn_dim + 1, fpn_dim, 3, padding=1)
            for _ in self.strides)
        # 레벨별 게이트 γ (init 0.1 — tanh(0.1)≈0.0997, 첫 스텝부터 grad 유입).
        self.gamma = nn.Parameter(
            torch.full((len(self.strides),), float(gamma_init)))

        # Sobel 커널(상수, 학습 불가). persistent=False → state_dict 를 늘리지 않는다
        # (init 에서 항상 재구성). 채널 평균 후 (B,1,H,W) 에 그룹 없는 단일 conv 로 적용.
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = kx.t().contiguous()
        self.register_buffer('_sobel_x', kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer('_sobel_y', ky.view(1, 1, 3, 3), persistent=False)

    def _edge_magnitude(self, depth: torch.Tensor) -> torch.Tensor:
        """depth (B,C,H,W) 정규화 텐서 → Sobel 그래디언트 크기 (B,1,H,W).

        채널 평균 후 Sobel(gx, gy) → sqrt(gx²+gy²). Sobel 커널 dtype 을 입력에 맞춰
        캐스팅한다(bf16 autocast 안전)."""
        g1 = depth.mean(dim=1, keepdim=True)                       # (B,1,H,W)
        wx = self._sobel_x.to(dtype=g1.dtype)
        wy = self._sobel_y.to(dtype=g1.dtype)
        gx = F.conv2d(g1, wx, padding=1)
        gy = F.conv2d(g1, wy, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def forward(self, pyramid: List[torch.Tensor],
                depth: torch.Tensor) -> List[torch.Tensor]:
        """pyramid([s4, s8, s16, s32]) 의 지정 레벨을 depth 에지로 정제해 돌려준다.

        SimpleFPN 순서 불변식(index 0 이 가장 조밀=stride-4)을 assert 로 지킨다."""
        assert pyramid[0].shape[-1] >= pyramid[1].shape[-1], (
            "[BREFINE] SimpleFPN pyramid 순서가 [s4, s8, ...] 이 아니다 — "
            "stride→레벨 매핑 전제가 깨졌다.")
        g_full = self._edge_magnitude(depth)                      # (B,1,H,W)
        out = list(pyramid)
        for li, idx in enumerate(self.pyr_index):
            assert 0 <= idx < len(out), (
                f"[BREFINE] pyramid 레벨 인덱스 {idx} 범위 밖(레벨 {len(out)}개).")
            s = out[idx]
            h, w = s.shape[-2:]
            g = F.adaptive_avg_pool2d(g_full, (h, w))             # 레벨 해상도로 avg-pool
            # 이미지별 표준화 → τ init 0 이 에지 맵 평균(제안서 τ=평균). DDP-안전
            # (배치 원소별, rank 간 lazy-init 없음).
            mu = g.mean(dim=(2, 3), keepdim=True)
            sd = g.std(dim=(2, 3), keepdim=True) + 1e-6
            g = (g - mu) / sd
            e = torch.sigmoid(self.k * (g - self.tau))            # (B,1,h,w)
            feat = torch.cat([s, s * e, e], dim=1)                # (B, 2C+1, h,w)
            ref = self.refine[li](feat)
            out[idx] = s + torch.tanh(self.gamma[li]) * ref
        return out

    def gate_values(self) -> List[float]:
        """로깅용 tanh(γ) 스칼라 리스트(학습 영향 0)."""
        with torch.no_grad():
            return [float(v) for v in torch.tanh(self.gamma.detach()).cpu()]
