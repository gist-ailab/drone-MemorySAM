"""[P51] CMLC — Cross-modal LoRA Coupling (인코딩-시간 cross-modal 결합).

FrozenViTEncoder 의 per-modality qkv LoRA 가 모달마다 따로 만드는 토큰 특징을,
**선택한 ViT block 경계에서** 저랭크 부분공간 안에서 서로 결합한다:

    z_m  = down_p(x_m)                                  (B, N, r)  저랭크 코드
    z̃_m = z_m + Σ_{k≠m} γ[m,k] · (z_k @ C_p[m,k])       코드 공간 결합
    x'_m = x_m + up_p(z̃_m − z_m)                     잔차: 결합 증분만 up-project

설계 계약 (P51 제안서):
  - 결합은 fusion 직전이 아니라 인코딩 **중간**(block 경계)에 일어나, 이후
    block 들이 결합된 표현으로 계속 인코딩하게 만든다 ("fusion 직전 결합"은
    이미 반증된 축이다).
  - 결합 행렬 C 는 모달-쌍별 r×r 소행렬 — "모달 k 의 코드를 모달 m 의 코드
    공간으로 옮기는" 선형 지도다. LoRA 의 저랭크 코드 공간을 그대로 재사용해
    full D×D mixing(M²·D² 파라미터)이 아니라 M²·r² 로 묶는다.
  - γ 게이트는 절대 zero-init 하지 않는다(init 1.0): 첫 스텝부터 결합 경로로
    gradient 가 흐른다(키1 — 수동 zero-결선 사금). 반대로 γ=0 이면 잔차 구조상
    CMLC 는 정확히 항등이 된다(off 모델과 수치 등가, 스모크 2).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossModalLoRACoupling(nn.Module):
    """결합점(ViT block 경계)마다 down/up projection + 모달-쌍별 r×r 결합 행렬.

    down/up 은 결합점별로 **별도** 둔다 — 얕은 block 과 깊은 block 의 코드
    분포가 다르기 때문에 같은 projection 을 공유할 근거가 없다. γ 는 결합점
    전체에서 공유(스칼라 축이 모달 쌍 하나뿐) — 결합 강도의 전역 스케일은
    하나로 두는 편이 ablation 해석이 쉽다.
    """

    def __init__(self, num_modalities: int, dim: int, r: int = 16,
                 num_couple_points: int = 1):
        super().__init__()
        M, P = int(num_modalities), int(num_couple_points)
        if M < 2:
            raise ValueError(f"CMLC needs >=2 modalities to couple, got {M}")
        self.num_modalities = M
        self.dim = dim
        self.r = int(r)
        self.num_couple_points = P
        # 결합점별 down/up (dim->r / r->dim, bias 없음 — LoRA 코드와 같은 관행)
        self.down = nn.ModuleList(
            nn.Linear(dim, self.r, bias=False) for _ in range(P))
        self.up = nn.ModuleList(
            nn.Linear(self.r, dim, bias=False) for _ in range(P))
        # C_p[m, k]: 모달 k 의 코드 → 모달 m 의 코드 (r×r). 작은 랜덤 init —
        # std=1/sqrt(r) 이면 z_k @ C 의 성분 분산이 z_k 분산과 같은 스케일로
        # 보존된다(zero-init 금지: 결합이 첫 스텝부터 살아 있어야 한다).
        self.c = nn.Parameter(
            torch.randn(P, M, M, self.r, self.r) / math.sqrt(self.r))
        # 결합 게이트 γ (init 1.0 — 절대 zero-init 금지). 대각(m==k)은 forward
        # 에서 마스크해 자기 자신과의 결합(=scale 변형)이 생기지 않게 한다.
        self.gamma = nn.Parameter(torch.ones(M, M))
        self.register_buffer('_offdiag', 1.0 - torch.eye(M), persistent=False)

    def forward(self, feats_stack: torch.Tensor, point_idx: int = 0) -> torch.Tensor:
        """feats_stack (M, B, N, D) — 같은 결합점의 M개 모달 토큰 → (M, B, N, D).

        point_idx 는 몇 번째 결합점(=COUPLE_LAYERS 의 몇 번째 block)인지 —
        결합점마다 C/down/up 이 따로 있기 때문에 caller(hook)가 지정한다.
        """
        M, B, N, D = feats_stack.shape
        assert M == self.num_modalities and D == self.dim, \
            f"feats_stack {tuple(feats_stack.shape)} != ({M},B,N,{D})"
        p = int(point_idx)
        z = self.down[p](feats_stack)                       # (M, B, N, r)
        # z_k @ C[m,k] 를 한 번의 einsum 으로: (수신 m, 송신 k, B, N, r)
        cross = torch.einsum('kbnr,mkrs->mkbns', z, self.c[p])
        w = self.gamma * self._offdiag                      # (M, M) 대각 0
        z_tilde = z + torch.einsum('mk,mkbns->mbns', w, cross)
        # 잔차: 결합 증분만 up-project → γ=0 이면 z̃==z, 증분 0, 정확히 항등
        return feats_stack + self.up[p](z_tilde - z)
