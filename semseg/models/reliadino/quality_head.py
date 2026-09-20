"""[P54-Q1] 품질 헤드 — 동결 E1 의 모달별 LoRA 출력 토큰 위에 붙는 작은 헤드.

제안서 §2-1. 모달 m 의 토큰 (B,C,h,w) 위에 학습 가능한 품질 query 1개가 attention
(1 head)으로 pooling → 모달 스칼라 η̂_m∈[0,1] 과, 2층 conv 헤드로 토큰별 η̂_{m,t}∈[0,1]
(B,h,w)를 낸다. η̂ = "열화(비신뢰)" 추정치(0=clean, 1=완전 열화/결측).

- 파라미터 ≈ 0.3M/모달 이내(dim=1024 기준: in_proj 1024×256=0.26M 이 대부분).
- **fp32 고정**: autocast 안에서도 float 로 계산한다(§2-1 게이트 fp32 규칙).
- 손실 `quality_loss(pred, labels)` = presence BCE + severity SmoothL1(존재 조건부)
  + 패치 마스크 BCE + 순위 힌지(margin 0.1). 반환 dict 에 각 항 분리.

presence 는 별도 보조 로짓으로 예측한다(η̂_m 은 severity 축이므로 결측/존재 축과
분리 — 손실 4항이 서로 다른 출력을 감독하게 하려는 의도). 문서화용 보조 출력이며
융합에는 η̂_m·η̂_{m,t} 만 쓴다.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _ModalQualityHead(nn.Module):
    """단일 모달용. 입력 (B,C,h,w) → (η̂_scalar (B,), η̂_token (B,h,w), presence_logit (B,))."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden)               # 토큰 투영(대부분의 파라미터)
        self.query = nn.Parameter(torch.randn(hidden) * 0.02)  # 품질 query 1개
        self.scalar_head = nn.Linear(hidden, 1)             # η̂_m
        self.presence_head = nn.Linear(hidden, 1)           # 보조 presence 로짓
        # 토큰별 η̂: 2층 conv 헤드
        self.token_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden // 4, 1, kernel_size=3, padding=1),
        )

    def forward(self, feat: Tensor):
        B, C, h, w = feat.shape
        tok = feat.flatten(2).transpose(1, 2).float()       # (B,N,C), fp32 고정
        z = self.in_proj(tok)                               # (B,N,hidden)
        # 1-head attention pooling: query 로 토큰을 attend
        scale = z.shape[-1] ** -0.5
        attn = torch.softmax((z * self.query).sum(-1) * scale, dim=1)  # (B,N)
        ctx = (attn.unsqueeze(-1) * z).sum(1)               # (B,hidden)
        eta_scalar = torch.sigmoid(self.scalar_head(ctx)).squeeze(-1)  # (B,)
        presence_logit = self.presence_head(ctx).squeeze(-1)           # (B,)
        zmap = z.transpose(1, 2).reshape(B, -1, h, w)       # (B,hidden,h,w)
        eta_token = torch.sigmoid(self.token_head(zmap)).squeeze(1)    # (B,h,w)
        return eta_scalar, eta_token, presence_logit


class QualityHead(nn.Module):
    """모달별 품질 헤드 묶음.

    forward(feats) : feats = list of M x (B,C,h,w) (융합 직전 LoRA 토큰).
    반환 dict:
      'eta_scalar'     (B,M)      η̂_m
      'eta_token'      (B,M,h,w)  η̂_{m,t}
      'presence_logit' (B,M)      보조 presence 로짓(손실 전용)
    """

    def __init__(self, dim: int, num_modalities: int, hidden: int = 256):
        super().__init__()
        self.num_modalities = num_modalities
        self.heads = nn.ModuleList(
            _ModalQualityHead(dim, hidden) for _ in range(num_modalities))

    def forward(self, feats: List[Tensor]) -> Dict[str, Tensor]:
        assert len(feats) == self.num_modalities, \
            f"got {len(feats)} modalities, expected {self.num_modalities}"
        scal, tok, pres = [], [], []
        for m, f in enumerate(feats):
            s, t, p = self.heads[m](f)
            scal.append(s)
            tok.append(t)
            pres.append(p)
        return {
            'eta_scalar': torch.stack(scal, dim=1),         # (B,M)
            'eta_token': torch.stack(tok, dim=1),           # (B,M,h,w)
            'presence_logit': torch.stack(pres, dim=1),     # (B,M)
        }


def quality_loss(pred: Dict[str, Tensor], labels: Dict[str, Tensor],
                 rank_margin: float = 0.1) -> Dict[str, Tensor]:
    """품질 손실. 각 항을 분리해 반환하고 'total' 에 합을 넣는다.

    pred   : QualityHead.forward 출력.
    labels : Degrader 라벨. presence (B,M), severity (B,M), mask16 (B,M,h,w).
    """
    eta_s = pred['eta_scalar'].float()                      # (B,M)
    eta_t = pred['eta_token'].float()                       # (B,M,h,w)
    pres_logit = pred['presence_logit'].float()             # (B,M)
    presence = labels['presence'].float().to(eta_s.device)  # 1=존재
    severity = labels['severity'].float().to(eta_s.device)
    mask16 = labels['mask16'].float().to(eta_s.device)

    # eta_token 을 라벨 mask16 해상도로 맞춘다(특징 h,w 와 다를 수 있음).
    if eta_t.shape[-2:] != mask16.shape[-2:]:
        B, M = eta_t.shape[:2]
        eta_t = F.interpolate(eta_t.reshape(B * M, 1, *eta_t.shape[-2:]),
                              size=mask16.shape[-2:], mode='bilinear',
                              align_corners=False).reshape(B, M, *mask16.shape[-2:])

    # 1) presence BCE — 보조 로짓으로 존재/결측 예측
    l_presence = F.binary_cross_entropy_with_logits(pres_logit, presence)

    # 2) severity SmoothL1 — 존재하는 모달에서만(결측 severity=1 은 제외)
    present = presence > 0.5
    if present.any():
        l_severity = F.smooth_l1_loss(eta_s[present], severity[present])
    else:
        l_severity = eta_s.sum() * 0.0

    # 3) 패치 마스크 BCE — 토큰별 열화 여부
    l_mask = F.binary_cross_entropy(eta_t.clamp(1e-6, 1 - 1e-6), mask16)

    # 4) 순위 힌지 — 같은 배치에서 severity 라벨이 큰 쪽의 η̂ 가 작으면 벌점.
    #    존재 모달만 대상, 배치·모달을 평탄화해 쌍 비교.
    l_rank = eta_s.sum() * 0.0
    ps = eta_s[present]                                     # (K,)
    sv = severity[present]
    if ps.numel() >= 2:
        diff_eta = ps[:, None] - ps[None, :]               # η̂_i - η̂_j
        diff_sev = sv[:, None] - sv[None, :]               # sev_i - sev_j
        pair = (diff_sev > 1e-4).float()                   # sev_i > sev_j 인 쌍
        # sev_i>sev_j 인데 η̂_i < η̂_j 이면 hinge(margin - (η̂_i-η̂_j))
        viol = F.relu(rank_margin - diff_eta) * pair
        denom = pair.sum().clamp(min=1.0)
        l_rank = viol.sum() / denom

    total = l_presence + l_severity + l_mask + l_rank
    return {
        'total': total,
        'presence': l_presence.detach(),
        'severity': l_severity.detach(),
        'mask': l_mask.detach(),
        'rank': l_rank.detach(),
    }
