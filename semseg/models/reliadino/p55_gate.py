"""[P55] Unsupervised multi-block modality gate.

연구 질문(user): 여러 인코더 블록(상대 깊이 [0.1, 0.25, 0.5])의 센서별 특징을
읽어 각 센서의 융합 기여를 곱셈으로 조절하는 게이트를, **품질/열화/조건 라벨 없이**
평소 세그멘테이션 손실만 융합을 통해 역전파해 학습하면, 상황에 따라 융합이 달라지는가
(예: 밤·비에 RGB 신뢰를 낮추고 depth/LiDAR/event 를 더 쓰는가) 아니면 상수로 붕괴하는가.

이 모듈은 게이트 본체다. 동결 인코더에서 **stop-grad** 로 잡은 다중 깊이 특징
(mixdepth: 분수별 LayerNorm + 학습된 softmax 가중합, tools/probe_quality_blocks.py 의
MixDepthHead 와 같은 구성)과, 각 센서가 나머지 센서와 같은 깊이에서 얼마나 일치하는지의
평균 코사인 지도(stop-grad)를 입력으로 받아 센서별 r_token∈(0,1) 을 낸다.

QAF 융합 인터페이스로의 변환:
  eta_token  = 1 − r_token          (η̂ = "열화/비신뢰", 1 = 완전 불신)
  eta_scalar = 1 − r_scalar
  floor: eta 를 ≤ 1 − r_floor 로 clamp (r_floor 기본 0.05) → 어떤 모달도 완전 제거되지 않는다.

초기화: 최종 conv 를 zero-weight + bias +3 으로 두어 시작 시 r ≈ sigmoid(3) ≈ 0.95 로
전 위치·전 센서 균일(= 거의 항등 융합, 붕괴 안전). 이후 세그 손실이 게이트만 조각한다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def frac_to_block(frac: float, n_blocks: int) -> int:
    """상대 깊이 분수 → 1-indexed 블록 번호 = max(1, round(frac*n_blocks)).

    tools/probe_quality_blocks.frac_to_block 과 동일 규약(백본 깊이 무관 상대 위치).
    tools 를 semseg 로 import 하면 순환(도구가 semseg 를 import)이라 여기 동일 사본을 둔다.
    """
    return max(1, int(round(frac * n_blocks)))


class P55Gate(nn.Module):
    """다중 블록 무감독 모달 게이트.

    forward(flat_feats) 입력 규약:
      flat_feats = 길이 n_fracs*M 의 **평탄 리스트**, frac-major 순서
      ([f0m0, f0m1, …, f0m(M-1), f1m0, …]), 각 원소 (B, C, h, w).
      tools/probe_quality_blocks.MixDepthHead 와 같은 배치 규약이라 같은 캡처를 재사용한다.

    반환:
      qaf_pred dict: 'eta_scalar' (B,M), 'eta_token' (B,M,h,w) — QAF 융합 경로 입력.
      r_token  (B,M,h,w)  게이트 원본(로깅·loo 타깃용)
      r_scalar (B,M)      공간 평균
    """

    def __init__(self, dim: int, num_modalities: int,
                 fractions: Tuple[float, ...] = (0.1, 0.25, 0.5),
                 hidden: int = 64, r_floor: float = 0.05,
                 init_bias: float = 3.0, use_agreement: bool = True):
        super().__init__()
        if num_modalities < 1:
            raise ValueError(f"P55Gate 는 모달 1개 이상 (got {num_modalities}).")
        self.m = int(num_modalities)
        self.fractions = tuple(float(f) for f in fractions)
        self.n_fracs = len(self.fractions)
        if self.n_fracs < 1:
            raise ValueError("P55.FRACTIONS 는 최소 1개.")
        self.r_floor = float(r_floor)
        self.use_agreement = bool(use_agreement)
        # 분수별 채널 LayerNorm (센서 간 공유) + 분수 softmax 가중 (MixDepthHead 구성).
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(self.n_fracs))
        self.frac_logits = nn.Parameter(torch.zeros(self.n_fracs))   # softmax → 균등 초기화
        in_ch = dim + (1 if self.use_agreement else 0)
        # 센서 간 공유 게이트 헤드(작게 유지, ≤~1M). 최종 conv zero-init + bias +3.
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, float(init_bias))

    def frac_weights(self) -> torch.Tensor:
        return torch.softmax(self.frac_logits, dim=0)

    def _ln(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        # (B,C,h,w) → 채널축 LayerNorm → (B,C,h,w) (MixDepthHead._ln 와 동일)
        return self.norms[idx](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def forward(self, flat_feats: List[torch.Tensor]) -> Tuple[Dict[str, torch.Tensor],
                                                               torch.Tensor, torch.Tensor]:
        M, Fn = self.m, self.n_fracs
        assert len(flat_feats) == Fn * M, \
            f"P55 mixdepth 특징 {len(flat_feats)}개 != n_fracs*M={Fn * M}"
        w = self.frac_weights()
        # 1) 분수별 LayerNorm (stop-grad: 동결 인코더 특징 detach — 게이트 파라미터만 학습).
        normed = [[self._ln(f, flat_feats[f * M + m].detach())
                   for m in range(M)] for f in range(Fn)]
        # 2) 센서별 mixdepth 결합.
        combined = [sum(w[f] * normed[f][m] for f in range(Fn)) for m in range(M)]
        # 3) (선택) 같은 깊이에서 타 센서와의 평균 코사인 일치 지도.
        agree: Optional[List[torch.Tensor]] = None
        if self.use_agreement and M >= 2:
            agree = []
            for m in range(M):
                per_frac = []
                for f in range(Fn):
                    xi = F.normalize(normed[f][m], dim=1)
                    others = torch.stack(
                        [F.normalize(normed[f][j], dim=1) for j in range(M) if j != m],
                        dim=0).mean(dim=0)
                    per_frac.append((xi * others).sum(dim=1, keepdim=True))  # (B,1,h,w)
                agree.append(torch.stack(per_frac, dim=0).mean(dim=0))
        elif self.use_agreement:
            B, _, h, w_ = combined[0].shape
            agree = [combined[0].new_zeros(B, 1, h, w_) for _ in range(M)]
        # 4) 게이트 헤드 → r_token.
        r_cols = []
        for m in range(M):
            inp = combined[m]
            if agree is not None:
                inp = torch.cat([inp, agree[m]], dim=1)
            r_cols.append(torch.sigmoid(self.head(inp)))       # (B,1,h,w)
        r_token = torch.cat(r_cols, dim=1)                     # (B,M,h,w)
        r_scalar = r_token.mean(dim=(-1, -2))                  # (B,M)
        eta_max = 1.0 - self.r_floor
        eta_token = (1.0 - r_token).clamp(min=0.0, max=eta_max)
        eta_scalar = (1.0 - r_scalar).clamp(min=0.0, max=eta_max)
        qaf_pred = {'eta_scalar': eta_scalar, 'eta_token': eta_token}
        return qaf_pred, r_token, r_scalar


# ===========================================================================
# 인코더 블록 캡처 — P55 가 켜졌을 때만 model 이 hook 을 건다.
# (tools/probe_quality_blocks.BlockTapHooks 와 같은 규약: forward 당 모달마다 1회 발화)
# ===========================================================================
class P55BlockCapture:
    """요청 블록(1-indexed)마다 forward-hook 을 걸어, 발화마다(=모달마다) 출력을 쌓는다.

    비-CMLC 경로는 encoder(x[i], i) 를 i=0..M-1 순차 호출 → hook 이 forward 당 블록마다
    1회 발화, 그 순서가 모달 순서다. clear() 후 순차 forward 를 돌리고 collect() 로 뽑는다.
    """

    def __init__(self, encoder, blocks_1indexed: List[int]):
        self.encoder = encoder
        self.blocks = list(blocks_1indexed)
        self.buf: Dict[int, list] = {k: [] for k in self.blocks}
        self.handles = []
        for k in self.blocks:
            self.handles.append(
                encoder.backbone.blocks[k - 1].register_forward_hook(self._mk(k)))

    def _mk(self, k):
        def hook(_m, _i, out):
            self.buf[k].append(out[0] if isinstance(out, tuple) else out)
        return hook

    def clear(self):
        for k in self.blocks:
            self.buf[k].clear()

    def collect_flat(self, h: int, w: int, M: int) -> List[torch.Tensor]:
        """frac-major 평탄 리스트 [b0m0, b0m1, …, b1m0, …] 를 (B,C,h,w) 맵으로 돌려준다.

        blocks 순서 = frac 순서(model 이 frac 오름차순 블록으로 등록). 각 블록 버퍼는
        모달 순서 M개여야 한다(순차 forward 가정 위반 시 시끄럽게 실패).
        """
        out = []
        for k in self.blocks:
            caps = self.buf[k]
            assert len(caps) == M, \
                f"[P55] block{k} 캡처 {len(caps)}개 != 모달 수 M={M} (순차 forward 가정 위반)"
            for t in caps:
                out.append(self.encoder._to_map(t, h, w))
        return out

    def remove(self):
        for hd in self.handles:
            hd.remove()
        self.handles = []
