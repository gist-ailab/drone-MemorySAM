"""[DETAIL_BRANCH] 고해상도 세부 가지 — ReliaDINO stride-4/8 세부 경로.

진단: 백본은 stride-16 토큰(768 입력 기준 48×48)만 내보내고, SimpleFPN 은 그 한 장을
ConvTranspose 로 stride-4/8 까지 **늘려 쓸 뿐** 원본 해상도 정보를 새로 들이지 않는다.
그래서 얇고 작은 객체(Pole·TrafficLight·Pedestrian·Static)의 세부가 파이프라인에 들어오지
못한다(입력 해상도를 1024 로 올려도 test 이득이 없던 것도 "토큰 격자가 아니라 세부 경로
부재"를 가리킨다). SOTA(MM SAM-adapter)는 ConvNeXt 측면 가지로 1024 세부를 따로 뽑는다.
이 모듈은 그보다 가벼운 합성곱 스템으로 같은 역할을 하되, **모든 모달리티**에 적용한다.

동작: 백본에 넣는 것과 같은 전처리 입력 이미지에 작은 CNN 스템을 걸어 stride-4·stride-8
세부 특징을 만들고, 각 레벨을 1×1 conv 로 fpn_dim 으로 투영한다. 모달별 세부 특징은
**합산**(TAPS 의 per_modal 합산과 같은 방식)한 뒤, 모델이 FPN pyramid 의 stride-4·stride-8
레벨에 게이트 잔차(`pyr[l] = pyr[l] + tanh(g_l)·detail[l]`, g_l init 0.1)로 더한다.

기본 off: build_reliadino 가 `MODEL.DETAIL_BRANCH.ENABLE=false` 면 이 모듈을 아예 만들지
않으므로 forward·state_dict 가 baseline 과 byte-동일하다(off 계약).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Union

import torch
import torch.nn as nn


def _norm2d(kind: str, ch: int) -> nn.Module:
    """stride 스템용 정규화. 기본 GroupNorm(BS1 안전) — num_groups 는 ch 를 나누고
    32 를 넘지 않는 최댓값(gcd)으로 잡는다. 'bn' 이면 BatchNorm2d."""
    kind = str(kind).lower()
    if kind == 'bn':
        return nn.BatchNorm2d(ch)
    if kind in ('gn', 'groupnorm'):
        return nn.GroupNorm(num_groups=math.gcd(32, ch), num_channels=ch)
    raise ValueError(f"[DETAIL] NORM 은 gn|bn 이어야 한다 (got {kind!r}).")


class DetailStem(nn.Module):
    """입력 이미지 → stride-4/8 세부 특징(fpn_dim, 모달 합산).

    스템 stage k(0-indexed)의 출력 stride = 2^(k+1), 채널 = stem_dim·2^k.
    LEVELS=[4,8], STEM_DIM=32 이면: stage0 stride2/32ch → stage1 stride4/64ch(레벨)
    → stage2 stride8/128ch(레벨). 각 레벨은 1×1 conv 로 fpn_dim 투영.

    MODE:
      · shared_stem : 스템 가중치는 모달 공유, 1×1 투영만 모달별(센서 특성은 투영에서).
                      모달별 입력 채널이 다르면 첫 conv 만 모달별, 나머지 공유로 폴백.
      · per_modal   : 스템도 모달별.
    """

    def __init__(self,
                 in_ch: Union[int, Sequence[int]],
                 fpn_dim: int,
                 stem_dim: int = 32,
                 levels: Sequence[int] = (4, 8),
                 mode: str = 'shared_stem',
                 num_modalities: int = 4,
                 norm: str = 'gn',
                 gate_init: float = 0.1):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in ('shared_stem', 'per_modal'):
            raise ValueError(f"[DETAIL] MODE 는 shared_stem|per_modal 이어야 한다 "
                             f"(got {mode!r}).")
        self.num_modalities = int(num_modalities)
        self.stem_dim = int(stem_dim)

        # ── 레벨 검증 및 stage 매핑 ────────────────────────────────────────────
        lv = [int(s) for s in levels]
        for s in lv:
            if s < 4 or (s & (s - 1)) != 0:
                raise ValueError(f"[DETAIL] LEVELS 는 4 이상의 2의 거듭제곱이어야 한다 "
                                 f"(got {levels}).")
        self.strides: List[int] = sorted(set(lv))
        self.max_stride = self.strides[-1]
        self.n_stages = int(round(math.log2(self.max_stride)))       # stride 2..max
        # stage k 출력 채널 = stem_dim·2^k (stage0=stride2, stage1=stride4, ...)
        self.stage_out = [self.stem_dim * (2 ** k) for k in range(self.n_stages)]
        # stride s → stage index (log2(s)-1) → 채널
        self._stage_of = {s: int(round(math.log2(s))) - 1 for s in self.strides}
        self.level_ch = {s: self.stage_out[self._stage_of[s]] for s in self.strides}
        # stride s → SimpleFPN pyramid index. SimpleFPN 은 [s4, s8, s16, s32] 순서라
        # index = log2(s/4)  (stride4→0, stride8→1). 모델이 assert 와 함께 소비한다.
        self.pyr_index = [int(round(math.log2(s / 4))) for s in self.strides]

        in_list = ([int(in_ch)] * self.num_modalities
                   if isinstance(in_ch, int) else [int(c) for c in in_ch])
        if len(in_list) != self.num_modalities:
            raise ValueError(f"[DETAIL] in_ch 길이({len(in_list)})가 num_modalities"
                             f"({self.num_modalities})와 다르다.")
        self._in_list = in_list

        # ── 스템 구성 ──────────────────────────────────────────────────────────
        self.shared_stem = None      # shared_stem, 모달 입력채널 동일
        self.first_blocks = None     # shared_stem, 입력채널 상이 → 첫 conv 만 모달별
        self.shared_tail = None      # shared_stem, 입력채널 상이 → stage1.. 공유
        self.stems = None            # per_modal → 모달별 전체 스템
        if self.mode == 'per_modal':
            self.stems = nn.ModuleList(
                self._make_stem(in_list[i], norm) for i in range(self.num_modalities))
        elif len(set(in_list)) == 1:
            self.shared_stem = self._make_stem(in_list[0], norm)
        else:
            # 첫 conv(stage0)만 모달별, 나머지(stage1..) 공유
            self.first_blocks = nn.ModuleList(
                self._make_block(c, self.stage_out[0], norm) for c in in_list)
            self.shared_tail = self._make_stem(self.stage_out[0], norm, start=1)

        # ── 모달별 1×1 투영(레벨당) ────────────────────────────────────────────
        self.proj = nn.ModuleList(
            nn.ModuleList(nn.Conv2d(self.level_ch[s], fpn_dim, 1)
                          for s in self.strides)
            for _ in range(self.num_modalities))

        # ── 레벨별 게이트 스칼라 (init 0.1, zero-init 금지) ────────────────────
        self.gate = nn.Parameter(
            torch.full((len(self.strides),), float(gate_init)))

    # ── 빌드 헬퍼 ─────────────────────────────────────────────────────────────
    def _make_block(self, cin: int, cout: int, norm: str) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
            _norm2d(norm, cout),
            nn.GELU())

    def _make_stem(self, in_ch: int, norm: str, start: int = 0) -> nn.ModuleList:
        blocks = []
        cin = in_ch
        for k in range(start, self.n_stages):
            cout = self.stage_out[k]
            blocks.append(self._make_block(cin, cout, norm))
            cin = cout
        return nn.ModuleList(blocks)

    # ── forward ───────────────────────────────────────────────────────────────
    def _run_stem(self, img, blocks, start: int = 0) -> dict:
        """blocks 를 순차 적용하며 stride 가 레벨인 stage 출력만 모은다."""
        outs = {}
        x = img
        for off, blk in enumerate(blocks):
            k = start + off
            x = blk(x)
            stride = 2 ** (k + 1)
            if stride in self.strides:
                outs[stride] = x
        return outs

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """x_list: 모달 리스트(각 (B, C_m, H, W)) → 레벨별 fpn_dim 세부 특징 리스트
        (self.strides 순서), 모달 합산 완료. 원본 dim 텐서를 리스트로 들고 있지
        않도록 모달마다 투영 후 즉시 누적한다(활성화 메모리 절약)."""
        if len(x_list) != self.num_modalities:
            raise ValueError(f"[DETAIL] 모달 수 불일치: 입력 {len(x_list)} vs "
                             f"num_modalities {self.num_modalities}.")
        detail: List = [None] * len(self.strides)
        for m in range(self.num_modalities):
            img = x_list[m]
            if self.mode == 'per_modal':
                outs = self._run_stem(img, self.stems[m])
            elif self.shared_stem is not None:
                outs = self._run_stem(img, self.shared_stem)
            else:
                x1 = self.first_blocks[m](img)               # stage0(stride2)
                outs = {}
                if 2 in self.strides:                        # LEVELS≥4 이면 미해당
                    outs[2] = x1
                outs.update(self._run_stem(x1, self.shared_tail, start=1))
            for li, s in enumerate(self.strides):
                pj = self.proj[m][li](outs[s])
                detail[li] = pj if detail[li] is None else detail[li] + pj
        return detail

    def gate_values(self) -> List[float]:
        """로깅용 tanh(g) 스칼라 리스트(학습 영향 0)."""
        with torch.no_grad():
            return [float(v) for v in torch.tanh(self.gate.detach()).cpu()]
