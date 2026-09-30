"""[P56-C] 센서 상태 조건부 LoRA 전문가 혼합 — state router.

설계 = .claude_logs/decisions/2026-09-30-p56-bc-modality-aware-design.md §2.
얕은 블록(BLOCK_FRAC 기본 0.25 → frac_to_block 규약으로 1-indexed 블록 6)의
토큰 출력(**stop-grad**) + 센서 임베딩으로 토큰별 혼합 계수 α∈[0,1] 을 낸다.
APPLY_FROM_BLOCK(기본 7) 이상 블록의 StateRoutedLoRAQKV(encoder.py)가

    ΔW x = α · shared_delta(x) + (1 − α) · sensor_delta_m(x)

혼합에 쓴다. 라우터 파라미터는 센서 간 공유(센서 임베딩으로 구분)이며 최종
Linear 는 zero-weight/bias → 초기 α=0.5(공유·센서별 절반)에서 출발한다.
정규화는 두지 않는다(P55 교훈 — 열화·충돌 학습 표본이 α 를 움직일 유인).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class StateRouter(nn.Module):
    """블록 k 토큰 (B,N,C) + 센서 임베딩 → α (B,N,1). PER_TOKEN=false 면 (B,1,1).

    forward 입력 tokens 는 호출부(attach_state_routed_alpha hook)가 stop-grad 로
    넘긴다 — 라우터 파라미터만 학습되고 인코더 그래프로 grad 가 새지 않는다
    (p55_gate 의 stop-grad 규약과 동일). α 자체는 grad 가 살아 있어 블록 k+1~
    의 LoRA 혼합을 통해 세그 손실로 학습된다.

    `_last_alpha_stats` 는 detach 스칼라(모달별 mean / 토큰 간 std)로, 로깅·붕괴
    감시(게이트 G3: α 토큰 std ≥ 0.05)에 쓴다. forward 시점에 아직 안 본 모달
    슬롯은 None.
    """

    def __init__(self, dim: int, num_modalities: int, hidden: int = 64,
                 per_token: bool = True):
        super().__init__()
        if num_modalities < 1:
            raise ValueError(f"[P56-C] StateRouter 는 모달 1개 이상 (got {num_modalities}).")
        self.m = int(num_modalities)
        self.per_token = bool(per_token)
        # LayerNorm → Linear(C,HIDDEN) → GELU → Linear(HIDDEN,1) → sigmoid.
        # 센서 임베딩을 입력에 더해 센서 간 공유 파라미터로 센서별 α 를 낸다.
        self.norm = nn.LayerNorm(dim)
        self.sensor_embed = nn.Embedding(num_modalities, dim)
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].weight)   # σ(0)=0.5 — 균등 혼합에서 출발
        nn.init.zeros_(self.head[-1].bias)
        # 로깅 스냅샷(모달별). forward 때마다 해당 모달 슬롯만 갱신한다.
        self._mod_stats: Dict[int, Tuple[float, float]] = {}
        self._last_alpha_stats: Dict[str, Optional[tuple]] = {'mean': None,
                                                              'std_token': None}

    def forward(self, tokens: torch.Tensor, modality_idx: int) -> torch.Tensor:
        x = self.norm(tokens.detach()) + self.sensor_embed.weight[modality_idx]
        logits = self.head(x)                          # (B,N,1)
        if not self.per_token:
            # 설계서 선택 C-2: 토큰 평균 스칼라 α — 표현력은 낮지만 안정.
            logits = logits.mean(dim=1, keepdim=True)  # (B,1,1)
        alpha = torch.sigmoid(logits)
        with torch.no_grad():
            a = alpha.detach()
            self._mod_stats[int(modality_idx)] = (float(a.mean()), float(a.std()))
            self._last_alpha_stats = {
                'mean': tuple(self._mod_stats.get(m, (None,))[0]
                              for m in range(self.m)),
                'std_token': tuple(self._mod_stats.get(m, (None, None))[1]
                                   for m in range(self.m)),
            }
        return alpha


def attach_state_routed_alpha(encoder, router: StateRouter, block_1idx: int):
    """블록 k(1-indexed) 출력 직후 forward-hook 으로 α 를 만들어 주입한다.

    P55BlockCapture(p55_gate.py)와 같은 hook 규약이지만 버퍼에 쌓지 않고 즉시
    라우터를 돌려 블록 k+1~ 의 StateRoutedLoRAQKV.alpha 에 넣는다 — 블록 k 출력은
    블록 k+1 forward 전에 이미 나와 있으므로 timm 블록 루프를 바꾸지 않고 순차
    forward 안에서 처리된다. 현재 모달은 encoder.set_modality 가 심은
    active_modality에서 읽는다(모달마다 1회 발화 = 순차 forward 가정).
    """
    layers = list(getattr(encoder, 'state_routed_layers', []))
    if not layers:
        raise RuntimeError(
            "[P56-C] encoder.state_routed_layers 가 비었다 — LORA_MODE=state_routed "
            "로 인코더를 만들었는지 확인하라.")
    if not (1 <= int(block_1idx) <= len(encoder.backbone.blocks)):
        raise ValueError(
            f"[P56-C] 라우터 블록 {block_1idx} 가 백본 블록 수 "
            f"{len(encoder.backbone.blocks)} 밖에 있다.")

    def hook(_module, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if t.dim() != 3:
            raise RuntimeError(
                f"[P56-C] 블록 {block_1idx} 출력이 (B,N,C) 3차원이어야 하는데 "
                f"{tuple(t.shape)} — α 를 만들 수 없다 (조용한 skip 금지).")
        m = encoder.lora_layers[0].active_modality
        alpha = router(t, m)
        for w in layers:
            w.alpha = alpha

    return encoder.backbone.blocks[int(block_1idx) - 1].register_forward_hook(hook)
