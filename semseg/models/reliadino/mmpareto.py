"""[P44-B1] MMPareto gradient integration (arXiv 2405.17730) — trainer-level.

문제: per-modal aux CE(deep-sup)와 주 CE를 **그냥 더하면**, 두 목표의 gradient가
충돌하는 파라미터에서 주 손실이 이긴다. img가 지배적인 우리 계보에서는 그게
곧 "비RGB 브랜치가 학습되지 않는다"(drop-modality dMIoU≈0)로 나타난다.

MMPareto의 두 단계 규칙:
  cos(g_main, g_aux) ≥ 0  →  g = g_main + g_aux           (이미 공통 하강 방향)
  cos(g_main, g_aux) < 0  →  g = Pareto(min-norm) 방향으로 충돌 성분을 제거한 뒤
                              **크기를 ‖g_main‖+‖g_aux‖ 수준으로 복원**
크기 복원이 핵심이다 — 순수 min-norm 방향은 크기가 줄어들고, 논문은 그 축소가
SGD 노이즈/일반화를 해쳐 성능이 떨어짐을 보인다(OGM/AGM/PMR이 uniform baseline
미달인 것과 같은 함정). 두 벡터의 min-norm 점은 닫힌형:
  α* = clamp((‖ga‖² − g·) / ‖gm − ga‖², 0, 1),  d = α*·gm + (1−α*)·ga
이고 d·gm = d·ga = ‖d‖² ≥ 0 이므로 두 목표 모두에 대한 공통 하강 방향이다.
모든 스칼라(내적·노름·‖d‖)는 **원소별 누적 스칼라만으로** 닫힌형 계산되므로
gradient를 하나의 큰 벡터로 이어붙이지 않는다(추가 메모리 0).

그룹 (spec: per-modal LoRA + shared trunk):
  - `lora_<modal>` : 그 모달의 LoRA 슬라이스 전부. MultiModalLoRAQKV는 a_q/b_q/
    a_v/b_v를 (M, ·, ·) 한 텐서로 들고 있고 modality m의 forward는 슬라이스 m만
    건드리므로, **합산 aux CE의 gradient가 슬라이스 단위로 이미 분리**돼 있다
    (∂Σ_m L_aux^m/∂a_q[m] = ∂L_aux^m/∂a_q[m]). 별도 backward가 필요 없다.
  - `shared`       : 나머지 학습 파라미터(fusion·head·aux decoder·m2f…) 전체.

🔴 DDP 결정 (설계 선택, 문서화 의무):
  **allreduce 이후에 결합한다.** 각 micro-step은 `model.no_sync()` 안에서
  `torch.autograd.grad`로 두 번 미분하고(=DDP reducer 미개입), accumulation
  경계에서 g_main·g_aux를 **각각** all_reduce(SUM)/world_size 한 다음 cos·α를
  계산한다. 따라서 (a) 전 rank가 **동일한** 전역 gradient 위에서 동일한 결합을
  계산 → 파라미터가 rank 간 절대 벌어지지 않고, (b) 의미론이 단일 GPU에서 전역
  배치로 돌린 것과 정확히 일치한다(로컬 배치별 cos를 쓰면 world마다 결과가
  달라짐). 비용은 optimizer step당 allreduce 2×(DDP 1× 대비).
  ⚠️ `autograd.grad`는 AccumulateGrad를 우회하므로 DDP hook이 절대 발화하지
  않는다 — no_sync 없이 forward하면 다음 iteration에서 "Expected to have
  finished reduction in the prior iteration"으로 죽는다. no_sync는 필수다.

AMP: 두 backward는 `scaler.scale(loss)`로 걸고, 누적 시 `inv_scale`(=1/scale)을
곱해 **cos 계산 전에 unscale**한다(bf16/AMP off면 scale=1). 누적·결합은 전부
fp32 파라미터 dtype에서 일어난다.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import distributed as dist

# MultiModalLoRAQKV의 per-modality 어댑터 파라미터 (encoder.py). 이름 끝
# 세그먼트로 판정 — 중첩 경로(encoder.vit.blocks.N.attn.qkv.a_q)와 평면 이름
# 모두에서 동작한다.
_LORA_LEAF = ('a_q', 'b_q', 'a_v', 'b_v')


class MMPareto:
    """gradient 통합기. OFF일 때는 아예 생성하지 않는다(트레이너가 분기)."""

    def __init__(self,
                 named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
                 num_modalities: int,
                 modal_names: Optional[Sequence[str]] = None,
                 interval: int = 1,
                 magnitude: str = 'sum_norm',
                 eps: float = 1e-12):
        self.num_modalities = int(num_modalities)
        self.modal_names = list(modal_names) if modal_names else [
            str(i) for i in range(self.num_modalities)]
        self.interval = max(1, int(interval))
        self.magnitude = str(magnitude).lower()
        self.eps = float(eps)

        self.params: List[torch.nn.Parameter] = []
        self.names: List[str] = []
        for n, p in named_parameters:
            if p.requires_grad:
                self.params.append(p)
                self.names.append(n)
        # 그룹 구성: entries = (param_index, slice_index or None)
        lora: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_modalities)]
        shared: List[Tuple[int, Optional[int]]] = []
        for i, (n, p) in enumerate(zip(self.names, self.params)):
            if (n.split('.')[-1] in _LORA_LEAF and p.dim() >= 1
                    and p.shape[0] == self.num_modalities):
                for m in range(self.num_modalities):
                    lora[m].append((i, m))
            else:
                shared.append((i, None))
        self.groups: List[Dict] = []
        for m in range(self.num_modalities):
            if lora[m]:
                self.groups.append({'name': f'lora_{self.modal_names[m]}',
                                    'entries': lora[m]})
        if shared:
            self.groups.append({'name': 'shared', 'entries': shared})
        self._aux: List[Optional[torch.Tensor]] = [None] * len(self.params)
        self.last_stats: Dict[str, float] = {}

    # ── lifecycle ────────────────────────────────────────────────────────
    def active(self, update_idx: int) -> bool:
        """이 optimizer-step 윈도에서 MMPareto를 쓸지 (INTERVAL 탈출 밸브)."""
        return (int(update_idx) % self.interval) == 0

    def _ensure_buffers(self) -> None:
        """전 rank가 **같은 텐서 집합**에 대해 collective를 호출하도록, 이번
        스텝에 gradient가 없던 파라미터도 0 버퍼를 갖게 한다 (path dropout 등
        확률 분기 때문에 rank마다 unused 집합이 다를 수 있음)."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if self._aux[i] is None:
                self._aux[i] = torch.zeros_like(p)

    def accumulate(self, grads_main: Sequence[Optional[torch.Tensor]],
                   grads_aux: Sequence[Optional[torch.Tensor]],
                   inv_scale: float = 1.0) -> None:
        """micro-step의 두 gradient를 누적. main은 p.grad에, aux는 내부 버퍼에.

        inv_scale = 1/GradScaler.scale (fp16). **여기서 unscale**하므로 이후
        cos/노름 계산은 전부 실 gradient 스케일에서 일어난다.
        """
        self._ensure_buffers()
        for i, p in enumerate(self.params):
            gm = grads_main[i]
            if gm is not None:
                p.grad.add_(gm.to(p.grad.dtype), alpha=inv_scale)
            ga = grads_aux[i]
            if ga is not None:
                self._aux[i].add_(ga.to(self._aux[i].dtype), alpha=inv_scale)

    def _allreduce(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        if world <= 1:
            return
        inv = 1.0 / world
        # 텐서별 all_reduce (flat 버퍼를 만들지 않는다 — 학습 파라미터 전체
        # 크기의 임시 사본 2개가 24GB 카드에서 부담이고, optimizer step당
        # 수백 회의 소형 collective는 1024² 스텝 시간에 비해 무시 가능).
        for i, p in enumerate(self.params):
            dist.all_reduce(p.grad)
            p.grad.mul_(inv)
            dist.all_reduce(self._aux[i])
            self._aux[i].mul_(inv)

    @torch.no_grad()
    def combine(self) -> Dict[str, float]:
        """accumulation 경계에서 호출. allreduce → 그룹별 결합 → p.grad에 기록.

        반환: 로깅용 stats (그룹별 cos, 충돌 여부, 복원 배율).
        """
        self._ensure_buffers()
        self._allreduce()
        stats: Dict[str, float] = {}
        for g in self.groups:
            entries = g['entries']
            # 🔴 스칼라는 **디바이스에서** 누적하고 그룹당 딱 한 번만 CPU로 내린다.
            # (엔트리마다 float()를 부르면 shared 그룹 ~200개 × 3회 = optimizer
            # step당 수백 번의 GPU 동기화가 되어 스텝 시간을 잡아먹는다.)
            acc = None
            for idx, sl in entries:
                gm = self.params[idx].grad
                ga = self._aux[idx]
                if sl is not None:
                    gm, ga = gm[sl], ga[sl]
                fm, fa = gm.flatten(), ga.flatten()
                cur = torch.stack([torch.dot(fm, fa), torch.dot(fm, fm),
                                   torch.dot(fa, fa)])
                acc = cur if acc is None else acc + cur
            dot, nm2, na2 = (acc.tolist() if acc is not None else (0.0, 0.0, 0.0))
            nm, na = max(nm2, 0.0) ** 0.5, max(na2, 0.0) ** 0.5
            cos = dot / max(nm * na, self.eps) if (nm > 0 and na > 0) else 0.0
            conflict = (dot < 0.0) and (nm > 0) and (na > 0)
            if not conflict:
                # 합의(또는 한쪽이 0) → 단순 합. 정확히 기존 trainer와 같은 방향.
                alpha, scale = None, 1.0
            else:
                denom = nm2 - 2.0 * dot + na2               # ‖gm − ga‖²
                alpha = (na2 - dot) / max(denom, self.eps)
                alpha = min(max(alpha, 0.0), 1.0)
                d2 = (alpha * alpha * nm2 + 2.0 * alpha * (1.0 - alpha) * dot
                      + (1.0 - alpha) * (1.0 - alpha) * na2)
                d = max(d2, 0.0) ** 0.5
                if d <= self.eps * max(nm + na, self.eps):
                    # 거의 정반대 → min-norm 방향이 0에 수렴. 여기서 크기를
                    # 복원하면 순수 노이즈를 증폭한다 → 주 gradient로 폴백.
                    alpha, scale = 1.0, 1.0
                    conflict = False
                elif self.magnitude == 'none':
                    scale = 1.0
                elif self.magnitude == 'main':
                    scale = nm / d
                else:                                       # 'sum_norm' (기본)
                    scale = (nm + na) / d
            for idx, sl in entries:
                gm_full = self.params[idx].grad
                ga_full = self._aux[idx]
                gm = gm_full if sl is None else gm_full[sl]
                ga = ga_full if sl is None else ga_full[sl]
                if alpha is None:
                    gm.add_(ga)                             # g_main + g_aux
                else:
                    gm.mul_(alpha * scale).add_(ga, alpha=(1.0 - alpha) * scale)
            stats[f"cos_{g['name']}"] = cos
            stats[f"conflict_{g['name']}"] = 1.0 if conflict else 0.0
        self.last_stats = stats
        return stats

    def reset(self) -> None:
        """optimizer.zero_grad 와 짝. aux 버퍼를 0으로 (해제하지 않고 재사용)."""
        for a in self._aux:
            if a is not None:
                a.zero_()
