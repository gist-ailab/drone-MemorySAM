"""[P47-2] Uni-modal Balance (구 D-2) — P39.1-rank 위 학습-전용 토글 모듈.

설계 정본: `.claude_logs/decisions/2026-08-03-p47-mub-muses-proposal.md` §3 D-2.
(문서는 "D-2"로 적혀 있다. 네이밍 규칙이 `P<모델>-<모듈>`로 바뀌어 코드/config는
`P47_2` / `p47_2`를 쓴다.)

진단
  **우리 모델에서** 모달을 추가할수록 성능이 떨어진다 — 4모달 val 82.35 / 공식
  test 79.571 < 3모달 82.62 / 79.788, drop-radar +0.13 (전부 within-method 실측).
  🔴 2026-08-04 철회: 초판은 "리더보드 camera-only 82.39 > C+L 81.07 > 4모달
  79.49"를 근거로 들었으나 이는 **서로 다른 방법론 간 비교라 교란**이다. 통제
  ablation은 정반대로 단조 증가한다(CAFuser Table IX: RGB 55.7 → +L 58.7 →
  +R 59.3 → +E 59.7). 즉 이것은 벤치의 법칙이 아니라 **우리 모델의 증상**이다.
  문헌 기제 = **modality laziness / greedy joint learning** — 융합 손실만으로
  학습하면 지배 모달(RGB)의 uni-modal feature가 under-optimize 된다
  (2305.01233 UMT · 1905.12681 Gradient-Blending · 2202.05306 · 2203.12221 이론증명).
  자체 확증 = P46-C3의 손해가 **clear/day(RGB 주도 조건)에 집중**
  (val Δclear −1.72 / Δday −1.29, fog +0.16).

처방
  각 모달의 encoder(frozen ViT + per-modal LoRA) 출력에 **모달마다 독립인** 경량
  head를 달고 동일 GT로 CE를 준다. 손실은 주 손실에 **직접 합산**된다
  (키1 — zero-init 잔차·수동 0-게이트 금지, 4연속 반증된 실패키).

🔴 **왜 `FUSION.AUX_CE_WEIGHT`(기존 aux_ce)로 충분하지 않은가** — 중요.
  base P39.1에는 이미 per-modal aux CE가 있다: `fusion.aux_decoders[i](feats[i])`에
  대한 CE를 모달 평균해 `AUX_CE_WEIGHT`(0.5)로 더한다. 그래도 P47-2가 no-op이
  아닌 이유는 셋이다.

  1. **목적 분리**. `fusion.aux_decoders`는 다목적이다 — 그 logit이
     `_compute_signals`의 rel_cal/corroboration/consistency bias, router anchor,
     `rbma_cal_loss`(correctness-contrastive 보정), P44 mutual-KL의 입력이다.
     즉 그 head는 "정확해져라"와 "잘 보정된 신뢰도 추정기가 돼라" 사이의
     타협점으로 최적화된다. P47-2 head는 **uni-modal 정확도만** 목표로 하고
     어떤 신뢰도/게이트 경로에도 연결되지 않는다.
  2. **가중을 모달별로 줄 수 있다**. 기존 aux_ce는 모달 평균이라 4모달이면
     모달당 실효 0.5/4 = 0.125로 고정이고, "RGB에만 더 걸어라"를 표현할 수
     없다. §1 진단이 지목하는 건 **RGB 본류 표현력**이므로 `MODALS: ['img']` +
     `LAMBDA_U`가 그 직접 레버다.
  3. **OGM-GE**(2203.15332) 결선이 없다. per-modal 학습속도 불균형 보정은
     per-modal 성능 추정치를 요구하는데, 기존 aux_ce는 그 값을 노출하지 않는다.

🔴 **추론 불변**: head는 학습 전용이다. `model.eval()` 경로에서 단 한 번도
   호출되지 않고 logit에 아무것도 더하지 않는다 → P39.1과 **완전 동일한 출력**
   (P46-C3와 같은 계약; `tools/smoke_p47.py`가 |Δ|max == 0으로 확인).
   ⚠️ 체크포인트에는 head 파라미터가 **들어간다**(nn.Module이므로). 로드는
   `strict=False` 관례를 따르는 경로에서 무해하고, off 모델로 로드해도 남는
   키는 추론에 쓰이지 않는다.

🔴 **추가 forward 없음**: 같은 forward가 이미 만든 `feats[i]`(stride-16 per-modal
   토큰맵)를 재사용한다 → P46-C2/C3에서 문제가 됐던 iteration당 forward 2회
   (broadcast_buffers / unused-param, ISSUE-028) 상황이 **구조적으로 생기지 않는다.**

메모리 (실측, autograd saved-tensor 계측; dim 1024 · 1024² 입력 · 4모달 · BS1 · bf16)
   HEAD=linear  GT_DIV=4  → **+51.7 MiB/스텝**, params 336 KiB (+AdamW state 1.0 MiB)
   HEAD=linear  GT_DIV=16 → +33.4 MiB
   HEAD=conv1x1 GT_DIV=4  → +69.6 MiB, params 4.1 MiB (+AdamW 12.3 MiB)
   내역: 모달당 정규화 출력 (1,1024,64,64) bf16 8 MiB + CE의 log_softmax
   (1,19,256,256) fp32 4.75 MiB. A100 40GB BS1 기준 0.13%.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# P47-2 · per-modal uni-modal aux head
# ─────────────────────────────────────────────────────────────────────────────

def _norm2d(dim: int) -> nn.Module:
    """head 앞 정규화 — `nn.GroupNorm`(융합 커널) 사용.

    🔴 여기서 `encoder.LayerNorm2d`(mean/pow/sqrt/div/mul/add를 파이썬으로 편 것)를
    쓰면 안 된다. 그 구현은 **모달당 (B,dim,h,w) 크기의 중간 텐서 3장**을 autograd
    그래프에 남긴다 — MUSES 1024²·dim 1024·4모달에서 실측 +96 MiB(bf16)/스텝이고,
    "경량 head"라는 계약을 혼자 깨뜨린다. GroupNorm은 단일 융합 op이라 입력 +
    (mean, rstd)만 저장한다(실측 4.6배 절감). 이 리포의 다른 head들(AuxDecoder,
    FPNSegHead)도 GroupNorm(32, ·) 관례를 쓴다.
    ⚠️ GroupNorm(g>1)은 채널을 그룹으로 묶어 (그룹채널×H×W) 통계로 정규화하므로
       채널별 LN2d와 **수학적으로 같지 않다**. 여기서는 head 입력 조건화가 목적이라
       무방하다(분포를 맞춰 초기 CE 폭주를 막는 것).
    """
    for g in (32, 16, 8, 4, 2, 1):
        if dim % g == 0:
            return nn.GroupNorm(g, dim)
    return nn.GroupNorm(1, dim)                              # 도달 불가(g=1이 항상 나눔)


class UniModalHead(nn.Module):
    """모달 하나의 uni-modal 분류 head (stride-16 토큰맵 → num_classes).

    mode='linear'  : GroupNorm → 1×1 conv        (선형 프로브 1층)
    mode='conv1x1' : GroupNorm → 1×1 → GELU → 1×1 (경량 2층 MLP)

    두 모드 모두 정규화를 앞에 둔다. frozen DINOv3 ViT-L 토큰은 채널 스케일이
    크고 모달마다 분포가 달라, 정규화 없는 맨 linear는 초기 CE가 폭주해 주 손실을
    밀어낸다. 정규화는 통계만 맞추므로 "선형 프로브"의 성격(모달 feature가
    **그 자체로** 분류 가능해야 한다)은 유지된다.
    """

    def __init__(self, dim: int, num_classes: int, mode: str = 'linear',
                 hidden: int = 256):
        super().__init__()
        mode = str(mode).lower()
        if mode not in ('linear', 'conv1x1'):
            raise ValueError(f"P47_2.HEAD must be linear|conv1x1, got {mode!r}")
        self.mode = mode
        if mode == 'linear':
            self.net = nn.Sequential(_norm2d(dim),
                                     nn.Conv2d(dim, num_classes, 1))
        else:
            self.net = nn.Sequential(_norm2d(dim),
                                     nn.Conv2d(dim, hidden, 1), nn.GELU(),
                                     nn.Conv2d(hidden, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UniModalBalance(nn.Module):
    """[P47-2] 모달별 독립 head + uni-modal CE (학습 전용).

    forward(feats, gt_mask, epoch, ...) -> (loss, None일 수 있음)
      loss 는 **이미 λ_u 가 곱해진** 값이다 (리포 관례: aux dict 항목은 pre-scaled).

    파라미터 공유 금지: head는 `nn.ModuleDict`에 활성 모달 인덱스별로 하나씩
    만든다. 공유하면 "각 모달이 스스로 분류 가능해야 한다"는 제약이 무너지고,
    강한 모달이 head를 독점해 약한 모달의 gradient가 오히려 왜곡된다.

    비활성 모달에는 head를 **아예 만들지 않는다** (unused parameter를 만들지
    않기 위해서). warmup 구간에는 head가 존재하지만 호출되지 않는데, 그건
    `find_unused_parameters=True`가 처리한다 (train_reliadino.py가 항상 켠다).

    진단 스태시(파이썬 float, 학습 영향 0):
      `last_ce[i]`  모달 i의 uni-modal CE (λ 미적용) — None이면 이번 스텝 미계산
      `last_acc[i]` 모달 i의 uni-modal 픽셀 정확도 (ignore 제외) — OGM-GE 입력
    """

    def __init__(self, dim: int, num_classes: int, num_modalities: int,
                 active: Sequence[int], head: str = 'linear', hidden: int = 256,
                 lambda_u: float = 0.4, warmup_ep: int = 0, gt_div: int = 4,
                 reduce: str = 'mean', ignore_label: int = 255):
        super().__init__()
        self.M = int(num_modalities)
        self.K = int(num_classes)
        self.active = sorted({int(i) for i in active if 0 <= int(i) < self.M})
        if not self.active:
            raise ValueError("[P47-2] 활성 모달이 없다 — MODEL.P47_2.MODALS 확인")
        self.lambda_u = float(lambda_u)
        self.warmup_ep = int(warmup_ep)
        self.gt_div = max(1, int(gt_div))
        self.reduce = str(reduce).lower()
        if self.reduce not in ('mean', 'sum'):
            raise ValueError(f"P47_2.REDUCE must be mean|sum, got {reduce!r}")
        self.ignore_label = int(ignore_label)
        self.heads = nn.ModuleDict({
            str(i): UniModalHead(dim, num_classes, head, hidden)
            for i in self.active})
        self.last_ce: List[Optional[float]] = [None] * self.M
        self.last_acc: List[Optional[float]] = [None] * self.M

    # ── GT 정합 ──────────────────────────────────────────────────────────────
    def _gt_at(self, gt_mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """기존 aux(P39 deep-sup / fusion aux_ce)와 **동일 규약**: 라벨 해상도의
        1/`gt_div` 에서 CE. GT는 nearest, logit은 bilinear로 그 크기에 맞춘다."""
        d = self.gt_div
        Ht = max(1, gt_mask.shape[-2] // d)
        Wt = max(1, gt_mask.shape[-1] // d)
        gt = F.interpolate(gt_mask.unsqueeze(1).float(), size=(Ht, Wt),
                           mode='nearest').squeeze(1).long()
        return gt, (Ht, Wt)

    def _img_masked_gt(self, gt: torch.Tensor, img_mask: Optional[torch.Tensor],
                       size: Tuple[int, int]) -> torch.Tensor:
        """[발견C / P44-B3와 동일] img가 지워진 곳은 ignore 처리.

        0-입력에서 GT를 맞히도록 img 브랜치를 학습시키면 위치 기반 장면 prior를
        환각하게 된다. fusion.forward의 aux_ce가 쓰는 것과 같은 규약이다.
        img_mask.dim()==1 → P42 전역(샘플 단위), 그 외 (B,1,H,W) → P44 국소.
        """
        if img_mask is None:
            return gt
        if img_mask.dim() == 1:
            keep = (img_mask < 0.5).view(-1, 1, 1)
            return torch.where(keep, gt, torch.full_like(gt, self.ignore_label))
        reg = F.interpolate(img_mask.float(), size=size, mode='nearest')[:, 0] > 0.5
        return torch.where(reg, torch.full_like(gt, self.ignore_label), gt)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, feats: Sequence[torch.Tensor], gt_mask: torch.Tensor,
                epoch: int = 0, img_mask: Optional[torch.Tensor] = None,
                img_idx: int = -1) -> Optional[torch.Tensor]:
        """활성 모달의 uni-modal CE. 반환값은 λ_u 적용 후(pre-scaled)."""
        self.last_ce = [None] * self.M
        self.last_acc = [None] * self.M
        if epoch < self.warmup_ep:
            return None
        gt0, size = self._gt_at(gt_mask)
        terms = []
        for i in self.active:
            gt = self._img_masked_gt(gt0, img_mask, size) if i == img_idx else gt0
            valid = gt != self.ignore_label
            if not bool(valid.any()):
                continue
            lg = self.heads[str(i)](feats[i])
            lg = F.interpolate(lg.float(), size=size, mode='bilinear',
                               align_corners=False)
            ce = F.cross_entropy(lg, gt, ignore_index=self.ignore_label)
            terms.append(ce)
            self.last_ce[i] = float(ce.detach())
            with torch.no_grad():
                hit = (lg.detach().argmax(1) == gt) & valid
                self.last_acc[i] = float(hit.sum()) / float(valid.sum())
        if not terms:
            return None
        agg = sum(terms) / (len(terms) if self.reduce == 'mean' else 1.0)
        return self.lambda_u * agg


# ─────────────────────────────────────────────────────────────────────────────
# P47-2 · OGM-GE (선택 토글, 기본 off) — On-the-fly Gradient Modulation
# ─────────────────────────────────────────────────────────────────────────────

class OGMGE:
    """[P47-2 opt] 모달별 학습속도 불균형 보정 (2203.15332, 기본 off).

    per-modal uni-modal 정확도 s_m 을 그 모달의 "기여도"로 보고, 평균보다 앞서
    가는 모달의 **자기 LoRA gradient만** 감쇠한다:

        ρ_m = s_m / mean_{j≠m} s_j
        k_m = clamp( 1 − tanh(α · relu(ρ_m − 1)) , min_k, 1 )

    ρ_m ≤ 1(평균 이하)이면 k_m = 1 = 무개입이다. 즉 뒤처진 모달을 **키우는 게
    아니라** 앞선 모달을 늦춘다 (원논문과 동일 방향; 손실 스케일을 건드리지
    않으므로 주 손실의 의미는 불변).

    적용 대상 = `MultiModalLoRAQKV`의 (M, …) 파라미터(`a_q/b_q/a_v/b_v`)의
    **모달 슬라이스**뿐이다. 이 리포는 모달별 LoRA를 하나의 텐서 0번 축에
    쌓아 두므로 `p.grad[m] *= k_m` 이 정확히 "모달 m의 인코더 gradient"다.
    fusion/FPN/head/trunk_exp 등 공유·후단 파라미터는 건드리지 않는다 —
    거기서 모달을 분리하는 건 정의되지 않고, 주 경로 손실을 왜곡한다.

    🔴 DDP: gradient는 backward의 DDP hook에서 이미 all-reduce 된 뒤다. 각
       rank가 자기 배치로 잰 s_m 으로 서로 다른 k를 곱하면 **rank 간 파라미터가
       갈라진다.** 그래서 k를 만들기 전에 s를 all_reduce(mean) 한다. 이 collective
       는 optimizer step마다 **전 rank가 대칭으로** 1회(크기 M) 호출한다
       (2026-07-16 NCCL 데드락은 rank0 단독 collective가 원인이었다 — 여기 해당 없음).

    AMP: GradScaler가 스케일한 gradient에 상수 k를 곱하는 것은 스케일과 교환
    가능하다(k·(s·g) = s·(k·g)) → `unscale_` 없이 step 직전에 적용해도 안전하다.

    GE(generalization enhancement, 원논문의 gradient Gaussian noise)는
    `ge_noise > 0` 일 때만 켠다. 기본 0 = OGM만.
    """

    def __init__(self, model: nn.Module, num_modalities: int, alpha: float = 0.5,
                 ema: float = 0.9, min_k: float = 0.1, ge_noise: float = 0.0):
        self.M = int(num_modalities)
        self.alpha = float(alpha)
        self.ema = float(ema)
        self.min_k = float(min_k)
        self.ge_noise = float(ge_noise)
        self.params = [p for n, p in model.named_parameters()
                       if n.endswith(('.a_q', '.b_q', '.a_v', '.b_v'))
                       and p.dim() >= 2 and p.shape[0] == self.M]
        if not self.params:
            raise RuntimeError(
                "[P47-2/OGM] 모달 슬라이스를 가진 LoRA 파라미터를 찾지 못했다 — "
                "encoder.MultiModalLoRAQKV 레이아웃이 바뀌었는지 확인하라")
        self._sum = torch.zeros(self.M, dtype=torch.float64)
        self._n = torch.zeros(self.M, dtype=torch.float64)
        self.score: Optional[torch.Tensor] = None      # EMA 후 s_m
        self.last_k = [1.0] * self.M

    # ── micro-step마다: per-modal 점수 관측 ─────────────────────────────────
    def observe(self, scores: Sequence[Optional[float]]) -> None:
        for i, s in enumerate(scores[:self.M]):
            if s is None:
                continue
            self._sum[i] += float(s)
            self._n[i] += 1.0

    # ── optimizer step 직전: k 계산 후 gradient 변조 ────────────────────────
    @torch.no_grad()
    def apply_(self) -> List[float]:
        seen = self._n > 0
        if not bool(seen.any()):
            self.last_k = [1.0] * self.M
            return self.last_k
        cur = torch.where(seen, self._sum / self._n.clamp(min=1.0),
                          torch.zeros_like(self._sum))
        self._sum.zero_(); self._n.zero_()
        if dist.is_available() and dist.is_initialized():
            # rank 간 동일한 k를 쓰기 위한 대칭 collective (위 docstring 참조).
            # nccl 백엔드는 CUDA 텐서만 받으므로 파라미터가 사는 device로 올린다.
            dev = self.params[0].device
            buf = torch.stack([cur, seen.to(cur.dtype)]).to(dev, torch.float32)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf = buf.double().cpu()
            cur = buf[0] / buf[1].clamp(min=1.0)
            seen = buf[1] > 0
        self.score = cur if self.score is None else (
            self.ema * self.score + (1.0 - self.ema) * cur)
        s = torch.where(seen, self.score, torch.zeros_like(self.score))
        tot, cnt = float(s[seen].sum()), int(seen.sum())
        k = torch.ones(self.M, dtype=torch.float64)
        if cnt >= 2:
            for i in range(self.M):
                if not bool(seen[i]):
                    continue
                others = (tot - float(s[i])) / (cnt - 1)
                if others <= 1e-8:
                    continue
                rho = float(s[i]) / others
                k[i] = 1.0 - torch.tanh(
                    torch.tensor(self.alpha * max(rho - 1.0, 0.0))).item()
            k.clamp_(min=self.min_k, max=1.0)
        self.last_k = [float(v) for v in k]
        for p in self.params:
            if p.grad is None:
                continue
            kk = k.to(device=p.grad.device, dtype=p.grad.dtype)
            p.grad.mul_(kk.view(self.M, *([1] * (p.grad.dim() - 1))))
            if self.ge_noise > 0:
                std = p.grad.detach().std()
                if torch.isfinite(std) and float(std) > 0:
                    p.grad.add_(torch.randn_like(p.grad) * (self.ge_noise * std))
        return self.last_k


# ─────────────────────────────────────────────────────────────────────────────
# config 해석 유틸
# ─────────────────────────────────────────────────────────────────────────────

def resolve_modals(spec, modalities: Sequence[str]) -> List[int]:
    """`MODEL.P47_2.MODALS` 를 모달 인덱스 리스트로.

    'all'(기본) | 이름 리스트(['img']) | 인덱스 리스트([0, 2]) 를 받는다.
    이름이 구성에 없으면 **조용히 무시하지 않고 죽는다** — 오타 하나로 P47-2가
    무음 no-op이 되면 300ep을 태우고서야 안다(ISSUE-024류).
    """
    mods = list(modalities)
    if spec is None or (isinstance(spec, str) and str(spec).lower() == 'all'):
        return list(range(len(mods)))
    if isinstance(spec, str):
        spec = [spec]
    out = []
    for v in spec:
        if isinstance(v, str):
            if v not in mods:
                raise ValueError(
                    f"[P47-2] MODALS 에 '{v}' 가 있으나 DATASET.MODALS={mods} 에 없다")
            out.append(mods.index(v))
        else:
            i = int(v)
            if not (0 <= i < len(mods)):
                raise ValueError(f"[P47-2] MODALS 인덱스 {i} 가 범위 밖 (M={len(mods)})")
            out.append(i)
    return sorted(set(out))
