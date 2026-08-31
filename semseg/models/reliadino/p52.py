"""[P52] RxDINO 자기-적응 컨트롤러 — C3-adaptive + UniBal-adaptive (학습 전용).

설계 정본: `.claude_logs/decisions/2026-08-31-p52-rxdino-adaptive-amendment.md`
(원 P52 "오프라인 진단이 벤치별 C3 on/off를 결정"의 개정 — 사람이 하는 per-dataset
튜닝과 구분 불가하다는 user 지적 수용. 개정 = **진단의 온라인화**: 학습 중 계산되는
병리 지표가 처방 강도를 연속 조절한다. 3벤치 완전 동일 config.)

  C3Adaptive    : train 배치 argmax↔GT 혼동 EMA에서 per-class "흡수도"
                  s_c = (1−recall_c)·conc_c 를 계산, λ_c = λ_max·clamp(s_c/τ,0,1).
                  붕괴 클래스(다른 클래스로 흡수되는)만 P46-C3 prototype 당김을
                  받는다(현행 = 전 클래스 균일 λ). [P52] G4의 DELIVER 예측 =
                  RailTrack λ↑.
  UniBalAdaptive: P47-2가 이미 계산하는 모달별 uni-modal CE L_m의 EMA에서
                  laziness g_m = clamp(L_m/mean_k L_k − 1, 0, CAP) 를 계산,
                  λ_u,m = λ_u_max·g_m/CAP. 혼자서 못 푸는(게으른) 모달만 보조
                  CE가 강화된다(선례 OGM-GE CVPR'22 — 학습-내 모달 기여 측정→
                  변조). [P52] G4의 MUSES 예측 = radar λ_u↑.

🔴 계약 (decisions 문서 §1 그대로):
  - 두 컨트롤러 모두 **학습 전용·train-배치 통계만 사용(val 불사용)**.
    model.eval() 경로에서는 단 한 번도 호출되지 않는다(추론 그래프 불변).
  - EMA 평활·warmup·λ 궤적 로깅(창발 증거 = 논문 그림, [C3-ADPT]/[UB-ADPT]).
  - buffer는 `persistent=False` — state_dict 키가 늘지 않으므로 기존 ckpt
    로드·저장이 어떤 조합에서도 깨지지 않는다(스펙 §3 가드).
  - 파라미터 0개 → DDP find_unused/optimizer와 무관.

⚠️ DDP: persistent=False buffer도 named_buffers()에는 나오므로
   broadcast_buffers=True(기본)가 매 forward마다 rank0의 값을 전 rank에
   덮어쓴다. 실효 동작 = "전 rank가 rank0이 본 혼동/손실 EMA를 공유" — rank0
   스트림도 iid 표본이라 수치적으로 무해하고 오히려 rank 간 λ가 정확히 일치해
   유리하다(RCA [P40-F1] 주석과 같은 결론. collective를 새로 만들지 않는다 =
   2026-07-16 NCCL 데드락 이력 회피).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

_EPS = 1e-8


def _names(seq: Optional[Sequence[str]], i: int, prefix: str = 'c') -> str:
    """로깅용 표시명 — 클래스/모달 이름이 없으면 인덱스로."""
    if seq is not None and 0 <= i < len(seq):
        return str(seq[i])
    return f"{prefix}{i}"


# ─────────────────────────────────────────────────────────────────────────────
# [P52] C3-adaptive — per-class λ_c 컨트롤러
# ─────────────────────────────────────────────────────────────────────────────

class C3Adaptive(nn.Module):
    """train 배치 혼동 EMA → per-class 붕괴 지표 s_c → λ_c.

    observe(logits, gt) 를 주 forward마다 호출(model이 보조 branch
    `_p46_replay_path`에서는 호출하지 않는다 — 마스킹/스타일 변주된 입력은
    "train 배치"의 통계가 아니므로 EMA를 오염시킨다).

    지표(decisions 문서 §1):
      M[c,j]   = GT가 c인 픽셀의 예측 분포(행 정규화)의 EMA
      recall_c = M[c,c]
      conc_c   = max_{j≠c} M[c,j] / max(Σ_{j≠c} M[c,j], eps)   (top-1 confuser 집중도)
      s_c      = (1 − recall_c) · conc_c        (해당 클래스 미출현 시 0)
      λ_c      = LAMBDA_MAX · clamp(s_c / TAU, 0, 1)
    recall만 보면 "그냥 어려운" 클래스까지 잡히고, conc만 보면 오답이 흩어진
    붕괴(=특정 클래스로 쏠리는 흡수)를 못 잡는다. 곱으로 결합해 **도메인 전이
    붕괴의 형태**(Wall→Building 식 흡수)에만 반응하게 한다.

    WARMUP_EP 동안은 λ_c=0(혼동 EMA 워밍업). LAMBDA_MAX 기본 0.1 = P46.C3_PROTO
    의 현행 균일 λ — 즉 모든 클래스가 s_c≥TAU면 균일 λ와 같아져 상한이 된다.
    """

    def __init__(self, num_classes: int, lambda_max: float = 0.1, tau: float = 0.25,
                 momentum: float = 0.99, warmup_ep: int = 5,
                 ignore_label: int = 255):
        super().__init__()
        self.K = int(num_classes)
        self.lambda_max = float(lambda_max)
        self.tau = max(float(tau), 1e-6)
        self.m = float(momentum)
        self.warmup_ep = int(warmup_ep)
        self.ignore_label = int(ignore_label)
        # (K,K) 혼동 EMA(행=GT c, 열=예측 j) + 클래스 출현 비트. persistent=False
        # → state_dict 불변(기존 ckpt 로드 가드).
        self.register_buffer('conf_ema', torch.zeros(self.K, self.K),
                             persistent=False)
        self.register_buffer('seen', torch.zeros(self.K), persistent=False)

    @torch.no_grad()
    def observe(self, logits: torch.Tensor, gt: torch.Tensor) -> None:
        """이번 train 배치의 argmax vs GT → 혼동 행렬 EMA 갱신."""
        K = self.K
        pred = logits.detach().reshape(logits.shape[0], logits.shape[1], -1)
        pred = pred.argmax(1).reshape(-1)
        g = gt.detach().reshape(-1)
        keep = (g >= 0) & (g < K) & (g != self.ignore_label)
        g, pred = g[keep], pred[keep]
        if g.numel() == 0:
            return
        mb = torch.bincount(g * K + pred, minlength=K * K).reshape(K, K).float()
        fresh_row = mb.sum(1) > 0
        prob = mb / mb.sum(1, keepdim=True).clamp(min=1.0)
        # 처음 본 클래스는 EMA warm-up 없이 즉시 반영(ClassLossEMA와 같은 규약 —
        # 0에서 서서히 오르는 동안 붕괴 클래스가 과소평가되는 역전 방지).
        is_fresh = fresh_row & (self.seen < 0.5)
        if bool(is_fresh.any()):
            self.conf_ema[is_fresh] = prob[is_fresh]
        is_old = fresh_row & (self.seen > 0.5)
        if bool(is_old.any()):
            self.conf_ema[is_old] = (self.m * self.conf_ema[is_old]
                                     + (1.0 - self.m) * prob[is_old])
        self.seen[fresh_row] = 1.0

    def scores(self) -> torch.Tensor:
        """per-class 붕괴 지표 s_c (K,). 관측된 적 없는 클래스는 0."""
        m = self.conf_ema
        row = m.sum(1)
        live = (self.seen > 0.5) & (row > _EPS)
        s = torch.zeros(self.K, device=m.device, dtype=m.dtype)
        if not bool(live.any()):
            return s
        diag = m.diagonal()
        recall = diag / row.clamp(min=_EPS)
        off = m - torch.diag_embed(diag)                       # 대각(정답) 제외
        conc = off.max(1).values.clamp(min=0.0) / (off.sum(1).clamp(min=0.0)
                                                   + _EPS)
        s = torch.where(live, (1.0 - recall) * conc,
                        torch.zeros_like(s))
        return s.clamp(min=0.0)

    def lambdas(self, epoch: int) -> torch.Tensor:
        """per-class 가중 λ_c (K,). WARMUP_EP 미만은 전 클래스 0."""
        if epoch < self.warmup_ep:
            return torch.zeros(self.K, device=self.conf_ema.device)
        return self.lambda_max * (self.scores() / self.tau).clamp(0.0, 1.0)

    def log_lines(self, epoch: int,
                  class_names: Optional[Sequence[str]] = None) -> List[str]:
        """train.log 로깅([C3-ADPT]). top-5 λ_c + s_c 분포 요약."""
        lam = self.lambdas(epoch)
        s = self.scores()
        lam_c, s_c = lam.detach().cpu(), s.detach().cpu()
        top5 = torch.argsort(lam_c, descending=True)[:5]
        l1 = (f"[C3-ADPT] ep{epoch} λ_c top5: " +
              " ".join(f"{_names(class_names, int(c))}:{float(lam_c[c]):.4f}"
                       for c in top5))
        n_pos = int((lam_c > 0).sum())
        s_arr = s_c[self.seen.detach().cpu() > 0.5]
        if s_arr.numel():
            smax_i = int(torch.argmax(s_arr))
            seen_names = [i for i in range(self.K)
                          if float(self.seen[i]) > 0.5]
            stat = (f"mean:{float(s_arr.mean()):.4f} "
                    f"p50:{float(s_arr.median()):.4f} "
                    f"max:{_names(class_names, seen_names[smax_i])}:"
                    f"{float(s_arr.max()):.4f}")
        else:
            stat = "no-observation"
        l2 = (f"[C3-ADPT] ep{epoch} s_c {stat} | λ_c>0: {n_pos}/{self.K} "
              f"| conf-EMA seen: {int((self.seen > 0.5).sum())}/{self.K}")
        return [l1, l2]


# ─────────────────────────────────────────────────────────────────────────────
# [P52] UniBal-adaptive — per-modal λ_u,m 컨트롤러
# ─────────────────────────────────────────────────────────────────────────────

class UniBalAdaptive(nn.Module):
    """모달별 uni-modal CE(L_m) EMA → laziness g_m → λ_u,m.

    L_m은 P47-2 UniModalBalance가 **이미 계산하는** 값(`last_ce[i]`, λ 미적용)
    이다 — 관측 비용 0. observe는 model이 p47_2.forward 직후에 호출한다.

    지표(decisions 문서 §1):
      g_m    = clamp(L_m_ema / mean_k(L_k_ema) − 1, 0, CAP)
      λ_u,m  = LAMBDA_U_MAX · g_m / CAP
    평균 대비 손실이 큰 모달 = 융합에 기대만는(스스로는 못 푸는) 게으른 모달 →
    보조 CE 강화. 균형 잡힌 모달은 →0. LAMBDA_U_MAX 기본 0.4 = P47_2 현행 λ_u.

    WARMUP_EP 동안은 균일 λ_u 소값(0.05) — L_m EMA가 수렴하기 전의 상대 비교
    는 노이즈다(스펙 §2). 관측된 모달이 2개 미만이면 mean_k 정의가 없으므로
    같은 소값을 유지한다(조용한 0 = 무음 no-op 방지, ISSUE-024류). 한 train
    step에 활성 모달의 CE가 전부 같이 계산되므로 실질적으로 첫 스텝 후 해소.
    """

    def __init__(self, num_modalities: int, active: Sequence[int],
                 lambda_u_max: float = 0.4, cap: float = 2.0,
                 momentum: float = 0.99, warmup_ep: int = 0,
                 warmup_small: float = 0.05):
        super().__init__()
        self.M = int(num_modalities)
        self.active = sorted({int(i) for i in active if 0 <= int(i) < self.M})
        self.lambda_u_max = float(lambda_u_max)
        self.cap = max(float(cap), 1e-6)
        self.m = float(momentum)
        self.warmup_ep = int(warmup_ep)
        self.warmup_small = float(warmup_small)
        # (M,) 손실 EMA + 관측 비트. persistent=False → state_dict 불변.
        self.register_buffer('l_ema', torch.zeros(self.M), persistent=False)
        self.register_buffer('observed', torch.zeros(self.M), persistent=False)

    @torch.no_grad()
    def observe(self, last_ce: Sequence[Optional[float]]) -> None:
        """UniModalBalance.last_ce(λ 미적용 uni-modal CE)를 EMA에 반영."""
        for i, c in enumerate(list(last_ce)[:self.M]):
            if c is None or not math.isfinite(float(c)):
                continue
            if float(self.observed[i]) < 0.5:
                self.l_ema[i] = float(c)          # 첫 관측 즉시 초기화
                self.observed[i] = 1.0
            else:
                self.l_ema[i].mul_(self.m).add_(float(c), alpha=1.0 - self.m)

    def lambdas(self, epoch: int) -> torch.Tensor:
        """per-modal λ_u,m (M,). 비활성 모달은 0(head가 없어 쓰이지 않는다)."""
        out = torch.zeros(self.M, device=self.l_ema.device)
        live = [i for i in self.active if float(self.observed[i]) > 0.5]
        if epoch < self.warmup_ep or len(live) < 2:
            for i in self.active:
                out[i] = self.warmup_small
            return out
        vals = torch.stack([self.l_ema[i] for i in live])
        mean = float(vals.mean())
        for i in live:
            g = min(max(float(self.l_ema[i]) / max(mean, _EPS) - 1.0, 0.0),
                    self.cap)
            out[i] = self.lambda_u_max * g / self.cap
        return out

    def log_lines(self, epoch: int,
                  modal_names: Optional[Sequence[str]] = None) -> List[str]:
        """train.log 로깅([UB-ADPT]). 모달별 λ_u,m · L_m_ema."""
        lam = self.lambdas(epoch).detach().cpu()
        act = set(self.active)
        l1 = (f"[UB-ADPT] ep{epoch} λ_u: " +
              " ".join(f"{_names(modal_names, i, 'm')}:{float(lam[i]):.4f}"
                       if i in act else f"{_names(modal_names, i, 'm')}:-"
                       for i in range(self.M)))
        l2 = (f"[UB-ADPT] ep{epoch} L_ema: " +
              " ".join(f"{_names(modal_names, i, 'm')}:"
                       f"{float(self.l_ema[i]):.4f}"
                       if float(self.observed[i]) > 0.5
                       else f"{_names(modal_names, i, 'm')}:-"
                       for i in range(self.M)))
        return [l1, l2]
