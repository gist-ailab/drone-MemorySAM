"""[P44-BMR] Balanced Modality Rebalance — pure functional pieces.

설계: decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md §3 (P44) · §4 (P45) · §7-b.
여기 있는 것은 **부작용 없는 순수 함수들**뿐이다 (모듈/파라미터 0). 결선은
model.py(B-3/V-1/P45) · fusion.py(B-2) · train_reliadino.py(B-1)가 한다.
파라미터를 새로 만들지 않는 이유 = 키1(zero-init 잔차·게이트 결선으로 모듈을
사장시키는 4연속 실패 패턴) 회피: P44는 전부 손실/입력/gradient 레벨이다.

구성:
  B-3  sample_region_mask   커버리지 패턴 국소 img 마스킹 (P42 전역 마스킹의 승격)
  B-2  mutual_kl            per-modal aux logit 간 대칭 KL (DML, teacher 없음)
  B-2  relational_correspondence   모달 간 관계형 대응 (feature copy 아님)
  V-1  presence_masks       결정론적 presence 마스크 (학습 파라미터 0)
  P45  style_perturb        img 브랜치 feature-space style 섭동 (픽셀 증강 아님)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-6


def ramp(epoch: int, warmup_ep: int) -> float:
    """P42/modal_dropout와 동일한 선형 커리큘럼: min(1, epoch/warmup_ep)."""
    if warmup_ep and warmup_ep > 0:
        return min(1.0, float(epoch) / float(warmup_ep))
    return 1.0


# ── B-3: coverage-pattern local masking ─────────────────────────────────────
def _rect_mask(mask: torch.Tensor, b: int, area_ratio: Sequence[float],
               num_regions: Sequence[int]) -> None:
    """mask[b] (1,H,W)에 랜덤 사각형 영역들을 1로 채운다 (in-place)."""
    _, H, W = mask[b].shape
    lo, hi = float(area_ratio[0]), float(area_ratio[1])
    n_lo, n_hi = int(num_regions[0]), int(num_regions[1])
    n = int(torch.randint(n_lo, n_hi + 1, (1,)).item()) if n_hi > n_lo else n_lo
    n = max(n, 1)
    total_area = (lo + (hi - lo) * float(torch.rand(1).item())) * H * W
    for _ in range(n):
        a = total_area / n
        # aspect ratio 0.5~2.0 (log-uniform 근사) — 세로/가로 편향 없이
        ar = float(torch.empty(1).uniform_(0.5, 2.0).item())
        rh = int(min(H, max(1, round((a * ar) ** 0.5))))
        rw = int(min(W, max(1, round(a / max(rh, 1)))))
        top = int(torch.randint(0, max(H - rh, 1), (1,)).item())
        left = int(torch.randint(0, max(W - rw, 1), (1,)).item())
        mask[b, :, top:top + rh, left:left + rw] = 1.0


def sample_region_mask(img: torch.Tensor,
                       frac: float,
                       mode: str = 'rect',
                       lidar: Optional[torch.Tensor] = None,
                       area_ratio: Sequence[float] = (0.1, 0.5),
                       num_regions: Sequence[int] = (1, 3),
                       coverage_dilate: int = 31,
                       blob_grid: int = 16,
                       blob_p: float = 0.5) -> Optional[torch.Tensor]:
    """[P44-B3] (B,1,H,W) float 마스크 (1 = 그 픽셀의 img를 0으로) 를 만든다.

    학습 전용 (호출자가 self.training을 검사). 샘플은 확률 `frac`로 독립 선택
    (P42의 [발견 A]와 동일 — round(B·frac) 양자화가 BS1/BS2에서 계단함수가
    되는 문제 회피).

    mode:
      'global'   선택된 샘플 전체를 마스킹 = P42 M-1과 동일 의미 (ablation 연속성)
      'rect'     랜덤 사각형 영역들 (area_ratio 비율)
      'coverage' **같은 샘플의 lidar 유효 리턴 패턴**에서 유도한 blob에 마스킹 —
                 "img 부재 / lidar 존재" 공간 폴백을 정확히 그 위치에서 연습시킨다
                 (§7-b: 학습↔추론 커버리지 분포 정렬). lidar가 사실상 없는
                 샘플은 'rect'로 폴백한다.
    반환 None = 이 배치에서 아무 샘플도 뽑히지 않음 (호출자는 입력을 그대로 사용).
    """
    if frac <= 0:
        return None
    B, _, H, W = img.shape
    sel = torch.rand(B, device=img.device) < frac
    if not bool(sel.any()):
        return None
    mask = img.new_zeros((B, 1, H, W))
    mode = str(mode).lower()
    if mode == 'global':
        mask[sel] = 1.0
        return mask

    cover = None
    if mode == 'coverage' and lidar is not None:
        valid = (lidar.abs().sum(1, keepdim=True) > EPS).float()      # (B,1,H,W)
        if coverage_dilate and coverage_dilate > 1:
            k = int(coverage_dilate) | 1                              # 홀수 강제
            # sparse 투영점(MUSES ~6.7%)을 blob으로 — max-pool = 형태학적 팽창
            valid = F.max_pool2d(valid, kernel_size=k, stride=1, padding=k // 2)
        gh = max(1, int(blob_grid))
        gw = max(1, int(round(blob_grid * W / max(H, 1))))
        blobs = (torch.rand((B, 1, gh, gw), device=img.device) < blob_p).float()
        blobs = F.interpolate(blobs, size=(H, W), mode='nearest')
        cover = valid * blobs

    for b in range(B):
        if not bool(sel[b]):
            continue
        if cover is not None and float(cover[b].sum()) > 0:
            mask[b] = cover[b]
        else:
            _rect_mask(mask, b, area_ratio, num_regions)              # 폴백 포함
    return mask


# ── B-2: peer mutual distillation ───────────────────────────────────────────
def mutual_kl(aux_logits: List[torch.Tensor], temperature: float = 1.0,
              pixel_weights: Optional[List[Optional[torch.Tensor]]] = None
              ) -> torch.Tensor:
    """[P44-B2] per-modal aux logit 간 대칭 KL — 전 순서쌍 (i≠j) 평균.

    teacher 없음 · **양쪽 다 stop-grad 없음**(DML, CVPR'18): gradient가 약한
    브랜치로도 들어가야 "미사용 모달"이 살아난다. 지식 증류 관행대로 T² 스케일.
    pixel_weights[i] = (B,1,h,w) {0,1} — 그 모달이 유효한 픽셀 (B-3로 img가
    지워진 영역은 0). 쌍 (i,j)의 가중은 두 마스크의 곱 (둘 다 유효한 픽셀만).
    """
    m = len(aux_logits)
    if m < 2:
        return aux_logits[0].new_zeros(())
    T = max(float(temperature), 1e-3)
    with torch.autocast(device_type=aux_logits[0].device.type, enabled=False):
        logp = [F.log_softmax(l.float() / T, dim=1) for l in aux_logits]
        prob = [lp.exp() for lp in logp]
        total = aux_logits[0].new_zeros((), dtype=torch.float32)
        n = 0
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                kl = (prob[i] * (logp[i] - logp[j])).sum(dim=1, keepdim=True)
                w = None
                if pixel_weights is not None:
                    wi, wj = pixel_weights[i], pixel_weights[j]
                    if wi is not None or wj is not None:
                        w = (wi if wi is not None else 1.0) * \
                            (wj if wj is not None else 1.0)
                if w is None:
                    total = total + kl.mean()
                else:
                    w = w.to(kl.dtype)
                    total = total + (kl * w).sum() / w.sum().clamp(min=1.0)
                n += 1
        return (T * T) * total / max(n, 1)


def relational_correspondence(feats: List[torch.Tensor], num_pairs: int = 2048,
                              mode: str = 'mse') -> torch.Tensor:
    """[P44-B2] 관계형 대응 — 모달 간 **토큰쌍 cos-sim 분포**를 맞춘다.

    feature를 서로 베끼게 하는(=CKA를 인위로 올리는) 대신, 각 모달이 "어떤
    토큰들이 서로 닮았는가"라는 **관계 구조**만 공유하게 한다 (RKD/AnySeg CMD
    계열). 모달 간 동일 인덱스쌍을 쓰므로 대응이 공간적으로 정합한다.
    ⚠️ rank/η² 목적(P41 반증)이 아니다 — 분산·rank를 직접 건드리지 않는다.
    VICReg(_vicreg_loss)의 서브샘플 관행을 따라 fp32 강제 + 토큰 서브샘플.
    """
    m = len(feats)
    if m < 2:
        return feats[0].new_zeros(())
    with torch.autocast(device_type=feats[0].device.type, enabled=False):
        z = [f.float().flatten(2).transpose(1, 2).reshape(-1, f.shape[1])
             for f in feats]                                   # m x (B*h*w, C)
        N = z[0].shape[0]
        P = max(1, min(int(num_pairs), N))
        ia = torch.randint(0, N, (P,), device=z[0].device)
        ib = torch.randint(0, N, (P,), device=z[0].device)
        sims = [F.cosine_similarity(zi[ia], zi[ib], dim=1) for zi in z]  # m x (P,)
        total = feats[0].new_zeros((), dtype=torch.float32)
        n = 0
        for i in range(m):
            for j in range(i + 1, m):
                if str(mode).lower() == 'kl':
                    pi = F.log_softmax(sims[i], dim=0)
                    pj = F.log_softmax(sims[j], dim=0)
                    total = total + 0.5 * (
                        F.kl_div(pj, pi, log_target=True, reduction='sum')
                        + F.kl_div(pi, pj, log_target=True, reduction='sum'))
                else:
                    total = total + F.mse_loss(sims[i], sims[j])
                n += 1
        return total / max(n, 1)


# ── V-1: deterministic presence / validity renormalization ──────────────────
def presence_masks(inputs: List[torch.Tensor], size: Tuple[int, int],
                   img_idx: int = -1, dilate: int = 1,
                   eps: float = EPS) -> torch.Tensor:
    """[P44-V1] 모달별 presence 마스크 (m,B,1,h,w), 1 = 그 픽셀에 데이터 존재.

    🔴 이것은 **품질 게이트가 아니다.** 반증된(3세대 no-op) 것은 "모델이 스스로
    추정한 신뢰도로 추론 시 재가중"하는 학습형 게이트이고, 여기 있는 것은
    **입력에 데이터가 있느냐 없느냐**라는 결정론적 사실이다 — 학습 파라미터 0,
    추정 0, 임계값 학습 0. (§7-b: zero-fill 입력에서도 백본이 그럴듯한 feature를
    만들어 router가 무효 데이터임을 모르는 문제의 처방.)

    규칙: img는 항상 전 픽셀 present(1) — RGB는 프레임 전체를 덮는다. 그 외
    모달은 "채널 절대합 > eps" (투영 리턴 존재). MUSES lidar/event·DELIVER는
    비-img 모달을 `/255`로만 정규화하므로 무반환 픽셀이 정확히 0으로 보존된다
    (muses.py: "empty pixels -> 0 in all channels"). thermal z-score(mean/std)
    정규화를 쓰는 데이터셋에서는 0이 보존되지 않아 전 픽셀 present로 퇴화한다
    = 보수적 폴백(재정규화 없음)이지 오작동이 아니다.
    """
    out = []
    for i, x in enumerate(inputs):
        if i == img_idx:
            out.append(x.new_ones((x.shape[0], 1, *size)))
            continue
        v = (x.abs().sum(dim=1, keepdim=True) > eps).float()
        if dilate and dilate > 0:
            k = 2 * int(dilate) + 1
            v = F.max_pool2d(v, kernel_size=k, stride=1, padding=int(dilate))
        # 토큰 해상도로 내릴 때 max — 패치 안에 리턴이 하나라도 있으면 present
        v = F.adaptive_max_pool2d(v, size)
        out.append(v)
    return torch.stack(out, dim=0)


def renormalize_over_present(w: torch.Tensor, presence: torch.Tensor,
                             eps: float = EPS) -> torch.Tensor:
    """[P44-V1] 모달 축(dim 0) softmax 가중 w를 present 모달 위에서만 재정규화.

    w: (m,B,*,h,w), presence: (m,B,1,h,w) — 클래스 축이 있으면 broadcast.
    전 모달이 absent인 픽셀(정상 데이터셋에선 img=1이라 발생 불가)은 원래 w를
    유지한다 (0/0 방지).
    """
    masked = w * presence
    s = masked.sum(dim=0, keepdim=True)
    alive = (s > eps).to(w.dtype)
    return alive * masked / s.clamp(min=eps) + (1.0 - alive) * w


# ── P45: feature-space style perturbation ───────────────────────────────────
def style_perturb(feat: torch.Tensor, prob: float, sigma: float
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """[P45-F1] 채널별 (mean, std) 를 곱셈 섭동한 AdaIN식 style 교란.

    fog = style (FIFO 2204.01587) 이라는 전제에서, **feature space에서만**
    스타일을 흔들고 예측 일관성을 요구한다. 🔴 픽셀 공간 fog/night 증강은
    physaug 공정성 라인 침범이라 금지 — 여기서 입력은 건드리지 않는다.
    B=1에서도 동작하도록 배치 통계가 아니라 자기 통계의 곱셈 섭동을 쓴다.
    반환: (섭동된 feature, applied (B,) float)
    """
    B = feat.shape[0]
    mu = feat.mean(dim=(2, 3), keepdim=True)
    sd = feat.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
    em = torch.randn_like(mu) * float(sigma)
    es = torch.randn_like(sd) * float(sigma)
    out = (feat - mu) / sd * (sd * (1.0 + es).clamp(min=0.1)) + mu * (1.0 + em)
    applied = (torch.rand(B, device=feat.device) < float(prob)).to(feat.dtype)
    out = torch.where(applied.view(-1, *([1] * (feat.dim() - 1))).bool(), out, feat)
    return out, applied
