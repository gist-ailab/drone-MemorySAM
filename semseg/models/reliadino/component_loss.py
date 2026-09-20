"""[COMPONENT / R2] 연결 성분 단위 손실 — 작은·원거리 성분에 가중.

진단(제안서 2026-09-20 §6 "Q3+R", 판정 대장 "원거리 손해 원인 규명"): 얇은 객체를
찾고도 윤곽을 못 그리며, 결함은 거리가 멀수록 커진다. 픽셀 평균 CE 는 큰 영역이
지배해 작은 성분의 경계 손실이 묻힌다.

이 손실은 GT 의 클래스별 **연결 성분**마다 soft-IoU 를 계산하고, 작은 성분(면적^−1/2)
과 원거리 성분(depth 구간)에 가중을 둔다. 손실 = 1 − 가중 평균 soft-IoU.

soft-IoU 는 성분의 **bbox 국소 영역**(pad 포함)에서 계산한다 — 이미지 전체로 잡으면
같은 클래스의 다른 인스턴스가 union 을 부풀려 성분별 신호가 흐려진다. bbox 국소화가
인스턴스 단위 경계 신호를 보존한다.

⚠️ depth 구간 가중(dist_weight): 이 파이프라인에서 forward/학습 루프에 들어오는 depth
는 **정규화된 HHA 텐서**(0~255 원본 아님, deliver.py:141·183). 제안서 A6 의 로그 스케일
5분위 구간 [8,47,111,255] 는 원본 0~255 기준이므로, 정규화 텐서에 그대로 적용하면 구간
경계가 어긋난다. 따라서 depth 가 주어지면 채널 평균 후 **min-max 로 0~255 로 되돌려**
근사 구간화한다(정확한 원본 depth 가 없을 때의 보수적 근사). 원본 depth 가 없거나
`dist_weight_enable=False` 면 거리 가중을 끈다(size 가중만).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy import ndimage as _ndimage
    _HAS_SCIPY = True
except Exception:                                   # pragma: no cover
    _ndimage = None
    _HAS_SCIPY = False


def _label_components(mask_np: np.ndarray) -> Tuple[np.ndarray, int]:
    """연결 성분 라벨링. scipy 가 있으면 4-이웃 label, 없으면 오류(설치 요구)."""
    if not _HAS_SCIPY:
        raise RuntimeError(
            "[COMPONENT] scipy.ndimage 가 필요하다 (연결 성분 라벨링). "
            "conda 환경 MMSS_SAM 에 scipy 설치 확인.")
    return _ndimage.label(mask_np)


class ComponentLoss(nn.Module):
    """클래스별 연결 성분 soft-IoU 손실(작은·원거리 성분 가중).

    forward(logits, target, depth=None) → (loss, n_comp)
      · logits : (B, K, h, w) 픽셀 헤드 로짓(모델 반환 = 입력 해상도)
      · target : (B, H, W) 라벨(ignore_index 포함). logits 해상도로 nearest 다운샘플.
      · depth  : (B, C, Hd, Wd) 정규화 depth 텐서 또는 None(거리 가중 off).
    """

    def __init__(self,
                 num_classes: int,
                 ignore_index: int = 255,
                 min_area: int = 1,
                 size_weight: str = 'inv_sqrt',
                 dist_weight: Sequence[float] = (1., 1., 1.5, 2., 2.),
                 depth_bins: Sequence[float] = (8., 47., 111., 255.),
                 dist_weight_enable: bool = True,
                 max_comp: int = 256,
                 bbox_pad: int = 2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.min_area = int(min_area)
        self.size_weight = str(size_weight).lower()
        if self.size_weight not in ('inv_sqrt', 'none'):
            raise ValueError(f"[COMPONENT] SIZE_WEIGHT 는 inv_sqrt|none 이어야 한다 "
                             f"(got {size_weight!r}).")
        self.dist_weight = [float(v) for v in dist_weight]
        self.depth_bins = [float(v) for v in depth_bins]
        if len(self.dist_weight) != len(self.depth_bins) + 1:
            raise ValueError(
                f"[COMPONENT] dist_weight 길이({len(self.dist_weight)})는 "
                f"depth_bins 길이+1({len(self.depth_bins)+1}) 이어야 한다 "
                f"(구간 = bins 사이 + 양끝).")
        self.dist_weight_enable = bool(dist_weight_enable)
        self.max_comp = int(max_comp)
        self.bbox_pad = int(bbox_pad)
        # 로깅 스냅샷(학습 영향 0).
        self._last_n_comp = 0

    def _dist_bin(self, med_val: float) -> int:
        """정규화 depth 채널평균의 성분 중앙값 → A6 근사 구간 인덱스."""
        # med_val 은 이미 0~255 근사 스케일(호출부에서 min-max 복원)로 넘어온다.
        return int(np.searchsorted(self.depth_bins, med_val, side='right'))

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                depth: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, int]:
        B, K, h, w = logits.shape
        probs = logits.float().softmax(dim=1)                     # (B,K,h,w) grad
        # 라벨을 logits 해상도로 nearest 다운샘플(fcr_loss 규약과 동일).
        tgt = F.interpolate(target.unsqueeze(1).float(), size=(h, w),
                            mode='nearest').squeeze(1).long()      # (B,h,w)

        use_dist = self.dist_weight_enable and depth is not None
        depth_ds = None
        if use_dist:
            d1 = depth.float().mean(dim=1, keepdim=True)           # (B,1,Hd,Wd)
            d1 = F.interpolate(d1, size=(h, w), mode='nearest')    # (B,1,h,w)
            # 정규화 HHA → 0~255 근사 복원(이미지별 min-max). 원본 depth 부재 시 근사.
            dmin = d1.amin(dim=(2, 3), keepdim=True)
            dmax = d1.amax(dim=(2, 3), keepdim=True)
            depth_ds = (255.0 * (d1 - dmin) / (dmax - dmin + 1e-6)).squeeze(1)  # (B,h,w)

        num = probs.new_zeros(())          # 가중 soft-IoU 합(grad 연결)
        den = 0.0                          # 가중 합(스칼라)
        n_comp = 0
        pad = self.bbox_pad
        for b in range(B):
            tgt_b = tgt[b]
            classes = torch.unique(tgt_b)
            for c in classes.tolist():
                if c == self.ignore_index or c < 0 or c >= self.num_classes:
                    continue
                cls_mask = (tgt_b == c)
                labeled, ncc = _label_components(
                    cls_mask.detach().cpu().numpy().astype(np.uint8))
                if ncc == 0:
                    continue
                labeled_t = torch.from_numpy(labeled).to(tgt_b.device)
                p_full = probs[b, c]                              # (h,w) grad
                for comp_id in range(1, ncc + 1):
                    if n_comp >= self.max_comp * B:
                        break
                    m = (labeled_t == comp_id)
                    area = int(m.sum().item())
                    if area < self.min_area:
                        continue
                    # bbox 국소화(+pad) — 인스턴스 단위 soft-IoU.
                    ys, xs = torch.where(m)
                    r0 = max(int(ys.min()) - pad, 0)
                    r1 = min(int(ys.max()) + pad + 1, h)
                    c0 = max(int(xs.min()) - pad, 0)
                    c1 = min(int(xs.max()) + pad + 1, w)
                    p_crop = p_full[r0:r1, c0:c1]
                    m_crop = m[r0:r1, c0:c1].float()
                    inter = (p_crop * m_crop).sum()
                    union = p_crop.sum() + m_crop.sum() - inter
                    siou = inter / (union + 1e-6)
                    # 가중: size(면적^−1/2) × dist(구간).
                    wgt = 1.0
                    if self.size_weight == 'inv_sqrt':
                        wgt *= 1.0 / (area ** 0.5)
                    if use_dist:
                        med = float(depth_ds[b][m].median().item())
                        wgt *= self.dist_weight[self._dist_bin(med)]
                    num = num + wgt * siou
                    den += wgt
                    n_comp += 1
        self._last_n_comp = n_comp
        if den <= 0.0:
            # 성분 없음(전부 ignore 등) → grad 연결된 0 반환(DDP unused-param 안전).
            return probs.sum() * 0.0, 0
        loss = 1.0 - num / den
        return loss, n_comp
