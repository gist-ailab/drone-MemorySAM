#!/usr/bin/env python3
"""oracle_assemble 순수 로직 스모크 (CPU, 모델·데이터셋 무관).

pytest 없이 assert + __main__ 로만 검증한다. 도구 모듈은 torch/val 을 main 안에서만
import 하므로, 여기서 oracle_assemble 만 import 해도 CUDA/sam2 없이 돈다.
"""
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.oracle_headroom_probe import oracle_assemble  # noqa: E402

IGN = 255


def _base_8x8():
    """8×8 gt(값 0~3) + 좌상단 2×2 ignore. cand1=좌반부 완벽, cand2=우반부 완벽, cand0=엉망."""
    gt = np.tile(np.arange(8) % 4, (8, 1)).astype(np.int64)   # (8,8)
    gt[0:2, 0:2] = IGN
    cand1 = gt.copy()
    cand1[:, 4:] = (gt[:, 4:] + 1) % 4          # 우반부 전부 오답
    cand2 = gt.copy()
    cand2[:, :4] = (gt[:, :4] + 1) % 4          # 좌반부 전부 오답
    cand0 = (gt + 2) % 4                         # 전 픽셀 오답(엉망)
    return gt, [cand0, cand1, cand2]


def test_win4_reconstructs_gt():
    gt, preds = _base_8x8()
    oracle, counts = oracle_assemble(preds, gt, win=4, ignore=IGN)
    valid = gt != IGN
    assert np.array_equal(oracle[valid], gt[valid]), "win4 oracle 이 non-ignore 에서 gt 와 불일치"
    assert counts.sum() == 4, f"8x8/win4 윈도우 수 4 != {counts.sum()}"
    assert counts[0] == 0 and counts[1] == 2 and counts[2] == 2, f"선택 분포 이상: {counts}"


def test_win_none_single_best():
    gt, preds = _base_8x8()
    oracle, counts = oracle_assemble(preds, gt, win=None, ignore=IGN)
    assert counts.sum() == 1, "전체 이미지 = 단일 윈도우"
    # 우반부(우 32 유효) > 좌반부(좌 28 유효, ignore 4) → cand2 선택.
    assert int(np.argmax(counts)) == 2, f"단일 최선 후보가 cand2 가 아님: {counts}"
    assert np.array_equal(oracle, preds[2]), "oracle 이 선택 후보 전체와 불일치"


def test_ties_prefer_index0():
    gt = np.full((4, 4), 1, dtype=np.int64)
    gt[0, 0] = IGN
    a = gt.copy()      # 완벽
    b = gt.copy()      # 완벽 (동점)
    _, counts = oracle_assemble([a, b], gt, win=None, ignore=IGN)
    assert counts[0] == 1 and counts[1] == 0, f"동점 시 index0 우선 실패: {counts}"


def test_ignore_pixels_dont_affect_choice():
    # candA: 유효 1픽셀 오답 + ignore 위치만 '정답'. candB: 유효 전부 정답, ignore 위치 오답.
    # ignore 를 세면 동점→A 선택. 제대로 제외하면 B(유효 정답 더 많음) 선택.
    gt = np.full((4, 4), 1, dtype=np.int64)
    gt[0, 0] = IGN
    candA = np.full((4, 4), 1, dtype=np.int64)
    candA[0, 0] = IGN                 # ignore 위치 '일치'(무의미해야 함)
    candA[1, 1] = 2                   # 유효 1픽셀 오답 → 유효 정답 14
    candB = np.full((4, 4), 1, dtype=np.int64)
    candB[0, 0] = 7                   # ignore 위치 불일치(무의미해야 함) → 유효 정답 15
    _, counts = oracle_assemble([candA, candB], gt, win=None, ignore=IGN)
    assert int(np.argmax(counts)) == 1, f"ignore 픽셀이 선택에 영향을 줌: {counts}"


def test_partial_windows_6x6():
    gt = np.tile(np.arange(6) % 3, (6, 1)).astype(np.int64)
    preds = [gt.copy(), (gt + 1) % 3]
    _, counts = oracle_assemble(preds, gt, win=4, ignore=IGN)
    # ys=[(0,4),(4,6)], xs=[(0,4),(4,6)] → 부분 윈도우 포함 4개.
    assert counts.sum() == 4, f"6x6/win4 윈도우 수 4 != {counts.sum()}"


def _miou(pred, gt, n_classes, ignore):
    """순수 numpy mIoU(존재하지 않는 클래스는 union=0 이라 건너뛴다). headroom 계산용."""
    valid = gt != ignore
    ious = []
    for c in range(n_classes):
        p = (pred == c) & valid
        g = (gt == c) & valid
        union = int(np.count_nonzero(p | g))
        if union == 0:
            continue
        ious.append(int(np.count_nonzero(p & g)) / union)
    return float(np.mean(ious)) if ious else 0.0


def test_headroom_zero_over_identical_candidates():
    # 동일 후보 N개(=모달 정보 무의미)에서는 oracle 이 clean(index0)만 고르므로 headroom=0.
    rng = np.random.default_rng(0)
    n_classes = 4
    gt = rng.integers(0, n_classes, size=(32, 32)).astype(np.int64)
    clean = rng.integers(0, n_classes, size=(32, 32)).astype(np.int64)  # 불완전 pred
    preds = [clean.copy() for _ in range(6)]                            # N개 동일
    oracle, _ = oracle_assemble(preds, gt, win=4, ignore=IGN)
    headroom = _miou(oracle, gt, n_classes, IGN) - _miou(preds[0], gt, n_classes, IGN)
    assert abs(headroom) < 1e-9, f"동일 후보인데 headroom≠0: {headroom}"


def test_headroom_positive_over_independent_noise():
    # 독립 난수 후보 N개는 GT 정보가 전혀 없는데도, 윈도우별 최댓값 선택만으로 headroom>0 이
    # 나온다 = 선택 편향(selection bias)의 직접 시연.
    rng = np.random.default_rng(1)
    n_classes = 4
    gt = rng.integers(0, n_classes, size=(32, 32)).astype(np.int64)
    preds = [rng.integers(0, n_classes, size=(32, 32)).astype(np.int64) for _ in range(6)]
    oracle, _ = oracle_assemble(preds, gt, win=4, ignore=IGN)
    headroom = _miou(oracle, gt, n_classes, IGN) - _miou(preds[0], gt, n_classes, IGN)
    assert headroom > 0, f"독립 난수 후보인데 선택 편향 headroom>0 이 아님: {headroom}"


if __name__ == "__main__":
    test_win4_reconstructs_gt()
    test_win_none_single_best()
    test_ties_prefer_index0()
    test_ignore_pixels_dont_affect_choice()
    test_partial_windows_6x6()
    test_headroom_zero_over_identical_candidates()
    test_headroom_positive_over_independent_noise()
    print("ALL PASS")
