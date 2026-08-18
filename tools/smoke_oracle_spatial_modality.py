#!/usr/bin/env python3
"""tools/smoke_oracle_spatial_modality.py — oracle_spatial_modality 정합성 스모크.

실제 ckpt·실제 데이터셋 없이 합성 미니 텐서로 몇 초 안에 끝난다. 정본 §6 필수 체크:

  1. 단조성: 오라클 mIoU ≥ full mIoU 가 **항상** 성립 (full 도 후보 부분집합).
     여러 랜덤 시드에서 assert.
  2. 강한 정합성: 오라클이 맞히는 픽셀 집합 ⊇ full 이 맞히는 픽셀 집합.
  3. 함수가 죽지 않고 정상 종료, 출력 shape/키가 맞는지.
  4. keep-subset 래퍼(_KeepSubsetDataset)가 유지 밖 모달만 zero-fill 하는지.

Usage:
  python tools/smoke_oracle_spatial_modality.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.oracle_spatial_modality import (  # noqa: E402
    enumerate_subsets, subset_bitmask, subset_label, hist_from_pred,
    miou_from_hist, oracle_synthesize, _KeepSubsetDataset,
)


def _rng(seed):
    return np.random.RandomState(seed)


def test_enumerate_subsets():
    subs = enumerate_subsets(4)
    assert len(subs) == 2 ** 4 - 1 == 15, subs
    # tie-break: 크기 오름차순 → 마지막이 full
    assert subs[-1] == (0, 1, 2, 3)
    sizes = [len(s) for s in subs]
    assert sizes == sorted(sizes), "부분집합이 크기 오름차순이 아님"
    # 비트마스크 유일성
    bms = [subset_bitmask(s) for s in subs]
    assert len(set(bms)) == len(bms)
    # 2모달 케이스: {a},{b},{a,b}
    subs2 = enumerate_subsets(2)
    assert subs2 == [(0,), (1,), (0, 1)], subs2
    print(f"[ok] enumerate_subsets: M=4 → {len(subs)} subsets, full 마지막; "
          f"M=2 → {[subset_label(s, ['a', 'b']) for s in subs2]}")


def test_monotonicity(n_trials=200):
    """랜덤 예측맵에서 오라클 mIoU ≥ full mIoU + 픽셀 정답 포함관계."""
    worst_delta = 1e9
    for seed in range(n_trials):
        r = _rng(seed)
        C = r.randint(3, 8)
        H, W = r.randint(8, 24), r.randint(8, 24)
        M = r.randint(1, 4)  # 1~3 modals
        subs = enumerate_subsets(M)
        full_index = len(subs) - 1
        ignore = 255

        gt = r.randint(0, C, size=(H, W)).astype(np.int64)
        # 일부 픽셀 ignore 로
        gt[r.rand(H, W) < 0.1] = ignore
        # 부분집합별 예측: 일부러 서로 다른 정답률
        preds = np.stack([r.randint(0, C, size=(H, W)) for _ in subs], axis=0)

        O, star = oracle_synthesize(preds, gt, full_index)
        assert O.shape == (H, W)
        assert star.shape == (H, W)

        P_full = preds[full_index]
        h_full = hist_from_pred(P_full, gt, C, ignore)
        h_orac = hist_from_pred(O, gt, C, ignore)
        _, miou_full = miou_from_hist(h_full)
        _, miou_orac = miou_from_hist(h_orac)

        # (1) 단조성
        assert miou_orac >= miou_full - 1e-9, \
            f"seed={seed}: oracle {miou_orac} < full {miou_full}"
        worst_delta = min(worst_delta, miou_orac - miou_full)

        # (2) 픽셀 정답 포함관계 (valid 픽셀에서)
        valid = (gt != ignore) & (gt < C)
        full_correct = (P_full == gt) & valid
        orac_correct = (O == gt) & valid
        assert np.all(orac_correct[full_correct]), \
            f"seed={seed}: full-correct 픽셀이 oracle 에서 틀림"

        # (3) star 정합: star>=0 인 픽셀은 그 부분집합이 실제로 정답
        for s in range(len(subs)):
            m = star == s
            assert np.all(preds[s][m] == gt[m]), f"seed={seed}: star={s} 오채택"
        # star==-1 픽셀은 어떤 부분집합도 정답 아님
        none_m = star == -1
        assert not (preds == gt[None])[:, none_m].any(), \
            f"seed={seed}: none 인데 정답 부분집합 존재"

    print(f"[ok] monotonicity/포함관계 {n_trials} trials 통과 "
          f"(min Δ over trials = {worst_delta:+.4f} ≥ 0)")


def test_oracle_beats_when_possible():
    """full 이 틀리고 다른 부분집합이 맞히는 픽셀이 있으면 Δ>0 이 되는지(비자명)."""
    C, H, W = 3, 4, 4
    subs = enumerate_subsets(2)  # (0,),(1,),(0,1)
    full_index = 2
    gt = np.zeros((H, W), dtype=np.int64)
    # full=예측 전부 오답(1), subset0=전부 정답(0)
    preds = np.stack([
        np.zeros((H, W), dtype=np.int64),       # (0,) 정답
        np.ones((H, W), dtype=np.int64),        # (1,) 오답
        np.ones((H, W), dtype=np.int64),        # full 오답
    ], axis=0)
    O, star = oracle_synthesize(preds, gt, full_index)
    _, miou_full = miou_from_hist(hist_from_pred(preds[full_index], gt, C, 255))
    _, miou_orac = miou_from_hist(hist_from_pred(O, gt, C, 255))
    assert miou_full == 0.0 and miou_orac > miou_full, (miou_full, miou_orac)
    # 전 픽셀 subset (0,) 이 tie-break 상 채택
    assert np.all(star == 0)
    print(f"[ok] 비자명 이득: full={miou_full:.2f} → oracle={miou_orac:.2f} "
          f"(subset 0 이 full 오답 복구)")


def test_keep_subset_dataset():
    """_KeepSubsetDataset 가 keep 밖 모달만 zero-fill 하는지."""
    import torch

    class _FakeBase:
        n_classes = 4
        CLASSES = ['a', 'b', 'c', 'd']
        ignore_label = 255

        def __len__(self):
            return 3

        def __getitem__(self, i):
            imgs = [torch.full((3, 8, 8), float(j + 1)) for j in range(4)]
            label = torch.zeros(8, 8, dtype=torch.long)
            return imgs, label

    base = _FakeBase()
    keep = [0, 2]
    ds = _KeepSubsetDataset(base, keep)
    assert len(ds) == 3
    imgs, label = ds[0]
    assert len(imgs) == 4 and label.shape == (8, 8)
    for j, t in enumerate(imgs):
        if j in keep:
            assert t.abs().sum() > 0, f"keep 모달 {j} 가 0 으로 지워짐"
        else:
            assert t.abs().sum() == 0, f"drop 모달 {j} 가 zero-fill 안 됨"
    print(f"[ok] _KeepSubsetDataset: keep={keep} → 유지 {keep}, "
          f"zero-fill {[j for j in range(4) if j not in keep]}")


def test_report_shapes():
    """합성 → miou_from_hist 출력 형태(길이 C per-class, mIoU float)."""
    C = 5
    hist = np.eye(C) * 10 + 1
    ious, miou = miou_from_hist(hist)
    assert len(ious) == C and isinstance(miou, float)
    print(f"[ok] miou_from_hist 출력 shape: per-class={len(ious)}, mIoU={miou:.2f}")


if __name__ == '__main__':
    print("=== smoke: oracle_spatial_modality ===")
    test_enumerate_subsets()
    test_keep_subset_dataset()
    test_report_shapes()
    test_oracle_beats_when_possible()
    test_monotonicity()
    print("\n✅ ALL SMOKE PASSED — 단조성(oracle≥full)·포함관계·keep-subset 정합 확인")
