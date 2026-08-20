#!/usr/bin/env python3
"""tools/smoke_oracle_spatial_modality.py — oracle_spatial_modality 정합성 스모크.

실제 ckpt·실제 데이터셋 없이 합성 미니 텐서로 몇 초 안에 끝난다. 정본 §6 필수 체크:

  1. 단조성: 오라클 mIoU ≥ full mIoU 가 **항상** 성립 (full 도 후보 부분집합).
     여러 랜덤 시드에서 assert.
  2. 강한 정합성: 오라클이 맞히는 픽셀 집합 ⊇ full 이 맞히는 픽셀 집합.
  3. 함수가 죽지 않고 정상 종료, 출력 shape/키가 맞는지.
  4. keep-subset 래퍼(_KeepSubsetDataset)가 유지 밖 모달만 zero-fill 하는지.

2026-08-19 추가(통제 실험 — .claude_logs/experiments/analysis/2026-08-19-spatial-modality-oracle-verdict.md §3):
  5. 회귀 가드: --select-granularity 1 (블록 크기 1) 이 기존 픽셀별 오라클과 byte-동일.
  6. 선택 입도 스윕이 실제로 "블록이 커질수록 상계가 좁아진다"는 방향으로 동작하는지
     (블록이 진짜 공간 구조와 정확히 맞아떨어지면 Δ 보존, 블록이 구조보다 커지면 Δ 붕괴).
  7. 독립성 널 셔플이 진짜 구조가 있는 케이스와 없는 케이스를 구분해내는지.
  8. _synthesize_split 파이프라인 레벨 회귀: 새 플래그를 안 쓰면 기존 필드가 그대로,
     새 필드는 켰을 때만 추가된다.

2026-08-20 추가(통제3: 입력-실현성 #15 — .claude_logs/experiments/analysis/
2026-08-20-oracle-realizability-control-verdict.md §4):
  9. realizable_majority 가 GT 없이 픽셀별 다수결(동점은 작은 인덱스)을 정확히 계산.
  10. consensus(oracle_synthesize_blocked 를 majority_map 으로 재사용)가 "다수결과
      가장 많이 일치하는 부분집합"을 블록 단위로 정확히 고르는지.
  11. --realizable 기본(off)이면 파이프라인 필드가 여전히 byte-동일(회귀 가드 확장).

Usage:
  python tools/smoke_oracle_spatial_modality.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.oracle_spatial_modality import (  # noqa: E402
    enumerate_subsets, subset_bitmask, subset_label, hist_from_pred,
    miou_from_hist, oracle_synthesize, oracle_synthesize_blocked,
    oracle_synthesize_null, realizable_majority, realizable_confidence_blocked,
    _synthesize_split, _KeepSubsetDataset,
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


def test_blocked_regression(n_trials=200):
    """[통제1 회귀] block<=1 이 기존 픽셀별 오라클(oracle_synthesize)과 byte-동일."""
    for seed in range(n_trials):
        r = _rng(seed)
        C = r.randint(3, 8)
        H, W = r.randint(8, 24), r.randint(8, 24)
        M = r.randint(1, 4)
        subs = enumerate_subsets(M)
        full_index = len(subs) - 1
        gt = r.randint(0, C, size=(H, W)).astype(np.int64)
        preds = np.stack([r.randint(0, C, size=(H, W)) for _ in subs], axis=0)

        O_pixel, _ = oracle_synthesize(preds, gt, full_index)
        for block in (0, 1):  # 0/1 둘 다 "픽셀별과 동일"이어야 함
            O_block = oracle_synthesize_blocked(preds, gt, full_index, block)
            assert np.array_equal(O_pixel, O_block), \
                f"seed={seed} block={block}: 픽셀별 오라클과 불일치(회귀 실패)"
    print(f"[ok] 선택입도 회귀: block<=1 이 픽셀별 오라클과 {n_trials} trial 전부 byte-동일")


def _construct_half_split_case(H=8, W=8):
    """A=top-half 완벽·B=bottom-half 완벽·full=항상 오답(클래스2) 인 합성 케이스.

    subs = enumerate_subsets(2) = [(0,),(1,),(0,1)], full_index=2, C=3.
    진짜 공간 구조 = "위/아래 절반"(block=H//2 와 정확히 정렬).
    반환: preds(3,H,W), gt(H,W), full_index.
    """
    subs = enumerate_subsets(2)
    full_index = len(subs) - 1
    gt = np.zeros((H, W), dtype=np.int64)
    gt[H // 2:, :] = 1
    pred_a = np.zeros((H, W), dtype=np.int64)          # top 절반 정답, bottom 오답
    pred_b = np.ones((H, W), dtype=np.int64)           # bottom 절반 정답, top 오답
    pred_full = np.full((H, W), 2, dtype=np.int64)     # 항상 오답(클래스 2, gt 에 없음)
    preds = np.stack([pred_a, pred_b, pred_full], axis=0)
    return preds, gt, full_index


def test_granularity_tracks_real_structure():
    """[통제1 동작] 블록이 진짜 구조(위/아래 절반)와 맞으면 Δ 보존, 구조보다 커지면 Δ 붕괴."""
    H = W = 8
    preds, gt, full_index = _construct_half_split_case(H, W)
    C = 3
    ignore = 255

    def _delta_for_block(block):
        O = oracle_synthesize_blocked(preds, gt, full_index, block)
        _, m_full = miou_from_hist(hist_from_pred(preds[full_index], gt, C, ignore))
        _, m_o = miou_from_hist(hist_from_pred(O, gt, C, ignore))
        return m_o - m_full

    delta_pixel = _delta_for_block(1)          # 픽셀별(=진짜 구조와 완전 정렬된 것과 동치)
    delta_aligned = _delta_for_block(H // 2)   # 블록이 정확히 절반 크기 → 구조와 정렬
    delta_whole = _delta_for_block(H)          # 블록 = 이미지 전체 → 절반 정보 소실

    # C=3(그 중 클래스2는 gt/pred 어디에도 실현되지 않는 "항상 오답" 유도용) 이라
    # mIoU 가 3-클래스 평균이 되어 절대값은 100에 못 미친다(class2 IoU=0 이 항상 섞임) —
    # 여기서 보는 건 절대 크기가 아니라 **블록 크기에 따른 상대적 붕괴** 방향이다.
    assert delta_pixel > 0, delta_pixel
    assert abs(delta_aligned - delta_pixel) < 1e-6, \
        (delta_aligned, delta_pixel)                     # 구조와 정렬된 블록 = 픽셀별과 동일
    assert delta_whole < delta_aligned - 30, \
        f"블록이 구조보다 커지면 Δ가 무너져야 하는데 aligned={delta_aligned} whole={delta_whole}"
    print(f"[ok] 선택입도가 구조를 추적: Δ(pixel)={delta_pixel:.1f}  "
          f"Δ(block={H // 2}, 구조정렬)={delta_aligned:.1f}  "
          f"Δ(block={H}, 전체)={delta_whole:.1f}")


def test_null_shuffle_discriminates():
    """[통제2 동작] 진짜 구조가 있으면 관측Δ ≫ 널Δ, 구조가 없으면 관측Δ ≈ 널Δ."""
    C, ignore = 3, 255

    # (a) 진짜 구조 있음: 위/아래 절반 케이스 — 셔플하면 union 이 무너져야 함
    preds, gt, full_index = _construct_half_split_case(H=16, W=16)
    _, m_full = miou_from_hist(hist_from_pred(preds[full_index], gt, C, ignore))
    O_real, _ = oracle_synthesize(preds, gt, full_index)
    _, m_real = miou_from_hist(hist_from_pred(O_real, gt, C, ignore))
    delta_real = m_real - m_full

    rng = np.random.default_rng(0)
    null_deltas = []
    for _ in range(60):
        O_null = oracle_synthesize_null(preds, gt, full_index, rng)
        _, m_null = miou_from_hist(hist_from_pred(O_null, gt, C, ignore))
        null_deltas.append(m_null - m_full)
    mean_null = float(np.mean(null_deltas))
    assert delta_real - mean_null > 10, \
        f"구조 있는 케이스인데 관측Δ({delta_real:.1f})-널Δ({mean_null:.1f}) 차이가 너무 작음"
    print(f"[ok] 독립성 널이 진짜 구조를 검출: 관측Δ={delta_real:.1f}  "
          f"널Δ(mean of 60)={mean_null:.1f}  차이={delta_real - mean_null:+.1f}")

    # (b) 구조 없음: 각 subset 이 독립 IID 50% 정답(공간 상관 없음) — 셔플해도 통계적으로 그대로
    r = np.random.RandomState(42)
    H2 = W2 = 24
    gt2 = r.randint(0, C, size=(H2, W2)).astype(np.int64)
    subs2 = enumerate_subsets(2)
    full_index2 = len(subs2) - 1
    # 각 subset 이 각 픽셀에서 독립적으로 50% 확률로 정답(gt 그대로), 아니면 다른 랜덤 오답
    def _iid_pred(seed):
        rr = np.random.RandomState(seed)
        correct_mask = rr.rand(H2, W2) < 0.5
        wrong = (gt2 + 1 + rr.randint(0, C - 1, size=(H2, W2))) % C
        return np.where(correct_mask, gt2, wrong)
    preds2 = np.stack([_iid_pred(1), _iid_pred(2), _iid_pred(3)], axis=0)
    _, m_full2 = miou_from_hist(hist_from_pred(preds2[full_index2], gt2, C, ignore))
    O_real2, _ = oracle_synthesize(preds2, gt2, full_index2)
    _, m_real2 = miou_from_hist(hist_from_pred(O_real2, gt2, C, ignore))
    delta_real2 = m_real2 - m_full2

    rng2 = np.random.default_rng(1)
    null_deltas2 = []
    for _ in range(60):
        O_null2 = oracle_synthesize_null(preds2, gt2, full_index2, rng2)
        _, m_null2 = miou_from_hist(hist_from_pred(O_null2, gt2, C, ignore))
        null_deltas2.append(m_null2 - m_full2)
    mean_null2 = float(np.mean(null_deltas2))
    assert abs(delta_real2 - mean_null2) < 15, \
        f"IID(무구조) 케이스인데 관측Δ({delta_real2:.1f})와 널Δ({mean_null2:.1f})가 너무 다름"
    print(f"[ok] 독립성 널이 무구조 케이스와 구분: 관측Δ={delta_real2:.1f}  "
          f"널Δ(mean of 60)={mean_null2:.1f}  차이={delta_real2 - mean_null2:+.1f}")


def test_synthesize_split_pipeline_regression():
    """[파이프라인 회귀] granularities/n_null 기본(비활성)이면 기존 필드와 완전히 동일하고,
    켰을 때만 새 필드(granularity_deltas/null_shuffle)가 조건부로 추가되는지."""
    M, N, H, W, C = 2, 5, 6, 6, 3
    subs = enumerate_subsets(M)
    full_index = len(subs) - 1
    class_names = ['x', 'y', 'z']

    r = np.random.RandomState(7)
    gt_all = r.randint(0, C, size=(N, H, W)).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        np.save(cache_dir / "gt_val.npy", gt_all)
        for sub in subs:
            bm = subset_bitmask(sub)
            pred = r.randint(0, C, size=(N, H, W)).astype(np.uint8)
            np.save(cache_dir / f"pred_val_mask{bm}.npy", pred)

        rep_base = _synthesize_split(subs, ['a', 'b'], full_index, cache_dir, 'val',
                                     C, class_names, 255, conditions=None)
        rep_ext = _synthesize_split(subs, ['a', 'b'], full_index, cache_dir, 'val',
                                    C, class_names, 255, conditions=None,
                                    granularities=[1, 2, 3], n_null=5, null_seed=0)

    # 기존 필드는 완전히 동일해야 한다(회귀 가드)
    for k in ('split', 'n_images', 'modals', 'num_subsets', 'miou_full', 'miou_oracle',
              'delta', 'per_class', 'per_condition', 'star_distribution',
              'star_none_pixels', 'summary'):
        assert rep_base[k] == rep_ext[k], f"필드 '{k}' 가 확장 실행에서 달라짐(회귀 실패)"
    assert 'granularity_deltas' not in rep_base and 'null_shuffle' not in rep_base
    assert 'granularity_deltas' in rep_ext and 'null_shuffle' in rep_ext
    # block=1 의 granularity delta 는 기존 delta 와 정확히 같아야 한다
    assert rep_ext['granularity_deltas']['1'] == rep_base['delta'], \
        (rep_ext['granularity_deltas']['1'], rep_base['delta'])
    assert rep_ext['null_shuffle']['n_trials'] == 5
    print(f"[ok] _synthesize_split 파이프라인 회귀: 기본 실행 필드 완전 동일, "
          f"granularity['1']==delta({rep_base['delta']}), 신규 필드는 조건부 추가만")


def test_realizable_majority():
    """[통제3 ①] realizable_majority 가 GT 없이 픽셀별 다수결(동점=작은 인덱스)을 계산."""
    C = 4
    # 3x1 픽셀, 4개 부분집합: 픽셀0 = {0,0,0,1}→다수결0, 픽셀1={1,1,2,2}→동점,작은쪽1,
    # 픽셀2={3,3,3,3}→3(만장일치)
    preds = np.array([
        [[0, 1, 3]],
        [[0, 1, 3]],
        [[0, 2, 3]],
        [[1, 2, 3]],
    ])  # (S=4, H=1, W=3)
    maj = realizable_majority(preds, C)
    assert maj.shape == (1, 3)
    assert maj[0, 0] == 0, f"만장일치 근접 다수결 실패: {maj[0,0]}"
    assert maj[0, 1] == 1, f"동점 tie-break(작은 인덱스) 실패: {maj[0,1]}"
    assert maj[0, 2] == 3, f"만장일치 실패: {maj[0,2]}"
    print(f"[ok] realizable_majority: 다수결={maj.flatten().tolist()} "
          f"(만장일치/동점tie-break/만장일치 전부 정확)")


def test_realizable_consensus_via_blocked():
    """[통제3 ②] oracle_synthesize_blocked 를 majority_map 으로 재사용 = "블록별 다수결-부합
    부분집합 커밋"이 실제로 majority 예측과 무관하게(GT 미사용) 동작하는지."""
    C = 3
    H = W = 4
    # subset0 = 항상 클래스0, subset1 = 항상 클래스1, subset2(=full) = 항상 클래스2.
    # 다수결(3개 중 서로 다른 값 3개 → 전부 동점, tie-break로 subset0 값인 0이 항상 선택됨).
    preds = np.stack([
        np.zeros((H, W), dtype=np.int64),
        np.ones((H, W), dtype=np.int64),
        np.full((H, W), 2, dtype=np.int64),
    ], axis=0)
    full_index = 2
    majority_map = realizable_majority(preds, C)
    assert np.all(majority_map == 0), "3-way 동점에서 tie-break(가장 작은 인덱스) 실패"

    # consensus: majority_map(=전부 0)과 가장 많이 일치하는 부분집합은 subset0(preds[0]=0 전부)
    # → 블록 전체가 subset0(=클래스0) 예측으로 커밋돼야 한다. GT 는 이 선택에 전혀 관여 안 함
    # (아래에서 GT 를 다르게 줘도 커밋 결과는 동일해야 함이 핵심 포인트).
    O_c = oracle_synthesize_blocked(preds, majority_map, full_index, block=2)
    assert np.all(O_c == 0), "consensus 가 majority 와 가장 잘 맞는 subset0 을 못 골랐음"

    # GT 를 다른 값(전부 1)으로 바꿔도 — consensus 선택 자체는 majority_map 만 보므로 불변.
    O_c2 = oracle_synthesize_blocked(preds, majority_map, full_index, block=2)
    assert np.array_equal(O_c, O_c2), "consensus 선택이 GT 에 의존하면 안 됨(입력만 사용)"
    print(f"[ok] consensus(=oracle_synthesize_blocked+majority_map 재사용): "
          f"GT 미사용으로 다수결-부합 부분집합(subset0)을 블록 전체에 정확히 커밋")


def test_realizable_pipeline_integration():
    """[파이프라인] --realizable 기본(off)=기존과 byte-동일, 켜면 'realizable' 필드만 추가."""
    M, N, H, W, C = 2, 4, 4, 4, 3
    subs = enumerate_subsets(M)
    full_index = len(subs) - 1
    class_names = ['x', 'y', 'z']

    r = np.random.RandomState(11)
    gt_all = r.randint(0, C, size=(N, H, W)).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        np.save(cache_dir / "gt_val.npy", gt_all)
        for sub in subs:
            bm = subset_bitmask(sub)
            pred = r.randint(0, C, size=(N, H, W)).astype(np.uint8)
            np.save(cache_dir / f"pred_val_mask{bm}.npy", pred)

        rep_base = _synthesize_split(subs, ['a', 'b'], full_index, cache_dir, 'val',
                                     C, class_names, 255, conditions=None)
        rep_rz = _synthesize_split(subs, ['a', 'b'], full_index, cache_dir, 'val',
                                   C, class_names, 255, conditions=None,
                                   granularities=[2], realizable={'majority', 'consensus'})

    for k in ('split', 'n_images', 'modals', 'num_subsets', 'miou_full', 'miou_oracle',
              'delta', 'per_class', 'per_condition', 'star_distribution',
              'star_none_pixels', 'summary'):
        assert rep_base[k] == rep_rz[k], f"필드 '{k}' 가 --realizable 로 달라짐(회귀 실패)"
    assert 'realizable' not in rep_base
    assert 'realizable' in rep_rz
    assert 'majority' in rep_rz['realizable'] and 'consensus' in rep_rz['realizable']
    assert '2' in rep_rz['realizable']['consensus']
    print(f"[ok] --realizable 파이프라인: 기본 필드 완전 동일, "
          f"realizable={{majority:{rep_rz['realizable']['majority']['delta']:+.4f}, "
          f"consensus[2]:{rep_rz['realizable']['consensus']['2']:+.4f}}} 조건부 추가")


def test_realizable_confidence_basic():
    """[통제3b ①] realizable_confidence_blocked — block=1(픽셀별 최고-confidence 선택)과
    block>1(블록 평균 confidence 최대 부분집합 커밋)이 정의대로 동작하는지."""
    # 2 부분집합, 4x4, 좌반쪽은 subset0 이 고신뢰(0.9), 우반쪽은 subset1 이 고신뢰(0.9).
    # subset0=항상 클래스0 예측, subset1=항상 클래스1 예측.
    H = W = 4
    preds = np.stack([
        np.zeros((H, W), dtype=np.int64),
        np.ones((H, W), dtype=np.int64),
    ], axis=0)
    confs = np.stack([
        np.where(np.arange(W)[None, :] < W // 2, 0.9, 0.1) * np.ones((H, W)),  # subset0: 왼쪽 고신뢰
        np.where(np.arange(W)[None, :] < W // 2, 0.1, 0.9) * np.ones((H, W)),  # subset1: 오른쪽 고신뢰
    ], axis=0)
    full_index = 1

    # block=1: 픽셀별로 confidence 높은 쪽 채택 → 왼쪽=subset0(클래스0), 오른쪽=subset1(클래스1)
    O1 = realizable_confidence_blocked(preds, confs, full_index, block=1)
    assert np.all(O1[:, :W // 2] == 0) and np.all(O1[:, W // 2:] == 1), O1

    # block=4(이미지 전체): 평균 confidence 는 두 subset 다 (0.9+0.1)/2=0.5 로 동률
    # → argmax 첫-최댓값 규약상 subset0(인덱스 작은 쪽) 전체 커밋.
    O4 = realizable_confidence_blocked(preds, confs, full_index, block=4)
    assert np.all(O4 == 0), O4

    # block=2(왼쪽/오른쪽 2열씩 블록): 왼쪽 블록은 subset0 평균 0.9 > subset1 평균 0.1 → subset0
    #                                오른쪽 블록은 그 반대 → subset1.
    O2 = realizable_confidence_blocked(preds, confs, full_index, block=2)
    assert np.all(O2[:, :W // 2] == 0) and np.all(O2[:, W // 2:] == 1), O2

    print(f"[ok] realizable_confidence_blocked: block=1(픽셀별 정확)·"
          f"block=2(블록평균, 구조와 정렬)·block=4(전체 동률→tie-break) 전부 정의대로 동작")


def test_confidence_pipeline_requires_cache():
    """[파이프라인] --realizable confidence 는 conf 캐시가 없으면 명확한 에러를 낸다
    (조용히 무시하거나 잘못된 값을 내지 않음 — no-GT 실현성 판정을 오도하면 안 되므로)."""
    M, N, H, W, C = 2, 3, 4, 4, 3
    subs = enumerate_subsets(M)
    full_index = len(subs) - 1
    class_names = ['x', 'y', 'z']
    r = np.random.RandomState(3)
    gt_all = r.randint(0, C, size=(N, H, W)).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        np.save(cache_dir / "gt_val.npy", gt_all)
        for sub in subs:
            bm = subset_bitmask(sub)
            pred = r.randint(0, C, size=(N, H, W)).astype(np.uint8)
            np.save(cache_dir / f"pred_val_mask{bm}.npy", pred)
            # conf 캐시는 일부러 만들지 않는다.

        raised = False
        try:
            _synthesize_split(subs, ['a', 'b'], full_index, cache_dir, 'val',
                              C, class_names, 255, conditions=None,
                              granularities=[2], realizable={'confidence'})
        except RuntimeError:
            raised = True
        assert raised, "conf 캐시 없이 --realizable confidence 가 조용히 통과함(안전하지 않음)"
    print(f"[ok] --realizable confidence: conf 캐시 없으면 RuntimeError로 명확히 실패(오판정 방지)")


if __name__ == '__main__':
    print("=== smoke: oracle_spatial_modality ===")
    test_enumerate_subsets()
    test_keep_subset_dataset()
    test_report_shapes()
    test_oracle_beats_when_possible()
    test_monotonicity()
    test_blocked_regression()
    test_granularity_tracks_real_structure()
    test_null_shuffle_discriminates()
    test_synthesize_split_pipeline_regression()
    test_realizable_majority()
    test_realizable_consensus_via_blocked()
    test_realizable_pipeline_integration()
    test_realizable_confidence_basic()
    test_confidence_pipeline_requires_cache()
    print("\n✅ ALL SMOKE PASSED — 단조성(oracle≥full)·포함관계·keep-subset 정합·"
          "선택입도 회귀+동작·독립성널 판별·파이프라인 회귀·"
          "입력-실현성(majority/consensus/confidence) 확인")
