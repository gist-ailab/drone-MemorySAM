#!/usr/bin/env python3
"""oracle_assemble + by-condition 순수 로직 스모크 (CPU, 모델·데이터셋 무관).

pytest 없이 assert + __main__ 로만 검증한다. 도구 모듈은 torch/val 을 main 안에서만
import 하므로, 여기서 순수 헬퍼만 import 해도 CUDA/sam2 없이 돈다.
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.baseline_failure import common  # noqa: E402
from tools.oracle_headroom_probe import (  # noqa: E402
    accum_case_hist, accum_oracle_hist, build_by_condition_summary,
    condition_group_keys, filter_emm_max_missing, group_miou_from_cm,
    new_group_state, oracle_assemble)

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


# ===========================================================================
# --emm_max_missing 필터 / by-condition 그룹 누적 (P55 확장)
# ===========================================================================
def _fake_case(cid, group, n_missing, present="img+depth+event+lidar"):
    return SimpleNamespace(id=cid, group=group, present_names=present,
                           n_missing=n_missing)


def test_filter_emm_max_missing():
    cases = [
        _fake_case("clean", "clean", 0),
        _fake_case("emm|present=depth+event+lidar", "emm", 1, "depth+event+lidar"),
        _fake_case("emm|present=event+lidar", "emm", 2, "event+lidar"),
        _fake_case("emm|present=lidar", "emm", 3, "lidar"),
        _fake_case("rmm0.25|present=event+lidar", "rmm", 2, "event+lidar"),
        _fake_case("null0", "null", 0, "null0"),
    ]
    kept = filter_emm_max_missing(cases, 1)
    assert [c.id for c in kept] == [
        "clean", "emm|present=depth+event+lidar", "rmm0.25|present=event+lidar",
        "null0"], f"n_missing<=1 EMM 만 남아야 함: {[c.id for c in kept]}"
    # 기본값(0)=필터 없음 → 전체 유지(순서 불변).
    assert ([c.id for c in filter_emm_max_missing(cases, 0)]
            == [c.id for c in cases]), "max_missing=0 인데 케이스가 걸러짐"


def test_group_accumulation_sums_equal_global():
    # 두 그룹(cond:fog/cond:night)에 각 2장씩 누적 → 그룹 합이 전역 직접계산과 동일해야 한다.
    n_classes = 4
    imgs = []
    for gname, seed in (("cond:fog", 0), ("cond:night", 1)):
        r = np.random.default_rng(seed)
        for _ in range(2):
            gt = r.integers(0, n_classes, size=(8, 8)).astype(np.int64)
            gt[0, 0] = IGN
            pred = r.integers(0, n_classes, size=(8, 8)).astype(np.int64)
            imgs.append((gname, pred, gt))
    states = {g: new_group_state() for g in ("cond:fog", "cond:night")}
    opred_cat, ocounts = [], None
    for gname, pred, gt in imgs:
        accum_case_hist(states[gname], "clean", pred, gt, n_classes, IGN)
        states[gname]["n_images"] += 1
        alt = (pred + 1) % n_classes
        opred, counts = oracle_assemble([pred, alt], gt, win=4, ignore=IGN)
        accum_oracle_hist(states[gname], "drop1", "win4", 2,
                          opred, counts, gt, n_classes, IGN)
        opred_cat.append(opred.ravel())
        ocounts = counts if ocounts is None else ocounts + counts
    # ① 케이스 혼동행렬: 그룹 합 = 전역 직접계산.
    gsum = (states["cond:fog"]["case_hists"]["clean"]
            + states["cond:night"]["case_hists"]["clean"])
    direct = common.confusion_matrix(
        np.concatenate([p.ravel() for _, p, _ in imgs]),
        np.concatenate([g.ravel() for _, _, g in imgs]), n_classes, IGN)
    assert np.array_equal(gsum, direct), "그룹 case 혼동행렬 합 ≠ 전역"
    # ② oracle 혼동행렬·선택 카운트: 그룹 합 = 전역.
    key = ("drop1", "win4")
    gosum = (states["cond:fog"]["oracle_hists"][key]
             + states["cond:night"]["oracle_hists"][key])
    gcsum = (states["cond:fog"]["oracle_choices"][key]
             + states["cond:night"]["oracle_choices"][key])
    odirect = common.confusion_matrix(
        np.concatenate(opred_cat),
        np.concatenate([g.ravel() for _, _, g in imgs]), n_classes, IGN)
    assert np.array_equal(gosum, odirect), "그룹 oracle 혼동행렬 합 ≠ 전역"
    assert np.array_equal(gcsum, ocounts), "그룹 선택 카운트 합 ≠ 전역"
    assert sum(s["n_images"] for s in states.values()) == len(imgs), "n_images 합 불일치"


def test_group_miou_ignores_absent_classes():
    # GT 에 클래스 0·1 만 존재(2·3 부재, 2는 예측에만 등장=FP). 그룹 규약(nan-aware)은
    # 존재 클래스만 평균 → (1.0+0.5)/2=75%. 전역 규약(부재 IoU=0 포함)은 37.5% 로 다르다.
    cm = np.zeros((4, 4), np.int64)
    cm[0, 0] = 8                       # 클래스0: IoU 1.0
    cm[1, 1] = 4                       # 클래스1: tp=4
    cm[1, 2] = 4                       # 클래스1 GT 가 클래스2로 예측(fn이자 fp)
    got = group_miou_from_cm(cm)
    assert abs(got - 75.0) < 1e-6, f"nan-aware mIoU 가 75 이 아님: {got}"
    glob = common.global_miou_from_cm(cm)[1]
    assert abs(glob - 37.5) < 1e-6, f"전역 규약 대조값이 37.5 이 아님: {glob}"


def test_condition_group_keys_deliver_paths():
    p1 = "/ds/DELIVER/img/fog/val/scene12_motionblur/000050_rgb_front.png"
    p2 = "/ds/DELIVER/img/night/val/scene3/000123_rgb_front.png"
    p3 = "/ds/DELIVER/img/sun/test/scene7_eventlowres/000001_rgb_front.png"
    assert condition_group_keys(p1) == ["cond:fog", "case:motionblur"], p1
    assert condition_group_keys(p2) == ["cond:night", "case:none"], p2
    assert condition_group_keys(p3) == ["cond:sun", "case:eventlowres"], p3


def test_build_by_condition_summary():
    # clean + emm(drop1) + null_n1 그룹 요약: per_case 키·clean Δ=0·net_headroom 규칙.
    n_classes = 4
    cases = [
        _fake_case("clean", "clean", 0, present="a+b"),
        _fake_case("emm|present=b", "emm", 1, present="b"),
        _fake_case("null0", "null", 0, present="null0"),
    ]
    cand_names = {"emm": ["a+b", "b"], "null_n1": ["a+b", "null0"]}
    rng = np.random.default_rng(3)
    st = new_group_state()
    gt = rng.integers(0, 2, size=(8, 8)).astype(np.int64)      # 클래스 0·1 만 존재
    clean = rng.integers(0, 2, size=(8, 8)).astype(np.int64)
    st["n_images"] = 5
    accum_case_hist(st, "clean", clean, gt, n_classes, IGN)
    accum_case_hist(st, "emm|present=b", (clean + 1) % 2, gt, n_classes, IGN)
    accum_case_hist(st, "null0", clean, gt, n_classes, IGN)
    for cs in cand_names:
        second = (clean + 1) % 2 if cs == "emm" else clean
        opred, counts = oracle_assemble([clean, second], gt, win=None, ignore=IGN)
        accum_oracle_hist(st, cs, "img", 2, opred, counts, gt, n_classes, IGN)
    out = build_by_condition_summary({"cond:fog": st}, cases, cand_names, ["img"])
    e = out["cond:fog"]
    assert e["n_images"] == 5
    assert set(e["per_case_mIoU"]) == {"a+b", "b", "null0"}, "per_case 키=present_names"
    assert abs(e["per_case_mIoU"]["a+b"]["mIoU"] - e["clean_mIoU"]) < 1e-9, \
        "clean 의 delta_vs_clean 는 0 이어야 함"
    d = e["candidate_sets"]["emm"]["oracles"]["img"]
    assert "net_headroom_vs_null" in d, "개수맞춤 null_n1 대비 net 이 없음"
    ref = e["candidate_sets"]["null_n1"]["oracles"]["img"]["headroom"]
    assert abs(d["net_headroom_vs_null"] - (d["headroom"] - ref)) < 1e-9, \
        "net = headroom − 개수맞춤 null headroom 규칙 위반"
    assert abs(sum(d["choice_fraction"].values()) - 1.0) < 1e-6, "선택률 합 ≠ 1"


def test_write_outputs_by_condition():
    # _write_oracle_outputs 종단: by_cond 누적 → oracle_summary.json by_condition +
    # oracle_by_condition.md 표. 모듈 글로벌(ORACLE_*)을 합성값으로 채워 부른다.
    import tools.oracle_headroom_probe as ohp
    from tools.baseline_failure import common as bf_common

    n_classes = 4
    cases = [
        _fake_case("clean", "clean", 0, present="a+b"),
        _fake_case("emm|present=b", "emm", 1, present="b"),
        _fake_case("null0", "null", 0, present="null0"),
    ]
    candsets = {"emm": [cases[0], cases[1]], "drop1": [cases[0], cases[1]],
                "null": [cases[0], cases[2]], "null_n1": [cases[0], cases[2]]}
    oracles = [("win64", 64), ("img", None)]
    ohp.ORACLE_CLEAN_HIST = np.zeros((n_classes, n_classes), np.int64)
    ohp.ORACLE_CAND_NAMES = {k: [c.present_names for c in v] for k, v in candsets.items()}
    ohp.ORACLE_HISTS = {(cs, o): np.zeros((n_classes, n_classes), np.int64)
                        for cs in candsets for o, _ in oracles}
    ohp.ORACLE_CHOICES = {(cs, o): np.zeros(len(candsets[cs]), np.int64)
                          for cs in candsets for o, _ in oracles}
    ohp.ORACLE_DISAGREE = {}

    rng = np.random.default_rng(5)
    st = new_group_state()
    for _ in range(2):
        gt = rng.integers(0, 2, size=(32, 32)).astype(np.int64)
        gt[0, 0] = IGN
        clean = gt.copy()
        clean[:, :16] = (gt[:, :16] + 1) % 2      # 좌반부 오답 → 결정적 열세
        per = {"clean": clean, "emm|present=b": gt.copy(), "null0": clean.copy()}
        for c in cases:
            pred = per[c.id]
            accum_case_hist(st, c.id, pred, gt, n_classes, IGN)
            if c.group == "clean":
                ohp.ORACLE_CLEAN_HIST += bf_common.confusion_matrix(
                    pred, gt, n_classes, IGN)
        st["n_images"] += 1
        for cs, cands in candsets.items():
            preds_list = [per[c.id] for c in cands]
            d = ohp.ORACLE_DISAGREE.setdefault(cs, [0, 0])
            for pp in preds_list[1:]:
                d[0] += int(np.count_nonzero(pp != preds_list[0]))
                d[1] += int(pp.size)
            for oname, win in oracles:
                opred, counts = oracle_assemble(preds_list, gt, win, IGN)
                ohp.ORACLE_HISTS[(cs, oname)] += bf_common.confusion_matrix(
                    opred, gt, n_classes, IGN)
                ohp.ORACLE_CHOICES[(cs, oname)] += counts
                accum_oracle_hist(st, cs, oname, len(cands), opred, counts,
                                  gt, n_classes, IGN)

    with tempfile.TemporaryDirectory() as td:
        base, _ = ohp._write_oracle_outputs(td, "val", [64],
                                            by_cond={"cond:fog": st}, cases=cases)
        js = json.loads((base / "oracle_summary.json").read_text())
        md = (base / "oracle_by_condition.md").read_text()
    e = js["by_condition"]["cond:fog"]
    assert e["n_images"] == 2
    assert set(e["per_case_mIoU"]) == {"a+b", "b", "null0"}
    assert "net_headroom_vs_null" in e["candidate_sets"]["drop1"]["oracles"]["win64"]
    assert "| group | n_images | clean mIoU | Δ−a | headroom | net_vs_null | top non-clean (frac) |" in md, md
    assert any(ln.startswith("| cond:fog |") for ln in md.splitlines()), md
    # emm(=b, 완벽)이 항상 선택돼야 한다: top non-clean = '−a (1.0)'.
    row = next(ln for ln in md.splitlines() if ln.startswith("| cond:fog |"))
    assert "−a (1.0)" in row, row
    o = e["candidate_sets"]["drop1"]["oracles"]["win64"]
    assert o["choice_fraction"]["b"] == 1.0 and o["headroom"] > 0, o


if __name__ == "__main__":
    test_win4_reconstructs_gt()
    test_win_none_single_best()
    test_ties_prefer_index0()
    test_ignore_pixels_dont_affect_choice()
    test_partial_windows_6x6()
    test_headroom_zero_over_identical_candidates()
    test_headroom_positive_over_independent_noise()
    test_filter_emm_max_missing()
    test_group_accumulation_sums_equal_global()
    test_group_miou_ignores_absent_classes()
    test_condition_group_keys_deliver_paths()
    test_build_by_condition_summary()
    test_write_outputs_by_condition()
    print("ALL PASS")
