#!/usr/bin/env python3
"""
tools/baseline_failure/ 합성 스모크 (CPU, 1분 이내).

합성 GT/pred(4장, 조건·케이스가 **상대 경로 image_id** 에 들어간 중첩 이름)로 파일
3(per_image_metrics)·4(failure_mining)·5(viz_panels)를 끝까지 돌려 다음을 assert:
  (a) per-image 혼동행렬 합에서 재계산한 mIoU 가 독립 계산값과 일치
  (b) 연결성분 recall 이 독립 손계산과 일치
  (c) 상위 목록·행렬·패널 파일이 생성됨
그리고 2차 보강 항목(요구 E):
  (E1) 합성 RLE JSON(3장, 두 조건 폴더에 같은 basename) → PNG 변환 →
       장수·값 범위·중복 basename 의 상대 경로 보존을 assert
  (E2) 오프셋 +1 로 일부러 어긋난 GT 에서 check_label_convention 이 이동량 +1 을 잡는지
A2~A7 확장 항목:
  (A2) per_image_metrics --gt_protocol native·resized1024 두 경로 모두 실행 — 완전히
       같은 예측·GT 에서는 두 프로토콜 모두 이미지별 mIoU 100, 파일명이 프로토콜별로
       갈라 서로 덮어쓰지 않는다
  (A5) 합성 체크포인트 3개(항상 나쁨·항상 좋음·절반만 좋음)로 ckpt_stability 가
       always_fail/always_pass/flip 을 규칙대로 분류하는지
  (A6) depth_bin_iou — 로그 5분위 경계·구간별 픽셀수 합·완벽 예측 pacc=1·diff 정합성
  (A7) zero_modal_in_batch 두 방식 — normalized 는 모달 평균 채움((x-mean)/std → 0),
       raw 는 0 채움(옛 동작), 평균을 찾지 못하는 cfg 는 에러
  (A8) rle_json_to_png — 저장은 항상 RLE 해상도(1024²) 그대로(--gt_dir 가 있어도
       변경 없음). --keep_1024 는 1024 단언 검증(1024² 면 통과, 아니면 에러) +
       요약 pred_resolution·keep_1024 기록.
       per_image_metrics resized1024 — 1024 예측 디렉터리 exact / 1042 예측 디렉터리
       resampled 로 기록, 같은 예측·GT 로 두 경로 모두 mIoU 100, 근사 note 는
       resampled 모델이 있을 때만
파일 1·2·6·7 은 실서버 전용이라 import·인자 파싱(+id 규약 동치성)만 검사.

실행: /home/jemo/anaconda3/envs/MMSS_SAM/bin/python tools/smoke_baseline_failure.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402

SPLIT = "smoke"
H = W = 40
ROAD = common.CLASSES.index("Road")       # 6  배경
WATER = common.CLASSES.index("Water")      # 20 (큰 영역 묶음)
WALL = common.CLASSES.index("Wall")        # 10 (큰 영역 묶음)
POLE = common.CLASSES.index("Pole")        # 4  (얇은 객체 묶음)

# image_id = 데이터셋 루트 기준 상대 경로(중첩). 조건은 경로에, 센서 고장 케이스는
# scene 폴더명 접미사에 인코딩. 같은 basename(000001_rgb_front)이 여러 폴더에 반복돼도
# 상대 경로가 다르면 서로 다른 image_id 로 구분된다.
IMAGES = ["img/night/test/scene_a_lidarjitter/000001_rgb_front",
          "img/night/test/scene_a/000001_rgb_front",   # 위와 같은 basename, 다른 scene
          "img/fog/test/scene_b_eventlowres/000003_rgb_front",
          "img/sun/test/scene_c/000004_rgb_front"]


def build_gt():
    """알려진 면적의 연결성분을 심은 GT 4장."""
    gts = {}
    g1 = np.full((H, W), ROAD, np.uint8)
    g1[2:14, 2:14] = WATER   # 12x12 = 144 (bin1)
    g1[20:23, 20:23] = POLE  # 3x3 = 9 (bin0)
    gts[IMAGES[0]] = g1

    g2 = np.full((H, W), ROAD, np.uint8)
    g2[5:20, 5:20] = WALL     # 15x15 = 225 (bin1)
    gts[IMAGES[1]] = g2

    g3 = np.full((H, W), ROAD, np.uint8)
    g3[30:34, 30:34] = WATER  # 4x4 = 16 (bin0)
    gts[IMAGES[2]] = g3

    gts[IMAGES[3]] = np.full((H, W), ROAD, np.uint8)  # 성분 없음
    return gts


def build_preds(gts):
    """세 모델의 예측. detection 여부를 결정적으로 통제해 recall 을 손계산 가능하게."""
    ours = {k: v.copy() for k, v in gts.items()}          # 완벽
    dgf = {k: np.full((H, W), ROAD, np.uint8) for k in gts}  # 모든 성분 놓침
    caf = {k: v.copy() for k, v in gts.items()}
    caf[IMAGES[2]][:] = ROAD                               # image3 Water 놓침
    return {"ours": ours, "dgf": dgf, "caf": caf}


def save_dir(arrs, d):
    """중첩 image_id 를 상대 경로 그대로 풀어 저장(common.save_label_png 로 폴더 생성)."""
    for k, v in arrs.items():
        common.save_label_png(Path(d) / f"{k}.png", v)


def independent_global_miou(pred, gt):
    """common 을 쓰지 않는 독립 경로의 전역 mIoU(%)."""
    n = common.N_CLASSES
    hist = np.zeros((n, n), np.int64)
    for k in gt:
        g = gt[k].astype(np.int64).ravel()
        p = pred[k].astype(np.int64).ravel()
        valid = (g != 255) & (g >= 0) & (g < n) & (p >= 0) & (p < n)
        for gg, pp in zip(g[valid], p[valid]):
            hist[gg, pp] += 1
    tp = np.diag(hist).astype(float)
    denom = hist.sum(0) + hist.sum(1) - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(iou.mean() * 100)


def independent_cc_recall(preds, gts):
    """failure_mining 과 무관한 독립 경로의 연결성분 recall(모델→bin→recall)."""
    from scipy import ndimage
    from tools.baseline_failure.failure_mining import AREA_EDGES, AREA_LABELS, DETECT_FRAC

    def binidx(area):
        for b in range(len(AREA_LABELS)):
            if AREA_EDGES[b] <= area < AREA_EDGES[b + 1]:
                return b
        return len(AREA_LABELS) - 1

    out = {}
    for name, pred in preds.items():
        counts = np.zeros((len(AREA_LABELS), 2), np.int64)
        for k, g in gts.items():
            for c in range(common.N_CLASSES):
                m = g == c
                if not m.any():
                    continue
                lab, nlab = ndimage.label(m)
                for j in range(1, nlab + 1):
                    cmask = lab == j
                    area = int(cmask.sum())
                    hit = int((pred[k][cmask] == c).sum())
                    b = binidx(area)
                    counts[b, 0] += int(hit >= DETECT_FRAC * area)
                    counts[b, 1] += 1
        rec = {}
        for b, labl in enumerate(AREA_LABELS):
            tot = counts[b, 1]
            rec[labl] = counts[b, 0] / tot if tot else None
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# 요구 E1 — 합성 RLE JSON → PNG 변환
# ---------------------------------------------------------------------------
def _encode_rle(mask):
    """(H,W) bool → COCO RLE(dict, counts 는 JSON 용 str)."""
    from pycocotools import mask as mask_util
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle = {"size": [int(mask.shape[0]), int(mask.shape[1])],
           "counts": rle["counts"].decode("utf-8")}
    return rle


def build_rle_json(dataset_root):
    """3장 · 두 조건 폴더에 같은 basename(000050_rgb_front)이 반복되는 예측 JSON."""
    files = [
        f"{dataset_root}/img/night/test/scene_a/000050_rgb_front.png",
        f"{dataset_root}/img/fog/test/scene_b/000050_rgb_front.png",   # 같은 basename
        f"{dataset_root}/img/sun/test/scene_c/000099_rgb_front.png",
    ]
    records = []
    for fn in files:
        lab = np.full((H, W), ROAD, np.uint8)
        lab[4:16, 4:16] = WATER      # 사각형 하나
        # category 별 마스크(전 픽셀 커버 → ignore 0). ROAD·WATER 두 종.
        for cid in (ROAD, WATER):
            m = lab == cid
            records.append({"file_name": fn, "category_id": int(cid),
                            "segmentation": _encode_rle(m)})
    return files, records


def smoke_rle(tmp):
    """(E1) RLE JSON → PNG 변환 후 장수·값 범위·중복 basename 상대 경로 보존 확인."""
    from tools.baseline_failure import rle_json_to_png as r2p
    droot = str(tmp / "fake_root")
    _files, records = build_rle_json(droot)
    json_path = tmp / "sem_seg_predictions.json"
    json_path.write_text(json.dumps(records), encoding="utf-8")

    out = tmp / "rle_out"
    argv = ["rle_json_to_png.py", "--json", str(json_path),
            "--dataset_root", droot, "--split", "test",
            "--out", str(out), "--expected_count", "3"]
    old = sys.argv
    try:
        sys.argv = argv
        r2p.main()
    finally:
        sys.argv = old

    pred_dir = out / "test" / "pred"
    idx = common.index_label_pngs(pred_dir)
    assert len(idx) == 3, f"변환 장수 {len(idx)} != 3"
    # 중복 basename 이 서로 다른 상대 경로로 보존됐는지(둘 다 존재).
    assert "img/night/test/scene_a/000050_rgb_front" in idx
    assert "img/fog/test/scene_b/000050_rgb_front" in idx
    # 값 범위 0~24 (+255) 만.
    valid = set(range(common.N_CLASSES)) | {255}
    for image_id, p in idx.items():
        u = set(np.unique(common.load_label_png(p)).tolist())
        assert u <= valid, f"{image_id} 값 범위 벗어남: {sorted(u)}"
    summ = json.loads((out / "test" / "summary.json").read_text(encoding="utf-8"))
    assert summ["num_images"] == 3 and summ["overlap_pixels"] == 0, summ
    print("[smoke] (E1) RLE→PNG 변환: 장수·값범위·중복 basename 상대경로 보존 ✓")
    return pred_dir


# ---------------------------------------------------------------------------
# 요구 E2 — 오프셋 +1 어긋난 GT 에서 이동량 탐지
# ---------------------------------------------------------------------------
def smoke_label_offset(tmp):
    """(E2) GT = pred+1 로 만든 뒤 check_label_convention 이 최적 이동량 +1 을 잡는지."""
    from tools.baseline_failure import check_label_convention as clc
    pred_dir = tmp / "off_pred"
    gt_dir = tmp / "off_gt"
    ids = ["img/night/test/s1/000001_rgb_front",
           "img/fog/test/s2/000002_rgb_front"]
    for image_id in ids:
        pred = np.full((H, W), ROAD, np.uint8)          # 6
        pred[5:20, 5:20] = WALL                          # 10
        gt = pred.astype(np.int32) + 1                   # 7, 11 (pred 보다 +1)
        common.save_label_png(pred_dir / f"{image_id}.png", pred)
        common.save_label_png(gt_dir / f"{image_id}.png", gt.astype(np.uint8))
    best, mious = clc.method_i_iou_shift(str(pred_dir), str(gt_dir), gt_raw=False, limit=0)
    assert best == 1, f"이동량 오탐: best={best}, mious={mious}"
    # 정합(0 오프셋)이면 이동량 0 을 잡는지도 확인.
    best0, _ = clc.method_i_iou_shift(str(pred_dir), str(pred_dir), gt_raw=False, limit=0)
    assert best0 == 0, f"정합 케이스 이동량 오탐: {best0}"
    print("[smoke] (E2) check_label_convention 이 이동량 +1/0 을 정확히 판정 ✓")


# ---------------------------------------------------------------------------
# A2 — per_image_metrics 의 GT 프로토콜(native vs resized1024)
# ---------------------------------------------------------------------------
def smoke_gt_protocol(tmp):
    """(A2) 두 프로토콜 모두 end-to-end 실행 — 같은 예측·GT 에서 mIoU 100,
    프로토콜별 파일명 분리, 요약 JSON 프로토콜 기록을 확인한다."""
    import csv as _csv
    from tools.baseline_failure import per_image_metrics as pim

    ids = ["img/night/test/s1/000001_rgb_front", "img/fog/test/s2/000002_rgb_front"]
    gt = {k: np.full((64, 64), ROAD, np.uint8) for k in ids}
    for k in ids:
        gt[k][20:36, 20:36] = WATER                      # Road + Water 2클래스
    perfect = {k: v.copy() for k, v in gt.items()}       # 예측 == GT
    degraded = {k: np.full((64, 64), ROAD, np.uint8) for k in ids}  # Water 전부 놓침

    gt_dir = tmp / "a2_gt"
    save_dir(gt, gt_dir)
    save_dir(perfect, tmp / "a2_perfect")
    save_dir(degraded, tmp / "a2_degraded")
    out = tmp / "a2_out"
    out.mkdir(parents=True, exist_ok=True)

    for proto in ("native", "resized1024"):
        argv = ["per_image_metrics.py", "--gt", str(gt_dir),
                "--models", f"ours={tmp}/a2_perfect", f"dgf={tmp}/a2_degraded",
                "--split", SPLIT, "--out", str(out), "--gt_protocol", proto]
        old = sys.argv
        try:
            sys.argv = argv
            pim.main()
        finally:
            sys.argv = old

    # 파일명이 프로토콜별로 갈라 서로 덮어쓰지 않는다(native 는 기존 이름 그대로).
    sfx = {"native": "", "resized1024": "_resized1024"}
    for proto, s in sfx.items():
        assert (out / f"per_image_ours_{SPLIT}{s}.csv").exists(), f"{proto} per_image CSV 없음"
        assert (out / f"joined_{SPLIT}{s}.csv").exists(), f"{proto} joined CSV 없음"
        summ = json.loads((out / f"summary_per_image_{SPLIT}_{proto}.json")
                          .read_text(encoding="utf-8"))
        assert summ["gt_protocol"] == proto, summ
        if proto == "resized1024":
            assert "정수배" in summ["note"], f"한계 note 없음: {summ['note']}"
        # 완전히 같은 예측·GT — 이미지별 mIoU(도구 규약 0~1)가 두 프로토콜 모두 1.0,
        # 즉 백분율 100 이다.
        with open(out / f"per_image_ours_{SPLIT}{s}.csv", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert rows and all(float(r["mIoU_img"]) * 100 == 100.0 for r in rows), \
            f"{proto}: mIoU_img != 100(%) — {[r['mIoU_img'] for r in rows]}"
        # 전역 mIoU(부재 클래스 0 포함 25클래스 평균)도 두 프로토콜이 같다(2/25*100=8).
        assert abs(summ["models"]["ours"]["global_miou"] - 8.0) < 1e-6, summ["models"]["ours"]
        # 열화 예측은 확실히 아래로 — 두 프로토콜 모두 수치가 타당하게 갈린다.
        assert summ["models"]["dgf"]["global_miou"] < summ["models"]["ours"]["global_miou"]
    n_summ = json.loads((out / f"summary_per_image_{SPLIT}_native.json")
                        .read_text(encoding="utf-8"))
    r_summ = json.loads((out / f"summary_per_image_{SPLIT}_resized1024.json")
                        .read_text(encoding="utf-8"))
    assert abs(n_summ["models"]["ours"]["global_miou"]
               - r_summ["models"]["ours"]["global_miou"]) < 1e-6, "완벽 예측인데 프로토콜 간 격차 발생"
    print("[smoke] (A2) gt_protocol native·resized1024 모두 동작 + mIoU 100 ✓")


# ---------------------------------------------------------------------------
# A5 — ckpt_stability 분류(항상 나쁨·항상 좋음·절반만 좋음 체크포인트 3개)
# ---------------------------------------------------------------------------
CKPT_IDS = ["img/night/test/s1/000001_rgb_front",
            "img/night/test/s1_lidarjitter/000002_rgb_front",
            "img/fog/test/s2/000003_rgb_front",
            "img/fog/test/s2/000004_rgb_front",
            "img/sun/test/s3/000005_rgb_front",
            "img/rain/test/s4/000006_rgb_front"]


def _ck_quality_maps(gts):
    """품질 3단계 예측 맵 — q0(성분 전부 놓침)·q1(절반만 적중)·q2(완벽).

    GT 는 Road 배경 + 16x16 Water 블록으로 모든 이미지가 동일하므로 각 품질의
    이미지별 mIoU 도 정확히 같아진다(중앙값 비교가 결정적).
    """
    q0 = {k: np.full((H, W), ROAD, np.uint8) for k in gts}
    q1 = {k: v.copy() for k, v in gts.items()}
    q2 = {k: v.copy() for k, v in gts.items()}
    for k in gts:
        q1[k][20:36, 20:28] = ROAD                      # Water 블록 절반만 적중
    return {"q0": q0, "q1": q1, "q2": q2}


def smoke_ckpt_stability(tmp):
    """(A5) bad=[q0,q0,q0,q1,q1,q1] / good=[q1,q1,q1,q2,q2,q2] / half=[q0,q0,q1,q1,q2,q2]
    로 만든 3개 체크포인트에서 규칙대로 always_fail·flip·always_pass 가 나오는지."""
    import csv as _csv
    from tools.baseline_failure import ckpt_stability as cs

    gts = {k: np.full((H, W), ROAD, np.uint8) for k in CKPT_IDS}
    for k in CKPT_IDS:
        gts[k][20:36, 20:36] = WATER
    q = _ck_quality_maps(gts)
    plans = {"bad": ["q0", "q0", "q0", "q1", "q1", "q1"],
             "good": ["q1", "q1", "q1", "q2", "q2", "q2"],
             "half": ["q0", "q0", "q1", "q1", "q2", "q2"]}

    gt_dir = tmp / "a5_gt"
    save_dir(gts, gt_dir)
    for tag, plan in plans.items():
        save_dir({k: q[plan[i]][k] for i, k in enumerate(CKPT_IDS)},
                 tmp / f"a5_{tag}")

    out = tmp / "a5_out"
    argv = ["ckpt_stability.py", "--gt", str(gt_dir),
            "--preds", *[f"{t}={tmp}/a5_{t}" for t in plans],
            "--split", SPLIT, "--out", str(out)]
    old = sys.argv
    try:
        sys.argv = argv
        cs.main()
    finally:
        sys.argv = old

    # 기대 라벨 — 손계산: bad 중앙값=(q0+q1)/2, good 중앙값=(q1+q2)/2, half 중앙값=q1.
    # q0 은 bad·good 중앙값의 아래(둘 다 미만), q1 은 half 중앙값(q1)과 같아 미만이 아님.
    # CSV 행은 image_id 정렬순이므로 기대값도 정렬된 id 순서로 맞춘다.
    plan_by_id = {CKPT_IDS[i]: {t: plans[t][i] for t in plans} for i in range(len(CKPT_IDS))}
    with open(out / f"ckpt_stability_{SPLIT}.csv", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    got, exp, got_n = [], [], []
    for r in rows:                       # r["image_id"] 는 정렬순
        qsel = plan_by_id[r["image_id"]]
        vals = {"q0": 42.0, "q1": 70.652174, "q2": 100.0}
        med = {"bad": 56.326087, "good": 85.326087, "half": 70.652174}
        n_below = sum(vals[qsel[t]] < med[t] for t in plans)
        exp.append(cs.classify(len(plans), n_below, 9.0 / 11.0))
        got.append(r["label"])
        got_n.append(int(r["n_below_median"]))
    assert got == exp, f"라벨 불일치: {got} != {exp}"
    assert sorted(got) == ["always_fail", "always_fail", "always_pass",
                           "always_pass", "always_pass", "flip"], got
    assert got_n.count(3) == 2 and got_n.count(0) == 3 and got_n.count(2) == 1, got_n
    summ = json.loads((out / f"ckpt_stability_{SPLIT}.json").read_text(encoding="utf-8"))
    assert summ["label_counts"] == {"always_fail": 2, "always_pass": 3, "flip": 1}, summ
    night_labels = {lab for lab, n in summ["label_dist_by_condition"]["night"].items() if n}
    assert night_labels == {"always_fail"}, summ["label_dist_by_condition"]
    assert len(summ["per_class_iou_std_across_ckpts"]) == common.N_CLASSES
    print("[smoke] (A5) ckpt_stability always_fail/always_pass/flip 분류 ✓")


# ---------------------------------------------------------------------------
# A6 — depth_bin_iou(로그 5분위 경계·구간별 IoU·모델 간 diff)
# ---------------------------------------------------------------------------
def smoke_depth_bin(tmp):
    """(A6) 합성 depth 로 로그 5분위 구간 — 경계 단조성·구간 픽셀수 합(독립 손계산)·
    완벽 예측의 구간별 pacc=1·diff 열 정합성을 확인한다."""
    import csv as _csv
    from tools.baseline_failure import depth_bin_iou as dbi

    gts = {k: np.full((H, W), ROAD, np.uint8) for k in CKPT_IDS}
    for k in CKPT_IDS:
        gts[k][20:36, 20:36] = WATER
        gts[k][0:4, :] = 255                             # ignore 영역(제외 검증용)
    bad = {k: np.full((H, W), ROAD, np.uint8) for k in CKPT_IDS}

    # depth: 행 기반 계단값(100 + 50r), 상단 8행은 0(무효 depth). uint16 PNG 로 저장
    # (common.save_label_png 는 uint8 로 캐스팅하므로 여기선 PIL 로 직접 저장).
    depth_arr = (100 + 50 * np.arange(H, dtype=np.int64)[:, None]).repeat(W, axis=1)
    depth_arr[0:8, :] = 0
    depth_root = tmp / "a6_root" / "depth"
    for k in CKPT_IDS:
        rel = k[len("img/"):].replace("_rgb", "_depth")
        p = depth_root / f"{rel}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(depth_arr.astype(np.uint16)).save(p)

    gt_dir = tmp / "a6_gt"
    save_dir(gts, gt_dir)
    save_dir({k: v.copy() for k, v in gts.items()}, tmp / "a6_perfect")
    save_dir(bad, tmp / "a6_bad")

    out = tmp / "a6_out"
    argv = ["depth_bin_iou.py", "--gt", str(gt_dir),
            "--depth_root", str(tmp / "a6_root"),
            "--models", f"perfect={tmp}/a6_perfect", f"bad={tmp}/a6_bad",
            "--split", SPLIT, "--out", str(out), "--bins", "5"]
    old = sys.argv
    try:
        sys.argv = argv
        dbi.main()
    finally:
        sys.argv = old

    summ = json.loads((out / f"depth_bin_iou_{SPLIT}.json").read_text(encoding="utf-8"))
    assert len(summ["log_edges"]) == 4 and np.all(np.diff(summ["log_edges"]) > 0), summ["log_edges"]
    assert np.allclose(np.exp(summ["log_edges"]), summ["depth_edges"]), "depth_edges != exp(log_edges)"
    # 구간별 픽셀수 합 == 독립 손계산((depth>0) & (gt!=255)).
    valid = (depth_arr > 0) & (gts[CKPT_IDS[0]] != 255)
    assert sum(summ["bin_pixels"]) == int(valid.sum()) * len(CKPT_IDS), \
        f"{sum(summ['bin_pixels'])} != {int(valid.sum()) * len(CKPT_IDS)}"
    assert all(b > 0 for b in summ["bin_pixels"]), "빈 구간 있음(분포 대비 이상)"

    with open(out / f"depth_bin_iou_{SPLIT}.csv", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 5 and all(float(r["pacc_perfect"]) == 1.0 for r in rows), \
        "완벽 예측의 구간별 pixel_acc != 1"
    for r in rows:
        d_expect = float(r["miou_perfect"]) - float(r["miou_bad"])
        assert abs(float(r["diff_perfect_minus_bad"]) - d_expect) <= 2e-4, r
        assert float(r["miou_bad"]) <= float(r["miou_perfect"]), r
    print("[smoke] (A6) depth_bin_iou 로그 5분위·구간별 지표·diff 정합 ✓")


# ---------------------------------------------------------------------------
# A7 — zero_modal_in_batch 개입 방식(normalized|raw)
# ---------------------------------------------------------------------------
def smoke_zero_mode():
    """(A7) 실제 모델 없이 합성 텐서·가짜 cfg 로 두 방식을 검증 — normalized 는 채운
    값이 모달 평균이고 (x-mean)/std 를 적용하면 0, raw 는 채운 값이 0, 평균을 찾지
    못하면 에러로 멈춘다."""
    import torch
    from tools.baseline_failure import probe_dgfusion as pdg

    class _ModelCfg:                       # 가짜 cfg.MODEL — 평균·표준편차만 가진 객체
        PIXEL_MEAN = [0.485, 0.456, 0.406]
        PIXEL_STD = [0.229, 0.224, 0.225]
    class _Cfg:
        MODEL = _ModelCfg()
        LIDAR_PIXEL_MEAN = [0.5]           # 보조 모달별 통계 키(탐색 경로 확인용)
    mean = torch.tensor(_ModelCfg.PIXEL_MEAN).view(-1, 1, 1)
    std = torch.tensor(_ModelCfg.PIXEL_STD).view(-1, 1, 1)

    def _batch():
        return [{"CAMERA": torch.randn(3, 8, 10) + 10.0,   # 0 이 아닌 값들
                 "image": torch.randn(3, 8, 10) + 10.0,
                 "LIDAR": torch.randn(1, 8, 10) + 10.0}]

    # raw — 옛 동작 그대로: 지정 모달(+image)만 원본이 0, 다른 모달은 그대로.
    b = _batch()
    pdg.zero_modal_in_batch(b, "CAMERA", mode="raw", cfg=_Cfg())
    assert (b[0]["CAMERA"] == 0).all() and (b[0]["image"] == 0).all()
    assert not (b[0]["LIDAR"] == 0).any(), "raw 가 지정 밖 모달을 건드렸다"

    # normalized — 원본이 채널별 평균으로 차고, (x-mean)/std 를 적용하면 정확히 0.
    b = _batch()
    pdg.zero_modal_in_batch(b, "CAMERA", mode="normalized", cfg=_Cfg())
    for k in ("CAMERA", "image"):
        assert torch.allclose(b[0][k], mean.expand_as(b[0][k])), k
        assert torch.allclose((b[0][k] - mean) / std, torch.zeros_like(b[0][k])), k
    assert not (b[0]["LIDAR"] == 0).any(), "normalized 가 지정 밖 모달을 건드렸다"

    # 보조 모달 키 탐색(LIDAR_PIXEL_MEAN) — LIDAR 도 모달 평균으로 채워진다.
    b = _batch()
    pdg.zero_modal_in_batch(b, "LIDAR", mode="normalized", cfg=_Cfg())
    assert torch.allclose(b[0]["LIDAR"], torch.full_like(b[0]["LIDAR"], 0.5))
    assert not (b[0]["CAMERA"] == 0).all(), "LIDAR 지정이 CAMERA 를 건드렸다"

    # 기본 모드는 normalized — cfg 없이는 호출 자체가 거부된다.
    try:
        pdg.zero_modal_in_batch(_batch(), "CAMERA")
        raise AssertionError("기본 모드가 normalized 인데 cfg 없이 통과했다")
    except ValueError:
        pass

    # 평균 부재 — EVENT 통계는 가짜 cfg 에 없다 → 명확한 에러(임의값 대입 금지).
    try:
        pdg.zero_modal_in_batch(_batch(), "EVENT", mode="normalized", cfg=_Cfg())
        raise AssertionError("EVENT 평균이 없는데 에러가 나지 않았다")
    except RuntimeError:
        pass
    print("[smoke] (A7) zero-modal normalized/raw·평균 부재 에러 ✓")


# ---------------------------------------------------------------------------
# A8 — keep_1024 저장(정확 경로) · resized1024 채점의 exact/resampled 구분
# ---------------------------------------------------------------------------
# 실제 규약 그대로: 기준선 RLE = 1024²(추론 해상도), 우리 GT 덤프 = native 1042².
A8_RLE_H = A8_RLE_W = 1024
A8_GT_H = A8_GT_W = 1042
A8_IDS = ["img/night/test/s1/000010_rgb_front", "img/fog/test/s2/000011_rgb_front"]


def _a8_gt():
    """Road 배경 + Water 블록 2장(native 1042²). 완벽 예측의 전역 mIoU = 2/25*100=8."""
    gt = {k: np.full((A8_GT_H, A8_GT_W), ROAD, np.uint8) for k in A8_IDS}
    for k in A8_IDS:
        gt[k][100:500, 200:900] = WATER
    return gt


def _a8_records(droot, labels):
    """{image_id: 라벨} → category 별 RLE 레코드 목록(file_name 은 droot 절대 경로)."""
    records = []
    for k, lab in labels.items():
        for cid in (ROAD, WATER):
            records.append({"file_name": f"{droot}/{k}.png", "category_id": int(cid),
                            "segmentation": _encode_rle(lab == cid)})
    return records


def _run_main(mod, argv):
    old = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old


def smoke_keep_1024(tmp):
    """(A8-1) 저장은 항상 RLE 해상도 그대로 — GT 덤프(1042²)를 줘도 기본 모드 저장은
    1024². --keep_1024 는 크기 단언(1024² 면 통과, 아니면 에러)이지 저장을 바꾸지 않는다."""
    from tools.baseline_failure import rle_json_to_png as r2p

    gt = _a8_gt()
    pred1024 = {k: common.resize_nearest(v, 1024, 1024) for k, v in gt.items()}
    droot = str(tmp / "a8_root2")
    json_path = tmp / "a8_predictions.json"
    json_path.write_text(json.dumps(_a8_records(droot, pred1024)), encoding="utf-8")
    gt_dir = tmp / "a8_gt"
    save_dir(gt, gt_dir)

    def run(out, extra):
        _run_main(r2p, ["rle_json_to_png.py", "--json", str(json_path),
                        "--dataset_root", droot, "--split", "test",
                        "--out", str(out), "--expected_count", "2",
                        "--gt_dir", str(gt_dir)] + extra)

    out_keep, out_def = tmp / "a8_keep", tmp / "a8_default"
    # keep_1024 의 재현 검산도 기본 로직(pred→GT 최근접) — 이 블록 배치에서는
    # 1042→1024→1042 왕복이 픽셀 정합이라 mIoU = 8.0(2/25 클래스).
    run(out_keep, ["--keep_1024", "--expected_miou", "8.0"])
    run(out_def, [])

    idx_keep = common.index_label_pngs(out_keep / "test" / "pred")
    idx_def = common.index_label_pngs(out_def / "test" / "pred")
    for image_id in A8_IDS:
        k = common.load_label_png(idx_keep[image_id])
        d = common.load_label_png(idx_def[image_id])
        # 저장은 모드와 무관하게 항상 RLE 해상도(1024²) — GT(1042²)를 무시한다.
        assert k.shape == (1024, 1024), f"keep_1024 저장 해상도 {k.shape} != 1024²"
        assert d.shape == (1024, 1024), \
            f"기본 저장이 RLE 해상도가 아니다: {d.shape} != 1024² (GT {A8_GT_H}² 을 무시해야)"
        assert np.array_equal(k, pred1024[image_id]) and np.array_equal(d, k), \
            "저장 내용이 RLE 디코드 원본이 아니다"

    sk = json.loads((out_keep / "test" / "summary.json").read_text(encoding="utf-8"))
    assert sk["keep_1024"] is True and sk["pred_resolution"] == 1024, sk
    assert sk["reproduced"] is True and sk["reproduce_miou"] == 8.0, sk
    sd = json.loads((out_def / "test" / "summary.json").read_text(encoding="utf-8"))
    assert sd["keep_1024"] is False and sd["pred_resolution"] == 1024, sd

    # 비-1024 RLE → keep_1024 는 임의 보정 없이 명확한 에러로 멈춘다.
    bad = tmp / "a8_bad_size.json"
    bad.write_text(json.dumps(_a8_records(droot, {A8_IDS[0]: np.full((32, 32), ROAD, np.uint8)})),
                   encoding="utf-8")
    try:
        _run_main(r2p, ["rle_json_to_png.py", "--json", str(bad),
                        "--dataset_root", droot, "--split", "test",
                        "--out", str(tmp / "a8_bad_out"), "--expected_count", "1",
                        "--keep_1024"])
        raise AssertionError("비-1024 RLE 인데 keep_1024 가 에러 없이 통과했다")
    except ValueError:
        pass
    print("[smoke] (A8-1) 저장=항상 RLE 해상도 · keep_1024 1024 단언(통과/에러) · 요약 기록 ✓")


def smoke_exact_1024(tmp):
    """(A8-2) resized1024 채점 — 1024 예측 디렉터리는 exact, 1042 예측 디렉터리는
    resampled 로 기록. 같은 예측·GT 로는 두 경로 모두 이미지별 mIoU 100."""
    import csv as _csv
    from tools.baseline_failure import per_image_metrics as pim

    gt = _a8_gt()
    gt_dir = tmp / "a8_gt3"
    save_dir(gt, gt_dir)
    # exact 경로용 예측 = GT 를 1024 최근접으로 줄인 것(RLE 복원 덤프와 동일 구성 = 1024²).
    # resampled 경로용 예측 = GT 원 해상도 그대로(1042 → 채점 시 1024 로 줄인다).
    save_dir({k: common.resize_nearest(v, 1024, 1024) for k, v in gt.items()},
             tmp / "a8_p1024")
    save_dir({k: v.copy() for k, v in gt.items()}, tmp / "a8_p1042")

    out = tmp / "a8_pim_out"
    out.mkdir(parents=True, exist_ok=True)
    _run_main(pim, ["per_image_metrics.py", "--gt", str(gt_dir),
                    "--models", f"exact={tmp}/a8_p1024", f"resamp={tmp}/a8_p1042",
                    "--split", SPLIT, "--out", str(out),
                    "--gt_protocol", "resized1024"])
    summ = json.loads((out / f"summary_per_image_{SPLIT}_resized1024.json")
                      .read_text(encoding="utf-8"))
    m = summ["models"]
    assert m["exact"]["pred_1024_path"] == "exact", m["exact"]
    assert m["exact"]["n_pred_1024_exact"] == 2 and m["exact"]["n_pred_1024_resampled"] == 0
    assert m["resamp"]["pred_1024_path"] == "resampled", m["resamp"]
    assert m["resamp"]["n_pred_1024_resampled"] == 2, m["resamp"]
    assert "정수배" in summ["note"], f"근사 경로가 있는데 근사 note 없음: {summ['note']}"
    # 같은 예측·GT — 두 경로 모두 완벽(이미지별 mIoU 100, 전역 2/25*100=8).
    for name in ("exact", "resamp"):
        with open(out / f"per_image_{name}_{SPLIT}_resized1024.csv", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert rows and all(float(r["mIoU_img"]) * 100 == 100.0 for r in rows), \
            f"{name}: mIoU_img != 100(%) — {[r['mIoU_img'] for r in rows]}"
        assert abs(m[name]["global_miou"] - 8.0) < 1e-6, m[name]

    # exact 만 있으면 note 에 근사 문구가 없어야 한다.
    out2 = tmp / "a8_pim_out_exact"
    out2.mkdir(parents=True, exist_ok=True)
    _run_main(pim, ["per_image_metrics.py", "--gt", str(gt_dir),
                    "--models", f"exact={tmp}/a8_p1024",
                    "--split", SPLIT, "--out", str(out2),
                    "--gt_protocol", "resized1024"])
    summ2 = json.loads((out2 / f"summary_per_image_{SPLIT}_resized1024.json")
                       .read_text(encoding="utf-8"))
    assert summ2["models"]["exact"]["pred_1024_path"] == "exact", summ2
    assert "정수배" not in summ2["note"], f"근사 경로가 없는데 근사 note 있음: {summ2['note']}"
    print("[smoke] (A8-2) resized1024 exact/resampled 기록 · 두 경로 모두 mIoU 100 ✓")


def run():
    tmp = Path(tempfile.mkdtemp(prefix="bf_smoke_"))
    gts = build_gt()
    preds = build_preds(gts)

    gt_dir = tmp / "gt"
    save_dir(gts, gt_dir)
    pred_dirs = {}
    for name, arrs in preds.items():
        pd = tmp / name / "pred"
        save_dir(arrs, pd)
        pred_dirs[name] = pd

    d2_out = tmp / "d2"
    d2_out.mkdir(parents=True, exist_ok=True)

    # ---- 파일 3: per_image_metrics (핵심 함수 직접 호출 + join) ----
    from tools.baseline_failure import per_image_metrics as pim
    models_pi = {}
    for name in preds:
        pi, miou, _ok, _chk, _p1024 = pim.score_model(name, pred_dirs[name], gt_dir, SPLIT, d2_out)
        models_pi[name] = pi
        # (a) 혼동행렬 합 mIoU == 독립 계산값
        ind = independent_global_miou(preds[name], gts)
        assert abs(miou - ind) < 1e-3, f"[a] {name} mIoU {miou} != 독립 {ind}"
        assert (d2_out / f"per_image_{name}_{SPLIT}.csv").exists()
    joined = pim.write_join(models_pi, SPLIT, d2_out)
    assert joined.exists(), "joined CSV 누락"
    print("[smoke] (a) per-image 혼동행렬 합 mIoU == 독립 계산값 ✓")

    # ---- 파일 4: failure_mining (main 을 argv 로 끝까지) ----
    from tools.baseline_failure import failure_mining as fm
    mining_out = tmp / "fm"
    argv = ["failure_mining.py",
            "--joined", str(joined), "--gt", str(gt_dir),
            "--models", *[f"{n}={pred_dirs[n]}" for n in preds],
            "--per-image", *[f"{n}={d2_out}/per_image_{n}_{SPLIT}.csv" for n in preds],
            "--split", SPLIT, "--out", str(mining_out), "--topk", "3"]
    old = sys.argv
    try:
        sys.argv = argv
        fm.main()
    finally:
        sys.argv = old
    mroot = mining_out / "mining"
    for f in ["d31_condition_case_matrix.csv", "d32_top_lists.json",
              "d33_top_confusions.md", "d34_cc_recall.csv", "d35_case_sensitivity.csv"]:
        assert (mroot / f).exists(), f"D3 산출물 누락: {f}"
    assert list((mroot / "confusion").glob("*.npy")), "혼동행렬 npy 누락"
    print("[smoke] (c) 목록·행렬·케이스표 생성 ✓")

    # (b) 연결성분 recall == 독립 손계산
    exp = independent_cc_recall(preds, gts)
    got = {}
    import csv as _csv
    with open(mroot / "d34_cc_recall.csv", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            got.setdefault(r["model"], {})[r["area_bin"]] = (
                None if r["recall"] == "" else float(r["recall"]))
    for name in preds:
        for b, ev in exp[name].items():
            gv = got[name][b]
            if ev is None:
                assert gv is None, f"[b] {name}/{b} 기대 None, 실제 {gv}"
            else:
                assert gv is not None and abs(gv - ev) < 1e-6, \
                    f"[b] {name}/{b} recall {gv} != 손계산 {ev}"
    # 통제된 설계 확인: ours 모든 bin recall=1, dgf=0
    assert exp["ours"]["100-1k"] == 1.0 and exp["dgf"]["100-1k"] == 0.0
    print("[smoke] (b) 연결성분 recall == 독립 손계산 ✓")

    # d32 목록 내용 sanity: dgf 가 못 맞춘 만큼 ours≫dgf 목록이 존재
    lists = json.loads((mroot / "d32_top_lists.json").read_text(encoding="utf-8"))
    assert any("dgf" in k and "worse_than_ours" in k for k in lists), \
        f"기대한 상위목록 키 없음: {list(lists)}"

    # ---- 파일 5: viz_panels (가짜 deliver-root — 모달은 검게, 패널은 생성) ----
    from tools.baseline_failure import viz_panels as vp
    viz_out = tmp / "viz"
    argv = ["viz_panels.py", "--gt", str(gt_dir),
            "--models", *[f"{n}={pred_dirs[n]}" for n in preds],
            "--deliver-root", str(tmp / "fake_deliver"), "--split", "test",
            "--out", str(viz_out), "--list-json", str(mroot / "d32_top_lists.json"),
            "--topn", "2"]
    old = sys.argv
    try:
        sys.argv = argv
        vp.main()
    finally:
        sys.argv = old
    panels = list((viz_out / "panels").glob("**/*.png"))
    assert panels, "패널 PNG 미생성"
    print(f"[smoke] (c) 패널 {len(panels)} 장 생성 ✓")

    # ---- 요구 E: RLE 변환 + 라벨 규약 이동량 ----
    smoke_rle(tmp)
    smoke_label_offset(tmp)

    # ---- A2~A8 확장: GT 프로토콜 · 체크포인트 안정성 · depth 구간 IoU · zero-out 규약 · 1024 정확 경로 ----
    smoke_gt_protocol(tmp)
    smoke_ckpt_stability(tmp)
    smoke_depth_bin(tmp)
    smoke_zero_mode()
    smoke_keep_1024(tmp)
    smoke_exact_1024(tmp)

    # ---- 파일 1·2·6·7: import + argparse + 규약 동치성 ----
    from tools.baseline_failure import dump_preds_ours, d2_dump_evaluator, \
        modality_zero_ablation, probe_dgfusion, probe_cafuser, \
        rle_json_to_png, check_label_convention
    for mod in (dump_preds_ours, d2_dump_evaluator, modality_zero_ablation,
                probe_dgfusion, probe_cafuser, rle_json_to_png, check_label_convention):
        assert hasattr(mod, "main"), f"{mod.__name__}.main 없음"
    # d2 evaluator 의 id 규약이 common 과 동일해야 join 이 성립(상대 경로).
    sample = "/data/DELIVER/img/night/test/scene_a_lidarjitter/000001_rgb_front.png"
    assert d2_dump_evaluator._image_id_from_rel(sample) == \
        common.image_id_from_rel(sample) == \
        "img/night/test/scene_a_lidarjitter/000001_rgb_front", "id 규약 불일치"
    # dataset_root 로 잘라도 같은 상대 경로가 나오는지.
    assert common.image_id_from_rel(sample, dataset_root="/data/DELIVER") == \
        "img/night/test/scene_a_lidarjitter/000001_rgb_front"
    # check_label_convention(GT PNG 규약): 0~24+255 판정
    rep = d2_dump_evaluator.check_label_convention(gt_dir, sample=4)
    assert rep["ok"] and rep["convention"] == "trainid_0_24", rep
    # 평탄 덤프 감지: 평탄 디렉터리에 require_nested → 에러
    flat_dir = tmp / "flat"
    common.save_label_png(flat_dir / "000001_rgb_front.png", gts[IMAGES[0]])
    try:
        common.index_label_pngs(flat_dir, require_nested=True)
        raise AssertionError("평탄 덤프인데 에러가 나지 않았다")
    except RuntimeError:
        pass
    # argparse 파싱만(무인자 → SystemExit)
    for mod in (dump_preds_ours, modality_zero_ablation, probe_dgfusion, probe_cafuser,
                rle_json_to_png, check_label_convention):
        try:
            old = sys.argv
            sys.argv = [f"{mod.__name__}.py"]
            mod.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old
    print("[smoke] 파일 1·2·6·7 import·argparse·id 규약 동치성·평탄감지 ✓")

    print(f"\n[smoke] 전체 통과. 산출물: {tmp}")


if __name__ == "__main__":
    run()
