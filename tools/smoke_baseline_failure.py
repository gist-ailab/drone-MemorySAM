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
  (A7·A12) zero_modal_in_batch — normalized 는 모델 버퍼 pixel_mean[3i:3i+3] 채움
       ((x-mean)/std → 0), raw 는 0 채움(옛 동작). 모달 인덱스 계산(probe·패치 동치),
       모델 슬라이스==cfg 검증 통과/불일치 에러, 버퍼 길이·모달 부재·버퍼 부재 에러 포함
  (A11) 모달 PIXEL_MEAN 탐색 — 마지막 세그먼트 정확 일치(부분일치 부정: EVENT_CAMERA 는
       CAMERA·EVENT 어느 쪽 후보도 아님) + 우선순위 채택(DELIVER 특화 > DATASETS.* >
       MODEL.*) + 같은 우선순위 값 불일치 모호 에러. probe_dgfusion 탐색과
       d2_zero_modality.patch 본문에서 추출한 탐색이 같은 입력에 같은 결과를 내는지
       (추출 영역에 A12 모달 순서 helper 도 포함 — A7 이 probe 와 동치 검증).
  (A8) rle_json_to_png — 저장은 항상 RLE 해상도(1024²) 그대로(--gt_dir 가 있어도
       변경 없음). --keep_1024 는 1024 단언 검증(1024² 면 통과, 아니면 에러) +
       요약 pred_resolution·keep_1024 기록.
       per_image_metrics resized1024 — 1024 예측 디렉터리 exact / 1042 예측 디렉터리
       resampled 로 기록, 같은 예측·GT 로 두 경로 모두 mIoU 100, 근사 note 는
       resampled 모델이 있을 때만
  (A9) recover_1024_from_1042 — 합성 1042² 라벨을 중심 정렬 최근접으로 1024² 복원:
       출력 1024²·값 집합이 입력의 부분집합·요약(approx_from_1042·resize_method) 기록.
       1024² 입력은 보정 없이 에러. floor(torch nearest) vs 중심(PIL NEAREST) 정렬이
       실제로 다른 결과를 냄을 가는 줄무늬로 단언. dump_preds_ours --save_1024 는
       모델이 필요해 argparse 인자 존재만 단언.
  (A10) d2_eval_batch.patch(BF_EVAL_BATCH — 평가 배치) — zero 패치와 양방향 병합 적용이
       합성 train_net/test_net 사본에서 모두 성공·패치 산출 블록이 복원 킷 정본과 byte
       정합. run_eval_autobatch.sh --dry_run — 24GB 가정 후보(16 12 8 6 4 2 1)·최종
       명령 전달·인자 파싱 에러를 GPU·모델 없이 단언.
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
# A7·A12 — zero_modal_in_batch 개입 방식(normalized|raw) · 모델 버퍼 구조
# ---------------------------------------------------------------------------
# A12 구조 — 모델은 모달별 3채널 평균을 하나의 긴 pixel_mean 버퍼로 이어 붙여 갖고
# 모달 순서(주 모달 먼저 + cfg.DATASETS.MODALITIES.ORDER 나머지)로 슬라이스해 쓴다.
A7_MODAL_ORDER = ("CAMERA", "LIDAR", "EVENT", "DEPTH")
A7_BUF_MEANS = {"CAMERA": (0.485, 0.456, 0.406),
                "LIDAR": (0.35, 0.36, 0.37),
                "EVENT": (0.1, 0.2, 0.3),
                "DEPTH": (0.7, 0.71, 0.72)}


def _a7_zero_cfg(order=("LIDAR", "EVENT", "DEPTH"), main="CAMERA", cfg_means=None):
    """가짜 cfg — DATASETS.MODALITIES(ORDER·MAIN_MODALITY) + DELIVER 모달별 PIXEL_MEAN.
    cfg_means 로 특정 모달의 cfg 평균을 덮어쓸 수 있다(모델 슬라이스 불일치 검증용)."""
    means = {k: list(v) for k, v in A7_BUF_MEANS.items()}
    for k, v in (cfg_means or {}).items():
        means[k] = list(v)
    return {"DATASETS": {"MODALITIES": {"ORDER": list(order), "MAIN_MODALITY": main},
                         "DELIVER": {"PIXEL_MEAN": means}}}


def _a7_fake_model(buf_means=None, n_modals=4):
    """가짜 모델 — pixel_mean 버퍼 = 모달 순서×3채널을 이어 붙인 긴 텐서(4모달=12).
    buf_means 로 일부 모달 슬라이스를 덮어쓰거나, n_modals 로 모달 수를 줄여 버퍼
    길이 불일치 상태를 만들 수 있다."""
    import torch

    class _Model:
        def __init__(self, pm):
            self.pixel_mean = pm

    means = {k: list(v) for k, v in A7_BUF_MEANS.items()}
    for k, v in (buf_means or {}).items():
        means[k] = list(v)
    vals = [x for m in A7_MODAL_ORDER[:n_modals] for x in means[m]]
    return _Model(torch.tensor(vals, dtype=torch.float32))


def smoke_zero_mode():
    """(A7·A12) 가짜 모델 버퍼(pixel_mean 길이 12, 모달 4개)·가짜 cfg 로: 모달 인덱스
    계산(probe·패치 추출본 동치), 모델 슬라이스==cfg 평균이면 검증 통과·다르면 에러,
    normalized 채움이 그 슬라이스 값이고 (x-mean)/std → 0, 버퍼 길이 불일치·모달 부재·
    버퍼 부재는 에러, raw 는 0 채움(옛 동작)을 단언한다."""
    import contextlib
    import io
    import torch
    from tools.baseline_failure import probe_dgfusion as pdg

    cfg = _a7_zero_cfg()
    model = _a7_fake_model()

    def _batch():
        return [{"CAMERA": torch.randn(3, 8, 10) + 10.0,   # 0 이 아닌 값들
                 "image": torch.randn(3, 8, 10) + 10.0,
                 "LIDAR": torch.randn(3, 8, 10) + 10.0,
                 "EVENT": torch.randn(3, 8, 10) + 10.0}]

    # 모달 순서·인덱스 — 주 모달 먼저, 나머지는 ORDER 순(ORDER 순서를 바꿔도 규칙 유지).
    order = pdg.modal_order_from_cfg(cfg)
    assert order == list(A7_MODAL_ORDER), order
    perm = _a7_zero_cfg(order=("DEPTH", "LIDAR", "EVENT"))
    assert pdg.modal_order_from_cfg(perm) == ["CAMERA", "DEPTH", "LIDAR", "EVENT"]
    patch_order = _a11_patch_ns()["_bf_modal_order"]
    assert patch_order(cfg) == order, "패치 추출본 모달 순서가 probe 와 다르다(cfg)"
    assert patch_order(perm) == pdg.modal_order_from_cfg(perm), \
        "패치 추출본 모달 순서가 probe 와 다르다(perm)"
    idx, mean = pdg.model_modal_mean(model, cfg, "LIDAR")
    assert idx == 1, idx
    assert all(abs(a - b) < 1e-6 for a, b in zip(mean, A7_BUF_MEANS["LIDAR"])), mean

    # 검증 통과 — 모델 슬라이스 == cfg 평균(DELIVER 특화 경로)이면 통과하고 로그를 남긴다.
    logbuf = io.StringIO()
    with contextlib.redirect_stdout(logbuf):
        pdg.assert_normalized_fill_is_zero(model, "LIDAR", cfg)
    assert "모달 LIDAR 인덱스 1" in logbuf.getvalue(), logbuf.getvalue()

    # 검증 에러 — 모델 슬라이스와 cfg 값이 다르면 멈추고 양쪽 값을 에러에 보인다.
    # (12.5 는 float32 로 정확히 표현되는 값 — 문자열 단언이 반올림에 흔들리지 않게.)
    try:
        pdg.assert_normalized_fill_is_zero(
            _a7_fake_model(buf_means={"LIDAR": (12.5, 12.5, 12.5)}), "LIDAR", cfg)
        raise AssertionError("모델 슬라이스≠cfg 평균인데 검증이 통과했다")
    except RuntimeError as e:
        assert "12.5" in str(e) and "0.35" in str(e), str(e)

    # normalized — 채움 값의 출처는 모델 버퍼 슬라이스. (x-mean)/std 를 적용하면 0.
    b = _batch()
    pdg.zero_modal_in_batch(b, "CAMERA", mode="normalized", cfg=cfg, model=model)
    mc = torch.tensor(A7_BUF_MEANS["CAMERA"]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    for k in ("CAMERA", "image"):
        assert torch.allclose(b[0][k], mc.expand_as(b[0][k])), k
        assert torch.allclose((b[0][k] - mc) / std, torch.zeros_like(b[0][k])), k
    assert not (b[0]["LIDAR"] == 0).any() and not (b[0]["EVENT"] == 0).any(), \
        "normalized 가 지정 밖 모달을 건드렸다"

    # cfg 와 값이 다른 모델 버퍼라도 채움은 그 모델의 슬라이스 값으로 한다(출처=버퍼).
    b = _batch()
    pdg.zero_modal_in_batch(b, "LIDAR", mode="normalized", cfg=cfg,
                            model=_a7_fake_model(buf_means={"LIDAR": (12.5, 12.5, 12.5)}))
    assert torch.allclose(b[0]["LIDAR"], torch.full_like(b[0]["LIDAR"], 12.5))

    # 버퍼 길이 불일치(9 != 3*4) — 에러로 멈춘다.
    try:
        pdg.zero_modal_in_batch(_batch(), "CAMERA", mode="normalized", cfg=cfg,
                                model=_a7_fake_model(n_modals=3))
        raise AssertionError("버퍼 길이가 모달 수와 안 맞는데 에러가 나지 않았다")
    except RuntimeError:
        pass

    # 모달 부재 — cfg 모달 순서에 없는 모달은 에러로 멈춘다.
    try:
        pdg.zero_modal_in_batch(_batch(), "DEPTH", mode="normalized",
                                cfg=_a7_zero_cfg(order=("LIDAR", "EVENT")),
                                model=model)
        raise AssertionError("모달이 cfg 순서에 없는데 에러가 나지 않았다")
    except RuntimeError:
        pass

    # 버퍼 부재(구조가 다른 모델) — 에러에 BF_ZERO_MODE=raw 대안 안내가 있다(구현 3).
    class _NoBufModel:
        pass
    try:
        pdg.zero_modal_in_batch(_batch(), "CAMERA", mode="normalized", cfg=cfg,
                                model=_NoBufModel())
        raise AssertionError("pixel_mean 버퍼가 없는데 에러가 나지 않았다")
    except RuntimeError as e:
        assert "BF_ZERO_MODE=raw" in str(e), str(e)

    # 기본 모드는 normalized — cfg·model 없이는 호출 자체가 거부된다.
    try:
        pdg.zero_modal_in_batch(_batch(), "CAMERA")
        raise AssertionError("기본 모드가 normalized 인데 cfg·model 없이 통과했다")
    except ValueError:
        pass

    # raw — 옛 동작 그대로: 지정 모달(+image)만 원본이 0, 다른 모달은 그대로.
    b = _batch()
    pdg.zero_modal_in_batch(b, "CAMERA", mode="raw", cfg=cfg, model=model)
    assert (b[0]["CAMERA"] == 0).all() and (b[0]["image"] == 0).all()
    assert not (b[0]["LIDAR"] == 0).any() and not (b[0]["EVENT"] == 0).any(), \
        "raw 가 지정 밖 모달을 건드렸다"
    b = _batch()
    pdg.zero_modal_in_batch(b, "LIDAR", mode="raw")     # raw 는 cfg·model 불필요
    assert (b[0]["LIDAR"] == 0).all()
    assert not (b[0]["CAMERA"] == 0).any(), "LIDAR 지정이 CAMERA 를 건드렸다"
    print("[smoke] (A7·A12) zero-modal 모델 버퍼 슬라이스 채움·검증·에러·raw ✓")


# ---------------------------------------------------------------------------
# A11 — 모달 PIXEL_MEAN 탐색: 마지막 세그먼트 정확 일치 + 우선순위 채택
# ---------------------------------------------------------------------------
# jarvis DGFusion 80k 실패 실측(2026-09-18)의 후보 구성 그대로 — 부분일치로
# EVENT_CAMERA 가 CAMERA·EVENT 양쪽에 섞여 "모호하다" 에러로 멈춘 사례.
A11_RGB_MEAN = (123.675, 116.28, 103.53)          # CAMERA 공통(DELIVER·DATASETS·MODEL)
A11_EVENT_MEAN = (0.12577528, 0.12728328, 0.0)    # EVENT_CAMERA(이벤트 카메라 평균)


def _a11_failure_cfg():
    """실패 실측과 같은 모양의 가짜 cfg(점 경로) + LIDAR 우선순위 확인용 두 경로."""
    return {"MODEL": {"PIXEL_MEAN": list(A11_RGB_MEAN),
                      "PIXEL_STD": [1.0, 1.0, 1.0]},
            "DATASETS": {"PIXEL_MEAN": {"CAMERA": list(A11_RGB_MEAN),
                                        "EVENT_CAMERA": list(A11_EVENT_MEAN),
                                        "LIDAR": [0.5]},
                         "DELIVER": {"PIXEL_MEAN": {"CAMERA": list(A11_RGB_MEAN),
                                                    "LIDAR": [0.35]}}}}


def _a11_patch_ns():
    """d2_zero_modality.patch 본문에서 PIXEL_MEAN 탐색·모달 순서 helper 블록을 추출해
    실행 가능하게 만든다 — 패치 쪽 규칙이 probe 쪽과 갈라지지 않았는지 같은 입력으로
    검증하기 위함(탐색=A11, 모달 순서=A7·A12)."""
    import textwrap
    patch = (_REPO_ROOT / "tools/baseline_failure/d2_zero_modality.patch"
             ).read_text(encoding="utf-8")
    added = "\n".join(l[1:] for l in patch.splitlines()
                      if l.startswith("+") and not l.startswith("+++"))
    start = added.index("def _bf_iter_cfg")
    start = added.rfind("\n", 0, start) + 1   # 함수 정의 줄 맨 앞(들여쓰기 포함)부터
    end = added.index("_bf_mean = None")   # helper 블록은 이 대입 직전까지
    ns = {}
    exec(compile(textwrap.dedent(added[start:end]),
                 "<d2_zero_modality.patch>", "exec"), ns)
    return ns


def smoke_a11_mean_search():
    """(A11) probe·패치 양쪽 탐색이 같은 입력에서: CAMERA 는 DELIVER 특화 값을 고르고
    EVENT_CAMERA 값은 절대 고르지 않는다. EVENT 는 후보 없음 에러(EVENT_CAMERA 는 부분
    일치 후보가 아니다). 우선순위가 다른 값 불일치는 채택으로 해결, 같은 우선순위 값
    불일치는 모호 에러로 멈춘다."""
    import contextlib
    import io
    from tools.baseline_failure import probe_dgfusion as pdg

    cfg = _a11_failure_cfg()
    finders = (("probe", pdg.find_modal_pixel_mean),
               ("patch", _a11_patch_ns()["_bf_find_modal_pixel_mean"]))

    # CAMERA — DELIVER 특화 경로를 골라 로그로 남기고, EVENT_CAMERA 값은 배제된다.
    for tag, finder in finders:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mean = finder(cfg, "CAMERA")
        assert mean == A11_RGB_MEAN and mean != A11_EVENT_MEAN, (tag, mean)
        assert "DATASETS.DELIVER.PIXEL_MEAN.CAMERA" in buf.getvalue(), \
            (tag, buf.getvalue())
        assert "EVENT_CAMERA" not in buf.getvalue(), (tag, buf.getvalue())

    # EVENT — EVENT_CAMERA 가 후보로 잡히지 않아 정확 일치 항목이 없다 → 후보 없음 에러.
    for _tag, finder in finders:
        try:
            finder(cfg, "EVENT")
            raise AssertionError("EVENT 후보가 없는데 에러가 나지 않았다")
        except RuntimeError:
            pass

    # 우선순위 — LIDAR 는 DELIVER 특화(0.35) 가 그 외 DATASETS.*(0.5) 보다 이긴다.
    for _tag, finder in finders:
        assert finder(cfg, "LIDAR") == (0.35,), f"{_tag}: 우선순위 채택이 아니다"

    # 같은 우선순위(둘 다 DATASETS.*) 안에서 값이 다르면 여전히 모호 에러.
    amb = {"DATASETS": {"PIXEL_MEAN": {"LIDAR": [0.5]},
                        "MUSES": {"PIXEL_MEAN": {"LIDAR": [0.9]}}}}
    for _tag, finder in finders:
        try:
            finder(amb, "LIDAR")
            raise AssertionError("같은 우선순위 값 불일치인데 에러가 나지 않았다")
        except RuntimeError:
            pass
    print("[smoke] (A11) PIXEL_MEAN 마지막 세그먼트 정확일치·우선순위 채택·probe/패치 동치 ✓")


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


# ---------------------------------------------------------------------------
# A9 — 1042 덤프 근사 복원(중심 정렬) · floor/중심 정렬 차이 · --save_1024 인자
# ---------------------------------------------------------------------------
A9_SRC = A8_GT_H          # 1042 — 우리 덤프가 복원돼 있는 native 해상도
A9_DST = A8_RLE_H         # 1024 — 정본 채점 격자
A9_IDS = ["img/night/test/s1/000020_rgb_front", "img/fog/test/s2/000021_rgb_front"]


def smoke_a9_recover(tmp):
    """(A9-2) 합성 1042² 라벨 → 중심 정렬 최근접 1024² 복원 — 해상도·값 집합 부분집합·
    요약 기록. 1024² 입력은 임의 보정 없이 명확한 에러로 멈춘다."""
    from tools.baseline_failure import recover_1024_from_1042 as rec

    src = {}
    for k in A9_IDS:
        a = np.full((A9_SRC, A9_SRC), ROAD, np.uint8)
        a[100:600, 200:900] = WATER
        a[0:50, :] = 255                                # ignore 포함(값 규약 확인용)
        src[k] = a
    pred_dir = tmp / "a9_src1042"
    save_dir(src, pred_dir)

    out = tmp / "a9_rec1024"
    _run_main(rec, ["recover_1024_from_1042.py", "--pred_dir", str(pred_dir),
                    "--split", "test", "--out", str(out), "--expected_count", "2"])
    idx = common.index_label_pngs(out / "test" / "pred")
    assert len(idx) == 2, f"복원 장수 {len(idx)} != 2"
    for k in A9_IDS:
        r = common.load_label_png(idx[k])
        assert r.shape == (A9_DST, A9_DST), f"{k} 복원 해상도 {r.shape} != 1024²"
        u_out = set(np.unique(r).tolist())
        assert u_out <= set(np.unique(src[k]).tolist()), \
            f"{k}: 복원 후 새 라벨 값 생김: {u_out}"
    summ = json.loads((out / "test" / "summary.json").read_text(encoding="utf-8"))
    assert summ["approx_from_1042"] is True, summ
    assert summ["resize_method"], f"resize_method 미기록: {summ}"
    assert summ["src_resolution"] == 1042 and summ["dst_resolution"] == 1024, summ
    assert summ["num_images"] == 2, summ

    # 이미 1024² 인 입력 — 되돌릴 것이 없으니 보정 없이 에러.
    p1024 = tmp / "a9_src1024"
    save_dir({k: common.resize_nearest(v, 1024, 1024) for k, v in src.items()}, p1024)
    try:
        _run_main(rec, ["recover_1024_from_1042.py", "--pred_dir", str(p1024),
                        "--split", "test", "--out", str(tmp / "a9_bad_out"),
                        "--expected_count", "2"])
        raise AssertionError("1024² 입력인데 에러 없이 통과했다")
    except ValueError:
        pass
    print("[smoke] (A9-2) 1042→1024 중심 정렬 복원·값 부분집합·1024 입력 에러·요약 기록 ✓")


def smoke_a9_alignment():
    """(A9) floor 정렬(torch F.interpolate mode='nearest')과 중심 정렬(PIL NEAREST)
    이 실제로 다른 결과를 냄 — 3픽셀 주기 가는 세로 줄무늬를 1042→1024 로 축소해
    클래스 픽셀수가 다름을 단언한다(같으면 구현이 floor 정렬을 쓰고 있다는 뜻)."""
    import torch
    import torch.nn.functional as F
    from tools.baseline_failure.recover_1024_from_1042 import resize_center_nearest

    a = np.full((A9_SRC, A9_SRC), ROAD, np.uint8)
    a[:, ::3] = POLE                                   # 가는 세로선(얇은 클래스)
    center = resize_center_nearest(a)
    t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).float()
    floor = F.interpolate(t, size=(A9_DST, A9_DST), mode="nearest")
    floor = floor.squeeze().numpy().astype(np.uint8)
    assert center.shape == floor.shape == (A9_DST, A9_DST)
    n_center = int((center == POLE).sum())
    n_floor = int((floor == POLE).sum())
    assert n_center != n_floor, (
        f"floor({n_floor}) == 중심({n_center}) — 두 정렬이 같은 결과를 냈다. "
        f"구현이 floor 정렬을 쓰고 있거나 대조군이 잘못됐다")
    print(f"[smoke] (A9-정렬) floor {n_floor} vs 중심 {n_center} 픽셀 — 실제로 다른 결과 ✓")


def smoke_a9_dump_arg():
    """(A9-1) dump_preds_ours --save_1024 — 모델이 필요해 실행은 못 하고 인자가
    argparse 에 등록돼 있는지만 단언한다(미등록이면 SystemExit=unrecognized)."""
    from tools.baseline_failure import dump_preds_ours as dpo
    old = sys.argv
    try:
        sys.argv = ["dump_preds_ours.py", "--cfg", "nonexistent.yaml",
                    "--model_path", "nonexistent.pth", "--split", "test",
                    "--out", "unused", "--save_1024"]
        dpo.main()
        raise AssertionError("--save_1024 로 main 이 정상 종료했다 — 비정상(cfg 가 없다)")
    except SystemExit:
        raise AssertionError("--save_1024 인자가 인식되지 않았다(argparse 미등록)")
    except Exception:
        pass    # 파싱은 통과하고 cfg 파일 부재 등으로 실패 — 인자 존재 단언에는 충분
    finally:
        sys.argv = old
    print("[smoke] (A9-1) dump_preds_ours --save_1024 인자 등록 ✓")


# ---------------------------------------------------------------------------
# A10 — 평가 배치 옵션 패치 충돌 검증 · autobatch dry-run
# ---------------------------------------------------------------------------
# 두 패치(d2_eval_batch: build_test_loader, d2_zero_modality: main)의 컨텍스트만 갖춘
# 합성 기준선 entry 최소 파일. 실제 원격 저장소(yeon dgfusion_train/lecun cafuser_train)
# 원본은 이 박스에 없어 스모크는 이 합성 사본으로 검증한다.
A10_SYNTH = (
    '"""\n'
    '합성 기준선 entry — 두 패치의 컨텍스트만 갖춘 최소 파일.\n'
    '"""\n'
    "import os\n"
    "\n"
    "\n"
    "class Trainer(DefaultTrainer):\n"
    "    @classmethod\n"
    "    def build_test_loader(cls, cfg, dataset_name):\n"
    '        """\n'
    "        Returns:\n"
    "            iterable\n"
    "        It now calls :func:`detectron2.data.build_detection_test_loader`.\n"
    "        Overwrite it if you'd like a different data loader.\n"
    '        """\n'
    '        if cfg.INPUT.DATASET_MAPPER_NAME == "muses_unified":\n'
    "            mapper = MUSESTestDatasetMapper(cfg, False)\n"
    '        elif cfg.INPUT.DATASET_MAPPER_NAME == "deliver_semantic":\n'
    "            mapper = DELIVERSemanticDatasetMapper(cfg, False)\n"
    "        else:\n"
    "            mapper = DatasetMapper(cfg, False)\n"
    "        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)\n"
    "\n"
    "\n"
    "def main(args):\n"
    "    cfg = setup(args)\n"
    "\n"
    "    if args.inference_only and args.eval_only:\n"
    '        raise Exception("You can only run inference or evaluation, not both at the same time.")\n'
    "\n"
    "    elif args.eval_only or args.inference_only:\n"
    "        model = Trainer.build_model(cfg)\n"
    "        net_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n"
    '        print("Total Params: {} M".format(net_params/1e6))\n'
    "        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(\n"
    "            cfg.MODEL.WEIGHTS, resume=args.resume\n"
    "        )\n"
    "        res = Trainer.test(cfg, model, eval_only=args.eval_only, inference_only=args.inference_only)\n"
    "        if cfg.TEST.AUG.ENABLED:\n"
    "            res.update(Trainer.test_with_TTA(cfg, model))\n")


def smoke_a10_patch(tmp):
    """(A10-1) d2_eval_batch.patch 가 d2_zero_modality.patch 와 함께 적용될 때 충돌하지
    않는지 — 합성 train_net.py/test_net.py 사본에 두 순서로 각각 적용해 모두 성공함을
    단언한다. 추가로 패치 산출 블록이 복원 킷 정본에 byte 일치하고, zero 패치가 배치
    옵션 반영 후의 킷 정본에도 적용되는지 확인한다."""
    import subprocess
    bf = _REPO_ROOT / "tools/baseline_failure"
    patch_b = bf / "d2_eval_batch.patch"
    patch_z = bf / "d2_zero_modality.patch"
    assert patch_b.exists() and patch_z.exists(), "A10 패치 파일 없음"

    # 실제 기준선 저장소 원본(원격 yeon/lecun)은 로컬에 없다 — 원본 대상 적용 검증은
    # 건너뛰고 아래 합성 사본으로 검증한다(킷 정본 대조는 리포 안 파일로 수행).
    print("[smoke] (A10-1) 실제 기준선 저장소 원본 없음 — 합성 파일로 패치 검증(원본 적용은 서버에서)")

    def apply_dir(tag, order):
        d = tmp / tag
        d.mkdir(parents=True)
        (d / "train_net.py").write_text(A10_SYNTH, encoding="utf-8")
        (d / "test_net.py").write_text(A10_SYNTH, encoding="utf-8")
        for p in order:
            r = subprocess.run(["git", "apply", str(p)], cwd=d,
                               capture_output=True, text=True)
            assert r.returncode == 0, f"{tag}/{p.name} 적용 실패:\n{r.stdout}{r.stderr}"
        return (d / "train_net.py").read_text(encoding="utf-8")

    o_bz = apply_dir("a10_bz", [patch_b, patch_z])
    o_zb = apply_dir("a10_zb", [patch_z, patch_b])
    assert o_bz == o_zb, "두 순서의 적용 결과가 다르다"
    for token in ("BF_EVAL_BATCH", "batch_size=_bf_eval_batch", "BF_ZERO_MODAL",
                  "register_forward_pre_hook"):
        assert token in o_bz, f"병합 결과에 {token} 없음"

    kit = _REPO_ROOT / "third_party/dgfusion_train_restore/train_net.py"
    if not kit.exists():
        print("[smoke] (A10-1) 복원 킷 정본이 없어 킷 대조를 건너뛴다: " + str(kit))
        return
    kit_text = kit.read_text(encoding="utf-8")
    patch_text = patch_b.read_text(encoding="utf-8")
    added = [l[1:] for l in patch_text.splitlines(keepends=True)
             if l.startswith("+") and not l.startswith("+++")]
    assert len(added) == 34, f"추가 줄 수 {len(added)} != 34(두 파일 × 17)"
    assert added[17:] == added[:17], "train_net/test_net 두 hunk 본문이 다르다"
    assert "".join(added[:17]) in kit_text, "킷 정본에 패치 추가 블록이 byte 일치하지 않는다"
    # 배치 옵션 반영 후의 킷 정본에 zero 패치도 그대로 적용돼야 한다(공존 확인).
    d = tmp / "a10_kit_zero"
    d.mkdir(parents=True)
    (d / "train_net.py").write_text(kit_text, encoding="utf-8")
    r = subprocess.run(["git", "apply", str(patch_z)], cwd=d,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"킷 정본에 zero 패치 적용 실패:\n{r.stderr}"
    print("[smoke] (A10-1) eval_batch·zero 패치 양방향 병합 + 킷 정본 byte 정합 ✓")


def smoke_a10_autobatch(tmp):
    """(A10-2) run_eval_autobatch.sh —dry_run — GPU·모델 없이 인자 파싱과 배치 후보
    계산을 확인한다. 24GB(24576MiB) 를 가정한 후보 목록이 16 12 8 6 4 2 1 인지,
    최종 명령에 config·weights·오버라이드가 그대로 실리는지, 잘못된 인자(홀수
    오버라이드·필수 인자 누락)가 명확한 에러(종료 코드 1)로 멈추는지 단언한다."""
    import os
    import subprocess
    script = _REPO_ROOT / "tools/baseline_failure/run_eval_autobatch.sh"
    assert script.exists(), f"스크립트 없음: {script}"
    env = dict(os.environ, BF_AUTOBATCH_TOTAL_MIB="24576")
    r = subprocess.run(
        ["bash", str(script), "--dry_run",
         "--repo", "/fake/dgfusion_train", "--cfg", "fake.yaml",
         "--weights", "model_final.pth", "--out", str(tmp / "ab_out"),
         "--log", str(tmp / "ab.log"), "MODEL.TEST.DEPTH_ON", "False"],
        capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"dry_run 실패:\n{r.stdout}{r.stderr}"
    assert "candidates: 16 12 8 6 4 2 1" in r.stdout, r.stdout
    for token in ("BF_EVAL_BATCH=16", "train_net.py", "--config-file fake.yaml",
                  "--eval-only", "MODEL.WEIGHTS model_final.pth",
                  "MODEL.TEST.DEPTH_ON False"):
        assert token in r.stdout, f"최종 명령에 {token} 없음:\n{r.stdout}"
    # 홀수 오버라이드 — KEY VALUE 쌍이 아니면 에러.
    r = subprocess.run(["bash", str(script), "--dry_run", "--repo", "r", "--cfg", "c",
                        "--weights", "w", "--out", "o", "--log", "l", "SOME.KEY"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "KEY VALUE" in r.stderr, (r.returncode, r.stderr)
    # 필수 인자 누락 — 에러.
    r = subprocess.run(["bash", str(script), "--dry_run", "--cfg", "c"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1, r.returncode
    # 48GB 급(48688MiB) 후보도 급 테이블대로 갈린다.
    env48 = dict(os.environ, BF_AUTOBATCH_TOTAL_MIB="48688")
    r = subprocess.run(["bash", str(script), "--dry_run", "--repo", "r", "--cfg", "c",
                        "--weights", "w", "--out", "o", "--log", "l"],
                       capture_output=True, text=True, env=env48)
    assert "candidates: 32 24 16 12 8 6 4 2 1" in r.stdout, r.stdout
    print("[smoke] (A10-2) autobatch dry-run 후보(24GB=16..1)·명령 전달·인자 에러 ✓")


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

    # ---- A2~A9 확장: GT 프로토콜 · 체크포인트 안정성 · depth 구간 IoU · zero-out 규약 · 1024 정확 경로 ----
    smoke_gt_protocol(tmp)
    smoke_ckpt_stability(tmp)
    smoke_depth_bin(tmp)
    smoke_zero_mode()
    smoke_a11_mean_search()
    smoke_keep_1024(tmp)
    smoke_exact_1024(tmp)
    smoke_a9_recover(tmp)
    smoke_a9_alignment()
    smoke_a9_dump_arg()

    # ---- A10 확장: 평가 배치 패치 충돌 검증 · autobatch dry-run ----
    smoke_a10_patch(tmp)
    smoke_a10_autobatch(tmp)

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
