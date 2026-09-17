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
        pi, miou, _ok = pim.score_model(name, pred_dirs[name], gt_dir, SPLIT, d2_out)
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
