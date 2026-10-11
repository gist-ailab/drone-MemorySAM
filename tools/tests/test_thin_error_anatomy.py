#!/usr/bin/env python3
"""thin_error_anatomy의 CPU 합성 장면 검증."""
import csv
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from tools.baseline_failure import common
from tools.baseline_failure import thin_error_anatomy as anatomy


def test_hand_count_metrics():
    c = common.THIN_CLASS_IDS[0]
    other = 0
    gt = np.array([[c, c, other], [c, other, other],
                   [common.IGNORE_LABEL, other, c]], dtype=np.uint8)
    pred = np.array([[c, other, c], [c, other, c],
                     [c, other, other]], dtype=np.uint8)
    cm = common.confusion_matrix(pred, gt, common.N_CLASSES, common.IGNORE_LABEL)
    metric = anatomy.class_metrics(cm)
    assert metric["tp"][c] == 2
    assert metric["fp"][c] == 2
    assert metric["fn"][c] == 2
    for key, expected in (("iou", 1/3), ("precision", 1/2),
                          ("recall", 1/2), ("fp_loss", 1/3),
                          ("fn_loss", 1/3)):
        assert np.isclose(metric[key][c], expected), (key, metric[key][c])
    print("PASS a: 원시 혼동행렬·precision·recall·FP/FN 손계산")


def test_distance_bins():
    gt = np.zeros((100, 100), dtype=bool)
    gt[40:60, 40:44] = True
    pred = gt.copy()
    pred[40:60, 39:45] = True
    near = pred & ~gt
    counts, _ = anatomy.distance_bin_counts(near, gt)
    assert counts[0] == int(near.sum()) == 40
    assert counts[1:].sum() == 0
    far = np.zeros_like(gt)
    far[0:4, 0:4] = True
    counts, _ = anatomy.distance_bin_counts(far, gt)
    assert counts[5] == int(far.sum()) == 16
    assert counts[:5].sum() == 0
    print("PASS b: 1px 두꺼운 막대 FP≤1, 먼 가짜 덩어리 FP>32")


def test_thickness():
    mask = np.ones((30, 8), dtype=bool)
    thickness, patches, bin_idx = anatomy.thickness_patch_bin(mask, 128, 16, 128)
    assert abs(thickness - 8) <= 1, thickness
    assert patches == 0.5
    assert anatomy.THICK_LABELS[bin_idx] == "0.5-1"
    print("PASS c: 폭 8px 성분=두께 8px·0.5패치 구간")


def test_full_scan_and_raw_cm():
    c = common.THIN_CLASS_IDS[0]
    gt = np.full((64, 64), 0, dtype=np.uint8)
    gt[20:28, 30:38] = c
    gt[0, 0] = common.IGNORE_LABEL
    pred1 = gt.copy()
    pred1[20:28, 29] = c
    pred2 = gt.copy()
    pred2[20:28, 30] = 0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = {}
        for name, arr in (("gt", gt), ("ours", pred1), ("base", pred2)):
            path = root / name / "cloud" / "test" / "scene" / "0001.png"
            path.parent.mkdir(parents=True)
            Image.fromarray(arr).save(path)
            paths[name] = common.index_label_pngs(root / name, require_nested=True)
        ids, cm_all, cm_cond, dist, sums_gt, sums_model, thick = anatomy.scan(
            paths["gt"], {"ours": paths["ours"], "base": paths["base"]},
            model_input_px=64, patch_px=16, workers=2)
        for name, pred in (("ours", pred1), ("base", pred2)):
            expected = common.confusion_matrix(pred, gt, common.N_CLASSES,
                                               common.IGNORE_LABEL)
            assert np.array_equal(cm_all[name], expected)
            assert np.array_equal(cm_cond[name]["cloud"], expected)
            metric = anatomy.class_metrics(expected)
            assert np.array_equal(dist[name][:, 0].sum(axis=1), metric["fp"])
            assert np.array_equal(dist[name][:, 1].sum(axis=1), metric["fn"])
        out = root / "out"
        anatomy.write_outputs(out, ["ours", "base"], ids, cm_all, cm_cond,
                              dist, sums_gt, sums_model, thick, "test", 16)
        assert np.array_equal(np.load(out / "raw_confusion" / "ours.npy"),
                              common.confusion_matrix(pred1, gt, common.N_CLASSES,
                                                      common.IGNORE_LABEL))
        assert np.array_equal(np.load(out / "raw_confusion" / "conditions" /
                                      "ours" / "cloud.npy"), cm_all["ours"])
        assert (out / "summary.md").is_file()
        for file in ("precision_recall.csv", "boundary_distance.csv", "thickness_patch.csv"):
            with (out / file).open(newline="", encoding="utf-8") as handle:
                assert list(csv.DictReader(handle)), file
    print("PASS d: 이미지 병렬 집계·저장 원시 혼동행렬=common.confusion_matrix")


if __name__ == "__main__":
    test_hand_count_metrics()
    test_distance_bins()
    test_thickness()
    test_full_scan_and_raw_cm()
    print("모든 합성 CPU 테스트 통과")
