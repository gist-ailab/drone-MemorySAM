#!/usr/bin/env python3
"""
D3 — 실패 사례 채굴. joined CSV(D2) + per-image CSV(D2) + 예측/GT 디렉터리로 설계서
D3 의 6항목을 만든다. 출력 루트 = `<out>/mining/`.

  D3-1 조건×케이스 행렬(모델별 mIoU·얇은 객체·큰 영역 + 모델 간 차)  → md + csv
  D3-2 상위 사례 목록 4종(각 --topk 장, 조건·케이스·클래스 요약)      → json
  D3-3 조건별 25×25 정규화 혼동행렬(모델별) + 상위 혼동쌍            → npy + md
  D3-4 GT 연결성분 면적 5구간 × 모델별 recall(성분 픽셀 50%↑ 검출)    → csv + md
  D3-5 같은 조건 안 case=none 대비 각 케이스 ΔmIoU(모델별)          → csv
  D3-6 클래스별 (this split) 데이터셋 IoU, 그리고 --per-class-other 주면 val−test → json + csv

예:
  python tools/baseline_failure/failure_mining.py \
    --joined .../joined_test.csv --gt .../ours/test/gt \
    --per-image ours=.../per_image_ours_test.csv dgf80k=... caf=... \
    --models ours=.../ours/test/pred dgf80k=... caf=... \
    --split test --out .../out
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402

N = common.N_CLASSES
CLASSES = common.CLASSES
# 연결성분 면적 5구간(픽셀, 로그 스케일). 데이터와 무관한 고정 경계 → 재현·손계산 가능.
AREA_EDGES = [0, 100, 1000, 10000, 100000, float("inf")]
AREA_LABELS = ["<100", "100-1k", "1k-10k", "10k-100k", ">=100k"]
DETECT_FRAC = 0.5   # 성분 픽셀의 이 비율 이상을 맞으면 검출


# ---------------------------------------------------------------------------
# 입력 파서
# ---------------------------------------------------------------------------
def read_joined(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        names = [h[len("mIoU_"):] for h in r.fieldnames if h.startswith("mIoU_")]
        for d in r:
            rec = {"image_id": d["image_id"], "condition": d["condition"],
                   "case": d["case"], "miou": {}, "thin": {}, "large": {}}
            for n in names:
                rec["miou"][n] = float(d[f"mIoU_{n}"])
                rec["thin"][n] = _f(d.get(f"thin_{n}"))
                rec["large"][n] = _f(d.get(f"large_{n}"))
            rows.append(rec)
    return names, rows


def read_per_image(path):
    """per_image CSV → id→iou_vec(np.array, NaN 포함)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            vec = np.array([_f(d[f"IoU_{i}"]) for i in range(N)], dtype=np.float64)
            out[d["image_id"]] = vec
    return out


def _f(x):
    if x is None or x == "":
        return float("nan")
    return float(x)


# ---------------------------------------------------------------------------
# D3-1 조건×케이스 행렬
# ---------------------------------------------------------------------------
def d31_matrix(names, rows, ref, out_dir):
    cells = defaultdict(lambda: defaultdict(list))   # (cond,case)->name->[miou]
    thin = defaultdict(lambda: defaultdict(list))
    large = defaultdict(lambda: defaultdict(list))
    for rec in rows:
        key = (rec["condition"], rec["case"])
        for n in names:
            cells[key][n].append(rec["miou"][n])
            thin[key][n].append(rec["thin"][n])
            large[key][n].append(rec["large"][n])

    csv_path = out_dir / "d31_condition_case_matrix.csv"
    header = ["condition", "case", "n"]
    for n in names:
        header += [f"mIoU_{n}", f"thin_{n}", f"large_{n}"]
    for n in names:
        if n != ref:
            header += [f"dmIoU_{ref}_{n}"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key in sorted(cells):
            cond, case = key
            n_img = len(next(iter(cells[key].values())))
            row = [cond, case, n_img]
            means = {}
            for n in names:
                means[n] = common.nanmean(cells[key][n])
                row += [_r(means[n]), _r(common.nanmean(thin[key][n])),
                        _r(common.nanmean(large[key][n]))]
            for n in names:
                if n != ref:
                    row += [_r(means[ref] - means[n])]
            w.writerow(row)
    print(f"[D3-1] 조건×케이스 행렬 -> {csv_path.name}")
    return csv_path


# ---------------------------------------------------------------------------
# D3-2 상위 사례 목록
# ---------------------------------------------------------------------------
def _class_summary(ids, per_image, name, by_id=None):
    """목록 이미지들에 대한 (loser 모델의) 클래스별 평균 IoU 최약 5 + 조건/케이스 분포."""
    summ = {"n": len(ids), "by_condition": defaultdict(int), "by_case": defaultdict(int)}
    if by_id is not None:
        for i in ids:
            summ["by_condition"][by_id[i]["condition"]] += 1
            summ["by_case"][by_id[i]["case"]] += 1
    if name in per_image:
        mat = np.vstack([per_image[name][i] for i in ids]) if ids else np.zeros((0, N))
        with np.errstate(all="ignore"):
            cls_mean = np.nanmean(mat, axis=0) if len(mat) else np.full(N, np.nan)
        order = np.argsort(np.nan_to_num(cls_mean, nan=1.0))[:5]
        summ["worst_classes"] = [
            {"class": CLASSES[c], "mean_iou": _r(cls_mean[c])} for c in order]
    return summ


def d32_lists(names, rows, per_image, ref, out_dir, topk):
    by_id = {rec["image_id"]: rec for rec in rows}
    ids = list(by_id.keys())
    dgf = next((n for n in names if "dgf" in n.lower()), None)
    caf = next((n for n in names if "caf" in n.lower()), None)
    lists = {}

    def entry(i, loser):
        rec = by_id[i]
        return {"image_id": i, "condition": rec["condition"], "case": rec["case"],
                "mIoU": {n: _r(rec["miou"][n]) for n in names}, "loser": loser}

    if ref in names and dgf:
        s = sorted(ids, key=lambda i: by_id[i]["miou"][dgf] - by_id[i]["miou"][ref],
                   reverse=True)[:topk]
        lists[f"{ref}_much_worse_than_{dgf}"] = {
            "items": [entry(i, ref) for i in s], "summary": _class_summary(s, per_image, ref, by_id)}
        s2 = sorted(ids, key=lambda i: by_id[i]["miou"][ref] - by_id[i]["miou"][dgf],
                    reverse=True)[:topk]
        lists[f"{dgf}_much_worse_than_{ref}"] = {
            "items": [entry(i, dgf) for i in s2], "summary": _class_summary(s2, per_image, dgf, by_id)}

    s3 = [i for i in ids if all(by_id[i]["miou"][n] < 0.40 for n in names)]
    s3 = sorted(s3, key=lambda i: np.mean([by_id[i]["miou"][n] for n in names]))[:topk]
    lists["all_models_below_40"] = {
        "items": [entry(i, "all") for i in s3], "summary": _class_summary(s3, per_image, ref if ref in names else names[0], by_id)}

    if dgf and caf:
        s4 = sorted(ids, key=lambda i: abs(by_id[i]["miou"][caf] - by_id[i]["miou"][dgf]),
                    reverse=True)[:topk]
        lists[f"{dgf}_vs_{caf}_diverge"] = {
            "items": [entry(i, "diverge") for i in s4], "summary": _class_summary(s4, per_image, dgf, by_id)}

    json_path = out_dir / "d32_top_lists.json"
    json_path.write_text(json.dumps(lists, indent=2, ensure_ascii=False,
                                    default=lambda o: dict(o) if isinstance(o, defaultdict) else o),
                         encoding="utf-8")
    print(f"[D3-2] 상위 사례 목록 {len(lists)} 종 -> {json_path.name}")
    return json_path, lists


# ---------------------------------------------------------------------------
# D3-3/4/6 를 한 번의 이미지 순회로 계산
# ---------------------------------------------------------------------------
def _bin_index(area):
    for b in range(len(AREA_LABELS)):
        if AREA_EDGES[b] <= area < AREA_EDGES[b + 1]:
            return b
    return len(AREA_LABELS) - 1


def scan_dirs(names_dirs, gt_dir, rows, out_dir, split):
    """예측/GT 를 한 번 순회하며 D3-3(혼동)·D3-4(연결성분 recall)·D3-6(클래스 IoU) 계산."""
    from scipy import ndimage

    cond_of = {rec["image_id"]: rec["condition"] for rec in rows}
    ids = sorted(cond_of.keys())

    # D3-3: model -> condition -> hist[gt,pred]
    conf = {n: defaultdict(lambda: np.zeros((N, N), dtype=np.int64)) for n in names_dirs}
    # D3-6: model -> global hist
    ghist = {n: np.zeros((N, N), dtype=np.int64) for n in names_dirs}
    # D3-4: model -> bin -> [detected, total]
    cc = {n: np.zeros((len(AREA_LABELS), 2), dtype=np.int64) for n in names_dirs}

    for image_id in ids:
        gt = common.load_label_png(Path(gt_dir) / f"{image_id}.png")
        cond = cond_of[image_id]
        # 클래스별 연결성분(모델 무관) 미리 계산.
        comps = []   # (mask_bool, area)
        for c in range(N):
            m = gt == c
            if not m.any():
                continue
            lab, nlab = ndimage.label(m)
            for k in range(1, nlab + 1):
                cmask = lab == k
                comps.append((c, cmask, int(cmask.sum())))
        for n, d in names_dirs.items():
            pred = common.load_label_png(Path(d) / f"{image_id}.png")
            if pred.shape != gt.shape:
                pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
            cm = common.confusion_matrix(pred, gt, N, common.IGNORE_LABEL)
            conf[n][cond] += cm
            ghist[n] += cm
            for c, cmask, area in comps:
                hit = int((pred[cmask] == c).sum())
                detected = hit >= DETECT_FRAC * area
                b = _bin_index(area)
                cc[n][b, 0] += int(detected)
                cc[n][b, 1] += 1

    # ---- D3-3 저장
    conf_dir = out_dir / "confusion"
    conf_dir.mkdir(parents=True, exist_ok=True)
    top_lines = ["# D3-3 조건별 상위 혼동쌍 (정규화 혼동행렬 off-diagonal)", ""]
    for n in names_dirs:
        for cond, hist in sorted(conf[n].items()):
            row_sum = hist.sum(1, keepdims=True)
            norm = np.divide(hist, np.maximum(row_sum, 1), dtype=np.float64)
            np.save(conf_dir / f"{n}__{cond}.npy", norm)
            off = [(norm[g, p], g, p) for g in range(N) for p in range(N) if g != p]
            off.sort(reverse=True)
            top_lines.append(f"## {n} / {cond}")
            for v, g, p in off[:8]:
                if v <= 0:
                    break
                top_lines.append(f"- GT **{CLASSES[g]}** → pred **{CLASSES[p]}** : {v:.3f}")
            top_lines.append("")
    (out_dir / "d33_top_confusions.md").write_text("\n".join(top_lines), encoding="utf-8")
    print(f"[D3-3] 조건별 혼동행렬 npy + 상위쌍 -> confusion/, d33_top_confusions.md")

    # ---- D3-4 저장
    cc_csv = out_dir / "d34_cc_recall.csv"
    with open(cc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "area_bin", "detected", "total", "recall"])
        for n in names_dirs:
            for b, lab in enumerate(AREA_LABELS):
                det, tot = int(cc[n][b, 0]), int(cc[n][b, 1])
                rec = det / tot if tot else float("nan")
                w.writerow([n, lab, det, tot, _r(rec)])
    print(f"[D3-4] 연결성분 recall -> {cc_csv.name}")

    # ---- D3-6 (this split) 클래스별 데이터셋 IoU
    perclass = {}
    for n in names_dirs:
        ious, _ = common.global_miou_from_cm(ghist[n])
        perclass[n] = {CLASSES[i]: ious[i] for i in range(N)}
    pc_path = out_dir / f"perclass_{split}.json"
    pc_path.write_text(json.dumps(perclass, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[D3-6] {split} 클래스별 데이터셋 IoU -> {pc_path.name}")

    return {"cc": cc, "perclass": perclass}


# ---------------------------------------------------------------------------
# D3-5 케이스 민감도
# ---------------------------------------------------------------------------
def d35_case_sensitivity(names, rows, out_dir):
    per = defaultdict(lambda: defaultdict(list))    # (cond,case)->name->list
    for rec in rows:
        for n in names:
            per[(rec["condition"], rec["case"])][n].append(rec["miou"][n])

    conds = sorted({c for (c, _k) in per})
    csv_path = out_dir / "d35_case_sensitivity.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "case"] + [f"dmIoU_{n}_vs_none" for n in names])
        for cond in conds:
            base = {n: common.nanmean(per.get((cond, common.CASE_NONE), {}).get(n, [np.nan]))
                    for n in names}
            cases = sorted({k for (c, k) in per if c == cond and k != common.CASE_NONE})
            for case in cases:
                row = [cond, case]
                for n in names:
                    cur = common.nanmean(per[(cond, case)][n])
                    row.append(_r(cur - base[n]))
                w.writerow(row)
    print(f"[D3-5] 케이스 민감도(vs none) -> {csv_path.name}")
    return csv_path


# ---------------------------------------------------------------------------
# D3-6 val−test 델타(선택)
# ---------------------------------------------------------------------------
def d36_val_test_delta(perclass_this, other_json, split, out_dir):
    other = json.loads(Path(other_json).read_text(encoding="utf-8"))
    # 파일명에서 상대 split 을 추정만; 계산은 this − other.
    csv_path = out_dir / f"d36_perclass_delta_{split}_vs_other.csv"
    names = [n for n in perclass_this if n in other]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class"] + [f"{n}_this" for n in names]
                   + [f"{n}_other" for n in names] + [f"{n}_delta" for n in names])
        for c in CLASSES:
            row = [c]
            row += [_r(perclass_this[n].get(c)) for n in names]
            row += [_r(other[n].get(c)) for n in names]
            row += [_r(perclass_this[n].get(c, np.nan) - other[n].get(c, np.nan)) for n in names]
            w.writerow(row)
    print(f"[D3-6] this vs other split 클래스별 델타 -> {csv_path.name}")
    return csv_path


def _r(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except TypeError:
        return ""
    return round(float(x), 6)


def _parse_kv(items, flag):
    out = {}
    for spec in items or []:
        if "=" not in spec:
            raise SystemExit(f"{flag} 는 name=값 형식이어야 함: {spec}")
        k, v = spec.split("=", 1)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joined", required=True, help="D2 joined_<split>.csv")
    ap.add_argument("--gt", required=True, help="공통 GT trainID PNG 디렉터리")
    ap.add_argument("--models", required=True, nargs="+", metavar="NAME=DIR",
                    help="모델별 예측 디렉터리")
    ap.add_argument("--per-image", nargs="*", metavar="NAME=CSV",
                    help="모델별 per_image CSV(클래스 요약용, 선택)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--ref", default=None, help="기준 모델(기본 ours 있으면 ours)")
    ap.add_argument("--per-class-other", default=None,
                    help="상대 split 의 perclass_*.json (주면 D3-6 델타 계산)")
    args = ap.parse_args()

    out_dir = Path(args.out) / "mining"
    out_dir.mkdir(parents=True, exist_ok=True)

    names, rows = read_joined(args.joined)
    ref = args.ref or ("ours" if "ours" in names else names[0])
    models_dirs = _parse_kv(args.models, "--models")
    models_dirs = {k: (Path(v) / "pred" if (Path(v) / "pred").is_dir() else Path(v))
                   for k, v in models_dirs.items()}
    per_image = {k: read_per_image(v) for k, v in _parse_kv(args.per_image, "--per-image").items()}

    d31_matrix(names, rows, ref, out_dir)
    d32_lists(names, rows, per_image, ref, out_dir, args.topk)
    scan = scan_dirs(models_dirs, args.gt, rows, out_dir, args.split)
    d35_case_sensitivity(names, rows, out_dir)
    if args.per_class_other:
        d36_val_test_delta(scan["perclass"], args.per_class_other, args.split, out_dir)

    print(f"[failure_mining] 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
