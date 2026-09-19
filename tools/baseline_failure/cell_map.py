#!/usr/bin/env python3
"""
셀 단위 실패 지도 — 저장된 분할 예측 PNG 만 읽어(GPU 불필요) "어떤 상황(조건×케이스)
의 어떤 클래스에서 우리 모델이 기준선에 지는가" 를 보여 주는 원자료 세 벌을 만든다.

설계 맥락 = .claude_logs/experiments/analysis/2026-09-17-baseline-failure-analysis-plan.md.
채점 규약(trainID·혼동행렬 축·IoU 정의)은 tools/baseline_failure/common.py 를 그대로
재사용하며 여기서 다시 구현하지 않는다. 순회 구조(GT·연결성분을 이미지마다 한 번만
계산해 모든 모델이 공유)는 failure_mining.py 의 scan_dirs 를 참고했다(그 파일은 고치지
않고 별도 파일로 둔다).

이 도구의 핵심 구분 — **셀 안에서 혼동행렬을 먼저 누적한 뒤 그 누적 행렬에서 클래스별
IoU 를 계산**한다. 이미지별 IoU 를 낸 다음 평균하는 방식(failure_mining 의 조건×케이스
행렬)과는 값이 다르다. 작은 객체가 드문 이미지들에서 이미지별 평균은 표본이 적은 IoU 를
동등 가중으로 섞어 과대·과소 추정하기 쉽지만, 누적 혼동행렬의 IoU 는 셀 전체 픽셀을
모아 한 번 계산하므로 셀의 실제 성능을 편향 없이 나타낸다. 두 방식을 섞어 인용하면 안
된다(README 에 경고).

산출물:
  ① <out>/cell_class_iou_<split>.csv   셀 × 클래스 IoU(셀 누적 혼동행렬 기반) + ref−최고기준선 델타
  ② <out>/cell_summary_<split>.csv     셀 × 클래스묶음(thin/large/rest/all) 평균 IoU + 델타
  ③ <out>/cell_cc_split_<split>.csv    GT 연결성분별 검출/영역묘사 분리표
  ④ <out>/README.md                    세 파일 설명·정의·주의

예:
  python tools/baseline_failure/cell_map.py \
    --gt   .../preds/ours_gt/test/gt \
    --models ours=.../preds/ours_E1conf_s1/test/pred \
             dgf=.../preds/dgf80k/test/pred \
             caf=.../preds/caf/test/pred \
    --ref ours --split test --out .../out/cell_map
"""
import argparse
import csv
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

# 클래스 묶음 인덱스. thin/large 는 common 의 단일 출처를 쓰고, rest 는 그 둘에 속하지
# 않는 나머지를 뜻한다. "all" 은 25개 전 클래스를 뜻하며 각 성분·클래스가 소속 묶음과
# "all" 에 동시에 집계된다.
THIN_IDS = set(common.THIN_CLASS_IDS)
LARGE_IDS = set(common.LARGE_CLASS_IDS)
REST_IDS = set(range(N)) - THIN_IDS - LARGE_IDS
GROUP_IDS = {
    "thin": sorted(THIN_IDS),
    "large": sorted(LARGE_IDS),
    "rest": sorted(REST_IDS),
    "all": list(range(N)),
}
GROUP_ORDER = ["thin", "large", "rest", "all"]

# 클래스 인덱스 → 소속 묶음 이름(thin/large/rest 중 하나). "all" 은 여기 넣지 않는다
# (모든 클래스가 "all" 에도 들어가므로 성분 집계 때 별도로 함께 더한다).
def _class_group(c):
    if c in THIN_IDS:
        return "thin"
    if c in LARGE_IDS:
        return "large"
    return "rest"


# 연결성분 면적 5구간(픽셀, 로그 스케일). failure_mining.py 의 _bin_index/AREA_LABELS 와
# **같은 경계**를 일부러 재현한다 — 두 도구의 면적 구간이 어긋나면 recall·검출 표를 나란히
# 비교할 수 없기 때문이다. import 하지 않고 값을 복제하되(그 파일은 손대지 않는다) 여기
# 주석으로 출처를 남긴다. 경계·라벨을 바꾸려면 두 파일을 함께 고쳐야 한다.
AREA_EDGES = [0, 100, 1000, 10000, 100000, float("inf")]
AREA_LABELS = ["<100", "100-1k", "1k-10k", "10k-100k", ">=100k"]


def _bin_index(area):
    """면적(픽셀) → area_bin 인덱스. failure_mining._bin_index 와 동일 로직."""
    for b in range(len(AREA_LABELS)):
        if AREA_EDGES[b] <= area < AREA_EDGES[b + 1]:
            return b
    return len(AREA_LABELS) - 1


def _r(x):
    """CSV 셀 포맷: None·NaN 은 빈 칸으로, 그 밖은 소수 6자리. 값을 추측해 채우지 않는다."""
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except TypeError:
        return ""
    return round(float(x), 6)


def _parse_kv(items, flag):
    """name=값 목록을 dict 로. failure_mining._parse_kv 와 같은 규약."""
    out = {}
    for spec in items or []:
        if "=" not in spec:
            raise SystemExit(f"{flag} 는 name=값 형식이어야 함: {spec}")
        k, v = spec.split("=", 1)
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# 한 번의 이미지 순회로 셀 누적 혼동행렬(①②) + 연결성분 분리표(③) 를 함께 만든다.
# GT 로드와 연결성분 라벨링은 이미지마다 한 번만 하고 모든 모델이 공유한다(성능).
# ---------------------------------------------------------------------------
def scan(gt_index, model_index, model_names, detect_frac, delin_iou, limit):
    from scipy import ndimage

    # 모든 모델이 가진 이미지만 다룬다(셀별 n_images 가 모델 간 일치하도록).
    common_ids = set(gt_index.keys())
    for n in model_names:
        common_ids &= set(model_index[n].keys())
    ids = sorted(common_ids)
    n_gt = len(gt_index)
    if len(ids) < n_gt:
        print(f"[cell_map] 경고: GT {n_gt}장 중 모든 모델이 공유하는 {len(ids)}장만 집계"
              f"(누락 이미지가 있는 모델 존재).")
    if limit and limit > 0:
        ids = ids[:limit]
    print(f"[cell_map] 집계 대상 {len(ids)}장 · 모델 {model_names}")

    # ① 셀 누적 혼동행렬: (cond,case) -> model -> hist[gt,pred]
    cell_cm = defaultdict(lambda: {n: np.zeros((N, N), dtype=np.int64) for n in model_names})
    # 셀별 이미지 수.
    cell_n = defaultdict(int)
    # ③ 연결성분 분리표: (cond,case) -> model -> group -> area_bin -> [n, det, well, poor]
    def _new_cc():
        return {n: {g: np.zeros((len(AREA_LABELS), 4), dtype=np.int64) for g in GROUP_ORDER}
                for n in model_names}
    cell_cc = defaultdict(_new_cc)

    for image_id in ids:
        cond, case = common.parse_condition_case(image_id)
        key = (cond, case)
        cell_n[key] += 1
        gt = common.load_label_png(gt_index[image_id])

        # 클래스별 GT 연결성분(모델 무관). bbox 를 함께 저장해 모델마다 다시 안 구한다.
        comps = []   # (class_c, rmin, rmax, cmin, cmax, sub_M(bool), area)
        for c in range(N):
            m = gt == c
            if not m.any():
                continue
            lab, nlab = ndimage.label(m)
            for k in range(1, nlab + 1):
                cmask = lab == k
                ys, xs = np.where(cmask)
                rmin, rmax = int(ys.min()), int(ys.max())
                cmin, cmax = int(xs.min()), int(xs.max())
                sub_M = cmask[rmin:rmax + 1, cmin:cmax + 1]
                comps.append((c, rmin, rmax, cmin, cmax, sub_M, int(cmask.sum())))

        for n in model_names:
            pred = common.load_label_png(model_index[n][image_id])
            if pred.shape != gt.shape:
                pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])

            # ① 셀 누적 혼동행렬.
            cm = common.confusion_matrix(pred, gt, N, common.IGNORE_LABEL)
            cell_cm[key][n] += cm

            # ③ 성분별 검출/영역묘사. 성분 마스크 M, 면적 A, 클래스 c 에 대해,
            #    바운딩 박스 B 안에서 P=(pred==c) 로 두고 inter=|M∩P|, union=|M∪P|,
            #    comp_iou=inter/union.
            for c, rmin, rmax, cmin, cmax, sub_M, area in comps:
                sub_P = pred[rmin:rmax + 1, cmin:cmax + 1] == c
                inter = int((sub_M & sub_P).sum())
                union = int((sub_M | sub_P).sum())
                comp_iou = inter / union if union else 0.0
                detected = inter >= detect_frac * area
                well = detected and comp_iou >= delin_iou
                poor = detected and comp_iou < delin_iou
                b = _bin_index(area)
                for g in (_class_group(c), "all"):   # 소속 묶음 + 전체
                    arr = cell_cc[key][n][g][b]
                    arr[0] += 1
                    arr[1] += int(detected)
                    arr[2] += int(well)
                    arr[3] += int(poor)

    return cell_cm, cell_n, cell_cc


# ---------------------------------------------------------------------------
# 산출물 ① 셀 × 클래스 IoU
# ---------------------------------------------------------------------------
def write_cell_class_iou(cell_cm, cell_n, model_names, ref, baselines, out_path):
    """셀 누적 혼동행렬에서 클래스별 IoU 를 계산해 CSV 로 쓴다.

    GT 에 그 클래스가 셀 안에서 한 화소도 없으면 IoU·델타를 빈 칸으로 둔다(0 으로
    채우지 않는다). per_image_iou 는 GT 행합이 0 인 클래스를 NaN 으로 돌려주므로
    그대로 재사용한다(누적 행렬에 먹여도 "부재=NaN" 규약이 그대로 성립한다)."""
    header = ["condition", "case", "n_images", "class_id", "class_name", "gt_pixels"]
    header += [f"iou_{n}" for n in model_names]
    header += ["best_baseline_iou", "best_baseline_name", "delta_ref_minus_best"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key in sorted(cell_cm):
            cond, case = key
            n_img = cell_n[key]
            # 모델별 per-class IoU(부재 클래스=NaN)와 GT 픽셀수.
            iou = {n: common.per_image_iou(cell_cm[key][n]) for n in model_names}
            # gt_pixels 는 모델 무관(같은 GT) — ref 의 혼동행렬 행합으로 읽는다.
            gt_pix = cell_cm[key][ref].sum(1) if ref in cell_cm[key] \
                else cell_cm[key][model_names[0]].sum(1)
            for c in range(N):
                px = int(gt_pix[c])
                row = [cond, case, n_img, c, CLASSES[c], px]
                for n in model_names:
                    row.append(_r(iou[n][c]))
                # 최고 기준선: 델타·묶음 평균에서 부재(NaN) 클래스는 제외한다.
                if px == 0:
                    row += ["", "", ""]      # 부재 클래스: 델타도 비운다.
                else:
                    cand = [(iou[b][c], b) for b in baselines
                            if not np.isnan(iou[b][c])]
                    if cand:
                        best_v, best_n = max(cand, key=lambda t: t[0])
                        ref_v = iou[ref][c] if ref in iou else float("nan")
                        delta = ref_v - best_v if not np.isnan(ref_v) else float("nan")
                        row += [_r(best_v), best_n, _r(delta)]
                    else:
                        row += ["", "", ""]
                w.writerow(row)
    print(f"[①] 셀 × 클래스 IoU -> {out_path.name}")


# ---------------------------------------------------------------------------
# 산출물 ② 셀 단위 요약(클래스 묶음별 평균 IoU)
# ---------------------------------------------------------------------------
def write_cell_summary(cell_cm, cell_n, model_names, ref, baselines, out_path):
    """묶음(thin/large/rest/all)별 평균 IoU 를 CSV 로 쓴다.

    평균은 셀 누적 혼동행렬의 클래스별 IoU 를, 그 셀에 실제로 존재하는 클래스(GT 픽셀>0
    → per_image_iou 가 NaN 이 아닌 클래스)에 대해서만 낸다(부재 클래스 제외, nanmean)."""
    header = ["condition", "case", "n_images", "group"]
    header += [f"miou_{n}" for n in model_names]
    header += ["best_baseline_miou", "delta_ref_minus_best"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key in sorted(cell_cm):
            cond, case = key
            n_img = cell_n[key]
            iou = {n: common.per_image_iou(cell_cm[key][n]) for n in model_names}
            for g in GROUP_ORDER:
                idx = GROUP_IDS[g]
                miou = {n: common.nanmean([iou[n][c] for c in idx]) for n in model_names}
                row = [cond, case, n_img, g]
                row += [_r(miou[n]) for n in model_names]
                cand = [(miou[b], b) for b in baselines if not np.isnan(miou[b])]
                if cand:
                    best_v = max(v for v, _ in cand)
                    ref_v = miou[ref] if ref in miou else float("nan")
                    delta = ref_v - best_v if not np.isnan(ref_v) else float("nan")
                    row += [_r(best_v), _r(delta)]
                else:
                    row += ["", ""]
                w.writerow(row)
    print(f"[②] 셀 단위 묶음 요약 -> {out_path.name}")


# ---------------------------------------------------------------------------
# 산출물 ③ 연결성분 분리표
# ---------------------------------------------------------------------------
def write_cc_split(cell_cc, model_names, out_path):
    header = ["condition", "case", "model", "group", "area_bin", "n_components",
              "detected", "well_delineated", "found_but_poor", "recall", "poor_ratio"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key in sorted(cell_cc):
            cond, case = key
            for n in model_names:
                for g in GROUP_ORDER:
                    for b, lab in enumerate(AREA_LABELS):
                        ncomp, det, well, poor = (int(x) for x in cell_cc[key][n][g][b])
                        if ncomp == 0:
                            continue   # 성분이 없는 (셀,묶음,구간) 은 행을 만들지 않는다.
                        recall = det / ncomp if ncomp else float("nan")
                        poor_ratio = poor / det if det else float("nan")
                        w.writerow([cond, case, n, g, lab, ncomp, det, well, poor,
                                    _r(recall), _r(poor_ratio)])
    print(f"[③] 연결성분 분리표 -> {out_path.name}")


# ---------------------------------------------------------------------------
# 산출물 ④ README
# ---------------------------------------------------------------------------
def write_readme(out_dir, split, ref, baselines, detect_frac, delin_iou):
    txt = f"""# 셀 단위 실패 지도 — 산출물 설명 ({split})

`tools/baseline_failure/cell_map.py` 가 저장된 분할 예측 PNG 만 읽어(GPU 불필요) 만든
세 벌의 원자료다. 기준 모델(ref) = `{ref}`, 기준선(baselines) = {baselines}.

## 집계 축 — "셀 안에서 누적한 전역 혼동행렬"

이미지를 (조건, 케이스) **셀**로 나눈다. 조건 = {common.KNOWN_CONDITIONS},
케이스 = {common.KNOWN_CASES} + '{common.CASE_NONE}'.

셀 안에서 **먼저 모든 이미지의 혼동행렬을 누적**한 뒤, 그 누적 행렬 하나에서 클래스별
IoU 를 계산한다. 이미지마다 IoU 를 낸 다음 평균하지 않는다. 두 방식은 값이 다르다 —
누적 혼동행렬의 IoU 는 셀 전체 픽셀을 한 번에 모아 계산하므로, 표본이 적은 이미지의
IoU 가 과대 가중되는 편향이 없다. **이 파일들의 IoU 를 failure_mining 의 이미지별
평균 mIoU(예: mining/d31_condition_case_matrix.csv)와 섞어 인용하면 안 된다.**

## ① cell_class_iou_{split}.csv — 셀 × 클래스 IoU

- 열: condition, case, n_images, class_id, class_name, gt_pixels,
  모델마다 iou_<모델>, best_baseline_iou, best_baseline_name, delta_ref_minus_best.
- iou_<모델> = 그 셀의 누적 혼동행렬에서 계산한 그 클래스의 IoU.
- gt_pixels = 그 셀에서 그 클래스로 표시된 GT 유효 픽셀 수.
- delta_ref_minus_best = iou_<ref> − best_baseline_iou (양수면 ref 가 앞선다).
- 그 셀에서 GT 에 그 클래스가 한 화소도 없으면(gt_pixels=0) IoU·델타를 **빈 칸**으로
  둔다(0 으로 채우지 않는다).

## ② cell_summary_{split}.csv — 셀 × 클래스묶음 평균 IoU

- 열: condition, case, n_images, group, 모델마다 miou_<모델>,
  best_baseline_miou, delta_ref_minus_best.
- group = thin(얇은 객체 {sorted(THIN_IDS)}) / large(큰 영역 {sorted(LARGE_IDS)}) /
  rest(나머지) / all(전 클래스).
- 평균 = 셀 누적 혼동행렬의 클래스별 IoU 를, 그 셀에 **존재하는 클래스에 대해서만**
  평균(부재 클래스 제외).

## ③ cell_cc_split_{split}.csv — 연결성분 분리표

GT 의 클래스별 연결 성분마다(scipy.ndimage.label) 아래를 센다. 성분 마스크 M,
면적 A, 클래스 c 라 할 때, M 의 바운딩 박스 B 안에서 P = (pred == c) 로 두고
inter = |M ∩ P|, union = |M ∪ P|, comp_iou = inter / union 이다.

- detected          = inter >= {detect_frac} * A            ("찾았다")
- well_delineated   = detected 이고 comp_iou >= {delin_iou}  ("영역도 그렸다")
- found_but_poor    = detected 이고 comp_iou <  {delin_iou}  ("찾긴 했는데 영역을 못 그렸다")

- 열: condition, case, model, group, area_bin, n_components, detected,
  well_delineated, found_but_poor, recall(= detected/n_components),
  poor_ratio(= found_but_poor/detected, detected 가 0 이면 빈 칸).
- area_bin 경계(픽셀) = {AREA_LABELS} (failure_mining.py 와 동일 경계).
- 한 성분은 소속 묶음(thin/large/rest)과 all 에 동시에 집계된다.

## 주의

- 값을 추측해 채우지 않는다. 계산할 수 없는 칸은 비운다.
- ①②의 IoU 는 "셀 누적 전역 혼동행렬" 기반이므로 이미지별 평균 지표와 혼용 금지.
"""
    (out_dir / "README.md").write_text(txt, encoding="utf-8")
    print(f"[④] README -> README.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="GT trainID PNG 루트")
    ap.add_argument("--models", required=True, nargs="+", metavar="NAME=DIR",
                    help="모델별 예측 PNG 루트(name=경로). 여러 번.")
    ap.add_argument("--ref", default="ours", help="우리 모델로 볼 이름(기본 ours)")
    ap.add_argument("--baselines", nargs="*", default=None,
                    help="기준선으로 볼 이름 목록(공백 구분). 기본은 ref 를 뺀 전부.")
    ap.add_argument("--out", required=True, help="출력 디렉터리")
    ap.add_argument("--split", default="test")
    ap.add_argument("--detect_frac", type=float, default=0.5,
                    help="성분을 '찾았다' 고 볼 겹침 비율(inter >= frac*A)")
    ap.add_argument("--delin_iou", type=float, default=0.5,
                    help="'영역을 그렸다' 고 볼 성분 단위 IoU 문턱")
    ap.add_argument("--limit", type=int, default=0, help="디버그용 앞 N 장만")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dirs = _parse_kv(args.models, "--models")
    # failure_mining 과 같은 규약: 경로 아래 pred/ 가 있으면 그쪽을 쓴다.
    models_dirs = {k: (Path(v) / "pred" if (Path(v) / "pred").is_dir() else Path(v))
                   for k, v in models_dirs.items()}
    model_names = list(models_dirs.keys())
    ref = args.ref
    if ref not in model_names:
        raise SystemExit(f"--ref '{ref}' 가 --models 에 없다: {model_names}")
    baselines = args.baselines if args.baselines else [n for n in model_names if n != ref]
    for b in baselines:
        if b not in model_names:
            raise SystemExit(f"--baselines '{b}' 가 --models 에 없다: {model_names}")

    # GT·예측 디렉터리를 상대 경로 image_id 로 인덱싱(중첩 보존). 평탄 덤프면 명확히 실패.
    gt_index = common.index_label_pngs(args.gt, require_nested=True)
    model_index = {n: common.index_label_pngs(d, require_nested=True)
                   for n, d in models_dirs.items()}

    cell_cm, cell_n, cell_cc = scan(
        gt_index, model_index, model_names,
        args.detect_frac, args.delin_iou, args.limit)

    write_cell_class_iou(cell_cm, cell_n, model_names, ref, baselines,
                         out_dir / f"cell_class_iou_{args.split}.csv")
    write_cell_summary(cell_cm, cell_n, model_names, ref, baselines,
                       out_dir / f"cell_summary_{args.split}.csv")
    write_cc_split(cell_cc, model_names, out_dir / f"cell_cc_split_{args.split}.csv")
    write_readme(out_dir, args.split, ref, baselines, args.detect_frac, args.delin_iou)

    print(f"[cell_map] 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
