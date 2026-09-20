#!/usr/bin/env python3
"""
원거리 손해 원인 분해 — 저장된 분할 예측 PNG·GT·원본 depth 만 읽어(GPU 불필요)
"원거리에서 손해가 나는 원인" 이 무엇인지 세 각도로 가른다.

설계 맥락 = .claude_logs/experiments/analysis/2026-09-17-baseline-failure-analysis-plan.md.
채점 규약(trainID·혼동행렬 축·IoU 정의·연결성분 검출 판정)은 tools/baseline_failure/
common.py 와 cell_map.py 의 것을 그대로 따르고, 여기서 다시 발명하지 않는다. 순회 구조
(GT·depth·연결성분을 이미지마다 한 번만 계산해 모든 모델이 공유) 역시 cell_map.py·
depth_bin_iou.py 를 그대로 재현한다(그 두 파일은 손대지 않는다).

이 도구가 답하려는 질문:
  ① 원거리 손해가 "객체가 작아서" 인가 "멀어서" 인가       -> size_by_distance
     (같은 area_bin 안에서 depth_bin 만 바꿔 가며 검출/영역묘사 지표를 읽는다)
  ② 원거리에서 경계를 못 그리는가                          -> boundary_quality
     (거리 구간별 boundary F-score; 누적 분자·분모로 계산)
  ③ 원거리에서 어디로 잘못 가는가                          -> far_misclass
     (거리 구간별 25x25 혼동행렬에서 GT 클래스별 오분류 상위 3개)

⚠️ 모든 거리 구간은 depth_bin_iou.py 가 남긴 경계(--edges_json)를 **그대로** 읽어 쓴다.
경계를 새로 추정하지 않는다 — 그래야 depth_bin_iou 의 표와 나란히 비교된다.

예:
  python tools/baseline_failure/far_range_diag.py \
    --gt .../preds/ours_gt/test/gt --depth_root /ailab_mat2/dataset/DELIVER \
    --models ours=.../preds/ours/test/pred dgf=.../preds/dgf/test/pred \
             caf=.../preds/caf/test/pred \
    --edges_json .../analysis/test/depth_bin_iou_test.json \
    --split test --out .../analysis/far_range/test
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
THIN_IDS = set(common.THIN_CLASS_IDS)

# 성분 면적 5구간(픽셀, 로그 스케일). cell_map.py 의 AREA_EDGES/AREA_LABELS 와 **같은
# 값**을 일부러 복제한다 — 두 도구의 면적 구간이 어긋나면 검출/recall 표를 나란히 비교할
# 수 없기 때문이다. import 하지 않고 값만 복제하되(cell_map.py 는 손대지 않는다), 경계를
# 바꾸려면 두 파일을 함께 고쳐야 한다. (cell_map.py 는 이 경계를 다시 failure_mining.py 와
# 맞춰 두었다.)
AREA_EDGES = [0, 100, 1000, 10000, 100000, float("inf")]
AREA_LABELS = ["<100", "100-1k", "1k-10k", "10k-100k", ">=100k"]

# 클래스 묶음. thin = common 의 얇은 객체 4개, 그 밖은 rest. 모든 성분은 "all" 에도 센다.
# cell_map.py 의 _class_group 정의와 같되 large 는 여기서 쓰지 않는다(원거리 손해의 관심은
# 얇은/작은 객체이므로 thin/rest/all 세 묶음만 둔다).
GROUP_ORDER = ["thin", "rest", "all"]

# 산출물 ①③ 의 초점 클래스 기본값 — 얇은 4개(Pole TrafficLight Pedestrian Static) +
# 원거리에서 표지·궤도·트럭·차와 헷갈리기 쉬운 4개. --classes 로 바꾼다.
DEFAULT_CLASSES = ["Pole", "TrafficLight", "Pedestrian", "Static",
                   "TrafficSign", "GroundRail", "Truck", "Cars"]


def _class_group(c):
    """클래스 인덱스 -> 소속 묶음(thin/rest). "all" 은 성분 집계 때 별도로 함께 더한다."""
    return "thin" if c in THIN_IDS else "rest"


def _bin_index(area):
    """면적(픽셀) -> area_bin 인덱스. cell_map._bin_index 와 동일 로직."""
    for b in range(len(AREA_LABELS)):
        if AREA_EDGES[b] <= area < AREA_EDGES[b + 1]:
            return b
    return len(AREA_LABELS) - 1


def _r(x):
    """CSV 셀 포맷: None·NaN 은 빈 칸으로, 그 밖은 소수 6자리. 값을 추측해 채우지 않는다.

    cell_map._r 와 같은 규약이다(계산할 수 없는 칸은 0 이 아니라 빈 칸)."""
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except TypeError:
        return ""
    return round(float(x), 6)


def _parse_kv(items, flag):
    """name=값 목록을 dict 로. cell_map._parse_kv 와 같은 규약."""
    out = {}
    for spec in items or []:
        if "=" not in spec:
            raise SystemExit(f"{flag} 는 name=값 형식이어야 함: {spec}")
        k, v = spec.split("=", 1)
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# depth 경로 규약 — depth_bin_iou.py 의 resolve_depth_dir/depth_path_for 를 복제한다.
# import 하지 않는 이유: depth_bin_iou.py 는 argparse main 을 가진 실행 스크립트이고
# 그 파일을 손대지 말라는 지시가 있어, 규칙만 이 파일에 그대로 옮긴다(그 파일을 읽어
# 같은 규칙을 구현). depth GT 는 DELIVER `depth/` 원본 단일 채널 uint8 PNG 다.
#
# GT 분할 PNG 경로에서 depth 를 찾는 치환 규약(과제 지시):
#   - 디렉터리:  semantic -> depth        (원본 DELIVER semantic 트리를 --gt 로 준 경우)
#   - 파일명:    _semantic -> _depth
# 그리고 depth_bin_iou.py 의 실제 규약(덤프 GT 는 image_id 가 `img/...` 로 시작):
#   - `img/` 접두어를 떼고,  _rgb -> _depth
# 두 경우를 모두 처리한다 — --gt 가 덤프 trainID GT(img/ 기반)든 원본 semantic 트리든
# 같은 depth PNG 를 가리키게 한다.
# ---------------------------------------------------------------------------
def resolve_depth_dir(depth_root):
    """--depth_root 는 DELIVER depth 디렉터리 자체 또는 데이터셋 루트를 받는다."""
    p = Path(depth_root)
    return p if p.name == "depth" else p / "depth"


def depth_path_for(depth_dir, image_id):
    """image_id(GT 루트 기준 상대 경로, 확장자 제외) -> 원본 depth PNG 경로.

    depth_bin_iou.depth_path_for 규약 + 과제의 semantic 치환 규약을 함께 적용한다.
    """
    rel = str(image_id)
    if rel.startswith("img/"):
        rel = rel[len("img/"):]
    elif rel.startswith("semantic/"):
        rel = rel[len("semantic/"):]
    # 파일명 접미사 치환(둘 다 시도해도 안전하다 — 존재하는 접미사만 바뀐다).
    rel = rel.replace("_rgb", "_depth").replace("_semantic", "_depth")
    # 중간 디렉터리에 semantic 이 남아 있으면(원본 semantic 트리) depth 로 바꾼다.
    rel = rel.replace("semantic/", "depth/")
    return Path(depth_dir) / f"{rel}.png"


def load_depth(path):
    """원본 depth PNG -> (H, W) float64(0 = 무효). depth_bin_iou.load_depth 와 동일."""
    from PIL import Image
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"depth 파일이 없다: {p} — --depth_root/치환 규약을 확인하라")
    arr = np.array(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float64)


def resize_nearest_float(arr, out_h, out_w):
    """float 맵(depth)용 최근접 리사이즈. depth_bin_iou.resize_nearest_float 와 동일."""
    if arr.shape[0] == out_h and arr.shape[1] == out_w:
        return arr
    from PIL import Image
    return np.array(Image.fromarray(arr.astype(np.float32))
                    .resize((out_w, out_h), Image.NEAREST), dtype=np.float64)


# ---------------------------------------------------------------------------
# 거리 구간 경계 읽기 — depth_bin_iou 의 JSON 에서 그대로 읽는다(새로 추정 금지).
# ---------------------------------------------------------------------------
def load_edges(edges_json):
    """--edges_json 에서 log_edges/depth_edges 를 읽어 (use_log, bin_edges, depth_edges) 반환.

    - log_edges 가 있으면 그것을 쓰고 픽셀 depth 는 로그를 취해 구간을 나눈다(depth_bin_iou
      가 `searchsorted(log_edges, log(depth), 'right')` 로 하는 것과 정확히 같다).
    - log_edges 가 없고 depth_edges 만 있으면 원본 depth 로 구간을 나눈다(단조 변환이라
      결과가 같다).
    - 둘 다 없으면 경계를 새로 추정하지 말고 명확한 에러로 멈춘다.
    """
    p = Path(edges_json)
    if not p.exists():
        raise SystemExit(f"--edges_json 파일이 없다: {p} (depth_bin_iou_<split>.json 을 주라)")
    obj = json.loads(p.read_text(encoding="utf-8"))
    log_edges = obj.get("log_edges")
    depth_edges = obj.get("depth_edges")
    if log_edges:
        log_edges = np.asarray(log_edges, dtype=np.float64)
        if depth_edges:
            depth_edges = np.asarray(depth_edges, dtype=np.float64)
        else:
            depth_edges = np.exp(log_edges)
        return True, log_edges, depth_edges
    if depth_edges:
        depth_edges = np.asarray(depth_edges, dtype=np.float64)
        return False, depth_edges, depth_edges
    raise SystemExit(
        f"--edges_json 에 log_edges/depth_edges 가 없다: {p} — 경계를 새로 추정하지 않는다.")


def bin_of_pixels(depth_valid, use_log, bin_edges):
    """유효 depth 값 배열 -> 구간 인덱스 배열. depth_bin_iou 와 동일 규약(side='right')."""
    vals = np.log(depth_valid) if use_log else depth_valid
    return np.searchsorted(bin_edges, vals, side="right")


def bin_of_scalar(depth_val, use_log, bin_edges):
    """스칼라 depth(성분 depth 중앙값) -> 구간 인덱스."""
    v = np.log(depth_val) if use_log else depth_val
    return int(np.searchsorted(bin_edges, v, side="right"))


def bin_lo_hi(b, nbins, depth_edges):
    """구간 인덱스 -> (depth_lo, depth_hi). depth_bin_iou CSV 의 lo/hi 규약과 동일."""
    lo = 0.0 if b == 0 else float(depth_edges[b - 1])
    hi = float("inf") if b == nbins - 1 else float(depth_edges[b])
    return lo, hi


# ---------------------------------------------------------------------------
# 경계(boundary) 유틸 — EDT 를 쓰지 않고 3x3 침식/팽창만 쓴다(과제 지시: EDT 는 느림).
# ---------------------------------------------------------------------------
_ST3 = np.ones((3, 3), dtype=bool)   # 3x3 구조 원소(8-이웃)


def boundary_of(mask):
    """마스크 껍질 = mask − (3x3 한 번 침식). 빈 마스크면 빈 마스크를 돌려준다."""
    from scipy import ndimage
    if not mask.any():
        return mask
    er = ndimage.binary_erosion(mask, structure=_ST3, border_value=0)
    return mask & ~er


def dilate(mask, tol):
    """3x3 구조로 tol 번 반복 팽창(iterations=tol). tol=0 이면 원본."""
    from scipy import ndimage
    if tol <= 0 or not mask.any():
        return mask
    return ndimage.binary_dilation(mask, structure=_ST3, iterations=tol, border_value=0)


# ---------------------------------------------------------------------------
# 한 번의 이미지 순회로 세 산출물의 원자료를 모두 누적한다.
# GT 로드·depth 로드·연결성분·구간 지도(bin_map)는 이미지마다 한 번만 계산해 모든 모델이
# 공유한다(성능). 이미지 하나를 다 처리하면 큰 배열을 붙들지 않는다(메모리 — 앞선 도구가
# 12GB 까지 올라간 적이 있어 crop 후 즉시 폐기).
# ---------------------------------------------------------------------------
def scan(gt_index, model_index, model_names, depth_dir, use_log, bin_edges, nbins,
         focus_ids, detect_frac, delin_iou, tolerances, limit):
    from scipy import ndimage

    common_ids = set(gt_index.keys())
    for n in model_names:
        common_ids &= set(model_index[n].keys())
    ids = sorted(common_ids)
    n_gt = len(gt_index)
    if len(ids) < n_gt:
        print(f"[far_range] 경고: GT {n_gt}장 중 모든 모델이 공유하는 {len(ids)}장만 집계"
              f"(누락 이미지가 있는 모델 존재).")
    if limit and limit > 0:
        ids = ids[:limit]
    print(f"[far_range] 집계 대상 {len(ids)}장 · 모델 {model_names} · "
          f"구간 {nbins}개 · 초점 클래스 {[CLASSES[c] for c in focus_ids]}")

    max_tol = max(tolerances) if tolerances else 0

    # ① size_by_distance: (group, area_bin, depth_bin, model) -> [n_comp, gt_pix, det, well, poor]
    agg1 = defaultdict(lambda: np.zeros(5, dtype=np.int64))
    # ② boundary_quality: (class, depth_bin, model, tol) -> [p_num, p_den, r_num, r_den]
    agg2 = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    # ②의 표본: (class, depth_bin) -> 그 구간에 그 클래스 GT 가 있었던 이미지 수
    n_images2 = defaultdict(int)
    # ③ far_misclass: (depth_bin, model) -> 25x25 혼동행렬(gt 축 × pred 축), 원 카운트
    conf = {(b, n): np.zeros((N, N), dtype=np.int64)
            for b in range(nbins) for n in model_names}

    for k_img, image_id in enumerate(ids):
        gt = common.load_label_png(gt_index[image_id])
        depth = load_depth(depth_path_for(depth_dir, image_id))
        if depth.shape != gt.shape:
            depth = resize_nearest_float(depth, gt.shape[0], gt.shape[1])

        # 픽셀별 구간 지도(무효 depth = -1). depth_bin_iou 와 같은 유효 픽셀 규약.
        valid_d = depth > 0
        bin_map = np.full(gt.shape, -1, dtype=np.int16)
        if valid_d.any():
            bin_map[valid_d] = bin_of_pixels(depth[valid_d], use_log, bin_edges).astype(np.int16)

        # --- GT 연결성분(모델 무관): 성분마다 면적·bbox·부분마스크·거리구간을 미리 구한다.
        #     성분의 거리 = 그 성분 화소의 depth 중앙값(유효 depth 만). 유효 depth 가 하나도
        #     없으면 구간을 정할 수 없어 ①에서 건너뛴다(계산 불가는 비운다).
        comps = []   # (class_c, rmin, rmax, cmin, cmax, sub_M, area, area_bin, depth_bin)
        for c in range(N):
            m = gt == c
            if not m.any():
                continue
            lab, nlab = ndimage.label(m)
            for kk in range(1, nlab + 1):
                cmask = lab == kk
                ys, xs = np.where(cmask)
                rmin, rmax = int(ys.min()), int(ys.max())
                cmin, cmax = int(xs.min()), int(xs.max())
                sub_M = cmask[rmin:rmax + 1, cmin:cmax + 1]
                area = int(cmask.sum())
                dvals = depth[rmin:rmax + 1, cmin:cmax + 1][sub_M]
                dvals = dvals[dvals > 0]
                if dvals.size == 0:
                    depth_bin = None
                else:
                    depth_bin = bin_of_scalar(float(np.median(dvals)), use_log, bin_edges)
                comps.append((c, rmin, rmax, cmin, cmax, sub_M, area,
                              _bin_index(area), depth_bin))

        # --- ② 준비: 초점 클래스마다, 그 클래스 GT 가 존재하는 구간의 GT 껍질을 미리 구한다.
        #     Bg 는 모델 무관이므로 여기서 한 번만 만든다. 팽창은 여기서 하지 않는다 —
        #     전체 이미지 팽창은 느리므로(과제 지시) 모델 루프에서 crop 을 뜬 뒤에 돌린다.
        gbits = {}   # (c, b) -> dict(Bg, bg_count)
        for c in focus_ids:
            mc = gt == c
            if not mc.any():
                continue
            for b in range(nbins):
                Mg = mc & (bin_map == b)
                if not Mg.any():
                    continue
                n_images2[(c, b)] += 1
                Bg = boundary_of(Mg)
                gbits[(c, b)] = {"Bg": Bg, "bg": int(Bg.sum())}

        # --- 모델 루프: 예측을 한 번만 로드해 ①③②에 모두 쓴다.
        for n in model_names:
            pred = common.load_label_png(model_index[n][image_id])
            if pred.shape != gt.shape:
                pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])

            # ① 성분별 검출/영역묘사 — cell_map.py 와 같은 정의.
            for (c, rmin, rmax, cmin, cmax, sub_M, area, abin, dbin) in comps:
                if dbin is None:
                    continue   # 거리 구간을 정할 수 없는 성분은 크기×거리 표에서 제외.
                sub_P = pred[rmin:rmax + 1, cmin:cmax + 1] == c
                inter = int((sub_M & sub_P).sum())
                union = int((sub_M | sub_P).sum())
                comp_iou = inter / union if union else 0.0
                detected = inter >= detect_frac * area
                well = detected and comp_iou >= delin_iou
                poor = detected and comp_iou < delin_iou
                for g in (_class_group(c), "all"):
                    arr = agg1[(g, abin, dbin, n)]
                    arr[0] += 1
                    arr[1] += area
                    arr[2] += int(detected)
                    arr[3] += int(well)
                    arr[4] += int(poor)

            # ③ 구간별 혼동행렬 — depth_bin_iou 와 같은 누적 규약(유효 픽셀만).
            valid = valid_d & (gt != common.IGNORE_LABEL)
            if valid.any():
                gv = gt[valid].astype(np.int64)
                pv = pred[valid].astype(np.int64)
                bv = bin_map[valid].astype(np.int64)
                keep = (gv >= 0) & (gv < N) & (pv >= 0) & (pv < N)
                gv, pv, bv = gv[keep], pv[keep], bv[keep]
                flat = np.bincount(bv * (N * N) + gv * N + pv,
                                   minlength=nbins * N * N).reshape(nbins, N, N)
                for b in range(nbins):
                    if flat[b].any():
                        conf[(b, n)] += flat[b]

            # ② 경계 품질 — 초점 클래스·구간마다 누적 분자·분모를 더한다.
            for (c, b), gd in gbits.items():
                Mp = (pred == c) & (bin_map == b)
                Bp = boundary_of(Mp)
                bp_count = int(Bp.sum())
                if bp_count == 0 and gd["bg"] == 0:
                    continue   # 양쪽 다 경계가 없으면 더할 것이 없다.
                # 성능: GT∪예측 껍질의 bbox 안(여유 max_tol+1)에서만 팽창을 돌린다.
                union_mask = gd["Bg"] | Bp
                if not union_mask.any():
                    continue
                ys, xs = np.where(union_mask)
                pad = max_tol + 1
                r0 = max(int(ys.min()) - pad, 0)
                r1 = min(int(ys.max()) + pad + 1, gt.shape[0])
                c0 = max(int(xs.min()) - pad, 0)
                c1 = min(int(xs.max()) + pad + 1, gt.shape[1])
                Bp_crop = Bp[r0:r1, c0:c1]
                Bg_crop = gd["Bg"][r0:r1, c0:c1]
                # crop 은 union bbox 를 max_tol+1 만큼 여유 두었으므로, tol(≤max_tol) 팽창은
                # crop 안에서 완결된다(경계 밖 참 화소가 없어 border_value=0 이 정확).
                for tol in tolerances:
                    dBg_crop = dilate(Bg_crop, tol)       # GT 껍질도 crop 안에서 팽창(전체 X)
                    dBp_crop = dilate(Bp_crop, tol)       # 예측 껍질은 모델마다 다르니 crop 후 팽창
                    arr = agg2[(c, b, n, tol)]
                    arr[0] += int((Bp_crop & dBg_crop).sum())   # p_num = |Bp ∩ dilate(Bg,tol)|
                    arr[1] += bp_count                           # p_den = |Bp|
                    arr[2] += int((Bg_crop & dBp_crop).sum())    # r_num = |Bg ∩ dilate(Bp,tol)|
                    arr[3] += gd["bg"]                           # r_den = |Bg|

        # 이미지 하나 끝 — 큰 배열을 붙들지 않는다(메모리).
        del gt, depth, bin_map, comps, gbits
        if (k_img + 1) % 200 == 0:
            print(f"[far_range] {k_img + 1}/{len(ids)} 장 처리")

    return agg1, agg2, n_images2, conf


# ---------------------------------------------------------------------------
# 산출물 ① size_by_distance
# ---------------------------------------------------------------------------
def write_size_by_distance(agg1, model_names, nbins, depth_edges, out_path):
    header = ["group", "area_bin", "depth_bin", "depth_lo", "depth_hi", "model",
              "n_components", "gt_pixels", "detected", "well_delineated",
              "found_but_poor", "recall", "poor_ratio"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for g in GROUP_ORDER:
            for abin, alab in enumerate(AREA_LABELS):
                for b in range(nbins):
                    lo, hi = bin_lo_hi(b, nbins, depth_edges)
                    for n in model_names:
                        key = (g, abin, b, n)
                        if key not in agg1:
                            continue   # 성분이 없는 칸은 행을 만들지 않는다.
                        ncomp, gtpix, det, well, poor = (int(x) for x in agg1[key])
                        if ncomp == 0:
                            continue
                        recall = det / ncomp if ncomp else None
                        poor_ratio = poor / det if det else None
                        w.writerow([g, alab, b, round(lo, 4),
                                    ("" if np.isinf(hi) else round(hi, 4)), n,
                                    ncomp, gtpix, det, well, poor,
                                    _r(recall), _r(poor_ratio)])
    print(f"[①] size_by_distance -> {out_path.name}")


# ---------------------------------------------------------------------------
# 산출물 ② boundary_quality
# ---------------------------------------------------------------------------
def write_boundary_quality(agg2, n_images2, model_names, nbins, depth_edges,
                           focus_ids, tolerances, out_path):
    header = ["class_id", "class_name", "depth_bin", "depth_lo", "depth_hi", "model",
              "tol", "n_gt_boundary_px", "n_pred_boundary_px",
              "precision", "recall", "f_score", "n_images"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for c in focus_ids:
            for b in range(nbins):
                if (c, b) not in n_images2:
                    continue   # 그 구간에 그 클래스 GT 가 한 장도 없으면 건너뛴다.
                lo, hi = bin_lo_hi(b, nbins, depth_edges)
                n_img = n_images2[(c, b)]
                for n in model_names:
                    for tol in tolerances:
                        key = (c, b, n, tol)
                        if key not in agg2:
                            continue
                        p_num, p_den, r_num, r_den = (int(x) for x in agg2[key])
                        precision = p_num / p_den if p_den else None
                        recall = r_num / r_den if r_den else None
                        if precision is None or recall is None or (precision + recall) == 0:
                            fscore = None
                        else:
                            fscore = 2 * precision * recall / (precision + recall)
                        w.writerow([c, CLASSES[c], b, round(lo, 4),
                                    ("" if np.isinf(hi) else round(hi, 4)), n, tol,
                                    r_den, p_den, _r(precision), _r(recall),
                                    _r(fscore), n_img])
    print(f"[②] boundary_quality -> {out_path.name}")


# ---------------------------------------------------------------------------
# 산출물 ③ far_misclass (+ 누적 혼동행렬 .npy)
# ---------------------------------------------------------------------------
def write_far_misclass(conf, model_names, nbins, depth_edges, focus_ids,
                       out_path, conf_dir):
    conf_dir.mkdir(parents=True, exist_ok=True)
    # 누적 혼동행렬 자체를 원 카운트로 저장(정규화하지 않는다).
    for n in model_names:
        for b in range(nbins):
            np.save(conf_dir / f"{n}__bin{b}.npy", conf[(b, n)])

    header = ["depth_bin", "depth_lo", "depth_hi", "model", "gt_class_id",
              "gt_class_name", "gt_pixels", "correct_pixels", "rank",
              "pred_class_name", "pred_pixels", "frac_of_gt"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for b in range(nbins):
            lo, hi = bin_lo_hi(b, nbins, depth_edges)
            hi_s = "" if np.isinf(hi) else round(hi, 4)
            for n in model_names:
                cm = conf[(b, n)]
                for c in focus_ids:
                    gt_pix = int(cm[c].sum())
                    if gt_pix == 0:
                        continue   # 그 구간에 그 GT 클래스가 없으면 오분류를 낼 수 없다.
                    correct = int(cm[c, c])
                    # 잘못 간 곳: 대각(정답)을 뺀 예측 클래스 중 픽셀수 상위 3개.
                    wrong = [(int(cm[c, j]), j) for j in range(N) if j != c and cm[c, j] > 0]
                    wrong.sort(key=lambda t: t[0], reverse=True)
                    for rank, (px, j) in enumerate(wrong[:3], start=1):
                        w.writerow([b, round(lo, 4), hi_s, n, c, CLASSES[c], gt_pix,
                                    correct, rank, CLASSES[j], px, _r(px / gt_pix)])
    print(f"[③] far_misclass -> {out_path.name} (혼동행렬 -> {conf_dir.name}/)")


# ---------------------------------------------------------------------------
# 산출물 ④ README
# ---------------------------------------------------------------------------
def write_readme(out_dir, split, edges_json, detect_frac, delin_iou, tolerances,
                 focus_ids):
    focus_names = [CLASSES[c] for c in focus_ids]
    txt = f"""# 원거리 손해 원인 분해 — 산출물 설명 ({split})

`tools/baseline_failure/far_range_diag.py` 가 저장된 분할 예측 PNG·GT·원본 depth 만
읽어(GPU 불필요) 만든 세 표다. 초점 클래스 = {focus_names}.

## 공통 규약

- 거리 구간 경계는 **직접 추정하지 않고** depth_bin_iou 가 남긴 JSON 을 그대로 읽어
  쓴다(`--edges_json {Path(edges_json).name}`). 그래서 depth_bin_iou 의 구간 표와 같은
  경계에서 나란히 비교된다. (다) — 이 파일의 depth_bin·depth_lo·depth_hi 는 depth_bin_iou
  의 것과 동일하다.
- 픽셀 depth 는 DELIVER `depth/` 원본 단일 채널 uint8 PNG 다. depth==0 또는 GT==255 인
  픽셀은 제외한다(depth_bin_iou 와 같은 유효 픽셀 규약).
- 채점·연결성분 검출 판정·혼동행렬 축(hist[gt,pred])은 common.py·cell_map.py 규약을
  그대로 따른다. 계산할 수 없는 칸은 0 이 아니라 **빈 칸**으로 둔다.

## ① size_by_distance_{split}.csv — 성분 크기 × 거리 2차원 표

GT 클래스별 연결 성분(scipy.ndimage.label)마다 면적 A 와 거리 구간을 정해, 묶음
(thin/rest/all) × area_bin × depth_bin × 모델로 검출 지표를 집계한다.

- (가) **성분의 거리 = 그 성분 화소들의 depth 중앙값**이 속하는 구간이다(유효 depth 만).
  유효 depth 가 하나도 없는 성분은 구간을 정할 수 없어 이 표에서 제외한다.
- 성분 단위 판정(cell_map.py 와 동일): 바운딩 박스 B 안에서 P=(pred==c), inter=|M∩P|,
  union=|M∪P|, comp_iou=inter/union. detected = inter >= {detect_frac}*A,
  well_delineated = detected 이고 comp_iou>={delin_iou}, found_but_poor = detected 이고
  comp_iou<{delin_iou}.
- area_bin 경계(픽셀) = {AREA_LABELS} (cell_map.py 와 동일 경계).
- 묶음: thin = 얇은 객체 {sorted(THIN_IDS)}, rest = 나머지, all = 전 클래스(모든 성분은
  소속 묶음과 all 에 동시에 집계).
- 열: group, area_bin, depth_bin, depth_lo, depth_hi, model, n_components, gt_pixels,
  detected, well_delineated, found_but_poor, recall(=detected/n_components),
  poor_ratio(=found_but_poor/detected, detected 가 0 이면 빈 칸).
- 읽는 법: **같은 area_bin 안에서 depth_bin 만 바꿔** recall·poor_ratio 가 어떻게 변하는지
  보면 "원거리 손해가 작아서인지(area_bin 이 낮은 쪽) 멀어서인지(같은 area_bin 인데 먼
  depth_bin 에서 나빠지는지)" 를 가를 수 있다.

## ② boundary_quality_{split}.csv — 경계 품질(boundary F-score)

초점 클래스 c·거리 구간 b 에서, 그 구간에 속한 화소만 남긴 GT 마스크 Mg 와 예측 마스크
Mp 를 만든다. 경계는 마스크에서 3x3 구조로 한 번 침식한 것을 뺀 껍질이다.
허용 화소 tol 마다 한 행이며, dilate 는 3x3 구조로 tol 번 반복(iterations=tol)한다
(거리 변환 EDT 는 쓰지 않는다 — 느림).

    precision = |Bp ∩ dilate(Bg, tol)| / |Bp|
    recall    = |Bg ∩ dilate(Bp, tol)| / |Bg|
    F         = 2PR / (P+R)

- (나) 집계는 **이미지별 F 의 평균이 아니다.** 네 수(|Bp∩dilate(Bg,tol)|, |Bp|,
  |Bg∩dilate(Bp,tol)|, |Bg|)를 이미지에 걸쳐 누적한 뒤 **마지막에 한 번** P·R·F 를 낸다.
- 성능: 클래스마다 GT∪예측 껍질의 바운딩 박스(여유 tol+1 화소) 안에서만 팽창을 돌린다.
  해당 구간에 그 클래스가 없으면 건너뛴다.
- 열: class_id, class_name, depth_bin, depth_lo, depth_hi, model, tol,
  n_gt_boundary_px(=|Bg| 누적), n_pred_boundary_px(=|Bp| 누적), precision, recall,
  f_score, n_images.
- 허용 화소 tol = {list(tolerances)}.

## ③ far_misclass_{split}.csv — 원거리 오분류 전이

거리 구간마다 모델마다 25x25 혼동행렬(GT 축 × 예측 축)을 원 카운트로 누적하고
(depth_bin_iou 와 같은 누적 규약), `confusion_by_bin/<model>__bin<b>.npy` 로도 저장한다
(정규화하지 않음). 초점 클래스(GT 축)마다 정답(대각)을 뺀 **잘못 간 곳 상위 3개**를 행으로
낸다.

- 열: depth_bin, depth_lo, depth_hi, model, gt_class_id, gt_class_name, gt_pixels,
  correct_pixels, rank(1~3), pred_class_name, pred_pixels, frac_of_gt(=pred_pixels/gt_pixels).

## (라) 표본 크기 열의 뜻

- ① `n_components` = 그 (묶음, area_bin, depth_bin) 에 든 GT 연결 성분의 개수.
  `gt_pixels` = 그 성분들의 총 화소 수.
- ② `n_images` = 그 (클래스, 거리 구간) 에 GT 화소가 한 개라도 있었던 이미지 수(모델·tol
  무관). `n_gt_boundary_px`/`n_pred_boundary_px` = 껍질 화소를 이미지에 걸쳐 누적한 수.
- ③ `gt_pixels` = 그 (거리 구간, GT 클래스) 의 유효 GT 화소 수(혼동행렬 행합).
  표본이 작은 칸의 비율(recall·poor_ratio·frac_of_gt)은 흔들리므로 표본 열과 함께 읽는다.

## 주의

- 값을 추측해 채우지 않는다 — 계산할 수 없는 칸은 비운다.
- ①②③ 모두 depth_bin_iou 의 경계에 매여 있으니, 다른 경계로 만든 표와 섞어 인용하지
  않는다.
"""
    (out_dir / "README.md").write_text(txt, encoding="utf-8")
    print(f"[④] README -> README.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="GT 분할 PNG(trainID) 루트")
    ap.add_argument("--depth_root", required=True,
                    help="DELIVER 루트 또는 depth 디렉터리(depth/ 와 같은 층)")
    ap.add_argument("--models", required=True, nargs="+", metavar="NAME=DIR",
                    help="모델별 예측 PNG 루트(name=경로). 여러 번.")
    ap.add_argument("--edges_json", required=True,
                    help="depth_bin_iou_<split>.json (거리 구간 경계). 없으면 멈춘다.")
    ap.add_argument("--out", required=True, help="출력 디렉터리")
    ap.add_argument("--split", default="test")
    ap.add_argument("--classes", nargs="*", default=None,
                    help=f"초점 클래스 이름들. 기본 = {DEFAULT_CLASSES}")
    ap.add_argument("--tolerances", nargs="*", type=int, default=[2, 4, 8],
                    help="경계 허용 화소(dilate 반복 횟수). 기본 2 4 8")
    ap.add_argument("--detect_frac", type=float, default=0.5,
                    help="성분을 '찾았다' 고 볼 겹침 비율(inter >= frac*A)")
    ap.add_argument("--delin_iou", type=float, default=0.5,
                    help="'영역을 그렸다' 고 볼 성분 단위 IoU 문턱")
    ap.add_argument("--limit", type=int, default=0, help="디버그용 앞 N 장만")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 초점 클래스 이름 -> 인덱스.
    class_names = args.classes if args.classes else DEFAULT_CLASSES
    focus_ids = []
    for nm in class_names:
        if nm not in CLASSES:
            raise SystemExit(f"--classes '{nm}' 는 DELIVER 클래스가 아니다: {CLASSES}")
        focus_ids.append(CLASSES.index(nm))

    tolerances = sorted(set(int(t) for t in args.tolerances if int(t) > 0))
    if not tolerances:
        raise SystemExit("--tolerances 에 1 이상 정수가 하나도 없다.")

    # 모델 경로 정규화 — cell_map 과 같은 규약(아래에 pred/ 있으면 그쪽).
    models_dirs = _parse_kv(args.models, "--models")
    models_dirs = {k: (Path(v) / "pred" if (Path(v) / "pred").is_dir() else Path(v))
                   for k, v in models_dirs.items()}
    model_names = list(models_dirs.keys())

    depth_dir = resolve_depth_dir(args.depth_root)
    if not depth_dir.is_dir():
        raise SystemExit(f"depth 디렉터리가 없다: {depth_dir} — --depth_root 를 확인하라")

    use_log, bin_edges, depth_edges = load_edges(args.edges_json)
    nbins = len(bin_edges) + 1
    print(f"[far_range] 거리 구간 {nbins}개 (경계 출처={Path(args.edges_json).name}, "
          f"use_log={use_log})")

    gt_index = common.index_label_pngs(args.gt, require_nested=True)
    model_index = {n: common.index_label_pngs(d, require_nested=True)
                   for n, d in models_dirs.items()}

    agg1, agg2, n_images2, conf = scan(
        gt_index, model_index, model_names, depth_dir, use_log, bin_edges, nbins,
        focus_ids, args.detect_frac, args.delin_iou, tolerances, args.limit)

    write_size_by_distance(agg1, model_names, nbins, depth_edges,
                           out_dir / f"size_by_distance_{args.split}.csv")
    write_boundary_quality(agg2, n_images2, model_names, nbins, depth_edges,
                           focus_ids, tolerances,
                           out_dir / f"boundary_quality_{args.split}.csv")
    write_far_misclass(conf, model_names, nbins, depth_edges, focus_ids,
                       out_dir / f"far_misclass_{args.split}.csv",
                       out_dir / "confusion_by_bin")
    write_readme(out_dir, args.split, args.edges_json, args.detect_frac,
                 args.delin_iou, tolerances, focus_ids)

    print(f"[far_range] 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
