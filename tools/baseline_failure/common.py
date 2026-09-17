#!/usr/bin/env python3
"""
DGFusion·CAFuser·ReliaDINO 실패 사례 분석 공통 유틸리티.

설계서 = .claude_logs/experiments/analysis/2026-09-17-baseline-failure-analysis-plan.md.
이 모듈은 세 모델의 덤프 산출물(trainID PNG)을 같은 규약으로 읽고 채점하는
바닥 함수들을 모아 둔다. 다른 도구는 이 함수들을 import 해서 쓰고, 채점 규약을
중복 구현하지 않는다.

핵심 규약 (semseg/datasets/deliver.py · semseg/metrics.py 와 정합):
- trainID = 0~24 유효, 255 = ignore.
- 혼동행렬 hist[gt, pred] (행=정답, 열=예측) — Metrics.update 와 동일한 축.
- 전역 mIoU = 25개 전 클래스 IoU 의 산술평균이며, 데이터셋 전체에서 한 번도
  등장하지 않은 클래스(분모 0)는 0 으로 넣어 평균에 포함한다(Metrics.compute_iou 재현).
- 이미지별 IoU 는 그 이미지의 GT 에 존재하지 않는 클래스를 NaN 으로 두어
  평균(mIoU_img = nanmean)에서 제외한다.
"""
import os
import re
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 클래스 정의 — 단일 출처는 semseg/datasets/deliver.py 의 DELIVER.CLASSES/PALETTE.
# import 가 실패하는 순수 분석 환경을 위해 동일 사본을 fallback 으로 두되,
# import 가 성공하면 반드시 사본과 일치하는지 단언해 표류를 막는다.
# ---------------------------------------------------------------------------
_CLASSES_FALLBACK = [
    "Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road",
    "SideWalk", "Vegetation", "Cars", "Wall", "TrafficSign", "Sky", "Ground",
    "Bridge", "RailTrack", "GroundRail", "TrafficLight", "Static", "Dynamic",
    "Water", "Terrain", "TwoWheeler", "Bus", "Truck",
]
_PALETTE_FALLBACK = np.array([
    [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60], [153, 153, 153],
    [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142],
    [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81], [150, 100, 100],
    [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160],
    [170, 120, 50], [45, 60, 150], [145, 170, 100], [0, 0, 230], [0, 60, 100],
    [0, 0, 70],
], dtype=np.uint8)


def get_classes():
    """DELIVER 클래스 이름 목록(길이 25). deliver.py 를 우선 사용한다."""
    try:
        from semseg.datasets.deliver import DELIVER
        cls = list(DELIVER.CLASSES)
        assert cls == _CLASSES_FALLBACK, (
            "common.py 의 클래스 사본이 deliver.py 와 어긋난다 — 사본을 갱신하라.")
        return cls
    except Exception as e:  # noqa: BLE001 — 순수 분석 환경(torch 부재)에서도 동작해야 한다.
        warnings.warn(f"[common] deliver.py import 실패({e}) — 내장 클래스 사본 사용.")
        return list(_CLASSES_FALLBACK)


def get_palette():
    """DELIVER trainID→RGB 팔레트 (25, 3) uint8. deliver.py 를 우선 사용한다."""
    try:
        from semseg.datasets.deliver import DELIVER
        pal = DELIVER.PALETTE.cpu().numpy().astype(np.uint8)
        assert pal.shape == (25, 3)
        return pal
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"[common] deliver.py 팔레트 import 실패({e}) — 내장 사본 사용.")
        return _PALETTE_FALLBACK.copy()


CLASSES = get_classes()
N_CLASSES = len(CLASSES)
IGNORE_LABEL = 255

# 설계서 D2/D3 의 관심 클래스 묶음.
THIN_CLASS_NAMES = ["Pole", "TrafficLight", "Pedestrian", "Static"]   # 얇은 객체 4
LARGE_CLASS_NAMES = ["RailTrack", "Wall", "Water"]                    # 큰 영역 3
THIN_CLASS_IDS = [CLASSES.index(n) for n in THIN_CLASS_NAMES]
LARGE_CLASS_IDS = [CLASSES.index(n) for n in LARGE_CLASS_NAMES]

# DELIVER 경로 규약.
KNOWN_CONDITIONS = ["cloud", "fog", "night", "rain", "sun"]
KNOWN_CASES = ["motionblur", "overexposure", "underexposure",
               "lidarjitter", "eventlowres"]
CASE_NONE = "none"

# 스플릿별 기대 장수(설계서 D2 검증 기준: DELIVER val 2005 / test 1897).
# 덤프·변환기는 저장이 끝난 뒤 이 수와 다르면 멈춘다(평탄화 덮어쓰기·필터 누락 감지).
EXPECTED_COUNTS = {"val": 2005, "test": 1897}


# ---------------------------------------------------------------------------
# image_id ↔ 조건/케이스
# ---------------------------------------------------------------------------
def image_id_from_rel(path, dataset_root=None, path_sub=None):
    """원본 이미지 경로 → 데이터셋 루트 기준 **상대 경로**(확장자 제외, '/' 구분).

    이 상대 경로가 곧 image_id 다. 덤프·변환기는 이 값을 그대로 중첩 디렉터리로
    풀어 `<out>/<split>/pred/<image_id>.png` 에 저장한다. 그래서 DELIVER 처럼 조건·
    케이스 하위 폴더마다 같은 basename(`000050_rgb_front.png`)이 반복돼도 뒤 파일이
    앞 파일을 덮어쓰지 않는다(1차 덤프의 평탄화로 test 1270/1897 만 남던 결함을 차단).

    - dataset_root 를 주면 그 접두어를 잘라 상대 경로를 만든다(기준선 detectron2 의
      `file_name` 은 절대 경로일 수 있다).
    - 못 주면 DELIVER 표준 마커 `/img/` 이후를 상대 경로로 삼되 `img/` 접두어를
      포함한다(우리 덤프와 변환기가 같은 RGB 경로에서 같은 id 를 내야 join 이 성립).
    - path_sub=(old, new) 를 주면 상대 경로 문자열에 치환을 적용한다(RGB `img` 접두어
      를 예측 경로 규약에 맞게 바꾸는 인자화된 규칙).
    """
    p = str(path).replace("\\", "/")
    rel = None
    if dataset_root:
        dr = str(dataset_root).replace("\\", "/").rstrip("/")
        if p == dr:
            raise ValueError(f"경로가 dataset_root 와 같다(파일이 아님): {p}")
        if p.startswith(dr + "/"):
            rel = p[len(dr) + 1:]
    if rel is None:
        if "/img/" in p:
            rel = "img/" + p.split("/img/", 1)[1]      # img/<cond>/<split>/<scene>/<file>
        elif p.startswith("img/"):
            rel = p
        else:
            raise ValueError(
                f"데이터셋 루트를 특정할 수 없어 상대 경로를 만들 수 없다: {p}. "
                f"--dataset_root 를 주거나 경로에 '/img/' 마커가 있어야 한다.")
    if path_sub is not None:
        old, new = path_sub
        rel = rel.replace(old, new)
    rel = re.sub(r"\.[^./]+$", "", rel)                 # 확장자 1개 제거
    return rel


# 하위 호환 별칭 — 예전 이름으로 부르는 호출부도 새 상대 경로 규약을 얻는다.
def image_id_from_rgb_path(rgb_path):
    """image_id_from_rel 의 하위 호환 별칭(데이터셋 루트 기준 상대 경로 반환)."""
    return image_id_from_rel(rgb_path)


def parse_condition_case(image_id_or_path):
    """image_id(또는 경로 문자열)에서 (condition, case) 를 뽑는다.

    합성 스모크 이름(예: `night__scene1_lidarjitter__000123`)에서도 동작하도록
    토큰·부분문자열 매칭을 함께 쓴다. 조건이 없으면 'unknown', 센서 고장 케이스가
    없으면 'none' 을 돌려준다.
    """
    s = str(image_id_or_path).replace("\\", "/")

    # 조건: image_id 형태면 첫 토큰, 경로면 `/<cond>/` 패턴을 찾는다.
    condition = "unknown"
    head = s.split("__", 1)[0].split("/")[-1]
    if head in KNOWN_CONDITIONS:
        condition = head
    else:
        for c in KNOWN_CONDITIONS:
            if re.search(rf"(?:^|[/_]){re.escape(c)}(?:[/_]|$)", s):
                condition = c
                break

    # 케이스: scene 폴더명 접미사(_lidarjitter 등)를 부분문자열로 찾는다.
    case = CASE_NONE
    for k in KNOWN_CASES:
        if k in s:
            case = k
            break
    return condition, case


# ---------------------------------------------------------------------------
# 라벨 PNG 입출력
# ---------------------------------------------------------------------------
def load_label_png(path):
    """trainID PNG 를 (H, W) uint8 배열로 읽는다(값 0~24, 255=ignore)."""
    from PIL import Image
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def save_label_png(path, arr):
    """(H, W) trainID 배열을 uint8 PNG 로 저장한다.

    path 에 하위 폴더가 들어 있으면(중첩 image_id) 상위 디렉터리를 만들어 준다 —
    상대 경로 보존 저장의 바탕.
    """
    from PIL import Image
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    Image.fromarray(np.asarray(arr).astype(np.uint8)).save(str(path))


def load_gt_deliver(gt_png_path):
    """DELIVER semantic PNG(원본 1~25, 255)를 deliver.py 규약대로 trainID 로 읽는다.

    semseg/datasets/deliver.py 155~157행과 동일: `label[label==255]=0; label-=1`
    (uint8 언더플로로 0→255=ignore). 결과 = 0~24 유효 + 255 ignore.
    """
    from PIL import Image
    arr = np.array(Image.open(gt_png_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    lab = arr.astype(np.int64)
    lab[lab == 255] = 0
    lab = lab - 1
    lab[lab < 0] = 255
    return lab.astype(np.uint8)


# ---------------------------------------------------------------------------
# 재귀 인덱싱 · 장수 검증 (상대 경로 image_id 규약)
# ---------------------------------------------------------------------------
def index_label_pngs(root, require_nested=False):
    """디렉터리를 재귀 순회해 {image_id: Path} 를 만든다.

    image_id = root 기준 상대 경로(확장자 제외, '/' 구분). rglob 이므로 image_id 는
    항상 유일하고, basename 이 여러 폴더에서 반복돼도 상대 경로로 구분된다.

    require_nested=True 면 **평탄 덤프**(파일이 하위 폴더 없이 root 바로 아래에 있는
    경우)를 명확한 에러로 막는다. 1차 덤프의 평탄화 저장은 DELIVER 의 반복 basename
    을 덮어써 앞 파일을 잃고(예: test 1270/1897), 그 상태로는 image_id join 이 조용히
    어긋난다 — 상대 경로를 보존한 중첩 덤프만 허용한다(설계서 결함 1).
    """
    root = Path(root)
    files = sorted(root.rglob("*.png"))
    if not files:
        raise FileNotFoundError(f"라벨 PNG 가 없다: {root}")
    flat = [f for f in files if f.parent == root]
    if require_nested and flat:
        raise RuntimeError(
            f"평탄 파일명 감지({root}): {len(flat)}장이 하위 폴더 없이 root 바로 아래에 "
            f"있다(예: {[f.name for f in flat[:3]]}). DELIVER 는 조건/케이스 하위 폴더마다 "
            f"같은 basename 이 반복되므로, 상대 경로를 보존한 **중첩** 덤프여야 image_id "
            f"join 이 성립한다(평탄 덤프는 앞 파일이 덮여 사라진다). 상대 경로 보존 덤프로 "
            f"다시 만들라.")
    out = {}
    for f in files:
        image_id = f.relative_to(root).with_suffix("").as_posix()
        out[image_id] = f
    return out


def assert_expected_count(n_saved, split, extra=""):
    """스플릿 기대 장수와 다르면 AssertionError. 미등록 split 은 경고만 하고 통과."""
    exp = EXPECTED_COUNTS.get(split)
    if exp is None:
        warnings.warn(
            f"[common] '{split}' 기대 장수 미정 — 장수 검증 생략(저장 {n_saved}장).")
        return
    assert n_saved == exp, (
        f"장수 불일치: split='{split}' 에 {n_saved}장 저장(기대 {exp}). {extra} "
        f"평탄화 덮어쓰기·필터 누락·경로 규약 오류를 의심하라(상대 경로 보존 확인).")


def resize_nearest(arr, out_h, out_w):
    """정수 라벨/맵을 최근접 보간으로 (out_h, out_w) 로 바꾼다(PIL, torch 불필요)."""
    if arr.shape[0] == out_h and arr.shape[1] == out_w:
        return arr
    from PIL import Image
    return np.array(
        Image.fromarray(arr.astype(np.uint8)).resize((out_w, out_h), Image.NEAREST))


# ---------------------------------------------------------------------------
# 혼동행렬·IoU (Metrics 재현)
# ---------------------------------------------------------------------------
def confusion_matrix(pred, gt, n=N_CLASSES, ignore=IGNORE_LABEL):
    """hist[gt, pred] 혼동행렬(n, n). Metrics.update 와 동일한 유효 픽셀 규약."""
    pred = np.asarray(pred).astype(np.int64).ravel()
    gt = np.asarray(gt).astype(np.int64).ravel()
    keep = (gt != ignore) & (gt >= 0) & (gt < n) & (pred >= 0) & (pred < n)
    idx = gt[keep] * n + pred[keep]
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def global_miou_from_cm(cm):
    """Metrics.compute_iou 재현: 부재 클래스 IoU=0 포함, 25클래스 평균×100.

    반환 = (per_class_iou_percent[list, 부재=0], miou_percent).
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / denom
    iou = np.where(np.isnan(iou), 0.0, iou)
    miou = float(iou.mean())
    return (iou * 100).round(4).tolist(), round(miou * 100, 4)


def per_image_iou(cm):
    """이미지별 per-class IoU. GT 에 없는 클래스(행합 0)는 NaN(평균 제외)."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt_sum = cm.sum(1)
    pred_sum = cm.sum(0)
    denom = tp + (pred_sum - tp) + (gt_sum - tp)   # = tp + fp + fn
    iou = np.full(cm.shape[0], np.nan)
    present = gt_sum > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        iou[present] = tp[present] / np.maximum(denom[present], 1e-9)
    return iou


def pixel_acc_from_cm(cm):
    """전체 유효 픽셀 정확도 = 대각합 / 총합."""
    total = cm.sum()
    if total == 0:
        return float("nan")
    return float(np.diag(cm).sum() / total)


def nanmean(values):
    """NaN 을 무시한 평균. 전부 NaN 이면 NaN."""
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(arr))
