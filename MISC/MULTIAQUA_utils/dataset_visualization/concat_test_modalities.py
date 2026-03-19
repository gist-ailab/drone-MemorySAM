#!/usr/bin/env python3
"""
MULTIAQUA Test 셋: RGB, Thermal, LiDAR, Segmentation 마스크를 concat 해서 저장.
동영상 프레임 제작 시 --frames-dir 로 번호 순 저장.

경로:
- test.txt: stem 목록
- data/zed: RGB {stem}.png
- data/thermal_processed: Thermal {stem}_thermal.png
- data/lidar_processed2: LiDAR {stem}_lidar.png
- annotations: 마스크 {stem}.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semseg.datasets.multiaqua import MULTIAQUA

from config import MULTIAQUA_ROOT, TEST_LIST, VAL_LIST, OUT_DIR, VAL_OUT_DIR, OUT_VIDEO_FRAMES

# Annotation: 0=Recording Boat, 1=Static, 2=Dynamic, 3=Water, 4=Sky (multiaqua.py 동일)
# 팔레트: 0=boat(회색), 1~4=MULTIAQUA._BASE_PALETTE[0~3]
_BASE = MULTIAQUA._BASE_PALETTE.numpy() if hasattr(MULTIAQUA._BASE_PALETTE, "numpy") else np.array(MULTIAQUA._BASE_PALETTE)
MASK_PALETTE = [[128, 128, 128]] + [_BASE[i].tolist() for i in range(len(_BASE))]


def load_stems(path: Path) -> list:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text().strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _to_uint8_display(arr: np.ndarray) -> np.ndarray:
    """그레이/16비트 등 -> 0-255 uint8 (시각화)."""
    if arr is None or arr.size == 0:
        return None
    if arr.ndim > 2:
        arr = arr.squeeze()
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    if arr.dtype == np.uint8:
        return arr
    lo, hi = np.percentile(arr, [1, 99])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
    else:
        arr = np.zeros_like(arr) if np.issubdtype(arr.dtype, np.floating) else arr
    return arr.astype(np.uint8)


def load_thermal_from_dirs(thermal_dir: Path, stem: str, normalize: bool = False) -> np.ndarray | None:
    """Thermal: {stem}_thermal.png. normalize=False면 원본 픽셀값 그대로 표시(정규화 없음)."""
    p = thermal_dir / f"{stem}_thermal.png"
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if normalize:
        u8 = _to_uint8_display(img)
        if u8 is None:
            return None
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    # 정규화 없음: 원본이 uint8이면 그대로, 16비트면 선형 스케일만 (0-65535 -> 0-255)
    if img.ndim > 2:
        img = img.squeeze()
    if img.dtype == np.uint8:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 16비트 등: min-max 선형 스케일 (percentile 아님)
    arr = np.nan_to_num(img.astype(np.float64), nan=0.0)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
    else:
        arr = np.zeros_like(arr)
    return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def load_lidar_from_dir(lidar_dir: Path, stem: str) -> np.ndarray | None:
    """LiDAR: lidar_processed(2)에서 시도: _lidar.png, _lidar_color.png, )_lidar_color.png(오타 버전)."""
    for name in (f"{stem}_lidar.png", f"{stem}_lidar_color.png", f"{stem})_lidar_color.png"):
        p = lidar_dir / name
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            u8 = _to_uint8_display(img)
            if u8 is not None:
                return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        else:
            # 이미 BGR 3채널 (e.g. _lidar_color.png)
            if img.shape[2] >= 3 and img.dtype == np.uint8:
                return img.copy()
            u8 = _to_uint8_display(img)
            if u8 is not None:
                return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    return None


def load_mask_from_dir(ann_dir: Path, stem: str) -> np.ndarray | None:
    """어노테이션 PNG 로드 (multiaqua와 동일: torchvision io 또는 PIL).
    파일 픽셀값: 0=boat, 1=Static, 2=Dynamic, 3=Water, 4=Sky, 255=ignore."""
    p = ann_dir / f"{stem}.png"
    if not p.exists():
        return None
    labels = None
    try:
        # 1) multiaqua와 동일: torchvision read_image 첫 채널 (단채널/그레이면 0,1,2,3,4,255)
        from torchvision import io as tv_io
        raw = tv_io.read_image(str(p))
        if raw.dim() >= 2:
            ch0 = raw[0, ...].numpy().astype(np.int32)
            uniq = np.unique(ch0)
            if np.all(np.isin(uniq, [0, 1, 2, 3, 4, 255])) or (uniq.size <= 6 and uniq.max() <= 255):
                labels = ch0
    except Exception:
        pass
    if labels is None:
        try:
            im = Image.open(str(p))
            arr = np.array(im)
            if arr.ndim == 2:
                labels = arr.astype(np.int32)
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                uniq = np.unique(arr[:, :, 0])
                if np.all(np.isin(uniq, [0, 1, 2, 3, 4, 255])):
                    labels = arr[:, :, 0].astype(np.int32)
                else:
                    labels = _rgb_mask_to_indices(arr, MASK_PALETTE)
            else:
                labels = arr.astype(np.int32).squeeze()
        except Exception:
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is not None:
                labels = raw[:, :, 0].astype(np.int32) if raw.ndim == 3 else raw.astype(np.int32)
    if labels is None:
        return None
    # 0=boat, 1~4=Static,Dynamic,Water,Sky, 255=ignore
    h, w = labels.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(MASK_PALETTE)):
        out[labels == i] = MASK_PALETTE[i]
    out[labels == 255] = [0, 0, 0]
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _rgb_mask_to_indices(rgb: np.ndarray, palette: list) -> np.ndarray:
    """(H,W,3) RGB 마스크를 팔레트와 매칭해 인덱스 (H,W)로 변환."""
    h, w = rgb.shape[:2]
    out = np.full((h, w), 255, dtype=np.int32)
    for i, c in enumerate(palette):
        if len(c) >= 3:
            r, g, b = int(c[0]), int(c[1]), int(c[2])
        else:
            continue
        match = (rgb[:, :, 0] == r) & (rgb[:, :, 1] == g) & (rgb[:, :, 2] == b)
        out[match] = i
    return out


def resize_to_size(img: np.ndarray, ref_h: int, ref_w: int, use_nearest: bool = False) -> np.ndarray:
    """이미지를 (ref_h, ref_w)로 리사이즈. use_nearest=True면 Mask 등 라벨 보존용."""
    if img is None or img.size == 0:
        return None
    inter = cv2.INTER_NEAREST if use_nearest else cv2.INTER_LINEAR
    return cv2.resize(img, (ref_w, ref_h), interpolation=inter)


def build_concat_panel(images: list, labels: list, ref_h: int, ref_w: int) -> np.ndarray:
    """원본 해상도(ref_h, ref_w)로 각 패널을 맞춰 가로 concat. Mask는 INTER_NEAREST."""
    panels = []
    for img, label in zip(images, labels):
        if img is None or img.size == 0:
            panel = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)
        else:
            panel = resize_to_size(img, ref_h, ref_w, use_nearest=(label == "Mask"))
            if panel is None:
                panel = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
                panel[:] = (40, 40, 40)
        cv2.putText(panel, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        panels.append(panel)
    return np.hstack(panels)


def main():
    ap = argparse.ArgumentParser(description="MULTIAQUA Test set concat (RGB, Thermal, LiDAR, Mask)")
    ap.add_argument("--root", type=str, default=None, help="MULTIAQUA 루트 (기본: config)")
    ap.add_argument("--split", type=str, default="test", choices=("test", "val"), help="test.txt 또는 val.txt 사용 (기본: test)")
    ap.add_argument("--test-list", type=str, default=None, help="stem 목록 txt 경로 (지정 시 --split 무시)")
    ap.add_argument("--out-dir", type=str, default=None, help="concat 이미지 저장 폴더")
    ap.add_argument("--frames-dir", type=str, default=None, help="영상용 프레임 폴더 (000000.png, ...)")
    ap.add_argument("--no-mask", action="store_true", help="마스크 패널 제외")
    ap.add_argument("--thermal-normalize", action="store_true", help="Thermal 픽셀값 percentile 정규화 적용 (기본: 원본/선형 스케일)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else MULTIAQUA_ROOT
    if args.test_list is not None:
        stem_list_path = Path(args.test_list)
        default_out = OUT_DIR
    else:
        stem_list_path = VAL_LIST if args.split == "val" else TEST_LIST
        default_out = VAL_OUT_DIR if args.split == "val" else OUT_DIR
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    # config 경로는 스크립트 기준이므로 --root만 data/annotations 상대로 반영
    data_root = root / "MULTIAQUA_night"
    ann_dir = data_root / "annotations"
    data_dir = data_root / "data"
    rgb_dir = data_dir / "zed"
    thermal_dir = data_dir / "thermal_processed"
    # LiDAR: lidar_processed2 우선, 없으면 lidar_processed
    lidar_dir = data_dir / "lidar_processed2"
    if not lidar_dir.exists():
        lidar_dir = data_dir / "lidar_processed"

    if not stem_list_path.exists():
        print("stem 목록이 없습니다:", stem_list_path)
        return
    stems = load_stems(stem_list_path)
    if not stems:
        print(f"{stem_list_path.name}에 항목이 없습니다.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    with_mask = not args.no_mask

    for idx, stem in enumerate(stems):
        images = []
        labels = []
        # RGB (원본 해상도 기준)
        rgb_path = rgb_dir / f"{stem}.png"
        rgb_img = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
        images.append(rgb_img)
        labels.append("RGB")
        # Thermal (기본: 정규화 없음)
        images.append(load_thermal_from_dirs(thermal_dir, stem, normalize=args.thermal_normalize))
        labels.append("Thermal")
        # LiDAR
        images.append(load_lidar_from_dir(lidar_dir, stem))
        labels.append("LiDAR")
        if with_mask:
            images.append(load_mask_from_dir(ann_dir, stem))
            labels.append("Mask")
        # 원본 해상도: RGB 크기 사용 (RGB 없으면 첫 번째 유효 이미지 크기)
        ref_h, ref_w = None, None
        for im in images:
            if im is not None and im.size > 0:
                ref_h, ref_w = im.shape[0], im.shape[1]
                break
        if ref_h is None or ref_w is None:
            continue
        canvas = build_concat_panel(images, labels, ref_h, ref_w)
        out_path = out_dir / f"{stem}_concat.png"
        cv2.imwrite(str(out_path), canvas)
        if frames_dir:
            cv2.imwrite(str(frames_dir / f"{idx:06d}.png"), canvas)
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  {idx+1}/{len(stems)} {stem} -> {out_path.name}")

    print("저장 경로:", out_dir)
    if frames_dir:
        print("영상용 프레임:", frames_dir, f"({len(stems)}장)")
    print("완료.")


if __name__ == "__main__":
    main()
