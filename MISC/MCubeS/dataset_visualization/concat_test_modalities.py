#!/usr/bin/env python3
"""
MCubeS Test 셋에 대해 모달리티별 이미지를 concat 해서 저장.
영상 제작을 위해 --frames-dir 로 프레임 번호 순 저장 가능.

Ref: https://github.com/kyotovision-public/multimodal-material-segmentation
데이터 구조: polL_color(RGB), polL_dolp(.npy), polL_aolp_sin/cos(.npy), NIR_warped
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from config import MCUBES_ROOT, TEST_LIST, VIS_MODALITIES, SEG_MASK_DIR, OUT_DIR, OUT_VIDEO_FRAMES

# 세그멘테이션 마스크용 클래스 컬러 (BGR). 0=검정, 1~N 구분.
MASK_COLORS = [
    (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 128, 255), (128, 255, 128), (255, 128, 128),
    (128, 255, 255), (255, 128, 255), (255, 255, 128), (64, 128, 255), (255, 128, 64),
]


def get_modality_dirs(root: Path) -> dict:
    """root 기준 모달리티 폴더 경로."""
    return {
        "RGB": root / "polL_color",
        "DoLP": root / "polL_dolp",
        "AoLP_sin": root / "polL_aolp_sin",
        "AoLP_cos": root / "polL_aolp_cos",
        "NIR": root / "NIR_warped",
    }


def load_stems(path: Path) -> list:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().strip().splitlines() if line.strip() and not line.strip().startswith("#")]


def npy_to_uint8(arr: np.ndarray, percentile: bool = True) -> np.ndarray:
    """1채널 .npy를 0-255 uint8로 (시각화용)."""
    if arr is None or arr.size == 0:
        return None
    if arr.ndim > 2:
        arr = arr.squeeze()
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    if percentile:
        lo, hi = np.percentile(arr, [1, 99])
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255)
        else:
            arr = np.zeros_like(arr)
    else:
        arr = np.clip(arr, 0, 1) * 255 if arr.max() <= 1 else np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def load_modality(modality_dirs: dict, modality: str, stem: str):
    """모달리티별 로드. RGB/NIR은 PNG, DoLP/AoLP는 .npy."""
    folder = modality_dirs.get(modality)
    if not folder or not folder.exists():
        return None
    if modality in ("RGB", "NIR"):
        path = folder / f"{stem}.png"
        if not path.exists():
            return None
        img = cv2.imread(str(path))
        return img
    # DoLP, AoLP_sin, AoLP_cos
    path = folder / f"{stem}.npy"
    if not path.exists():
        return None
    arr = np.load(path)
    u8 = npy_to_uint8(arr)
    if u8 is None:
        return None
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def get_seg_mask_dir(root: Path) -> Path:
    """세그멘테이션 마스크 폴더 (GT)."""
    return root / "GT"


def load_seg_mask(seg_dir: Path, stem: str) -> np.ndarray | None:
    """세그멘테이션 마스크 로드 후 클래스별 컬러 BGR 이미지로 변환."""
    if not seg_dir or not seg_dir.exists():
        return None
    for ext in (".png", ".npy"):
        path = seg_dir / f"{stem}{ext}"
        if not path.exists():
            continue
        if ext == ".png":
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                return None
            if raw.ndim == 3:
                # palette/grayscale stored as 3ch: use first channel as label index
                labels = np.asarray(raw[:, :, 0], dtype=np.int32)
            else:
                labels = np.asarray(raw, dtype=np.int32)
        else:
            labels = np.load(path)
            if labels.ndim > 2:
                labels = labels.squeeze()
            labels = np.asarray(labels, dtype=np.int32)
        h, w = labels.shape[:2]
        out = np.zeros((h, w, 3), dtype=np.uint8)
        uniq = np.unique(labels)
        for idx in uniq:
            if idx < 0:
                continue
            c = MASK_COLORS[idx % len(MASK_COLORS)]
            out[labels == idx] = c
        return out
    return None


def resize_to_height(img: np.ndarray, ref_h: int) -> np.ndarray:
    """ref_h 높이에 맞춰 비율 유지 리사이즈."""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    scale = ref_h / h
    nw = int(w * scale)
    return cv2.resize(img, (nw, ref_h), interpolation=cv2.INTER_AREA)


def build_concat_panel(images: list, labels: list, ref_h: int, panel_w: int) -> np.ndarray:
    """이미지들을 ref_h로 맞춰 가로 concat. 각 패널 너비 panel_w, 라벨 추가."""
    panels = []
    for img, label in zip(images, labels):
        if img is None or img.size == 0:
            panel = np.zeros((ref_h, panel_w, 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)
        else:
            resized = resize_to_height(img, ref_h)
            if resized.shape[1] > panel_w:
                resized = cv2.resize(resized, (panel_w, ref_h), interpolation=cv2.INTER_AREA)
            panel = np.zeros((ref_h, panel_w, 3), dtype=np.uint8)
            panel[:] = (30, 30, 30)
            x0 = (panel_w - resized.shape[1]) // 2
            panel[:, x0 : x0 + resized.shape[1]] = resized
        cv2.putText(panel, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        panels.append(panel)
    return np.hstack(panels)


def main():
    ap = argparse.ArgumentParser(description="MCubeS Test set modality concat and save")
    ap.add_argument("--root", type=str, default=None, help="MCubeS 루트 (기본: config.MCUBES_ROOT)")
    ap.add_argument("--test-list", type=str, default=None, help="test.txt 경로")
    ap.add_argument("--out-dir", type=str, default=None, help="concat 이미지 저장 폴더")
    ap.add_argument("--frames-dir", type=str, default=None, help="영상용 프레임 저장 폴더 (번호순 000000.png, ...)")
    ap.add_argument("--ref-h", type=int, default=360, help="패널 높이")
    ap.add_argument("--panel-w", type=int, default=400, help="패널당 너비")
    ap.add_argument("--modalities", type=str, nargs="+", default=VIS_MODALITIES, help="모달리티 순서")
    ap.add_argument("--no-mask", action="store_true", help="세그멘테이션 마스크(GT) 패널 제외")
    args = ap.parse_args()

    root = Path(args.root) if args.root else MCUBES_ROOT
    test_list = Path(args.test_list) if args.test_list else TEST_LIST
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    modalities = args.modalities

    if not root.exists():
        print("MCubeS 루트가 없습니다:", root)
        return
    if not test_list.exists():
        print("test 목록이 없습니다:", test_list)
        return

    stems = load_stems(test_list)
    if not stems:
        print("test.txt에 항목이 없습니다.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    ref_h = args.ref_h
    panel_w = args.panel_w

    modality_dirs = get_modality_dirs(root)
    with_mask = not args.no_mask
    seg_dir = get_seg_mask_dir(root) if with_mask else None

    for idx, stem in enumerate(stems):
        images = []
        labels = []
        for mod in modalities:
            img = load_modality(modality_dirs, mod, stem)
            images.append(img)
            labels.append(mod)
        if with_mask and seg_dir:
            mask_img = load_seg_mask(seg_dir, stem)
            images.append(mask_img)
            labels.append("GT")
        canvas = build_concat_panel(images, labels, ref_h, panel_w)
        out_path = out_dir / f"{stem}_concat.png"
        cv2.imwrite(str(out_path), canvas)
        if frames_dir:
            frame_path = frames_dir / f"{idx:06d}.png"
            cv2.imwrite(str(frame_path), canvas)
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  {idx+1}/{len(stems)} {stem} -> {out_path.name}")

    print("저장 경로:", out_dir)
    if frames_dir:
        print("영상용 프레임:", frames_dir, f"({len(stems)}장)")
    print("완료.")


if __name__ == "__main__":
    main()
