#!/usr/bin/env python3
"""
MULTIAQUA Night Augmentation Visualizer

Config 파일의 NIGHT_AUG 설정을 읽어 원본과 augmented 이미지를 비교.
모달리티 3개(RGB, LiDAR, Thermal)를 한 화면에서 비교.

Usage:
    python visualize_night_aug.py --cfg configs/archive/levine-multiaqua_rgbtl_P13_hardaug4.yaml --split train
    python visualize_night_aug.py --cfg configs/multiaqua/levine-multiaqua_rgbtl_P9_hardaug4.yaml  --split val --force

Keyboard:
    n / → / d  : 다음 이미지
    p / ← / a  : 이전 이미지
    r          : 같은 이미지에 재적용 (새 random seed)
    Space      : Force 모드 토글 (p=1.0으로 강제 적용)
    s          : 현재 화면 PNG로 저장
    q / ESC    : 종료
"""

import sys
import os
import copy
import random
import argparse
from pathlib import Path

# Project root를 Python path에 추가
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Config & Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_stems(root: Path, split: str) -> list:
    txt_path = root / f"{split}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Split file not found: {txt_path}")
    with open(txt_path) as f:
        stems = [line.strip() for line in f if line.strip()]
    return stems


def _load_modal(path: Path, H: int, W: int) -> torch.Tensor:
    """이미지 파일을 로드해 (3, H, W) uint8 텐서로 반환."""
    if not path.exists():
        print(f"  [WARN] missing: {path}")
        return torch.zeros(3, H, W, dtype=torch.uint8)
    img = tvio.read_image(str(path))
    C = img.shape[0]
    if C == 1:
        img = img.repeat(3, 1, 1)
    elif C == 4:
        img = img[:3]
    # 크기 불일치 시 보정
    if img.shape[1] != H or img.shape[2] != W:
        img = TF.resize(img, [H, W], TF.InterpolationMode.BILINEAR, antialias=True)
    return img


def load_raw_sample(root: Path, stem: str, modals: list) -> dict:
    """Raw uint8 텐서 로드. mask 없이 img/lidar/thermal만."""
    sample = {}

    # multiaqua.py와 동일: root / "MULTIAQUA_night" / "data" / ...
    data_root = root / "MULTIAQUA_night"

    rgb_path = data_root / "data" / "zed" / f"{stem}.png"
    img = tvio.read_image(str(rgb_path))[:3]  # (3, H, W) uint8
    sample['img'] = img
    H, W = img.shape[1:]

    if 'lidar' in modals:
        sample['lidar'] = _load_modal(
            data_root / "data" / "lidar_processed" / f"{stem}_lidar.png", H, W)

    if 'thermal' in modals:
        sample['thermal'] = _load_modal(
            data_root / "data" / "thermal_processed" / f"{stem}_thermal.png", H, W)

    return sample


def resize_sample_to_square(sample: dict, size: int) -> dict:
    """
    MULTIAQUA 전용: 가로가 긴 이미지를 aspect-ratio 유지 + 패딩으로 size x size로.
    ResizeWidthPadToSquare 로직과 동일하나 mask 없이도 동작.
    """
    H, W = sample['img'].shape[1:]
    if W >= H:
        scale = size / W
        nH, nW = round(H * scale), size
        pad_top = (size - nH) // 2
        pad_bottom = size - nH - pad_top
        padding = [0, pad_top, 0, pad_bottom]
    else:
        scale = size / H
        nH, nW = size, round(W * scale)
        pad_left = (size - nW) // 2
        pad_right = size - nW - pad_left
        padding = [pad_left, 0, pad_right, 0]

    out = {}
    for k, v in sample.items():
        resized = TF.resize(v, [nH, nW], TF.InterpolationMode.BILINEAR, antialias=True)
        out[k] = TF.pad(resized, padding, fill=0)
    return out


def resize_display(sample: dict, size: int) -> dict:
    """단순 리사이즈 (display 패널용)."""
    out = {}
    for k, v in sample.items():
        out[k] = TF.resize(v, [size, size], TF.InterpolationMode.BILINEAR, antialias=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def build_night_transforms(night_cfg: dict, force_p: bool = False) -> list:
    """
    Night aug 전용 transform 리스트를 반환.
    force_p=True이면 NightSim만 p=1.0으로 강제 적용.
    CRM/ZeroOut은 force 여부 무관하게 config 확률 유지
    (ZeroOut p=1.0이면 RGB가 완전 검정이 되어 시각화 무의미).
    """
    from semseg.augmentations_mm import (
        RandomRGBNightSimulation,
        RandomRGBComplementaryMasking,
        RandomRGBZeroOut,
    )

    if not night_cfg.get('ENABLE', False):
        return []

    dark_range = tuple(night_cfg['DARK_RANGE']) if 'DARK_RANGE' in night_cfg else None
    mod_range = tuple(night_cfg['MODERATE_RANGE']) if 'MODERATE_RANGE' in night_cfg else None

    nightsim_p = 1.0 if force_p else night_cfg.get('NIGHT_SIM_P', 0.45)

    transforms = [
        RandomRGBNightSimulation(
            p=nightsim_p,
            brightness_range=tuple(night_cfg.get('BRIGHTNESS_RANGE', [0.03, 0.45])),
            contrast_range=tuple(night_cfg.get('CONTRAST_RANGE', [0.3, 0.7])),
            gamma_range=tuple(night_cfg.get('GAMMA_RANGE', [0.4, 0.8])),
            noise_std=night_cfg.get('NOISE_STD', 0.02),
            brightness_sampling=night_cfg.get('BRIGHTNESS_SAMPLING', 'dark_biased'),
            dark_biased_ratio=night_cfg.get('DARK_BIASED_RATIO', 0.7),
            dark_range=dark_range,
            moderate_range=mod_range,
        )
    ]

    crm_p = night_cfg.get('CRM_P', 0)
    if crm_p > 0:
        transforms.append(RandomRGBComplementaryMasking(
            p=crm_p,
            mask_ratio_range=tuple(night_cfg.get('CRM_MASK_RATIO', [0.2, 0.5])),
        ))

    zero_p = night_cfg.get('ZERO_P', 0)
    if zero_p > 0:
        transforms.append(RandomRGBZeroOut(p=zero_p))

    return transforms


def apply_aug(sample: dict, transforms: list, seed: int) -> tuple:
    """
    transforms를 sample 복사본에 적용.
    Returns: (augmented_sample, applied_info_str)
    """
    s = copy.deepcopy(sample)
    orig_img = sample['img'].clone()

    random.seed(seed)
    torch.manual_seed(seed)

    for t in transforms:
        s = t(s)

    # 무슨 aug가 적용됐는지 휴리스틱 판단
    diff = (orig_img.float() - s['img'].float()).abs()
    mean_diff = diff.mean().item()

    if mean_diff < 0.3:
        info = "no aug (dice miss)"
    elif s['img'].max() < 3:
        info = f"ZERO-OUT (RGB blacked out)"
    else:
        # CRM 감지: 큰 rectangular zero 영역 존재?
        rgb_aug = s['img'].float()
        zero_ratio = (rgb_aug < 2).float().mean().item()
        if zero_ratio > 0.05:
            info = f"Night + CRM mask  diff={mean_diff:.1f}  zero_ratio={zero_ratio:.2f}"
        else:
            bright_mean = s['img'].float().mean().item()
            info = f"Night sim  diff={mean_diff:.1f}  aug_brightness={bright_mean:.1f}"

    return s, info


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────────────

MODAL_LABEL = {'img': 'RGB', 'lidar': 'LiDAR', 'thermal': 'Thermal'}
MODAL_COLOR_BGR = {
    'img':     (80,  220, 80),   # green
    'lidar':   (255, 160, 80),   # orange
    'thermal': (255, 200, 0),    # cyan-ish
}


def to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) uint8 → (H, W, 3) BGR numpy."""
    arr = tensor.permute(1, 2, 0).numpy()  # RGB
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def put_text(img: np.ndarray, text: str, pos: tuple,
             color=(220, 220, 220), scale=0.5, bold=False):
    """외곽선 있는 텍스트."""
    th = 2 if bold else 1
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)


def draw_diff_heatmap(orig: torch.Tensor, aug: torch.Tensor, size: int) -> np.ndarray:
    """RGB 차이를 히트맵으로 시각화."""
    diff = (orig.float() - aug.float()).abs().mean(dim=0)  # (H, W)
    diff_np = diff.numpy()
    diff_np = (diff_np / (diff_np.max() + 1e-6) * 255).astype(np.uint8)
    diff_np = cv2.resize(diff_np, (size, size), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(diff_np, cv2.COLORMAP_INFERNO)
    return heatmap


def build_frame(
    orig: dict, aug: dict,
    stem: str, idx: int, total: int,
    modals: list, aug_info: str,
    night_cfg: dict, panel: int,
    force_mode: bool,
    show_diff: bool,
    cfg_name: str,
) -> np.ndarray:
    """
    Layout:
        Header: 파일명, 상태
        Grid (2 rows × n cols):
            Row 0 "ORIGINAL"  : [modal_0] [modal_1] [modal_2]
            Row 1 "AUGMENTED" : [modal_0] [modal_1] [modal_2]
        (Optional) Row 2 "DIFF" : [diff_heatmap] [-] [-]
        Footer: aug params + key hints
    """
    n = len(modals)
    HEADER_H = 60
    ROW_LBL_W = 88
    SEP = 3          # separator px
    FOOTER_H = 75
    n_rows = 3 if show_diff else 2

    W = ROW_LBL_W + n * panel + (n - 1) * SEP
    H = HEADER_H + n_rows * (panel + SEP) + FOOTER_H

    canvas = np.full((H, W, 3), 22, dtype=np.uint8)

    # ── Header ────────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (W, HEADER_H), (42, 42, 42), -1)
    put_text(canvas, f"[{idx+1}/{total}]  {stem}", (8, 20), (210, 210, 210), 0.48)
    put_text(canvas, f"config: {cfg_name}", (8, 38), (130, 130, 130), 0.38)

    mode_label = "FORCE ON (Space:off)" if force_mode else "RANDOM  (Space:force)"
    mode_color = (0, 220, 220) if force_mode else (200, 190, 40)
    put_text(canvas, mode_label, (ROW_LBL_W, 38), mode_color, 0.42)

    aug_color = (80, 200, 255) if "no aug" not in aug_info else (120, 120, 120)
    put_text(canvas, f"aug: {aug_info}", (ROW_LBL_W, 55), aug_color, 0.42)

    # Column headers (modal names)
    for ci, modal in enumerate(modals):
        x = ROW_LBL_W + ci * (panel + SEP) + panel // 2 - 28
        put_text(canvas, MODAL_LABEL.get(modal, modal),
                 (x, HEADER_H - 6), MODAL_COLOR_BGR.get(modal, (200, 200, 200)),
                 0.55, bold=True)

    # ── Grid rows ─────────────────────────────────────────────────────────────
    row_specs = [("ORIGINAL", orig), ("AUGMENTED", aug)]
    if show_diff:
        row_specs.append(("DIFF", None))

    row_label_colors = [(150, 255, 150), (255, 150, 150), (150, 200, 255)]

    for ri, (row_label, source) in enumerate(row_specs):
        y0 = HEADER_H + ri * (panel + SEP)

        # Row label block
        cv2.rectangle(canvas, (0, y0), (ROW_LBL_W - 2, y0 + panel), (48, 48, 48), -1)
        # Vertical center text
        text_y = y0 + panel // 2 + 6
        put_text(canvas, row_label, (4, text_y), row_label_colors[ri], 0.43, bold=True)

        # Modal panels
        for ci, modal in enumerate(modals):
            x0 = ROW_LBL_W + ci * (panel + SEP)

            if row_label == "DIFF":
                if modal == 'img':
                    panel_bgr = draw_diff_heatmap(orig['img'], aug['img'], panel)
                else:
                    # LiDAR/Thermal: unchanged → gray panel with text
                    panel_bgr = np.full((panel, panel, 3), 35, dtype=np.uint8)
                    put_text(panel_bgr, "unchanged", (panel // 2 - 45, panel // 2),
                             (100, 100, 100), 0.5)
            else:
                panel_bgr = to_bgr(source[modal])
                # Augmented row: LiDAR/Thermal에 "unchanged" 오버레이
                if row_label == "AUGMENTED" and modal != 'img':
                    panel_bgr = panel_bgr.copy()
                    cv2.rectangle(panel_bgr, (0, 0), (panel, 22), (0, 0, 0), -1)
                    put_text(panel_bgr, "unchanged", (4, 16), (110, 110, 110), 0.4)

            canvas[y0:y0 + panel, x0:x0 + panel] = panel_bgr

            # Column separator
            if ci < n - 1:
                canvas[y0:y0 + panel, x0 + panel:x0 + panel + SEP] = (55, 55, 55)

        # Row separator
        sep_y = y0 + panel
        canvas[sep_y:sep_y + SEP, ROW_LBL_W:] = (65, 65, 65)

    # ── Footer ────────────────────────────────────────────────────────────────
    fy = HEADER_H + n_rows * (panel + SEP) + 8

    if night_cfg.get('ENABLE', False):
        cfg_line = (
            f"NightSim p={night_cfg.get('NIGHT_SIM_P')}  "
            f"Brt={night_cfg.get('BRIGHTNESS_RANGE')}  "
            f"sampling={night_cfg.get('BRIGHTNESS_SAMPLING')}  "
            f"CRM p={night_cfg.get('CRM_P', 0)}  "
            f"Zero p={night_cfg.get('ZERO_P', 0)}"
        )
        put_text(canvas, cfg_line, (8, fy + 14), (120, 170, 120), 0.36)
    else:
        put_text(canvas, "NIGHT_AUG: disabled in this config", (8, fy + 14), (100, 100, 200), 0.4)

    keys = "n/→:next   p/←:prev   r:re-apply   Space:force-toggle   d:diff-heatmap   s:save   q/ESC:quit"
    put_text(canvas, keys, (8, fy + 38), (120, 120, 175), 0.37)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MULTIAQUA Night Augmentation Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--cfg',   required=True,
                        help='Config YAML (e.g. configs/archive/levine-multiaqua_rgbtl_P13_hardaug4.yaml)')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--panel', type=int, default=380,
                        help='Each modal panel size in pixels (default: 380)')
    parser.add_argument('--force', action='store_true',
                        help='Start with force-apply mode (aug p=1.0)')
    parser.add_argument('--diff',  action='store_true',
                        help='Start with diff heatmap row shown')
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.cfg)
    dataset_cfg = cfg['DATASET']
    root       = Path(dataset_cfg['ROOT'])
    modals     = dataset_cfg.get('MODALS', ['img', 'lidar', 'thermal'])
    night_cfg  = dataset_cfg.get('NIGHT_AUG', {})
    cfg_name   = os.path.basename(args.cfg)

    print(f"Config  : {args.cfg}")
    print(f"Split   : {args.split}")
    print(f"Root    : {root}")
    print(f"Modals  : {modals}")
    print(f"NIGHT_AUG enabled: {night_cfg.get('ENABLE', False)}")

    if not night_cfg.get('ENABLE', False):
        print("[WARN] NIGHT_AUG.ENABLE is False — aug transforms will be empty.")

    # ── Load stems ───────────────────────────────────────────────────────────
    stems = get_stems(root, args.split)
    print(f"Stems   : {len(stems)}")

    # ── State ─────────────────────────────────────────────────────────────────
    idx        = 0
    seed       = random.randint(0, 99999)
    force_mode = args.force
    show_diff  = args.diff

    # ── OpenCV window ────────────────────────────────────────────────────────
    win_name = f"MULTIAQUA Night Aug — {args.split}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # ── Pre-compute working size (2x panel for hi-res aug, then downscale) ───
    working_size = args.panel * 2

    while True:
        stem = stems[idx]

        # Load raw uint8 sample
        raw = load_raw_sample(root, stem, modals)

        # Resize to square at working resolution (preserve aspect via padding)
        prepared = resize_sample_to_square(raw, working_size)

        # Build transforms based on current force_mode
        transforms = build_night_transforms(night_cfg, force_p=force_mode)

        # Apply augmentation
        aug_result, aug_info = apply_aug(prepared, transforms, seed)

        # Downscale both to display panel size
        orig_disp = resize_display(prepared,  args.panel)
        aug_disp  = resize_display(aug_result, args.panel)

        # Build and show frame
        frame = build_frame(
            orig_disp, aug_disp,
            stem, idx, len(stems),
            modals, aug_info, night_cfg,
            args.panel, force_mode, show_diff,
            cfg_name,
        )
        cv2.imshow(win_name, frame)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):           # q / ESC → quit
            break
        elif key in (ord('n'), 83):         # n / →   → next
            idx  = (idx + 1) % len(stems)
            seed = random.randint(0, 99999)
        elif key in (ord('p'), 81):         # p / ←   → prev
            idx  = (idx - 1) % len(stems)
            seed = random.randint(0, 99999)
        elif key == ord('r'):               # r → re-apply (new seed)
            seed = random.randint(0, 99999)
        elif key == ord(' '):               # Space → toggle force mode
            force_mode = not force_mode
        elif key == ord('d'):               # d → toggle diff heatmap row
            show_diff = not show_diff
        elif key == ord('s'):               # s → save PNG
            save_name = f"aug_viz_{stem.replace('/', '_')}_{seed}.png"
            cv2.imwrite(save_name, frame)
            print(f"Saved: {save_name}")

    cv2.destroyAllWindows()
    print("Visualizer closed.")


if __name__ == '__main__':
    main()
