#!/usr/bin/env python3
"""PhysAug Augmentation 시각화 뷰어.

주간 RGB 이미지에 PhysAug (Filter + Fourier) 적용 결과를
인터랙티브하게 확인. NightSim과의 결합 효과도 함께 표시.

사용법:
    python MISC/physaug_viewer.py
    python MISC/physaug_viewer.py --sigma 2.0 --mean-str 8.0 --decay 0.3
    python MISC/physaug_viewer.py --src-dir /path/to/images --src-prefix bl1

컨트롤:
    ← →        : 소스 이미지 전환
    F           : Filter on/off 토글
    W           : Fourier (wave) on/off 토글
    N           : NightSim 결합 on/off 토글
    R           : 파라미터 랜덤 리샘플링
    S           : 현재 결과를 save_dir에 저장
    Q / ESC     : 종료

슬라이더:
    sigma x100  : Filter noise 강도 (0~400 → 0.0~4.0)
    mean_str x10: Fourier wave 강도 (10~200 → 1.0~20.0, 높을수록 약함)
    decay x100  : Gaussian decay (0~100 → 0.0~1.0)
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as io

WINDOW = 'PhysAug Viewer'

# ─── PhysAug Core (semseg/augmentations_mm.py의 RandomPhysAug와 동일 로직) ──

def apply_filter(img, sigma, kernel_size=3):
    """Random convolution: identity kernel + Gaussian noise.

    Args:
        img: (C, H, W) float [0, 1]
        sigma: noise 강도
        kernel_size: conv 커널 크기
    Returns:
        (C, H, W) float [0, 1], per-channel min-max normalized.
    """
    if sigma < 0.01:
        return img.clone()

    C, H, W = img.shape
    pad = kernel_size // 2

    delta = torch.zeros(kernel_size, kernel_size)
    delta[kernel_size // 2, kernel_size // 2] = 1.0
    conv_weight = sigma * torch.randn(kernel_size, kernel_size) + delta
    conv_weight = conv_weight.unsqueeze(0).unsqueeze(0)

    filtered = torch.zeros_like(img)
    for c in range(C):
        inp = img[c].unsqueeze(0).unsqueeze(0)
        out = F.conv2d(inp, conv_weight, padding=pad)
        filtered[c] = out.squeeze()

    filtered = filtered.abs()

    for c in range(C):
        ch = filtered[c]
        mn, mx = ch.min(), ch.max()
        if mx - mn > 1e-6:
            filtered[c] = (ch - mn) / (mx - mn)
        else:
            filtered[c] = ch.clamp(0.0, 1.0)

    return filtered


def apply_fourier(img, groups=(1, 513), mean_str=8.0, decay=0.3,
                  granularity=256, f_cut=1, p_cut=1):
    """Planar wave perturbation + atmospheric light.

    Args:
        img: (C, H, W) float [0, 1]
    Returns:
        (C, H, W) float [0, 1]
    """
    C, H, W = img.shape

    # meshgrid
    _x = np.linspace(-H / 2, H / 2, H)
    _y = np.linspace(-W / 2, W / 2, W)
    mesh_x, mesh_y = np.meshgrid(_x, _y, indexing='ij')
    mesh_x = torch.tensor(mesh_x, dtype=torch.float32)
    mesh_y = torch.tensor(mesh_y, dtype=torch.float32)

    # 주파수/위상 후보
    freqs_pool = np.array([g / 1024.0 for g in range(groups[0], groups[1])], dtype=np.float32)
    phases_pool = -np.pi * np.linspace(0.0, 1.0, num=granularity)
    num_groups = len(freqs_pool)
    num_phases = len(phases_pool)

    # 샘플링
    f_idx = np.random.randint(0, num_groups, (1, C, f_cut, 1))
    p_idx = np.random.randint(0, num_phases, (1, C, f_cut, p_cut))
    freqs = torch.tensor(freqs_pool[f_idx], dtype=torch.float32)
    phases = torch.tensor(phases_pool[p_idx], dtype=torch.float32)

    # strength
    strengths = np.random.exponential(1.0 / mean_str, (1, C, f_cut, p_cut))
    strengths_t = torch.tensor(strengths, dtype=torch.float32)

    # planar wave 생성
    f = freqs.unsqueeze(-1).unsqueeze(-1)
    p = phases.unsqueeze(-1).unsqueeze(-1)
    eps_scale = 1024.0 / 32.0

    waves = torch.sin(
        2 * math.pi * f * (mesh_x * torch.cos(p) + mesh_y * torch.sin(p))
        - math.pi / 4
    )
    norm = torch.norm(waves, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    waves = waves / norm * eps_scale

    # einsum
    aug = torch.einsum('bcfp,bcfphw->bchw', strengths_t, waves)
    aug = aug / (f_cut * p_cut)

    if aug.shape[-2:] != (H, W):
        aug = F.interpolate(aug, size=(H, W), mode='bilinear', align_corners=False)

    aug_hwc = aug[0].permute(1, 2, 0)

    # gaussian decay
    if decay > 0:
        center_x = random.randint(0, max(0, H - 13))
        center_y = random.randint(0, max(0, W - 13))
        sigma_x, sigma_y = H / 6.0, W / 6.0
        x = torch.arange(H, dtype=torch.float32) - center_x
        y = torch.arange(W, dtype=torch.float32) - center_y
        X, Y = torch.meshgrid(x, y, indexing='ij')
        gaussian = torch.exp(-((X ** 2 / (2 * sigma_x ** 2)) + (Y ** 2 / (2 * sigma_y ** 2))))
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min() + 1e-6)
        decay_map = (1 - decay) + decay * gaussian
        aug_hwc = aug_hwc * decay_map.unsqueeze(-1)

    img_hwc = img.permute(1, 2, 0)
    result = torch.clamp(img_hwc + aug_hwc, 0.0, 1.0)

    # atmospheric light
    log_sample = random.uniform(-3, -1)
    L_inf = 10 ** log_sample
    dx = random.uniform(0, 10)
    L = L_inf * (1 - math.exp(-dx))
    result = torch.clamp(result + L / 255.0, 0.0, 1.0)

    return result.permute(2, 0, 1)


def apply_nightsim(img, brightness=0.1, contrast=0.5, gamma=0.6, noise_std=0.02):
    """NightSim 시뮬레이션 (간략 버전).

    Args:
        img: (C, H, W) float [0, 1]
    Returns:
        (C, H, W) float [0, 1]
    """
    img = img * brightness
    img = (img - img.mean()) * contrast + img.mean()
    img = torch.clamp(img, 1e-6, 1.0) ** gamma
    if noise_std > 0:
        img = img + torch.randn_like(img) * noise_std
    return torch.clamp(img, 0.0, 1.0)


# ─── Image I/O ─────────────────────────────────────────────────────────────

def load_image(path):
    """이미지 로드 → (C, H, W) float [0, 1]."""
    img = io.read_image(str(path))[:3, ...]
    return img.float() / 255.0


def tensor_to_bgr(t):
    """(C, H, W) float [0, 1] → (H, W, 3) uint8 BGR."""
    arr = (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def diff_map(a, b, scale=5.0):
    """두 텐서의 차이를 시각화 (amplified)."""
    diff = (a.float() - b.float()).abs().mean(dim=0)  # (H, W)
    diff = (diff * scale).clamp(0, 1)
    diff_rgb = diff.unsqueeze(0).repeat(3, 1, 1)
    return tensor_to_bgr(diff_rgb)


# ─── Viewer ────────────────────────────────────────────────────────────────

class PhysAugViewer:
    def __init__(self, src_paths, sigma=1.5, mean_str=8.0, decay=0.3,
                 nightsim_brightness=0.1, save_dir=None):
        self.src_paths = src_paths
        self.src_idx = 0
        self.sigma = sigma
        self.mean_str = mean_str
        self.decay = decay
        self.ns_brightness = nightsim_brightness
        self.save_dir = save_dir
        self.filter_on = True
        self.fourier_on = True
        self.nightsim_on = True

    def _load_src(self):
        return load_image(self.src_paths[self.src_idx])

    def _build_layout(self, original, filtered, fouriered, combined, ns_combined):
        """2행 4열 레이아웃 구성."""
        h, w = original.shape[1], original.shape[2]
        font_scale = max(0.4, h / 1200)
        thickness = max(1, int(h / 500))

        def put_text(bgr, text, color=(0, 255, 255)):
            cv2.putText(bgr, text, (10, int(30 * font_scale * 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            return bgr

        # Row 1: images
        orig_bgr = put_text(tensor_to_bgr(original),
                            f"Original [{self.src_idx+1}/{len(self.src_paths)}]")
        filt_bgr = put_text(tensor_to_bgr(filtered),
                            f"Filter (sigma={self.sigma:.2f})")
        four_bgr = put_text(tensor_to_bgr(fouriered),
                            f"Fourier (str={self.mean_str:.1f}, decay={self.decay:.2f})")
        comb_bgr = put_text(tensor_to_bgr(combined),
                            f"PhysAug {'F' if self.filter_on else '-'}{'W' if self.fourier_on else '-'}")

        row1 = np.concatenate([orig_bgr, filt_bgr, four_bgr, comb_bgr], axis=1)

        # Row 2: diffs + nightsim
        diff_filt = put_text(diff_map(filtered, original), "Diff: Filter", (0, 200, 200))
        diff_four = put_text(diff_map(fouriered, original), "Diff: Fourier", (0, 200, 200))
        diff_comb = put_text(diff_map(combined, original), "Diff: Combined", (0, 200, 200))
        ns_bgr = put_text(tensor_to_bgr(ns_combined),
                          f"PhysAug + NightSim (b={self.ns_brightness:.2f})")

        row2 = np.concatenate([diff_filt, diff_four, diff_comb, ns_bgr], axis=1)

        return np.concatenate([row1, row2], axis=0)

    def _on_sigma(self, val):
        self.sigma = val / 100.0

    def _on_mean_str(self, val):
        self.mean_str = max(1.0, val / 10.0)

    def _on_decay(self, val):
        self.decay = val / 100.0

    def _on_ns_brightness(self, val):
        self.ns_brightness = max(0.01, val / 100.0)

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('sigma x100', WINDOW, int(self.sigma * 100), 400, self._on_sigma)
        cv2.createTrackbar('mean_str x10', WINDOW, int(self.mean_str * 10), 200, self._on_mean_str)
        cv2.createTrackbar('decay x100', WINDOW, int(self.decay * 100), 100, self._on_decay)
        cv2.createTrackbar('NS bright x100', WINDOW, int(self.ns_brightness * 100), 50, self._on_ns_brightness)

        need_update = True

        while True:
            if need_update:
                src = self._load_src()
                # Filter only
                filtered = apply_filter(src, self.sigma) if self.filter_on else src.clone()
                # Fourier only
                fouriered = apply_fourier(src, mean_str=self.mean_str, decay=self.decay) if self.fourier_on else src.clone()
                # Combined
                combined = src.clone()
                if self.filter_on:
                    combined = apply_filter(combined, self.sigma)
                if self.fourier_on:
                    combined = apply_fourier(combined, mean_str=self.mean_str, decay=self.decay)
                # NightSim on combined
                if self.nightsim_on:
                    ns_combined = apply_nightsim(combined, brightness=self.ns_brightness)
                else:
                    ns_combined = combined.clone()

                canvas = self._build_layout(src, filtered, fouriered, combined, ns_combined)
                cv2.imshow(WINDOW, canvas)
                need_update = False

            key = cv2.waitKey(50) & 0xFF

            # Check slider changes
            new_sigma = cv2.getTrackbarPos('sigma x100', WINDOW) / 100.0
            new_mean_str = max(1.0, cv2.getTrackbarPos('mean_str x10', WINDOW) / 10.0)
            new_decay = cv2.getTrackbarPos('decay x100', WINDOW) / 100.0
            new_ns = max(0.01, cv2.getTrackbarPos('NS bright x100', WINDOW) / 100.0)

            if (abs(new_sigma - self.sigma) > 0.005 or
                abs(new_mean_str - self.mean_str) > 0.05 or
                abs(new_decay - self.decay) > 0.005 or
                abs(new_ns - self.ns_brightness) > 0.005):
                self.sigma = new_sigma
                self.mean_str = new_mean_str
                self.decay = new_decay
                self.ns_brightness = new_ns
                need_update = True

            if key == ord('q') or key == 27:
                break
            elif key == 81 or key == 2:  # Left
                self.src_idx = (self.src_idx - 1) % len(self.src_paths)
                need_update = True
            elif key == 83 or key == 3:  # Right
                self.src_idx = (self.src_idx + 1) % len(self.src_paths)
                need_update = True
            elif key == ord('f'):
                self.filter_on = not self.filter_on
                print(f"Filter: {'ON' if self.filter_on else 'OFF'}")
                need_update = True
            elif key == ord('w'):
                self.fourier_on = not self.fourier_on
                print(f"Fourier: {'ON' if self.fourier_on else 'OFF'}")
                need_update = True
            elif key == ord('n'):
                self.nightsim_on = not self.nightsim_on
                print(f"NightSim: {'ON' if self.nightsim_on else 'OFF'}")
                need_update = True
            elif key == ord('r'):
                need_update = True
                print("Re-sampled (new random waves/filter)")
            elif key == ord('s') and self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                stem = self.src_paths[self.src_idx].stem
                cv2.imwrite(os.path.join(self.save_dir, f"{stem}_physaug.png"), canvas)
                print(f"Saved: {stem}_physaug.png")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="PhysAug Augmentation Viewer")
    parser.add_argument('--src-dir', type=str,
                        default='/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/zed',
                        help='Source image directory')
    parser.add_argument('--src-prefix', type=str, default='bl1',
                        help='Source filename prefix filter (default: bl1 = daytime)')
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='Initial filter sigma')
    parser.add_argument('--mean-str', type=float, default=8.0,
                        help='Initial fourier mean_str (higher=weaker)')
    parser.add_argument('--decay', type=float, default=0.3,
                        help='Initial Gaussian decay')
    parser.add_argument('--ns-brightness', type=float, default=0.10,
                        help='NightSim brightness')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results')
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    all_paths = sorted(src_dir.glob('*.png'))
    if args.src_prefix:
        src_paths = [p for p in all_paths if p.stem.startswith(args.src_prefix)]
    else:
        src_paths = all_paths

    if not src_paths:
        print(f"No images found in {src_dir} (prefix={args.src_prefix})")
        sys.exit(1)

    print(f"Source images : {len(src_paths)} (dir={src_dir}, prefix={args.src_prefix})")
    print(f"Initial sigma : {args.sigma}")
    print(f"Initial mean_str: {args.mean_str}")
    print(f"Initial decay : {args.decay}")
    print(f"\nControls:")
    print(f"  ← →       : Source image 전환")
    print(f"  F          : Filter on/off 토글")
    print(f"  W          : Fourier (wave) on/off 토글")
    print(f"  N          : NightSim 결합 on/off 토글")
    print(f"  R          : 파라미터 랜덤 리샘플링")
    print(f"  S          : 결과 저장")
    print(f"  Q / ESC    : 종료")

    viewer = PhysAugViewer(
        src_paths=src_paths,
        sigma=args.sigma,
        mean_str=args.mean_str,
        decay=args.decay,
        nightsim_brightness=args.ns_brightness,
        save_dir=args.save_dir,
    )
    viewer.run()


if __name__ == '__main__':
    main()
