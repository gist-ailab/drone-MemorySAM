#!/usr/bin/env python3
"""FDA (Fourier Domain Adaptation) Augmentation 시각화 뷰어.

주간 RGB 이미지에 야간 style reference의 저주파 amplitude를 전이한 결과를
인터랙티브하게 확인할 수 있는 OpenCV 기반 뷰어.

사용법:
    # 기본: 주간 train 이미지 → lj4 야간 reference로 FDA 적용
    python MISC/fda_augmentation_viewer.py

    # beta, blend_ratio 커맨드라인 지정
    python MISC/fda_augmentation_viewer.py --beta 0.05 --blend 0.7

    # 특정 소스/타겟 폴더 지정
    python MISC/fda_augmentation_viewer.py \
        --src-dir /path/to/day/images \
        --tgt-dir /path/to/night/images \
        --tgt-prefix lj4

    # 결과 이미지 저장 (save_dir에 원본+FDA 결과 PNG 저장)
    python MISC/fda_augmentation_viewer.py --save-dir ./fda_preview

컨트롤:
    ← →        : 소스 이미지 전환
    ↑ ↓        : 타겟 reference 이미지 전환
    R           : 타겟 이미지 랜덤 선택
    S           : 현재 결과를 save_dir에 저장
    Q / ESC     : 종료

슬라이더:
    beta        : 저주파 대역 비율 (0.01~0.10)
    blend       : 원본↔FDA 블렌딩 (0.0=원본, 1.0=full FDA)
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.io as io
import torchvision.transforms.functional as TF

# ─── FDA Core (semseg/augmentations_mm.py의 RandomFDA._fda_swap 동일) ────────

def fda_swap(src_img, tgt_img, beta=0.001):
    """src의 저주파 amplitude를 tgt의 것으로 교체.

    Args:
        src_img: (3, H, W) float [0, 1] — 주간 이미지
        tgt_img: (3, H, W) float [0, 1] — 야간 reference
        beta: 저주파 대역 비율 (0.01~0.10)
    Returns:
        (3, H, W) float [0, 1] — FDA 적용 결과
    """
    _, h, w = src_img.shape
    src_fft = torch.fft.fft2(src_img, dim=(-2, -1))
    tgt_fft = torch.fft.fft2(tgt_img, dim=(-2, -1))

    src_amp = torch.abs(src_fft)
    src_phase = torch.angle(src_fft)
    tgt_amp = torch.abs(tgt_fft)

    h_cut = max(1, int(h * beta))
    w_cut = max(1, int(w * beta))

    # fft2: corners = 저주파 → 4개 코너의 amplitude 교체
    new_amp = src_amp.clone()
    new_amp[:, :h_cut, :w_cut] = tgt_amp[:, :h_cut, :w_cut]
    new_amp[:, -h_cut:, :w_cut] = tgt_amp[:, -h_cut:, :w_cut]
    new_amp[:, :h_cut, -w_cut:] = tgt_amp[:, :h_cut, -w_cut:]
    new_amp[:, -h_cut:, -w_cut:] = tgt_amp[:, -h_cut:, -w_cut:]

    fda_fft = new_amp * torch.exp(1j * src_phase)
    fda_img = torch.fft.ifft2(fda_fft, dim=(-2, -1)).real
    return torch.clamp(fda_img, 0.0, 1.0)


def apply_fda(src_img, tgt_img, beta=0.03, blend_ratio=1.0):
    """FDA 적용 + 블렌딩.

    Args:
        src_img: (3, H, W) float [0, 1]
        tgt_img: (3, H, W) float [0, 1]
        beta: 저주파 대역 비율
        blend_ratio: 1.0=full FDA, <1.0=원본과 혼합
    Returns:
        (3, H, W) float [0, 1]
    """
    # target을 src 크기에 맞춤
    if tgt_img.shape[1:] != src_img.shape[1:]:
        tgt_img = TF.resize(tgt_img, list(src_img.shape[1:]),
                            TF.InterpolationMode.BILINEAR)

    fda_img = fda_swap(src_img, tgt_img, beta)

    if blend_ratio < 1.0:
        fda_img = (1 - blend_ratio) * src_img + blend_ratio * fda_img
        fda_img = torch.clamp(fda_img, 0.0, 1.0)

    return fda_img


def reconstruct_from_fda(src_img, tgt_img, beta=0.03):
    """clamp 없는 FDA → 역 FDA로 완벽 복원하여 semantic 보존 증명.

    실제 학습 파이프라인에서는 FDA 후 clamp(0,1)을 적용하는데,
    이 비선형 연산 때문에 역 FDA가 완벽하지 않음.
    여기서는 clamp 없이 forward+reverse를 수행하여
    FDA가 본질적으로 가역적이고 semantic을 보존함을 시각적으로 증명.

    Args:
        src_img: (3, H, W) float [0, 1] — 원본
        tgt_img: (3, H, W) float [0, 1] — 야간 reference
        beta: 저주파 대역 비율
    Returns:
        (3, H, W) float [0, 1] — 복원 결과 (원본과 거의 동일해야 함)
    """
    _, h, w = src_img.shape

    src_fft = torch.fft.fft2(src_img, dim=(-2, -1))
    tgt_fft = torch.fft.fft2(tgt_img, dim=(-2, -1))

    src_amp = torch.abs(src_fft)
    src_phase = torch.angle(src_fft)
    tgt_amp = torch.abs(tgt_fft)

    h_cut = max(1, int(h * beta))
    w_cut = max(1, int(w * beta))

    # Forward FDA (clamp 없이): src의 저주파 → tgt의 저주파
    fwd_amp = src_amp.clone()
    fwd_amp[:, :h_cut, :w_cut] = tgt_amp[:, :h_cut, :w_cut]
    fwd_amp[:, -h_cut:, :w_cut] = tgt_amp[:, -h_cut:, :w_cut]
    fwd_amp[:, :h_cut, -w_cut:] = tgt_amp[:, :h_cut, -w_cut:]
    fwd_amp[:, -h_cut:, -w_cut:] = tgt_amp[:, -h_cut:, -w_cut:]

    fda_fft = fwd_amp * torch.exp(1j * src_phase)
    fda_img_noclamp = torch.fft.ifft2(fda_fft, dim=(-2, -1)).real
    # clamp 생략 → 완벽한 주파수 정보 유지

    # Reverse FDA: fda 결과의 저주파 → 원본의 저주파로 복원
    fda2_fft = torch.fft.fft2(fda_img_noclamp, dim=(-2, -1))
    fda2_amp = torch.abs(fda2_fft)
    fda2_phase = torch.angle(fda2_fft)

    rev_amp = fda2_amp.clone()
    rev_amp[:, :h_cut, :w_cut] = src_amp[:, :h_cut, :w_cut]
    rev_amp[:, -h_cut:, :w_cut] = src_amp[:, -h_cut:, :w_cut]
    rev_amp[:, :h_cut, -w_cut:] = src_amp[:, :h_cut, -w_cut:]
    rev_amp[:, -h_cut:, -w_cut:] = src_amp[:, -h_cut:, -w_cut:]

    rev_fft = rev_amp * torch.exp(1j * fda2_phase)
    reconstructed = torch.fft.ifft2(rev_fft, dim=(-2, -1)).real
    return torch.clamp(reconstructed, 0.0, 1.0)


# ─── FFT Amplitude Spectrum 시각화 ───────────────────────────────────────────

def get_amplitude_spectrum(img_tensor):
    """(3,H,W) float [0,1] → grayscale log-amplitude spectrum (H,W) uint8."""
    gray = img_tensor.mean(dim=0, keepdim=True)  # (1, H, W)
    fft = torch.fft.fft2(gray, dim=(-2, -1))
    amp = torch.abs(torch.fft.fftshift(fft))
    log_amp = torch.log1p(amp)
    log_amp = log_amp.squeeze(0)  # (H, W)
    # normalize to [0, 255]
    mn, mx = log_amp.min(), log_amp.max()
    if mx > mn:
        log_amp = (log_amp - mn) / (mx - mn) * 255
    return log_amp.numpy().astype(np.uint8)


# ─── Image I/O helpers ───────────────────────────────────────────────────────

def load_image(path):
    """이미지 로드 → (3, H, W) float [0, 1]."""
    img = io.read_image(str(path))[:3, ...]
    return img.float() / 255.0


def tensor_to_bgr(tensor):
    """(3, H, W) float [0,1] → (H, W, 3) uint8 BGR for OpenCV."""
    img = (tensor * 255).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).numpy()  # (H, W, 3) RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ─── Main Viewer ─────────────────────────────────────────────────────────────

WINDOW = "FDA Augmentation Viewer  [</>: src, Up/Down: tgt, R: random tgt, S: save, Q: quit]"


class FDAViewer:
    def __init__(self, src_paths, tgt_paths, beta=0.03, blend=1.0, save_dir=None):
        self.src_paths = src_paths
        self.tgt_paths = tgt_paths
        self.src_idx = 0
        self.tgt_idx = 0
        self.beta = beta
        self.blend = blend
        self.save_dir = save_dir
        self.save_count = 0

        # 슬라이더 값 (int: beta*1000, blend*100)
        self.beta_slider = int(beta * 1000)
        self.blend_slider = int(blend * 100)

    def _load_src(self):
        return load_image(self.src_paths[self.src_idx])

    def _load_tgt(self):
        return load_image(self.tgt_paths[self.tgt_idx])

    def _build_layout(self, src, tgt, fda_result, reconstructed):
        """4열 레이아웃: [Source | Target Ref | FDA Result | Reconstructed]
           + 하단에 FFT amplitude spectrum 행 추가.
           원본 해상도 유지 — 리사이즈 없이 concat.
        """
        src_bgr = tensor_to_bgr(src)
        tgt_bgr = tensor_to_bgr(tgt)
        fda_bgr = tensor_to_bgr(fda_result)
        rec_bgr = tensor_to_bgr(reconstructed)

        # target이 src와 다른 해상도일 수 있으므로 src 기준으로 통일
        h, w = src_bgr.shape[:2]
        if tgt_bgr.shape[:2] != (h, w):
            tgt_bgr = cv2.resize(tgt_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # 라벨 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, h / 1000)
        thickness = max(1, int(h / 500))
        y1, y2 = int(30 * h / 500), int(55 * h / 500)

        cv2.putText(src_bgr, f"Source (Day) [{self.src_idx+1}/{len(self.src_paths)}]",
                    (10, y1), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(src_bgr, Path(self.src_paths[self.src_idx]).stem,
                    (10, y2), font, font_scale * 0.75, (200, 200, 200), thickness, cv2.LINE_AA)

        cv2.putText(tgt_bgr, f"Target (Night) [{self.tgt_idx+1}/{len(self.tgt_paths)}]",
                    (10, y1), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(tgt_bgr, Path(self.tgt_paths[self.tgt_idx]).stem,
                    (10, y2), font, font_scale * 0.75, (200, 200, 200), thickness, cv2.LINE_AA)

        cv2.putText(fda_bgr, f"FDA (beta={self.beta:.3f}, blend={self.blend:.2f})",
                    (10, y1), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        # 복원 이미지 — 원본과의 차이(PSNR) 표시
        diff = (src.float() - reconstructed.float()).abs().mean().item() * 255
        cv2.putText(rec_bgr, f"Reconstructed (MAE={diff:.1f})",
                    (10, y1), font, font_scale, (0, 200, 255), thickness, cv2.LINE_AA)

        # FFT amplitude spectrum 행 (원본 해상도 유지)
        def make_spectrum_panel(tensor, label):
            spec = get_amplitude_spectrum(tensor)
            if spec.shape[:2] != (h, w):
                spec = cv2.resize(spec, (w, h), interpolation=cv2.INTER_AREA)
            spec_bgr = cv2.applyColorMap(spec, cv2.COLORMAP_JET)
            cv2.putText(spec_bgr, label, (10, y1), font, font_scale * 0.75,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            return spec_bgr

        spec_src = make_spectrum_panel(src, "FFT: Source")
        spec_tgt = make_spectrum_panel(tgt, "FFT: Target")
        spec_fda = make_spectrum_panel(fda_result, "FFT: FDA")
        spec_rec = make_spectrum_panel(reconstructed, "FFT: Reconstructed")

        # 조합: 상단 4열, 하단 4열
        n_cols = 4
        sep_v = np.ones((h, 2, 3), dtype=np.uint8) * 128
        sep_h = np.ones((2, w * n_cols + 2 * (n_cols - 1), 3), dtype=np.uint8) * 128

        top_row = np.hstack([src_bgr, sep_v, tgt_bgr, sep_v, fda_bgr, sep_v, rec_bgr])
        bot_row = np.hstack([spec_src, sep_v, spec_tgt, sep_v, spec_fda, sep_v, spec_rec])

        canvas = np.vstack([top_row, sep_h, bot_row])
        return canvas

    def _on_beta(self, val):
        self.beta_slider = max(val, 1)
        self.beta = self.beta_slider / 1000.0

    def _on_blend(self, val):
        self.blend_slider = val
        self.blend = self.blend_slider / 100.0

    def _save_current(self, src, tgt, fda_result):
        if not self.save_dir:
            print("[WARN] --save-dir not specified, skipping save.")
            return
        os.makedirs(self.save_dir, exist_ok=True)
        src_stem = Path(self.src_paths[self.src_idx]).stem
        tgt_stem = Path(self.tgt_paths[self.tgt_idx]).stem

        prefix = f"{src_stem}__tgt_{tgt_stem}__b{self.beta:.3f}_bl{self.blend:.2f}"

        src_bgr = tensor_to_bgr(src)
        tgt_bgr = tensor_to_bgr(tgt)
        fda_bgr = tensor_to_bgr(fda_result)

        cv2.imwrite(os.path.join(self.save_dir, f"{prefix}_src.png"), src_bgr)
        cv2.imwrite(os.path.join(self.save_dir, f"{prefix}_tgt.png"), tgt_bgr)
        cv2.imwrite(os.path.join(self.save_dir, f"{prefix}_fda.png"), fda_bgr)
        self.save_count += 1
        print(f"[SAVED #{self.save_count}] {self.save_dir}/{prefix}_*.png")

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("beta x1000", WINDOW, self.beta_slider, 100, self._on_beta)
        cv2.createTrackbar("blend x100", WINDOW, self.blend_slider, 100, self._on_blend)

        need_update = True
        src, tgt, fda_result, reconstructed = None, None, None, None

        while True:
            if need_update:
                src = self._load_src()
                tgt = self._load_tgt()
                fda_result = apply_fda(src, tgt, self.beta, self.blend)
                reconstructed = reconstruct_from_fda(src, tgt, self.beta)
                canvas = self._build_layout(src, tgt, fda_result, reconstructed)
                cv2.imshow(WINDOW, canvas)
                need_update = False

            key = cv2.waitKey(30) & 0xFF

            # beta/blend 슬라이더 변경 감지
            new_beta = cv2.getTrackbarPos("beta x1000", WINDOW) / 1000.0
            new_blend = cv2.getTrackbarPos("blend x100", WINDOW) / 100.0
            if abs(new_beta - self.beta) > 1e-4 or abs(new_blend - self.blend) > 1e-3:
                self.beta = max(new_beta, 0.001)
                self.blend = new_blend
                fda_result = apply_fda(src, tgt, self.beta, self.blend)
                reconstructed = reconstruct_from_fda(src, tgt, self.beta)
                canvas = self._build_layout(src, tgt, fda_result, reconstructed)
                cv2.imshow(WINDOW, canvas)

            if key == ord('q') or key == 27:  # Q / ESC
                break
            elif key == 83 or key == ord('l') or key == 3:  # → or L
                self.src_idx = (self.src_idx + 1) % len(self.src_paths)
                need_update = True
            elif key == 81 or key == ord('h') or key == 2:  # ← or H
                self.src_idx = (self.src_idx - 1) % len(self.src_paths)
                need_update = True
            elif key == 82 or key == ord('k') or key == 0:  # ↑ or K
                self.tgt_idx = (self.tgt_idx - 1) % len(self.tgt_paths)
                need_update = True
            elif key == 84 or key == ord('j') or key == 1:  # ↓ or J
                self.tgt_idx = (self.tgt_idx + 1) % len(self.tgt_paths)
                need_update = True
            elif key == ord('r'):  # R: random target
                self.tgt_idx = random.randint(0, len(self.tgt_paths) - 1)
                need_update = True
            elif key == ord('s'):  # S: save
                self._save_current(src, tgt, fda_result)

        cv2.destroyAllWindows()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def collect_images(directory, prefix=None, exts=('.png', '.jpg', '.jpeg')):
    """디렉토리에서 이미지 파일 수집 (prefix 필터링 옵션)."""
    paths = []
    for f in sorted(Path(directory).iterdir()):
        if f.suffix.lower() in exts:
            if prefix and not f.stem.startswith(prefix):
                continue
            paths.append(str(f))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="FDA Augmentation 시각화 뷰어",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--src-dir', type=str, default=None,
                        help='Source (day) 이미지 디렉토리 (default: MULTIAQUA train zed)')
    parser.add_argument('--src-prefix', type=str, default=None,
                        help='Source 파일명 prefix 필터 (예: bl1)')
    parser.add_argument('--tgt-dir', type=str, default=None,
                        help='Target (night) reference 이미지 디렉토리')
    parser.add_argument('--tgt-prefix', type=str, default='lj4',
                        help='Target 파일명 prefix 필터 (default: lj4)')
    parser.add_argument('--beta', type=float, default=0.09,
                        help='FDA 저주파 대역 비율 (default: 0.03)')
    parser.add_argument('--blend', type=float, default=1.0,
                        help='원본↔FDA 블렌딩 비율 (default: 1.0 = full FDA)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='결과 이미지 저장 디렉토리 (S키로 저장)')
    parser.add_argument('--max-src', type=int, default=200,
                        help='소스 이미지 최대 수 (default: 200)')
    args = parser.parse_args()

    # Default 경로 설정 (MULTIAQUA 데이터셋)
    default_data = '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data'
    local_data = os.path.expanduser('~/drone-demo/MULTIAQUA_night/MULTIAQUA_night/data')

    if args.src_dir is None:
        for candidate in [default_data, local_data]:
            src_candidate = os.path.join(candidate, 'zed')
            if os.path.isdir(src_candidate):
                args.src_dir = src_candidate
                break
        if args.src_dir is None:
            parser.error("--src-dir를 지정하세요. 기본 경로를 찾을 수 없습니다.")

    if args.tgt_dir is None:
        args.tgt_dir = args.src_dir  # 같은 폴더에서 lj4_ prefix로 필터링

    # 이미지 수집
    src_paths = collect_images(args.src_dir, prefix=args.src_prefix)
    tgt_paths = collect_images(args.tgt_dir, prefix=args.tgt_prefix)

    if not src_paths:
        print(f"[ERROR] No source images in {args.src_dir} (prefix={args.src_prefix})")
        sys.exit(1)
    if not tgt_paths:
        print(f"[ERROR] No target images in {args.tgt_dir} (prefix={args.tgt_prefix})")
        sys.exit(1)

    # Source에서 night 이미지(lj4) 제외 (day만 남기기)
    if args.src_prefix is None and args.tgt_prefix:
        day_paths = [p for p in src_paths if not Path(p).stem.startswith(args.tgt_prefix)]
        if day_paths:
            src_paths = day_paths

    # 소스 이미지 수 제한
    if len(src_paths) > args.max_src:
        src_paths = src_paths[:args.max_src]

    print(f"Source images : {len(src_paths)} (dir={args.src_dir}, prefix={args.src_prefix})")
    print(f"Target images : {len(tgt_paths)} (dir={args.tgt_dir}, prefix={args.tgt_prefix})")
    print(f"Initial beta  : {args.beta}")
    print(f"Initial blend : {args.blend}")
    print()
    print("Controls:")
    print("  ← →       : Source image 전환")
    print("  ↑ ↓       : Target reference 전환")
    print("  R          : Target 랜덤 선택")
    print("  S          : 현재 결과 저장")
    print("  Q / ESC    : 종료")
    print()

    viewer = FDAViewer(
        src_paths=src_paths,
        tgt_paths=tgt_paths,
        beta=args.beta,
        blend=args.blend,
        save_dir=args.save_dir,
    )
    viewer.run()


if __name__ == '__main__':
    main()
