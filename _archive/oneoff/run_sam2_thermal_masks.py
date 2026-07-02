#!/usr/bin/env python3
"""
SAM2 vanilla 모델로 MULTIAQUA thermal 이미지 전체에 대해 automatic mask generation 수행.
- tmp/: 원본 마스크 데이터 (npz) 저장
- result/: 입력과 동일한 파일명의 눈에 보이는 segmentation mask PNG + thermal|mask concat 시각화

사용 (MMSS_SAM 환경):
  python run_sam2_thermal_masks.py --thermal_dir /ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_camera
  python run_sam2_thermal_masks.py --thermal_dir /path/to/thermal_camera --out_dir ./output_thermal_sam2
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import cv2

def _log(msg: str) -> None:
    print(msg, flush=True)

# 프로젝트 루트 (체크포인트 경로 등)
PROJECT_ROOT = Path(__file__).resolve().parent

# sam2 패키지만 로드 (semseg 전체 미사용 → einops 등 불필요)
_sam2_parent = PROJECT_ROOT / "semseg" / "models" / "sam2"
if str(_sam2_parent) not in sys.path:
    sys.path.insert(0, str(_sam2_parent))

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def _build_sam2_vanilla(config_file, ckpt_path, device="cuda"):
    """
    프로젝트의 SAM2Base가 num_multimask_outputs=25로 수정되어 있어 vanilla 체크포인트(=3)와
    불일치. 모델 빌드 후 mask_decoder만 원본 크기로 재생성하고 체크포인트 전체를 로드.
    """
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.modeling.sam2_utils import MLP
    import torch.nn as nn

    cfg = compose(config_name=config_file, overrides=[
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
    ])
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    # mask_decoder를 원본 크기(num_multimask_outputs=3, num_mask_tokens=4)로 복구
    dec = model.sam_mask_decoder
    orig_n = 3
    num_tokens = orig_n + 1  # 4
    dec.num_multimask_outputs = orig_n
    dec.num_mask_tokens = num_tokens
    dec.mask_tokens = nn.Embedding(num_tokens, dec.transformer_dim)
    dec.output_hypernetworks_mlps = nn.ModuleList([
        MLP(dec.transformer_dim, dec.transformer_dim, dec.transformer_dim // 8, 3)
        for _ in range(num_tokens)
    ])
    dec.iou_prediction_head = MLP(
        dec.transformer_dim, 256, num_tokens, 3
    )

    sd = torch.load(ckpt_path, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    msg = model.load_state_dict(sd, strict=False)
    if msg.missing_keys:
        _log(f"  load warning - missing: {len(msg.missing_keys)} keys")
    if msg.unexpected_keys:
        _log(f"  load warning - unexpected: {len(msg.unexpected_keys)} keys")

    model = model.to(device)
    model.eval()
    return model


# 기본 경로
DEFAULT_THERMAL_DIR = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_camera"
CHECKPOINT = PROJECT_ROOT / "semseg" / "models" / "sam2" / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
CONFIG = "sam2_hiera_b+.yaml"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _auto_crop_padding(gray: np.ndarray, threshold: int = 5):
    """검정 패딩 영역을 자동으로 크롭. (cropped, y0, x0) 반환. 패딩이 없으면 그대로."""
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return gray, 0, 0
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return gray[y0:y1+1, x0:x1+1], int(y0), int(x0)


def _enhance_thermal_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Thermal 이미지 전처리 (이미 CLAHE 적용된 입력 가정):
    1) Bilateral filter: 노이즈를 스무딩하되 실제 경계(수면/사람/구름)는 보존
    2) Percentile-based min-max stretch (5%~95%): 극단값 제외한 부드러운 정규화
       → 노이즈 차이를 과도하게 증폭하지 않음
    """
    content = gray[gray > 0] if np.any(gray > 0) else gray.ravel()
    if content.std() < 0.5:
        return gray

    # bilateral filter: 노이즈 제거 + 경계 보존
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=40, sigmaSpace=40)

    sc = smoothed[smoothed > 0] if np.any(smoothed > 0) else smoothed.ravel()
    p_lo, p_hi = np.percentile(sc, [5, 95])
    if p_hi - p_lo < 1.0:
        p_lo, p_hi = float(sc.min()), float(sc.max())
    if p_hi - p_lo < 1.0:
        return gray

    stretched = (smoothed.astype(np.float32) - p_lo) / (p_hi - p_lo)
    stretched = np.clip(stretched * 255, 0, 255).astype(np.uint8)
    return stretched


def load_thermal_as_rgb(path: Path):
    """
    Thermal 이미지 로드 → 패딩 크롭 → contrast stretch → CLAHE → HWC uint8 3채널.
    반환: (enhanced_rgb, crop_bbox) where crop_bbox = (y0, x0, crop_h, crop_w, orig_h, orig_w)
    """
    img = np.array(Image.open(path).convert("L"))
    orig_h, orig_w = img.shape[:2]

    cropped, y0, x0 = _auto_crop_padding(img, threshold=5)
    crop_h, crop_w = cropped.shape[:2]
    enhanced = _enhance_thermal_contrast(cropped)
    rgb = np.stack([enhanced, enhanced, enhanced], axis=-1)

    crop_bbox = (y0, x0, crop_h, crop_w, orig_h, orig_w)
    return rgb, crop_bbox


def resize_image_max_side(img: np.ndarray, max_size: int):
    """이미지를 긴 변이 max_size 이하가 되도록 리사이즈. (리사이즈된 이미지, 원본 H, 원본 W) 반환."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img, h, w
    scale = max_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    out = np.array(
        Image.fromarray(img).resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    )
    return out, h, w


def masks_to_colored_viz(masks: list, h: int, w: int, seed: int = 0) -> np.ndarray:
    """마스크 리스트를 겹쳐서 색칠한 HWC uint8 이미지 생성 (배경 검정). output_mode=binary_mask면 segmentation은 ndarray."""
    np.random.seed(seed)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for ann in masks:
        seg = ann["segmentation"]
        if isinstance(seg, np.ndarray):
            pass
        else:
            try:
                from sam2.utils.amg import rle_to_mask
                seg = rle_to_mask(seg)
            except Exception:
                continue
        if seg.shape[:2] != (h, w):
            from PIL import Image as PILImage
            seg = np.array(
                PILImage.fromarray(seg.astype(np.uint8)).resize(
                    (w, h), resample=PILImage.NEAREST
                )
            )
        color = (np.random.random(3) * 200 + 55).astype(np.uint8)
        out[seg > 0] = color
    return out


def main():
    _log("run_sam2_thermal_masks.py 시작")
    parser = argparse.ArgumentParser(description="SAM2 automatic mask generation on thermal images")
    parser.add_argument(
        "--thermal_dir",
        type=str,
        default=DEFAULT_THERMAL_DIR,
        help="Thermal 이미지 폴더 경로",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="출력 루트 (기본: 프로젝트/thermal_sam2_output). 아래에 tmp/ result/ 생성",
    )
    parser.add_argument(
        "--points_per_side",
        type=int,
        default=24,
        help="Automatic mask generator points_per_side (기본 24, 줄이면 빠름)",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="inference 시 긴 변 최대 픽셀 (OOM 방지, 기본 1024)",
    )
    parser.add_argument(
        "--points_per_batch",
        type=int,
        default=16,
        help="한 번에 처리할 point 수 (낮을수록 메모리 적음, 기본 16)",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=500,
        help="이 픽셀 수보다 작은 마스크 제거 (기본 500, 노이즈 마스크 필터링)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    thermal_dir = Path(args.thermal_dir)
    if not thermal_dir.exists():
        _log(f"Thermal 디렉토리가 없습니다: {thermal_dir}")
        sys.exit(1)

    out_root = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "thermal_sam2_output")
    tmp_dir = out_root / "tmp"
    result_dir = out_root / "result"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        _log(f"출력 디렉토리 생성 실패 (쓰기 권한/경로 확인): {out_root}\n  오류: {e}")
        sys.exit(1)
    _log(f"출력 루트: {out_root.resolve()}")

    image_paths = [
        p
        for p in sorted(thermal_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_paths:
        _log(f"이미지가 없습니다: {thermal_dir}")
        sys.exit(1)
    _log(f"총 {len(image_paths)}개 thermal 이미지 처리 예정. 출력: tmp={tmp_dir}, result={result_dir}")

    # SAM2 + AutomaticMaskGenerator 로드 (프로젝트의 build_sam.py는 mask decoder 가중치를
    # 건너뛰므로, 여기서 직접 빌드+전체 로드)
    ckpt_path = Path(CHECKPOINT)
    if not ckpt_path.exists():
        _log(f"체크포인트 없음: {ckpt_path}")
        sys.exit(1)
    _log("SAM2 모델 로딩 (vanilla, 전체 가중치)...")
    sam2 = _build_sam2_vanilla(CONFIG, str(ckpt_path), device=args.device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=args.min_area,
        output_mode="binary_mask",
    )
    _log("모델 로드 완료. (단일 이미지 inference, max_size=%d, points_per_batch=%d)" % (args.max_size, args.points_per_batch))

    for i, img_path in enumerate(image_paths):
        name = img_path.stem
        ext = img_path.suffix
        _log(f"[{i+1}/{len(image_paths)}] {img_path.name}")
        try:
            # 1) 로드: 패딩 크롭 + contrast stretch + CLAHE
            image_cropped, crop_bbox = load_thermal_as_rgb(img_path)
            y0, x0, crop_h, crop_w, orig_h, orig_w = crop_bbox

            # 2) OOM 방지: inference는 max_size 이하로 리사이즈
            image_infer, _, _ = resize_image_max_side(image_cropped, args.max_size)
            h_infer, w_infer = image_infer.shape[:2]

            # 3) SAM2 inference (단일 이미지)
            masks = mask_generator.generate(image_infer)
            n_masks = len(masks) if masks else 0
            _log(f"  → {n_masks} masks, crop=({crop_h}x{crop_w}), infer=({h_infer}x{w_infer})")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(mask_generator, "predictor") and hasattr(mask_generator.predictor, "reset_predictor"):
                mask_generator.predictor.reset_predictor()

            # 4) 마스크를 크롭 좌표 → 원본 좌표로 매핑
            if not masks:
                mask_viz_full = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                seg_stack = np.zeros((0, orig_h, orig_w), dtype=np.uint8)
            else:
                mask_viz_infer = masks_to_colored_viz(masks, h_infer, w_infer, seed=i)
                # infer size → crop size로 복원
                mask_viz_crop = np.array(
                    Image.fromarray(mask_viz_infer).resize(
                        (crop_w, crop_h), resample=Image.Resampling.NEAREST
                    )
                )
                # crop size → 원본 좌표에 embed
                mask_viz_full = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                mask_viz_full[y0:y0+crop_h, x0:x0+crop_w] = mask_viz_crop

                seg_list = []
                for ann in masks:
                    s = ann["segmentation"]
                    if not isinstance(s, np.ndarray):
                        try:
                            from sam2.utils.amg import rle_to_mask
                            s = rle_to_mask(s)
                        except Exception:
                            continue
                    # infer size → crop size
                    s_crop = np.array(
                        Image.fromarray(s.astype(np.uint8)).resize(
                            (crop_w, crop_h), resample=Image.Resampling.NEAREST
                        )
                    )
                    # crop → 원본 좌표
                    s_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    s_full[y0:y0+crop_h, x0:x0+crop_w] = s_crop
                    seg_list.append(s_full)

                seg_stack = np.stack(seg_list, axis=0) if seg_list else np.zeros((0, orig_h, orig_w), dtype=np.uint8)

            # 5) 저장
            np.savez_compressed(
                tmp_dir / f"{name}_masks.npz",
                masks=seg_stack,
                count=seg_stack.shape[0],
            )
            Image.fromarray(mask_viz_full).save(result_dir / f"{name}{ext}")

            # 시각화용 concat: enhanced thermal | mask overlay(thermal 위에 반투명 마스크)
            viz_input = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            viz_input[y0:y0+crop_h, x0:x0+crop_w] = image_cropped
            overlay = viz_input.copy()
            mask_pixels = mask_viz_full.sum(axis=2) > 0
            overlay[mask_pixels] = (
                viz_input[mask_pixels].astype(np.float32) * 0.4 +
                mask_viz_full[mask_pixels].astype(np.float32) * 0.6
            ).astype(np.uint8)
            concat = np.concatenate([viz_input, overlay], axis=1)
            Image.fromarray(concat).save(result_dir / f"{name}_concat{ext}")

        except Exception as e:
            _log(f"  오류: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    _log("완료.")
    _log(f"  tmp:   {tmp_dir}")
    _log(f"  result: {result_dir} (동일 파일명 mask + *_concat 시각화)")


if __name__ == "__main__":
    main()
