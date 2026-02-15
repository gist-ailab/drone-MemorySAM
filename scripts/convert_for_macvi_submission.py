#!/usr/bin/env python3
"""
MaCVi Multimodal Semantic Segmentation Challenge 제출용 변환 스크립트.

문제: 로컬 모델은 0-indexed (0=Static, 1=Dynamic, 2=Water, 3=Sky)를 출력하지만,
     MaCVi 서버는 1-indexed (1=Static, 2=Dynamic, 3=Water, 4=Sky)를 기대합니다.
     이 불일치로 인해 리더보드 mIoU(7.9%)와 로컬 val mIoU(93.12%)가 크게 차이납니다.

해결: 예측 마스크의 각 픽셀에 +1을 적용 (0→1, 1→2, 2→3, 3→4)

사용:
  python scripts/convert_for_macvi_submission.py \\
    --input_dir outputs/.../val_pred/seg \\
    --output_dir outputs/.../val_pred_macvi
  python scripts/convert_for_macvi_submission.py \\
    --input_dir outputs/.../test_pred/seg \\
    --output_dir outputs/.../test_pred_macvi
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_mask_0_to_1indexed(arr: np.ndarray) -> np.ndarray:
    """
    0-indexed (0,1,2,3) → 1-indexed (1,2,3,4)
    - 0 (Static)  → 1
    - 1 (Dynamic) → 2
    - 2 (Water)  → 3
    - 3 (Sky)    → 4
    ignore(255)는 그대로 유지 (있다면)
    """
    out = arr.copy().astype(np.uint8)
    valid = (arr >= 0) & (arr <= 3)
    out[valid] = arr[valid] + 1
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert segmentation masks for MaCVi submission")
    parser.add_argument("--input_dir", type=str, required=True, help="seg/ 폴더 경로 (0-indexed)")
    parser.add_argument("--output_dir", type=str, required=True, help="변환된 마스크 저장 경로 (1-indexed)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files in {input_dir}")

    for fp in tqdm(files, desc="Converting"):
        arr = np.array(Image.open(fp))
        if arr.ndim > 2:
            arr = arr[:, :, 0]
        converted = convert_mask_0_to_1indexed(arr)
        Image.fromarray(converted).save(output_dir / fp.name)

    print(f"Converted {len(files)} masks: {input_dir} -> {output_dir}")
    print("MaCVi expects: 1=Static, 2=Dynamic, 3=Water, 4=Sky (1-indexed)")


if __name__ == "__main__":
    main()
