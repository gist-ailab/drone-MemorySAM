"""
Thermal 이미지를 RGB 이미지 평면에 맞춰 동일 해상도로 만드는 도구.

1) 대화형 정렬: RGB 위에 thermal을 오버레이하고, 키보드로 확대/축소/회전/이동하여
   겹치는 영역을 맞춘 뒤, 설정(scale, angle, tx, ty)을 저장.
2) 배치 적용: 저장된 설정으로 전체 thermal에 대해 RGB 크기 캔버스에 0 패딩 + 해당 영역만 thermal 적용하여 저장.
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

RGB_DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/zed")
THERMAL_DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_camera")
OUTPUT_DIR = Path("/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_min_max")
PARAMS_FILE = Path(__file__).resolve().parent / "thermal_align_params.json"

# 키보드 스텝
STEP_SCALE = 0.05
STEP_ANGLE = 2.0  # degrees
STEP_TX = 5
STEP_TY = 5


def build_pairs(
    rgb_dir: Path = RGB_DIR,
    thermal_dir: Path = THERMAL_DIR,
    rgb_ext: str = ".png",
    thermal_ext: str = ".png",
) -> List[dict]:
    """RGB와 thermal 파일명 stem 기준으로 페어 목록 생성."""
    pairs = []
    rgb_dir = Path(rgb_dir)
    thermal_dir = Path(thermal_dir)
    if not thermal_dir.exists():
        return pairs
    for thermal_path in sorted(thermal_dir.glob(f"*{thermal_ext}")):
        stem = thermal_path.stem
        if stem.endswith("_thermal"):
            stem = stem[:-8]
        rgb_path = rgb_dir / f"{stem}{rgb_ext}"
        for cand in [rgb_path, rgb_dir / f"{stem}_rgb{rgb_ext}"]:
            if cand.exists():
                pairs.append({"stem": stem, "rgb": str(cand), "thermal": str(thermal_path)})
                break
    return pairs


def _affine_matrix_inv(scale: float, angle_deg: float, tx: float, ty: float) -> np.ndarray:
    """
    Thermal (u,v) -> RGB (x,y) = scale * R(angle) * (u,v) + (tx, ty).
    warpAffine용 역행렬: RGB (x,y) -> thermal (u,v).
    """
    angle_rad = math.radians(angle_deg)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    inv_scale = 1.0 / max(1e-6, scale)
    # (u,v) = R(-angle) * ((x-tx, y-ty) * inv_scale)
    M = np.array([
        [c * inv_scale, s * inv_scale, (-tx * c - ty * s) * inv_scale],
        [-s * inv_scale, c * inv_scale, (tx * s - ty * c) * inv_scale],
    ], dtype=np.float64)
    return M


def thermal_to_rgb_canvas(
    thermal: np.ndarray,
    rgb_shape: Tuple[int, int],
    scale: float,
    angle_deg: float,
    tx: float,
    ty: float,
    out_channels: int = 1,
) -> np.ndarray:
    """
    RGB와 동일 크기 (H,W) 캔버스를 0으로 채운 뒤, 변환된 thermal만 해당 영역에 적용.
    thermal: (Ht,Wt) or (Ht,Wt,3). rgb_shape: (H, W).
    out_channels: 1이면 그레이스케일 캔버스, 3이면 3채널 (thermal을 복제).
    """
    H, W = rgb_shape[0], rgb_shape[1]
    if thermal.ndim == 3:
        thermal_1ch = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
    else:
        thermal_1ch = thermal
    M = _affine_matrix_inv(scale, angle_deg, tx, ty)
    warped = cv2.warpAffine(
        thermal_1ch, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if out_channels == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    return warped


def thermal_for_display(thermal_canvas: np.ndarray) -> np.ndarray:
    """
    표시용: 파란빛 제거, 그레이스케일, min-max 정규화로 구분력 향상.
    thermal_canvas는 1채널. 저장 시에는 사용하지 않음(원본 그대로 저장).
    """
    if thermal_canvas.ndim == 3:
        thermal_1ch = cv2.cvtColor(thermal_canvas, cv2.COLOR_BGR2GRAY)
    else:
        thermal_1ch = np.asarray(thermal_canvas, dtype=np.float64)
    mask = thermal_1ch > 0
    if not np.any(mask):
        return np.zeros((thermal_1ch.shape[0], thermal_1ch.shape[1]), dtype=np.uint8)
    vmin, vmax = np.min(thermal_1ch[mask]), np.max(thermal_1ch[mask])
    if vmax <= vmin:
        out = np.where(mask, 255, 0).astype(np.uint8)
    else:
        out = np.zeros_like(thermal_1ch, dtype=np.float64)
        out[mask] = (thermal_1ch[mask] - vmin) / (vmax - vmin) * 255
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def overlay_thermal_on_rgb(
    rgb: np.ndarray,
    thermal_canvas: np.ndarray,
    alpha: float = 0.5,
    use_grayscale_norm: bool = True,
) -> np.ndarray:
    """
    RGB 위에 thermal 오버레이.
    use_grayscale_norm=True: 표시용으로 그레이스케일 + min-max 정규화(구분력 향상).
    thermal_canvas는 1채널. 저장 시에는 이 처리가 적용되지 않은 원본 사용.
    """
    if use_grayscale_norm:
        thermal_disp = thermal_for_display(thermal_canvas)
        thermal_color = cv2.cvtColor(thermal_disp, cv2.COLOR_GRAY2BGR)
    else:
        thermal_disp = thermal_canvas if thermal_canvas.ndim == 2 else cv2.cvtColor(thermal_canvas, cv2.COLOR_BGR2GRAY)
        thermal_color = cv2.cvtColor(thermal_disp, cv2.COLOR_GRAY2BGR)
    if thermal_color.shape[:2] != rgb.shape[:2]:
        thermal_color = cv2.resize(thermal_color, (rgb.shape[1], rgb.shape[0]))
    mask = (thermal_canvas > 0) if thermal_canvas.ndim == 2 else (np.any(thermal_canvas > 0, axis=2))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    out = rgb.astype(np.float32)
    out[mask] = (1 - alpha) * rgb[mask] + alpha * thermal_color[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def run_interactive(
    rgb_dir: Path = RGB_DIR,
    thermal_dir: Path = THERMAL_DIR,
    params_path: Path = PARAMS_FILE,
) -> None:
    """
    OpenCV 창에서 RGB + thermal 오버레이 표시. RGB 영역 안에 thermal이 어디에 겹치는지 맞춘 뒤 설정 저장.

    트랙바:
      - index: 이전/다음 이미지
      - overlay alpha (0-100): thermal 오버레이 투명도
    키보드:
      - + / = : 확대, - : 축소
      - [ / ] : 회전 (왼쪽/오른쪽)
      - 방향키: thermal 이동 (tx, ty)
      - S : 현재 scale, angle, tx, ty를 JSON으로 저장 (다음 배치에 사용)
      - B : 저장된(또는 현재) 설정으로 전체 thermal 배치 처리
      - Q / ESC : 종료
    """
    pairs = build_pairs(rgb_dir, thermal_dir)
    if not pairs:
        print("No RGB–thermal pairs found.")
        return
    n = len(pairs)
    # 저장된 파라미터 로드
    if params_path.exists():
        with open(params_path) as f:
            saved = json.load(f)
        scale = saved.get("scale", 1.0)
        angle = saved.get("angle", 0.0)
        tx = saved.get("tx", 0.0)
        ty = saved.get("ty", 0.0)
        print(f"Loaded params from {params_path}: scale={scale}, angle={angle}, tx={tx}, ty={ty}")
    else:
        scale, angle, tx, ty = 1.0, 0.0, 0.0, 0.0

    win = "thermal align: [+/-] scale, [/] rotate, arrows move, S save, B batch, Q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    idx = [0]
    alpha_val = [50]  # 0..100

    def on_trackbar_index(val):
        idx[0] = min(max(0, val), n - 1)

    def on_trackbar_alpha(val):
        alpha_val[0] = val

    cv2.createTrackbar("index", win, 0, max(0, n - 1), on_trackbar_index)
    cv2.createTrackbar("overlay alpha (0-100)", win, 50, 100, on_trackbar_alpha)

    state = {"scale": scale, "angle": angle, "tx": tx, "ty": ty}

    while True:
        i = idx[0]
        pair = pairs[i]
        rgb = cv2.imread(pair["rgb"])
        thermal = cv2.imread(pair["thermal"])
        if rgb is None or thermal is None:
            cv2.setTrackbarPos("index", win, min(i + 1, n - 1))
            continue
        H, W = rgb.shape[0], rgb.shape[1]
        canvas = thermal_to_rgb_canvas(
            thermal, (H, W),
            state["scale"], state["angle"], state["tx"], state["ty"],
            out_channels=1,
        )
        alpha = alpha_val[0] / 100.0
        disp = overlay_thermal_on_rgb(rgb, canvas, alpha=alpha)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(disp, f"{pair['stem']} [{i+1}/{n}] scale={state['scale']:.2f} angle={state['angle']:.1f} tx={state['tx']:.0f} ty={state['ty']:.0f}", (10, 30), font, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, disp)
        cv2.setTrackbarPos("index", win, i)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q") or key == ord("Q"):
            break
        if key == ord("+") or key == ord("="):
            state["scale"] = min(5.0, state["scale"] + STEP_SCALE)
        if key == ord("-"):
            state["scale"] = max(0.1, state["scale"] - STEP_SCALE)
        if key == ord("["):
            state["angle"] -= STEP_ANGLE
        if key == ord("]"):
            state["angle"] += STEP_ANGLE
        if key == 81 or key == 2:  # left
            state["tx"] -= STEP_TX
        if key == 83 or key == 3:  # right
            state["tx"] += STEP_TX
        if key == 82 or key == 0:  # up
            state["ty"] -= STEP_TY
        if key == 84 or key == 1:  # down
            state["ty"] += STEP_TY
        if key == ord("s") or key == ord("S"):
            p = Path(params_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump({
                    "scale": state["scale"],
                    "angle": state["angle"],
                    "tx": state["tx"],
                    "ty": state["ty"],
                }, f, indent=2)
            print(f"Saved params to {p}")
        if key == ord("b") or key == ord("B"):
            run_batch(
                rgb_dir=rgb_dir,
                thermal_dir=thermal_dir,
                output_dir=OUTPUT_DIR,
                params_path=Path(params_path),
                params={
                    "scale": state["scale"],
                    "angle": state["angle"],
                    "tx": state["tx"],
                    "ty": state["ty"],
                },
            )
            print("Batch done.")

    cv2.destroyAllWindows()


def run_batch(
    rgb_dir: Path = RGB_DIR,
    thermal_dir: Path = THERMAL_DIR,
    output_dir: Optional[Path] = None,
    params_path: Path = PARAMS_FILE,
    params: Optional[dict] = None,
    out_channels: int = 1,
) -> List[str]:
    """
    저장된(또는 인자로 준) scale, angle, tx, ty로 전체 thermal에 대해
    RGB 크기 캔버스에 0 패딩 + 해당 영역만 thermal 적용한 이미지 저장.
    """
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if params is None:
        if not Path(params_path).exists():
            raise FileNotFoundError(f"Params not found: {params_path}. Run interactive first and press S.")
        with open(params_path) as f:
            params = json.load(f)
    scale = params["scale"]
    angle = params["angle"]
    tx = params["tx"]
    ty = params["ty"]

    pairs = build_pairs(rgb_dir, thermal_dir)
    out_paths = []
    for pair in pairs:
        rgb = cv2.imread(pair["rgb"])
        thermal = cv2.imread(pair["thermal"])
        if rgb is None or thermal is None:
            continue
        H, W = rgb.shape[0], rgb.shape[1]
        canvas = thermal_to_rgb_canvas(thermal, (H, W), scale, angle, tx, ty, out_channels=out_channels)
        stem = pair["stem"]
        out_path = output_dir / f"{stem}_thermal.png"
        cv2.imwrite(str(out_path), canvas)
        out_paths.append(str(out_path))
    return out_paths


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Thermal alignment and batch process")
    p.add_argument("--interactive", "-i", action="store_true", help="Run interactive alignment (default)")
    p.add_argument("--batch", "-b", action="store_true", help="Run batch with saved params")
    p.add_argument("--rgb-dir", type=str, default=None)
    p.add_argument("--thermal-dir", type=str, default=None)
    p.add_argument("--output-dir", "-o", type=str, default=None)
    p.add_argument("--params", type=str, default=None)
    args = p.parse_args()
    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else RGB_DIR
    thermal_dir = Path(args.thermal_dir) if args.thermal_dir else THERMAL_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    params_path = Path(args.params) if args.params else PARAMS_FILE

    if args.batch:
        run_batch(rgb_dir=rgb_dir, thermal_dir=thermal_dir, output_dir=output_dir, params_path=params_path)
    else:
        run_interactive(rgb_dir=rgb_dir, thermal_dir=thermal_dir, params_path=params_path)
