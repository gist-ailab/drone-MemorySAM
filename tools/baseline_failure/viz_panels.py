#!/usr/bin/env python3
"""
D5 — 실패 사례 패널 시각화. D3-2 목록의 상위 N 장에 대해
`[RGB | Depth | Event | LiDAR | GT | <모델별 예측...> | <모델별 오류맵...>]`
패널 PNG 를 만든다. 팔레트는 리포 DELIVER 팔레트를, 모달 로더는 deliver.py 경로
규칙(img→hha/lidar/event/semantic, _rgb→_depth/_lidar/_event/_semantic)을 그대로 쓴다.

예:
  python tools/baseline_failure/viz_panels.py \
    --list-json .../mining/d32_top_lists.json --topn 12 \
    --deliver-root /ailab_mat2/dataset/DELIVER --split test \
    --gt .../ours/test/gt \
    --models ours=.../ours/test/pred dgf80k=... caf=... \
    --out .../mining
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.baseline_failure import common  # noqa: E402

PALETTE = common.get_palette()   # (25,3) uint8


def rgb_path_from_id(deliver_root, split, image_id):
    """image_id(데이터셋 루트 기준 상대 경로) → DELIVER RGB 경로.

    새 규약에서 image_id = `img/<cond>/<split>/<scene>/<stem>_rgb_front` 이므로 RGB
    경로는 `<deliver_root>/<image_id>.png` 로 곧장 복원된다(split 인자는 상대 경로에
    이미 들어 있어 별도로 쓰지 않는다).
    """
    rel = image_id if image_id.startswith("img/") else f"img/{image_id}"
    return str(Path(deliver_root) / f"{rel}.png")


def modality_paths(rgb_path):
    """deliver.py __getitem__ 의 경로 치환 규칙(141~144행)을 그대로 재현."""
    return {
        "img": rgb_path,
        "depth": rgb_path.replace("/img", "/hha").replace("_rgb", "_depth"),
        "lidar": rgb_path.replace("/img", "/lidar").replace("_rgb", "_lidar"),
        "event": rgb_path.replace("/img", "/event").replace("_rgb", "_event"),
    }


def load_rgb(path, shape=None):
    from PIL import Image
    if not Path(path).exists():
        return np.zeros((shape[0], shape[1], 3), np.uint8) if shape else np.zeros((64, 64, 3), np.uint8)
    img = np.array(Image.open(path).convert("RGB"))
    if shape and img.shape[:2] != tuple(shape):
        img = np.array(Image.fromarray(img).resize((shape[1], shape[0]), Image.BILINEAR))
    return img


def colorize(label):
    """trainID (H,W) → RGB. 255(ignore)=회색."""
    h, w = label.shape
    out = np.zeros((h, w, 3), np.uint8)
    for c in range(len(PALETTE)):
        out[label == c] = PALETTE[c]
    out[label == 255] = (30, 30, 30)
    return out


def error_map(pred, gt):
    """오답=빨강, 정답=검정, ignore=회색."""
    h, w = gt.shape
    out = np.zeros((h, w, 3), np.uint8)
    valid = gt != 255
    wrong = valid & (pred != gt)
    out[wrong] = (220, 40, 40)
    out[gt == 255] = (30, 30, 30)
    return out


def make_panel(image_id, deliver_root, split, gt_dir, models_dirs, out_path):
    rgb_p = rgb_path_from_id(deliver_root, split, image_id)
    mpaths = modality_paths(rgb_p)
    gt = common.load_label_png(Path(gt_dir) / f"{image_id}.png")
    H, W = gt.shape

    cols = [("RGB", load_rgb(mpaths["img"], (H, W))),
            ("Depth", load_rgb(mpaths["depth"], (H, W))),
            ("Event", load_rgb(mpaths["event"], (H, W))),
            ("LiDAR", load_rgb(mpaths["lidar"], (H, W))),
            ("GT", colorize(gt))]

    preds = {}
    for n, d in models_dirs.items():
        p = common.load_label_png(Path(d) / f"{image_id}.png")
        if p.shape != gt.shape:
            p = common.resize_nearest(p, H, W)
        preds[n] = p
        cols.append((f"pred:{n}", colorize(p)))
    for n in models_dirs:
        cols.append((f"err:{n}", error_map(preds[n], gt)))

    ncol = len(cols)
    fig, axes = plt.subplots(1, ncol, figsize=(2.4 * ncol, 2.8))
    if ncol == 1:
        axes = [axes]
    for ax, (title, im) in zip(axes, cols):
        ax.imshow(im)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(image_id, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _parse_kv(items, flag):
    out = {}
    for spec in items or []:
        if "=" not in spec:
            raise SystemExit(f"{flag} 는 name=dir 형식이어야 함: {spec}")
        k, v = spec.split("=", 1)
        out[k] = Path(v) / "pred" if (Path(v) / "pred").is_dir() else Path(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--models", required=True, nargs="+", metavar="NAME=DIR")
    ap.add_argument("--deliver-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--list-json", default=None, help="d32_top_lists.json")
    ap.add_argument("--ids", nargs="*", default=None, help="직접 image_id 목록")
    ap.add_argument("--topn", type=int, default=12)
    args = ap.parse_args()

    models_dirs = _parse_kv(args.models, "--models")
    out_root = Path(args.out) / "panels"

    tasks = []   # (subdir, image_id)
    if args.ids:
        tasks += [("ids", i) for i in args.ids[:args.topn]]
    if args.list_json:
        lists = json.loads(Path(args.list_json).read_text(encoding="utf-8"))
        for lname, obj in lists.items():
            for it in obj.get("items", [])[:args.topn]:
                tasks.append((lname, it["image_id"]))
    if not tasks:
        raise SystemExit("--list-json 또는 --ids 중 하나는 필요")

    made = 0
    for sub, image_id in tasks:
        out_path = out_root / sub / f"{image_id}.png"
        try:
            make_panel(image_id, args.deliver_root, args.split, args.gt, models_dirs, out_path)
            made += 1
        except Exception as e:  # noqa: BLE001 — 한 장 실패가 전체를 막지 않게.
            print(f"[viz] {image_id} 실패: {e}")
    print(f"[viz_panels] {made}/{len(tasks)} 패널 -> {out_root}")


if __name__ == "__main__":
    main()
