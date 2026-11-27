#!/usr/bin/env python3
"""``viz_fused_stability.py`` 가 두 모델(A·B)에서 낸 패널·통계를 비교 그림으로 합친다.

- 공통 패널 이미지마다 그림 한 장: 행=설정(--rows), 열=[B PCA, A PCA, B 예측, A 예측, GT].
- 막대 그림: 두 모델의 제거 센서별 평균 cosine(오차 막대=표준편차).

색·글꼴은 labdesign 의 carbon_palette.py 를 importlib 로 불러 쓰고(없으면 matplotlib 기본),
그림마다 <파일>.json 사이드카로 design_source 를 남긴다. 학습·모델·기존 도구는 수정하지 않는다.
"""
import argparse
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CARBON_PATH = Path("/home/jemo/cl-jemo/labdesign/build/py/carbon_palette.py")


def load_carbon():
    """carbon_palette.py 를 importlib 로 불러 (module, version) 반환. 없으면 (None, 'unknown')."""
    if not _CARBON_PATH.exists():
        print(f"[warn] carbon_palette.py 없음({_CARBON_PATH}) — matplotlib 기본값 사용", flush=True)
        return None, "unknown"
    spec = importlib.util.spec_from_file_location("carbon_palette", _CARBON_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    version = str(getattr(mod, "VERSION", getattr(mod, "__version__", "unknown")))
    return mod, version


def apply_carbon_rc(carbon):
    if carbon is not None:
        try:
            matplotlib.rcParams.update(carbon.rcparams(theme="light"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] carbon rcparams 적용 실패({exc}) — 기본값", flush=True)


def _categorical(carbon):
    if carbon is not None and getattr(carbon, "CATEGORICAL", None):
        return list(carbon.CATEGORICAL)
    return list(plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e"]))


def _sidecar(path, design_source):
    Path(str(path) + ".json").write_text(
        json.dumps({"design_source": f"carbon@{design_source}"}, ensure_ascii=False),
        encoding="utf-8")


def _get_palette():
    from tools.baseline_failure import common
    return common.get_palette()


def colorize_label(arr, palette):
    """trainID (H,W) uint8 → RGB. 255(ignore)·범위 밖은 검정."""
    arr = np.asarray(arr)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    valid = (arr >= 0) & (arr < len(palette))
    out[valid] = palette[arr[valid]]
    return out


def _load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _load_label(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.uint8)


def _safe_id(image_id):
    return image_id.replace("/", "__").replace("\\", "__")


def list_panels(out_dir):
    """<out>/panels 아래 gt.png 를 가진 image_id(상대 경로) 집합."""
    panels = Path(out_dir) / "panels"
    ids = set()
    if not panels.exists():
        return ids
    for gt in panels.rglob("gt.png"):
        ids.add(gt.parent.relative_to(panels).as_posix())
    return ids


def make_panel_fig(image_id, a_dir, b_dir, rows, palette, out_dir, name_a, name_b,
                   design_source):
    a_panel = Path(a_dir) / "panels" / image_id
    b_panel = Path(b_dir) / "panels" / image_id
    col_titles = [f"{name_b} PCA", f"{name_a} PCA", f"{name_b} pred",
                  f"{name_a} pred", "GT"]
    gt = colorize_label(_load_label(b_panel / "gt.png"), palette)
    n_rows = len(rows)
    rgb_in = _load_rgb(b_panel / "rgb.png") if (b_panel / "rgb.png").exists() else None
    row_names = {"clean": "All sensors", "no_img": "RGB removed", "no_depth": "Depth removed",
                 "no_event": "Event removed", "no_lidar": "LiDAR removed"}
    titles = ["RGB input (reference)", f"{name_b}: fused feature (PCA)", f"{name_a}: fused feature (PCA)",
              f"{name_b}: prediction", f"{name_a}: prediction", "GT"]
    fig, axes = plt.subplots(n_rows, 6, figsize=(6 * 2.6, n_rows * 2.6), squeeze=False)
    for r, setting in enumerate(rows):
        cells = [
            rgb_in if rgb_in is not None else np.ones((8, 8, 3)),
            _load_rgb(b_panel / f"pca_{setting}.png"),
            _load_rgb(a_panel / f"pca_{setting}.png"),
            colorize_label(_load_label(b_panel / f"pred_{setting}.png"), palette),
            colorize_label(_load_label(a_panel / f"pred_{setting}.png"), palette),
            gt,
        ]
        for c, cell in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(cell)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_names.get(setting, setting), fontsize=11)
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
    fig.tight_layout()
    safe = _safe_id(image_id)
    png = Path(out_dir) / f"panel_{safe}.png"
    pdf = Path(out_dir) / f"panel_{safe}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _sidecar(png, design_source)
    _sidecar(pdf, design_source)
    return png


def make_bar_fig(a_summary, b_summary, name_a, name_b, out_dir, colors, design_source):
    removed = a_summary.get("removed_modals") or list(a_summary["per_setting"].keys())
    means_a, std_a, means_b, std_b = [], [], [], []
    for s in removed:
        means_a.append(a_summary["per_setting"][s]["mean_cosine"])
        std_a.append(a_summary["per_setting"][s]["std_cosine"])
        means_b.append(b_summary["per_setting"][s]["mean_cosine"])
        std_b.append(b_summary["per_setting"][s]["std_cosine"])
    x = np.arange(len(removed))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(5, len(removed) * 1.4), 4))
    ax.bar(x - width / 2, means_b, width, yerr=std_b, capsize=3,
           label=name_b, color=colors[0 % len(colors)])
    ax.bar(x + width / 2, means_a, width, yerr=std_a, capsize=3,
           label=name_a, color=colors[1 % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(removed)
    ax.set_ylabel("mean cosine (clean vs removed)")
    ax.set_title("융합 특징 안정성 — 센서 제거별")
    ax.legend()
    fig.tight_layout()
    png = Path(out_dir) / "cosine_bar.png"
    pdf = Path(out_dir) / "cosine_bar.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _sidecar(png, design_source)
    _sidecar(pdf, design_source)
    with (Path(out_dir) / "cosine_bar.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["removed_modal", f"{name_b}_mean", f"{name_b}_std",
                    f"{name_a}_mean", f"{name_a}_std"])
        for i, s in enumerate(removed):
            w.writerow([s, means_b[i], std_b[i], means_a[i], std_a[i]])
    return png


def _run(a_dir, b_dir, name_a, name_b, out_dir, rows):
    carbon, version = load_carbon()
    apply_carbon_rc(carbon)
    colors = _categorical(carbon)
    palette = _get_palette()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_ids = sorted(list_panels(a_dir) & list_panels(b_dir))
    made = []
    for image_id in common_ids:
        made.append(make_panel_fig(image_id, a_dir, b_dir, rows, palette, out_dir,
                                    name_a, name_b, version))

    a_summary = json.loads((Path(a_dir) / "stats_summary.json").read_text(encoding="utf-8"))
    b_summary = json.loads((Path(b_dir) / "stats_summary.json").read_text(encoding="utf-8"))
    bar = make_bar_fig(a_summary, b_summary, name_a, name_b, out_dir, colors, version)
    print(f"panels={len(made)}, bar={bar} -> {out_dir}")
    return made, bar


def _selftest():
    palette = np.arange(25 * 3, dtype=np.uint8).reshape(25, 3)
    rows = ["clean", "no_img", "no_depth"]
    settings = ["clean", "no_img", "no_depth", "no_event", "no_lidar"]
    tmp = Path(tempfile.mkdtemp(prefix="fused_fig_selftest_"))
    for model in ("a", "b"):
        for image_id in ("night/scene1/000001", "sun/scene2/000002"):
            pdir = tmp / model / "panels" / image_id
            pdir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
                pdir / "rgb.png")
            Image.fromarray(np.random.randint(0, 25, (40, 40), dtype=np.uint8)).save(
                pdir / "gt.png")
            for s in settings:
                Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)).save(
                    pdir / f"pca_{s}.png")
                Image.fromarray(np.random.randint(0, 25, (40, 40), dtype=np.uint8)).save(
                    pdir / f"pred_{s}.png")
        summary = {
            "removed_modals": ["no_img", "no_depth", "no_event", "no_lidar"],
            "per_setting": {s: {"mean_cosine": 0.9, "std_cosine": 0.05, "n_images": 2}
                            for s in ["no_img", "no_depth", "no_event", "no_lidar"]},
        }
        (tmp / model / "stats_summary.json").write_text(
            json.dumps(summary), encoding="utf-8")
    out_dir = tmp / "fig"
    # 팔레트 주입(순수 검증용) — _get_palette 대신 합성 팔레트.
    global _get_palette
    _orig = _get_palette
    _get_palette = lambda: palette      # noqa: E731
    try:
        made, bar = _run(tmp / "a", tmp / "b", "A", "B", out_dir, rows)
    finally:
        _get_palette = _orig
    assert len(made) == 2 and all(Path(p).exists() for p in made), "패널 그림 미생성"
    assert Path(bar).exists(), "막대 그림 미생성"
    assert (out_dir / "cosine_bar.csv").exists(), "cosine_bar.csv 미생성"
    assert (out_dir / "cosine_bar.png.json").exists(), "사이드카 미생성"
    print("SELFTEST_OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--a", help="모델 A out 폴더")
    ap.add_argument("--b", help="모델 B out 폴더")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", help="그림 출력 폴더")
    ap.add_argument("--rows", default="clean,no_img,no_depth",
                    help="행에 쓸 설정 이름, 쉼표 구분")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    for name in ("a", "b", "out"):
        if not getattr(args, name):
            ap.error(f"--{name} 가 필요합니다(--selftest 제외)")
    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    if not rows:
        ap.error("--rows 가 비었습니다")
    _run(args.a, args.b, args.name_a, args.name_b, args.out, rows)


if __name__ == "__main__":
    main()
