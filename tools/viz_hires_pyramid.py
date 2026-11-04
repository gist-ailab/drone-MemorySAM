#!/usr/bin/env python3
"""DRN-261104-02: ReliaDINO 고해상 경로 특징 시각화와 H-V1 노름 비율을 계산한다.
픽셀 헤드의 s4 입력을 (b) FPN + (c) taps + (d) high-res path = (e) sum으로 분해한다.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402


SOURCES = ("b", "c", "d", "e")
TITLES = ("(a) input", "(b) FPN", "(c) taps", "(d) high-res path", "(e) sum")


def _ratio(thin_sum, thin_count, rest_sum, rest_count):
    if not thin_count or not rest_count or rest_sum == 0:
        return None
    return (thin_sum / thin_count) / (rest_sum / rest_count)


class RatioTotals:
    def __init__(self):
        self.thin_sum = 0.0
        self.rest_sum = 0.0
        self.n_thin_px = 0
        self.n_rest_px = 0
        self.n_images_with_thin = 0
        self.ratios = []

    def add(self, norm, thin, rest):
        n_thin = int(thin.sum())
        n_rest = int(rest.sum())
        thin_sum = float(norm[thin].sum(dtype=np.float64))
        rest_sum = float(norm[rest].sum(dtype=np.float64))
        ratio = _ratio(thin_sum, n_thin, rest_sum, n_rest)
        self.thin_sum += thin_sum
        self.rest_sum += rest_sum
        self.n_thin_px += n_thin
        self.n_rest_px += n_rest
        self.n_images_with_thin += int(n_thin > 0)
        if ratio is not None:
            self.ratios.append(ratio)
        return (thin_sum / n_thin if n_thin else None,
                rest_sum / n_rest if n_rest else None, ratio)

    def summary(self):
        return {
            "ratio_mean_over_images": float(np.mean(self.ratios)) if self.ratios else None,
            "ratio_pooled": _ratio(self.thin_sum, self.n_thin_px,
                                   self.rest_sum, self.n_rest_px),
            "n_images_with_thin": self.n_images_with_thin,
        }


def _native_norm(feature, size):
    norm = feature.float().norm(dim=1, keepdim=True)
    norm = F.interpolate(norm, size=size, mode="bilinear", align_corners=False)
    return norm[0, 0].cpu().numpy()


def _pca_rgb(feature, size):
    # PCA is fit only to this image's s4 pixels; never mix images or sources.
    flat = feature[0].float().permute(1, 2, 0).reshape(-1, feature.shape[1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(flat, q=3, center=False)
    projected = flat @ components[:, :3]
    colors = []
    for channel in projected.unbind(dim=1):
        lo, hi = torch.quantile(channel, torch.tensor([0.01, 0.99], device=channel.device))
        colors.append(((channel.clamp(lo, hi) - lo) / (hi - lo).clamp_min(1e-12)))
    rgb = torch.stack(colors, dim=1).T.reshape(1, 3, *feature.shape[-2:])
    rgb = F.interpolate(rgb, size=size, mode="nearest")
    return (rgb[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def _save_panel(out_root, image_id, img_path, gt, features, norms):
    panel_dir = out_root / "panels" / image_id
    panel_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.shape[:2] != gt.shape:
        raise ValueError(f"RGB/GT 크기 불일치: {image_id}: {rgb.shape[:2]} vs {gt.shape}")
    palette = common.get_palette()
    gt_rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = gt != 255
    gt_rgb[valid] = palette[gt[valid]]
    Image.fromarray(rgb).save(panel_dir / "rgb.png")
    Image.fromarray(gt_rgb).save(panel_dir / "gt.png")

    norm_images, pca_images = [], []
    cmap = plt.get_cmap("viridis")
    for source in SOURCES:
        norm = norms[source]
        p99 = float(np.percentile(norm, 99))
        scaled = np.clip(norm / p99, 0, 1) if p99 > 0 else np.zeros_like(norm)
        colored = (cmap(scaled)[..., :3] * 255).round().astype(np.uint8)
        pca = _pca_rgb(features[source], gt.shape)
        Image.fromarray(colored).save(panel_dir / f"{source}_norm.png")
        Image.fromarray(pca).save(panel_dir / f"{source}_pca.png")
        norm_images.append(colored)
        pca_images.append(pca)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for column, title in enumerate(TITLES):
        axes[0, column].set_title(title)
    for axis, image in zip(axes.flat, [rgb, *norm_images, gt_rgb, *pca_images]):
        axis.imshow(image)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(panel_dir / "grid.png", dpi=150)
    plt.close(fig)


def _read_ids(path):
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
           if line.strip()]
    if not ids:
        raise ValueError("--ids-file에 image_id가 없습니다")
    if len(ids) != len(set(ids)):
        raise ValueError("--ids-file에 중복 image_id가 있습니다")
    return set(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True, help="ReliaDINO DELIVER 평가 YAML")
    ap.add_argument("--model_path", required=True, help="체크포인트 경로")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--out", required=True, help="통계 및 패널 출력 디렉터리")
    ap.add_argument("--mode", default="stats", choices=("stats", "panels", "both"))
    ap.add_argument("--ids-file", help="패널 대상 image_id 목록, 한 줄에 하나")
    ap.add_argument("--limit", type=int, default=0, help="통계 대상 앞 N장 (0=전체)")
    ap.add_argument("--thin", default="Pole,TrafficLight,Pedestrian,Static",
                    help="얇은 객체 클래스 이름, 쉼표 구분")
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("--limit은 0 이상이어야 합니다")
    if args.mode in ("panels", "both") and not args.ids_file:
        ap.error("--mode panels/both에는 --ids-file이 필요합니다")
    thin_names = [name.strip() for name in args.thin.split(",") if name.strip()]
    if not thin_names or len(thin_names) != len(set(thin_names)):
        ap.error("--thin에 중복되지 않은 클래스 이름을 지정하세요")
    unknown = set(thin_names) - set(common.CLASSES)
    if unknown:
        ap.error(f"알 수 없는 클래스: {sorted(unknown)}")
    selected = _read_ids(args.ids_file) if args.mode in ("panels", "both") else set()
    want_stats = args.mode in ("stats", "both")
    out_root = Path(args.out)

    # val.py has heavy imports, so load it only after argument validation.
    import val as valmod

    with open(args.cfg, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg["MODEL"].get("NAME") != "ReliaDINO" or cfg["DATASET"].get("NAME", "").upper() != "DELIVER":
        raise ValueError("ReliaDINO/DELIVER 평가 config가 필요합니다")
    dataset_cfg = cfg["DATASET"]
    eval_cfg = cfg["EVAL"]
    image_size = (eval_cfg["IMAGE_SIZE"] if args.split == "val" else
                  cfg.get("TEST", {}).get("IMAGE_SIZE", eval_cfg["IMAGE_SIZE"]))
    transform = valmod.get_val_augmentation(image_size, dataset_cfg=dataset_cfg)
    dataset, has_gt = valmod.create_dataset(dataset_cfg, args.split, transform,
                                            args.split, macvi=False, eval_day=False)
    if not has_gt:
        raise ValueError("native GT가 필요한 작업입니다")
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=False,
                        collate_fn=valmod._collate_fn)
    device = torch.device(cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    valmod.setup_cudnn()
    model = valmod.load_model(cfg, Path(args.model_path), device).eval()
    if (model.taps_mode != "per_modal" or model.p43_lateral_off or
            model.detail is not None or model.boundary_refine is not None or
            model.hires is None or model.hires_infer_mode != "on"):
        raise ValueError("분해에는 per_modal taps, 활성 high-res path, 비활성 detail/boundary가 필요합니다")
    assert model.p43_lateral_levels[0] == 0
    if sum(level == 0 for level in model.p43_lateral_levels) != 1:
        raise ValueError("s4에 더해지는 tap이 정확히 하나여야 합니다")

    captured = {}

    def fpn_hook(_module, _inputs, output):
        captured["b"] = output[0].detach()

    def head_pre_hook(_module, inputs):
        captured["e"] = inputs[0][0].detach()

    hooks = [model.fpn.register_forward_hook(fpn_hook),
             model.head.register_forward_pre_hook(head_pre_hook)]
    totals = defaultdict(RatioTotals)
    rows = []
    seen = set()
    found = set()
    decomposition_check = None
    n_images = 0
    thin_ids = [common.CLASSES.index(name) for name in thin_names]

    try:
        with torch.no_grad():
            for images, _labels, metas in tqdm(loader, desc=f"hires [{args.split}]"):
                meta = metas[0]
                img_path = meta["paths"]["img"]
                image_id = common.image_id_from_rel(img_path)
                if image_id in seen:
                    raise RuntimeError(f"중복 image_id: {image_id}")
                seen.add(image_id)
                do_stats = want_stats and (not args.limit or n_images < args.limit)
                do_panel = image_id in selected
                if not do_stats and not do_panel:
                    if args.mode == "stats" or (args.limit and found == selected):
                        break
                    continue

                captured.clear()
                images = [image.to(device) for image in images]
                model(images, multimask_output=True)
                if set(captured) != {"b", "e"}:
                    raise RuntimeError("FPN/head hook이 s4 특징을 캡처하지 못했습니다")
                b, e = captured["b"], captured["e"]
                if model._p43_taps is None or model._hires_feats is None:
                    raise RuntimeError("tap 또는 high-res 특징이 없습니다")
                c = F.interpolate(model._p43_taps[0], size=b.shape[-2:],
                                  mode="bilinear", align_corners=False).detach()
                d = model._hires_feats[0].detach()
                if d.shape[-2:] != b.shape[-2:]:
                    d = F.interpolate(d, size=b.shape[-2:], mode="bilinear",
                                      align_corners=False)
                if not (b.shape == c.shape == d.shape == e.shape):
                    raise RuntimeError(f"분해 텐서 크기 불일치: {[x.shape for x in (b, c, d, e)]}")
                if decomposition_check is None:
                    denom = e.float().abs().max()
                    decomposition_check = float((b.float() + c.float() + d.float() - e.float()
                                                 ).abs().max() / denom)
                    print(f"decomposition_check={decomposition_check:.8g}", flush=True)
                    assert decomposition_check < 1e-3, "s4 분해 검증 실패"

                gt_tensor = meta.get("orig_label")
                if gt_tensor is None:
                    raise RuntimeError(f"native GT 없음: {image_id}")
                gt = gt_tensor.cpu().numpy().astype(np.uint8)
                features = dict(zip(SOURCES, (b, c, d, e)))
                norms = {source: _native_norm(feature, gt.shape)
                         for source, feature in features.items()}
                if do_stats:
                    parts = image_id.split("/")
                    if len(parts) < 2:
                        raise ValueError(f"condition을 읽을 수 없는 image_id: {image_id}")
                    condition = parts[1]
                    thin = np.isin(gt, thin_ids)
                    valid = gt != 255
                    rest = valid & ~thin
                    class_masks = {name: gt == common.CLASSES.index(name)
                                   for name in thin_names}
                    row = {"image_id": image_id, "condition": condition,
                           "n_thin_px": int(thin.sum()), "n_rest_px": int(rest.sum())}
                    for source, norm in norms.items():
                        thin_mean, rest_mean, ratio = totals[("all", "", source)].add(norm, thin, rest)
                        totals[("condition", condition, source)].add(norm, thin, rest)
                        row[f"{source}_thin_mean"] = thin_mean if thin_mean is not None else ""
                        row[f"{source}_rest_mean"] = rest_mean if rest_mean is not None else ""
                        row[f"{source}_ratio"] = ratio if ratio is not None else ""
                        for name in thin_names:
                            class_mask = class_masks[name]
                            totals[("class", name, source)].add(norm, class_mask,
                                                                   valid & ~class_mask)
                    rows.append(row)
                    n_images += 1
                if do_panel:
                    _save_panel(out_root, image_id, img_path, gt, features, norms)
                    found.add(image_id)
                if args.mode == "stats" and args.limit and n_images >= args.limit:
                    break
                if args.mode == "both" and args.limit and n_images >= args.limit and found == selected:
                    break
    finally:
        for hook in hooks:
            hook.remove()

    missing = selected - found
    if missing:
        raise ValueError(f"--ids-file의 image_id를 찾지 못했습니다: {sorted(missing)}")
    if want_stats:
        if not n_images:
            raise RuntimeError("통계를 계산할 이미지가 없습니다")
        out_root.mkdir(parents=True, exist_ok=True)
        fields = ["image_id", "condition", "n_thin_px", "n_rest_px"]
        fields += [f"{source}_{metric}" for source in SOURCES
                   for metric in ("thin_mean", "rest_mean", "ratio")]
        with (out_root / "stats_per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        conditions = sorted({row["condition"] for row in rows})
        summary = {
            "decomposition_check": decomposition_check,
            "cfg": str(args.cfg), "ckpt": str(args.model_path),
            "split": args.split, "n_images": n_images,
            "sources": {
                source: {
                    **totals[("all", "", source)].summary(),
                    "per_class": {name: totals[("class", name, source)].summary()
                                  for name in thin_names},
                    "per_condition": {name: totals[("condition", name, source)].summary()
                                      for name in conditions},
                } for source in SOURCES
            },
        }
        (out_root / "stats_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"processed stats={n_images}, panels={len(found)}, decomposition={decomposition_check}")


if __name__ == "__main__":
    main()
