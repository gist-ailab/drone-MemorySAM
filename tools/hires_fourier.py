#!/usr/bin/env python3
"""Fourier relative log-amplitude curves for ReliaDINO pixel-head s4 features.

The hooks and b/c/d/e decomposition follow tools/viz_hires_pyramid.py.
Curves use the shifted 2D FFT log amplitude relative to the zero frequency.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _log_amp(feat):
    if feat.ndim != 4 or feat.shape[0] != 1:
        raise ValueError(f"expected feature shape (1, C, H, W), got {tuple(feat.shape)}")
    h, w = feat.shape[-2:]
    k = min(h, w) // 2
    if k < 2:
        raise ValueError(f"feature is too small for a frequency curve: {tuple(feat.shape)}")
    spectrum = torch.fft.fftshift(torch.fft.fft2(feat.float()), dim=(-2, -1))
    return torch.log(spectrum.abs() + 1e-12), k


def rel_log_amp_diag(feat):
    """Channel-mean shifted FFT log amplitude along the lower-right diagonal."""
    logamp, k = _log_amp(feat)
    h, w = feat.shape[-2:]
    offsets = torch.arange(k, device=feat.device)
    curve = logamp[0, :, h // 2 + offsets, w // 2 + offsets].mean(dim=0)
    return curve - curve[0]


def rel_log_amp_radial(feat):
    """Channel- and ring-mean shifted FFT log amplitude by integer radius."""
    logamp, k = _log_amp(feat)
    h, w = feat.shape[-2:]
    yy = torch.arange(h, device=feat.device) - h // 2
    xx = torch.arange(w, device=feat.device) - w // 2
    radius = torch.floor(torch.sqrt(yy[:, None].float().square()
                                    + xx[None, :].float().square())).long()
    valid = radius < k
    ring = radius[valid]
    pixel_mean = logamp[0].mean(dim=0)[valid]
    sums = torch.bincount(ring, weights=pixel_mean, minlength=k)
    counts = torch.bincount(ring, minlength=k)
    curve = sums / counts
    return curve - curve[0]


def _selftest():
    torch.manual_seed(0)
    constant = torch.ones(1, 32, 64, 64)
    noise = torch.randn(1, 64, 64, 64)
    lowpass = F.interpolate(F.avg_pool2d(noise, kernel_size=4, stride=4),
                            size=noise.shape[-2:], mode="bilinear", align_corners=False)
    for curve_fn in (rel_log_amp_diag, rel_log_amp_radial):
        constant_curve = curve_fn(constant)
        white_curve = curve_fn(noise)
        low_curve = curve_fn(lowpass)
        assert (constant_curve[1:] < 0).all(), f"{curve_fn.__name__}: constant spectrum"
        assert abs(float(white_curve.mean())) < 1.0, f"{curve_fn.__name__}: white noise"
        high = torch.arange(white_curve.numel()) / white_curve.numel() >= 0.5
        assert low_curve[high].mean() < white_curve[high].mean(), (
            f"{curve_fn.__name__}: low-pass high-frequency band")
    print("SELFTEST_OK")


def _file_md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_indices(length, num):
    if length < 1:
        raise ValueError("dataset is empty")
    return sorted(set(np.linspace(0, length - 1, num).round().astype(int).tolist()))


def _run(args):
    # val.py has heavy imports; the standalone selftest never imports it.
    import yaml
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    import val as valmod
    from tools.baseline_failure import common

    with Path(args.cfg).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg["MODEL"].get("NAME") != "ReliaDINO" or cfg["DATASET"].get("NAME", "").upper() != "DELIVER":
        raise ValueError("ReliaDINO/DELIVER evaluation config is required")
    dataset_cfg = cfg["DATASET"]
    eval_cfg = cfg["EVAL"]
    image_size = (eval_cfg["IMAGE_SIZE"] if args.split == "val" else
                  cfg.get("TEST", {}).get("IMAGE_SIZE", eval_cfg["IMAGE_SIZE"]))
    transform = valmod.get_val_augmentation(image_size, dataset_cfg=dataset_cfg)
    dataset, _has_gt = valmod.create_dataset(dataset_cfg, args.split, transform,
                                             args.split, macvi=False, eval_day=False)
    indices = _selected_indices(len(dataset), args.num)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=4,
                        pin_memory=False, collate_fn=valmod._collate_fn)
    device = torch.device(cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    valmod.setup_cudnn()
    model = valmod.load_model(cfg, Path(args.model_path), device).eval()

    if not args.no_hires:
        if (model.taps_mode != "per_modal" or model.p43_lateral_off or
                model.detail is not None or model.boundary_refine is not None or
                model.hires is None or model.hires_infer_mode != "on"):
            raise ValueError("decomposition requires per_modal taps, active high-res path, "
                             "and inactive detail/boundary")
        assert model.p43_lateral_levels[0] == 0
        if sum(level == 0 for level in model.p43_lateral_levels) != 1:
            raise ValueError("exactly one tap must be added at s4")

    captured = {}

    def fpn_hook(_module, _inputs, output):
        captured["b"] = output[0].detach()

    def head_pre_hook(_module, inputs):
        captured["e"] = inputs[0][0].detach()

    hooks = [model.fpn.register_forward_hook(fpn_hook),
             model.head.register_forward_pre_hook(head_pre_hook)]
    curves = defaultdict(list)
    image_ids = []
    seen_ids = set()
    decomposition_check = None
    feature_shape = None
    expected_components = None
    try:
        with torch.no_grad():
            for images, _labels, metas in tqdm(loader, desc=f"fourier [{args.split}]"):
                captured.clear()
                images = [image.to(device) for image in images]
                model(images, multimask_output=True)
                if set(captured) != {"b", "e"}:
                    raise RuntimeError("FPN/head hooks did not capture s4 features")
                b, e = captured["b"], captured["e"]
                features = {"b": b, "e": e}
                if not args.no_hires:
                    if model._p43_taps is None or model._hires_feats is None:
                        raise RuntimeError("tap or high-res features are missing")
                    c = F.interpolate(model._p43_taps[0], size=b.shape[-2:],
                                      mode="bilinear", align_corners=False).detach()
                    d = model._hires_feats[0].detach()
                    if d.shape[-2:] != b.shape[-2:]:
                        d = F.interpolate(d, size=b.shape[-2:], mode="bilinear",
                                          align_corners=False)
                    if not (b.shape == c.shape == d.shape == e.shape):
                        raise RuntimeError(f"decomposition shape mismatch: "
                                           f"{[tuple(x.shape) for x in (b, c, d, e)]}")
                    features.update(c=c, d=d, cd=c + d)
                    if decomposition_check is None:
                        denom = e.float().abs().max().clamp_min(1e-12)
                        decomposition_check = float((b.float() + c.float() + d.float()
                                                     - e.float()).abs().max() / denom)
                        print(f"decomposition_check={decomposition_check:.8g}", flush=True)
                        assert decomposition_check < 1e-3, "s4 decomposition failed"
                else:
                    taps = getattr(model, "_p43_taps", None)
                    if taps is not None and len(taps) > 0 and taps[0] is not None:
                        features["c"] = F.interpolate(taps[0], size=b.shape[-2:],
                                                       mode="bilinear", align_corners=False).detach()

                components = set(features)
                if expected_components is None:
                    expected_components = components
                    feature_shape = list(b.shape)
                elif components != expected_components or list(b.shape) != feature_shape:
                    raise RuntimeError("component set or s4 feature shape changed between images")
                image_id = common.image_id_from_rel(metas[0]["paths"]["img"])
                if image_id in seen_ids:
                    raise RuntimeError(f"duplicate image_id: {image_id}")
                seen_ids.add(image_id)
                image_ids.append(image_id)
                for component, feat in features.items():
                    for kind, fn in (("diag", rel_log_amp_diag),
                                     ("radial", rel_log_amp_radial)):
                        curves[(component, kind)].append(fn(feat).cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected_ids.txt").write_text("".join(f"{x}\n" for x in image_ids),
                                          encoding="utf-8")
    high_frequency_mean = {}
    with (out / "curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("component", "kind", "freq", "mean", "std", "n_images"))
        for component in ("b", "c", "d", "cd", "e"):
            for kind in ("diag", "radial"):
                samples = curves.get((component, kind))
                if not samples:
                    continue
                values = np.stack(samples).astype(np.float64)
                mean = values.mean(axis=0)
                std = values.std(axis=0)
                freq = np.arange(values.shape[1]) / values.shape[1]
                high_frequency_mean.setdefault(component, {})[kind] = float(mean[freq >= 0.5].mean())
                for x, y, s in zip(freq, mean, std):
                    writer.writerow((component, kind, float(x), float(y), float(s), len(samples)))
    summary = {
        "cfg": str(args.cfg), "ckpt_md5": _file_md5(args.model_path),
        "n_images": len(image_ids), "feature_shape": feature_shape,
        "decomposition_check": decomposition_check,
        "high_frequency_mean": high_frequency_mean,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(f"processed n_images={len(image_ids)}, decomposition={decomposition_check}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", help="ReliaDINO DELIVER evaluation YAML")
    parser.add_argument("--model_path", help="checkpoint path")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--num", type=int, default=100, help="number of evenly spaced images")
    parser.add_argument("--no-hires", action="store_true", help="analyze model without high-res path")
    parser.add_argument("--selftest", action="store_true", help="run CPU-only Fourier assertions")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    for required in ("cfg", "model_path", "out"):
        if getattr(args, required) is None:
            parser.error(f"--{required} is required unless --selftest is used")
    if args.num < 1:
        parser.error("--num must be positive")
    _run(args)


if __name__ == "__main__":
    main()
