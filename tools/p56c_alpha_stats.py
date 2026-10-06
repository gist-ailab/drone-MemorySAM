#!/usr/bin/env python3
"""P56-C test 이미지의 센서별 LoRA 혼합 계수 α 분포를 측정한다."""

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEATHER = ("cloud", "fog", "night", "rain", "sun")
CASES = ("motionblur", "overexposure", "underexposure", "lidarjitter", "eventlowres")


class Moments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.sumsq = 0.0

    def add(self, count, total, sumsq):
        self.count += int(count)
        self.total += float(total)
        self.sumsq += float(sumsq)

    @property
    def mean(self):
        return self.total / self.count if self.count else None

    @property
    def std(self):
        if not self.count:
            return None
        return math.sqrt(max(0.0, self.sumsq / self.count - self.mean ** 2))


class AlphaStats:
    def __init__(self, modal_names):
        self.modal_names = list(modal_names)
        self.global_stats = {m: Moments() for m in modal_names}
        self.limits = {m: [math.inf, -math.inf] for m in modal_names}
        self.hist = {m: [0] * 10 for m in modal_names}
        self.image_means = {m: Moments() for m in modal_names}
        self.within_sums = {m: [0.0, 0] for m in modal_names}
        self.conditions = {}
        self.current = {}

    def hook(self, _module, inputs, output):
        idx = int(inputs[1])
        if not 0 <= idx < len(self.modal_names):
            raise RuntimeError(f"라우터 modality_idx 범위 오류: {idx}")
        name = self.modal_names[idx]
        # α 전체를 보관하지 않고 호출마다 CPU에서 집계한다.
        a = output.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if not a.numel() or not torch.isfinite(a).all():
            raise RuntimeError(f"{name} 라우터 α가 비었거나 비유한값을 포함한다")
        if a.min().item() < 0.0 or a.max().item() > 1.0:
            raise RuntimeError(f"{name} 라우터 α가 [0, 1] 밖이다")
        values = a.double()
        n, total, sumsq = a.numel(), values.sum().item(), values.square().sum().item()
        self.global_stats[name].add(n, total, sumsq)
        self.current.setdefault(name, Moments()).add(n, total, sumsq)
        self.limits[name][0] = min(self.limits[name][0], a.min().item())
        self.limits[name][1] = max(self.limits[name][1], a.max().item())
        bins = torch.clamp((a * 10).long(), 0, 9)
        counts = torch.bincount(bins, minlength=10).tolist()
        self.hist[name] = [old + new for old, new in zip(self.hist[name], counts)]

    def finish_image(self, path):
        missing = set(self.modal_names) - set(self.current)
        if missing:
            raise RuntimeError(f"이미지 {path}: 라우터가 호출되지 않은 센서 {sorted(missing)}")
        weather, case = path_conditions(path)
        # 날씨와 결함 케이스를 각각 조건 축으로 센다.
        for name, stats in self.current.items():
            mean, std = stats.mean, stats.std
            self.image_means[name].add(1, mean, mean * mean)
            self.within_sums[name][0] += std
            self.within_sums[name][1] += 1
            for condition in (weather, case):
                if condition is None:
                    continue
                slot = self.conditions.setdefault(condition, {}).setdefault(
                    name, {"mean_sum": 0.0, "std_sum": 0.0, "n": 0})
                slot["mean_sum"] += mean
                slot["std_sum"] += std
                slot["n"] += 1
        self.current.clear()

    def report(self, ckpt, md5, n_images):
        per_modality = {}
        for name in self.modal_names:
            moments = self.global_stats[name]
            if not moments.count:
                raise RuntimeError(f"{name} 라우터 α 측정값이 없다")
            per_modality[name] = {
                "count": moments.count,
                "mean": moments.mean,
                "token_std_global": moments.std,
                "within_image_token_std_mean": (self.within_sums[name][0] /
                                                self.within_sums[name][1]),
                "between_image_std": self.image_means[name].std,
                "min": self.limits[name][0], "max": self.limits[name][1],
                "hist": self.hist[name],
            }
        per_condition = {}
        for condition, modals in sorted(self.conditions.items()):
            per_condition[condition] = {
                name: {"mean": slot["mean_sum"] / slot["n"],
                       "within_image_token_std_mean": slot["std_sum"] / slot["n"],
                       "n": slot["n"]}
                for name, slot in modals.items()
            }
        return {"ckpt": ckpt, "md5": md5, "n_images": n_images,
                "per_modality": per_modality, "per_condition": per_condition,
                "gate": {"alpha_token_std_ge_0.05": all(
                    row["token_std_global"] >= 0.05 for row in per_modality.values())}}


def path_conditions(path):
    parts = [p.lower() for p in Path(path).parts]
    weather = next((c for c in WEATHER if c in parts), None)
    # 케이스는 semseg/datasets/deliver.py 와 같은 규칙(경로 문자열 포함 여부)으로 판정한다.
    low = str(path).lower()
    case = next((c for c in CASES if c in low), "none")
    return weather, case


def file_md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(cfg, model_path, device, smoke):
    from semseg.models.reliadino import build_reliadino

    model_cfg = copy.deepcopy(cfg)
    model_cfg["MODEL"]["PRETRAINED_BACKBONE"] = False
    if smoke:
        # 같은 P56-C 라우터를 작은 무작위 백본에 연결한다.
        model_cfg = {
            "MODEL": {
                "BACKBONE_TIMM": "vit_tiny_patch16_224",
                "BACKBONE_FALLBACK": "vit_tiny_patch16_224",
                "PRETRAINED_BACKBONE": False,
                "LORA_R": 2, "LORA_MODE": "state_routed",
                "LORA_ROUTER": copy.deepcopy(cfg["MODEL"]["LORA_ROUTER"]),
                "FPN_DIM": 64,
                "FUSION": {"NUM_LAYERS": 1, "NUM_HEADS": 4, "MLP_RATIO": 1.0,
                           "AUX_HIDDEN": 32, "ATTN_BIAS": {"ENABLE": False}},
            },
            "DATASET": {"MODALS": list(cfg["DATASET"]["MODALS"])},
            "TRAIN": {"IMAGE_SIZE": [64, 64]},
        }
    dataset_cfg = cfg["DATASET"]
    n_classes = cfg["MODEL"].get("LORA_NUM_CLASSES", dataset_cfg.get("NUM_CLASSES", 25))
    model = build_reliadino(model_cfg, n_classes)
    if model.p56c_router is None:
        raise ValueError("MODEL.LORA_MODE=state_routed 인 P56-C config가 필요하다")
    if not smoke:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            raise KeyError("체크포인트에 model_state_dict 키가 없다")
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[p56c] checkpoint strict=True: missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
    if hasattr(model, "_current_epoch"):
        model._current_epoch = 9999
    return model.to(device).eval()


def print_report(report, out):
    print(f"[p56c] ckpt={report['ckpt']} md5={report['md5']} "
          f"images={report['n_images']}")
    print("모달 | tokens | mean | global std | 이미지내 std 평균 | 이미지간 std | min | max | hist[0..9]")
    for name, row in report["per_modality"].items():
        print(f"{name} | {row['count']} | {row['mean']:.5f} | "
              f"{row['token_std_global']:.5f} | "
              f"{row['within_image_token_std_mean']:.5f} | "
              f"{row['between_image_std']:.5f} | {row['min']:.5f} | "
              f"{row['max']:.5f} | {row['hist']}")
    print("조건 | 모달 | 이미지 수 | mean | 이미지내 std 평균")
    for condition, modals in report["per_condition"].items():
        for name, row in modals.items():
            print(f"{condition} | {name} | {row['n']} | {row['mean']:.5f} | "
                  f"{row['within_image_token_std_mean']:.5f}")
    print(f"gate alpha_token_std_ge_0.05: "
          f"{report['gate']['alpha_token_std_ge_0.05']}")
    print(f"[p56c] JSON: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--model_path")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--max_images", type=int, default=400)
    parser.add_argument("--subset_every", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and not args.model_path:
        parser.error("--model_path 는 실제 측정에 필요하다")
    if args.max_images < 1 or args.subset_every < 1:
        parser.error("--max_images 와 --subset_every 는 양수여야 한다")
    if not args.smoke and args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA를 사용할 수 없다. --device cpu 를 지정하거나 CUDA를 확인하라")

    with open(args.cfg, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg["DATASET"]["NAME"].upper() != "DELIVER":
        parser.error("DELIVER config만 지원한다")
    if cfg["MODEL"].get("LORA_MODE") != "state_routed":
        parser.error("P56-C MODEL.LORA_MODE=state_routed 가 필요하다")
    device = torch.device("cpu" if args.smoke else args.device)

    if args.smoke:
        records = [
            (f"/img/{WEATHER[i]}/test/none/smoke_{i}.png",
             [torch.randn(1, 3, 64, 64) for _ in cfg["DATASET"]["MODALS"]])
            for i in range(2)
        ]
    else:
        import val as valmod

        valmod.setup_cudnn()
        image_size = cfg.get("TEST", {}).get("IMAGE_SIZE", cfg["EVAL"]["IMAGE_SIZE"])
        transform = valmod.get_val_augmentation(image_size, dataset_cfg=cfg["DATASET"])
        dataset, _ = valmod.create_dataset(cfg["DATASET"], args.split, transform,
                                           args.split, macvi=False, eval_day=False)
        indices = list(range(0, len(dataset.files), args.subset_every))[:args.max_images]
        if not indices:
            raise RuntimeError("선택된 test 이미지가 없다")
        selected = Subset(dataset, indices)
        loader = DataLoader(selected, batch_size=1, num_workers=0,
                            pin_memory=False, collate_fn=valmod._collate_fn)
        records = ((metas[0]["paths"]["img"], [x.to(device) for x in images])
                   for images, _labels, metas in loader)

    model = load_model(cfg, args.model_path, device, args.smoke)
    stats = AlphaStats(cfg["DATASET"]["MODALS"])
    handle = model.p56c_router.register_forward_hook(stats.hook)
    amp_name = str(cfg.get("TRAIN", {}).get("AMP_DTYPE", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_name in ("bf16", "bfloat16") else torch.float16
    use_amp = bool(cfg.get("TRAIN", {}).get("AMP", False)) and device.type == "cuda"
    n_images = 0
    try:
        with torch.no_grad():
            for path, inputs in records:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    model(inputs, multimask_output=True)
                stats.finish_image(path)
                n_images += 1
                if n_images % 50 == 0:
                    print(f"[p56c] {n_images} images", flush=True)
    finally:
        handle.remove()
    report = stats.report(None if args.smoke else str(args.model_path),
                          None if args.smoke else file_md5(args.model_path), n_images)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_report(report, out)


if __name__ == "__main__":
    main()
