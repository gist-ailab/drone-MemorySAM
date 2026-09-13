"""Measure per-module max |activation| (fp32, eval mode) for DGFusion checkpoints.

Usage (from the DGFusion repo root, env dgfusion, PYTHONPATH=OneFormer:repo):
    python act_probe.py <ckpt> [<ckpt> ...]
"""
import sys

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser

import test_net

FP16_MAX = 65504.0
N_BATCHES = 8


def iter_tensors(o):
    if torch.is_tensor(o):
        yield o
    elif isinstance(o, (list, tuple)):
        for x in o:
            yield from iter_tensors(x)
    elif isinstance(o, dict):
        for x in o.values():
            yield from iter_tensors(x)


def probe(ckpt, cfg):
    model = test_net.Tester.build_model(cfg)
    DetectionCheckpointer(model).load(ckpt)
    model.eval()
    maxes = {}

    def make_hook(name):
        def hook(_m, _i, out):
            for t in iter_tensors(out):
                if t.is_floating_point() and t.numel():
                    v = t.detach().abs().max().item()
                    if v > maxes.get(name, 0.0):
                        maxes[name] = v
        return hook

    for name, m in model.named_modules():
        if name:
            m.register_forward_hook(make_hook(name))
    loader = test_net.Tester.build_test_loader(cfg, "deliver_semantic_val")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            model(batch)
            if i + 1 >= N_BATCHES:
                break
    return maxes


def main():
    ckpts = sys.argv[1:]
    args = default_argument_parser().parse_args([
        "--config-file", "configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml",
        "--eval-only", "MODEL.IS_TRAIN", "False", "MODEL.TEST.DEPTH_ON", "False",
        "OUTPUT_DIR", "output/act_probe",
    ])
    cfg = test_net.setup(args)
    results = {c: probe(c, cfg) for c in ckpts}
    last = results[ckpts[-1]]
    top = sorted(last.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print(f"\nfp16 max = {FP16_MAX:.0f}; per-module max|act| over {N_BATCHES} val images")
    print("module".ljust(70) + "".join(c.split("model_")[-1][:7].rjust(12) for c in ckpts))
    for name, _ in top:
        row = "".join(f"{results[c].get(name, float('nan')):12.1f}" for c in ckpts)
        print(name[:70].ljust(70) + row)
    overall = {c: max(results[c].values()) for c in ckpts}
    print("\noverall max per ckpt:", {k.split('model_')[-1][:7]: round(v, 1) for k, v in overall.items()})
    print("ratio to fp16 max:", {k.split('model_')[-1][:7]: round(v / FP16_MAX, 3) for k, v in overall.items()})


if __name__ == "__main__":
    main()
