#!/usr/bin/env python3
"""
D4 — 우리 모델 모달 제거(zero-out) ablation. 각 모달 입력을 0 으로 두고 mIoU 를
재는 추론을 모달 수만큼 반복해, 어느 모달에 실제로 의존하는지 본다.

val.py 의 load_model/create_dataset 을 재사용한다. 기준선(detectron2)용 zero-out 은
tools/baseline_failure/d2_zero_modality.patch (BF_ZERO_MODAL) 를 쓴다 — README 참조.
tools/module_ablation.py 는 모듈 토글(rbma_off 등)만 다루고 모달 zero-out 기능은 없어
여기서 별도 구현한다.

예:
  python tools/baseline_failure/modality_zero_ablation.py \
    --cfg configs/eval/<ours>.yaml --model_path <ckpt> \
    --split test --out .../mining/modal_zero_ours.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402


def _run_once(valmod, model, loader, device, n_classes, ignore, model_size, zero_idx):
    hist = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for images, labels, metas in tqdm(loader, desc=f"zero={zero_idx}", leave=False):
            images = [x.to(device) for x in images]
            if zero_idx is not None:
                images[zero_idx] = torch.zeros_like(images[zero_idx])
            output, _ = model(images, multimask_output=True)
            probs = output.softmax(dim=1)
            pred_labels = valmod._argmax_pred(probs, n_classes, None)
            for b in range(pred_labels.shape[0]):
                meta = metas[b]
                orig_label = meta.get("orig_label")
                if orig_label is None:
                    continue
                pr = valmod._unpad_resize_to_orig(
                    pred_labels[b], meta["orig_h"], meta["orig_w"], model_size=model_size)
                hist += common.confusion_matrix(
                    pr.cpu().numpy(), orig_label.cpu().numpy(), n_classes, ignore)
    _, miou = common.global_miou_from_cm(hist)
    return miou


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import val as valmod
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    device = torch.device(cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    valmod.setup_cudnn()

    dataset_cfg = cfg["DATASET"]
    eval_cfg = cfg["EVAL"]
    test_cfg = cfg.get("TEST", {})
    mode = args.split
    image_size = (eval_cfg["IMAGE_SIZE"] if mode == "val"
                  else test_cfg.get("IMAGE_SIZE", eval_cfg["IMAGE_SIZE"]))
    model_size = image_size if isinstance(image_size, int) else int(image_size[0])
    transform = valmod.get_val_augmentation(image_size, dataset_cfg=dataset_cfg)
    dataset, _ = valmod.create_dataset(dataset_cfg, mode, transform, mode)
    if args.limit:
        dataset.files = dataset.files[:args.limit]
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=4,
                        pin_memory=False, collate_fn=valmod._collate_fn)
    model = valmod.load_model(cfg, Path(args.model_path), device)
    model.eval()

    modals = list(dataset_cfg["MODALS"])
    n_classes = dataset.n_classes
    ignore = dataset.ignore_label

    results = {"base": _run_once(valmod, model, loader, device, n_classes, ignore, model_size, None)}
    for i, m in enumerate(modals):
        results[f"zero_{m}"] = _run_once(
            valmod, model, loader, device, n_classes, ignore, model_size, i)

    report = {"cfg": args.cfg, "model_path": args.model_path, "split": mode,
              "modals": modals, "mIoU": results,
              "drop_vs_base": {k: round(results["base"] - v, 3)
                               for k, v in results.items() if k != "base"}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["mIoU"], indent=2, ensure_ascii=False))
    print(f"[modality_zero] -> {args.out}")


if __name__ == "__main__":
    main()
