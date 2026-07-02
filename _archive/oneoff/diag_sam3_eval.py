"""Localize the SAM3-RBMA train-good / val-bad gap (train loss ~1.4 but val mIoU ~8).

Loads a trained checkpoint and reports per-class IoU on:
  (1) VAL set (val transform)                      <- the broken number
  (2) a TRAIN subset evaluated WITH THE VAL TRANSFORM (no PhysAug)  <- "can it segment
      images it trained on, under the eval pipeline?"

Interpretation:
  train-subset mIoU ~= val mIoU (both low) -> NOT a train/val gap; the model genuinely
      can't segment under eval (feature/architecture/eval-resolution issue).
  train-subset mIoU >> val mIoU            -> a train/val discrepancy: eval pipeline
      (resize/normalize/label) or distribution shift, NOT model capacity.

Run (single GPU):
  PYTHONPATH=semseg/models/sam3 python diag_sam3_eval.py \
    --cfg configs/b200-deliver_rgbdel_SAM3RBMA_physaug.yaml \
    --ckpt outputs/MMSam3RBMA/b200_deliver_rgbdel_SAM3RBMA_physaug/last.pth
"""
import os, sys, argparse, yaml, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "semseg/models/sam3"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from torch.utils.data import DataLoader, Subset
from semseg.datasets import *                       # noqa
from semseg.augmentations_mm import get_val_augmentation
from train_sam3_rbma import build_model, evaluate, print_iou_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_train", type=int, default=200, help="train images to eval (val transform)")
    args = ap.parse_args()
    cfg = yaml.load(open(args.cfg), Loader=yaml.SafeLoader)
    dcfg, ecfg, mc = cfg["DATASET"], cfg["EVAL"], cfg["MODEL"]
    device = torch.device(cfg.get("DEVICE", "cuda"))
    num_classes = mc.get("LORA_NUM_CLASSES", 4)
    ignore = dcfg["IGNORE_LABEL"]
    img = ecfg["IMAGE_SIZE"][0] if isinstance(ecfg["IMAGE_SIZE"], (list, tuple)) else ecfg["IMAGE_SIZE"]

    # model + trained checkpoint (build_model loads sam3.pt backbone; ckpt overwrites all)
    model = build_model(cfg, device)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model", ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {args.ckpt} epoch={ck.get('epoch','?')} miou={ck.get('miou', ck.get('best','?'))} "
          f"| missing={len(miss)} unexpected={len(unexp)}")

    va_t = get_val_augmentation(ecfg["IMAGE_SIZE"], dataset_cfg=dcfg)
    kw = {}
    if dcfg["NAME"] == "MULTIAQUA" and "NUM_CLASSES" in dcfg:
        kw["n_classes"] = dcfg["NUM_CLASSES"]
    bs = ecfg.get("BATCH_SIZE", 1); nw = ecfg.get("NUM_WORKERS", 4)

    valset = eval(dcfg["NAME"])(dcfg["ROOT"], "val", va_t, dcfg["MODALS"], **kw)
    names = getattr(valset, "CLASSES", [str(i) for i in range(num_classes)])
    valloader = DataLoader(valset, batch_size=bs, num_workers=nw, pin_memory=True)

    # TRAIN split but with the VAL transform (no PhysAug) -> isolates pipeline vs capacity
    trainset_eval = eval(dcfg["NAME"])(dcfg["ROOT"], "train", va_t, dcfg["MODALS"], **kw)
    n = min(args.n_train, len(trainset_eval))
    train_sub = Subset(trainset_eval, list(range(n)))
    trainloader = DataLoader(train_sub, batch_size=bs, num_workers=nw, pin_memory=True)

    print(f"\n===== VAL ({len(valset)} imgs) =====")
    v_miou, v_ious = evaluate(model, valloader, device, num_classes, ignore, img_size=img)
    print_iou_table("VAL", ck.get("epoch", 0), v_miou, v_ious, names)

    print(f"\n===== TRAIN-as-VAL ({n} imgs, val transform, no aug) =====")
    t_miou, t_ious = evaluate(model, trainloader, device, num_classes, ignore, img_size=img)
    print_iou_table("TRAIN", ck.get("epoch", 0), t_miou, t_ious, names)

    print("\n===== VERDICT =====")
    print(f"val mIoU={v_miou:.2f}  train-as-val mIoU={t_miou:.2f}")
    if t_miou - v_miou > 10:
        print(">>> TRAIN >> VAL: train/val gap = eval-pipeline or distribution issue (NOT capacity).")
        print(">>> Inspect: val transform resize/normalize vs train, label alignment, input scale.")
    else:
        print(">>> TRAIN ~= VAL (both low): NOT a train/val gap. Model genuinely under-segments")
        print(">>> under the eval pipeline -> feature/decoder/eval-resolution. (train loss low can")
        print(">>> still mean low mIoU if OHEM focuses on frequent classes.)")


if __name__ == "__main__":
    main()
