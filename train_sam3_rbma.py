"""
Dedicated trainer for LoRA_Sam3_RBMA (RBMA on SAM3).

Why a separate script: train_sam2_lora_paper.py is SAM2-specific (build_sam2, sam2
checkpoint, 1024 input, proto/gate/MI loss coupling). This script reuses the dataset /
augmentation / loss / scheduler / metric utilities but has a clean loop using the
model's own forward + compute_losses (main semantic CE + per-modality aux CE).

Run (single GPU):
  PYTHONPATH=semseg/models/sam3 python train_sam3_rbma.py --cfg configs/b200-multiaqua_rgbtl_SAM3RBMA_hardaug8_physaug.yaml
Run (DDP):
  PYTHONPATH=semseg/models/sam3 torchrun --nproc_per_node=8 train_sam3_rbma.py --cfg <cfg>

Notes: SAM3 weights (facebook/sam3) are gated → set MODEL.CHECKPOINT_PATH after approval
(else random init). Input is 1008 (SAM3), not 1024.
"""
import os, sys, argparse, yaml, math, time
from datetime import timedelta
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import torch.distributed as dist

# make `import sam3` work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "semseg/models/sam3"))

from semseg.datasets import *                       # noqa  (DELIVER, MULTIAQUA, ...)
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.metrics import Metrics
from semseg.utils.utils import fix_seeds, setup_cudnn
from sam3_lora_rbma import LoRA_Sam3_RBMA


def build_datasets(cfg):
    dcfg, tcfg, ecfg = cfg["DATASET"], cfg["TRAIN"], cfg["EVAL"]
    tr_t = get_train_augmentation(tcfg["IMAGE_SIZE"], seg_fill=dcfg["IGNORE_LABEL"], dataset_cfg=dcfg)
    va_t = get_val_augmentation(ecfg["IMAGE_SIZE"], dataset_cfg=dcfg)
    kw = {}
    if dcfg["NAME"] == "MULTIAQUA":
        kw["night_translation"] = bool(dcfg.get("NIGHT_TRANSLATION", False))
        if "NUM_CLASSES" in dcfg:
            kw["n_classes"] = dcfg["NUM_CLASSES"]
    trainset = eval(dcfg["NAME"])(dcfg["ROOT"], "train", tr_t, dcfg["MODALS"], **kw)
    valset = eval(dcfg["NAME"])(dcfg["ROOT"], "val", va_t, dcfg["MODALS"], **kw)
    testset = None
    if dcfg["NAME"] != "MULTIAQUA":      # DELIVER 등 test split 보유 → SAM2처럼 Test mIoU도 로깅
        try:
            testset = eval(dcfg["NAME"])(dcfg["ROOT"], "test", va_t, dcfg["MODALS"], **kw)
        except Exception as e:
            print(f"[INFO] test set not available: {e}")
    return trainset, valset, testset


def print_iou_table(tag, epoch, miou, ious, names):
    """SAM2-style per-class IoU log (reveals dead classes, e.g. Water=0 / Dynamic=0)."""
    vals = ious.tolist() if hasattr(ious, "tolist") else list(ious)
    print(f"[ep{epoch}] {tag} mIoU={miou:.2f}")
    line = " | ".join(f"{names[i] if i < len(names) else i}:{float(v):.1f}"
                      for i, v in enumerate(vals))
    print(f"    {tag} per-class IoU: {line}")


def build_model(cfg, device):
    mc = cfg["MODEL"]
    model = LoRA_Sam3_RBMA(
        r=mc.get("LORA_R", 4),
        num_modalities=len(cfg["DATASET"]["MODALS"]),
        num_classes=mc.get("LORA_NUM_CLASSES", 4),
        checkpoint_path=(mc.get("CHECKPOINT_PATH") or None),
        load_from_HF=mc.get("LOAD_FROM_HF", False),
        lambda_bias_init=mc.get("LAMBDA_BIAS_INIT", 1.0),
        decoder_high_res=mc.get("DECODER_HIGH_RES", False),
    ).to(device)
    return model


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_label,
             amp_dtype=torch.bfloat16, img_size=1008):
    model.eval()
    torch.cuda.empty_cache()
    metric = Metrics(num_classes, ignore_label, device)
    for sample, lbl in loader:
        sample = [x.to(device, non_blocking=True) for x in sample]
        # SAM3 ViT needs a FIXED input size (RoPE freqs_cis is precomputed for img_size);
        # the val transform (Resize: aspect-preserving + 32-align) may not yield img_size²
        # → force exactly (img_size, img_size). Output is resized back to label size below.
        sample = [F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
                  for x in sample]
        lbl = lbl.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):          # match train precision/memory
            sem = model(sample, multimask_output=True)         # (B,C,img_size,img_size)
        sem = F.interpolate(sem.float(), size=lbl.shape[-2:], mode="bilinear", align_corners=False)
        metric.update(sem.softmax(dim=1), lbl)
    ious, miou = metric.compute_iou()
    return miou, ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    cfg = yaml.load(open(args.cfg), Loader=yaml.SafeLoader)
    mc, dcfg, tcfg, ecfg = cfg["MODEL"], cfg["DATASET"], cfg["TRAIN"], cfg["EVAL"]

    # DDP / device
    ddp = tcfg.get("DDP", False) and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        # Long timeout: only rank 0 runs eval (val+test, ~4k imgs single-GPU). The other
        # ranks idle at the next-epoch collective meanwhile; the default 600s NCCL watchdog
        # would abort them mid-eval. 2h covers a long eval window.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world, device = 0, 1, torch.device(cfg.get("DEVICE", "cuda"))
    is_main = (rank == 0)
    fix_seeds(42); setup_cudnn()

    num_classes = mc.get("LORA_NUM_CLASSES", 4)
    save_dir = cfg["SAVE_DIR"]; os.makedirs(save_dir, exist_ok=True)

    trainset, valset, testset = build_datasets(cfg)
    train_sampler = DistributedSampler(trainset, world, rank, shuffle=True) if ddp else RandomSampler(trainset)
    lk = dict(num_workers=tcfg.get("NUM_WORKERS", 4), pin_memory=True, drop_last=True)
    trainloader = DataLoader(trainset, batch_size=tcfg["BATCH_SIZE"], sampler=train_sampler, **lk)
    # Larger eval batch + more workers => higher GPU util during validation (smaller
    # idle window where the box looks free). For segmentation mIoU under model.eval(),
    # batch size does NOT change the metric (per-pixel accumulation, BN uses running
    # stats) — only speed/VRAM. Requires uniform val image sizes (DELIVER = uniform).
    valloader = DataLoader(valset, batch_size=ecfg.get("BATCH_SIZE", 1),
                           num_workers=ecfg.get("NUM_WORKERS", 4),
                           pin_memory=True) if is_main else None
    eval_test = ecfg.get("EVAL_TEST", True)   # set false to skip test eval (saves time)
    testloader = DataLoader(testset, batch_size=ecfg.get("BATCH_SIZE", 1),
                            num_workers=ecfg.get("NUM_WORKERS", 4), pin_memory=True) \
        if (is_main and eval_test and testset is not None) else None
    class_names = getattr(trainset, "CLASSES", [str(i) for i in range(num_classes)])

    model = build_model(cfg, device)
    core = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False,
            # static_graph: reliab_head & sem_decoder are used m times per forward
            # (modality-as-frame), and sem_decoder.iou_head is unused (not in loss).
            # find_unused_parameters=True errors ("marked ready twice") on such multi-use
            # params; static_graph handles both multi-use and unused params (graph is
            # structurally identical every iteration). broadcast_buffers off: backbone frozen.
            static_graph=True)

        core = model.module

    loss_fn = get_loss(cfg["LOSS"]["NAME"], dcfg["IGNORE_LABEL"], None)
    aux_w = mc.get("AUX_WEIGHT", 0.5)
    optimizer = torch.optim.AdamW(core.trainable_parameters(),
                                  lr=cfg["OPTIMIZER"]["LR"], weight_decay=cfg["OPTIMIZER"]["WEIGHT_DECAY"])
    upe = len(trainloader); epochs = tcfg["EPOCHS"]
    scheduler = get_scheduler(cfg["SCHEDULER"]["NAME"], optimizer, (epochs + 1) * upe,
                              cfg["SCHEDULER"]["POWER"], upe * cfg["SCHEDULER"]["WARMUP"],
                              cfg["SCHEDULER"]["WARMUP_RATIO"])
    use_amp = tcfg.get("AMP", False)
    amp_dtype = torch.bfloat16 if str(tcfg.get("AMP_DTYPE", "bfloat16")) == "bfloat16" else torch.float16

    if is_main:
        n_tr = sum(p.numel() for p in core.trainable_parameters())
        print(f"[SAM3-RBMA] trainable params: {n_tr/1e6:.2f}M | classes={num_classes} | "
              f"img={tcfg['IMAGE_SIZE']} | ddp={ddp}(world={world}) | ckpt={'random' if not mc.get('CHECKPOINT_PATH') else mc['CHECKPOINT_PATH']}")

    # ── resume (model + optimizer + scheduler + epoch + best) ──
    best = -1.0
    start_epoch = 0
    resume_path = mc.get("RESUME_PATH", "") if mc.get("RESUME_ENABLE", False) else ""
    # AUTO_RESUME: get killed -> just rerun the SAME command -> continue from save_dir/last.pth
    # (saved every epoch BEFORE eval, so a kill costs <=1 epoch). Opt-in. Do NOT enable right
    # after changing model code — it would resume stale/incompatible weights. Start fresh by
    # turning AUTO_RESUME off or deleting last.pth.
    if not resume_path and mc.get("AUTO_RESUME", False):
        cand = os.path.join(save_dir, "last.pth")
        if os.path.isfile(cand):
            resume_path = cand
            if is_main: print(f"[AUTO_RESUME] found -> resuming: {cand}")
        elif is_main:
            print(f"[AUTO_RESUME] no last.pth in {save_dir} -> starting fresh")
    if resume_path and os.path.isfile(resume_path):
        ck = torch.load(resume_path, map_location="cpu")
        miss, unexp = core.load_state_dict(ck.get("model", ck), strict=False)
        start_epoch = int(ck.get("epoch", -1)) + 1
        best = float(ck.get("best", ck.get("miou", -1.0)))
        if "optimizer" in ck:
            try: optimizer.load_state_dict(ck["optimizer"])
            except Exception as e: print(f"[resume] optimizer state skipped: {e}")
        if "scheduler" in ck:
            try: scheduler.load_state_dict(ck["scheduler"])
            except Exception as e: print(f"[resume] scheduler state skipped: {e}")
        if is_main:
            print(f"[resume] {resume_path} → start_epoch={start_epoch} best={best:.2f} "
                  f"(missing={len(miss)} unexpected={len(unexp)})")
    elif resume_path and is_main:
        print(f"[resume] RESUME_PATH not found: {resume_path} → start from scratch")

    for epoch in range(start_epoch, epochs):
        model.train()
        if ddp: train_sampler.set_epoch(epoch)
        t0 = time.time(); run = 0.0
        for it, (sample, lbl) in enumerate(trainloader):
            sample = [x.to(device, non_blocking=True) for x in sample]
            lbl = lbl.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                sem = model(sample, multimask_output=True, gt_mask=lbl)
                loss, parts = core.compute_losses(sem, lbl, loss_fn, aux_weight=aux_w,
                                                  ignore_index=dcfg["IGNORE_LABEL"])
            loss.backward()
            optimizer.step(); scheduler.step()
            run += float(loss)
            if is_main and it % 20 == 0:
                print(f"ep{epoch} it{it}/{upe} loss={float(loss):.4f} "
                      f"(main={float(parts['main']):.4f} aux={float(parts['aux']):.4f}) "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if is_main:
            print(f"[ep{epoch}] mean loss={run/max(upe,1):.4f} time={time.time()-t0:.0f}s")
            # save checkpoint FIRST (full state for resume; so a val crash never loses the epoch)
            torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best},
                       os.path.join(save_dir, "last.pth"))
            if epoch >= tcfg.get("EVAL_START", 0) and (epoch % tcfg.get("EVAL_INTERVAL", 1) == 0):
                try:
                    _img = tcfg["IMAGE_SIZE"][0] if isinstance(tcfg["IMAGE_SIZE"], (list, tuple)) else tcfg["IMAGE_SIZE"]
                    miou, ious = evaluate(core, valloader, device, num_classes,
                                          dcfg["IGNORE_LABEL"], amp_dtype, img_size=_img)
                    print_iou_table("val", epoch, miou, ious, class_names)
                    if testloader is not None:
                        t_miou, t_ious = evaluate(core, testloader, device, num_classes,
                                                  dcfg["IGNORE_LABEL"], amp_dtype, img_size=_img)
                        print_iou_table("test", epoch, t_miou, t_ious, class_names)
                    if miou > best:
                        best = miou
                        torch.save({"model": core.state_dict(), "epoch": epoch, "miou": miou},
                                   os.path.join(save_dir, f"best_ep{epoch}_{miou:.2f}.pth"))
                        print(f"  NEW BEST {best:.2f}")
                except Exception as e:
                    import traceback
                    print(f"[ep{epoch}] eval FAILED (training continues, last.pth saved): {e}")
                    traceback.print_exc()
        # Resync ALL ranks here so non-main ranks wait at this barrier (not at the next
        # epoch's collective) while rank 0 saves/evals → clean, deterministic sync point.
        if ddp:
            dist.barrier()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
