# Detection training — per-model / per-server guide

All detection models share the **ReliaDINO** backbone (frozen DINOv3 ViT-L/16 +
per-modality LoRA + reliability-gated fusion). They differ only in the **head** and
the aux **seg mechanism** on the backbone. Everything lives on `develop` — a server
just needs `git checkout develop` + the data + (for RF-DETR heads) the COCO weights.

## Models

| config (`configs/det/`) | DET_MODEL | head | seg mechanism | COCO init |
|---|---|---|---|---|
| `det_P34_final_full.yaml`      | ReliaDINODetector      | FCOS + NMS     | RBMA bias + consistency | no |
| `det_P35_final_full.yaml`      | ReliaDINODetector      | FCOS + NMS     | paper-freeze (bias/cons off) | no |
| `det_P36_final_full.yaml`      | ReliaDINODetector      | FCOS + NMS     | + PerClassRouter | no |
| `det_P37_rfdetr_full.yaml`     | ReliaDINORFDETRDetector| RF-DETR NMS-free| P34 recipe | **yes** |
| `det_P37a_cefr_yeon.yaml`      | ReliaDINORFDETRDetector| RF-DETR NMS-free| **CEFR** (2-pass blend) | **yes** |
| `det_P37b_classtoken_yeon.yaml`| ReliaDINORFDETRDetector| RF-DETR NMS-free| **ClassToken** residual | **yes** |
| `det_P38_m2f_yeon.yaml`        | ReliaDINOM2FDetector   | **M2F queries** (NMS-free) | M2F is the head | no (from scratch) |

P37a/P37b/P38 share one backbone recipe (ATTN_BIAS/CONSISTENCY off, GATE/VETO/
CALIBRATION on) so their fused features are identical — the head/mechanism is the
only variable.

## Per-server setup (2 paths + weights)

1. **Data** — `DATASET.ROOT` + `ANNOTATION_TRAIN/VAL` default to
   `/SSDb/jemo_maeng/dset/poongsan_v2` (consistent on the lab servers: bengio,
   yeon, lecun, levine). On a server with a different layout (e.g. hpca100
   `~/SSDb/...`), edit these three paths.
2. **COCO weights** (RF-DETR heads only — P37/P37a/P37b): `MODEL.COCO_CKPT` is the
   repo-relative `weights/rf-detr-large-2026.pth`. Put the 130 MB file there once
   per checkout (it is not in git). P38/M2F needs no weights.
3. **Nothing else** — the DINOv3 backbone downloads via timm on first run.

## Critical flags (do not change)

- `MODEL.GRAD_CHECKPOINT: false` — REQUIRED. encoder `active_modality` state is not
  preserved across checkpoint recompute, so grad-ckpt trains all-but-last modality
  on noise (corrupts 2 of our 3 modalities).
- RF-DETR heads: `TRAIN.GRAD_CLIP: 0.1` (DETR standard). The original P37 run
  collapsed with the old hardcoded 10.0 (≈no clipping). `grad_clip` is env-overridable
  via `DET_GRAD_CLIP`.

## Launch (torchrun, pick free GPUs first)

```bash
cd <repo>
conda activate openmmlab          # or the server's env with timm>=1.0
export OMP_NUM_THREADS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=<free gpus>          # verify free first!
export DET_GRAD_CLIP=0.1                          # RF-DETR heads
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # ~23GB/24GB at BS1
torchrun --standalone --nproc_per_node=<N> --master_port=<port> \
    train_det.py --cfg configs/det/<config>.yaml > logs/<name>.log 2>&1
```

VRAM ≈ 23 GB at BS1 (no grad-ckpt) — needs a 24 GB GPU. Effective batch = N × BS ×
GRAD_ACCUM_STEPS. Eval + checkpoint every `SAVE_INTERVAL` epoch.
