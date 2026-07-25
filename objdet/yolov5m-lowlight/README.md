# YOLOv5m low-light — architecture-frozen training-recipe study

Retrain a **classic YOLOv5m (RGB-only)** on poongsan_v2 and lift its **low-light**
detection **without changing the model**, so the result stays **i.MX-NPU portable**.
Only train-time levers are used: augmentation, loss terms, optimizer/schedule.
Presentation deliverable = a clean ablation ladder with a night/normal breakdown.

## Why "classic" YOLOv5m (not `yolov5mu.pt`)

The earlier baseline used `yolov5mu.pt` — the ultralytics **anchor-free** (v8-head)
variant. That is **not** the i.MX export target. NXP eIQ / the i.MX model zoo
support the **classic anchor-based YOLOv5** graph, so this package builds on
`ultralytics/yolov5` (`train.py` + `hyp.yaml`) with `yolov5m.pt` weights. Every
intervention below is train-time only → the exported ONNX/TFLite is byte-for-byte
the stock YOLOv5m graph.

## The one rule: inference graph is frozen

Enforced *by construction* — `train_lowlight.py` imports stock YOLOv5 and
**monkeypatches only two runtime things**, gated by env flags:

| lever | env flag | what it touches | i.MX-safe |
|---|---|---|---|
| dark-tail low-light aug | `YOLO_NIGHTAUG=1` | `Albumentations` pipeline | ✅ train-only |
| EIoU box loss | `YOLO_EIOU=1` | `ComputeLoss` bbox_iou | ✅ loss-only |
| focal cls+obj | `fl_gamma` (hyp) | loss weighting | ✅ loss-only |
| label smoothing | `--label-smoothing` | target encoding | ✅ train-only |
| mixup / copy_paste | hyp | data mixing | ✅ train-only |

No layers added, no head/anchor change, no activation swap. Remove the flags →
bit-identical stock YOLOv5m.

## Data finding that set the augmentation (measured, not assumed)

Per-clip RGB luma (0–255), poongsan_v2:

| split | mean luma | p10 (dark tail) | dark-pixel % |
|---|---|---|---|
| train (5 clips) | 117.2 | 78–99 | 8–14 |
| **test (3 clips)** | 108.6 | **48.6** | **up to 22** |

The gap is **not** a global offset (only −8.6) — it is a **deep low-light tail
present in test but missing from train**. So the augmentation *manufactures that
tail*: `night_aug.py` darkens a fraction of train frames toward luma ~40–70
(gamma ≈ 2.0) and adds low-light sensor noise. (A naïve brightness jitter would
have *brightened* — measuring first caught the wrong direction.)

## Ablation ladder (each rung = one slide = one training)

| rung | adds | isolates | script |
|---|---|---|---|
| **b0** | — (stock YOLOv5m) | control | `train_ladder.sh b0` |
| **b1** | + dark-tail night-aug | augmentation | `train_ladder.sh b1` |
| **b2** | + EIoU + focal + label-smooth | loss | `train_ladder.sh b2` |
| **b3** | + mixup + copy_paste | data mixing (rare classes) | `train_ladder.sh b3` |

Money slide: `eval_lowlight.sh` gives **all / lowlight / normal** mAP50 per rung —
show the night gap shrinking b0→b3 while normal stays flat.

## Run (on a GPU box, e.g. hinton `conda activate yolo`)

```bash
git clone https://github.com/ultralytics/yolov5        # classic anchor-based
export YOLOV5_DIR=$PWD/yolov5
pip install albumentations                             # if not present
# 0) verify the levers BEFORE training (repo policy: measure, don't assume)
python night_aug.py /path/to/a_train_rgb.png          # confirm luma reaches ~50 tail
python eiou.py                                         # EIoU sanity
# 1) ladder (pick free GPU first)
DATA_YAML=/SSDd/.../poongsan_v2_yolo_rgb/poongsan_v2_rgb.yaml \
  bash train_ladder.sh b0 <gpu> 100
  bash train_ladder.sh b1 <gpu> 100
  bash train_ladder.sh b2 <gpu> 100
  bash train_ladder.sh b3 <gpu> 100
# 2) night/normal breakdown for each best.pt
bash eval_lowlight.sh runs/y5m_b3_full/weights/best.pt $DATA_YAML <OUT_dir_with_test_txt> <gpu>
```

Data conversion (RGB-only, emits test_all/lowlight/normal lists):
`python convert_final_yolo.py <ANN_DIR> <DATA_BASE> <OUT_DIR>`.

## Context / honest framing

On this dataset RGB-only already rivals the multimodal stack on mAP50
(YOLO11m 0.864; D1 ViT-S+ 3-modal 0.9081, different eval scope). So the story is
**"a tiny i.MX-portable RGB detector reaches near-multimodal accuracy, and a
low-light-targeted training recipe defends the dark tail"** — not "recover a huge
multimodal gap". A multimodal→RGB **distillation** rung (teacher = D1) is a
possible b4, but only worth the effort if b0→b3 leaves a night gap to close;
validate the cheap rungs first.

## Files

- `train_lowlight.py` — monkeypatch wrapper (imports stock YOLOv5, patches aug+loss)
- `night_aug.py` — dark-tail augmentation + standalone calibration self-test
- `eiou.py` — EIoU box loss (bbox_iou-compatible) + sanity self-test
- `hyp.b2_loss.yaml`, `hyp.lowlight.yaml` — b2 / b3 hyperparameters
- `train_ladder.sh` / `eval_lowlight.sh` — run the ladder / night-normal eval
- `convert_final_yolo.py` — COCO → YOLO(RGB), lowlight/normal test lists
