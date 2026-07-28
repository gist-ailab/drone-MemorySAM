"""GISTOLO step 1 — dump the multimodal teacher's detections on the TRAIN frames.

Teacher = D1 ReliaDINO (DINOv3 ViT-S+ + RF-DETR head), 3-modal (RGB+LiDAR+Thermal).
We run it over the exact modal-complete train frames the RGB student trains on, and
save its predictions (COCO format) to distil into the YOLOv5m student.

Reuses the certification inference stack (tools/_det_common) verbatim, so the
teacher outputs here are identical in spec to the certified eval.

  python teacher_dump.py --cfg configs/det/det_D1_vitsp_jarvis.yaml \
      --ckpt weights/det_D1_vitsp_20260723/best_checkpoint.pth \
      --data-root ~/poongsan_v2_train3modal --out runs_gistolo/teacher_train_preds.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # objdet/gistolo -> repo root
for p in (REPO, os.path.join(REPO, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from _det_common import (build_detector, build_loader, load_cfg,          # noqa: E402
                         load_det_checkpoint, run_inference)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-root', required=True, help='train 3-modal root (clip/{rgb,depth_map_lidar,thermal_aligned})')
    ap.add_argument('--ann', default=None, help='train annotation (default: <root>/_final_ann/instances_train_egofill.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--score-thresh', type=float, default=0.05)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    r = args.data_root.rstrip('/')
    ann = args.ann or f'{r}/_final_ann/instances_train_egofill.json'
    cfg['DATASET']['ROOT'] = r
    cfg['DATASET']['ANNOTATION_VAL'] = ann          # run_inference reads the 'val' split
    cfg['DATASET']['REQUIRE_ALL_MODALITIES'] = True  # keep only modal-complete frames

    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ds, loader = build_loader(cfg, 'val', workers=4)
    n_classes = cfg['MODEL'].get('N_CLASSES') or ds.n_classes
    model = build_detector(cfg, dev, n_classes)
    load_det_checkpoint(model, args.ckpt, dev)
    model.eval()
    print(f"[teacher] frames kept (modal-complete): {len(ds)}  classes={n_classes}")

    preds, id2file = run_inference(model, ds, loader, cfg, dev, score_thresh=args.score_thresh)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'preds': preds,
                   'id2file': {str(k): v for k, v in id2file.items()},
                   'ann': ann, 'score_thresh': args.score_thresh}, f)
    print(f"[teacher] dumped {len(preds)} boxes over {len(id2file)} images -> {args.out}")


if __name__ == '__main__':
    main()
