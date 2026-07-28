"""Fair-scope teacher eval: run D1 on the 2066 modal-complete TEST frames and
score it ANNOTATION-scope over exactly those 2066 (COCOeval imgIds = the kept
ids) — the same scope yolov5 val uses for the RGB student. Prints AP/AP50/AP75
under both scopes so predicted vs annotation is explicit.

  python teacher_eval_fair.py --cfg configs/det/det_D1_vitsp_jarvis.yaml \
    --ckpt weights/det_D1_vitsp_20260723/best_checkpoint.pth --data-root ~/poongsan_v2
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.join(REPO, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
from _det_common import (build_detector, build_loader, load_cfg,           # noqa: E402
                         load_det_checkpoint, run_inference, eval_overall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    r = args.data_root.rstrip('/')
    cfg['DATASET']['ROOT'] = r
    cfg['DATASET']['ANNOTATION_VAL'] = f'{r}/_final_ann/instances_test_common.json'
    cfg['DATASET']['REQUIRE_ALL_MODALITIES'] = True
    ann = cfg['DATASET']['ANNOTATION_VAL']

    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ds, loader = build_loader(cfg, 'val', workers=4)
    model = build_detector(cfg, dev, cfg['MODEL'].get('N_CLASSES') or ds.n_classes)
    load_det_checkpoint(model, args.ckpt, dev)
    model.eval()

    preds, id2file = run_inference(model, ds, loader, cfg, dev, score_thresh=0.05)
    kept_ids = sorted(int(k) for k in id2file)          # the 2066 modal-complete frames
    print(f"[teacher] ran on {len(kept_ids)} frames, {len(preds)} boxes")

    # ANNOTATION-scope over the 2066 (== yolov5 val scope on the same frames)
    a = eval_overall(ann, preds, img_ids=kept_ids)
    print(f"[annotation-scope / {len(kept_ids)} frames]  "
          f"mAP50={a['AP50']:.4f}  mAP50-95={a['AP']:.4f}  mAP75={a['AP75']:.4f}")

    # PREDICTED-scope: restrict to frames that got >=1 prediction (D1 cert scope)
    with_pred = sorted({int(p['image_id']) for p in preds})
    p_ = eval_overall(ann, preds, img_ids=with_pred)
    print(f"[predicted-scope  / {len(with_pred)} frames]  "
          f"mAP50={p_['AP50']:.4f}  mAP50-95={p_['AP']:.4f}  mAP75={p_['AP75']:.4f}")
    print(f"[no-prediction frames excluded by predicted-scope: {len(kept_ids)-len(with_pred)}]")

    # NIGHT / NORMAL breakdown over the kept 2066 (annotation-scope)
    import json as _json
    gt = _json.load(open(ann))
    keptset = set(kept_ids)
    night = sorted(im['id'] for im in gt['images'] if im.get('low_light') and im['id'] in keptset)
    normal = sorted(im['id'] for im in gt['images'] if not im.get('low_light') and im['id'] in keptset)
    n = eval_overall(ann, preds, img_ids=night)
    nm = eval_overall(ann, preds, img_ids=normal)
    print(f"[night  / {len(night)} frames]  mAP50={n['AP50']:.4f}  mAP50-95={n['AP']:.4f}")
    print(f"[normal / {len(normal)} frames]  mAP50={nm['AP50']:.4f}  mAP50-95={nm['AP']:.4f}")


if __name__ == '__main__':
    main()
