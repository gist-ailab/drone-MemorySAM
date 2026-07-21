"""Per-scene (per-image) mIoU audit for MUSES fog cells — P39.1 선행 분석.

Why: P39/P37a의 fog 결손(-12.7 vs clear)은 문헌 전형(-3~-5)의 3배로 비정상이고
night>clear 역전은 발표된 MUSES 시스템 전체와 반대 패턴이다(2026-07-21 fog 리서치).
소수 파국 장면이 평균을 끌어내리는지(수리 가능, 큰 헤드룸) vs 균일 저하인지
(소표본 노이즈 가능, 헤드룸 과대평가)를 아키텍처 베팅 전에 판별한다.

Usage (jarvis GPU6 예):
  python tools/p39_fog_scene_audit.py \
    --cfg configs/jarvis-muses_rgbel_P39_dpc.yaml \
    --ckpt P39=outputs/.../epoch146_81.52_top3_checkpoint.pth \
    --ckpt P38=outputs/.../epoch156_*_checkpoint.pth \
    --dataset-root <MUSES_ROOT> --conditions fog_night,fog_day,clear_night \
    --split val --gpu 0 --out /tmp/fog_audit

Output: <out>.tsv (cond, image, per-ckpt per-image mIoU, delta) +
        <out>.md  (per-cond summary: mean/median/worst-5 + 파국장면 판정 힌트)

Loading follows tools/feature_stats.py conventions exactly (V.load_model /
V.create_dataset with DATASET.CASE per condition).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def per_image_miou(pred, label, num_classes):
    """Present-class mean IoU for one image. pred/label: (H,W) long."""
    import torch
    valid = label != 255
    ious = []
    for c in torch.unique(label[valid]).tolist():
        p = (pred == c) & valid
        g = label == c
        inter = (p & g).sum().item()
        union = (p | g).sum().item()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / max(len(ious), 1), len(ious)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', action='append', required=True, metavar='LABEL=PATH')
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--conditions', default='fog_night,fog_day,clear_night')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--max-imgs', type=int, default=-1)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    import torch
    import yaml
    import val as V

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['RESUME_ENABLE'] = False
    ds_cfg = cfg['DATASET']
    if args.dataset_root:
        ds_cfg['ROOT'] = args.dataset_root
    if isinstance(ds_cfg.get('PHYSAUG'), dict):
        ds_cfg['PHYSAUG']['ENABLE'] = False
    device = torch.device(cfg['DEVICE'])
    V.setup_cudnn()
    isz = cfg.get('TEST', {}).get('IMAGE_SIZE', cfg['EVAL']['IMAGE_SIZE'])
    transform = V.get_val_augmentation(isz, dataset_cfg=ds_cfg)

    ckpts = []
    for kv in args.ckpt:
        label, path = kv.split('=', 1)
        ckpts.append((label, path))

    conds = [c.strip() for c in args.conditions.split(',') if c.strip()]
    # rows[(cond, img_id)] = {label: (miou, n_cls)}
    rows = {}
    order = []
    for label, path in ckpts:
        model = V.load_model(cfg, Path(path), device)
        model.eval()
        for cond in conds:
            ds_cfg['CASE'] = cond
            dataset, _ = V.create_dataset(ds_cfg, args.split, transform,
                                          args.split, macvi=False, eval_day=False)
            n = len(dataset) if args.max_imgs < 0 else min(args.max_imgs, len(dataset))
            print(f"[{label}] {cond}: {n} images")
            for idx in range(n):
                sample = dataset[idx]
                images, gt = sample[0], sample[1]
                meta = sample[2] if len(sample) > 2 else None
                if isinstance(meta, dict):
                    img_id = str(meta.get('stem', f"idx{idx}"))
                    ip = (meta.get('paths', {}) or {}).get('img', '')
                    for c in ('fog', 'rain', 'snow', 'clear'):
                        if f"/{c}/" in ip:
                            for t in ('day', 'night'):
                                if f"/{t}/" in ip:
                                    img_id = f"{c}_{t}:{img_id}"
                            break
                else:
                    img_id = str(meta) if meta is not None else f"idx{idx}"
                imgs = [im.unsqueeze(0).to(device) for im in images]
                with torch.no_grad():
                    out = model(imgs, multimask_output=True)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                pred = logits[0].argmax(0).cpu()
                gt = gt.squeeze().long().cpu()
                if pred.shape != gt.shape:
                    pred = torch.nn.functional.interpolate(
                        pred[None, None].float(), size=gt.shape,
                        mode='nearest')[0, 0].long()
                miou, ncls = per_image_miou(pred, gt, None)
                key = (cond, img_id)
                if key not in rows:
                    rows[key] = {}
                    order.append(key)
                rows[key][label] = (100.0 * miou, ncls)
        del model
        torch.cuda.empty_cache()

    labels = [l for l, _ in ckpts]
    with open(args.out + '.tsv', 'w') as f:
        f.write("cond\timage\t" + "\t".join(f"{l}_miou\t{l}_ncls" for l in labels)
                + ("\tdelta" if len(labels) == 2 else "") + "\n")
        for key in order:
            cond, img = key
            vals = [rows[key].get(l, (float('nan'), 0)) for l in labels]
            line = f"{cond}\t{img}\t" + "\t".join(
                f"{v:.2f}\t{n}" for v, n in vals)
            if len(labels) == 2:
                line += f"\t{vals[0][0] - vals[1][0]:+.2f}"
            f.write(line + "\n")

    with open(args.out + '.md', 'w') as f:
        f.write("# fog per-scene audit\n\n")
        for cond in conds:
            f.write(f"## {cond}\n\n")
            for l in labels:
                vs = sorted([(rows[k][l][0], k[1]) for k in order
                             if k[0] == cond and l in rows[k]])
                if not vs:
                    f.write(f"- {l}: no images\n")
                    continue
                mean = sum(v for v, _ in vs) / len(vs)
                med = vs[len(vs) // 2][0]
                worst = ", ".join(f"{img}({v:.1f})" for v, img in vs[:5])
                f.write(f"- **{l}** n={len(vs)} mean {mean:.2f} · median {med:.2f}"
                        f" · worst5: {worst}\n")
                # 파국 장면 힌트: median-mean 괴리가 크면 소수 장면이 평균을 끌어내림
                f.write(f"  - skew(median−mean) = {med - mean:+.2f} "
                        f"({'소수 파국 장면 의심' if med - mean > 3 else '균일 저하 쪽'})\n")
            f.write("\n")
    print("wrote", args.out + '.tsv', args.out + '.md')


if __name__ == '__main__':
    main()
