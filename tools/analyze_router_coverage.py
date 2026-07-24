#!/usr/bin/env python3
"""
tools/analyze_router_coverage.py — 학습0 검증 2건 (P43~P45 제안 §7, 2026-07-25)

  [V1] per-class x drop-modality IoU 매트릭스 + router per-class 모달 가중 히트맵
       -> "클래스마다 실제 다른 모달을 골랐나, 전부 RGB에 비율만 다른가" 판별 (§7-a)
  [V2] 커버리지 안/밖 router 가중 대조 (lidar 유효/무반환 픽셀 분할)
       -> 커버리지 밖에서 lidar 가중이 안 떨어지면 presence-mask(V-1) 필수 근거 (§7-b)

무학습·eval 전용. 모델 코드 무수정(PerClassRouter forward hook로 w (m,B,K,h,w) 캡처).
router가 없는 모델이면 V1의 drop 매트릭스만 산출하고 router 절은 skip.

Usage:
  python tools/analyze_router_coverage.py --cfg <muses.yaml> --model_path <ckpt> \
    --conditions clear,fog,night --split val --max-imgs 60 --drop-n 24 \
    --lidar-key lidar --gpu 0 --out <prefix>

Output: <out>.json + <out>.md
"""
import argparse, os, sys, json
from pathlib import Path

# --gpu 는 torch import 전에 반영 (module_diagnostics --gpu 선반영 픽스와 동일 규약)
if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpu') + 1]
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import val as V

WR = 256


def rs_nn(t, hw):
    return F.interpolate(t[None, None].float(), size=(hw, hw), mode='nearest')[0, 0].long()


def iou_per_class(cf):
    C = cf.shape[0]
    return np.array([cf[c, c] / max(1, cf[c, :].sum() + cf[:, c].sum() - cf[c, c]) for c in range(C)])


def lidar_footprint(x):
    """x: (1,C,H,W) 정규화된 lidar 입력. 무반환/패딩 픽셀은 채널별 상수(=per-channel mode)로
    깔리므로, 전 채널이 mode 값과 일치하는 픽셀을 absent로 판정 (정규화 무관·결정론)."""
    v = (x[0] * 1000).round()                      # (C,H,W) 양자화로 부동소수 동치 회피
    C = v.shape[0]
    flat = v.reshape(C, -1)
    mode = torch.mode(flat, dim=1).values          # (C,)
    absent = (v == mode.view(C, 1, 1)).all(dim=0)  # (H,W) bool
    return ~absent                                 # True = lidar 유효(커버리지 안)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--conditions', default='clear,fog,night')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--max-imgs', type=int, default=60, help='router 통계 이미지 수/조건')
    ap.add_argument('--drop-n', type=int, default=24, help='drop-modality 매트릭스 이미지 수/조건')
    ap.add_argument('--lidar-key', default='lidar')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import yaml
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
    model = V.load_model(cfg, Path(args.model_path), device)
    model.eval()

    modals = list(ds_cfg['MODALS'])
    Mn = len(modals)
    li = modals.index(args.lidar_key) if args.lidar_key in modals else -1

    # PerClassRouter 탐색 + hook (없으면 router 절 skip)
    router = None
    for _, mod in model.named_modules():
        if type(mod).__name__ == 'PerClassRouter':
            router = mod
            break
    holder = {}
    if router is not None:
        def _hook(_m, _i, out):
            holder['w'] = out[0].detach().float()  # (m,B,K,h,w)
        router.register_forward_hook(_hook)

    report = {'model': Path(args.model_path).stem, 'modals': modals,
              'lidar_idx': li, 'router_found': router is not None, 'conditions': {}}

    for cond in [c.strip() for c in args.conditions.split(',') if c.strip()]:
        ds_cfg['CASE'] = cond
        dataset, _ = V.create_dataset(ds_cfg, args.split, transform, args.split, macvi=False)
        n = len(dataset)
        CLASSES = list(dataset.CLASSES)
        C = len(CLASSES)
        ign = getattr(dataset, 'ignore_label', 255)
        print(f"[{cond}] {n} imgs, {C} classes")

        # ---- [V1a] router per-class 가중 + [V2] 커버리지 대조 ----
        w_cls_sum = np.zeros((Mn, C))              # per-class 모달 가중 누적 (공간 평균)
        w_cls_cnt = 0
        lid_in_sum = np.zeros(C); lid_in_cnt = np.zeros(C)   # 커버리지 안 lidar 가중 (per-class)
        lid_out_sum = np.zeros(C); lid_out_cnt = np.zeros(C)  # 커버리지 밖
        cover_frac_sum = 0.0

        # ---- [V1b] per-class x drop-modality confusion ----
        cf_full = np.zeros((C, C), np.int64)
        cf_drop = [np.zeros((C, C), np.int64) for _ in range(Mn)]

        n_stat = min(args.max_imgs, n)
        n_drop = min(args.drop_n, n)
        for idx in range(max(n_stat, n_drop)):
            images, label, _ = dataset[idx]
            base = [im.unsqueeze(0).to(device) for im in images]
            gt = rs_nn(torch.as_tensor(np.asarray(label)).to(device), WR).cpu().numpy()
            valid = gt != ign

            holder.pop('w', None)
            with torch.no_grad():
                mo, _ = model(base, multimask_output=True)

            if idx < n_drop:
                fp = rs_nn(mo[0].argmax(0), WR).cpu().numpy()
                np.add.at(cf_full, (gt[valid], fp[valid]), 1)
                for i in range(Mn):
                    dz = [(torch.zeros_like(base[j]) if j == i else base[j]) for j in range(Mn)]
                    with torch.no_grad():
                        moi, _ = model(dz, multimask_output=True)
                    fpi = rs_nn(moi[0].argmax(0), WR).cpu().numpy()
                    np.add.at(cf_drop[i], (gt[valid], fpi[valid]), 1)

            if router is not None and idx < n_stat and 'w' in holder:
                w = holder['w'][:, 0]              # (m,K,h,w)
                w_cls_sum += w.mean(dim=(2, 3)).cpu().numpy()
                w_cls_cnt += 1
                if li >= 0:
                    fpm = lidar_footprint(base[li].cpu())           # (H,W) bool
                    fpm_r = F.interpolate(fpm[None, None].float(),
                                          size=w.shape[-2:], mode='nearest')[0, 0] > 0.5
                    cover_frac_sum += float(fpm_r.float().mean())
                    wl = w[li].cpu()                                # (K,h,w)
                    inm, outm = fpm_r, ~fpm_r
                    if inm.any():
                        lid_in_sum += wl[:, inm].mean(dim=1).numpy()
                        lid_in_cnt += 1
                    if outm.any():
                        lid_out_sum += wl[:, outm].mean(dim=1).numpy()
                        lid_out_cnt += 1

        iou_full = iou_per_class(cf_full) * 100
        drops = {}
        for i in range(Mn):
            di = iou_per_class(cf_drop[i]) * 100
            drops[modals[i]] = {
                'miou_delta': round(float(np.nanmean(iou_full) - np.nanmean(di)), 2),
                'per_class_delta': {CLASSES[c]: round(float(iou_full[c] - di[c]), 2) for c in range(C)},
            }
        centry = {
            'n_stat': n_stat, 'n_drop': n_drop,
            'miou_full_WR': round(float(np.nanmean(iou_full)), 2),
            'per_class_iou_full': {CLASSES[c]: round(float(iou_full[c]), 2) for c in range(C)},
            'drop_matrix': drops,
        }
        if router is not None and w_cls_cnt:
            wm = w_cls_sum / w_cls_cnt
            centry['router_class_weights'] = {
                CLASSES[c]: {modals[i]: round(float(wm[i, c]), 4) for i in range(Mn)} for c in range(C)}
            centry['router_argmax_nonRGB_classes'] = [
                CLASSES[c] for c in range(C) if int(np.argmax(wm[:, c])) != 0]
        if router is not None and li >= 0 and lid_in_cnt.max() > 0:
            lin = lid_in_sum / np.maximum(lid_in_cnt, 1)
            lout = lid_out_sum / np.maximum(lid_out_cnt, 1)
            centry['coverage'] = {
                'cover_frac': round(cover_frac_sum / max(w_cls_cnt, 1), 4),
                'lidar_w_inside_mean': round(float(lin.mean()), 4),
                'lidar_w_outside_mean': round(float(lout.mean()), 4),
                'inside_minus_outside_per_class': {
                    CLASSES[c]: round(float(lin[c] - lout[c]), 4) for c in range(C)},
            }
        report['conditions'][cond] = centry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f'{out}.json', 'w') as f:
        json.dump(report, f, indent=1, ensure_ascii=False)

    lines = [f"# Router/coverage 학습0 검증 — `{report['model']}` (modals={modals})", '']
    for cond, ce in report['conditions'].items():
        lines += [f"## {cond} (stat n={ce['n_stat']}, drop n={ce['n_drop']}, mIoU@WR {ce['miou_full_WR']})", '']
        lines += ['### [V1b] drop-modality ΔmIoU: ' + ', '.join(
            f"{m} {ce['drop_matrix'][m]['miou_delta']:+.2f}" for m in modals)]
        top = {m: sorted(ce['drop_matrix'][m]['per_class_delta'].items(),
                         key=lambda kv: -kv[1])[:5] for m in modals}
        for m in modals:
            lines += [f"  - {m} top-Δ classes: " + ', '.join(f"{k} {v:+.1f}" for k, v in top[m])]
        if 'router_class_weights' in ce:
            lines += ['', f"### [V1a] router argmax가 비RGB인 클래스: "
                      f"{ce['router_argmax_nonRGB_classes'] or '없음(전 클래스 RGB 1위)'}"]
        if 'coverage' in ce:
            cv = ce['coverage']
            lines += ['', f"### [V2] lidar 가중 커버리지 안 {cv['lidar_w_inside_mean']} vs "
                      f"밖 {cv['lidar_w_outside_mean']} (cover_frac {cv['cover_frac']})"]
        lines += ['']
    with open(f'{out}.md', 'w') as f:
        f.write('\n'.join(lines))
    print(f"[done] {out}.json / {out}.md")


if __name__ == '__main__':
    main()
