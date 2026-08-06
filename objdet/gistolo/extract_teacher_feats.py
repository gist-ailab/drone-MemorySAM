"""[C/D] Teacher 특징 + 모달 게이트 추출 — feature-level 크로스모달 증류의 재료.

라벨(박스)만 옮기던 기존 GISTOLO 와 달리, 여기서는 **멀티모달이 융합된 표현 자체**를
꺼낸다. ReliaDINO ViT-L 의 SimpleFPN stride-16 특징(256ch)과, 그 위치에서 어느
모달을 믿었는지 알려주는 **게이트 가중치**(m=3: img/lidar/thermal)를 함께 저장한다.

  - [C] feature KD : teacher FPN 특징 -> student neck 특징 정합
  - [D] modality-aware weighting : 1 - w_img (= 비-RGB 의존도) 를 KD 손실 가중치로.
        RGB 만으로는 못 봤을 위치에서 증류를 더 세게 건다.

teacher 는 768x768 로 도는데 student(YOLOv5m 640) 의 stride-16 격자는 40x40 이므로
저장 시점에 미리 40x40 으로 리샘플해 둔다(학습 루프를 가볍게).

  python extract_teacher_feats.py --cfg configs/det/det_D1_vitl_gistolo.yaml \
      --ckpt weights/det_D1_recovered_20260723/best_checkpoint.pth \
      --data-root ~/poongsan_v2_train3modal --out ~/dset/gistolo_teacher_feats --grid 40
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.join(REPO, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--ann', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--grid', type=int, default=40, help='student stride-16 격자 (640/16)')
    ap.add_argument('--level', type=int, default=2, help='pyramid index (0:s4 1:s8 2:s16 3:s32)')
    ap.add_argument('--multiscale', action='store_true',
                    help='[2] s8/s16/s32 세 레벨을 함께 저장 (다중 스케일 KD)')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from _det_common import (build_detector, build_loader, load_cfg, load_det_checkpoint)

    cfg = load_cfg(args.cfg)
    r = args.data_root.rstrip('/')
    ann = args.ann or f'{r}/_final_ann/instances_train_egofill.json'
    cfg['DATASET']['ROOT'] = r
    cfg['DATASET']['ANNOTATION_VAL'] = ann
    cfg['DATASET']['REQUIRE_ALL_MODALITIES'] = True

    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ds, loader = build_loader(cfg, 'val', workers=4)
    model = build_detector(cfg, dev, cfg['MODEL'].get('N_CLASSES') or ds.n_classes)
    load_det_checkpoint(model, args.ckpt, dev)
    model.eval()
    seg = model.seg_model

    # --- 게이트 가중치 포획: fusion._gate 를 감싸 마지막 w 를 보관 ---
    gate_store = {}
    fusion = seg.fusion
    if hasattr(fusion, '_gate'):
        orig_gate = fusion._gate

        def gate_hook(*a, **k):
            w, ent = orig_gate(*a, **k)
            gate_store['w'] = w.detach()          # (m, B, 1, h, w)
            return w, ent
        fusion._gate = gate_hook
    else:
        print('[warn] fusion._gate 없음 — modality weight 없이 특징만 저장')

    os.makedirs(args.out, exist_ok=True)
    n = 0
    with torch.no_grad():
        for batch in loader:
            # detector 의 정식 진입점을 쓴다 — 내부에서 MODALS 순서로 정렬해
            # seg_model.extract_det_pyramid(list) 를 호출한다 (dict 를 직접 주면
            # list(dict) 가 키 문자열이 되어 터진다).
            modals = [k for k in batch
                      if isinstance(batch[k], torch.Tensor) and batch[k].dim() == 4]
            sample = {m: batch[m].to(dev) for m in modals}
            gate_store.clear()
            pyr = model.extract_fpn_features(sample)
            # extract_fpn_features 는 config 의 DET_LEVELS 만 돌려준다(D1 은 [2]=stride-16).
            # 따라서 리스트가 1개면 그게 곧 우리가 원하는 stride-16 레벨이다.
            feat = pyr[args.level] if len(pyr) > args.level else pyr[0]
            feat = F.interpolate(feat, size=(args.grid, args.grid),
                                 mode='bilinear', align_corners=False)

            w = gate_store.get('w')
            if w is not None:
                # (m,B,1,h,w) -> (B,m,grid,grid)
                w = w.squeeze(2).permute(1, 0, 2, 3)
                w = F.interpolate(w, size=(args.grid, args.grid),
                                  mode='bilinear', align_corners=False)

            # [2] 다중 스케일: teacher 는 DET_LEVELS 상 단일 레벨만 주므로
            # 그 레벨을 student 의 각 격자(80/40/20)로 리샘플해 저장한다.
            multi = {}
            if args.multiscale:
                base = pyr[args.level] if len(pyr) > args.level else pyr[0]
                for g in (args.grid * 2, args.grid, args.grid // 2):     # 80,40,20
                    multi[f'feat{g}'] = F.interpolate(
                        base, size=(g, g), mode='bilinear', align_corners=False)

            for i in range(feat.shape[0]):
                fn = batch['file_name'][i]
                stem = os.path.splitext(fn.replace('/rgb/', '_').replace('/', '_'))[0]
                d = {'feat': feat[i].half().cpu().numpy()}
                for k2, v2 in multi.items():
                    d[k2] = v2[i].half().cpu().numpy()
                if w is not None:
                    d['gate'] = w[i].float().cpu().numpy()    # (m, grid, grid) img,lidar,thermal
                np.savez_compressed(f'{args.out}/{stem}.npz', **d)
                n += 1
            if n % 500 < feat.shape[0]:
                print(f'  ..{n}')

    print(f'[teacher-feat] {n} 개 저장 -> {args.out}  '
          f'(feat {feat.shape[1]}x{args.grid}x{args.grid} fp16'
          f'{", gate 3ch" if w is not None else ""})')


if __name__ == '__main__':
    main()
