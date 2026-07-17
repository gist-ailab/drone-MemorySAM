#!/usr/bin/env python3
"""
tools/probe_cefr_routing.py — [P37a] CEFR per-class 라우팅 분화 프로브.

질문: "CEFR가 실제로 클래스별로 다른 모달리티를 골랐는가?"
전역 평균 w̄(p37/cefr_w_*)는 uniform으로 보여도 per-class w_{i,k}는 분화될 수
있다(P30 router의 'uniform 평균' 함정과 동일). 이 프로브는 eval 스태시
`fusion.cefr._last_cefr_w`(m,B,K,h,w)를 N장에 걸쳐 집계해:

  - per-class × per-modality 평균 w (K×m 매트릭스)
  - per-class winner modality + winner margin (분화 크기)
  - per-class 라우팅 엔트로피 (0=완전 commit, log m=완전 uniform)
  - 분화 랭킹: |w_winner − 1/m| 상위/하위 클래스
  - σ(a) blend 개방도, (있으면) P36 router w 평균과 병기

Usage:
  python tools/probe_cefr_routing.py --cfg <p37a.yaml> --model_path <ckpt> \
    --dataset-root <MUSES|DELIVER> --split val --max-imgs 60 --gpu 0 --out <prefix>

Output: <out>.json + <out>.md
"""
import argparse, os, sys, json, math
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import val as V


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--split', default='val')
    ap.add_argument('--max-imgs', type=int, default=60)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

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
    core = model.module if hasattr(model, 'module') else model
    cefr = getattr(getattr(core, 'fusion', None), 'cefr', None)
    if cefr is None:
        sys.exit('[probe_cefr] 이 모델에는 CEFR가 없습니다 (fusion.cefr 부재)')

    dataset, _ = V.create_dataset(ds_cfg, args.split, transform, args.split,
                                  macvi=False, eval_day=False)
    CLASSES = list(dataset.CLASSES)
    K = len(CLASSES)
    modals = ds_cfg.get('MODALS', [])
    n = min(args.max_imgs, len(dataset))

    w_sum = None      # (m, K) Σ spatial-mean of per-class routing weight
    m = None
    for idx in range(n):
        s = dataset[idx]
        images = s[0]
        imgs = [im.unsqueeze(0).to(device) for im in images]
        with torch.no_grad():
            model(imgs, multimask_output=True)
        w = cefr._last_cefr_w            # (m, B, K, h, w)
        if w is None:
            sys.exit('[probe_cefr] _last_cefr_w 미기록 — fusion.py 스태시 패치 확인')
        wm = w.float().mean(dim=(1, 3, 4)).cpu().numpy()   # (m, K)
        if w_sum is None:
            m = wm.shape[0]
            w_sum = np.zeros_like(wm, dtype=np.float64)
        w_sum += wm
    W = w_sum / n                                           # (m, K) mean routing
    names = (modals + [f'mod{i}' for i in range(m)])[:m]
    uni = 1.0 / m
    ent = -(W * np.clip(np.log(W + 1e-8), -50, 0)).sum(axis=0)          # (K,)
    winner = W.argmax(axis=0)
    margin = W.max(axis=0) - uni

    rows = []
    for k in range(K):
        rows.append({'class': CLASSES[k],
                     'w': {names[i]: round(float(W[i, k]), 4) for i in range(m)},
                     'winner': names[int(winner[k])],
                     'margin_vs_uniform': round(float(margin[k]), 4),
                     'routing_entropy': round(float(ent[k]), 4)})
    rows_sorted = sorted(rows, key=lambda r: -r['margin_vs_uniform'])
    report = {
        'model': Path(args.model_path).stem, 'n_imgs': n, 'modals': names,
        'sigma_a': getattr(cefr, '_last_sigma_a', None),
        'uniform_ref': round(uni, 4), 'max_entropy_ref': round(math.log(m), 4),
        'global_mean_w': {names[i]: round(float(W[i].mean()), 4) for i in range(m)},
        'mean_margin': round(float(margin.mean()), 4),
        'n_classes_committed_10pt': int((margin > 0.10).sum()),
        'per_class': rows_sorted,
    }
    Path(args.out + '.json').write_text(json.dumps(report, indent=1))

    lines = [f"# CEFR per-class routing probe — `{report['model']}` ({args.split}, n={n})",
             f"- σ(a) blend 개방도: **{report['sigma_a']}** (init 0.018; 클수록 CEFR 경로 채택)",
             f"- 전역 평균 w: {report['global_mean_w']} (uniform={uni:.3f})",
             f"- **분화 판정**: margin>0.10 클래스 **{report['n_classes_committed_10pt']}/{K}**,"
             f" 평균 margin {report['mean_margin']:.3f}, max-entropy={math.log(m):.3f}",
             "", "| class | " + " | ".join(names) + " | winner | margin | entropy |",
             "|---|" + "---|" * (m + 3)]
    for r in rows_sorted:
        lines.append("| " + r['class'] + " | "
                     + " | ".join(f"{r['w'][nm]:.3f}" for nm in names)
                     + f" | **{r['winner']}** | {r['margin_vs_uniform']:.3f} | {r['routing_entropy']:.3f} |")
    Path(args.out + '.md').write_text('\n'.join(lines))
    print(f"[probe_cefr] committed(>10pt)={report['n_classes_committed_10pt']}/{K} "
          f"mean_margin={report['mean_margin']:.3f} sigma_a={report['sigma_a']}")
    print(f"[probe_cefr] wrote {args.out}.json / .md")


if __name__ == '__main__':
    main()
