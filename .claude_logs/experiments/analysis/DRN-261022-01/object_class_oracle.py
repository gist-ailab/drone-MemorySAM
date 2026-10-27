#!/usr/bin/env python3
"""객체 단위 클래스 오라클(DRN-261022-01 (a), DELIVER).

예측 맵에서 클래스별 8연결 성분을 '예측 객체'로 보고, 각 성분의 라벨을 그 성분 안 유효 GT
(255 제외) 픽셀의 다수 클래스로 바꾼다. 성분 안에 유효 GT 가 없으면 원래 라벨을 둔다.
마스크(성분 모양)는 고정하고 클래스만 바꾸므로, 오라클 mIoU − 원래 mIoU = '클래스만 맞히면 얻는 여유'.
채점은 tools/baseline_failure/common.py 규약(hist[gt,pred], 부재 클래스 IoU=0 포함 25클래스 평균).
GT 와 예측 크기가 다르면 예측을 GT 크기로 최근접 보간(thin_error_anatomy 와 동일).

사용: python object_class_oracle.py --repo <repo> --gt <gt_dir> --models NAME=DIR ... --out <dir> [--workers N]
"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--gt', required=True)
ap.add_argument('--models', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--restrict', default='', help='쉼표로 준 클래스 이름 집합. 주면 예측 클래스와 GT 다수 클래스가 둘 다 이 집합에 속할 때만 라벨을 바꾼다(사물↔사물 혼동만 보정)')
args = ap.parse_args()
sys.path.insert(0, args.repo)
from tools.baseline_failure import common  # noqa: E402
N = common.N_CLASSES
S8 = np.ones((3, 3), dtype=np.uint8)
RESTRICT = None
if args.restrict:
    names = [x.strip() for x in args.restrict.split(',') if x.strip()]
    bad = [x for x in names if x not in common.CLASSES]
    assert not bad, f'알 수 없는 클래스: {bad}'
    RESTRICT = np.zeros(N, dtype=bool)
    RESTRICT[[common.CLASSES.index(x) for x in names]] = True


def relabel(pred, gt):
    """반환: 오라클 예측, (원래 클래스→새 클래스) 바뀐 픽셀 수 행렬 N×N."""
    out = pred.copy()
    moved = np.zeros((N, N), dtype=np.int64)
    gtc = np.where((gt < N), gt, N).astype(np.int64)  # 255 → N(무효)
    for c in np.unique(pred):
        if c >= N:
            continue
        lab, n = ndimage.label(pred == c, structure=S8)
        if n == 0:
            continue
        cnt = np.bincount(lab.ravel() * (N + 1) + gtc.ravel(), minlength=(n + 1) * (N + 1)).reshape(n + 1, N + 1)
        valid = cnt[:, :N]
        has = valid.sum(1) > 0
        maj = valid.argmax(1)
        if RESTRICT is not None:
            has = has & RESTRICT[c] & RESTRICT[maj]
        new = np.where(has, maj, c).astype(np.uint8)
        new[0] = 0  # 배경 성분(0)은 쓰지 않음
        m = lab > 0
        newmap = new[lab[m]]
        out[m] = newmap
        changed = newmap != c
        if changed.any():
            moved[c] += np.bincount(newmap[changed].astype(np.int64), minlength=N)
    return out, moved


def work(item):
    image_id, gt_path, pred_paths = item
    gt = common.load_label_png(gt_path)
    res = []
    for p in pred_paths:
        pred = common.load_label_png(p)
        if pred.shape != gt.shape:
            pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
        orc, moved = relabel(pred, gt)
        res.append((common.confusion_matrix(pred, gt), common.confusion_matrix(orc, gt), moved))
    return res


def main():
    names, dirs = [], []
    for spec in args.models:
        n, d = spec.split('=', 1)
        d = Path(d); d = d / 'pred' if (d / 'pred').is_dir() else d
        names.append(n); dirs.append(d)
    gt_idx = common.index_label_pngs(args.gt, require_nested=True)
    m_idx = [common.index_label_pngs(d, require_nested=True) for d in dirs]
    ids = sorted(set(gt_idx).intersection(*[set(m) for m in m_idx]))
    if args.limit:
        ids = ids[:args.limit]
    print(f'[oracle] images={len(ids)} models={names}', flush=True)
    items = [(i, gt_idx[i], [m[i] for m in m_idx]) for i in ids]
    acc = [[np.zeros((N, N), np.int64) for _ in range(3)] for _ in names]
    with ProcessPoolExecutor(args.workers) as ex:
        for k, res in enumerate(ex.map(work, items, chunksize=8)):
            for j, (a, b, c) in enumerate(res):
                acc[j][0] += a; acc[j][1] += b; acc[j][2] += c
            if (k + 1) % 200 == 0:
                print(f'[oracle] {k + 1}/{len(items)}', flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {'n_images': len(ids), 'classes': common.CLASSES, 'restrict': args.restrict, 'models': {}}
    for j, n in enumerate(names):
        cm0, cm1, mv = acc[j]
        iou0, m0 = common.global_miou_from_cm(cm0)
        iou1, m1 = common.global_miou_from_cm(cm1)
        np.save(out / f'{n}_cm_orig.npy', cm0); np.save(out / f'{n}_cm_oracle.npy', cm1); np.save(out / f'{n}_moved.npy', mv)
        pairs = sorted(((int(mv[a, b]), common.CLASSES[a], common.CLASSES[b]) for a in range(N) for b in range(N) if mv[a, b] > 0), reverse=True)[:15]
        summary['models'][n] = {'mIoU_orig': m0, 'mIoU_oracle': m1, 'delta': round(m1 - m0, 4),
                                'per_class_orig': dict(zip(common.CLASSES, iou0)),
                                'per_class_oracle': dict(zip(common.CLASSES, iou1)),
                                'top_relabel_pairs_px': [{'pred': a, 'oracle': b, 'px': px} for px, a, b in pairs]}
        print(f'[oracle] {n}: orig {m0:.4f} oracle {m1:.4f} delta {m1 - m0:+.4f}', flush=True)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print('[oracle] done', out, flush=True)


if __name__ == '__main__':
    main()
