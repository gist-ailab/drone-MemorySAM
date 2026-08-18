#!/usr/bin/env python3
"""tools/oracle_spatial_modality.py — Spatial-Modality Oracle Probe.

정본 설계: .claude_logs/decisions/2026-08-18-spatial-modality-oracle-probe-proposal.md
(§2 설계, §3 게이트, §6 실행).

목적: SoftMoE-LoRA(spatial×modality routing) 재개방의 **천장(상계)**을 학습 없이
측정한다. 각 비공집합 부분집합 S ⊆ MODALS 에 대해 S 밖 모달을 zero-fill 한 뒤
forward → per-pixel argmax 예측맵 P_S(x). 각 픽셀에서 GT를 맞히는 부분집합이 하나라도
있으면 그것을 채택하는 오라클 합성맵 O(x) 를 만들고, Δ = mIoU(O) − mIoU(P_full) 를 낸다.
Δ 는 "임의의 하드 픽셀별 모달-부분집합 라우터"의 상계다(오라클은 GT 치팅).

⚠️ 순수 추론 — 학습 코드 없음. 모든 forward 는 torch.no_grad().

참조: tools/eval_reliadino_ckpt.py 의 _DropModalityDataset(단일 모달 zero-fill)를
keep-subset(임의 부분집합 유지, 나머지 zero-fill)로 **일반화**했다.

핵심 정합성(스모크에서 assert): full 도 후보 부분집합 중 하나이므로 오라클은
full 을 이길 수만 있고 질 수 없다 → mIoU(O) ≥ mIoU(P_full) 항상 성립(단조성).
(증명: 오라클은 full-정답 픽셀은 그대로 두고 full-오답 픽셀 일부만 정답으로 바꾼다.
바뀐 픽셀은 오답→정답이므로 confusion 행렬에서 각 클래스 TP↑·FP↓·FN↓ → per-class
IoU 비감소 → mIoU 비감소.)

Usage (실제 ckpt — GPU forward 는 별도 승인 후 기동):
  PYTHONPATH=pylibs_p34:. python tools/oracle_spatial_modality.py \
    --cfg configs/jarvis-deliver_rgbdel_P46_ctr_c3only_lam005.yaml \
    --model_path outputs/.../epoch70_67.79_top1_checkpoint.pth \
    --mode both --gpu 0 \
    --out_dir /drone_nas/.../analysis_logs/oracle_spatial_modality_deliver

  # 일부 부분집합만 (재)캐시:
  ... --subsets img,depth,img+depth   # 나머지 캐시가 다 있어야 합성 단계 실행
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# CUDA_VISIBLE_DEVICES 는 torch import 전에 argv 에서 선반영 (eval_reliadino_ckpt.py 관례)
if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpu') + 1]
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')


# ───────────────────────── 순수 로직 (스모크에서 재사용, torch 불요) ─────────────────────────

def enumerate_subsets(n_modals):
    """비공집합 부분집합 전부를 tie-break 정본 순서로 반환.

    순서 = (크기 오름차순, 그다음 인덱스 튜플 사전순) — 결정적. 작은 부분집합 우선
    (모달 수가 적은 라우팅에 credit). 전체(full) 부분집합이 항상 리스트의 마지막.
    반환: list[tuple[int]] (각 원소 = 유지할 모달 인덱스들의 정렬 튜플).
    """
    subs = []
    for k in range(1, n_modals + 1):
        for c in itertools.combinations(range(n_modals), k):
            subs.append(tuple(c))
    subs.sort(key=lambda s: (len(s), s))
    return subs


def subset_bitmask(subset):
    """부분집합 → 비트마스크 int (캐시 파일명용, 안정적)."""
    bm = 0
    for i in subset:
        bm |= (1 << i)
    return bm


def subset_label(subset, modal_names):
    """부분집합 → 사람이 읽는 라벨, 예: ('img','depth') → 'img+depth'."""
    return '+'.join(modal_names[i] for i in subset)


def hist_from_pred(pred, gt, num_classes, ignore_label):
    """argmax 예측맵(pred)과 GT 로 confusion 히스토그램 누적분을 만든다.

    semseg.metrics.Metrics 와 동일 시맨틱: ignore·범위밖 픽셀 제외, hist[gt, pred].
    pred/gt: np.ndarray[int] 같은 shape. 반환: np.ndarray[float64] (C, C).
    """
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    keep = (gt != ignore_label) & (gt < num_classes) & (pred < num_classes)
    if not keep.any():
        return np.zeros((num_classes, num_classes), dtype=np.float64)
    g = gt[keep].astype(np.int64)
    p = pred[keep].astype(np.int64)
    flat = g * num_classes + p
    bc = np.bincount(flat, minlength=num_classes ** 2)[: num_classes ** 2]
    return bc.reshape(num_classes, num_classes).astype(np.float64)


def miou_from_hist(hist):
    """confusion 히스토그램 → (per-class IoU %, mIoU %). Metrics.compute_iou 와 동일식."""
    hist = np.asarray(hist, dtype=np.float64)
    diag = np.diag(hist)
    denom = hist.sum(0) + hist.sum(1) - diag
    with np.errstate(divide='ignore', invalid='ignore'):
        ious = np.where(denom > 0, diag / denom, 0.0)
    ious = np.nan_to_num(ious, nan=0.0)
    miou = float(ious.mean())
    return (ious * 100).tolist(), round(miou * 100, 4)


def oracle_synthesize(preds, gt, full_index):
    """픽셀별 오라클 합성.

    preds: np.ndarray[int] (S, H, W) — 부분집합별 argmax 예측맵 (tie-break 순서).
    gt:    np.ndarray[int] (H, W).
    full_index: full 부분집합의 preds 축 인덱스 (fallback 예측).

    반환:
      O:    (H, W) 합성 예측맵. any-correct 픽셀은 GT, 아니면 P_full.
      star: (H, W) int — 채택된 부분집합의 preds 인덱스(tie-break상 첫 정답),
            정답 부분집합이 하나도 없으면 -1.
    """
    preds = np.asarray(preds)
    gt = np.asarray(gt)
    S = preds.shape[0]
    correct = (preds == gt[None])            # (S, H, W)
    any_correct = correct.any(axis=0)        # (H, W)

    O = preds[full_index].copy()
    O[any_correct] = gt[any_correct]

    # star = tie-break 순서상 첫 정답 부분집합. preds 축은 이미 tie-break 순서라고 가정.
    star = np.full(gt.shape, -1, dtype=np.int64)
    for s in range(S):
        take = correct[s] & (star == -1)
        star[take] = s
    return O, star


# ───────────────────────── torch 경로 (main 전용, 실제 forward) ─────────────────────────

class _KeepSubsetDataset:
    """MODALS 기반 데이터셋을 감싸, keep-subset 밖 모달만 zero-fill 한다.

    eval_reliadino_ckpt._DropModalityDataset(단일 모달 zero-fill)의 일반화:
    keep_indices 에 없는 모든 모달 텐서를 0 으로 채운다(모델 구조 불변, 입력만 치환).
    """

    def __init__(self, base, keep_indices):
        import torch  # noqa
        self._torch = torch
        self.base = base
        self.keep = set(keep_indices)
        self.n_classes = base.n_classes
        self.CLASSES = base.CLASSES
        self.ignore_label = getattr(base, 'ignore_label', 255)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        imgs, label = self.base[i]
        imgs = list(imgs)
        for j in range(len(imgs)):
            if j not in self.keep:
                imgs[j] = self._torch.zeros_like(imgs[j])
        return imgs, label


def _deliver_condition(path):
    """DELIVER 파일 경로에서 weather 조건(cloud/fog/night/rain/sun)을 추출.

    경로 구조: <root>/img/<condition>/<split>/<scene>/<file>.png → 'img' 다음 파트.
    실패 시 'unknown'.
    """
    parts = Path(path).parts
    try:
        k = parts.index('img')
        return parts[k + 1]
    except (ValueError, IndexError):
        return 'unknown'


def _cache_predictions(model, base_ds, subset, num_classes, ignore_label, device,
                       batch_size, npy_path, gt_path):
    """subset 하나에 대한 argmax 예측맵 전체를 forward 해 npy 로 저장(+GT 없으면 함께).

    이미 npy_path 가 있으면 skip(재개 가능). 반환: (n_images,)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    if npy_path.exists() and (gt_path.exists() or subset is None):
        print(f"  [skip] {npy_path.name} 이미 존재 (재개)")
        return None

    save_gt = not gt_path.exists()
    wrapped = _KeepSubsetDataset(base_ds, subset)
    loader = DataLoader(wrapped, batch_size=batch_size, num_workers=4,
                        pin_memory=True, shuffle=False)

    preds_all, gts_all = [], []
    model.eval()
    with torch.no_grad():
        for bi, (images, labels) in enumerate(loader):
            images = [x.to(device, non_blocking=True) for x in images]
            labels = labels.to(device, non_blocking=True)
            output, _ = model(images, True)
            if output.shape[-2:] != labels.shape[-2:]:
                output = F.interpolate(output, size=labels.shape[-2:],
                                       mode='bilinear', align_corners=False)
            pred = output.argmax(dim=1).to(torch.uint8).cpu().numpy()  # (B,H,W)
            preds_all.append(pred)
            if save_gt:
                gts_all.append(labels.to(torch.int16).cpu().numpy())
            if bi % 20 == 0:
                print(f"    forward {bi * batch_size}/{len(wrapped)}", flush=True)

    preds = np.concatenate(preds_all, axis=0)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, preds)
    print(f"  [saved] {npy_path.name}  shape={preds.shape}")
    if save_gt:
        gts = np.concatenate(gts_all, axis=0)
        # GT 는 0-24 + 255(ignore). int16 로 저장(255 안전).
        np.save(gt_path, gts)
        print(f"  [saved] {gt_path.name}  shape={gts.shape}")
    return preds.shape[0]


def _synthesize_split(subsets, modal_names, full_index, cache_dir, split,
                      num_classes, class_names, ignore_label, conditions=None):
    """캐시된 예측맵들로 오라클 합성 + Δ/분포 리포트 dict 생성.

    conditions: None 또는 길이 N 의 조건 라벨 리스트(per-condition Δ 용).
    """
    gt_path = cache_dir / f"gt_{split}.npy"
    gt_all = np.load(gt_path, mmap_mode='r')
    N = gt_all.shape[0]

    # 부분집합별 예측 memmap 열기
    pred_maps = []
    for sub in subsets:
        p = cache_dir / f"pred_{split}_mask{subset_bitmask(sub)}.npy"
        arr = np.load(p, mmap_mode='r')
        if arr.shape[0] != N:
            raise RuntimeError(
                f"캐시 길이 불일치: {p.name} n={arr.shape[0]} != gt n={N}. "
                f"--limit 를 바꿨거나 캐시가 오염됨 — 캐시를 지우고 재실행.")
        pred_maps.append(arr)

    C = num_classes
    hist_full = np.zeros((C, C), dtype=np.float64)
    hist_oracle = np.zeros((C, C), dtype=np.float64)
    # per-condition
    cond_hist = {}  # cond -> [hist_full, hist_oracle]
    # S* 분포 (부분집합 인덱스 -> valid 픽셀 credit 수), -1 = none
    star_counts = np.zeros(len(subsets) + 1, dtype=np.int64)  # 마지막 슬롯 = none(-1)

    for i in range(N):
        gt = np.asarray(gt_all[i]).astype(np.int64)
        preds = np.stack([np.asarray(pm[i]).astype(np.int64) for pm in pred_maps], axis=0)
        O, star = oracle_synthesize(preds, gt, full_index)
        P_full = preds[full_index]

        h_full = hist_from_pred(P_full, gt, C, ignore_label)
        h_orac = hist_from_pred(O, gt, C, ignore_label)
        hist_full += h_full
        hist_oracle += h_orac

        # S* 분포 (valid 픽셀만)
        valid = (gt != ignore_label) & (gt < C)
        sv = star[valid]
        star_counts[:-1] += np.bincount(sv[sv >= 0], minlength=len(subsets))
        star_counts[-1] += int((sv < 0).sum())

        if conditions is not None:
            c = conditions[i]
            if c not in cond_hist:
                cond_hist[c] = [np.zeros((C, C)), np.zeros((C, C))]
            cond_hist[c][0] += h_full
            cond_hist[c][1] += h_orac

        if i % 100 == 0:
            print(f"    synth {i}/{N}", flush=True)

    iou_full, miou_full = miou_from_hist(hist_full)
    iou_orac, miou_orac = miou_from_hist(hist_oracle)
    delta = round(miou_orac - miou_full, 4)

    per_class = []
    for ci, name in enumerate(class_names):
        per_class.append({
            'class': name,
            'iou_full': round(iou_full[ci], 2),
            'iou_oracle': round(iou_orac[ci], 2),
            'delta': round(iou_orac[ci] - iou_full[ci], 2),
        })

    # S* 분포 정리
    total_valid = int(star_counts.sum())
    star_dist = []
    for si, sub in enumerate(subsets):
        cnt = int(star_counts[si])
        star_dist.append({
            'subset': subset_label(sub, modal_names),
            'size': len(sub),
            'pixels': cnt,
            'frac': round(cnt / total_valid, 4) if total_valid else 0.0,
        })
    none_cnt = int(star_counts[-1])
    # 요약 지표: full 채택 비율, 단일모달 채택 비율
    frac_full = next((d['frac'] for d in star_dist
                      if d['subset'] == subset_label(subsets[full_index], modal_names)), 0.0)
    frac_single = round(sum(d['frac'] for d in star_dist if d['size'] == 1), 4)
    frac_none = round(none_cnt / total_valid, 4) if total_valid else 0.0

    per_condition = None
    if cond_hist:
        per_condition = {}
        for c, (hf, ho) in sorted(cond_hist.items()):
            _, mf = miou_from_hist(hf)
            _, mo = miou_from_hist(ho)
            per_condition[c] = {
                'miou_full': mf, 'miou_oracle': mo, 'delta': round(mo - mf, 4)}

    return {
        'split': split,
        'n_images': int(N),
        'modals': modal_names,
        'num_subsets': len(subsets),
        'miou_full': miou_full,
        'miou_oracle': miou_orac,
        'delta': delta,
        'per_class': per_class,
        'per_condition': per_condition,
        'star_distribution': star_dist,
        'star_none_pixels': none_cnt,
        'summary': {
            'frac_full': frac_full,
            'frac_single_modal': frac_single,
            'frac_no_correct_subset': frac_none,
        },
    }


def _print_report(rep):
    print(f"\n{'=' * 68}")
    print(f"[oracle-spatial-modality][{rep['split']}] n={rep['n_images']}  "
          f"modals={rep['modals']}  #subsets={rep['num_subsets']}")
    print(f"  mIoU(full)   = {rep['miou_full']:.4f}")
    print(f"  mIoU(oracle) = {rep['miou_oracle']:.4f}")
    print(f"  Δ            = {rep['delta']:+.4f}")
    print(f"  [게이트] Δ<0.3 폐쇄 · 0.3~1.0 경계 · ≥1.0 여지실재 "
          f"(정본: 2026-08-18-spatial-modality-oracle-probe-proposal §3)")
    s = rep['summary']
    print(f"  S* 분포: full={s['frac_full'] * 100:.1f}%  "
          f"single-modal={s['frac_single_modal'] * 100:.1f}%  "
          f"no-correct-subset={s['frac_no_correct_subset'] * 100:.1f}%")
    print(f"  per-class Δ (상위 변화):")
    top = sorted(rep['per_class'], key=lambda d: -abs(d['delta']))[:8]
    for d in top:
        print(f"    {d['class']:<14} full={d['iou_full']:6.2f} "
              f"oracle={d['iou_oracle']:6.2f}  Δ={d['delta']:+6.2f}")
    if rep['per_condition']:
        print(f"  per-condition Δ:")
        for c, v in rep['per_condition'].items():
            print(f"    {c:<14} full={v['miou_full']:6.2f} "
                  f"oracle={v['miou_oracle']:6.2f}  Δ={v['delta']:+6.2f}")


def _write_reports(rep, out_dir):
    split = rep['split']
    (out_dir / f"report_{split}.json").write_text(json.dumps(rep, indent=2))
    # per-class CSV
    lines = ['class,iou_full,iou_oracle,delta']
    for d in rep['per_class']:
        lines.append(f"{d['class']},{d['iou_full']},{d['iou_oracle']},{d['delta']}")
    (out_dir / f"report_{split}_perclass.csv").write_text('\n'.join(lines) + '\n')
    # S* 분포 CSV
    lines = ['subset,size,pixels,frac']
    for d in rep['star_distribution']:
        lines.append(f"{d['subset']},{d['size']},{d['pixels']},{d['frac']}")
    (out_dir / f"report_{split}_stardist.csv").write_text('\n'.join(lines) + '\n')
    print(f"  [written] {out_dir}/report_{split}.json (+perclass/stardist csv)")


def _parse_subset_filter(spec, subsets, modal_names):
    """--subsets 스펙 → 이번 실행에서 forward 할 부분집합 리스트.

    'all'(기본) 또는 콤마목록. 각 항목 = 모달명 '+'-조인 또는 'full'.
    """
    if spec in (None, '', 'all'):
        return list(subsets)
    label2sub = {subset_label(s, modal_names): s for s in subsets}
    full_label = subset_label(subsets[-1], modal_names)
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok == 'full':
            out.append(subsets[-1])
            continue
        # 'img+depth' 또는 단일 'img'; 모달 순서 무관하게 정규화
        names = tok.split('+')
        idxs = tuple(sorted(modal_names.index(n) for n in names))
        if idxs not in label2sub.values() and idxs not in [tuple(s) for s in subsets]:
            raise ValueError(f"--subsets 항목 '{tok}' 이 유효한 부분집합이 아님")
        out.append(idxs)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True,
                    help='평가 대상 ckpt (하드코딩 금지 — 반드시 인자로).')
    ap.add_argument('--mode', default='both', choices=['val', 'test', 'both'])
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--batch', type=int, default=None,
                    help='기본 = cfg EVAL.BATCH_SIZE (프로토콜 유지).')
    ap.add_argument('--dataset-root', default=None,
                    help='DATASET.ROOT 만 서버-로컬 경로로 override.')
    ap.add_argument('--out_dir', required=True,
                    help='결과 루트 (cache/ + report_*.json/csv). 캐시 재사용으로 재개 가능.')
    ap.add_argument('--subsets', default='all',
                    help="이번 실행에서 forward 할 부분집합('all' 또는 'img,img+depth,full' 등). "
                         "합성 단계는 15개 캐시가 모두 있을 때만 실행.")
    ap.add_argument('--limit', type=int, default=None,
                    help='디버그용: 앞 N 장만 사용(캐시 길이를 바꾸므로 기존 캐시와 섞지 말 것).')
    ap.add_argument('--no-synth', action='store_true',
                    help='forward/캐시만 하고 오라클 합성은 건너뛴다.')
    args = ap.parse_args()

    import yaml
    import torch
    from torch.utils.data import Subset
    from semseg.augmentations_mm import get_val_augmentation
    from semseg.datasets import DELIVER, MUSES  # noqa: F401 (eval 로 동적 접근)
    import semseg.datasets as _ds_mod  # noqa
    from semseg.models.reliadino.model import build_reliadino

    cfg = yaml.safe_load(open(args.cfg))
    dataset_cfg, eval_cfg = cfg['DATASET'], cfg['EVAL']
    if args.dataset_root:
        dataset_cfg['ROOT'] = args.dataset_root
    device = torch.device('cuda')
    ds_name = dataset_cfg['NAME']
    modal_names = list(dataset_cfg['MODALS'])
    M = len(modal_names)
    ignore_label = dataset_cfg.get('IGNORE_LABEL', 255)

    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], dataset_cfg=dataset_cfg)
    DS = getattr(_ds_mod, ds_name)
    probe = DS(dataset_cfg['ROOT'], 'val', valtransform, modal_names)
    num_classes, class_names = probe.n_classes, probe.CLASSES

    subsets = enumerate_subsets(M)
    full_index = len(subsets) - 1  # enumerate_subsets 상 full 이 마지막
    assert subsets[full_index] == tuple(range(M))
    print(f"[oracle] modals={modal_names}  #subsets={len(subsets)} "
          f"(full={subset_label(subsets[full_index], modal_names)})")

    # 모델 로드 (eval_reliadino_ckpt 와 동일 계약)
    model = build_reliadino(cfg, num_classes)
    ck = torch.load(args.model_path, map_location='cpu')
    state = ck.get('model_state_dict', ck)
    msg = model.load_state_dict(state, strict=False)
    print(f"[oracle] ckpt={Path(args.model_path).name} epoch={ck.get('epoch', '?')} "
          f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    assert len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0, \
        f"state_dict mismatch: {msg.missing_keys[:3]} {msg.unexpected_keys[:3]}"
    model = model.to(device)

    bs = args.batch or eval_cfg['BATCH_SIZE']
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'subset_legend.json').write_text(json.dumps(
        {subset_bitmask(s): subset_label(s, modal_names) for s in subsets}, indent=2))

    to_run = _parse_subset_filter(args.subsets, subsets, modal_names)
    splits = ['val', 'test'] if args.mode == 'both' else [args.mode]

    for split in splits:
        print(f"\n===== split={split} =====")
        base_ds = probe if split == 'val' else DS(
            dataset_cfg['ROOT'], split, valtransform, modal_names)
        if args.limit is not None:
            base_ds = Subset(base_ds, range(min(args.limit, len(base_ds))))
            base_ds.n_classes = num_classes
            base_ds.CLASSES = class_names
            base_ds.ignore_label = ignore_label

        gt_path = cache_dir / f"gt_{split}.npy"
        for sub in to_run:
            bm = subset_bitmask(sub)
            npy_path = cache_dir / f"pred_{split}_mask{bm}.npy"
            print(f"  subset={subset_label(sub, modal_names)} (mask {bm})")
            _cache_predictions(model, base_ds, sub, num_classes, ignore_label,
                               device, bs, npy_path, gt_path)

        if args.no_synth:
            continue
        # 합성: 전 부분집합 캐시가 있어야 함
        missing = [subset_label(s, modal_names) for s in subsets
                   if not (cache_dir / f"pred_{split}_mask{subset_bitmask(s)}.npy").exists()]
        if missing or not gt_path.exists():
            print(f"  [synth skip] 캐시 미완: {missing if missing else 'GT'} — "
                  f"나머지 부분집합을 채운 뒤 재실행하면 합성이 돈다.")
            continue

        # per-condition (DELIVER 만)
        conditions = None
        raw = base_ds.dataset if isinstance(base_ds, Subset) else base_ds
        if hasattr(raw, 'files'):
            idxs = base_ds.indices if isinstance(base_ds, Subset) else range(len(raw.files))
            conditions = [_deliver_condition(raw.files[i]) for i in idxs]

        rep = _synthesize_split(subsets, modal_names, full_index, cache_dir, split,
                                num_classes, class_names, ignore_label, conditions)
        _print_report(rep)
        _write_reports(rep, out_dir)


if __name__ == '__main__':
    main()
