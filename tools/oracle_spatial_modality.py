#!/usr/bin/env python3
"""tools/oracle_spatial_modality.py — Spatial-Modality Oracle Probe.

정본 설계: .claude_logs/decisions/2026-08-18-spatial-modality-oracle-probe-proposal.md
(§2 설계, §3 게이트, §6 실행). 통제 실험(선택 입도 스윕·독립성 널) 정본:
.claude_logs/experiments/analysis/2026-08-19-spatial-modality-oracle-verdict.md §3.

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

## 통제 실험 (2026-08-19 추가, --select-granularity / --null-shuffle)

첫 측정(Δ=+8.66 val/+8.29 test)이 사전등록 게이트(≥1.0)를 8배 초과했으나, 이 오라클은
느슨한 상계다: ① union-over-15 팽창(15개 중 하나라도 맞으면 성공 = 예측기가 서로 다르게
틀리기만 해도 union 이 커진다) ② zero-fill 이 OOD 입력이라 예측이 요동(추가 팽창). 두
통제로 "진짜 공간 구조"와 "팽창 바닥"을 분리한다:

- **선택 입도 스윕**(`--select-granularity`): 픽셀별 자유 선택을 B×B 블록 단위 강제
  선택으로 좁힌다. 진짜 공간 응집(인접 픽셀이 같은 모달을 필요로 함)이 있다면 블록이
  커져도 Δ가 어느 정도 유지되고, 픽셀 단위 노이즈일 뿐이라면 블록이 커질수록 Δ가 급격히
  무너진다. B=1 은 기존 픽셀별 오라클과 **byte-동일**해야 한다(회귀 가드).
- **독립성 널**(`--null-shuffle`): 각 부분집합의 정답맵(correct=pred==gt)을 개수는
  보존한 채 공간적으로 무작위 셔플한 뒤 같은 합성을 하면, 실제 공간 정렬 없이 순수
  union 통계 효과만으로 나오는 Δ("팽창 바닥")를 얻는다. 관측 Δ가 이 널과 비슷하면
  전부 팽창, 유의하게 크면 실제 구조.

두 플래그 다 **합성 단계 전용**이다 — 이미 캐시(pred_*.npy/gt_*.npy)가 완비돼 있으면
forward 를 다시 하지 않고, 모델 로드조차 생략한다(GPU 불요, 순수 numpy 후처리).

Usage (실제 ckpt — GPU forward 는 별도 승인 후 기동):
  PYTHONPATH=pylibs_p34:. python tools/oracle_spatial_modality.py \
    --cfg configs/jarvis-deliver_rgbdel_P46_ctr_c3only_lam005.yaml \
    --model_path outputs/.../epoch70_67.79_top1_checkpoint.pth \
    --mode both --gpu 0 \
    --out_dir /drone_nas/.../analysis_logs/oracle_spatial_modality_deliver

  # 일부 부분집합만 (재)캐시:
  ... --subsets img,depth,img+depth   # 나머지 캐시가 다 있어야 합성 단계 실행

  # 통제 실험 (캐시 완비 후, GPU 불요 — model_path/cfg 는 그대로 넘겨도 무해):
  ... --select-granularity 1,8,16,32,64 --null-shuffle 20
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


def oracle_synthesize_blocked(preds, gt, full_index, block):
    """[통제1: 선택 입도] B×B 블록 단위로 부분집합을 강제 선택하는 오라클 합성.

    block<=1 이면 oracle_synthesize 와 **완전히 동일한 O** 를 낸다(회귀 보장 — 호출부에서
    직접 검증 가능하도록 그대로 위임한다).

    block>1: 이미지를 겹치지 않는 B×B 블록(가장자리는 나머지 크기 그대로)으로 나누고,
    각 블록 안에서 정답 픽셀 수(correct=pred==gt, oracle_synthesize 와 동일 정의)가
    최대인 부분집합을 골라 그 예측을 블록 **전체**(맞은 픽셀뿐 아니라 틀린 픽셀도)에
    대입한다. 동점은 tie-break 순서상 먼저 나오는(=인덱스가 작은) 부분집합 —
    np.argmax 의 첫-최댓값 반환 동작과 그대로 일치한다. 블록 안에 정답 부분집합이
    하나도 없으면(전부 count=0) full_index 로 폴백한다(oracle_synthesize 의
    no-correct-subset 폴백과 동일 규약 — 이래야 block=1 이 픽셀별과 정확히 같아진다).

    preds: (S,H,W), gt: (H,W). 반환: O_block (H,W), preds.dtype.
    """
    preds = np.asarray(preds)
    gt = np.asarray(gt)
    if block <= 1:
        O, _ = oracle_synthesize(preds, gt, full_index)
        return O

    S, H, W = preds.shape
    correct = (preds == gt[None])  # (S,H,W) — oracle_synthesize 와 동일 정의(ignore 별도 필터 없음)
    O = np.empty((H, W), dtype=preds.dtype)
    for y0 in range(0, H, block):
        y1 = min(y0 + block, H)
        for x0 in range(0, W, block):
            x1 = min(x0 + block, W)
            counts = correct[:, y0:y1, x0:x1].reshape(S, -1).sum(axis=1)
            best_s = int(np.argmax(counts))
            if counts[best_s] == 0:
                best_s = full_index
            O[y0:y1, x0:x1] = preds[best_s, y0:y1, x0:x1]
    return O


def oracle_synthesize_null(preds, gt, full_index, rng):
    """[통제2: 독립성 널] 정확도-보존 공간 셔플 — "팽창 바닥" 1회 trial.

    각 부분집합의 correct 맵(H,W, boolean, True 개수 불변)을 이미지 안에서 무작위
    순열(위치만 재배치)한 뒤, 그 셔플된 correctness 로 any_correct 를 다시 계산해
    O_null(x) = GT(x) if any_correct_shuffled(x) else P_full(x) 를 합성한다.
    실제 P_full 예측값 자체는 셔플하지 않는다(그대로 사용) — 오직 "어느 픽셀이
    어떤 subset 에 의해 맞았는가"라는 위치 정보만 흔들어, 진짜 공간 정렬 없이
    union(15) 만으로 통계적으로 얼마나 부풀려지는지를 측정한다.

    preds: (S,H,W), gt: (H,W), rng: np.random.Generator. 반환: O_null (H,W).
    """
    preds = np.asarray(preds)
    gt = np.asarray(gt)
    S, H, W = preds.shape
    correct = (preds == gt[None]).reshape(S, -1)  # (S, H*W)
    shuffled = np.empty_like(correct)
    for s in range(S):
        idx = rng.permutation(H * W)
        shuffled[s] = correct[s][idx]
    any_correct = shuffled.any(axis=0).reshape(H, W)

    O = preds[full_index].copy()
    O[any_correct] = gt[any_correct]
    return O


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
                      num_classes, class_names, ignore_label, conditions=None,
                      granularities=None, n_null=0, null_seed=0):
    """캐시된 예측맵들로 오라클 합성 + Δ/분포 리포트 dict 생성.

    conditions: None 또는 길이 N 의 조건 라벨 리스트(per-condition Δ 용).
    granularities: None/[] 또는 int 리스트 — [통제1] 선택 입도 스윕(예: [1,8,16,32,64]).
                   비어있으면 기존과 완전히 동일한 dict 를 반환한다(새 필드 없음).
    n_null: [통제2] 독립성 널 셔플 trial 수. 0 이면 계산 안 함(새 필드 없음).
    null_seed: 널 셔플 재현용 시드.

    🔴 회귀 가드: granularities/n_null 이 기본값(둘 다 비활성)이면 기존 필드
    (miou_full/miou_oracle/delta/per_class/per_condition/star_distribution/
    star_none_pixels/summary) 는 이전 버전과 byte-동일하다 — 이 계산 경로는
    전혀 변경하지 않았다(신규 필드는 조건부로만 추가).
    """
    granularities = list(granularities) if granularities else []

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

    # [통제1] 입도별 누적 히스토그램
    hist_blocked = {b: np.zeros((C, C), dtype=np.float64) for b in granularities}
    # [통제2] 널 셔플 trial별 누적 히스토그램
    rng = np.random.default_rng(null_seed) if n_null > 0 else None
    hist_null = [np.zeros((C, C), dtype=np.float64) for _ in range(n_null)]

    for i in range(N):
        gt = np.asarray(gt_all[i]).astype(np.int64)
        preds = np.stack([np.asarray(pm[i]).astype(np.int64) for pm in pred_maps], axis=0)
        O, star = oracle_synthesize(preds, gt, full_index)
        P_full = preds[full_index]

        h_full = hist_from_pred(P_full, gt, C, ignore_label)
        h_orac = hist_from_pred(O, gt, C, ignore_label)
        hist_full += h_full
        hist_oracle += h_orac

        for b in granularities:
            if b <= 1:
                hist_blocked[b] += h_orac  # 픽셀별과 동일 — 재계산 없이 재사용(회귀 보장)
            else:
                Ob = oracle_synthesize_blocked(preds, gt, full_index, b)
                hist_blocked[b] += hist_from_pred(Ob, gt, C, ignore_label)

        for t in range(n_null):
            Onull = oracle_synthesize_null(preds, gt, full_index, rng)
            hist_null[t] += hist_from_pred(Onull, gt, C, ignore_label)

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

    result = {
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

    if granularities:
        granularity_deltas = {}
        for b in granularities:
            _, m_b = miou_from_hist(hist_blocked[b])
            granularity_deltas[str(b)] = round(m_b - miou_full, 4)
        result['granularity_deltas'] = granularity_deltas

    if n_null > 0:
        null_deltas = []
        for t in range(n_null):
            _, m_t = miou_from_hist(hist_null[t])
            null_deltas.append(round(m_t - miou_full, 4))
        result['null_shuffle'] = {
            'n_trials': n_null,
            'seed': null_seed,
            'mean_delta': round(float(np.mean(null_deltas)), 4),
            'std_delta': round(float(np.std(null_deltas)), 4),
            'trials': null_deltas,
        }

    return result


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
    if 'granularity_deltas' in rep:
        print(f"  [통제1] 선택 입도별 Δ:")
        for b, d in rep['granularity_deltas'].items():
            print(f"    block={b:<4} Δ={d:+.4f}")
        print(f"    게이트: 블록16 Δ<1.0 & 관측≈널 → H16 반증 / "
              f"블록16 Δ≥2.0 & 관측≫널 → 공간응집 실재")
    if 'null_shuffle' in rep:
        ns = rep['null_shuffle']
        print(f"  [통제2] 독립성 널(n={ns['n_trials']}): "
              f"mean_Δ={ns['mean_delta']:+.4f}  std={ns['std_delta']:.4f}  "
              f"관측Δ={rep['delta']:+.4f}  (관측−널={rep['delta'] - ns['mean_delta']:+.4f})")


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


def _parse_int_list(spec):
    """콤마 구분 정수 목록 문자열 → list[int]. 빈 문자열/None 이면 []."""
    if not spec:
        return []
    return [int(x) for x in spec.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True,
                    help='평가 대상 ckpt (하드코딩 금지 — 반드시 인자로). '
                         '캐시가 이미 완비돼 있으면 실제로 로드되지 않는다.')
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
    ap.add_argument('--select-granularity', default='',
                    help='[통제1] 콤마 구분 블록 크기 목록(예: 1,8,16,32,64). 합성 단계 전용 — '
                         '캐시가 이미 있으면 forward 재실행 없이 numpy 후처리만 한다. '
                         'B=1 은 기존 픽셀별 오라클과 byte-동일(회귀 가드).')
    ap.add_argument('--null-shuffle', type=int, default=0,
                    help='[통제2] 정확도-보존 공간 셔플 trial 수(0=off, 기본). '
                         '캐시 재사용, GPU/forward 불요.')
    ap.add_argument('--null-seed', type=int, default=0,
                    help='--null-shuffle 재현용 시드(기본 0).')
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(open(args.cfg))
    dataset_cfg, eval_cfg = cfg['DATASET'], cfg['EVAL']
    if args.dataset_root:
        dataset_cfg['ROOT'] = args.dataset_root
    ds_name = dataset_cfg['NAME']
    modal_names = list(dataset_cfg['MODALS'])
    M = len(modal_names)
    ignore_label = dataset_cfg.get('IGNORE_LABEL', 255)

    subsets = enumerate_subsets(M)
    full_index = len(subsets) - 1  # enumerate_subsets 상 full 이 마지막
    assert subsets[full_index] == tuple(range(M))
    print(f"[oracle] modals={modal_names}  #subsets={len(subsets)} "
          f"(full={subset_label(subsets[full_index], modal_names)})")

    granularities = _parse_int_list(args.select_granularity)
    n_null = args.null_shuffle

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'subset_legend.json').write_text(json.dumps(
        {subset_bitmask(s): subset_label(s, modal_names) for s in subsets}, indent=2))

    to_run = _parse_subset_filter(args.subsets, subsets, modal_names)
    splits = ['val', 'test'] if args.mode == 'both' else [args.mode]

    def _split_needs_forward(split):
        gt_path = cache_dir / f"gt_{split}.npy"
        if to_run and not gt_path.exists():
            return True
        for sub in to_run:
            bm = subset_bitmask(sub)
            if not (cache_dir / f"pred_{split}_mask{bm}.npy").exists():
                return True
        return False

    needs_forward = any(_split_needs_forward(s) for s in splits)

    from semseg.augmentations_mm import get_val_augmentation
    from semseg.datasets import DELIVER, MUSES  # noqa: F401 (eval 로 동적 접근)
    import semseg.datasets as _ds_mod  # noqa

    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'], dataset_cfg=dataset_cfg)
    DS = getattr(_ds_mod, ds_name)
    probe = DS(dataset_cfg['ROOT'], 'val', valtransform, modal_names)
    num_classes, class_names = probe.n_classes, probe.CLASSES

    # 모델은 forward 가 실제로 필요할 때만 로드한다 — 통제 실험(합성 단계 전용,
    # --select-granularity/--null-shuffle)은 캐시만 있으면 GPU/ckpt 없이 돈다.
    model = None
    device = None
    if needs_forward:
        import torch
        from semseg.models.reliadino.model import build_reliadino

        device = torch.device('cuda')
        model = build_reliadino(cfg, num_classes)
        ck = torch.load(args.model_path, map_location='cpu')
        state = ck.get('model_state_dict', ck)
        msg = model.load_state_dict(state, strict=False)
        print(f"[oracle] ckpt={Path(args.model_path).name} epoch={ck.get('epoch', '?')} "
              f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        assert len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0, \
            f"state_dict mismatch: {msg.missing_keys[:3]} {msg.unexpected_keys[:3]}"
        model = model.to(device)
    else:
        print("[oracle] 캐시 완비 — forward/모델로드 생략(합성/통제 전용 실행, GPU 불요)")

    bs = args.batch or eval_cfg['BATCH_SIZE']

    for split in splits:
        print(f"\n===== split={split} =====")
        base_ds = probe if split == 'val' else DS(
            dataset_cfg['ROOT'], split, valtransform, modal_names)
        if args.limit is not None:
            from torch.utils.data import Subset
            base_ds = Subset(base_ds, range(min(args.limit, len(base_ds))))
            base_ds.n_classes = num_classes
            base_ds.CLASSES = class_names
            base_ds.ignore_label = ignore_label

        gt_path = cache_dir / f"gt_{split}.npy"
        if needs_forward:
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
        raw = base_ds.dataset if hasattr(base_ds, 'dataset') else base_ds
        if hasattr(raw, 'files'):
            idxs = base_ds.indices if hasattr(base_ds, 'indices') else range(len(raw.files))
            conditions = [_deliver_condition(raw.files[i]) for i in idxs]

        rep = _synthesize_split(subsets, modal_names, full_index, cache_dir, split,
                                num_classes, class_names, ignore_label, conditions,
                                granularities=granularities, n_null=n_null,
                                null_seed=args.null_seed)
        _print_report(rep)
        _write_reports(rep, out_dir)


if __name__ == '__main__':
    main()
