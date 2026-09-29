#!/usr/bin/env python3
"""[P55] 조건별 게이트 분석 — 연구 질문의 핵심 산출물.

게이트-학습 ckpt 를 로드해 DELIVER val(clean 입력)을 돌리고, 이미지마다 모달별 평균 r_m 을
기록한다. 경로에서 조건(cloud/fog/night/rain/sun)·케이스(motionblur/overexposure/…)를
파싱(tools/baseline_failure/common.parse_condition_case)하되, 라벨은 **분석에만** 쓴다.

출력:
  - 조건×케이스 그룹별 모달별 r 의 mean±std 표 + RGB-vs-others 비율.
  - r 붕괴 검사: 이미지 간 r std(≈0 이면 정적 게이트 = 상수 붕괴).
  - night/rain/fog vs sun/cloud 에서 r_img 가 낮아지는지: r_img 의 night-vs-clean AUROC
    (+ scipy 있으면 Mann-Whitney U p-value).
  json + md 를 쓴다.

  python tools/p55_gate_by_condition.py --cfg <P55 eval cfg> --ckpt <gate full ckpt> \
    --split val --out <analysis_logs>/p55_by_condition_YYYYMMDD [--subset_every 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.baseline_failure.common import parse_condition_case, KNOWN_CONDITIONS  # noqa: E402

# clean 정의: 센서 고장 없음 + 맑은 조건(sun/cloud). night/rain/fog 는 "악조건".
CLEAN_CONDS = ('sun', 'cloud')
DARK_CONDS = ('night', 'rain', 'fog')


# ===========================================================================
# 순수 계산 — 스모크가 합성 레코드로 직접 부른다(모델·데이터 불필요)
# ===========================================================================
def _stat(a):
    a = np.asarray(a, dtype=np.float64)
    return (float(a.mean()), float(a.std())) if a.size else (float('nan'), float('nan'))


def rank_auroc(scores, labels):
    """점수가 높을수록 양성(label=1)일 확률. Mann-Whitney U 통계 = AUROC.

    양성/음성 표본이 각 1개 이상 없으면 nan.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # 동점 평균 순위
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {i: (csum[i] - counts[i] + 1 + csum[i]) / 2.0 for i in range(len(counts))}
    ranks = np.array([avg[i] for i in inv])
    n1 = pos.size
    r1 = ranks[labels == 1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return float(u1 / (n1 * neg.size))


def compute_condition_report(r, conds, cases, modal_names):
    """r (N,M) per-image 평균 게이트, conds/cases 길이 N → 리포트 dict.

    r 는 게이트 신뢰도(1=완전 신뢰). 표는 조건×케이스 그룹별 모달 mean±std +
    RGB-vs-others 비율(r_img / mean(r_others)). 붕괴 검사 = 전 이미지 r std.
    night 검사 = r_img 의 (night∪rain∪fog) vs (sun∪cloud, 고장 없음) AUROC.
    """
    r = np.asarray(r, dtype=np.float64)
    conds = np.asarray(conds)
    cases = np.asarray(cases)
    M = len(modal_names)
    assert r.shape[1] == M, f"r 열 {r.shape[1]} != 모달 {M}"
    img_idx = modal_names.index('img') if 'img' in modal_names else 0
    other_idx = [i for i in range(M) if i != img_idx]

    # 그룹 = (condition, case) 조합 중 실제로 나타난 것.
    groups = {}
    keys = sorted(set(zip(conds.tolist(), cases.tolist())))
    for (c, cs) in keys:
        sel = (conds == c) & (cases == cs)
        if not sel.any():
            continue
        rows = r[sel]
        per_modal = {modal_names[i]: {'mean': _stat(rows[:, i])[0],
                                      'std': _stat(rows[:, i])[1]} for i in range(M)}
        r_img = rows[:, img_idx].mean()
        r_oth = rows[:, other_idx].mean() if other_idx else float('nan')
        groups[f"{c}|{cs}"] = {
            'condition': c, 'case': cs, 'n': int(sel.sum()),
            'per_modal': per_modal,
            'rgb_vs_others_ratio': float(r_img / r_oth) if r_oth else float('nan')}

    # 붕괴 검사: 전 이미지에서 모달별 r 의 across-image std.
    collapse = {modal_names[i]: float(r[:, i].std()) for i in range(M)}
    collapse_overall = float(np.mean([collapse[m] for m in modal_names]))

    # night/rain/fog vs clean(sun/cloud & case==none) 에서 r_img AUROC(낮을수록 악조건).
    is_clean = np.array([(c in CLEAN_CONDS) and (cs == 'none')
                         for c, cs in zip(conds, cases)])
    is_dark = np.array([c in DARK_CONDS for c in conds])
    # r_img 가 악조건에서 낮아지길 기대 → "r_img 낮음"을 양성으로: score = −r_img.
    sel2 = is_clean | is_dark
    labels = is_dark[sel2].astype(np.int64)
    score = -r[sel2, img_idx]
    auroc_dark = rank_auroc(score, labels)
    mann = None
    try:
        from scipy.stats import mannwhitneyu
        if is_dark.sum() > 0 and is_clean.sum() > 0:
            u, p = mannwhitneyu(r[is_dark, img_idx], r[is_clean, img_idx],
                                alternative='less')   # 악조건 r_img < clean r_img?
            mann = {'U': float(u), 'p_value': float(p)}
    except Exception:
        mann = None

    return {
        'modal_names': list(modal_names),
        'n_images': int(r.shape[0]),
        'groups': groups,
        'collapse_std_per_modal': collapse,
        'collapse_std_overall': collapse_overall,
        'r_img_dark_vs_clean_auroc': auroc_dark,
        'r_img_dark_vs_clean_mannwhitney': mann,
        'n_clean': int(is_clean.sum()), 'n_dark': int(is_dark.sum()),
        'r_mean_per_modal': {modal_names[i]: float(r[:, i].mean()) for i in range(M)},
    }


def build_md(report):
    mn = report['modal_names']
    lines = ["# P55 게이트 조건별 분석 (r = 게이트 신뢰도, 1=완전 신뢰)\n",
             f"- 이미지 수: {report['n_images']}  · clean={report['n_clean']} dark={report['n_dark']}",
             f"- **붕괴 검사** across-image r std (overall): {report['collapse_std_overall']:.4f} "
             "(≈0 이면 상수 붕괴 = 정적 게이트)",
             f"- r_img dark(night/rain/fog)-vs-clean AUROC: {report['r_img_dark_vs_clean_auroc']:.4f} "
             "(>0.5 = 악조건에서 r_img 낮음 = 상황 적응)"]
    if report['r_img_dark_vs_clean_mannwhitney']:
        mw = report['r_img_dark_vs_clean_mannwhitney']
        lines.append(f"- Mann-Whitney U(악조건 r_img < clean): U={mw['U']:.1f} p={mw['p_value']:.3e}")
    lines.append("\n| condition | case | n | "
                 + " | ".join(f"r_{m}" for m in mn) + " | rgb/others |")
    lines.append("|" + "---|" * (len(mn) + 4))
    for gk, g in report['groups'].items():
        cells = [f"{g['per_modal'][m]['mean']:.3f}±{g['per_modal'][m]['std']:.3f}" for m in mn]
        lines.append(f"| {g['condition']} | {g['case']} | {g['n']} | "
                     + " | ".join(cells) + f" | {g['rgb_vs_others_ratio']:.3f} |")
    return "\n".join(lines)


# ===========================================================================
# 실제 수집 — DELIVER val(clean) forward → per-image 평균 r
# ===========================================================================
def collect_gate_r(model, loader, files, device, limit=0):
    import torch
    assert len(files) == len(loader.dataset), \
        f"파일 {len(files)} != 데이터셋 {len(loader.dataset)} (순서 정합 실패)"
    r_all, conds, cases, idx = [], [], [], 0
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if limit and bi >= limit:
                break
            imgs = [x.to(device) for x in batch[0]]
            B = imgs[0].shape[0]
            model(imgs, multimask_output=True)
            rt = model._last_p55_r_token            # (B,M,h,w) cpu
            if rt is None:
                raise RuntimeError("[P55] _last_p55_r_token 이 없다 — MODEL.P55.ENABLE=true "
                                   "config·게이트 ckpt 인지 확인.")
            r_all.append(rt.float().mean(dim=(-1, -2)).numpy())   # (B,M)
            for j in range(B):
                c, cs = parse_condition_case(files[idx + j])
                conds.append(c)
                cases.append(cs)
            idx += B
    return np.concatenate(r_all, axis=0), conds, cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True, help='게이트-학습 전체 ckpt(p55_gate_full.pth).')
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--subset_every', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    import torch
    import yaml
    from torch.utils.data import DataLoader, Subset
    import val as valmod

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device if args.device else
                          cfg.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
    valmod.setup_cudnn()
    ds_cfg, eval_cfg = cfg['DATASET'], cfg['EVAL']
    test_cfg = cfg.get('TEST', {})
    image_size = eval_cfg['IMAGE_SIZE'] if args.split == 'val' \
        else test_cfg.get('IMAGE_SIZE', eval_cfg['IMAGE_SIZE'])
    tfm = valmod.get_val_augmentation(image_size, dataset_cfg=ds_cfg)
    dataset, _ = valmod.create_dataset(ds_cfg, args.split, tfm, args.split,
                                       macvi=False, eval_day=False)
    if not hasattr(dataset, 'files'):
        raise RuntimeError("[P55] 데이터셋이 .files 를 노출하지 않아 조건 파싱 불가.")
    files = list(dataset.files)
    if args.subset_every > 1:
        keep = list(range(0, len(dataset), args.subset_every))
        files = [files[i] for i in keep]
        dataset = Subset(dataset, keep)
        print(f"[P55] 부분집합 {len(keep)}장(every {args.subset_every}) — 스크린용")
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=False,
                        collate_fn=valmod._collate_fn)

    model = valmod.load_model(cfg, Path(args.ckpt), device)
    if getattr(model, 'p55_gate', None) is None:
        raise SystemExit("[P55] 모델에 P55 게이트가 없다 (MODEL.P55.ENABLE=true config 필요).")
    modal_names = list(ds_cfg['MODALS'])

    r, conds, cases = collect_gate_r(model, loader, files, device, limit=args.limit)
    report = compute_condition_report(r, conds, cases, modal_names)
    report['cfg'] = str(args.cfg)
    report['ckpt'] = str(args.ckpt)
    report['split'] = args.split

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'p55_by_condition.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    md = build_md(report)
    (out_dir / 'p55_by_condition.md').write_text(md, encoding='utf-8')
    print(md)
    print(f"\n[P55] → {out_dir / 'p55_by_condition.json'}")


if __name__ == '__main__':
    main()
