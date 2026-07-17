#!/usr/bin/env python3
"""
tools/compare_models.py — [분석항목 4] 모델별 클래스×도메인 성능 비교 → "어디를 극복해야"
자동 digest. GPU/데이터 불필요 — eval_per_domain.py 산출 로그 디렉토리들만 파싱.

Input: N개의 `label=dir` 쌍. 각 dir = tools/eval_per_domain.py의 --out-dir
(로그 파일명 `<ckpt-label>__<condition>.log`, per-class IoU 표 포함 — analyze_per_domain과
동일 파서 재사용).

Output:
  - 모델 × 도메인 mIoU 표
  - 모델 × 클래스 (도메인 평균) IoU 표 + best-model 마킹
  - 자동 분류 digest:
      · STRUCTURAL   : 모든 모델에서 max IoU < dead_thresh — 설계 반복으로 못 푼 클래스
                       (backbone/데이터 레벨 개입 필요)
      · DESIGN-GAP   : 모델 간 spread > gap_thresh — 설계에 민감, 이긴 설계를 채택/분석
      · DOMAIN-GAP   : (모델별) 도메인 간 spread > gap_thresh — 조건 취약, 타깃 증강 후보
      · SOLVED       : 전 모델 IoU > solved_thresh — 더 볼 필요 없음

Usage:
  python tools/compare_models.py \
    --run P28=/mnt/HDD2/src/logs/P28_eval/perdomain \
    --run P29=/mnt/HDD2/src/logs/P29_eval_20260630/perdomain \
    --run P31=<dir> --out compare_P28_P29_P31.md
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_per_domain import parse_log  # 동일 파서 재사용 (val.py per-class IoU 표)


def load_run(d):
    """`DIR[:GLOB]` of eval_per_domain logs -> {condition: {class: iou}}, {condition: mean}.
    GLOB(기본 *.log)로 같은 dir의 여러 ckpt 로그 중 하나를 고른다 (예: dir:ep146__*.log)."""
    pattern = '*.log'
    if ':' in d and not Path(d).exists():
        d, pattern = d.rsplit(':', 1)
    per_cond, means = {}, {}
    for logf in sorted(Path(d).glob(pattern)):
        cond = logf.stem.split('__')[-1]
        cls_iou, mean = parse_log(logf)
        if cls_iou:
            per_cond[cond] = cls_iou
            means[cond] = mean
    return per_cond, means


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, metavar='LABEL=DIR',
                    help='repeatable: model label = eval_per_domain out-dir')
    ap.add_argument('--dead-thresh', type=float, default=10.0)
    ap.add_argument('--gap-thresh', type=float, default=15.0)
    ap.add_argument('--solved-thresh', type=float, default=75.0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        label, d = spec.split('=', 1)
        per_cond, means = load_run(d)
        if not per_cond:
            print(f"[compare] WARNING: no parsable logs in {d} — skipping {label}")
            continue
        runs[label] = (per_cond, means)
    if len(runs) < 1:
        sys.exit('[compare] no runs loaded')

    labels = list(runs)
    conds = sorted({c for pc, _ in runs.values() for c in pc})
    classes = sorted({cl for pc, _ in runs.values() for c in pc.values() for cl in c})

    # model × class matrix (mean over available conditions) + per-model domain spread
    mat = np.full((len(labels), len(classes)), np.nan)
    dom_spread = np.full((len(labels), len(classes)), np.nan)
    for li, lb in enumerate(labels):
        pc, _ = runs[lb]
        for ci, cl in enumerate(classes):
            vals = [pc[c][cl] for c in pc if cl in pc[c]]
            if vals:
                mat[li, ci] = float(np.mean(vals))
                dom_spread[li, ci] = float(np.max(vals) - np.min(vals))

    lines = [f"# Model comparison — {', '.join(labels)}",
             f"- thresholds: dead<{args.dead_thresh}, design/domain gap>{args.gap_thresh}, "
             f"solved>{args.solved_thresh}", ""]

    lines += ["## 모델 × 도메인 mIoU", "| model | " + " | ".join(conds) + " | mean |",
              "|---|" + "---|" * (len(conds) + 1)]
    for lb in labels:
        _, means = runs[lb]
        row = [f"{means.get(c, float('nan')):.2f}" if means.get(c) is not None else '—' for c in conds]
        mv = [means[c] for c in conds if means.get(c) is not None]
        lines.append(f"| {lb} | " + " | ".join(row) + f" | **{np.mean(mv):.2f}** |")
    lines.append("")

    lines += ["## 모델 × 클래스 (도메인 평균 IoU, ★=best)",
              "| class | " + " | ".join(labels) + " | spread |", "|---|" + "---|" * (len(labels) + 1)]
    order = np.argsort(np.nanmax(mat, axis=0))     # 어려운 클래스부터
    for ci in order:
        vals = mat[:, ci]
        if np.all(np.isnan(vals)):
            continue
        best = np.nanargmax(vals)
        cells = [f"{'★' if li == best else ''}{vals[li]:.1f}" if not np.isnan(vals[li]) else '—'
                 for li in range(len(labels))]
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        lines.append(f"| {classes[ci]} | " + " | ".join(cells) + f" | {spread:.1f} |")
    lines.append("")

    structural, design_gap, solved = [], [], []
    domain_gap = {}
    for ci, cl in enumerate(classes):
        vals = mat[:, ci]
        if np.all(np.isnan(vals)):
            continue
        if np.nanmax(vals) < args.dead_thresh:
            structural.append(cl)
        elif np.nanmin(vals) > args.solved_thresh:
            solved.append(cl)
        elif len(labels) > 1 and (np.nanmax(vals) - np.nanmin(vals)) > args.gap_thresh:
            best = labels[int(np.nanargmax(vals))]
            design_gap.append(f"{cl} (best={best}, {np.nanmax(vals):.1f} vs {np.nanmin(vals):.1f})")
        ds = np.nanmax(dom_spread[:, ci])
        if not np.isnan(ds) and ds > args.gap_thresh and np.nanmax(vals) >= args.dead_thresh:
            domain_gap[cl] = round(float(ds), 1)

    lines += ["## 🎯 극복 대상 digest (자동 분류)",
              f"- **STRUCTURAL (전 모델 사망, IoU<{args.dead_thresh})**: "
              + (', '.join(structural) or '없음')
              + " → fusion/헤드 반복으로 불가, backbone unfreeze/사전학습/데이터 레벨 개입",
              f"- **DESIGN-GAP (모델 간 {args.gap_thresh}pt+ 격차)**: "
              + ('; '.join(design_gap) or '없음') + " → 이긴 설계의 기전 분석 후 채택",
              f"- **DOMAIN-GAP (도메인 간 {args.gap_thresh}pt+ 편차)**: "
              + (', '.join(f"{k}({v})" for k, v in sorted(domain_gap.items(), key=lambda x: -x[1]))
                 or '없음') + " → 조건 타깃 증강 후보",
              f"- **SOLVED (전 모델 >{args.solved_thresh})**: " + (', '.join(solved) or '없음'), ""]

    Path(args.out).write_text('\n'.join(lines))
    print(f"[compare] wrote {args.out}  (models={labels}, classes={len(classes)}, conds={conds})")
    print(f"[compare] STRUCTURAL={structural}")
    print(f"[compare] DESIGN-GAP={design_gap}")


if __name__ == '__main__':
    main()
