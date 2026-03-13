"""
P24+ 실험 결과 분석 스크립트 — detailed_log.json 기반
UAMM/AMF 상수 수렴 여부, per-class IoU, MoE routing, quality gating 분석

사용법:
  # 단일 split 분석
  python MISC/analyze_detailed_log.py --val_log <val_detailed_log.json>
  python MISC/analyze_detailed_log.py --test_log <test_detailed_log.json>

  # Val + Test 동시 분석
  python MISC/analyze_detailed_log.py \
      --val_log outputs/MMSamP24/.../val_pred_P24/detailed_log.json \
      --test_log outputs/MMSamP24/.../test_pred_P24/detailed_log.json

  # 특정 실험 디렉토리 자동 탐색
  python MISC/analyze_detailed_log.py --exp_dir outputs/MMSamP24/.../epoch36_93.89_top1

  # MACVi summary.json 포함 분석
  python MISC/analyze_detailed_log.py --exp_dir outputs/MMSamP24/.../epoch36_93.89_top1 \
      --summary outputs/MMSamP24/.../*_results/summary.json
"""

import argparse
import json
import os
import glob
import numpy as np
from collections import defaultdict


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def analyze_uamm_amf(data, split_name=""):
    """UAMM/AMF 상수 수렴 여부 분석"""
    images = data['images']
    modals = data['meta']['modals']

    uamm_vals = {m: [] for m in modals}
    amf_vals = {m: [] for m in modals}

    for img_name, img_data in images.items():
        if 'uamm' in img_data:
            for m in modals:
                uamm_vals[m].append(img_data['uamm'].get(m, 0))
        if 'amf' in img_data:
            for m in modals:
                amf_vals[m].append(img_data['amf'].get(m, 0))

    print(f"\n{'='*60}")
    print(f"  UAMM/AMF 분석 [{split_name}] (N={len(images)})")
    print(f"{'='*60}")

    print(f"\n  {'Modal':<10} {'Mean':>8} {'Std':>10} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*44}")
    print("  [UAMM]")
    for m in modals:
        vals = uamm_vals[m]
        if vals:
            arr = np.array(vals)
            print(f"  {m:<10} {arr.mean():>8.4f} {arr.std():>10.6f} {arr.min():>8.4f} {arr.max():>8.4f}")

    print("  [AMF]")
    for m in modals:
        vals = amf_vals[m]
        if vals:
            arr = np.array(vals)
            print(f"  {m:<10} {arr.mean():>8.4f} {arr.std():>10.6f} {arr.min():>8.4f} {arr.max():>8.4f}")

    # 상수 수렴 판단
    all_stds = []
    for m in modals:
        if uamm_vals[m]:
            all_stds.append(np.std(uamm_vals[m]))
        if amf_vals[m]:
            all_stds.append(np.std(amf_vals[m]))

    if all_stds and max(all_stds) < 0.01:
        print(f"\n  ⚠ 상수 수렴: 모든 UAMM/AMF의 std < 0.01 → gating이 작동하지 않음")
    elif all_stds and max(all_stds) < 0.05:
        print(f"\n  ⚠ 준상수: std < 0.05 → 매우 제한적 변화")
    else:
        print(f"\n  ✓ 이미지별 변화 있음 (max std={max(all_stds):.4f})")


def analyze_quality_gating(data, split_name=""):
    """Quality gating 관련 데이터 분석 (있는 경우)"""
    images = data['images']
    first_img = list(images.values())[0]

    # quality_gating 키가 있는지 확인
    quality_keys = [k for k in first_img.keys() if 'quality' in k.lower() or 'gating' in k.lower()]
    if not quality_keys:
        print(f"\n  [Quality Gating 데이터 없음 - {split_name}]")
        return

    print(f"\n{'='*60}")
    print(f"  Quality Gating 분석 [{split_name}]")
    print(f"{'='*60}")
    for key in quality_keys:
        print(f"\n  Key: {key}")
        vals = []
        for img_data in images.values():
            if key in img_data:
                vals.append(img_data[key])
        if vals and isinstance(vals[0], dict):
            # dict형태면 모달리티별 분석
            modals = list(vals[0].keys())
            for m in modals:
                m_vals = [v[m] for v in vals if m in v]
                arr = np.array(m_vals)
                print(f"    {m}: mean={arr.mean():.4f}, std={arr.std():.6f}, min={arr.min():.4f}, max={arr.max():.4f}")


def analyze_per_class_iou(data, split_name=""):
    """Val per-class IoU 분석"""
    images = data['images']
    classes = ['Static', 'Dynamic', 'Water', 'Sky']

    class_ious = {c: [] for c in classes}
    for img_data in images.values():
        if 'per_class_iou' in img_data:
            for c in classes:
                if c in img_data['per_class_iou']:
                    v = img_data['per_class_iou'][c]
                    if v is not None and v >= 0:  # -1이면 해당 클래스 없음
                        class_ious[c].append(v)

    if not any(class_ious.values()):
        return

    print(f"\n{'='*60}")
    print(f"  Per-class IoU [{split_name}] (from detailed_log)")
    print(f"{'='*60}")
    print(f"\n  {'Class':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>5}")
    print(f"  {'-'*47}")

    total_mean = []
    for c in classes:
        vals = class_ious[c]
        if vals:
            arr = np.array(vals)
            mean = arr.mean()
            total_mean.append(mean)
            print(f"  {c:<10} {mean:>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f} {len(vals):>5}")

    if total_mean:
        print(f"  {'mIoU':<10} {np.mean(total_mean):>8.4f}")


def analyze_pred_confidence(data, split_name=""):
    """Prediction confidence/entropy 분석"""
    images = data['images']
    mean_entropies = []
    high_unc_ratios = []

    for img_data in images.values():
        if 'pred_confidence' in img_data:
            pc = img_data['pred_confidence']
            mean_entropies.append(pc.get('mean_entropy', 0))
            high_unc_ratios.append(pc.get('high_uncertainty_ratio', 0))

    if not mean_entropies:
        return

    print(f"\n{'='*60}")
    print(f"  Prediction Confidence [{split_name}]")
    print(f"{'='*60}")
    me = np.array(mean_entropies)
    hur = np.array(high_unc_ratios)
    print(f"  Mean entropy:        avg={me.mean():.4f}, std={me.std():.4f}")
    print(f"  High uncertainty %:  avg={hur.mean():.4f}, std={hur.std():.4f}")


def analyze_moe_routing_summary(data, split_name=""):
    """MoE routing 요약 — Block 0, 중간, 마지막의 entropy_ratio 평균"""
    images = data['images']
    first_img = list(images.values())[0]

    if 'moe_routing' not in first_img:
        return

    # 어떤 블록들이 있는지 확인
    blocks = sorted(first_img['moe_routing'].keys())
    modals = data['meta']['modals']

    print(f"\n{'='*60}")
    print(f"  MoE Routing 요약 [{split_name}] — entropy_ratio (낮을수록 분화)")
    print(f"{'='*60}")

    # 대표 블록만 표시 (처음, 중간, 마지막)
    if len(blocks) > 6:
        show_blocks = [blocks[0], blocks[len(blocks)//2], blocks[-1]]
    else:
        show_blocks = blocks

    for block in show_blocks:
        print(f"\n  [{block}]")
        print(f"  {'Modal':<10} {'ent_ratio':>10} {'max_wt':>8} {'top1_frac':>10}")
        print(f"  {'-'*38}")
        for m in modals:
            ent_ratios = []
            max_wts = []
            for img_data in images.values():
                if block in img_data.get('moe_routing', {}):
                    bd = img_data['moe_routing'][block]
                    if m in bd:
                        ent_ratios.append(bd[m]['entropy_ratio'])
                        max_wts.append(bd[m]['per_token_max'])
            if ent_ratios:
                er = np.mean(ent_ratios)
                mw = np.mean(max_wts)
                # argmax fraction for top expert
                top1 = []
                for img_data in images.values():
                    if block in img_data.get('moe_routing', {}):
                        bd = img_data['moe_routing'][block]
                        if m in bd and 'argmax_fraction' in bd[m]:
                            top1.append(max(bd[m]['argmax_fraction'].values()))
                t1 = np.mean(top1) if top1 else 0
                print(f"  {m:<10} {er:>10.4f} {mw:>8.4f} {t1:>10.4f}")


def analyze_summary_json(summary_path):
    """MACVi summary.json 분석"""
    data = load_json(summary_path)
    summary = data.get('summary', data)

    print(f"\n{'='*60}")
    print(f"  MACVi Challenge Results")
    print(f"{'='*60}")
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def find_logs_from_exp_dir(exp_dir):
    """exp_dir에서 val/test detailed_log.json 자동 탐색"""
    val_log = None
    test_log = None
    summary = None

    # 패턴: {exp_dir}_val_pred_P*/detailed_log.json
    base = os.path.basename(exp_dir)
    parent = os.path.dirname(exp_dir)

    for d in os.listdir(parent):
        full = os.path.join(parent, d)
        if not os.path.isdir(full):
            continue
        if base in d and 'val_pred' in d:
            log = os.path.join(full, 'detailed_log.json')
            if os.path.exists(log):
                val_log = log
        elif base in d and 'test_pred' in d:
            log = os.path.join(full, 'detailed_log.json')
            if os.path.exists(log):
                test_log = log

    # summary.json 탐색
    for d in os.listdir(parent):
        full = os.path.join(parent, d)
        if os.path.isdir(full) and '_results' in d and base.split('_')[0] in d:
            s = os.path.join(full, 'summary.json')
            if os.path.exists(s):
                summary = s

    return val_log, test_log, summary


def main():
    parser = argparse.ArgumentParser(description='P24+ detailed_log.json 분석')
    parser.add_argument('--val_log', type=str, help='Val detailed_log.json 경로')
    parser.add_argument('--test_log', type=str, help='Test detailed_log.json 경로')
    parser.add_argument('--exp_dir', type=str, help='실험 디렉토리 (자동 탐색)')
    parser.add_argument('--summary', type=str, help='MACVi summary.json 경로')
    parser.add_argument('--no_moe', action='store_true', help='MoE routing 분석 스킵')
    args = parser.parse_args()

    if args.exp_dir:
        val_log, test_log, summary = find_logs_from_exp_dir(args.exp_dir)
        if not args.val_log and val_log:
            args.val_log = val_log
        if not args.test_log and test_log:
            args.test_log = test_log
        if not args.summary and summary:
            args.summary = summary

    if not args.val_log and not args.test_log:
        print("Error: --val_log, --test_log, 또는 --exp_dir 중 하나를 지정해주세요")
        return

    # Val 분석
    if args.val_log:
        print(f"\n{'#'*60}")
        print(f"  VAL: {args.val_log}")
        print(f"{'#'*60}")
        val_data = load_json(args.val_log)
        analyze_uamm_amf(val_data, "VAL")
        analyze_quality_gating(val_data, "VAL")
        analyze_per_class_iou(val_data, "VAL")
        analyze_pred_confidence(val_data, "VAL")
        if not args.no_moe:
            analyze_moe_routing_summary(val_data, "VAL")

    # Test 분석
    if args.test_log:
        print(f"\n{'#'*60}")
        print(f"  TEST: {args.test_log}")
        print(f"{'#'*60}")
        test_data = load_json(args.test_log)
        analyze_uamm_amf(test_data, "TEST")
        analyze_quality_gating(test_data, "TEST")
        analyze_pred_confidence(test_data, "TEST")
        if not args.no_moe:
            analyze_moe_routing_summary(test_data, "TEST")

    # MACVi summary
    if args.summary:
        analyze_summary_json(args.summary)

    # Val vs Test UAMM/AMF 비교
    if args.val_log and args.test_log:
        val_data = load_json(args.val_log)
        test_data = load_json(args.test_log)
        modals = val_data['meta']['modals']

        print(f"\n{'='*60}")
        print(f"  VAL vs TEST UAMM/AMF 비교")
        print(f"{'='*60}")
        print(f"\n  {'Modal':<10} {'Val UAMM':>10} {'Test UAMM':>10} {'Val AMF':>10} {'Test AMF':>10}")
        print(f"  {'-'*50}")
        for m in modals:
            val_uamm = np.mean([v['uamm'][m] for v in val_data['images'].values() if 'uamm' in v])
            test_uamm = np.mean([v['uamm'][m] for v in test_data['images'].values() if 'uamm' in v])
            val_amf = np.mean([v['amf'][m] for v in val_data['images'].values() if 'amf' in v])
            test_amf = np.mean([v['amf'][m] for v in test_data['images'].values() if 'amf' in v])
            print(f"  {m:<10} {val_uamm:>10.4f} {test_uamm:>10.4f} {val_amf:>10.4f} {test_amf:>10.4f}")


if __name__ == '__main__':
    main()
