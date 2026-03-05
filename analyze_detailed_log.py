"""
Analyze detailed_log.json from val_multiaqua_detailed.py
========================================================

val_multiaqua_detailed.py가 생성한 JSON 로그를 읽어서
UAMM/AMF fusion, MoE routing, prediction confidence 등을 요약.

사용:
  python analyze_detailed_log.py <path_to_detailed_log.json>
  python analyze_detailed_log.py <save_dir>  # save_dir/detailed_log.json 자동 탐색
"""
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_json(path):
    path = Path(path)
    if path.is_dir():
        path = path / "detailed_log.json"
    if not path.exists():
        print(f"[ERROR] JSON not found: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f), path


def print_header(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"{'=' * 64}")


def analyze_fusion(images, modals):
    """UAMM / AMF fusion weight 분석."""
    uamm_all = defaultdict(list)
    amf_all = defaultdict(list)

    for stem, d in images.items():
        if 'uamm' in d:
            for m in modals:
                uamm_all[m].append(d['uamm'].get(m, 0.0))
        if 'amf' in d:
            for m in modals:
                amf_all[m].append(d['amf'].get(m, 0.0))

    if not uamm_all and not amf_all:
        return

    print_header("Fusion Weights (UAMM / AMF)")

    for tag, data in [("UAMM", uamm_all), ("AMF", amf_all)]:
        if not data:
            continue
        print(f"\n  [{tag}]  (across {len(list(data.values())[0])} images)")
        print(f"  {'Modality':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for m in modals:
            vals = np.array(data[m])
            print(f"  {m:<12} {vals.mean():8.4f} {vals.std():8.4f} "
                  f"{vals.min():8.4f} {vals.max():8.4f}")

    # Dominant modality 분석
    if amf_all:
        dominant_counts = defaultdict(int)
        for i in range(len(list(amf_all.values())[0])):
            dominant = max(modals, key=lambda m: amf_all[m][i])
            dominant_counts[dominant] += 1
        total = sum(dominant_counts.values())
        print(f"\n  [AMF Dominant Modality]")
        for m in modals:
            cnt = dominant_counts.get(m, 0)
            print(f"  {m:<12} {cnt:>4} images ({100*cnt/total:.1f}%)")


def analyze_confidence(images):
    """Prediction confidence (softmax entropy) 분석."""
    entropies = []
    max_entropies = []
    high_unc_ratios = []

    for stem, d in images.items():
        if 'pred_confidence' not in d:
            continue
        pc = d['pred_confidence']
        entropies.append(pc['mean_entropy'])
        max_entropies.append(pc['max_entropy'])
        high_unc_ratios.append(pc['high_uncertainty_ratio'])

    if not entropies:
        return

    print_header("Prediction Confidence")

    entropies = np.array(entropies)
    high_unc = np.array(high_unc_ratios)

    print(f"\n  Mean Entropy    : {entropies.mean():.4f} +/- {entropies.std():.4f}  "
          f"(range: {entropies.min():.4f} ~ {entropies.max():.4f})")
    print(f"  High Unc. Ratio : {high_unc.mean():.4f} +/- {high_unc.std():.4f}  "
          f"(range: {high_unc.min():.4f} ~ {high_unc.max():.4f})")

    # Uncertainty 기반 이미지 분류
    low_conf = sum(1 for e in entropies if e > 0.3)
    mid_conf = sum(1 for e in entropies if 0.1 <= e <= 0.3)
    high_conf = sum(1 for e in entropies if e < 0.1)
    n = len(entropies)
    print(f"\n  Confidence Distribution:")
    print(f"    High (entropy < 0.1) : {high_conf:>4} ({100*high_conf/n:.1f}%)")
    print(f"    Mid  (0.1 ~ 0.3)    : {mid_conf:>4} ({100*mid_conf/n:.1f}%)")
    print(f"    Low  (entropy > 0.3) : {low_conf:>4} ({100*low_conf/n:.1f}%)")

    # Worst-10 by mean entropy
    sorted_imgs = sorted(images.items(),
                         key=lambda x: x[1].get('pred_confidence', {}).get('mean_entropy', 0),
                         reverse=True)
    print(f"\n  [Top-10 Highest Uncertainty Images]")
    print(f"  {'#':>3} {'Stem':<30} {'MeanEnt':>8} {'HighUncR':>9}")
    for i, (stem, d) in enumerate(sorted_imgs[:10]):
        pc = d.get('pred_confidence', {})
        print(f"  {i+1:>3} {stem:<30} {pc.get('mean_entropy', 0):8.4f} "
              f"{pc.get('high_uncertainty_ratio', 0):9.4f}")


def analyze_per_class_iou(images, classes=None):
    """Per-class IoU 분석 (val only)."""
    class_ious = defaultdict(list)
    has_iou = False

    for stem, d in images.items():
        if 'per_class_iou' not in d:
            continue
        has_iou = True
        for cls_name, iou in d['per_class_iou'].items():
            if iou is not None:
                class_ious[cls_name].append(iou)

    if not has_iou:
        return

    print_header("Per-Class IoU (Val Only)")

    # Class-level summary
    all_class_means = []
    print(f"\n  {'Class':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>5}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    for cls_name in sorted(class_ious.keys()):
        vals = np.array(class_ious[cls_name])
        all_class_means.append(vals.mean())
        print(f"  {cls_name:<12} {vals.mean():8.4f} {vals.std():8.4f} "
              f"{vals.min():8.4f} {vals.max():8.4f} {len(vals):5d}")
    if all_class_means:
        mean_iou = np.mean(all_class_means)
        print(f"  {'mIoU':<12} {mean_iou:8.4f}")

    # Worst images by mIoU
    img_mious = []
    for stem, d in images.items():
        if 'per_class_iou' not in d:
            continue
        ious = [v for v in d['per_class_iou'].values() if v is not None]
        if ious:
            img_mious.append((stem, np.mean(ious), d['per_class_iou']))

    img_mious.sort(key=lambda x: x[1])
    print(f"\n  [Top-10 Worst Images by mIoU]")
    print(f"  {'#':>3} {'Stem':<30} {'mIoU':>7}  Class IoUs")
    for i, (stem, miou, per_cls) in enumerate(img_mious[:10]):
        cls_str = "  ".join(f"{k}={v:.2f}" for k, v in per_cls.items() if v is not None)
        print(f"  {i+1:>3} {stem:<30} {miou:7.4f}  {cls_str}")


def analyze_moe_routing(images, modals):
    """MoE routing 패턴 분석."""
    q_entropy_ratios = []
    v_entropy_ratios = []
    q_top2_gaps = []

    for stem, d in images.items():
        summary = d.get('moe_summary', {})
        if 'Q' in summary and summary['Q'].get('avg_entropy_ratio') is not None:
            q_entropy_ratios.append(summary['Q']['avg_entropy_ratio'])
            if summary['Q'].get('avg_top2_gap') is not None:
                q_top2_gaps.append(summary['Q']['avg_top2_gap'])
        if 'V' in summary and summary['V'].get('avg_entropy_ratio') is not None:
            v_entropy_ratios.append(summary['V']['avg_entropy_ratio'])

    if not q_entropy_ratios:
        return

    print_header("MoE Routing Summary")

    q_er = np.array(q_entropy_ratios)
    print(f"\n  [Q Layers]  ({len(q_er)} images)")
    print(f"    Entropy Ratio : {q_er.mean():.4f} +/- {q_er.std():.4f}  "
          f"(range: {q_er.min():.4f} ~ {q_er.max():.4f})")
    if q_top2_gaps:
        q_gap = np.array(q_top2_gaps)
        print(f"    Top-2 Gap     : {q_gap.mean():.4f} +/- {q_gap.std():.4f}")

    if v_entropy_ratios:
        v_er = np.array(v_entropy_ratios)
        print(f"\n  [V Layers]  ({len(v_er)} images)")
        print(f"    Entropy Ratio : {v_er.mean():.4f} +/- {v_er.std():.4f}  "
              f"(range: {v_er.min():.4f} ~ {v_er.max():.4f})")

    # Expert collapse check (from first image's summary)
    first_img = list(images.values())[0]
    summary = first_img.get('moe_summary', {})
    if 'Q' in summary and 'expert_usage' in summary['Q']:
        eu = summary['Q']['expert_usage']
        collapsed_any = False
        for mname, usage in eu.items():
            if 'collapsed_experts' in usage:
                collapsed_any = True
                print(f"\n  [WARNING] Expert collapse in Q/{mname}: "
                      f"{usage['collapsed_experts']}")
        if not collapsed_any:
            print(f"\n  Expert collapse: None detected (Q)")

    # Per-modality expert preference (from routing data)
    # 첫 이미지의 대표 블록에서 각 모달리티의 expert 선호도 확인
    routing = first_img.get('moe_routing', {})
    mid_block_key = None
    for k in routing:
        if '_Q' in k:
            mid_block_key = k
    if mid_block_key and mid_block_key in routing:
        print(f"\n  [Expert Selection Example: {mid_block_key}]")
        block_data = routing[mid_block_key]
        for mname in modals:
            if mname in block_data:
                fracs = block_data[mname].get('argmax_fraction', {})
                frac_str = "  ".join(f"{k}={v:.1%}" for k, v in fracs.items())
                print(f"    {mname:<10}: {frac_str}")


def analyze_fusion_confidence_correlation(images, modals):
    """Fusion weight vs confidence 상관 분석."""
    amf_dominant = []
    entropies = []

    for stem, d in images.items():
        if 'amf' not in d or 'pred_confidence' not in d:
            continue
        dominant_w = max(d['amf'].get(m, 0) for m in modals)
        amf_dominant.append(dominant_w)
        entropies.append(d['pred_confidence']['mean_entropy'])

    if len(amf_dominant) < 5:
        return

    amf_arr = np.array(amf_dominant)
    ent_arr = np.array(entropies)
    corr = np.corrcoef(amf_arr, ent_arr)[0, 1]

    print_header("Fusion-Confidence Correlation")
    print(f"\n  AMF dominant weight vs Mean Entropy: r = {corr:.4f}")
    if abs(corr) > 0.3:
        direction = "higher" if corr > 0 else "lower"
        print(f"  -> Stronger modality dominance correlates with {direction} uncertainty")
    else:
        print(f"  -> No strong correlation")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_detailed_log.py <detailed_log.json or save_dir>")
        sys.exit(1)

    data, json_path = load_json(sys.argv[1])
    meta = data.get('meta', {})
    images = data.get('images', {})
    modals = meta.get('modals', ['img', 'lidar', 'thermal'])
    split = meta.get('split', 'unknown')

    print_header(f"Detailed Log Analysis: {json_path.name}")
    print(f"  Split       : {split}")
    print(f"  Images      : {meta.get('n_images', len(images))}")
    print(f"  Model       : {meta.get('lora_model', '?')}")
    print(f"  Modals      : {modals}")
    print(f"  MoE Blocks  : Q={meta.get('num_moe_blocks_q', '?')}, "
          f"V={meta.get('num_moe_blocks_v', '?')}")

    analyze_fusion(images, modals)
    analyze_confidence(images)
    if split == 'val':
        analyze_per_class_iou(images)
    analyze_moe_routing(images, modals)
    analyze_fusion_confidence_correlation(images, modals)

    print(f"\n{'=' * 64}")
    print(f"  Analysis complete. ({len(images)} images)")
    print(f"{'=' * 64}\n")


if __name__ == '__main__':
    main()
