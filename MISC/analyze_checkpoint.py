#!/usr/bin/env python3
"""
MACVi 체크포인트 분석 스크립트 (범용)

사용법:
  # 단일 체크포인트 분석
  python MISC/analyze_checkpoint.py outputs/MMSamP21/.../epoch94_94.17_top1_16792_results/

  # 여러 체크포인트 비교 (디렉토리 여러 개)
  python MISC/analyze_checkpoint.py \
    outputs/MMSamP21/.../epoch94_94.17_top1_16792_results/ \
    outputs/MMSamP21/.../epoch101_94.25_top1_16798_results/ \
    outputs/MMSamP21/.../epoch115_94.26_top1_16814_results/

  # 특정 모델 output 디렉토리 아래의 모든 *_results/ 자동 탐색
  python MISC/analyze_checkpoint.py --scan outputs/MMSamP21/levine_multiaqua_rgbtl_P21_hardaug8_physaug/MULTIAQUA_CMNeXt-B2_ilt/

  # P9 기준선 포함
  python MISC/analyze_checkpoint.py --baseline "P9 ep131,93.54,70.41,81.98,81.30,21.86,94.61,76.54" \
    outputs/MMSamP21/.../epoch94_*_results/

  # 학습 곡선 출력 (train.log 경로)
  python MISC/analyze_checkpoint.py --training-curve outputs/MMSamP21/.../train.log --from-epoch 50

기능:
  1. summary.json → val_mIoU, test_mIoU, M-score
  2. frames_test.csv → per-class IoU 평균 + Dynamic/Sky 분포 분석
  3. detailed_log.json → UAMM/AMF 통계 + MoE routing entropy
  4. train.log → epoch별 Day-Val / Night-Val 추이
"""

import argparse
import csv
import json
import os
import glob
import statistics
from pathlib import Path


def find_result_dirs(scan_dir):
    """*_results/ 디렉토리 자동 탐색"""
    pattern = os.path.join(scan_dir, "*_results")
    return sorted(glob.glob(pattern))


def find_detailed_log(result_dir):
    """result_dir과 매칭되는 detailed_log.json 경로 찾기"""
    # epoch94_94.17_top1_16792_results → epoch94_94.17_top1_test_pred_P*/detailed_log.json
    base = os.path.dirname(result_dir)
    result_name = os.path.basename(result_dir.rstrip("/"))

    # "epoch94_94.17_top1_16792_results" → "epoch94_94.17_top1"
    parts = result_name.rsplit("_", 2)  # split off "{submission_id}_results"
    if len(parts) >= 3:
        prefix = parts[0]  # e.g. "epoch94_94.17_top1"
    else:
        prefix = result_name.replace("_results", "")

    # Search for matching test pred directory
    pattern = os.path.join(base, f"{prefix}_test_pred_P*", "detailed_log.json")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def analyze_frames_test(csv_path):
    """Per-class IoU 분석"""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)
    static = [float(r["IoU_static_obstacle"]) for r in rows]
    dynamic = [float(r["IoU_dynamic_obstacle"]) for r in rows]
    water = [float(r["IoU_water"]) for r in rows]
    sky = [float(r["IoU_sky"]) for r in rows]

    result = {
        "n_images": n,
        "per_class": {
            "Static": {"mean": sum(static) / n, "std": statistics.stdev(static)},
            "Dynamic": {
                "mean": sum(dynamic) / n,
                "std": statistics.stdev(dynamic),
                "zero": sum(1 for d in dynamic if d == 0.0),
                "lt10": sum(1 for d in dynamic if d < 10.0),
                "gte50": sum(1 for d in dynamic if d >= 50.0),
            },
            "Water": {"mean": sum(water) / n, "std": statistics.stdev(water)},
            "Sky": {
                "mean": sum(sky) / n,
                "std": statistics.stdev(sky),
                "zero": sum(1 for s in sky if s == 0.0),
                "lt10": sum(1 for s in sky if s < 10.0),
            },
        },
    }
    return result


def analyze_detailed_log(log_path):
    """UAMM/AMF + MoE routing 분석"""
    with open(log_path) as f:
        data = json.load(f)

    uamm = {"img": [], "lidar": [], "thermal": []}
    amf = {"img": [], "lidar": [], "thermal": []}
    entropy_q, entropy_v = [], []

    for img_name, d in data["images"].items():
        for m in ["img", "lidar", "thermal"]:
            uamm[m].append(d["uamm"][m])
            amf[m].append(d["amf"][m])

        for key, routing in d.get("moe_routing", {}).items():
            for modal in ["img", "lidar", "thermal"]:
                if modal in routing:
                    er = routing[modal]["entropy_ratio"]
                    if "_Q" in key:
                        entropy_q.append(er)
                    elif "_V" in key:
                        entropy_v.append(er)

    result = {
        "uamm": {
            m: {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0}
            for m, v in uamm.items()
        },
        "amf": {
            m: {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0}
            for m, v in amf.items()
        },
    }

    if entropy_q:
        result["moe_entropy_q"] = statistics.mean(entropy_q)
    if entropy_v:
        result["moe_entropy_v"] = statistics.mean(entropy_v)

    return result


def analyze_training_curve(log_path, from_epoch=0):
    """train.log에서 epoch별 Day-Val/Night-Val 추출"""
    with open(log_path) as f:
        lines = f.readlines()

    epoch_data = {}
    for line in lines:
        if "[Day-Val]" in line:
            parts = line.split()
            ep, miou, loss = None, None, None
            for i, p in enumerate(parts):
                if "epoch:" in p:
                    ep = int(p.replace("epoch:", ""))
                if p == "mIoU:" and i + 1 < len(parts):
                    miou = float(parts[i + 1])
                if p == "Loss:" and i + 1 < len(parts):
                    loss = float(parts[i + 1])
            if ep is not None:
                epoch_data.setdefault(ep, {})["day"] = miou
                if loss is not None:
                    epoch_data[ep]["loss"] = loss

        elif "[Night-Val]" in line and "epoch:" in line and "Best:" not in line.split("epoch:")[1].split("mIoU:")[0]:
            parts = line.split()
            ep, miou = None, None
            for i, p in enumerate(parts):
                if "epoch:" in p:
                    ep = int(p.replace("epoch:", ""))
                if p == "mIoU:" and i + 1 < len(parts):
                    miou = float(parts[i + 1])
            if ep is not None:
                epoch_data.setdefault(ep, {})["night"] = miou

    return {ep: d for ep, d in sorted(epoch_data.items()) if ep >= from_epoch}


def print_comparison_table(results, baseline=None):
    """비교 테이블 출력"""
    print("\n" + "=" * 120)
    print(f"{'Checkpoint':<25} {'Val mIoU':>10} {'Test mIoU':>10} {'M-score':>10} | "
          f"{'Static':>8} {'Dynamic':>8} {'Water':>8} {'Sky':>8}")
    print("-" * 120)

    for name, r in results.items():
        s = r["summary"]
        c = r["frames"]["per_class"]
        print(f"{name:<25} {s['val_mIoU']:>10.2f} {s['test_mIoU']:>10.2f} {s['M']:>10.2f} | "
              f"{c['Static']['mean']:>8.2f} {c['Dynamic']['mean']:>8.2f} "
              f"{c['Water']['mean']:>8.2f} {c['Sky']['mean']:>8.2f}")

    if baseline:
        print("-" * 120)
        b = baseline.split(",")
        print(f"{b[0]:<25} {float(b[1]):>10.2f} {float(b[2]):>10.2f} {float(b[3]):>10.2f} | "
              f"{float(b[4]):>8.2f} {float(b[5]):>8.2f} {float(b[6]):>8.2f} {float(b[7]):>8.2f}")

    # Dynamic/Sky distribution
    print("\n--- Dynamic Class Distribution ---")
    print(f"{'Checkpoint':<25} {'mean':>8} {'zero':>6} {'<10%':>6} {'>=50%':>6}")
    for name, r in results.items():
        d = r["frames"]["per_class"]["Dynamic"]
        print(f"{name:<25} {d['mean']:>8.2f} {d['zero']:>6d} {d['lt10']:>6d} {d['gte50']:>6d}")

    print("\n--- Sky Class Distribution ---")
    print(f"{'Checkpoint':<25} {'mean':>8} {'zero':>6} {'<10%':>6}")
    for name, r in results.items():
        s = r["frames"]["per_class"]["Sky"]
        print(f"{name:<25} {s['mean']:>8.2f} {s['zero']:>6d} {s['lt10']:>6d}")

    # UAMM/AMF if available
    has_detailed = any("detailed" in r for r in results.values())
    if has_detailed:
        print("\n--- UAMM/AMF Weights ---")
        print(f"{'Checkpoint':<25} {'UAMM_img':>10} {'UAMM_lid':>10} {'UAMM_thm':>10} | "
              f"{'AMF_img':>8} {'AMF_lid':>8} {'AMF_thm':>8} | {'MoE_Q':>6} {'MoE_V':>6}")
        for name, r in results.items():
            if "detailed" in r:
                d = r["detailed"]
                u = d["uamm"]
                a = d["amf"]
                mq = d.get("moe_entropy_q", 0)
                mv = d.get("moe_entropy_v", 0)
                print(f"{name:<25} {u['img']['mean']:>10.4f} {u['lidar']['mean']:>10.4f} "
                      f"{u['thermal']['mean']:>10.4f} | {a['img']['mean']:>8.4f} "
                      f"{a['lidar']['mean']:>8.4f} {a['thermal']['mean']:>8.4f} | "
                      f"{mq:>6.4f} {mv:>6.4f}")

    print("=" * 120)


def print_training_curve(curve_data, markers=None):
    """학습 곡선 출력"""
    print("\n=== Training Curve ===")
    print(f"{'Ep':<5} {'Day-Val':>8} {'Night-Val':>10} {'Loss':>8}")
    print("-" * 40)

    for ep, d in curve_data.items():
        day = d.get("day", 0)
        night = d.get("night", 0)
        loss = d.get("loss", 0)
        marker = ""
        if markers and ep in markers:
            marker = f"  <-- {markers[ep]}"
        print(f"{ep:<5} {day:>8.2f} {night:>10.2f} {loss:>8.3f}{marker}")

    max_ep = max(curve_data.keys())
    best_day_ep = max(curve_data, key=lambda e: curve_data[e].get("day", 0))
    best_night_ep = max(curve_data, key=lambda e: curve_data[e].get("night", 0))
    print(f"\nLast epoch: {max_ep}")
    print(f"Best Day-Val:   ep{best_day_ep} = {curve_data[best_day_ep].get('day', 0):.2f}")
    print(f"Best Night-Val: ep{best_night_ep} = {curve_data[best_night_ep].get('night', 0):.2f}")


def main():
    parser = argparse.ArgumentParser(description="MACVi 체크포인트 분석")
    parser.add_argument("dirs", nargs="*", help="*_results/ 디렉토리 경로들")
    parser.add_argument("--scan", help="이 디렉토리 아래의 모든 *_results/ 자동 탐색")
    parser.add_argument("--baseline", help="기준선 (이름,val,test,M,static,dynamic,water,sky)")
    parser.add_argument("--training-curve", help="train.log 경로")
    parser.add_argument("--from-epoch", type=int, default=0, help="학습 곡선 시작 epoch")
    parser.add_argument("--markers", help="학습 곡선 마커 (ep:label,ep:label,...)")
    args = parser.parse_args()

    # Collect result directories
    result_dirs = list(args.dirs) if args.dirs else []
    if args.scan:
        result_dirs.extend(find_result_dirs(args.scan))

    if not result_dirs and not args.training_curve:
        parser.print_help()
        return

    # Analyze each checkpoint
    if result_dirs:
        results = {}
        for d in result_dirs:
            d = d.rstrip("/")
            name = os.path.basename(d).replace("_results", "")
            # Shorten name: remove submission ID part
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                name = parts[0]

            result = {}

            # summary.json
            summary_path = os.path.join(d, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    result["summary"] = json.load(f)

            # frames_test.csv
            frames_path = os.path.join(d, "frames_test.csv")
            if os.path.exists(frames_path):
                result["frames"] = analyze_frames_test(frames_path)

            # detailed_log.json
            log_path = find_detailed_log(d)
            if log_path:
                result["detailed"] = analyze_detailed_log(log_path)

            if result:
                results[name] = result

        if results:
            print_comparison_table(results, baseline=args.baseline)

    # Training curve
    if args.training_curve:
        markers = {}
        if args.markers:
            for m in args.markers.split(","):
                ep, label = m.split(":", 1)
                markers[int(ep)] = label

        curve = analyze_training_curve(args.training_curve, from_epoch=args.from_epoch)
        print_training_curve(curve, markers=markers)


if __name__ == "__main__":
    main()
