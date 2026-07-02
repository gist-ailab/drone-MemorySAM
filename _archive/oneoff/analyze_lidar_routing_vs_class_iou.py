#!/usr/bin/env python3
"""
Analyze correlation between LiDAR routing/fusion weights and per-class segmentation performance in P12.
Combines frames_test.csv (per-class IoU) with test_pred_P12/detailed_log.json (fusion weights).
Also analyzes val split for comparison.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/media/jemo/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM/outputs/MMSamP12/levine_multiaqua_rgbtl_P12_hardaug4/MULTIAQUA_CMNeXt-B2_ilt"


def load_split(split):
    """Load CSV (per-class IoU) and JSON (fusion weights) for a split, return merged DataFrame."""
    csv_path = f"{BASE}/P12_15949_results/frames_{split}.csv"
    json_path = f"{BASE}/{split}_pred_P12/detailed_log.json"

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        log = json.load(f)

    records = []
    for _, row in df.iterrows():
        img = row["image"]
        entry = log["images"][img]

        rec = {
            "image": img,
            "Static_IoU": row["IoU_static_obstacle"],
            "Dynamic_IoU": row["IoU_dynamic_obstacle"],
            "Water_IoU": row["IoU_water"],
            "Sky_IoU": row["IoU_sky"],
            "mIoU": row["mIoU"],
            # UAMM weights
            "uamm_img": entry["uamm"]["img"],
            "uamm_lidar": entry["uamm"]["lidar"],
            "uamm_thermal": entry["uamm"]["thermal"],
            # AMF weights
            "amf_img": entry["amf"]["img"],
            "amf_lidar": entry["amf"]["lidar"],
            "amf_thermal": entry["amf"]["thermal"],
            # Prediction confidence
            "mean_entropy": entry["pred_confidence"]["mean_entropy"],
            "high_uncertainty_ratio": entry["pred_confidence"]["high_uncertainty_ratio"],
        }

        # MoE routing: average lidar per_token_max across all blocks (Q only)
        lidar_ptm_q = []
        lidar_entropy_q = []
        lidar_top2_gap_q = []
        for block_key, block_val in entry["moe_routing"].items():
            if "_Q" in block_key and "lidar" in block_val:
                lidar_ptm_q.append(block_val["lidar"]["per_token_max"])
                lidar_entropy_q.append(block_val["lidar"]["entropy_ratio"])
                lidar_top2_gap_q.append(block_val["lidar"]["top2_gap"])

        rec["moe_lidar_ptm_mean"] = np.mean(lidar_ptm_q) if lidar_ptm_q else np.nan
        rec["moe_lidar_entropy_mean"] = np.mean(lidar_entropy_q) if lidar_entropy_q else np.nan
        rec["moe_lidar_top2gap_mean"] = np.mean(lidar_top2_gap_q) if lidar_top2_gap_q else np.nan

        records.append(rec)

    return pd.DataFrame(records)


def analyze_split(df, split_name):
    """Run all analyses on a split."""
    print("=" * 80)
    print(f"  SPLIT: {split_name.upper()} ({len(df)} images)")
    print("=" * 80)

    classes = ["Static", "Dynamic", "Water", "Sky"]
    lidar_features = [
        ("amf_lidar", "AMF LiDAR weight"),
        ("uamm_lidar", "UAMM LiDAR weight"),
        ("moe_lidar_ptm_mean", "MoE LiDAR per-token-max (mean)"),
        ("moe_lidar_entropy_mean", "MoE LiDAR entropy ratio (mean)"),
        ("moe_lidar_top2gap_mean", "MoE LiDAR top2-gap (mean)"),
    ]

    # ==========================================
    # 1. Basic statistics of lidar weights
    # ==========================================
    print("\n--- 1. Basic Statistics of LiDAR Fusion Weights ---")
    for feat, label in lidar_features:
        vals = df[feat].dropna()
        print(f"  {label}:")
        print(f"    mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}  range={vals.max()-vals.min():.4f}")

    # Also show all modalities for AMF and UAMM
    print("\n  AMF weight distribution (all modalities):")
    for m in ["img", "lidar", "thermal"]:
        vals = df[f"amf_{m}"]
        print(f"    {m:>8s}: mean={vals.mean():.4f}  std={vals.std():.4f}  range=[{vals.min():.4f}, {vals.max():.4f}]")

    print("\n  UAMM weight distribution (all modalities):")
    for m in ["img", "lidar", "thermal"]:
        vals = df[f"uamm_{m}"]
        print(f"    {m:>8s}: mean={vals.mean():.4f}  std={vals.std():.4f}  range=[{vals.min():.4f}, {vals.max():.4f}]")

    # ==========================================
    # 2. High vs Low IoU comparison for Dynamic
    # ==========================================
    print("\n--- 2. Dynamic Class: High IoU (>50) vs Low IoU (<10) ---")
    high_dyn = df[df["Dynamic_IoU"] > 50]
    low_dyn = df[df["Dynamic_IoU"] < 10]
    print(f"  High Dynamic IoU (>50): {len(high_dyn)} frames, mean IoU={high_dyn['Dynamic_IoU'].mean():.1f}")
    print(f"  Low Dynamic IoU  (<10): {len(low_dyn)} frames, mean IoU={low_dyn['Dynamic_IoU'].mean():.1f}")

    if len(high_dyn) > 0 and len(low_dyn) > 0:
        for feat, label in lidar_features:
            h = high_dyn[feat].dropna()
            l = low_dyn[feat].dropna()
            if len(h) > 1 and len(l) > 1:
                t_stat, p_val = stats.ttest_ind(h, l, equal_var=False)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"    {label}:  high={h.mean():.4f}+-{h.std():.4f}  "
                      f"low={l.mean():.4f}+-{l.std():.4f}  "
                      f"diff={h.mean()-l.mean():.4f}  p={p_val:.4f} {sig}")
            else:
                print(f"    {label}:  high={h.mean():.4f} (n={len(h)})  low={l.mean():.4f} (n={len(l)})  [too few for t-test]")
    else:
        print("  [Not enough frames in one or both groups]")

    # ==========================================
    # 3. High vs Low IoU comparison for Water
    # ==========================================
    print("\n--- 3. Water Class: High IoU (>90) vs Low IoU (<50) ---")
    high_water = df[df["Water_IoU"] > 90]
    low_water = df[df["Water_IoU"] < 50]
    print(f"  High Water IoU (>90): {len(high_water)} frames, mean IoU={high_water['Water_IoU'].mean():.1f}")
    print(f"  Low Water IoU  (<50): {len(low_water)} frames, mean IoU={low_water['Water_IoU'].mean():.1f}")

    if len(high_water) > 0 and len(low_water) > 0:
        for feat, label in lidar_features:
            h = high_water[feat].dropna()
            l = low_water[feat].dropna()
            if len(h) > 1 and len(l) > 1:
                t_stat, p_val = stats.ttest_ind(h, l, equal_var=False)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"    {label}:  high={h.mean():.4f}+-{h.std():.4f}  "
                      f"low={l.mean():.4f}+-{l.std():.4f}  "
                      f"diff={h.mean()-l.mean():.4f}  p={p_val:.4f} {sig}")
            else:
                print(f"    {label}:  high={h.mean():.4f} (n={len(h)})  low={l.mean():.4f} (n={len(l)})  [too few for t-test]")
    else:
        print("  [Not enough frames in one or both groups]")

    # ==========================================
    # 4. Correlation: LiDAR weights vs each class IoU
    # ==========================================
    print("\n--- 4. Pearson Correlation: LiDAR Features vs Per-Class IoU ---")
    print(f"  {'Feature':<40s} | {'Static':>8s} | {'Dynamic':>8s} | {'Water':>8s} | {'Sky':>8s} | {'mIoU':>8s}")
    print("  " + "-" * 90)
    for feat, label in lidar_features:
        row = f"  {label:<40s} |"
        for cls in classes + ["m"]:
            col = f"{cls}_IoU" if cls != "m" else "mIoU"
            valid = df[[feat, col]].dropna()
            if len(valid) > 2:
                r, p = stats.pearsonr(valid[feat], valid[col])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                row += f" {r:+.4f}{sig:>3s} |"
            else:
                row += f"   {'N/A':>5s} |"
        print(row)

    # ==========================================
    # 5. Spearman Correlation (for robustness)
    # ==========================================
    print(f"\n--- 5. Spearman Correlation: LiDAR Features vs Per-Class IoU ---")
    print(f"  {'Feature':<40s} | {'Static':>8s} | {'Dynamic':>8s} | {'Water':>8s} | {'Sky':>8s} | {'mIoU':>8s}")
    print("  " + "-" * 90)
    for feat, label in lidar_features:
        row = f"  {label:<40s} |"
        for cls in classes + ["m"]:
            col = f"{cls}_IoU" if cls != "m" else "mIoU"
            valid = df[[feat, col]].dropna()
            if len(valid) > 2:
                r, p = stats.spearmanr(valid[feat], valid[col])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                row += f" {r:+.4f}{sig:>3s} |"
            else:
                row += f"   {'N/A':>5s} |"
        print(row)

    # ==========================================
    # 6. Per-block MoE analysis (lidar routing consistency)
    # ==========================================
    print("\n--- 6. LiDAR MoE Routing Variation Across Blocks ---")
    json_path = f"{BASE}/{split_name}_pred_P12/detailed_log.json"
    with open(json_path) as f:
        log = json.load(f)

    block_lidar_ptm_all = {}
    for img_key, entry in log["images"].items():
        for block_key, block_val in entry["moe_routing"].items():
            if "_Q" in block_key and "lidar" in block_val:
                if block_key not in block_lidar_ptm_all:
                    block_lidar_ptm_all[block_key] = []
                block_lidar_ptm_all[block_key].append(block_val["lidar"]["per_token_max"])

    print(f"  {'Block':<12s} | {'mean':>7s} | {'std':>7s} | {'min':>7s} | {'max':>7s} | {'range':>7s}")
    print("  " + "-" * 60)
    for bk in sorted(block_lidar_ptm_all.keys(), key=lambda x: (int(x.split("Block")[1].split("_")[0]), x)):
        vals = np.array(block_lidar_ptm_all[bk])
        print(f"  {bk:<12s} | {vals.mean():7.4f} | {vals.std():7.4f} | {vals.min():7.4f} | {vals.max():7.4f} | {vals.max()-vals.min():7.4f}")

    # Cross-image variation
    print(f"\n--- 7. Cross-Image Variation Summary ---")
    for feat, label in [("amf_lidar", "AMF LiDAR"), ("uamm_lidar", "UAMM LiDAR"),
                         ("amf_img", "AMF RGB"), ("amf_thermal", "AMF Thermal"),
                         ("uamm_img", "UAMM RGB"), ("uamm_thermal", "UAMM Thermal")]:
        vals = df[feat].dropna()
        cv = vals.std() / vals.mean() * 100 if vals.mean() != 0 else 0
        print(f"  {label:<16s}: mean={vals.mean():.4f}  std={vals.std():.6f}  CV={cv:.2f}%  range={vals.max()-vals.min():.6f}")

    return df


def main():
    print("LiDAR Routing vs Per-Class IoU Analysis for P12")
    print("Model: LoRA_Sam_P12 (Input-Conditioned Soft MoE LoRA)")
    print()

    # Analyze test split (night, the challenging one)
    df_test = load_split("test")
    analyze_split(df_test, "test")

    print("\n\n")

    # Analyze val split (day, for comparison)
    df_val = load_split("val")
    analyze_split(df_val, "val")

    # ==========================================
    # Cross-split comparison
    # ==========================================
    print("\n\n" + "=" * 80)
    print("  CROSS-SPLIT COMPARISON: Test (Night) vs Val (Day)")
    print("=" * 80)
    print(f"\n  {'Metric':<30s} | {'Val (Day)':>12s} | {'Test (Night)':>12s} | {'Diff':>10s}")
    print("  " + "-" * 70)
    for feat, label in [("amf_lidar", "AMF LiDAR"),
                         ("amf_img", "AMF RGB"),
                         ("amf_thermal", "AMF Thermal"),
                         ("uamm_lidar", "UAMM LiDAR"),
                         ("uamm_img", "UAMM RGB"),
                         ("uamm_thermal", "UAMM Thermal"),
                         ("moe_lidar_ptm_mean", "MoE LiDAR PTM"),
                         ("moe_lidar_entropy_mean", "MoE LiDAR Entropy"),
                         ("mean_entropy", "Pred Entropy")]:
        v = df_val[feat].mean()
        t = df_test[feat].mean()
        print(f"  {label:<30s} | {v:12.4f} | {t:12.4f} | {t-v:+10.4f}")

    # Also show per-class IoU comparison
    print(f"\n  {'Class IoU':<30s} | {'Val (Day)':>12s} | {'Test (Night)':>12s} | {'Diff':>10s}")
    print("  " + "-" * 70)
    for cls in ["Static", "Dynamic", "Water", "Sky", "m"]:
        col = f"{cls}_IoU" if cls != "m" else "mIoU"
        label = f"{cls} IoU" if cls != "m" else "mIoU"
        v = df_val[col].mean()
        t = df_test[col].mean()
        print(f"  {label:<30s} | {v:12.2f} | {t:12.2f} | {t-v:+10.2f}")


if __name__ == "__main__":
    main()
