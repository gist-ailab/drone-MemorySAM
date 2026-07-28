#!/usr/bin/env python
"""
tools/muses_motioncomp_analysis.py — MUSES LiDAR motion-compensation의 실제 효과를
픽셀 변위로 정량화하고, baseline vs 신규 투영을 RGB 위에 겹쳐 정렬을 눈으로 검증한다.

Motion compensation measurement: project the SAME lidar sweep twice (mc off / mc on),
match points by identity (same input order, same filtering), and report the per-point
pixel displacement distribution. This isolates the geometric effect from any dilation
difference (both runs use dilation OFF here).

Requires the vendored MUSES devkit at third_party/MUSES/MUSES (imported below).

Example:
  python tools/muses_motioncomp_analysis.py --muses_root /ailab_mat2/dataset/MUSES \
    --baseline projected_to_rgb --new projected_to_rgb_dgf \
    --outdir ~/muses_motioncomp --n_frames 3 --n_motion 8
"""
import argparse
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'third_party', 'MUSES', 'MUSES')))

from processing.utils import (load_muses_calibration_data, load_meta_data,  # noqa: E402
                              filter_and_project_pcd_to_image, motion_compensate_pcd)
from processing.lidar_processing import load_lidar_data  # noqa: E402

TARGET = (1920, 1080)


def uv_for(muses_root, entry, calib, mc):
    """UV coords of the lidar sweep with/without motion compensation (no dilation)."""
    pcd = load_lidar_data(os.path.join(muses_root, entry['path_to_lidar']))
    if mc:
        pcd = motion_compensate_pcd(muses_root, entry, pcd,
                                    calib['extrinsics']['lidar2gnss'], ts_channel_num=5)
    uv, pts = filter_and_project_pcd_to_image(
        pcd, calib['extrinsics']['lidar2rgb'], calib['intrinsics']['rgb']['K'], TARGET)
    return uv, pts


def motion_stats(muses_root, entry, calib):
    """Per-point pixel displacement induced by motion compensation.

    We re-project without the image-boundary filter so points correspond 1:1.
    """
    pcd = load_lidar_data(os.path.join(muses_root, entry['path_to_lidar']))
    from processing.utils import (filter_points_by_distance, project_pcd_to_image,
                                  rescale_K)
    base = filter_points_by_distance(pcd.copy(), min_distance=1.0)
    comp = motion_compensate_pcd(muses_root, entry, base.copy(),
                                 calib['extrinsics']['lidar2gnss'], 5)
    K = rescale_K(calib['intrinsics']['rgb']['K'], 1080, 1920)
    uv0 = project_pcd_to_image(K, base[:, :3], calib['extrinsics']['lidar2rgb'])
    uv1 = project_pcd_to_image(K, comp[:, :3], calib['extrinsics']['lidar2rgb'])
    inb = ((uv0[0] > 0) & (uv0[0] < 1919) & (uv0[1] > 0) & (uv0[1] < 1079))
    d = np.linalg.norm((uv1[:2] - uv0[:2])[:, inb], axis=0)
    xyz = np.linalg.norm(base[:, :3] - comp[:, :3], axis=1)[inb]
    return dict(n_points=int(inb.sum()),
                px_mean=float(d.mean()), px_p50=float(np.percentile(d, 50)),
                px_p95=float(np.percentile(d, 95)), px_max=float(d.max()),
                px_gt1=float((d > 1).mean() * 100), px_gt3=float((d > 3).mean() * 100),
                xyz_mean_m=float(xyz.mean()), xyz_max_m=float(xyz.max()))


def colorize(png, modality):
    """Projected PNG -> BGR uint8 heat overlay of the range/count channel."""
    if modality == 'event_camera':
        v = png[:, :, :2].max(axis=2).astype(np.float32)
        m = v > 0
    else:
        r = png.astype(np.float32) / 150.0 - 100.0
        v = r[:, :, 0]
        m = (png != 15000).any(axis=2)
        v = np.clip(v, 0, 80)
    v = (v - v[m].min()) / max(v[m].max() - v[m].min(), 1e-6) if m.any() else v
    hm = cv2.applyColorMap((np.clip(v, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    hm[~m] = 0
    return hm, m


def overlay(rgb, png, modality, alpha=1.0):
    hm, m = colorize(png, modality)
    out = rgb.copy()
    out[m] = (alpha * hm[m] + (1 - alpha) * rgb[m]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--muses_root', default='/ailab_mat2/dataset/MUSES')
    ap.add_argument('--baseline', default='projected_to_rgb')
    ap.add_argument('--new', default='projected_to_rgb_dgf')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--n_frames', type=int, default=3)
    ap.add_argument('--n_motion', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    calib = load_muses_calibration_data(a.muses_root)
    meta = load_meta_data(a.muses_root)

    # --- motion compensation stats over a few frames
    keys = [k for k, v in meta.items() if '/val/' in v['path_to_lidar']][:a.n_motion]
    print('=== LiDAR motion-compensation effect (dilation off, per-point displacement)')
    rows = []
    for k in keys:
        s = motion_stats(a.muses_root, meta[k], calib)
        rows.append(s)
        print(f'{k}: n={s["n_points"]} px_mean={s["px_mean"]:.2f} p50={s["px_p50"]:.2f} '
              f'p95={s["px_p95"]:.2f} max={s["px_max"]:.2f} >1px={s["px_gt1"]:.1f}% '
              f'>3px={s["px_gt3"]:.1f}% xyz_mean={s["xyz_mean_m"]*100:.1f}cm')
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print('MEAN over frames:', {k: round(v, 3) for k, v in agg.items()})

    # --- overlays: baseline vs new, side by side
    vis_keys = keys[:a.n_frames]
    for k in vis_keys:
        e = meta[k]
        rgb = cv2.imread(os.path.join(a.muses_root, e['path_to_frame_camera']))
        if rgb is None:
            print('no rgb for', k)
            continue
        for modality, relkey, ext in (('lidar', 'path_to_lidar', '.bin'),
                                      ('event_camera', 'path_to_event_camera', '.h5'),
                                      ('radar', 'path_to_radar', None)):
            rel = e[relkey]
            rel_png = rel if ext is None else rel.replace(ext, '.png')
            pb = os.path.join(a.muses_root, a.baseline, rel_png)
            pn = os.path.join(a.muses_root, a.new, rel_png)
            if not (os.path.exists(pb) and os.path.exists(pn)):
                print('missing', rel_png)
                continue
            ib, inw = cv2.imread(pb, cv2.IMREAD_UNCHANGED), cv2.imread(pn, cv2.IMREAD_UNCHANGED)
            ob, on = overlay(rgb, ib, modality), overlay(rgb, inw, modality)
            for img, tag in ((ob, 'baseline'), (on, 'dgfusion')):
                cv2.putText(img, tag, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 4)
            pair = np.concatenate([ob, on], axis=0)
            pair = cv2.resize(pair, (1280, 1440))
            out = os.path.join(a.outdir, f'{k}_{modality}_baseline_vs_dgf.jpg')
            cv2.imwrite(out, pair, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print('wrote', out)


if __name__ == '__main__':
    main()
