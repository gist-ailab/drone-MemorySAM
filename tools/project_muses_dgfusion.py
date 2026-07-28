#!/usr/bin/env python
"""
Re-project MUSES sensors (lidar / event_camera / radar) to the RGB plane using the
*DGFusion* parameter set, into a NEW output folder (never touching the existing
`projected_to_rgb/`, which is the historical baseline that produced test 78.979).

WHY THIS SCRIPT EXISTS
----------------------
DGFusion (github.com/timbroed/DGFusion) does NOT precompute projections: its config sets
`LOAD_PROJECTED=False` for every modality and `muses_loader.load_modality_from_raw()`
calls the vendored MUSES-SDK `load_*_projection(..., enlarge_*_points=False)` per sample
and then applies dilation *itself* with the kernel from the config.

We keep our precomputed-PNG pipeline, so we bake the same operations into the PNG.
This is equivalent because DGFusion's projection+dilation both happen at full
(1920, 1080) resolution before any augmentation/resize.

DGFusion params (configs/muses/swin/dgfusion_swin_tiny_bs8_180k_muses_clre.yaml):
    LIDAR:        LOAD=True, MOTION_COMPENSATION=True,  DILATION.KERNAL=(7,7)
    EVENT_CAMERA: LOAD=True,                            DILATION.KERNAL=(3,3)
    RADAR:        LOAD=True, MOTION_COMPENSATION=True,  DILATION.KERNAL=(9,9)

We call the devkit projection functions with enlarge=False and dilate here, exactly
mirroring DGFusion. No devkit source is patched.

TWO ENGINES (why)
-----------------
* 'official' = third_party/MUSES/MUSES  (upstream timbroed/MUSES devkit)
  Used for lidar+radar. VERIFIED to reproduce the existing baseline PNGs bit-exactly.
* 'ours'     = third_party/MUSES        (repo's fixed copy)
  Used for event_camera ONLY, because the upstream event path is BROKEN here:
  each MUSES event .h5 holds exactly the last 30 ms (t is uint32 spanning 0..29999 us,
  `ms_to_idx` has 30 entries), and upstream `accumulate_events(path, 30000)` computes
  `min_t = np.amax(t) - 30000` on uint32 -> UNDERFLOWS to 4294967295 -> selects ZERO
  events. The intended semantics ("last 30 ms") == the whole file, so we use the fixed
  implementation with accumulate_us=0 (= whole file = intended 30 ms window). This
  matches DGFusion's intent and also reproduces the existing baseline.

Both engines expose a `processing` package, so they collide on sys.path; hence exactly
one engine per invocation (lidar/radar together, event_camera separately).

Output encoding matches the devkit / existing data:
    lidar, radar : uint16 PNG, value = (raw + 100) * 150   (background raw 0 -> 15000)
    event        : uint8  PNG, ch0 = +polarity count, ch1 = -polarity count, ch2 = 0
"""
import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Pin BLAS/OpenMP to 1 thread BEFORE numpy/cv2 import. We parallelise with processes;
# letting each worker also spawn a thread pool oversubscribes the box (measured: 48
# workers -> runnable queue ~400 on a 96-core host) and starves co-tenant GPU jobs.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import cv2

cv2.setNumThreads(0)

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICIAL = os.path.abspath(os.path.join(_HERE, '..', 'third_party', 'MUSES', 'MUSES'))
_OURS = os.path.abspath(os.path.join(_HERE, '..', 'third_party', 'MUSES'))

TARGET_SHAPE = (1920, 1080)  # (w, h)

# ---- parameter sets ---------------------------------------------------------
DGF = {
    'lidar':        dict(motion_compensation=True,  kernel=(7, 7)),
    'event_camera': dict(motion_compensation=None,  kernel=(3, 3)),
    'radar':        dict(motion_compensation=True,  kernel=(9, 9)),
}
# Our existing baseline (each verified bit-exactly against projected_to_rgb/)
OURS_PS = {
    'lidar':        dict(motion_compensation=False, kernel=(2, 2)),
    'event_camera': dict(motion_compensation=None,  kernel=(2, 2)),
    'radar':        dict(motion_compensation=True,  kernel=(9, 9)),
}
PARAMSETS = {'dgfusion': DGF, 'ours': OURS_PS}

# Height-channel offset for lidar dilation. The devkit uses 255; DGFusion uses 50.
# Only needs to exceed |min height| so cv2.dilate's max-op ranks real points above
# background(0); identical results for any offset > |min z|. We use the devkit's.
_HEIGHT_OFFSET = 255.0

ENGINE_OF = {'lidar': 'official', 'radar': 'official', 'event_camera': 'ours'}

_M = {}  # per-process module/function cache


def _init(engine, muses_root):
    cv2.setNumThreads(0)  # workers are forked; re-assert in case of spawn
    sys.path.insert(0, _OFFICIAL if engine == 'official' else _OURS)
    if engine == 'official':
        from processing.utils import (load_muses_calibration_data,
                                      rescale_and_shift_image, enlarge_points_in_image)
        from processing.lidar_processing import load_lidar_projection
        from processing.radar_processing import load_radar_projection
        _M.update(calib=load_muses_calibration_data(muses_root),
                  rescale=rescale_and_shift_image, dilate=enlarge_points_in_image,
                  lidar=load_lidar_projection, radar=load_radar_projection)
    else:
        from processing.utils_muses import (load_muses_calibration_data,
                                            enlarge_points_in_image)
        from processing.event_camera_processing import load_event_camera_projection
        _M.update(calib=load_muses_calibration_data(muses_root),
                  dilate=enlarge_points_in_image, event=load_event_camera_projection)


def project_one(args):
    muses_root, out_folder, entry, modality, p, overwrite = args
    try:
        if modality == 'lidar':
            rel = entry['path_to_lidar']
            out = os.path.join(muses_root, out_folder, rel.replace('.bin', '.png'))
            if not overwrite and os.path.exists(out):
                return ('skip', rel, None)
            img = _M['lidar'](os.path.join(muses_root, rel), _M['calib'], entry,
                              p['motion_compensation'], muses_root, TARGET_SHAPE,
                              enlarge_lidar_points=False)   # dilate below, like DGFusion
            mask = img[:, :, 2] != 0.
            img[mask, 2] += _HEIGHT_OFFSET
            img = _M['dilate'](img, kernel_shape=p['kernel'])
            mask_d = img[:, :, 2] != 0.
            img[mask_d, 2] -= _HEIGHT_OFFSET
            png = _M['rescale'](img, 150, 100)

        elif modality == 'radar':
            rel = entry['path_to_radar']
            out = os.path.join(muses_root, out_folder, rel)
            if not overwrite and os.path.exists(out):
                return ('skip', rel, None)
            img = _M['radar'](os.path.join(muses_root, rel), _M['calib'], entry,
                              p['motion_compensation'], muses_root, TARGET_SHAPE,
                              enlarge_radar_points=False)
            img = _M['dilate'](img, kernel_shape=p['kernel'])
            png = _M['rescale'](img, 150, 100)

        elif modality == 'event_camera':
            rel = entry['path_to_event_camera']
            out = os.path.join(muses_root, out_folder, rel.replace('.h5', '.png'))
            if not overwrite and os.path.exists(out):
                return ('skip', rel, None)
            # accumulate_us=0 -> whole file -> the intended "last 30 ms" (see docstring)
            img = _M['event'](os.path.join(muses_root, rel), _M['calib'], TARGET_SHAPE,
                              enlarge_event_camera_points=False, accumulate_us=0)
            png = _M['dilate'](img, kernel_shape=p['kernel'])
        else:
            raise ValueError(modality)

        assert png.shape == (1080, 1920, 3), f'bad shape {png.shape}'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = f'{out}.{os.getpid()}.tmp.png'  # keep .png so cv2 picks a writer
        assert cv2.imwrite(tmp, png), 'imwrite failed'
        os.replace(tmp, out)
        return ('ok', rel, None)
    except Exception:
        return ('fail', modality, traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--muses_root', default='/ailab_mat2/dataset/MUSES')
    ap.add_argument('--output_folder', default='projected_to_rgb_dgf')
    ap.add_argument('--paramset', default='dgfusion', choices=list(PARAMSETS))
    ap.add_argument('--modalities', default='lidar,radar')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    assert a.output_folder != 'projected_to_rgb', 'refusing to overwrite the baseline'
    mods = a.modalities.split(',')
    engines = {ENGINE_OF[m] for m in mods}
    assert len(engines) == 1, (f'mixed engines {engines}: run lidar/radar in one '
                               f'invocation and event_camera in another')
    engine = engines.pop()
    params = PARAMSETS[a.paramset]

    _init(engine, a.muses_root)  # also validates imports before forking
    if engine == 'official':
        from processing.utils import load_meta_data
    else:
        from processing.utils_muses import load_meta_data
    meta = load_meta_data(a.muses_root)
    entries = list(meta.values())
    if a.limit:
        entries = entries[:a.limit]

    jobs = [(a.muses_root, a.output_folder, e, m, params[m], a.overwrite)
            for m in mods for e in entries]
    print(f'engine={engine} paramset={a.paramset} {[(m, params[m]) for m in mods]}')
    print(f'{len(entries)} entries x {len(mods)} = {len(jobs)} jobs, workers={a.workers}'
          f' -> {a.muses_root}/{a.output_folder}', flush=True)

    n_ok = n_skip = n_fail = 0
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(engine, a.muses_root)) as ex:
        futs = [ex.submit(project_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            st, rel, tb = f.result()
            n_ok += st == 'ok'; n_skip += st == 'skip'; n_fail += st == 'fail'
            if st == 'fail':
                print(f'FAIL {rel}\n{tb}', flush=True)
            if i % 250 == 0 or i == len(futs):
                print(f'[{i}/{len(futs)}] ok={n_ok} skip={n_skip} fail={n_fail}', flush=True)
    print(f'DONE ok={n_ok} skip={n_skip} fail={n_fail}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
