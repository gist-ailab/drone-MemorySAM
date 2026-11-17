# MUSES four-sensor robustness evaluation

This folder supplies the DGFusion and CAFuser eval-only hook, native MUSES scorer,
and runners. It does not train or change either checkpoint. Use the official MUSES
val split (250 images, native 1080×1920, 19 train IDs).

## Protocol

The sensor order is RGB (`img`/`CAMERA`), LiDAR, Event (`EVENT_CAMERA`), Radar.
All corruptions are applied in the **public benchmark input space** (DELIVER
`augmentations_mm.Normalize` @b3a76c3 :34-40, used by arXiv 2503.18445): RGB is
/255 + ImageNet-normalized, the other sensors are raw/255. EMM evaluates all 15
nonempty present subsets; the full set is clean; a missing sensor is 0 in that
space (RGB -> channel mean, LiDAR/Event/Radar -> raw 0). RMM evaluates the same 15
subsets at each `r` in 0.25, 0.5, 0.75: for sensors outside the present subset each
element becomes 0 (same space) when a seeded uniform draw is less than `r`. The
`mean15` row is the arithmetic average of the 15 mIoUs, including clean. Each setting
uses a fresh CPU Torch generator with the specified seed (default 0).
NM (default `--nm-scope all` / `MUSES_ROBUST_NM_SCOPE=all`, changed 2026-11-16 by user
decision to follow prior work) follows upstream `val_mm_NM.py:64-84,104-113` exactly:
salt-and-pepper on every sensor first (per-tensor max/min, coordinates with
replacement, batch 1), then additive Gaussian on every sensor except Event, with
(density, sigma) = (0.05, 0.1) / (0.1, 0.2) / (0.2, 0.5); sigma is in normalized units
for RGB and raw/255 units for the other sensors. Radar is not in the upstream code
(DELIVER has no radar) and is treated like LiDAR. `rgb` scope = RGB only (figure dump).
`python tools/verify_muses_nm.py` checks the input space (upstream Normalize == ours) and
our runner, the DELIVER tool and this hook against the verbatim upstream functions
(same seed -> max abs diff 0 / 1e-6 for the hook).
History: the hook before 2026-11-16 corrupted non-camera sensors in d2-normalized units and
filled missing ones with the channel mean (not upstream); a 2026-11-17 morning version wrongly
used raw 0-255 for non-RGB (255x weaker noise) and was reverted the same day.

Definitions reproduced from the read-only benchmark:

- `third_party/mmss_robust_bench/upstream/val_mm_RMM.py:94-124`: 15 missing
  combinations; `torch.rand(image.shape) < r` independently per tensor element.
- `third_party/mmss_robust_bench/upstream/val_mm_NM.py:64-83,96-113`: S&P
  coordinates and the three density/standard-deviation pairs.
- `tools/missing_modality_eval.py:64-111,155-175,252-286`: present subset
  enumeration, seeded CPU mask, and the S&P/Gaussian order used here.

The ReliaDINO runner receives exactly the upstream space (RGB normalized, other sensors
raw/255; `NORM_ALL_MODALS` false, asserted) and corrupts it directly. DGFusion and CAFuser
receive raw 0-255 tensors; the hook maps the camera with `model.pixel_mean/std[0:3]` and the
other sensors with /255, corrupts, and maps back. Camera changes are
also written to the duplicate `image` key. The hook reads `model.modalities`
to get the actual buffer order and checks that the four MUSES modalities and 12
mean/std entries are present. For RGB NM it normalizes raw RGB with those same
buffers, applies the DELIVER operations, then maps back to raw before forward.
These are the same interventions at the model input. The models' LiDAR/Radar
projections, resize, and learned preprocessing differ; those upstream inputs
are not made identical by zero fill.

## Install on jarvis

Copy this `muses_robust/` directory to
`/SSDb/jemo_maeng/dgfusion_train/muses_robust/`. The runner adds this folder's
parent to `PYTHONPATH`, so `muses_robust.muses_robust_hook` resolves even when
the runner is invoked from another directory. Before the first evaluation it
dry-runs and applies `train_net_muses_robust.patch` if the hook marker is absent.
The patch is applied once; subsequent invocations reuse it.

The patch targets the staged `train_net.py` at
`/mnt/SSD2/stage_drn261111/dgf_src/train_net.py`, which already has the prior
DELIVER hooks. The MUSES hook is a no-op when `MUSES_ROBUST_PROTOCOL` is unset
or `clean`, so other users of that repo are unaffected. The bundled CAFuser
config is copied to `configs/muses/swin/` if absent; an existing different
file stops the runner. Its `_BASE_` chain is
`configs/muses/oneformer_R50_bs8_90k_muses.yaml` →
`configs/muses/Base-MUSES-UnifiedSegmentation.yaml`, both present in the staged
jarvis repo copy. The runner checks both files before evaluation.
The runners require `datasets/muses` to link to `/SSDb/jemo_maeng/dset/MUSES`,
the `dgfusion` Python at `/home/jemo_maeng/miniconda3/envs/dgfusion/bin/python`
(or set `DGF_PYTHON` to an executable), and matching public weights. The
runner exports the repo's `OneFormer` and root import paths plus the hook path.
Each runner refuses a GPU with memory use over 2000 MiB or utilization over
10%. Each evaluation uses a random localhost detectron2 dist port to avoid
collisions between parallel GPU runs. It runs `score_dgf_muses.py` after each
detectron2 inference and writes
`report.json`, `hist_full.npy`, per-condition `hist_*.npy`, and native trainID
uint8 predictions at `<case_out>/pred/<weather>/<tod>/<stem>_frame_camera.png`.
For a single setting, `<case_out>` is exactly the runner's `OUT` argument.

## Commands

Set `DGF_WEIGHTS`, `CAF_WEIGHTS`, `OURS_CKPT`, and output paths to real files.
All commands below are examples and run GPU inference only when invoked.

```bash
OURS_CFG=configs/yeon-muses_rgbelr_P39_1_physaugoff_taps_screen40_A2_s2_768.yaml
python tools/muses_robust_eval.py --cfg "$OURS_CFG" --ckpt "$OURS_CKPT" --gpu 0 --out "$OUT/ours/clean" --protocol clean
python tools/muses_robust_eval.py --cfg "$OURS_CFG" --ckpt "$OURS_CKPT" --gpu 0 --out "$OUT/ours/emm" --protocol emm
python tools/muses_robust_eval.py --cfg "$OURS_CFG" --ckpt "$OURS_CKPT" --gpu 0 --out "$OUT/ours/rmm" --protocol rmm
python tools/muses_robust_eval.py --cfg "$OURS_CFG" --ckpt "$OURS_CKPT" --gpu 0 --out "$OUT/ours/nm" --protocol nm
# Four required prediction dumps (single setting puts pred/ directly under --out):
python tools/muses_robust_eval.py --cfg "$OURS_CFG" --ckpt "$OURS_CKPT" --gpu 0 --out "$OUT/ours/rgb_missing" --protocol emm --subset lidar,event,radar --dump
python tools/muses_robust_eval.py --cfg "$OURS_CFG" --ckpt "$OURS_CKPT" --gpu 0 --out "$OUT/ours/lidar_missing" --protocol emm --subset img,event,radar --dump
python tools/muses_robust_eval.py --cfg "$OURS_CFG" --ckpt "$OURS_CKPT" --gpu 0 --out "$OUT/ours/lidar_event_missing" --protocol emm --subset img,radar --dump
python tools/muses_robust_eval.py --cfg "$OURS_CFG" --ckpt "$OURS_CKPT" --gpu 0 --out "$OUT/ours/rgb_high" --protocol nm --level high --dump

bash muses_robust/run_dgf_muses_robust.sh "$DGF_WEIGHTS" "$OUT/dgf/clean" clean clean 0
bash muses_robust/run_dgf_muses_robust.sh "$DGF_WEIGHTS" "$OUT/dgf" emm all 0
bash muses_robust/run_dgf_muses_robust.sh "$DGF_WEIGHTS" "$OUT/dgf" rmm 0.25 0
bash muses_robust/run_dgf_muses_robust.sh "$DGF_WEIGHTS" "$OUT/dgf" rmm 0.5 0
bash muses_robust/run_dgf_muses_robust.sh "$DGF_WEIGHTS" "$OUT/dgf" rmm 0.75 0
for level in low mid high; do
  bash muses_robust/run_dgf_muses_robust.sh "$DGF_WEIGHTS" "$OUT/dgf/nm_$level" nm "$level" 0
done

bash muses_robust/run_caf_muses_robust.sh "$CAF_WEIGHTS" "$OUT/caf/clean" clean clean 0
bash muses_robust/run_caf_muses_robust.sh "$CAF_WEIGHTS" "$OUT/caf" emm all 0
bash muses_robust/run_caf_muses_robust.sh "$CAF_WEIGHTS" "$OUT/caf" rmm 0.25 0
bash muses_robust/run_caf_muses_robust.sh "$CAF_WEIGHTS" "$OUT/caf" rmm 0.5 0
bash muses_robust/run_caf_muses_robust.sh "$CAF_WEIGHTS" "$OUT/caf" rmm 0.75 0
for level in low mid high; do
  bash muses_robust/run_caf_muses_robust.sh "$CAF_WEIGHTS" "$OUT/caf/nm_$level" nm "$level" 0
done
```

For a single baseline EMM dump, substitute `emm lidar,event,radar`,
`emm img,event,radar`, or `emm img,radar` for `emm all`; pass a distinct `OUT`
directory and its `pred/` tree is directly beneath it. Full `emm all` already
saves those three dumps. The NM high
command saves the RGB High dump. Baseline outputs are scored from their
`sem_seg_predictions.json` using the staged scorer's trainID mapping and
native MUSES GT. No GPU evaluation has been run as part of this code change.
