# MUSES four-sensor robustness evaluation

This folder supplies the DGFusion and CAFuser eval-only hook, native MUSES scorer,
and runners. It does not train or change either checkpoint. Use the official MUSES
val split (250 images, native 1080×1920, 19 train IDs).

## Protocol

The sensor order is RGB (`img`/`CAMERA`), LiDAR, Event (`EVENT_CAMERA`), Radar.
EMM evaluates all 15 nonempty present subsets. The full set is clean. A missing
sensor is zero **after** model normalization. RMM evaluates the same 15 subsets
at each `r` in 0.25, 0.5, 0.75. For sensors outside the present subset, each
channel and pixel independently becomes zero when a seeded uniform draw is less
than `r`; the full-set case is clean. The `mean15` row is the arithmetic average
of the 15 mIoUs, including clean. Each setting uses a fresh CPU Torch generator
with the specified seed (default 0), so separate runner processes reproduce the
same setting. NM corrupts **RGB only** at Low/Mid/High strengths: salt-and-pepper
density and additive Gaussian standard deviation `(0.05, 0.1)`, `(0.1, 0.2)`,
`(0.2, 0.5)` in normalized space. The S&P operation samples independent tensor
coordinates with replacement, then applies Gaussian noise.

Definitions reproduced from the read-only benchmark:

- `third_party/mmss_robust_bench/upstream/val_mm_RMM.py:94-124`: 15 missing
  combinations; `torch.rand(image.shape) < r` independently per tensor element.
- `third_party/mmss_robust_bench/upstream/val_mm_NM.py:64-83,96-113`: S&P
  coordinates and the three density/standard-deviation pairs.
- `tools/missing_modality_eval.py:64-111,155-175,252-286`: present subset
  enumeration, seeded CPU mask, and the S&P/Gaussian order used here.

The ReliaDINO runner masks tensors returned by the MUSES val transform, after
`Normalize`. DGFusion and CAFuser normalize each raw `batched_input[modality]`
using `model.pixel_mean[3*i:3*i+3]` and the corresponding standard deviation.
The hook fills a missing raw sensor with that exact mean slice, giving normalized
zero. RMM selects raw pixels or the mean slice per element. Camera changes are
also written to the duplicate `image` key. The hook reads `model.modalities`
to get the actual buffer order and checks that the four MUSES modalities and 12
mean/std entries are present. For RGB NM it normalizes raw RGB with those same
buffers, applies the DELIVER operations, then maps back to raw before forward.
These are the same interventions at the model input. The models' LiDAR/Radar
projections, resize, and learned preprocessing differ; those upstream inputs
are not made identical by zero fill.

## Install on jarvis or bengio

Copy this `muses_robust/` directory to
`/SSDb/jemo_maeng/dgfusion_train/muses_robust/`, then from that repo run:

```bash
patch --dry-run -p1 < muses_robust/train_net_muses_robust.patch
patch -p1 < muses_robust/train_net_muses_robust.patch
```

The patch targets the staged `train_net.py` at
`/mnt/SSD2/stage_drn261111/dgf_src/train_net.py`, which already has the prior
DELIVER hooks. The MUSES hook is inactive when `MUSES_ROBUST_PROTOCOL` is unset.
The runners require `datasets/muses` to link to `/SSDb/jemo_maeng/dset/MUSES`,
the `dgfusion` conda environment at the path in the common runner, and the
matching public MUSES config/weights. The CAFuser config must exist at
`configs/muses/swin/cafuser_swin_tiny_bs8_180k_muses_clre.yaml` on the server.
Each runner refuses a GPU with memory use over 2000 MiB or utilization over
10%. It runs `score_dgf_muses.py` after each detectron2 inference and writes
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
