# Segmentation analysis pipeline (model-agnostic)

One driver — `tools/seg_analysis_pipeline.py` — runs four analysis dimensions over **any**
DELIVER segmentation checkpoint (SAM2 family `LoRA_Sam_P8..P33` **and** SAM3-RBMA
`LoRA_Sam3_RBMA`), auto-detecting the model family and which diagnostic hooks the checkpoint
actually exposes, then running each dimension with **graceful skip + a logged capability
matrix** so a blank panel is never mistaken for "covered".

## The four dimensions

| Dim | Question | Tool(s) | Model-agnostic? |
|---|---|---|---|
| **D1** per-class / per-domain metrics | which classes drop, is it domain shift | `eval_per_domain.py` → `analyze_per_domain.py` | ✅ GT-based, any model |
| **D2** per-modality encoder info | what each img/depth/event/lidar encoder captures | `viz_features.py` (R2) + `module_diagnostics.py` (B) | 🟡 needs per-modal hooks |
| **D3** adapter / LoRA / router health | is the adapter actually adapting? dead layers? | **`adapter_health.py`** (static) + `module_diagnostics.py` (C/E/F) | ✅ adapter_health is agnostic |
| **D4** post-fusion feature info | fused feature, reliability/competence, UAMM alloc | `viz_features.py` (R2/R3/R5) + `module_diagnostics.py` (D) | 🟡 needs fusion hooks |

**Always run regardless of family:** D1 (metrics come from GT, not hooks) and the
`adapter_health.py` part of D3 (reads the state_dict statically — no forward, no GPU, no data).

## `adapter_health.py` — the model-agnostic gap-filler (NEW)

The previous toolkit could only infer "is the adapter adapting?" indirectly (drop-ΔmIoU).
This reads the checkpoint's state_dict directly and, for every injected qkv site, computes the
effective delta-weight `dW = B @ A`:

- **plain LoRA** (`*.linear_a_q/b_q/a_v/b_v.weight`) — SAM2 plain blocks **and** SAM3-RBMA
  (`sam3_lora_rbma.inject_plain_lora`, B init-0).
- **SoftMoE LoRA** (`*.experts_a.{i}/experts_b.{i}.weight`, `*.gate.weight`) — SAM2 P8+ SoftMoE.

Reports per site: `||dW||_F`, `||B||_F` (B is init-0 → `||B||≈0` after training = **dead adapter**),
ratio `||dW||/||W_base||` (if the frozen qkv is in the ckpt), and for MoE the per-expert `||dW||`
+ expert-usage CV (≈0 = collapse) + gate norm.

```bash
# instant, CPU-only, no data — works on ANY .pth (raw or {'model_state_dict':...})
python tools/adapter_health.py --ckpt <model.pth> --out health.json
```

## Full pipeline

```bash
python tools/seg_analysis_pipeline.py \
  --cfg configs/b200-deliver_rgbdel_P33_1_physaug.yaml \
  --model_path outputs/MMSamP33/.../best_checkpoint.pth \
  --dataset-root /path/to/DELIVER --out-dir <out> --gpu <free-gpu> \
  [--stages D1,D2,D3,D4] [--conditions cloud,fog,night,rain,sun] \
  [--viz-case sun --viz-contains RailTrack --viz-num 2] \
  [--max-imgs 120] [--skip-per-domain]     # skip the heavy 5-condition eval
```

Outputs under `<out-dir>/`:
- `report.md` — consolidated summary + **capability matrix** (which `_last_*` hooks are live) + per-stage status.
- `capability.json` — family, modals, live-hook map, forward-ok.
- `adapter_health.json` — per-layer LoRA health (D3, always).
- `per_domain/` + `per_domain_analysis.md` — per-domain × per-class IoU + failure classes (D1).
- `module_diag.json` — modal-competence / reliability-AUROC / UAMM alloc / drop-ΔmIoU / MoE (D2-4).
- `viz/panel_*.png` — per-modal encoder + fused + reliability + UAMM panels (D2/D4).

## Family behavior (why graceful skip matters)

- **SAM2 family** (`LoRA_Sam_P*`, incl. P32/P33.1) sets rich hooks
  (`_last_per_modal_feats/outputs`, `_last_uamm_spatial`, `_last_amf_weights`, `_last_moe_gates`)
  → D2/D3/D4 fully populate.
- **SAM3-RBMA** exposes **only** `_last_reliab`/`_last_reliab_logits`. The pipeline detects this
  via the capability probe and **skips** the UAMM/per-modal panels (logging the reason) rather
  than emitting silently-blank output. D1 + adapter_health still give a full report.

## Known remaining gaps (documented, not yet built)
- Numeric per-modality encoder **stats** (feature norm/activation/dead-channel) — D2 is PCA-visual + per-class recall only.
- Fused-feature **quantitative** quality (separability/entropy) — D4 is PCA + allocation only.
- A SAM3-RBMA-native renderer for `_last_reliab` as competence/quality maps (RBMA UAMM-equivalent).
