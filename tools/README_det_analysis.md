# Detection analysis suite

Standard analysis for **any** detection model in this repo. Do not write new
analysis code per model — run these and, if a genuinely new module appears, add one
line to the toggle registry.

Counterpart of the seg suite (`tools/module_ablation.py`,
`tools/seg_analysis_pipeline.py`) and follows the same contract: config+checkpoint
driven, JSON + Markdown output, toggles that auto-skip when absent.

## The three questions -> which tool

| question | tool | output |
|---|---|---|
| per-class performance | `det_eval_breakdown.py` (D1) | AP / AP50 / n_gt per class, incl. per-class night vs normal |
| night (low-light) performance | `det_eval_breakdown.py` (D1) | mAP triplet on night clips vs the rest + the delta |
| did the module actually work | `det_module_ablation.py` (D2) | ΔmAP50 + detection agreement per module -> ACTIVE / NO-OP |

## Run

```bash
# both stages
python tools/det_analysis_pipeline.py \
    --cfg configs/det/det_P37a_cefr_yeon.yaml \
    --ckpt outputs/det_final_P37a_cefr_yeon/det_P37a_cefr_yeon/best_checkpoint.pth \
    --out-dir analysis/P37a --gpu 0

# or a single stage
python tools/det_eval_breakdown.py  --cfg <cfg> --ckpt <ckpt> --out analysis/P37a_breakdown
python tools/det_module_ablation.py --cfg <cfg> --ckpt <ckpt> --out analysis/P37a_ablation
```

`--limit N` caps the images for a cheap smoke run. `--mode test` for the test split.

## Night split

Defaults to the poongsan `final` test clips `capture_20260618_114021` +
`capture_20260618_115624` (1,768 frames) vs `capture_20260618_114808` (1,471
normal). Override with `--lowlight-clips a,b` — matching is a substring test on
`file_name`, so any dataset with clip-structured paths works.

## Module toggles (auto-skip when the attribute is absent)

| toggle | disables | applies to |
|---|---|---|
| `p36_router_det_off` | `det_router_alpha` -> 0 (router->det seam) | P36-Det+ |
| `p37b_classtoken_det_off` | `det_classtoken_alpha` -> 0 | P37b-Det |
| `p36_router_off` | `fusion.router_alpha` -> 0 | P36+ |
| `p37a_cefr_off` | `fusion.cefr.a` -> -20 (sigma~0) | P37a |
| `attn_bias_off` / `consistency_off` | RBMA bias / consistency | P34 lineage |
| `p38_m2f_beta_off` | `m2f.beta` -> 0 | P38 |
| `p39_modalsrc_off` | V2 modal-token queries -> fused-only | P39 |
| `p39_anchored_off` | V3 anchored queries -> free only | P39 |
| `p39_trunkexp_off` / `p39_query_off` | V1 / query path | P39 |

**Adding a future model:** if its module exposes a scalar/flag on `seg_model`,
`seg_model.fusion` or `seg_model.m2f`, add one `_attr(...)` line in
`det_module_ablation.make_toggles`. Nothing else changes.

## Verdicts

`ACTIVE(+)` turning the module off hurts -> it contributes.
`ACTIVE(-)` turning it off helps -> net negative, consider removing.
`NO-OP` |ΔmAP50| < 0.005 **and** top-10 detection agreement > 0.99 -> dead weight.

The no-op check exists because four consecutive zero-init-residual mechanisms
(P36 router, P37a CEFR, P37b classtoken, P38 m2f) shipped without contributing.
A module is only "working" if switching it off moves the predictions.
