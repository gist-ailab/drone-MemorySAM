# Figure plan — ReliaDINO RA-L submission

Status: 2026-07-15. Owner: figures agent. Sources of truth: `notes/fact_method.md`,
`notes/fact_experiments.md`, `notes/fact_related.md`. Naming rule: paper text says
**ReliaDINO** (full = with PRR router) and **ReliaDINO (no router)**; internal codes
(P34/P35/P36) never appear in figure text or captions.

TikZ drafts live in `latex/figures/`:
- `fig_teaser.tex` — compilable TikZ (single column)
- `fig_arch.tex` — compilable TikZ (two-column `figure*`)
- `fig_router.tex` — compilable TikZ (single column)
- `fig_qual.tex` — `\framebox` placeholder + caption (assets TODO)
- `fig_analysis.tex` — `\framebox` placeholder + caption (plots TODO)

All five files are BODY-only and meant to be `\input` inside a `figure`/`figure*`
environment in the section files. Required preamble (add to `root.tex`):

```latex
\usepackage{tikz, xcolor, amsmath, amssymb, graphicx}
\usetikzlibrary{positioning, arrows.meta, calc, shapes.geometric}
```

All five files were compile-tested (pdflatex, ieeeconf.cls, 2026-07-15) with the
harness above plus a stub `\newcommand{\todo}[1]{{\color{red}[TODO: #1]}}`;
`fig_arch.tex` is wrapped in `\resizebox{\textwidth}{!}{...}`.

`\todo{}` macro assumed defined in `root.tex`.

Shared color code (used consistently in fig_arch and fig_router; keep for fig_qual
overlays and fig_analysis bars so modality colors mean the same thing everywhere;
Okabe–Ito, colorblind-safe):

| element | RGB | meaning |
|---|---|---|
| `cRGB` | 213, 94, 0 (vermillion) | RGB modality |
| `cDep` | 0, 114, 178 (blue) | Depth modality |
| `cEvt` | 230, 159, 0 (orange) | Event modality |
| `cLid` | 0, 158, 115 (green) | LiDAR modality |
| `cRel` | 136, 78, 160 (purple) | reliability signals (r_m, v_m) |
| `cGray` | 120, 120, 120, dashed | training-only components |
| `OursBlue` | 0, 90, 181 | ReliaDINO markers/accents |
| `SupRed` | 165, 50, 45 | extra-supervision methods (teaser) |

---

## Fig. 1 — fig:teaser (concept: supervision budget vs. DELIVER test mIoU)

**Purpose.** One-glance statement of C3 + headline: ReliaDINO exceeds the prior
DELIVER test SOTA (DGFusion, 56.71) while using *segmentation labels only*, whereas
the strongest baselines buy their accuracy with extra supervision (GT log-depth +
condition meta-text for DGFusion; condition labels + CLIP for CAFuser).

**Content spec.** Single-column scatter, ~8.4 cm wide.
- y-axis: DELIVER test mIoU (CLDE, 4 modalities), range 52–58, ticks every 1.0,
  light horizontal gridlines.
- x-axis: three categorical columns, left → right = decreasing supervision budget:
  1. "GT depth + condition text" (red): DGFusion 56.7.
  2. "condition labels + CLIP" (light red): CAFuser-CAA 55.2, CAFuser (CA²) 55.6.
  3. "segmentation labels only" (blue): CMNeXt 53.0, StitchFusion 53.4,
     GeminiFusion 54.5 (gray dots), ReliaDINO (no router) 57.6 as open blue circle,
     **ReliaDINO (full) = blue star** at 57.14 with `\todo{final}` tag.
- Dashed red horizontal line at 56.71 labeled "prior test SOTA (DGFusion)".
- Short blue arrow from the SOTA line up to the ReliaDINO star, labeled "+0.43".
- Thin arrow under the x-axis: "decreasing extra supervision →".
- Marker semantics: circle = baseline, open circle = our ablation variant,
  star = our full model.

**Data source.** `notes/fact_experiments.md` §(a) test-CLDE cluster (ultimately
BENCH:129–134) + §(0) our numbers (MON:539, MON:582/584). All numbers already in the
draft; the ReliaDINO star and the "+0.43" arrow must be updated when the final run
finishes (both carry `\todo`). NEVER add val-cluster numbers to this plot.

**Caveats.** 57.60 (no router) is test-selected; the caption says so and points to
the checkpoint-protocol section. If the final full-model number changes, update the
star y-position (y = (mIoU − 52) × 0.9 cm) and the delta label.

**Hand-drawing description.** Draw an L: vertical line ~5.5 cm (label "mIoU", ticks
53…58 bottom-to-top, 0.9 cm apart), horizontal line ~7.5 cm. Split the horizontal
span into three invisible columns; write two-line column titles below the axis at
x ≈ 1.3 / 3.4 / 6.2 cm ("GT depth + condition text" in red, "condition labels +
CLIP" in faded red, "segmentation labels only" in blue). Dots (y in cm above axis =
(mIoU−52)×0.9): red dot at (1.0, 4.24)=DGFusion 56.7; two faded-red dots at
(2.9, 2.88)=CAFuser-CAA 55.2 and (2.9, 3.24)=CAFuser 55.6; gray dots at (6.4, 0.90)
CMNeXt 53.0, (6.4, 1.26) StitchFusion 53.4, (6.4, 2.25) GeminiFusion 54.5; open
blue circle at (6.4, 5.04) "ReliaDINO (no router) 57.6"; filled blue 5-point star at
(6.4, 4.63) "ReliaDINO (ours) 57.1 [final TODO]". Names written to the LEFT of the
column-3 markers, to the RIGHT elsewhere. Dashed red horizontal line across the plot
at height 4.24 cm, small label "prior test SOTA 56.7" near the left end, sitting on
the line. At x ≈ 7.2 draw a short vertical blue arrow from the dashed line up to
star height, write "+0.43" beside it. Below everything, a thin left-to-right arrow
captioned "decreasing extra supervision".

---

## Fig. 2 — fig:arch (full architecture)

**Purpose.** Full model overview making C1 and C2 legible: ONE shared frozen DINOv3
multiplexed by color-coded per-modality LoRA; reliability signals derived from
per-modality posteriors; reliability applied ONLY at the output gate and the
per-class router (the attention-logit bias site is shown crossed out — C4);
training-only parts dashed.

**Content spec.** Two-column `figure*`, ~17.5 cm wide. Left → right main flow on the
top row, signal/aux path on the bottom row, losses dashed at the very bottom.
1. Inputs: 4 stacked squares (RGB/Depth/Event/LiDAR), border in modality colors,
   caption "4 × (3, 1024²)".
2. Big box "Frozen DINOv3 ViT-L/16 — single shared encoder, 24 blocks" with a
   snowflake-substitute "(frozen)" tag, containing 4 small color-coded chips
   "LoRA_rgb / LoRA_depth / LoRA_event / LoRA_lidar (r=8, Q/V)". One arrow per
   modality in, annotation "4 sequential passes; modality identity = active LoRA".
3. Output stacked cards "F_m (1024, 64²)".
4. Top row: box "Cross-modal attention ×2 (shared weights) — Q: own tokens; K,V:
   other modalities" with a gray struck-through mini-tag "reliability logit bias —
   REMOVED (negative result, Sec. V)". Then box "calibrated competence gate
   softmax_m(r_m/τ) + training-free veto floor (v_m<δ)" → "Σ_m w_m ⊙ F̃_m" →
   "SimpleFPN {4,8,16,32}" → "seg head" → ⊕ → ŷ.
5. Bottom row: "aux decoders D_m ×4" → stacked "z_m (25, 64²)" → purple box
   "reliability signals: r_m = 1 − H(softmax(z_m/T_m))/log C ; v_m (veto blend)".
   Purple arrows: signals → gate (label "r_m, v_m"), signals → router (label
   "sg[r_m]").
6. Router box (blue accent): "Per-class Reliability-anchored Router (PRR)" with
   inputs F_m (thin bus from the F_m stack) and z_m; output arrow up into ⊕ labeled
   "α · z_route (α zero-init)".
7. Dashed gray loss nodes: "L_aux + L_cal" under z_m; "L_route" under router;
   "L_OHEM" above ŷ.
8. Legend strip: solid = inference graph; gray dashed = training-only; purple =
   reliability signals (computed from detached logits).

**Data source.** Structure only — `notes/fact_method.md` §0–§9 (shapes, τ=0.25,
r=8, δ/κ, zero-init α). No numbers to update except none.

**Hand-drawing description.** Landscape sheet. Far left, stack four small squares
vertically, outline them red/blue/orange/green top-to-bottom, label RGB, Depth,
Event, LiDAR. Draw four arrows into one large rectangle labeled "Frozen DINOv3
ViT-L/16 (shared)"; inside it, near the right edge, four small pills in the same
four colors labeled LoRA_rgb…LoRA_lidar. From its right side, draw a fan of four
slightly offset overlapping small rectangles = feature stack "F_m". From F_m two
paths: (top) arrow to a box "cross-modal attention ×2"; write under that box, in
gray with a line struck through it, "reliability logit bias (removed)". (bottom)
arrow down-right to "aux decoders ×4", then to a small stack "z_m", then to a purple
box "reliability r_m, v_m". Top path continues: attention box → "competence gate +
veto" box → "SimpleFPN" → "head" → a ⊕ circle → ŷ. Draw purple arrows from the
purple box up into the gate box ("r_m, v_m") and right into a bottom-right blue box
"per-class router (PRR)"; also connect z_m → router and a thin line from the F_m
stack along the bottom into the router. From the router draw an arrow up into the ⊕
circle, labeled "α·z_route, α=0 at init". Finally add three dashed gray bubbles:
"L_aux, L_cal" hanging off z_m, "L_route" hanging off the router, "L_OHEM" attached
to ŷ. Bottom corner: 3-line legend (solid / dashed gray = training-only / purple =
reliability).

---

## Fig. 3 — fig:router (Per-class Reliability-anchored Router mechanism)

**Purpose.** Mechanism detail for C2's router half: why per-CLASS routing (the
scalar gate anti-selects per class) and why it is collapse-safe (zero-init head +
reliability anchor + zero-init residual α).

**Content spec.** Single column, vertical flow, ~8.4 cm wide.
1. Top inputs (one per modality, colors): F_m (pre-fusion features, 1024×64²),
   z_m (aux logits, 25×64²), purple r_m (calibrated reliability, 1×64²).
2. F_m → box "g_m: Conv1×1 1024→64 → ReLU → Conv1×1 64→C (zero-init)".
3. ⊕ node: g_m(F_m) + λ_anc · sg[r_m] (r_m broadcast over C; sg = stop-gradient),
   λ_anc = 1.
4. Box "softmax over modalities m" → tensor tag "w_r ∈ R^{M×C×h×w} (per-class,
   per-pixel modality weights)".
5. ⊗ node with z_m, then Σ_m → "z_route (C×64²)".
6. "× α (zero-init, learnable)" → arrow labeled "+ residual to head output"
   feeding a small "Head(FPN(fused))" stub → ŷ.
7. Right-hand side note card (gray, thin border): "at init: g_m ≡ 0, α = 0 ⇒
   weights = softmax(λ_anc r_m), output = no-router model (collapse-safe)".
8. Bottom note card: "decisive regularizer L_route = λ_reg (H_pix − H_bar):
   commit per pixel/class, stay diverse on average (λ_reg = 0.01)".

**Data source.** `notes/fact_method.md` §6 (equations, zero-init details,
λ_anc=1.0, λ_reg=0.01, α zero-init). Motivation numbers for the CAPTION only:
night RoadLine per-modality competence rgb .798 vs depth .001 while the scalar gate
assigns depth .432 (fact_method §6). No plot data.

**Hand-drawing description.** Portrait strip. Top row, three small labeled tokens
left-to-right: "F_m" (4-color stack), "r_m" (purple square), "z_m" (small stack).
From F_m draw an arrow down into a rectangle "g_m: 1×1 conv → ReLU → 1×1 conv
(zero-init)". Its output arrow and a purple arrow from r_m (annotated
"λ_anc·sg[r_m]") meet at a ⊕ circle. Arrow down into a box "softmax over
modalities"; under it write "w_r: M×C×h×w — per-class per-pixel weights". Arrow
down to a ⊗ circle whose second input is a long arrow from z_m down the right
margin; under ⊗ write "Σ_m". Arrow down to "z_route", then through a small box
"×α (zero-init)" into a final ⊕ circle whose other input comes from a left stub
"Head(FPN(fused))"; output arrow "ŷ". To the right of steps 2–4, a gray sticky
note: "init ⇒ routing = softmax(r_m); model = no-router variant". At the bottom, a
second note: "L_route = λ(H_pix − H_bar): decisive per pixel, diverse on average".

---

## Fig. 4 — fig:qual (qualitative grid) — ASSETS TODO

**Purpose.** Visual evidence under adverse conditions: ReliaDINO vs the strongest
public baseline on DELIVER test frames (night / fog / rain / sensor-corruption),
showing cleaner thin structures (poles, traffic lights, road lines) and stable
predictions where RGB dies.

**Content spec.** Two-column `figure*`. Grid 4 rows × 4 cols.
- Rows (conditions): night, fog, rain, corruption case (e.g., LiDAR-jitter or
  event-noise DELIVER split — pick the frame with visible RGB failure).
- Cols: (1) RGB input, (2) GT, (3) CMNeXt or CAFuser prediction `\todo{which
  baseline + its checkpoint}`, (4) ReliaDINO (full). Row label on the left edge,
  column titles on top. DELIVER 25-class palette; add a compact class legend strip
  under the grid for classes visible in the crops.
- Optional 5th column if space allows: router weight map argmax (which modality
  each class trusts) — decide after assets exist.

**Data source / how to produce.**
- Ours: per-condition predicted masks already exist for the no-router ep140
  checkpoint under `/drone_nas/drone/analysis_logs/P34_eval_20260713/per_domain/masks_best_<cond>/`
  (cond ∈ cloud/fog/night/rain/sun); regenerate for the FINAL full-model checkpoint
  with repo tool `tools/eval_per_domain.py` (or `tools/infer_mm.py` for single
  frames) — `--cfg configs/b200-deliver_rgbdel_P36_router.yaml --model_path <final ckpt>`.
- Feature/reliability panels (if 5th column added): `tools/viz_features.py`; night
  panels exist at `/drone_nas/drone/analysis_logs/P34_eval_20260713/viz/panel_night_*.png`.
- Baseline column: `\todo{run official CMNeXt (or CAFuser) checkpoint on the same
  frames — no baseline predictions exist locally}`.
- Frame selection: pick frames where per-class audit says conditions diverge
  (Fence night, TrafficLight sun, Water night — see fact_experiments §(c)).

**Hand-drawing description.** Draw a 4×4 grid of equal rectangles (~4 cm wide
each). Left margin: rotate row labels "night / fog / rain / corruption". Top:
column titles "RGB", "GT", "baseline", "ReliaDINO". Inside each cell just write
the cell id (e.g., "night-GT"). Under the grid, a thin strip of small color swatches
with class names (Road, Car, Pedestrian, Pole, TrafficLight, RoadLine, …).

---

## Fig. 5 — fig:analysis (2-panel analysis figure) — PLOTS TODO

**Purpose.** C4 evidence in one figure: (a) the first per-class DELIVER *test*
audit — where the remaining gap lives (dead classes = benchmark ceiling, not model
failure); (b) reliability signals are meaningful and all four modalities carry
usable information (balanced AUROC) while drop-modality deltas identify depth as
the dominant complement and event as ≈0.

**Content spec.** Two-column `figure*`, two panels side by side.
- Panel (a): horizontal bar chart, 25 DELIVER classes sorted by 5-condition-mean
  test IoU (ReliaDINO no-router ep140 checkpoint = the audited model; footnote
  which checkpoint). Dead classes (Other 3.9, Wall 7.1, Bridge 0.1 — 5-cond means)
  hatched/red with a bracket "dead for official CMNeXt too `\todo{M0-a per-class
  numbers}`". Optional thin overlay markers for the official-CMNeXt per-class test
  IoU when the M0-a audit lands.
- Panel (b): left half — grouped bars, reliability AUROC per modality at night:
  RGB .851, Depth .784, Event .869, LiDAR .695 (modality colors; dashed line at 0.5
  = chance). Right half — grouped bars, drop-modality ΔmIoU per condition
  (cloud/fog/night/rain/sun) per modality: depth ≈ +9.5–11.4 dominates, event ≈ 0
  everywhere (annotate "event ≈ 0 in all 5 conditions").

**Data source.**
- (a): `/drone_nas/drone/analysis_logs/P34_eval_20260713/per_domain_analysis.md`
  lines 3–29 (per-class × condition IoU; take row means) — produced by
  `tools/analyze_per_domain.py`.
- (b) AUROC: `/drone_nas/drone/analysis_logs/P34_eval_20260713/module_diag.json`
  key `conditions/night/reliability_auroc` — produced by `tools/module_diagnostics.py`
  (also `tools/eval_reliability_auroc.py`).
- (b) drop-modality: same JSON, `conditions/*/drop_modality_dmiou`.
- Plot generation: small matplotlib script TODO (put under
  `_paper_submission/figures/scripts/`, read the JSON/md directly; export PDF).
  Regenerate both panels for the FINAL full-model checkpoint before camera-ready;
  keep the no-router audit if the final run's diagnostics are not rerun in time,
  and say so in the caption.

**Hand-drawing description.** Two panels side by side. (a) left: horizontal bars,
one per class, longest ("Sky", "Road" ≈ 97) at top down to "Bridge" ≈ 0 at bottom;
color the bottom three bars (Other, Wall, Bridge) red with diagonal hatching and a
curly bracket labeled "dead — also 0 for official CMNeXt". (b) right, split in two:
upper strip = 4 bars (red/blue/orange/green) between 0.5 and 1.0 labeled "night
reliability AUROC" with values .85/.78/.87/.70; lower strip = 5 groups of 4 bars
(one group per condition), blue (depth) bars ~10 units tall, red (RGB) 3–8, green
(LiDAR) ~1, orange (event) flat at 0 with an arrow annotation "event contributes
≈ 0".

---

## Open TODOs (figure-level)

1. Final full-model number → update `fig_teaser.tex` star position + "+Δ" label +
   caption (`\todo` markers are in place).
2. fig:qual — all image assets missing: final-checkpoint masks (tool above),
   baseline predictions (no local baseline ckpt), frame selection.
3. fig:analysis — write the matplotlib export script; decide whether panel (a)
   shows the audited no-router checkpoint or the final full model.
4. Official-CMNeXt per-class test numbers (M0-a) for the panel-(a) overlay/bracket.
5. If reviewers need it: MUSES teaser variant — blocked on official-protocol MUSES
   result (fact_experiments gap 2).
