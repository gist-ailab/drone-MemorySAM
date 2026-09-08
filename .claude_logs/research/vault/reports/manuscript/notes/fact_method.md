# METHOD fact sheet — ReliaDINO (for the Method section writer)

Status: 2026-07-15. Sources: code (`semseg/models/reliadino/{encoder,fusion,model}.py`, `train_reliadino.py`,
worktree `p33-impl` for the P36 router), configs (`configs/b200-deliver_rgbdel_P34_reliadino.yaml`,
worktree `configs/b200-deliver_rgbdel_P36_router.yaml`), vault architecture notes
(`architecture/P34_full_architecture_P35_changes_20260713.md` v3, `architecture/P35_design_20260713.md` v2),
repo `.claude_logs/models/arch-evolution.md` (P30/P31 router sections).

NAMING: paper name = **ReliaDINO** (full model = internal P36; "ReliaDINO (no router)" = internal P34;
P35 = stripped-recipe internal ablation). Internal codes must NOT appear in paper text.

---

## 0. One-paragraph overview (data flow)

Four sensor modalities (RGB, depth, event, LiDAR — each rendered as a 3-channel image) are encoded by a
SINGLE shared **frozen DINOv3 ViT-L/16** whose fused qkv projections carry **independent per-modality LoRA
adapters** (rank 8, Q and V slices only, all 24 blocks). Each modality yields a stride-16 token map
(B, 1024, 64, 64) at 1024² input. A light per-modality **auxiliary decoder** produces per-modality class
posteriors from which all reliability signals are computed (calibrated self-entropy reliability
`rel_cal`, veto-blended corroboration `corr_veto`, centered consistency `B_cons`). Modalities exchange
information through **2 shared cross-modal attention layers** (each modality queries the concatenation of
the other modalities' tokens; a pre-softmax additive reliability bias exists in code but is **OFF/removed
in the final model** — 5-generation negative result). The per-modality fused maps are combined by a
**calibrated competence gate** (softmax over modalities of `rel_cal`/τ) with a **training-free veto floor**.
The fused map goes through a **ViTDet-style SimpleFPN** (strides {4,8,16,32}) and a light **FPN seg head**
(sum at stride 4 → 2 conv blocks → 1×1 classifier). The full model additionally runs the **Per-class
Reliability-anchored Router (PRR)**: per-class, per-pixel modality weights = softmax over modalities of
(zero-init learned logits + λ·`rel_cal`), which reweight the per-modality aux logits; the routed map is
added to the head output as a residual scaled by a zero-init learnable scalar α.

```
x_m (4×) → frozen ViT-L/16 + LoRA_m → F_m (B,1024,64,64)
   F_m → AuxDec_m → z_m → {rel_cal, corr_veto, B_cons}
   F_m ×2 cross-modal attn (pure, no bias in final) → F̃_m
   fused = Σ_m w^g_m ⊙ F̃_m           (competence gate + veto floor)
   routed = Σ_m w^r_m ⊙ z_m           (PRR, per-class per-pixel; full model only)
   ŷ = Head(FPN(fused)) + α·up(routed)
```

---

## 1. Proposed notation table

| Symbol | Meaning | Shape / value |
|---|---|---|
| $\mathcal{M}=\{\mathrm{rgb},\mathrm{depth},\mathrm{event},\mathrm{lidar}\}$, $M=|\mathcal{M}|=4$ | modality set | — |
| $m, j$ | modality indices | — |
| $C$ | number of classes (DELIVER) | 25 |
| $x_m$ | input image of modality $m$ | $(B,3,H,W)$, $H{=}W{=}1024$ |
| $E(\cdot;\theta_0)$ | frozen ViT-L/16 backbone (DINOv3) | 24 blocks, $d{=}1024$ |
| $\Delta\theta_m = \{A^{q}_{m,\ell},B^{q}_{m,\ell},A^{v}_{m,\ell},B^{v}_{m,\ell}\}_{\ell=1}^{24}$ | per-modality LoRA | $r{=}8$ |
| $F_m$ | stride-16 token map of modality $m$ | $(B,d,h,w)$, $h{=}w{=}64$ |
| $\tilde F_m$ | cross-modally attended features | $(B,d,h,w)$ |
| $z_m = D_m(F_m)$ | aux-decoder logits (per-modality posterior source) | $(B,C,h,w)$ |
| $T_m = \exp(\theta^{T}_m)$ | learned per-modality calibration temperature | scalar, clamp $[0.05,20]$ |
| $r_m$ (paper) / `rel_cal` (code) | calibrated self-entropy reliability | $(B,1,h,w)$, stack $(M,B,1,h,w)$ |
| $v_m$ / `corr_veto` | veto-blended corroboration | $(B,1,h,w)$ |
| $b_m$ / `B_cons` | centered leave-one-out Bhattacharyya consistency | $(B,1,h,w)$ |
| $w^{g}_m$ | competence-gate weight | $(B,1,h,w)$, $\sum_m w^g_m = 1$ pixelwise |
| $w^{r}_{m}$ | PRR router weight (per class $k$, per pixel) | $(B,C,h,w)$ |
| $\alpha$ | router residual scale (learnable, zero-init) | scalar |
| $\tau$ | gate temperature | 0.25 |
| $\lambda_{\mathrm{anc}}$ | router reliability anchor | 1.0 |
| $\lambda_1,\lambda_2$ | attention-bias scales (learnable) | init 1.0 / 0.5 — **OFF in final** |
| $\hat y$ | final logits | $(B,C,H,W)$ |

---

## 2. Module 1 — Frozen DINOv3 encoder + per-modality LoRA (C1)

File: `semseg/models/reliadino/encoder.py` (`FrozenViTEncoder`, `MultiModalLoRAQKV`).

- Backbone: timm `vit_large_patch16_dinov3` (DINOv3 ViT-L/16, LVD-1689M pretrain), `num_classes=0`,
  dynamic image size. Fallback chain in code (DINOv2 → random init) exists but real runs used DINOv3.
  ALL backbone parameters frozen (`requires_grad=False`).
- ONE shared backbone for all 4 modalities; modality identity enters ONLY through the active LoRA
  adapter (`set_modality(idx)` flips `active_modality` on every wrapped block before that modality's
  forward). 4 sequential forwards per training step (one per modality).
- LoRA is injected on the **fused qkv linear of every one of the 24 attention blocks**, on the **Q and V
  output slices only (K untouched)** — matches the SAM2-lineage convention. For the Eva/DINOv3 attention,
  `qkv_bias_separate=True` is forced so the wrapper's forward actually runs.

Exact LoRA equation (code: `MultiModalLoRAQKV.forward`). For block $\ell$, active modality $m$, token
input $x\in\mathbb{R}^{d}$, with frozen fused projection $W^{qkv}_{\ell}$ ($3d\times d$):

$$
[q;\,k;\,v] = W^{qkv}_{\ell}x + b^{qkv}_{\ell},\qquad
q \mathrel{+}= \tfrac{\alpha_L}{r}\, B^{q}_{m,\ell} A^{q}_{m,\ell}\, x,\qquad
v \mathrel{+}= \tfrac{\alpha_L}{r}\, B^{v}_{m,\ell} A^{v}_{m,\ell}\, x
$$

with $A\in\mathbb{R}^{r\times d}$ (Kaiming-uniform init, $a=\sqrt5$), $B\in\mathbb{R}^{d\times r}$
(**zero-init** → identity at start), $r=8$, $\alpha_L=r$ so the scale $\alpha_L/r = 1$
(config `LORA_R: 8`, `LORA_ALPHA: null` → α=r).

- LoRA parameter count: $4$ matrices $\times\, r\,d = 4\cdot8\cdot1024 = 32{,}768$ per (block, modality);
  $\times 24$ blocks $\times 4$ modalities $= 3{,}145{,}728 \approx 3.15$M (computed from code, cite as ≈3.1M).
- Output: prefix tokens stripped, tokens reshaped to $(B, 1024, H/16, W/16) = (B,1024,64,64)$.
- No SAM2 dependency anywhere in the package.

## 3. Module 2 — Auxiliary decoders + reliability signals

File: `fusion.py` (`AuxDecoder`, `_compute_signals`).

AuxDecoder $D_m$ (one per modality, ~2.4M params each): Conv3×3(1024→256, no bias) → GroupNorm(32) →
GELU → Conv1×1(256→25), at token resolution (stride 16). $z_m = D_m(F_m) \in \mathbb{R}^{B\times C\times h\times w}$.

**IMPORTANT for the paper**: the aux decoders serve BOTH (a) training-only deep supervision (aux CE +
calibration loss) AND (b) the inference-time source of reliability for the gate and the router. In the
final full model (gate + calibration + router ON) they ARE part of the inference graph. Only in the fully
stripped ablation variant would they be training-only.

All signals are computed from **detached** logits (training-free w.r.t. the decoders through this path);
gradients through the signals reach only $T_m$ (and only when the gate-entropy reg is on, which it never
was — coeff 0 in all runs).

Calibrated self-entropy reliability (`rel_cal`, port of P33 `_stash_comp_rel`):
$$
\hat p_m = \mathrm{softmax}\!\big(z_m / T_m\big),\qquad
r_m = 1 - \frac{H(\hat p_m)}{\log C},\qquad
H(\hat p) = -\sum_{k=1}^{C}\hat p_k \log \hat p_k
$$
with $T_m=\exp(\theta^T_m)$ clamped to $[0.05, 20]$, $\theta^T_m$ zero-init (so $T_m{=}1$ at start).

Uncalibrated self-confidence used inside the veto (temperature-free, matches the validated P32
diagnostic): $p_m=\mathrm{softmax}(z_m)$, $s_m = 1 - H(p_m)/\log C$.

Leave-one-out Bhattacharyya corroboration (port of P32 `_compute_bias_source`):
$$
c_m = \frac{1}{M-1}\Big(\sum_{j} p_j - p_m\Big)_{+},\qquad
\mathrm{corr}_m = \sum_{k=1}^{C}\sqrt{p_{m,k}\, c_{m,k}}
$$

Unique-confidence veto blend (protects a modality that is alone-confident where the consensus is blind):
$$
g_m = \mathrm{clamp}\big(s_m - \max_{j\neq m} s_j,\ 0,\ 1\big),\qquad
v_m = \mathrm{clamp}\big(g_m\, s_m + (1-g_m)\,\mathrm{corr}_m,\ 0,\ 1\big)
$$

Centered signals (zero-mean across modalities — used only by the attention bias, OFF in final):
$$
B^{\mathrm{cal}}_m = r_m - \tfrac1M\textstyle\sum_j r_j,\qquad
B^{\mathrm{cons}}_m = \mathrm{corr}_m - \tfrac1M\textstyle\sum_j \mathrm{corr}_j
$$

Empirical constraints honored by design (from P32/P33 analysis, stated in the code docstring):
(1) the gate uses `rel_cal`, NEVER `corr_veto` (corroboration mis-ranks dead modalities high — it rewards
agreeing with consensus in easy regions); (2) consistency is only ever a secondary additive bias term;
(3) the veto floor is the only place `corr_veto` acts, and it is kept training-free (outside learning).

## 4. Module 3 — Cross-modal attention (2 layers, shared weights)

File: `fusion.py` (`CrossModalAttentionLayer`). SAM2 memory-attention generalization with "frames" =
modalities, re-implemented without SAM2.

Per modality $m$: queries = own tokens $t_m\in\mathbb{R}^{B\times N\times d}$ ($N=hw=4096$);
keys/values = concatenation of the OTHER modalities' tokens ($N_k = (M{-}1)N = 12{,}288$). Two pre-norm
blocks, weights **shared across modalities**; $n_h=8$ heads, MLP ratio 4.0, residual connections:

$$
\mathrm{Attn}(t_m) = \mathrm{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d/n_h}} \;+\; \underbrace{\lambda_1 B^{\mathrm{cal}} + \lambda_2 B^{\mathrm{cons}}}_{\text{key bias — OFF in final}}\Big)V
$$
$$
t_m \leftarrow t_m + W_{\mathrm{proj}}\,\mathrm{Attn}(t_m);\qquad t_m \leftarrow t_m + \mathrm{MLP}(\mathrm{LN}(t_m))
$$

The bias (when on) is a per-KEY-token scalar broadcast over heads and queries — a key token at location
$s$ of modality $j$ carries $\lambda_1 B^{\mathrm{cal}}_j(s) + \lambda_2 B^{\mathrm{cons}}_j(s)$, added
pre-softmax (the RBMA "additive logit bias" form). $\lambda_1,\lambda_2$ learnable scalars (init 1.0, 0.5).

**FINAL MODEL: `ATTN_BIAS.ENABLE: false`, `CONSISTENCY.ENABLE: false`** — the attention is pure
$\mathrm{softmax}(QK^\top/\sqrt{d_h})V$. Evidence: 3 generations of $|\Delta|\le0.03$ with output
cosine $=1.0$; P32 significance test $p=4.5\times10^{-22}$ (attention bias a significant net harm);
G0c full-resolution toggle confirms bias+consistency+veto sum $\approx 0$. This is contribution C4's
negative-result axis, and removing the bias restores the fused flash-attention kernel (speed).
Note: in the P34 (no-router) TRAINING both biases were ON; they are removed in the final recipe.
Output per modality: $\tilde F_m$ (reshaped back to $(B,d,h,w)$).

## 5. Module 4 — Calibrated competence gate + veto floor (C2)

File: `fusion.py` (`_gate`). Output-level fusion of the attended maps:

$$
w^{g} = \mathrm{softmax}_{m}\!\big(r_m / \tau\big),\qquad \tau = 0.25
$$

Training-free veto floor (the only consumer of $v_m$): modalities that are consensus-contradicted AND
self-unconfident get their gate capped, then weights are renormalized:
$$
w^{g}_m \leftarrow \min\!\big(w^{g}_m,\ \kappa\big)\ \ \text{where}\ v_m < \delta,\qquad
w^{g} \leftarrow w^{g} / \textstyle\sum_m w^{g}_m
$$
with $\delta = 0.10$ (`VETO_FLOOR.THRESH`), $\kappa = 0.05$ (`VETO_FLOOR.CAP`). Fused map:
$$
F^{\mathrm{fused}} = \sum_{m} w^{g}_m \odot \tilde F_m \qquad (B,d,h,w)
$$

Optional hinge-entropy regularizer (penalize only when the modality-mixing entropy of $w^g$ drops below a
floor of 0.5 — anti-collapse, not push-to-uniform): coefficient `GATE.ENTROPY_REG` = **0.0 in every run**
(present in code, never active — do not claim it).

Evidence for C2 (from G0c, full-res, full test set, ep120 of the no-router model): removing gate +
calibration changes test 56.64 → 56.38 (**gate+calibration = +0.26 test contribution**, val −0.25
trade). Bias/consistency/veto sum ≈ 0. (Low-res n=40 sub-1pt probes had suggested the opposite sign —
superseded by G0c.)

## 6. Module 5 — Per-class Reliability-anchored Router (PRR; full model only)

Files: worktree `p33-impl`: `semseg/models/reliadino/fusion.py` (`PerClassRouter`, forward step 4b),
`model.py` (residual add), `train_reliadino.py` (loss term). Port of the P31 `ReliabilityAnchoredRouter`
(`semseg/models/sam2/sam2/modules/reliability.py`) — the only large-contribution module in the SAM2
lineage. Config: `configs/b200-deliver_rgbdel_P36_router.yaml` (base = frozen P35 recipe + `MODEL.ROUTER`).

Motivation (documented, quantitative): the scalar competence gate anti-selects per class — measured
night RoadLine: per-modality competence img .798 / depth .001, yet the gate assigns depth .432. A
per-CLASS router lets each class route to the modality that sees it. (Caveat carried from the design doc:
P31's +10–13 gain lived on collapsed SAM2 features (FUSED rank 1.26, CKA .02–.16); on ReliaDINO features
(rank 10.2, CKA .80–.91) the expected gain is sub-1pt to ~1.5.)

Router weights — per class, per pixel, over modalities; heads see the PRE-fusion features $F_m$; the
anchor is the **detached** calibrated reliability (training-free signal, P31 convention):
$$
w^{r} = \mathrm{softmax}_{m}\!\Big( g_m(F_m) + \lambda_{\mathrm{anc}}\, \mathrm{sg}[r_m] \Big)
\in \mathbb{R}^{M\times B\times C\times h\times w},\qquad \lambda_{\mathrm{anc}}=1.0
$$
where $g_m$ = Conv1×1(1024→64) → ReLU → Conv1×1(64→C), **last conv zero-init** (weights AND bias) →
routing is purely reliability-driven at start (collapse-safe; fixes the documented P10–P27
"gate constant-convergence" failure), then learns per-class ratios end-to-end. $r_m$ broadcasts over the
$C$ class channels. sg[·] = stop-gradient. Router params ≈ 0.27M (4 heads × (1024·64+64 + 64·25+25)).

Routed prediction = reliability-routed mixture of the per-modality aux logits, added to the head output
as a residual with a **zero-init learnable scalar** $\alpha$ (`ALPHA_INIT: 0.0`) — the model is exactly
the no-router model at initialization:
$$
z^{\mathrm{route}} = \sum_m w^{r}_m \odot z_m,\qquad
\hat y = \mathrm{Head}(\mathrm{FPN}(F^{\mathrm{fused}})) + \alpha\cdot \mathrm{up}_{\times4}(z^{\mathrm{route}})
$$
(routed map bilinearly upsampled from stride 16 to the head's stride-4 resolution before the add; the sum
is then upsampled to $H\times W$). The router runs in **train AND eval** — it is part of the prediction.
Gradients reach $\alpha$, the router heads, and the aux decoders through this decision path.

"Decisive" regularizer (P31; the earlier 'diversity' reward provably pushed TOWARD uniform): with
per-pixel mixing entropy $H_{\mathrm{pix}} = \mathbb{E}_{b,k,s}\big[-\sum_m w^r_m \log w^r_m\big]$ and
batch-marginal entropy $H_{\mathrm{bar}}$ computed on $\bar w_m = \mathbb{E}_{b,s}[w^r_m]$ (per class,
then averaged):
$$
\mathcal{L}_{\mathrm{route}} = \lambda_{\mathrm{reg}}\,\big(H_{\mathrm{pix}} - H_{\mathrm{bar}}\big),
\qquad \lambda_{\mathrm{reg}} = 0.01
$$
i.e. reward = commit per pixel/class (low local entropy) while keeping all modalities used on average
(high marginal entropy — no global single-modality collapse). Same confident+diverse pairing as the SDC
clustering loss in the lineage.

## 7. Module 6 — SimpleFPN + segmentation head

Files: `encoder.py` (`SimpleFPN`, `LayerNorm2d`), `model.py` (`FPNSegHead`).

SimpleFPN (ViTDet recipe) runs ONCE on the fused stride-16 map (not per modality — deliberate: 4×
pre-fusion FPNs would 4× the cost for no design gain; kept as an ablation seam in the doc only):
- stride-4 branch: ConvT2×2(1024→512) → LayerNorm2d → GELU → ConvT2×2(512→256)
- stride-8 branch: ConvT2×2(1024→512); stride-16: identity; stride-32: MaxPool2×2.
- each level then a lateral: Conv1×1(→256, no bias) → LN2d → Conv3×3(256, no bias) → LN2d. `FPN_DIM: 256`.

FPNSegHead (query-free, GOOSE/SemanticFPN-style, deliberately simple so backbone/fusion effects are
readable): bilinearly upsample all 4 levels to stride 4 and SUM → [Conv3×3(256)+GN(32)+GELU]×2 →
Conv1×1(256→25) → bilinear ×4 to input resolution.

## 8. Losses and training recipe

Total loss (trainer `train_reliadino.py`; router term only in the full model):
$$
\mathcal{L} = \mathcal{L}_{\mathrm{OHEM}}(\hat y, y)
+ \lambda_{\mathrm{aux}} \mathcal{L}_{\mathrm{auxCE}}
+ \lambda_{\mathrm{cal}} \mathcal{L}_{\mathrm{cal}}
+ \mathcal{L}_{\mathrm{gate}}
+ \mathcal{L}_{\mathrm{route}}
$$
$\lambda_{\mathrm{aux}} = 0.5$ (`FUSION.AUX_CE_WEIGHT`), $\lambda_{\mathrm{cal}} = 0.1$
(`CALIBRATION.LAMBDA`), $\mathcal{L}_{\mathrm{gate}} = 0$ (entropy reg off in all runs),
$\lambda_{\mathrm{reg}} = 0.01$.

- Main loss: OHEM cross-entropy (`OhemCrossEntropy`, no class weights), ignore index 255.
- Aux CE (deep supervision): $\mathcal{L}_{\mathrm{auxCE}} = \frac1M\sum_m \mathrm{CE}\big(\mathrm{up}(z_m),\ y{\downarrow}_{4}\big)$
  — per-modality aux logits bilinearly resized to 1/4 label resolution, GT nearest-downsampled ×4.
- Correctness-contrastive calibration loss (exact P31 port; drives $T_m$ AND the aux decoders): at 1/4
  label resolution, with normalized entropy $e_m = H(\mathrm{softmax}(z_m/T_m))/\log C$ and
  $\hat k = \arg\max$ prediction of the calibrated aux logits,
$$
\mathcal{L}_{\mathrm{cal}} = \frac1M\sum_m \Big[
\underbrace{\mathbb{E}_{\,\hat k \neq y}\big[\,1-e_m\,\big]}_{\text{wrong → raise entropy}}
+ \underbrace{\mathbb{E}_{\,\hat k = y}\big[\,e_m\,\big]}_{\text{correct → lower entropy}} \Big]
$$
  i.e. confidence is pushed to rank correctness (directly optimizes reliability AUROC). Per-modality
  reliability AUROC (Mann-Whitney) is stashed each step and logged per epoch (the AUROC-balance figure;
  e.g. night AUROC [.85, .78, .87, .70] for rgb/depth/event/lidar in the no-router run).

Recipe (identical across P34/P35/P36 unless noted):

| Item | Value |
|---|---|
| Dataset | DELIVER, 25 classes, modalities `['img','depth','event','lidar']` |
| Input | 1024×1024 (→ 64×64 tokens/modality) |
| Optimizer | AdamW, lr 6e-4, weight decay 0.01 (trainables only: LoRA+fusion+decoders+head) |
| Schedule | warmup-poly, power 0.9, 10-epoch warmup, warmup ratio 0.1 |
| Epochs | 200 |
| Batch | 4/GPU × 4 GPUs (B200), gradient accumulation to effective batch 16 |
| Precision | AMP bfloat16; DDP (torchrun); `find_unused_parameters=True` |
| Seed | 3407 |
| Eval | every 2 epochs, val + test, per-class IoU; top-5 checkpoints by val AND by test kept |
| PhysAug | ON in P34 (P=0.4, Gaussian filter σ∈[0,1.5] k3 + Fourier-domain aug) — **OFF in P35/P36 (fairness vs DGFusion)** |
| Modality dropout | code seam exists, **OFF in all runs** |
| LoRA norm cap | seam `TRAIN.LORA_NORM_CAP`, **0 = off in the T1/P36 freeze** |
| Checkpoint selection | **val-only** for headline claims (legal-protocol contribution); test-selected numbers disclosed as such |
| TTA / MSF | none (single-scale, no flip) for headline |

Parameter counts (logged by the trainer at launch, no-router model): **total 349.9M / trainable 46.8M
(13.4%)**. Breakdown (computed from code, approximate): LoRA ≈3.1M; 2 cross-attn layers ≈25M; 4 aux
decoders ≈9.5M; SimpleFPN + head ≈9M; temperatures/λ negligible. Router adds ≈0.27M + α (full model);
\todo{exact P36 total/trainable from its launch log}.

## 9. Configuration matrix — what is ON in the final model

| Component | Final full model (P36) | ReliaDINO (no router) = P34 ablation | Role at inference |
|---|---|---|---|
| Frozen DINOv3 ViT-L/16 + per-modality LoRA (r8, Q/V, 24 blocks) | ON | ON | inference graph |
| Cross-modal attention ×2 (pure) | ON | ON | inference graph |
| Attention bias $\lambda_1 B^{\mathrm{cal}}$ | **REMOVED** | ON during its training (dead: Δ≈0, cos=1.0) | — (negative result, C4) |
| Consistency bias $\lambda_2 B^{\mathrm{cons}}$ | **REMOVED** | ON during its training (dead) | — (negative result, C4) |
| Aux decoders $D_m$ | ON | ON | inference graph (signal source for gate/router) + train-time deep supervision |
| Calibration ($T_m$ + $\mathcal{L}_{\mathrm{cal}}$, λ=0.1) | ON | ON | $T_m$ used at inference; loss train-only |
| Competence gate (τ=0.25) | ON | ON | inference graph (+0.26 test, G0c) |
| Veto floor (δ=0.10, κ=0.05) | ON | ON | inference graph, training-free |
| Gate entropy hinge reg | coeff 0 (never active) | coeff 0 | — |
| PRR router (λ_anc=1.0, decisive λ_reg=0.01, α zero-init) | **ON** | OFF | inference graph (residual is part of prediction) |
| Modality dropout | OFF | OFF | — |
| PhysAug | **OFF** | ON (aug-ablation row) | train-only |
| SimpleFPN + FPNSegHead | ON | ON | inference graph |

P35 (internal stripped-recipe row, at most an ablation line): same as P36 minus router
(= gate/veto/calibration KEPT, bias/consistency removed, PhysAug off, val-only selection; died at ep120,
val 67.61 / test 56.14).

---

## 10. Gaps / TODO for the writer

1. **P36 final numbers are TODO** — training still running on B200 (finishes 2026-07-15); best so far
   val 67.74@ep52 / test 57.14@ep58. Every P36 result in the paper must be \todo until the run ends and
   the val-selected checkpoint is evaluated.
2. **P36 code/config location**: the router implementation and `b200-deliver_rgbdel_P36_router.yaml` were
   read from the local worktree `.claude/worktrees/p33-impl/` (not merged to `develop` at fact-sheet time);
   the canonical training copy is on B200. Code semantics above are from the worktree — re-verify against
   the B200 copy before camera-ready if anything was hot-fixed there. (I did NOT find a P35 config file
   locally; the P35/T1 freeze is documented in the P36 config header and `P35_design_20260713.md` §6.)
3. **Exact P36 parameter count** (349.9M/46.8M is the no-router launch log; router adds ≈0.27M):
   \todo{read from the P36 train.log}.
4. **P34 vs final-recipe attention-bias status**: the shipped P34 checkpoints were TRAINED with
   bias+consistency ON (dead at inference by toggle); P35/P36 trained with them OFF. If the ablation table
   compares P34 rows to P36 rows, footnote this train-time difference.
5. **Calibration-LOSS (train-time) performance contribution is UNVERIFIED** — the D5/G0c toggles only cut
   the inference calibration path. Only claim: (a) the +0.26 gate+calibration inference contribution
   (G0c), (b) the AUROC-balance diagnostic. Do NOT claim the calibration loss itself improves mIoU
   (a T3 "calib-pair" training run was optional and \todo{likely never run}).
6. **Head/heads dimension detail**: cross-attention uses $d/n_h = 128$ per head; DINOv3 ViT-L internal
   attention heads = 16 (backbone, frozen) — do not confuse the two in the text.
7. Single-seed everywhere (seed 3407); sub-1pt claims are excluded by project convention — keep that
   sentence in the experimental-setup or limitations text.
8. Eval-protocol parity vs DGFusion (G0d, resize convention ~2pt swing risk): \todo{confirm G0d outcome
   before quoting head-to-head deltas below ~2pt}.
