# FACT SHEET — Experiments / Numbers (ReliaDINO, RA-L draft)

> Compiled 2026-07-15 (~15:30 KST). Every number below carries its source `path:line`.
> Naming map (NEVER print internal codes in paper text):
> P36 = **ReliaDINO (full, with PRR router)** — FLAGSHIP, training still running.
> P34 = **ReliaDINO (no router)** — ablation variant. P35 = stripped-recipe internal variant (ablation row at most).
> Abbrev: `MON` = repo `.claude_logs/experiments/monitor-log.md` · `BENCH` = `/nas_jm/Research/26_MultimodalSeg/relatedworks/09_benchmark_tables_deliver_muses_mcubes.md` · `ARCH` = `/nas_jm/Research/26_MultimodalSeg/architecture/P34_full_architecture_P35_changes_20260713.md` · `AN34` = `/drone_nas/drone/analysis_logs/P34_eval_20260713/` · `BBR` = `/drone_nas/drone/analysis_logs/backbone_report_20260713/backbone_lineage_report.md`

---

## (0) OUR NUMBERS — training-curve best points

### P36 = ReliaDINO (full, router) — ⚠ STILL TRAINING, final numbers TODO
- Best so far: **val 67.74 @ep52 / test 57.14 @ep58** (test-SOTA DGFusion 56.71 **+0.43**). Source: MON:582 (2026-07-14 19:51 entry), reconfirmed MON:584.
- LATEST monitor entry (newest in file as of compile time), **verbatim** (MON:584):
  > `| 2026-07-15 00:20 | 112 | 63.18 (best **67.74@ep52**, **60ep 미갱신**; vs 68.6 −0.86) | 56.38 (best **57.14@ep58**, 54ep 미갱신; vs 56.71 **+0.43**) | val ep52 / test ep58 | G2-5 98%, 39 proc | **⚠️ val 열화 확정(미회복 36ep)**. ep76 dip 이후 val 62~63 밴드 고착(ep108 62.85/110 63.66/112 63.18) — 게이트/라우터 붕괴 **영구화**. test는 56.38까지 회복하나 57.14 미달. **best 사실상 잠김 → P36 최종 = val 67.74/test 57.14, P34(68.19/57.60)에 val −0.45/test −0.46 미달 확정적**. 단 **test-SOTA(56.71) +0.43 돌파는 유지**. ep112/200(~12:20 완주, 마감 여유). |`
- Expected completion ~2026-07-15 12:20 KST (MON:584). **No newer entry exists in monitor-log at compile time → final ep200 numbers = TODO (re-check MON RUN-23 + B200 backup re-sync).**
- Curve milestones: val 66.28@ep8/test 54.50@ep6 (MON:579) → val 66.43@ep26/test 55.42@ep30 (MON:580) → val 67.03@ep34/test 55.44@ep46 (MON:581) → val 67.74@ep52/test 57.14@ep58 (MON:582). Reached test-SOTA at **ep58** vs P34's ep116 (MON:582). Post-ep76 val degradation to 62–63 band, unrecovered (MON:583–584).
- vs same-epoch P34: ep30 val +0.6/test +2.0 (MON:580); ep46 val +1.6/test +2.4 (MON:581); ep76-pt test +0.88 (MON:582) → **router contribution visible mid-training, esp. test**.
- Config: `configs/b200-deliver_rgbdel_P36_router.yaml`, output `outputs/ReliaDINO/b200_deliver_rgbdel_P36_router/` on B200 (MON:573). = P35 recipe + MODEL.ROUTER (P31 Per-class Reliability-anchored Router port) (MON:574).
- Mid-run ckpt snapshot backed up: `/nas_jm/drone_ckpts/B200_backup_20260715/` — `test_epoch58_57.14_top1`, `epoch52_67.74_top1` (MON:592). **Needs re-sync after run finishes** (MON:595).

### P34 = ReliaDINO (no router) — FINISHED 200ep (2026-07-13 15:34)
- Final best: **val 68.19 @ep120 / test 57.60 @ep140** — test-SOTA DGFusion 56.71 **+0.89**; val-SOTA CAFuser-CAA 68.6 **−0.41** (MON:539).
- Curve milestones: val 60.30@ep4 (MON:526); 65.86@ep10 (MON:528); 67.24@ep28 = first val>66.51 (MON:530); test 57.06@ep116 = first test>56.71 (MON:536); test 57.60@ep140 (MON:537).
- Ckpts (verified on NAS): `/nas_jm/drone_ckpts/P34_final_20260713/` — `epoch120_68.19_top1`, `test_epoch140_57.6_top1`, `train.log` (MON:539, MON:593). B200 output dir: `outputs/ReliaDINO/b200_deliver_rgbdel_P34_reliadino/DELIVER_ReliaDINO-ViTL16_idel/` (MON:513).
- Config: `configs/b200-deliver_rgbdel_P34_reliadino.yaml` (MON:513). ⚠ P34 trained **with PhysAug** and best-ckpt bookkeeping includes test-best → see fairness (d) and protocol (f).

### P35 = stripped recipe variant — DIED at ep120 (B200 reboot, not config fault), internal only
- Best: **val 67.61 @ep78 / test 56.14 @ep90** (MON:569; snapshot entries MON:567–568).
- Ckpts backed up: `/nas_jm/drone_ckpts/B200_backup_20260715/` — `test_epoch90_56.14_top1`, `epoch78_67.61_top1` (MON:592). Config `configs/b200-deliver_rgbdel_P35_paper.yaml` (MON:556).
- Use at most as ablation row; note incomplete (120/200 ep, killed by system event, MON:569).

### Predecessor (SAM2-lineage) internal baselines — for backbone-swap ablation
| model | backbone | val best | test best | src |
|---|---|---|---|---|
| P28-RBMA | SAM2 Hiera-B+ frozen | 63.40 | 55.27 | BBR:24 |
| P29 | SAM2 Hiera-B+ | 63.20 | 54.34 | BBR:25 |
| P31 (+router) | SAM2 Hiera-B+ | 63.20 | 54.85 | BBR:26 |
| P32 (+CoRB) | SAM2 Hiera-B+ | 64.12 @ep98 | 55.01 @ep158 | BBR:27; ckpt `/nas_jm/drone_ckpts/B200_backup_20260715/` MON:592 |
- SAM2 4 generations (P28→P32) total val gain = **+0.7**; one backbone swap (P32→P34) = **+4.1** (BBR:30).

---

## (a) MAIN COMPARISON TABLE DATA — DELIVER

⚠ PROTOCOL: DELIVER numbers form clusters (val vs test × backbone). Comparison table MUST use the **test (CLDE)** cluster; report val in a separate/dual column (CAFuser Tab.III format). Never mix with the B0-val cluster (e.g. CMNeXt 59.18, MemorySAM 65.38 = val). Sources: BENCH:321–338 (U1), BENCH:402–407 (U8).

### DELIVER test set, CLDE (= Camera+LiDAR+Depth+Event), mIoU-test — the headline table
| Method | Backbone | CLE test | CLDE test | src |
|---|---|---|---:|---|
| CMNeXt | MiT-B2 | 50.3 | 53.0 | BENCH:129 (DGFusion Tab.III), BENCH:295 |
| StitchFusion | MiT-B2 | 50.8 | 53.4 | BENCH:130, BENCH:296 |
| GeminiFusion | MiT-B2 | 50.5 | 54.5 | BENCH:131, BENCH:297 |
| CAFuser-CAA | Swin-T | 51.2 | 55.2 | BENCH:132, BENCH:298 |
| CAFuser (CA²) | Swin-T | 51.3 | 55.6 | BENCH:133, BENCH:299 |
| DGFusion | Swin-T | 51.6 | **56.7** (56.71 per convention memo) | BENCH:134 |
| **ReliaDINO no-router [P34]** | frozen DINOv3-L | TODO(CLE not run) | **57.60** (test-best) / 56.64 (legal val-selected, see (f)) | MON:539; ARCH:102 |
| **ReliaDINO full [P36]** | frozen DINOv3-L | TODO | **57.14** so far, final TODO | MON:582/584 |

⚠ SCOOP/caveat row (concurrent Sept-2025, must scope claims): **MM-SAM-adapter** (2509.10408), SAM ViT-L + ConvNeXt-S side-adapter, 2-modality: DELIVER test **RGB-D 57.35 / RGB-L 57.14 / RGB-E 55.70** (BENCH:376–378). Any "VFM multimodal SOTA" headline is false unless scoped (4-modality / supervision-minimal / etc.).

### DELIVER val split (B2/Swin "high" cluster) — dual column
| Method | Backbone | val mIoU | src |
|---|---|---:|---|
| CMNeXt | MiT-B2 | 66.30 (66.3) | BENCH:355, BENCH:295 |
| GeminiFusion | MiT-B2 | 66.9 | BENCH:354 |
| MAGIC | SegFormer-B2 | 67.66 | BENCH:352 |
| CAFuser-CA² | Swin-T | 67.8 | BENCH:351 |
| OmniSegmentor | DFormer-L | 68.0 | BENCH:349 |
| StitchFusion | MiT-B2 | 68.18 | BENCH:348 |
| **CAFuser-CAA** | Swin-T | **68.6** (= val-SOTA per our convention) | BENCH:347 |
| StitchFusion | Swin-Tiny-1k | 70.34 (⚠ split inferred, not from own caption) | BENCH:345, caveat BENCH:411 |
| **ReliaDINO no-router [P34]** | frozen DINOv3-L | **68.19** @ep120 | MON:539 |
| **ReliaDINO full [P36]** | frozen DINOv3-L | 67.74 so far, final TODO | MON:582 |

### MUSES (for planned-experiments / future-work section)
MUSES semantic seg **test** set (server-evaluated), CLRE, mIoU (BENCH:112–123, DGFusion Tab.II):
Mask2Former(C) 70.7 · SegFormer(C) 72.5 · OneFormer(C) 72.8 · CMNeXt 72.1 (CAFuser's own table says 72.4 — keep source-specific, BENCH:277/284,383) · GeminiFusion 75.3 (number exists only in CAFuser Tab.II, BENCH:383) · CAFuser-CAA 78.5 · CAFuser 78.2 · **DGFusion 79.5**. Plus MM-SAM-adapter test **RGB-L 81.07 / RGB-E 79.92** (BENCH:384). MUSES panoptic PQ test: DGFusion 61.03, CAFuser 59.70, CAFuser-CAA 59.38 (BENCH:108–110); DGFusion weakest conditions = Night 58.97 / Fog 58.86 PQ (BENCH:385).
⚠ Memory note "MUSES SOTA 79.72/79.49" conflicts with BENCH DGFusion 79.5 — resolve before citing (gap G-7).
Our MUSES status: first run launched 2026-07-15 00:50 KST on B200 (P34 recipe, [img,lidar,event], 300ep) — internal letterbox-1024 val 74.24@ep10, **NOT official protocol, not comparable to the 78–79.5 leaderboard** (MON:622–637). No test-server submission yet.

---

## (b) ABLATION TABLE DATA

### b-1. Router on/off (headline ablation, per user directive)
- **Router ON = P36 (full ReliaDINO)**: val 67.74@ep52 / test 57.14@ep58 (so far; final TODO) — MON:582/584.
- **Router OFF = P34 (no-router ReliaDINO)**: val 68.19@ep120 / test 57.60@ep140 (MON:539); legal val-selected: val 68.20 / test 56.64 @ep120 (ARCH:102).
- ⚠ CONFOUND: P36 = **P35 recipe** + router (MON:574), P34 = original recipe → P34 vs P36 is not a pure router toggle. Clean router pair = P35 (recipe, no router: 67.61/56.14, died ep120) vs P36 (recipe + router). Mid-training same-epoch deltas (router vs P34: test +2.0~+2.4 at ep30–46, MON:580–581) support router benefit on test. Final framing TODO after P36 finishes.

### b-2. Reliability-module toggles on P34 ep140 ckpt (D5 module ablation)
Verdicts (ARCH:52–62 v2, superseded in part by ARCH:100–102 v3 = full-res G0a/G0c):
| module | verdict (v3, full-res) | number | src |
|---|---|---|---|
| attention-logit bias λ₁ + consistency λ₂ | **DEAD** (also at full res ≈0) | \|Δ\|≤0.03, feat cos=1.0, 3 generations consistent; P32 CoRB significance p=4.5e-22 (harmful) | ARCH:57, ARCH:102; raw per-condition: AN34/module_ablation.md:7–47 (bias_off Δ ∈ {+0.01,−0.00,−0.00,+0.01,+0.01}; cons_off ∈ {−0.02,−0.00,−0.03,−0.01,−0.02}) |
| competence gate + calibration path | **KEEP — real contribution at full res**: **+0.26 test** (removal: 56.64 → 56.38) | ARCH:102 (G0c full-res reversal of v2's low-res "slightly harmful" verdict at ARCH:58–59) |
| veto floor | minor, only meaningful on top of gate | +0.02~+0.23 (low-res, n=40) | ARCH:60; raw AN34/module_ablation.md:10,19,28,37,46 |
| calibration LOSS (training-time) | unverified as performance claim (toggle only touches inference temperature path) — do NOT claim | indirect evidence = balanced AUROC (see (c)) | ARCH:61 |
- ⚠ Sign convention: AN34/module_ablation.md raw table header vs ARCH §3 read the sign oppositely; the reconciled paper-facts are the ARCH v3 (§8) full-res numbers above. Low-res n=40/condition rows are sub-1pt → per multi-seed policy do not build claims on them (ARCH:62).
- Raw per-condition base mIoU of the D5 subset (n=40/cond, low-res): cloud 45.76 / fog 42.39 / night 43.13 / rain 47.60 / sun 44.76 (AN34/module_ablation.md:4,13,22,31,40) — internal diagnostic scale only, not comparable to full-res 57.60.

### b-3. Backbone probe (controlled: frozen backbone + linear head only)
| probe (DELIVER test) | DINOv3 | SAM2 | Δ | src |
|---|---:|---:|---:|---|
| img mIoU | **40.51** | 28.90 | +11.6 | BBR:52 |
| depth mIoU | **35.48** | 25.34 | +10.1 | BBR:53 |
| TrafficLight IoU | 32.2 | 8.27 | 3.9× | BBR:54 |
- Feature mechanism (night, P31-SAM2 vs P34-DINOv3): per-modal effective rank 1.1–6.1 vs 10.7–19.6; FUSED eff-rank **1.26 vs 10.18**; cross-modal CKA 0.02–0.16 vs **0.80–0.91** (BBR:62–66).

### b-4. LoRA adaptation evidence (adapter on/off, P34 ep140, D3B)
Per-condition Δacc when adapter ON (AN34/modal_adaptation.md, n=40/cond): **lidar +0.157~+0.202** (cloud .2023:11, fog .1571:19, night .1743:27, rain .1758:35, sun .1953:43); **depth +0.110~+0.153**; event +0.020~+0.050; img +0.032~+0.064. Adaptation present in **5/5 conditions**; **dead adapters 0/48** (ARCH:50; raw `AN34/adapter_health.json`).

---

## (c) PER-CLASS / PER-CONDITION DATA (P34 ep140 test-best ckpt, full DELIVER test)

### Per-condition mIoU (AN34/per_domain_analysis.md:30)
cloud **55.64** / fog **56.32** / night **54.56** / rain **56.31** / sun **55.41** — per-domain spread **1.76** → gap vs val is NOT domain shift; it is per-class transfer (AN34/per_domain_analysis.md:32).

### Per-class × condition IoU (full table = AN34/per_domain_analysis.md:3–29; per-class copy below, cloud/fog/night/rain/sun)
Building 87.1/88.2/86.3/87.0/87.5 · Fence 45.5/43.4/37.0/52.4/53.3 · Other 3.9/5.6/2.8/2.8/4.2 · Pedestrian 76.6/78.1/71.6/73.5/70.5 · Pole 49.2/47.8/50.4/50.4/45.2 · RoadLine 79.7/79.6/77.0/77.7/77.2 · Road 97.5/97.4/96.8/97.2/97.1 · SideWalk 83.5/79.3/78.3/76.1/77.8 · Vegetation 84.4/82.7/81.6/84.1/81.5 · Cars 91.0/90.6/91.8/92.3/92.5 · Wall 8.7/3.7/7.5/8.1/7.4 · TrafficSign 47.1/47.0/47.9/42.7/56.2 · Sky 97.7/97.6/97.8/97.6/97.9 · Ground 14.9/5.3/9.9/8.6/7.6 · Bridge 0.0/0.5/0.0/0.0/0.0 · RailTrack 61.0/85.2/70.2/58.8/29.8 · GroundRail 76.6/78.8/79.1/75.8/74.9 · TrafficLight 40.1/40.3/38.5/36.4/27.6 · Static 36.7/38.2/34.9/42.4/43.4 · Dynamic 13.1/7.3/14.0/13.8/9.2 · Water 0.3/2.1/0.1/2.9/21.8 · Terrain 71.7/68.0/64.3/71.9/68.5 · TwoWheeler 55.4/63.9/52.3/73.3/72.1 · Bus 76.9/85.0/89.4/93.3/92.5 · Truck 92.4/92.3/84.5/88.7/89.5.
- Domain-invariant dead (max IoU<10): **Other (5.6), Wall (8.7), Bridge (0.5)** (AN34/per_domain_analysis.md:33).
- Domain-sensitive (spread>12): Fence ±16 (worst night 37), TrafficSign ±14 (rain 43), RailTrack ±55 (sun 30), TrafficLight ±13 (sun 28), Water ±22 (night 0), TwoWheeler ±21 (night 52), Bus ±16 (cloud 77) (AN34/per_domain_analysis.md:34).

### Class-level lineage gains (5-condition mean, BBR:89–95)
| class | P29 | P31 | P32 | P34 ep140 |
|---|---:|---:|---:|---:|
| Water | 0.0 | 0.0 | 0.2 | **5.4** (12.0 @ep40) |
| TrafficLight | 12.4 | 13.8 | 20.7 | **36.6** |
| Static | 29.4 | 29.9 | 23.5 | **39.1** |
| RailTrack | 47.9 | 58.4 | 54.0 | **61.0** |
| Pole | 45.0 | 46.8 | 45.6 | **48.6** |
- Residual dead classes (Bridge/Other/Wall) are also 0 for the DINOv3 probe AND official CMNeXt → dataset/benchmark ceiling, C4 audit (BBR:98; ARCH:88). **Exact official-CMNeXt per-class test numbers = TODO (source M0-a run; not in the files read here).**

### Reliability AUROC per modality [img, depth, event, lidar] (P34 ep140; AN34/module_diag.json `conditions/*/reliability_auroc`)
cloud [.848,.770,.834,.669] · fog [.826,.789,.685,.636] · **night [.851,.784,.869,.695]** (≈ the .85/.78/.87/.70 headline) · rain [.812,.753,.806,.650] · sun [.852,.790,.834,.671].
Generation contrast (night AUROC, BBR:74–79): P29 [.84,.63,.26,.38] · P31 [.43,.92,.55,.97] (calibration loss repaired geometry but sacrificed img) · P32 [.86,.70,.35,.34] · **P34 [.85,.78,.87,.70] — only P34 balances all 4 modalities without repair**.

### Drop-modality ΔmIoU [img, depth, event, lidar] (P34 ep140; AN34/module_diag.json `conditions/*/drop_modality_dmiou`)
cloud [2.79, 9.57, −0.02, 1.00] · fog [4.99, 11.43, −0.01, 0.63] · night [8.04, 10.62, −0.11, 0.89] · rain [7.63, 9.49, −0.24, 1.19] · sun [6.46, 9.48, −0.87, 0.11].
→ depth is the dominant complementary modality (~+9.5–11.4); **event contributes ≈0 in all 5 conditions** (known 4-generation issue, BBR:104).

---

## (d) FAIRNESS / SUPERVISION TABLE DATA (source: ARCH §5:73–76, DGFusion repo audit)

| method | extra supervision beyond seg labels | src |
|---|---|---|
| DGFusion | **GT log-depth supervision + condition meta-text (CLIP contrastive)** | ARCH:75 |
| CAFuser | condition labels + CLIP (condition meta-text pipeline) | ARCH:75/91; task directive |
| **ReliaDINO (ours)** | **none — segmentation labels only** (C3) | ARCH:86/104 |

- UNFAIR-OURS: PhysAug (competitors use standard aug only) → headline should be PhysAug-off run; and test-best ckpt selection → replaced by val-only selection (ARCH:74; see (f)).
- UNFAIR-THEIRS: DGFusion GT log-depth + meta-text (ARCH:75).
- DISCLOSE: backbone asymmetry — frozen DINOv3-L **349.9M total / 46.8M trainable (13.4%)** (ARCH:49) vs trainable Swin-T ~79M (ARCH:76). Defense: trainable 46.8M < CMNeXt-B2 total ~58M (ARCH:94); CMNeXt-B2 params 58.69–58.73M (BENCH:201, StitchFusion Tab.7).
- FAIR: no TTA, matched modalities, training budget ours 200ep vs their ≈400ep-equivalent (ARCH:76).
- Blocking-level check outstanding: eval resize-convention parity (**G0d**) — internal jitter 2pt > winning margin 0.89 (ARCH:76).

---

## (e) IMPLEMENTATION DETAILS (ARCH:35–49, MON:513/556/573)

- Backbone: frozen **DINOv3 ViT-L/16** (timm `vit_large_patch16_dinov3`, pretrained; MON:514), shared single encoder, 24 blocks.
- Per-modality LoRA on qkv, rank 8, independent adapters per modality × 24 blocks (`MultiModalLoRAQKV`, ARCH:37).
- Fusion: AuxDecoder×4 (per-modality 25-class predictions) → reliability signals → **2-layer cross-modal attention** (+λ₁B_cal+λ₂B_cons bias in P34; bias dead) → **calibrated competence gate + veto** weighted sum (ARCH:39–47). SimpleFPN stride16→{4,8,16,32} + FPNSegHead → (B,25,H,W) (ARCH:44–45).
- Router (P36 only): **Per-class Reliability-anchored Router (PRR)**, ported from P31 (MON:574). Architecture detail = fact_method.md scope.
- Params: total 349.9M / trainable **46.8M (13.4%)** (ARCH:49). Zero SAM2 dependency (ARCH:49).
- Training: 1024² crops, loss = OHEM CE + 0.5·aux_ce + 0.1·calib, AdamW lr 6e-4, **200 epochs**, batch 4, 4-GPU DDP (B200), eval every 2 epochs (ARCH:49; MON:513). ~8 min/epoch on B200 (MON:535). Trainer `train_reliadino.py` (MON:513).
- Dataset: DELIVER, 4 modalities [img, depth, event, lidar]; splits 3,983 train / 2,005 val / 1,897 test, 25 classes, 1042×1042 (BENCH:331).
- Configs: P34 `configs/b200-deliver_rgbdel_P34_reliadino.yaml` (MON:513) · P35 `configs/b200-deliver_rgbdel_P35_paper.yaml` (MON:556) · P36 `configs/b200-deliver_rgbdel_P36_router.yaml` (MON:573).
- Augmentation: P34 = PhysAug ON (ARCH:74). P35/P36 recipe = PhysAug-off + val-only selection + LoRA-norm regularization per design (ARCH:26) — **confirm from P36 yaml before stating in paper (gap G-6)**.

---

## (f) CHECKPOINT-SELECTION PROTOCOL (legal = val-only selection)

- Problem: P34 bookkeeping kept test-best checkpoints (test_epochN_*_top1) → selecting ep140 by test = illegal test peeking (ARCH:74).
- **Legal protocol: select by val only.** P34 legal checkpoint = **ep120 (val-best): val 68.20 / test 56.64** → vs DGFusion **+1.69 val / −0.07 test** (ARCH:102, G0a measurement of ep120 on test).
  - ⚠ rounding: monitor log prints val 68.19@ep120 (MON:539) vs ARCH:102 "68.20" — same checkpoint; pick one rendering and footnote (gap G-8).
- Both numbers can be reported: legal val-selected row (68.20/56.64) + oracle/test-best row (57.60@ep140) clearly labeled; the protocol itself is a contribution (C4, ARCH:104).
- P36: apply same protocol at ep200 — val-best so far = ep52 (67.74) whose test = 57.14 was measured on the same schedule; final legal pair TODO after completion.

---

## (g) MISSING EXPERIMENTS / GAPS (for the "planned/limitations" list and \todo markers)

1. **P36 final numbers** — training ends ~2026-07-15 12:20 KST; monitor-log has no entry past 00:20 (ep112). Re-read MON RUN-23 tail + re-sync `/nas_jm/drone_ckpts/B200_backup_20260715/` (MON:584, 595). B200 hard deadline 07-15 23:59 KST.
2. **MUSES official-protocol result** — only internal letterbox val 74.24@ep10 exists (RUN-24, MON:634, explicitly non-comparable MON:637); no full-res re-eval, no test-server submission (semantic/panoptic/AUPQ tracks, BENCH:405).
3. **MULTIAQUA** — no ReliaDINO run; only dataset facts (BENCH:392–394: CMNeXt-DH 93.58 val-day / 74.25 test-night) — planned condition-shift showcase.
4. **Multi-seed** — everything single-seed; policy = no sub-1pt claims (ARCH:94).
5. **PhysAug-off headline (T1)** — P35 (stripped+PhysAug-off) died at ep120 (67.61/56.14, MON:569); P36 carries the recipe but confound with router. A clean no-router+PhysAug-off completed run does not exist.
6. **PhysAug/val-selection settings of P35/P36 configs unverified** — inferred from design note (ARCH:26/74); confirm from yaml on B200 backup before paper claims.
7. **MUSES SOTA number discrepancy** — memory "79.72/79.49" vs BENCH DGFusion 79.5 (BENCH:123); resolve source before citing.
8. **68.19 vs 68.20 (P34 ep120 val)** rounding mismatch (MON:539 vs ARCH:102).
9. **G0d eval-resize parity check** not done — internal jitter 2pt > 0.89 margin (ARCH:76). Blocking-level for the +0.89 claim.
10. **Official CMNeXt per-class DELIVER-test numbers (M0-a audit)** — claim "Wall/Bridge/Water dead for official CMNeXt" (ARCH:88, BBR:98) but the raw per-class table was not in files read here; locate M0-a artifact before printing numbers.
11. **CLE (3-modality) column for ReliaDINO** — never evaluated; competitors' tables have it (BENCH:129–134).
12. **repo `experiments/log.md` has NO P34/P35/P36 entries** (latest = P33.1) — checkpoint/eval provenance for the paper relies on monitor-log + NAS backup only.
13. **MM-SAM-adapter scoping** — concurrent work beats us 2-modal on DELIVER test (57.35 RGB-D) and MUSES (81.07); claims must be scoped to 4-modality/supervision-minimal setting (BENCH:376–378).
14. **Per-condition DELIVER test breakdowns of competitors do not exist in print** (only val; DGFusion publishes aggregate test only, BENCH:398) — our per-condition test table (c) is first-of-kind but has no external comparison column.
