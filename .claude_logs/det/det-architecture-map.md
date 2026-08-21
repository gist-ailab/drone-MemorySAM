# Detection architecture map — how each model's detector is actually built

"Build a detector on the segmentation encoder" has been implemented **three
structurally different ways** across P29→P39. The difference is the **tap point**:
where the detector takes features out of the segmentation model. Everything else
(neck, head, NMS, how a seg mechanism reaches detection) follows from that choice.

Params measured on CPU from each det config (frozen trunk = DINOv3 ViT-L, which
never trains; trainable = LoRA + fusion + FPN + head).

## The three construction patterns

### A. SAM2 mid-level features + FCOS  (P29, P30)
`seg_model.extract_det_features()` returns three **heterogeneous** SAM2 Hiera maps
— `fpn0` 32ch/s4, `fpn1` 64ch/s8, `mem` 256ch/s16 — so the detector must run its
own **FPNNeck** to align channels before an FCOS head. NMS required.

### B. ReliaDINO pyramid + a head  (P34, P35, P36, P37, P37a, P37b)
`extract_det_pyramid()` returns a **uniform, already-fused** 4-level pyramid (all
256ch, s4/8/16/32). Cross-modal fusion happens inside the backbone, so the detector
does no modality fusion and needs **no neck**. Two head generations:
  - FCOS + NMS (P34/P35/P36)
  - **RF-DETR decoder, COCO-pretrained, one-to-one Hungarian → no NMS** (P37 family)

### C. The segmentation head *is* the detector  (P38, P39)
`extract_m2f_output()` returns the **M2F query outputs** directly. Mask2Former is
already a query-based one-to-one (Hungarian), NMS-free architecture, so there is no
separate det head at all — only a per-query **box head** was added. No NMS.

## Model table

| model | base segmentation model | tap point | neck | det head | NMS | total | trainable | det-head |
|---|---|---|---|---|---|---|---|---|
| P29-Det | SAM2 Hiera-B+ + LoRA + RBMA | `extract_det_features` (3 mixed maps) | **FPNNeck** | FCOS | yes | n/m | n/m | n/m |
| P30-Det | + SDC / reliability router | `extract_det_features` | **FPNNeck** | FCOS + object-query decoder | yes | n/m | n/m | n/m |
| P34-Det | ReliaDINO (attn-bias + consistency on) | `extract_det_pyramid` | — | FCOS | yes | 351.7M | 48.7M | 5.0M |
| P35-Det | ReliaDINO paper-freeze (bias/cons off) | `extract_det_pyramid` | — | FCOS | yes | 351.7M | 48.7M | 5.0M |
| P36-Det | + PerClassRouter | `extract_det_pyramid` + router→det seam | — | FCOS | yes | 351.9M | 48.9M | 5.0M |
| P37-Det | P34 backbone | `extract_det_pyramid` (stride-16 only) | — | **RF-DETR (COCO init)** | **no** | 356.6M | 53.5M | 9.9M |
| P37a-Det | + CEFR (2-pass feature blend) | `extract_det_pyramid` (CEFR applied) | — | RF-DETR | no | 357.0M | 53.9M | 9.9M |
| P37b-Det | + ClassToken | `extract_det_pyramid` + classtoken→det seam | — | RF-DETR | no | 359.7M | 56.6M | 9.9M |
| P38-Det | + M2F (Mask2Former-lite) | `extract_m2f_output` (queries) | — | **M2F queries + box head** | no | 352.1M | 49.0M | 5.4M |
| P39-Det | + Dual-Path Compete (V2 modal-token queries, V3 anchored) | `extract_m2f_output` | — | M2F queries + box head | no | 352.1M | 49.0M | 5.4M |

n/m = not measured here (the SAM2 path needs deps absent from this env).
Frozen DINOv3 trunk is 303.1M in every ReliaDINO row.

## What the numbers say

- **The head is a rounding error.** 5.0M (FCOS) / 9.9M (RF-DETR) / 5.4M (M2F) against
  ~352M total — 1.4–2.8%. Detection quality is decided by the shared backbone and by
  whether the head carries pretrained priors, not by head capacity.
- **Only ~14% of the model trains** (48.7M of 351.7M). The DINOv3 trunk is frozen
  throughout; LoRA + fusion + SimpleFPN + head are the learnable surface.
- **P37b costs +2.7M over P37a and returns nothing** — the class-token measured NO-OP
  in detection (agreement 1.000). Dead weight, quantified.
- **P37 family is the only one with a pretrained head** (RF-DETR COCO). P38/P39's M2F
  query head trains from scratch, so their comparison to P37 is not head-for-head.

## How a segmentation mechanism reaches detection (the recurring trap)

| where the mechanism acts | reaches detection? |
|---|---|
| modifies `fused` (feature level) — e.g. CEFR, P39-V1 | **automatically**, via the pyramid |
| modifies the seg **head logits** — router, class-token, m2f β | **no** — det never reads the seg head; needs an explicit seam |
| its own query branch — M2F, P39-V2/V3 | only if the det path calls it with the right inputs |

Every output-level mechanism therefore needed a hand-built seam (a 1×1 projection +
zero-init α residual into the pyramid). Measured verdict on those seams:

| mechanism | ΔmAP50 | agreement | verdict |
|---|---|---|---|
| P37a CEFR (feature level) | −0.0197 | 0.000 | **ACTIVE** |
| P36 router (output level, both seams) | 0.0000 | 1.000 | NO-OP |
| P37b class-token (output level) | 0.0000 | 1.000 | NO-OP |

Feature-level blends survive; output-level zero-init residuals stay at zero. This is
the detection-side confirmation of the failure key P39 was designed around, and it is
why P39 abandons passive residuals for competing paths.

Reproduce: `tools/det_analysis_pipeline.py` (see `tools/README_det_analysis.md`).
