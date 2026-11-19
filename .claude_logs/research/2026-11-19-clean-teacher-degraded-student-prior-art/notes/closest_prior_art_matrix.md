# Closest prior art to scheme S (clean frozen teacher + corrupted-input student, multi-sensor segmentation, frozen VFM + LoRA) — adversarial novelty check

Scope note. S = frozen DINOv3 shared by all sensors + sensor-specific LoRA; each batch used twice: (1) clean pass with supervised loss, (2) corrupted pass where each sensor independently (p=0.5) is zero-filled or hit by a sensor-specific synthetic degradation with a severity curriculum, trained with supervised loss + KL toward the logits of a frozen same-architecture clean-trained teacher that sees the clean input.
Axes: (A) clean-input frozen teacher + degraded/missing-input student with logit/feature distillation; (B) paired clean pass + corrupted pass in the same step; (C) multi-modal / multi-sensor; (D) semantic segmentation / dense prediction; (E) synthetic sensor-specific degradation, not only removal; (F) frozen foundation backbone + adapters.
Verification levels used below: **[FT]** = method read from full text this session; **[ABS]** = abstract / listing page only; **[SR]** = search-result summary only. Search date 2026-11-19. About 40 tool calls; the Semantic Scholar citation graphs were NOT traversed (see Gaps).

---

## Key Question 0 (objective): ranked table of the closest papers with the A–F match matrix

### Takeaway
No single paper found matches all six axes, but every axis and almost every pair/triple is occupied: FuseAdapt (NeurIPS 2026) occupies C+D+F with a full-modality teacher and the same independent Bernoulli-0.5 modality masking; RMMSS/AnySeg occupy A+C+D with exactly CE + KL from a frozen full-modality teacher; CRM occupies B+C+D; RobustSAM occupies A+D+E+F for a single camera. The only cell not found occupied is axis E inside a multi-sensor segmentation distillation scheme (sensor-specific degradation of the student input instead of removal), and its conjunction with the others.

### Ranked match matrix (closest first)

| # | Paper | arXiv id | Venue / year | A | B | C | D | E | F | Residual difference from S | Verif. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FuseAdapt: Parameter-Efficient Multimodal Fusion for Semantic Segmentation with Missing Modalities (Zhang, Tan) | 2610.21779 | NeurIPS 2026 (paper footer) | partial | partial | yes | yes | no | yes | Teacher is an EMA copy of the adapters (not a frozen clean-trained model) and always gets full input; main loss is latent prediction of the masked modality (cosine), logit KL appears only as an ablation control ("Output KD"); modalities are only masked (independent Bernoulli 0.5), never synthetically degraded; full and masked inputs are mixed per iteration (p=0.5), not both passes per batch; DINOv2, not DINOv3. | FT |
| 2 | RMMSS / "RobustSeg" (Tan, Zheng, Liu) | 2505.12861 | arXiv v2 carries an AAAI 2026 copyright line; a CVPR 2026 poster by the same authors exists under a different title (see KQ3) | yes | no | yes | yes | no | no | Its *baseline* ("Basic", L = CE + λ·KL to a frozen pretrained full-modality teacher, student gets randomly dropped modalities) is S minus degradation, minus the clean pass, minus the VFM; modality dropout only; SegFormer/CMNeXt trained end-to-end. | FT |
| 3 | AnySeg: Learning Robust Anymodal Segmentor with Unimodal and Cross-modal Distillation (Zheng et al.) | 2411.17141 | arXiv ("Preprint. Under review" in v2, May 2025; later venue not verified) | yes | no | yes | yes | no | no | Frozen PML-pretrained multimodal teacher sees the full batch, student sees "anymodal dropout" batch; prediction-level KL + multi-scale feature KL + cross-modal correspondence KL; removal only; SegFormer-B0 trained. | FT |
| 4 | CRM: Complementary Random Masking for RGB-Thermal Semantic Segmentation (Shin et al.) | 2303.17386 | ICRA 2024 | partial | yes | yes | yes | partial | no | Same-network self-distillation (no separate frozen teacher), L1 on class logits between the clean RGB-T prediction and masked-pair predictions; clean pass is supervised in the same step; corruption = random patch masking (not sensor-specific); two modalities; Swin fine-tuned. | FT (ar5iv) |
| 5 | RobustSAM: Segment Anything Robustly on Degraded Images (Chen et al.) | 2406.09627 | CVPR 2024 (Highlight) | yes | partial | no | yes | yes | yes | Single camera; clean image → frozen original SAM gives target mask-features/output-token, degraded image → SAM + small trainable modules, consistency (feature/token) + segmentation loss; 15 synthetic degradations + identity; feature/token consistency instead of logit KL; promptable not semantic segmentation. | FT (ar5iv) |
| 6 | md4all: Robust Monocular Depth Estimation under Challenging Conditions (Gasperini et al.) | 2308.09711 | ICCV 2023 | yes | no | no | yes (depth) | partial | no | Frozen clear-day baseline sees the original image; student sees a mix of original and translated (night/rain) images and "learns to follow B at the output level"; only a distillation loss (L1), no supervised term; GAN translation, not parametric sensor degradation; single camera. | FT (ar5iv) |
| 7 | Robo3D density-insensitive training (Kong et al.) | 2303.17597 | ICCV 2023 | partial | yes | no | yes (LiDAR seg) | partial | no | Teacher gets the full point cloud, student a randomly point-masked cloud; both trained (teacher not frozen); L = L_full + L_part + partial-to-full + full-to-partial consistency; LiDAR only. | FT (ar5iv) |
| 8 | Distilling Vision Transformers for Distortion-Robust Representation Learning (Alexis et al.) | 2604.22529 | arXiv, Apr 2026 | yes | no | no | no | partial | no | Frozen pretrained DINO ViT-B/16 teacher on clean images, student sees only distorted counterparts (pixel masking, Gaussian noise, blur); embedding/patch/attention distillation; classification only; one encoder per distortion. | FT (alphaXiv md) |
| 9 | M3L: Missing Modality Robustness in Semi-Supervised Multi-Modal Semantic Segmentation (Maheshwari, Liu, Kira) | 2304.10756 | arXiv 2023 (venue not listed on abs page) | partial | n/v | yes | yes | no | no | "Multi-modal Teacher for Masked Modality Learning": full-modality teacher supervises masked-modality segmentation (semi-supervised); teacher type/loss not verified from full text. | ABS + description in FuseAdapt related work |
| 10 | MMANet: Margin-aware Distillation and Modality-aware Regularization for Incomplete Multimodal Learning (Wei et al.) | 2304.08028 | CVPR 2023 | partial | no | yes | partial | no | no | Teacher transfers "comprehensive multimodal information" to a deployment network that handles incomplete inputs; margin-aware (uncertainty-weighted) distillation; classification + segmentation; removal only. | ABS |
| 11 | Compass: Degradation-Simulated Reciprocal Learning … Multimodal Crack Segmentation under Missing Modalities (Liu et al.) | 2608.03559 | ACM MM 2026 (per abs page) | partial | partial | yes | yes | partial | no | "Degradation Simulation Distillation": a degradation-simulation stream that "mimics more severe missing conditions" and reciprocal (two-way) distillation with the original stream; roles, losses and whether degradation goes beyond missingness not verified; RGB(+depth) crack segmentation; no VFM. | ABS |
| 12 | UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation (Zhao et al.) | 2509.16170 | NeurIPS 2025 | partial | n/v | yes | yes | no (n/v) | partial | Input-level modality reconstruction, feature-level contrastive learning with a lightweight adapter on a frozen encoder, output-level consistency across modality combinations; no clean-trained frozen teacher described; removal-centric. | SR |
| 13 | GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth (Jaganathan, Vela) | 2604.11585 | CVPR 2026 URVIS Workshop | no | no | yes | yes | partial | partial | Frozen RGB-D segmenter; small trainable prompt/recovery modules trained with segmentation loss only; handles "severe depth corruptions" (how they are simulated in training not verified); no teacher/distillation. | ABS |
| 14 | Large-Scale Domain Adaptation via Teacher-Student Learning (Li, Seltzer, Wang, Zhao, Gong) | 1708.05466 | arXiv 2017 (speech; venue not on abs page) | yes | no | no | no | partial | no | Source-domain (clean) teacher posteriors on parallel data replace labels for a target-domain (noisy) student; speech recognition; loss form not confirmed from abstract. | ABS |
| 15 | Adapting Models to Signal Degradation using Distillation ("cross-quality distillation", Su & Maji) | 1604.00433 | BMVC 2017 | yes (abs-level) | no | no | no | partial | no | Distillation between aligned high-quality and synthetically degraded versions (low resolution, non-localized, line drawings); image classification. | ABS |

Checked and judged NOT close on A/B (kept for completeness):
- MAGIC (2407.11344, ECCV 2024): no teacher or distillation; robustness from similarity-based modality ranking/selection [ABS].
- MM SAM-adapter (2509.10408, arXiv 2025): adapter injecting fused multimodal features into SAM; no distillation, no missing-modality/corruption training mentioned; whether SAM is frozen is not stated on the abstract page [ABS].
- Sensor-failure benchmark (2503.18445): benchmark only, no training method [FT].
- Condition Dropout (2607.20326, arXiv Jul 2026): frozen original encoders + trained copied encoders with zero-init feature injection; random complete / RGB-missing / depth-missing simulation; no teacher, no synthetic degradation mentioned [ABS].
- Purify-then-Align (2604.05584, CVPR 2026 workshop): purified multimodal "clean teacher" → single-modality students via diffusion KD; human sensing; "clean" refers to purified fusion, not clean-vs-corrupted input [ABS].
- "Towards robust multimodal 3D object detection via visual foundation models" (2609.23541): SAM-AD teacher distilled into a point-cloud network; whether teacher sees clean and student corrupted is not stated [ABS].
- Grace-BEV, "Can BEV Perception Gracefully Degrade under Sensor Failures?" (2605.30983): three-phase training with modality dropout; no clean-teacher distillation mentioned [ABS].
- RayD3D (2603.07493): frozen LiDAR teacher → camera student, trained on clean data, evaluated on corruptions; cross-modal KD, not clean→corrupted [SR].
- Diff-KD (2604.02061): teacher-student for collaborative perception, trained only on clean data [SR].

### Cited Findings
- FuseAdapt keeps a frozen VFM (DINOv2-B for DELIVER/MCubeS, DINOv2-S for MUSES), gives each modality private Q/V LoRA plus routed FFN low-rank residuals, and adds Predictive Modality Modeling: "PMM masks a target modality during training and predicts its representation from the visible modalities, supervised by a teacher target computed under full-modality input… Its EMA teacher supplies a temporally smoothed, stop-gradient target." — [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)
- FuseAdapt masking: "After the full-input warm-up, training mixes full and masked inputs. In the masked branch, modalities are independently retained with probability 0.5, conditioned on a nonempty visible subset"; appendix: "with probability 0.5 we retain all modalities; otherwise we draw independent Bernoulli-0.5 visibility masks… The teacher always receives the full input." — [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)
- FuseAdapt ablates the exact logit-distillation variant: "Output KD matches partial-input student logits to full-context EMA teacher logits through pixelwise KL divergence"; on MCubeS missing-modality mean mIoU, output-logit KD = 49.84, fused-feature KD = 50.57, PMM = 52.89. — [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)
- FuseAdapt itself disclaims the mechanism as new: "Masking, EMA, and teacher–student optimization are established mechanisms; the prediction target and modality conditioning distinguish our objective." It describes CRM as combining "complementary RGB-thermal masking with clean-to-masked self-distillation" and M3L as using "a full-modality teacher to supervise masked-modality segmentation". — [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)
- FuseAdapt's training augmentations include "random resizing (0.5–2.0), horizontal flipping, color jitter, blur, and random cropping" (standard augmentation, not sensor-specific degradation of a student branch). — [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)
- RMMSS "Preliminary": "The initial self-distillation and modality masking framework typically consists of a teacher model pretrained on full-modality data and a student model trained with randomly dropped modalities… We construct the basic loss function for the student model… L_origin = L_CE + λ L_KL." — [arXiv 2505.12861](https://arxiv.org/abs/2505.12861)
- RMMSS ablation Table 4 has a "Basic" row (CE + KL self-distillation with modality masking only): EMM 43.45 / RMM 51.69 / Full 57.50; + HPDM 47.45 / 53.20 / 57.46; + FSM 47.94 / 53.63 / 59.89. — [arXiv 2505.12861](https://arxiv.org/abs/2505.12861)
- RMMSS related work states the general idea as known: "In self-distillation, perturbations such as modality masking and noisy inputs are introduced during student training, while a full-modality teacher model is used to guide the student toward correct learning (Zhou et al. 2023; Shin et al. 2024; Hou et al. 2024)." — [arXiv 2505.12861](https://arxiv.org/abs/2505.12861)
- AnySeg: "the multimodal teacher is first trained using PML, then frozen for student distillation"; "the input multimodal mini-batch r, d, l, e undergoes random masking to generate an anymodal batch, where modality data is randomly dropped from the batch, ensuring that at least one modality is retained"; losses L_total = L_sup + λ L_mad (prediction-level KL to teacher) + α L_umd + β L_cmd; backbone SegFormer-B0; datasets DELIVER and MUSES. — [arXiv 2411.17141](https://arxiv.org/abs/2411.17141)
- CRM: one network predicts from clean and masked RGB-T pairs; self-distillation terms are L1 distances between class logits of the clean pair and of complementary-masked / singly-masked inputs; L_total = L_MWS + L_SDC + L_SDN with a supervised term on the clean pair; Swin backbones initialised from ImageNet-1K. — [ar5iv 2303.17386](https://ar5iv.labs.arxiv.org/html/2303.17386); abstract: "self-distillation loss between clean and masked input modalities" — [arXiv 2303.17386](https://arxiv.org/abs/2303.17386)
- RobustSAM: "we keep the pre-trained SAM parameters frozen, focusing only on optimizing the proposed modules for robustness"; training applies "a degradation augmentation to a clear input image and then feeding the resulting degraded image into RobustSAM" while "the original clear image is processed by standard SAM to extract clear versions" of features/token used as consistency targets (L_MFC, L_TC) plus a segmentation loss; 15 synthetic degradation types + identity. — [ar5iv 2406.09627](https://ar5iv.labs.arxiv.org/html/2406.09627)
- md4all-DD: a baseline trained on clear-day images is frozen and sees only the original image; the student "is fed m_i, i.e., the same mix of e_i and h_i^c" and "learns to follow B at the output level, even when fed the problematic h_i^c"; translated fraction x = |C|/(|C|+1) (66% nuScenes, 50% RobotCar). — [ar5iv 2308.09711](https://ar5iv.labs.arxiv.org/html/2308.09711)
- Robo3D: the sparser input "is generated by randomly masking the points from the original point cloud with a ratio β"; objective L = L_full + L_part + α1·L_part2full + α2·L_full2part; "We initialize both teacher and student networks with the same baseline model." — [ar5iv 2303.17597](https://ar5iv.labs.arxiv.org/html/2303.17597)
- 2604.22529: frozen pretrained DINO ViT-B/16 teacher processes clean images while the student "observes only their distorted counterparts" (pixel masking, Gaussian noise, Gaussian blur at two severities); classification benchmarks only. — [alphaXiv 2604.22529](https://www.alphaxiv.org/abs/2604.22529)
- Sensor-failure benchmark defines EMM / RMM / NM; NM = additive Gaussian noise + salt-and-pepper noise on all modalities (no Gaussian noise for events), three severities; models are "Trained from complete multi-modal data"; modality failure is modelled as independent Bernoulli(p). It proposes no training method. — [arXiv 2503.18445](https://arxiv.org/abs/2503.18445)
- MMANet (CVPR 2023): deployment network + teacher network + regularization network; margin-aware distillation from a teacher carrying "comprehensive multimodal information". — [arXiv 2304.08028](https://arxiv.org/abs/2304.08028)
- M3L = "Multi-modal Teacher for Masked Modality Learning", semi-supervised multi-modal segmentation. — [arXiv 2304.10756](https://arxiv.org/abs/2304.10756)
- Compass (ACM MM 2026): Degradation Simulation Distillation "constructs a degradation simulation stream that mimics more severe missing conditions and performs reciprocal distillation with the original stream". — [arXiv 2608.03559](https://arxiv.org/abs/2608.03559)
- UniMRSeg (NeurIPS 2025): hierarchical compensation at input (modality reconstruction), feature (contrastive + lightweight reverse-attention adapter on a frozen encoder) and output (consistency) levels. — [arXiv 2509.16170](https://arxiv.org/html/2509.16170v1); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/119265)
- GeomPrompt adapts "the fourth channel of a frozen RGB-D semantic segmentation model", trained with downstream segmentation supervision only; CVPR 2026 URVIS Workshop. — [arXiv 2604.11585](https://arxiv.org/abs/2604.11585)

### Inferences
- Ranking rationale: FuseAdapt is #1 because it shares S's architecture family (frozen DINO + per-sensor LoRA), task, datasets, full-input teacher and the Bernoulli-0.5 per-sensor masking, and it even reports the logit-KL variant as a weaker control. RMMSS/AnySeg are #2/#3 because their loss on the student is literally CE + KL to a frozen full-modality teacher.
- S's distinguishing content relative to the table is concentrated in two places: axis E (sensor-specific degradation operators with a severity curriculum inside the student branch) and the exact pairing of a supervised clean pass with the distilled corrupted pass for the trainable model (axis B) while the teacher is a separately trained frozen clean model.
- A hostile reviewer can assemble S as "RMMSS Basic (CE+KL, frozen full-modality teacher) + RobustSAM-style degradation augmentation + FuseAdapt backbone", each with a citation.

### Gaps
- Citation graphs (Semantic Scholar) of AnySeg, RMMSS, MAGIC and the sensor-failure benchmark were not traversed; coverage of 2026 citing papers relies on two alphaXiv topical searches.
- Rows 9–15 are abstract- or search-level only; their A/B cells could move by one step after a full-text read.
- AnySeg's final venue and whether RMMSS was accepted at AAAI 2026 were not confirmed (only the copyright line in the arXiv v2 PDF was seen).

---

## Key Question 1: Any paper (any task) doing exactly "frozen clean teacher sees clean input, student sees corrupted version, KD loss", and what is it called?

### Takeaway
Yes — this exact pattern is old and recurs under several names: "teacher-student (T/S) learning" for noise-robust speech (2017), "cross-quality distillation" / "adapting models to signal degradation using distillation" (2016–2017), md4all-DD in depth estimation (ICCV 2023), RobustSAM's clean-to-degraded consistency (CVPR 2024), and a 2026 DINO "distortion-robust" distillation paper. The professor's recollection is justified for axis A.

### Cited Findings
- Speech: "posterior probabilities generated by the source-domain model can be used in lieu of labels to train the target-domain model" using unlabeled parallel data (source/target pairs), evaluated on adapting a clean model to noisy speech. — [arXiv 1708.05466](https://arxiv.org/abs/1708.05466)
- Su & Maji, "Adapting Models to Signal Degradation using Distillation" (BMVC 2017; first posted 2016): distillation across aligned high-quality / degraded data (low resolution, non-localized objects, line drawings), with aligned data generated synthetically. — [arXiv 1604.00433](https://arxiv.org/abs/1604.00433)
- md4all-DD: frozen clean-condition teacher on the original image, student on translated adverse-condition image, output-level distillation. — [ar5iv 2308.09711](https://ar5iv.labs.arxiv.org/html/2308.09711)
- RobustSAM: frozen SAM on the clear image supplies targets for the degraded-image branch; 15 synthetic degradations. — [ar5iv 2406.09627](https://ar5iv.labs.arxiv.org/html/2406.09627)
- 2604.22529: "a pretrained teacher processes clean images while the student model observes only their distorted counterparts". — [arXiv 2604.22529](https://arxiv.org/pdf/2604.22529)
- Related same-network variant: AugMix uses "a Jensen-Shannon Divergence consistency loss" between predictions on the clean image and augmented versions to improve corruption robustness. — [AugMix, arXiv 1912.02781](https://alphaxiv.org/abs/1912.02781)
- A web search for clean-teacher/corrupted-student KD also returned Arani et al. (noisy collaboration in KD; main noise reported on teacher labels) and adversarially robust distillation (ARD, arXiv 1905.09747), which are neighbouring but not the same setup. — [Arani et al. PDF](https://navinfo.eu/wp-content/uploads/2022/08/improving-generalization-and-robustness-with-noisy-collaboration-in-knowledge-distillation.pdf); [ARD, arXiv 1905.09747](https://arxiv.org/pdf/1905.09747.pdf)

### Inferences
- Recognisable names a reviewer would use: "teacher-student learning with parallel clean/noisy data", "cross-quality distillation", "clean-to-degraded (clean-to-masked) self-distillation", "degradation-robust / distortion-robust distillation", "consistency regularisation with a clean anchor".
- The "3–5 most cited" ranking by citation count could not be established (no citation counts retrieved). By venue and age, the best-known verified instances are Li et al. 2017 (speech T/S), Su & Maji 2017, md4all (ICCV 2023), RobustSAM (CVPR 2024 Highlight), and AugMix for the same-network consistency form.

### Gaps
- Citation counts were not retrieved, so "most cited" is not established.
- Candidates recalled from memory but NOT re-verified this session (ids from memory; treat as leads only): Noisy Student (arXiv 1911.04252; student noised, teacher un-noised), FixMatch (2001.07685; weak-view target, strong-view student), generalized distillation / learning using privileged information (Lopez-Paz et al., 1511.03643), Depth Anything (2401.10891; strongly perturbed student vs. clean teacher pseudo-labels), UDA self-training such as DAFormer (2111.14887). These would further weaken an axis-A/B novelty claim if confirmed.

---

## Key Question 2: Multi-modal papers combining modality dropout / missing-modality training with distillation from a full-modality teacher AND additionally corrupting (not just dropping) modalities

### Takeaway
Full-modality-teacher + modality-dropout distillation is standard in multi-modal segmentation (AnySeg, RMMSS, M3L, MMANet, FuseAdapt, CRM). I found no verified multi-modal paper that adds sensor-specific synthetic degradation of the student input to that recipe; the nearest are CRM (patch masking), Compass (a "degradation simulation stream", abstract-level, appears to simulate heavier missingness) and RMMSS's own related-work sentence that lists "noisy inputs" as a known perturbation.

### Cited Findings
- AnySeg, RMMSS, FuseAdapt: student perturbation is modality removal only (quotes in Key Question 0). — [2411.17141](https://arxiv.org/abs/2411.17141); [2505.12861](https://arxiv.org/abs/2505.12861); [2610.21779](https://arxiv.org/abs/2610.21779)
- RMMSS cites prior self-distillation where "perturbations such as modality masking and noisy inputs are introduced during student training, while a full-modality teacher model is used to guide the student" (citing Zhou et al. 2023; Shin et al. 2024; Hou et al. 2024). — [arXiv 2505.12861](https://arxiv.org/abs/2505.12861)
- CRM uses random complementary patch masking of RGB and thermal plus clean-to-masked logit self-distillation. — [ar5iv 2303.17386](https://ar5iv.labs.arxiv.org/html/2303.17386)
- Compass builds "a degradation simulation stream that mimics more severe missing conditions" with reciprocal distillation (RGB-depth crack segmentation, evaluated at 90% depth missing). — [arXiv 2608.03559](https://arxiv.org/abs/2608.03559)
- FuseAdapt notes that a full-context teacher is "not a guarantee of superior predictions for every sample, especially when a present sensor is corrupted", i.e. it discusses corruption but does not train with it. — [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)
- Driving / BEV fusion results found: Grace-BEV (modality dropout, three-phase training), SB-BEVFusion (fusion module for missing/corrupted sensor), RayD3D (frozen LiDAR teacher → camera student, trained on clean data), Diff-KD (trained only on clean data); none was shown to use a clean-teacher / corrupted-student scheme. — [2605.30983](https://arxiv.org/abs/2605.30983); [2605.11799](https://www.alphaxiv.org/abs/2605.11799); [2603.07493](https://arxiv.org/html/2603.07493); [2604.02061](https://arxiv.org/pdf/2604.02061)
- Medical multi-modal: several 2025–2026 missing-modality self-distillation papers exist (e.g. CCSD, 2511.14599) and one corruption-aware fusion paper ("When Fusion Fails", 2609.10261); only titles/abstract snippets were seen. — [2511.14599](https://www.alphaxiv.org/abs/2511.14599); [2609.10261](https://www.alphaxiv.org/abs/2609.10261)
- Audio-visual: OP-CAD "On-Policy Clean-Audio Distillation for Robust Audio-Visual Reasoning" (2609.39150) surfaced by title only (clean-audio distillation for noisy input in omni-modal LLMs). — [2609.39150](https://www.alphaxiv.org/abs/2609.39150)

### Inferences
- The combination "drop OR sensor-specific degrade, per sensor, with distillation from a clean full-modality teacher" was not found in multi-sensor segmentation; this is the most defensible residual of S, but it is a combination of two well-known ingredients (modality-dropout distillation; degradation augmentation with a clean anchor as in RobustSAM/md4all).
- Because RMMSS already names "noisy inputs" as a standard student perturbation, a reviewer can argue that even the idea level of E is anticipated; what is missing in prior work is an actual instantiation with sensor-specific operators and a curriculum.

### Gaps
- The papers RMMSS cites for "noisy inputs" (Zhou et al. 2023; Shin et al. 2024 = CRM; Hou et al. 2024 = SDSTrack) were not all read; SDSTrack (multi-modal tracking, self-distillation, adapters) is a likely A/B/C/F-relevant precedent and was not verified.
- OP-CAD, CCSD, "When Fusion Fails", SB-BEVFusion: abstract/title level only. MetaBEV, RoboFusion, UniBEV and other sensor-failure fusion detectors were not checked.
- RGB-D / RGB-T segmentation with noisy modality beyond CRM, GeomPrompt, Condition Dropout and RTFDNet (title only, 2603.09149) was not searched exhaustively.

---

## Key Question 3: Verification of the suspected candidates (and corrections)

### Takeaway
All listed arXiv ids resolve to real papers and the descriptions are broadly right, with four corrections: (1) "RobustSeg" is the earlier name of the same arXiv entry 2505.12861 (v1 title) rather than a separately verified follow-up, and a CVPR 2026 poster by the same authors exists under another title; (2) FuseAdapt uses DINOv2 with an EMA teacher of the adapters and latent prediction, not a frozen teacher with logit KD (logit KD is only its ablation control); (3) MAGIC and MM SAM-adapter have no teacher/distillation at all; (4) the sensor-failure benchmark's "noisy modality" is Gaussian + salt-and-pepper noise, not sensor-specific degradation, and it proposes no training method.

### Cited Findings
- AnySeg (2411.17141): exists; "Learning Robust Anymodal Segmentor with Unimodal and Cross-modal Distillation"; frozen PML teacher, anymodal dropout student, feature + prediction KL; SegFormer; v2 header "Preprint. Under review" (15 May 2025). — [arXiv 2411.17141](https://arxiv.org/abs/2411.17141)
- RMMSS (2505.12861): exists; v2 title "RMMSS: Towards Advanced Robust Multi-Modal Semantic Segmentation with Hybrid Prototype Distillation and Feature Selection"; two-stage (robust teacher via HPDM, then FSM-fused dual-teacher distillation); "Basic" CE+KL row verified in Table 4; v2 PDF carries "Copyright © 2026, Association for the Advancement of Artificial Intelligence". — [arXiv 2505.12861](https://arxiv.org/abs/2505.12861)
- RobustSeg: arXiv 2505.12861 v1 was titled "Robust Multimodal Segmentation with Representation Regularization and Hybrid Prototype Distillation" and is associated with the name RobustSeg (code github.com/RobustSeg/RobustSeg per search summary); a CVPR 2026 poster "Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation" by Jiaqi Tan, Xu Zheng, Yang Liu is listed. — [arXiv 2505.12861v1](https://arxiv.org/html/2505.12861v1); [CVPR 2026 poster 39865](https://cvpr.thecvf.com/virtual/2026/poster/39865)
- FuseAdapt (2610.21779): exists; NeurIPS 2026 footer; NUS (Xin Zhang, Robby T. Tan); frozen DINOv2 + modality-private Q/V LoRA + routed FFN low-rank residuals + PMM with EMA full-input teacher; DELIVER / MUSES / MCubeS. — [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)
- MMANet (2304.08028): exists; CVPR 2023; margin-aware distillation + modality-aware regularization for incomplete multimodal learning. — [arXiv 2304.08028](https://arxiv.org/abs/2304.08028)
- MAGIC (2407.11344): exists; ECCV 2024; "Centering the Value of Every Modality: Towards Efficient and Resilient Modality-agnostic Semantic Segmentation"; aggregation + arbitrary-modal selection modules; no teacher/distillation on the abstract page. — [arXiv 2407.11344](https://arxiv.org/abs/2407.11344)
- GeomPrompt: exists as arXiv 2604.11585, CVPR 2026 URVIS Workshop; frozen RGB-D segmenter + prompt modules; no distillation. — [arXiv 2604.11585](https://arxiv.org/abs/2604.11585)
- UniMRSeg: exists as arXiv 2509.16170, NeurIPS 2025. — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/119265)
- CRM: exists as arXiv 2303.17386, ICRA 2024; same-network clean-to-masked logit self-distillation (L1). — [ar5iv 2303.17386](https://ar5iv.labs.arxiv.org/html/2303.17386)
- MM SAM-adapter (2509.10408): exists; Univ. of Bologna; adapter injecting fused multimodal features into SAM; DeLiVER, FMB, MUSES; no distillation or missing-modality training on the abstract page. — [arXiv 2509.10408](https://arxiv.org/abs/2509.10408)
- Sensor-failure benchmark (2503.18445): exists; evaluates CMNeXt, GeminiFusion, MAGIC, MAGIC++, StitchFusion (MiT-B2) on DELIVER under EMM / RMM / NM; "This work provides the first dedicated benchmark for MMSS robustness". — [arXiv 2503.18445](https://arxiv.org/abs/2503.18445)

### Inferences
- The claim "RobustSeg is a CVPR 2026 follow-up of RMMSS with a basic distillation (CE+KL) baseline row" should be reworded: the CE+KL "Basic" row is verified in arXiv 2505.12861 v2 (RMMSS); the CVPR 2026 paper is by the same authors and, from titles, looks like a version of the same line of work, but its text and tables were not read.
- The AAAI copyright line in RMMSS v2 and the CVPR 2026 poster listing are in tension (the same work would not appear at both); the relation between the two versions is unresolved.

### Gaps
- CVPR 2026 RobustSeg paper text (and whether it keeps a CE+KL baseline row) not read.
- MM SAM-adapter: whether SAM is frozen was not verified from full text. UniMRSeg, MMANet, GeomPrompt: abstract/search level only.

---

## Key Question 4: 2025–2026 work on robust multi-modal segmentation with foundation models that may have been missed

### Takeaway
The most important 2026 item is FuseAdapt (already suspected); additional 2026 entries worth citing or checking are Condition Dropout (RGB-D, frozen encoders + copied trainable encoders), GeomPrompt, Compass (degradation-simulation distillation), 2604.22529 (clean-teacher / distorted-student DINO distillation), RobustMedSAM, and SAM-based multi-modal adaptation papers (MLE-SAM, MM SAM-adapter, SegRGB-X, SpectraDINO). None of them was found to implement S in full.

### Cited Findings
- MLE-SAM, "Customize Segment Anything Model for Multi-Modal Semantic Segmentation with Mixture of LoRA Experts" (2412.04220, HKUST(GZ)); FuseAdapt describes it as combining "modality-specific LoRA experts and routing with multi-scale feature/head fusion". — [alphaXiv 2412.04220](https://www.alphaxiv.org/abs/2412.04220); [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)
- "Toward Reliable RGB-D Semantic Segmentation: Handling Missing Modalities via Condition Dropout" (2607.20326, Jul 2026): second-stage training that "randomly simulates complete, RGB-missing, and depth-missing inputs" with frozen original encoders and zero-initialised feature injection. — [arXiv 2607.20326](https://arxiv.org/abs/2607.20326)
- RobustMedSAM (2604.09814), SegRGB-X (2603.28023), SpectraDINO (2605.02258), "Robust Cross-Modal Foundation Model Perception for Underwater Robots under Degraded Visual Conditions" (2608.19710), UMSS (2607.12372), SGMA (2603.02505), "Robust Multispectral Semantic Segmentation under Missing or Full Modalities via Structured Latent Projection" (2604.15856) surfaced by title/abstract snippet only. — [alphaXiv 2604.09814](https://www.alphaxiv.org/abs/2604.09814); [2603.28023](https://www.alphaxiv.org/abs/2603.28023); [2605.02258](https://www.alphaxiv.org/abs/2605.02258); [2608.19710](https://www.alphaxiv.org/abs/2608.19710); [2607.12372](https://www.alphaxiv.org/abs/2607.12372); [2603.02505](https://www.alphaxiv.org/abs/2603.02505); [2604.15856](https://www.alphaxiv.org/abs/2604.15856)
- FuseAdapt cites further 2025–2026 MMSS work not examined here: "Zheng et al., 2025" (modality regularization; reported DELIVER missing-modality mean 48.03 with MiT-B2), CHARM (Wen et al., 2026), M4-SAM (Liu et al., 2026). — [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)
- FuseAdapt reports DELIVER missing-modality mean mIoU 56.22 (ViT-B, 14.01M trainable) vs 48.03 for the prior best, and MUSES full-modality 77.28 / 80.16 mIoU with DINOv2-S / DINOv2-B (best-validation). — [arXiv 2610.21779](https://arxiv.org/abs/2610.21779)

### Inferences
- FuseAdapt pre-empts the framing "frozen VFM shared by all sensors + sensor-specific LoRA + full-input teacher for missing-modality robustness" on the same benchmarks; S must be positioned against it explicitly, and S's robustness numbers will be compared with its DELIVER/MUSES missing-modality tables.
- DINOv3 vs DINOv2 is a backbone swap and will not be accepted as a novelty axis.

### Gaps
- No 2026 paper was found that uses DINOv3 for multi-sensor segmentation with sensor-failure training, but the search used only two topical queries; absence is weak evidence.
- RobustMedSAM, SegRGB-X, SpectraDINO, CHARM, M4-SAM, "Zheng et al. 2025" were not read.

---

## Calibrated verdict (per the objective)

### Takeaway
"Degrading the input and distilling from a clean teacher" is not first proposed in S; it is a long-standing pattern, and in multi-sensor segmentation the CE + KL-to-frozen-full-modality-teacher recipe is described by RMMSS as the typical baseline. What remains unoccupied in the evidence collected is narrow: sensor-specific synthetic degradation (with severity curriculum) as the student perturbation inside that recipe, on a frozen VFM + per-sensor LoRA, with a supervised clean pass in the same step.

### Cited Findings
- Clearly not novel, single axis A: 1708.05466 (speech T/S), 1604.00433 (signal degradation distillation), md4all 2308.09711, RobustSAM 2406.09627, 2604.22529. — [1708.05466](https://arxiv.org/abs/1708.05466); [1604.00433](https://arxiv.org/abs/1604.00433); [2308.09711](https://ar5iv.labs.arxiv.org/html/2308.09711); [2406.09627](https://ar5iv.labs.arxiv.org/html/2406.09627); [2604.22529](https://arxiv.org/pdf/2604.22529)
- Clearly not novel, single axis B (clean + corrupted pass in one step with a consistency term): AugMix (same network), CRM, Robo3D. — [1912.02781](https://alphaxiv.org/abs/1912.02781); [2303.17386](https://ar5iv.labs.arxiv.org/html/2303.17386); [2303.17597](https://ar5iv.labs.arxiv.org/html/2303.17597)
- Clearly not novel, A+C+D (frozen full-modality teacher, CE+KL, multi-sensor segmentation): AnySeg, RMMSS "Basic". — [2411.17141](https://arxiv.org/abs/2411.17141); [2505.12861](https://arxiv.org/abs/2505.12861)
- Clearly not novel, B+C+D: CRM. — [2303.17386](https://ar5iv.labs.arxiv.org/html/2303.17386)
- Clearly not novel, C+D+F with full-input teacher and independent Bernoulli-0.5 modality masking, no inference overhead: FuseAdapt. — [2610.21779](https://arxiv.org/abs/2610.21779)
- Clearly not novel, A+D+E+F for a single camera: RobustSAM. — [2406.09627](https://ar5iv.labs.arxiv.org/html/2406.09627)
- Not novel as an evaluation idea: noisy-modality and Bernoulli sensor-failure protocols. — [2503.18445](https://arxiv.org/abs/2503.18445)

### Inferences
- Unoccupied in what was found: (i) axis E combined with A+C+D — a per-sensor "remove OR sensor-specific degrade" student input (camera blur/exposure, depth holes, LiDAR beam drop, event down-sampling) with a severity curriculum, distilled from a clean frozen teacher; (ii) the six-axis conjunction A∧B∧C∧D∧E∧F.
- Defensible wording: "to our knowledge, the first to combine sensor-specific degradation simulation with clean-teacher logit distillation for multi-sensor segmentation on a frozen foundation backbone". Not defensible: "we propose degradation + clean-teacher distillation", "first self-distillation for robust multi-sensor segmentation", "first frozen-VFM + LoRA multi-sensor segmentor with a teacher".
- Expected reviewer attack: "S = RMMSS Basic + RobustSAM-style degradation augmentation + FuseAdapt-style backbone; where is the contribution beyond the combination?" The paper needs head-to-head rows against (a) CE+KL with dropout only, (b) degradation augmentation without the teacher, (c) FuseAdapt/PMM, to show the combination is not additive common sense.
- Confidence in "unoccupied": moderate at best. The search was about 40 calls without citation-graph traversal; SDSTrack and the driving-fusion robustness literature (MetaBEV, RoboFusion, UniBEV) were not checked.

### Gaps
- A positive proof of absence is impossible; in particular 2026 papers citing AnySeg/RMMSS/the benchmark were not enumerated.

---

## Search phrases a reviewer would use to find the closest papers

### Takeaway
The closest papers are reached within one or two queries using "missing modality" + "self-distillation" + "semantic segmentation" and "degraded images" + "consistency" + "foundation model".

### Cited Findings
- Queries that actually surfaced the closest papers in this session: "missing modality multi-modal semantic segmentation distillation sensor failure robust corruption" (returned RMMSS, the benchmark, FuseAdapt, Compass, 2604.22529) and "DINOv2 LoRA multi-modal segmentation missing modality foundation model DELIVER MUSES" (returned FuseAdapt, MLE-SAM, Condition Dropout, GeomPrompt, MM SAM-adapter). — [alphaXiv search results, e.g. FuseAdapt](https://www.alphaxiv.org/abs/2610.21779)
- "RobustSeg multi-modal semantic segmentation CVPR 2026 distillation missing modality" returns the CVPR 2026 poster and arXiv 2505.12861. — [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/39865)

### Inferences
- Suggested exact phrases: "anymodal segmentation distillation"; "modality-incomplete semantic segmentation full-modality teacher"; "self-distillation modality masking multi-modal segmentation"; "clean-to-masked self-distillation RGB-thermal"; "robust multi-modal semantic segmentation hybrid prototype distillation"; "parameter-efficient multimodal fusion frozen DINOv2 LoRA missing modalities"; "segment anything degraded images consistency loss"; "robust depth estimation adverse conditions distillation frozen teacher"; "distortion-robust distillation clean teacher distorted student"; "teacher-student learning parallel clean noisy data"; "cross-quality distillation"; "sensor failure benchmark multi-modal semantic segmentation noisy modality"; "density-insensitive training partial-to-full consistency LiDAR".

### Gaps
- Phrases were not tested on Google Scholar / Semantic Scholar / OpenReview; ranking behaviour there is unknown.
