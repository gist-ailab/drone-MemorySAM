---
title: VFM Multimodal Dense-Prediction Landscape 2024→2026-07 — Synthesis & Taxonomy (Track 1)
tags: [related-work, synthesis, vfm, sam2, sam3, dinov2, dinov3, multimodal-segmentation, deliver, taxonomy, rbma]
created: 2026-07-02
source: parallel deep-research Track 1 (sources/07_parallel_research_prompts_2026-07-02.md) + adversarial verification (skeptic1/skeptic2)
status: verified-draft
---

# VFM 멀티모달 dense-prediction 지형 종합 (2024→2026-07)

Track 1 synthesis note. Per-paper detail notes: [[relatedworks/53_mm_sam_adapter_relatedwork]], [[relatedworks/54_omnisegmentor_relatedwork]], [[relatedworks/55_sam2_memory_attention_occupants]], [[relatedworks/56_sam_family_multimodal_periphery]], [[relatedworks/57_sam3_pe_multiscale_gap]].

⚠️ 이 노트의 모든 novelty 주장은 2026-07-02 adversarial verification(2인 skeptic 교차검증)을 거쳐 **정정된 버전**이다. 원래 Track 1 findings의 3개 주장이 refuted, 2개가 uncertain으로 판정됨 — 아래에 counter-evidence와 함께 반영. 원래 주장을 그대로 인용하지 말 것.

## Problem setting

Vision Foundation Model(SAM/SAM2/SAM3, DINOv2/v3, Perception Encoder)을 멀티 센서(RGB-D-E-L, RGB-T) semantic segmentation의 백본으로 쓰는 방법론의 2024→2026-07 지형. 우리(RBMA on MemorySAM)의 novelty cell이 아직 비어 있는지, DELIVER 리더보드의 현재 1위가 누구인지, SAM3의 single-scale 한계를 누가 다뤘는지가 핵심 질문.

## Q2 — SAM2 memory attention을 cross-MODALITY fusion에 쓰는 방법 (novelty 직접 위협)

**원래 주장 "MemorySAM이 유일" → REFUTED (skeptic1).**

- **SAM4D (ICCV 2025, arXiv:2506.21547)** — "Segment Anything in Camera and LiDAR Streams". **Motion-aware Cross-modal Memory Attention (MCMA)**: SAM2-스타일 memory attention이 카메라·LiDAR 모달리티를 **가로질러** attend, unified multi-modal positional encoding 사용. 단, **promptable segmentation**이지 DELIVER-스타일 semantic segmentation이 아님.
- 따라서 셀 상태를 두 층으로 분리해야 함:
  1. "memory attention을 cross-modality fusion 메커니즘으로 사용" (base mechanism) — **occupants: MemorySAM + SAM4D (2편)**. 더 이상 unoccupied라 쓸 수 없음.
  2. "멀티모달 **semantic** segmentation을 memory attention으로 fusion" (narrow cell) — **여전히 MemorySAM 단독** (skeptic2 confirmed: M⁴-SAM은 memory=temporal-only, OmniSAM은 panorama window 간, SHIFNet은 SACF 모듈 fusion, SAM-DAQ는 depth-guided query, CRISP-SAM2는 text-visual medical). 단 universal negative이므로 search-coverage 한계 명시 필요.
- 논문 문구 가이드: "MemorySAM is the only method using SAM2 memory attention for modality fusion" ❌ → "Among multimodal *semantic* segmentation methods, only MemorySAM fuses modalities through SAM2 memory attention; SAM4D applies a related cross-modal memory attention to promptable camera-LiDAR segmentation" ✅. **SAM4D는 반드시 인용.**

## Q2b — Reliability를 additive pre-softmax attention-logit bias로 주입한 선행 (RBMA cell)

**원래 landscape 서술("nearest는 learned gate/MoE/loss-level뿐") → REFUTED (skeptic1); 셀 자체는 no-found-occupant (skeptic2 uncertain).**

더 가까운 두 occupant가 확인됨:

1. **DAMM-Diffusion (arXiv:2503.09491, Eq. 9)** — UACA: $\mathrm{UACA}(Q,K,V,U)=\mathrm{softmax}\!\left(\frac{QK^\top\cdot(1-U)}{\sqrt d}\right)V$. **학습된** uncertainty map $U$를 cross-attention logits에 **pre-softmax로 주입** — 멀티모달(histology) fusion. 단 **multiplicative**, additive 아님, 학습 필요.
2. **SAM2Long (ICCV 2025, arXiv:2410.16268)** — **training-free** reliability(occlusion score) 가중치 $w\in[0.95,1.05]$가 SAM2 memory key를 곱셈 스케일 → memory cross-attention logits의 pre-softmax reweighting. 단 **temporal** memory이지 modality가 아님.
3. 인접: **SAE (arXiv:2603.16558)** — training-free, entropy 기반 reliability로 inference-time attention-logit 조정 (LVLM vision-text hallucination; additive 여부 abstract에서 미확인). [[relatedworks/45_sae_additive_logit_entropy_lvlm_nearmiss]] 참조.

**정정된 RBMA novelty 문장**: 정확한 셀 = *training-free + ADDITIVE pre-softmax bias + cross-MODALITY (SAM2 memory attention) + dense semantic seg* — 이 조합의 occupant는 미발견. 그러나 **novelty margin은 메커니즘 클래스가 아니라 설계 축 1개 차이**(additive vs multiplicative / modality vs temporal)임을 논문에서 인정하고, DAMM-Diffusion·SAM2Long·SAE를 related work에서 정면으로 다뤄야 함. "any other paper" 식의 광범위 negative 문구 금지 — scope을 "multimodal sensor-fusion segmentation"으로 한정. → [[relatedworks/42_attention_logit_bias_novelty_defense]] 갱신 필요 (Track 2 관할).

## Q3 — DINOv2/v3 기반 멀티모달 seg가 DELIVER에서 SAM2 계열을 위협하는가

**원래 주장("DINO 백본으로 DELIVER 숫자 보고한 방법 없음") → skeptic1 uncertain / skeptic2 REFUTED (literal).**

- **반례: MMMS (arXiv:2509.12963)** — "Multi-Modal Multi-Surface Interactive Segmentation". **DINOv2-pretrained ViT-B/14**를 RGB 인코더(최고 성능 백본)로 사용, depth/event/lidar는 SegFormer 인코더 + cross-attention(MMFuser)으로 fusion, **DELIVER 숫자 보고** (NoC@90 최대 1.28 클릭 감소, depth+event NoC 14.50). [ABSTRACT-ONLY]
- 뉘앙스: MMMS는 **interactive segmentation (NoC metric)**이지 semantic-seg mIoU가 아님 → **"DELIVER mIoU 리더보드에서 경쟁하는 DINO 기반 방법은 여전히 없음"**이 살아남는 정확한 결론 (skeptic1 2회 독립 adversarial search에서도 mIoU 반례 zero, moderate confidence).
- 주변 생태계: AnyThermal (2602.06203, ICRA'26, thermal-only distillation), MM-DINOv2 (MICCAI'25, medical MRI), DINO-in-the-Room (2503.18944, 3D). DINOv3 공식 릴리즈는 ADE20K seg + NYU depth probe만.
- 시사점: open flank는 여전히 열려 있음(DINOv3-멀티모달 진입자 경계). RBMA는 cross-attention fusion이면 어디든 이식 가능하다는 문장 1개 논문에 넣을 것.

## Q4 — DELIVER 현재 #1 (프로토콜별 top-5) [skeptic1+2 both CONFIRMED]

MemorySAM 65.38은 **CMNeXt-프로토콜 클러스터에서 더 이상 #1이 아님** — 이 사실은 양쪽 skeptic 모두 primary source에서 재현.

### Cluster A — CMNeXt protocol (RGB-D-E-L, 2005-image val-as-test)

| # | Method | Backbone | mIoU | Split | Tag | Source |
|---|--------|----------|------|-------|-----|--------|
| 1 | StitchFusion (ACMMM'25) | MiT (B2로 추정, B4 여부 미확정) | **68.18** (EQUISeg 인용치 68.20) | [val-as-test] | [VERIFIED-PDF] | 2408.01343 |
| 2 | OmniSegmentor (NeurIPS'25) | DFormer-L | 68.0 | [val-as-test, protocol-inference] | [VERIFIED-PDF, Table 1f] | 2509.15096 |
| 3 | EQUISeg | MiT-B2 | 67.90 | [val-as-test] | [VERIFIED-PDF, Table 1] | 2509.24505 |
| 4 | GeminiFusion (ICML'24) | MiT-B2 | 66.9 | [val] | [VERIFIED-PDF, Table 1] | 2406.01210 |
| 5 | CMNeXt (CVPR'23) | MiT-B2 | 66.30 | [val] | [vault-verified] | 2303.01480 |
| 6 | MemorySAM | SAM2/Hiera | 65.38 | [unknown — Track 3] | [ABSTRACT-ONLY] | 2503.06700 |
| 7 | MLE-SAM | SAM ViT | 64.08 | [unknown] | [UNVERIFIED-BLOG] | 2412.04220 |

⚠️ **StitchFusion "Swin-Tiny 70.3" 는 UNSUBSTANTIATED** — skeptic2가 어디서도 검증 실패, 공식 GitHub README는 Swin 실험을 TODO로만 기재. 커뮤니티 인용치도 68.20 (EQUISeg). **70.3을 인용하지 말 것**; 리더보드 1위는 68.18–68.20으로 기재. 또한 68.18의 MiT variant(B2 vs B4)도 직접 미확인 — 인용 시 "MiT backbone"으로만.

### Cluster B — CAFuser protocol (1897-image test split, CLDE)

| # | Method | mIoU (test, CLDE) | Tag | Source |
|---|--------|-------------------|-----|--------|
| 1 | DGFusion (RA-L'26) | **56.7** (vs CAFuser 55.6, +1.1) | [test] [VERIFIED-PDF, v3] | 2509.09828 |
| 2 | CAFuser CA² (RA-L'25) | 55.6 | [test] [vault-verified] | 2410.10791 |
| 3 | CMNeXt (rerun) | 53.0 | [test] [vault-verified] | — |

DGFusion MUSES test 79.5도 confirmed (2509.09828v3).

### 제3 프로토콜 변형 — MM SAM-adapter의 자체 test 파티션

MM SAM-adapter (2509.10408): DELIVER **test** "All" RGB-D 57.35 / RGB-L 57.14 [VERIFIED-PDF, Table 4] — 2-modality + 자체 easy/hard 파티션이므로 **어느 클러스터와도 행을 합치지 말 것**. 상세는 [[relatedworks/53_mm_sam_adapter_relatedwork]].

**Bottom line: 단일 #1 없음.** Clean-val 리더 = StitchFusion 68.18–68.20; test-CLDE 리더 = DGFusion 56.7; SAM-family test 리더 = MM SAM-adapter (2-modal). **우리 SOTA 스토리는 clean-val mIoU 우위가 아니라 condition-adaptive/test-protocol(Cluster B) + robustness에서 승부해야 함.**

## Q5 — SAM3/PE의 multi-scale 한계 (우리 SAM3-RBMA ~24 mIoU plateau 설명)

**원래 주장("multi-scale neck을 SAM3/PE에 붙인 published work 없음") → REFUTED (skeptic1) / narrow-scope에서만 생존 (skeptic2).** 상세는 [[relatedworks/57_sam3_pe_multiscale_gap]].

- **SAM3-UNet (arXiv:2512.01789)**: frozen SAM3 PE-ViT(+per-block adapter)에 정확히 이 neck을 이미 구현 — 4개 1×1 conv projection → H/4–H/32 multi-scale map → U-Net decoder. 단 **binary dense seg (mirror/salient-object)**, multi-class·multimodal 아님.
- **Detect Anything in Real Time (arXiv:2603.11441)**: SAM3 ViT-H feature에 ~5M-param FPN adapter 학습.
- PE 논문 자체(arXiv:2504.13181)의 dense recipe = **ViTDet simple feature pyramid + windowed attention** (skeptic2 confirmed).
- **정정된 포지셔닝**: "multi-scale neck on frozen SAM3/PE"라는 **메커니즘 자체는 published** — 우리는 이를 novelty로 주장하지 말고 SAM3-UNet·DART를 인용하며, 남은 빈 칸인 **"(multimodal) semantic segmentation에의 적용"**만 engineering contribution으로 주장. plateau fix 레시피는 ViTDet simple-FPN (stride-{4,8,16,32} conv, 마지막 1/16 map에서 분기).

## Synthesis taxonomy — mechanism-class 축 (verification 반영판)

| Method | arXiv | Venue | VFM | Modality entry | Fusion mechanism-class | DELIVER | Reliability/condition signal |
|---|---|---|---|---|---|---|---|
| MemorySAM | 2503.06700 | arXiv'25 | SAM2 | modalities-as-frames | **memory cross-attention** | 65.38 [A][ABSTRACT-ONLY] | none |
| **SAM4D** | 2506.21547 | ICCV'25 | SAM2-style | camera+LiDAR streams | **cross-modal memory attention (MCMA)** | n/a (promptable) | motion-aware PE, not reliability |
| MM SAM-adapter | 2509.10408 | arXiv'25 | SAM1 ViT-L | ConvNeXt-S side encoders | learned-gate (deform. cross-attn injection) | 57.35 RGB-D [test][VERIFIED-PDF] | none (architectural) |
| FusionSAM | 2408.13980 | arXiv'24 | SAM1 | VQ latent tokens | prompt-level fusion | n/a | none |
| MLE-SAM | 2412.04220 | arXiv'24 | SAM1 | per-modality LoRA experts | learned-gate (modality-ID MoE routing) | 64.08 [unknown][UNVERIFIED-BLOG] | modality-ID only |
| SHIFNet | 2503.02581 | IROS'25 | SAM2 | shared encoder + SACF | learned-gate (text-guided affinity) | n/a (PST900 89.8, FMB 67.8) | text/CLIP |
| M⁴-SAM | 2605.11760 | arXiv'26 | SAM2 | modality dispatcher (MoE-LoRA in encoder) | learned-gate; memory=**temporal-only** | n/a (VSOD) | modality-ID routing |
| MM-SAM | 2408.09085 | NeurIPS'24? | SAM1 | per-modality embeds | weakly-sup fusion | n/a | none |
| MMMS | 2509.12963 | arXiv'25 | **DINOv2** ViT-B/14 (RGB) | SegFormer aux encoders | cross-attention (MMFuser) | NoC metric only (interactive) | none |
| OmniSegmentor | 2509.15096 | NeurIPS'25 | none (DFormer) | per-modality stems, add+LN | feature-add + enhancement (pretraining axis) | 68.0 [A][VERIFIED-PDF] | none |
| StitchFusion | 2408.01343 | ACMMM'25 | none (MiT) | multi-dir adapter weaving | feature-share (MultiAdapter) | 68.18 [A][VERIFIED-PDF] | none |
| EQUISeg | 2509.24505 | arXiv'25 | none (MiT) | equal 4-stage encoding | cross-attn + distill | 67.90 [A][VERIFIED-PDF] | none (loss-balance) |
| GeminiFusion | 2406.01210 | ICML'24 | none (MiT) | pixel-wise cross-attn | feature-level attention | 66.9 [A,val][VERIFIED-PDF] | none |
| DGFusion | 2509.09828 | RA-L'26 | none | depth-GT aux loss | condition-token + local depth tokens | 56.7 [B,test][VERIFIED-PDF] | learned, LiDAR-depth GT 필요 |
| CAFuser | 2410.10791 | RA-L'25 | none | CLIP condition token | condition-token (CA²/CAA) | 55.6 [B,test][vault-verified] | learned, CLIP |
| DAMM-Diffusion | 2503.09491 | '25 | none (diffusion) | histology multimodal | **multiplicative pre-softmax uncertainty (UACA)** | n/a (medical) | **learned uncertainty map U** |
| SAM2Long | 2410.16268 | ICCV'25 | SAM2 | (temporal, 단일 모달) | **multiplicative pre-softmax memory-key scaling** | n/a (VOS) | **training-free occlusion score** |
| RBMA (ours) | — | — | SAM2 | modalities-as-frames | **ADDITIVE pre-softmax logit bias in memory attention** | — | **training-free predictive entropy** |

[A] = CMNeXt-protocol val-as-test 2005; [B] = CAFuser-protocol test 1897 CLDE.

**정정된 cell 진술**: "logit-bias × reliability" 메커니즘 클래스는 DAMM-Diffusion(multiplicative·learned·modality)과 SAM2Long(multiplicative·training-free·temporal)이 이미 점유. 우리의 미점유 조합 = **additive × training-free × cross-modality × semantic seg** — 정확히 이 4축 교집합만 주장 가능.

## Improvement directions (paper-level)

1. Related work의 SAM-adaptation 분류: adapter-injection (MM SAM-adapter) / prompt-fusion (FusionSAM) / LoRA-MoE (MLE-SAM, M⁴-SAM) / text-conditioned (SHIFNet) / **memory-attention (MemorySAM, SAM4D → ours)** — SAM4D를 memory-attention 클래스에 반드시 포함.
2. Claim 정밀도: "first *reliability-biased* memory attention for multimodal semantic segmentation" (additive·training-free 한정) — "first SAM2 multimodal"(MemorySAM), "best DELIVER mIoU"(StitchFusion/OmniSegmentor), "first memory-attention modality fusion"(SAM4D 존재) 모두 금지.
3. 평가 프로토콜: Cluster B (test/CLDE + per-condition)를 primary로; sensor-failure benchmark (2503.18445, CVPRW'25) + MM SAM-adapter식 RGB-easy/hard split을 robustness 증거로 추가.
4. SAM3 plateau: ViTDet simple-FPN neck — SAM3-UNet/DART 인용 하에 "first for multimodal semantic seg"로만 포지셔닝 (secondary contribution).
5. Watchlist (재확인 대상): StitchFusion Swin 결과 공개 여부, MLE-SAM split, SHIFNet full PDF, SAM4D 후속(semantic화 여부 — **최대 스쿠프 경로**), DINOv3-멀티모달 진입자, M⁴-SAM journal version, SAE(2603.16558) full text.

## Comparison to RBMA-P29-P30 (mechanism-class)

- RBMA (logit-additive-bias, training-free): 최근접 = SAM2Long (training-free, multiplicative, temporal) / DAMM-Diffusion (learned, multiplicative, modality). 차별 축 = additive + modality + semantic seg.
- P29 SDC (condition-token/FiLM routing): 이 landscape에서 condition 신호를 쓰는 것은 DGFusion(depth-GT 필요)·CAFuser(CLIP 필요)뿐 — unsupervised image-derived latent는 미점유 (Track 6 확정 대상).
- P30 (class-token decoder on fused memory): SAM-DAQ가 depth-guided query로 인접하나 memory-feature 위 class query는 미발견 (Track 7 확정 대상).

## Application to ours (RBMA/P29/P30 적용방향)

1. **RBMA**: novelty 문장을 4축 교집합으로 다시 쓰고, SAM4D·DAMM-Diffusion·SAM2Long·SAE 4편을 related work에 선제 배치 (리뷰어가 먼저 찾기 전에). SAM2Long의 [0.95,1.05] 범위 제약은 우리 λ·B 스케일 설계의 참고치.
2. **성능 스토리**: clean-val에서 MemorySAM+RBMA가 68.2를 못 넘으면 leaderboard 주장 불가 → per-condition/test-CLDE/sensor-failure에서의 delta를 main table로.
3. **stacking 기회**: OmniSegmentor의 ImageNeXt pretraining은 orthogonal — "composable with modality-pretraining" 문장 + 후속 실험 후보.
4. **SAM3 트랙**: ViTDet simple-FPN을 붙여 plateau 해소 실험 — SAM3-UNet의 4×(1×1 conv) 레시피가 구현 청사진.

## Related-work paragraph candidate (English)

> Vision foundation models have recently been adapted to multi-sensor dense prediction along several distinct mechanism classes. Adapter-injection methods such as MM SAM-adapter fuse auxiliary modalities through deformable cross-attention injectors into a SAM ViT, while LoRA-based mixtures (MLE-SAM, M4-SAM) route per-modality experts by modality identity, and SHIFNet balances RGB-thermal contributions with text-guided affinity. A separate line reinterprets fusion as memory reasoning: MemorySAM treats modalities as frames fused by SAM2 memory attention for semantic segmentation, and SAM4D extends cross-modal memory attention to promptable camera-LiDAR streams. Orthogonally, OmniSegmentor improves multimodal segmentation purely through multi-modal pretraining, and StitchFusion through inter-encoder feature weaving; neither models sensor reliability. Closest to our mechanism, DAMM-Diffusion injects a learned uncertainty map multiplicatively into pre-softmax cross-attention logits for medical image fusion, and SAM2Long scales SAM2 memory keys by a training-free occlusion score for long-video tracking. To our knowledge, however, no prior work injects a training-free per-modality reliability signal as an additive pre-softmax bias into memory cross-attention for multimodal semantic segmentation, which is precisely the cell our method occupies.
