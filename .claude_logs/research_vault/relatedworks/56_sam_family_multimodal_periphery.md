---
title: SAM-Family Multimodal Periphery — SHIFNet, MLE-SAM, FusionSAM, MM-SAM, MMMS + citing sweep
tags: [related-work, sam, sam2, multimodal-segmentation, rgb-thermal, moe-lora, dinov2, taxonomy-rows]
created: 2026-07-02
source: arXiv:2503.02581, 2412.04220, 2408.13980, 2408.09085, 2509.12963 + MemorySAM S2 citation sweep
status: verified-draft
---

# SAM-family 멀티모달 주변부 — taxonomy row 모음

Track 1 Q1/Q3 보조 노트, 2026-07-02. 개별 full note가 필요 없는 중소 위협들의 검증된 요약. Mechanism-class 표는 [[relatedworks/52_vfm_multimodal_landscape_synthesis]].

## SHIFNet (arXiv:2503.02581, IROS 2025) [ABSTRACT-ONLY]

- SAM2 encoder, RGB/T shared weights, 32.27M trainable (adapter-scale). Code: https://github.com/iAsakiT3T/SHIFNet
- **SACF 모듈**: text-guided affinity learning으로 모달리티 기여 balance + Heterogeneous Prompting Decoder.
- 수치: PST900 **89.8**, FMB **67.8** [ABSTRACT-ONLY].
- **SAM2 memory attention을 fusion에 쓰지 않음** (fusion은 encoder feature 위 별도 SACF; language-guided) — [skeptic2 재확인: near-miss로 재분류, occupant 아님]. 단 abstract-level negative → 인용 전 PDF 확인.
- vs ours: mechanism-class = **learned-gate (text-conditioned affinity)**; text/CLIP 필요. 우리 SDC는 label-free·text-free, RBMA는 training-free.

## MLE-SAM (arXiv:2412.04220) [UNVERIFIED-BLOG/html snippet]

- SAM(1) + per-modality LoRA experts (MoE routing). **DELIVER RGB-D-E-L 64.08 mIoU** [unknown split — Track 3 재검증 필요]; RGB-only 대비 +8.85.
- Routing은 **modality-identity 기반**, condition/reliability 기반 아님 → P29 SDC cell은 이쪽에서 여전히 열려 있음.
- 표에 CWSAM, SAM-LoRA, MemorySAM 등 baseline 포함 — full-table verbatim 미확보 (재수집 대상).

## FusionSAM (arXiv:2408.13980) [ABSTRACT-ONLY]

- SAM(1), autonomous driving. Latent Space Token Generation (모달리티별 VQ latent) + cross-attention inter-domain fusion → fused feature를 SAM decoder의 **prompt**로 사용.
- "+4.1% average mIoU over SOTA" 주장 — 데이터셋/숫자 abstract에 없음, code URL 없음. 저위협, taxonomy row (mechanism-class: **prompt-level fusion**).

## MM-SAM (arXiv:2408.09085, NeurIPS'24 추정) [ABSTRACT-ONLY]

- Xiao et al. SAM의 sensor-suite 확장: unsupervised cross-modal transfer + weakly-supervised multi-modal fusion. MFNet/SUN RGB-D/SemanticKITTI 평가 — **promptable class-agnostic 설정, DELIVER 없음**. Code: https://github.com/weihao1115/mm-sam
- Mechanism-class: per-modality patch embed + tuning. Reliability 없음.

## MMMS (arXiv:2509.12963) — DINOv2 쪽 반례 [ABSTRACT-ONLY, skeptic2 발굴]

- "Multi-Modal Multi-Surface Interactive Segmentation". **DINOv2-pretrained ViT-B/14 RGB 인코더**(최고 성능 백본) + SegFormer 보조 인코더 + cross-attention (MMFuser).
- **DELIVER 숫자 보고** — 단 interactive segmentation **NoC metric** (NoC@90 최대 1.28 클릭 감소, depth+event NoC 14.50), mIoU 아님.
- 의미: "DINO 백본으로 DELIVER 숫자를 보고한 방법 없음"(원래 Track 1 주장)은 **문자 그대로는 false**; "DELIVER **mIoU** 리더보드에 DINO 기반 경쟁자 없음"은 유지. 관련 서술 시 반드시 이 note 인용.

## 기타 one-liner

- **SAM-DAQ (2511.09870)** [ABSTRACT-ONLY]: depth-guided adaptive query, RGB-D VSOD — P30 related work (query on multimodal features) 소재.
- **SARTM (2505.01950)** [ABSTRACT-ONLY]: SAM2 RGB-T + language-aided distillation.
- **SAM3-Adapter (2511.19425)** [ABSTRACT-ONLY]: SAM3 → camouflage/shadow/medical, 단일 모달리티.
- **X-SAM (2508.04655)**: "unified multimodal segmentation" = MLLM/text-멀티모달 — **sensor-멀티모달 아님, related work에서 혼동 금지.**
- **AnyThermal (2602.06203, ICRA'26)** [ABSTRACT-ONLY]: DINOv2 distillation → thermal-only 백본 (fusion 아님); TartanRGBT 플랫폼.
- **MM-DINOv2 (MICCAI'25)**: 의료 MRI 시퀀스 DINOv2 — sensor fusion 아님.
- **Underwater condition-aware MMSS (ICRCV 2025, arXiv 미발견)** [ABSTRACT-ONLY via S2]: MemorySAM 인용, 수중 condition-aware — Track 2의 condition-token 선행 점검 대상.

## MemorySAM 인용 논문 sweep (S2, 2026-07-02, 13 hits)

MaCVi'26 overview (2604.13244), PanoEnv (2602.21992), BiXFormer (2506.03675, TMM), EGFormer (2505.14014, Any-modal Scoring + Modal Dropping, -88% params) [ABSTRACT-ONLY], Reducing-Unimodal-Bias (2505.06635, ICCV'25 — functional-entropy **loss** regularizer; RBMA novelty defense에 인용), Sensor-failure benchmark (2503.18445, CVPRW'25 best paper — robustness 평가 프로토콜로 채택), AnySeg (2411.17141), Med-K2N (2510.02815), underwater ICRCV'25. **memory attention을 수정한 논문 없음** → [[relatedworks/55_sam2_memory_attention_occupants]].

## Limitations

- 이 노트의 다수 항목이 [ABSTRACT-ONLY] — negative claim (특히 SHIFNet의 memory 미사용)은 인용 전 PDF 필수.
- MLE-SAM 64.08의 split 미해결; FusionSAM/MLE-SAM full-table verbatim 미확보.
- Google Scholar 인용 sweep 미실시 (S2보다 넓음) — Track 8.

## Improvement directions

- SHIFNet·MLE-SAM PDF 정독 1회로 [ABSTRACT-ONLY] 꼬리표 제거 (특히 negative 2건).
- EGFormer의 Any-modal Scoring Module은 "scoring → fusion" 인접 메커니즘 — Track 2 taxonomy에 편입 검토.

## Comparison to RBMA-P29-P30 (mechanism-class)

| Method | Class | RBMA와의 구분 |
|---|---|---|
| SHIFNet | learned-gate (text affinity) | text 필요 / feature-level |
| MLE-SAM, M⁴-SAM | learned-gate (modality-ID MoE) | routing ≠ reliability, 학습 필요 |
| FusionSAM | prompt-level fusion | logit 접근 없음 |
| MM-SAM | weakly-sup per-modality tuning | 신호 없음 |
| MMMS | cross-attention (MMFuser) | interactive/NoC, 신호 없음 |
| Reducing-Unimodal-Bias | loss-level entropy | 학습 시 regularizer, inference 무관 |

## Application to ours (RBMA/P29/P30 적용방향)

1. Related work의 SAM-adaptation 5분류(adapter-injection / prompt-fusion / LoRA-MoE / text-conditioned / memory-attention)에 이 노트가 3개 클래스의 대표 인용을 공급.
2. MMMS 때문에 DINO-side 서술은 "no DINO-based method competes on DELIVER **mIoU**"로 한정.
3. P29: MLE-SAM/M⁴-SAM의 modality-ID routing과 "unsupervised condition-latent routing"의 대비 문장 소재.

## Related-work paragraph candidate (English)

> Beyond memory-based fusion, SAM-family multimodal adaptations span several mechanism classes: FusionSAM fuses vector-quantized modality latents at the prompt level, MLE-SAM routes per-modality LoRA experts by modality identity, SHIFNet balances RGB-thermal features through text-guided affinity on a SAM2 encoder, and MM-SAM tunes per-modality embeddings for promptable sensor-suite segmentation. On the DINO side, MMMS employs a DINOv2 RGB encoder with cross-attention fusion for interactive multi-sensor segmentation on DELIVER, though no DINO-based method yet competes on the DELIVER semantic-segmentation leaderboard. Common to all of these, modality weighting is either learned, text-driven, or identity-driven; none derives a training-free reliability estimate from the model's own predictions, and none modifies attention logits.
