---
title: SAM3 / Perception Encoder Multi-Scale Gap — ViTDet-FPN Necks and the SAM3-RBMA Plateau
tags: [related-work, sam3, perception-encoder, vitdet, fpn, multi-scale, dense-prediction, engineering-gap]
created: 2026-07-02
source: arXiv:2511.16719, 2504.13181, 2512.01789, 2603.11441, 2512.04585 + adversarial verification skeptic1/skeptic2
status: verified-draft
---

# SAM3/PE single-scale 한계와 multi-scale neck 선행 (Q5)

Track 1 Q5, 2026-07-02. 우리 SAM3-RBMA ~24 mIoU plateau의 원인 가설(plain-ViT single-scale)과 처방. **원래 "선행 없음" 주장은 REFUTED — 메커니즘은 이미 published, 적용처(멀티모달 semantic seg)만 미점유.**

## Problem setting

SAM3의 vision encoder = Perception Encoder (plain ViT, single-scale). Semantic-seg 류 dense task에 그대로 쓰면 multi-scale feature 부재로 성능 저하 — 우리 SAM3-RBMA 실험의 ~24 mIoU plateau와 부합. 누가 이미 neck을 붙였는가?

## 검증된 사실

- **SAM3 (arXiv:2511.16719)** [VERIFIED-PDF-lite]: PE(plain ViT) + DETR-style detection decoder + pixel decoder; semantic head는 prompt-conditioned fused feature 위 FCN-style dense module.
- **PE (arXiv:2504.13181, Meta 2025)** [VERIFIED-PDF, OpenReview copy; skeptic2 confirmed]: dense/spatial task용 Meta 자체 레시피 = **ViTDet simple feature pyramid + windowed attention (+소수 global block)** — COCO는 ViTDet + Mask R-CNN @1024. 즉 plain ViT의 마지막 1/16 map에서 strided conv/deconv로 stride-{4,8,16,32} 피라미드 생성.
- **SAM3-I (2512.04585)** [VERIFIED-PDF-lite]: SAM3 pixel decoder의 "FPN connections"에 parallel residual adapter 삽입 — pixel decoder에 FPN-류 연결이 존재함을 확인.

## 선행 (원래 주장을 뒤집은 counter-evidence — 반드시 인용)

1. **SAM3-UNet (arXiv:2512.01789, 2025-12)** [skeptic1 발굴]: frozen SAM3 PE-ViT (+per-block adapter)에 **정확히 이 neck** — 4개 1×1-conv projection을 H/4–H/32로 resize한 multi-scale map → U-Net decoder. 단 **binary dense segmentation** (mirror detection, salient object detection) — multi-class·multimodal semantic seg 아님.
2. **Detect Anything in Real Time (arXiv:2603.11441)**: SAM3 ViT-H feature에 ~5M-param **FPN adapter** 학습 (detection).
3. SAM3-Adapter (2511.19425), SegEarth-OV3 (2512.08730), medical SAM3 (2603.25945): downstream 적응이나 multi-scale × multimodal은 다루지 않음.

## 정정된 gap 진술

- ❌ "No published work adds a multi-scale/FPN neck to SAM3/PE" — **false** (SAM3-UNet, DART).
- ✅ "No published work adds one **for (multimodal) semantic segmentation**" — narrow scope에서만 생존 [skeptic1 refuted-broad / skeptic2 uncertain-narrow]. 이 architectural move는 **novelty가 아니라 documented precedent** — 인용하고 활용할 것.

## Quantitative results

이 트랙에서 SAM3-UNet/DART의 수치는 미추출 (mirror/SOD 벤치마크라 우리 표와 비교 불가). 우리 내부 참고치: SAM3-RBMA plateau **~24 mIoU** [internal, unverified vs 최신 로그 — drone .claude_logs 확인 필요].

## Limitations

- SAM3-UNet의 adapter 구성·수치 상세는 html-lite 수준 — 구현 참고 전 PDF 정독.
- "multimodal semantic seg에 적용한 사례 없음"도 universal negative — 제출 전 재스윕.

## Improvement directions

1. **처방**: SAM3/PE 마지막 1/16 feature에 ViTDet simple-FPN (stride-{4,8,16,32}) 또는 SAM3-UNet식 4×(1×1 conv → resize) neck → 기존 decoder. 두 레시피 모두 published blueprint 있음 → 구현 리스크 낮음.
2. plateau가 neck으로 해소되는지 단일-모달 대조 실험 먼저 (RBMA와 교락 방지).

## Comparison to RBMA-P29-P30 (mechanism-class)

- 이 노트는 mechanism novelty가 아니라 **engineering enabler**: multi-scale neck은 RBMA(logit-additive-bias)와 직교하며, SAM3 host에서 RBMA가 공정하게 평가받기 위한 전제 조건.
- P30 class-token decoder도 multi-scale pixel feature를 전제 — 같은 neck을 공유 가능 (ViTDet simple-FPN + query head 선행은 Track 7 Q4).

## Application to ours (RBMA/P29/P30 적용방향)

1. SAM3-RBMA 트랙 재개 시 첫 실험 = simple-FPN neck 추가. 성공 시 논문에는 "following the ViTDet recipe adopted by PE and SAM3-UNet"으로 서술 — **novelty 주장 금지, 적용처(first for multimodal semantic seg)만 minor contribution**.
2. SAM3-I가 확인한 pixel-decoder FPN 연결은 adapter 주입 지점 후보 (P29 LoRA 배치와 병행 검토).

## Related-work paragraph candidate (English)

> SAM3 inherits the single-scale plain-ViT design of the Perception Encoder, whose own dense-prediction recipe compensates with a ViTDet-style simple feature pyramid and windowed attention. Recent adaptations confirm this path: SAM3-UNet attaches four convolutional projections at H/4 to H/32 scales to a frozen SAM3 encoder for binary dense segmentation, and real-time detection adapters train lightweight FPN necks on SAM3 features. We follow this established recipe to obtain multi-scale features from SAM3, and are, to our knowledge, the first to apply it to multimodal semantic segmentation.
