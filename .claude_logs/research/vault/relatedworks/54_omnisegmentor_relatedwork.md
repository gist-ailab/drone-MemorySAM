---
title: OmniSegmentor — Multi-Modal Pretraining (ImageNeXt) for Semantic Segmentation
tags: [related-work, key-paper, pretraining, multimodal-segmentation, deliver, dformer, neurips2025]
created: 2026-07-02
source: arXiv:2509.15096 (html v1 verified, NeurIPS 2025); code https://github.com/VCIP-RGBD/DFormer
status: verified-draft
---

# OmniSegmentor (arXiv:2509.15096, NeurIPS 2025)

Yin, Cao, Zhang, Chen, Cheng, Hou (VCIP/Nankai). Track 1 deep-research, 2026-07-02.

## Problem setting

멀티모달(RGB + D/E/L/T) semantic segmentation을 **pretraining-data 축**에서 공략. VFM 아님 — DFormer 계열 인코더 (ResNet-101/MiT-B2 variant 포함). "First flexible pretrain-and-finetune pipeline for semantic segmentation with increasing visual modalities" 주장.

## Novelty

**ImageNeXt**: ImageNet-1K를 5개 모달리티로 확장한 pretraining 데이터셋 — Depth(Omnidata), Event(N-ImageNet), pseudo-LiDAR, learned RGB→thermal. Pretraining 시 매 스텝 RGB + 무작위 보조 모달리티 1개를 샘플링.

## Method

- Finetune: 모달리티별 separate stem + MLP → **addition + LayerNorm**으로 aggregation → learned enhancement module이 RGB와 fusion.
- Fusion mechanism-class: **feature-add + enhancement** — attention-logit 조작 없음, reliability/condition 신호 없음, memory 없음.

## Quantitative results (verbatim rows)

| Dataset | Config | Backbone | mIoU | Split | Tag |
|---|---|---|---|---|---|
| DELIVER | RGB-D-E-L | DFormer-L | **68.0** | [val-as-test, protocol-inference — DFormer/CMNeXt lineage; Track 3 확정 대상] | [VERIFIED-PDF, Table 1f] |
| DELIVER | RGB-D-E-L | MiT-B2 | 67.5 | [동일] | [VERIFIED-PDF, Table 1f] |
| DELIVER | RGB-D-L | DFormer-L | 67.2 | [동일] | [VERIFIED-PDF] |
| DELIVER | RGB-D-E | DFormer-L | 65.9 | [동일] | [VERIFIED-PDF] |
| NYUv2 | RGB-D | DFormer-L | 57.6 | [unknown] | [VERIFIED-PDF] |
| MFNet | RGB-T | DFormer-L | 60.6 | [unknown] | [VERIFIED-PDF] |
| EventScape | RGB-E | DFormer-L | 67.6 | [unknown] | [VERIFIED-PDF] |
| SUN RGBD | RGB-D | DFormer-L | 52.8 | [unknown] | [VERIFIED-PDF] |
| KITTI-360 | | DFormer-L | 69.2 | [unknown] | [VERIFIED-PDF] |

Adversarial verification (skeptic1/2): NeurIPS'25 SOTA-on-DELIVER 주장 및 68.0 (Table 1f) 재현 확인 — MemorySAM 65.38 대비 +2.6.

## Limitations

- (저자 명시) 벤치마크가 5개 모달리티의 부분집합만 커버; synthetic/pseudo 모달리티 의존.
- 조건-적응 없음: 악천후에서 어느 모달리티를 신뢰할지에 대한 메커니즘 부재 — 순수 representation 품질 개선.
- VFM prior 미활용 (SAM2/DINO 없음).

## Improvement directions

- ImageNeXt pretraining + reliability-aware fusion은 직교 → 결합 시 양쪽 gain이 합산될 가능성. 우리 후속 실험 후보.

## Comparison to RBMA-P29-P30 (mechanism-class)

- **Mechanism-class: feature-add + enhancement (pretraining axis)** — RBMA의 logit-additive-bias와 완전히 직교.
- MemorySAM을 clean DELIVER에서 +2.6 이기는 **pretraining-data 경쟁자**이지 fusion-mechanism 경쟁자가 아님.
- P29 SDC 관점: condition 신호 전무 → SDC cell 비위협. P30 관점: decoder는 표준 (query 없음).

## Application to ours (RBMA/P29/P30 적용방향)

1. Related work 프레이밍: "modality-pretraining axis (OmniSegmentor) vs fusion-mechanism axis (ours) — composable".
2. 실용: ImageNeXt-style pseudo-modal pretraining을 RBMA 밑에 쌓는 ablation은 리더보드 격차(65.38 → 68.0+) 해소 경로.
3. 우리 clean-val 표에서 68.0이 우리의 상한 비교 대상 중 하나임을 인지 — condition-adaptive 표로 승부.

## Related-work paragraph candidate (English)

> OmniSegmentor attacks multimodal segmentation from the pretraining axis: it extends ImageNet-1K into the five-modality ImageNeXt corpus and pretrains a DFormer-style encoder by sampling RGB plus one random auxiliary modality per step, reaching 68.0 mIoU on DELIVER with simple additive fusion at finetuning time. This demonstrates that representation quality alone can outperform architecture-centric fusion, yet the approach remains condition-agnostic — it offers no mechanism to modulate the contribution of a degraded sensor at inference, and is therefore orthogonal and complementary to reliability-aware fusion such as ours.
