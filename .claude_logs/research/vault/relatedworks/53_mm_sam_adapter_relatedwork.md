---
title: MM SAM-adapter — Multimodal Adapter Injection into SAM for RGB-X Segmentation
tags: [related-work, key-paper, sam, adapter, multimodal-segmentation, deliver, muses, fmb]
created: 2026-07-02
source: arXiv:2509.10408 (html v1 verified); code https://github.com/iacopo97/Multimodal-SAM-Adapter
status: verified-draft
---

# MM SAM-adapter (arXiv:2509.10408)

Curti, Zama Ramirez, Petrelli, Di Stefano (U. Bologna). arXiv 2025-09 (v3 exists; conference tag 미확인). Track 1 deep-research, 2026-07-02.

## Problem setting

RGB-X (2-modality) semantic segmentation — DELIVER, FMB, MUSES. SAM(1)의 RGB prior를 유지하면서 보조 모달리티를 "이득이 될 때만" 사용하는 adapter 설계. **SAM1 (ViT-L, 24 layers, 1024-d)이며 SAM2가 아님** — memory 메커니즘 없음.

## Novelty

Fused multimodal feature를 SAM의 RGB feature 스트림에 **주입(inject)**하는 비대칭 adapter: auxiliary modality가 도움이 될 때만 선택적으로 반영되도록 학습됨. 추가로 RGB-easy / RGB-hard 평가 split 제안 (RGB만으로 충분한 장면 vs 보조 모달리티가 필수인 장면 분리).

## Method

- 비대칭 구조: SAM ViT-L RGB branch (fine-tuned; frozen ablation은 55.35 vs 57.14, Table 13) + 경량 **ConvNeXt-Small** modality encoder.
- 업스트림 "Road-Fusion" 모듈이 RGB+X를 fusion → fused feature를 **multi-scale deformable cross-attention injector/extractor**로 SAM 레이어 6개마다 (4 blocks) 주입.
- Reliability 신호 없음 — "selective incorporation"은 학습된 cross-attention의 architectural 효과이지 explicit 신호가 아님. Attention logit에 스칼라 bias를 더하는 메커니즘도 없음.

## Quantitative results (verbatim rows)

| Dataset | Config | mIoU | Split | Tag |
|---|---|---|---|---|
| DELIVER "All" | RGB-D | **57.35** | [test] (자체 파티션) | [VERIFIED-PDF, Table 4] |
| DELIVER "All" | RGB-L | 57.14 | [test] | [VERIFIED-PDF, Table 4] |
| DELIVER "All" | RGB-E | 55.70 | [test] | [VERIFIED-PDF, Table 4] |
| DELIVER RGB-hard | RGB-L | 45.46 | [test] | [VERIFIED-PDF, Table 4] |
| (동일 표 경쟁자) CMNeXt RGB-L | | 51.32 | [test] | [VERIFIED-PDF, Table 4] |
| (동일 표) GeminiFusion | | 50.57 | [test] | [VERIFIED-PDF, Table 4] |
| (동일 표) RoadFormer+ | | 54.56 | [test] | [VERIFIED-PDF, Table 4] |
| FMB | RGB-T | 66.10 | [test] | [VERIFIED-PDF, Table 5] (CMNeXt 61.66, GeminiFusion 64.75) |
| MUSES | RGB-L | **81.07** | [test] | [VERIFIED-PDF, Table 6] (RoadFormer+ 80.38, CMNeXt 72.36) |
| MUSES | RGB-E | 79.92 | [test] | [VERIFIED-PDF, Table 6] |

⚠️ 프로토콜 주의: DELIVER 숫자는 **test split + 자체 easy/hard "All" 파티션 + 2 modalities** — CMNeXt-프로토콜 클러스터(66–68, RGB-D-E-L val-as-test)와도 CAFuser CLDE 클러스터(53–57, 4-modal)와도 **행을 합치지 말 것** (제3 프로토콜 변형). MUSES 81.07은 2-modality semantic-seg mIoU로, CAFuser 78.2와 protocol/metric 맥락이 다름 — Track 3에서 reconcile. **MemorySAM·DGFusion과는 논문 내 비교 없음.**

## Limitations

- **입력 2 모달리티만 지원** (Road-Fusion 제약; 저자 명시). DELIVER 4-modal 설정 불가.
- FLOPs/latency 미보고. SAM1 기반이라 memory-attention 계열 fusion과 구조적으로 다른 축.
- Reliability가 implicit — 어떤 조건에서 보조 모달리티를 얼마나 신뢰했는지 해석 불가.

## Improvement directions

- Road-Fusion을 N-modality로 일반화하면 SAM-family test-split 리더로서 강력해질 것 — 스쿠프 감시 대상.
- RGB-easy/hard split은 우리도 채택 가능한 robustness 평가 프로토콜 (리뷰어 선제 대응).

## Comparison to RBMA-P29-P30 (mechanism-class)

- **Mechanism-class: learned-gate (deformable cross-attn injection)** — feature-level, 학습 필요, 신호 없음.
- RBMA(logit-additive-bias, training-free)와는 클래스가 다름: 이들의 "selective incorporation"은 아키텍처가 암묵적으로 배우는 것, 우리는 explicit reliability 스칼라를 attention logit에 더함.
- SAM1이라 memory attention 부재 → RBMA의 직접 위협 아님. P30의 query-decoder와도 무관.

## Application to ours (RBMA/P29/P30 적용방향)

1. Related work에서 "adapter-injection class vs memory-attention class"의 대표 대조군으로 인용.
2. 이들의 DELIVER-test 57.35 (RGB-D)는 우리가 test-split(Cluster B 인접)에서 보고할 때 SAM-family 참고점 — 단 2-modal이므로 각주로 프로토콜 차이 명시.
3. RGB-easy/hard 스타일 stress split을 우리 실험 계획에 추가 (sensor-failure benchmark 2503.18445와 병행).

## Related-work paragraph candidate (English)

> MM SAM-adapter augments a SAM ViT-L backbone with lightweight ConvNeXt modality encoders whose fused features are injected into the RGB stream through multi-scale deformable cross-attention adapters, so that auxiliary modalities are exploited only when beneficial. While this achieves strong two-modality results on the DELIVER test split, the selectivity is an implicit property of learned attention rather than an explicit, interpretable reliability signal, and the framework is restricted to two input modalities without any memory mechanism.
