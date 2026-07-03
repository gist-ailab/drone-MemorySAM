---
title: 멀티모달 세그멘테이션 군집형 관련연구 리뷰
tags: [material, pdf-ready, related-work, multimodal-segmentation, rbma, korean]
created: 2026-06-25
source: [[relatedworks/90_clustered_relatedwork_synthesis]]
status: pdf-ready
---

# 멀티모달 세그멘테이션 군집형 관련연구 리뷰

## 초록

이 문서는 [[26_MultimodalSeg]] 볼트의 관련연구 노트를 논문 작성용으로 재구성한 학습/리뷰 자료이다. 현재 문헌은 여섯 군집으로 정리된다. 첫째, 직접적인 멀티모달 의미론적 세그멘테이션, 둘째, 멀티모달 객체 검출, 셋째, 어댑터·LoRA·기반모델 적응, 넷째, 세그멘테이션/검출 헤드, 다섯째, 불확실성·신뢰도·독창성 방어, 여섯째, 벤치마크와 데이터셋이다. 핵심 결론은 기존 연구가 feature fusion, SAM/SAM2 적응, anymodal distillation, condition-aware perception, BEV/query detection 측면에서 강한 기준선을 제공하지만, **SAM2 메모리 attention 내부에 신뢰도를 pre-softmax attention-logit bias로 주입하는 방식**은 아직 명확한 공백으로 남아 있다는 점이다.

## 1. 연구 질문

**RBMA 방식의 멀티모달 의미론적 세그멘테이션 논문은 기존 멀티모달 세그멘테이션, 멀티모달 검출, adapter/LoRA, 기반모델, 불확실성, 벤치마크 문헌과 어떻게 차별화되어야 하는가?**

가장 방어 가능한 답은 “fusion control의 위치”를 중심으로 관련연구를 구성하는 것이다. 기존 연구는 주로 feature, decoder, training objective, modality selection, late fusion을 개선한다. RBMA는 attention 경쟁 자체를 바꾸는 방법으로 설명해야 한다.

## 2. 군집형 문헌 지도

| 군집 | 대표 연구 | 핵심 교훈 | RBMA 관점의 공백 |
|---|---|---|---|
| 직접 멀티모달 세그멘테이션 | MemorySAM, DGFusion, CMX, TokenFusion, MAGIC++, CAFuser, StitchFusion, AnySeg | memory, feature, token, adapter, condition, anymodal fusion 기준선 | 신뢰도가 암묵적이거나 proxy 기반이며 memory-attention logit 내부에 있지 않음 |
| 멀티모달 객체 검출 | BEVFusion, TransFusion, DeepInteraction, FUTR3D | 공유 공간, query attention, modality identity 보존이 강건한 센서 융합에 중요 | 검출 메커니즘을 dense semantic prediction으로 변환해야 함 |
| Adapter / LoRA / 기반모델 | LoRA, AdaptFormer, VPT, ViT-Adapter, SAM-Adapter, SAMed, MedSAM, MoE-LoRA SAM | PEFT는 큰 encoder를 저비용으로 domain/sensor에 적응시킴 | 적응은 corrupted sensor의 신뢰도 판단을 직접 해결하지 않음 |
| Head | SegFormer, Mask2Former, OneFormer, DETR, Deformable DETR, DINO, MaskDINO, YOLO, Mask R-CNN | head는 출력 형식과 metric을 결정 | head 자체는 reliability mechanism이 아님 |
| 신뢰도 / 독창성 | UTFNet, HyperDUM, TMC, DGFusion, CAFuser, unimodal-bias regularization | uncertainty와 modality collapse는 이미 중요한 문제로 인식됨 | 다수 방법은 feature/output/loss를 조절하며 pre-softmax memory attention은 아님 |
| 벤치마크 | DeLiVER, MUSES, MCubeS | semantic/panoptic multimodal 평가에 적합 | 모든 수치 주장은 source table 검증이 필요 |

## 3. 직접 멀티모달 의미론적 세그멘테이션

가장 가까운 기준선은 MemorySAM, DGFusion, CMX, TokenFusion, MAGIC++, CAFuser, StitchFusion, AnySeg, Reducing Unimodal Bias이다. MemorySAM은 modality를 SAM2의 frame-like input으로 보고 memory mechanism을 사용하기 때문에 가장 직접적인 구조적 비교 대상이다. DGFusion과 CAFuser는 driving scene에서 condition-aware 또는 depth-guided fusion을 사용한다. StitchFusion은 pretrained encoder들을 lightweight adapter로 연결하며, AnySeg 계열은 missing 또는 arbitrary modality를 distillation으로 처리한다.

정리는 명확하다. 이 방법들은 멀티모달 세그멘테이션 성능을 높이지만, reliability를 SAM2 memory attention의 pre-softmax prior로 직접 구현하지 않는다. 따라서 RBMA는 representation/adaptation 전략 위에 놓이는 **신뢰도 제어 계층**으로 위치시키는 것이 좋다.

## 4. 멀티모달 객체 검출 문헌의 역할

객체 검출 논문은 직접적인 semantic segmentation 기준선은 아니지만 설계 원칙을 제공한다. BEVFusion은 camera와 LiDAR evidence를 BEV 공간에서 결합하는 장점을 보여준다. TransFusion은 hard geometric association보다 learned query attention이 더 강건할 수 있음을 보인다. DeepInteraction은 modality-specific stream을 보존하는 것이 정보 손실을 줄일 수 있음을 주장한다. FUTR3D는 query-based feature sampling으로 camera/LiDAR/radar 조합을 유연하게 다룬다.

논문에서는 이 검출 연구들을 “learned multimodal association이 왜 중요한가”를 설명하는 보조 문헌으로 사용하는 것이 적절하다. 단, 이들을 semantic segmentation의 직접 경쟁 방법처럼 과장해서는 안 된다.

## 5. 기반모델 적응: Adapter와 LoRA

LoRA, AdaptFormer, VPT, ViT-Adapter는 transformer backbone을 효율적으로 적응시키는 방법을 설명한다. SAM-Adapter, MedSAM, SAMed, MemorySAM, MoE-LoRA SAM, SAM-FuseNet, ClassWise-SAM-Adapter는 SAM 계열 모델도 medical, SAR, RGB-thermal, multimodal semantic segmentation 영역에서 domain-specific customization이 필요함을 보여준다.

중요한 구분은 adaptation과 reliability가 다르다는 점이다. Adapter는 “이 modality/domain을 어떻게 표현할 것인가?”에 답한다. RBMA는 “현재 위치와 상황에서 이 modality를 얼마나 믿을 것인가?”에 답한다. 이 구분을 introduction과 related work에서 명확히 해야 한다.

## 6. Head와 평가 인터페이스

의미론적 세그멘테이션 중심 실험에서는 SegFormer, UPerNet, DeepLabv3+ 계열 head로 mIoU를 보고하는 것이 가장 깔끔하다. Mask2Former와 OneFormer는 semantic, instance, panoptic segmentation을 모두 지원한다. DETR, Deformable DETR, DINO, MaskDINO, YOLO, Mask R-CNN은 detection 또는 instance segmentation 확장에 적합하다.

첫 논문에서는 semantic segmentation과 reliability-aware fusion을 중심으로 유지하는 편이 좋다. detection/panoptic head는 부가 실험 또는 향후 확장으로 두면 핵심 기여가 흐려지지 않는다.

## 7. 신뢰도와 독창성 방어

가장 강한 독창성 방어는 수학적 위치에 있다. Feature scaling은 feature 크기를 바꾼다. Late fusion은 output aggregation을 바꾼다. Evidential fusion은 branch-level confidence를 결합한다. Modality selection은 sensor나 feature group을 선택한다. Loss regularization은 학습 동역학을 바꾼다. Condition/depth token은 context를 추가한다. RBMA는 다르다. RBMA는 softmax 이전 attention logit에 reliability를 더해서 memory-token 경쟁 자체를 바꾼다.

논문에서 필요한 ablation은 다음과 같다.

1. Reliability bias가 없는 MemorySAM-style fusion.
2. Feature-level reliability scaling.
3. Output-level uncertainty weighting.
4. Explicit uncertainty가 없는 learned gate.
5. Global modality reliability와 local patch/token reliability 비교.
6. Dark RGB, thermal saturation, event noise, sparse LiDAR/depth 등 corruption-specific test.
7. ECE, uncertainty-error correlation 같은 calibration test.

## 8. 벤치마크와 정량 보고

현재 벤치마크 노트는 DeLiVER, MUSES, MCubeS를 핵심 데이터셋으로 지지한다. 추출된 표에는 MemorySAM, DGFusion, Reducing Unimodal Bias, StitchFusion, AnySeg, MAGIC++, CAFuser의 source-table-backed row가 포함되어 있다. 관련연구와 motivation에는 사용할 수 있지만, 최종 원고에서는 모든 수치에 source table 번호와 metric을 보존해야 한다.

## 9. 바로 사용할 수 있는 관련연구 문단

최근 멀티모달 의미론적 세그멘테이션 연구는 feature rectification, token fusion, modality selection, adapter exchange, condition-aware modulation, distillation을 통해 강건성을 높여 왔다. CMX와 TokenFusion은 feature/token fusion 기준선이고, MAGIC++와 AnySeg는 arbitrary 또는 missing modality 설정을 다루며, CAFuser와 DGFusion은 driving scene perception을 위해 condition 및 depth-guided fusion을 도입한다. StitchFusion은 pretrained encoder를 adapter로 엮고, MemorySAM은 modality를 SAM2 memory attention 구조로 매핑한다. 이 연구들은 강한 멀티모달 기준선을 제공하지만, reliability를 attention 경쟁 내부에 명시적으로 넣는 경우는 드물다. RBMA는 explicit reliability prior를 softmax 이전 memory-attention logit에 주입하여 unreliable modality-memory token의 영향력을 fusion 과정에서 낮춘다는 점에서 차별화된다.

## 10. 참고

전체 citation metadata와 검증 상태는 [[relatedworks/90_clustered_relatedwork_synthesis]] 및 연결된 per-paper note를 참조한다. 최종 원고의 인용은 discovery DB가 아니라 검증된 per-paper note에서 가져와야 한다.
