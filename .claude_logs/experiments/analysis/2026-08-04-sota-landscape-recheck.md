# 🔴 벤치 SOTA 지형 재확인 — 기준 수치 정정 (2026-08-04)

> 계기: "DGFusion도 3모달>4모달인가?" 질의. arXiv 원문 직접 확인 결과 **우리가 써 온 SOTA 기준이 낡았다**.
> 출처: DGFusion 2509.09828v3 · CAFuser 2410.10791v2 · **MM SAM-adapter 2509.10408v1**(신규 확인)

## 1. 🔴 DELIVER — SOTA 기준 정정

| Method | Modalities | val | test |
|---|---|---|---|
| **MM SAM-adapter** | **RGB+Depth (2모달)** | **69.60** ← val 1위 | **57.35** ← test 1위 |
| MM SAM-adapter | RGB+LiDAR | 61.89 | 57.14 |
| DGFusion | CLDE | 66.51 | 56.7 |
| CAFuser | CLDE | 67.80 | 55.60 |
| CAFuser-CAA | CLDE | 68.79 | 55.2 |
| **Ours (P46-3 λ0.05, legal)** | CLDE | **68.57** | **55.62/55.69** |

**정정 전 → 후**: val SOTA `68.79 → **69.60**` · test SOTA `56.71 → **57.35**`
**우리 격차**: val `−0.22 → **−1.03**` · test `−1.02 → **−1.66**`

⚠️ 기존 메모리([[seg-report-sota-gap]])의 68.79/56.71은 **CAFuser/DGFusion 계보만 본 값**이었다. 이후 모든 DELIVER 보고는 위 수치로 델타를 계산한다.

## 2. MUSES test (semantic mIoU) — 상위권은 전부 2모달/카메라단독

| 순위 | Method | Modalities | mIoU |
|---|---|---|---|
| 1 | GtA (리더보드, **익명·논문 없음**) | camera only | 82.39 |
| 2 | **MM SAM-adapter** | RGB+LiDAR | **81.07** |
| 3 | RoadFormer+ (MM SAM-adapter 저자 학습본) | RGB+LiDAR | 80.38 |
| 4 | MM SAM-adapter | RGB+Event | 79.92 |
| **5** | **Ours 4모달 seed2** | **CLRE** | **79.571** |
| 6 | DGFusion | CLRE | 79.5 |
| 7 | CAFuser-CAA / CAFuser | CLRE | 78.5 / 78.2 |

- ✅ **우리는 4모달(CLRE) 계보 1위**(DGFusion +0.07). 이건 정직하게 주장 가능.
- 🔴 전체 1~4위가 **카메라단독 또는 2모달**이다. MM SAM-adapter는 **radar를 명시적으로 배제**했다 — 원문: *"we excluded Radar due to its sparsity and insufficient information for multimodal segmentation."*
- ⚠️ 단 MM SAM-adapter는 **구조적으로 2모달만 지원**(논문이 한계로 명시). 3·4모달을 시도조차 못 했으므로 **"모달을 줄이는 게 낫다"의 within-method 근거는 아니다**.

## 3. 모달 개수 논쟁의 최종 정리

| 근거 유형 | 결론 |
|---|---|
| **within-method ablation** (CAFuser Table IX: RGB 55.7→+L 58.7→+R 59.3→+E 59.7) | **모달 추가는 단조 이득** |
| within-method (DGFusion: C+L 60.19 → CLRE 61.03 / DELIVER CLE 51.6 → CLDE 56.7) | **모달 추가는 이득** |
| within-method (**우리**: 3모달 79.788 → 4모달 79.571, val 82.62→82.35) | 🔴 **우리만 손해** |
| cross-method 리더보드 (2모달·카메라단독이 상위) | **판단 근거 아님**(방법론 교란) |

→ **"모달↑ = 성능↓"은 벤치의 법칙이 아니라 우리 모델의 증상**이다. 회수 가능한 결함.

## 4. 🎯 SOTA 공략에 생긴 카드

1. **MUSES RGB-L 2모달 런** — 상위권 구성과 직접 비교(81.07 대비). 우리 3모달>4모달 추세가 이어지면 2모달이 더 나올 수 있다. 비용 = 런 1개.
2. **frozen backbone 반례** — MM SAM-adapter Table 13: SAM frozen 55.35 vs fine-tuned 57.14 (**−1.79**), 저자들이 "fine-tuning 필수"라고 자기 논문에서 인정. 우리는 **frozen DINOv3 + per-modal LoRA**. 같은 대역 도달 시 직접 반박 = 노벨티 축.
3. **PQ 진입** — MM SAM-adapter는 panoptic을 "future research"로 남겼다(결론부). DGFusion/CAFuser만 PQ를 낸다. 우리가 PQ를 내면 2모달 계열과 차별된다.
4. **DELIVER RGB-D 2모달** — MM SAM-adapter의 최고 구성(57.35). 우리 CLE/CLDE 외에 이것도 비교 대상.

## 5. 후속 조치

- [ ] 메모리 `seg-report-sota-gap`의 DELIVER SOTA 수치 갱신(68.79/56.71 → 69.60/57.35)
- [ ] `research/novelty-and-related-work.md`에 MM SAM-adapter 행 추가(최근접 선행 + frozen 반례 축)
- [ ] MUSES RGB-L 런 대기열 등록
