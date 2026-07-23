---
created: 2026-07-23
scope: P42 — lidar-강제(조건부 img 마스킹). P41 게이트 부정(fusion-rank 아님) 후 fog 분석 + fog 딥리서치(arXiv) 기반. MUSES 병목 = 비RGB 미사용(img 과지배), fog에서 치명적.
gates: drop-modality dMIoU(lidar) 상승 + fog mIoU 개선 + val ≥ P38 82.22. ep30 조기.
---

# P42 — lidar-강제: 조건부 img 마스킹 + aux deep-sup

## 0. 진단 (P41 이후)

- **P41 게이트 부정**(fusion rank/η² 아님, §C7). MUSES 병목 재탐색 → fog 분석(2026-07-23).
- fog 분석: **drop-modality dMIoU(lidar,event)≈0**(clear (21.8,0.34,−0.03), fog (15.0,0.14,0.42)) = img 과지배, 비RGB 미사용. fog에서 img 열화(fused mIoU clear 68→fog 49) 시 폴백 없어 붕괴. GT mIoU clear 75.85/**fog 62.67**/night 78.05.
- fog 딥리서치(arXiv 근거): lidar는 **"저정보 아니라 미사용"**(AnySeg MUSES lidar-only 32.13, MUSES ablation +5.8 PQ, 리더보드 2위 **MM SAM-Adapter cam+lidar fog 74.12**=frozen-VFM 선례). **event은 저정보 근접**(제외). fog는 물리 천장 아님(lidar 쓰는 기법 fog갭 −5~−7 vs 우리 −13).
- **판별(진행 중)**: P38 모달별 단독 readout mIoU로 "lidar 미사용" 최종 확정(전제).

## 1. 처방 (frozen 백본 = loss/input 레벨만 유효)

| # | 변경 | 근거(arXiv) | 상태 |
|---|---|---|---|
| **M-1 (주 변수)** | **조건부 균형 img 마스킹** — 학습 배치의 FRAC(0.5)에서 img 입력 0 → fusion이 lidar/event로 풀도록 강제. 커리큘럼 ramp(WARMUP_EP 20). **추론은 항상 full-modality** | MCRM 2603.17705(균형 마스킹 +0.65 frozen-PEFT) | ✅ 구현(`model.py::_p42_mask_img`, config `P42.MASK_IMG`) |
| **M-2** | per-modal aux deep-sup CE — **P38에 이미 존재**(`FUSION.AUX_CE_WEIGHT 0.5`, fusion.py:586) | MLE-SAM 2412.04220(MUSES +4.7) | ✅ 기존 |
| M-3 (2차) | hard-pixel aux 집중 — fused 틀린 픽셀에서 per-modal aux CE 가중 | MCRM 핵심 | ⏸ M-1 무이득 시 추가 토글 |

**반증 회피**: RCA(P40 추론감쇠·유해)❌ — M-1은 학습시만·추론 full. 무조건 dropout(P33 실패)❌ — 균형 분할·ramp. fusion-rank(P41)❌. 외부신호❌. zero-init 잔차❌ — M-1은 입력 마스킹, M-2는 주손실 aux.

## 2. 게이트 (사전등록·falsifiable)

- **① drop-modality dMIoU(lidar) 상승** — 미사용→사용의 직접 지표(fog 분석과 동일 측정). ≈0에서 유의미 양수로.
- **② fog mIoU 개선** — fog 62.67 → 목표 상위기법 갭(−6) 수준 = ~+7pt fog.
- **③ val ≥ P38 82.22** — ep30 동에폭 P38 이상 유지(P41처럼 조기 판정).
- **falsify**: dMIoU(lidar) 안 오르면 M-1이 lidar를 못 살림 = 재설계 / dMIoU↑인데 mIoU 무이득 = "lidar 정보가 task에 무용" = MUSES 천장 근접(정직한 종결).
- 공정성: physaug 정합·val-best·radar 미포함.

## 3. 정직한 상한 (fog 딥리서치)

현실적 이득 = **+1~2pt(val, fog-국소)**지 SOTA 점프 아님(test SOTA GtA 82.39=카메라단독). 단 **논문 서사 강력**: "멀티센서 융합이 카메라단독을 이기는 조건은 fog뿐(74.12 vs 72.64)" = 융합 존재이유를 fog로 입증. user SOTA 목표엔 이 레버가 유일하게 근거 있는 길.

## 4. 실행

1. 판별(lidar 단독 mIoU ≫ trivial) 확정 → 코드검수(fresh-eyes + 스모크 grad/등가) → **A100 기동**(hpca100, BS2 DDP, ep30 게이트).
2. ablation: M-1 on/off(주 변수) + FRAC sweep(0.3/0.5) + (2차)M-3.

**근거 arXiv**: 2603.17705 · 2412.04220 · 2509.10408 · 2411.17141 · 2108.05249.
