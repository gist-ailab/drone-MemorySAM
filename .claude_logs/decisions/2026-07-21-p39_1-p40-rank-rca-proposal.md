---
created: 2026-07-21
scope: P39.1(즉시 수리) + P40(신모델 RCA-Fusion) 제안 — P39-MUSES 표준분석(2026-07-21)·fognight 원인규명(07-20) 기반, 관련연구 딥리서치 3편(rank collapse / modality imbalance / fog 물리) 교차 검증
gates: DELIVER = P36 fair 67.74/55.62 + thin-class · MUSES = P38 val 82.22/test 79.025 + fog_night ≥74
---

# P39.1 + P40 제안 — 2단 실패(저rank 압축 + 융합 미활용)의 문헌 기반 처방

## 0. 진단 ↔ 문헌 대응 (딥리서치 결과 요지)

| 우리 실측 | 문헌 근거 | 함의 |
|---|---|---|
| lidar rank 4.7 붕괴, adapter가 압축 주체(feat_cos 0.115), V1 선형 투영 유력 | **선형 cascaded 경로의 암묵적 저rank 편향** (deep matrix factorization 1905.13655, DirectCLR 2110.09348) — V1 선형 P_m + LoRA BA가 정확히 그 구조. LoRA "intruder dimensions"(2410.21228)가 feat_cos 0.115의 기제, **r 상향이 완화**하되 **rsLoRA(α/√r, 2312.03732) 없이는 무효** | V1을 비선형·게이트로 바꾸면 붕괴 압력 제거, VICReg류로 복원 |
| 카메라 편중(clear_night img 의존 24.3), fog_night에서 lidar 대체 실패(6.33→1.24) | **modality laziness/imbalance** 문헌 전체(OGM-GE, PMR, MMPareto…)가 같은 병리. 단 gradient-modulation 계열은 **frozen backbone에서 지렛대 없음** — 입력/손실 레벨만 유효. 무조건 드롭아웃은 **역효과까지 실증**(2403.04245) = 우리 P33 no-op 재현 | 신뢰도 **조건부** 강모달 감쇠 + lidar 보조 손실이 유일한 full-strength 레버 |
| fog 62.4가 최약, night는 강함 | **필드와 역전**: 문헌 fog 결손은 −3~−5(우리 −12.7), night가 필드 최약. **fog에서 가장 죽는 센서는 lidar 자신**(MUSES: fog 인스턴스 22.3% 리턴 0개) — 카메라가 여전히 최대 캐리어(CAFuser fog-night RGB ~48%) | 초과 결손 8~9pt = 헤드룸 실재. 단 **fog val split per-scene 감사 선행**(소표본/파국장면 판별). "lidar로 fog를 다 메운다"는 물리적 상한 있음 — 목표는 P38 수준(74) 복원 + α |
| 조건부 드롭아웃 노벨티 | "adaptive modality dropout" 자체는 선행: **OPM**(T-PAMI'24 2410.11582, 배치 레벨·라벨 유도·분류), **SGMA**(2603.02505, 샘플링 빈도 조건화) | **미점유 조합**: ①자기-추정 per-sample 신뢰도 신호 ②강모달 입력 감쇠를 예측된 실패 조건에 표적 ③dense prediction ④frozen-backbone 제약 논증. "first adaptive dropout" 주장 금지, 위 4축 조합으로 포지셔닝 |

## 1. P39.1 — 즉시 수리 (rank 복원, 주 변수 1개)

**동결**: V2(modal-token attention)·V3(앵커)·V4(쿼터)·router 직접감독·deep-sup. **M-2 동반**: gate/calib config off (fog_night 유해 실증, ablation 행 분리).

| # | 변경 | 근거 | 시작 하이퍼 |
|---|---|---|---|
| **R-1 (주 변수)** | V1 교체: `fused += P_m(f_m)` → **`fused += tanh(γ_m)·MLP_m(f_m)`**, MLP=LN→1×1(1024→256)→GELU→1×1(→1024), γ zero-init | V1의 night +2.50 기여는 보존하되 선형 지름길의 저rank 편향 제거 (2110.09348, ReZero/LLaMA-Adapter 게이트) | hidden 256 |
| **R-2** | **VICReg var+cov 정규화**를 per-modal 토큰에 (lidar ×1.0, img/event ×0.25) | 붕괴 "복원"용 (VICRegL 2210.01571, Shuffled-DBN 스펙트럼 복원 실증) | λ_var 0.1, λ_cov 0.01, 토큰 2048 서브샘플, fp32, per-GPU |
| **R-3 (조건부 2차)** | ep30 게이트 미달 시: **전모달 r 8→16 + rsLoRA α/√r** + AdaLoRA 직교항 0.1 | intruder-dim 완화(2410.21228); rsLoRA 없인 무효 | — |

**ep30 판정 게이트**: `feature_stats`로 **lidar rank ≥15** & fog_night drop-lidar **≥4.0** (fognight 문서 M-1 게이트 유지). 미달 시 R-3 적용 재기동, R-1/R-2 무효면 V2 원인설로 전환.
**선행 분석(학습 전, 분석 세션 위임)**: ① MUSES fog val split per-scene 감사(파국 장면 vs 균일 저하 — night>clear 역전이 문헌 전체와 반대라 소표본 아티팩트 배제 필요) ② P39 ckpt에서 trunk_exp off 시 lidar rank 재측정(V1 원인 확정).

## 2. P40 — RCA-Fusion (Reliability-Conditioned Attenuation) : 신모델·논문 주장 모듈

**서사**: 신뢰도 기계가 5세대 동안 *추론-시 재가중*으로는 무효(bias/gate/CEFR 반증 완주)였다. P40은 같은 신호를 **학습-시 조건화**로 옮긴다 — "모델 스스로 카메라가 나쁘다고 추정하면, 그 샘플은 카메라 없이 풀 수 있어야 한다"는 자기-일관성 루프. 외부 신호 0 유지.

| 구성 | 내용 |
|---|---|
| **C-1 신뢰도 신호 확장** | 기존 rel_cal + **lidar 리턴 유효성 통계**(입력에서 유도한 per-region density/zero-return 맵 — 내부 신호). CAFuser(전역 CLIP condition token)·DGFusion(depth 값 감독)과 구조적 구별: per-region·물리 유도·무감독 |
| **C-2 조건부 감쇠 학습** | 학습 중 per-sample로 카메라 신뢰도가 배치 하위 분위(예: 30%)면 확률 p로 img 입력을 **soft 감쇠**(α∈[0.1,0.5] 스케일, hard-zero 금지 — 2403.04245의 "missing 지름길" 역효과 회피). curriculum ramp(ep20까지 0→p_max 0.5), floor 유지 |
| **C-3 약모달 readout 보조 손실** | 감쇠된 샘플에 **lidar(+event) 단독 보조 seg 손실**(UMT식 readout) — 감쇠만으로는 fusion이 "저카메라 모드 암기"로 빠질 수 있어 gradient 출구 필요 (R1 리서치 권고) |
| **C-4 사전 검증 게이트** | 신뢰도 추정기 자체가 카메라 편중이면 무조건 드롭아웃으로 퇴화 — 학습 전 fog_night에서 rel AUROC(img) ≥0.75 확인(P39 0.70/P38 0.79), 미달 시 C-1 통계 신호를 주 신호로 |
| 부수(비주장) | fognight 스펙 M-3(Λ 온도 배타화)·M-4(앵커 클래스 균형 가중)·M-5(ckpt 복합 선택) 동반, ablation 분리 |

**노벨티 포지셔닝(정직)**: "reliability-conditioned modality attenuation for condition-robust dense fusion". 최근접 = OPM(T-PAMI'24)·SGMA — 차별 4축(자기추정 per-sample / 강모달 입력 감쇠의 조건 표적화 / dense / frozen-VFM 제약 논증). 지지 증거 = 자체 P33 무조건 드롭아웃 no-op + 2403.04245(무조건 드롭아웃 역효과) + 20% 균일 드롭은 CAFuser/DGFusion도 사용(공정선 내, 조건화가 delta). 관련연구 신규 편입 필요: 2505.22483(ICML'25 multimodal representation collapse)·2511.06450(rank-targeted fusion) — 우리 rank 스토리의 인접 선행.

**게이트(사전 등록)**: MUSES test ≥79.025 & **fog_night ≥74**(P38 복원) 우선, 이후 fog 전체 66~69(초과결손 절반) 도전 — 물리 상한 감안한 현실 목표(R3 리서치). DELIVER = P36 fair + thin-class 유지.

## 3. 실행 순서

1. **분석 선행 2건**(fog per-scene 감사, trunk_exp-off rank 재측정) — 학습 0, 기존 ckpt
2. **P39.1 구현·투입**(R-1+R-2, M-2 동반; 주 변수 R-1) — jarvis/hpca100 첫 슬롯, ep30 게이트
3. **P40 구현 병행**(C-1~C-4) — P39.1 rank 게이트 통과 확인 후 투입 (rank가 죽은 채면 C-3 lidar readout이 헛돎)
4. ablation 표: R-1/R-2/R-3/C-2/C-3 개별 토글 + 무조건 드롭아웃 대조행(우리 반증 재사용)
