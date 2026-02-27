# 프로젝트 현황 (Project Status)

> 최종 업데이트: 2026-02-27

## 현재 상태: P9가 최선 모델 (M=81.47), P15 설계 완료 (Calibrated Spatial Entropy Fusion)

### P14 실험 결과 (2026-02-27 완료)

- **M-score: 74.27** (P9: 81.47 대비 **-7.20, 심각한 하락**)
- Submission #16062. Checkpoint: `night_epoch47_90.75_top1_checkpoint.pth`
- **Per-class Test**: Static 62.57 / Dynamic 22.87 / Water 92.92 / **Sky 36.47**
- **Sky collapse**: 73/200 프레임 <10%, 56/200 프레임 <1%
- **핵심 문제**: LiDAR UAMM=1.000 고정 (200장 전부) + RGB 억제 (UAMM 0.555)
- **Aux mask 품질**: P13 대비 개선됐으나 여전히 GT 대비 매우 부정확. 모달리티 간 비교 불가 수준
- **CRM/ZERO 제거 효과**: hardaug5에서 제거했으나 Sky collapse 여전 → ISSUE-007은 부분 원인
- **교훈**: image-level scalar fusion의 근본 한계. Spatial-wise 접근 필요 (P15)

### P13 실험 결과 (2026-02-26 완료)

- **M-score: 81.21** (P9: 81.47 대비 -0.26)
- Submission ID: 15997 (epoch17), **16044 (epoch39 — test crash)**
- Checkpoint: `night_epoch17_87.71_checkpoint.pth` (night-val 기준 선택)
- **Dynamic IoU: 27.41** (P9: 21.86, **+5.55 개선** — 가장 큰 단일 클래스 개선)
- Static -1.49pp, Sky -1.42pp 하락 → Dynamic 개선을 상쇄
- Val mIoU: 92.45 (P9: 93.32, **-0.87 하락**) → M-score에서 불리
- **설계 목표 달성**:
  1. Expert collapse 해결: **실패** (17.4%, P12와 동일 수준)
  2. Energy Score fusion: **부분 성공** (UAMM 변동성 5-22x 증가, 하지만 test LiDAR 고정)
- 상세 분석: `.claude_logs/06_result_analysis_P13.md`

#### P13 Epoch39 Test Crash (Submission #16044)

- **Night-val: 89.53** (epoch17: 87.71, +1.82 개선)
- **Test mIoU: 50.48** (epoch17: 69.98, **-19.50pp 폭락**)
- **M-score: 71.67** (epoch17: 81.21, -9.54pp)
- **원인**: CRM/ZERO overfitting — Night Aug의 exact-zero 마스킹 패턴에 과적합
  - Sky 클래스 -51.76pp 붕괴 (crash의 67%), 80/200 프레임에서 Sky IoU=0
  - Night-val에도 CRM/ZERO 적용 → 오염된 proxy로 checkpoint 선택
  - 학습 샘플 44%에 exact-zero 패턴 → 실제 test에는 없는 artifact
- **교훈**: Night-val은 CRM/ZERO 제거 후 사용해야 신뢰 가능한 test proxy
- 상세 분석: `.claude_logs/06_result_analysis_P13.md` §10

### P12 실험 결과 (2026-02-25 완료)

- **M-score: 80.80** (P9: 81.47 대비 -0.67)
- Submission ID: 15949
- Dynamic IoU: 25.27 (P9: 21.25, **+4.02 개선**)
- Sky IoU: Test에서 P9 대비 **-6.81pp 하락** → 전체 M-score 하락의 주원인
- Expert collapse가 P9보다 심화 (Block0 lidar 단일 expert 독점)
- 상세 분석: `.claude_logs/05_result_analysis_P9_P12.md`

### Night Augmentation 포화 판정 (2026-02-26)

- P8 동일 아키텍처에서 4가지 aug 변종 실험: basic-aug → best hardaug 차이 **+1.43pp만**
- no-aug → basic-aug: **+26.57pp** (전체 gain의 80%)
- **Aug 튜닝은 포화 상태. M=85 달성에는 +7.4pp 필요하나 aug로는 +1~2pp가 한계.**
- 병목 클래스: Dynamic(gap -38pp), Sky(gap -21pp) — 전역 밝기 변환으로 해결 불가
- **필요한 접근**: Diffusion 기반 night 합성, TTA, Ensemble 등 근본적으로 다른 방법

### 최선 모델

- **P9 hardaug4**: M-score **81.47** (Challenge 최고)
- Checkpoint: `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug4/MULTIAQUA_CMNeXt-B2_ilt/epoch47_94.18_checkpoint.pth`
- Val mIoU: 93.32, Test mIoU: 69.62

### 진행 타임라인

1. **P8 (기본 Soft-MoE LoRA + ConfidenceHeadV2)**
   - 5가지 실험 완료 (no-aug, basic-aug, hardaug, hardaug2, hardaug3)
   - 최선: P8 hardaug(기본) M=78.45 → hardaug3 M=77.46 (test 성능 불안정)
   - 문제 발견: sigmoid saturation → UAMM 점수 항상 ~1.0, AMF 항상 ~1/3 uniform

2. **P9 (CrossModalFusionHead + max-norm UAMM)**
   - P8의 sigmoid 문제 해결: cross-modal relative comparison + softmax
   - hardaug4 적용, M-score **81.47** 달성 (P8 대비 +3.0)
   - Test mIoU 69.62 → P8 대비 크게 향상
   - **현재까지 최선 모델**

3. **P10 (CrossModalFusionHeadV2 + ModalAuxHead + oracle KL loss)**
   - 취소: 복잡도 증가가 test generalization을 악화시킴 (M=79.27)

4. **P11 (P10 + MI routing loss)**
   - 취소: loss 추가가 해결책이 아님 (M=77.09)

5. **MoE Gate 진단 (diagnose_moe_gate.py)**
   - 핵심 발견: MoE gate는 실제로 분화되어 있음! "uniform"은 공간 평균의 측정 artifact

6. **P12 (Input-Conditioned Soft MoE LoRA)**
   - Dynamic +4.02pp 개선했으나 Sky -6.81pp 하락. M=80.80 (P9 미달)

7. **P13 (Energy Score Fusion + Expert Collapse Fix)**
   - Dynamic +5.55pp 개선. Energy Score fusion으로 UAMM 변동성 증가.
   - 하지만 val -0.87pp 하락으로 M=81.21 (P9 미달)
   - Expert collapse 해결 실패 (collapse rate 동일)

8. **P14 (Per-Modality Separate Aux Decoders + hardaug5)**
   - ConfidenceAuxHead×1(공유) → ModalAuxDecoder×3(독립) + CRM/ZERO 제거
   - **M=74.27 — P9 대비 -7.20 심각한 하락**
   - Sky IoU 36.47%, LiDAR UAMM=1.0 고정, RGB 억제 (0.555)
   - Aux mask 품질 여전히 불충분 → Energy Score 신뢰도 낮음
   - Image-level scalar fusion의 근본 한계 확인 → P15 동기

9. **P15 (Calibrated Spatial Entropy Fusion) — 설계 완료, 구현 대기**
   - P12~P14 실패 분석에서 도출된 4가지 수정사항 통합:
     1. `.detach()` gradient 격리 (ISSUE-008)
     2. Energy Score → Calibrated Entropy (ISSUE-009: "confident but wrong" 해결)
     3. Spatial-wise `(B, m, H, W)` 가중치 (ISSUE-004)
     4. Aux Warmup Schedule (초기 N epoch uniform → 이후 활성화)
   - Baseline(단순평균) < P9(학습된상수) 확인 → UAMM/AMF 개념 유효
   - P9 val/test 가중치 완전 동일 (std≈0) → "학습된 상수"이므로 적응형 개선 여지 있음
   - P13이 낮/밤 적응 실제 수행 (img AMF: 0.404→0.289) → 방향 유효, 정확도가 병목
   - 상세 설계: `.claude_logs/02_model_arch.md` P15 섹션

### 미해결 과제

1. **🟡 ISSUE-007: CRM/ZERO Overfitting**: hardaug5에서 제거 완료. 하지만 Sky collapse 여전 → 부분 원인에 불과
2. **M=85 목표**: 현재 81.47 → +3.53pp 필요. Night Aug 포화로 새로운 접근 필수
3. **Val vs Test 갭 (93% vs 70%)**: 야간 test에서의 성능 저하가 여전히 핵심 문제
4. **Dynamic 클래스 IoU**: Test에서 21-27%. Gap -38pp이 가장 심각한 병목
5. **TTA (Test Time Augmentation)**: val_multiaqua_P9.py에 `--tta` 플래그 추가됨, **효과 미검증**
6. **P13 best day-val checkpoint test 재평가**: epoch14(93.48)로 M-score 역전 가능성 확인 필요
7. **Diffusion 기반 Night 합성** (ISSUE-005): M=85 도달을 위한 최유력 접근, 미구현
8. **Ensemble**: P9(Sky 우세) + P13(Dynamic 우세) 상보성 활용 가능
9. **🔴 ISSUE-009: Energy Score "confident but wrong"**: P15에서 calibrated entropy로 교체 예정

### 중요 발견사항

- **🔴 CRM/ZERO Overfitting 발견**: 학습 44%에 exact-zero RGB → test에는 없는 shortcut 학습. P13 epoch39에서 test -19.5pp crash, Sky -51.76pp. Night-val도 오염됨 (CRM/ZERO 동일 적용)
- **NIGHT_AUG hardaug4가 최적이나 포화 상태**: 추가 튜닝으로는 +1~2pp가 한계
- **MoE gate는 정상 작동**: spatial mean이 uniform처럼 보이는 것은 CLT artifact
- **모델 복잡도 ≠ 성능**: P10/P11/P12/P13 모두 P9보다 복잡하지만 M-score는 낮음
- **P9의 단순한 CrossModalFusionHead가 가장 효과적**: near-constant이지만 좋은 기본 비율
- **LiDAR routing 야간 고정**: 모든 P 버전에서 공통. LiDAR 데이터의 물리적 한계 (물 반사 없음, 원거리 미감지)
- **Energy Score fusion 방향은 유효**: Dynamic +5.55pp 개선. aux head 정확도가 관건
- **Night-val checkpoint 선택**: CRM/ZERO 제거 후에만 신뢰 가능한 test proxy
- **P14: Aux decoder 독립화만으로는 불충분**: 모달리티별 head 분리해도 frozen backbone feature 기반이라 mask 품질 한계. image-level scalar fusion은 Sky/Water 등 영역별 차이 반영 불가 → P15 spatial-wise가 필수
- **P9 CrossModalFusionHead = 학습된 상수**: val(낮)/test(밤) 345장에서 UAMM/AMF std≈0.0000. 입력에 따라 변하지 않는 고정 비율 (img:27.5%, lidar:35.5%, thermal:37.0%). SAM2 memory attention이 implicit adaptation 수행
- **Baseline(단순평균) < P9(UAMM/AMF)**: UAMM/AMF 개념의 가치 확인. 에너지 스코어 정확도 개선이 올바른 방향
- **Energy Score = confidence, not correctness (ISSUE-009)**: logit magnitude 기반 → "자신있게 틀리는" LiDAR에 높은 점수 → Sky collapse. Calibrated entropy가 대안
