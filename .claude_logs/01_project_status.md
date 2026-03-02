# 프로젝트 현황 (Project Status)

> 최종 업데이트: 2026-03-03

## 현재 상태: Night2 기반 P9/P17/P19 config 생성 완료, levine 학습 대기

### Night2 실험 Config 생성 완료 (2026-03-03)

- P9/P17/P19 × (train + eval) = **6개 config 생성 완료**
- 공통: `ROOT→MULTIAQUA_night2`, `NIGHT_TRANSLATION: true`, hardaug4, levine 서버
- 학습 명령어:
  - `python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P9_hardaug4_night2.yaml`
  - `python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P17_hardaug4_night2.yaml`
  - `python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P19_hardaug4_night2.yaml`

### 유틸: SAM2 Thermal 전체 마스크 인퍼런스 (2026-03-01)

- **스크립트**: `run_sam2_thermal_masks.py` — SAM2 vanilla(automatic mask generator)로 thermal 이미지 전체에 대해 segmentation 마스크 생성.
- **입력**: MULTIAQUA thermal_camera 폴더 (또는 임의 thermal 이미지 폴더).
- **출력**: `out_dir/tmp/`: 원본 마스크 npz; `out_dir/result/`: 입력과 동일 파일명의 시각화 마스크 PNG + `*_concat.png` (thermal|mask 이어붙임).
- **실행**: `conda activate MMSS_SAM` 후 `python run_sam2_thermal_masks.py --thermal_dir /path/to/thermal_camera [--out_dir ./output_thermal_sam2]`.

### P19 / P9+hardaug6 실험 결과 (2026-03-03 완료)

- **P19 hardaug5 M=69.63** (P9 대비 **-11.84**, P16급 최악)
  - SpatialCrossModalFusionHead: multi-scale FPN + DWConv spatial softmax `(B,m,H,W)`
  - Sky IoU **3.77%** (169/200 프레임 Sky=0) — 학습된 spatial fusion이 LiDAR 편향 수렴
  - AMF: lidar=0.403 (P9: 0.355) → P9의 thermal 우세 균형 파괴
  - Submission #16313
- **P9 hardaug6 M=75.95** (best: epoch20, P9 hardaug4 대비 **-5.52**)
  - Broader augmentation(BRIGHTNESS [0.01,0.60], GAMMA [0.20,1.50]) 전략 실패
  - Sky IoU: epoch20=56.87, epoch85=39.90 (학습 길수록 Sky 하락)
  - Submission #16339 (ep85), #16340 (ep20)
- **핵심 발견**:
  1. **CRM/ZERO는 P9에 유익**: aux decoder 없으므로 shortcut 문제 없음, multimodal 강제 학습에 도움
  2. **Broader aug ≠ Better**: test에 없는 조건(밝은 야간, gamma>1)에 capacity 낭비
  3. **학습 가능 fusion은 계속 실패**: P19의 spatial fusion도 P12~P17과 동일 패턴 (LiDAR 편향 → Sky 붕괴)
  4. **Early stopping 중요**: epoch20 > epoch85 on test (Sky -16.97pp 차이)

### P16/P17 실험 결과 (2026-02-27 완료)

- **P16 M=68.42 (역대 최악)**, P17 M=73.23 (부분 회복, 여전히 P9 대비 -8.24)
- **P16**: Calibrated entropy + 4 Fixes 통합. Sky IoU **3.17%** (157/200 프레임 Sky=0). Thermal UAMM=0.923으로 지배.
- **P17**: Multi-Scale FPN Aux Decoder (fpn[0,1,2] 352ch). Sky **33.35%** (+30.18 vs P16). 변동성 2x 증가.
- **핵심 결론**: P9 이후 모든 adaptive fusion이 P9의 고정 상수보다 나쁨
  - P12(-0.67), P13(-0.26), P14(-7.20), P15(-10.42), **P16(-13.05)**, P17(-8.24)
  - Aux mask 품질 부족 → entropy/energy 추정 무의미 → thermal 편향 → Sky 붕괴
  - SAM2 memory attention이 이미 implicit cross-modal adaptation 수행 → 외부 dynamic fusion 불필요?
- **다음 실험 방향**: P9 기반 점진적 개선 (A1: P9+hardaug5, A2: TTA, A3: Ensemble)

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

9. **P15 (Spatial Energy Fusion) — 역대 최악 M=71.05**
   - Spatial-wise `(B, m, H, W)` 가중치 + Energy Score 기반
   - 설계 4가지 Fix 중 Fix 3(spatial-wise)만 단독 적용 → ❌ **오히려 악화**
   - **M=71.05** (P9 대비 -10.42, P14 대비 -3.22) — Submission #16087
   - **Sky IoU 16.66%**: 111/200 프레임 Sky=0% (P14: 36.47%, P9: 76.54%)
   - **"Spatial Amplification"**: 부정확한 energy를 pixel-level로 전파 → noise 증폭
   - UAMM 적응 자체는 작동 (img -21%, lidar +15%), LiDAR 1.0 미고정
   - **하지만** aux mask 부정확 + energy 부정확 + no-detach → spatial이 피해만 증폭
   - Checkpoint: epoch46 (day-val best). Night-val best(epoch45) 미평가

10. **P16 (Calibrated Spatial Entropy Fusion) — 실험 완료, M=68.42 (역대 최악)**
    - 4가지 Fix 통합: detach, calibrated entropy, spatial, warmup
    - **Sky IoU 3.17%** (157/200 프레임 Sky=0), thermal UAMM=0.923 지배
    - Submission #16106

11. **P17 (Multi-Scale FPN Aux Decoder) — 실험 완료, M=73.23 (P16 대비 +4.81)**
    - P16 + MultiScaleModalAuxDecoder (fpn[0,1,2] 352ch)
    - Sky 33.35% (+30.18 vs P16), UAMM 변동성 2x 증가
    - 하지만 P9(81.47) 대비 여전히 -8.24. Dynamic fusion 방향의 한계 확인
    - Submission #16107 (night_ep35), #16108 (ep28)

12. **P18 (Trainable ResNet-18 Aux Backbone) — 구현 완료, 학습 대기**
    - P17 기반 + ResNet-18 aux backbone. P18-A(scalar), P18-B(entropy) 두 변형
    - ~20M trainable (ResNet-18 ~11.2M 추가)

13. **P19 (Learned Spatial Cross-Modal Fusion) — 실험 완료, M=69.63 (실패)**
    - P9 base + SpatialCrossModalFusionHead (multi-scale FPN + DWConv)
    - Sky IoU 3.77%, LiDAR 편향 수렴 (AMF lidar=0.403)
    - Submission #16313

14. **P9+hardaug6 (Diversity Augmentation) — 실험 완료, M=75.95 (실패)**
    - P9 아키텍처 + 넓은 범위 augmentation [0.01, 0.60]
    - Sky 56.87(ep20)/39.90(ep85). Broader aug가 역효과
    - Submission #16339, #16340

### 다음 실험 계획

#### 실험 A: P9 + hardaug5 (Aug Ablation)

- **목적**: hardaug4 vs hardaug5 영향 분리. P14~P17 하락이 아키텍처인지 augmentation인지 판별
- **필요 파일**:
  1. `configs/levine-multiaqua_rgbtl_P9_hardaug5.yaml`
  2. `configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug5.yaml`
- **LORA_MODEL**: `LoRA_Sam_P9` (아키텍처 변경 없음)
- **SAVE_DIR**: `./outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug5`
- **변경 내용**: P9 hardaug4에서 NIGHT_AUG만 hardaug5로 교체
  - CRM/ZERO 제거, NIGHT_SIM_P 0.45→0.60, BRIGHTNESS [0.03,0.45]→[0.02,0.20]
- **상태**: 계획 완료, 구현 대기

#### 실험 B: P9 + hardaug6 (Diversity Augmentation)

- **목적**: test 분포에 맞추는 대신, 훨씬 넓고 다양한 augmentation으로 robustness 극대화
- **가설**: hardaug3~5가 test 통계에 맞추려다 오히려 좁은 일반화 → 넓은 범위가 더 유리할 수 있음
  - 근거: hardaug3(실측정렬)이 hardaug2(넓은범위)보다 나빴음 (M 77.46 vs 78.37)
- **필요 파일**:
  1. `configs/levine-multiaqua_rgbtl_P9_hardaug6.yaml`
  2. `configs/eval_config/levine-multiaqua_rgbtl_P9_hardaug6.yaml`
- **LORA_MODEL**: `LoRA_Sam_P9` (아키텍처 변경 없음)
- **SAVE_DIR**: `./outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug6`
- **hardaug6 설계** (Broader Range + Diversity):

| 파라미터 | hardaug4 | hardaug5 | **hardaug6** | 변경 이유 |
| --- | --- | --- | --- | --- |
| NIGHT_SIM_P | 0.45 | 0.60 | **0.50** | 중간값, 주간 데이터도 충분히 보존 |
| BRIGHTNESS | [0.03, 0.45] | [0.02, 0.20] | **[0.01, 0.60]** | 매우 넓은 범위: 극저조도~약간 밝은 야간 |
| SAMPLING | dark_biased | dark_biased | **dark_biased** | 유지 |
| DARK_RATIO | 0.60 | 0.70 | **0.50** | 50%만 dark, 나머지 uniform → 다양성 확보 |
| DARK_RANGE | [0.03, 0.12] | [0.02, 0.06] | **[0.01, 0.10]** | 극저조도 포함하되 범위 넓게 |
| MODERATE_RANGE | [0.12, 0.45] | [0.06, 0.20] | **[0.10, 0.60]** | 밝은 야간도 포함 |
| CONTRAST | [0.3, 0.7] | [0.20, 0.65] | **[0.15, 0.85]** | 넓은 contrast 변동 |
| GAMMA | [0.4, 0.8] | [0.30, 0.75] | **[0.20, 1.50]** | gamma>1.0도 포함 (밝기 반전 효과) |
| NOISE_STD | 0.02 | 0.025 | **0.03** | 노이즈 다양성 증가 |
| CRM_P | 0.35 | 제거 | **제거** | CRM/ZERO overfitting 방지 (유지) |
| ZERO_P | 0.09 | 제거 | **제거** | |

- **핵심 철학**: "test 분포에 맞추지 말고, 모든 조건에서 robust하게 만든다"
- **상태**: 계획 완료, 구현 대기

#### 실험 C: P17 + hardaug6

- **목적**: hardaug6가 P9에서 효과적이면, P17에서도 aux mask 학습에 도움될 수 있는지 확인
- **가설**: 넓은 augmentation이 aux decoder에 더 다양한 학습 신호 제공 → entropy 추정 품질 향상
- **P17은 P9보다 augmentation 의존도가 높을 수 있음**: aux decoder가 다양한 brightness 조건을 경험할수록 각 모달리티의 entropy를 더 정확하게 추정
- **필요 파일**:
  1. `configs/levine-multiaqua_rgbtl_P17_hardaug6.yaml`
  2. `configs/eval_config/levine-multiaqua_rgbtl_P17_hardaug6.yaml`
- **LORA_MODEL**: `LoRA_Sam_P17`
- **SAVE_DIR**: `./outputs/MMSamP17/levine_multiaqua_rgbtl_P17_hardaug6`
- **우선순위**: P9+hardaug6 결과 확인 후 진행 (조건부)
- **상태**: 계획 완료, P9+hardaug6 결과 대기

#### 실험 E: P19 — Learned Spatial Cross-Modal Fusion

- **목적**: P9의 GAP 기반 스칼라 fusion을 학습 가능한 spatial fusion으로 교체
- **가설**: LiDAR 포인트 밀도, Thermal 패딩, RGB 밝기 차이 등 위치별 모달리티 퀄리티를 학습하면 per-location fusion 가능
- **아키텍처**: P9 base + SpatialCrossModalFusionHead (multi-scale FPN + DWConv + spatial softmax)
- **핵심 차이**: (B,m) scalar → (B,m,H,W) spatial, fpn[0] only → fpn[0,1,2], aux decoder 없음
- **파라미터 증가**: ~8K (23K fusion head vs 15K CrossModalFusionHead) — negligible
- **Config**: `configs/levine-multiaqua_rgbtl_P19_hardaug5.yaml`
- **상태**: **구현 완료, 학습 대기**

#### 실험 D: P18 — Trainable Aux Backbone (ResNet-18)

- **목적**: frozen SAM2 FPN feature 위 lightweight decoder의 aux mask 품질 한계 돌파
- **가설**: 2,952장 train data + ImageNet pretrain → ResNet-18이 MULTIAQUA 4-class 특화 feature 학습 가능
- **LORA_MODEL**: `LoRA_Sam_P18` (신규 클래스 필요)

**아키텍처 설계**:

```
Input (3ch RGB / 1ch LiDAR / 1ch Thermal)
  ├─→ SAM2 Hiera B+ (frozen) → backbone_fpn → memory attention → final prediction
  │                                     ↓ (fpn[0] for m_feat fusion)
  └─→ ResNet-18 (trainable, pretrained) → multi-scale features
                                            ↓
                             MultiScaleAuxDecoder → aux_logits
                                            ↓ (.detach())
                             compute_spatial_entropy_confidence → UAMM/AMF weights
```

**ResNet-18 적용 방식**:
- **Input adapter**: LiDAR(1ch)와 Thermal(1ch)는 3ch로 repeat 또는 별도 stem conv(1→64) 추가
- **Feature extraction**: ResNet-18의 layer2(128ch, 64×64) + layer3(256ch, 32×32) 사용
  - SAM2 FPN과 다른 스케일/채널 → 상호 보완적 정보
- **Aux decoder**: `MultiScaleModalAuxDecoder` 변형 — ResNet feature를 proj→concat→decode
- **학습**: ResNet-18은 aux CE loss로만 학습. Main pipeline은 P9과 동일 (고정 상수 또는 entropy)
- **핵심**: SAM2 main pipeline(tracking, memory attention, mask decoder)은 **전혀 변경 없음**

**두 가지 서브 옵션**:

| 옵션 | UAMM/AMF 방식 | 기대 효과 |
| --- | --- | --- |
| **P18-A**: ResNet aux + 고정상수 fusion | P9처럼 고정 비율 (entropy 미사용) | aux mask CE loss만으로 backbone fine-tune. UAMM은 P9 그대로 → 안전한 baseline |
| **P18-B**: ResNet aux + entropy fusion | P17처럼 spatial entropy → UAMM/AMF | 정확한 aux mask → 정확한 entropy → dynamic fusion 비로소 작동? |

**우선순위**: P18-A 먼저 (P9 고정상수 + ResNet aux로 backbone만 학습), 그 후 P18-B

**파라미터 수 추가**:
- ResNet-18: ~11.2M (ImageNet pretrained)
- 기존 trainable: LoRA ~700K + aux decoder ~159K
- **총**: ~12M trainable (기존 대비 15x 증가, 하지만 2,952장 × 200 epoch이면 충분)

**리스크**:
- ResNet-18이 주간 데이터에 과적합 → 야간에서 부정확한 aux mask → 역효과
- Mitigation: dropout, data augmentation (hardaug6), early stopping by night-val
- 추론 latency 증가 (ResNet-18 forward pass 추가)

**구현 파일** (2026-03-01 완료):

1. `sam_lora_image_encoder_seg.py` — `ResNetAuxBackbone`, `ResNetAuxDecoder`, `LoRA_Sam_P18` 클래스
2. `configs/levine-multiaqua_rgbtl_P18_hardaug5.yaml` (training)
3. `configs/eval_config/levine-multiaqua_rgbtl_P18_hardaug5.yaml` (eval)
4. `train_sam2_lora_paper.py` — `use_entropy_fusion` dispatch 추가
5. `val_multiaqua.py` / `val_multiaqua_detailed.py` — P18 지원 추가

- **상태**: **구현 완료, 학습 대기** (P18-A: `USE_ENTROPY_FUSION: false`)

---

#### 실험 F: P9 + hardaug4-noCRM (CRM/ZERO Ablation) ⭐ 최우선

- **목적**: CRM/ZERO가 P9 성능에 기여하는 정도를 정확히 분리 (Critical Ablation)
- **배경**:
  - P9+hardaug4(M=81.47)과 P9+hardaug6(M=75.95) 사이 **-5.52pp** 차이
  - hardaug6은 CRM/ZERO 제거 + brightness/gamma 범위 변경 **두 가지를 동시에 바꿈** → 원인 불명
  - P9에는 aux decoder가 없으므로 CRM/ZERO shortcut 문제 없음
  - 가설: CRM/ZERO가 P9에서 multimodal 강제 학습에 유익하다면, 이걸 제거하면 하락할 것
  - **이 ablation이 확인해야 할 것**: hardaug6 하락이 CRM/ZERO 제거 때문인지, 범위 변경 때문인지
- **설계**: hardaug4의 **모든 파라미터를 동일하게 유지**, CRM_P와 ZERO_P만 0.0으로 변경

| 파라미터 | hardaug4 (원본) | **hardaug4-noCRM** | 변경 여부 |
| --- | --- | --- | --- |
| NIGHT_SIM_P | 0.45 | 0.45 | 유지 |
| BRIGHTNESS_RANGE | [0.03, 0.45] | [0.03, 0.45] | 유지 |
| BRIGHTNESS_SAMPLING | dark_biased | dark_biased | 유지 |
| DARK_BIASED_RATIO | 0.6 | 0.6 | 유지 |
| DARK_RANGE | [0.03, 0.12] | [0.03, 0.12] | 유지 |
| MODERATE_RANGE | [0.12, 0.45] | [0.12, 0.45] | 유지 |
| CONTRAST_RANGE | [0.3, 0.7] | [0.3, 0.7] | 유지 |
| GAMMA_RANGE | [0.4, 0.8] | [0.4, 0.8] | 유지 |
| NOISE_STD | 0.02 | 0.02 | 유지 |
| **CRM_P** | **0.35** | **0.0** | **제거** |
| CRM_MASK_RATIO | [0.2, 0.5] | (미사용) | — |
| **ZERO_P** | **0.09** | **0.0** | **제거** |

- **필요 파일**:
  1. `configs/bengio-multiaqua_rgbtl_P9_hardaug4noCRM.yaml` (학습)
  2. `configs/eval_config/bengio-multiaqua_rgbtl_P9_hardaug4noCRM.yaml` (평가)
- **LORA_MODEL**: `LoRA_Sam_P9` (아키텍처 변경 없음)
- **SAVE_DIR**: `./outputs/MMSamP9/bengio_multiaqua_rgbtl_P9_hardaug4noCRM`
- **학습 하이퍼파라미터**: hardaug4와 완전 동일 (LR=0.0006, EPOCHS=200, DDP=True, BATCH_SIZE=1, WARMUP_EPOCHS=10)
- **서버**: bengio

- **예상 결과 해석**:
  - hardaug4-noCRM ≈ hardaug4 (M≥80) → CRM/ZERO 영향 미미, hardaug6 하락은 범위 변경 때문
  - hardaug4-noCRM ≪ hardaug4 (M≤78) → CRM/ZERO가 핵심 기여 요소, P9에서 유지해야 함
  - hardaug4-noCRM ≈ hardaug6 (M≈76) → CRM/ZERO 제거가 주된 하락 원인, 범위 변경은 무관
- **상태**: 계획 완료, 구현 대기

### Night2 데이터셋 기반 실험 우선순위 (2026-03-03)

#### 배경: img2img Day-to-Night Translation

- **데이터셋**: `MULTIAQUA_night2` — 기존 MULTIAQUA에 img2img 변환으로 생성한 야간 RGB 추가
- **경로**: `/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2`
- **구조**: `data/zed` (주간 원본 3,298장) + `data/zed_night` (야간 변환 3,298장)
- **Config 변경**: `DATASET.ROOT` → `MULTIAQUA_night2`, `NIGHT_TRANSLATION: true`
- **효과**: train 시 `zed` + `zed_night` 모두 로드 → **2,952 × 2 = 5,904 학습 샘플**
- **val/test는 원본 zed만 사용** (코드에서 `split == "train"` 일 때만 night_translation 적용)

#### 핵심 패러다임 전환

기존 실험의 근본 한계: **train=주간만, test=야간만** → 모든 모델이 야간 일반화 실패 (val 93% vs test 70%).

Night2 데이터로 **야간 RGB를 직접 학습**하면:
1. Backbone(SAM2 Hiera)은 frozen이지만, **LoRA adapter가 야간 feature 패턴 학습** 가능
2. Aux decoder가 **야간 이미지에서의 entropy/energy를 직접 경험** → 추정 정확도 향상
3. Fusion head가 **야간 조건에서의 modality 신뢰도를 직접 학습** → 정확한 가중치

이것은 P12~P19가 실패한 **직접적 원인**을 해결할 수 있음:
- **"Dynamic Fusion 실패"의 원인**: adaptive mechanism이 주간에서만 학습 → 야간에서 잘못된 가중치
- **Night2로 해결 가능성**: adaptive mechanism이 야간 패턴도 학습 → 정확한 야간 가중치

#### 우선순위 1: P9 + night2 (Baseline) ⭐⭐

- **근거**: 현재 최선 모델(M=81.47). Night2로 가장 안전한 성능 향상 기대
- **변경**: `ROOT` → night2, `NIGHT_TRANSLATION: true`, 나머지 hardaug4 동일
- **기대 효과**:
  - LoRA adapter가 야간 RGB feature를 직접 학습 → test mIoU 향상
  - CrossModalFusionHead는 여전히 "learned constant"로 수렴하겠지만, 야간 데이터 포함 학습으로 constant 비율 자체가 더 최적화
  - NIGHT_AUG와 night2가 상호보완: NIGHT_AUG는 brightness/gamma 다양성, night2는 realistic texture/structure
- **리스크**: 낮음. P9은 가장 안정적인 아키텍처
- **예상 향상**: test mIoU +3~8pp (M 83~86 가능)

#### 우선순위 2: P13 + night2 (Energy Score Fusion) ⭐

- **근거**: P9 대비 -0.26 (M=81.21)로 **가장 근접**. 실패 원인이 night2로 직접 해결됨
- **P13이 night2에서 P9을 넘을 수 있는 이유**:
  1. P13의 Energy Score fusion은 **방향이 맞았음** — 야간에 RGB↓ LiDAR↑ 적응 실제 수행
  2. 실패 원인: aux head가 주간만 학습 → **야간 LiDAR를 항상 "가장 confident"로 오판** (Sky에서 LiDAR가 Water로 확신있게 틀림)
  3. Night2로 aux head가 야간 패턴 학습 → **LiDAR의 Sky 오예측을 정확히 감지** → energy 낮음 → RGB 가중치 유지 → Sky 보존
  4. P9는 고정 상수로 야간 적응 불가. P13은 **야간에 맞는 dynamic 가중치 가능**
- **기대 효과**:
  - Aux mask 야간 품질 향상 → Energy Score 정확도 향상 → dynamic fusion 비로소 작동
  - Dynamic IoU: 기존 27.41 (P9 대비 +5.55)에서 추가 향상 가능
  - Sky IoU: LiDAR 맹신 해소 → P9 수준(76%) 회복 가능
- **리스크**: 중간. Energy Score "confident but wrong" 문제가 night2에서도 잔존할 수 있음
- **예상 향상**: P9+night2보다 +1~3pp 추가 가능 (M 84~88)

#### 우선순위 3: P17 + night2 (Multi-Scale FPN Entropy Fusion)

- **근거**: 가장 정교한 aux decoder (159K params, 3-level FPN). Night2로 aux mask 품질 극대화
- **P17이 night2에서 도약할 수 있는 이유**:
  1. P17은 P16 대비 Sky +30pp 개선 → multi-scale FPN의 효과 입증
  2. 하지만 M=73.23 (P9 대비 -8.24) — aux mask가 야간에서 여전히 부정확해서 entropy 오추정
  3. Night2로 **aux decoder가 야간 multi-scale feature의 entropy를 직접 학습** → calibrated entropy 정확도 급상승
  4. `.detach()` + warmup + spatial entropy — 4가지 Fix가 night2와 결합하면 비로소 의도대로 작동
  5. **P13 대비 장점**: multi-scale feature(352ch)가 야간에서 더 풍부한 정보 제공
- **리스크**: 중-높음. 복잡한 파이프라인(4-fix + aux + entropy)이 여전히 불안정할 수 있음
- **예상 향상**: aux mask 야간 품질이 충분하면 P13 이상 가능, 불충분하면 P13 이하

#### 우선순위 4: P19 + night2 (Spatial Cross-Modal Fusion)

- **근거**: Aux decoder 없는 깔끔한 spatial fusion. Night2로 공간 가중치 학습 패턴 변화 기대
- **P19가 night2에서 회복할 수 있는 이유**:
  1. P19 실패 원인: SpatialCrossModalFusionHead가 **주간 패턴에서 LiDAR 편향으로 수렴** (AMF lidar=0.403)
  2. Night2에서: 야간 RGB가 학습에 포함 → fusion head가 "야간엔 RGB 신뢰도 낮음"을 직접 학습
  3. 주간 vs 야간에서 **다른 spatial 가중치**를 출력할 수 있음 → 상황 적응적 fusion
  4. Aux decoder 없음 → 파이프라인 단순, entropy 추정 오류 위험 없음
- **P9보다 나을 수 있는 시나리오**: 야간 특정 영역(ex: 수면 반사가 강한 곳)에서 RGB가 유리한 경우, spatial fusion이 per-pixel로 RGB 가중치를 올릴 수 있음. P9의 고정 상수로는 불가능
- **리스크**: 중-높음. LiDAR 편향 수렴이 night2에서도 반복될 가능성
- **예상 향상**: 수렴 패턴에 따라 크게 달라짐 (P9 이상 또는 여전히 이하)

#### 우선순위 5: P18-A + night2 (ResNet-18 Aux Backbone)

- **근거**: Trainable ResNet-18이 야간 도메인 특화 feature 학습 → 가장 높은 aux mask 품질
- **Night2와의 시너지**:
  1. ResNet-18(11.2M)은 ImageNet pretrain → night2로 야간 MULTIAQUA fine-tune
  2. 기존 우려: "주간에만 학습하면 야간 aux mask 부정확" → **night2로 직접 해결**
  3. 5,904장 × 200 epoch = 충분한 학습량 (기존 2,952장보다 overfitting 위험도 감소)
- **리스크**: 높음. 파라미터 20M으로 가장 크고, 학습 시간 ~15h+, 여전히 4-class 데이터 규모에 과적합 위험
- **예상 향상**: 성공 시 가장 큰 폭의 향상 가능 (aux mask 품질 급상승 → entropy fusion 최적 작동), 실패 시 overfitting

#### Night2 실험 매트릭스

| 순서 | 모델 | 아키텍처 특성 | Night2 기대 효과 | 리스크 | 왜 P9을 넘을 수 있나 |
| --- | --- | --- | --- | --- | --- |
| **1** | **P9** | 고정 상수 fusion | LoRA 야간 학습 | 낮음 | baseline, 안전한 향상 |
| **2** | **P13** | Energy Score adaptive | Aux mask 야간 품질↑ → 정확한 dynamic fusion | 중간 | 야간 dynamic adaptation |
| **3** | P17 | Entropy spatial + multi-scale | Aux decoder 야간 entropy 정확도↑ | 중-높 | 정교한 spatial 적응 |
| **4** | P19 | Learned spatial fusion | Spatial head 야간 패턴 학습 | 중-높 | Per-pixel 야간 가중치 |
| **5** | P18-A | ResNet-18 trainable aux | ResNet 야간 도메인 fine-tune | 높음 | 최고 aux mask 품질 |

#### Config 공통 변경 사항

모든 night2 실험에 공통 적용:
```yaml
DATASET:
  ROOT: '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night2'
  NIGHT_TRANSLATION: true    # zed + zed_night 모두 로드
  # 나머지 동일

# NIGHT_AUG: hardaug4 유지 (night2와 상호보완)
# — night2: realistic texture/structure
# — NIGHT_AUG: brightness/gamma diversity
```

- Config 이름 패턴: `{server}-multiaqua_rgbtl_{model}_hardaug4_night2.yaml`
- SAVE_DIR 패턴: `./outputs/MMSam{model}/{server}_multiaqua_rgbtl_{model}_hardaug4_night2`
- 평가 시: `ROOT`는 night2 유지 (val은 원본 zed만 사용하므로 결과 동일)
- `TEST.FILE`도 night2로 설정 (test도 원본 zed만 사용)

### 이전 실험 우선순위 (night2 이전, 참고용)

| 순서 | 실험 | 서버 | 목적 | 상태 |
| --- | --- | --- | --- | --- |
| ~~1~~ | P9+hardaug4-noCRM | bengio | CRM/ZERO ablation | 계획 완료 (night2로 대체 가능) |
| ~~2~~ | P9+hardaug5 | levine | Aug ablation | 계획 완료 |
| ~~3~~ | ~~P9+hardaug6~~ | ~~levine~~ | ~~Diversity aug~~ | **완료 (M=75.95, 실패)** |
| ~~4~~ | P18-A | bengio | ResNet aux backbone | 구현 완료 |
| ~~5~~ | ~~P19~~ | ~~levine~~ | ~~Learned spatial fusion~~ | **완료 (M=69.63, 실패)** |

### 미해결 과제

1. **🟡 ISSUE-007: CRM/ZERO Overfitting**: hardaug5에서 제거 완료. 하지만 Sky collapse 여전 → 부분 원인에 불과
2. **M=85 목표**: 현재 81.47 → +3.53pp 필요. Night Aug 포화로 새로운 접근 필수
3. **Val vs Test 갭 (93% vs 70%)**: 야간 test에서의 성능 저하가 여전히 핵심 문제
4. **Dynamic 클래스 IoU**: Test에서 21-27%. Gap -38pp이 가장 심각한 병목
5. **TTA (Test Time Augmentation)**: val_multiaqua_P9.py에 `--tta` 플래그 추가됨, **효과 미검증**
6. **P13 best day-val checkpoint test 재평가**: epoch14(93.48)로 M-score 역전 가능성 확인 필요
7. **Diffusion 기반 Night 합성** (ISSUE-005): M=85 도달을 위한 최유력 접근, 미구현
8. **Ensemble**: P9(Sky 우세) + P13(Dynamic 우세) 상보성 활용 가능
9. **🟢 ISSUE-009: Energy Score "confident but wrong"**: P16에서 calibrated entropy로 교체 완료 (구현됨, 학습 대기)

### 중요 발견사항

- **🔴 Dynamic Fusion 실패 확정**: P12~P17 6개 실험 모두 P9(고정 상수)보다 나쁨. 복잡한 adaptive fusion이 오히려 해로움
- **🔴 CRM/ZERO Overfitting 발견**: 학습 44%에 exact-zero RGB → test에는 없는 shortcut 학습. P13 epoch39에서 test -19.5pp crash
- **🔴 Sky collapse가 핵심 병목**: P14~P17에서 Sky IoU 3~36% (P9: 76%). Adaptive fusion이 RGB를 suppress → sky 인식 파괴
- **NIGHT_AUG hardaug4가 최적이나 포화 상태**: 추가 튜닝으로는 +1~2pp가 한계
- **P9의 단순한 CrossModalFusionHead가 가장 효과적**: 고정 비율 img:27.5%, lidar:35.5%, thermal:37.0%. SAM2 memory attention이 implicit adaptation 수행
- **Multi-Scale FPN은 효과 있음**: P16(3.17)→P17(33.35) Sky +30pp. 하지만 근본 해결은 아님
- **Aux mask 품질이 근본 한계** (ISSUE-008): frozen backbone 위의 lightweight decoder로는 entropy/energy 추정 신뢰도 확보 불가
- **CRM/ZERO는 P9에 유익**: aux decoder 없으므로 shortcut 없고, multimodal 강제 학습에 도움. P9+hardaug5(CRM/ZERO만 제거) ablation 필요
- **Broader aug 실패**: hardaug6 [0.01, 0.60] + gamma>1.0은 test에 없는 조건에 capacity 낭비
- **학습 가능 fusion 실패 확정**: P12~P19 8개 실험 전부 P9(고정 상수)보다 나쁨
- **다음 방향**: Night2 데이터셋 기반 재학습 (P9 → P13 → P17 → P19 → P18 순)
