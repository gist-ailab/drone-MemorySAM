# 프로젝트 현황 (Project Status)

> 최종 업데이트: 2026-03-01

## 현재 상태: P9가 최선 모델 (M=81.47), P19 구현 완료 (학습 대기)

### 유틸: SAM2 Thermal 전체 마스크 인퍼런스 (2026-03-01)

- **스크립트**: `run_sam2_thermal_masks.py` — SAM2 vanilla(automatic mask generator)로 thermal 이미지 전체에 대해 segmentation 마스크 생성.
- **입력**: MULTIAQUA thermal_camera 폴더 (또는 임의 thermal 이미지 폴더).
- **출력**: `out_dir/tmp/`: 원본 마스크 npz; `out_dir/result/`: 입력과 동일 파일명의 시각화 마스크 PNG + `*_concat.png` (thermal|mask 이어붙임).
- **실행**: `conda activate MMSS_SAM` 후 `python run_sam2_thermal_masks.py --thermal_dir /path/to/thermal_camera [--out_dir ./output_thermal_sam2]`.

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

13. **P19 (Learned Spatial Cross-Modal Fusion) — 구현 완료, 학습 대기**
    - P9 base + SpatialCrossModalFusionHead (multi-scale FPN + DWConv)
    - (B,m) scalar → (B,m,H,W) spatial 가중치. Aux decoder 없음
    - P13~P18의 aux-dependent fusion 대신 backbone feature에서 직접 학습
    - ~8.5M trainable (P9과 동일 수준)

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

### 실험 우선순위 (실행 순서)

| 순서 | 실험 | 서버 | 예상 소요 | 목적 |
| --- | --- | --- | --- | --- |
| 1 | **P9+hardaug5** | levine | ~12h | Aug ablation (빠르게 판별) |
| 2 | **P9+hardaug6** | levine | ~12h | Diversity aug 효과 검증 |
| 3 | P17+hardaug6 | levine/bengio | ~12h | Diversity가 P17에도 도움되는지 (조건부) |
| 4 | **P19** | levine | ~12h | Learned spatial fusion (P9 base) |
| 5 | P18-A | bengio | ~15h | ResNet aux backbone 효과 검증 |
| 6 | P18-B | bengio | ~15h | ResNet + entropy fusion (조건부) |

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
- **다음 방향**: P9 기반 점진적 개선 (hardaug5 재학습, TTA, P9+P13 ensemble)
