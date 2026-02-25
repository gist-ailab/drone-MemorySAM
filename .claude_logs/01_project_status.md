# 프로젝트 현황 (Project Status)

> 최종 업데이트: 2026-02-25

## 현재 상태: P9가 최선 모델, P13 구현 완료 (학습 대기), Night-Val 평가 파이프라인 추가 완료

### 다음 단계: P13 학습 실행

- 설계 가이드: `.claude_logs/P13_design_guide.md`
- Config: `configs/levine-multiaqua_rgbtl_P13_hardaug4.yaml`
- 핵심 변경 2가지:
  1. CrossModalFusionHead → ConfidenceAuxHead + Energy Score 기반 fusion weight
  2. SoftMoE_LoRA_Layer experts_b zero-init → kaiming*0.01 (expert collapse 방지)
- P10/P11과 달리 fusion weight에 학습 가능 파라미터/GT supervision 없음
- P12는 스킵 (MoE gate는 이미 정상 작동, 모달리티 conditioning 불필요)

### P13 구현 상세 (2026-02-25 완료)

**변경 파일:**
- `sam_lora_image_encoder_seg.py` (line 2427~2743): `ConfidenceAuxHead`, `compute_energy_confidence()`, `LoRA_Sam_P13` 추가
- `configs/levine-multiaqua_rgbtl_P13_hardaug4.yaml`: 새 config (LORA_MODEL=P13, LAMBDA_AUX=0.3, LORA_NUM_CLASSES=4)
- `train_sam2_lora_paper.py`: lambda_aux, 3-tuple dispatch, P13 aux CE loss, pbar/TensorBoard 로깅
- `val_mm_sam.py`: wildcard import + config 기반 동적 모델 선택

**디자인 가이드 대비 의도적 차이:**
- experts_b init: `sam_lola_utils.py` 수정 대신 P13 `__init__`에서 직접 재초기화 (P9 체크포인트 호환성 유지)
- Aux loss: GT downsample 대신 logits upsample (P10/P11과 동일 패턴)

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
   - P9의 near-constant cross-modal weight 문제 해결 시도
   - Multi-pool (GAP+GMP+Std) + per-modality auxiliary segmentation으로 oracle 생성
   - Val mIoU 93.23 (P9과 유사)이지만 Test mIoU **65.30** (P9 대비 -4.3)
   - M-score 79.27 → **P9보다 나쁨**
   - hardaug3은 더 나쁨 (M=76.05)
   - **취소 결정**: 복잡도 증가가 test generalization을 악화시킴

4. **P11 (P10 + MI routing loss)**
   - MoE gate uniform 문제 해결 시도: Mutual Information loss 추가
   - Val mIoU 93.17, Test mIoU **61.01** → P10보다도 나쁨
   - M-score 77.09 → **P9 대비 -4.4**
   - **취소 결정**: loss 추가가 해결책이 아님

5. **MoE Gate 진단 (diagnose_moe_gate.py)**
   - 지도교수 피드백: "loss를 넣어볼게 아니라 왜 gating이 안되는지 분석이 먼저"
   - 4가지 가설 검증 (H1~H4)
   - **핵심 발견**: MoE gate는 실제로 분화되어 있음! "uniform"은 공간 평균의 측정 artifact
   - Per-token 분석: entropy_ratio=0.55, max_weight=0.72 → 결정적 routing 수행 중
   - Block9_Q: lidar→E2(83%), thermal→E0(84%)

6. **val_multiaqua_detailed.py (상세 분석 스크립트)** ← val_multiaqua_P9.py에서 리네임
   - P8~P13 모든 모델 지원, config의 LORA_MODEL로 동적 선택
   - 4-row grid: 모달리티 입력 / GT+Prediction+Overlay / Per-block stats / Spatial routing map
   - Val/Test 모두 지원 (Test에는 GT 대신 Legend)
   - **전체 블록 Q+V gating 로깅** → `detailed_log.json` (기존 `uamm_amf_moe_log.json` 대체)
   - 추가 로깅: per-class IoU, prediction confidence(entropy), expert collapse detection, top2_gap, logit_std

### P10/P11 취소 이유 상세

**P10 취소 이유:**
- Val에서는 P9과 거의 동일 (93.23 vs 93.32)하지만 Test에서 크게 하락 (65.30 vs 69.62)
- ModalAuxHead + oracle KL이 학습 데이터(주간)에 과적합
- CrossModalFusionHeadV2의 multi-pool이 test(야간)에서 부정확한 quality estimation
- 파라미터 증가 (aux_heads × 3) 대비 성능 저하

**P11 취소 이유:**
- P10의 문제를 해결하지 못하고 MI loss만 추가 → 오히려 악화
- MI loss가 gate를 강제로 분산시키지만, expert가 이미 분화되어 있으므로 불필요
- 지도교수 피드백에 따라 loss 추가가 아닌 근본 원인 분석으로 방향 전환
- 진단 결과 MoE gate는 이미 정상 작동 중이었음

### 미해결 과제

1. **Val vs Test 갭 (93% vs 70%)**: 야간 test에서의 성능 저하가 여전히 큰 문제
2. **Dynamic 클래스 IoU**: Test에서 Dynamic IoU가 특히 낮음 (21-28%)
3. **P12 설계 완료**: Input-Conditioned Soft MoE LoRA (cond_dim으로 모달리티 조건부 gating) — 아직 학습 미실행
4. **TTA (Test Time Augmentation)**: val_multiaqua_P9.py에 `--tta` 플래그 추가됨, 효과 미검증

### 중요 발견사항

- **NIGHT_AUG hardaug4가 최적**: hardaug2/3보다 test 성능이 일관되게 좋음
- **MoE gate는 정상 작동**: spatial mean이 uniform처럼 보이는 것은 Central Limit Theorem에 의한 artifact
- **모델 복잡도 ≠ 성능**: P10/P11의 추가 모듈이 오히려 test generalization 악화
- **P9의 CrossModalFusionHead가 가장 효과적**: 간단한 relative comparison이 최선
