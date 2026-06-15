# 프로젝트 현황 (Project Status)

> 최종 업데이트: 2026-06-15

### P28 = RBMA (Reliability-Biased Memory Attention) 구현 완료 (학습 대기) — 2026-06-15

**상태**: 구현·등록·검증 완료 → B200 학습 대기.

**배경 (진단)**: P25/P27의 SpatialQualityGating(SQG)이 야간 Pred Q에서 **정적 RGB-붕괴 + lidar/thermal 평탄**(B-1), 원인은 SQG 예측기가 frozen-encoder feature로 학습돼 underfit, teacher(exp(-CE))는 멀쩡(B-2). 상세: [03_experiment_log.md](03_experiment_log.md) 진단 1~3, [10_related_work.md](10_related_work.md).

**노벨티 근거 (deep-research)**: SAM2 memory-attention에서 reliability를 **attention LOGIT에 additive bias**로 거는 전례 0 (선행연구는 feature-multiply/output-scale/loss). 신호는 **training-free decoder 예측 불확실성** (UTFNet/HyperDUM의 학습 evidential/HD head 대비 차별). 차별화 대상: UTFNet, HyperDUM, TMC/ETMC, ReliFusion/READ.

**구현 (P28 = P27 기구 + A 신호)**:
- `LoRA_Sam_P28(LoRA_Sam_P27)` — `_compute_bias_source` 오버라이드만. P27 forward에 hook 메서드 추가(기본 identity=SQG, 안전).
- bias 신호: `reliability_i = 1 - H(softmax(per_modal_decoder_i(f_i)))/logC`, 모달리티 간 per-pixel 센터링, `λ=lambda_bias`(학습) 곱 → memory cross-attn logit에 가산.
- 모달리티 단독 디코드(memory 융합 전) → **순환 없음, GT 불필요**. SQG는 학습용 유지(점진 제거는 ablation).

**변경 파일**:
- `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`: P27 `_compute_bias_source` 훅 추가 + `LoRA_Sam_P28` 클래스.
- `train_sam2_lora_paper.py`: `QUALITY_GATE_MODELS`에 P28 추가.
- `val_multiaqua_detailed.py`: `_model_map`에 P28 추가. (val_multiaqua.py는 eval()로 자동.)
- 신규 config: `configs/b200-deliver_rgbdel_P28_physaug.yaml`, `configs/b200-multiaqua_rgbtl_P28_hardaug8_physaug.yaml` (둘 다 `AMF_MODE: uniform`=순수 RBMA).

**다음**: B200 학습 → 성능 확인 → ablation (SoftMoE LoRA / SQG / AMF=sqg_quality 제거해도 유지되는지).
⚠️ multiaqua B200 config의 ROOT/PRETRAINED 경로는 deliver 템플릿 기반 placeholder — 실제 B200 마운트 확인 필요.

---

### B200 학습 파이프라인 처리량 개선 (2026-04-15)

**문제**: B200 단일 GPU util이 10~80% 왔다갔다 → 데이터 파이프라인 병목 패턴

**적용한 조치 4건**

| 항목 | 위치 | 변경 |
|------|------|------|
| (a) AMP on | `configs/b200-deliver_rgbdel_P27_physaug.yaml` | `AMP: false → true` — B200 bf16 Tensor Core 활용 |
| (b) DataLoader tuning | [train_sam2_lora_paper.py:687-697](train_sam2_lora_paper.py#L687-L697) | `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4` (num_workers>0 조건부) |
| (c) synchronize 제거 | [train_sam2_lora_paper.py:863](train_sam2_lora_paper.py#L863) | `torch.cuda.synchronize()` 제거 — DDP all-reduce + scaler.step이 이미 동기화 제공. legacy 코드로 추정 |
| (d) non_blocking=True | [train_sam2_lora_paper.py:782-783, 1010-1011](train_sam2_lora_paper.py#L782) | `.to(device)` → `.to(device, non_blocking=True)` train/val 경로 모두 |

**기존 코드 호환성**
- 다른 모델 config(P9~P26)는 그대로 유지됨 — AMP 설정은 config별 독립
- DataLoader 변경은 `num_workers>0` 조건 분기로 단일 GPU 케이스 보호
- DDP + persistent_workers 조합은 PyTorch 2.x 표준 패턴

**주의사항**
- P27 + AMP 조합은 1 iter dry-run으로 `lambda_bias` grad 흐름 / attn_mask float bias SDPA 수치 확인 필요
- 여전히 util 불안정하면 PhysAug `P: 0.40 → 0.20` 하향 또는 Fourier를 GPU로 이동 (별도 리팩토링)

**관련 파일**
- 학습 스크립트: [train_sam2_lora_paper.py](train_sam2_lora_paper.py)
- B200 config: [configs/b200-deliver_rgbdel_P27_physaug.yaml](configs/b200-deliver_rgbdel_P27_physaug.yaml)
- B200 MULTIAQUA config: [configs/b200-multiaqua_rgbtl_P27_hardaug8_physaug.yaml](configs/b200-multiaqua_rgbtl_P27_hardaug8_physaug.yaml)

### P27 DELIVER config 생성 (2026-04-15)

- **파일**: [configs/b200-deliver_rgbdel_P27_physaug.yaml](configs/b200-deliver_rgbdel_P27_physaug.yaml)
- **기반**: `levine-deliver_rgbdel_P26_physaug.yaml` + P27 차이점
  - `LORA_MODEL: LoRA_Sam_P27`
  - `LAMBDA_BIAS_INIT: 1.0` 추가
  - `SAVE_DIR: MMSamP27/b200_deliver_rgbdel_P27_physaug`
  - DELIVER 4 modalities (`img, depth, event, lidar`), `GRADIENT_CHECKPOINT: true` 유지
- **사유**: B200 환경에서 P27 ablation + DELIVER 비교를 위한 베이스 config

### P27 구현 완료 — Additive Attention Bias on Cross-Modal Memory Attention (2026-04-14)

**상태**: 설계 안 → 구현 완료 (학습 대기)

**구현 내용**
- **RoPEAttention 확장** ([transformer.py](semseg/models/sam2/sam2/modeling/sam/transformer.py))
  - `_p27_attn_bias` 인스턴스 속성 추가 (기본 None)
  - `_sdpa_with_optional_bias()` helper: bias 있으면 Flash kernel 우회, math/mem-efficient kernel에서 `attn_mask=bias` 경로 사용
  - `RoPEAttention.forward`가 자신의 `_p27_attn_bias`를 자동 주입
- **LoRA_Sam_P27 클래스** ([sam_lora_image_encoder_seg.py:7498+](semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py))
  - `LoRA_Sam_P26` 상속 — SoftMoE LoRA, SQG, KL teacher, per-modal decoder, multi-scale FPN 그대로 유지
  - 추가: `self.lambda_bias = nn.Parameter(torch.tensor(1.0))` — 학습 가능 스칼라
  - **MemoryAttention forward pre-hook**: track_step 호출 시 memory tensor shape에서 N_k / obj_ptr token 수를 산출 → 각 source modality의 `quality_logit`을 memory spatial resolution에 interpolate → `(B, 1, 1, N_k)` bias 구성 → `layer.cross_attn_image._p27_attn_bias`에 주입
  - **Phase 3 변경**: `level_feat * q_flat` UAMM multiplication 제거 (bias 주입 전에 state만 set → track_step → 클리어)
  - **Phase 4**: `amf_mode='sqg_quality'` 기본 유지 + `'uniform'` ablation 옵션 추가
- **Train/Val 스크립트 통합**
  - [train_sam2_lora_paper.py](train_sam2_lora_paper.py): `QUALITY_GATE_MODELS`에 `LoRA_Sam_P27` 추가, `LAMBDA_BIAS_INIT` config 전달 경로
  - [val_multiaqua.py](val_multiaqua.py), [val_multiaqua_detailed.py](val_multiaqua_detailed.py): import 및 `_model_map` / kwargs 등록
- **Config**
  - 학습: [configs/levine-multiaqua_rgbtl_P27_hardaug8_physaug.yaml](configs/levine-multiaqua_rgbtl_P27_hardaug8_physaug.yaml)
  - 평가: [configs/eval_config/levine-multiaqua_rgbtl_P27_hardaug8_physaug.yaml](configs/eval_config/levine-multiaqua_rgbtl_P27_hardaug8_physaug.yaml)

**Bias 산출 로직**
```
frame f (>0) 처리 시:
  memory N_k = (f × spatial_tokens_per_frame) + num_obj_ptr_tokens
  for j in 0..f-1:
    quality_logit[j] (B,1,H_fpn,W_fpn) → interpolate to (h_mem, w_mem) → flatten → (B, spatial_tokens)
  obj_ptr 토큰 영역: 0 (학습 가능 bias 없음)
  bias = λ · concat(...) → (B, 1, 1, N_k)  ← 모든 Q · head에 broadcast
```

**검증**
- `py_compile` 통과 (transformer / sam_lora_image_encoder_seg / train / val 스크립트)
- `LoRA_Sam_P27` import + signature 확인 OK (lambda_bias_init=1.0)

**다음 단계**
1. 1 iter dry-run으로 bias injection 동작 확인 (λ=1.0 기준 attention 분포 변화 확인)
2. `MISC/diagnose_memory_attention.py`에 P27 mode 추가 — 학습 완료 후 attention content-sensitivity 재측정
3. Phase 2 ablation: `amf_mode=uniform` vs `sqg_quality` 비교
4. Phase 3 검증: Val >93%, Test >70%, M-score >82.10 목표



### P26 Ablation B — Single LoRA (Parameter-Matched) 구현 완료 (2026-04-10)

**동기 — 리뷰어 예상 비판 대응**

P26 SoftMoE LoRA (4 experts × rank 4)의 성능 향상이 단순 파라미터 증가 효과인지,
MoE routing의 가치인지 분리 검증 필요. 동일 파라미터 수의 Single LoRA 대조군 구축.

**파라미터 매칭 결과** (DELIVER, 4 modalities)

| 컴포넌트 | P26 (SoftMoE 4×r4, cond_dim=8) | AblB (Single LoRA r=18) |
|---|---:|---:|
| LoRA/MoE+embed | **719,648** | **717,696** |
| SQG | 369,412 | 369,412 |
| Per-modal decoder | 28,680,296 | 28,680,296 |
| FPN proj | 24,672 | 24,672 |
| **Total trainable** | **44,475,587** | **44,473,635** |

- LoRA 파라미터 매칭률: **99.7%** (차이 1,952개)
- Total 파라미터 차이: **0.00%**
- SQG/decoder/FPN 등 나머지 모든 컴포넌트 동일

**구현 변경**

- **새 클래스**: `LoRA_Sam_P26_AblB` ([sam_lora_image_encoder_seg.py:7283+](semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py))
  - P26의 Phase 1~4 전체 forward flow 동일 유지
  - SoftMoE LoRA → `_LoRA_qkv` (Single LoRA r=18)로 교체
  - 제거: `modal_embed`, `cond_dim`, `set_condition()`, MoE gate collection
  - 유지: ① Per-Modality SQG, ② UAMM softmax, ③ KL teacher loss + aux CE, ④ AMF (sqg_quality), ⑥ Multi-Scale FPN, ⑦ Per-Modality Decoder
- **train script**: [`train_sam2_lora_paper.py:647`](train_sam2_lora_paper.py#L647) `is_p24` 리스트에 `'LoRA_Sam_P26_AblB'` 추가
- **신규 config**: [`configs/levine-deliver_rgbdel_P26_AblB_physaug.yaml`](configs/levine-deliver_rgbdel_P26_AblB_physaug.yaml) — DELIVER, LORA_R=18

**기각 가설 (해석 전략)**

- AblB ≥ P26: SoftMoE LoRA의 routing 효과는 없음 → P26 contribution 약화
- AblB < P26 (유의미한 차이): SoftMoE LoRA의 modality routing이 단순 파라미터 증가 이상의 가치 제공
- AblB ≈ P26: SoftMoE LoRA의 효과는 expressive capacity 증가에 가깝고 routing 자체는 부차적

**확장 ablation 후보** (선택적, 우선순위 낮음)
- D: Per-modal LoRA (3 × rank 4, 명시적 모달리티 분리)
- E: D의 파라미터 매칭 버전

**상태**: 구현 완료, 학습 대기 (DELIVER 4 modalities)

### P27 설계 안 — Additive Attention Bias on Cross-Modal Memory Attention (2026-04-10)

**동기 — P26까지의 구조적 비판**

P26 v6까지 SQG 기반 gating은 feature에 element-wise multiply (`feature × uamm_weight`) 구조.
이 방식의 세 가지 이론적 문제:
1. **정보 병목**: SQG 오판단 시 해당 모달리티 feature가 0에 눌려 영구 소실, 후속 attention에서 복구 불가
2. **기울기 충돌**: Quality gate 학습 경로와 feature 학습 경로가 얽힘 (`∂L/∂feat = uamm * ∂L/∂modulated`)
3. **이중 가중치**: Feature의 상대 중요도가 gating weight로 왜곡되고 attention softmax가 다시 가중치 적용

또한 2026-03-25 Memory Attention 진단에서 확인된 사실:
- RGB 95% 어둡게 해도 cross-attention mass 변화 ≤7% → **memory attention이 품질에 반응 못 함**
- P9~P26의 UAMM/SQG는 feature 쪽만 건드리고 attention routing에는 개입 안 함 → 병목 미해결

**두 대안 비교 결과**

| 기준 | A) Additive Attention Bias | H) Quality-Conditioned Memory Bank |
|------|---------------------------|----------------------------------|
| Memory attention content-insensitivity 해결 | ✅ logit 직접 주입 | ⚠️ 우회 (저장 제한) |
| 정보 병목 | ✅ Value 보존 | ⚠️ 제외 토큰 영구 소실 |
| 미분 가능성 | ✅ 완전 미분 | 🔴 Gumbel/ST estimator 필요 |
| 상수 수렴 재발 위험 | 🟡 중간 (Q,K 입력 의존 경로 존재) | 🔴 높음 (threshold 자체 상수화) |
| Spatial-local 보완성 (LiDAR 물체 영역만) | ✅ soft gating | 🔴 binary 소실 |
| 이 프로젝트 memory bank 크기 이점 | N/A | ❌ 이미 작음 (모달별 1프레임) |
| 쓰기 시점 품질 결정 → context 변화 대응 | ✅ 읽기 시 재평가 | 🔴 쓰기 시 고정 |
| 구현 난이도 | 🟡 attention forward 수정 | 🟢 memory encoder 출력 gating |

**결정: A 채택, H 기각**

- A의 결정적 우위: 현재 최대 병목(attention content-insensitivity)을 직접 해결.
  UAMM `feature × scalar`는 상수 수렴 시 identity가 되어 무력화됐지만, A는 softmax 입력에 bias를
  더하므로 bias가 상수여도 Q,K 입력 의존 내적과 결합되어 attention 분포가 변함
- H 기각 이유:
  1. P10/P11/P14~P25 전부에서 본 "gate 상수 수렴" 패턴이 H의 threshold에서도 재현 가능성 높음
  2. SAM2 memory bank가 모달별 1프레임만 저장 → "메모리 비대화 방지" 이점 없음
  3. Binary gating이 LiDAR의 물체 영역 보완성 제거

**P27 아키텍처 초안**

```
기존: Attention = softmax(QK^T / √d) V
변경: Attention = softmax((QK^T / √d) + λ * quality_logit) V
```

- `quality_logit`: P26의 SQG logit 재활용 (이미 KL loss로 학습됨)
- `λ`: `nn.Parameter(torch.tensor(1.0))` — 학습 가능 스칼라
- Cross-attention만 수정, self-attention은 유지

**변경 사항 (P26 v6 → P27)**:
- [제거] UAMM feature multiplication
- [제거] AMF (SQG 기반 fusion weight) — ablation으로 유지 여부 결정
- [추가] Cross-attention additive bias (`RoPEAttention.forward`)
- [유지] SQG 모듈 + KL teacher loss
- [유지] SAM2 backbone, memory attention 구조, SoftMoE LoRA

**구현 위치**:
- `semseg/models/sam2/sam2/modeling/sam/transformer.py` — `RoPEAttention.forward`
- `F.scaled_dot_product_attention(q, k, v, attn_mask=float_bias)` 경로로 additive bias 주입

**리스크 및 완화**:
1. logit scale 불일치 → `λ` 학습 가능 스칼라로 자동 조정
2. 모든 layer 적용 시 학습 불안정 → 우선 cross-attention만, 필요시 단계적 확장
3. `λ` → 0 수렴 → L2 reg 또는 `0.5 + sigmoid(raw_λ)`로 범위 제한 고려
4. bias 추가로도 content-insensitivity 해결 안 될 가능성 → **Phase 0 진단으로 선행 검증**

**실험 순서**:
1. **Phase 0 (진단)**: P26 학습 완료 후 `MISC/diagnose_memory_attention.py` 재실행 — zero_rgb/zero_thermal/zero_lidar 극단 조건에서 attention mass 측정. 여전히 insensitive면 A 채택 정당성 확보
2. **Phase 1 (구현)**: `LoRA_Sam_P27` 클래스 + `RoPEAttention` cross-attention bias 경로 수정
3. **Phase 2 (ablation)**: A-only / A+UAMM / A+light-H 비교
4. **Phase 3 (검증)**: Val >93%, Test >70%, M-score >82.10 목표 + memory attention 재진단으로 content-sensitivity 개선 확인

**선결 조건**:
- P26 v6 학습 결과 확인 (baseline 비교 대상)
- Memory Attention Phase 0 진단 재실행

### P26 DELIVER 단일 GPU 메모리 프로브 추가 (2026-04-10)

- **변경**: `tmp/p26_amp_gc_probe/` 폴더 추가
  - `probe_p26_memory.py`: P26 학습 경로를 그대로 따라가며 1-step warmup + 측정 step에서 `torch.cuda.max_memory_allocated()` / `reserved()` / step time을 기록하는 단일 GPU 프로브
  - `levine-deliver_rgbdel_P26_physaug_single_gpu_baseline.yaml`: DELIVER P26 baseline 단일 GPU 측정용 config (`AMP=false`, `GRADIENT_CHECKPOINT=true`, `DDP=false`, `EVAL_INTERVAL=999999`)
  - `levine-deliver_rgbdel_P26_physaug_amp_on_gc_off.yaml`: 비교 config (`AMP=true`, `GRADIENT_CHECKPOINT=false`, `DDP=false`, `EVAL_INTERVAL=999999`)
- **사유**: RTX TITAN / RTX 3090 24GB 단일 GPU에서 `AMP on + gradient checkpointing off` 조합이 실제로 한 step을 버티는지, baseline 대비 peak VRAM이 얼마인지 빠르게 검증하기 위함.
- **검증**: `python -m py_compile tmp/p26_amp_gc_probe/probe_p26_memory.py` 통과
- **1차 결과**: RTX TITAN 단일 GPU에서 baseline(`AMP=false`, `GC=true`)도 여유 VRAM 부족으로 OOM. 다만 프로브 peak는 약 16.5~17.5GB로 관측되어, 현재 로컬에서 이미 점유 중인 약 6GB가 해소된 24GB 환경에서는 baseline 통과 가능성이 높음.
- **2차 결과**: 3모달 lower-bound(`img/depth/event`)에서는 baseline이 `peak_allocated≈16.56GB`, `peak_reserved≈17.64GB`로 통과했지만, 동일 조건의 `AMP=true + GC=false`는 `peak_allocated≈16.91GB`에서 OOM. 즉 TITAN 기준으로는 AMP 절감폭이 checkpoint 해제 증가분을 상쇄하지 못함. 3090은 FlashAttention 경로가 있어 별도 실측 필요.

### 문서: AGENTS.md Codex/Cursor 정렬 (2026-04-09)

- **변경**: 루트 [`AGENTS.md`](../AGENTS.md)를 MemorySAM 워크플로(conda `MMSS_SAM`, `train_sam2_lora_paper.py` / `val_multiaqua.py`, `.claude_logs` 세션 규칙)에 맞게 재작성. 기존 pnpm/Node 전용 템플릿 제거.
- **`.codex/AGENTS.md`**: 루트 `AGENTS.md`·`CLAUDE.md`를 가리키도록 단순화(중복 지침 방지).
- **사유**: Codex가 `AGENTS.md`를 우선 읽을 때 프로젝트와 무관한 지침이 섞이지 않게 함.

## 현재 상태: P9 ep131 & P22 ep120 **공동 최선 (M=82.10)**, P25 학습 중, **P26 v6 구현 완료 (학습 대기)**, **Memory Attention 진단 도구 구현 완료**

### Memory Attention 진단 — 1단계 문제 존재 증명 (2026-03-25)

- **동기**: 선배 피드백 — P9~P26의 UAMM/SQG가 "degraded modality가 attention을 오염시킨다"는 가정 위에 설계되었지만, 실제 검증 없음
- **구현**: `MISC/diagnose_memory_attention.py` — SAM2 memory attention의 cross-attention weight를 hooking해서 추출
  - `RoPEAttention.forward()`를 monkey-patch → manual attention weight 계산 (F.scaled_dot_product_attention은 weight 비노출)
  - memory token 구성 추적 (어떤 frame=modality에서 온 토큰인지)
  - Normal vs Degraded 입력 비교, per-layer/per-modality attention mass 통계
  - 시각화: bar chart, spatial attention map, box plot
- **초기 결과** (P9 hardaug8_physaug, val 10장, dark_rgb ×0.05):
  - Frame 2(thermal)의 cross-attention: memory=[img, lidar]
  - RGB 95% 어둡게 해도 attention 변화 ≤7% — img에 여전히 ~74% attention
  - 해석: (A) memory attention이 content-insensitive → degraded에 부적절하게 높은 attention 유지, 또는 (B) backbone이 이미 degraded를 유의미하게 인코딩
- **다음 단계**: zero_rgb, zero_thermal 등 극단적 degradation으로 추가 검증 필요

### P26 v6 구현 완료 — tau_amf 제거, UAMM/AMF weight 통합 (2026-03-25)

- **변경**: tau_amf 파라미터 완전 제거 — UAMM과 AMF가 동일한 `softmax(q_logit_stack / tau_uamm)` weight 사용
- **수정 파일**: `sam_lora_image_encoder_seg.py` (__init__, Phase 4), configs (×2), scripts (×4)
- **Feature fusion**: `q_uamm_norm` 재사용 (별도 softmax 계산 제거)

### P26 v6 설계 수정 — Per-Modal Decoder 역할 분리 + AMF SQG 기반 변경 (2026-03-24)

- **v5 문제**: per-modal decoder가 teacher(no_grad) + track_step(inference) 양쪽에 사용됨
- **v6 설계 변경 (구현 대기)**:
  - Per-modal decoder (m개): **학습 전용 auxiliary head** — 직접 CE supervision + SQG quality target 생성
  - Shared inference decoder (1개): track_step 추론 경로 — 학습 + 추론 모두 사용
  - 총 decoder: m+1개 (학습), 1개 (추론)
  - SQG = knowledge distillation: per-modal decoder → SQG로 quality 예측 능력 증류
  - **AMF: output entropy → SQG quality map 기반으로 변경** — "confident but wrong" 문제 해결
    - entropy는 "모델이 확신하는가"(calibration 의존), SQG는 "모델이 실제로 잘 하는가"(CE 기반) 측정
    - tau_amf 불필요: UAMM과 동일한 sqg_weight 재사용 (SQG logit이 KL loss로 이미 calibrate)
    - overconfident 방지: SQG target이 softmax(-CE/tau)로 항상 soft distribution + min_quality=0.3
- **구현 대기**: `sam_lora_image_encoder_seg.py` Phase 2.5/3/4 수정, `train_sam2_lora_paper.py` aux CE loss 추가
- **상세**: `.claude_logs/02_model_arch.md` — "P26 v6 설계 수정" 섹션

### P26 DELIVER 런타임 버그 6건 해결 + 통합 val.py 생성 (2026-03-24)

- **P26 버그 수정** (ISSUE-016): CheckpointError (tensor count mismatch) + CUDA OOM 해결
  - **Nested checkpointing** 도입: outer per-modality + inner per-block
  - `_encode_single_modality()` 안에서 `set_condition()` 호출 → recomputation 시 올바른 MoE condition 보장
  - 하위 호환성 유지 (다른 P 버전 영향 없음)
- **통합 val.py 생성**: MULTIAQUA + DELIVER 공용 평가 스크립트
  - `--detailed` 플래그로 per-token MoE routing 분석 (기존 `val_multiaqua_detailed.py` 기능 통합)
  - `--macvi` 플래그로 MACVi challenge 제출 (MULTIAQUA only)
  - Dataset factory 패턴으로 향후 데이터셋 추가 용이
- **P26 Detailed Visualization 추가** (2026-03-24):
  - per-modality mask 추론 결과 (colored segmap)
  - SQG quality map (viridis heatmap, 0~1)
  - per-modality entropy map (hot heatmap)
  - UAMM spatial weight map (plasma heatmap, softmax normalized)
  - AMF spatial weight map (coolwarm heatmap)
  - Feature comparison: pre-UAMM per-modal features + post-UAMM fused feature (magma)
  - JSON log에 uamm_spatial, amf_spatial, per_modal_entropy 통계 추가
  - Model 변경: `_last_uamm_spatial`, `_last_amf_spatial`, `_last_entropy_maps` 버퍼 추가
- **DELIVER return_meta 지원**: `semseg/datasets/deliver.py`에 `return_meta` 파라미터 추가
- **DELIVER eval config 생성**: `configs/eval_config/levine-deliver_rgbdel_P26_physaug.yaml`
- **다음 단계**: P26 DELIVER 학습 실행 → nested checkpointing 동작 검증

### P26 v5 전체 구현 완료 (2026-03-23)

- **핵심**: P25 기반 8가지 구조적 개선
  - ①~⑤: Per-Modality SQG, UAMM softmax, KL teacher, AMF entropy, MemMod 제거
  - ⑥ Multi-Scale FPN (fpn[0,1,2] concat → 96ch SQG input)
  - ⑦ Per-Modality Decoder (decoder ×m deepcopy, _swap_decoder)
  - ⑧ Modality-Conditioned MoE LoRA Gate (nn.Embedding + cond_dim=8)
- **구현 완료 파일**:
  - `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P26` 클래스
  - `train_sam2_lora_paper.py` — P26 kwargs, KL loss, is_p24 체크
  - `val_multiaqua.py` / `vis_feature_analysis.py` — P26 params 전달
  - `val.py` — 통합 평가 스크립트 (MULTIAQUA + DELIVER, --detailed 지원)
  - `configs/hpca100-multiaqua_rgbtl_P26_hardaug8_physaug.yaml` — HPC config
  - `configs/eval_config/hpca100-multiaqua_rgbtl_P26_hardaug8_physaug.yaml` — eval config
  - `configs/eval_config/levine-deliver_rgbdel_P26_physaug.yaml` — DELIVER eval config
  - `configs/levine-deliver_rgbdel_P26_physaug.yaml` — DELIVER levine config (4모달)
- **선결 조건**: P25 학습 결과 확인 후 착수 여부 결정
- **상세**: `.claude_logs/02_model_arch.md` — "P26" 섹션

### P22 ep120 평가 완료 — M=82.10 공동 1위 (2026-03-18)

- **P9 ep131 재제출 (#16710)**: Val 93.29 / Test 70.91 / **M=82.10**
- **P22 ep120 (#16932)**: Val 93.42 / Test 70.77 / **M=82.10**
- 두 모델 동률이나 특성이 다름:
  - P22: Dynamic **+8.85pp** (30.71 vs P9 21.86), Static/Sky 각 -2pp
  - P9 재제출: Test mIoU가 원본 대비 +0.5 (70.41→70.91)
- UAMM/AMF 여전히 상수 수렴 — scoring 병목 미해결
- 상세: `03_experiment_log.md` — P22 실험 결과 섹션

### P25 구현 완료: Unified Spatial Quality Fusion (2026-03-14)

- **핵심**: CrossModalFusionHead 제거 → SpatialQualityGating의 quality map으로 UAMM + AMF + Memory 3곳 통합
- **변경 사항**:
  - UAMM: scalar `(B, m)` → spatial max-norm `(B, 1, H, W)` — pixel-wise max-norm 후 각 vision_feat level에 interpolate
  - AMF: scalar softmax `(B, m)` → spatial softmax `(B, 1, H, W)` (pixel별 weighted fusion)
  - Memory: P24 동일 (spatial modulation)
  - CrossModalFusionHead ~3K params 제거, SpatialQualityGating ~12.5K만 유지
- **구현 완료 파일**:
  - `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P25` 클래스 추가
  - `train_sam2_lora_paper.py` — `is_p24` 분기에 P25 포함
  - `val_multiaqua_detailed.py` — P25 import/model_map 추가
  - `configs/hpca100-multiaqua_rgbtl_P25_hardaug8_physaug.yaml` — HPC 학습 config
  - `configs/eval_config/hpca100-multiaqua_rgbtl_P25_hardaug8_physaug.yaml` — HPC eval config
- **학습 명령어**: `python train_sam2_lora_paper.py --cfg configs/hpca100-multiaqua_rgbtl_P25_hardaug8_physaug.yaml`
- **리스크**: student 오류 cascade (quality map 하나가 3곳 동시 결정), quality ≠ 최적 fusion weight
- **상세**: `.claude_logs/02_model_arch.md` — "P25" 섹션

### P24 sigmoid 기반 ep36 평가 완료 (2026-03-13)

- **Val mIoU**: 93.89 (ep36), **Val per-class**: Static 94.41, Dynamic 47.17, Water 98.15, Sky 97.27
- **UAMM/AMF**: 완전 상수 수렴 (std < 0.001, 모든 이미지 동일) — gating이 작동하지 않음
  - UAMM: img=0.659, lidar=1.000, thermal=1.000
  - AMF: img=0.248, lidar=0.376, thermal=0.376
- **Quality Gating 데이터**: detailed_log에 미기록 → 코딩봇에게 quality map 로깅 추가 요청 필요
- **Test**: MACVi 미제출 (sigmoid 버그로 의미 없음)
- **Pred confidence**: Val mean_entropy=0.14, Test mean_entropy=0.38 (야간 불확실성 높음)

**Teacher Signal 버그 (ISSUE-013)**:
- sigmoid confidence 기반 target이 epoch 40에서 전부 1.0 포화 → gating net supervision 소멸
- 원인: `torch.abs(sigmoid(logits) - 0.5) * 2` — GT 미사용, decoder 자체 자신감만 측정
- **수정 필요**: 4-class CE 기반 `exp(-CE(logits_4class, gt))` → 모달리티 구조적 약점이 signal로 유지
- **추가 문제**: 현재 코딩봇 수정이 BCE(binary)로 되어있으나, teacher decoder가 4-class logits를 출력하도록 변경 필요
  - Binary BCE: "물체 있는지" → LiDAR 수면/Sky에서 의미 있는 quality 차이 못 잡음
  - 4-class CE: Static/Dynamic/Water/Sky 구분 → semantic 오류 기반 풍부한 quality signal
- **상세**: `.claude_logs/04_issues_and_fixes.md` — ISSUE-013

### P21 학습 완료 — ep94가 best (2026-03-12)

- **Best**: ep94 (M=81.77) — P9 ep131 (M=81.98) 대비 -0.21pp, P9을 넘지 못함
- ep94 이후 test mIoU 지속 하락 (70.36→67.95 at ep133) — 오버피팅 확정
- Dynamic +15pp (P9 대비) 개선 확인, Static -5.6pp / Sky -7.5pp 하락
- **근본 원인**: CrossModalFusionHead의 GAP→상수 수렴 병목이 DeBA-FP 효과를 상쇄
- 상세: `.claude_logs/03_experiment_log.md` — "P21 실험 결과" 섹션

### Scoring 함수 개선 방향 연구 (2026-03-12)

#### 문제 정의
- CrossModalFusionHead: GAP→MLP→softmax가 학습된 상수로 수렴 (std≈0, 모든 입력에 동일 weight)
- P12~P19의 8개 adaptive fusion 시도 모두 P9(고정 상수)보다 나쁨
- 핵심 원인 3가지:
  1. GAP이 spatial quality 정보 소실
  2. SAM2 frozen encoder가 modality 간 feature 분포 정규화
  3. Zero-init이 안정적 uniform 고정점 생성

#### 관련 연구 조사 결과
- **CAFuser (RA-L 2025)**: CLIP Condition Token + Cross-Attention² — 우리는 CLIP/text 없으므로 직접 차용 불가
- **DGFusion, arXiv 2509.09828**: 입력 condition 정보 추출 → fusion 비율 결정
- **Learnable Query Attention Pooling (CLIP, Perceiver, Q-Former)**: GAP 대체 — learnable query가 feature map에 cross-attend하여 입력 의존적 summary 생성 → 상수 수렴 원리적 불가
- **AECF (arXiv 2025)**: gate collapse 방지 — per-instance entropy regularization + adversarial curriculum masking
- **EMOE (CVPR 2025)**: unimodal distillation으로 각 modality 독립 예측력 유지 강제
- **TokenFusion (CVPR 2022)**: per-token MLP scoring → 분포 기반 modality quality 판단
- **GSoP (CVPR 2019)**: Second-order covariance pooling — 채널 co-activation 패턴으로 GAP보다 풍부한 입력 의존 표현
- **CGGM (NeurIPS 2024)**: 학습 시 gradient modulation — dominant modality 억제

#### 실험 계획: Quality-aware Memory Gating via Per-Modality Decoder Distillation (P24 후보)

##### 해결하려는 문제

1. **CrossModalFusionHead 상수 수렴**: GAP→MLP→softmax가 모든 입력에 동일 weight 출력 → adaptive fusion 불가
2. **열화된 RGB의 memory 전파 오염**: 야간 장면에서 RGB가 frame_idx=0 (is_init_cond_frame=True)으로 먼저 처리 → 열화된 RGB memory가 memory bank에 저장 → LiDAR/Thermal이 이 memory를 attend하면서 오염 전파
3. **Per-modality 보완성 보존**: LiDAR는 전체 mask quality는 낮지만(sky/water 못 잡음), 해상 객체 영역에서는 RGB를 보완 → scalar gating으로는 이 영역별 기여가 무시됨

##### 핵심 아이디어: Spatial Quality Map으로 Memory 기여도 조절

**학습 시 (Teacher)**:
- 각 모달리티 feature를 SAM2 decoder에 single-frame으로 입력 (`is_init_cond_frame=True`, memory attention skip)
- Per-pixel CE loss를 GT mask 대비 계산 → spatial quality map 생성
- 이 quality map이 gating network의 supervision target

**학습 시 (Student)**:
- Gating network가 encoded feature만 보고 spatial quality map을 예측
- 예측된 quality map으로 memory bank 저장 시 memory features를 spatial modulation
- 열화된 영역의 memory 기여 ↓, 잘 예측하는 영역의 memory 기여 ↑

**추론 시**:
- Per-modality decoding 없이 gating network만 실행 → spatial quality map 예측
- Memory modulation은 학습 시와 동일하게 적용
- 추가 비용: gating network forward만 (매우 경량)

##### 구체적 파이프라인

```
학습 시:
  for m in [img, lidar, thermal]:
    # Teacher: per-modality single-frame decoding
    feat_m → SAM2 decoder(is_init=True, no memory) → logits_m  (B, 4, 1024, 1024)
    loss_map_m = CE(logits_m, gt_mask, reduction='none')         (B, 1024, 1024)
    quality_target_m = downsample(1/(1+loss_map_m), 64×64)       (B, 1, 64, 64)

    # Student: gating network
    predicted_quality_m = gating_net(feat_m)                      (B, 1, 64, 64)
    gate_loss += MSE(predicted_quality_m, quality_target_m.detach())

  # Main pipeline (기존과 동일 + memory modulation)
  for frame_idx, m in enumerate(modalities):
    output_m = track_step(feat_m, ...)   # memory attention 포함
    maskmem_m = memory_encoder(output_m)
    maskmem_m = maskmem_m * predicted_quality_m   ← memory modulation
    → memory bank에 저장 → 다음 모달리티가 참조

  final = AMF(outputs)  # 기존 AMF 유지 또는 quality 기반으로 대체
  main_loss = CE(final, gt_mask)
  total_loss = main_loss + λ_gate * gate_loss

추론 시:
  for m in modalities:
    predicted_quality_m = gating_net(feat_m)          # gating network만 실행
    output_m = track_step(feat_m, ...)
    maskmem_m = memory_encoder(output_m) * predicted_quality_m
    → memory bank
  final = AMF(outputs)
```

##### P14~P17 aux decoder 접근과의 핵심 차이

| 항목 | P14~P17 | 이 제안 |
|------|---------|---------|
| Mask 생성기 | 경량 Conv 2~3개 (~290 params) | **SAM2 full decoder** |
| Mask 품질 | 낮음 (ISSUE-008 근본 한계) | **높음** (SAM2 decoder 능력 활용) |
| Quality 측정 | Entropy (confident하게 틀려도 low) | **Per-pixel CE loss** (GT 대비 실제 품질) |
| Scoring 시점 | 추론 시 aux prediction 필요 | **학습 시만** → KD로 gating net 학습 |
| 추론 비용 | Aux decoder M번 추가 | **Gating net만** (경량) |
| 적용 위치 | 최종 output 합산 weight | **Memory bank 저장 시 modulation** |
| 해결 대상 | UAMM/AMF scoring | **Memory 전파 오염 차단** |

##### Memory Modulation이 해결하는 것

```
예시: 야간 장면

RGB (frame_idx=0):
  gating_net → quality_map: sky=0.7, water=0.3, object=0.2 (전반적 열화)
  maskmem_rgb = maskmem_rgb * quality_map → 약한 memory

LiDAR (frame_idx=1):
  memory attention: Q=LiDAR, K/V=약한 RGB memory
  → 열화된 RGB memory를 별로 안 가져감 → LiDAR 오염 방지
  gating_net → quality_map: sky=0.0, water=0.1, object=0.9
  maskmem_lidar = maskmem_lidar * quality_map → object 영역만 강한 memory

Thermal (frame_idx=2):
  memory attention: Q=Thermal, K/V=약한 RGB + object-focused LiDAR memory
  → LiDAR의 object 정보는 활용, RGB/LiDAR의 약한 영역은 무시
```

##### Gating Network 구조 (후보)

Learnable Query Attention Pooling 기반:
```python
class SpatialQualityGating(nn.Module):
    def __init__(self, feat_dim=256, n_queries=4):
        self.queries = nn.Parameter(randn(n_queries, feat_dim))
        self.cross_attn = MultiheadAttention(feat_dim, num_heads=4)
        self.conv_head = nn.Sequential(
            Conv2d(feat_dim, 64, 1), ReLU(),
            Conv2d(64, 1, 1), Sigmoid()
        )

    def forward(self, feat_map):  # (B, C, H, W) = (B, 256, 64, 64)
        # Cross-attention으로 quality-aware feature 생성
        tokens = feat_map.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        q = self.queries.unsqueeze(1).expand(-1, B, -1)
        attended, attn_weights = self.cross_attn(q, tokens, tokens)
        # attended: (n_queries, B, C) → spatial map으로 변환
        quality_feat = attn_weights @ tokens  # 또는 다른 spatial 복원 방식
        quality_map = self.conv_head(quality_feat)  # (B, 1, 64, 64)
        return quality_map
```

##### 리스크 및 열린 질문

1. **Domain gap**: 학습(주간+aug)에서 배운 "feature→quality" 매핑이 야간 test에서 전이되는가?
   - 완화: hardaug8_physaug의 극저조도 시뮬레이션이 커버
   - 검증: val(주간)에서 quality map이 합리적인지 시각화로 확인

2. **학습 비용**: SAM2 decoder × 3 추가 실행 (학습 시만)
   - Encoder(Hiera)는 1번만 → decoder는 상대적으로 가벼움
   - `torch.no_grad()`로 teacher decoder gradient 차단 가능 (memory 절약)
   - 하지만 gating net의 gradient가 encoder까지 전파되어야 할 수 있음 → 설계 결정 필요

3. **Gating net spatial output 구조**: Learnable Query → spatial map 복원 방법
   - 단순: feat_map에 Conv2d 몇 개로 직접 quality map 생성 (P19과 유사하지만 supervision이 다름)
   - 복잡: Cross-attention 기반 (더 input-adaptive하지만 spatial 복원이 자명하지 않음)

4. **Memory modulation 강도**: quality_map ∈ [0,1]에서 0에 가까우면 memory가 거의 사라짐
   - Min-clamp (e.g., max(quality, 0.1))로 최소 기여 보장?
   - 또는 residual: maskmem = maskmem * (0.5 + 0.5 * quality) → [0.5, 1.0] 범위

5. **UAMM/AMF와의 관계**: memory modulation이 충분하면 UAMM/AMF를 상수로 유지해도 되는가?
   - 가설: memory 단계에서 quality-aware fusion이 일어나면, 최종 output fusion은 단순 평균이어도 충분
   - 또는: quality map의 global average를 AMF weight로도 활용 (dual-use)

### P23 구현 완료 (2026-03-10)

- **실험 M 구현**: MoE DeBA-BB — SoftMoE-LoRA를 MoE 기반 deformable bottleneck adapter로 교체
- **핵심 구조**: `x → GAP gate → Σ w_i × [W_d(shared) → DCM_i(3×3) → LN(shared) → GELU → W_u(shared)] → α_m × out`
- **설계 결정 사항**:
  1. **Gating**: GAP gating (per-image/per-window) — training noise std=0.1
  2. **Expert 차별화**: Multi-scale (×1, ×2) — per-expert DCM, shared W_d/W_u
  3. **Cross-layer sharing**: DCM per-expert (shared across layers), LN shared, W_d/W_u per-stage, gate per-stage, α per-modality
  4. **Single adapter per block**: delta added to both Q and V (DeBA-BB 개념에 맞게 feature refinement)
- **파라미터**: ~325K (deba_bb) + ~15K (cross_modal_head) = ~340K trainable adapter params (P9 MoE LoRA ~538K 대비 37% 감소)
- **수정 파일**:
  1. `semseg/models/sam2/sam2/sam_lola_utils.py` — `MoE_DeBA_BB`, `_MoE_DeBA_BB_qkv` 추가
  2. `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P23` 추가
  3. `train_sam2_lora_paper.py` — `deba_scales`, `deba_gate_noise_std` dispatch 추가
  4. `configs/levine-multiaqua_rgbtl_P23_hardaug8_physaug.yaml` — 학습 config
  5. `configs/eval_config/levine-multiaqua_rgbtl_P23_hardaug8_physaug.yaml` — 평가 config
- **학습 명령**: `python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P23_hardaug8_physaug.yaml`
- **상세**: `.claude_logs/02_model_arch.md` — "P23: MoE DeBA-BB" 섹션

### val_multiaqua_detailed.py 호환 fix (2026-03-10)

- P20 test 실행 시 `SoftMoE_LoRA_Layer_V2`에 `gate` 속성 없어 에러 발생
- V1(`module.gate`) / V2(`module._shared_gate`) 호환 처리 완료
- 상세: `.claude_logs/04_issues_and_fixes.md` — RESOLVED-005

### P22 구현 완료 (2026-03-09)

- **실험 L 구현**: P9 + Multi-Scale DeBA-FP (all FPN levels, Phase 1)
- **P21 대비 핵심 차이**:
  - P21: fpn[0] only, Phase 2 → CrossModalFusionHead와 m_feat에만 영향
  - **P22: fpn[0,1,2] all, Phase 1 → 전체 파이프라인(vision_feats, tracking, decoder, fusion) 전파**
- **핵심 변경**:
  - `DeBAFP_MultiScale`: cross-layer sharing (DCM+norm 공유, W_d/W_u per-level, α per-modality)
  - FPN 채널: [32, 64, 256] (after conv_s0/conv_s1 in forward_image)
  - 추가 파라미터: ~98K (P21 ~56K 대비 +42K, per-level W_d/W_u 추가)
- **수정 파일**:
  1. `semseg/models/sam2/sam2/sam_lola_utils.py` — `DeBAFP_MultiScale` 클래스 추가
  2. `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P22` 추가
  3. `configs/levine-multiaqua_rgbtl_P22_hardaug8_physaug.yaml` — 학습 config
  4. `configs/eval_config/levine-multiaqua_rgbtl_P22_hardaug8_physaug.yaml` — 평가 config
- **학습 명령**: `python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P22_hardaug8_physaug.yaml`

### P21 구현 완료 (2026-03-09)

- **실험 K 구현**: P9 + DeBA-FP (Deformable Bottleneck Adapter for Feature Pyramid)
- **Ref**: CVPR 2026 — "Rethinking Deformable Convolution as an Adapter with Cross-layer Weight Sharing"
- **핵심 변경**:
  - FPN[0] → DeBA-FP → refined FPN[0] → CrossModalFusionHead (UAMM/AMF는 P9 동일)
  - Cross-modal weight sharing: DCM, norm, W_d, W_u 공유, α만 per-modality
  - DeBA-BB 미적용 (SAM2 Hiera ≠ DINOv2, 향후 과제)
  - 추가 파라미터: ~85K (P9 대비 12% 증가)
- **수정 파일**:
  1. `semseg/models/sam2/sam2/sam_lola_utils.py` — `DeBAFP` 클래스 추가
  2. `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P21` 추가
  3. `train_sam2_lora_paper.py` — `deba_bottleneck_dim` dispatch 추가
  4. `configs/levine-multiaqua_rgbtl_P21_hardaug8_physaug.yaml` — 학습 config
  5. `configs/eval_config/levine-multiaqua_rgbtl_P21_hardaug8_physaug.yaml` — 평가 config
- **학습 명령**: `python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P21_hardaug8_physaug.yaml`

### P9+hardaug8_physaug ep131 — 역대 최고 M-score 달성 (2026-03-09)

- **M=81.98** (이전 최선 P9 hardaug4 M=81.47 대비 **+0.51**)
- Val 93.54 / Test **70.41** (처음으로 test 70 돌파) / Test Obstacle **32.85**
- **핵심 성과**: Dynamic IoU 21.86→**33.50** (+11.64pp), Dynamic=0 프레임 38→13 (66% 감소)
- Sky 76.54→73.75 (-2.79pp), Static 81.30→76.64 (-4.66pp) — Dynamic 개선이 압도
- 학습 궤적: ep83(M=80.57)→ep94(80.75)→ep131(81.98) — 장기 학습에서 지속 개선
- AMF: img 0.239, lidar 0.371, thermal 0.390 (여전히 완전 상수, RGB 의존도 추가 감소)
- Checkpoint: `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug8_physaug/MULTIAQUA_CMNeXt-B2_ilt/epoch131_94.41_top1_checkpoint.pth`
- Submission #16683

### P20 구현 완료 (2026-03-05)

- **실험 J-A 구현**: SharedGateMLP + SoftMoE_LoRA_Layer_V2 + LoRA_Sam_P20
- **핵심 변경**:
  - Gate: `Linear(C→3)` → `SharedGateMLP(C→C//4→ReLU→C//4→3)` 2-layer MLP
  - Gate 공유: 동일 dim 블록들이 1개 MLP 공유 (48개→4개, ~268K)
  - Rank: 4 → 8
- **수정 파일**:
  1. `semseg/models/sam2/sam2/sam_lola_utils.py` — `SharedGateMLP`, `SoftMoE_LoRA_Layer_V2` 추가
  2. `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P20` 추가
  3. `train_sam2_lora_paper.py` — `gate_hidden_ratio` dispatch 추가
  4. `configs/levine-multiaqua_rgbtl_P20_hardaug8_physaug.yaml` — 학습 config
  5. `configs/eval_config/levine-multiaqua_rgbtl_P20_hardaug8_physaug.yaml` — 평가 config
- **학습 명령**: `python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P20_hardaug8_physaug.yaml`

### P9+hardaug8 학습 궤적 — Non-monotonic, ep131=peak (2026-03-07~09)

- ep83(M=80.57) → ep94(80.75) → **ep131(M=81.98) ★ peak** → ep188(M=80.84, **regression**)
- **ep188 분석** (#16702): Day-val 94.43(>ep131의 94.41) but test 68.20(<ep131의 70.41)
  - **Day-val은 야간 test 성능의 신뢰할 수 있는 proxy가 아님** — 주간 과적합
  - Dynamic 33.50→25.88 (-7.62pp), 162/200 프레임 ep131 대비 악화
  - Per-class 상관: Static(r=0.59), Dynamic(r=0.60)이 test mIoU 결정에 가장 중요
- **핵심 발견**:
  - ep131이 sweet spot — 추가 학습은 역효과
  - Dynamic IoU가 M-score 변동의 핵심 driver
  - **AMF/UAMM 여전히 완전한 상수** (std≈0.0000)

### I2I Translation 실험 실패 (2026-03-05)

- **실험 II: Day-Trans** (test night→day, img2img-turbo) → **M=78.90 (baseline 81.47 대비 -2.57)**
  - Test mIoU 64.50 (-5.12pp), Sky -9.80pp, Static -5.33pp
  - 171/200 이미지 하락, hallucinated texture가 segmentation 악화
  - Submission #16478
- **실험 III: Night2** (day→night I2I로 학습 데이터 확장) → **M=73.04 (baseline 대비 -8.43)**
  - Test mIoU 53.18 (-16.44pp), **Sky IoU 26.92 (-49.62pp)** — Sky 붕괴 130/200장
  - I2I artifact 학습 + cross-modal 불일치 + NightSim 이중 적용
  - Submission #16482
- **근본 원인**: Day/Night 정보 비대칭. Real night은 센서 단계에서 정보 비가역 소실 → I2I로 복원/모사 불가. Pixel-level domain bridging의 정보이론적 한계

### Gamma TTA 실험 실패 (2026-03-04)

- **실험 I: P9 hardaug4 + Gamma TTA [1.0, 1.5, 2.0, 2.5]** → **M=76.10 (baseline 81.47 대비 -5.37)**
- Test mIoU 58.89 (-10.73pp), **Sky IoU 47.70 (-28.84pp)** — Sky 붕괴가 하락의 75%
- Sky=0 프레임 5→29개 (5.8x), Dynamic=0 프레임 46→72개 (1.6x)
- **원인**: 높은 gamma(1.5~2.5)가 OOD 입력 생성 + equal-weight soft voting이 좋은 예측을 dilute
- **결론**: Multi-gamma TTA는 이 모델에서 확정적 실패. Single mild gamma(1.2~1.3) 탐색 여지 있음
- Submission #16412

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

15. **P20 (Shared MLP Gate + Rank 8) — 구현 완료, 학습 대기**
    - P9 base + SharedGateMLP(2-layer) + SoftMoE_LoRA_Layer_V2 + rank 8
    - Gate 공유: dim별 4개 MLP만 사용 (48개 독립 gate → 4개 공유 MLP)
    - hardaug8_physaug config (CRM 0.35→0.20 완화 + PhysAug + shot noise)

16. **P21 (DeBA-FP: Deformable Bottleneck Adapter) — 구현 완료, 학습 대기**
    - P9 base + DeBA-FP (deformable conv bottleneck on FPN features)
    - Cross-modal weight sharing: DCM/norm/W_d/W_u 공유, α만 per-modality
    - DeBA-BB 미적용 (SAM2 Hiera ↔ DINOv2 구조 차이, 향후 과제)
    - ~85K 추가 파라미터 (P9 대비 12% 증가)

17. **P22 (Multi-Scale DeBA-FP: all FPN levels, Phase 1) — 구현 완료, 학습 대기**
    - P9 base + DeBAFP_MultiScale (fpn[0,1,2] 전부, Phase 1 적용)
    - P21 대비: fpn[0] only → fpn[0,1,2] all, Phase 2 → Phase 1
    - Refined features가 vision_feats/tracking/decoder/fusion 전체 파이프라인 전파
    - Cross-layer sharing: DCM/norm 공유(levels+modalities), W_d/W_u per-level, α per-modality
    - ~98K 추가 파라미터 (P21 대비 +42K)

18. **P23 (MoE DeBA-BB: backbone adapter) — 구현 완료, 학습 대기**
    - SoftMoE-LoRA를 MoE DeBA-BB로 교체 (deformable conv bottleneck as MoE expert)
    - GAP gating (per-image/per-window), multi-scale (×1, ×2), cross-layer weight sharing
    - ~340K trainable adapter params (P9 ~538K 대비 37% 감소)
    - Single adapter per block → Q + V에 동시 적용 (DeBA-BB feature refinement 개념)
    - Ref: ConvLoRA (ICLR 2024) + DeBA (CVPR 2026)
    - P21/P22(DeBA-FP)와 보완적 — BB=backbone 내부, FP=FPN 출력

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

#### 실험 J: MLP Gate + Rank Scaling (MoE LoRA 강화) — P9 기반

- **목적**: 현재 P9의 약한 gate(`Linear(C→3)`)와 낮은 expert capacity(rank=4)를 동시에 강화
- **배경 분석**:
  - P9 MoE gating이 사실상 상수화 (모달리티별 거의 동일한 gate weight, entropy_ratio 0.55~0.86)
  - P11에서 MI loss로 gate 분화 강제 → 실패 (M=77.09). 하지만 이는 **약한 gate로는 의미 있는 routing을 못하는데 억지로 분화시킨 것**이 원인일 가능성
  - 최신 연구(LD-MoLE, DynMoLE)는 단순 Linear가 아닌 MLP 기반 gate 사용
  - Expert rank=4도 매우 낮아 expert 간 specialization 여지 부족

- **변경 사항**:

| 구성요소 | 현재 P9 | 실험 J |
| --- | --- | --- |
| Gate | `Linear(C → 3)` | `Linear(C → C//4) → ReLU → Linear(C//4 → 3)` (2-layer MLP) |
| Rank | 4 | 8 (J-A), 16 (J-B) |
| Expert 수 | 3 | 3 (모달리티 수 유지) |
| Gate 공유 | 없음 (48개 독립) | **같은 C 차원의 layer끼리 MLP 공유** (LD-MoLE 방식) |

- **Gate MLP 상세**:
  - Stage 0 (C=112): `Linear(112→28) → ReLU → Linear(28→3)` — hidden=28
  - Stage 1 (C=224): `Linear(224→56) → ReLU → Linear(56→3)` — hidden=56
  - Stage 2-3 (C=768): `Linear(768→192) → ReLU → Linear(192→3)` — hidden=192
  - **Stage별 공유**: 같은 C 차원의 모든 block이 1개 MLP 공유 → 파라미터 절약
    - Stage 0: 2개 block(B0-B1) × Q/V = 4개 layer가 1개 MLP 공유
    - Stage 1: 3개 block(B2-B4) × Q/V = 6개 layer가 1개 MLP 공유
    - Stage 2-3: 19개 block(B5-B23) × Q/V = 38개 layer가 1개 MLP 공유
    - → 총 3개 MLP만 필요 (현재 48개 Linear → 3개 MLP)

- **Computational cost**:
  - Gate MLP 추가: QKV 연산 대비 ~8% 수준 → 전체 forward에서 체감 미미
  - Rank 8: expert 파라미터 2배 증가 → 전체 모델 대비 여전히 작음 (~1.8M)
  - Rank 16: expert 파라미터 4배 증가 → ~3.5M (SAM 80M 대비 4.4%)

- **실험 순서**:
  1. **J-A**: MLP gate + rank 8 (안전한 첫 실험)
  2. **J-B**: MLP gate + rank 16 (J-A 결과에 따라)
  3. 선택: gate만 MLP로 바꾸고 rank 4 유지하는 ablation도 고려

- **수정 필요 파일**:
  1. `semseg/models/sam2/sam2/sam_lola_utils.py` — `SoftMoE_LoRA_Layer`의 gate를 MLP로 교체, shared gate 지원
  2. `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P9` (또는 P20 신규) gate 공유 로직
  3. `configs/` — `LORA_R: 8`, `GATE_TYPE: mlp`, `GATE_HIDDEN_RATIO: 4` 등 config 항목 추가
  4. `train_sam2_lora_paper.py` — gate config 파싱 및 모델 생성 로직

- **기대 효과**:
  - MLP gate → 비선형 결정 경계 학습 가능 → 모달리티/공간/컨텐츠 기반 의미 있는 routing
  - Rank 증가 → expert 간 실질적 차이 발생 → gate 분화에 대한 gradient 신호 강화
  - 두 가지가 시너지: 강한 gate + 강한 expert → MoE가 비로소 의도대로 작동할 가능성

- **리스크**: 중간. 파라미터 증가로 오버피팅 가능 (MULTIAQUA 데이터 적음). Gate 공유로 완화
- **LORA_MODEL**: `LoRA_Sam_P20` (신규)
- **상태**: **구현 완료, 학습 대기** (J-A: rank=8, MLP gate, hardaug8_physaug)

---

#### 실험 H: RandomGammaWide — 독립 Contrast Invariance Augmentation

- **목적**: NightSim과 독립적인 wide gamma augmentation으로 contrast-invariant feature 학습
- **배경 관찰** (2026-03-03):
  - 실제 야간 test 이미지에 gamma 조절 시, 육안으로 수면-하늘-장애물 boundary 구분 가능
  - 이는 semantic 정보가 존재하지만 contrast curve가 다를 뿐이라는 증거
  - 현재 NightSim gamma range [0.4, 0.8]은 일방향(어둡게만), 범위도 좁음
- **⚠️ hardaug6과의 차이점** (hardaug6은 M=75.95로 실패):
  - hardaug6: NightSim **내부**의 gamma를 [0.20, 1.50]으로 변경 + CRM/ZERO 제거 + brightness 변경 → 3가지 동시 변경
  - **본 실험**: NightSim과 **완전 독립적인 별도 augmentation**으로 적용
  - hardaug6은 "야간 시뮬레이션 범위 확장" → 비현실적 야간 이미지 생성
  - 본 실험은 "밝은 주간 이미지도 다양한 gamma로 볼 수 있게" → **Domain Randomization 철학**
  - NightSim의 파라미터(gamma, brightness 등)는 hardaug4 그대로 유지
- **설계**:
  ```python
  class RandomGammaWide:
      """NightSim과 독립. 순수 contrast invariance 학습용.
      NightSim 적용 여부와 무관하게 모든 RGB 이미지에 적용 가능."""
      def __init__(self, gamma_range=(0.3, 2.5), p=0.5):
          self.gamma_range = gamma_range
          self.p = p

      def __call__(self, sample):
          if random.random() > self.p:
              return sample
          gamma = random.uniform(*self.gamma_range)
          img = sample['img'].float() / 255.0
          img = torch.clamp(img, 1e-6, 1.0) ** gamma
          sample['img'] = (img * 255).to(torch.uint8)
          return sample
  ```
- **적용 위치**: `get_train_augmentation()` 내에서 NightSim **이전**에 배치
  - NightSim 전에 gamma → NightSim이 이미 다양한 contrast의 이미지를 어둡게 만듦
  - 또는 NightSim **이후**에 배치 → 이미 어두운 이미지에도 gamma 적용 (더 extreme)
  - **권장: NightSim 이전** (주간 이미지에 gamma 적용 → NightSim → 다양한 야간)
- **Config 설계** (hardaug7):
  ```yaml
  NIGHT_AUG:
    # hardaug4 파라미터 전부 유지
    ENABLE: true
    NIGHT_SIM_P: 0.45
    BRIGHTNESS_RANGE: [0.03, 0.45]
    # ... (hardaug4 동일)
    CRM_P: 0.35
    ZERO_P: 0.09
    # 새로운 파라미터
    RANDOM_GAMMA_WIDE:
      ENABLE: true
      GAMMA_RANGE: [0.3, 2.5]
      P: 0.5
  ```
- **가설 검증**:
  - hardaug4 + RandomGammaWide ≥ hardaug4 (M≥81.47) → contrast invariance가 유익
  - hardaug4 + RandomGammaWide < hardaug4 → gamma 다양성이 capacity 낭비 (hardaug6과 동일 결론)
- **리스크**: 중간. hardaug6 실패와 유사할 수 있으나, NightSim 파라미터 유지 + CRM/ZERO 유지가 차이점
- **LORA_MODEL**: `LoRA_Sam_P9` (아키텍처 변경 없음)
- **상태**: 설계 완료, 구현 대기

---

#### 실험 I: Test-time Gamma Preprocessing ⭐ (학습 불필요, 즉시 검증 가능)

- **목적**: 기존 P9 hardaug4 체크포인트를 그대로 사용, test 이미지에 gamma 전처리만 적용
- **배경 관찰** (2026-03-03):
  - 야간 test RGB 이미지에 gamma=1.5~2.5 적용 시, 어두운 영역의 semantic boundary가 육안으로 구분 가능
  - 모델이 주간 이미지(밝은)로 학습했으므로, 야간 이미지를 밝게 만들면 학습 분포에 가까워짐
  - **학습 없이 즉시 검증 가능** → 가장 빠른 실험
- **구현 방식**:
  ```python
  # val_multiaqua.py에 --test_gamma 플래그 추가
  # test 이미지 로드 후, 모델 입력 전에 gamma 적용
  def apply_test_gamma(img_tensor, gamma):
      """img_tensor: (B, 3, H, W) uint8 또는 float [0,1]"""
      img = img_tensor.float() / 255.0 if img_tensor.dtype == torch.uint8 else img_tensor
      img = torch.clamp(img, 1e-6, 1.0) ** (1.0 / gamma)  # gamma > 1 → 밝게
      return img
  ```
- **실험 설계**:
  - `--test_gamma 1.0`: baseline (현재와 동일, M=81.47 확인)
  - `--test_gamma 1.5`: 약간 밝게
  - `--test_gamma 2.0`: 중간
  - `--test_gamma 2.5`: 강하게 밝게
  - `--test_gamma 3.0`: 극단
  - **val에는 적용하지 않음** (val은 주간이므로 gamma 불필요)
- **변형: Multi-gamma TTA**:
  - 여러 gamma 값으로 예측 → soft voting (확률 평균)
  - `--test_gamma_tta 1.0,1.5,2.0`: 3개 gamma로 예측 후 평균
  - Ensemble 효과로 단일 gamma보다 robust할 수 있음
- **적용 위치**: `val_multiaqua.py`의 test 평가 루프, normalize 이전에 gamma 적용
- **리스크**: 낮음. 학습 변경 없음, 기존 체크포인트 그대로 사용
- **기대 효과**:
  - 야간 RGB를 밝게 → SAM2가 주간에서 학습한 feature와 더 유사한 입력
  - 특히 Sky, Static 영역에서 RGB feature 품질 향상 기대
  - val mIoU는 변화 없음 (val에는 미적용)
  - test mIoU +1~5pp 가능 → M-score 82~84 잠재력
- **주의사항**:
  - gamma 적용은 normalize **이전**에 해야 함 (정규화 후 적용하면 분포 파괴)
  - 현재 `Normalize`가 augmentation pipeline의 마지막 → 별도로 gamma를 삽입해야 함
  - test_gamma는 RGB에만 적용 (thermal, lidar는 별도 전처리)
- **우선순위**: ⭐ 최우선 — 학습 없이 즉시 효과 검증 가능
- **상태**: 설계 완료, 구현 대기

---

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
5. **🔴 TTA (Test Time Augmentation)**: Gamma TTA [1.0,1.5,2.0,2.5] 실험 완료 → **M=76.10, 실패 확정** (baseline 81.47 대비 -5.37). OOD gamma가 Sky 붕괴(-28.84pp) 유발. Single mild gamma(1.2~1.3) 단독 적용은 미검증
6. **P13 best day-val checkpoint test 재평가**: epoch14(93.48)로 M-score 역전 가능성 확인 필요
7. **Diffusion 기반 Night 합성** (ISSUE-005): M=85 도달을 위한 최유력 접근, 미구현
8. **Ensemble**: P9(Sky 우세) + P13(Dynamic 우세) 상보성 활용 가능
9. **🟢 ISSUE-009: Energy Score "confident but wrong"**: P16에서 calibrated entropy로 교체 완료 (구현됨, 학습 대기)
10. **🔴 Test-time Gamma TTA**: 실험 I 완료, **실패 확정 (M=76.10)**. Multi-gamma soft voting은 해로움. Single gamma(1.2~1.3) 단독은 미검증
11. **🔴 I2I Translation**: Day-Trans(M=78.90), Night2(M=73.04) 양방향 모두 실패. Pixel-level domain bridging 한계 확정. day2night2day≈day 관찰 → I2I의 synthetic night은 정보 보존된 fake night
12. **🟢 FDA Augmentation**: Fourier Domain Adaptation — low-freq amplitude만 교체하여 style transfer. I2I 대비 hallucination 없음. **구현 완료 (학습 대기)**. Config: `levine-multiaqua_rgbtl_P9_hardaug4_fda.yaml`
13. **🟡 Self-training**: Pseudo-label 기반 unsupervised domain adaptation. Real test image 그대로 사용 → 정보 한계 우회
14. **🟡 MoE Routing 분석 시각화 개선**: `detailed_log.json` 기반 3-chart 시각화 구현 필요 (아래 스펙 참조)

#### MoE Routing 시각화 스펙 (구현 대기)

`val_multiaqua_detailed.py`의 `detailed_log.json` 출력 기반. 기존 argmax map은 routing strength를 반영하지 못하므로, 아래 3개 차트를 추가해야 함.

**데이터 소스**: `detailed_log.json` → 이미지별 `moe_routing` 딕셔너리 내 `Block{N}_{Q/V}` → 모달리티별 stats

**Chart 1: Per-Modal Expert Soft Weight (Grouped Stacked Bar)**
- x축: Block index (B0, B3, B9 등)
- y축: soft weight (0~1)
- 모달리티별 3개 bar 묶음, 각 bar를 E0/E1/E2 stacked
- 데이터 필드: `spatial_mean` (argmax가 아닌 실제 soft weight)
- 목적: 모달리티마다 expert 선호가 다른가? (cross-modal specialization)

**Chart 2: Routing Strength Heatmap**
- x축: Block index
- y축: 모달리티 (img, lidar, thermal) × Q/V
- 색상: `top2_gap` 값 (0~0.5+)
- 목적: 어느 블록/모달리티에서 routing 결정이 확실한가? 연한 셀 = argmax map 신뢰 불가

**Chart 3: Spatial Adaptiveness (Line Plot)**
- x축: Block index
- y축: `per_token_max_std`
- 선 3개: img, lidar, thermal
- 목적: routing이 공간적으로 adaptive한가? (높으면 영역별로 다른 expert 사용)

**해석 조합**: Chart1에서 모달리티간 패턴 다름 + Chart2에서 top2_gap 큼 + Chart3에서 std 높음 → MoE가 진짜 adaptive. 셋 다 낮으면 → 사실상 단일 LoRA처럼 동작.

### 중요 발견사항

- **🔴 Dynamic Fusion 실패 확정**: P12~P17 6개 실험 모두 P9(고정 상수)보다 나쁨. 복잡한 adaptive fusion이 오히려 해로움
- **🔴 CRM/ZERO Overfitting 발견**: 학습 44%에 exact-zero RGB → test에는 없는 shortcut 학습. P13 epoch39에서 test -19.5pp crash
- **🔴 Sky collapse가 핵심 병목**: P14~P17에서 Sky IoU 3~36% (P9: 76%). Adaptive fusion이 RGB를 suppress → sky 인식 파괴
- **🔴 Gamma TTA 실패 확정** (2026-03-04): Gamma TTA [1.0,1.5,2.0,2.5] M=76.10 (baseline 대비 -5.37). OOD gamma→Sky 붕괴(-28.84pp). 육안으로 boundary 보이는 것 ≠ 모델 성능 향상. Memory attention 연쇄 오염으로 모든 모달리티 악화
- **NIGHT_AUG hardaug4가 최적이나 포화 상태**: 추가 튜닝으로는 +1~2pp가 한계
- **P9의 단순한 CrossModalFusionHead가 가장 효과적**: 고정 비율 img:27.5%, lidar:35.5%, thermal:37.0%. SAM2 memory attention이 implicit adaptation 수행
- **Multi-Scale FPN은 효과 있음**: P16(3.17)→P17(33.35) Sky +30pp. 하지만 근본 해결은 아님
- **Aux mask 품질이 근본 한계** (ISSUE-008): frozen backbone 위의 lightweight decoder로는 entropy/energy 추정 신뢰도 확보 불가
- **CRM/ZERO는 P9에 유익**: aux decoder 없으므로 shortcut 없고, multimodal 강제 학습에 도움. P9+hardaug5(CRM/ZERO만 제거) ablation 필요
- **Broader aug 실패**: hardaug6 [0.01, 0.60] + gamma>1.0은 test에 없는 조건에 capacity 낭비. 단, NightSim **내부** gamma 변경이었음 — 독립 RandomGammaWide는 다른 접근
- **학습 가능 fusion 실패 확정**: P12~P19 8개 실험 전부 P9(고정 상수)보다 나쁨
- **🔴 I2I Translation 양방향 실패 확정** (2026-03-05): Day-Trans M=78.90(-2.57), Night2 M=73.04(-8.43). Night2 Sky 붕괴 -49.62pp(130/200장). Real night의 정보 비가역 소실 → pixel-level I2I로 domain gap 해소 불가. day2night2day≈day 관찰이 증거 (I2I의 "night"은 정보 보존된 fake night)
- **다음 방향**: FDA augmentation (frequency domain style transfer) 또는 Self-training (pseudo-label) 검토

---

## Object Detection 확장 계획 (2026-03-19)

### 배경

- MemorySAM의 memory attention cross-modal fusion을 유지하면서 object detection으로 확장
- P22가 fpn[0,1,2] 3레벨 전부 사용 (DeBA-FP MultiScale) → multi-scale detection에 적합
- mmdetection 이사 대신 **현재 코드베이스에서 detection head만 추가**하는 방식 채택
  - 이유: SAM2 memory attention, MoE LoRA, UAMM/AMF가 SAM2 내부 API에 깊이 의존 → mmdet 이식 비용이 처음부터 만드는 것과 동일

### 변경 범위

**변경 없음 (그대로 재사용)**:
- SAM2 Encoder (Hiera-B+)
- MoE LoRA (SoftMoE_LoRA_Layer × 48)
- Memory Attention (cross-modal fusion)
- FPN (fpn[0]=32ch, fpn[1]=64ch, fpn[2]=256ch)
- UAMM / AMF
- DeBA-FP (P22)

**신규 구현 필요**:
- Detection Head (FCOS anchor-free 또는 DETR transformer decoder)
- Loss (Focal Loss + L1/GIoU)
- Post-processing (NMS)
- Dataset loader (bbox annotation 로딩)
- Detection 전용 augmentation (bbox-aware)
- Evaluation metrics (mAP, AP50, AP75)

### 파일 구조 계획

```
drone-MemorySAM/
├── semseg/                              # 기존 segmentation (변경 없음)
│   └── models/sam2/sam2/
│       └── sam_lora_image_encoder_seg.py # P9/P22 공유 backbone
│
├── objdet/                              # 신규 detection 모듈
│   ├── __init__.py
│   ├── models/
│   │   └── heads/
│   │       ├── __init__.py
│   │       ├── fcos_head.py             # Anchor-free detection head
│   │       └── detr_head.py             # Transformer decoder detection head
│   ├── datasets/
│   │   └── multiaqua_det.py             # bbox annotation 로딩
│   ├── losses.py                        # Focal + GIoU + L1
│   ├── metrics.py                       # mAP 계산
│   ├── augmentations_det.py             # bbox-aware augmentation
│   └── utils/
│       └── nms.py                       # NMS post-processing
│
├── train_det.py                         # detection 학습 스크립트
├── val_multiaqua_det.py                 # detection 평가 스크립트
└── configs/det/                         # detection 전용 config
```

### Detection Head 후보

1. **FCOS (anchor-free)**: 변경량 최소, 현재 segmentation head와 구조 유사 (per-pixel prediction), ~200줄
2. **DETR (transformer decoder)**: SAM2 memory attention과 구조적으로 가장 자연스러운 확장, object query가 cross-modal fused feature를 참조, ~300줄

### Dataset: COCO Format + 모달리티별 Path Prefix

COCO JSON format을 기반으로 하되, `images`의 `file_name`에 모달리티별 prefix를 부여하여 멀티모달 로딩을 지원한다.

**Annotation JSON 예시**:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "000001.png",
      "width": 2048,
      "height": 1024
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 200, 50, 80],
      "area": 4000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "boat"}
  ]
}
```

**Config에서 모달리티별 경로 지정**:

```yaml
DATASET:
  FORMAT: coco
  ANNOTATION: /path/to/annotations.json
  MODALITIES:
    img:
      ROOT: /path/to/data/zed          # file_name 앞에 이 경로를 붙임
    lidar:
      ROOT: /path/to/data/lidar_processed
    thermal:
      ROOT: /path/to/data/thermal_processed_fieldscale
```

**로딩 로직**: `images[i]["file_name"]`은 공통이고, 각 모달리티의 `ROOT`를 prefix로 붙여서 로드:

```python
for modality, cfg in modalities.items():
    path = os.path.join(cfg['ROOT'], image_info['file_name'])
    img = load_image(path)  # RGB / depth / thermal 각각
```

이렇게 하면 기존 COCO 도구(pycocotools)와 호환되면서도 멀티모달 로딩이 가능.

### 구현 우선순위

1. `objdet/` 폴더 구조 생성
2. Dataset loader (COCO format + 모달리티별 path prefix)
3. Detection Head (FCOS 우선 → DETR 후속)
4. Loss 함수 (Focal + GIoU)
5. train_det.py 작성
6. NMS + mAP evaluation (pycocotools 활용)
7. val_multiaqua_det.py 작성

### 상태: ✅ 구현 완료 (2026-03-19) — 데이터셋 준비 후 학습 가능

**구현 완료 파일**:
- `objdet/` 모듈 전체 (패키지 구조)
  - `datasets/multimodal_det.py`: COCO JSON + 멀티모달 path prefix dataset loader
  - `models/heads/fcos_head.py`: FCOS anchor-free detection head (3-level FPN: 32/64/256ch)
  - `models/det_model.py`: MemorySAMDetector (seg backbone + FCOS head wrapper)
  - `losses.py`: FCOSLoss (Focal + GIoU + Centerness BCE)
  - `utils/nms.py`: class-aware batched NMS
  - `metrics.py`: pycocotools 기반 COCO mAP 평가
  - `augmentations_det.py`: bbox-aware augmentation (hflip, brightness, crop)
- `train_det.py`: 학습 스크립트 (AMP, warmup, cosine LR)
- `val_det.py`: 평가 스크립트 (bbox visualization 포함)
- `configs/det/det_P9_base.yaml`: P9 backbone 기본 config 템플릿

**사용법**:
```bash
# 학습
python train_det.py --cfg configs/det/det_P9_base.yaml

# 평가
python val_det.py --cfg configs/det/det_P9_base.yaml \
    --det_checkpoint outputs/det/det_P9_base/best_checkpoint.pth \
    --mode val --save_vis
```

**TODO (구현 시 반드시 확인)**:

1. COCO format detection annotation JSON 준비 (외부 데이터셋)
2. seg_model이 `_last_backbone_fpn` 속성을 노출하도록 P9/P22 forward에 hook 추가
3. 실제 학습 돌려서 검증
4. `configs/det/det_P9_base.yaml`의 경로 수정:
   - `ANNOTATION_TRAIN` / `ANNOTATION_VAL`: COCO JSON 경로
   - `MODALITIES.img.ROOT` / `lidar.ROOT` / `thermal.ROOT`: 모달리티별 이미지 디렉토리
5. `det_P9_base.yaml`의 `N_CLASSES`: 실제 detection 데이터셋 클래스 수로 변경
6. Dataset loader(`objdet/datasets/multimodal_det.py`)의 COCO JSON 구조:
   - `images[i].file_name`은 모든 모달리티가 공유하는 공통 파일명
   - 각 모달리티는 config의 `MODALITIES.{modal}.ROOT` + `file_name`으로 이미지 로드
   - `annotations[].bbox`: COCO format `[x, y, w, h]`
   - `categories[].id`: 원본 카테고리 ID → 내부적으로 0-based contiguous index로 매핑
