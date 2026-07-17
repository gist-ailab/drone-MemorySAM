---
legacy_id: 11
legacy_file: 11_sam3_rbma_plan.md
moved: 2026-07-08
---

# SAM3 RBMA 포팅 — 진행 플랜 & 체크리스트

> 시작: 2026-06-16
> 목표: SAM2 기반 RBMA(P28)를 SAM3(`semseg/models/sam3`)로 이식.
> 원칙: **plain LoRA로 시작** (SoftMoE/SQG 미이식), RBMA bias는 `cross_attn_image(attn_mask=memory_mask)`에 주입.
> 표기: `[ ]` 미진행 · `[~]` 진행중 · `[x]` 완료 · `[!]` 막힘/보류

---

## 확정된 사실 (코드 레벨, done)
- [x] SAM3 tracker = `Sam3TrackerBase` (`build_tracker()`), SAM2 등가 빌딩블록 보유 (`forward_image`, `_prepare_backbone_features`, `_prepare_memory_conditioned_features`, `_forward_sam_heads`).
- [x] LoRA 타겟 = `tracker.backbone.visual.trunk.blocks[*].attn.qkv` (SAM3 `vitdet.ViT`, SAM2 `trunk.blocks`와 동형).
- [x] RBMA bias 주입점 = `sam3/model/encoder.py` `TransformerEncoderLayer.cross_attn_image(attn_mask=memory_mask)` (`MultiheadAttentionWrapper`, 인자 이미 존재 → SDPA 패치 불필요).
- [x] memory(key) 순서 = spatial memory(프레임=모달 블록) → obj_ptr (`_prepare_memory_conditioned_features`의 `to_cat_prompt`).
- [x] `Sam3TrackerBase.forward`는 NotImplementedError → 우리가 modality-as-frame forward를 자체 조립 (P27 방식).

---

## Phase 1 — 모델 생성 + plain LoRA + 백본 forward
- [x] 1.1 새 모듈 `semseg/models/sam3/sam3_lora_rbma.py` 생성, `_LoRA_qkv`(shape-무관) + `inject_plain_lora`(named_modules 스캔, qkv out=3×in, name_filter). 단위테스트 통과(3D/4D forward, ViT만 선택·decoder 제외, B=0 init-identity).
- [x] 1.2 `build_sam3_tracker()` + `LoRA_Sam3_RBMA.__init__`: `build_tracker(apply_temporal_disambiguation=False, with_backbone=True)` → `Sam3TrackerPredictor`(random init OK, 오프라인). 가중치는 `strict=False` 로드 훅(checkpoint_path/load_from_HF).
- [x] 1.3 `inject_plain_lora`로 **32 ViT 블록 qkv(q,v)** 주입, encoder freeze. LoRA는 `_LoRA_qkv` 내부 단일 등록(state_dict 중복 0). trainable=129(LoRA 128 + lambda_bias).
- [x] 1.4 스모크: build + `forward_image(1,3,1008,1008)` → FPN [(1,32,288,288),(1,64,144,144),(1,256,72,72)]. PASS.

> 메모: 모듈 = `semseg/models/sam3/sam3_lora_rbma.py`. PYTHONPATH=`semseg/models/sam3` 필요. 빌드/forward는 PYTHONPATH + `HF_HUB_OFFLINE=1`로 가중치 없이 동작. ViT 경로 = `backbone.vision_backbone.trunk.blocks[*].attn.qkv` (32블록, dim 1024). 입력 = 1008×1008.
> 설치한 deps: pycocotools, psutil, ftfy, regex, iopath, einops, hydra-core, python-rapidjson, numba, opencv-python.

> ⚠️ **가중치 블로커**: `facebook/sam3` gated, `jmmh` 계정 라이선스 동의했으나 **Meta 수동 승인 대기 중**("awaiting a review"). 코드/shape 검증은 random-init로 진행, 실제 가중치는 학습 시 필요(승인 후 다운로드 or B200에서).
> 🔒 보안: HF 토큰이 대화에 노출됨 → 작업 후 재발급 권장.

## Phase 2 — modality-as-frame forward (bias 없이, 등가중) ✅
- [x] 2.1 모달별 `forward_image` → `_prepare_backbone_features`
- [x] 2.2 memory bank(output_dict: cond=frame0, non_cond=나머지) + `track_step` 프레임 루프 (SAM3 track_step == SAM2 + `image` 인자)
- [x] 2.3 `track_step` 내부 `_prepare_memory_conditioned_features`+`_forward_sam_heads` → `pred_masks_high_res` 등가중 평균
- [x] 2.4 스모크: 3-modality forward(random init, 1008²) → fused (1,1,1008,1008), per-modal (1,1,1008,1008)×3. PASS (GPU).

> 발견: `Sam3TrackerBase.track_step` 존재(929) → P25/P27 forward 거의 그대로 미러. 출력 M=1(무프롬프트) → Phase 4에서 semantic num_classes 개조 필요. `forward`가 `self._last_per_modal_out` 보관(Phase 3/4 재사용).

## Phase 3 — RBMA bias (핵심)
- [x] 3.1 per-modality predictive entropy → reliability map: `_sem_logits`+`_reliability_from_logits` (1-norm_entropy). 신호원 = SemanticHead(standalone backbone feat). Phase 4.1로 구현됨.
- [ ] 3.2 `_compute_memory_mask`: reliability를 memory(key) 시퀀스의 모달 블록 컬럼에 broadcast (memory layout: `to_cat_prompt` = 프레임별 spatial 블록 + obj_ptr; Sk=k.shape[1])
- [x] 3.3 bias 주입: **RoPEAttention SDPA 패치** + forward_pre/post hook으로 `cross_attn_image._rbma_attn_bias` set/clear
- [x] 3.4 attn 의미 확인: ⚠️ **정정** — tracker cross-attn은 `MultiheadAttention`이 아니라 **RoPEAttention**(`q,k,v,num_k_exclude_rope`, SDPA attn_mask 없음). model_builder의 MHA는 detector용. → `sam3/sam/transformer.py` SDPA를 P27식으로 패치(`_rbma_attn_bias`, math/mem-efficient 커널).
- [x] 3.5 스모크: 메모리 키 절반 suppress → 출력 변화(mean|diff|=3e-4), fn 8회(4층×2프레임), q=(1,5184,256) k=(1,10376,64). **bias 주입점 검증 완료.**

> **메커니즘 검증 완료.** 남은 3.1/3.2(실제 reliability 신호 + 모달→키컬럼 매핑)는 Phase 4(semantic decoder로 entropy 신호) 이후 마무리.
> 변경: `semseg/models/sam3/sam3/sam/transformer.py` (RoPEAttention/Attention SDPA에 `_rbma_attn_bias` 경로 추가). `sam3_lora_rbma.py`에 hook(set/clear) + `_mem_bias_fn`.
> bias shape: (B,1,1,Sk) broadcast → SDPA logits (B,nh,Lq,Sk). Sk=memory 토큰수(프레임별 블록+obj_ptr).

## Phase 4 — decoder semantic 개조 + 불확실성 신호 확정
- [x] 4.1 **SemanticHead**(conv 256→128→num_classes) 추가 — SAM3 mask decoder는 SAM 마스크(M=1/3)만 내므로 MemorySAM식 별도 conv head. per-modality 로짓(1,4,72,72) + reliability 검증.
- [x] 4.2 출력 융합: `_prepare_memory_conditioned_features` 반환(pix_feat_with_mem) 캡처 래퍼 → 각 모달 **memory-conditioned feat에 sem_head** → 평균 → upsample. **출력이 RBMA bias에 결합됨 검증**(suppress vs normal diff 5.8e-4).
- [x] 4.3 loss: `compute_losses` = main CE(fused) + per-modality aux CE(×0.5). `_LoRA_qkv`를 non-in-place(cat delta)로 변경 → backward autograd-safe(단위검증). fused=mean(per_modal_sem)이라 main CE만으로도 sem_head/LoRA/λ 전부 grad 흐름.

## Phase 3.2 — 완료 ✅
- [x] 3.2 `_rbma_bias_fn`: per-modality reliability를 memory 키 컬럼에 매핑. `num_obj_ptr`는 encoder pre-hook으로 캡처, 나머지 spatial을 프레임수 균등분할(tokens_per_frame=√), 각 블록에 해당 모달 reliability(resize+flatten), 프레임 간 센터링, ×λ, (B,1,1,Sk) 반환. **전체 RBMA forward 동작 + 출력 결합 검증.**
  - ⚠️ 가정(작은 m): memory 키 블록이 프레임 순서. m=3~4에선 성립(frame0=cond, 나머지 non-cond 최근). 큰 m이나 memory 선택 변경 시 순서 재확인 필요.

## Phase 5 — 학습 파이프라인 통합 ✅
- [x] 5.1 전용 트레이너 `train_sam3_rbma.py` (SAM2 train의 proto/gate/MI 결합 회피). 데이터셋/증강/OhemCrossEntropy/scheduler/Metrics 재사용, loss=`compute_losses`, AMP bf16, DDP(torchrun) 지원, best/last 체크포인트.
- [x] 5.2 config: `configs/b200-multiaqua_rgbtl_SAM3RBMA_hardaug8_physaug.yaml`(4cls), `configs/b200-deliver_rgbdel_SAM3RBMA_physaug.yaml`(25cls). IMAGE_SIZE=1008, CHECKPOINT_PATH(gated) 옵션.
- [x] 5.3 배선 검증: 트레이너 compile OK, config→build_model(random) 0.82M trainable, get_loss/scheduler/Metrics 연결 OK.
  - 실행: `PYTHONPATH=semseg/models/sam3 [torchrun ...] python train_sam3_rbma.py --cfg <cfg>`

## Phase 6 — 검증
- [x] 6.1 컴포넌트 스모크: eval forward(출력 (B,C,1008,1008)), RBMA bias 출력결합(suppress diff), `_LoRA_qkv` backward, sem_head/reliability, compute_losses 배선 — 전부 PASS(random init, 로컬).
- [ ] 6.2 전체 학습 스모크 1-step(3-modal forward+backward): 로컬 GPU 메모리 부족으로 보류 → **서버/B200에서** (메모리 충분, 가중치 승인 후).
- [ ] 6.3 B200 학습 (DeLiVER + MULTIAQUA).

## 상태 요약 (2026-06-16)
**RBMA-on-SAM3 구현 완료.** plain LoRA + modality-as-frame + RBMA logit-bias(패치된 RoPEAttention) + semantic head + training-free entropy reliability + 전용 트레이너 + config. 모든 컴포넌트 random-init 검증. 남은 것: 서버에서 (가중치 승인 후) 실학습.
**변경/신규 파일**: `semseg/models/sam3/sam3_lora_rbma.py`(신규), `semseg/models/sam3/sam3/sam/transformer.py`(RBMA SDPA 패치), `train_sam3_rbma.py`(신규), `configs/b200-*-SAM3RBMA-*.yaml`(신규×2).
**설치 deps**: pycocotools psutil ftfy regex iopath einops hydra-core python-rapidjson numba opencv-python.

---

## TODO (future work, 별도)
- [ ] 입력/모달리티 조건부 **expert LoRA** (SoftMoE 대체, 환경 신호 주입 — gate 붕괴 회피). plain LoRA + RBMA 검증 후.

## 막힘/메모
- (없음)
