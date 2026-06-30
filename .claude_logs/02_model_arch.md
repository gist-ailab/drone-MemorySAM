# 모델 아키텍처 상세 (Model Architecture Details)

> 최종 업데이트: 2026-06-30

## P30-Det — P30 백본 detection 확장: Reliability-router 융합 + Object-Query decoder + FCOS aux (2026-06-30)

**상태**: **구현 완료 (CPU smoke 통과, lecun 학습 대기)**. 브랜치 `worktree-p30-det` (develop 기준, P30 seg 보유). 계보: `LoRA_Sam_P30_Det(LoRA_Sam_P30)` — RBMA(P27/P28) + SDC(P29) + P30 두 기구를 **그대로 상속**하고, P30 seg의 두 노벨티를 **detection 헤드로 번역**. P29-Det(P28 기반, mean 융합 + FCOS)의 후속.

**동기**: P29-Det는 detection FPN 융합을 **단순 mean**(`MODALITY_FUSE: mean`, `AMF_MODE: uniform`)으로 처리 → 이 프로젝트의 핵심 노벨티(RBMA/SQG 신뢰도 기반 융합)가 detection feature 융합엔 미사용이었음. 실내 RGB+LiDAR+Thermal은 모달리티별 신뢰도 편차가 큰데 mean이 이를 버림. 사용자 결정: **"memory attention을 활용하되, 현재 P30 seg 모델 기반으로 detection 확장"** → P30 seg의 ①②를 detection으로 이식.

**Detection feature 브릿지 (P27에 이식, P29-Det서 확장)**: `extract_det_features()`가 encoder + cross-modal memory-attention(track_step loop)을 그대로 돌려 in-graph 캡처:
- `fpn0` (B,32,s4) · `fpn1` (B,64,s8) — encoder detail (per modality)
- `mem` (B,256,s16) — memory-conditioned coarse (frame0 = +no_mem_embed; frame≥1 = memory attention + RBMA bias) (per modality)
- `output` (B,Cseg,s4) — per-modality seg logits → **training-free reliability `1−H(softmax)/logCseg`** 소스 (**P30-Det 신규 노출**, `_capture_det_features` 플래그로 behaviour-neutral)

**기구 ① (P30 ② 이식) — Reliability-anchored router 모달 융합**: P29-Det의 mean을 **per-level `ReliabilityAnchoredRouter`**(sam_lola_utils.py 재사용)로 교체. 각 FPN level에서 `w = softmax_modality(learned_logits(feat_i) + λ·reliability_i)`, reliability는 위 per-modal seg `output`에서 도출해 level 해상도로 resize. zero-init conv head → 초기 reliability-구동(상수수렴 방지), 이후 자동 학습. fused level = `Σ w_i·feat_i`. 선택적 diversity reg `router_reg`(`ROUTER_REG_LAMBDA>0` → `−λ·entropy` 가산).

**기구 ② (P30 ① 이식) — Object-Query decoder (primary head)**: P30 seg의 class-token decoder를 **per-class mask → per-object (box+class)**로 번역. N개 object query가 **융합된 `mem`(memory-conditioned, RBMA bias 보유)**에 cross-attend(DETR류 transformer decoder, sine PE) → `pred_logits (B,N,C+1)` + `pred_boxes (B,N,4 cxcywh)`. **Hungarian set loss**(CE+L1+GIoU, no-object class, `eos_coef=0.1`). 이것이 사용자가 강조한 **"memory attention 활용" 헤드라인**. (`objdet/models/heads/query_decoder.py`: `ObjectQueryDecoder`/`HungarianMatcher`/`SetCriterion`/`decode_queries`)

**FCOS aux**: 기존 P29-Det FCOS dense head를 **보조**로 유지(`USE_FCOS_AUX: true`) — 융합 FPN 공유, 조기 수렴 안정화. 총손실 = `W_QUERY·query_set_loss + W_FCOS·fcos_loss − ROUTER_REG_LAMBDA·router_entropy`. eval은 **query detection(primary)** 반환(`decode_queries` → NMS).

**구현/검증**: `MemorySAMDetectorP30`(`objdet/models/det_model.py`) = routers(per-level) + FPNNeck + ObjectQueryDecoder + FCOS aux. train_det는 `MODEL.DET_MODEL: MemorySAMDetectorP30`로 분기. config `configs/det/det_P30_indoor.yaml`. **CPU smoke 통과**(백본 mock): loss 유한, grad가 query_decoder·routers·neck·FCOS·백본까지 전파, eval px-space detection, state_dict 왕복. **scipy 설치**(Hungarian; 미설치 시 greedy fallback 내장). 미검증: SAM2 풀로드 e2e(=lecun 1-GPU forward 권장), AP.

**리스크**: (1) query decoder는 소규모 indoor에서 DETR 수렴 느릴 수 있음 → FCOS aux로 완화. (2) reliability는 seg-class 엔트로피 기반(detection-class 아님) — 모달 신뢰도 proxy로는 타당하나 직접 신호 아님. (3) IMG_SIZE 1024 필수(SAM2 assert), mem=64² cross-attn 토큰 4096.

---

## P30 — Class-token decoder + Reliability-anchored learned modality router (2026-06-28)

**상태**: **구현 완료 (학습 대기, P28 종료 후 GPU 2,3)**. 계보: `LoRA_Sam_P30(LoRA_Sam_P29)` — P29(SDC)+RBMA 상속, 두 기구 추가(둘 다 config-gated, 기본 OFF → P28/P29 불변).

**동기 (P28 실패 분석에서 직접 도출, `analyze_failures.py`)**:
- **실패는 weather가 아니라 class-driven**: per-condition mIoU는 타이트(night 0.526 … rain 0.561)인데 per-class가 양극화 — Road/Sky/Car ~0.9+ vs **Water 0.00, Bridge 0.00, Wall 0.035, Other 0.054, Dynamic 0.083, Ground 0.097, TrafficLight 0.137**. ~7개 thin/rare class가 mIoU를 끌어내림 = 70 갭 거의 전부.
- **융합이 2-모달로 퇴화**: ablation(cloud) drop-depth Δ−0.224, drop-RGB Δ−0.097, **drop-event Δ−0.000, drop-LiDAR Δ+0.001 = event/LiDAR 사실상 미사용**. 현 융합 `m_feat=Σ q_uamm_norm[i]·f_i`(`sam_lora_image_encoder_seg.py:7140`)는 **class-agnostic per-pixel scalar**(`q_uamm_norm` (B,1,H,W), SQG quality softmax `:7033`)라 "Water엔 LiDAR 가중" 표현 불가.

**기구 ① Class-token decoder (rare-class fix)**: SAM2 mask decoder를 class token으로 repurpose하는 아이디어를 이식 — C개 학습 class query가 **융합된 cross-modal memory feature `m_feat`**(전 모달 + RBMA bias 보유, `:7140`)에 cross-attention해 per-class mask `(B,C,H,W)`를 직접 생성. thin/rare class에 능동적 query 메커니즘 부여(per-pixel argmax에서 지배 class에 밀리는 구조 제거). **SAM3-RBMA에서 decoder repurpose가 class-collapse를 깸(val 8.49→16.27)** 의 SAM2 이식. **근사 구현**(faithful approximation): `ClassTokenDecoder`(sam_lola_utils.py) = 경량 transformer-decoder block(self+cross attn+FFN) + dynamic-kernel dot-product. 실제 `sam_mask_decoder` 가중치 수술 아님(명시). 통합: `LoRA_Sam_P30.forward`에서 super 반환의 grad-attached `m_feat`에 post-hoc 적용 → end-to-end 학습. config `MODEL.CLASS_TOKEN_DECODER{ENABLE, DIM}`.

**기구 ② Reliability-anchored learned router (dead-modality fix)**: 고정 UAMM scalar를 **학습 router**로 교체하되 **RBMA reliability로 anchor**해 상수수렴(P10–P27 'gate 상수수렴', ISSUE-002/015) 방지. `w = softmax_modality(learned_logits(feat_i) + λ·reliability_i)`, reliability = `1 − H(softmax(output_i))/logC`(training-free). 학습 conv head **zero-init → 초기 w는 reliability-구동(붕괴 없음)**, 이후 비율을 자동 학습(사용자 요구: 라벨 통계가 아니라 모델이 자동 학습). `per_class=true` → per-class 모달 가중(B,C,H,W)로 "class가 자기를 보는 모달에 라우팅" → event/LiDAR 부활. 통합: P26 fusion을 overridable hook `_fuse_outputs`로 추출(기본 byte-identical → P26~P29 불변), `LoRA_Sam_P30._fuse_outputs`가 router 적용. `ReliabilityAnchoredRouter`(sam_lola_utils.py). 선택적 diversity reg `self._router_reg`(모달-mixing entropy) → trainer가 `−λ_router·reg`로 가산. config `MODEL.LEARNED_ROUTER{ENABLE, PER_CLASS, ANCHOR_LAMBDA, REG_LAMBDA}`.

**왜 각 finding을 고치나**: ①은 rare-class collapse(finding 1) 직격(class query가 자기 영역 능동 탐색); ②는 dead-modality(finding 2) 직격(reliability-anchored 학습 router가 event/LiDAR에 의미 가중) + per_class로 rare-class가 geometry 모달을 끌어씀 → 두 finding의 coupling 해소.

**Ablation 계획**: ① class-token decoder on/off (rare-class IoU: Water/Wall/Bridge 0 탈출?), ② router scalar vs per_class vs 고정 UAMM (event/LiDAR ablation Δ가 유의미 음수로 바뀌나 = 부활 확인), ③ anchor λ sweep(0=순수 학습 vs 큰 λ=reliability 지배; 상수수렴 여부), ④ `analyze_failures.py`로 per-condition×class + modality-ablation을 P28/P29 대비 측정. 성공 기준: Water/Wall/Bridge>0, event/LiDAR Δ 유의미 음수.

**리스크**: (1) **frozen-backbone 천장(ISSUE-008)** — rare class가 frozen SAM2 feature에 애초에 안 담겼으면 ①②로도 한계; multi-scale FPN/②의 모달 부활로 완화. (2) class-token decoder는 근사 구현이라 실제 SAM decoder 대비 약할 수 있음. (3) **런타임 미검증**: 두 모듈은 CPU dummy smoke(forward+backward+grad+reliability-anchor 초기성) 통과했으나, `LoRA_Sam_P30`의 full forward(track_step 내부와 _fuse_outputs 상호작용)는 SAM2 로드 없이 compile-only 검증 → 학습 전 main이 GPU 1-forward로 확인 권장. 노벨티 = [12_novelty_and_related_work.md](12_novelty_and_related_work.md) §2.8.

---

## P29 — Self-Derived Condition (SDC) 라우팅: label-free 조건 latent + prototype bank → FiLM Soft-MoE LoRA gate (2026-06-27)

**상태**: **설계 완료 (구현 대기)**. 계보: `LoRA_Sam_P29(LoRA_Sam_P28)` — RBMA 기구(P27/P28 memory-attention logit bias)는 그대로 두고, **Soft-MoE LoRA의 라우팅(gate) 조건화**를 재설계.

**동기 / 근본 원인 (라우팅 비특화 진단)**:
- **조건(day/night/snow-rain)이 라우터에 구조적으로 안 보임**. P28의 gate 조건은 `self.modal_embed(modal_idx)`(`nn.Embedding(num_modalities, cond_dim=8)`, `sam_lora_image_encoder_seg.py:6715, 6801-6803`)뿐 → 라우팅을 바꿀 수 있는 입력은 **"어느 모달리티냐"가 전부**. 환경 조건은 입력이 아니므로 per-condition 특화가 원천 불가. (P12는 RGB mean/std 통계 `:1621` — RGB-only·전역 스칼라로 매우 약함.)
- **존재하는 조건화도 너무 약함**. `SoftMoE_LoRA_Layer.forward`(`sam_lola_utils.py:690-709`)는 조건을 gate logit에 **가산 bias**(`cond_proj`)로 주입하는데 **zero-init**(`:677-679`)이라 초기 기여 0, 전 토큰 broadcast로 per-token `gate(x)`와 가산 경쟁 → modal_embed가 near-constant로 표류 가능.
- **특화 압력 부재 + collapse 유발 init**. 순수 soft-blend(`:725-730`, top-k/load-balance 없음; P11 MI-loss는 취소). `experts_b` zero-init(`:684`)→초기 expert 출력 0→gate gradient≈0→rich-get-richer. 측정(ISSUE-002): Block9 argmax E1≈0~10%(img)/0~0.5%(lidar) = **E1 dead expert**, soft-MoE가 사실상 평균 단일 LoRA로 동작(ISSUE-015 #7 "gate 상수수렴").
- **"붕괴" vs "오측"은 축이 다름**. viz 콜백은 **spatial-mean** gate 저장(`:714-716`)→uniform처럼 보이는 artifact. per-token 분석(CLAUDE.md)은 entropy_ratio≈0.55/max_weight≈0.72 → **per-token/region 라우팅은 분화**. 그러나 **per-modality 특화는 약하고(E1 dead), per-condition 특화는 설계상 부재**.

**P29 설계 (Proposal A = 헤드라인): SDC latent + prototype bank → FiLM router**
- **SDC 모듈**: RGB/초기 backbone feature(`fpn[0]`)에서 전역 시각 조건 descriptor 산출 = **GAP + 채널 mean/std** → projection으로 조건 latent `z_c`(latent_dim).
- **Condition-prototype bank**: 학습되는 K개(K≈4~8) prototype에 `z_c`를 cosine/VQ로 **soft-assign**(label-free). 학습은 entropy/contrastive **clustering term + 본 seg loss**만 사용(조건 라벨/텍스트 불사용) → day/night/snow가 prototype으로 자연 출현하도록.
- **라우터 주입(FiLM)**: gate 입력을 `[modal_embed ⊕ z_c]`로 구성하고, gate logit에 **FiLM(scale+shift) 변조**로 주입 → 기존 zero-init 가산 `cond_proj`(`:705-709`)를 대체. (multiplicative라 zero-init no-op·가산 약점 탈출.)
- **텐서 흐름/플러그 지점**: SDC는 encoder당 1회(이미지당) 계산, `SoftMoE_LoRA_Layer.set_condition`이 `[modal_embed, z_c]`를 받도록 확장, forward `:705-709`의 가산 블록을 FiLM으로 교체.
- **제안 config 키**: `MODEL.SDC: {ENABLE: true, K: 6, LATENT_DIM: 32, CLUSTER_WEIGHT: ...}`, gate `COND_MODE: film`(vs `add`/`none`), label-free term 가중.

**P29-B (확장, optional combine): Reliability-Routed Experts — RBMA를 라우팅으로 확장**
- RBMA의 **training-free 신뢰도** `B_i = 1 − H(softmax(Decoderᵢ(fᵢ)))/log C`를 **라우터 prior**로 재사용: 신뢰도가 어느 expert가 켜질지/gate를 bias(선택적으로 expert군↔신뢰도 regime 경량 supervision).
- 의의: **하나의 reliability field가 두 곳을 구동** — 기존 RBMA의 memory-attention logit bias + 신규 LoRA expert routing. 무감독 soft-softmax(uniform-collapse 원인)를 GT-free 의미 신호로 대체. RBMA를 "융합 전용"에서 "라우팅+융합 통합 reliability 프레임워크"로 확장.

**Proposal C (지원 ablation): reliability/importance 기반 pruning** — dead expert(ISSUE-002 E1)·rank를 신뢰도×utilization 중요도로 구조적 prune, 또는 per-token 신뢰도-salient 채널만 보존(feature pruning). RBMA 신뢰도를 **중요도 기준**으로 재사용. 헤드라인 아님, A/B의 "kept expert가 의미 있다"는 분석용.

**Ablation 계획**: ① modal-only(P28) vs +SDC, ② 가산 bias vs FiLM, ③ K sweep, ④ **prototype↔DELIVER 조건 라벨 post-hoc probe**(무감독 latent가 day/night/cloud/rain/sun/fog/night를 복원하는지), ⑤ **per-condition mIoU** 분해(night/rain에서 이득 기대), ⑥ P29-B reliability-prior gate vs 학습 gate(특화도 = per-modality argmax·per-token entropy_ratio).

**리스크**: (1) 무감독 prototype이 **nuisance 요인**(장면 레이아웃 등)으로 군집될 수 있음 → probe + 필요시 약한 self-supervised 조건 contrast(단 label-free 유지). (2) **노벨티≠mIoU**: 실제 천장이 frozen-backbone feature 품질일 수 있음(ISSUE-008) → 라우팅 재설계가 방법론적 기여여도 수치는 소폭일 수 있음, per-condition 분해로 방어. (3) 리뷰어 "왜 DELIVER 조건 라벨 안 씀?" → **무라벨 야간 드론 배치 전제**(배포 시 조건 라벨 없음)로 답, label-free latent가 라벨 조건과 일치함을 probe로 입증.

**노벨티 포지셔닝**: label-free·image-derived·router-level 조건화는 CAFuser(CLIP/text 조건)·DGFusion(depth+depth-GT) 어느 쪽과도 다름. 상세 = [12_novelty_and_related_work.md](12_novelty_and_related_work.md) §2.7.

---

## P28 — RBMA: Reliability-Biased Memory Attention (2026-06-15)

**계보**: `LoRA_Sam_P28(LoRA_Sam_P27)`. P27의 additive memory-attention logit-bias 기구는 그대로, **bias 신호만** 교체.

**P27 기구 (재사용)**: cross-modal memory attention에서
`attn = softmax(QK^T/√d + λ·B) V`, `B[memory_token]` = 그 토큰 source-modality의 신뢰도 맵을 memory grid에 대응, `λ`=학습 스칼라(`self.lambda_bias`). `RoPEAttention._p27_attn_bias`(SDPA `attn_mask`)로 pre-softmax logit에 주입, `memory_attention`의 forward_pre_hook에서 매 frame 설정.

**P27 → P28 변경 (신호)**:
- P27 bias 신호 = SpatialQualityGating(SQG) quality_logits → **B-2 진단: frozen-feature 예측기 underfit, lidar/thermal 평탄, 정적 RGB-붕괴.**
- P28 bias 신호 = **per-modality decoder의 training-free 예측 불확실성**:
  - `aux_logits_i = _auxiliary_decode_single(per_modal_decoders[i], vision_feats[i], ...)` (모달리티 단독 디코드, memory 융합 이전)
  - `H_i = -Σ_c softmax(aux_logits_i) log softmax(aux_logits_i) / log C` (per-pixel, [0,1])
  - `reliability_i = 1 - H_i`, 모달리티 간 per-pixel 평균 0 센터링 → `_p27_attn_bias` 신호
  - `torch.no_grad()`로 detach → per_modal_decoders는 aux-CE로만 학습, bias는 파생 routing 신호(λ만 학습)
- **순환 없음**: 불확실성은 단독 디코드(융합 전)에서, bias는 융합 attention에 주입.

**구현**: P27에 `_compute_bias_source(quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m)` 훅 추가(기본 identity=SQG). P28은 이 메서드만 오버라이드.

**설계 의의**:
- 노벨티 축 = **attention LOGIT additive bias** (선행연구는 feature-multiply/output-scale/loss; 전례 0). 신호의 차별점 = **training-free**(학습 evidential/HD head 불필요, vs UTFNet/HyperDUM).
- B-2 병목(SQG)을 bias 경로에서 제거. 데이터셋 무관(uncertainty는 보편 → DeLiVER/MUSES/MCubeS 공통, coverage mask 불필요).

**평가 설정**: `AMF_MODE: uniform`(출력 융합 등가중) → 적응은 오직 RBMA bias = 순수 효과 측정. configs: `b200-deliver_rgbdel_P28_physaug.yaml`, `b200-multiaqua_rgbtl_P28_hardaug8_physaug.yaml`.

**예정 ablation**: SoftMoE LoRA→단일 LoRA, SQG/KL teacher 제거, AMF uniform↔sqg_quality, λ 고정↔학습.

---

## 공통 기반: MemorySAM

### 핵심 아이디어

SAM2의 시간축 메모리 어텐션을 **모달리티 축**으로 전용:
1. 각 모달리티(RGB, LiDAR, Thermal)를 별도 "프레임"으로 인코딩
2. SAM2의 memory attention으로 모달리티 간 상호 참조
3. 모달리티별 가중치(UAMM/AMF)로 adaptive fusion

### SAM2 Backbone: Hiera-B+

- `embed_dim=112`, stages=(2,3,16,3) = 24 blocks, `dim_mul=2.0`
- Block별 차원:
  - Blocks 0-2: dim=112 (3개)
  - Blocks 3-5: dim=224 (3개)
  - Blocks 6-20: dim=448 (15개)
  - Block 21: dim=448→896 (전환)
  - Blocks 22-23: dim=896 (2개)
- Pretrained: `semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt`

### Soft-MoE LoRA Layer (공통)

파일: `semseg/models/sam2/sam2/sam_lola_utils.py` (line 521)

```python
class SoftMoE_LoRA_Layer:
    gate: Linear(dim, num_experts)         # routing network
    experts_a: ModuleList[Linear(dim, rank)]   # down-projection (LoRA A)
    experts_b: ModuleList[Linear(rank, dim)]   # up-projection (LoRA B)
```

- **Soft-MoE**: softmax gating → 모든 expert가 참여 (top-k 아님)
- **초기화**: gate.weight N(0, 0.01), gate.bias=0, experts_a=kaiming, experts_b=zeros
- **총 48개 layer**: 24 blocks × 2 (Q, V)
- `rank=4`, `num_experts=3` (모달리티 수와 동일)

### Forward 흐름 (공통)

```
Phase 1: 모달리티별 인코딩
  for modal in [img, lidar, thermal]:
    backbone_feat = SAM2_encoder(modal)        # Hiera-B+ + SoftMoE_LoRA
    memory_attention(backbone_feat, memory)     # cross-modal attention
    memory.append(backbone_feat)

Phase 2: 모달리티 가중치 계산
  cross_weights = Head(all_backbone_feats)      # 방법은 버전별 상이

Phase 3: UAMM (Unified Attention Modulation Memory)
  modulated_feats = backbone_feats * uamm_scores  # feature 조절

Phase 4: Tracking + AMF (Adaptive Modality Fusion)
  outputs = [track(modulated_feat) for feat in modulated_feats]
  final = sum(amf_weights[i] * outputs[i])     # 가중 평균
```

---

## P8: ConfidenceHeadV2 + Sigmoid UAMM

파일: `sam_lora_image_encoder_seg.py` line 1134, 클래스: `LoRA_Sam_P8`

### 아키텍처

```
backbone_feats → ConfidenceHeadV2(fusion_dim) → logits → sigmoid → scores
                                                         ↓
UAMM: scores (0~1, 각 모달리티 독립)
AMF:  normalized_scores = scores / sum(scores)
```

### ConfidenceHeadV2

- GAP(backbone_feat) → Linear → sigmoid
- 각 모달리티에 대해 **독립적**으로 0~1 점수 산출
- 모달리티 간 상대 비교 없음

### 한계점

1. **Sigmoid saturation**: logit > 3 → score ≈ 1.0, logit < -3 → score ≈ 0.0
   - 학습 진행 시 모든 모달리티의 logit이 양수로 → 전부 ~1.0
2. **AMF uniform**: 모든 score ≈ 1.0 → normalized = 1/3씩 uniform 분배
3. **UAMM 무의미**: 모든 feature에 ~1.0 곱함 → modulation 효과 없음

### 실험 결과 요약

| Config | Val mIoU | Test mIoU | M-score |
| --- | --- | --- | --- |
| no-aug (beforeAug) | 93.10 | 35.93 | 64.51 |
| basic-aug | 93.13 | 62.50 | 77.82 |
| hardaug (기본) | 92.96 | 63.93 | 78.45 |
| hardaug2 | 93.29 | 63.45 | 78.37 |
| hardaug3 | 93.36 | 61.57 | 77.46 |

---

## P9: CrossModalFusionHead + Max-Norm UAMM (현재 최선)

파일: `sam_lora_image_encoder_seg.py` line 1355, 클래스: `LoRA_Sam_P9`

### P8에서의 변경 동기

P8의 sigmoid 독립 평가 → 모달리티 간 상대 비교 부재 → uniform AMF
→ **해결**: 모든 모달리티를 동시에 비교하는 cross-modal head

### 아키텍처

```
all_backbone_feats → CrossModalFusionHead → softmax → cross_weights (B, m)
                                                       ↓
UAMM: max_w = max(cross_weights)
       uamm_scores = cross_weights / max_w  → 최선 모달리티=1.0, 나머지 상대적
AMF:  amf_weights = cross_weights (softmax 출력 그대로)
```

### CrossModalFusionHead

```python
class CrossModalFusionHead:
    # GAP → compress → 모든 모달리티 concat → compare → softmax
    gap = AdaptiveAvgPool2d(1)
    compress = Linear(in_channels, in_channels // 4)  # 차원 축소
    compare = Linear(in_channels // 4 * num_modalities, num_modalities)  # 상대 비교
```

- 핵심: **모든 모달리티의 feature를 concat** 후 비교 → 상대적 품질 평가
- softmax 출력 → 합=1 보장, 상대적 가중치

### Max-Norm UAMM

```python
max_w = cross_weights.max(dim=1, keepdim=True)[0]
uamm_scores = cross_weights / (max_w + 1e-8)
# 최선 모달리티 = 1.0 (feature 보존), 나머지 < 1.0 (억제)
```

- P8의 sigmoid와 달리, 최선 모달리티의 feature는 **완전 보존**
- 나쁜 모달리티만 상대적으로 억제

### 한계점 (관찰됨)

1. **Cross-modal weight near-constant**: 특정 이미지에서 thermal≈1.0, lidar≈0.96, img≈0.74 패턴 반복
2. 단순 GAP만 사용 → 텍스처/노이즈 정보 반영 부족
3. 그러나 test generalization은 P8 대비 크게 향상 → 이 방식이 효과적

### 실험 결과

| Config | Val mIoU | Test mIoU | M-score |
| --- | --- | --- | --- |
| hardaug4 | 93.32 | 69.62 | **81.47** |

---

## P10: CrossModalFusionHeadV2 + ModalAuxHead + Oracle KL (취소됨)

파일: `sam_lora_image_encoder_seg.py` line 1859, 클래스: `LoRA_Sam_P10`

### P9에서의 변경 동기

P9의 cross-modal weight가 near-constant → gating이 충분히 adaptive하지 않음
→ **시도**: quality-aware multi-pool + oracle supervision으로 gating 학습 강화

### 아키텍처 변경

```
all_backbone_feats → CrossModalFusionHeadV2 → softmax → cross_weights
                  ↘ ModalAuxHead(각 모달리티) → per-modal segmentation
                     ↓
                  oracle_weights = softmax(per_modal_iou)  # 학습 시 GT와 비교
                  KL(amf_weights || oracle_weights)        # gating 지도학습
```

### CrossModalFusionHeadV2

```python
class CrossModalFusionHeadV2:
    # Multi-pool: GAP + GMP + Channel Std
    gap = AdaptiveAvgPool2d(1)
    gmp = AdaptiveMaxPool2d(1)
    # Std = channel-wise std (텍스처/노이즈 indicator)
    compress_per_modal = ModuleList[Linear(in_ch * 3, in_ch // 4)]  # per-modality
    compare = Linear(in_ch // 4 * num_modalities, num_modalities)
```

- GAP (평균) + GMP (최대값) + Std (변동성) → 품질 정보 풍부
- Per-modality compress → 각 모달리티 독립 특징 추출

### ModalAuxHead

```python
class ModalAuxHead:
    # 각 모달리티별 경량 segmentation head
    conv1x1 → BN → ReLU → conv1x1 → num_classes
```

- 각 모달리티의 backbone feature로 독립 segmentation 수행
- GT와 비교하여 per-modal IoU 계산 → oracle weight 생성
- `LAMBDA_GATE: 0.5`

### 취소 이유

1. **Test 성능 하락**: M-score 79.27 (P9: 81.47, **-2.2**)
2. Test mIoU 65.30 (P9: 69.62, **-4.3**)
3. Oracle supervision이 주간(Val) 데이터에 과적합
4. Multi-pool의 Std feature가 야간에서 부정확한 quality estimation
5. Aux head 추가로 파라미터 증가 → overfitting 가속

### 실험 결과

| Config | Val mIoU | Test mIoU | M-score |
| --- | --- | --- | --- |
| hardaug4 | 93.23 | 65.30 | 79.27 |
| hardaug3 | 93.18 | 58.93 | 76.05 |

---

## P11: P10 + MI Routing Loss (취소됨)

파일: `sam_lora_image_encoder_seg.py` line 2130, 클래스: `LoRA_Sam_P11`

### P10에서의 변경 동기

MoE gate weights가 "uniform"으로 수렴하는 문제 (당시 spatial mean 기준)
→ **시도**: Mutual Information (MI) loss로 expert 분화 강제

### 아키텍처 변경

```
P10 구조 그대로 +
MI loss = H(gate|input) - H(gate_marginal)
LAMBDA_MI: 1.0

UAMM: softmax with temperature (τ=2.0) 로 변경 (max-norm 대신)
```

- Gate distribution을 gradient 유지한 채 수집 (`_grad_gate_collector`)
- Per-modal gate distribution → MI loss 계산
- UAMM: `softmax(logits / τ) * m` (temperature-scaled)

### 취소 이유

1. **Test 성능 더 악화**: M-score 77.09 (P10: 79.27, P9: 81.47)
2. Test mIoU 61.01 → P10보다도 나쁨
3. 지도교수 피드백: "loss를 넣어볼게 아니라 왜 gating이 안되는지 분석이 먼저"
4. **후속 진단에서 핵심 발견**: MoE gate는 이미 정상 작동!
   - "Uniform"은 spatial mean의 CLT artifact
   - Per-token entropy_ratio=0.55, max_weight=0.72
   - MI loss가 불필요하고, 오히려 이미 잘 작동하는 routing을 방해

### 실험 결과

| Config | Val mIoU | Test mIoU | M-score |
| --- | --- | --- | --- |
| hardaug4 | 93.17 | 61.01 | 77.09 |

---

## P12: Input-Conditioned Soft MoE LoRA

파일: `sam_lora_image_encoder_seg.py` line 1585, 클래스: `LoRA_Sam_P12`

### P9에서의 변경 동기

MoE gate 진단 결과 정상이었으나, 모달리티별로 다른 routing 패턴이 필요하다는 가설
→ RGB 채널 통계(mean+std)를 gate에 condition으로 주입

### 아키텍처 변경

```
gate(x) + cond_proj(condition) → softmax → weights
condition = RGB channel mean+std (cond_dim=6), lidar/thermal은 cond=None
cond_proj: Linear(cond_dim, num_experts), zero-init
```

### 실험 결과

- M-score 80.80 (P9: 81.47, **-0.67**)
- Dynamic +4.02pp 개선, Sky -6.81pp 하락
- Expert collapse P9보다 심화 (15% → 20%)
- Test LiDAR routing 48/48 블록 완전 고정

---

## P13: Energy Score Fusion + Expert Collapse Fix

파일: `sam_lora_image_encoder_seg.py` line 2483, 클래스: `LoRA_Sam_P13`

### P9에서의 변경 동기

1. CrossModalFusionHead의 near-constant 출력 문제 (ISSUE-003) → 학습 가능 파라미터 없는 fusion weight
2. SoftMoE_LoRA_Layer의 expert collapse (ISSUE-002) → 비영 초기화로 대칭 깨기

### 아키텍처

```
Phase 2: Aux Prediction + Energy Confidence (P9 Phase 2 대체)
  all_backbone_feats → ConfidenceAuxHead(공유) → aux_logits_list
  aux_logits_list → compute_energy_confidence(T=1.0) → cross_weights (B, m)

나머지 Phase (1, 3, 4)는 P9과 동일
```

### ConfidenceAuxHead

```python
class ConfidenceAuxHead(nn.Module):
    # 공유 1개 (모든 모달리티가 동일 head 사용)
    head = Sequential(
        Conv2d(in_ch, in_ch//4, 1),  # mid_channels = max(in_ch//4, 32)
        BatchNorm2d, ReLU,
        Conv2d(mid_ch, num_classes, 1),
    )
    # 출력: raw logits (B, C, H, W)
```

### compute_energy_confidence

```python
def compute_energy_confidence(aux_logits_list, temperature=1.0):
    for z in aux_logits_list:
        energy = -T * logsumexp(z / T, dim=1)  # (B, H, W)
        conf = -energy.mean(dim=[1, 2])          # (B,) spatial average
    weights = softmax(stack(confs) / T, dim=1)   # (B, m)
    return weights
```

핵심 특징:
- **학습 가능 파라미터 없음** — computed signal이므로 상수 수렴 불가
- **학습/추론 동일 메커니즘** — P10의 oracle-at-train / guess-at-test 불일치 없음
- aux head는 학습됨 (seg loss + λ_aux * aux_CE)

### Expert Collapse Fix

```python
# P13 __init__에서 experts_b 재초기화
for expert_b in moe_q.experts_b:
    nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
    expert_b.weight.data *= 0.01
```

### 실험 결과 및 설계 목표 달성 여부

| 설계 목표 | 판정 | 결과 |
| --- | --- | --- |
| Expert collapse 해결 | **실패** | collapse rate 17.4% (P12: 16.0%와 동일) |
| Energy Score fusion | **부분 성공** | UAMM CV 5-22x 증가, Dynamic +5.55pp |

- M-score 81.21 (P9: 81.47, **-0.26**)
- Val mIoU 92.45 (-0.87), Test mIoU 69.98 (+0.36)
- Night-val checkpoint 선택으로 test 개선 but val 희생

### 한계점 (관찰됨)

1. **Expert collapse 미해결**: kaiming*0.01 init은 resume 학습으로 무력화, 스케일도 미미
2. **Test LiDAR UAMM = 1.0 고정**: aux head가 LiDAR를 항상 "가장 confident"로 판정 (실제 LiDAR 품질은 가장 낮음)
3. **Val mIoU 하락**: Energy Score의 adaptive weight가 P9의 안정적 상수 비율보다 val에서 불리
4. **17 epochs 학습**: P9(47 epochs) 대비 짧지만, P9도 epoch 17(93.57) → 46(94.18)은 +0.61pp만 개선

---

## P14: Per-Modality Separate Aux Decoders

파일: `sam_lora_image_encoder_seg.py` line 2780, 클래스: `LoRA_Sam_P14`

### P13에서의 변경 동기

P13의 ConfidenceAuxHead는 **공유 1개** → 모든 모달리티가 동일 decoder를 공유.
RGB 텍스처, LiDAR 점군, Thermal gradient는 특성이 완전히 다름 → 공유 head로는 각 모달리티에 특화된 예측 불가.
시각화에서 aux mask 품질이 모두 GT와 큰 괴리 확인.

### 아키텍처 변경

```
P13: ConfidenceAuxHead×1 (공유) → 모든 모달리티 동일 head
P14: ModalAuxDecoder×3 (독립) → 모달리티별 전용 head
     · 첫 conv를 3×3으로 변경 → 텍스처/경계 패턴 특화
     · 각 모달리티가 고유 파라미터 → inter-modality gradient interference 제거
```

나머지(Energy Score fusion, UAMM max-norm, AMF, MoE init)는 P13과 동일.

### 상태

- **구현 완료**, 학습 대기 (hardaug5 config 준비됨)
- hardaug5: CRM/ZERO 완전 제거 + test셋 실측 밝기 분포 정렬

---

## P15: Calibrated Spatial Entropy Fusion (설계 단계)

### 변경 동기 — P12~P14 실패 분석에서의 교훈

**1. UAMM/AMF 개념은 유효하다**

| 모델 | Fusion | Val mIoU | 비고 |
| --- | --- | --- | --- |
| Baseline (LoRA_Sam) | 단순 평균 (1/3) | 92.86 | AMF 없음 |
| **P9** | UAMM + AMF (학습된 가중치) | 93.32 | **Baseline 대비 개선** |

Baseline(단순 평균) < P9(UAMM/AMF) → modality fusion 개념 자체의 가치 확인.

**2. Energy Score 방향은 맞지만 정확도가 부족**

P13의 Energy Score fusion은 **낮/밤 적응을 실제로 수행**:

| 모달리티 | P9 Val AMF | P9 Test AMF | P13 Val AMF | P13 Test AMF |
| --- | --- | --- | --- | --- |
| img | 0.275 | 0.275 (**동일**) | 0.404 | **0.289 (↓28%)** |
| lidar | 0.355 | 0.355 (**동일**) | 0.429 | **0.517 (↑20%)** |
| thermal | 0.370 | 0.370 (**동일**) | 0.167 | 0.194 |

P9는 345장 전체에서 소수점 4자리까지 동일한 **학습된 상수** (std ≈ 0.0000).
P13은 밤에 RGB↓ LiDAR↑ 적응 → **방향은 맞지만** LiDAR Sky 맹신으로 실패.

**3. 실패의 직접 원인 3가지**

1. **Energy Score = confidence, not correctness** → "confident but wrong" (ISSUE-008)
2. **Gradient 오염**: `.detach()` 없음 → main loss가 aux head 왜곡
3. **Image-level scalar**: 위치별 모달리티 차이 무시

P15는 이 3가지를 동시에 수정.

### P15 핵심 변경 4가지

#### Fix 1: Gradient 격리 — `.detach()`

```python
# P13/P14 (현재 — gradient 오염)
cross_weights = compute_energy_confidence(aux_logits_list, ...)

# P15 (수정 — gradient 차단)
cross_weights = compute_spatial_entropy_confidence(
    [z.detach() for z in aux_logits_list], ...
)
```

aux head는 **자기 자신의 CE loss만으로** 학습 → 정직한 confidence 출력.
Main loss gradient가 energy→aux→LoRA로 역전파되는 경로 차단.

#### Fix 2: Energy Score → Calibrated Entropy 교체

Energy Score 문제: `E(x) = -T * logsumexp(z/T)` → logit magnitude 기반.
LiDAR가 4클래스 중 하나에 높은 logit → 높은 energy → "confident" → **하지만 틀림** (Sky에서).

Entropy 기반 대안: **예측 분포의 불확실성**을 직접 측정.

```python
# P15: Calibrated Spatial Entropy Confidence
def compute_spatial_entropy_confidence(aux_logits_list, temperature=1.0, num_classes=4):
    """
    Energy Score 대신 calibrated entropy로 per-pixel confidence 계산.

    핵심 차이:
    - Energy: logit magnitude → "자신있게 틀리면" 높은 점수 (dangerous)
    - Entropy: 확률 분포 균등도 → 4클래스에 골고루 분산 = 낮은 confidence (safe)

    LiDAR가 Sky에서 Water로 확신있게 오예측 → Energy 높음 (나쁨)
    LiDAR가 Sky에서 불확실 → Entropy 높음 → confidence 낮음 (좋음)
    """
    conf_maps = []
    for z in aux_logits_list:  # z: (B, C, H, W), C=num_classes
        # Temperature scaling for calibration
        probs = F.softmax(z / temperature, dim=1)               # (B, C, H, W)
        log_probs = F.log_softmax(z / temperature, dim=1)       # (B, C, H, W)
        entropy = -(probs * log_probs).sum(dim=1)               # (B, H, W)
        # Normalize: 0 (완전 확신) ~ 1 (완전 균등)
        max_entropy = math.log(num_classes)
        confidence = 1.0 - entropy / max_entropy                # (B, H, W)
        conf_maps.append(confidence)

    stacked = torch.stack(conf_maps, dim=1)                     # (B, m, H, W)
    weights = F.softmax(stacked / temperature, dim=1)           # (B, m, H, W)
    return weights
```

Entropy의 장점:
- **"자신있게 틀리는" 케이스 감지**: LiDAR가 Sky에서 단일 클래스(Water)에 높은 확률을 주면 aux head가 정확해야만 높은 confidence → aux head가 부정확하면 자연스럽게 분산된 예측 → 높은 entropy → 낮은 confidence
- **Calibration 가능**: temperature T를 val에서 최적화하여 confidence를 보정

#### Fix 3: Spatial-wise (공간별 가중치)

기존 `(B, m)` 스칼라 → `(B, m, H, W)` spatial map:

```python
# UAMM: vision_feats 각 level에 spatial weight 적용
spatial_score = uamm_scores[:, frame_idx]                 # (B, H, W)
for level, feat in enumerate(vision_feats):
    h, w = feat_sizes[level]
    score_resized = F.interpolate(
        spatial_score.unsqueeze(1), size=(h, w), mode='bilinear'
    )  # (B, 1, h, w)
    score_flat = score_resized.flatten(2).permute(2, 0, 1)  # (h*w, B, 1)
    modulated_feat = feat * score_flat

# AMF: output fusion에 spatial weight 적용
w_i = F.interpolate(
    amf_weights[:, i:i+1], size=output[0].shape[2:], mode='bilinear'
)  # (B, 1, H_out, W_out)
m_output += output[i] * w_i
```

#### Fix 4: Aux Warmup Schedule

Aux head가 충분히 학습된 후에 UAMM/AMF 활성화:

```python
# Config
TRAIN:
  AUX_WARMUP_EPOCHS: 10    # 초기 N epoch는 aux CE만 학습
  LAMBDA_AUX: 0.3

# Forward에서
if current_epoch < aux_warmup_epochs:
    # Uniform weights (P9의 near-constant와 유사)
    cross_weights = torch.ones(B, m, H, W) / m
else:
    # Calibrated entropy weights
    cross_weights = compute_spatial_entropy_confidence(
        [z.detach() for z in aux_logits_list], ...
    )
```

첫 N epoch 동안:
- Aux head: CE loss로 학습 → 기본적인 segmentation 능력 확보
- UAMM/AMF: uniform(1/m) → P9처럼 안정적 학습
- Main decoder: 정상 학습

N epoch 이후:
- Aux head의 entropy가 UAMM/AMF에 반영 시작
- 점진적 전환 (abrupt하지 않도록 linear ramp 고려)

### 전체 Forward 흐름 (P15)

```
Phase 1: 모달리티별 인코딩 (P14 동일)
  for modal in [img, lidar, thermal]:
    backbone_feat = SAM2_encoder(modal)  # Hiera-B+ + SoftMoE_LoRA

Phase 2: Spatial Entropy Confidence
  aux_logits[i] = aux_heads[i](backbone_feat[i])        # 독립 aux decoder × 3
  conf_maps = entropy_confidence([z.detach() for z])     # (B, m, H, W)

Phase 3: Spatial UAMM + Tracking
  for each modality:
    spatial_uamm = conf_maps[:, i, :, :]                 # (B, H, W)
    modulated_vision_feats = vision_feats * spatial_uamm  # level별 interpolate
    output[i] = track_step(modulated_vision_feats, memory)

Phase 4: Spatial AMF
  amf_weights = conf_maps                                # (B, m, H, W)
  final = sum(output[i] * interpolate(amf_weights[:, i]))
```

### P15 vs 이전 버전 차이 요약

| 구분 | P13 | P14 | **P15** |
| --- | --- | --- | --- |
| Confidence 방식 | Energy Score (logit) | Energy Score (logit) | **Calibrated Entropy** |
| Gradient 격리 | 없음 (오염) | 없음 (오염) | **`.detach()` 적용** |
| Weight 형태 | `(B, m)` 스칼라 | `(B, m)` 스칼라 | **`(B, m, H, W)` spatial** |
| Aux Decoder | 공유 1개 | 독립 3개 | 독립 3개 (P14 유지) |
| Warmup | 없음 | 없음 | **AUX_WARMUP_EPOCHS** |
| UAMM | max-norm 스칼라 | max-norm 스칼라 | **spatial max-norm** |
| AMF | energy softmax 스칼라 | energy softmax 스칼라 | **spatial entropy softmax** |

### 구현 시 주의사항

1. **해상도 정합**: aux head 출력 `(H_feat, W_feat)`와 vision_feats/output의 해상도가 다름 → `F.interpolate` 필수
2. **vision_feats 형상**: SAM2 Hiera는 `(num_tokens, B, C)` 형태의 flattened feature 사용 → reshape/flatten 처리 필요
3. **feat_sizes**: `_prepare_backbone_features()`에서 반환하는 각 level의 (h, w) 사용
4. **backward compatibility**: train 시 `(output, m_feat, aux_logits_list)` 반환 형식 유지
5. **Temperature 최적화**: `temperature` 파라미터를 config에 노출 (기본 1.0, val에서 grid search 가능)
6. **Warmup→Active 전환**: abrupt 전환은 학습 불안정 유발 가능 → linear ramp (N~N+5 epoch) 고려

---

## P16: Calibrated Spatial Entropy Fusion (P15 설계의 구현 버전)

파일: `sam_lora_image_encoder_seg.py` 끝부분, 클래스: `LoRA_Sam_P16`

### P15에서의 변경 동기

P15는 기존 Energy Score (spatial)를 사용하여 Levine에서 학습 진행 중.
P15 설계 문서에서 제시한 4가지 수정사항을 별도 버전으로 구현하여 P15와 비교 실험.
P15 코드를 직접 수정하지 않고 **새 버전 P16으로 분리** (P15 학습 결과 보존).

### 핵심 변경 4가지 (P12~P14 실패 분석에서 도출)

#### 1. `.detach()` Gradient 격리 (ISSUE-008)

```python
# P13/P14/P15: gradient 오염
cross_weights = compute_spatial_energy_confidence(aux_logits_list, ...)

# P16: gradient 차단
cross_weights = compute_spatial_entropy_confidence(
    [z.detach() for z in aux_logits_list], ...
)
```

Aux head는 자기 CE loss만으로 학습 → 정직한 confidence 출력.

#### 2. Energy Score → Calibrated Entropy (ISSUE-009)

```python
def compute_spatial_entropy_confidence(aux_logits_list, temperature=1.0, num_classes=4):
    conf_maps = []
    max_entropy = math.log(num_classes)
    for z in aux_logits_list:
        probs = F.softmax(z / temperature, dim=1)
        log_probs = F.log_softmax(z / temperature, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)          # (B, H, W)
        confidence = 1.0 - entropy / max_entropy            # (B, H, W)
        conf_maps.append(confidence)
    stacked = torch.stack(conf_maps, dim=1)                 # (B, m, H, W)
    weights = F.softmax(stacked / temperature, dim=1)       # (B, m, H, W)
    return weights
```

Energy는 logit magnitude 기반 → "자신있게 틀리면" 높은 점수. Entropy는 분포 균등도 → 불확실하면 낮은 confidence.

#### 3. Spatial-wise `(B, m, H, W)` 가중치 (P15에서 유지)

UAMM/AMF 모두 pixel-level 가중치 사용. `F.interpolate`로 vision_feats/output 해상도에 맞춤.

#### 4. Aux Warmup Schedule (신규)

```python
# 3단계: uniform → linear ramp → full entropy
warmup_epochs = 10  # config: TRAIN.AUX_WARMUP_EPOCHS
if epoch < warmup_epochs:
    cross_weights = uniform(1/m)                  # P9처럼 안정적
elif epoch < warmup_epochs + 5:
    ramp = (epoch - warmup_epochs) / 5.0          # 0→1 linear
    cross_weights = (1-ramp)*uniform + ramp*entropy
else:
    cross_weights = entropy                       # full adaptive
```

Aux head가 충분히 학습된 후에 UAMM/AMF 활성화. `_current_epoch` 속성을 train script에서 매 epoch 설정.

### P15 vs P16 차이

| 구분 | P15 (Levine 학습 중) | P16 |
| --- | --- | --- |
| Confidence 함수 | `compute_spatial_energy_confidence` | **`compute_spatial_entropy_confidence`** |
| Gradient 격리 | 없음 | **`.detach()` 적용** |
| Warmup | 없음 | **10ep uniform + 5ep ramp** |
| Weight 형태 | `(B, m, H, W)` spatial | `(B, m, H, W)` spatial (동일) |
| Aux Decoder | ModalAuxDecoder×3 (독립) | ModalAuxDecoder×3 (동일) |

### 구현 상태

- **구현 완료** (2026-02-27)
- Config: `configs/levine-multiaqua_rgbtl_P16_hardaug5.yaml`
- Eval config: `configs/eval_config/levine-multiaqua_rgbtl_P16_hardaug5.yaml`
- 학습 스크립트: `train_sam2_lora_paper.py` (warmup epoch 전달 + `_current_epoch` 설정)
- 로깅: TensorBoard + trackio (전면 교체)

### 추가 개선사항 (P16과 함께 구현)

1. **5-epoch 주기 체크포인트 저장**: `periodic_epoch{N}_checkpoint.pth`
2. **trackio 로깅**: TensorBoard 대체, 전체 메트릭 로깅 (per-class IoU/acc/f1, warmup_ramp 등)
3. **tqdm 개선**: 0값 loss 숨김, warmup 상태 표시

---

## P17: Multi-Scale FPN Aux Decoder + Calibrated Spatial Entropy Fusion

파일: `sam_lora_image_encoder_seg.py` 끝부분, 클래스: `LoRA_Sam_P17`

### P16에서의 변경 동기

P13~P16의 aux decoder는 **`backbone_fpn[0]`(32ch, 256×256) 하나만** 사용.
SAM2 Hiera B+는 3개 FPN 레벨을 계산하지만 나머지 2개는 aux decoder에서 전혀 미활용:
- `backbone_fpn[1]`: 64ch, 128×128
- `backbone_fpn[2]`: 256ch, 64×64

이것이 ISSUE-008(frozen backbone bottleneck)의 실질적 원인:
32채널 단일 스케일 → 352채널(32+64+256) 멀티스케일 = **11배 정보량 증가, 추가 backbone 연산 0**

### 핵심 변경: MultiScaleModalAuxDecoder

```python
class MultiScaleModalAuxDecoder(nn.Module):
    """3개 FPN 레벨을 모두 활용하는 aux segmentation decoder."""

    def __init__(self, fpn_channels=(32, 64, 256), proj_dim=32, num_classes=4):
        # 각 FPN 레벨을 proj_dim(32)으로 project (1×1 conv + BN + ReLU)
        self.proj_layers = nn.ModuleList([
            nn.Sequential(Conv2d(ch, 32, 1), BN, ReLU) for ch in fpn_channels
        ])
        # Concat(32×3=96) → 3×3 conv(96→48) → 1×1 conv(48→4)
        self.decoder = nn.Sequential(
            Conv2d(96, 48, 3, padding=1), BN, ReLU,
            Conv2d(48, num_classes, 1),
        )

    def forward(self, fpn_feats):  # [fpn0, fpn1, fpn2]
        # 모든 레벨을 fpn[0] 해상도로 upsample → project → concat → decode
        target_size = fpn_feats[0].shape[2:]
        projected = [proj(feat) → interpolate if needed for each level]
        return self.decoder(torch.cat(projected, dim=1))
```

**파라미터 수** (~53K per modality, ×3 = ~159K total):
- proj_layers: 32×32 + 64×32 + 256×32 = ~11.3K
- decoder: 96×48×3×3 + 48×4 = ~41.7K

기존 ModalAuxDecoder: ~290 params per modality → **정보량 11배 증가 대비 합리적 파라미터 증가**

### P16과의 차이

| 구분 | P16 | **P17** |
| --- | --- | --- |
| Aux Decoder | ModalAuxDecoder (fpn[0] only, 32ch) | **MultiScaleModalAuxDecoder (fpn[0,1,2], 352ch)** |
| Aux 입력 | `backbone_fpn[0]` (32ch, 256×256) | `backbone_fpn[0,1,2]` (32+64+256ch, multi-scale) |
| Aux 파라미터 | ~290/modality | **~53K/modality** |
| Confidence | Calibrated Entropy (동일) | Calibrated Entropy (동일) |
| Gradient 격리 | `.detach()` (동일) | `.detach()` (동일) |
| Warmup | 10ep+5ep ramp (동일) | 10ep+5ep ramp (동일) |
| Spatial UAMM/AMF | (B, m, H, W) (동일) | (B, m, H, W) (동일) |

### Forward Phase 2 변경

```python
# P16:
all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
aux_logits_list = [self.aux_heads[i](feat) for i, feat in enumerate(all_backbone_feats)]

# P17:
all_fpn_feats = [
    [image_embedding[i]['backbone_fpn'][j] for j in range(3)]
    for i in range(m)
]  # all_fpn_feats[modality][level]
aux_logits_list = [self.aux_heads[i](all_fpn_feats[i]) for i in range(m)]
all_backbone_feats = [all_fpn_feats[i][0] for i in range(m)]  # m_feat용
```

Phase 3 (Spatial UAMM + Tracking), Phase 4 (AMF Fusion)는 P16과 동일.

### 구현 상태

- **구현 완료** (2026-02-27)
- Config: `configs/bengio-multiaqua_rgbtl_P17_hardaug5.yaml`
- Eval config: `configs/eval_config/bengio-multiaqua_rgbtl_P17_hardaug5.yaml`
- 학습 스크립트 변경 불필요 (기존 inspect 기반 분기 + 3-tuple 리턴 호환)

---

## P18: Trainable ResNet-18 Aux Backbone + Configurable Fusion

파일: `sam_lora_image_encoder_seg.py` 끝부분, 클래스: `LoRA_Sam_P18`

### P17에서의 변경 동기

P13~P17 모두 frozen SAM2 Hiera B+ FPN feature로 aux decoder를 학습 (ISSUE-008).
P17이 3개 FPN 레벨을 사용해도 feature 자체가 MULTIAQUA 도메인에 특화되지 않아 aux mask 품질 한계.

해결: **ImageNet pretrained ResNet-18**을 trainable aux backbone으로 추가.
aux CE loss로 MULTIAQUA 4-class에 직접 fine-tune → 도메인 특화 feature 학습.

### 핵심 변경: ResNet-18 Aux Pipeline

```
Input (3ch) → ResNetAuxBackbone → layer2(128ch, H/8) + layer3(256ch, H/16)
                                    ↓
              ResNetAuxDecoder → aux_logits (B, 4, 128, 128)
                                    ↓
                    aux CE loss (trains ResNet) + optional entropy fusion
```

**ResNetAuxBackbone** (~11.2M):
- 3개 per-modality stems (Conv7×7+BN+ReLU, pretrained conv1 복제 초기화)
- 1개 shared body (maxpool + layer1 + layer2 + layer3)
- layer4 미사용 (해상도 32×32로 너무 낮음)

**ResNetAuxDecoder** (~53K per modality):
- layer2(128ch)+layer3(256ch) → proj(32ch×2) → concat(64ch) → 3×3 conv → 4ch logits

### Two Sub-Variants: `use_entropy_fusion` 플래그

| | P18-A (False) | P18-B (True) |
|---|---|---|
| Fusion | P9-style CrossModalFusionHead (scalar) | P17-style spatial entropy |
| UAMM | scalar max-norm `(B, m)` | spatial max-norm `(B, m, H, W)` |
| AMF | scalar softmax `(B, m)` | spatial entropy softmax `(B, m, H, W)` |
| ResNet역할 | aux CE loss로만 학습 (fusion 미영향) | aux logits → entropy → fusion 구동 |
| Warmup | 불필요 (entropy 미사용) | 10ep+5ep ramp |

### P17 vs P18 차이

| 구분 | P17 | **P18-A** | **P18-B** |
|---|---|---|---|
| Aux feature source | SAM2 FPN (frozen) | **ResNet-18 (trainable)** | **ResNet-18 (trainable)** |
| Aux decoder input | fpn[0,1,2] (352ch) | ResNet l2+l3 (384ch) | ResNet l2+l3 (384ch) |
| Aux decoder | MultiScaleModalAuxDecoder | **ResNetAuxDecoder** | **ResNetAuxDecoder** |
| Fusion | spatial entropy | **P9 scalar** | spatial entropy |
| Trainable aux params | ~159K | **~11.4M** | **~11.4M** |
| Total trainable | ~8.7M | **~20M** | **~20M** |

### 구현 상태

- **구현 완료** (2026-03-01)
- Config: `configs/levine-multiaqua_rgbtl_P18_hardaug5.yaml`
- Eval config: `configs/eval_config/levine-multiaqua_rgbtl_P18_hardaug5.yaml`
- 학습 스크립트: `use_entropy_fusion` inspect dispatch 추가

---

## P19: Learned Spatial Cross-Modal Fusion (SpatialCrossModalFusionHead)

파일: `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` (LoRA_Sam_P19)
융합 헤드: `semseg/models/sam2/sam2/sam_lola_utils.py` (SpatialCrossModalFusionHead)

### 핵심 아이디어

P9 CrossModalFusionHead는 GAP로 공간정보 소실 → 스칼라 (B,m) 가중치.
P10은 GAP+GMP+Std 시도했으나 같은 fpn[0]에서 pooling 변형만으로 실패 (M -2.2).
P17은 aux entropy로 spatial (B,m,H,W)를 만들지만, aux mask 품질 의존 (ISSUE-008).

**P19: 학습 가능한 SpatialCrossModalFusionHead로 backbone feature에서 직접 spatial 가중치 학습.**

### 아키텍처

```
Phase A: Multi-Scale FPN Projection (shared across modalities)
  fpn[0] (32ch, 256²) → Conv1×1(32→32) → BN → ReLU ────────────→ (B, 32, 256, 256)
  fpn[1] (64ch, 128²) → Conv1×1(64→32) → BN → ReLU → ×2 upsample→ (B, 32, 256, 256)
  fpn[2] (256ch, 64²) → Conv1×1(256→32) → BN → ReLU → ×4 upsample→ (B, 32, 256, 256)
                                                            concat → (B, 96, 256, 256)

Phase B: Per-Modality Spatial Context (shared across modalities)
  DWConv 3×3(96, groups=96) → BN → ReLU → Conv1×1(96→32) → BN → ReLU
  → (B, 32, 256, 256)  -- local context: LiDAR density, Thermal padding, RGB illumination

Phase C: Cross-Modal Spatial Comparison
  concat m modalities → (B, 96, 256, 256)
  → Conv1×1(96→64) → BN → ReLU
  → DWConv 3×3(64, groups=64) → BN → ReLU  -- spatial coherence
  → Conv1×1(64→3) [zero-init]
  → softmax(dim=1) → (B, 3, 256, 256)
```

### P9 vs P19 비교

| | P9 | P19 |
| --- | --- | --- |
| Fusion Head | CrossModalFusionHead (GAP) | SpatialCrossModalFusionHead (DWConv) |
| FPN Input | fpn[0] only (32ch) | fpn[0]+[1]+[2] (32+64+256ch) |
| Weight Shape | (B, m) scalar | (B, m, H, W) spatial |
| UAMM | scalar broadcast | per-level F.interpolate (P17 패턴) |
| AMF | `.view(-1,1,1,1)` | `_resize_weight()` (P17 패턴) |
| Aux Decoder | 없음 | 없음 |
| Return | 2-tuple | 2-tuple |
| Fusion Head Params | ~15K | ~23K |
| Total Trainable | ~8.5M | ~8.5M |

### 구현 상태

- **구현 완료** (2026-03-01)
- Config: `configs/levine-multiaqua_rgbtl_P19_hardaug5.yaml`
- Eval config: `configs/eval_config/levine-multiaqua_rgbtl_P19_hardaug5.yaml`
- 학습 스크립트 변경 없음 (P9과 동일 시그니처)

---

## P20: Shared MLP Gate + Higher Rank MoE (실험 J-A)

파일: `sam_lora_image_encoder_seg.py` 끝부분, 클래스: `LoRA_Sam_P20`
Gate/MoE: `sam_lola_utils.py` — `SharedGateMLP`, `SoftMoE_LoRA_Layer_V2`

### P9에서의 변경 동기

P9의 MoE gate `Linear(C→3)`는 단일 선형 레이어로 비선형 결정경계 학습 불가.
Per-token entropy_ratio=0.55로 분화되어 있지만, 모달리티별 의미 있는 routing 차이는 부족.
Expert rank=4도 매우 낮아 expert 간 specialization 여지 부족.

### 핵심 변경 3가지

#### 1. SharedGateMLP (2-layer MLP Gate)

```python
class SharedGateMLP(nn.Module):
    """Linear(C → C//4) → ReLU → Linear(C//4 → num_experts)"""
    def __init__(self, in_features, num_experts, hidden_ratio=4):
        hidden = max(in_features // hidden_ratio, 16)
        self.net = Sequential(
            Linear(in_features, hidden),
            ReLU(inplace=True),
            Linear(hidden, num_experts),
        )
    # init: kaiming + zeros(bias) + normal(0.01, last layer weight)
```

- 비선형 결정경계 학습 가능 → 모달리티/공간/컨텐츠 기반 routing
- `hidden_ratio=4` → C//4 hidden dim

#### 2. Gate 공유 전략

동일 `in_features` 차원의 블록들이 하나의 MLP gate 공유:

| Stage | Blocks | dim | hidden | Q/V layers | 공유 MLP |
| --- | --- | --- | --- | --- | --- |
| 0 | 0-1 | 112 | 28 | 4 | 1개 |
| 1 | 2-4 | 224 | 56 | 6 | 1개 |
| 2 | 5-20 | 448 | 112 | 32 | 1개 |
| 3 | 21-23 | 896 | 224 | 6 | 1개 |
| **합계** | | | | **48** | **4개** |

- 독립 gate 48개(~2.8M) → 공유 gate 4개(~268K) — **과적합 방지**
- `LoRA_Sam_P20.shared_gates` (nn.ModuleDict, key=str(dim))

#### 3. Rank 상향: 4 → 8

- Expert capacity 2배 증가 → expert 간 실질적 차이 발생 가능
- Gate 분화에 대한 gradient 신호 강화

### SoftMoE_LoRA_Layer_V2

```python
class SoftMoE_LoRA_Layer_V2(nn.Module):
    """외부 공유 gate 참조, 자체 gate 없음"""
    def __init__(self, in_features, rank, num_experts=4):
        self.experts_a = ModuleList[Linear(in_features, rank, bias=False)]
        self.experts_b = ModuleList[Linear(rank, in_features, bias=False)]
        self._shared_gate = None  # Python attribute, not nn.Module

    def set_shared_gate(self, gate_module):
        self._shared_gate = gate_module

    def forward(self, x):
        gate_logits = self._shared_gate(x)
        gate_weights = softmax(gate_logits, dim=-1)
        # weighted sum of experts
```

- `_shared_gate`는 Python attribute → state_dict에 포함 안 됨
- Gate는 `LoRA_Sam_P20.shared_gates`에서 소유/저장

### P9 vs P20 비교

| | P9 | P20 |
| --- | --- | --- |
| Gate | `Linear(C→3)` × 48 | `SharedGateMLP(C→C//4→3)` × 4 |
| Gate 파라미터 | ~268K (48 independent) | ~268K (4 shared MLP) |
| MoE Layer | SoftMoE_LoRA_Layer | **SoftMoE_LoRA_Layer_V2** |
| Rank | 4 | **8** |
| Expert 파라미터 | ~700K | **~1.4M** |
| Fusion Head | CrossModalFusionHead | CrossModalFusionHead (동일) |
| UAMM | max-norm scalar | max-norm scalar (동일) |
| AMF | softmax scalar | softmax scalar (동일) |
| Forward | 2-tuple (output, feat) | 2-tuple (동일) |

### Save/Load 전략

```python
# save_lora_parameters:
merged_dict = {
    **moe_params,          # moe_q_{i:03d}, moe_v_{i:03d} (experts only, no gate)
    **shared_gate_params,  # shared_gate.{dim}.net.{0,2}.{weight,bias}
    **cross_modal_tensors,
    **prompt_encoder_tensors,
    **mask_decoder_tensors,
}
```

- MoE expert state_dict에는 gate 없음 (V2는 자체 gate 미소유)
- Shared gates는 별도 prefix `shared_gate.`로 저장/로드

### 구현 상태

- **구현 완료** (2026-03-05)
- Config: `configs/levine-multiaqua_rgbtl_P20_hardaug8_physaug.yaml`
- Eval config: `configs/eval_config/levine-multiaqua_rgbtl_P20_hardaug8_physaug.yaml`
- Train script: `gate_hidden_ratio` inspect dispatch 추가
- Augmentation: hardaug8_physaug (CRM 0.20 + PhysAug + shot noise)

---

## P21: DeBA-FP (Deformable Bottleneck Adapter for Feature Pyramid) (실험 K)

파일: `sam_lora_image_encoder_seg.py` 끝부분, 클래스: `LoRA_Sam_P21`
DeBA-FP: `sam_lola_utils.py` — `DeBAFP`

### 동기

P9의 FPN feature(fpn[0])는 GAP → CrossModalFusionHead에 직접 입력. Spatial refinement 없이
global average만으로 모달리티 중요도를 산출. Day→Night domain gap에서 경계/형태 같은
구조적 정보는 domain-invariant인데, 이를 명시적으로 포착하는 메커니즘 부재.

DeBA (CVPR 2026)는 deformable convolution으로 domain-invariant structural information을 포착.
특히 LaRS(수면 환경) 벤치마크에서 SOTA → MULTIAQUA와 직접 관련.

### P9 대비 변경

P9 구조 완전 유지 + DeBA-FP 모듈만 fpn[0]과 CrossModalFusionHead 사이에 삽입.

```
P9:  fpn[0] ──────────────────→ CrossModalFusionHead → UAMM/AMF
P21: fpn[0] → DeBA-FP(shared) → CrossModalFusionHead → UAMM/AMF
```

### DeBA-FP 구조

```python
class DeBAFP(nn.Module):
    """
    feat' = feat + α_m * W_u(GELU(LN(DCM(W_d(feat)))))

    Shared across modalities: W_d, DCM, LN, W_u
    Per-modality: α (init=0 → identity at start)
    """
    # W_d: Conv2d(256→64, 1×1) — bottleneck down projection
    # offset_mask_conv: Conv2d(64→27, 3×3) — DCNv2 offset+mask prediction
    # dcm_weight: Parameter(64, 64, 3, 3) — deformable conv weight
    # norm: LayerNorm(64) — shared θ_norm
    # W_u: Conv2d(64→256, 1×1) — up projection
    # alpha: ParameterList([zeros(1)] × num_modalities)
```

**핵심 설계 결정**:

1. **Cross-modal weight sharing**: 모든 learnable 레이어(W_d, DCM, LN, W_u)를 3개 모달리티가 공유
   - 2,952 학습 샘플로 최대한 regularization
   - α만 per-modality → 각 모달리티가 다른 강도로 adaptation 가능
2. **α=0 init**: 학습 시작 시 DeBA-FP = identity → P9과 동일한 출발점
3. **Offset zero-init**: DCM offset이 0부터 시작 → regular conv로 시작, 점진적으로 deformable
4. **fpn[0] only**: P9가 fpn[0]만 사용하므로 다른 FPN 레벨은 불필요

### 원본 DeBA와의 차이

| 항목 | 원본 DeBA | P21 |
| --- | --- | --- |
| Backbone | DINOv2 ViT | SAM2 Hiera B+ |
| DeBA-BB | ViT 블록 사이 삽입 | **미적용** |
| DeBA-FP | FPN 4-level | **fpn[0] only** |
| Cross-layer sharing | 레이어 간 DCM/norm 공유 | **모달리티 간** 공유 |
| d_b | 64 | 64 (동일) |
| Norm | LayerNorm | LayerNorm (동일) |
| DCN version | DCNv4 | **DCNv2** (torchvision) |

### 파라미터 추가량

| 구성 | 파라미터 |
| --- | --- |
| W_d: Conv2d(256→64, 1×1) | 16,448 |
| offset_mask_conv: Conv2d(64→27, 3×3) | 15,579 |
| dcm_weight: (64, 64, 3, 3) | 36,864 |
| LayerNorm(64) | 128 |
| W_u: Conv2d(64→256, 1×1) | 16,640 |
| α × 3 | 3 |
| **합계** | **~85K** |

P9 LoRA ~700K 대비 12% 증가. 전체 trainable ~785K.

### Save/Load

```python
# save_lora_parameters:
merged_dict = {
    **moe_params,          # P9 동일
    **cross_modal_tensors, # P9 동일
    **deba_fp_tensors,     # prefix "deba_fp." (신규)
    **prompt_encoder_tensors,
    **mask_decoder_tensors,
}
```

### DeBA-BB 향후 과제

SAM2 Hiera의 블록 구조(MultiScaleBlock with dim changes)가 DINOv2 ViT(일정 dim)와 다르므로
DeBA-BB를 직접 삽입하려면 Hiera-specific adapter 설계가 필요. 현재는 DeBA-FP만 적용.
DeBA-FP만으로 충분한 효과가 있으면 BB 추가 불필요, 불충분하면 BB 설계 진행.

### 구현 상태

- **구현 완료** (2026-03-09)
- Config: `configs/levine-multiaqua_rgbtl_P21_hardaug8_physaug.yaml`
- Eval config: `configs/eval_config/levine-multiaqua_rgbtl_P21_hardaug8_physaug.yaml`

---

## P22: Multi-Scale DeBA-FP (all FPN levels, Phase 1) (실험 L)

파일: `sam_lora_image_encoder_seg.py` 끝부분, 클래스: `LoRA_Sam_P22`
DeBA-FP MultiScale: `sam_lola_utils.py` — `DeBAFP_MultiScale`

### 동기

P21은 fpn[0]만 Phase 2에서 DeBA-FP 적용. vision_feats(tracking/memory attention에 사용)는
raw FPN에서 생성되어 DeBA-FP 효과가 도달하지 않음. P22는 Phase 1에서 fpn[0,1,2] 전부 적용하여
refined features가 전체 파이프라인으로 전파.

### P21 대비 변경

```
P21: encode → fpn[0] → DeBA-FP → CrossModalFusionHead (vision_feats는 raw)
P22: encode → fpn[0,1,2] → DeBA-FP_MS → _prepare_backbone_features → vision_feats (refined)
                                       → CrossModalFusionHead (refined)
```

| 항목 | P21 | P22 |
| --- | --- | --- |
| 적용 범위 | fpn[0] only | fpn[0,1,2] all |
| 적용 위치 | Phase 2 | Phase 1 |
| 영향 범위 | fusion weights only | 전체 pipeline |
| FPN 채널 | [32] | [32, 64, 256] |
| Cross-layer sharing | 모달리티 간 | 모달리티 간 + FPN 레벨 간 |
| 추가 파라미터 | ~56K | ~98K |

### DeBAFP_MultiScale 구조

```python
class DeBAFP_MultiScale(nn.Module):
    """
    feat'_l = feat_l + α_m * W_u_l(GELU(LN(DCM(W_d_l(feat_l)))))

    Shared across levels + modalities: DCM (offset+deform conv), LayerNorm
    Per-level: W_d_l, W_u_l (different in_channels)
    Per-modality: α_m (shared across levels, init=0)
    """
    # W_d_list: [Conv2d(32→64), Conv2d(64→64), Conv2d(256→64)]
    # W_u_list: [Conv2d(64→32), Conv2d(64→64), Conv2d(64→256)]
    # offset_mask_conv: Conv2d(64→27, 3×3) — shared
    # dcm_weight: (64, 64, 3, 3) — shared
    # norm: LayerNorm(64) — shared
    # alpha: ParameterList([zeros(1)] × 3) — per-modality
```

### 파라미터 추가량

| 구성 | 파라미터 |
| --- | --- |
| W_d_list: 3 × Conv2d(C_l→64, 1×1) | 22,720 |
| W_u_list: 3 × Conv2d(64→C_l, 1×1) | 22,688 |
| offset_mask_conv: Conv2d(64→27, 3×3) | 15,579 |
| dcm_weight: (64, 64, 3, 3) | 36,864 |
| LayerNorm(64) | 128 |
| α × 3 | 3 |
| **합계** | **~98K** |

P9 대비 14% 증가. P21 대비 +42K (per-level W_d/W_u 추가분).

### Forward Flow

```python
# Phase 1: encode + DeBA-FP MultiScale
for i in range(m):  # 각 모달리티
    img_emb = self.sam.forward_image(batched_input[i])
    # backbone_fpn channels: [32, 64, 256] (after conv_s0, conv_s1)
    for level in range(len(img_emb['backbone_fpn'])):
        img_emb['backbone_fpn'][level] = self.deba_fp_ms(
            img_emb['backbone_fpn'][level], modality_idx=i, level_idx=level)
    # _prepare_backbone_features → vision_feats (now refined!)
    bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)

# Phase 2: CrossModalFusionHead (fpn[0] already refined from Phase 1)
all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
cross_weights, cross_logits = self.cross_modal_head(all_backbone_feats)
# → UAMM/AMF는 P9 동일
```

### Save/Load

```python
merged_dict = {
    **moe_params,             # P9 동일
    **cross_modal_tensors,    # P9 동일
    **deba_fp_ms_tensors,     # prefix "deba_fp_ms." (P22 신규)
    **prompt_encoder_tensors,
    **mask_decoder_tensors,
}
```

### 구현 상태

- **구현 완료** (2026-03-09)
- Config: `configs/levine-multiaqua_rgbtl_P22_hardaug8_physaug.yaml`
- Eval config: `configs/eval_config/levine-multiaqua_rgbtl_P22_hardaug8_physaug.yaml`
- Train script: `deba_bottleneck_dim` inspect dispatch 추가
- Augmentation: hardaug8_physaug (P9 h8과 동일)

---

## P23: MoE DeBA-BB (구현 완료, 학습 대기) (실험 M)

### 동기 및 핵심 아이디어

현재 P9의 MoE LoRA는 linear adapter (`Linear_down → Linear_up`). ConvLoRA(ICLR 2024)와 DeBA(CVPR 2026)를 결합하여 **deformable conv bottleneck adapter를 MoE expert로** 사용하는 구조.

- **ConvLoRA** (Zhong et al., ICLR 2024): LoRA bottleneck에 conv 삽입 → ViT에 local inductive bias 주입
- **DeBA-BB** (Anonymous, CVPR 2026): backbone layer 사이에 deformable conv bottleneck adapter 삽입
- **P23 제안**: DeBA-BB 구조를 MoE expert로, GAP gating으로 per-image routing

### 참고 논문

- ConvLoRA: "Convolution Meets LoRA: Parameter Efficient Finetuning for SAM" (ICLR 2024) — https://arxiv.org/abs/2401.17868
- DeBA: "Rethinking Deformable Convolution as an Adapter with Cross-layer Weight Sharing" (CVPR 2026)

### 설계 결정 기록 (2026-03-10 논의)

#### 1. LoRA → DeBA-BB 교체 근거

DeBA-BB는 정확히 LoRA와 같은 위치(backbone layer 사이)에 적용되는 adapter. Drop-in replacement 가능:
```
P9 MoE LoRA:  x → gate → Σ w_i × [Linear_a_i(C→r) → Linear_b_i(r→C)]
P23 MoE DeBA: x → gate → Σ w_i × [W_d(C→d_b) → reshape(H,W) → DCM_i(3×3) → LN → GELU → reshape(HW) → W_u(d_b→C)]
```

Hiera의 token은 2D spatial grid를 유지하고 있어 DCM 적용이 자연스러움 (DINOv2에서 DeBA-BB가 이미 성공).

#### 2. Gating: GAP gating 선택 (per-image routing)

**비교 검토**:
| 방식 | 구조 | Routing 단위 | 적합성 |
|---|---|---|---|
| P9 Linear | `Linear(C→E)` | per-token | overfitting 위험 (145장 × HW decisions) |
| P20 MLP | `Linear→ReLU→Linear` | per-token | 더 복잡, 같은 위험 |
| **GAP (선택)** | `GAP → W_g·x + noise → softmax` | **per-image** | 안정적, 소규모 데이터 적합 |

**GAP 선택 근거**:
1. P9의 AMF/UAMM가 상수 수렴 (std≈0.0000) → 모델이 per-modality global decision을 선호
2. 145장으로 per-token spatial routing 학습은 overfitting 위험 극대
3. 주요 variation 축이 modality 간 차이 (RGB vs thermal vs LiDAR), spatial 내 차이는 attention이 처리
4. ConvLoRA의 noise term (학습 중 exploration 강제, inference시 제거)이 gate collapse 방지에 효과적

#### 3. Expert 차별화: Multi-scale upsampling

**Dilation은 DCM에 무의미**: DCM의 learned offset (Δp ∈ ℝ²)이 임의 위치로 sampling point 이동 가능 → dilation의 base grid 간격을 offset이 흡수/보상. 학습 후 수렴하면 dilation 차이 소멸.

**Kernel size 차이는 유의미**: 3×3(9점) vs 5×5(25점)은 sampling point **개수** 자체가 다름 → DCM이 흡수 불가. 단, 파라미터 증가.

**Multi-scale upsampling (선택)**:
```
Expert 1: W_d(shared) → ×1 → DCM_1(3×3) → W_u(shared)       (원본 해상도)
Expert 2: W_d(shared) → upsample ×2 → DCM_2(3×3) → downsample → W_u(shared)  (2배 해상도)
```
- 3×3 DCM 유지하면서 해상도 변경으로 effective scale 차별화
- 각 expert는 자기만의 DCM 보유 (DCM 공유 시 MoE 무의미)
- W_d/W_u는 shared → 파라미터 효율
- Scale factor는 ×1, ×2 정도로 보수적 (compute overhead 제한)

#### 4. Cross-layer weight sharing

DeBA 원본 전략 유지:
- **Shared (layers 간 + modalities 간)**: LN (normalization)
- **Per-expert**: DCM weights, offset_mask_conv
- **Shared (experts 간)**: W_d, W_u
- **Per-stage**: W_d, W_u 차원 (Hiera stage별 dim 상이: 112, 224, 448, 896)
- **GAP gate**: stage별 공유 (P20의 SharedGateMLP과 유사한 dim-grouping)

#### 5. P21/P22와의 관계

- P21/P22: **DeBA-FP** (FPN 레벨의 adapter) — backbone 출력 후
- P23: **DeBA-BB** (backbone layer 사이의 adapter) — backbone 내부
- 두 접근은 **보완적** (BB=backbone refinement, FP=FPN refinement)
- 향후 P23 + P22 결합 가능 (DeBA-BB + DeBA-FP)

#### 6. 열린 질문

- Expert 수: 3 (modality) vs 4 (여분 shared expert)?
- Upsample scale factor: ×2만? ×2 + ×4?
- W_d/W_u sharing: expert 간 완전 공유 vs per-expert?
- DeBA-FP(P21/P22)와 동시 적용 시 학습 안정성?

### 구현 상태

- **구현 완료** (2026-03-10)

### 구현 결정 (열린 질문 해결)

1. **Expert 수**: 2 (×1, ×2 scale) — 최소한의 multi-scale 차별화
2. **Upsample scale**: ×2만 — compute 효율과 boundary 보존
3. **W_d/W_u sharing**: expert 간 완전 공유 (파라미터 효율)
4. **Single adapter per block**: Q/V에 같은 delta 적용 — DeBA-BB의 "feature refinement" 개념에 충실

### 구현 상세

**파일**: `sam_lola_utils.py` — `MoE_DeBA_BB`, `_MoE_DeBA_BB_qkv`
**파일**: `sam_lora_image_encoder_seg.py` — `LoRA_Sam_P23`

**MoE_DeBA_BB (단일 공유 모듈, ~325K params)**:
- 2 × DCM (per-expert, shared across 24 blocks): offset_mask_conv + dcm_weight
- 1 × LayerNorm (shared across all)
- 4 × W_d, W_u (per-stage: dim→64, 64→dim)
- 4 × Gate (per-stage: Linear(dim→E))
- 3 × α (per-modality, init=0)

**_MoE_DeBA_BB_qkv (QKV wrapper)**:
- 단일 adapter delta → Q[:dim] += delta, V[-dim:] += delta
- shared_deba_bb reference (cross-layer sharing 달성)

**Block-to-Stage mapping (Hiera-B+)**:
- Blocks 0-2 → Stage 0 (dim=112, 3 blocks)
- Blocks 3-5 → Stage 1 (dim=224, 3 blocks)
- Blocks 6-21 → Stage 2 (dim=448, 16 blocks)
- Blocks 22-23 → Stage 3 (dim=896, 2 blocks)

**Parameter count**:
- deba_bb: 325,361 (~325K)
- cross_modal_head: 14,659 (~15K)
- Total trainable adapter: ~340K (P9 MoE LoRA ~538K 대비 37% 감소)

**학습 명령**: `python train_sam2_lora_paper.py --cfg configs/levine-multiaqua_rgbtl_P23_hardaug8_physaug.yaml`

---

## P24: P9 + Quality-aware Memory Gating via Per-Modality Decoder Distillation (실험 N)

파일: `sam_lora_image_encoder_seg.py` 끝부분, 클래스: `LoRA_Sam_P24`
Quality Head: `sam_lola_utils.py` — `SpatialQualityGating`

### 동기

P9의 memory attention은 모든 모달리티의 memory를 동일하게 취급. UAMM이 vision_feats를 scalar로 modulate하지만, **memory bank에 저장되는 maskmem_features는 무조건 원본 그대로**. 야간에 RGB encoder가 생성한 저품질 feature가 memory에 그대로 저장되면, 이후 모달리티의 memory attention이 오염된 memory를 참조.

### 핵심 아이디어: Teacher-Student Quality Distillation

각 모달리티 feature의 **공간적 품질(spatial quality)**을 예측하는 lightweight head를 학습하고, 예측된 quality map으로 memory bank 저장 시 maskmem_features를 modulate.

### Teacher Signal: Per-pixel CE from SAM2 Decoder

```
Teacher (학습 시만, torch.no_grad):
  per-modality vision_feats → SAM2 decoder (no memory) → teacher_logits (B, C, H, W)
  → F.cross_entropy(teacher_logits, gt_mask, reduction='none') → ce_map (B, H, W)
  → quality_target = exp(-CE) ∈ (0, 1]
  → downsample to FPN size

Student:
  fpn[0] feature → SpatialQualityGating → quality_logits (B, 1, H, W)
  → Loss: BCE_with_logits(logits, quality_target), ignore_mask 적용
```

**CE 기반 target의 장점** (ISSUE-013 참조):
- Decoder가 수렴해도 모달리티별 **구조적 약점은 남음** (LiDAR→하늘, RGB→암전 등)
- Signal이 epoch에 걸쳐 소멸하지 않음 (GT 대비 절대적 오차)

### SpatialQualityGating 구조

```python
class SpatialQualityGating(nn.Module):
    head = Sequential(
        Conv2d(in_channels, 64, 3, padding=1),  # spatial context
        ReLU(),
        Conv2d(64, 64, 3, padding=1),
        ReLU(),
        Conv2d(64, 1, 1),                        # quality logit
    )
    # Init: kaiming + last bias=+1.0 → sigmoid ≈ 0.73 (optimistic start)

    def logits_to_quality(logits):
        return sigmoid(logits) * (1 - min_quality) + min_quality
        # → [min_quality, 1.0] 범위, min_quality=0.1 (완전 zeroing 방지)
```

**파라미터**: ~12.5K (Conv2d 32→64→64→1)

### Memory Modulation (Phase 3)

```python
# Phase 3에서 track_step 후:
maskmem = multi_mask_output_step["maskmem_features"]  # (B, C, H_mem, W_mem)
q_map = quality_maps[frame_idx]  # (B, 1, H_fpn, W_fpn)
q_map_resized = F.interpolate(q_map, size=maskmem.shape[-2:], ...)
multi_mask_output_step["maskmem_features"] = maskmem * q_map_resized
```

- Quality 높은 영역: memory 유지 (×1.0에 가까움)
- Quality 낮은 영역: memory 억제 (×min_quality=0.1까지)
- 이후 모달리티가 이 memory를 참조할 때 열화 영역의 영향 감소

### P9 vs P24 비교

| | P9 | P24 |
| --- | --- | --- |
| MoE LoRA | SoftMoE_LoRA (동일) | SoftMoE_LoRA (동일) |
| Fusion Head | CrossModalFusionHead (동일) | CrossModalFusionHead (동일) |
| UAMM | scalar max-norm (동일) | scalar max-norm (동일) |
| AMF | scalar softmax (동일) | scalar softmax (동일) |
| **Memory Modulation** | 없음 | **SpatialQualityGating** |
| **Teacher Signal** | 없음 | **per-modality decoder CE → exp(-CE)** |
| **추가 Loss** | 없음 | **BCE(quality_logits, target) × λ_gate** |
| 학습 반환 | (o, f) | **(o, f, gate_loss_data)** |
| 추가 파라미터 | 0 | **~12.5K** (SpatialQualityGating) |
| Gradient Checkpoint | 미적용 | **적용 가능** (`GRADIENT_CHECKPOINT: true`) |

### Config 주요 설정

```yaml
MODEL:
  LORA_MODEL    : LoRA_Sam_P24
  LORA_R        : 4
  QUALITY_GATE:
    HIDDEN_DIM  : 64        # SpatialQualityGating 중간 채널
    MIN_QUALITY : 0.1       # quality map 최솟값 (완전 zeroing 방지)

TRAIN:
  LAMBDA_GATE   : 0.5       # quality loss 가중치
  GRADIENT_CHECKPOINT : true # encoder activation checkpointing
  AMP           : false      # P24는 AMP 비활성
```

### Gradient Checkpointing (ISSUE-012 대응)

P24 config에서 `GRADIENT_CHECKPOINT: true` 적용:
```python
# train_sam2_lora_paper.py:416-417
if train_cfg.get('GRADIENT_CHECKPOINT', False):
    model.sam.image_encoder.trunk.gradient_checkpointing = True
```
- Hiera trunk의 각 block activation을 backward 시 재계산 → VRAM 절약
- P23 OOM 문제의 해결책으로도 적용 가능

### Save/Load

```python
merged_dict = {
    **moe_params,              # P9 동일 (gate + experts)
    **cross_modal_tensors,     # P9 동일
    **quality_gating_tensors,  # prefix "quality_gating." (P24 신규)
    **prompt_encoder_tensors,
    **mask_decoder_tensors,
}
```

### 알려진 이슈

- **ISSUE-013**: Teacher signal이 원래 sigmoid confidence로 구현되어 epoch 40에서 포화 → CE 기반으로 수정됨
- Teacher decoder가 **binary mask** 출력 (SAM2 원본 `_forward_sam_heads`) → 4-class CE가 아닌 제한된 signal
- 4-class teacher logits를 위해서는 main decoder의 segmentation head 공유 또는 별도 head 필요

### 시각화

- `train_sam2_lora_paper.py:178-243` — `save_p24_quality_vis()`: 매 epoch 1st batch의 predicted/target quality map 저장
- 출력 위치: `{save_dir}/quality_vis/`

### 구현 상태

- **구현 완료** (2026-03-11)
- Config: `configs/hpca100-multiaqua_rgbtl_P24_hardaug8_physaug.yaml`, `configs/bengio-multiaqua_rgbtl_P24_hardaug8_physaug.yaml`
- Eval config: 미확인
- 학습 스크립트: `gate_loss_data` 3-tuple return 처리, `LAMBDA_GATE` loss 가중치, quality vis 저장
- Augmentation: hardaug8_physaug

---

## P25: Unified Spatial Quality Fusion — Quality Map으로 UAMM + AMF + Memory 통합 (설계 중)

파일: 미구현
기반: P24 (SpatialQualityGating + Teacher-Student CE distillation)

### 동기

P24는 SpatialQualityGating으로 memory modulation만 수행하면서, UAMM/AMF는 여전히 P9의 CrossModalFusionHead(GAP→MLP→softmax)를 사용. 그런데:

1. **CrossModalFusionHead는 8번 실패**: P9~P21까지 모든 variant에서 학습된 상수로 수렴 (std≈0.0000). GAP이 spatial 정보를 소실하고, frozen SAM2 encoder가 modality 간 분포를 정규화하여 입력 의존성이 사라짐.
2. **P24에 이미 spatial quality map 존재**: teacher-supervised quality map이 "각 modality가 각 pixel에서 얼마나 정확한가"를 spatial하게 표현. 이걸 memory modulation에만 쓰는 것은 과소활용.
3. **아키텍처 중복**: 상수 수렴하는 CrossModalFusionHead를 유지할 이유가 없음. Quality map으로 통합하면 파라미터 감소 + 학습 경로 단순화.

### 핵심 변경: CrossModalFusionHead 제거 → Quality Map Triple-Duty

```
P24 (현재):
  CrossModalFusionHead → scalar (B, m) → UAMM, AMF  (상수 수렴, adaptive 불가)
  SpatialQualityGating → spatial (B, 1, H, W) → Memory modulation만

P25 (제안):
  CrossModalFusionHead 제거
  SpatialQualityGating → spatial (B, 1, H, W) × m개 → UAMM + AMF + Memory 모두
```

### Phase별 Quality Map 활용

```
Phase 1: 모달리티별 인코딩 (P9 동일)
  for modal in [img, lidar, thermal]:
    backbone_feat = SAM2_encoder(modal)
    memory_attention(backbone_feat, memory)
    memory.append(backbone_feat)

Phase 2: Quality Map 예측 (P24 Student 활용, CrossModalFusionHead 삭제)
  for i, modal in enumerate(modalities):
    quality_maps[i] = SpatialQualityGating(fpn_feats[i])  # (B, 1, H_fpn, W_fpn)

Phase 3: Spatial UAMM (기존 scalar → spatial)
  for i, modal in enumerate(modalities):
    q_uamm = F.interpolate(quality_maps[i], size=vision_feats.shape[-2:])  # (B, 1, H_v, W_v)
    # max-norm: 가장 높은 quality를 가진 modality를 1.0으로 정규화
    # 3개 모달리티의 quality map을 stack → pixel별 max로 나누기
    vision_feats[i] = vision_feats[i] * q_uamm_normalized[i]

Phase 3.5: Memory Modulation (P24 동일)
  maskmem[i] = maskmem[i] * quality_maps[i]  # memory bank 저장 시 modulation

Phase 4: track_step (P9 동일)
  각 모달리티별 SAM2 decoder 실행

Phase 5: Spatial AMF (기존 scalar → spatial)
  for i, modal in enumerate(modalities):
    q_amf = F.interpolate(quality_maps[i], size=(H_out, W_out))  # (B, 1, H_out, W_out)
  # pixel별 softmax normalization
  q_stack = torch.stack([q_amf_0, q_amf_1, q_amf_2], dim=0)  # (m, B, 1, H, W)
  q_norm = q_stack / q_stack.sum(dim=0, keepdim=True)          # pixel별 비율
  fused = sum(q_norm[i] * output[i] for i in range(m))
```

### Teacher Signal (P24에서 계승, 변경 없음)

```
Teacher (학습 시만, torch.no_grad):
  per-modality vision_feats → SAM2 decoder (no memory) → teacher_logits (B, 4, H, W)
  → F.cross_entropy(teacher_logits, gt_mask, reduction='none') → ce_map (B, H, W)
  → quality_target = exp(-CE) ∈ (0, 1]
  → downsample to FPN size
  → BCE_with_logits(student_logits, quality_target), ignore_mask 적용

Student:
  SpatialQualityGating (P24 동일, ~12.5K params)
```

**주의**: P24의 ISSUE-013이 해결된 상태여야 함 (4-class CE 기반 teacher signal)

### P9 vs P24 vs P25 비교

| | P9 | P24 | **P25** |
| --- | --- | --- | --- |
| CrossModalFusionHead | ✅ (상수 수렴) | ✅ (상수 수렴) | **❌ 제거** |
| SpatialQualityGating | 없음 | ✅ (Memory만) | **✅ (Triple-Duty)** |
| UAMM | scalar max-norm `(B, m)` | scalar max-norm `(B, m)` | **spatial max-norm `(B, 1, H, W)`** |
| AMF | scalar softmax `(B, m)` | scalar softmax `(B, m)` | **spatial softmax `(B, 1, H, W)`** |
| Memory Modulation | 없음 | spatial quality | **spatial quality (동일)** |
| Teacher Signal | 없음 | CE-based | **CE-based (동일)** |
| 추가 파라미터 | CrossModalFusionHead ~3K | CrossModalFusionHead ~3K + SQG ~12.5K | **SQG ~12.5K만** |
| Scoring 근거 | 입력 무관 상수 | Memory만 adaptive | **UAMM + AMF + Memory 모두 adaptive** |

### Spatial UAMM 상세

P9의 scalar UAMM:
```python
# scores: (B, m) — 모든 pixel에 동일
scores_norm = scores / scores.max(dim=1, keepdim=True).values  # max-norm
vision_feats[i] = vision_feats[i] * scores_norm[:, i:i+1, None, None]  # broadcast
```

P25의 spatial UAMM:
```python
# quality_maps: list of (B, 1, H_fpn, W_fpn) — pixel별 다른 quality
q_stack = torch.stack(quality_maps, dim=1)  # (B, m, 1, H, W)
q_max = q_stack.max(dim=1, keepdim=True).values  # (B, 1, 1, H, W)
q_norm = q_stack / q_max.clamp(min=1e-6)  # pixel별 max-norm, (B, m, 1, H, W)
for i in range(m):
    q_i = F.interpolate(q_norm[:, i], size=vision_feats[i].shape[-2:])
    vision_feats[i] = vision_feats[i] * q_i
```

### Spatial AMF 상세

P9의 scalar AMF:
```python
# cross_weights: (B, m) — softmax normalized
fused = sum(cross_weights[:, i:i+1, None, None] * seg_output[i] for i in range(m))
```

P25의 spatial AMF:
```python
# quality_maps를 output resolution으로 interpolate
q_amf = [F.interpolate(q, size=(H_out, W_out), mode='bilinear') for q in quality_maps]
q_stack = torch.stack(q_amf, dim=0)  # (m, B, 1, H, W)
q_softmax = F.softmax(q_stack, dim=0)  # pixel별 softmax across modalities
fused = sum(q_softmax[i] * seg_output[i] for i in range(m))
```

### 논리적 타당성 평가: 80~85%

**강점**:
1. CrossModalFusionHead 8연패 → 구조적 교체 필요성이 명확히 입증됨
2. Teacher supervision이 P12~P19 실패 원인(GT 없는 scoring)을 근본적으로 해결
3. Spatial 정보 보존 — GAP의 정보 소실 문제 해소
4. 아키텍처 단순화 (모듈 1개 제거, quality map 하나로 통합)
5. P24에서 이미 quality map 인프라 구축 → 증분 변경만 필요

**리스크**:
1. **Student 오류 cascade**: quality map 하나가 3곳을 동시 결정 → 예측 오류 시 동시 영향
   - 완화: min_quality=0.1로 완전 zeroing 방지, 초기 bias=+1.0으로 optimistic start
2. **Quality ≠ 최적 fusion weight**: "정확도"와 "fusion 기여도"는 미묘하게 다를 수 있음
   - 그러나 실용적으로 quality가 높을수록 더 기여해야 하는 건 맞으므로 좋은 proxy
3. **Domain gap**: 학습(주간+aug)의 quality 패턴이 야간 test에서 전이되는지 불확실
   - 완화: hardaug8_physaug의 극저조도 시뮬레이션, P24 결과로 사전 검증 가능

### 구현 시 주의사항

1. **P24 결과 먼저 확인**: P24의 quality map이 합리적 spatial pattern을 보이는지 확인 후 P25 진행
2. **ISSUE-013 선결**: 4-class CE teacher signal이 구현되어야 quality map의 semantic 품질이 보장됨
3. **Gradient flow**: SpatialQualityGating → quality_map → UAMM/AMF/Memory 세 경로로 gradient 전파 → loss landscape 변화 가능
4. **기존 CrossModalFusionHead 코드 제거**: UAMM/AMF에서 `cross_weights` 참조하는 모든 곳을 `quality_maps`로 교체

### 구현 상태

- **설계 완료** (2026-03-14)
- 구현 대기: P24 학습 결과 + ISSUE-013 해결 후 착수
- 기반 코드: P24의 `LoRA_Sam_P24` + `SpatialQualityGating`

---

## 버전 비교 총괄

### 표 A: P8~P19 (Fusion Head 중심 계열)

| 구분 | P8 | P9 | P10 | P11 | P12 | P13 | P14 | P15 | P16 | P17 | P19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Head | ConfHeadV2 | CrossModalFH | CrossModalFHV2 | CrossModalFHV2 | CrossModalFH | AuxHead+Energy | AuxDec×3+Energy | AuxDec×3+Energy(spatial) | AuxDec×3+Entropy(spatial) | MSAuxDec×3+Entropy(spatial) | SpatialCrossFH |
| UAMM | sigmoid | max-norm | max-norm | softmax+τ | max-norm | max-norm | max-norm | spatial max-norm | spatial max-norm | spatial max-norm | spatial max-norm |
| AMF | norm(sig) | softmax | softmax | softmax | softmax | energy softmax | energy softmax | spatial energy | spatial entropy | spatial entropy | spatial softmax |
| Aux Head | 없음 | 없음 | AuxHead×3 | AuxHead×3 | 없음 | 공유×1 | 독립×3 | 독립×3 | 독립×3 | MS독립×3 | 없음 |
| 추가 Loss | 없음 | 없음 | KL(0.5) | KL+MI(1.0) | 없음 | auxCE(0.3) | auxCE(0.3) | auxCE(0.3) | auxCE(0.3) | auxCE(0.3) | 없음 |
| MoE | SoftMoE LoRA | SoftMoE LoRA | SoftMoE LoRA | SoftMoE LoRA | SoftMoE LoRA | SoftMoE LoRA | SoftMoE LoRA | SoftMoE LoRA | SoftMoE LoRA | SoftMoE LoRA | SoftMoE LoRA |
| 학습 반환 | (o,f) | (o,f) | (o,f,aux,w) | (o,f,aux,w,g) | (o,f) | (o,f,aux) | (o,f,aux) | (o,f,aux) | (o,f,aux) | (o,f,aux) | (o,f) |
| FPN 레벨 | fpn[0] | fpn[0] | fpn[0] | fpn[0] | fpn[0] | fpn[0] | fpn[0] | fpn[0] | fpn[0] | fpn[0,1,2] | fpn[0,1,2] |
| 최선 M-score | 78.45 | **81.47** | 79.27 | 77.09 | 80.80 | 81.21 | 74.27 | 71.05 | 68.42 | 73.23 | 구현 완료 |

### 표 B: P20~P24 (MoE/Adapter/Memory 강화 계열, P9 기반)

| 구분 | P9 (기준) | P20 | P21 | P22 | P23 | P24 |
| --- | --- | --- | --- | --- | --- | --- |
| 기반 | — | P9 | P9 | P9 | P9 | P9 |
| 핵심 변경 | — | MLP Gate + Rank↑ | DeBA-FP (fpn[0]) | DeBA-FP MultiScale | MoE DeBA-BB | Quality Memory Gating |
| MoE Layer | SoftMoE_LoRA | **SoftMoE_LoRA_V2** | SoftMoE_LoRA | SoftMoE_LoRA | **MoE_DeBA_BB** | SoftMoE_LoRA |
| Gate | Linear(C→3)×48 | **SharedGateMLP×4** | Linear(C→3)×48 | Linear(C→3)×48 | **GAP+Linear×4** | Linear(C→3)×48 |
| Rank | 4 | 4 | 4 | 4 | N/A (conv) | 4 |
| DeBA-FP | 없음 | 없음 | **fpn[0] only** | **fpn[0,1,2]** | 없음 | 없음 |
| DeBA-BB | 없음 | 없음 | 없음 | 없음 | **24 blocks** | 없음 |
| Memory Mod | 없음 | 없음 | 없음 | 없음 | 없음 | **SpatialQualityGating** |
| Fusion Head | CrossModalFH | CrossModalFH | CrossModalFH | CrossModalFH | CrossModalFH | CrossModalFH |
| UAMM/AMF | scalar | scalar | scalar | scalar | scalar | scalar |
| 추가 Loss | 없음 | 없음 | 없음 | 없음 | 없음 | **BCE(quality) ×λ** |
| 학습 반환 | (o,f) | (o,f) | (o,f) | (o,f) | (o,f) | **(o,f,gate_data)** |
| 추가 파라미터 | 0 | +700K (rank↑) | +85K | +98K | +325K | +12.5K |
| Grad Ckpt | 없음 | 없음 | 없음 | 없음 | 필요(OOM) | **적용** |
| 최선 M-score | **81.98** | 학습 대기 | 학습 대기 | 학습 대기 | OOM (ISSUE-012) | 학습 중 |

### 표 C: P25~P26 (Spatial Quality Fusion 계열, CrossModalFusionHead 제거)

| 구분 | P25 | **P26 (설계 v4)** |
| --- | --- | --- |
| 기반 | P9 + P24 SQG | P25 |
| 핵심 변경 | CrossModalFH 제거, spatial quality triple-duty | **SQG 분리 + Multi-Scale FPN + Per-Modal Decoder + Modal-Cond MoE + triple-duty 해소 + UAMM softmax + Memory mod 제거** |
| SQG 입력 | fpn[0] only (32ch) | **fpn[0,1,2] concat (96ch)** |
| SQG | 1개 (공유, 12.5K) | **3개 (독립, ~42K) + fpn proj ~19K** |
| SAM2 Decoder | 1개 (공유, 3회 호출) | **3개 (모달리티별 독립, ~4M×3)** |
| UAMM | spatial max-norm `(B,1,H,W)` | **spatial softmax `(B,m,1,H,W)` — smooth, 불연속 제거** |
| AMF | spatial softmax (SQG 기반) | **output entropy 기반 confidence — SQG와 분리** |
| Memory Mod | spatial quality (maskmem×q) | **제거 — UAMM에서 이미 조절, 이중 페널티 방지** |
| Fusion Head | **없음** | **없음** |
| Teacher target | `exp(-CE)` 절대 quality | **`softmax(-CE_stack/tau)` relative quality (모달리티 간 경쟁)** |
| 추가 Loss | BCE(quality) ×λ | **KL(pred_dist, target_dist) ×λ** |
| min_quality | 0.1 | **0.3** |
| DeBA-FP | 없음 | **옵션 (config on/off, ablation용)** |
| MoE LoRA Gate | input-only, 상수 수렴 | **modality embedding conditioned** |
| 추가 파라미터 | +12.5K | **~8M (decoder ×2) + 61K (SQG+proj) + ~수십 (modal embed)** |
| VRAM 추가 | — | **~0.13GB (weight+optimizer만, activation 동일)** |
| 최선 M-score | 학습 중 | 설계 완료 (P25 결과 대기) |

---

## P26: Per-Modality SQG + Multi-Scale + Per-Modality Decoder + Modal-Cond MoE + UAMM Softmax (설계 v5, 2026-03-23)

### 동기

P25의 비판적 분석 (6가지) + 추가 분석 결과, 아래 구조적 문제를 확인:
1. **SQG 가중치 공유**: 3개 모달리티에 하나의 SQG → multi-task 충돌 (ISSUE-015)
2. **Triple-duty**: quality map이 UAMM/AMF/Memory 3곳에 공유 → optimization conflict
3. **Teacher target 분포**: `exp(-CE)` 대부분 ~1.0 → 유의미한 variation이 경계 일부에만 존재
4. **Pixel-wise max-norm 불연속**: max modality 전환 시 정규화 기준 불연속
5. **Memory modulation 이중 페널티**: UAMM에서 이미 조절된 feature의 memory를 다시 깎음
6. **min_quality=0.1 연쇄 약화**: 3곳에 동시 적용 시 복합 효과로 정보 소실
7. **Shared Decoder 충돌**: SAM2 decoder 1개가 3개 모달리티의 서로 다른 feature 분포를 처리 → SQG와 동일한 multi-task 충돌. VRAM 추가 ~0.13GB로 무시 가능

### P26 설계 — 6가지 변경

#### 변경 ①: 모달리티별 독립 SQG + Multi-Scale FPN 입력 (ISSUE-015 해결)

```python
# P25: fpn[0]만 사용, SQG 1개 공유
self.quality_gating = SpatialQualityGating(in_channels=256, ...)  # 1개, 12.5K

# P26: fpn[0,1,2] multi-scale + SQG 모달리티별 독립
# fpn[0]: (B, 32, 256, 256)  — high-res, fine detail
# fpn[1]: (B, 64, 128, 128)  — mid-res
# fpn[2]: (B, 256, 64, 64)   — low-res, semantic

# Multi-scale fusion: fpn[1,2]를 fpn[0] 해상도로 resize 후 project & concat
self.fpn_proj1 = nn.Conv2d(64, 32, 1)    # fpn[1] channel → fpn[0] channel
self.fpn_proj2 = nn.Conv2d(256, 32, 1)   # fpn[2] channel → fpn[0] channel
# concat 후 SQG 입력: in_channels = 32 * 3 = 96

self.quality_gating_rgb = SpatialQualityGating(in_channels=96, hidden_dim=64, min_quality=0.3)
self.quality_gating_thr = SpatialQualityGating(in_channels=96, hidden_dim=64, min_quality=0.3)
self.quality_gating_lid = SpatialQualityGating(in_channels=96, hidden_dim=64, min_quality=0.3)
# 총 ~42K params (SQG) + ~19K (proj) ≈ 61K
```

**Multi-scale 적용 방식**:
```python
def _fuse_fpn_multiscale(self, backbone_fpn):
    """fpn[0,1,2]를 fpn[0] 해상도로 합쳐서 SQG 입력 생성"""
    f0 = backbone_fpn[0]  # (B, 32, 256, 256)
    f1 = F.interpolate(self.fpn_proj1(backbone_fpn[1]), size=f0.shape[-2:], mode='bilinear')
    f2 = F.interpolate(self.fpn_proj2(backbone_fpn[2]), size=f0.shape[-2:], mode='bilinear')
    return torch.cat([f0, f1, f2], dim=1)  # (B, 96, 256, 256)
```

**DeBA-FP (선택적)**: Config `DEBA_FP: true/false`로 on/off
- on: P22의 DeBA-FP로 cross-scale deformable attention refinement 후 위 fusion 적용 (+~98K params)
- off: 단순 project + resize + concat (기본값)
- Ablation에서 비교하여 DeBA 효과 검증

각 SQG가 해당 모달리티의 multi-scale feature에 특화 학습. KD도 모달리티별로 독립 수행.

#### 변경 ②: UAMM softmax 정규화 (max-norm → softmax)

```python
# P25: pixel-wise max-norm (불연속)
q_max = q_stack.max(dim=1, keepdim=True).values
q_uamm_norm = q_stack / q_max.clamp(min=1e-6)

# P26: pixel-wise softmax (연속, smooth)
q_uamm_norm = F.softmax(q_logit_stack / tau_uamm, dim=1)  # (B, m, 1, H, W)
# tau_uamm: temperature (config 설정, default=1.0)
```

- max-norm의 불연속 문제 해소 — softmax는 연속이고 미분 가능
- 합이 1로 보장되어 "경쟁" 구조 자연스러움
- max modality가 바뀌는 경계에서도 가중치 smooth 전환

#### 변경 ③: Relative Quality Teacher Target

```python
# P25: 절대 quality per modality
quality_target[i] = exp(-CE[i])  # 독립, 대부분 ~1.0

# P26: 모달리티 간 상대적 비교
ce_stack = torch.stack([CE_rgb, CE_thr, CE_lid], dim=0)  # (3, B, H, W)
quality_target_dist = F.softmax(-ce_stack / tau_teacher, dim=0)  # (3, B, 1, H, W)
# tau_teacher: teacher temperature (config 설정, default=0.5~1.0)
```

- 쉬운 픽셀(sky 내부): 3개 다 CE≈0 → softmax ≈ uniform → 균등 fusion (올바름)
- 어려운 픽셀(경계): CE 차이 큼 → sharp routing → 차등 fusion (필요한 곳에서만)
- Loss: `BCE` → `KL divergence`로 변경 (분포 간 비교)

```python
# P25: BCE per-modality
loss = sum(BCE_with_logits(pred[i], target[i]) for i in range(m)) / m

# P26: KL divergence (모달리티 간 관계 학습)
pred_dist = F.log_softmax(torch.stack(pred_logits, dim=0) / tau_uamm, dim=0)
loss = F.kl_div(pred_dist, quality_target_dist.detach(), reduction='batchmean')
```

#### 변경 ④: AMF를 output entropy 기반으로 분리 (triple-duty 해소)

```python
# P25: AMF도 SQG quality map 사용 (triple-duty)
q_amf_norm = quality_maps / sum(quality_maps)  # SQG에 의존

# P26: AMF는 decode 결과의 자체 확신도 사용 (SQG와 독립)
amf_weights = []
for i in range(m):
    prob = F.softmax(output[i], dim=1)  # (B, 4, H, W) — 4 class probabilities
    entropy = -(prob * prob.log().clamp(min=-100)).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    confidence = 1.0 - entropy / math.log(num_classes)  # normalized to [0, 1]
    amf_weights.append(confidence)

amf_stack = torch.stack(amf_weights, dim=0)  # (m, B, 1, H, W)
amf_norm = amf_stack / amf_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)
m_output = sum(amf_norm[i] * output[i] for i in range(m))
```

~~핵심 (v5): SQG quality map은 **UAMM에만** 사용. AMF는 모델의 decode output 자체 확신도로 판단.~~
~~- UAMM: "encoding quality — memory attention 전 input 조정" (SQG, teacher 학습)~~
~~- AMF: "decoding confidence — memory attention 후 output 융합" (output entropy, 학습 불필요)~~
~~- 역할 분리 → optimization conflict 제거~~

**v6 수정**: AMF도 SQG quality map 기반으로 변경 (output entropy 제거). 상세는 "v6 설계 수정" 섹션 참조.

#### 변경 ⑤: Memory Modulation 제거

```python
# P25:
maskmem_features = maskmem * quality_map_resized  # 이중 페널티

# P26: 제거 (UAMM에서 이미 quality-aware modulation 완료)
# maskmem 그대로 memory bank에 저장
```

UAMM에서 quality가 낮은 모달리티의 vision_feats를 이미 줄였으므로, track_step에서 생성된 maskmem은 이미 quality-aware. 거기에 다시 곱하면 이중 페널티.
Memory attention 자체가 attention mechanism이므로, query-key 매칭을 통해 유용한 정보를 알아서 선택.

#### 변경 ⑥: Per-Modality Decoder + Shared Inference Decoder (역할 분리)

**Decoder 구성: 총 m+1개 (모달리티별 m개 + 추론용 1개)**

```python
# P25: 1개 decoder, 3번 호출 (학습+추론 공유)
for frame_idx in range(m):
    output = self.sam.track_step(vision_feats[frame_idx], ...)  # 같은 decoder

# P26: 모달리티별 auxiliary decoder (m개) + shared inference decoder (1개)
# (1) Per-modal decoder: 학습 시 직접 CE supervision + SQG quality target 생성
self.per_modal_decoders = nn.ModuleList([
    deepcopy(sam_model.sam_mask_decoder) for _ in range(m)
])  # m × ~4M params

# (2) Shared inference decoder: track_step (memory attention 포함) 추론 경로
# = sam_model.sam_mask_decoder (원본 유지)
```

**역할 분리 — 핵심 설계**:

| Decoder | 학습 | 추론 | 역할 |
|---------|------|------|------|
| `per_modal_decoders[i]` | ✅ 직접 CE loss (auxiliary) | ❌ 미사용 | Quality oracle: 모달리티별 spatial quality 측정 → SQG target |
| `sam_mask_decoder` (shared) | ✅ main CE loss (AMF fused) | ✅ track_step | 실제 추론: memory attention 결과를 decoding |

**학습 시 gradient 흐름**:
```
Per-modal decoder[i] path (auxiliary):
  encoder → vision_feats[i] → per_modal_decoder[i] → CE(pred, GT)
  → gradient: encoder ✓, per_modal_decoder[i] ✓
  → CE map → quality target for SQG (KL loss)

Shared decoder path (main):
  encoder → vision_feats[i] → UAMM 가중 → memory attention → shared_decoder → AMF → CE(fused, GT)
  → gradient: encoder ✓, memory attention ✓, shared_decoder ✓, SQG (via UAMM) ✓
```

**Per-modal decoder가 추론에 불필요한 이유**:
- Per-modal decoder의 목적은 모달리티별 CE map 생성 → SQG 학습 target
- SQG가 충분히 학습되면, encoder 피쳐만으로 quality map 예측 가능 (knowledge distillation)
- 추론 시 SQG가 per-modal decoder를 **대체** → decoder m개 불필요, SQG (경량 conv head)만 사용

**Per-modal CE loss가 memory attention을 개선하는 경로 (간접적)**:
- Per-modal CE loss는 memory attention 파라미터를 **직접 학습시키지 않음** (gradient 경로에 없음)
- 개선은 **두 가지 간접 경로**를 통해 이루어짐:
  1. **Encoder 피쳐 품질 향상**: per-modal CE → encoder에 추가 gradient → "이 모달리티 피쳐만으로도 segmentation 가능해야 한다"는 압력 → encoder가 모달리티별 더 informative한 피쳐 생성 → memory attention의 **입력**이 좋아짐
  2. **SQG quality target 정확도 향상**: 잘 학습된 per-modal decoder → 정확한 CE map → SQG가 정확한 quality map 학습 → UAMM이 memory attention 입력을 **공간적으로 정확하게 가중**
- 비유: per-modal CE는 **재료(입력)**를 좋게 만들고, main CE는 **조리법(memory attention 파라미터)**을 학습시킴
- Memory attention 파라미터 자체는 main path의 CE loss에서만 학습됨

**동기**: SQG와 동일한 문제 — RGB/Thermal/LiDAR의 feature 분포가 근본적으로 다른데 하나의 decoder weight로 quality를 측정하면 multi-task 충돌. 모달리티별 decoder는 각 모달리티의 spatial quality를 정확히 측정하기 위한 auxiliary head.

**비용**: Weight+Optimizer ~0.13GB 추가 (14GB 대비 무시 가능). 추론 시에는 shared decoder 1개만 사용하므로 추론 비용 증가 없음.

**분리 대상 vs 공유 유지**:
| 모듈 | P26 | 이유 |
|------|-----|------|
| Per-modal Decoder | **분리 ×m** (학습 전용) | 모달리티별 quality oracle, 직접 CE supervision |
| Shared Inference Decoder | **×1** (학습+추론) | track_step의 실제 decoding, memory attention 결과 처리 |
| Memory Attention | **공유 ×1** | cross-modal interaction이 목적, 분리하면 의미 없음 |
| Memory Encoder | **공유 ×1** | memory bank format 통일 필요 |

#### 변경 ⑦: Modality-Conditioned MoE LoRA Gate

**문제**: MoE LoRA gate(`Linear(C, 3) + softmax`)가 입력과 무관하게 고정 비율로 수렴.
- 모든 expert가 항상 참여(soft routing) → 특화 압력 약함
- Gate 전용 loss 없음, segmentation loss에서 gate까지 gradient 경로가 너무 김
- 결과: expert weight는 다르게 학습되더라도 mixing 비율이 상수 → 사실상 단일 LoRA

**해결**: 모달리티 identity embedding을 gate condition으로 추가

```python
# P25: gate_logits = self.gate(x)  # token feature만, 모달리티 구분 없음
# P26: modality embedding 추가
self.modal_embed = nn.Embedding(3, cond_dim)  # RGB=0, THR=1, LID=2
# cond_dim은 기존 P12의 cond_dim 인프라 활용

# Encoder forward 시:
for i, modal in enumerate([RGB, THR, LID]):
    modal_cond = self.modal_embed(torch.tensor(i, device=device))  # (cond_dim,)
    for layer in self.moe_layers_q + self.moe_layers_v:
        layer.set_condition(modal_cond.unsqueeze(0).expand(B, -1))  # (B, cond_dim)
    image_embedding[i] = self.sam.forward_image(modal)

# SoftMoE_LoRA_Layer.forward 내부 (기존 P12 인프라):
gate_logits = self.gate(x) + self.cond_proj(self._condition)
# → token feature 기반 routing + modality identity bias
```

**핵심**: Quality가 아닌 **modality identity**로 conditioning
- Quality conditioning의 문제: thermal quality가 항상 RGB보다 낮으면 condition이 상수화 → 또 고정 비율
- Modality embedding: "이 모달리티의 feature 특성에 맞는 expert 조합"을 학습 → quality 순서와 무관

**비용**: `nn.Embedding(3, cond_dim)` ≈ 수십 파라미터 + 기존 `cond_proj` 재사용 → 거의 0

**관련 연구**:
- **VLMo/BEiT-3** (NeurIPS'22, CVPR'23): Mixture-of-Modality-Experts (MoME) — 모달리티별 전용 FFN expert (hard routing). 우리는 soft 버전
- **Mod-Squad** (CVPR'23): Modality-aware sparse MoE + aux loss로 expert 특화 유도
- **MoE-Adapters4CL** (NeurIPS'24): **LoRA-level MoE + task/domain identity embedding** — 우리 설계와 가장 유사. "task identity → LoRA expert routing"을 "modality identity → LoRA expert routing"으로 대응
- **AdaMoLE** (arXiv'24): Soft MoE LoRA, input-conditioned gate — 우리 P12 기반 구조와 거의 동일
- **Uni-MoE** (ACL'24): Top-k + modality balancing loss로 routing collapse 방지

### Forward 흐름 (P25 대비 변경점 ★ 표시)

```
Phase 1: Image Encoding ★ Modality-Conditioned MoE LoRA
  for i, modal in enumerate([RGB, THR, LID]):
    ★ set MoE gate condition = modal_embed[i]
    SAM2_encoder(modal) → backbone FPN features + vision_feats

Phase 2: Spatial Quality Map ★ Multi-Scale FPN + 모달리티별 독립 SQG
  fpn_RGB[0,1,2] → proj+resize+concat → (B,96,256,256) → SQG_rgb → q_logit₀
  fpn_THR[0,1,2] → proj+resize+concat → (B,96,256,256) → SQG_thr → q_logit₁
  fpn_LID[0,1,2] → proj+resize+concat → (B,96,256,256) → SQG_lid → q_logit₂

Phase 2.5 (Training): Per-Modal Auxiliary CE + SQG Target ★ 직접 supervision
  for each modality:
    ★ per_modal_pred[i] = per_modal_decoder[i](vision_feats[i])  # no memory attention
    ★ aux_CE_loss[i] = cross_entropy(per_modal_pred[i], gt)      # 직접 supervision → decoder 학습
    CE_map[i] = per-pixel cross_entropy(per_modal_pred[i], gt)   # spatial quality 측정
  CE_stack = [CE_rgb, CE_thr, CE_lid]
  quality_target_dist = softmax(-CE_stack / tau_teacher, dim=0)  # SQG KL target
  sqg_loss = KL(log_softmax(q_logit_stack / tau_uamm), quality_target_dist)
  ★ total_aux_loss = sum(aux_CE_loss) / m + sqg_loss

Phase 3: Spatial UAMM + Shared Decoder ★ softmax 정규화 + 추론용 decoder
  q_uamm = softmax(q_logit_stack / tau_uamm, dim=modality)
  for each modality:
    vision_feats[i] *= interpolated(q_uamm[i])
    ★ track_step with shared_decoder(modulated_vision_feats)  # 추론용 decoder 1개
    ★ Memory Modulation 없음 (maskmem 그대로 저장)

Phase 4: AMF ★ SQG quality map 기반 (v6: entropy 제거, tau 제거)
  ★ sqg_weight = softmax(q_logit_stack, dim=modality)  # UAMM과 동일 weight 재사용
  m_output = Σ sqg_weight[i] × output[i]
```

### Config 변경

```yaml
MODEL:
  LORA_MODEL    : LoRA_Sam_P26
  QUALITY_GATE:
    HIDDEN_DIM   : 64
    MIN_QUALITY  : 0.3          # P25: 0.1 → P26: 0.3 (UAMM 전용, 연쇄 약화 방지)
    PER_MODALITY : true         # P26 신규
    TAU_UAMM     : 1.0          # P26 신규: UAMM softmax temperature
    TAU_TEACHER  : 0.5          # P26 신규: teacher target temperature
    MEMORY_MOD   : false        # P26 신규: memory modulation 비활성화
    AMF_MODE     : sqg_quality     # v6 변경: output_entropy → sqg_quality (SQG quality map 재사용, tau 없음)
    PER_MODALITY_DECODER : true   # P26 신규: 모달리티별 auxiliary decoder (학습 전용, 추론 시 미사용)
    AUX_CE_WEIGHT        : 0.5    # P26 신규: per-modal auxiliary CE loss 가중치
    MULTI_SCALE_SQG : true        # P26 신규: fpn[0,1,2] multi-scale SQG 입력
    DEBA_FP         : false       # P26 옵션: DeBA-FP cross-scale refinement (ablation용)
  LORA_COND_DIM   : 8            # P26 신규: modality embedding dimension for MoE gate conditioning
  MODAL_COND_MOE  : true         # P26 신규: modality-conditioned MoE LoRA gate
```

### 관련 연구 참조 — DGFusion (arxiv 2509.09828)

DGFusion은 **depth를 proxy로** spatial fusion을 가이드하는 방법으로, depth token을 cross-attention의 조건으로 사용.
- 유사점: 입력 조건에 따라 spatially varying fusion weight를 학습
- 차이점: DGFusion은 일반 cross-attention fusion, MemorySAM은 **SAM2 memory attention pipeline**을 모달리티 축으로 전용하고 그 전에 quality-aware modulation(UAMM) 적용
- **우리의 novelty**: memory attention 입력 전 quality-guided spatial modulation은 DGFusion과 근본적으로 다른 파이프라인

### 리스크

1. **AMF output entropy의 calibration**: 모델 출력의 entropy가 실제 품질을 반영하는지? 과도하게 confident한 잘못된 예측은 entropy가 낮아도 quality가 나쁨 → "confident but wrong" 문제
   - 완화: 학습이 진행되면 calibration이 자연스럽게 개선됨. 초반에는 AMF가 ~uniform에 가까움 (모든 output이 비슷하게 uncertain)
2. **tau 하이퍼파라미터 민감도**: tau_uamm과 tau_teacher가 routing sharpness를 결정 → grid search 필요
   - 완화: tau=1.0을 baseline으로 시작, 결과 보고 조정
3. **Night aug 충분성**: teacher가 augmented night image에서 CE를 계산하므로 night 분포는 커버됨. 단, 완전 새로운 열화 패턴(안개, 비)에서의 일반화는 한계

### 구현 상태

- **v5 구현 완료** (2026-03-23), **v6 구현 완료** (2026-03-24)
  - ① Per-Modality SQG (ModuleList)
  - ② UAMM softmax
  - ③ Relative quality teacher + KL loss
  - ④ AMF output entropy
  - ⑤ Memory mod 제거
  - ⑥ Multi-Scale FPN (fpn_proj1/2 + concat → 96ch SQG input)
  - ⑦ Per-Modality Decoder: **v6 역할 분리 완료**
    - `per_modal_decoders` (×m): 학습 전용 auxiliary CE head (`_auxiliary_decode_single`)
    - `self.sam.sam_mask_decoder`: shared inference decoder (학습+추론)
    - Phase 2.5: `no_grad` 제거, per_modal_decoder에 grad flow → SQG target 정확도 향상
    - Phase 3: `_swap_decoder` 제거 → shared decoder만 사용
    - `_encode_single_modality`: shared decoder의 conv_s0/s1만 사용
  - ⑧ Modality-Conditioned MoE LoRA Gate (nn.Embedding + cond_dim=8)
- `LoRA_Sam_P26` 클래스: `sam_lora_image_encoder_seg.py` 끝에 추가
- Train/Val/Vis 스크립트 모두 P26 v6 대응 완료
- `train_sam2_lora_paper.py`: `lambda_aux_ce` (AUX_CE_WEIGHT, default 0.5) 추가
- Configs:
  - `configs/hpca100-multiaqua_rgbtl_P26_hardaug8_physaug.yaml` (MULTIAQUA, HPC)
  - `configs/eval_config/hpca100-multiaqua_rgbtl_P26_hardaug8_physaug.yaml` (MULTIAQUA eval)
  - `configs/levine-deliver_rgbdel_P26_physaug.yaml` (DELIVER, levine, 4모달)
- **미구현**: DeBA-FP (옵션, ablation용 — config `DEBA_FP: true`로 활성화 시 구현 필요)
- **선결 조건**: P25 학습 결과 확인 후 학습 시작

#### ✅ v6 구현 완료 — Per-Modal Decoder 역할 분리 (2026-03-24)

**v5의 문제**: per_modal_decoder가 teacher (no_grad) + track_step (inference) 모두에서 사용됨
- Teacher: `no_grad` → per-modal decoder가 학습되지 않음 → quality target이 초기 상태에 고정
- track_step: per-modal decoder가 추론에도 사용 → 추론 시 decoder 3개 필요 (비효율)

**v6 변경 (구현 대기)**:
1. **Per-modal decoder (m개)**: 학습 전용 auxiliary head
   - 직접 CE loss로 학습 (no_grad 제거) → 모달리티별 decoding 능력 향상
   - CE map → SQG quality target (KL distillation)
   - 추론 시 **사용 안 함**
2. **Shared decoder (1개)**: track_step 추론 경로
   - memory attention 결과를 decoding
   - main CE loss (AMF fused) 로 학습
   - 학습 + 추론 모두 사용
3. **총 decoder 수**: m + 1 = 4개 (학습 시), 1개 (추론 시)
4. **SQG = knowledge distillation**: per-modal decoder의 지식을 경량 conv head로 증류
   - 추론 시 per-modal decoder 대신 SQG가 quality 예측
5. **AMF: output entropy → SQG quality map 기반으로 변경** (변경 ④ 수정)
   - v5의 output entropy AMF 제거 → SQG quality map을 UAMM과 AMF에 모두 사용
   - entropy 기반의 "confident but wrong" 문제 해결: SQG는 per-modal decoder CE로 학습되므로 실제 정확도를 반영
   - P25의 triple-duty 문제 재발 우려 없음: Memory modulation 제거(⑤)로 dual-duty, per-modal SQG(①)로 충돌 없음
   - **tau_amf 불필요 → 제거, UAMM과 AMF가 동일한 weight 사용**:
     - SQG logit 스케일이 KL loss로 이미 calibrate됨 → 추가 temperature 중복
     - learnable tau도 불필요: SQG 자체가 logit 크기를 학습하므로 tau와 역할 중복
     - UAMM/AMF 모두 "이 위치에서 이 모달리티를 얼마나 신뢰하는가"에 대한 동일한 답 → 일관된 적용이 자연스러움
     ```python
     # UAMM + AMF 공통 weight (한 번만 계산)
     sqg_weight = F.softmax(q_logit_stack, dim=0)  # (m, B, 1, H, W), tau 없음

     # UAMM: vision_feats[i] *= sqg_weight[i]  (feature modulation)
     # AMF:  m_output = Σ sqg_weight[i] * output[i]  (output fusion)
     ```
   - **Overconfident 방지 안전장치**:
     - SQG target = `softmax(-CE/tau_teacher)` → CE가 정확히 0이 아닌 한 one-hot 불가능
     - SQG 출력 범위 = `sigmoid * 0.7 + 0.3` → min_quality=0.3, 어떤 모달리티든 완전 무시 불가
     - GT 대비 실제 정확도 기반이므로 entropy처럼 "confident but wrong"에 취약하지 않음

**구현 계획 (코딩봇용)**:

##### 1. `sam_lora_image_encoder_seg.py` — `LoRA_Sam_P26.__init__` 수정

```python
# 현재 v5: per_modal_decoders만 있음 (teacher + track_step 양쪽에 사용)
self.per_modal_decoders = nn.ModuleList([
    copy.deepcopy(sam_model.sam_mask_decoder) for _ in range(num_modalities)
])

# v6: per_modal_decoders (학습 전용) + shared decoder (추론용) 분리
# (1) Per-modal decoder: 학습 시 auxiliary CE + SQG target 생성용
self.per_modal_decoders = nn.ModuleList([
    copy.deepcopy(sam_model.sam_mask_decoder) for _ in range(num_modalities)
])
# (2) Shared inference decoder: track_step 추론 경로용
# sam_model.sam_mask_decoder를 그대로 유지 (self.sam.sam_mask_decoder)
# → 별도 선언 불필요, 기존 self.sam.sam_mask_decoder가 shared decoder 역할
```

핵심: `self.sam.sam_mask_decoder`는 per_modal_decoders에 deepcopy되지 않고 **원본 그대로 유지**. 이것이 shared inference decoder.

v5 대비 `__init__` 변경: `amf_mode` 관련 파라미터 정리. `tau_amf` 불필요 (UAMM과 동일 weight 사용).

##### 2. `sam_lora_image_encoder_seg.py` — `forward()` Phase 2.5 수정

```python
# 현재 v5 Phase 2.5: per-modal decoder로 teacher decode (no_grad)
with torch.no_grad():
    self._swap_decoder(i)  # per_modal_decoder[i]로 swap
    teacher_logits = self._teacher_decode_single(vision_feats[i], ...)

# v6 Phase 2.5: per-modal decoder로 직접 CE loss (grad 있음, no_grad 제거)
# _swap_decoder 대신 per_modal_decoder[i]를 직접 호출
per_modal_pred = self._auxiliary_decode_single(
    self.per_modal_decoders[i], vision_feats[i], vision_pos_embeds[i], feat_sizes[i]
)
# (1) Auxiliary CE loss 수집
aux_ce_loss = F.cross_entropy(per_modal_pred_resized, gt_safe, ignore_index=255)
aux_losses.append(aux_ce_loss)
# (2) CE map for SQG target (detach — SQG target은 gradient 차단)
with torch.no_grad():
    ce_map = F.cross_entropy(per_modal_pred_resized, gt_safe, reduction='none')
    ce_maps.append(ce_map)
```

`_auxiliary_decode_single` 새 메서드: `_teacher_decode_single`과 동일하되 `torch.no_grad()` 없이 실행. per_modal_decoder를 인자로 받아 해당 decoder의 forward 호출.

##### 3. `sam_lora_image_encoder_seg.py` — `forward()` Phase 3 수정

```python
# 현재 v5 Phase 3: per-modal decoder로 track_step
self._swap_decoder(frame_idx)  # per_modal_decoder[i]로 swap
output_step = self.sam.track_step(...)

# v6 Phase 3: shared decoder (self.sam.sam_mask_decoder 원본)로 track_step
# _swap_decoder 호출하지 않음 — sam.sam_mask_decoder가 이미 shared decoder
output_step = self.sam.track_step(...)
```

주의: Phase 1의 `_encode_single_modality`에서 conv_s0/s1이 decoder에 속하므로,
encoding 시에는 **shared decoder의 conv_s0/s1**을 사용 (모든 모달리티 공통).
v5처럼 모달리티별 conv_s0/s1을 쓰지 않음.

##### 4. `sam_lora_image_encoder_seg.py` — `forward()` Phase 4 (AMF) 수정

```python
# 현재 v5 Phase 4: output entropy 기반 AMF
prob = F.softmax(output[i], dim=1)
entropy = -(prob * (prob + 1e-8).log()).sum(dim=1, keepdim=True)
confidence = 1.0 - entropy / math.log(num_classes)
amf_norm = confidence / sum(confidence)

# v6 Phase 4: SQG quality map 기반 AMF (entropy 코드 전부 제거)
# q_uamm_norm은 Phase 3에서 이미 계산됨 — 그대로 재사용
# UAMM weight == AMF weight (동일한 SQG quality, tau 없음)
sqg_weight = q_uamm_norm  # (m, B, 1, H, W), Phase 3에서 계산된 것 재사용
# output 해상도에 맞춰 interpolate
for i in range(m):
    w_i = F.interpolate(sqg_weight[i], size=output[i].shape[-2:], mode='bilinear', align_corners=False)
    ...
m_output = sum(w_i * output[i] for i in range(m))
```

핵심: entropy 관련 코드 전부 삭제. `_last_entropy_maps` 버퍼도 제거 가능.

##### 5. `sam_lora_image_encoder_seg.py` — `forward()` return 수정

```python
# v6: gate_loss_data에 aux_ce_loss 추가
gate_loss_data = {
    'predicted_logits': quality_logits,
    'quality_target_dist': quality_target_dist,
    'ignore_mask': ignore_mask_fpn,
    'loss_type': 'kl',
    'aux_ce_losses': aux_losses,  # 신규: list of m scalar losses
}
```

##### 6. `train_sam2_lora_paper.py` — auxiliary CE loss 반영

```python
# 현재 v5: KL loss만 사용
if gate_loss_data is not None:
    kl_loss = compute_kl_loss(gate_loss_data)
    total_loss = ce_loss + kl_weight * kl_loss

# v6: KL loss + auxiliary CE loss
if gate_loss_data is not None:
    kl_loss = compute_kl_loss(gate_loss_data)
    aux_ce = sum(gate_loss_data['aux_ce_losses']) / len(gate_loss_data['aux_ce_losses'])
    total_loss = ce_loss + kl_weight * kl_loss + aux_ce_weight * aux_ce
```

##### 7. Config 추가

```yaml
MODEL:
  QUALITY_GATE:
    AUX_CE_WEIGHT: 0.5    # per-modal auxiliary CE loss 가중치
    AMF_MODE: sqg_quality  # v6: SQG quality map 기반 AMF (tau 없음, UAMM과 동일 weight)
```

##### 8. 추론 시 per_modal_decoder 미사용 확인

- `forward()`에서 `self.training`이 False일 때 Phase 2.5 전체 스킵 (현재 v5도 동일)
- Phase 3에서 `_swap_decoder` 호출 안 함 → shared decoder만 사용
- `save_lora_parameters` / `load_lora_parameters`: per_modal_decoder weights 저장/로드 유지 (학습 재개용)

##### 9. `_last_*` 버퍼 변경

- 기존 `_last_per_modal_outputs`: Phase 3의 track_step 결과 (shared decoder) 저장 — 변경 없음
- `_last_entropy_maps`: v6에서 AMF가 entropy 미사용이므로 **제거 가능** (또는 디버깅용 유지)
- `_last_amf_spatial`: SQG quality map 기반으로 변경 — `amf_weight` 저장
- per_modal_decoder의 auxiliary output: 별도 버퍼 `_last_aux_per_modal_outputs` 추가 가능 (선택)

**수정 필요 파일 요약**:
- `sam_lora_image_encoder_seg.py`: `LoRA_Sam_P26.__init__`, `_auxiliary_decode_single` 신규, `forward()` Phase 2.5/3 수정
- `train_sam2_lora_paper.py`: auxiliary CE loss 추가
- Config yaml: `AUX_CE_WEIGHT` 파라미터 추가

---

## Object Detection 확장 아키텍처 (설계 2026-03-19)

### 설계 원칙

- SAM2 Encoder + MoE LoRA + Memory Attention + FPN + UAMM/AMF **전체 재사용**
- Segmentation Head만 Detection Head로 교체
- P22 기반 권장 (fpn[0,1,2] 3레벨 → multi-scale detection에 필수)

### Forward 흐름 (Detection)

```
Phase 1: 모달리티별 인코딩 (P22 동일)
  for modal in [img, lidar, thermal]:
    backbone_feat = SAM2_encoder(modal)           # Hiera-B+ + SoftMoE_LoRA
    DeBA-FP(backbone_feat, fpn[0,1,2])            # multi-scale refinement
    memory_attention(backbone_feat, memory)        # cross-modal attention
    memory.append(backbone_feat)

Phase 2: Cross-Modal 가중치 (P22 동일)
  cross_weights = CrossModalFusionHead(fpn[0])    # (B, m)

Phase 3: UAMM + Memory Tracking (P22 동일)
  modulated_vision_feats = vision_feats * uamm_scores
  track_step(modulated_vision_feats, memory)

Phase 4: AMF — multi-scale fused features
  for level in [fpn0, fpn1, fpn2]:
    fused_level[i] = sum(amf_weights[:, j] * level_feat[j] for j in range(m))

Phase 5: Detection Head (신규)
  Option A — FCOS:
    for level in fused_levels:
      cls_score = cls_branch(level)     # (B, num_classes, H_l, W_l)
      bbox_pred = reg_branch(level)     # (B, 4, H_l, W_l)
      centerness = ctr_branch(level)    # (B, 1, H_l, W_l)
    → NMS → final detections

  Option B — DETR:
    object_queries (learnable, N개)
    for layer in decoder_layers:
      queries = cross_attn(queries, fused_features)
    box_pred = box_head(queries)        # (B, N, 4)
    cls_pred = cls_head(queries)        # (B, N, num_classes)
    → Hungarian matching → loss
```

### Phase 4 변경점 (Seg → Det)

| | Segmentation (현재) | Detection (확장) |
| --- | --- | --- |
| AMF 대상 | fpn[0]만 fusion | fpn[0,1,2] 전부 fusion |
| 출력 | single fused feature → 1x1 Conv | multi-scale fused features → Det Head |
| Phase 5 | argmax → per-pixel class | NMS 또는 Hungarian matching → bbox + class |

### Loss 구성

| Loss | 용도 | 참고 |
| --- | --- | --- |
| Focal Loss | classification (class imbalance 대응) | torchvision 제공 |
| L1 Loss | bbox regression | 표준 |
| GIoU Loss | bbox regression (scale-invariant) | torchvision.ops |
| (Optional) UAMM/AMF loss | P25 quality gating 사용 시 | 기존 BCE quality loss |

### MLE-SAM (CVPR 2026) 대비 차별점

| | MLE-SAM | MemorySAM-Det (제안) |
| --- | --- | --- |
| Task | Semantic Segmentation | Object Detection |
| Cross-Modal Fusion | GAP + top-k hard routing | Memory Attention + UAMM/AMF soft routing |
| Routing 레벨 | per-image (GAP) | per-pixel (P25) 또는 per-image (P9/P22) |
| FPN 활용 | 3레벨 독립 routing | 3레벨 DeBA-FP refined + unified routing |
| Missing modality | top-k 자동 처리 | 미지원 (향후 확장 가능) |
| Detection Head | 없음 (seg only) | FCOS 또는 DETR |

### 상태: 설계 완료 (구현 대기)
