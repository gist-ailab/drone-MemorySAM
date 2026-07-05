# P32 아키텍처 제안 모음 — 조건 적응형 라우팅 재설계 (Seg)

> 작성: 2026-07-05 (proposal 상태 — 전부 미구현). 사용자 지시: "P31까지와 아키텍처가 달라도 되니, 논리적으로 말이 되는 여러 컨셉의 모델 구조 proposal".
> 근거 문서: [16_failure_analysis_P28_P29.md](16_failure_analysis_P28_P29.md)(실패모드), [02_model_arch.md](02_model_arch.md)(P9~P31 상세), [12_novelty_and_related_work.md](12_novelty_and_related_work.md) + `research_vault/relatedworks/{41,47,50,55}`(선행연구 빈칸 검증), [20_p31_design_proposal.md](20_p31_design_proposal.md)(직전 제안).
> 목표 대비: 공식 목표 DELIVER val ≥66.51 / test ≥56.71 (현 P31 63.20/54.75), MUSES SOTA 79.72/79.49, MULTIAQUA(현 M 82.10).

---

## 0. 왜 지금까지의 라우팅이 실패했나 — 진단 종합 (설계의 출발점)

P9~P31 로그에서 **라우팅 실패는 서로 다른 4개의 근본 원인**으로 분해된다. 새 proposal은 각각을 어떤 기구로 깨는지 명시해야 한다.

| # | 근본 원인 | 증거 |
|---|-----------|------|
| **R1. 게이트 입력에 조건 정보가 없다** | CrossModalFusionHead = GAP→Linear→softmax: 공간 평균 + frozen encoder의 정규화로 조건(야간/비) 신호가 입력 단계에서 소실 → 상수 수렴(std≈0, 8개 변형 전부; ISSUE-003). SQG도 동일(PredQ std 0.05 vs TargetQ 0.40, 16 §B-2) | 02:1592, 03:53, 03:472-489 |
| **R2. 상수 출력이 loss의 지름길이다** | task CE만으로는 게이트가 상수여도 벌점이 없음(융합 뒤 feature가 보상). 강제로 adaptation시킨 P15는 Test 48.94로 최악. MoE는 zero-init `experts_b`→gate grad≈0→rich-get-richer(E1 dead, ISSUE-002). P12 입력조건화·P13 init 수정 모두 실패 | 02:82-83, 03:371-391, 02:413, 02:479 |
| **R3. reliability 신호 자체가 깨져 있다** | RBMA `1−H(softmax(D_i(f_i)))/logC`는 **모달의 정보량이 아니라 per-modal decoder의 용량**을 측정 → event/LiDAR처럼 decoder가 약한 모달은 항상 고엔트로피 = 항상 저신뢰. AUROC [img .77 / depth .62 / **event .30 / lidar .22**] = 우연보다 나쁨(anti-calibrated) | 16 §7 (02:9 인용) |
| **R4. soft 융합은 select를 못 한다** | 융합 가중 ≈ uniform [.27,.28,.23,.23]인데 실제 기여(drop-Δ) [8.4, 23.5, **0.02, 0.01**] → RGB+Depth 2-모달 퇴화(Mode C). soft softmax는 "약간 덜 준다"만 가능, "안 본다/골라 본다"가 불가 | 16 §Mode C |

**추가 제약 2개 (proposal의 정직한 스코프):**
- **Mode B 경고**: DELIVER의 day→test 갭은 야간/악천후가 아니라 **class-transfer 실패**(Wall 62→2, per-condition mIoU는 night .526/rain .561로 타이트). **조건 라우팅만으로 DELIVER test 갭은 안 풀린다는 실증(P29)이 이미 있음** → 라우팅 제안의 주 무대는 **Mode C(dead modality) 부활 + MULTIAQUA(주간 val/야간 test 실제 조건 갭) + MUSES(per-condition split)**. DELIVER 공식목표는 backbone unfreeze·타깃증강(P31 레버) 병행이 전제.
- **SAM3/텍스트 재해석**: 실제로 돌린 SAM3 실험은 **language가 아니라 RBMA 포팅(plain LoRA)**이었고, 부진(val ~16-24 plateau)의 로그상 원인은 **ViT single-scale 구조**(11, 19:27). "텍스트 기반 조건 인지가 무효"라는 증거는 아직 없다 — 다만 SAM3의 텍스트는 *객체 개념* 프롬프트용이라 *센서 열화* 서술에 그대로 쓰는 건 원래 부적합(아래 A2에서 재활용 방식 제안).

**설계 원칙 (모든 proposal 공통 요구사항)**
1. 조건/신뢰도 신호는 **구성상(by construction) 상수로 수렴할 수 없어야** 한다 — (a) training-free 통계이거나, (b) task loss가 아닌 **자체 supervision**으로 학습되거나, (c) 이산 선택이라 gradient 지름길이 없어야 함.
2. R3를 먼저 고치지 않으면 어떤 라우팅도 무의미 — **신호의 AUROC>0.5가 라우팅 설계보다 선행**.
3. 선행연구 점유 셀 회피: depth-GT conditioning(DGFusion) · inference-time CLIP(CAFuser) · supervised weather 분류(AW-MoE) · pooled-stat gate-input(MLE-SAM/DAMP) · load-stat logit bias(LFB) — vault 47/50/55 기준.

---

## 0.5. 코드 검증 — 각 제안이 얹힐 실제 seam (직접 확인, 2026-07-05)

`sam_lola_utils.py` / `sam_lora_image_encoder_seg.py`를 직접 열어 위 진단과 구현 지점을 확증했다. 모든 proposal은 아래 **격리된 seam 하나**를 override하면 되며, 기존 P28~P31은 config-gate로 byte-identical 유지 가능(P29/P30이 쓴 패턴 그대로).

| 확인 대상 | 코드 위치 | 확인 결과 (진단 확증) | 관련 proposal seam |
|-----------|-----------|----------------------|--------------------|
| **RBMA 신호 계산** | `sam_lora_image_encoder_seg.py:8028` `LoRA_Sam_P28._compute_bias_source` | `p=softmax(aux_logits)` → `ent=−Σp·log p/logC` → `rel=1−ent` → 모달간 center. **정확히 self-entropy, per-modal 독립**(R3 확증). `torch.no_grad()`+λ만 학습 | **P32-B**: 이 메서드만 override. 루프 안에서 이미 per-modal `p_i`를 다 계산하므로 leave-one-out 합의 추가가 자연스러움 |
| **출력 융합 라우터** | `sam_lora_image_encoder_seg.py:8197` `LoRA_Sam_P30._fuse_outputs` | anchor `rel=1−H(softmax(output_i))/logC` (또 self-entropy). `w=softmax_modality(...)` → `m_feat=Σ w_i·feat_i`. **soft 가중, hard select 불가**(R4 확증) | **P32-C/D**: 이 메서드 override (P30이 P26 훅을 override한 그대로). C=여기에 Gumbel-top-k/null-token, D=`w` 대신 cluster-table lookup |
| **SoftMoE 게이트** | `sam_lola_utils.py:655,700` `SoftMoE_LoRA_Layer.gate/forward` | `self.gate=nn.Linear(in_features,num_experts)`, 입력=per-token `x`. 조건은 `set_condition()`→FiLM `logits*(1+γ)+β`로 **공유 게이트 위 modulation**(line 715-722) | **P32-A/E**: `set_condition`으로 z_c 주입(경로 이미 존재). E는 dual-view로 gate 출력에 contrastive |
| **P29 SDC 조건화** | `sam_lola_utils.py:754` `SelfDerivedCondition` + `seg.py:8129` `P29.forward` | RGB 채널통계(mean⊕std)→proj→prototype **soft-assign**(K=6). z_c를 gate에 FiLM. **여전히 공유 gate 함수 유지 = R2 미탈출**(Test 무이득과 정합) | **P32-D**: 이 soft-assign을 hard cluster + per-cluster table로 교체(게이트 함수 자체 제거) / **P32-A**: z_c를 θ로 supervise |
| **라우터 붕괴 방지 reg** | `sam_lola_utils.py:810` `ReliabilityAnchoredRouter`, `reg_mode` | P30 'diversity'=per-pixel mixing entropy(=**uniform 쪽으로 밈**, 자기모순 확인), P31 'decisive'=marginal−pixel entropy | **P32-E**: 여기에 condition-contrastive term 추가 가능. C의 hard-select는 이 reg를 대체 |
| **memory-attn bias 주입** | (P27 계보) `RoPEAttention._p27_attn_bias`, `memory_attention` forward_pre_hook | bias가 SDPA `attn_mask`로 pre-softmax 가산. `_compute_bias_source` 반환값이 이 자리로 감 | **P32-B/C** 모두 이 주입점 재사용(신호만 교체) — RBMA 노벨티 배관 계승 |

**핵심 함의 3가지**:
1. **P32-B가 가장 저비용·저위험**: `_compute_bias_source` 한 메서드 override, 학습 파라미터 λ만(기존과 동일), 주입 배관 그대로. 코드가 이미 per-modal `p_i`를 stack하므로 corroboration은 몇 줄 추가.
2. **self-entropy가 두 곳(`_compute_bias_source`의 aux_logits, `_fuse_outputs`의 output)에서 독립 계산됨** — corroboration 교체 시 두 곳 다 적용 가능하나 **우선순위는 memory-attn bias**(RBMA 헤드라인). fusion 쪽은 P32-C가 어차피 재작성.
3. **`_fuse_outputs`는 이미 grad-attached per-modal `output`/`feat`를 받음** → P32-C(pruning)·D(table)가 새 forward 배관 없이 이 훅만 교체하면 됨. P30_Det이 backbone fuse를 우회하는 것도 확인(det는 별도 경로).

---

## 1. 제안 요약표

| ID | 이름 | 한 줄 컨셉 | 직격하는 원인 | 학습 필요 | 무학습 사전검증 가능 |
|----|------|-----------|--------------|----------|:---:|
| **P32-A** | **PhysCond** | 증강 파라미터(우리가 이미 아는 열화 GT)로 **자가지도 조건 인코더**를 학습해 라우팅 구동 | R1, R2 | 조건 인코더(자체 loss) | △ |
| **P32-B** | **CoRB** | reliability 신호를 self-entropy → **cross-modal 상호검증(corroboration)** 으로 교체 (training-free 유지) | **R3** | ✕ (신호는 무학습) | **✅ 기존 ckpt로 AUROC 측정** |
| **P32-C** | **PruneMem** | memory attention을 soft bias → **reliability 기반 token pruning(hard select)** + null token + 확률적 modality dropout | **R4**, R2 | pruning 자체는 무학습 | **✅ 기존 ckpt inference sweep** |
| **P32-D** | **ProtoTable** | 라우팅을 "학습된 게이트 함수" → **"training-free 클러스터 → per-cluster 학습 테이블 lookup"** 으로 교체 | R1, R2 | 테이블(K×m 파라미터) | △ |
| **P32-E** | **CCR** | 같은 장면·다른 증강 조건이면 라우팅이 **달라야 한다는 contrastive 정칙화** (A~D 어디에나 부착) | R2 | loss만 추가 | ✕ |
| (A2) | CLIP-distill | CLIP zero-shot 조건 점수를 **학습 시에만** teacher로 증류(추론은 CLIP-free) — A의 변형 | R1 | 증류 loss | △ |

권장 조합·순서는 §7.

---

## 2. P32-A: PhysCond — 증강 파라미터 자가지도 조건 인코더

**컨셉 한 줄**: DGFusion은 depth-GT로, CAFuser는 CLIP 텍스트로, AW-MoE는 사람 라벨로 "환경"을 안다. **우리는 이미 physaug/NIGHT_AUG가 열화를 '생성'하므로, 그 파라미터(밝기 스케일·노이즈 강도·LiDAR dropout률 등)가 공짜 조건 GT다** — 이걸로 조건 인코더를 직접 지도학습한다.

**구조**
```
train:  x_aug = Aug(x; θ)          θ = (brightness, gamma, noise σ, rain, lidar-drop, …)  ← 이미 코드가 샘플링
        z_c = ψ(stats(x_aug), f_shallow)        # ψ: 작은 MLP/conv, 모달별 저수준 통계 + 얕은(비정규화) encoder feat
        L_cond = ‖g(z_c) − θ‖²  (+ InfoNCE: 같은 θ끼리 당김)   # task loss와 독립된 자체 loss
route:  memory-attn bias  b_i = MLP_b(z_c)[i]   그리고/또는  LoRA gate FiLM: g̃ = γ(z_c)∘g + β(z_c)
```
- 라우터 입력은 **z_c만** (content feature 차단) → 데이터셋 prior(상수)를 학습할 수 없음.
- ψ는 L_cond로 직접 지도되므로 **출력 분산이 supervision에 의해 강제**됨 — R1(조건 부재)·R2(상수 지름길) 동시 차단. P25 SQG가 무너진 이유(frozen feat에서 quality를 '추측'해야 했음)와 정반대로, 여기서는 정답 θ를 알고 가르친다.
- 추론: 실제 야간/비 영상이 학습된 열화 공간으로 사상됨(physaug가 야간 시뮬을 위해 설계됐다는 전제 활용).

**왜 붕괴 안 하는가**: 게이트가 아니라 **신호를 학습**하고, 신호는 자체 GT(θ)를 가진다. 상수 z_c는 L_cond가 즉시 벌점.

**선행연구 차별**: AW-MoE(수동 weather 라벨 CE)와 달리 라벨 0; DAMP(hand-crafted 통계를 gate 입력에 concat)와 달리 **학습된 조건 공간 + FiLM/bias 주입**; MLE-SAM(pooled feat stat = modality 신호)과 달리 **condition 신호**. "augmentation 파라미터를 조건 supervision으로 쓰는 멀티모달 seg 라우팅" 셀은 vault 50 기준 미점유(단, restoration 쪽 degradation estimation 계열 lit-check 필요 — §8).

**리스크**: ① sim-to-real 갭 — 증강이 못 만드는 열화(진짜 야간의 색분포)는 조건 공간 밖. ② 조건 축이 증강 축에 한정. → 완화: DELIVER/MUSES는 **메타데이터에 실제 condition 라벨**(night/rain/fog split)이 있으므로 θ-지도와 병행해 val에서 z_c의 조건 분리도(선형 probe acc)를 무료로 검증 가능.

**최소 검증 실험**: (1일) ψ만 단독 학습 → val에서 z_c로 day/night/rain 선형 분류 acc ≥90% 확인 → 통과 시에만 라우팅 연결. MULTIAQUA가 주 타깃(val 주간/test 야간 = 진짜 조건 갭).

**변형 A2 (CLIP-distill)**: ψ의 teacher로 θ 대신(또는 병행) CLIP zero-shot 조건 점수(`"a photo at night"`, `"heavy rain"` 등 vs RGB)를 **학습 시에만** 증류. CAFuser와의 차별 = **추론 시 CLIP-free**(경량) + text가 fusion을 직접 변조하지 않음. SAM3에서 못 살린 "언어로 환경 인지" 아이디어의 올바른 재활용 위치.

---

## 3. P32-B: CoRB — Corroboration-Biased Memory Attention (R3 직격, 최우선)

**컨셉 한 줄**: "이 모달이 스스로 얼마나 자신 있나(self-entropy)" 대신 **"이 모달의 주장이 다른 모달들의 합의와 얼마나 상호검증(corroborate)되나"** 를 training-free reliability로 쓴다. RBMA의 노벨티(무학습 신호 + logit additive bias)는 그대로 유지하고 **신호의 의미만 교체**.

**왜 이게 근본 수리인가**: R3의 본질 — self-entropy는 decoder 용량과 얽혀(confounded) 있다. event decoder는 *어디서나* 불확실 → 신호가 열화가 아니라 모달 정체성을 인코딩(그래서 AUROC .30). 반면 **일치도는 용량과 분리**된다: 약한 decoder라도 "그 위치에서 다수 모달과 같은 클래스를 가리키는가"는 정보 유무를 반영. P31 Seg-B가 consistency를 *2차* bias로 스케치했지만(02:13), 여기서는 **corroboration을 1차 신호로 승격**하고 self-entropy를 버리는 것이 차이.

**구조 (training-free, 기존 RBMA 배관 재사용)**
```
p_i = softmax(D_i(f_i))                          # per-modal standalone posterior (기존 그대로)
합의: p̄_{−i} = Σ_{j≠i} w_j·p_j / Σ w_j           # leave-one-out, w_j = 1차 근사로 uniform → 2-pass에서 corroboration으로 재가중(EM 1스텝)
corroboration_i(x) = 1 − D_B(p_i(x), p̄_{−i}(x))   # Bhattacharyya/JS, per-pixel ∈[0,1]
B_i = centered(corroboration_i)                    # 기존 _p27_attn_bias 자리에 그대로 주입 (λ 학습 유지)
```
- **유일정보 보호 장치(중요)**: 합의 기반의 알려진 위험 = 야간에 thermal만 보는 물체는 "불일치"로 벌점. → **veto 조건**: `conf_i 높음 ∧ 불일치 ∧ 나머지 전원 conf 낮음`이면 벌점 면제(오히려 boost). 즉 "다수가 아무것도 못 보는 곳에서 혼자 자신 있는 모달"은 살린다. 이 게이팅도 training-free(임계값 2개).
- P31 Seg-A(온도 보정 + calibration loss)와 직교 — corroboration에도 per-modal 온도를 그대로 적용 가능.

**왜 붕괴 안 하는가**: 학습 파라미터가 λ뿐(기존 RBMA와 동일). 신호는 입력에서 계산되므로 상수 수렴 자체가 성립 안 함.

**선행연구 차별**: RSGMamba의 consistency gate `g_c`는 **learned MLP**(vault 47) — 우리는 무학습 통계. DGFusion과는 신호(합의 vs depth-GT)·기구(logit bias vs token conditioning) 양축 상이. "training-free cross-modal corroboration을 attention logit bias로" 셀은 vault 42/47 기준 미점유. 논문 서사도 강력: *"신뢰도는 자기확신이 아니라 상호검증이다"* — AUROC .30→(목표).7 반전이 Figure 1감.

**리스크**: ① RGB+Depth가 담합해 낮에 event/LiDAR를 계속 누를 수 있음(단 지금도 Δ≈0이라 하방 없음). ② 4-모달 전원이 특정 클래스에서 같이 틀리면(공통 실패) 상호검증 무력 — 이건 어떤 reliability로도 안 됨.

**최소 검증 실험 (★무학습, 최우선 실행 후보)**: 기존 P28/P29/P31 ckpt에서 DELIVER val 1-pass — per-modal `p_i` 저장 → corroboration vs self-entropy의 **correctness-AUROC 비교** (16 §7과 동일 프로토콜). **event/LiDAR AUROC가 0.5를 넘으면 신호 수리 입증** → 그때만 학습 재개. 비용: eval 1회 + 스크립트(tools/에 `eval_reliability_auroc.py`로 재사용화).

---

## 4. P32-C: PruneMem — Reliability-Pruned Memory Attention (R4 직격)

**컨셉 한 줄**: soft bias는 "덜 본다"까지만 가능하고 게이트는 uniform으로 도망간다(R4). **memory attention에 들어가는 모달리티 토큰을 위치별로 top-k만 남기고 이산적으로 프루닝**해서 모델이 *고르도록 강제*한다. 사용자가 말한 "token pruning을 MemorySAM에 넣는다"의 구체화이자, RBMA의 자연 극한(bias→−∞ = pruning)이라 서사가 이어진다.

**구조**
```
score_i(x) = corroboration_i(x)                    # P32-B 신호 재사용 (보정된 신호가 전제!)
train:  위치 x마다 Gumbel-top-k(score, k=2/4모달) → 선택 토큰만 K,V에 참여 (straight-through)
        + 확률적 modality dropout: p_drop ∝ score → 높은 score의 RGB도 가끔 강제 탈락
inference: deterministic top-k (또는 임계값 τ 이하 컷)
+ null token: 위치별 학습된 "abstain" 토큰 1개를 항상 후보에 포함
```
- **왜 hard가 R2/R4를 깨나**: 이산 선택은 "모두에게 0.25씩"이라는 지름길이 **존재하지 않는 해공간**. gradient가 선택된 토큰으로만 흐름 → 모달 경로가 특화됨(hard-MoE의 고전적 anti-collapse 성질).
- **modality dropout이 Mode C를 직격**: RGB 토큰이 확률적으로 제거되면 모델은 event/LiDAR에서 정보를 뽑도록 *학습을 강요*당함 — drop-Δ [0.02, 0.01]의 직접 처방. (MAGIC/AnySeg의 modality-dropout 교훈을 memory-token 수준으로 이식.)
- **null token의 역할**: softmax는 합=1이라 모든 모달이 나쁜 위치에서도 attention 질량이 어딘가로 감 → 쓰레기 흡수용 abstain 토큰이 이를 받아냄(soft RBMA에는 없던 안전판).
- 부수 효과: 메모리 토큰 수 4→k로 감소 = **memory attention 연산량 절감** (효율 selling point, B200 아닌 서버들에 실익).

**선행연구 차별**: token pruning 문헌은 단일모달 *효율* 목적(EViT 등). "**reliability 기반 cross-modal memory-token pruning**"은 vault 46/55 스캔 기준 미점유. M⁴-SAM은 encoder MoE(memory 불개입), SAM4D는 pruning 없음.

**리스크**: ① k 선택 민감(k=1은 과격, k=3은 무의미) — τ-임계 변형으로 완화. ② Gumbel ST의 학습 불안정 — warmup 동안 soft bias(기존 RBMA)로 시작해 온도 annealing으로 hard화하는 스케줄 권장. ③ 신호가 깨진 채(P32-B 이전) 적용하면 event/LiDAR를 영구 프루닝 — **반드시 B 이후**.

**최소 검증 실험 (★무학습)**: 기존 P31 ckpt에서 inference-time pruning sweep — corroboration 순위 기반 top-k∈{1,2,3} / τ-컷으로 DELIVER val 측정. **k=3(최저 1개 모달 제거)에서 val이 안 떨어지면** "프루닝 여지 존재" 입증 + k별 곡선이 그대로 논문 Figure.

---

## 5. P32-D: ProtoTable — 조건 클러스터 lookup 라우팅 (게이트 함수의 제거)

**컨셉 한 줄**: 상수 수렴의 주범은 "task loss로 학습되는 **공유 게이트 함수**"라는 형식 자체다(R2). 게이트 함수를 없애고 — **조건 할당은 training-free 클러스터링, 라우팅 가중은 클러스터별 독립 파라미터 테이블 lookup**으로 바꾼다.

**구조**
```
offline: 학습셋 전체에서 조건 서술자 s(x) 계산(P32-A의 z_c, 또는 무학습 저수준 통계: 휘도 히스토그램+노이즈 추정+LiDAR 밀도)
         k-means → K개 조건 프로토타입 {c_k} (K≈4-8, training-free)
train:   k*(x) = argmin_k ‖s(x)−c_k‖ (hard, no-grad) → 라우팅 가중 W[k*] ∈ ℝ^{m}(또는 ℝ^{m×C}) lookup
         W[k*]만 gradient 수신 (다른 클러스터 테이블은 그날 안 배움)
infer:   같은 lookup. W를 그대로 출력하면 per-cluster 모달 프로파일 해석/시각화 가능
```
- **왜 붕괴 안 하는가**: 클러스터 할당은 gradient 밖(무학습), 테이블들은 **서로 다른 데이터 부분집합으로만 학습**되므로 공동 수렴(co-collapse)할 채널이 없음. 전 테이블이 우연히 같은 값이 되는 것만이 "상수"인데, inter-cluster decorrelation reg 한 줄로 차단.
- P29 SDC와의 관계: SDC는 "조건 latent → FiLM → **여전히 공유 게이트**"라 R2를 못 벗어났다(Test 무이득 실증). ProtoTable은 게이트 함수 자체를 제거하는 것이 요점.
- 파라미터 극소(K×m ~ 수백 개), 해석성 최고("클러스터 3=야간: thermal .45 / RGB .15" 식으로 표 한 장).

**선행연구 차별**: MoCLE(텍스트 클러스터→gate *입력*), c-BTM(문서 클러스터→LM 앙상블), ClusIR(learnable 클러스터, restoration) — "training-free **시각 조건** 클러스터 → **per-cluster 라우팅 테이블**, 멀티모달 dense seg" 셀은 vault 50 기준 미점유. DAMP는 통계를 학습 게이트의 입력으로 concat(함수 유지) — 우리는 함수 제거.

**리스크**: ① 클러스터 경계에서 라우팅 불연속 — soft-assign(top-2 보간)으로 완화. ② K가 실제 조건 다양성과 불일치 — DELIVER 조건 라벨로 클러스터 순도(purity) 사전 측정 가능(무학습). ③ 표현력이 테이블 수준으로 제한 — 의도된 트레이드오프(표현력↓ 대신 붕괴 불가능성↑). MULTIAQUA/MUSES처럼 조건이 이산적인 벤치마크에 적합.

**최소 검증 실험**: s(x) 클러스터링 → DELIVER 조건 라벨 대비 purity ≥0.8 확인(무학습, 반나절) → 통과 시 P31 위에 `_fuse_outputs`만 교체해 단기 학습.

---

## 6. P32-E: CCR — Condition-Contrastive Router 정칙화 (부착형)

**컨셉 한 줄**: 어떤 학습형 라우터가 남아 있든(P31 router, A의 MLP_b, D의 soft-assign), **"같은 장면·다른 증강 조건 → 라우팅 출력이 달라야 하고, 같은 조건 → 비슷해야 한다"** 는 contrastive loss를 부착해 상수 해를 loss 지형에서 제거한다.

```
x, x' = Aug(x; θ₁), Aug(x; θ₂)      # 같은 원본, 다른 조건 (physaug가 공짜 생성)
L_ccr = −sim(r(x), r(x̃)|θ 유사) + sim(r(x), r(x')|θ 상이)    # r = 라우터 출력 (InfoNCE 형)
(+ 선택) L_MI = −I(routing; condition-bin)                    # Mod-Squad의 task↔expert MI를 condition↔modality로 전치 (vault 50 개선방향 그대로)
```
- **왜 유효한가**: R2의 "상수 = 무벌점"을 정면 제거 — 상수 라우터는 L_ccr이 최대. P15의 실패(무조건 분산 강제 → test 붕괴)와 달리, **조건이 다를 때만** 다르라고 요구하므로 과잉 adaptation을 강제하지 않음.
- 단독 proposal이 아니라 **A·D·P31 router에 얹는 정칙화**. 구현 = loss 함수 하나 + 배치 내 dual-view 샘플링.
- 위치: 게이트 붕괴 계보(ISSUE-002/015)의 직접 처방이라 ablation 가치 높음(±L_ccr로 router 분산·AUROC·mIoU 3종 비교).

**리스크**: dual-view로 배치 실효 크기 절반(B200 VRAM 여유로 흡수 가능) / θ 유사도 bin 설계 필요.

---

## 7. 권장 조합과 실행 순서 (로드맵)

원칙: **무학습 진단 먼저(B·C·D는 기존 ckpt로 사전 검증 가능)** → 신호 수리(B) 없이는 어떤 라우팅(C·D)도 착수 금지.

```
Phase 0  (무학습, ~2-3일, P31 ckpt 재사용)
  ① corroboration vs self-entropy AUROC (P32-B 판정)   ← 최우선. event/LiDAR >0.5 반전 여부
  ② inference-time pruning sweep k∈{1,2,3} (P32-C 여지 측정)
  ③ 조건 서술자 클러스터 purity (P32-D/A 타당성)
Phase 1  P32-B 학습 = "RBMA v2" (P31 Seg-A 온도보정 유지, 신호만 corroboration 교체)
         → 헤드라인 후보: training-free 신호의 의미 전환(자기확신→상호검증), 기존 노벨티 서사 계승
Phase 2  + P32-C (soft→hard 스케줄, modality dropout) → Mode C(event/LiDAR 부활) 직격, drop-Δ 재측정으로 판정
Phase 3  + P32-A(또는 D) → MULTIAQUA/MUSES의 실제 조건 갭 공략 (DELIVER Mode B에는 기대 걸지 않음)
상시    P32-E는 Phase 1부터 남아있는 모든 학습형 라우터에 부착 (ablation 확보)
```

**벤치마크별 기대 기여**: DELIVER = Phase 1-2(Mode C 부활분) + P31 기존 레버(backbone unfreeze·Seg-C) 병행으로 66.51/56.71 도전 / MUSES = 조건 split이 명시적이라 Phase 3의 주 무대(벤치마크 셋업 선행 필요) / MULTIAQUA = Phase 3 직격(주간 val→야간 test).

**논문 서사(합치면)**: *"Reliability는 (1) 자기확신이 아니라 상호검증으로 측정해야 하고(B), (2) soft 가중이 아니라 이산 선택으로 작용해야 하며(C), (3) 조건 인지는 외부 신호(depth-GT/CLIP/라벨) 없이 열화 생성 과정 자체에서 배울 수 있다(A)"* — 세 축 모두 DGFusion/CAFuser/AW-MoE가 점유하지 않은 셀.

---

## 8. Lit-check TODO (proposal 채택 전 필수 확인, 12번 문서 §4에 병합할 것)

1. **(B)** training-free cross-modal agreement/corroboration을 fusion 가중이나 attention에 쓴 선행 — TTA/co-training 계열(pseudo-label agreement)과의 구분 논리 포함. RSGMamba(learned g_c)와 한 문장 차별 필수.
2. **(C)** reliability/uncertainty 기반 **token pruning**이 멀티모달 fusion에서 쓰인 전례 — 효율 목적 pruning(EViT/ToMe)과 구분. + Gumbel-top-k modality selection 계열(DynamicMoE) 스캔.
3. **(A)** augmentation-parameter regression을 condition supervision으로 쓴 전례 — 저수준 restoration(degradation estimation)엔 존재 가능성 높음 → "멀티모달 seg 라우팅 구동" 으로 스코프 한정 필요(DAMP 교훈: 전칭 부정 금지).
4. **(D)** training-free 클러스터 + per-cluster 파라미터 테이블 라우팅 — c-BTM/ClusIR과의 차별 문장화.
5. **(E)** routing-output contrastive 정칙화 전례 (MoE load-balance와 목적 구분: balance가 아니라 condition-sensitivity).

---

## 변경 이력
- 2026-07-05: 최초 작성 (P32-A~E 5개 컨셉 + A2 변형, Phase 0 무학습 검증 계획 포함). 상태 = **proposal (전부 미구현)**.
