---
legacy_id: 02
legacy_file: 02_model_arch.md
moved: 2026-07-08
---

# 모델 아키텍처 상세 (Model Architecture Details)

> 최종 업데이트: 2026-08-04

## P47-2 — UniBal (Uni-modal Balance): 모달별 독립 aux head + uni-modal CE (2026-08-04)

**상태**: **구현 완료 (학습 대기)**. 제안 = [decisions/2026-08-03-p47-mub-muses-proposal.md](../decisions/2026-08-03-p47-mub-muses-proposal.md) §3 **D-2**(문서 표기; 네이밍 규칙 변경으로 코드·config는 `P47_2`/`p47_2`). Base = **P39.1-rank 4모달 seed2 동결**(val 82.35 완주분 — gated_mlp trunk + VICReg + P36 router + M2F, 하이퍼 무변경). 파일: `semseg/models/reliadino/p47.py`(신규), `model.py`·`train_reliadino.py`(배선), `tools/smoke_p47.py`(신규 CPU 스모크). config: `configs/hpca100-muses_rgbelr_P47_2_unibal_4modal.yaml`(MUSES **4모달** img/lidar/event/radar, EPOCHS 300).

**동기(우리 실측)**: **우리 모델에서** 모달을 더할수록 성능이 떨어진다 — 4모달 val 82.35 / 공식 test 79.571 < 3모달 82.62 / 79.788 (drop-radar +0.13). ⚠️ 리더보드 역상관은 방법론 교란으로 **2026-08-04 철회**(통제 ablation은 단조 증가 — CAFuser Table IX: RGB 55.7→+L 58.7→+R 59.3→+E 59.7). 유효한 근거는 **within-method 실측뿐**. 문헌 기제 = **modality laziness / greedy joint learning**(융합 손실만으로 학습하면 지배 모달의 uni-modal feature가 under-optimize; 2305.01233 UMT · 1905.12681 Gradient-Blending · 2202.05306 · 2203.12221 이론증명). 자체 확증 = P46-C3(MUSES)의 손해가 **clear/day(RGB 주도 조건)에 집중**(val Δclear −1.72 / Δday −1.29, fog +0.16 / rain +0.21; 공식 test도 day −1.15, fog −0.07) → RGB 본류 표현력이 병목이라는 직접 증거. SOTA 격차의 실체도 clear −5.85 / day −4.37(night는 −2.69로 최소, fog는 +4.86 전체 1위).

**기제 (학습 전용·내부신호만·주손실 직결 = 키1, zero-init 잔차 아님)**: 각 모달의 encoder(frozen ViT + per-modal LoRA) 출력 `feats[i]`(stride-16)에 **모달마다 독립인** 경량 head(GroupNorm → 1×1 conv → K)를 달고 **동일 GT로 CE**. `aux['p47_2_uni']`로 λ_u pre-scale 후 반환 → trainer가 주 손실에 그대로 합산.

**🔴 기존 `FUSION.AUX_CE_WEIGHT`(aux_ce)와 무엇이 다른가** — base P39.1에 이미 per-modal aux CE가 **있다**(`fusion.aux_decoders[i](feats[i])`의 CE를 모달 평균해 0.5 가중). 그럼에도 P47-2가 no-op이 아닌 이유 3가지:

| | 기존 `aux_ce` | **P47-2** |
|---|---|---|
| head의 목적 | 다목적 — 그 logit이 `rel_cal`/corroboration/consistency bias·router anchor·`rbma_cal_loss`·P44 mutual-KL의 **신호원**이다. "정확해져라"와 "잘 보정된 신뢰도 추정기가 돼라"의 타협점으로 최적화됨 | **uni-modal 정확도만** 목표. 어떤 신뢰도/게이트 경로에도 연결되지 않음 |
| 모달별 가중 | 모달 평균 고정 → 4모달이면 모달당 실효 0.5/4=0.125, "RGB에만 더 걸어라" 표현 불가 | `MODALS`(all/`['img']`/인덱스) × `LAMBDA_U` — §1 진단이 지목하는 **RGB 본류**에 직접 레버 |
| OGM-GE | per-modal 성능 추정치를 노출하지 않아 결선 불가 | `last_acc[m]`을 내보내 gradient 변조 가능 |

**선택 토글 `OGM_GE` (기본 off, 2203.15332)**: per-modal uni-modal 정확도 s_m으로 ρ_m = s_m / mean_{j≠m} s_j, k_m = clamp(1 − tanh(α·relu(ρ_m−1)), MIN_K, 1)을 만들어 **앞서 가는 모달의 자기 LoRA 슬라이스 gradient만** 감쇠(ρ≤1이면 k=1 = 무개입). 이 리포는 모달별 LoRA를 하나의 텐서 0번 축에 쌓으므로(`MultiModalLoRAQKV.a_q/b_q/a_v/b_v` = (M,…)) `p.grad[m] *= k_m`이 정확히 "모달 m의 인코더 gradient"다. 공유·후단 파라미터(fusion/FPN/head/trunk_exp)는 건드리지 않는다. `GE_NOISE>0`이면 원논문의 gradient Gaussian noise도 적용(기본 0).

**🔴 DDP 계약**:
1. **추가 forward 없음** — 같은 forward의 `feats[i]`를 재사용하므로 P46-C2/C3에서 문제였던 iteration당 2-forward(broadcast_buffers in-place 사망 / unused-param 집합 불일치, ISSUE-028)가 **구조적으로 발생하지 않는다**. `broadcast_buffers` 변경 불필요.
2. **warmup 구간의 head는 unused parameter**가 된다 → trainer가 항상 켜는 `find_unused_parameters=True`가 처리(스모크 G가 `WARMUP_EP=5`로 실제 재현·통과).
3. **OGM-GE는 collective 1회 추가**. gradient는 backward 시점에 이미 all-reduce되므로, rank마다 자기 배치로 잰 s_m으로 다른 k를 곱하면 **rank 간 파라미터가 갈라진다**. 그래서 k 계산 전에 s를 all_reduce(mean)한다 — optimizer step마다 **전 rank가 대칭으로** 1회(크기 M). 2026-07-16 NCCL 데드락은 rank0 단독 collective가 원인이라 해당 없음. AMP는 상수배와 교환 가능해 `unscale_` 불요. P44 MMPareto와는 동시 사용 금지(둘 다 step 직전 `p.grad` 재작성) — trainer가 `RuntimeError`.

**추론 불변**: head는 `self.training and gt_mask is not None`에서만 호출되고 logit에 아무것도 더하지 않는다 → `model.eval()` 출력은 P39.1과 **완전 동일**. 스모크 C가 all-on/img-only 두 경우 모두 `|Δ|max = 0`으로 검증(+ 신규 state_dict 키가 `p47_2.*`뿐임도 확인). **DELIVER 무영향**: DELIVER config에 `MODEL.P47_2` 키가 없어 `ENABLE=False` → `self.p47_2 is None` → 학습 forward의 다른 손실 항까지 전부 불변(스모크 E). 모듈 생성은 `__init__` **최말단**이라 off일 때 init RNG 스트림도 안 건드린다(seed 재현 보존).

**메모리(실측, autograd saved-tensor 계측; dim 1024·1024²·4모달·BS1·bf16)**: HEAD=linear/GT_DIV=4 → **+51.7 MiB/스텝** + params 336 KiB(+AdamW state 1.0 MiB). GT_DIV=16이면 +33.4 MiB, HEAD=conv1x1이면 +69.6 MiB. A100 40GB BS1 기준 0.13% — BS 조정 불필요. ⚠️ head 앞 정규화에 `encoder.LayerNorm2d`(파이썬으로 편 elementwise 체인)를 쓰면 모달당 full-size 중간텐서 3장이 그래프에 남아 증분이 2배가 된다 → `nn.GroupNorm`(융합 op, 리포의 AuxDecoder/FPNSegHead 관례) 사용.

**검증**: `tools/smoke_p47.py` (CPU, tiny ViT, 4모달) 전항목 PASS — ① off / on(all) / on(img only) 3케이스 1-step fwd+bwd ② **키1**: uni-aux 손실 **단독 backward** 시 per-modal LoRA `b_q/b_v` **모달 슬라이스별** grad — all이면 4모달 전부 >0, img-only면 img만 >0이고 나머지 **정확히 0**(모달별 독립의 실증) ③ eval 등가성 |Δ|=0 + 신규 키 전부 `p47_2.*` ④ head 파라미터 공유 없음 ⑤ 부수효과 없음(다른 aux 손실 값 불변 + `encoder.forward` 호출 4회로 동일 = 추가 forward 없음) ⑥ OGM-GE가 앞선 모달만 k<1로 감쇠하고 실제 grad 비율이 k와 일치 ⑦ `--ddp`(gloo 2-proc) warmup 미달 스텝 포함 rank 간 gradient·OGM k 완전 일치. **실데이터 학습 미기동**. ⚠️ LoRA `a_q`는 `b`가 zero-init이라 step 0 grad가 정의상 0 → 판정은 `b_q/b_v`로(smoke_p46과 동일 규약).

**판정 게이트(사전등록, 제안서 §4 — 4모달 기준)**: Primary **MUSES 4모달 val ≥ 82.62**(= 4모달이 3모달 기록을 넘는 것). Stretch val ≥ 83.0. Secondary Codabench test ≥ 79.788. 🔴 falsifiable = **modality balance 적용 시 4모달 > 3모달로 역전** — 실패 시 "radar는 실제로 정보가 없다"로 확정하고 3모달 회귀. 부가 = drop-radar dMIoU +0.13 → ≥+0.5. ep30 조기 kill = 4모달 base 동일 ep 대비 −1.0 이하. ep30 즉검(무음 no-op 검출) = 로그 `[P47-2] per-modal acc`가 **모달별로 갈라지는가**(전부 붙어 있으면 압력 미형성 → λ_u 상향), `img`만 홀로 치솟으면 RGB 지배 잔존.

**⚠️ D-1과 분리**: 이 config는 `DATASET.PROJ_DIR`을 **넣지 않는다**(= SDK 기본 `projected_to_rgb`). D-1(LiDAR 투영 밀도화 `projected_to_rgb_dgf`)은 별도 단독 실험이고, 합본(D-1+D-2)은 두 단독 결과가 나온 뒤 별도 config로 만든다.

**노벨티(정직)**: uni-modal aux loss(UMT/Gradient-Blending)·OGM-GE 모두 **선행 존재 → 기법 단위 노벨티 아님**. 차별 축은 조합 ① frozen-VFM + per-modal LoRA 위에서의 modality balance(선행은 end-to-end fine-tune 전제) ② 내부신호만(조건 라벨·CLIP text·GT-depth 배제) ③ 진단주도(리더보드 모달수↔순위 역상관 + 우리 clear/day 손실 집중을 직접 표적). "first X" 주장 불가.

---

## P46 — CTR (Class-Transfer Recovery): RCS + Masked-Context Consistency + Class-Prototype (2026-07-29)

**상태**: **구현 완료 (학습 대기)**. 제안 = [decisions/2026-07-28-p46-classtransfer-recovery-proposal.md](../decisions/2026-07-28-p46-classtransfer-recovery-proposal.md). Base = **P39.1-rank 동결**(gated_mlp trunk + VICReg + P36 router + M2F, 하이퍼 무변경). 파일: `semseg/models/reliadino/p46.py`(신규), `model.py`·`train_reliadino.py`(배선), `tools/smoke_p46.py`(신규 CPU 스모크). config: `configs/hpca100-deliver_rgbdel_P46_ctr.yaml`(DELIVER 4모달 img/depth/event/lidar, EPOCHS 200).

**동기(우리 실측)**: DELIVER val→test 하락의 지배 원인 = **per-class 도메인 전이 붕괴** — Wall 62→2, TrafficLight 81→13, Water 33→0, Bridge 46→0. per-domain spread는 작고(P38 2.58) 모달 융합은 이미 천장(P39.1이 val·test 모두 baseline 미돌파) → 남은 지렛대는 **클래스 표현의 전이**뿐. 하위원인 (a) rare-class under-learning, (b) 도메인 간 class 표현 붕괴.

**3 구성요소 (전부 학습 전용·내부신호만·주손실 직결 = 키1, zero-init 잔차 아님)**:

| | 기제 | 근거 | 결선 위치 |
|---|---|---|---|
| **C-1 RCS** | train 라벨 전수 1회 스캔 → `P(c) ∝ exp((1−f_c)/T)`(T=0.01) → class 샘플 → 그 class를 담은 이미지 샘플. 런타임 per-class CE **EMA**로 `P(c)·(1+w·ĝ_c)` blend | DAFormer 2111.14887 | 데이터로더(`RareClassSampler`가 DistributedSampler 대체) — **주 CE/M2F 손실이 이 데이터를 본다** |
| **C-2 MCC** | student = 패치 마스킹(ratio 0.5 / patch 64, 전 모달 동일 위치) 입력, teacher = **EMA 복사본 + 원본** 입력. 마스킹 영역에서만 pseudo-label CE(conf≥0.75 게이트) | MIC 2212.01322 — UDA를 **source-only DG**로 변형(target 도메인 불요) | trainer 보조 branch + `EMATeacher` |
| **C-3 PROTO** | per-class EMA prototype bank(K×D) + prototype-contrastive CE(`CE(cos(f,P)/τ, y)` = 자기 prototype 당김 + 타 prototype 밀어냄). ColorAugSSD 등가 스타일 2-view를 **같은 bank**로 당겨 도메인불변화 | dual-prototypical 2309.14282 / SCSD 2412.12050 | `PrototypeBank`(model 서브모듈, 학습 파라미터 0·버퍼만) → `aux['p46_proto']` |

**보조 branch 설계(비용 상한)**: C-2와 C-3 2-view는 **하나의 추가 forward를 공유**한다(입력 = 선택적 스타일 변주 → 선택적 패치 마스킹). 토글을 더 켜도 forward는 늘지 않아 스텝 비용이 ≈2.2×(student 2 + teacher 1)로 묶인다.

**🔴 DDP 계약 (스모크가 실측으로 잡아낸 2건)**:
1. **버퍼 in-place 사망**: DDP는 매 forward 시작마다 rank0 버퍼를 in-place 복사한다. 보조 branch가 켜지면 2번째 forward의 브로드캐스트가 1번째 그래프가 저장한 버퍼(M2F `empty_weight`)를 갈아엎어 backward가 `modified by an inplace operation`으로 죽는다 → 보조 branch가 있을 때만 `broadcast_buffers=False`. P39.1의 버퍼는 상수라 의미 변화 없음(prototype bank는 rank-로컬 EMA가 된다).
2. **unused-param 집합 불일치**: `find_unused_parameters=True`는 **마지막** forward의 그래프로 unused를 정한다. 두 forward의 경로가 갈리면 한쪽에서만 쓰인 파라미터가 "unused로 ready 처리된 뒤 hook이 또 발화"해 reducer가 죽는다 → 보조 branch는 (a) `gt_mask`를 **똑같이** 넘기고(내부 aux 손실 결선 동일, 반환값은 버림) (b) P39 path-dropout 추첨을 **재생**(`_p46_replay_path`)한다. 그래서 확률적 입력 모듈(MODAL_DROPOUT/P42/P44.LOCAL_MASK/RCA/P45)과는 동시 사용 불가 — trainer가 명시적으로 `RuntimeError`를 던진다.

**추론 불변**: teacher·bank·샘플러는 전부 학습 루프 소속. `model.eval()` 경로는 P39.1과 **완전 동일**하며, 스모크 C가 all-on ↔ all-off eval logits `|Δ|max = 0`으로 검증한다.

**검증**: `tools/smoke_p46.py` (CPU, tiny ViT) 전항목 PASS — ① 6개 토글 조합 1-step fwd+bwd ② 각 aux 손실 **단독 backward** 시 LoRA `b_q`/fusion/FPN/head grad>0(키1) ③ eval 등가성 |Δ|=0 ④ RCS가 희소 클래스를 실제 up-weight + loss-EMA blend 작동 ⑤ 합성 DELIVER 라벨 PNG로 1-25→0-24 디코딩·캐시 왕복 일치 ⑥ `--ddp`(gloo 2-proc) 보조 branch 2-forward에서 rank 간 gradient 완전 일치. **실데이터 학습 미기동**. ⚠️ LoRA `a_q`는 `b`가 zero-init이라 step 0에서 grad가 정의상 0(baseline seg 손실도 동일) → 합격 판정은 `b_q`로 한다.

**판정 게이트(사전등록, 제안서 §4)**: 목표 **test ≥56.71**(DGFusion SOTA 돌파) & **val ≥68**. 🔴 falsifiable 예측 = collapse 클래스 test IoU 회복 Wall 2→≥13 / TrafficLight 13→≥40 / Water 0→≥9 / Bridge 0→≥20 / RailTrack ≥62 유지 — **회복 없으면 class-transfer 가설 반증 → 설계 폐기**. ep30 조기 kill = collapse 클래스 test IoU 합이 P39.1 대비 하락/무변화. ablation = C1/C2/C3 각 토글 분해. 무음 no-op 조기 검출 지표: `p46/rcs_class_entropy`(C-1이 실제로 rare를 뽑나) · `p46/mcc_pseudo_rate`(0이면 teacher가 conf 문턱을 못 넘음) · `p46/proto_coverage`.

**노벨티(정직)**: RCS·MIC·prototype DG **개별 기제는 전부 선행 존재 → 노벨티 아님**. 차별 축은 조합 ① multimodal frozen-VFM(선행은 전부 단일 RGB) ② 내부신호만(CLIP-text·GT-depth·조건라벨 배제) ③ 단일 아키가 DELIVER+MUSES 공용 ④ 진단주도(관측된 per-class val→test 붕괴 표적). "first X" 주장 불가.

---

## P43 — PanopticDual: 독립 주손실 mask-classification 헤드 (2026-07-25)

**상태**: **구현 완료 (학습 대기)**. 제안 = [decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md](../decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md) §2. 파일: `semseg/models/reliadino/panoptic_head.py`(신규 `MaskClsHead`), `encoder.py`(블록 tap 훅), `model.py`·`train_reliadino.py`(배선), `tools/module_ablation.py`(토글), `tools/smoke_p43.py`(신규 CPU 스모크). configs: `jarvis-muses_rgbel_P43_pdual.yaml` / `hpca100-deliver_rgbdel_P43_pdual.yaml` / `yeon-deliver_rgbdel_P43_pdual_smoke.yaml`.

**동기**: MUSES mIoU 보드는 융합 방법에게 죽은 축(1위 GtA 82.39 카메라단독, 2위 MM-SAM-Adapter 81.07). **PQ가 유일한 현실적 SOTA 축**인데(DGFusion 61.03 / CAFuser 59.70 / M2F baseline 53.60, frozen-VFM 참가자 0) 우리 per-pixel head는 PQ 산출이 구조적으로 불가.

**P38·P30과 무엇이 다른가 (이 설계의 전부)**:

| | 병합 방식 | 결과 |
|---|---|---|
| P30 | query decoder가 conv head를 **대체** | 소물체 붕괴 |
| P38 | `logits + β·sem_q`, β **zero-init** | 추론 no-op(β 0.13 정체), off Δ +0.04~0.12 (실패-키 1) |
| **P43** | **잔차 없음** — 두 헤드가 각자 주손실 | `L = L_pixel + λ(t)·L_mask`, 공유는 SimpleFPN 트렁크뿐 |

pixel logits에 더하지도, 게이트하지도, 대체하지도 않는다. 스모크가 이를 assert한다: mask 손실만 backward하면 pixel head(`head.cls`, `head.fuse`) grad = **정확히 0**이고, pixel 손실만 backward하면 query decoder grad = **정확히 0**. 두 손실이 만나는 유일한 지점은 공유 SimpleFPN·fusion·LoRA다.

**Head B 구조** (Mask2Former 2112.01527): N=100 learned query가 SimpleFPN {1/32,1/16,1/8} 레벨을 **round-robin**(layer i ↔ level i%3, coarse first)으로 masked cross-attn. 매 layer 공유 cls(K+1)/mask-embed head 적용 = deep supervision. attn mask는 **현재 layer의 공유-head 예측**에서 생성(P37b `mask_proj` 무-gradient 버그 구조적 배제). mask = mask_embed · `mask_feat_proj(1/4 SimpleFPN 레벨)` — pixel head의 내부 feature가 아니라 **트렁크 레벨을 직접** 읽는다. 손실 = Hungarian + CE(no-obj 0.1) + point-sampled BCE/Dice(2/5/5), **PointRend importance sampling**(uniform 매칭 / oversample 3× → 불확실 75% + uniform 25% 손실). 타깃 = MaskFormer semantic 모드(2107.06278 §3.3, 이미지에 존재하는 클래스마다 이진 마스크 1장) — **지금 있는 semantic GT만으로 학습된다**.

**λ 스케줄**: `λ(t) = LAMBDA·(0.1 + 0.9·min(1, ep/LAMBDA_WARMUP_EP))`, 기본 LAMBDA 1.0 / WARMUP 5ep. 모델이 pre-scale해 `aux['p43_mask_loss']`로 반환, trainer는 합산만. 모니터: `train/p43_mask_loss`, `p43/lambda`.

**T-2 lateral (PMT 2603.25398, 실증 +2.2 PQ)**: frozen DINOv3 중간 블록 3곳(ViT-L 24블록 → **5/11/17**)을 forward hook으로 tap, 모달 간 **고정 균등 평균**(학습·추론 재가중 아님) 후 `LayerNorm2d + 1×1 conv`로 SimpleFPN 레벨 {1/4,1/8,1/16}에 가산. zero-init·게이트 **아님** — 주 경로라 첫 스텝부터 gradient(실패-키 1). 훅 방식이라 `forward_features`(백본별 pos-embed/RoPE 처리)를 건드리지 않아 tap-off 경로는 bit-identical.

**추론**: semantic = **Head A 그대로**(기존 eval 도구 전부 무수정 동작, `EVAL_HEAD: false`면 평가 forward에서 헤드를 아예 실행하지 않아 속도도 baseline과 동일). PQ = `model.panoptic_inference()`(표준 M2F 후처리, `THING_IDS`로 thing/stuff 구분). 분석용 `model.semantic_from_queries()` = 쿼리 분기만의 semantic(P30/P38/P43 3-way ablation 측정용). `SEM_SOURCE`(pixel|query|sum)는 **eval 전용** — 학습은 언제나 pixel head 단독으로 CE를 받는다.

**토글**: `P43.M2F_HEAD`(false ⇒ LATERAL도 따라 꺼져 forward가 baseline과 byte-identical — 스모크 C가 state_dict 키 + `|Δ|max=0`으로 검증) / `P43.LATERAL`(단독 ablation arm 가능) / eval-time `p43_lateral_off`·`p43_m2f_off`(`tools/module_ablation.py`). ⚠️ `p43_m2f_off`는 **SEM_SOURCE≠pixel일 때만 등록**된다 — 설계상 헤드가 semantic 출력을 안 건드리므로 무조건 등록하면 Δ=0 행이 "죽은 모듈"로 오독된다.

**검증**: `tools/smoke_p43.py` (CPU, 실백본 tiny ViT) 전항목 PASS — fwd/bwd 유한, 신규·기존 파라미터 전부 grad 수신, 헤드 독립성(위 표), OFF 등가성, λ 스케줄, panoptic 세그먼트 유효성, bf16 autocast에서 손실 fp32 유지, 토글 동작·복원. **실데이터 스모크 미실행**(GPU 미점유) — 본학습 전 `yeon-deliver_rgbdel_P43_pdual_smoke.yaml` 2ep 선행.

**판정 게이트(사전등록)**: ep30 ① val PQ 상승 & PQ_thing>0(P30 붕괴 시그니처) ② Head A thin-class IoU가 dense-only 대비 −1pt 이내 ③ 쿼리 비어있지 않음. 완주 MUSES val mIoU ≥82.22 유지 & test PQ ≥59.7(CAFuser), 업사이드 61.03(DGFusion). DELIVER = P36 fair(67.74/55.62) + thin-class(Wall≥13/Water≥9.5/RailTrack≥62).

---

## P44 — BMR (Balanced Multimodal Reliability) + P45 FogStyle (2026-07-25)

**상태**: **구현 완료 (학습 대기)**. 제안 = [decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md](../decisions/2026-07-24-p43-p45-cvpr-sota-proposal.md) §3(P44)·§4(P45)·§7-b(coverage-aware). P42 후계, P42 코드 경로와 직교(`P44.LOCAL_MASK.MODE: global`이 P42 의미 재현). 구현 기록(상세) = [models/p44-bmr-implementation.md](p44-bmr-implementation.md).

**구성 요소**: B-1 MMPareto gradient 통합(2405.17730, `mmpareto.py` 신규 — 주 CE·per-modal aux CE gradient를 합산이 아니라 Pareto 방향+크기 복원으로 통합, allreduce 이후 결합·`no_sync` 필수) / B-2 peer 상호증류(`p44.py`, aux logit 전 순서쌍 대칭 KL + 토큰쌍 relational correspondence) / B-3 coverage-pattern 국소 img 마스킹(P42 전역 drop의 국소화 승격, lidar 리턴 패턴에서 blob 샘플링) / M-3 hard-pixel aux(기본 off) / V-1 결정론적 presence 재정규화(데이터 부재 픽셀에서 gate/router softmax를 present 모달 위로 재정규화, P44에서 유일한 추론 경로 변경) / P45 FogStyle(feature-space fog style 섭동+일관성, img 브랜치 한정, 기본 off). 전 항목 신규 학습 파라미터·zero-init 잔차·attn-bias·학습형 추론 게이트 0개(전부 손실/gradient/입력 레벨).

**결선**: `mmpareto.py`(신규) + `p44.py`(신규) + `fusion.py`·`model.py`·`train_reliadino.py` 배선. configs `{jarvis-muses,hpca100-deliver}_P44_bmr.yaml` + `yeon-deliver_P44_bmr_smoke.yaml`.

**검증**: `tools/smoke_p44.py`(CPU, 86 assert) 전건 PASS — all-off 시 baseline bitwise 동일, 충돌 목표쌍은 두 목표 모두 하강 방향으로 투영, KL gradient가 전 모달 브랜치 도달, coverage 마스크 ⊆ lidar 발자국, V-1 부재 픽셀 가중 정확히 0, tiny-ViT end-to-end 토글 전부 on에서 aux gradient가 3개 모달 LoRA 그룹 전부 도달·eval 결정론. **실데이터 학습 미기동**.

**병합 후 통합 검증(2026-07-25)**: P43+P44+P45 동시-on forward/backward 유한, eval 결정론, `panoptic_inference()` 동작 PASS(develop 병합 35ddbe0 이후).

**판정 게이트(사전등록)**: ep30 ① dMIoU(lidar) 0 근처→>1 ② CKA(img,lidar) ≥0.5 유지 ③ MUSES val ≥82.22 유지. fog 목표 = +2~5pt(P45 결합 시).

---

## P38 — MaskQueryLite: Mask2Former-lite Query Head (2026-07-17)

**상태**: **구현 완료 (학습 대기)**. 계보: P36 공정 레시피(`GATE`·`VETO`·`CALIB`·`ROUTER` on / `ATTN_BIAS`·`CONSISTENCY` off / `PHYSAUG` off / `DGFUSION_AUG` on) **동결** + Mask2Former-lite query head 추가. P37a(CEFR-Head)·P37b(ClassToken-lite-Learned)는 **off** — 깨끗한 1-변수 비교(P36 대비 변경점 = M2F head 단 하나)로 설계. config `configs/bengio-deliver_rgbdel_P38_m2f.yaml`(200ep, 768², bs2, 8-GPU 상정) / 스모크 `configs/yeon-deliver_rgbdel_P38_m2f_smoke.yaml`(2ep). 파일: `semseg/models/reliadino/m2f_head.py`(신규), `model.py`·`train_reliadino.py`(배선). 커밋 3bb2c41(develop 병합 tip 6d922bd).

**동기 (3가지)**:
1. **PQ 서사 정합**: DGFusion/CAFuser는 OneFormer(mask-classification) 스택이라 MUSES 주표가 **PQ** — 우리 기존 per-pixel head는 구조적으로 PQ 산출이 불가능했음. mask-classification 헤드로 전환해야 동일 잣대 비교가 성립.
2. **thin/희소 클래스 우세(문헌)**: mask-classification은 Wall/Water/RailTrack류 thin/희소 클래스에서 per-pixel 대비 +1~3 mIoU 우세로 보고됨.
3. **head confound 제거**: P36 위에 head 하나만 추가해 고정하면, 남는 성능차는 신뢰도 라우팅 융합(RBMA 계보의 핵심 노벨티)의 몫으로 귀속할 수 있음 — 모듈 기여 분리를 위한 통제.

**구조**: 100개 learned query가 gated fused stride-16 feature map 위에서 6-layer masked cross-attention(P37b `_TokenDecoderLayer` 재사용)을 수행. 공유 cls(K+1)/mask-embed head를 매 layer에 적용해 deep supervision. **attn mask**는 이전 layer의 공유-head mask 예측을 stride4→16으로 리사이즈해 다음 layer의 cross-attn을 제한(Mask2Former 표준 관행) — P37b의 `mask_proj`처럼 threshold-only 비미분 경로가 아니라, 예측을 직접 리사이즈해 사용하므로 gradient가 흐른다 (`_attn_bias`, 아래 ISSUE-024 참조).

**손실**: Hungarian matching(scipy) + CE(no-object weight 0.1) + point-sampled BCE/Dice(가중 2/5/5, 12544 pts/query). 매칭·손실 계산은 모델 내부(`m2f_head.py`)에서 수행되고 `aux['m2f_loss']`로 노출 — trainer는 `LOSS_W: 0.5`로 가중합만 수행.

**Collapse-safe 병합**: 최종 세그멘테이션 출력 = `conv_head + β·sem_query + router_alpha·routed`, `β`는 **zero-init**. 따라서 학습 시작 시점에는 M2F 쿼리 브랜치가 출력에 아무 기여도 하지 않아 **P36과 byte-identical** — 합성 스모크에서 on/off 출력차 `|Δ|max = 0.0`으로 검증됨. `β`가 학습되면서 점진적으로 mask-classification 신호가 섞여 들어가는 구조라, 초기화 실패로 인한 성능 열화 리스크가 없다.

**panoptic_inference()**: 표준 Mask2Former 후처리(클래스 확률 × 마스크 확률 → per-pixel argmax + overlap/threshold 필터)를 구현·포함. MUSES **PQ** 산출의 전제 조건 — 기존 per-pixel head에는 이 경로 자체가 없었음.

**검증**: 로컬 합성 스모크 PASS — fwd/bwd 유한, query/cls 양쪽 grad 흐름 확인, β-zero 등가성 exact, panoptic 경로 동작. **실데이터 스모크 미실행**(yeon 8-GPU 전부 점유) — 서버 확보 시 본학습 직전 2ep 스모크가 선행 조건. 파라미터 ~5.2M.

**판정 게이트**: P36 fair(val 67.74/test 55.62) 대비 mIoU + thin-class(Wall/Water/RailTrack) IoU.

---

## P39 — Dual-Path Compete (DPC) (2026-07-20)

**상태**: **구현 완료 (학습 대기)**. 계보: P38(MaskQueryLite) 위에 5개 변경(전부 토글 가능)을 얹음 — 베이스는 유지(frozen DINOv3-L + per-modal LoRA(r8) + cross-modal fusion + SimpleFPN + FPNSegHead + per-class router). 제안 근거 = [decisions/2026-07-20-p39-dual-path-compete-proposal.md](../decisions/2026-07-20-p39-dual-path-compete-proposal.md)(실패-키 문서 전 키 반영). **단일 아키텍처로 DELIVER·MUSES 모두 커버**(user 지정) — 데이터셋 적응은 학습된 모듈로만.

**동기 (실패-키 → 규칙 변환)**: P30~P38 계보에서 반복된 5개 실패 패턴을 규칙으로 역변환해 설계에 내장.

| 키 | 실패 패턴 | P39 규칙 |
|---|---|---|
| 키1 | zero-init 잔차 사장 (4연속 no-op) | 소극 잔차 결선 전면 금지 — 신규 경로는 주 손실을 직접 받거나(V1) 기존 경로와 경쟁(path dropout, V5) |
| 키2 | router 유일 실적 + co-adaptation | router를 직접 감독(CE)으로 "의존"에서 "기여"로 전환, deep-sup 의존 해소는 유지 |
| 키3 | FUSED rank 7/256 병목 | 로짓 근처 모듈 추가 금지 — 융합 트렁크 rank를 직접 확장(V1), query 경로는 병목 우회(V2) |
| 키4 | 문제 위치 상이(클래스축 vs 도메인축) | per-class 학습 중재(V5 Λ)로 데이터셋별 상이한 병목에 같은 기제가 다르게 적응 |
| 키5 | event 기여 = 데이터셋 속성 | 모달 하드 제거 없음 — query가 모달 토큰을 직접 attend해 데이터셋별로 스스로 배분(V2) |
| 반증 C1~C6 | — | attn-bias·gate류 신규 없음 · conv head 즉시 대체 없음(P30 재발 방지) · 무감독 게이트 없음 |

**구조 (P38 대비 변경 5개)**:
- **V1 — Trunk Rank Expansion**(키3): `fused' = fused + Σ_m P_m(f_m)` — 모달별 선형 투영(1024→1024, small-random init, **zero-init 아님**)을 트렁크에 가산 합류. 게이트 뒤 소실된 모달 부분공간을 주 경로에 복원해 rank 상한을 Σ modal rank로 확장, 주 경로 소속이라 첫 스텝부터 CE gradient 수신(키1 충족). +4.2M params.
- **V2 — Modal-token Query Attention**(키3 우회 + 키5): m2f query의 cross-attn 소스를 fused map(N tokens)에서 **per-modal 토큰 합집합(M·N tokens + modality embedding)**으로 교체. query가 융합 병목(rank 7)을 거치지 않고 인코더 피쳐(rank 20~36)를 직접 보아, 모달 배분이 attention 학습으로 데이터셋별 자동 적응(단일 모델 제약 충족). mask dot-product는 기존대로 FPN stride-4.
- **V3 — Anchored + Free Queries**(키2 thin-class + P30/P37b 교훈): query 100개 = 앵커 K개(클래스 고정 할당, Hungarian 없음 — P37b 방식이되 마스크 손실 직접 감독) + 자유 (100−K)개(Hungarian, 인스턴스/PQ 담당). 앵커 query가 thin/희소 클래스(Wall/Bridge/Other)의 Hungarian 기아를 구조적으로 제거.
- **V4 — Balanced Point Sampling**(P38 요인2 직접 수정): mask BCE/dice 포인트 샘플링을 uniform 12,544에서 **GT 영역별 최소 쿼터(클래스당 ≥256pt) + 잔여 uniform**으로 교체. thin 마스크(RoadLine·Wall·RailTrack)가 포인트 예산에서 소멸하는 문제 제거, 추론 불변.
- **V5 — Compete-and-Arbitrate 결합**(키1 핵심, β 잔차 폐기): dense 경로(conv head+router 잔차)와 query 경로(anchored+free 조립)를 zero-init β 대신 ① **학습: path dropout 경쟁**(p_d=0.25 dense-only CE / p_q=0.25 query-only CE / 나머지 결합 CE — 양쪽 다 주 손실을 단독 감당해 무임승차 불가) ② **추론: per-class 학습 중재** `final_k = dense_k + softplus(Λ_k)·query_k`(Λ init 0 → softplus≈0.69, 죽은 시작 아님) ③ **router 직접 감독** `CE(up(routed_logits), gt)`(w=0.4) 추가로 router를 자립 기여로 전환.

**손실**: `L = CE_compete(final|dense|query) + w_r·CE(routed) + mask-cls(anchored 고정매칭 + free Hungarian, V4 샘플링, 2/5/5, deep-sup) + aux_ce + cal`(기존 유지).

**판정 게이트 (사전 등록)**: DELIVER = P36 fair(val 67.74/test 55.62) + thin-class 복원(**Wall≥13/Water≥9.5/RailTrack≥62**, P36 수준) · MUSES = **P38 val 82.22 이상**(신규 내부 최고). 모듈 판정 = 학습 직후 `module_ablation.py` 토글 즉검(완주 후 발견 금지) — `p39_query_off`/`p39_trunkexp_off`/`p39_anchored_off`/`router_off` 각각 |Δ|>0.5 & agreement<0.99 (no-op 조기 탈락 기준).

**리스크와 방어**: P30 재발(query-only 붕괴) 방지 = p_q=0.25 한정 + V3 앵커 + V4 쿼터가 원인(소물체 기아)을 직접 제거, dense 경로는 추론에 항상 존재. 변경 5개는 다변수(1-변수 5세대는 deadline상 불가 — 결합 문제는 단독 변경으로 안 풀림)라 **전 항목 토글 구현 의무화**(`p39_query_off`/`p39_trunkexp_off` 등 + config off)로 ablation 표에서 분해. physaug는 공정선 유지(헤드라인 off, ablation 행만).

**검증**: 합성 스모크 **PASS** — 5지점 grad 흐름 확인, 토글 전부 유효, det(query-only 등) 폴백 확인, β/Λ 초기화 경로에서 **P38 호환 등가성** 확인. **실데이터 스모크 미실행**(선행조건 — yeon 배치 예정).

**config**: `configs/hpca100-deliver_rgbdel_P39_dpc.yaml`(200ep) · `configs/jarvis-muses_rgbel_P39_dpc.yaml` · `configs/yeon-deliver_rgbdel_P39_dpc_smoke.yaml`(2ep, 실데이터 스모크). 커밋 **c31dcd5**(develop).

**실행 계획**: ① 구현 완료(V1~V5+토글 5종+config 3벌) ② yeon 빈 GPU 2ep 스모크 → hpca100/jarvis 첫 빈 슬롯 투입(DELIVER 우선, MUSES는 jarvis P38 완주 후 이어달리기) ③ ep30 조기판정(module_ablation 즉검 + val 궤적 vs P36/P38 동 epoch, no-op 검출 시 조기 중단 — 2026-07-16 EPOCHS 사고 규칙).

---

## P39.1 — Rank 수리 (gated_mlp trunk + VICReg) (2026-07-21)

**상태**: **구현 완료 (학습 대기)**. 계보: P39(DPC) 위에서 V2(modal-token attention)·V3(앵커)·V4(쿼터)·router 직접감독·deep-sup은 **동결**, V1(trunk rank expansion)만 교체 + 신규 정규화 1종 추가. 제안 근거 = P39-MUSES 표준분석(2026-07-21)·fog_night 원인규명(07-20)·관련연구 딥리서치 3편(rank collapse / modality imbalance / fog 물리) 교차검증 = [decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md](../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md). 커밋 **ac5c7fe**(develop).

**동기**: P39-MUSES 표준분석이 **lidar effective-rank 4.7 붕괴**(adapter가 압축 주체, feat_cos 0.115)와 **fog_night 62.68 붕괴**(전 제출 최저)를 지목. 문헌 대응 — V1의 선형 투영(`P_m`)+LoRA BA가 딱 "선형 cascaded 경로의 암묵적 저rank 편향" 구조(deep matrix factorization/DirectCLR, LoRA intruder dimensions)이고, **rsLoRA 없는 단순 r 상향은 무효**하다는 근거. gate/calib은 fog_night에서 유해로 재판정(3세대 no-op→유해)돼 M-2로 off.

**구성 (P39 대비 변경 2개, R-3는 조건부)**:
- **R-1 (주 변수)**: V1 `fused += P_m(f_m)`(선형, small-random init)을 **`fused += tanh(γ_m)·MLP_m(f_m)`**로 교체 — MLP = LN→1×1(1024→256)→GELU→1×1(→1024), γ는 **모달별 스칼라, init 0.1**. **γ=0(완전 zero-init)이 아니라 0.1로 잡은 이유**: 0이면 tanh(0)=0이라 MLP 브랜치가 첫 스텝부터 gradient를 전혀 받지 못해(키1 "zero-init 잔차 사장" 재판) 학습 자체가 시작되지 않음 — 스모크로 실증. 0.1은 초기 기여를 작게 유지하면서 gradient 흐름은 보장하는 절충.
- **R-2**: **VICReg var+cov 정규화**를 per-modal 토큰에 적용(lidar 가중 ×1.0, img/event ×0.25, λ_var 0.1/λ_cov 0.01, 토큰 2048 서브샘플, fp32, per-GPU) — lidar rank 붕괴 직접 복원용(VICRegL·Shuffled-DBN 스펙트럼 복원 문헌 근거).
- **R-3 (조건부 2차, 미구현 대기)**: ep30 게이트 미달 시 전모달 r 8→16 + rsLoRA(α/√r) + AdaLoRA 직교항 0.1로 재기동.
- **M-2**: gate/calib/veto config off(fog_night ablation에서 유해 실증 반영, ablation 행으로 분리).
- **로깅**: eval마다 per-modal effective-rank(RankMe) 계산해 `p391/rank_*`로 노출 — ep30 게이트(lidar rank) 판독용. train 쪽 `train/vicreg` 로깅도 추가.

**판정 게이트 (사전 등록, ep30)**: `feature_stats`로 측정한 **lidar effective-rank ≥15** & **fog_night drop-lidar ≥4.0**(fognight 분석 M-1 게이트 유지). 미달 시 R-3 적용 후 재기동, R-1/R-2 모두 무효로 판정되면 V2(modal-token attention)를 원인으로 전환해 재설계.

**선행 분석 (학습 0, 분석 세션 위임)**: ① MUSES fog val split per-scene 감사(night>clear 역전이 문헌과 반대라 소표본 아티팩트 배제 필요) ② P39 ckpt에서 trunk_exp off 시 lidar rank 재측정(V1 원인 확정). 둘 다 기존 ckpt로 수행, 학습 불필요.

**검증**: 합성 스모크 **PASS** — γ/MLP grad 흐름 확인, eval 결정론, linear 모드(구 V1) 하위호환 확인.

**config**: `configs/jarvis-muses_rgbel_P39_1_rank.yaml` · `configs/hpca100-deliver_rgbdel_P39_1_rank.yaml`.

---

## P40 — RCA-Fusion: Reliability-Conditioned Attenuation (2026-07-21)

**상태**: **구현 완료 (학습 대기, P39.1 rank 게이트 통과 후 투입)**. 계보: P39.1 위에 조건부 감쇠 모듈(C-1~C-3) 추가. 제안 근거 = [decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md](../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md). 커밋 **ac5c7fe**(develop).

**서사**: 신뢰도 기계가 5세대(P28→P39) 동안 *추론-시 재가중*(attn-bias/gate/CEFR)으로는 반증 완주된 무효 시도였다. P40은 같은 신뢰도 신호를 **학습-시 조건화**로 옮긴다 — "모델 스스로 카메라가 나쁘다고 추정하면, 그 샘플은 카메라 없이도 풀 수 있어야 한다"는 자기-일관성 루프. 외부 라벨/신호 0 유지(내부 신호만).

**구성**:
- **C-1 (신뢰도 신호 확장)**: 기존 rel_cal에 **lidar 리턴 유효성 통계**(입력에서 유도한 per-region density/zero-return 맵, 내부 신호)를 더함 → RCA 가드 + 분석 스태시로 사용. CAFuser(전역 CLIP condition token)·DGFusion(depth 값 감독)과 구조적으로 구별(per-region·물리 유도·무감독).
- **C-2 (조건부 감쇠 학습)**: 학습 중 per-sample로 자기추정 rel(img)가 배치 **하위 분위(30%)**면, 해당 샘플의 img feature를 **soft 감쇠**(α∈[0.1,0.5], **hard-zero 금지** — 무조건 드롭아웃의 "missing 지름길" 역효과 문헌 회피). curriculum ramp(ep20까지 0→p_max 0.5).
- **C-3 (약모달 readout 보조 손실)**: 감쇠된 샘플 한정 **lidar readout 보조 CE**(w=0.5) — 감쇠만으로는 fusion이 "저카메라 모드 암기"로 빠질 수 있어 gradient 출구로 추가.
- **C-4 (사전 검증 게이트, 학습 전)**: 신뢰도 추정기 자체가 카메라 편중이면 무조건 드롭아웃으로 퇴화 — fog_night rel AUROC(img) ≥0.75 확인(P39 0.70/P38 0.79 실측), 미달 시 C-1 통계 신호를 주 신호로 전환.

**노벨티 포지셔닝**: "reliability-conditioned modality attenuation for condition-robust dense fusion". 최근접 선행 = OPM(T-PAMI'24, 배치레벨·라벨유도·분류)·SGMA(샘플링 빈도 조건화) — 차별 4축(자기추정 per-sample 신뢰도 신호 / 강모달 입력 감쇠의 조건 표적화 / dense prediction / frozen-VFM 제약). 지지 증거 = 자체 P33 무조건 드롭아웃 no-op 재현 + 무조건 드롭아웃 역효과 문헌.

**판정 게이트 (사전 등록)**: MUSES **test ≥79.025**(P38-m2f 현 최고) & **fog_night ≥74**(P38 수준 복원, 이후 fog 전체 66~69 도전은 물리 상한 감안한 2차 목표) · DELIVER = P36 fair(val 67.74/test 55.62) + thin-class 유지.

**검증**: 합성 스모크 **PASS** — RCA pick 발생 확인, C-1 가드(lidar 부재 샘플 제외) 동작, vicreg/readout 손실 유한, grad 흐름 확인, eval 결정론.

**config**: `configs/jarvis-muses_rgbel_P40_rca.yaml` · `configs/hpca100-deliver_rgbdel_P40_rca.yaml` · `configs/yeon-deliver_rgbdel_P40_rca_smoke.yaml`(스모크).

**실행 순서**: ① 분석 선행 2건(P39.1 절 참조) ② P39.1 투입·ep30 rank 게이트 통과 확인 ③ P40 투입(rank가 죽은 채면 C-3 lidar readout이 헛돎 — 순차 필수).

---

## P32 — CoRB: Corroboration-Biased Memory Attention (2026-07-06)

**계보**: `LoRA_Sam_P32(LoRA_Sam_P31)` — RBMA 배관(P27 memory-attn logit additive bias) 그대로. **변경점은 신뢰도 신호의 의미뿐**: self-entropy → cross-modal corroboration.

**동기(진단)**: RBMA 신뢰도 `rel_i=1−H(softmax(D_i(f_i)))/logC`는 per-modal decoder 용량과 confound → event/LiDAR anti-calibrated(correctness-AUROC .30/.22, [16] §7). "자기확신"이 아니라 "상호검증"으로 측정해야 한다.

**신호 (training-free, corr_veto — Phase 0 v2 확정)**:
```
p_i = softmax(D_i(f_i))                              # per-modal posterior (무학습)
p̄_{−i} = mean_{j≠i} p_j                              # leave-one-out 합의
corr_i = Σ_c √(p_i · p̄_{−i})                         # Bhattacharyya coeff ∈[0,1]
selfent_i = 1 − H(p_i)/logC
g_i = clamp(selfent_i − max_{j≠i} selfent_j, 0, 1)   # unique-info veto gate (threshold-free)
rel_i = g_i·selfent_i + (1−g_i)·corr_i               # veto blend
bias_i = λ·(rel_i − mean_j rel_j)                    # RBMA 배관, λ만 학습
```
- **veto 근거**: 순수 corroboration은 "다수가 못 보는 곳에서 홀로 confident한 workhorse"(P31 depth)를 벌해 AUROC .90→.28 붕괴 → g_i가 uniquely-confident 모달을 self-confidence로 보호. corr_veto가 depth .71 회복·event/LiDAR >.6 유지.
- 구현: `sam_lora_image_encoder_seg.py` `LoRA_Sam_P32._compute_bias_source` override(temperature-free, `tools/eval_reliability_auroc.py` corr_veto와 동일). config `CORROBORATION.ENABLE/VETO`, OFF→P31 byte-identical. corr 수식은 P31 `consistency_bias`(2차 항)를 1차 신호로 승격한 것.

**한계(실측, [16]/[24]·`/mnt/HDD2/src/logs/P32_eval_20260706/`)**: 신호 AUROC는 반전(event/lidar .59/.85)됐으나 **drop-modality Δ event/lidar≈0 = 여전히 미사용(Mode C)**. soft attention-bias는 feature/decoder가 약한 모달(competence≈0)을 못 살림 → Test 53.45<P28 55.27. **후속 P32-C(PruneMem: hard token pruning+modality dropout)** 가 이 R4(soft 융합이 select 못함)를 직격.

**선행연구 차별**: RSGMamba consistency gate=learned MLP; 우리는 무학습 통계+attention logit bias. "training-free cross-modal corroboration을 attention logit bias로" 셀 미점유(vault 42/47).

## P31 — Calibrated Dual-Reliability RBMA + Multi-scale HR Class-Token Decoding (2026-07-02)

**상태**: **구현 완료 (학습 대기, B200 타깃)**. 계보: `LoRA_Sam_P31(LoRA_Sam_P30)` — P30(CTD/router)+P29(SDC)+RBMA 상속. 설계 근거 = `20_p31_design_proposal.md`(P31-Seg core ①②) + `16_failure_analysis_P28_P29.md` §7 정량 진단. 전 기구 config-gated (전부 OFF → P30 byte-identical). config `configs/b200-deliver_rgbdel_P31_physaug.yaml`.

**동기 (doc 16 §7 실측)**: ① reliability AUROC [img .77, depth .62, **event .30, lidar .22**] — geometry 모달이 anti-calibrated(틀린 곳에서 과확신) → RBMA bias 신호 무의미. ② 융합 가중치 거의 uniform [.27,.28,.23,.23] vs 실기여 drop-Δ [8.4,23.5,0.02,0.01] → **router가 모달리티를 adaptive하게 선택 못함**(질량 ~45% 낭비, TrafficLight misalloc 0.37). ③ m_feat 단일 저해상(32ch,s4) 질의 → thin-class(Bridge/Water/Wall) 경계 muffle. ④ frozen-backbone ceiling(ISSUE-008): Bridge/Other modal_competence [0,0,0,0].

**기구 (모델 `sam_lora_image_encoder_seg.py` 말미, 유틸 `sam_lola_utils.py`)**:
- **[Seg-A] RBMA reliability 재보정** (`RBMA_CALIB.ENABLE`): per-modal 학습 temperature `T_i`(rbma_log_temp)로 `rel_i = 1−H(softmax(D_i(f_i)/T_i))/logC` + **correctness-contrastive calibration loss**(틀린 픽셀 entropy↑·맞은 픽셀 entropy↓, 1/4 해상도, `gate_loss_data['rbma_cal_loss']`, weight `RBMA_CALIB.LAMBDA=0.1`) → reliability의 정답 AUROC 직접 최적화. Phase 2.5 aux logits를 `_auxiliary_decode_single` override로 stash(재디코딩 0). 추론 bias는 여전히 GT-free training-free.
- **[Seg-B] Consistency 2차 bias** (`RBMA_CALIB.CONSISTENCY_BIAS`, **기본 OFF — A 성공(AUROC>0.5) 후 조건부**): `B_cons_i = mean_j Bhattacharyya(p_i,p_j)` centered → `softmax(QKᵀ/√d + λ_ent·B_ent + λ_ent·λ_cons·B_cons)` dual-axis training-free 항. λ_cons 학습 스칼라.
- **[Seg-A2] Reliability-proportional AMF** (`RBMA_CALIB.AMF_RELIABILITY`, 기본 OFF): 출력 융합 `w=softmax_m(rel/τ)`. learned_router OFF일 때만 유효(우선순위: router > rel-AMF > amf_mode). doc16 §7-2 "보정 후에만 전환" 게이트.
- **[Seg-C] Multi-scale HR class-token decoder** (`CLASS_TOKEN_DECODER.MULTI_SCALE`): `ClassTokenDecoderMS(ClassTokenDecoder)` — m_feat(s4)에서 simple-FPN {4,8,16,32} 피라미드(ViTDet 레시피) → class token이 coarse→fine cross-attend(scale embed) → **학습형 ConvTranspose ×UP 고해상 pixel-embed**(HR 프로토타입 흡수) + **training-only aux per-pixel CE head @H/4**(GOOSE-M2F, `gate_loss_data['ctd_aux_ce']`, weight `AUX_CE_WEIGHT=0.4`, 추론 비용 0). drop-in 동일 `(feat)` 시그니처, +1.28M params.
- **[레버①] backbone 부분 unfreeze** (`UNFREEZE_LAST_N_BLOCKS=3`): Hiera trunk 마지막 stage(blocks 21-23) unfreeze — ISSUE-008 구조적 dead-class의 유일 지렛대. optimizer가 `UNFREEZE_LR_SCALE=0.1` 감쇠 LR 그룹으로 분리(`semseg/optimizers.py` backbone_prefix).
- **[레버②] Router 'decisive' reg** (`LEARNED_ROUTER.REG_MODE: decisive`): 기존 'diversity' reg(mixing-entropy 보상)는 **uniform 방향으로 미는 모순** → `reg = batch-marginal entropy − per-pixel entropy`(per-pixel commit + 전역 다양성; SDC loss와 동형 confident+diverse 쌍). `ReliabilityAnchoredRouter`에 `reg_mode` 추가(기본 'diversity'=P30 불변) + `_last_w_mean` 모니터링 stash.

**trainer 변경**: `QUALITY_GATE_MODELS`+='LoRA_Sam_P31'; `ctd_multi_scale` sig-guard 배선; loss `+λ_cal·rbma_cal_loss + λ_ctd_aux·ctd_aux_ce`; `get_optimizer(backbone_lr_scale, backbone_prefix)` 확장(기본값 = 기존과 byte-identical).

**검증**: py_compile 4파일 PASS + CPU smoke PASS — CTD-MS shape(up=2→2×)/grad(feat·upsampler·aux_head)/aux head training-only, router decisive reg가 "committed-but-diverse > uniform" 순서 확인(diversity reg는 역방향 = 문서화된 결함 재현), calibration loss 유한·T grad 도달, P31 시그니처/오버라이드 체인 확인. **full SAM2 forward는 GPU 미검증**(P30과 동일 단서 — 학습 전 1-GPU sanity 권장).

**Ablation 세트(doc 20)**: RBMA 단일 vs dual-bias / uniform vs rel-proportional AMF / CTD single vs multi-scale / unfreeze 0 vs 3 / router reg diversity vs decisive. 성공 기준: event/lidar AUROC>0.5, router `_last_w_mean` 비uniform, Water/Wall/Bridge/Pole Test IoU 상승.

**P31.1 수정 (2026-07-03, 비판 리뷰 `/mnt/HDD2/src/logs/P31_review_20260702/` 검증 반영)**:
- **P30-seg 실측 붕괴 확인** (B200 ep188): Day-Val best 49.76@ep136 / Test best 44.10@ep146 = **P29 대비 −13.4/−10.2**. det E0.1(같은 ckpt에서 query head 0.256 vs FCOS-aux 0.431)과 동일 패턴 → 경량 class-token decoder가 최종 출력을 대체하는 P30 구조가 주범 후보.
- **① CTD 강등** (`CLASS_TOKEN_DECODER.AUX_ONLY: true`, `ctd_aux_only`): 최종 출력 = SAM decoder 융합(m_output) 복원, CTD(MS)는 학습 시 `ctd_seg_ce`(full-res) + `ctd_aux_ce`(@H/4) aux loss로만 rare-class gradient 공급. 추론 경로에서 CTD 완전 제거.
- **② AUROC 게이트 로깅** (리뷰 R1): `_calibration_loss`가 per-modal reliability AUROC(Mann-Whitney)·μ·σ를 stash → trainer가 epoch마다 `p31/rel_auroc_*`, `p31/rel_std_*`, `p31/router_w_*`, `p31/cal_loss`를 tb+wandb 기록 + 콘솔 출력. 판정: AUROC>0.5 = Seg-A 성공(B/AMF 전환 가능), σ→0 = 엔트로피 상수화 퇴화.
- **③ SDC OFF** (리뷰 R2): doc 16 실증 net −1.08 → P31 기본 config에서 제거 (P28식 add 게이트로 복귀).

---

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

**리스크**: (1) **frozen-backbone 천장(ISSUE-008)** — rare class가 frozen SAM2 feature에 애초에 안 담겼으면 ①②로도 한계; multi-scale FPN/②의 모달 부활로 완화. (2) class-token decoder는 근사 구현이라 실제 SAM decoder 대비 약할 수 있음. (3) **런타임 미검증**: 두 모듈은 CPU dummy smoke(forward+backward+grad+reliability-anchor 초기성) 통과했으나, `LoRA_Sam_P30`의 full forward(track_step 내부와 _fuse_outputs 상호작용)는 SAM2 로드 없이 compile-only 검증 → 학습 전 main이 GPU 1-forward로 확인 권장. 노벨티 = [research/novelty-and-related-work.md](../research/novelty-and-related-work.md) §2.8.

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

**노벨티 포지셔닝**: label-free·image-derived·router-level 조건화는 CAFuser(CLIP/text 조건)·DGFusion(depth+depth-GT) 어느 쪽과도 다름. 상세 = [research/novelty-and-related-work.md](../research/novelty-and-related-work.md) §2.7.

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

---
---

# 📕 P27 ~ P48 세대별 회고 기록 (재시도 방지 문서 · 작성 2026-08-05)

> **이 절의 존재 이유는 하나다 — 다음 세션이 이미 시도해서 실패한 모델을 다시 제안하지 않게 하는 것.**
> 따라서 "무엇을 했나"보다 **"왜 실패했나 / 무엇이 반증됐나"** 를 우선한다. 성공 사례 나열이 아니다.
>
> **파일 상단 P28~P47 절과의 관계**: 이 파일 위쪽(L11~L380)의 P28~P47 절은 **설계 시점 기술**이다 —
> 무엇을 구현했고, 스모크가 무엇을 통과했고, 어떤 게이트를 사전등록했는가. 대부분 "구현 완료 (학습 대기)"에서
> 멈춰 있다. **이 절은 그 뒤에 실제로 나온 결과·판정·반증을 사후 정리한 것**이다. 상단 절은 수정하지 않았다.
>
> 각 세대는 5항 고정 포맷: **① 무엇을 바꿨나 ② 왜 ③ 결과(legal만) ④ 판정 ⑤ 🔴 재시도 금지**.

## §0 — 이 절을 읽는 규약 (수치를 인용하기 전에 반드시 읽을 것)

**0-a. 체크포인트 규약 (user 논의 2026-08-05)**
- **val-best**(val 최고 epoch의 test)와 **final-iter**(마지막 epoch의 test) **둘 다 legal**이다.
- 🔴 **그러나 런마다 유리한 쪽을 고르는 행위 자체가 test peeking**이다. → **모든 런에 두 값을 항상 병기**하고,
  규약을 고르지 않는다. 이 절의 표에서 한쪽만 적혀 있으면 다른 쪽은 **측정되지 않은 것**이다("기록 없음").
- **test-best**(`test_epoch*_*.pth` 계열)는 **어느 프로토콜로도 legal이 아니다**. 인용 시 반드시
  "test-peeking, 사용 불가"로 표기한다. 2026-07-15·08-03·08-04 세 차례 이 규약 위반으로 헤드라인이 철회됐다.
- 베이스라인은 각자 발표 수치를 그대로 인용하되 **어느 규약인지 표에 명시**한다:
  **CMNeXt = val-best / CAFuser·DGFusion = final-iter**.

**0-b. 🔴 DELIVER 내부최고 정정 (2026-08-05 확정)**
- 그동안 헤드라인이던 **`val 67.74 / test 56.62`는 존재하지 않는 모델의 수치다** —
  **val은 P36, test는 P34**를 이어붙인 쌍이고, **P34는 PhysAug ON = 공정선 밖**이다
  (원문 = [decisions/2026-07-16-p36-novelty-critical-review.md](../decisions/2026-07-16-p36-novelty-critical-review.md) §0 표에서 P34 행에 "PhysAug ON = 공정선 밖(unfair-ours)"로 명기,
  physaug 배제는 [status/current.md](../status/current.md):40의 user 판정 2026-07-20).
- ✅ **공정·legal 내부최고 DELIVER = P36 val 67.74 / test 55.62** (둘 다 같은 val-best ckpt, PhysAug off).
- **혼입 시점**: 07-20/21/24 제안서들은 "P36 fair 67.74/**55.62**"로 정확했고,
  **2026-07-28 P46 제안서에서 56.62로 바뀌어 이후 문서 전체에 전파**됐다.
- **파급**: 56.62 기준으로 계산된 모든 델타는 −1.00만큼 과대하다. 특히 P46 λ0.2-seed2의 "내부최고 미달 −1.07"은
  공정 55.62 기준으로는 **−0.07(사실상 동률)** 이다(단 그 런의 val −2.03 하락이라는 별개 실패 신호는 유효).
- **SOTA 기준도 낡았었다**([analysis/2026-08-04-sota-landscape-recheck.md](../experiments/analysis/2026-08-04-sota-landscape-recheck.md)):
  DELIVER SOTA = ~~68.79/56.71~~ → **MM SAM-adapter(RGB-D 2모달) val 69.60 / test 57.35**. 우리 격차 **val −1.03 / test −1.73**.

**0-c. 비교를 무효화하는 이슈 (수치 인용 전 확인)**
| 이슈 | 무엇을 무효화하나 |
|---|---|
| **ISSUE-026** (ColorAugSSD brightness, ✅픽스 07-21) | **07-16 이후 `DGFUSION_AUG:true` DELIVER 학습 전부** — P37a-DELIVER, P38-DELIVER 완주분(= P38 게이트 미달 판정에 쓰인 그 런), P39-DPC resume. **P36 fair(67.74/55.62)는 07-16 이전이라 정상 RGB** → **P36 vs P37+/P38/P39 DELIVER 비교는 불공정했다.** P39.1이 픽스 후 첫 클린 DELIVER 런. MUSES 전 계보 무영향 |
| **ISSUE-025** (MUSES radar 디코딩 3중 버그, ✅픽스 07-21) | **4모달(radar 포함) 실험만** — P34-4모달 test 78.256("radar 유해 −0.72" 판정 보류). **3모달 전 계보 무영향** |
| **ISSUE-022** (`P27.forward`가 `_fuse_outputs` 훅 미호출, ✅픽스 P31.2) | **P30 learned router가 200ep 내내 미실행** → P30의 −13.4/−10.2 붕괴는 "router가 나빴다"가 아니라 **CTD가 conv head를 대체한 효과 단독**으로 재해석해야 한다 |
| **ISSUE-024** (P37b `mask_proj` 무-gradient, 🟡조건부 OPEN) | P37b의 masked attention이 사실상 random 마스킹 — P37b 수치는 "설계가 실패"가 아니라 "버그로 실행되지 않음" |

**0-d. 🔴 피쳐 통계 정정 — "lidar 표현이 4.7차원으로 붕괴"는 오기록이다**
- 2026-08-05 `tools/feature_stats.py`(DELIVER test) 실측: eff.rank **img 55.9 / depth 48.9 / event 97.8 / lidar 231.3**,
  **FUSED_pf 37.5 → PREHEAD 19.0 → FUSED 4.68**. → **4.68은 lidar가 아니라 FUSED(decode 피쳐)** 이고,
  **lidar의 랭크는 오히려 전 모달 중 최고(231.3)** 다.
- 이 두 수를 뒤바꿔 "lidar가 4.7차원으로 붕괴했다"고 적은 기록이 있으면 **오기록**이다(2026-08-05 시점 확인된 것 =
  `experiments/monitor-log.md` 12:30 KST 엔트리. monitor-log는 편집 금지 대상이라 여기에 정정을 남긴다).
- ⚠️ 혼동 주의: **P39-DPC 시절의 "lidar eff-rank 4.7"은 별개의 정당한 측정**이다
  ([analysis/2026-07-21-p39-muses-standard-analysis.md](../experiments/analysis/2026-07-21-p39-muses-standard-analysis.md), MUSES, per-modal 측정).
  그건 P39.1의 VICReg으로 78.5~100.3까지 복구됐다. 08-05의 4.68(DELIVER, FUSED)과 **같은 숫자지만 다른 대상**이다.

---

## §0.5 — 🔴 이 계보 전체를 관통하는 두 개의 실측 (다음 설계의 출발점)

이 두 개가 P43~P47이 왜 전부 실패했는지를 설명한다. 새 제안은 이 둘을 통과해야 한다.

### (1) 이중 중복 — dense 경로와 query 경로가 각각 단독으로 전체 성능을 재현한다

**2026-08-05 S1 측정** (`tools/module_ablation.py`, MUSES 4모달 seed2 ep260, 조건당 60장 —
원시 = `experiments/monitor-log.md` 2026-08-05 12:30 KST, 도구 = `p39_dense_off` 토글 신설 커밋 `a8016ea`):

| 조건 | base | query_off | **dense_off** | router_off | trunkexp_off | **쿼리 단독 mIoU** |
|---|---|---|---|---|---|---|
| day | 75.26 | −0.25 | **+0.60** | +0.54 | +3.51 | **74.66** |
| night | 75.87 | +0.63 | **+0.38** | +6.07 | +6.86 | **75.49** |
| fog_night | 40.50 | +0.13 | **−0.29** | +3.20 | +1.71 | **40.79** |

- **dense 단독 ≈ 전체의 99%, 쿼리 단독 ≈ 전체의 99%.** 양쪽 어느 하나를 꺼도 성능이 거의 그대로다
  = **두 경로가 완전 중복** 상태.
- 선행 측정과 정합([analysis/2026-08-05-p46-module-ablation-query-nooop.md](../experiments/analysis/2026-08-05-p46-module-ablation-query-nooop.md), DELIVER):
  `p39_query_off` 평균 Δ **−0.09**, `pred_agreement` **0.9883~0.9903**, `feat_cos` 1.0
  — 쿼리를 꺼도 픽셀의 98.9%가 동일 예측.
- **기제**: P39-V5의 path dropout이 쿼리에 강제한 과제가 *semantic*이고, 그건 dense가 이미 푸는 문제다.
  쿼리 경로의 최적해가 **"dense를 베끼는 것"** 이 되어버렸다.

🔴 **함의(가장 중요) — 이미 이중 중복인 시스템에 모듈을 더 얹는 형태는 중복에 흡수된다.**
P43(독립 주손실 헤드)·P44(gradient 통합/상호증류)·P46(prototype/RCS/MCC)·P47(uni-modal head)이
**전부 이 형태**였고 전부 P39.1을 못 넘었다. → **"기존 경로 위에 모듈/손실을 얹어 semantic mIoU를 올린다"는
접근 자체가 반증됐다.**

⚠️ 단서: **MUSES에서는 router/trunk_exp가 night에 크게 기여**(+6.07/+6.86)하는데
**DELIVER에서는 그렇지 않다**(+0.78/+1.11). **데이터셋 간 모듈 기여를 교차 인용하지 말 것.**

### (2) 피쳐 통계 — 압축은 양성, 모달 잉여의 기제는 둘로 갈린다

**2026-08-05 `tools/feature_stats.py`, DELIVER test 실측** (⚠️ 이 측정 전용 분석 문서는 **아직 없음** — 수치의 현재 출처는 이 절이다):

| 지표 | img | depth | event | lidar | FUSED_pf | PREHEAD | FUSED |
|---|---|---|---|---|---|---|---|
| eff.rank | 55.9 | 48.9 | 97.8 | **231.3** | 37.5 | 19.0 | **4.68** |
| η²(클래스분리 정렬) | 0.158 | 0.165 | 0.096 | **0.033** | 0.136 | 0.317 | **0.710** |

cross-modal CKA: img~depth 0.73 / img~event 0.72 / **img~lidar 0.34**. stage CKA: **lidar~PREHEAD 0.22(최저)**.

**해석(그대로 인용할 것)**:
1. **랭크 231 → 4.7 감소는 병리가 아니라 양성 압축**이다 — η²가 단조 상승(0.136 → 0.317 → 0.710)한다.
   "rank 붕괴를 고치면 성능이 오른다"는 P39.1/P41 시대의 전제는 여기서도 지지되지 않는다.
2. **모달 잉여의 기제가 둘로 갈린다** —
   **depth/event는 중복**(CKA 0.59~0.74로 img와 겹침), **lidar는 무관**(CKA 0.34인데 η² 0.033
   = 겹치지도 않는데 클래스 분리에도 기여 안 함 = **고엔트로피 노이즈**). 같은 "잉여"라도 처방이 달라야 한다.
3. **FUSED의 eff.rank 4.7 / idim90 10인데 DELIVER는 25 클래스** → **판별 차원 부족**이
   class-transfer(지배 오류, 복구 상한 **+7.9pt**)의 기계적 설명 후보다. 미검증 가설.

### (2-b) 입력 스케일 불일치 — 실재하나 원인은 아니고, 🔴 노벨티도 아니다

- `augmentations_mm.Normalize`가 **`img`만** /255 후 ImageNet 정규화하고 나머지 모달은 /255만 한다.
  `semseg/models/reliadino/encoder.py:273`(`self.backbone.forward_features(x)`)은 **모달별 정규화 없이**
  frozen ViT에 그대로 투입한다 → img ≈[−2.1,+2.6] vs depth/event [0,1] vs **lidar [0,0.38]**(실측).
- **정합화는 필요하다.** 그러나 위 (2)의 피쳐 통계로 **"lidar 표현 붕괴의 원인"이라는 가설은 반증**됐다
  (lidar의 랭크는 231.3으로 최고다 — 붕괴하지 않았다).
- 🔴 **이건 버그 수정이지 노벨티가 아니다**(user 지정 2026-08-05). 논문에서 기여로 주장하지 말 것.

---

## P27: Additive Attention Bias on Cross-Modal Memory Attention (RBMA 전구체, 2026-04-14)

**① 무엇을 바꿨나** — `LoRA_Sam_P27`(`semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py`) 신설.
P26의 SQG(SpatialQualityGating) logit을 **memory cross-attention의 pre-softmax logit에 additive bias**로 주입.
SAM2 memory attention이 `MultiheadAttention`이 아니라 **`RoPEAttention`**(SDPA에 attn_mask 없음)이라
`sam3/sam/transformer.py`의 SDPA를 `_rbma_attn_bias`로 패치해 bias 경로를 뚫었다.
`lambda_bias`(학습 스칼라, init 1.0) 1개만 추가. `train_sam2_lora_paper.py`의 `QUALITY_GATE_MODELS`에 등록.
configs: `b200-deliver_rgbdel_P27_physaug.yaml`, `levine-multiaqua_rgbtl_P27_hardaug8_physaug.yaml`.

**② 왜** — P24~P26 SQG 계열이 야간에서 **정적 RGB-붕괴 + lidar/thermal 평탄**으로 진단됐고
(`../status/history-2026H1.md`), SQG 예측기가 frozen-encoder feature로 학습돼 underfit이라는 원인이 나왔다.
"신뢰도를 feature에 곱하는" 대신 **attention 경쟁에서 누르되 Value는 보존**(정보 병목 회피)하자는 것이 P27의 동기.
이 기구가 이후 P28~P34의 **RBMA 배관 전체**가 된다.

**③ 결과** — 🔴 **P27 단독의 학습 수치 기록 없음.** 구현·config·1-iter dry-run 기록만 존재한다.
계보의 실측은 P28부터.

**④ 판정** — **기구로는 성립(배관 확보), 성능 판정 불가.**
다만 이 기구 위에 올라간 후속 5세대(P28·P31·P32·P34 λ1/λ2)가 **전부 Δ≈0 또는 순손해**로 나왔으므로,
P27의 주입 지점 자체가 **사후적으로 반증**됐다고 봐야 한다.

**⑤ 🔴 재시도 금지**
- **pre-softmax additive bias로 모달 신뢰도를 주입하는 것** — 4세대·2백본(SAM2/DINOv3)에서 효과 0 또는 순손해.
- ⚠️ 부수 교훈: `P27.forward`가 `_fuse_outputs` 훅을 호출하지 않아 **P30 router가 200ep 내내 실행되지 않았다**
  (ISSUE-022). **상속 계보에 훅을 추가할 때는 부모의 forward가 실제로 그 훅을 부르는지 확인할 것.**

---

## P28: RBMA — Reliability-Biased Memory Attention (2026-06-15)

**① 무엇을 바꿨나** — `LoRA_Sam_P28(LoRA_Sam_P27)`. P27의 배관은 그대로 두고 **bias 신호만** 교체:
SQG 예측 → **training-free self-entropy** `rel_i = 1 − H(softmax(D_i(f_i)))/log C`.
학습 파라미터 추가는 λ 1개뿐. `_compute_bias_source` 훅 신설(기본 identity=SQG라 P27 하위호환).

**② 왜** — SQG 예측기 자체가 underfit이라는 P25/P27 진단(`../status/history-2026H1.md` 진단 1~3).
"신뢰도를 학습으로 추정하지 말고 **모달 자신의 예측 불확실성**에서 공짜로 뽑자."

**③ 결과** — DELIVER 4모달(img/depth/event/lidar), PhysAug **ON**.
- B200 RUN-1: **val 57.87@ep12 / test 50.61@ep12** (ep16 사망, val-best 기준) — `experiments/registry.md`
- ⚠️ **모순되는 기록**: 이후 문서들이 P28 비교 기준을 **"55.27 / 63.40"**(test/val)으로 인용한다
  (`experiments/log.md` §P32, `../status/current.md`). 같은 P28의 다른(더 긴) 런으로 보이나
  **그 런의 로그·ckpt 경로가 registry에 없다.** 두 수치를 섞어 쓰지 말 것.

**④ 판정** — **불완전 실패(조기 사망).** 신호는 계보 표준이 됐지만 성능 판정에 필요한 완주가 없다.
후속 분석에서 이 계열의 근본 문제가 드러났다: **reliability AUROC가 geometry 모달에서 반전**
(img .77 / depth .62 / **event .30 / lidar .22** — 틀린 곳에서 과확신)이라 bias 신호 자체가 무의미했다
([analysis/2026-06-30-p28-p29-failure-analysis.md](../experiments/analysis/2026-06-30-p28-p29-failure-analysis.md) §7).

**⑤ 🔴 재시도 금지**
- **self-entropy를 보정 없이 신뢰도로 쓰는 것** — geometry 모달에서 anti-calibrated임이 실측됨.
  (P31의 calibration loss가 lidar AUROC .38→.97로 이걸 수리했다. 신호를 쓰려면 calibration이 선행이다.)

---

## P29: SDC — Self-Derived Condition 라우팅 (2026-06-27)

**① 무엇을 바꿨나** — `LoRA_Sam_P29(LoRA_Sam_P28)`. RBMA 배관 유지, **Soft-MoE LoRA의 gate 조건화**를 교체:
label-free image-derived 조건 latent + prototype bank → FiLM으로 MoE gate 변조.

**② 왜** — P10~P27 전 세대에서 관측된 **"MoE gate 상수수렴"**(ISSUE-002/015). 원인 진단 =
"조건이 라우터에 부재 + zero-init 가산 bias + 무감독 soft-softmax". 조건을 **라벨 없이** 이미지에서 파생해 주입.

**③ 결과** — DELIVER 4모달, PhysAug ON, B200.
**val 63.20@ep100 / test 54.34@ep146** (val-best) — 목표(66.51/56.71) 대비 −3.31/−2.37.
ep34부터 60~63 정체. `experiments/registry.md` · `../status/current.md`.

**④ 판정** — **정체(미달).** P28 대비로는 전진했으나 목표선에 크게 못 미쳤고, 진단이 더 중요했다:
`drop-modality Δ = [img 8.4, depth 23.5, **event 0.02, lidar 0.01**]`인데
융합 가중치는 거의 uniform `[.27,.28,.23,.23]` → **라우터가 모달을 적응적으로 선택하지 못함**(질량 ~45% 낭비).
이 진단이 P30 router와 P31의 직접 동기다.

**⑤ 🔴 재시도 금지**
- **무감독 soft-softmax gate**(anchor·직접감독 없는 학습 gate) — P10~P29에서 반복적으로 상수수렴.
  gate를 쓰려면 anchor(P30/P36 방식) 또는 직접 CE 감독(P39 방식)이 필수.

---

## P30: Class-token decoder + Reliability-anchored learned router (2026-06-28)

**① 무엇을 바꿨나** — 두 기구 동시 투입: ① **CTD**(class-token decoder)가 conv head를 **대체**,
② 고정 UAMM scalar를 **reliability로 anchor한 학습 router**로 교체
(`w = softmax_m(learned_logits(f_i) + λ·rel_i)`, conv head zero-init → 초기 w는 reliability 구동, per-class).

**② 왜** — P28 실패분석의 rare-class collapse(Water/Bridge=0) + event/LiDAR 미사용(Δ≈0) 직격.

**③ 결과** — DELIVER 4모달, PhysAug ON, B200.
**val 49.76@ep136 / test 44.10@ep146** — P29 대비 **−13.4 / −10.2**. 🔴 dead(회귀 확정).
det에서도 재현: P30-Det mAP50 0.256 (P29-Det 0.446 대비 붕괴, 소물체 사망).

**④ 판정** — **명백한 실패, 그러나 원인 귀속이 뒤집혔다.**
🔴 **ISSUE-022**: `P27.forward`가 `_fuse_outputs` 훅을 부르지 않아 **router는 200ep 내내 실행되지 않았다**.
→ 이 −13.4/−10.2는 **router의 실패가 아니라 "CTD가 conv head를 즉시 대체한 것" 단독의 효과**다.
실제로 같은 router를 P31/P36에서 **훅 픽스 후** 다시 켰을 때는 계보 유일의 대형 유효 모듈이 됐다.

**⑤ 🔴 재시도 금지**
- **query/class-token decoder로 conv head를 즉시 대체하는 것** — 소물체·thin-class 붕괴. seg·det 양쪽 재현.
  (P38·P43은 이 교훈으로 "대체" 대신 "병렬/잔차"를 택했다. 단 그건 그것대로 §0.5(1)의 중복 문제를 낳았다.)
- ⚠️ **"P30 router가 실패했다"는 서술을 재인용하지 말 것** — 미실행이었다.

---

## P31: Calibrated Dual-Reliability RBMA + Multi-scale HR Class-Token Decoding (2026-07-02)

**① 무엇을 바꿨나** — ① per-modal temperature + **correctness-contrastive calibration loss**(신뢰도 수리),
② `ClassTokenDecoderMS`를 **aux-only로 강등**(P31.1 수정 — 최종 출력은 SAM decoder 복원, CTD는 training-only aux CE),
③ Hiera 마지막 3 block unfreeze(LR×0.1), ④ router 'decisive' reg, ⑤ SDC OFF.

**② 왜** — P30의 붕괴(당시엔 CTD 대체가 원인으로 추정)를 되돌리면서, P28 진단의 anti-calibrated AUROC를
calibration loss로 직접 수리. `../decisions/2026-07-02-p31-redesign-proposal.md`.

**③ 결과** — DELIVER 4모달, PhysAug ON, B200.
**val 63.20@ep106 / test 54.75@ep158**(registry) — P29 대비 test +0.41, val 동률.
⚠️ **모순되는 기록**: `../status/current.md` P32 행은 P31 test를 **54.85**로 인용한다(registry는 54.75).
- ✅ **calibration은 성공**: lidar reliability AUROC **.38 → .97**, 계보 최초 4모달 균형 AUROC 달성.
- ✅ **router는 대형 유효**: off 시 −10.7~13.8 (SAM2 계보), thin-class 회복 실적.

**④ 판정** — **성능은 정체, 기제 2건은 성공.** 이 세대가 남긴 것은 수치가 아니라
**"calibration은 신호를 고치고, router는 thin-class를 살린다"** 는 두 개의 재사용 가능한 사실이다.
반대로 **RBMA bias는 여기서도 Δ≈0**이었다.

**⑤ 🔴 재시도 금지**
- **CTD를 최종 출력 경로에 두는 것**(P31.1이 이미 aux-only로 후퇴시킨 설계). P30 항목과 동일.
- 없음(신규 반증 없음) — 이 세대의 두 기제(calibration·router)는 **유지 대상**이다.

---

## P32: CoRB — Corroboration-Biased Memory Attention (2026-07-06)

**① 무엇을 바꿨나** — `LoRA_Sam_P32._compute_bias_source` override 하나. RBMA 배관 그대로,
**신뢰도의 의미만** self-entropy → **cross-modal corroboration + unique-info veto**로 교체
(`corr_i` = Bhattacharyya 계수, `g_i` = threshold-free veto gate, `rel_i = g_i·selfent_i + (1−g_i)·corr_i`).
학습 파라미터는 λ 1개. config `b200-deliver_rgbdel_P32_physaug.yaml`.

**② 왜** — "자기확신"이 per-modal decoder 용량과 confound되니 **"상호검증"** 으로 재정의하자.
Phase-0 무학습 게이트를 통과했다: event/lidar AUROC **.30/.22 → .54/.81 반전**
([analysis/p32-phase0-results.md](../experiments/analysis/p32-phase0-results.md)).

**③ 결과** — DELIVER 4모달, PhysAug ON, B200. 🔴 **같은 실험에 두 계열의 수치가 기록돼 있다:**
| 출처 | 값 | 맥락 |
|---|---|---|
| `experiments/log.md` §P32 | test **53.45**@ep40 / Day-Val **61.65**@ep30 | plateau 시점(ep26~30) 스냅샷 |
| `../status/current.md` P32 행 | 최종 Day-Val **64.12**@ep98 / test **55.00** | 완주분 |
| `experiments/log.md` §P33.1 "비교 기준" | P32 final test **55.01**@ep158 / val 64.12@ep98 | 완주분(test 55.00 vs 55.01 불일치) |

**④ 판정** — 🔴 **실패, 그리고 계보에서 가장 결정적인 반증.**
4축 독립검증 결과 **CoRB attn-bias는 유의한 순손해**(ΔmIoU −0.013, **p = 4.5e-22**).
기제: **신호 AUROC ≠ routing 이득.** corroboration 신호는 좋아졌는데
`drop-modality Δ [img 6.2, depth 15.6, **event ~0, lidar ~0**]` = event/LiDAR는 **여전히 미사용**.
**soft attention-bias로는 feature/decoder가 약한 모달(competence≈0)을 부활시킬 수 없다.**

**⑤ 🔴 재시도 금지**
- **RBMA/CoRB attn-bias 계열 전체** — 여기서 통계적으로 사망(p=4.5e-22). 4세대·2백본에서 효과 0 또는 순손해.
- **"신뢰도 신호의 AUROC를 개선하면 성능이 오른다"는 전제** — 반증됨. 신호 품질과 mIoU 사이에 경로가 없다.
- **soft 융합으로 죽은 모달을 살리려는 시도** — competence≈0인 모달은 가중치를 올려줘도 살아나지 않는다.

---

## P33: CG-MoD — Competence-Gated Hard Fusion + Modality Dropout (2026-07-07)

**① 무엇을 바꿨나** — `LoRA_Sam_P33(LoRA_Sam_P32)`, config-gated 3기구:
**M1** competence-weighted hard fusion(TAU 0.25, top-k 정규화) / **M2** 비대칭 modality dropout /
**M3** calibration 복원. 신호는 corr_veto가 아니라 **calibrated self-entropy**를 썼다
(corr_veto는 죽은 lidar를 0.85로 오상향해 융합 신호로 부적합 — P33 설계 문서의 실측).
설계 = [decisions/2026-07-07-p33-cgmod-design.md](../decisions/2026-07-07-p33-cgmod-design.md).

**② 왜** — P32가 남긴 "신호는 맞고 라우팅이 실패"를 1:1로 처방. soft bias가 안 되니 **hard 선택 + 강제 사용**.

**③ 결과** — **P33.1**(M1+M3, C1 단독) 학습 시작 2026-07-09, B200 GPU2-5, 200ep.
🔴 **완주 수치·판정 기록 없음.** `experiments/log.md`에 launch 기록과 스모크 PASS만 있고 결과가 없다.
(B200 마감이 07-15였고, 그 시점 우선순위가 P34-ReliaDINO로 넘어간 것으로 보이나 **명시적 종료 기록은 없다.**)
- 다만 **무조건 modality dropout의 no-op**은 이후 세대들이 P33의 결과로 반복 인용한다
  ([decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md](../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md),
  [decisions/2026-07-23-p42-lidar-forcing-proposal.md](../decisions/2026-07-23-p42-lidar-forcing-proposal.md)) — 근거 로그는 미확인.

**④ 판정** — **미완결(기록 소실).** 성능 판정 불가.

**⑤ 🔴 재시도 금지**
- **무조건(unconditional) modality dropout** — 우리 P33 no-op 인용 + 외부 문헌(2403.04245)의 역효과 실증.
  드롭아웃을 쓰려면 **조건부**(P40 RCA)거나 **균형 분할·ramp**(P42)여야 한다.
  ⚠️ 단 우리 쪽 1차 근거 로그가 없으므로, 재시도를 원한다면 **먼저 P33.1 로그를 찾거나 재측정**할 것.
- **P33-v2의 CLIP-text anchor** — "내부신호만" 방침으로 폐기됨(P46 제안서 §노벨티가 이를 확인).

---

## P34: ReliaDINO — DINOv3-L frozen + per-modal LoRA (계보 전환점, 2026-07-13 완주)

**① 무엇을 바꿨나** — **백본 교체**. SAM2 Hiera → **DINOv3 ViT-L/16 frozen** + **per-modal LoRA(r8)**
+ cross-modal attention fusion + SimpleFPN + FPNSegHead + reliability gate/calib/veto + RBMA attn-bias(λ1)/consistency(λ2).
신규 패키지 `semseg/models/reliadino/`(`encoder.py`/`fusion.py`/`model.py`) — 이후 P35~P48 전부가 이 코드베이스다.
모달별 LoRA는 하나의 텐서 0번 축에 쌓는다(`MultiModalLoRAQKV.a_q/b_q/a_v/b_v` = (M,…)).

**② 왜** — P29~P32 표준분석의 결정적 발견
([analysis/2026-07-12-p29-p34-standard-analysis.md](../experiments/analysis/2026-07-12-p29-p34-standard-analysis.md)):
SAM2 계열 피쳐가 **rank-1 붕괴**(depth 1.1 / fused 1.26) + **모달 비정렬**(CKA ~0.1)인데
DINOv3는 rank 10~20 + 정렬 0.85. **백본이 지배 변수**라는 통제 probe(+11.6)까지 확보.

**③ 결과** — 🔴 **PhysAug ON = 공정선 밖. 헤드라인으로 쓰지 말 것.**
| 데이터셋 | 프로토콜 | val | test |
|---|---|---|---|
| DELIVER 4모달(idel) | **val-best**, PhysAug **ON** | 68.19@ep120 | **56.62** |
| DELIVER 4모달 | final-iter | 기록 없음 | 기록 없음 |
| MUSES 3모달(ile) | val-best (공식 재평가) | 80.86@ep276 (내부 81.02) | **공식 test 78.979** |
| MUSES 4모달(iler) | val-best | 기록 없음 | 공식 test 78.256 — **ISSUE-025 오염, 무효** |
- 참고: "test 57.60@ep140"은 **test-best라 사용 불가**(2026-07-15 철회).

**④ 판정** — ✅ **계보 최대의 성공. 단 성공의 원천이 제안 모듈이 아니다.**
module_ablation 실측: **attn-bias(λ1)/consistency(λ2)/veto ≈ 0.00 (no-op)**, gate ±0.7 혼재, calib 평균 −0.3.
→ **"P34의 우위는 백본 + per-modal LoRA + FPN 구조 자체"** 다.
그리고 P34→P35 하락(−1.12)의 **전부가 physaug 제거분**(Static −13.8, Pole −5)이었다
= **증강 하나가 우리 제안 모듈 전체보다 큰 단일 변수**다.

**⑤ 🔴 재시도 금지**
- **RBMA attn-bias / consistency / veto를 DINOv3 계보에서 다시 켜는 것** — 3세대 연속 no-op 재확인.
- **P34 수치를 공정 비교선으로 쓰는 것**(PhysAug ON). §0-b 참조.
- ✅ 반대로 **유지 대상**: DINOv3-L frozen + per-modal LoRA 프레임. "건드리지 말 것"이 실패-키 D-1의 결론이다.

---

## P35: 공정 레시피 동결 (P34 − ATTN_BIAS − CONSISTENCY − PhysAug, 2026-07-15)

**① 무엇을 바꿨나** — 아키텍처 변경 0. **레시피만 동결**: RBMA attn-bias off, consistency off, **PhysAug off**,
DGFUSION_AUG on. config diff 실측 기준 `P35 = P34 − ATTN_BIAS − CONSISTENCY − PhysAug`.

**② 왜** — DGFusion/CAFuser와 **같은 증강 조건**에서 비교하기 위한 공정선 구축.
P34의 우위가 증강 덕이라는 의심을 통제하려는 것.

**③ 결과** — DELIVER 4모달, **val-best, PhysAug off**: **val 67.61@ep78 / test 55.52**. final-iter 기록 없음.

**④ 판정** — **의도한 대로 작동(공정선 확립), 성능은 −1.12.**
이 −1.12가 **physaug의 실제 크기**이고, 이 수치가 이후 모든 "제안 모듈 vs 증강" 논쟁의 기준이 됐다.

**⑤ 🔴 재시도 금지**
- **physaug를 켠 수치를 헤드라인·게이트 비교에 쓰는 것**(user 판정 2026-07-20).
  ablation 행으로만 병기한다.

---

## P36: Per-Class Reliability-Anchored Router (= P35 + router, 2026-07-15 완주)

**① 무엇을 바꿨나** — P35에 **per-class reliability-anchored router**(P31에서 포트) 하나 추가.
`fusion.py`의 gate 뒤(현행 코드 기준 `fusion.py:596~600` "4b) [P36] per-class reliability-anchored router" 블록):
anchor = **detached `rel_cal`**(training-free), 헤드는 **pre-fusion feats**를 보고,
routed logits가 per-modal aux logits를 재가중해 head 출력에 `router_alpha` 배로 가산된다.

**② 왜** — P31에서 router만이 유일한 대형 유효 모듈이었고 thin-class 회복 실적이 있었다.
DINOv3 계보에서도 재현되는지 확인하는 1-변수 실험.

**③ 결과** — DELIVER 4모달, **val-best, PhysAug off**:
- ✅ **val 67.74@ep52 / test 55.62** ← **이것이 공정·legal 내부최고 DELIVER다**(§0-b).
- final-iter 기록 없음. 완주는 ep200/200이나 best 이후 148ep 미갱신, val은 끝까지 61.45로 열화.
- P35 대비 **legal +0.13 val / +0.10 test**, D1 5-cond mean +0.76.
- router off 시 −38~42 (**의존이지 기여가 아니다** — 이 수치를 기여로 쓰면 자멸).
- thin-class 회복(D1): **Wall 6.0→13.3 · Water 5.3→9.5 · RailTrack 56.1→62.5**.
- P36+physaug(ep64 중단): val **68.76**(계보 최고) / test 54.18 — **공정선 밖 + test 꼴찌**라 헤드라인 불가.

**④ 판정** — ✅ **유일하게 살아남은 모듈 노벨티, 그러나 증분은 노이즈 대역(+0.10 test).**
[decisions/2026-07-16-p36-novelty-critical-review.md](../decisions/2026-07-16-p36-novelty-critical-review.md)의
판정을 그대로 옮긴다: **"현 상태로 '새 모듈 제안 + SOTA' 논문은 성립하지 않는다."**
router의 공격 지점 3개(MoE 변형 아니냐 / off Δ+40은 의존이지 기여 아님 / 제안 모듈 < 증강 하나)는 미해결.

**⑤ 🔴 재시도 금지**
- **`router_off Δ = +38~42`를 기여로 보고하는 것.** 그건 co-adaptation 의존도다. 증분 가치는 **+0.10~+0.76**뿐.
- **"RBMA" 브랜딩으로 성능을 주장하는 것** — 실측과 정면 모순. bias는 negative finding으로만 쓴다.

---

## P37a / P37b: CEFR-Head · ClassToken-lite-Learned (2026-07-17~18)

**① 무엇을 바꿨나**
- **P37a — CEFR**(Class-Expected Feature Routing): per-class 모달 라우팅 헤드를 만들어
  `σ(a)` blend(init 0.018)로 기존 gate-fused 경로와 섞는다. CA² anchor에 λ2(t)·log p̂ 항.
- **P37b — ClassToken-lite-Learned**(`semseg/models/reliadino/classtoken.py`): class-token decoder를 얹고
  `mask_proj`로 attn mask 예측.

**② 왜** — P36 router가 per-class로 작동한 것이 유일한 실적이었으니, **라우팅을 클래스 축으로 명시화**하면
더 크게 먹을 것이라는 기대. thin-class를 클래스 전용 토큰으로 직접 감독하려는 것(P37b).

**③ 결과**
- **P37a, MUSES 3모달, val-best**: **81.16@ep110** (P34-3모달 80.86 대비 +0.30). ep190에 81.57(미제출).
  DELIVER 학습분은 **ISSUE-026 오염**. test는 미제출 → 기록 없음.
- **P37b seg: 수치 기록 없음** — bengio GPU5 HW 고장으로 ep1~2에서 사망 확정(2026-07-18).
  det 계보에서는 측정됨: **Δ 0.0000 / agreement 1.000 = 완전 NO-OP**, 파라미터만 +2.7M
  (`det/det-architecture-map.md`).

**④ 판정** — 🔴 **둘 다 실패. 그리고 실패 방식이 서로 다르다.**
- **P37a: 구조는 채택됐으나 가설은 미실현**
  ([analysis/2026-07-18-p37a-muses-cefr-output-analysis.md](../experiments/analysis/2026-07-18-p37a-muses-cefr-output-analysis.md)):
  σ(a) 0.018→**0.121**(6.7× 개방)인데 **per-class 라우팅 분화는 0/19 커밋**,
  라우팅 엔트로피 1.085~1.092(max 1.099) = 사실상 uniform, **winner가 19/19 전부 event**(전역 틸트로 퇴화).
  `cefr_off` Δ **+0.16 = no-op**. → **"+0.30은 클래스 라우팅의 성과라고 주장할 수 없다."**
  ⚠️ P30 router의 "공간 평균 uniform은 측정 artifact" 함정과는 다르다 — **per-class로 갈라서 봐도 uniform**이다.
- **P37b: 버그로 실행되지 않음**(ISSUE-024) — `mask_proj`가 threshold 비교(비미분)로만 소비돼
  gradient를 전혀 못 받아 **영구 random init**. masked attention이 random 마스킹으로 동작했다.

**⑤ 🔴 재시도 금지**
- **CEFR class-expected routing** — per-class 미분화(0/19), 전역 재가중으로 퇴화. 실패-키 C-3.
- **zero-init/σ(a) blend로 새 경로를 "살짝 얹는" 결선** — 4연속 사장(m2f β 0.133, CEFR σ(a) 0.121, …).
  = **실패-키 1**, 이 계보 최상위 키.
- **무감독 threshold mask 게이트**(P37b `mask_proj`) — 영구 random. 마스크는 **예측을 리사이즈**해서 쓸 것
  (P38 `m2f_head.py::_attn_bias`가 올바른 패턴).

---

## P38: MaskQueryLite — Mask2Former-lite Query Head (2026-07-18 launch)

**① 무엇을 바꿨나** — P36 공정 레시피 **동결** + `semseg/models/reliadino/m2f_head.py` 신규.
100 learned query가 gated fused stride-16 map 위에서 6-layer masked cross-attn,
공유 cls(K+1)/mask-embed head를 매 layer에 적용(deep supervision).
손실 = Hungarian + CE(no-obj 0.1) + point-sampled BCE/Dice(2/5/5, 12544 pts), `aux['m2f_loss']`(LOSS_W 0.5).
출력 = `conv_head + β·sem_query + router_alpha·routed`, **β는 zero-init**.
`panoptic_inference()` 포함 = **PQ 산출 경로 최초 확보**.

**② 왜** — ① DGFusion/CAFuser가 OneFormer 스택이라 MUSES 주표가 PQ인데 우리 per-pixel head는 구조적으로 PQ 불가.
② mask-classification이 thin/희소 클래스에서 +1~3 mIoU 우세라는 문헌. ③ head를 통제 변수로 고정해
남는 성능차를 신뢰도 라우팅에 귀속시키려는 confound 제거.

**③ 결과**
| 데이터셋 | 프로토콜 | val | test |
|---|---|---|---|
| DELIVER 4모달 (⚠️ **ISSUE-026 오염**) | val-best | 65.19@ep28 | **기록 없음**(해당 ep의 test 미측정) |
| DELIVER 4모달 | (참고) test-best | — | 55.05@ep62 — **test-peeking, 사용 불가** |
| MUSES 3모달 | val-best | **82.22** | **공식 test 79.025** |
- D1 5-cond mean 53.66 = P36 −1.63 / P34 −1.99 (⚠️ ISSUE-026으로 **이 판정은 보류 상태**).
- MUSES 이득은 압도적으로 fog에서 나왔다(+6.64, fog_night +10.3) —
  P34-4모달이 fog에서 train IoU 0.00 완전사멸했던 것이 P38에서 100.00으로 복구
  ([analysis/MUSES_TEST_RESULTS_INDEX.md](../experiments/analysis/MUSES_TEST_RESULTS_INDEX.md)).

**④ 판정** — 🔴 **모듈은 실패, 부산물 두 개는 성공.**
- **실패**: `p38_m2f_off` Δ **+0.04~+0.12 = 추론 no-op**(β가 0.133까지만 열림 = 실패-키 1 재판).
  thin-class는 오히려 퇴행(Wall −6, RailTrack sun 붕괴) — "mask-cls로 thin-class 회복" 가설 **반증**.
- **성공 부산물 1**: `p36_router_off`가 +38~42 → **+1.6~2.4로 급감**.
  **deep-supervision이 router 단일 의존을 해소할 수 있다는 유일한 실증**(실패-키 D-4).
- **성공 부산물 2**: MUSES test 79.025로 당시 내부 최고 갱신, PQ 경로 확보.
- 정직한 정리: **"P38의 우위는 추론 시 m2f 로짓이 아니라 학습 시 deep-supervision"** 이고,
  그것조차 P36 fair를 못 넘었으므로 **"P37b 버그를 고친 것"이지 "P36 대비 전진"이 아니다.**

**⑤ 🔴 재시도 금지**
- **m2f semantic 잔차 헤드(β zero-init 결선)** — 추론 no-op + thin-class 퇴행. 실패-키 C-5.
- **"mask-classification 헤드를 달면 thin-class가 회복된다"는 문헌 기대** — 우리 세팅에서 반증(Wall −6).

---

## P39: DPC — Dual-Path Compete (2026-07-20)

**① 무엇을 바꿨나** — P38 위에 5개(전부 토글):
- **V1** trunk rank expansion `fused' = fused + Σ_m P_m(f_m)`(선형, small-random init — zero-init 아님)
- **V2** modal-token query attention(query가 fused map 대신 **per-modal 토큰 합집합**을 직접 attend)
- **V3** anchored(K개 클래스 고정) + free query
- **V4** balanced point sampling(클래스당 ≥256pt 쿼터)
- **V5** compete-and-arbitrate: **β 잔차 폐기** → 학습은 **path dropout 경쟁**(dense-only 25%/query-only 25%/결합 50%),
  추론은 per-class 중재 `final_k = dense_k + softplus(Λ_k)·query_k`, + **router 직접 CE(w 0.4)**
configs: `hpca100-deliver_rgbdel_P39_dpc.yaml` · `jarvis-muses_rgbel_P39_dpc.yaml`.

**② 왜** — [analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md](../experiments/analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md)의
5개 실패-키를 **규칙으로 역변환**해 설계에 내장한 첫 세대. 키1(zero-init 금지)→V5, 키2(router 직접감독)→V5,
키3(rank 병목)→V1·V2, 키4→V5 per-class Λ, 키5→V2.

**③ 결과**
| 데이터셋 | 프로토콜 | val | test |
|---|---|---|---|
| DELIVER 4모달 (⚠️ ISSUE-026 오염) | val-best | 65.68@ep64 (P38 65.19 첫 돌파) | **test 5-cond 평균 50.98 = 3시점 중 최저** |
| MUSES 3모달 | val-best | 81.52@ep146 | **공식 test 78.881** (P38 79.025 −0.144) |
- 🔴 **val↔test 순위 역전**: val 순위 ep64>ep38>ep60인데 test 순위 ep60>ep38>ep64.
  MUSES 공식 제출에서도 같은 역전(val 81.52 > P34 81.02인데 test 78.881 < 78.979)
  → **내부 val은 이 계보에서 모델 선택 지표로 신뢰할 수 없다**는 두 독립 증거.
- 주야 격차 5.14→3.73(−1.41) 개선했으나 **fog_night 62.68(−12.05, 전 제출 최저)** 가 전부 상쇄.
- DELIVER 손실의 대부분이 **RailTrack 단독 −20.4**(cloud 59.2→6.4).

**④ 판정** — 🔴 **기제는 성공, 성능 전환은 실패.**
- ✅ **5세대 만의 첫 non-no-op**: V1 off-Δ **+0.76~2.89**(양 벤치 최대 기여),
  V5 query MUSES 전 조건 +(최대 +1.09), **router 의존 +22~40 → +0.4~2.3으로 해소**, arb λ 0.69→1.0~2.3 성장.
  실패-키 처방(경쟁결합·직접감독·rank확장)이 **기제 수준에서 전부 유효**했다.
- 🔴 그런데 **성능이 안 따라왔다.** 원인 = query·router가 같은 클래스를 상충 점유(RailTrack)
  + gate/calib이 thin-class에 **유해**(off 시 +35.9/+26.0 — 3세대 no-op에서 **유해로 판정 변경**).
- thin-class 게이트 **세 시점 모두 0/3 미달.**

**⑤ 🔴 재시도 금지**
- **reliability gate/calib/veto를 켜두는 것** — no-op을 넘어 **thin-class·fog_night에 유해**로 재판정.
  P39.1의 M-2가 이미 config off 처리했다. 다시 켜지 말 것.
- **"모듈 토글 Δ가 크면 성능이 오른다"는 추론** — V1이 최대 기여인데 test는 최저였다. Δ는 필요조건일 뿐이다.

---

## P39.1: Rank 수리 — gated_mlp trunk + VICReg (2026-07-21) ★ 현행 기준선

**① 무엇을 바꿨나** — P39에서 V2·V3·V4·router 직접감독·deep-sup은 **동결**, V1만 교체 + 정규화 1종:
- **R-1**: `fused += P_m(f_m)`(선형) → **`fused += tanh(γ_m)·MLP_m(f_m)`**
  (LN→1×1(1024→256)→GELU→1×1(→1024), γ = 모달별 스칼라 **init 0.1**).
  🔴 γ=0으로 두지 않은 이유가 중요하다 — **0이면 tanh(0)=0이라 브랜치가 첫 스텝부터 gradient를 못 받아
  학습이 시작되지 않는다**(스모크로 실증 = 실패-키 1의 재판).
- **R-2**: per-modal 토큰에 **VICReg var+cov**(lidar ×1.0 / img·event ×0.25, λ_var 0.1 / λ_cov 0.01, 2048 서브샘플, fp32).
- **M-2**: gate/calib/veto config **off**(P39 유해 실증 반영).
- eval마다 per-modal effective-rank(RankMe)를 `p391/rank_*`로 로깅.

**② 왜** — P39-MUSES 표준분석이 **lidar effective-rank 4.7 붕괴**(adapter가 아니라 **트렁크가 압축 주체**)와
fog_night 62.68을 지목. 문헌 교차(deep matrix factorization / LoRA intruder dimensions):
V1의 선형 투영 + LoRA BA가 딱 "선형 cascaded 경로의 암묵적 저rank 편향" 구조이고,
**rsLoRA 없는 단순 r 상향은 무효**. 근거 = [decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md](../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md).

**③ 결과** — 🔴 **legal 수치 (모든 후속 세대의 비교 기준선)**
| 데이터셋 | 프로토콜 | val | test |
|---|---|---|---|
| MUSES 3모달, **seed2** | val-best | **82.62@ep208** | **공식 test 79.788** ← 우리 MUSES 최고 |
| MUSES 3모달, 5-seed | val-best | 82.03 / **82.62** / 81.89 / 81.92 / 81.70 (**평균 82.03, 범위 0.92**) | seed2만 제출 |
| MUSES 4모달(+radar), seed2 | val-best | 82.35@ep260 | **공식 test 79.571** (3모달 −0.217) |
| DELIVER 4모달(픽스 후 첫 클린 런) | **val-best** | 67.60@ep106 | **54.34** |
| DELIVER 4모달 | **final-iter(ep200)** | 65.88 | **53.95** |
- ⚠️ **seed 분산 0.92는 이 계보에서 논하는 대부분의 델타보다 크다.** 0.1~0.3 차이로 내린 판정들은 재검토 대상.

**④ 판정** — ✅ **기제 실증 성공 + MUSES 내부 최고.** 그러나 **여기서 4세대가 멈춰 있다.**
- R-2(VICReg)가 lidar eff-rank를 **78.5~100.3**까지 확장(P43의 VICReg-off 23.5~28.0의 3~4배) = 기제 실증.
- R-1(trunk) `p39_trunkexp_off` 전 조건 **+2.05~+6.78** 순기여, router +0.5~+4.5 순기여,
  drop-lidar 야간·adverse(fog_night 7.39 / snow_night 7.6 / rain_night 7.57) 인과 기여 확인.
- 흠: arbiter query가 일부 야간 조건에서 **미세 유해**(rain −0.26 / night −0.37 / clear_night −0.29)
  — §0.5(1)의 중복 문제가 여기서 이미 보이고 있었다.
- 🔴 **rank를 고쳐도(4.7→100) test는 P38 79.025 → 79.788로 +0.76에 그쳤고, 그 뒤 4세대가 이 선을 못 넘었다.**

**⑤ 🔴 재시도 금지**
- **완전 zero-init 잔차**(γ=0, β=0) — 학습이 시작조차 안 된다. 최소 init(γ=0.1)이 하한이다.
- **"per-modal rank를 올리면 성능이 오른다"의 추가 투자** — VICReg으로 rank를 20배 올렸는데 test는 +0.76이었고,
  **P41이 fusion 쪽에서 같은 가설을 결정적으로 기각**했다(아래).
- **단일 seed val-best 0.1~0.3 차이로 판정하는 것** — seed 분산 0.92.

---

## P40: RCA-Fusion — Reliability-Conditioned Attenuation (2026-07-21)

**① 무엇을 바꿨나** — P39.1 위에 C-1 lidar 리턴 유효성 통계(입력 유도 내부 신호, 가드/분석) +
**C-2 조건부 감쇠**(학습 중 자기추정 rel(img)가 배치 하위 30%인 샘플의 img feature를 soft 감쇠,
α∈[0.1,0.5], **hard-zero 금지**, curriculum ramp ep20까지) + **C-3 감쇠 샘플 한정 lidar readout 보조 CE**(w 0.5).
configs: `jarvis-muses_rgbel_P40_rca.yaml` · `hpca100-deliver_rgbdel_P40_rca.yaml`. 커밋 ac5c7fe.

**② 왜** — "신뢰도 기계가 5세대(P28→P39) 동안 *추론-시 재가중*으로는 반증 완주된 무효 시도였다.
P40은 같은 신호를 **학습-시 조건화**로 옮긴다." 무조건 드롭아웃은 역효과가 실증됐으므로(2403.04245 + 우리 P33)
**조건부 감쇠 + 약모달 보조손실**만 남은 레버라는 판단.

**③ 결과** — 🔴 **학습 미기동. 결과 기록 없음.**
구현 완료(develop ac5c7fe) + 합성 스모크 PASS까지만. 대기열 #2로 "P39.1 rank 게이트 통과 후 투입"이었으나
**투입 기록이 없다** — P41(fusion) → P42(lidar 강제) → P43~P45로 방향이 전환됐다.

**④ 판정** — **미실행(보류).** 성능 판정 불가.
⚠️ 단 **P39-DPC 분석에서 gate/calib이 "유해"로 재판정된 것**과, P44-BMR의 야간편중 lidar 사용 가설이
test에서 완전 반증된 것을 고려하면, P40의 전제("img를 죽이면 lidar가 산다")는 **간접적으로 흔들려 있다**.

**⑤ 🔴 재시도 금지**
- 없음(직접 반증 없음). **단 재시도 전에 P44-BMR 결과를 먼저 읽을 것** — P44가 "비RGB 사용을 늘린다"는
  같은 목표를 loss/gradient 레벨에서 시도해 **완전히 실패**했고(fog_night −13.2 파국),
  P42가 입력 마스킹 레벨에서 시도해 **역시 실패**했다. P40은 세 번째 각도의 같은 목표다.

---

## P41: FCR — Fusion Spectral Collapse / Fused Class-alignment Regularizer (2026-07-22~23)

**① 무엇을 바꿨나** — 2단 구조.
- **Phase 0 (학습 0)**: `tools/feature_stats.py`에 **LDA-rank(η²)** 와 **modality-ablated 스펙트럼** 추가.
  기존 P38 ckpt로 "fusion 저rank가 양성(neural collapse)인가 유해(EBR 모달 억압)인가"를 **학습 없이 판별**.
- **Phase 1**: **F-1 FCR** — fused(T3)에 supervised between-class 분산 규제
  `L_fcr = −λ·η²(fused, gt_mask)`(aux 손실, 주손실 경로, zero-init 잔차 아님).

**② 왜** — 실패-키 **키3**: per-modal rank 20~36인데 **FUSED rank 6.8~8.0/256으로 붕괴**,
제안 모듈들이 전부 fusion 이후·로짓 근처에서 작동해 no-op이 된 것과 정합.
"병목 위에서 뭘 더해도 안 변한다 → 융합 단계 자체를 건드려라."
근거 = [decisions/2026-07-22-p41-fusion-spectral-discrimination-proposal.md](../decisions/2026-07-22-p41-fusion-spectral-discrimination-proposal.md).

**③ 결과** — MUSES 3모달.
- **Phase 0**: FUSED_pf η² **0.32~0.35**(전 조건), FUSED(decode) η² 0.63
  → 저rank(9)인데 η²가 1이 아님 ⇒ **"neural-collapse 양성 압축이라 개입 무의미"라는 반대 가설은 반증**.
  P0-A: **img를 빼면 fused rank가 오히려 상승**(clear 9.21→14.53, night 7.02→9.48) = **img 과지배·압축**.
  단 fog는 반대(img 빼면 rank↓) = fog에서는 img가 진짜 캐리어.
- **Phase 1 (ep90+)**: **η² 0.35 → clear 0.9482 / fog 0.9339 / night 0.9381 = 2.7× 상승**,
  **mIoU는 P38과 사실상 동일**(ep42 79.83 vs P38 79.94, ep50대 80.4~81.2 = P38 수준).
  val-best/final-iter 최종 수치는 기록 없음(ep90+에서 게이트 부정으로 중단).

**④ 판정** — 🔴 **결정적 기각(airtight falsification). 이 계보에서 가장 깨끗한 negative다.**
사전등록한 falsification 케이스("η²↑인데 mIoU 불변이면 fusion은 병목이 아니다")가 **정확히 실현**됐다.
→ **fusion rank/η²는 성능 레버가 아니다.** fused를 거의 완전히 클래스-정렬(η² 0.94)시켜도 무이득.
**기제**: decode가 이미 클래스 정보를 추출하므로(P38 decode η² 0.63) **fusion 사전정렬은 head와 중복**이다.
- ✅ **방법론적으로는 성공**: 완주 후 실패를 발견한 P39.1/P40과 달리
  **학습0 Phase-0 판별 → 사전등록 게이트 → 조기 확정**으로 끝냈다. 이 패턴을 표준으로 삼을 것.
- 🔴 **2026-08-05 피쳐 통계가 이 결론을 재확인한다**(§0.5(2)): DELIVER에서도 η²가
  FUSED_pf 0.136 → PREHEAD 0.317 → FUSED 0.710으로 **단조 상승**한다 = 압축은 양성이다.

**⑤ 🔴 재시도 금지**
- **fusion rank / effective-rank / η²를 올리는 개입 전부** — 결정적으로 기각.
- **eff_rank 단독을 KPI로 쓰는 것** — 무정보 차원에 오염된다(2312.04000). 최소한 η²와 병용할 것.
  ⚠️ 그리고 **η²조차 성능을 직접 예측하지 못한다**(P41이 실증). 대리지표를 게이트로 삼지 말 것.

---

## P42: lidar-강제 — 조건부 균형 img 마스킹 (2026-07-23)

**① 무엇을 바꿨나** — **M-1**: 학습 배치의 FRAC 비율에서 **img 입력을 0으로** 마스킹해 융합이 lidar/event로 풀도록 강제
(`model.py::_p42_mask_img`, config `P42.MASK_IMG`, curriculum ramp WARMUP_EP 20, **추론은 항상 full-modality**).
M-2(per-modal aux deep-sup CE)는 P38에 **이미 존재**(`FUSION.AUX_CE_WEIGHT 0.5`).

**② 왜** — P41이 fusion을 기각한 뒤 병목을 재탐색한 fog 분석:
`drop-modality dMIoU(lidar,event) ≈ 0`(clear (21.8, 0.34, −0.03) / fog (15.0, 0.14, 0.42)) = **img 과지배, 비RGB 미사용**.
fog에서 img가 열화되면(fused mIoU clear 68 → fog 49) **폴백이 없어 붕괴**.
문헌: lidar는 "저정보가 아니라 미사용"(AnySeg MUSES lidar-only 32.13 / MM SAM-Adapter cam+lidar fog 74.12).
근거 = [decisions/2026-07-23-p42-lidar-forcing-proposal.md](../decisions/2026-07-23-p42-lidar-forcing-proposal.md).

**③ 결과** — MUSES 3모달, FRAC 스윕 (**전부 val-best; test는 미제출 = 기록 없음**):
| FRAC | 서버 | val-best | 완주 |
|---|---|---|---|
| 0.3 | jarvis | **81.53@ep218** | ✅ 300/300 |
| 0.5 | hpca100 | **80.85@ep124** | ✅ 300/300 |
| 0.7 | yeon | 79.13@ep96 (ep98 시점) | 🔴 **완주 수치 기록 없음** |
비교선: P38 82.22 / P39.1 5-seed 평균 82.03(최고 82.62).

**④ 판정** — 🔴 **실패(전 FRAC 미달), 그리고 단조 열화.**
FRAC이 클수록 나쁘다(0.3 81.53 > 0.5 80.85 > 0.7 79.13) = **img를 가릴수록 손해**.
게이트 ③(val ≥ P38 82.22)을 **어떤 FRAC도 통과 못 했다.**
게이트 ①(dMIoU(lidar) 상승)의 측정 기록은 없다 — 즉 **"lidar를 실제로 쓰게 됐는지"조차 확인되지 않은 채
성능만 떨어졌다.**

**⑤ 🔴 재시도 금지**
- **강모달 입력 마스킹/드롭으로 약모달 사용을 강제하는 것** — FRAC 0.3/0.5/0.7 전부 단조 열화.
  (P33 무조건 드롭아웃 no-op → P42 균형 마스킹 실패 → P44 B-3 국소 마스킹도 실패. **세 변형 모두 사망.**)
- ⚠️ P40 RCA(조건부 감쇠)는 이 계열의 **미실행 4번째 변형**임을 인지할 것.

---

## P43: PanopticDual — 독립 주손실 mask-classification 헤드 (2026-07-25)

**① 무엇을 바꿨나** — `semseg/models/reliadino/panoptic_head.py`(`MaskClsHead`) 신규 + `encoder.py` 블록 tap 훅.
**잔차 없음** — 두 헤드가 각자 주손실을 받는다: `L = L_pixel + λ(t)·L_mask`, 공유는 SimpleFPN 트렁크뿐.
스모크가 이를 assert한다(mask 손실만 backward → pixel head grad **정확히 0**, 그 역도 성립).
+ **T-2 lateral**: frozen DINOv3 중간 블록(5/11/17)을 forward hook으로 tap해 모달 간 **고정 균등 평균** 후
SimpleFPN {1/4,1/8,1/16}에 가산(**zero-init·게이트 아님** — 주 경로).

**② 왜** — P38의 β 잔차가 no-op이었으니(실패-키 1) **잔차를 아예 없애고 두 헤드를 경쟁시킨다**.
전략적으로는: MUSES mIoU 보드가 융합에 죽은 축(1위 GtA 82.39 카메라단독)이고
**PQ가 유일한 현실적 SOTA 축**(DGFusion 61.03, frozen-VFM 참가자 0)인데 우리는 PQ 산출이 구조적으로 불가했다.

**③ 결과** — MUSES 3모달: **val-best 82.51 → 공식 test 79.351** (seed2 79.788 대비 −0.44, **우리 2위**).
final-iter 기록 없음. 조건별: day 80.81(seed2보다 우세) / night 75.19(열세) / fog_night 67.76(열세).
- 표준분석: lidar eff-rank 23.5~28(**VICReg off인데도 건강** — P39-DPC의 4.7 붕괴는 트렁크가 자초한 것이고
  P43은 그 구조를 피했다는 뜻), LATERAL 기여 +0.3~1.9(feat_cos ~0.75, no-op 아님),
  router +4.7~11.3, drop-lidar fog_night **+7.19**.
- 🔴 **PQ 실측은 이 시점에 못 냈다** — MUSES panoptic GT 미확보로 판단했으나
  **실제로는 `/ailab_mat2/dataset/MUSES/gt_panoptic/`에 있었다**(2026-08-04 확인, 샌드박스 경로 가시성 문제).

**④ 판정** — 🔴 **실패(P39.1 미돌파). 실패-키 1을 완벽히 준수했는데도 못 넘었다는 것이 핵심이다.**
잔차 결선을 없애고 독립 주손실을 줬는데 결과는 −0.44. → **결선 방식이 문제의 전부가 아니었다.**
이것이 §0.5(1)의 이중 중복 발견으로 이어지는 첫 신호다 — 두 헤드가 **독립 손실을 받아도 같은 해로 수렴**한다.

**⑤ 🔴 재시도 금지**
- **"잔차 대신 독립 주손실을 주면 새 경로가 산다"는 처방** — P43이 정확히 그렇게 했고 못 넘었다.
  실패-키 1의 처방 (a)"주 손실을 직접 받게 하라"는 **필요조건이지 충분조건이 아니다.**
- ⚠️ **PQ가 안 나온다고 데이터가 없다고 단정하는 것** — 경로 가시성 문제였다. 데이터 부재는 실제 경로로 확인할 것.

---

## P44: BMR — Balanced Multimodal Reliability (+ P45 FogStyle) (2026-07-25)

**① 무엇을 바꿨나** — 전부 **loss/gradient/입력 레벨**(신규 학습 파라미터·zero-init 잔차·attn-bias·학습형 추론 게이트 0개):
**B-1** MMPareto gradient 통합(`mmpareto.py` 신규 — 주 CE와 per-modal aux CE의 gradient를 합산이 아니라
Pareto 방향+크기 복원으로 통합, allreduce 이후 결합·`no_sync` 필수) / **B-2** peer 상호증류(`p44.py`, 대칭 KL + relational correspondence) /
**B-3** coverage-pattern **국소** img 마스킹(P42 전역 drop의 국소화 승격) / **V-1** 결정론적 presence 재정규화(P44 유일의 추론 경로 변경) /
**P45** FogStyle(feature-space fog 스타일 섭동 + 일관성, img 브랜치 한정, **기본 off**).

**② 왜** — P42의 전역 마스킹이 실패했으니 **더 정교한 레벨**(gradient·국소 패턴)에서 같은 목표를 재시도.
사전등록 게이트 = **dMIoU(lidar) 0 근처 → >1** & CKA(img,lidar) ≥0.5 & MUSES val ≥82.22.

**③ 결과** — MUSES 3모달: **val-best 80.71@ep156 → 공식 test 78.429**.
seed2(79.788) 대비 **−1.36**, P38(79.025)·P34(78.979)보다도 낮다. final-iter 기록 없음.
🔴 **fog_night 56.443 = seed2 69.61 대비 −13.2pt 파국.**
DELIVER에서도 P44-BMR 66.31 < P39.1-rank 67.60.
- 사전등록 게이트 ①(dMIoU(lidar) >1) **미달**: drop-lidar day **−0.42** (seed2의 day 4.24보다 **낮다**).
- 유일한 특징은 lidar 사용의 야간 편중(fog_night 6.71 vs day −0.42).
- **P45 FogStyle은 별도 학습 기록 없음**(P44 위 토글, 기본 off).

**④ 판정** — 🔴 **완전 실패 + 가설의 test 반증.**
BMR은 **"비RGB 사용을 늘린다"는 자기 목표조차 달성하지 못했다**(drop-lidar가 오히려 baseline보다 낮다).
그리고 유일한 특징이었던 "야간편중 lidar 사용이 유리하다"는 가설은
**test에서 정확히 반대로 나왔다**(fog_night −13.2). **BMR 방향 종료.**

**⑤ 🔴 재시도 금지**
- **MMPareto / peer 상호증류 / coverage-aware 국소 마스킹** — 셋 다 이 런에 포함됐고 결과는 −1.36.
  (개별 ablation은 없으므로 "어느 하나가 나빴다"는 귀속은 불가하다. 그러나 조합은 사망.)
- **"lidar를 야간에 더 쓰게 하면 adverse 조건이 좋아진다"** — test에서 fog_night −13.2로 반증.
- **gradient-level 균형화 일반** — frozen backbone에서 gradient-modulation 계열은 지렛대가 약하다는
  P39.1 딥리서치 경고가 실측으로 확인됐다.

---

## P45: FogStyle (2026-07-25) — 미실행

**① 무엇을 바꿨나** — FIFO식 feature-space fog 스타일 섭동 + 일관성 손실, img 브랜치 한정. P44 위 토글, 기본 off.

**② 왜** — fog가 MUSES 진짜 병목(clear 75.85 / **fog 62.67** / night 78.05, −13pt)이라는 P41 이후 방향 전환.

**③ 결과** — 🔴 **단독 학습 기록 없음.** 구현·병합(35ddbe0)·동시-on 통합 검증 PASS까지만.

**④ 판정** — **미실행(보류).**

**⑤ 🔴 재시도 금지** — 없음. ⚠️ 단 재시도 전제로 삼던 P44가 사망했으므로, P45는 **base를 P39.1로 되돌려**
단독 변수로 다시 설계해야 한다.

---

## P46: CTR — Class-Transfer Recovery (RCS + MCC + Prototype) (2026-07-29 ~ 2026-08-05)

**① 무엇을 바꿨나** — `semseg/models/reliadino/p46.py` 신규, P39.1-rank base 동결. 전부 학습 전용:
- **C-1 RCS**: train 라벨 전수 스캔 → `P(c) ∝ exp((1−f_c)/T)`(T=0.01) rare-class 샘플링
  (`RareClassSampler`가 `DistributedSampler`를 대체) + 런타임 per-class CE EMA blend.
- **C-2 MCC**: 패치 마스킹(ratio 0.5 / patch 64) student vs **EMA teacher + 원본** → 마스킹 영역 pseudo-label CE(conf≥0.75).
- **C-3 PROTO**: per-class EMA prototype bank(K×D) + prototype-contrastive CE `CE(cos(f,P)/τ, y)`.
configs: `jarvis-deliver_rgbdel_P46_ctr_c3only_lam{005,01,015,02,03}.yaml` 등, `hpca100-muses_rgbel_P46_c3only_lam02.yaml`.

**② 왜** — DELIVER val→test 하락의 **지배 원인이 per-class 도메인 전이 붕괴**
(Wall 62→2, TrafficLight 81→13, Water 33→0, Bridge 46→0; per-domain spread는 작고 융합은 이미 천장).
복구 상한 +7.9pt. 근거 = [decisions/2026-07-28-p46-classtransfer-recovery-proposal.md](../decisions/2026-07-28-p46-classtransfer-recovery-proposal.md).

**③ 결과**

**(a) 구성요소 귀속** ([analysis/2026-07-30-p46-c3only-vs-c1c3-attribution.md](../experiments/analysis/2026-07-30-p46-c3only-vs-c1c3-attribution.md), ep40 중간 ckpt):
| 구성 | RailTrack test@768 | overall test@768 |
|---|---|---|
| base (P39.1) | 4.02 | 52.47 |
| C1+C3 | 59.10 | 54.92 |
| **C3-only** | **64.13** | **55.64** |
→ **C-1(RCS)을 빼는 게 더 낫다 = C-1은 순유해.** C-2(MCC)는 4090 OOM(ISSUE-028) → A100 재시도했으나
**SIGHUP으로 ep0 도달 전 사망 → C-2 순기여는 끝내 측정되지 않음(기록 없음).**

**(b) DELIVER λ 스윕 — 🔴 legal 두 프로토콜 병기**
([analysis/2026-08-03-p46-c3only-lambda-sweep.md](../experiments/analysis/2026-08-03-p46-c3only-lambda-sweep.md), @768 동일 프로토콜):
| λ | val-best@ep | test @ val-best | final-iter(ep200) test | final-iter val |
|---|---|---|---|---|
| — (base P39.1) | 67.60@106 | 54.34 | 53.95 | 65.88 |
| **0.05** | **68.57@62** | **55.62** | **55.69** | 65.81 |
| 0.1 | 67.79@70 | *미감사* | *미감사* | — |
| 0.15 | 67.02@92 (ep135 kill) | 54.63 | — | — |
| 0.2 | 67.47@118 | 54.60 | **55.69** | 66.78 |
| 0.2-seed2 | 67.74@62 | 55.55 | 55.31 | 65.71 |
| 0.3 | 67.83@170 | 54.52 | 55.04 | 67.67 |
- 🔴 "test 57.05 = SOTA 돌파"는 **test-best라 무효, 전면 철회**(2026-08-04).
  λ0.2-seed2의 56.30@ep146도 **test-peeking, 사용 불가**.
- **legal 최고 test ≈ 55.7** — DGFusion 56.71 대비 −1.0, MM SAM-adapter 57.35 대비 −1.65.
- ✅ base 대비 **효과는 실재**: test **+1.35(val-best) / +1.74(final-iter)**, val **+0.97**.
- **val과 test가 서로 다른 λ를 선호한다**(final-iter val 65.81→66.78→67.67 단조 증가 vs test 55.69→55.69→55.04).

**(c) MUSES 이식** — C3-only λ0.2, 3모달: **val-best 81.65@ep136 → 공식 test 79.023**
(seed2 −0.765, P38과 동률). 손해가 **clear/day에 집중**(val Δclear −1.72 / Δday −1.29; fog +0.16 / rain +0.21).

**④ 판정** — 🔴 **혼합. 기제는 실재하나 SOTA·내부최고 어느 쪽도 못 넘었고, 재현성 검증에서 최종 반증.**
- ✅ RailTrack 사전등록 falsifiable 게이트(4→≥40)는 **압도적 통과**(64.13) → class-transfer 가설 자체는 확증.
- 🔴 그러나 **RailTrack 회복이 overall 돌파로 직결되지 않았다** — 다른 붕괴 클래스가 천장.
- 🔴 **재현성 실패**: λ0.2-seed2가 원본과 같은 미달 대역으로 재현 → **λ0.2 SOTA 돌파 최종 반증**.
- 🔴 **EPOCHS200 과다**: λ0.2-seed2의 val이 ep62 67.74 → ep200 **65.71(−2.03)**.
  138 epoch 추가 학습이 val을 악화시켰다.
- ⚠️ **미해명**: RailTrack이 **val(18~20) < test(59~64)** 로 역전되는 현상이 전 런에서 일관 재현. 원인 미상.
- 🔴 **정정**: 공정 55.62 기준(§0-b)으로 재계산하면 λ0.2-seed2 val-best test 55.55는 **−0.07 = 사실상 동률**이다.
  "내부최고 미달 −1.07"이라는 기존 서술은 56.62 혼입에 기반한 것으로 **철회**한다.

**⑤ 🔴 재시도 금지**
- **C-1 RCS(rare-class sampling)** — C3-only가 전 지표에서 C1+C3를 상회. 순유해로 판정.
- **λ ≥ 0.3** — 악화 확정. λ 최적은 0.05~0.2 평탄.
- **λ0.2 + EPOCHS200** — val −2.03 하락. 후속 λ 실험은 EPOCHS를 재검토할 것.
- 🔴 **`Best:` 필드 두 개를 나란히 인용하는 것** — 학습 로그의 val `Best:`와 test `Best:`는 **독립적으로 갱신**된다.
  둘을 이어붙이면 자동으로 test-peeking이 된다. 이번 오보의 구조적 원인이다.
  **작업 규칙**: 스윕 표에서 순위를 매기기 전에 런마다 val-best epoch N을 찾고
  `[Test] epoch:N` 줄의 `mIoU:`를 읽어라.
- ⚠️ **미측정으로 남은 것**: C-2(MCC) 순기여, λ0.1의 legal test. 재제안 전 이 둘을 먼저 채울 것.

---

## P47-1: LiDAR 투영 밀도화 (구 D-1, 2026-08-03)

**① 무엇을 바꿨나** — **모델 코드 0줄 변경.** `DATASET.PROJ_DIR`만 SDK 기본 `projected_to_rgb` →
**`projected_to_rgb_dgf`**(DGFusion식 (7,7) 팽창 + motion-compensation, 유효 픽셀 32.6% = **4.99×**).
config `hpca100-muses_rgbelr_P47_d1_dgfproj_4modal.yaml`. base = P39.1-rank MUSES 4모달 seed2 동결.

**② 왜** — MUSES lidar 투영이 희소해 frozen ViT가 쓸 신호가 부족하다는 가설. DGFusion의 전처리를 그대로 이식.
사전등록 게이트: Primary **4모달 val ≥ 82.62**(3모달 역전), D-1 falsifiable = **drop-lidar day dMIoU ≥ 6**.

**③ 결과** — MUSES 4모달(img/lidar/event/radar), EPOCHS 300:
- **val-best 82.58@ep172** — 3모달 seed2 **82.62 대비 −0.04**.
- final-iter(ep300) **기록 없음**(2026-08-05 14:30 시점 ep294/300, ETA 15:30 — 이 문서 작성 시점 미완주).
- test는 **해당 없음** — MUSES는 학습 중 test 평가가 구조적으로 불가(GT 비공개). 공식 제출로만 획득 가능.
- drop-lidar day dMIoU(D-1 falsifiable 게이트) **측정 기록 없음.**
- 궤적: ep172 이후 **122 epoch 무갱신**(밴드 82.27~82.58).

**④ 판정** — 🔴 **음성 결과 확정(효과 없음).**
82.58은 밴드 상단 +0.04에 불과한 **노이즈 봉우리**이고, P39.1 5-seed 분산이 **0.92**라 −0.04는 구별 불가다.
→ **DGFusion식 투영 밀도화는 우리 모델에서 효과가 없다.**
- ⚠️ **파생 경고(중요)**: 같은 논리로 **내부최고 MUSES val 82.62 자체도 노이즈 봉우리일 수 있다.**
  val-best 단일 수치로 0.1~0.3 차이를 논한 판정들은 전부 재검토 대상이다.
- ⚠️ ISSUE-031: 이 런의 `BATCH_SIZE:1`이 A100(40GB) 재프로파일 없이 3090/4090 값을 그대로 쓴 것
  (실측 24.6GB/40GB = 60%). 프로세스 결함으로 등재, 이 런은 1-변수 순수성 때문에 변경하지 않음.

**⑤ 🔴 재시도 금지**
- **입력 투영 밀도화(dilation + motion-comp)** — 4.99× 밀도화가 −0.04. 데이터 전처리 축은 이쪽으로 소진됐다.
- **val-best 단일 수치의 ±0.3 이내 차이로 우열을 판정하는 것** — seed 분산 0.92, 밴드 폭 0.3.

---

## P47-2: UniBal — Uni-modal Balance (구 D-2, 2026-08-04)

**① 무엇을 바꿨나** — `semseg/models/reliadino/p47.py` 신규(`UniModalBalance`/`UniModalHead`/`OGMGE`).
각 모달 encoder(frozen ViT + per-modal LoRA) 출력 `feats[i]`에 **모달마다 독립인** 경량 head
(GroupNorm → 1×1 conv → K)를 달고 **동일 GT로 CE** → `aux['p47_2_uni']`로 λ_u pre-scale 후 주손실에 합산.
**추론 불변**(eval `|Δ|max = 0`), **추가 forward 없음**(feats 재사용 → ISSUE-028 무관), **DELIVER 무영향**.
선택 토글 **OGM-GE**(기본 off): per-modal 정확도로 앞선 모달의 **자기 LoRA 슬라이스 gradient만** 감쇠.
configs: `jarvis-muses_rgbel_P47_2_unibal_img.yaml`(arm① img-only) / `..._unibal_all.yaml`(arm② all).

**② 왜** — **within-method 실측**: 우리 모델에서 모달을 더할수록 나빠진다
(4모달 val 82.35 / test 79.571 < 3모달 82.62 / 79.788). 문헌 기제 = **modality laziness / greedy joint learning**.
자체 확증 = P46-C3의 손해가 **clear/day(RGB 주도 조건)에 집중**.
⚠️ **리더보드 역상관 논거는 2026-08-04 철회됨**(방법론 교란) — 통제 ablation은 단조 증가
(CAFuser Table IX: RGB 55.7 → +L 58.7 → +R 59.3 → +E 59.7). **유효한 근거는 within-method 실측뿐.**

**③ 결과** — MUSES 3모달, val-best(같은 시점 best-so-far 대조):
| 시점 | base(3모달 seed2) | arm① img-only(λ_u 0.4 전량) | arm② all(모달당 0.133) |
|---|---|---|---|
| ep30 | 78.73 | 78.45 (−0.28) | 79.16 (**+0.43**) |
| ep48 | 79.66 | 79.08 (−0.58) | 79.60 (−0.06) |
| ep60 | 80.73 | 80.05 (−0.68) | 80.43 (−0.30) |
| ep80 | 80.87 | **80.38 (−0.49)** ← user 승인 kill(ep83) | 80.83 (−0.04) |
| 최종 | **82.62** | (중단) | **81.93@ep182 (−0.69)** |
- arm② final-iter 기록 없음(2026-08-05 14:30 시점 ep250/300, ETA 20:10 — 작성 시점 미완주).
- test는 **해당 없음**(MUSES, 미제출).
- 4모달 arm(radar per-modal CE 확인용)은 **미기동**.

**④ 판정** — 🔴 **실패, 그리고 진단 자체가 실측으로 반증됐다.**
🔴 **결정적 증거 — per-modal 로그**:
| | arm①(img에 λ 0.4) | arm②(img에 0.133) |
|---|---|---|
| ep30 img acc / ce | 0.949 / 0.143 | 0.951 / 0.141 |
| ep49 img acc / ce | 0.957 / 0.120 | 0.956 / 0.124 |
→ **RGB에 3.2× 가중을 줘도 RGB uni-modal 성적이 20 epoch째 불변.**
= **"RGB가 under-optimize돼 있다"는 전제의 직접 반증.** RGB 보조 헤드는 이미 포화 상태다.
arm②의 초기 +0.43은 RGB가 아니라 **lidar(ce 0.362)·event(ce 0.407)에 감독이 붙은 것**에서 왔고,
그 이득조차 ep48에 소멸했다.

**⑤ 🔴 재시도 금지**
- **"RGB(지배 모달)가 under-optimize돼 있다"는 진단** — per-modal 실측으로 반증. 다시 쓰지 말 것.
- **uni-modal aux CE 추가 / OGM-GE 계열 gradient 변조** — 두 arm 모두 base 미달.
- **리더보드의 "모달 수 ↔ 순위" 역상관을 근거로 쓰는 것** — 방법론 교란. 철회됨.
  ✅ 대신 쓸 수 있는 정확한 서술: **"모달↑=성능↓는 벤치의 법칙이 아니라 우리 모델의 증상"**
  (통제 ablation은 CAFuser·DGFusion 모두 단조 증가).
- ⚠️ **radar 판정의 정밀화**: "radar 무익"이 아니라 **"주간엔 소폭 유익(+0.19), 야간엔 유해(−0.376, fog_night −5.37)"**
  ([analysis/2026-08-04-muses-radar-night-harm.md](../experiments/analysis/2026-08-04-muses-radar-night-harm.md)).
  그리고 CAFuser Fig.5의 정답지는 **"야간에도 radar를 믿지 마라"**(전 조건 5~7% 고정, 야간 폴백은 events·lidar가 받음)
  — 우리의 "야간에 radar로 폴백" 가설과 정확히 반대 방향이다.

---

## P48: 쿼리 경로 인스턴스 감독 (2026-08-05) — 제안 단계

**① 무엇을 바꿨나(예정)** — `m2f_head.py:317`의 타깃 `torch.unique(gt_s4[b])`(= **클래스 단위**)를
**connected-component 인스턴스 단위**로 교체(P48-a) + Hungarian matcher 확장(P48-b).
stuff는 기존대로 클래스 병합. P48-c(task token)는 2차. **semantic 경로·추론 배선은 변경 0**(`|Δ|max == 0` assert).
제안 = [decisions/2026-08-05-p48-instance-supervision-proposal.md](../decisions/2026-08-05-p48-instance-supervision-proposal.md).

**② 왜** — §0.5(1)의 이중 중복. 쿼리가 받는 **유일한 감독이 semantic이고 그건 dense가 이미 푸는 문제**라
쿼리의 최적해가 "dense 베끼기"가 됐다. → **dense가 원리적으로 풀 수 없는 과제(인스턴스 분리)** 를 줘서
중복을 **구조적으로 불가능하게** 만든다. 부수 목표 = PQ의 병목인 RQ 붕괴 복구
(MUSES val **PQ 43.35 = SQ 79.51 × RQ 44.51**; SOTA PQ 59.26 대비 SQ는 −2.8인데 **RQ가 −28.6**).

**③ 결과** — 🔴 **미실행.** 학습 0 선행 측정만 완료:
- **S0 ✅**: MUSES `gt_panoptic/val.json` 실제 `segments_info` 전수 파싱 — things **8.66 inst/img**, singleton **39.0%**.
  GT things 2,165개를 클래스단위 타깃은 **584 마스크**로만 표현 → things RQ 상한 **42.5**,
  SQ 0.79 적용 시 **things PQ 상한 ≈ 33.6**(측정 18.33 = 상한의 절반 남짓).
- **S1 ✅**: §0.5(1)의 dense_off/query_off 측정 → **P48 진행 판정**(쿼리는 유능한데 과제가 겹쳐 안 쓰일 뿐).
- 🔴 **DELIVER는 P48 대상에서 제외**: COCO 주석이 `Human`/`Vehicle` **2개 카테고리뿐**이라 25개 시맨틱 클래스와
  정렬되지 않는다. **P48은 MUSES 전용.**

**④ 판정** — **제안 단계(미판정).**
사전등록 게이트: ep30 즉검 = `p39_query_off`의 `pred_agreement`가 **0.95 미만**으로 떨어질 것(현행 0.989),
미달 시 **kill**. 완주 = MUSES val PQ ≥ 46.9 **그리고** semantic val mIoU ≥ 82.22 유지.
🔴 falsifiable = **things PQ 33.6 초과**(30이 아니다 — 30은 현행 타깃 구성 안에서도 도달 가능해 검정력이 없다).
18.33~33.6 사이는 **기제 미확정**이므로 성공으로 보고하지 말 것.

**⑤ 🔴 재시도 금지**
- **DELIVER semantic GT를 connected-component로 세어 인스턴스 수 대용으로 쓰는 것** —
  CC는 43.89 inst/img인데 실제 COCO 주석은 3.33 inst/img로 **13배 과대**(CARLA 마스크 파편화).
- **"인스턴스 감독"을 노벨티로 주장하는 것** — 표준 기법(MaskFormer 계열 전부)이다.
  P48은 노벨티가 아니라 **정합화(버그 수정에 가까움)** 다 — panoptic head를 달아놓고 semantic 타깃으로 학습시키고 있었다.
- **things PQ 게이트를 30으로 잡는 것** — 검정력 없음(2026-08-05 상향 확정).

---

# 📊 표 1 — 세대 총괄 (P27~P48)

> 🔴 **모든 수치는 legal만**(val-best 또는 final-iter). test-best는 제외했다.
> `—` = 해당 없음, **`기록 없음`** = 측정·기록이 실제로 존재하지 않음(= 다음 작업 목록).
> MUSES test는 GT 비공개라 학습 중 산출 불가 — **공식 Codabench 제출분만** 기재했다.

| 세대 | 핵심 변경 | 데이터셋(모달) | legal val / test (프로토콜) | 판정 | 🔴 재시도 금지 |
|---|---|---|---|---|---|
| **P27** | memory-attn pre-softmax additive bias 배관 | DELIVER 4모달 | 기록 없음 | 기구만 성립 | pre-softmax additive bias 주입 |
| **P28** | bias 신호 = training-free self-entropy | DELIVER 4모달(PhysAug on) | 57.87@ep12 / 50.61 (val-best, ep16 사망) ⚠️ 55.27/63.40 병존 기록 | 조기 사망 | 무보정 self-entropy를 신뢰도로 사용 |
| **P29** | SDC label-free 조건 latent → FiLM gate | DELIVER 4모달(PhysAug on) | 63.20@ep100 / 54.34@ep146 (val-best) | 정체(미달) | 무감독 soft-softmax gate |
| **P30** | class-token decoder가 conv head 대체 + anchored router | DELIVER 4모달(PhysAug on) | 49.76@ep136 / 44.10@ep146 (val-best) | 🔴 붕괴 | query decoder로 conv head **즉시 대체** |
| **P31** | calibration loss + CTD aux-only 강등 + router | DELIVER 4모달(PhysAug on) | 63.20@ep106 / 54.75@ep158 (val-best) ⚠️ 54.85 병존 | 정체·기제 2건 성공 | CTD를 최종 출력 경로에 두기 |
| **P32** | bias 신호 = cross-modal corroboration + veto | DELIVER 4모달(PhysAug on) | 64.12@ep98 / 55.00~55.01@ep158 (완주) · 61.65/53.45(plateau) | 🔴 **순손해 p=4.5e-22** | attn-bias 계열 전체 · "AUROC↑⇒성능↑" |
| **P33** | competence-gated hard fusion + modality dropout | DELIVER 4모달 | **기록 없음**(P33.1 launch만) | 미완결 | 무조건 modality dropout · CLIP-text anchor |
| **P34** | 🔵 **백본 교체 DINOv3-L frozen + per-modal LoRA** | DELIVER 4모달(**PhysAug on = 공정선 밖**) / MUSES 3모달 | 68.19@ep120 / 56.62 (val-best) · MUSES 80.86 / **78.979** | ✅ 최대 성공(단 원천은 백본) | RBMA bias·consistency·veto 재점화 · P34를 공정선으로 사용 |
| **P35** | 공정 레시피 동결(−bias −cons −PhysAug) | DELIVER 4모달 | 67.61@ep78 / 55.52 (val-best) | 공정선 확립(−1.12) | physaug 수치를 헤드라인·게이트에 사용 |
| **P36** | + per-class reliability-anchored router | DELIVER 4모달 | **67.74@ep52 / 55.62 (val-best)** ← **공정 내부최고** | ✅ 유일 생존 모듈(증분 +0.10) | `router_off Δ+40`을 기여로 보고 |
| **P37a** | CEFR class-expected routing head | MUSES 3모달 | 81.16@ep110 / 미제출 (val-best) | 🔴 라우팅 0/19 미분화, no-op | CEFR · σ(a)/zero-init blend 결선 |
| **P37b** | ClassToken-lite-Learned | (seg 사망) / det | **기록 없음**(seg) · det Δ 0.0000 = NO-OP | 🔴 버그로 미실행(ISSUE-024) | 무감독 threshold mask 게이트 |
| **P38** | Mask2Former-lite query head(β zero-init 잔차) | DELIVER 4모달(⚠️ISSUE-026) / MUSES 3모달 | DELIVER 65.19@ep28 / **기록 없음** · MUSES **82.22 / 79.025** | 🔴 추론 no-op·thin-class 퇴행 | m2f semantic 잔차 헤드 · "mask-cls⇒thin-class 회복" |
| **P39** | DPC (V1 rank확장·V2 modal-token·V3 앵커·V4 쿼터·V5 경쟁) | DELIVER 4모달(⚠️ISSUE-026) / MUSES 3모달 | DELIVER 65.68@ep64 / 5-cond 평균 50.98 · MUSES 81.52 / **78.881** | 🔴 기제 성공·성능 실패, val↔test 역전 | gate/calib/veto 재점화(유해) · "토글 Δ↑⇒성능↑" |
| **P39.1** | ★ gated_mlp trunk(γ=0.1) + VICReg + gate off | MUSES 3모달 / 4모달 / DELIVER 4모달 | MUSES seed2 **82.62 / 79.788**(val-best, 5-seed 평균 82.03) · 4모달 82.35 / **79.571** · DELIVER **67.60 / 54.34**(val-best) · **65.88 / 53.95**(final-iter) | ✅ **현행 기준선** | 완전 zero-init 잔차 · per-modal rank 추가 투자 · 0.1~0.3 델타 판정 |
| **P40** | RCA 조건부 강모달 감쇠 + 약모달 readout aux | — | **기록 없음**(미기동) | 미실행 | 없음(단 P42·P44 실패를 먼저 읽을 것) |
| **P41** | FCR — fused η² supervised 규제 | MUSES 3모달 | η² 0.35→**0.94(2.7×)**, mIoU 불변(P38 수준) | 🔴 **결정적 기각** | fusion rank/η² 개입 전부 · eff_rank 단독 KPI |
| **P42** | 조건부 균형 img 마스킹(FRAC 스윕) | MUSES 3모달 | FRAC0.3 **81.53@ep218** / 0.5 **80.85@ep124** / 0.7 79.13(미완주) — test 미제출 | 🔴 실패(단조 열화) | 강모달 입력 마스킹으로 약모달 강제 |
| **P43** | PanopticDual — 독립 주손실 mask-cls 헤드 + lateral | MUSES 3모달 | **82.51 / 79.351**(val-best) | 🔴 실패(−0.44) | "잔차 대신 독립 주손실이면 산다"는 처방 |
| **P44** | BMR — MMPareto + peer 증류 + 국소 마스킹 | MUSES 3모달 / DELIVER | **80.71@ep156 / 78.429**(val-best) · DELIVER 66.31 | 🔴 완패(fog_night −13.2) | MMPareto·상호증류·국소마스킹 조합 · "야간 lidar 편중이 유리" |
| **P45** | FogStyle(feature-space 스타일 불변) | — | **기록 없음**(미기동) | 미실행 | 없음(base를 P39.1로 되돌려 재설계할 것) |
| **P46** | CTR — C1 RCS / C2 MCC / C3 prototype | DELIVER 4모달 / MUSES 3모달 | DELIVER λ0.05 **68.57@ep62 / 55.62**(val-best) · **65.81 / 55.69**(final-iter) · MUSES **81.65 / 79.023** | 🔴 SOTA 미달·재현성 실패(단 base 대비 +1.35~1.74는 실재) | C-1 RCS · λ≥0.3 · λ0.2+EPOCHS200 · 두 `Best:` 이어붙이기 |
| **P47-1** | LiDAR 투영 밀도화(코드 0줄) | MUSES 4모달 | **82.58@ep172**(val-best) / final-iter **기록 없음** · test 미제출 | 🔴 음성 확정(−0.04 = 노이즈) | 입력 투영 밀도화 · ±0.3 델타 판정 |
| **P47-2** | UniBal — 모달별 독립 aux head + uni-modal CE | MUSES 3모달 | arm② **81.93@ep182**(val-best, −0.69) / final-iter **기록 없음** · arm① 80.38@ep80 kill | 🔴 실패 + **진단 반증** | "RGB under-optimization" 진단 · uni-modal aux CE / OGM-GE · 리더보드 모달수 역상관 |
| **P48** | 쿼리 경로 인스턴스 감독(타깃 변경) | MUSES 전용(DELIVER 제외 확정) | **미실행** (선행 측정만 완료) | 제안 단계 | DELIVER CC를 인스턴스 수 대용 · 인스턴스 감독을 노벨티로 주장 · things PQ 게이트 30 |

---

# 🔴 표 2 — 반증된 설계 목록 (단일 출처)

> **새 제안이 이 표의 항목에 해당하면 그대로 제안하지 마라.** 재시도하려면 왜 이번엔 다른지를
> 해당 반증 근거를 직접 반박하는 형태로 먼저 쓸 것.

| # | 반증된 설계 | 어디서 죽었나 | 반증 근거 |
|---|---|---|---|
| **1** | 🔴 **이중 중복 경로 위에 모듈을 얹어 semantic mIoU 개선** *(신규·최상위)* | P43·P44·P46·P47 4세대 연속 | [analysis/2026-08-05-p46-module-ablation-query-nooop.md](../experiments/analysis/2026-08-05-p46-module-ablation-query-nooop.md) + 2026-08-05 S1(§0.5-1): **dense 단독 ≈ 전체의 99%, query 단독 ≈ 전체의 99%**. 이미 이중 중복인 시스템에 얹은 것은 중복에 흡수된다 |
| **2** | **RBMA / CoRB pre-softmax additive bias 계열** | P27→P28→P31→P32→P34 (4세대·2백본) | P32 CoRB **ΔmIoU −0.013, p=4.5e-22**(순손해); P34 λ1/λ2 toggle Δ≈0; [analysis/2026-07-12-p29-p34-standard-analysis.md](../experiments/analysis/2026-07-12-p29-p34-standard-analysis.md), [decisions/2026-07-16-p36-novelty-critical-review.md](../decisions/2026-07-16-p36-novelty-critical-review.md) §1 |
| **3** | **추론 시 reliability gate / calib / veto 재가중** | P34·P37a·P38·P39 (3세대 no-op → **유해로 재판정**) | 각 세대 module_ablation \|Δ\|≤0.5; P39-DELIVER에서 off 시 thin-class **+35.9/+26.0** = 켜두면 유해. [analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md](../experiments/analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md) C-2, [analysis/2026-07-20-p39-deliver-3ckpt-compare.md](../experiments/analysis/2026-07-20-p39-deliver-3ckpt-compare.md) |
| **4** | **CEFR class-expected routing** | P37a (MUSES) | **per-class 커밋 0/19**, 라우팅 엔트로피 1.085~1.092(max 1.099), winner 19/19 event, `cefr_off` Δ+0.16=no-op. [analysis/2026-07-18-p37a-muses-cefr-output-analysis.md](../experiments/analysis/2026-07-18-p37a-muses-cefr-output-analysis.md) |
| **5** | **무감독 threshold mask 게이트** | P37b `mask_proj` | 비미분 경로라 gradient 미도달 → **영구 random init**(ISSUE-024). 올바른 패턴 = P38 `m2f_head.py::_attn_bias`(예측을 리사이즈) |
| **6** | **수동 zero-init 잔차 결선으로 새 경로 얹기** (실패-키 1) | P37a σ(a) 0.121 · P38 β 0.133 · P39.1이 γ=0에서 학습 불가 실증 · 4연속 사장 | [analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md](../experiments/analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md) 키1. ⚠️ **단 처방 (a)"주손실 직접"도 충분조건이 아니다** — P43이 잔차를 완전히 없애고도 실패(#1) |
| **7** | **query/class-token decoder로 conv head 즉시 대체** | P30 (seg **−13.4/−10.2**, det mAP50 0.446→0.256) | `experiments/registry.md`; ⚠️ 단 그 런은 router 미실행(ISSUE-022)이라 붕괴 귀속은 **CTD 대체 단독** |
| **8** | **fusion rank / effective-rank / η² 개입** | P41 FCR (MUSES) | η² **0.35→0.94(2.7×)인데 mIoU 불변** = 사전등록 falsification 정확히 실현. [decisions/2026-07-22-p41-fusion-spectral-discrimination-proposal.md](../decisions/2026-07-22-p41-fusion-spectral-discrimination-proposal.md) §Phase1 결과. 2026-08-05 DELIVER 피쳐 통계(η² 단조 상승)가 재확인 |
| **9** | **모달 드롭아웃/마스킹으로 약모달 사용 강제**(3변형 전멸) | P33 무조건 dropout(no-op) → P42 균형 마스킹(FRAC 0.3/0.5/0.7 **단조 열화**) → P44 B-3 국소 마스킹(−1.36) | `experiments/plan.md` P42 완주 수치; [analysis/2026-07-28-p44-bmr-muses-standard-analysis.md](../experiments/analysis/2026-07-28-p44-bmr-muses-standard-analysis.md) |
| **10** | **gradient-level 모달 균형화**(MMPareto·peer 증류·OGM-GE) | P44 BMR(공식 test **78.429**, fog_night **−13.2**) · P47-2(−0.69, 진단 반증) | 위 P44 분석 + `experiments/monitor-log.md` 2026-08-04 P47-2 per-modal 로그(**RGB에 3.2× 가중을 줘도 uni-modal 성적 불변**) |
| **11** | **"RGB(지배 모달)가 under-optimize돼 있다"는 진단** | P47-2 | 위와 동일 — arm①/arm② img acc 0.949/0.951, ce 0.143/0.141로 **가중치 무관** |
| **12** | **입력 투영 밀도화(DGFusion식 dilation+motion-comp)** | P47-1 (MUSES 4모달) | 4.99× 밀도화 → val 82.58 = 3모달 82.62 **−0.04(노이즈 내)**, ep172 이후 122ep 무갱신 |
| **13** | **"lidar 표현이 4.7차원으로 붕괴했다"는 서술** | 2026-08-05 피쳐 통계 | lidar eff.rank는 **231.3(전 모달 최고)**, 4.68은 **FUSED**다(§0-d). ⚠️ P39-DPC 시절의 lidar 4.7은 **별개의 정당한 측정**이며 VICReg으로 78~100으로 복구됨 |
| **14** | **입력 스케일 불일치를 "lidar 붕괴의 원인"으로 지목** / **그것을 노벨티로 주장** | 2026-08-05 | 불일치는 **실재**(img z-score vs lidar [0,0.38], `encoder.py:273`)하나 위 #13으로 원인 가설은 반증. 그리고 **버그 수정이지 노벨티가 아니다**(user 지정 2026-08-05) |
| **15** | **리더보드의 "모달 수 ↔ 순위" 역상관을 설계 근거로 사용** | 2026-08-04 철회 | 통제 within-method ablation은 **전부 단조 증가**(CAFuser Table IX: 55.7→58.7→59.3→59.7; DGFusion CLE 51.6→CLDE 56.7). [analysis/2026-08-04-sota-landscape-recheck.md](../experiments/analysis/2026-08-04-sota-landscape-recheck.md) §3 |
| **16** | **test-best 체크포인트 수치**(헤드라인·λ선택·게이트 전부) | 2026-07-15 / 08-03 / 08-04 세 차례 철회 | [analysis/2026-08-03-p46-c3only-lambda-sweep.md](../experiments/analysis/2026-08-03-p46-c3only-lambda-sweep.md) §4: **규칙마다 λ 순위가 뒤집힌다**(test-best λ0.2 > λ0.3 > λ0.05 vs val-best λ0.05 > λ0.2 > λ0.3) |
| **17** | **`val 67.74 / test 56.62`를 DELIVER 내부최고로 인용** | 2026-08-05 정정 | val=P36 · test=P34(**PhysAug ON = 공정선 밖**)를 이어붙인 쌍. 공정 legal 최고 = **P36 67.74 / 55.62**(§0-b) |

---

> **다음 세션에게**: 새 구조를 제안하기 전에 **§0.5의 두 실측**과 **표 2**를 통과하는지 먼저 확인하라.
> 특히 §0.5(1)(이중 중복)을 깨지 못하는 제안은 P43~P47과 같은 방식으로 흡수될 것이다.
> 그리고 어떤 수치를 인용하든 **val-best와 final-iter를 병기**하고, 데이터셋·모달 수·PhysAug 여부를 함께 적어라.
