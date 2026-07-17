---
title: "P34-ReliaDINO 설계 — DINOv3-RBMA (카드 A) 구현 스펙"
tags: [P34, ReliaDINO, dinov3, rbma, design, proposal, card-A]
created: 2026-07-12
source: "[[../material/brainstorm_next_arch_20260708|카드 A]] × [[../P33_CGMoD/P33_v2_설계개정_20260708|P33-v2 M3/corr_veto]] × P32/P33 실측 제약 4종"
status: proposal
---

# P34-ReliaDINO — DINOv3-RBMA 구현 설계 (2026-07-12)

> **⚠ NAS vault sync pending** — 로컬(worktree) 작성본. `scripts/sync_vault.sh` 반영 전까지 canonical은 이 파일.
> **구현 완료** (`semseg/models/reliadino/`, `train_reliadino.py`, configs) — CPU smoke 통과, A-1 feasibility probe 확정 시 즉시 학습 착수 가능.

## 0. 한 줄 요약

frozen **DINOv3 ViT-L/16** (per-modality LoRA r8) → **cross-modal memory-style attention**(RBMA-v2 pre-softmax bias λ₁·B_cal + λ₂·B_cons) → **competence gate**(calibrated self-entropy softmax + training-free corr_veto floor) → SimpleFPN + 경량 conv head. SAM2 의존 완전 제거, RBMA를 VFM-agnostic 프레임워크로 승격.

## 1. 아키텍처

```
 x_img ──┐                                          (모달별 독립 LoRA_i, 백본 공유·frozen)
 x_dep ──┤   ┌────────────────────────────┐   f_i (B,1024,h16,w16)
 x_evt ──┼──▶│ FrozenViTEncoder           │──┬────────────────────────────────┐
 x_lid ──┘   │  DINOv3 ViT-L/16 (frozen)  │  │                                │
             │  + LoRA_i on qkv (Q/V, r8) │  │  ┌──────────────────────────┐  │
             └────────────────────────────┘  ├─▶│ AuxDecoder_i (conv, s16) │  │
                                             │  └───────┬──────────────────┘  │
                                             │     aux_logits_i (B,25,h,w)    │
                                             │          │                     │
                                             │   ┌──────▼───────────────────┐ │
                                             │   │ signals (training-free)  │ │
                                             │   │ rel_cal_i = 1−H(p_i/T_i) │ │
                                             │   │ corr_i    = BC(p_i,p̄₋ᵢ)  │ │
                                             │   │ corr_veto = g·se+(1−g)·c │ │
                                             │   └──┬──────────────┬────────┘ │
                                             │      │ B_cal,B_cons │ veto     │
             ┌───────────────────────────────▼──────▼──┐           │          │
             │ ReliabilityGatedFusion (×2 layers)       │          │          │
             │  Q=tokens_i, K/V=concat(tokens_{j≠i})    │          │          │
             │  softmax(QKᵀ/√d + λ₁B_cal[j] + λ₂B_cons[j])V        │          │
             └──────────────────┬─────────────────────┘            │          │
                     h_i (모달별 enhanced tokens)                   │          │
                                ▼                                  ▼          │
             fused = Σ_i w_i·h_i,  w = softmax(rel_cal/τ) ⊙ veto-floor-clamp ◀┘
                                │ (B,1024,h16,w16)
                    ┌───────────▼───────────┐
                    │ SimpleFPN (ViTDet)     │ {×4,×2,×1,×½} → 256ch
                    └───────────┬───────────┘
                    ┌───────────▼───────────┐
                    │ FPNSegHead (query-free)│ sum@s4 → 2×conv → 1×1 → 25cls
                    └───────────┬───────────┘
                          logits (B,25,H,W)   +  aux{rbma_cal_loss, aux_ce, gate_entropy}
```

## 2. 모듈 스펙 (파일 = `semseg/models/reliadino/`)

| 모듈 | 파일 | 내용 | 계보 |
|---|---|---|---|
| `FrozenViTEncoder` | encoder.py | timm `vit_large_patch16_dinov3`(기본, **HF timm/ 저장소 un-gated 확인**) → 실패 시 `vit_large_patch14_reg4_dinov2` 자동 폴백 → 최후 random-init(경고). 전체 frozen + `MultiModalLoRAQKV`: fused qkv Linear 래핑, **Q/V 슬라이스에만** per-modality A/B(r8, B=0 init). Eva(dinov3)의 `F.linear(x, qkv.weight)` 우회 경로는 `qkv_bias_separate=True` 강제로 차단(래퍼 forward 보장) | MLE-SAM 계보, SAM2 `_LoRA_qkv` 패턴 재구현(의존 제거) |
| `SimpleFPN` | encoder.py | 단일 s16 맵 → ConvT×2/ConvT/id/MaxPool = {s4,s8,s16,s32}, lateral 1×1+LN+3×3+LN → 256ch | ViTDet |
| `ReliabilityGatedFusion` | fusion.py | ① AuxDecoder_i(conv 3×3+GN+1×1, s16) ② 신호 3종(§3) ③ cross-modal attn ×2 (가중치 모달 공유 = SAM2 memory-attn 일반화, per-key additive bias) ④ competence gate + veto floor ⑤ aux_ce(@gt/4) + calibration loss | P31/P32/P33 수식 포팅(동일 수치) |
| `ReliaDINO` | model.py | encoder→fusion→FPN→head, `_maybe_drop_modality`(M2 seam, 기본 OFF). train 반환 `(logits, m_feat, aux_dict)` / eval `(logits, m_feat)` — 기존 `evaluate()` 호환 | P33 forward 계약 |
| `FPNSegHead` | model.py | 전 레벨 s4 업샘플 합 → conv×2 → 1×1 → 25cls. **query-free 1차 버전** | GOOSE aux-CE 교훈: P30 query-head 붕괴 회피 |

**Mask2Former-lite(고정 per-class token) = stage-2 TODO** — 카드 A 원안의 head이나, P30 실측 붕괴(val −13.4) 전례 + 속도 우선으로 1차는 conv head. fusion/backbone 효과 판독이 목적.

## 3. 신호 정의 (실측 제약 준수 — 어기지 말 것)

| 신호 | 수식 | 소비처 | 근거 |
|---|---|---|---|
| **rel_cal** (1차) | `1 − H(softmax(D_i(f_i)/T_i))/log C` | **fusion gate** `w=softmax(rel_cal/τ)` + B_cal(centered) | 제약1: corr_veto는 죽은 모달을 오상향(P33 docstring) → gate는 반드시 calibrated self-entropy |
| **B_cons** (2차) | leave-one-out Bhattacharyya `Σ√(p_i·p̄₋ᵢ)` centered | attn bias **2차 additive 항만** (λ₂) | 제약2: P32 교훈 — soft attn bias 단독은 결정 불변, 신호는 gate에서 작동해야 |
| **corr_veto** | `g·selfent + (1−g)·corr`, `g=clamp(se_i − max_{j≠i} se_j, 0,1)` | **training-free veto floor만**: corr_veto<0.10 → w_i ≤ 0.05 clamp 후 renorm (게이트 학습 밖) | P33-v2 M3 "hard floor를 게이트 밖에 유지" |
| **calibration** | P31 correctness-contrastive(틀린 픽셀 ent↑/맞은 픽셀 ent↓, @gt/4) + per-modal AUROC stash | T_i 학습 + `p34/rel_auroc_*` 로깅 | M3; img AUROC>0.7 유지 모니터링(P31 over-rotate 전례) |

grad 규약(P33 동일): 신호는 detach된 decoder logits에서 계산(무학습), T_i만 `GATE.ENTROPY_REG>0`일 때 gate로 grad 전달. hinge-entropy는 floor **미만만** 벌점(AECF 붕괴 경고 — uniform으로 밀지 않음).

## 4. Config seam (`configs/*_P34_reliadino.yaml`)

`MODEL.{BACKBONE_TIMM, BACKBONE_FALLBACK, PRETRAINED_BACKBONE, LORA_R:8, LORA_ALPHA, FPN_DIM:256}` ·
`FUSION.{NUM_LAYERS:2, NUM_HEADS:8, AUX_HIDDEN, AUX_CE_WEIGHT:0.5, ATTN_BIAS.{ENABLE, LAMBDA1_INIT}}` ·
`CONSISTENCY.{ENABLE, LAMBDA2_INIT:0.5}` ·
`GATE.{ENABLE, TAU:0.25, ENTROPY_REG:0(hinge), ENTROPY_FLOOR:0.5, VETO_FLOOR.{ENABLE, THRESH:0.10, CAP:0.05}}` ·
`CALIBRATION.{ENABLE:true, LAMBDA:0.1}` ·
`MODAL_DROPOUT.{ENABLE:**false**, P, TARGETS, WARMUP_EP}` (제약4: mid-run 무이득 → seam만, 기본 OFF)

## 5. Staged ablation plan

| Stage | 구성 | 게이트/판정 |
|---|---|---|
| **P34.1** (본선) | RGB+depth+event+lidar, gate+bias+calib ON (본 config) | val > P32 64.12 / test > 55.01; **Bridge/Water/Wall test IoU가 0대를 벗어나는가**(frozen-ceiling 가설 직접 검증 = A-1 probe의 풀스케일 확인) |
| Abl-1 `-gate` | `GATE.ENABLE:false` (uniform 평균) | gate 기여 분리 — adverse-split(night/fog) delta 기준 ≥+0.5 기대 |
| Abl-2 `-consistency` | `CONSISTENCY.ENABLE:false` | B_cons 2차 항 기여 (P32 반례와 대조 — 예상 ≈0, 논문 방어용) |
| Abl-3 backbone | 동일 fusion을 SAM2 Hiera-B+ 출력에 (기존 P33 스택) vs DINOv3 | "RBMA 프레임워크의 백본 일반화" 서사의 핵심 표 |
| P34.2 (조건부) | `MODAL_DROPOUT.ENABLE:true` (+distill 검토) | drop-Δ(event/lidar) ≥ +1.0일 때만 채택 (P33-v2 M2 게이트 동일) |

## 6. Training recipe

- **trainer**: `train_reliadino.py` (SAM2 import 0) — OHEM CE + 0.1·cal + 0.5·aux_ce (+hinge), effective batch 16(grad-accum 자동), bf16 AMP, DDP(torchrun), val+test every 2ep + per-class IoU, top-5 ckpt `{'model_state_dict',...}` 포맷·명명 동일(기존 tools/eval_per_domain.py 등 호환), AUTO_RESUME.
- **B200**: `torchrun --standalone --nproc_per_node=4 train_reliadino.py --cfg configs/b200-deliver_rgbdel_P34_reliadino.yaml` — 1024², BS 4/GPU(보수적; 첫 nvidia-smi 후 6–8 상향 여지), adamw lr 6e-4 / wd 0.01 / warmuppolylr(warmup 10) / 200ep. trainable ≈ LoRA(qkv Q/V×24블록×4모달, r8) + fusion + FPN + head ≈ 40–60M vs frozen 300M.
- **hinton(24GB)**: `configs/hinton-deliver_rgbdel_P34_reliadino.yaml` — 768², BS1, grad-checkpoint ON (디버그/probe용).
- 학습 전 필수: 빈 GPU 확인(CLAUDE.md 규약), DELIVER 경로 확인, 첫 로그에서 `[ReliaDINO] backbone=vit_large_patch16_dinov3` (폴백/랜덤 경고 없음) 확인.

## 7. Success gates

| 기준 | 값 | 판정 |
|---|---|---|
| vs P32 (계보 최고) | val 64.12 / test 55.01 | P34.1이 **둘 다** 상회해야 카드 A 지속 |
| vs SOTA (공식 목표) | val 66.51(CMNeXt) / test 56.71(DGFusion) | val ≥66.51 = 카드 A 본명 입증; 미달 시 dead-class delta로 재평가 |
| Dead-class | Bridge/Water/Wall test IoU > 0 이탈 | 핵심 가설(제약3: frozen-SAM2 ceiling → DINOv3가 존재 이유) |
| 신호 위생 | 전 모달 rel AUROC>0.5, img>0.7 유지; gate w̄ 비-uniform | `p34/rel_auroc_*`, `p34/gate_w_*` 로그로 매 epoch 감시 |

## 8. Risks

1. **DINOv3 가중치**: HF `timm/vit_large_patch16_dinov3.lvd1689m` **un-gated 확인**(2026-07-12 API probe) — 리스크 해소. 단 B200 최초 실행 시 ~1.2GB 다운로드 필요(인터넷 OK). 폴백 dinov2 경로는 smoke 검증됨.
2. **attn bias + sdpa 메모리**: float attn_mask 사용 시 flash 커널 대신 mem-efficient/math 폴백 가능 → 4096×12288 어텐션이 무거워질 수 있음. 완화: `ATTN_BIAS.ENABLE:false`가 즉시 flash 복귀 스위치; NUM_LAYERS 1 축소 여지.
3. **RGB-특화 SSL의 약모달 부적합**(A-2 probe 미검증): event/lidar에서 rel AUROC<0.5 지속 시 AnyThermal식 증류 경로 검토.
4. **conv head 상한**: thin-class 경계는 M2F-lite(stage-2)까지 유보 — 1차 결과에서 경계-병목 확인 시 착수.
5. **모달 4× forward 비용**: frozen이라 activation은 LoRA-backward용으로만 유지되나 여전히 4×(단일 백본 공유로 파라미터는 1×). BS 상향 전 profile 필수.

## 9. 카드 A 원안 대비 의도적 차이

| 원안 | 구현 | 사유 |
|---|---|---|
| Seg head = Mask2Former-lite | **query-free conv head (1차)** | P30 query-head 실측 붕괴 + "training can start immediately" — M2F-lite는 stage-2 |
| FPN 위치 미명시 | **fusion 후 단일 FPN** (per-modality FPN 아님) | ViTDet가 단일 s16에서 전 피라미드 생성; pre-fusion 4×FPN은 비용 4×에 설계 이득 없음(ablation seam으로만 문서화) |
| B_cal = "교정된 예측 엔트로피" | 동일 + **gate 신호와 공유**(rel_cal) | P33 실측: 융합 가중은 반드시 calibrated self-entropy — 카드 A 표기보다 P33-v2가 우선 |
| thermal/LiDAR AnyThermal 증류 옵션 | 미구현 | A-2 probe 결과 대기; LoRA-only로 1차 |

관련: [[../material/brainstorm_next_arch_20260708]] · [[../P33_CGMoD/P33_v2_설계개정_20260708]] · repo `semseg/models/reliadino/` · `train_reliadino.py`
