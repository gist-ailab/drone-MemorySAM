# P33 설계 — Competence-Gated Hard Fusion + Modality Dropout (CG-MoD)

> 근거: [`2026-07-07_P32_perimage_analysis.md`](2026-07-07_P32_perimage_analysis.md)(ep108 best weight, test 전체 1897장). P32 결론 = **"신호는 맞고 라우팅은 실패"** + **융합이 competence 무시(misalloc 51.6%)** + **event/LiDAR 사망(competence≈16)** + **도메인 전이 붕괴**.
> 계보: P28(self-entropy RBMA) → P31(calibration+SDC router) → P32(corroboration bias) → **P33(competence-gated hard fusion + modality dropout)**. roadmap `23_seg_arch_proposals_P32.md`의 P32-C(PruneMem)를 흡수·구체화.

---

## 1. P32가 남긴 4개 실패와 P33 처방(1:1)

| # | P32 실패(정량) | 근본 원인 | P33 처방 |
|---|---|---|---|
| F1 | **misalloc 51.6%**, UAMM 균일(0.27/0.27/0.23/0.23) | 출력 융합이 비적응(`AMF_MODE:uniform`) | **C1. Competence-Gated Fusion** — reliability+corroboration로 구동되는 sharpened/top-k 융합 게이트 |
| F2 | event/LiDAR **competence 16.9/15.3**, drop-Δ≈0 | 약한 모달 feature/decoder에 정보 없음 | **C2. Stochastic Modality Dropout(PruneMem)** — RGB/depth 확률적 제거로 event/LiDAR 사용 강제 |
| F3 | corroboration Δlogit 0.25인데 **flip 0.046%**, ΔmIoU −0.013 | bias 권위 부족 + 비보정 디코더 위 작동 | **C3. Calibration 복원 + gate 승격** — corroboration을 attn-bias(soft)에서 **fusion gate 입력(hard)**로 승격, P31 calibration loss 부활 |
| F4 | Wall/Bridge/Water/TrafficLight = **val 생존/test 사망** | day→night class-transfer(용량 아님) | **C4. Thin-class 도메인 강건화** — class-balanced/focal loss + night-consistency + (옵션)backbone 마지막 stage unfreeze |

---

## 2. 아키텍처 (P32 대비 변경점만)

P33 = `LoRA_Sam_P33(LoRA_Sam_P32)` — P32의 corroboration 기구는 유지하되 **신호의 소비 지점**을 바꾼다.

### C1. Competence-Gated Fusion (F1 직격, 최우선)
- **문제**: 현재 `AMF_MODE:uniform`은 4모달 등가중 → depth(43.7)와 event(16.9)를 똑같이 섞음.
- **설계**: per-pixel per-modal 융합 가중치 `w_i(x)`를 **학습 게이트**로 산출.
  - 입력: `[reliability_i(calibrated), corroboration_i(veto), per-modal feature summary]`.
  - **Sharpening**: `w = softmax(g(·)/τ)`, τ<1로 뾰족하게(또는 학습가능 τ). 균일 붕괴 방지 위해 **entropy 정규화 페널티**(너무 균일하면 벌점) 또는 **top-k(k=2) hard selection**(하위 모달 0).
  - 초기화: uniform에서 시작해 competence 방향으로 학습(안정성).
- **config seam**: `AMF_MODE: competence_gate`(신규), `GATE.TAU`, `GATE.TOPK`, `GATE.ENTROPY_REG`.
- **기대**: misalloc↓, depth 지배 복원 → per-image mIoU 즉효 상승(가장 싼 레버).

### C2. Stochastic Modality Dropout / PruneMem (F2 직격)
- **문제**: RGB+depth만 쓰고 event/LiDAR는 미사용(drop-Δ≈0) → 두 모달이 학습 신호를 못 받아 competence가 안 큼(악순환).
- **설계**: 학습 시 각 스텝마다 확률 `p_drop`로 **주력 모달(img/depth) 중 1개를 memory/입력에서 제거** → 남은 모달로 세그를 맞추도록 강제. (memory token pruning = roadmap PruneMem의 hard 버전)
  - 스케줄: `p_drop` 0→0.3 warmup. event/LiDAR는 드롭 안 함(이미 약하므로 살리는 게 목표).
  - eval은 전모달(드롭 없음).
- **config seam**: `MODAL_DROPOUT.ENABLE`, `MODAL_DROPOUT.P`, `MODAL_DROPOUT.TARGETS:[img,depth]`, `MODAL_DROPOUT.WARMUP_EP`.
- **기대**: event/LiDAR competence↑, drop-Δ가 양수로 → C1 게이트가 실제로 라우팅할 정보가 생김. **F1과 상보**(게이트가 있어도 정보가 없으면 무의미, 정보가 있어도 게이트 없으면 무시 → 둘 다 필요).

### C3. Calibration 복원 + corroboration 승격 (F3)
- **문제**: P32는 P31 calibration loss를 버려 event/lidar 디코더가 anti-calibrated(self-entropy rel 0.29/0.36). corroboration은 attn-bias(soft)로만 작동해 flip 0.046%.
- **설계**:
  1. **P31 calibration loss 부활**(per-modal decoder logit shaping) → reliability가 신뢰 가능해짐(Phase 0: corr_veto는 **보정된 모델 위에서 최상**).
  2. corroboration을 **C1 게이트의 입력**으로 승격(soft attn-bias는 보조로 유지 or 제거). 결정경계에 직접 작용.
- **config seam**: `CALIBRATION.ENABLE:true`, `CORROBORATION.CONSUME: gate`(기존 `bias`에서).
- **기대**: 신호가 출력 결정에 도달(flip↑, ΔmIoU 양수화).

### C4. Thin-class 도메인 강건화 (F4)
- **문제**: Wall/Bridge/Water/TrafficLight val 생존/test 사망 = 일반화 실패(용량 아님).
- **설계**(직교, 저위험부터):
  1. **Class-balanced / focal loss** — thin·rare 클래스(Wall,Fence,TrafficLight,TwoWheeler,Ground…) 가중.
  2. **Night-consistency reg** — NIGHT_AUG 강화본과 원본 예측 일치(이미 physaug 존재, consistency term 추가).
  3. (옵션·고위험) **backbone 마지막 stage unfreeze**(P31 CTD 레버) — geometry 사망 클래스(Bridge/Water)용. 비용·과적합 주의 → ablation 후 결정.
- **config seam**: `LOSS.CLASS_BALANCED`, `LOSS.FOCAL_GAMMA`, `NIGHT_CONSISTENCY.ENABLE`, `UNFREEZE_LAST_N_BLOCKS`.

---

## 3. 우선순위 & Ablation 계획
P28 순수 ablation 관례 유지(한 번에 한 축). 순서 = **효과/비용 비**:

1. **P33.1 = C1 단독** (competence-gate, calibration 복원 포함) — 융합만 적응화. **misalloc·per-image mIoU 즉효 검증**. 가장 싸고 확실.
2. **P33.2 = C1 + C2** (게이트 + modality dropout) — event/LiDAR competence·drop-Δ 상승 검증. **Mode C 부활의 본 처방.**
3. **P33.3 = + C3 완전형** (corroboration을 gate 입력으로) — 신호→결정 도달 검증(flip↑).
4. **P33.4 = + C4** (thin-class/도메인) — 사망 클래스 회수.

각 단계 게이트: (a) test mIoU vs P32 54.79 / P28 55.27, (b) **misalloc% ↓**, (c) **modal competence(event/lidar) ↑**, (d) corroboration flip% ↑ — `viz_features_full.py`로 동일 측정 재사용.

## 4. 성공 기준 & 리스크
- **정량 목표**: test mIoU **> P28 55.27 돌파**(P32는 54.79로 −0.48), 궁극 DELIVER SOTA(DGFusion 56.71)·official test 56.71 지향. misalloc < 25%, event/lidar competence > 25.
- **리스크**:
  - C1 sharpening이 depth 단일모달 붕괴로 수렴 → entropy-reg/top-k로 방지, dropout(C2)이 견제.
  - C2 dropout이 주력(depth) 성능 일시 하락 → warmup·p_drop 상한으로 완화.
  - C4 unfreeze 과적합/비용 → 마지막 단계, ablation 후.
- **불변**: 실패해도 F1/F2는 독립 레버 → 부분 채택 가능.

## 5. 구현 seam (코드 위치)
- 모델: `semseg/models/sam2/sam2/sam_lora_image_encoder_seg.py` — `LoRA_Sam_P33(LoRA_Sam_P32)`. 융합 게이트는 P32의 UAMM/AMF 융합 지점(`_fuse_outputs`/AMF) 오버라이드; corroboration은 이미 `_compute_bias_source` 존재(gate 입력으로 재배선).
- config: `configs/b200-deliver_rgbdel_P33_physaug.yaml` (P32 config 복제 + 위 seam 키).
- 검증 도구: `tools/viz_features_full.py`(per-image + corrb ON/OFF), `tools/module_diagnostics.py`, `tools/eval_reliability_auroc.py` — 모두 `--cfg/--model_path` 교체로 재사용.

## 6. 근거 데이터
[`2026-07-07_P32_perimage_analysis.md`](2026-07-07_P32_perimage_analysis.md) + `/mnt/HDD2/src/logs/P32_perimage_20260707/ep108/`. 핵심 수치: competence depth 43.7≫event 16.9≈lidar 15.3, UAMM 균일 0.27, misalloc 51.6%, corroboration flip 0.046%·ΔmIoU −0.013, dead-on-test/alive-on-val(Wall·Bridge·Water·TrafficLight).

---

## 7. 관련연구 & 노벨티 방어 (2026-07-08, research_vault 마이닝 — 3-agent)

> 방법: `.claude_logs/research_vault/relatedworks/`(97노트) + [12](../12_novelty_and_related_work.md)/[10](../10_related_work.md) 전수. **핵심 함의**: P33은 RBMA 신호를 **attn-logit-bias → 출력-fusion-gate**로 옮기는데, 이 지점은 **선행연구가 더 많이 점유**한 셀이다 → 노벨티는 조합·비대칭·진단주도로만 방어 가능. 아래 vault 노트번호는 `research_vault/relatedworks/`.

### C1 (Competence-Gated Fusion) — 근접 선행 & 위협
| 선행 | 무엇 | 노트 |
|---|---|---|
| **Decouple-Recouple** (2603.07486) | 학습 reliability-softmax 모달 게이트 **+ 동일한 entropy 붕괴방지 reg `−ΣWlogW`** | `78` |
| **MLE-SAM** (2412.04220) | **SAM2 안**에서 pooled-feature 게이트 + **top-k 모달 라우팅** | `88`,`50` |
| MAGIC++ / UNO / HyperDUM·UTFNet / CAFuser | 계층적 hard 모달선택 / uncertainty-temp 출력융합 / feature-level uncertainty 가중 / condition-scalar 가중 | `07`,`43`,`44`,`40` |

- **⚠️ 위협**: **Decouple-Recouple(`78`)이 최근접** — "학습 reliability softmax 모달게이트 + entropy anti-collapse reg"를 이미 함 → **P33의 entropy-penalty는 단독 노벨티 아님**(차용 machinery로 프레이밍). **MLE-SAM(`88`)**이 "SAM2 내 top-k 출력융합" 점유 → C1은 "MLE-SAM + reliability 게이트 입력"으로 보일 위험.
- **방어 가능한 조합만**: *training-free reliability **+ cross-modal corroboration** 게이트 입력 × per-pixel hard top-k × SAM2 memory-fused feature*. "gated/competence fusion이 새롭다"는 표현 금지.

### C2 (Stochastic Modality Dropout) — 근접 선행 & 위협
| 선행 | 무엇 | 노트 |
|---|---|---|
| **EQUISeg** (2509.24505) | **동일 질병**(dominant-modality collapse) 겨냥, 랜덤 모달-holdout(prototype teacher/student) | `62` |
| Reducing Unimodal Bias (2505.06635) | 같은 목표를 functional-entropy/Fisher **loss reg**로 | `03`,`41` |
| AnySeg·CMNeXt·MAGIC++·CrossWeaver·UniMRSeg | arbitrary-modal 학습(=모달 부분집합 drop) — **DELIVER 표준 레시피** | `05`,`07`,`65`,`87`,`12` |
| UMSE (2305.02504) | 부재 모달 **−∞ logit 마스킹** (token-pruning analogue) | `40`,`42` |

- **⚠️ 위협**: "학습 중 모달 dropout" 자체는 **비노벨티**(arbitrary-modal 라인 전체가 함). SAM2-memory-token pruning 프레이밍도 이미 점유(UMSE·SAM2Long·SAM4D, `42`). **EQUISeg가 최근접**(같은 질병+랜덤 holdout).
- **차별점은 backbone이 아니라 비대칭+진단**: *dominant-only drop(weak never) × measured competence/drop-Δ 진단 구동*. Medium risk. EQUISeg·Reducing-Unimodal-Bias를 필수 cite-and-distinguish.

### C3 (Calibration 복원 + 신호 승격) — 근접 선행 & **자기잠식 위협**
| 선행 | 무엇 | 노트 |
|---|---|---|
| **DAMP** (2512.20251) | **training-free 통계를 MoE 게이트 입력에 concat** | `50` |
| MLE-SAM / UNO-TempNet / MEFN | pooled-feat 게이트입력 / 학습 temperature 보정 / evidential 보정자 | `88`,`43` |
| PRIMED (2605.07154) | 학습 prior를 **pre-softmax attn-logit bias**로(=현 RBMA 위치) | `60` |

- **⚠️ 최대 위협 = 자기잠식**: RBMA의 발표 노벨티 근거가 "출력/feature/gate-input reliability 가중은 이미 점유됐고 attn-logit-bias 셀만 비었다"인데, C3가 신호를 **gate 입력으로 되돌리면 점유 영역으로 회귀**(DAMP·MLE-SAM·UNO). → **노벨티 주장이 아니라 ablation 발견**("결정에 작용 > attention nudging")으로 프레이밍 필수.
- **⚠️ lit-check 필요**: **correctness-contrastive calibration loss**(entropy가 정답을 예측하도록)는 vault 미커버 — ConfidNet/correctness-ranking/trust-score 계열은 일반 calibration 문헌에 흔함 → **외부 서치 전 노벨티 주장 보류**(점유 가정).

### C4 (Thin-class) — ⚠️ **타당성 위협 (프로젝트 자체 데이터가 부분 반박)**
- **(1) night-consistency reg는 DELIVER에 레버-미스매치**: DELIVER per-condition mIoU spread 2.7~3.6뿐 → 갭은 조명이 아니라 class-transfer. [16](../16_failure_analysis_P28_P29.md) §1-B 명시 *"조명 문제 아님 → night-aug류로 해결 안 됨"*. night-consistency는 **MULTIAQUA(day-train/night-test)에서만** 올바른 레버(`64`).
- **(2) focal/class-balanced = imbalance용, transfer용 아님**: P29 reweighting이 **val rare class는 개선했으나 test는 사망 유지, net −1.1, TrafficLight 41.3→9.6 붕괴**([16] §7) → 같은 category 레버는 "val 이득, test 무이득" 재현 예상.
- **(3) Bridge/Water/Other는 val·test 양쪽 사망(competence [0,0,0,0])= frozen-backbone 용량한계(ISSUE-008)**, transfer 아님. → **C4가 두 기구를 혼동**: Wall(56→0.9=진짜 transfer) vs Bridge/Water(≈0 양쪽=용량). 서로 다른 처방 필요. *(주의: P32 per-image 분석은 이들을 "val-alive"로 적었으나 doc 16의 competence는 양쪽 사망 — **P32 held-out val 실측 per-class IoU로 재확인 필요**.)*
- **vault 처방**(`16` §5-6, `20`): 양쪽사망=backbone unfreeze / 조건민감(RailTrack·TwoWheeler)=class-targeted aug / rare-class starvation=GOOSE-M2F aux-CE head(`51`, val측 이득) / night-consistency=MULTIAQUA 전용. **focal/class-balanced 문헌은 vault 미커버 → lit-check 필요**.

### 노벨티 종합 판정
- **C1**: 조합(무학습 reliability+corroboration × hard top-k × SAM2)만 방어 가능. entropy-reg·gated-fusion은 비노벨티.
- **C2**: 비대칭+진단주도만 방어 가능. Medium risk.
- **C3**: 신호 위치 노벨티 주장 **금지**(자기잠식) → ablation 발견으로. calibration-loss lit-check 필수.
- **C4**: 노벨티보다 **레버-핏 재설계 우선**(night-consistency→DELIVER 부적합, 사망클래스=용량 vs transfer 분리).
- **필수 cite**: Decouple-Recouple`78`·MLE-SAM`88`·DAMP`50`·EQUISeg`62`·UNO`43`·PRIMED`60`·HyperDUM`44`·MAGIC++`07`·Reducing-Unimodal-Bias`03`·GOOSE-M2F`51`.
- **lit-check TODO(외부, [12](../12_novelty_and_related_work.md) §4 병합)**: (a) correctness-contrastive/confidence-ranking calibration loss, (b) asymmetric modality-dropout 전례, (c) focal/class-balanced/Lovász seg loss.
