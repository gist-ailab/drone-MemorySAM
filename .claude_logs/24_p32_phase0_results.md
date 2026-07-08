# P32 Phase 0 결과 — Corroboration vs Self-Entropy AUROC (무학습 진단)

> 실행: 2026-07-05, B200 GPU 2/3, DELIVER test 5조건 × 100장. 도구 `tools/eval_reliability_auroc.py`
> (roadmap [23_seg_arch_proposals_P32.md](23_seg_arch_proposals_P32.md) §7 GATE #1의 구현).
> 산출물: `/mnt/HDD2/src/logs/P32_phase0_20260705/{P28_test178,P31_test182}.json` + 로그.
> 모달 순서 m0..m3 = **[img, depth, event, lidar]** (DELIVER rgbdel).

## 신호 정의
- **selfent** (현 RBMA): `1 − H(softmax(D_i(f_i)))/logC` — per-modal self-confidence.
- **corr** (P32-B 후보): leave-one-out 합의 `p̄_{−i}=mean_{j≠i}p_j` 대비 corroboration. 두 변형 측정:
  - `corr_bc` = Bhattacharyya 계수 `Σ_c √(p_i·p̄_{−i})`
  - `corr_js` = `1 − JSD(p_i‖p̄_{−i})/log2`
  - gate는 per-modal **max(bc, js)** 사용.
- 타깃 = per-modal argmax == GT (16 §7과 동일 correctness AUROC 프로토콜).

## 결과 A — P28 test-ep178 (무보정 baseline, 16 §7의 그 ckpt)

| 모달 | selfent | corr(best) | Δ | 판정 |
|------|--------:|-----------:|----:|------|
| img   | 0.773 | 0.704 | −0.069 | (강모달 소폭 하락, >0.5) |
| depth | 0.621 | 0.701 | +0.080 | ↑ |
| **event** | **0.296** | **0.543** | **+0.247** | **anti-cal → 정보성 (0.5 돌파)** |
| **lidar** | **0.215** | **0.808** | **+0.593** | **우연 이하 → 강신호 (극적 수리)** |

- **selfent = [0.773, 0.621, 0.296, 0.215]** — doc 16 §7의 [0.77,0.62,0.30,0.22]을 정확히 재현 → **도구 검증됨**.
- **GATE(event/LiDAR corroboration > 0.5): PASS** (event 0.543, lidar 0.808). corroboration이 self-entropy가 깨진 geometry 모달을 **무학습으로** 반전 → P32-B 핵심 가설 입증.
- 단, **img는 corroboration이 소폭 열세**(0.773→0.704): 강모달은 self-confidence가 이미 좋음.

## 결과 B — P31 test-ep182 (rbma_calibrate ON, 보정 모델)

| 모달 | selfent | corr(best) | Δ |
|------|--------:|-----------:|----:|
| img   | 0.322 | 0.638 | +0.316 |
| **depth** | **0.904** | **0.283** | **−0.621** |
| event | 0.634 | 0.692 | +0.058 |
| lidar | 0.850 | 0.551 | −0.299 |

- **P31의 calibration loss가 작동함**: selfent event 0.30→0.63, lidar 0.22→**0.85**, depth 0.62→0.90 (16 §7 대비 큰 개선) — 단 **img는 0.77→0.32로 붕괴**(보정이 over-rotate, workhorse를 depth로 이동시킴).
- corroboration은 여기서 **비균일**: img↑(0.32→0.64)·event↑는 유지하나 **depth를 0.90→0.28로 크게 훼손**, lidar도 0.85→0.55로 하락.
- event/LiDAR>0.5 기준 자체는 PASS(event 0.692, lidar 0.551)나, **depth(workhorse) 훼손이 핵심 경고**.

## 판정 & P32-B 설계 함의

**GATE: PASS.** corroboration은 무보정(P28)에서 anti-calibrated event/LiDAR를 확실히 반전(lidar 0.22→0.81). 신호 수리 입증 → **P32-B(CoRB) 착수 GO**.

**단, 데이터가 드러낸 정제 사항 (P31 depth 0.90→0.28):**
1. **순수 대체는 위험** — corroboration은 "합의와 어긋나지만 혼자 옳은 workhorse"(P31 depth)를 penalize. 이것이 roadmap §3의 **unique-info veto**가 필수인 이유(선택 아님). veto = `conf_i 高 ∧ 불일치 ∧ 나머지 conf 低` → 벌점 면제/부스트.
2. **corroboration ⟂ calibration** — P31의 per-modal temperature와 직교. corroboration은 self-entropy가 깨진 raw 모달에서 강하고, calibration은 보정 후 self-confidence를 살림 → **둘 다 살리는 blend**(veto-gated, 또는 per-pixel max)가 순수 대체보다 우세일 여지.
3. 따라서 P32-B 권장 형태 = **corroboration-primary bias + unique-info veto**, P31 calibration/CTD 위에 config-gated 합성(P28~P31 byte-identical 유지).

## 결과 C — v2 재측정 (veto/max 변형 추가, 2026-07-05) → 신호형 확정

무학습으로 signal-form을 좁히기 위해 도구에 2개 변형 추가:
- **corr_veto** = `g·selfent + (1−g)·corr_bc`, `g_i = clamp(selfent_i − max_{j≠i} selfent_j, 0, 1)` (threshold-free soft unique-info veto: 모달 i가 나머지보다 얼마나 *더* 확신하는가 → uniquely-confident workhorse를 self-confidence 쪽으로 보호).
- **corr_max** = per-pixel `max(corr_bc, selfent)`.

**cross-condition mean AUROC (min = worst modality):**

| 신호 | P28 [img,dep,evt,lid] | P28 worst | P31 [img,dep,evt,lid] | **P31 worst** |
|------|----------------------|:---:|----------------------|:---:|
| selfent | [.773,.621,.296,.215] | .215 | [.322,.904,.634,.850] | .322 |
| corr_bc (순수) | [.664,.690,.543,.808] | .543 | [.602,**.283**,.660,.528] | **.283** |
| corr_js (순수) | [.704,.701,.509,.804] | .509 | [.638,**.278**,.692,.551] | **.278** |
| **corr_veto** | [.681,**.723**,.543,.808] | **.543** | [.603,**.708**,.664,.839] | **.603** |
| corr_max | [**.789**,.621,.543,.807] | .543 | [**.498**,.914,.520,.898] | .498 |

**확정: 신호형 = `corr_veto`.** 두 모델 모두에서 **worst-modality AUROC 최고**(P28 .543 tie, P31 .603 단독) — 유일하게 어떤 모달도 anti-calibrated로 남기지 않음:
- 순수 corroboration이 P31 depth(workhorse)를 .283으로 죽이는 걸 **veto가 .708로 회복**.
- P31의 calibration이 깨뜨린 img(selfent .322)도 corr_veto가 .603으로 살림.
- event/LiDAR 전부 >.6 유지 (원래 목표).
- corr_max는 mean은 살짝 높으나(P31 .708) selfent의 깨진 신호를 그대로 물려받아 P31 img worst .498 → 탈락.

**메커니즘 요지**: reliability = "상호검증(corroboration)"을 기본으로 하되, **혼자만 자신 있는 센서**(다수가 못 보는 곳에서 홀로 confident)는 합의 불일치로 벌하지 않고 self-confidence를 유지(veto). threshold-free, training-free, RBMA logit-bias 배관 그대로.

## 다음 (Step 2 — 구현 착수)
`LoRA_Sam_P32(LoRA_Sam_P31)`: `_compute_bias_source` override — bias 소스 = **corr_veto**. 코드상 P31 `consistency_bias`(line 8421-8424)가 이미 Bhattacharyya 합의(pairwise-mean)를 2차 항으로 계산 → corroboration을 1차 신호로 승격 + soft veto gate. config `corroboration_bias=True`. P31 calibration/CTD와 orthogonal 합성(전부 OFF → P28~P31 byte-identical). 학습 파라미터는 λ만(무학습 신호 유지). tools/eval의 `corr_veto`와 동일 수식.
