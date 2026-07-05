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

## 다음 (Step 2)
`LoRA_Sam_P32(LoRA_Sam_P31)`: `_compute_bias_source` override — bias 소스를 self-entropy → corroboration(+veto)로. 코드상 P31 `consistency_bias`(line 8421-8424)가 이미 Bhattacharyya 합의를 2차 항으로 계산 → 이를 1차 신호로 승격 + veto 추가. config `corroboration_bias` 플래그.
