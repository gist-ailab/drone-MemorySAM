# P37a-CEFR MUSES 출력 분석 — "구조가 역할했는가, 라우팅이 됐는가" (2026-07-18)

**대상**: P37a-CEFR **MUSES 3모달(img/lidar/event)** val-best `epoch110_81.16_top1_checkpoint.pth` (hpca100 학습, val 81.16 = P34-3modal 80.86 +0.30)
**실행**: yeon GPU0, worktree `/SSDb/jemo_maeng/src/dm_analysis` @ develop 2ed3076
**도구**: `tools/probe_cefr_routing.py` (val 60장) + `tools/module_ablation.py` (val 40장, 토글 4종)
**산출물**: yeon `/SSDb/jemo_maeng/analysis/P37a_muses_20260718/` → NAS `analysis_logs/P37a_muses_20260718/` 회수

## 한줄 판정

**구조는 채택됐지만(σ(a) 0.018→0.121), 핵심 가설인 "클래스별 모달리티 라우팅"은 실현되지 않았다** — CEFR 가중치는 19클래스 전부에서 사실상 동일한 전역 틸트(event 0.385 > lidar 0.336 > img 0.279)이고, 순기여는 +0.16 mIoU(서브셋)로 no-op 수준이다.

## 1. σ(a) blend 개방도 — 구조 "채택"은 됨

- σ(a) = **0.1212** (init 0.018). 학습이 CEFR 경로를 6.7× 키웠다 → 옵티마이저가 이 경로에서 손실 감소를 찾긴 했음.
- 그러나 87.9%는 여전히 기존 gate-fused 경로. "채택"이지 "의존"이 아님.

## 2. per-class 라우팅 분화 — **실패 (핵심 가설 미실현)**

probe (val 60장, `_last_cefr_w` (m,B,K,h,w) 집계):

| 지표 | 값 | 판정 기준 | 판정 |
|---|---|---|---|
| margin>0.10 커밋 클래스 | **0/19** | 분화됐다면 다수 클래스 >0.10 | ❌ |
| 평균 margin (vs uniform 0.333) | 0.052 | | ❌ |
| 라우팅 엔트로피 | 1.085~1.092 (max 1.099) | 낮을수록 commit | ❌ 사실상 uniform |
| winner | **19/19 전부 event** | 클래스별로 달라야 분화 | ❌ |

- 클래스 간 w 편차가 ±0.02 이내 (예: img 0.259~0.308) — **클래스 조건부 신호가 거의 없음**. per-class 헤드가 만든 w가 클래스 축으로 접히지 않고, 전역 모달 재가중(global re-weighting)으로 수렴.
- P30 router의 "공간 평균 uniform은 측정 artifact" 함정과 다름: 이번엔 **per-class로 갈라서 봐도** uniform이다. 진짜 미분화.
- 가능 원인(설계 진단): (a) CA2 anchor의 λ2(t)·log p̂ 항이 per-class zero-init 헤드 출력을 지배(앵커가 클래스 무관), (b) zero-init 헤드가 σ(a) 게이트에 눌려 gradient를 충분히 못 받음, (c) MUSES 3-modal에서 클래스별 최적 모달이 실제로 갈리지 않을 가능성(단, DELIVER P36 router 분석에선 갈렸으므로 데이터 탓만은 아님).

## 3. 모듈 A/B (val 40장, base mIoU 73.71 — 서브셋 프로토콜, full 81.16과 비교 불가)

| toggle | ΔmIoU(off, +=기여) | pred agree | 판정 |
|---|---|---|---|
| **p37_cefr_off** | **+0.16** | 0.9983 | CEFR 순기여 노이즈 수준 — 사실상 no-op |
| p36_router_off | +34.66 | 0.5599 | **의존이지 기여 아님** — P31(+38~42)·P36과 동일한 co-adaptation 시그니처. 라우터 경로에 표현이 얹혀 있어 끄면 붕괴하는 것 |
| p34_gate_off | −0.04 | 0.9979 | no-op (오히려 미세 악화 방지 없음) |
| p34_calib_off | +0.08 | 0.9981 | no-op |

- gate/calib no-op은 P34 계보의 기존 발견(reliability 장치 무효)과 **MUSES에서도 재현** — 백본·데이터셋을 바꿔도 같은 결론.
- cefr_off의 클래스별 Δ 상위가 rider +1.2, pole +0.5 등 소형·희소 클래스인 점은 방향성으로만 참고 (n=40 노이즈 범위).

## 4. 종합 판정과 설계 함의

1. **P37a의 +0.30(81.16 vs P34-3modal 80.86)은 클래스 라우팅의 성과라고 주장할 수 없다.** CEFR 자체 기여(+0.16, 서브셋)와 라우팅 미분화가 그 주장을 지지하지 않음. 전역 모달 틸트 또는 학습 동역학 차이로 보는 게 정직하다.
2. **CEFR는 현재 형태로는 RBMA bias의 전철(구조는 있는데 일을 안 함)을 밟는 중.** 논문에 넣으려면 (a) a init 상향/게이트 제거로 경로를 강제 개방, (b) 라우팅 commitment 정규화(엔트로피 페널티), (c) anchor 항 축소로 per-class 헤드가 신호를 내게 하는 재실험이 필요. 아니면 제외가 안전.
3. **event가 전 클래스 최고 가중(0.385)** — 이전 세대(DELIVER)에서 event 기여 ≈0였던 것과 대조. 단 라우팅 가중 ≠ 기여이므로, MUSES에서 event adapter on/off(D3B)로 검증할 가치는 있음.
4. router 의존(+34.66)은 "라우터가 +34 기여"로 읽으면 안 됨 (P31/P36 분석과 동일한 주의).

## 재현 커맨드

```bash
PY tools/probe_cefr_routing.py --cfg configs/hpca100-muses_rgbel_P37a_cefr.yaml \
  --model_path <ckpt> --dataset-root <MUSES> --split val --max-imgs 60 --gpu 0 --out <out>/cefr_routing
PY tools/module_ablation.py --cfg ... --split val --conditions all --max-imgs 40 \
  --toggles p37_cefr_off,p36_router_off,p34_gate_off,p34_calib_off --viz-num 2 --gpu 0 --out <out>/module_ablation
```
(전제: develop ≥ 2ed3076 — val.py MUSES=19 폴백 + fusion.py `_last_cefr_w` 스태시)
