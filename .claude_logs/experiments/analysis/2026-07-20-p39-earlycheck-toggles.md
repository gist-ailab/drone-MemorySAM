# P39-DPC 조기판정 (토글 즉검) — MUSES ep66 + DELIVER ep38 (2026-07-20)

**목적**: 제안서(2026-07-20-p39-dual-path-compete-proposal.md) §3의 사전 등록 게이트 — "학습 직후 module_ablation 즉검, no-op 조기 탈락 기준 |Δ|>0.5 & agreement<0.99" 집행.
**ckpt**: MUSES `epoch66_80.08_top1`(jarvis, ep72/300 진행 중 스냅샷) · DELIVER `epoch38_65.04_top1`(hpca100, ep53/200 진행 중 스냅샷).
**실행**: yeon GPU0/1 병렬, n=40/조건. **산출물**: NAS `analysis_logs/P39_earlycheck_20260720/`.

## 판정: V1·V5·router 전부 생존 — 5세대 만에 처음으로 신규 모듈이 no-op이 아님

| toggle | DELIVER (5-cond, ep38) | MUSES (6-cond, ep66) | 판정 |
|---|---|---|---|
| **p39_trunkexp_off** (V1 rank 확장) | **+0.76 ~ +2.67 (전 조건 최대 기여)** | +0.77 ~ +2.89 | ✅ 생존 — 키3(FUSED rank 병목) 처방이 실병목을 맞춘 인과 확인 |
| **p39_query_off** (V5 query 경로) | +0.65(cloud)/+0.20(fog)/**−0.24(night)/−0.23(rain)**/sun 소폭 | +0.05 ~ +1.09 (전 조건 +) | ✅ 생존(MUSES) / ⚠️ DELIVER 악조건에서 미세 역효과 — "query류=MUSES 체질" 비대칭이 P39에서도 재현 |
| p36_router_off | +0.02 ~ +0.58 | +0.5 ~ +2.30 | ✅ **의존→기여 전환 성공** — P37a +22~36 붕괴가 정상 기여로 (키2 처방 성공, 단일 실패점 해소) |
| p34_gate_off / calib_off | ≤+0.35 | ≤+0.77(fog) 대부분 ≤0.4 | 기존 결론 유지(≈no-op) |

**모듈 모니터 정합**: arb λ mean(softplus) MUSES 1.02 / DELIVER 1.41(max 2.33) — init 0.69에서 성장, "열리다 만"(β 0.13·σ(a) 0.12) 4세대 패턴과 결별. m2f legacy β=0(반증 경로 폐기 확인). MUSES router w̄ 분화(img .526/lidar .133/event .341).

## 성능 전망 (조기)

- DELIVER: val 65.04@ep38 후 63~65 정체 — **게이트(P36 fair 67.74) 도달 어려움**, P38형 조기포화. test 54.66@ep48 (P38 55.05, 계보상 test 피크는 늦으므로 관찰 지속). **조건부 중단점 제안: ep100~120에 test 55.05 미돌파 시 P39.1(physaug 복원 1순위)로 교체.**
- MUSES: 80.08@ep66 상승 중 (P38 피크는 ep156 — 완주까지 판단 유보). ETA 07-21 새벽.

## 함의

1. 키1(경쟁 결합)·키2(router 직접 감독)·키3(rank 확장) 처방은 **기제 수준에서 전부 유효** — 실패-키 문서의 인과 진단이 맞았음.
2. 남은 문제는 기제가 아니라 **성능 전환**: 모듈이 살아 있어도 DELIVER val 정체는 반복 — 남은 상한은 레시피(physaug, 키6)와 thin-class 손실 설계(V3/V4 효과는 D1 per-class로 완주 후 검증)에 있을 가능성.
3. DELIVER night/rain에서 query 경로 미세 역효과 — V2 modal-token attention이 악조건 모달 노이즈를 그대로 들여오는지 완주 후 D4/D5 심층 확인 필요.
