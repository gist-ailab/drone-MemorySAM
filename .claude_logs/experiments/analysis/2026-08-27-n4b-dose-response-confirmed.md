---
created: 2026-08-27
type: N4b 판정 — 진단-구동 dose-response 예측 적중 (논문 핵심 주장 검증)
---

# N4b 판정 — MCubeS C3-on: 사전등록 예측 2/2 적중, dose-response 성립 (2026-08-27)

> 실측 = 모니터링 세션(hpca100, N4와 유일차=C3_PROTO on, 동일 시드 3407). 예측 사전등록 = 2026-08-25 (per-class published 대조 후, 기동 전). 판정 = discussion 세션(fable).

## 1. 결과 vs 사전등록 예측

| 예측 | 등록값 | 실측 | 판정 |
|---|---|---|---|
| **P1 (기제)**: C3-on 시 rubber 회복 | ≥ +5 (18.80→24+) | **+9.76 (18.80→28.56)** | ✅ **적중** — published 대역(26.5~29.7) 안까지 완전 회복 |
| **P2 (총량 단조성)**: overall Δ가 MUSES(−0.77)와 DELIVER(+1.4) 사이 | −0.5 ~ +1.5 | **−0.10** (57.83 vs 57.93) | ✅ **적중** |

## 2. dose-response 3점 완성 — 진단-구동 주장의 실증

| 벤치 | 병리(class-transfer 붕괴) 강도 | C3 효과 (overall) | 표적 클래스 효과 |
|---|---|---|---|
| DELIVER | **심함** (RailTrack 4 vs 남들 64) | **+1.4** | RailTrack 4→68 |
| **MCubeS** | **중간** (rubber −11 vs published) | **−0.10** (중립) | **rubber +9.76 (published 대역 복귀)** |
| MUSES | **없음** | **−0.77** (유해) | — |

- **C3 효과가 붕괴 강도에 단조**: 심함→대이득 / 중간→표적만 회복·총량 중립 / 없음→유해. 사전등록 예측이 제3의 벤치에서 적중 = 진단-구동 프레임워크가 **사후 설명이 아니라 예측력**을 가짐을 실증.
- 특히 MCubeS 패턴이 정보적: **표적(rubber)은 크게 회복하면서 총량은 중립** — C3가 "붕괴 클래스를 되살리되, 병리 없는 나머지에는 소폭 비용"이라는 기제 해석(DELIVER/MUSES 양극단을 잇는 중간 거동)과 정확히 일치.

## 3. 함의

1. **논문 헤드라인 기둥 확정**: "진단-구동 학습신호 배치"가 3벤치 dose-response로 검증됨 — B-서사의 린치핀. N3 검출기(붕괴 강도 정량화)를 정식화해 "진단 지표 → 처방 강도" 규칙으로 논문에 기술.
2. **P52의 MCubeS 구성**: 총량 기준으론 C3-off(57.93)가 소폭 우세하나 rubber 회복 가치가 있음 — P52는 진단 규칙이 정하는 대로(중간 병리 → C3 약하게 or 표적 클래스 가중) 설계. λ 스윕은 비용 대비 후순위, 논문엔 on/off 두 점으로 dose-response 표만으로 충분.
3. 검출기 정교화(2026-08-25 판정의 요구) 재확인: 회복가능(rubber형) vs 공통난제(plaster형) 구분이 작동함 — plaster는 C3-on에서도 회복 안 됐을 것(확인 필요시 per-class에서 추출).

## 4. 기록

- ledger **H20 신설**(✓ 확인): "C3 prototype 손실의 효과는 class-transfer 붕괴 강도에 단조(dose-response)" — 3벤치 사전등록 검증.
- plan N4b 완료. P52 설계 재료로 편입.

관련: [2026-08-25-n4-mcubes-first-entry-verdict.md](2026-08-25-n4-mcubes-first-entry-verdict.md)(예측 등록) · [research/hypothesis-ledger.md](../../research/hypothesis-ledger.md) H8·H20 · 원시 hpca100 N4b SAVE_DIR
