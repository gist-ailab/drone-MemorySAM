---
created: 2026-08-20
type: 융합-기제 이중 판정 (xattn A/B #12 + no-GT 라우터 실현성 #15) — H16/H17
---

# 융합-기제 이중 음성 — cross-attn A/B 패배 + no-GT 라우터 실현성 음성 (2026-08-20)

> 두 실험은 독립적으로 같은 명제("융합 **기제**를 바꾸면/고르면 성능이 오르는가")를 친다. 둘 다 음성 → "믹서는 병목이 아니다"의 이중 실증. 실측 = 모니터링 세션(yeon). 판정 = discussion 세션(fable).

## A. Cross-attn 트렁크 A/B (#12, user 가설 통제검증) — MLP 우위 재확인

| | 트레이너 val-best | val.py legal @1024 | @768 |
|---|---|---|---|
| xattn 트렁크(대칭 모달간 attn, LayerScale 1.0) | 68.62@ep150 | **54.94** | 53.53 |
| 기준팔 gated-MLP(대표 c3only λ0.1) | 67.79@ep70 | **56.99** | — |
| **Δ (legal)** | | **−2.05** | |

- 사전등록 게이트(≥+0.3 채택 / ±0.3 무차이 / **≤−0.3 MLP 우위**) → **"MLP 우위 재확인" 구간**, 그것도 −2.05로 크게. **파라미터 15.9×(33.6M vs 2.11M)로도 못 이김** = 사전 명시한 "무차이/열세면 결론 강화" 케이스.
- ⚠️ 트레이너 지표(68.62)는 오히려 기준(67.79)보다 높았으나 legal에서 역전 — xattn이 768-리사이즈-GT 낙관지표에 더 과적합(train→legal 낙차 −13.7 vs 기준 −10.8). **legal 프로토콜의 필요성 재확인 사례.**
- **정직한 한계**: 통제 A/B는 "동일 레시피 drop-in 교체"를 잰 것 — xattn 전용 튜닝(LR/warmup/LayerScale)은 안 했다. 따라서 "어떤 cross-attn도 못 이긴다"는 증명이 아니라 **"drop-in 교체는 −2.05로 진다"**. 단 (i)손실 폭 큼 (ii)문헌(MM1 connector 2차·StitchFusion>GeminiFusion) (iii)원장 전반이 믹서 비병목 → **추가 튜닝의 기대가치 낮음, 계열 종료 권고.**

## B. no-GT 라우터 실현성 하한 (#15, val) — 값싼 실현신호 부재

오라클 여지(+8.66, GT-회수)가 **입력만으로 접근 가능한가**의 하한 측정. 캐시(argmax) 전용:

| no-GT 변형 | val Δ vs full |
|---|---|
| majority(다수결 앙상블) | **−1.447** |
| consensus(블록별 다수결-부합 커밋) block1 | −1.447 |
| … block64 | −1.019 |

- **전 변형 음수**, +1.0 게이트는커녕 0도 못 넘음. (test split 진행 중 — val이 이미 "reachability 확인" 바를 명확히 미달.)
- **기제 해석(중요)**: 오라클 여지는 부분집합들이 **함께 틀리는** thin-class 경계에 있다(#14 널: 양의 오차상관). 그 픽셀에서 옳은 모달은 **소수 의견**이라, 다수결/합의 라우터는 정확히 거기서 **틀린 쪽을 고른다** → 음수. 즉 **회수 여지는 "반-합의(anti-consensus)"라, 합의·동의 기반 라우터로는 구조적으로 접근 불가.**

## C. 종합 판정

1. **융합 기제 축은 두 방향에서 음성**: 연속 attention(xattn, −2.05)도, 이산 라우팅(no-GT, −1.0~1.4)도 실현 가능한 이득 없음. **"믹서가 병목"은 이중 반증.** MM1/원장과 정합.
2. **H16 = 여전히 개방이나 전망 하향**: GT-회수 여지(+4~8)는 실재하되 **합의/다수결로는 접근 불가(anti-consensus)**. 남은 유일 가능성 = 학습 게이트가 **비자명한 anti-consensus 신호**를 발견하는 것 — 낮은 사전확률.
3. **마지막 값싼 문 = confidence 라우터(#15b)**: consensus와 **기제적으로 다른** 유일한 실현 신호. 소수-옳은 모달이 그 픽셀에서 **고신뢰**면 confidence-max가 잡을 수 있다(합의는 못 잡음). 캐시가 argmax만 가져 **forward 재실행 필요(1 GPU 수 시간)**. 게이트: block16 Δ≥+1.0 → 학습 게이트 정당화 / <+0.5 → **모든 실현 신호 소진 → H16 실용적 폐쇄**(학습 게이트는 confidence조차 못 쓰는 신호를 찾아야 = 매우 낮은 사전확률, user 최종 판단).

## D. 논문 회수 (성패 무관 확정 자산)

- **믹서 비병목의 통제 실증 2건**(연속·이산) — "우리 단순 gated-MLP 트렁크가 정통"(MM1) 주장을 자체 A/B로 뒷받침.
- **모달 상보성의 전역/국소 분해 + anti-consensus 구조**: 전역 잉여(drop≈0) ∧ 국소 상보(16~64px +2~4.6) ∧ 그 상보가 합의로 접근 불가(소수-옳음). 융합 실패의 **위치와 성격**을 정량화 — limitation/분석 절의 강한 소재.

## E. 원장/계획 반영

- **H17 신설**: "명시적 모달간 cross-attention 트렁크가 gated-MLP를 이긴다" → ✗ 반증(legal −2.05, 15.9× params). connector-2차(MM1) 실증.
- **H16**: 개방 유지, 전망 하향(anti-consensus) + #15b confidence가 마지막 관문. #15 val 음성, test 대기.
- plan: #12 완료(MLP 우위), #15 val-음성/test-대기 + #15b(confidence, forward 재실행) 신설.

관련: [2026-08-20-oracle-realizability-control-verdict.md](2026-08-20-oracle-realizability-control-verdict.md) · [research/hypothesis-ledger.md](../../research/hypothesis-ledger.md) H16/H17 · [decisions/2026-08-17-p50-map-modal-alignment-pretraining-proposal.md](../../decisions/2026-08-17-p50-map-modal-alignment-pretraining-proposal.md)(믹서 2차 → 정렬 사전학습 1차, 이 판정이 근거 강화)
