---
created: 2026-08-08
author: fable (MMSAM discussion 세션)
---

# 가설 원장 (Hypothesis Ledger) — canonical

> **역할**: 계보 전체(P1~P48)에서 제기된 가설의 **검증 수단·판정·근거를 한 표로 유지**하는 단일 출처.
> 새 모델을 제안하기 전 **반드시 이 표를 확인** — ✗ 판정 가설의 재제안 금지, ⚠️ 미결 가설은 재판정 절차부터.
> 논문 Table(ablation/analysis 구성)의 뼈대이기도 하다. 갱신 규칙: 판정이 바뀌면 행을 덮어쓰고 근거 링크를 교체(이력은 근거 문서에 남는다).

## 판정 표

| # | 가설 | 검증 수단 | 판정 | 핵심 수치 | 근거 |
|---|---|---|---|---|---|
| H1 | 학습된 게이트로 모달 가중 (UAMM·SoftMoE·quality gating) | P10~P27 12세대 | ✗ 반증 | 전원 P9 미돌파 (gate 상수수렴) | [models/arch-evolution.md](../models/arch-evolution.md), [status/history-2026H1.md](../status/history-2026H1.md) |
| H2 | 무학습 신뢰도를 attention logit에 additive bias 주입 (RBMA) | P28~P36, 2백본 | ✗ 반증 | P32 순손해 p=4.5e-22; DINOv3 계보 Δ≈0 | [experiments/analysis/p32-verification-p33v2.md](../experiments/analysis/p32-verification-p33v2.md) |
| H3 | 추론 시 재가중 (gate/calib/veto) | P36~P39 토글 A/B | ✗ 반증(유해) | fog_night·thin-class에서 off 시 +26~36 | [experiments/analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md](../experiments/analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md) |
| H4 | **추출 수준 조건 전문화** (조건×클래스 어댑터, CEA) | **oracle 프로브** (조건 GT + 전용 가중치 = 상계) | ✗ 반증 | oracle Δ(fog_night) +0.21 < 게이트 +1.0; night가 주야갭 4.33 중 +0.02만 회수 | [decisions/2026-08-08-condexpert-adapter-probe-proposal.md](../decisions/2026-08-08-condexpert-adapter-probe-proposal.md) §6·§7 |
| H4′ | 평균 최적성 함정 (조건 혼합 학습이 조건별 최적을 가림) | H4 프로브의 G-P2 | ✗ 반증 | 회수할 잉여 없음 (+0.02/4.33) | 동상 §7 |
| H5 | 조건×모달 상호작용의 실재 | drop-modal ablation | ✓ 확인 | drop-lidar dMIoU: day 0.64 vs fog_night 7.19~7.39 | SOTA 진단 artifact(2026-08-08) B절 |
| H5′ | (H5의 귀결) 그 상호작용은 **현행 정적 구조가 이미 소진** — 남은 갭은 배분이 아니라 **정보** 부족 | H4+H5 종합 | ✓ 채택 | oracle 상계 +0.2 | H4 §7 |
| H6 | 표현력(백본)이 지배 변수 | SAM2↔DINOv3 통제 프로브 | ✓ 확인 | +11.6 (계보 최대 단일 변수) | ProbeA1(analysis_logs `ProbeA1_dinov3_20260712/`), [research/novelty-and-related-work.md](novelty-and-related-work.md) |
| H7 | 학습 해상도 | 768² vs 1024² 동일 모델·레시피 | ✓ 확인 | val +1.15 / test +2.01 (양쪽 val.py@1024) | [status/current.md](../status/current.md) 활성런 절, 커밋 d23bc39 |
| H8 | 학습 전용 클래스 prototype 손실 (P46-C3, 클래스축) | λ sweep + base 대조 + seed | ✓ 확인(축 특이적) | DELIVER test +1.35~1.74, RailTrack 4.02→67.69; **MUSES 이식 −0.765**(조건축엔 무효) | [experiments/analysis/2026-08-06-p46-c3only-fair-eval-final.md](../experiments/analysis/2026-08-06-p46-c3only-fair-eval-final.md) |
| H9 | 쿼리 경로 이중화가 기여 | module ablation | ✗ 무효(복제) | query 순기여 −0.09, 픽셀 98.9% dense와 동일 | [experiments/analysis/2026-08-05-p46-module-ablation-query-nooop.md](../experiments/analysis/2026-08-05-p46-module-ablation-query-nooop.md) |
| H10 | 쿼리에 인스턴스 감독을 주면 things PQ 회복 (P48) | PQ 게이트 | ⚠️ 미결·동결 (취소 2026-08-10, user) | PQ는 비경쟁축(경쟁 논문 전부 mIoU) — 재판정 실험 취소, 미결인 채 동결 | [experiments/analysis/2026-08-06-pq-perclass-vs-instance-density.md](../experiments/analysis/2026-08-06-pq-perclass-vs-instance-density.md) |
| H11 | radar 모달 기여 (MUSES) | drop ablation + 공식 test 2회 | ✗ 반증 | 4모달 −0.217, 야간 −0.376 | [experiments/analysis/2026-08-04-muses-radar-night-harm.md](../experiments/analysis/2026-08-04-muses-radar-night-harm.md) |
| H12 | L 이상으로 백본을 더 키우면(H+, 840M) 표현력 천장이 더 열린다 | ProbeA2 — frozen S+/B/L/H+ + 공용 head, MUSES RGB | ⚠️ 중간대역(포화 근접) | Δ(H+−L)=+0.52(게이트 +1.5 미만·+0.5 근접) — S+→L 구간(+8.82)과 대조적으로 급격한 수확체감. 조건별로는 야간·악천후 +3.7~4.7 vs clear_day −3.06(상쇄) | [experiments/analysis/2026-08-09-probea2-backbone-scaling.md](../experiments/analysis/2026-08-09-probea2-backbone-scaling.md) |
| H12′ | (하한 방어) S+(~29M, Swin-T 용량 정합)에서도 우리 스택 성능이 유지된다 | 동 프로브의 G-A2-하한 | ✗ 반증 | Δ(L−S+)=+8.82 (>3.0) — "방법 기여는 대형 백본 전제"로 논문 스코프 정직 공개 필요, 용량 정합 방어 불가 | 동상 |

## 종합 — 계보가 확립한 명제 (2026-08-09 갱신)

1. **적응 가설 계열은 3단계 전부에서 닫혔다**: 융합 가중(H1·H2) → 추론 재가중(H3) → 추출 수준 oracle 상계(H4). 치팅을 줘도 +0.2라는 상계가 있으므로, 학습 가능한 어떤 조건부 메커니즘도 이를 넘을 수 없다. **적응형 기제의 재제안 금지.**
2. **상호작용은 실재하되 이미 소진됐다**(H5·H5′): 남은 격차의 원인은 *배분*이 아니라 *정보*. 이득은 표현력(H6)·해상도(H7)·학습 신호(H8)·데이터에서만 나온다.
3. **축 특이성**(H8): 클래스축 처방(prototype)은 클래스축 붕괴(DELIVER)에만 통한다. 축이 다른 벤치에 이식하면 손해 — 처방 전에 축 진단 선행.
4. **표현력도 L 이상에서는 급격히 소진된다**(H12): SAM2→DINOv3-L(+11.6, H6)·S+→L(+8.82, H12′)은 크지만, L→H+(+0.52, H12)는 수확체감 — "더 큰 고정 백본으로 바꾸면 된다"는 단순한 해법의 여지도 좁다. 단 완전히 닫힌 결론은 아니다(7B 미측정, 중간대역).
5. **격차 축소 시도 세 갈래(배분·해상도·순수 스케일 확대) 모두 정체 근처**: 배분(H4, 폐기) · 학습 해상도 상승(H7 자체는 확인이지만 08-09 val/test 역발산으로 DELIVER 게이트는 미달) · 백본 추가 확대(H12, 중간대역/사실상 포화)는 전부 "더 하면 된다"가 아니었다. 남은 유력 축은 **데이터**(양·품질) 또는 **아키텍처 신규 설계**로 좁혀진다.
6. **논문 서사**: ①상호작용 실재(H5) → ②적응 기제 전 수준 실패 + oracle 상계(H1~H4) → ③작동하는 것은 표현+학습신호(H6~H8), 단 표현력은 L 근방에서 포화(H12) → ④두 벤치 (근)SOTA + 정직한 스코프 공개(대형 백본 전제, H12′). 실패가 서사의 증거가 되는 구조.

관련: [novelty-and-related-work.md](novelty-and-related-work.md)(노벨티 canonical) · SOTA 진단 artifact "MemorySAM — SOTA까지 무엇으로 가는가"(2026-08-08) · [decisions/2026-08-08-condexpert-adapter-probe-proposal.md](../decisions/2026-08-08-condexpert-adapter-probe-proposal.md)
