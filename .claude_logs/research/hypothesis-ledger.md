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
| H10 | 쿼리에 인스턴스 감독을 주면 things PQ 회복 (P48) | PQ 게이트 | ⚠️ 미결 | 게이트 적용 시점 오류 논란 — 재판정 필요 | [experiments/analysis/2026-08-06-pq-perclass-vs-instance-density.md](../experiments/analysis/2026-08-06-pq-perclass-vs-instance-density.md) |
| H11 | radar 모달 기여 (MUSES) | drop ablation + 공식 test 2회 | ✗ 반증 | 4모달 −0.217, 야간 −0.376 | [experiments/analysis/2026-08-04-muses-radar-night-harm.md](../experiments/analysis/2026-08-04-muses-radar-night-harm.md) |

## 종합 — 계보가 확립한 명제 (2026-08-08 시점)

1. **적응 가설 계열은 3단계 전부에서 닫혔다**: 융합 가중(H1·H2) → 추론 재가중(H3) → 추출 수준 oracle 상계(H4). 치팅을 줘도 +0.2라는 상계가 있으므로, 학습 가능한 어떤 조건부 메커니즘도 이를 넘을 수 없다. **적응형 기제의 재제안 금지.**
2. **상호작용은 실재하되 이미 소진됐다**(H5·H5′): 남은 격차의 원인은 *배분*이 아니라 *정보*. 이득은 표현력(H6)·해상도(H7)·학습 신호(H8)·데이터에서만 나온다.
3. **축 특이성**(H8): 클래스축 처방(prototype)은 클래스축 붕괴(DELIVER)에만 통한다. 축이 다른 벤치에 이식하면 손해 — 처방 전에 축 진단 선행.
4. **논문 서사**: ①상호작용 실재(H5) → ②적응 기제 전 수준 실패 + oracle 상계(H1~H4) → ③작동하는 것은 표현+학습신호(H6~H8) → ④두 벤치 (근)SOTA. 실패가 서사의 증거가 되는 구조.

관련: [novelty-and-related-work.md](novelty-and-related-work.md)(노벨티 canonical) · SOTA 진단 artifact "MemorySAM — SOTA까지 무엇으로 가는가"(2026-08-08) · [decisions/2026-08-08-condexpert-adapter-probe-proposal.md](../decisions/2026-08-08-condexpert-adapter-probe-proposal.md)
