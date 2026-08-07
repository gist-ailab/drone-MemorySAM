# 🗺 experiments/ — MOC (Map of Content)

> 폴더 역할: **실험 기록** — 결과 로그(canonical), 실시간 학습 모니터, 실험 레지스트리, 심층 분석.

> 갱신 2026-08-08: analysis/ 전체 재등록(29건 미등록 상태였음). 새 분석 문서 생성 시 이 표에 행 추가할 것.

| 파일 | 한줄설명 | legacy_id |
|------|----------|-----------|
| [plan.md](plan.md) | **🗓 실험 계획·큐 (미래)** — 대기열/우선순위/**GPU 예약 현황**/완주 ETA. **GPU 잡기 전 필독, 띄우면 여기 갱신.** registry=과거·현재, plan=미래 (신설 2026-07-16) | — (신규) |
| [registry.md](registry.md) | **실험 레지스트리 허브** — 핵심 실험 ID/config/서버/ckpt/상태/수치 한눈표 (신설 2026-07-08) | — (신규) |
| [log.md](log.md) | 전체 결과 M-score 표 + 버전별 상세 + 진단 — **실험 canonical** | 03 |
| [monitor-log.md](monitor-log.md) | 진행 중 학습 실시간 모니터 로그 (RUN-N 단위, `/loop` 세션이 append) | 15 |
| [analysis/2026-06-30-p28-p29-failure-analysis.md](analysis/2026-06-30-p28-p29-failure-analysis.md) | P28(RBMA)·P29(SDC) 체계적 실패분석 + P30 커버리지 판정 + P31 프로토타입 | 16 |
| [analysis/2026-07-07-p32-perimage-analysis.md](analysis/2026-07-07-p32-perimage-analysis.md) | **P32(CoRB) ep108 per-image 전체 test(1897장) 분석** — corroboration ON/OFF diff(flip 0.046%, ΔmIoU −0.013), UAMM 균일·misalloc 51.6%, event/LiDAR competence≈16. 도구=[`tools/viz_features_full.py`](../../tools/viz_features_full.py) | — (2026-07-28 회수) |
| [analysis/2026-07-12-p29-p34-standard-analysis.md](analysis/2026-07-12-p29-p34-standard-analysis.md) | **P29·P31·P32·P34 표준분석 종합(동일 프로토콜)** — P34 전도메인 1위·Water 부활, SAM2 피쳐 rank-1 붕괴 vs DINOv3 정렬, additive-bias 3세대 no-op, P31 router +10~13 기여. 산출물=NAS `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/` | — (2026-07-12) |
| [analysis/2026-07-15-p34-muses-test-official.md](analysis/2026-07-15-p34-muses-test-official.md) | **P34 MUSES 공식 test 상세 (78.979)** — 조건/서브카테고리/per-class×조건 전수 사본 + 판독. CAFuser +0.48(모달 −1), Night −3.45 강건성 | — (2026-07-15) |
| (repo) [tools/README_seg_analysis.md](../../tools/README_seg_analysis.md) | **표준 분석항목 1–4 ↔ 도구 매핑 (canonical)** — 모델 분석 지시를 받으면 **먼저 읽기**. adapter 적응도(D3B)·피쳐 통계(D2N)·모듈 A/B(D5)·멀티모델 비교(compare_models) 전부 model-agnostic, P31/32/33/34+ 재사용. **새 모델 분석 코드를 새로 짜지 말 것** | — (2026-07-12) |
| [benchmark_roadmap.md](benchmark_roadmap.md) | **벤치마크 & 모달리티 확장 로드맵** — Tier-1(DELIVER/MUSES) vs Tier-2(MCubeS/FMB/PST900/MULTIAQUA) 착수 순서 + modality ablation(leave-one-out·radar 서사·센서 열화·random drop) 계획. 착수 게이트=P38 DELIVER 판정 | — (2026-07-17) |
| [launch-runbook.md](launch-runbook.md) | **학습 런치 런북** — 서버별 레시피·함정(전 서버 공통 불변식: GRADIENT_CHECKPOINT 금지, YAML LR 지수표기 금지, 기동 검증 4종 등) | — (신규) |

## analysis/ 전체 목록

| 파일 | 한줄설명 |
|------|----------|
| [analysis/2026-08-06-pq-perclass-vs-instance-density.md](analysis/2026-08-06-pq-perclass-vs-instance-density.md) | per-class PQ ↔ 인스턴스 밀도 상관 — D2 진단(작고 얇은 인스턴스 PQ 붕괴)의 직접 확인, 새 실험 없이 두 측정 교차 |
| [analysis/2026-08-06-pq-first-measurement-p48-gate.md](analysis/2026-08-06-pq-first-measurement-p48-gate.md) | PQ 첫 측정 — P48 사전등록 게이트(things PQ>30) 판정용, P47-MUB D-1 ep172 MUSES val PQ 실측 |
| [analysis/2026-08-06-p46-c3only-fair-eval-final.md](analysis/2026-08-06-p46-c3only-fair-eval-final.md) | P46-CTR C3-only fair-eval 최종 판정(val-best ckpt만 사용) — 본/seed2 val·test·RailTrack 정리표 |
| [analysis/2026-08-05-p46-module-ablation-query-nooop.md](analysis/2026-08-05-p46-module-ablation-query-nooop.md) | P46 모듈 기여도 A/B — 쿼리 경로 순기여 0(dense 복제), module_ablation.py 토글 결과 |
| [analysis/2026-08-04-sota-landscape-recheck.md](analysis/2026-08-04-sota-landscape-recheck.md) | 벤치 SOTA 지형 재확인 — DGFusion/CAFuser/MM SAM-adapter arXiv 원문 대조로 기준 수치 정정 |
| [analysis/2026-08-04-muses-radar-night-harm.md](analysis/2026-08-04-muses-radar-night-harm.md) | MUSES radar 손해의 조건별 국소화 — 공식 test 3모달 vs 4모달(+radar) 대조, 손해가 야간에 집중됨을 확인 |
| [analysis/2026-08-03-p46-c3only-lambda-sweep.md](analysis/2026-08-03-p46-c3only-lambda-sweep.md) | P46-3(C3-only) λ 스윕 DELIVER 판정 — 🔴 원 "SOTA 돌파" 주장 철회 후 두 합법 프로토콜로 재계산 |
| [analysis/2026-08-03-muses-official-test-P46-c3only-lam02.md](analysis/2026-08-03-muses-official-test-P46-c3only-lam02.md) | MUSES 공식 test 결과 — P46 C3-only λ0.2 제출, Overall mIoU 79.023 |
| [analysis/2026-07-30-p46-ctr-c1c3-railtrack-gate.md](analysis/2026-07-30-p46-ctr-c1c3-railtrack-gate.md) | P46-CTR C1+C3 ep40 — RailTrack 게이트 통과(재타깃 가설 확증), overall SOTA는 미달 |
| [analysis/2026-07-30-p46-c3only-vs-c1c3-attribution.md](analysis/2026-07-30-p46-c3only-vs-c1c3-attribution.md) | P46-CTR C3-only vs C1+C3 — 기여 귀속 판정(ep40, test-SOTA 예비 도달), C3 단독이 핵심 기제로 확정 |
| [analysis/2026-07-30-muses-drop-radar-ablation.md](analysis/2026-07-30-muses-drop-radar-ablation.md) | MUSES P39.1-rank 4모달 seed2 — drop-radar ablation, radar 무익 확정(dMIoU +0.13, 오차범위 내) |
| [analysis/2026-07-28-p44-bmr-muses-standard-analysis.md](analysis/2026-07-28-p44-bmr-muses-standard-analysis.md) | P44-BMR MUSES 표준분석 — BMR이 비RGB 사용을 P39.1/seed2 대비 못 늘림, val 이득 없음 |
| [analysis/2026-07-27-seed2-p39_1-muses-standard-analysis.md](analysis/2026-07-27-seed2-p39_1-muses-standard-analysis.md) | P39.1-seed2 MUSES 표준분석 — 프로젝트 최고 모델(val 82.62/test 79.788), VICReg가 lidar rank 대폭 확장 |
| [analysis/2026-07-27-p43-pdual-muses-standard-analysis.md](analysis/2026-07-27-p43-pdual-muses-standard-analysis.md) | P43-PanopticDual MUSES 표준분석 — lidar adapter 강하게 작동, lidar rank 회복(VICReg OFF에도 건강), 융합 병목 잔존 |
| [analysis/2026-07-25-router-coverage-verification.md](analysis/2026-07-25-router-coverage-verification.md) | 학습0 검증 2건 판정 — router per-class/coverage(P43~P45 제안 §7), "클래스별 모달 특화" 강해석 기각 |
| [analysis/2026-07-22-p38-muses-feature-characterization.md](analysis/2026-07-22-p38-muses-feature-characterization.md) | P38-m2f MUSES 피쳐 특성화(§0.5 tap×method) — fusion 직후 rank·차원 급붕괴(FUSED_pf) 수치화 |
| [analysis/2026-07-21-p39-muses-standard-analysis.md](analysis/2026-07-21-p39-muses-standard-analysis.md) | P39-DPC MUSES 3모달 표준분석(공백 보완) — "lidar adapter가 죽었다" 가설 반증, 저rank 압축+융합 미활용 2단 실패로 정정 |
| [analysis/2026-07-21-p39-fog-scene-audit.md](analysis/2026-07-21-p39-fog-scene-audit.md) | MUSES fog per-scene 감사 — 파국 장면 가설 기각(worst5도 균일 분포) |
| [analysis/2026-07-20-p39-muses-fognight-rootcause.md](analysis/2026-07-20-p39-muses-fognight-rootcause.md) | P39-MUSES fog_night 붕괴 원인 규명 + P39.1 스펙 — 원인은 attention 노이즈가 아니라 lidar 표현 자체 붕괴(rank 4.6~4.8) |
| [analysis/2026-07-20-p39-earlycheck-toggles.md](analysis/2026-07-20-p39-earlycheck-toggles.md) | P39-DPC 조기판정(토글 즉검) — V1·V5·router 전부 생존, 5세대 만에 처음 신규 모듈이 no-op 아님 |
| [analysis/2026-07-20-p39-deliver-3ckpt-compare.md](analysis/2026-07-20-p39-deliver-3ckpt-compare.md) | P39-DELIVER 세 시점(ep38/60/64) 비교 — val 신기록이 test로 전이 안 됨, 손실 대부분 RailTrack 단일 클래스 |
| [analysis/2026-07-20-muses-official-test-P38-m2f-ep156.md](analysis/2026-07-20-muses-official-test-P38-m2f-ep156.md) | MUSES 공식 test 결과 — P38-m2f 3모달 ep156, Overall mIoU 79.025(프로젝트 당시 신기록) |
| [analysis/2026-07-20-module-visual-report.md](analysis/2026-07-20-module-visual-report.md) | 모듈·제안영역 시각 리포트 포인터 — 본문+그림 8장은 NAS, 실패-키 문서 수치를 그림·표로 고정 |
| [analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md](analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md) | 종합 실패-키: P38-DELIVER × P37a-MUSES 표준분석(차기 구조 설계 인계 문서) — "어디가 문제이고 어디서 실패했는가" 키 목록 |
| [analysis/2026-07-19-p38-m2f-standard-analysis.md](analysis/2026-07-19-p38-m2f-standard-analysis.md) | P38-m2f 표준분석(항목①~④) — m2f 분지는 추론 시 사실상 no-op, router 의존도 급감(구조적 신호) |
| [analysis/2026-07-18-p37a-muses-cefr-output-analysis.md](analysis/2026-07-18-p37a-muses-cefr-output-analysis.md) | P37a-CEFR MUSES 출력 분석 — 구조는 채택됐지만 "클래스별 모달리티 라우팅" 가설은 실현 안 됨(전역 틸트로 퇴화) |
| [analysis/MUSES_TEST_RESULTS_INDEX.md](analysis/MUSES_TEST_RESULTS_INDEX.md) | **MUSES 공식 test 결과 통합 인덱스(canonical)** — Codabench comp 14005 제출 전체 시간순 표(07-15~08-03) + 출처 링크 |
| [analysis/p32-verification-p33v2.md](analysis/p32-verification-p33v2.md) | P32 검증 + P33-v2 개정 포인터 문서 — 상세는 옵시디언 볼트, CoRB attn-bias 순손해 확정·지배원인=class-transfer 붕괴 |
| [analysis/p32-phase0-results.md](analysis/p32-phase0-results.md) | P32 Phase 0 결과 — Corroboration vs Self-Entropy AUROC(무학습 진단), DELIVER test 5조건 |
