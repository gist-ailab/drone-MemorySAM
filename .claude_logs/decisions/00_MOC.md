# 🗺 decisions/ — MOC (Map of Content)

> 폴더 역할: **설계 제안·계획·감사 문서** — 날짜 prefix(`YYYY-MM-DD-`)로 시점 고정. 결정의 실행 결과는 status/experiments에 기록.

> 갱신 2026-08-08: P36~P49 설계 계보 8건 일괄 등록(미등록 상태였음).

| 파일 | 한줄설명 | legacy_id |
|------|----------|-----------|
| [2026-08-08-condexpert-adapter-probe-proposal.md](2026-08-08-condexpert-adapter-probe-proposal.md) | **조건×클래스 어댑터(CEA) 방향 — oracle 프로브 선행 제안** (가칭 P49) — 상태: 제안, 프로브 게이트 대기(학습 미기동) | — |
| [2026-08-05-p48-instance-supervision-proposal.md](2026-08-05-p48-instance-supervision-proposal.md) | P48 — 쿼리 경로 인스턴스 감독(Instance Supervision) 제안 — 🔴 **상태: 폐기(2026-08-06)**, 사전등록 게이트(things PQ>30) 미달(실측 22.87) | — |
| [2026-08-03-p47-mub-muses-proposal.md](2026-08-03-p47-mub-muses-proposal.md) | **P47-MUB MUSES Uni-modal Balance & 투영 밀도화** — 진단 재정의(병목=clear/day RGB under-optimization, 야간 아님). D-1 lidar projected_to_rgb_dgf 교체(비용0) + D-2 modality-laziness 억제 aux head. 게이트: val≥82.62(seed2 초과) | - |
| [2026-07-28-p46-classtransfer-recovery-proposal.md](2026-07-28-p46-classtransfer-recovery-proposal.md) | P46-CTR Class-Transfer Recovery — DELIVER SOTA 공략 제안, per-class 도메인 전이 붕괴가 지배 원인이라는 진단 기반 | — |
| [2026-07-24-p43-p45-cvpr-sota-proposal.md](2026-07-24-p43-p45-cvpr-sota-proposal.md) | **P43 PanopticDual(PQ SOTA 축)·P44 BMR(모달 재균형)·P45 FogStyle** — 딥리서치 6축 교차, MUSES PQ가 유일한 현실적 SOTA 축 판정(Codabench 실측) | - |
| [2026-07-23-p42-lidar-forcing-proposal.md](2026-07-23-p42-lidar-forcing-proposal.md) | P42 — lidar-강제(조건부 img 마스킹) + aux deep-sup 제안, MUSES 병목=비RGB 미사용(img 과지배) 진단 | — |
| [2026-07-22-p41-fusion-spectral-discrimination-proposal.md](2026-07-22-p41-fusion-spectral-discrimination-proposal.md) | P41 — Fusion Spectral Collapse: 판별 우선·조건부 개입 제안, fusion 단계 rank 붕괴 진단(양성/음성 판별 필요) | — |
| [2026-07-21-p39_1-p40-rank-rca-proposal.md](2026-07-21-p39_1-p40-rank-rca-proposal.md) | P39.1(즉시 수리)+P40(RCA-Fusion) 제안 — 저rank 압축+융합 미활용 2단 실패의 문헌 기반 처방(딥리서치 3편 교차) | — |
| [2026-07-20-p39-dual-path-compete-proposal.md](2026-07-20-p39-dual-path-compete-proposal.md) | P39 — Dual-Path Compete(DPC) 제안 — 실패-키 문서(2026-07-20) 전 키를 규칙으로 변환, 단일 아키텍처로 DELIVER·MUSES 모두 커버 | — |
| [2026-07-16-p36-novelty-critical-review.md](2026-07-16-p36-novelty-critical-review.md) | P36 노벨티 비판적 리뷰 — "무엇이 살아남고 무엇을 내려놓아야 하나", legal(val-best ckpt) 기준 정직한 수치 정리 | — |
| [2026-07-07-p33-cgmod-design.md](2026-07-07-p33-cgmod-design.md) | **P33 CG-MoD 설계** — competence-gated hard fusion + 비대칭 modality dropout. P32 per-image 진단(misalloc 51.6%, event/LiDAR competence≈16) 1:1 처방 + 관련연구 노벨티 방어(C1~C4). 상태=설계 | — (2026-07-28 회수, 구 `experiments/2026-07-07_P33_design.md`) |
| [2026-07-05-p32-seg-arch-proposals.md](2026-07-05-p32-seg-arch-proposals.md) | **P32 아키텍처 제안 5종(A~E)** — 조건 적응형 라우팅 재설계. 라우팅 실패 근본원인 R1~R4 분해 + lit-check TODO. 상태=proposal(전부 미구현) | 23 (2026-07-28 회수) |
| [2026-07-03-train-eval-optimization-audit.md](2026-07-03-train-eval-optimization-audit.md) | 학습/평가 코드 최적화 감사 (hierarchy 리팩토링 Phase D) | 20 |
| [2026-07-02-p31-redesign-proposal.md](2026-07-02-p31-redesign-proposal.md) | P31 재설계 제안 — research vault 전수 매핑 기반 (Seg core + 학습 레버 + Det 분리 트랙) | 20 |
| [2026-06-16-sam3-porting-plan.md](2026-06-16-sam3-porting-plan.md) | SAM3 RBMA 포팅 플랜 & 체크리스트 | 11 |
