# 🗺 decisions/ — MOC (Map of Content)

> 폴더 역할: **설계 제안·계획·감사 문서** — 날짜 prefix(`YYYY-MM-DD-`)로 시점 고정. 결정의 실행 결과는 status/experiments에 기록.

| 파일 | 한줄설명 | legacy_id |
|------|----------|-----------|
| [2026-07-24-p43-p45-cvpr-sota-proposal.md](2026-07-24-p43-p45-cvpr-sota-proposal.md) | **P43 PanopticDual(PQ SOTA 축)·P44 BMR(모달 재균형)·P45 FogStyle** — 딥리서치 6축 교차, MUSES PQ가 유일한 현실적 SOTA 축 판정(Codabench 실측) | - |
| [2026-07-07-p33-cgmod-design.md](2026-07-07-p33-cgmod-design.md) | **P33 CG-MoD 설계** — competence-gated hard fusion + 비대칭 modality dropout. P32 per-image 진단(misalloc 51.6%, event/LiDAR competence≈16) 1:1 처방 + 관련연구 노벨티 방어(C1~C4). 상태=설계 | — (2026-07-28 회수, 구 `experiments/2026-07-07_P33_design.md`) |
| [2026-07-05-p32-seg-arch-proposals.md](2026-07-05-p32-seg-arch-proposals.md) | **P32 아키텍처 제안 5종(A~E)** — 조건 적응형 라우팅 재설계. 라우팅 실패 근본원인 R1~R4 분해 + lit-check TODO. 상태=proposal(전부 미구현) | 23 (2026-07-28 회수) |
| [2026-07-03-train-eval-optimization-audit.md](2026-07-03-train-eval-optimization-audit.md) | 학습/평가 코드 최적화 감사 (hierarchy 리팩토링 Phase D) | 20 |
| [2026-07-02-p31-redesign-proposal.md](2026-07-02-p31-redesign-proposal.md) | P31 재설계 제안 — research vault 전수 매핑 기반 (Seg core + 학습 레버 + Det 분리 트랙) | 20 |
| [2026-06-16-sam3-porting-plan.md](2026-06-16-sam3-porting-plan.md) | SAM3 RBMA 포팅 플랜 & 체크리스트 | 11 |
