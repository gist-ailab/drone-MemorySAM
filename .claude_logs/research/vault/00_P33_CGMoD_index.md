---
title: "P33_CGMoD — 폴더 인덱스 / 로그"
tags: [P33, CG-MoD, index, folder-readme]
created: 2026-07-08
updated: 2026-07-10
---

# P33_CGMoD — 폴더 인덱스

> **이 폴더는 뭔가**: MemorySAM 계열 **P33 (CG-MoD = Competence-Gated fusion + Modality Dropout)** 설계·검증 문서를 모은 곳. P32(CoRB) 실패 분석에서 도출된 차기 세그 모델 트랙.
> **P33 핵심**: P32의 결론("신호는 맞고 라우팅은 실패" + 지배 원인은 class-transfer)에 따라, ① class-transfer 복구(RCS+text-anchor+masked-consistency)를 1순위로, ② dropout+distillation으로 event/LiDAR drop-Δ 양수화, ③ soft competence gate(corr_veto 입력), ④ CoRB attn-bias 제거.

## 파일 목록

| 파일 | 무엇 | 비고 |
|---|---|---|
| `00_P33_CGMoD_index.md` | **이 파일** | 다른 세션 진입점 |
| `P33_v2_설계개정_20260708.md` | **P33-v2 개정 설계** — 원안(CG-MoD) 적대적 비판 + 문헌 3축 반영, 모듈/config/ablation/kill criteria | **여기부터 읽어라** |
| `architecture/P33_v2_설계확장_구현상세_20260710.md` | **P33-v2 설계확장(구현 상세)** — 미구현 모듈(M0 절차·M1 class-transfer·M2 KD·M3 완전형 게이트)을 코드 seam(파일:라인)·공식·config·ablation 게이트 수준으로 확장. 코드 "M1/M2/M3" ↔ 설계-v2 번호 명칭 충돌 정리 포함 | **구현 착수 시 이것부터** |
| `../P32_CoRB/P32_정량검증_실패분석_20260708.md` | 설계 근거가 되는 P32 검증 리포트 | [[issues/P32_정량검증_실패분석_20260708]] |
| (repo) `26_p33_design.md` | 원안 CG-MoD (perimage-viz 워크트리 브랜치) | `git show fcf3857:.claude_logs/26_p33_design.md` |

## 상태

- 원안(CG-MoD) 설계: ✅ (2026-07-07, repo doc 26)
- P32 4축 독립 검증: ✅ PASSED with corrections (2026-07-08 멀티에이전트)
- **P33-v2 개정안: ✅ 작성 완료 (2026-07-08)**
- 부분 구현 develop 병합 (2026-07-08, c441f1d/b7dbcee): CAL+CF-lite(P33.1 학습 중, B200 RUN-16), MD(P33.2 staged), CoRB-off — **설계-M1(class-transfer)·distill·완전형 게이트는 미구현**
- **설계확장(구현 상세): ✅ 작성 완료 (2026-07-10)** — `architecture/P33_v2_설계확장_구현상세_20260710.md`, ablation 사다리 재정렬(P33.3=RCS+text-anchor, P33.4=masked consistency+KD)
- 다음 게이트: **M0-a SOTA per-class test 삼각측량 + M0-c corr_veto AUROC 재측정** (P33.2 종료 전 완료) → M1 타깃(Wall/TL) 확정 → P33.3 구현 착수
- Global escape: P33.2 후 test <55.5 또는 val <65.5 → 카드 A(DINOv3-RBMA) 전환 ([[ideas/brainstorm_next_arch_20260708]])

## 관련

- [[00_P32_CoRB_index]] · [[00_MOC_26_MultimodalSeg]] · [[PROJECT_TRACKING_26_MultimodalSeg]]
