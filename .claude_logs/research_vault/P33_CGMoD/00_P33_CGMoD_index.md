---
title: "P33_CGMoD — 폴더 인덱스 / 로그"
tags: [P33, CG-MoD, index, folder-readme]
created: 2026-07-08
updated: 2026-07-08
---

# P33_CGMoD — 폴더 인덱스

> **이 폴더는 뭔가**: MemorySAM 계열 **P33 (CG-MoD = Competence-Gated fusion + Modality Dropout)** 설계·검증 문서를 모은 곳. P32(CoRB) 실패 분석에서 도출된 차기 세그 모델 트랙.
> **P33 핵심**: P32의 결론("신호는 맞고 라우팅은 실패" + 지배 원인은 class-transfer)에 따라, ① class-transfer 복구(RCS+text-anchor+masked-consistency)를 1순위로, ② dropout+distillation으로 event/LiDAR drop-Δ 양수화, ③ soft competence gate(corr_veto 입력), ④ CoRB attn-bias 제거.

## 파일 목록

| 파일 | 무엇 | 비고 |
|---|---|---|
| `00_P33_CGMoD_index.md` | **이 파일** | 다른 세션 진입점 |
| `P33_v2_설계개정_20260708.md` | **P33-v2 개정 설계** — 원안(CG-MoD) 적대적 비판 + 문헌 3축 반영, 모듈/config/ablation/kill criteria | **여기부터 읽어라** |
| `../P32_CoRB/P32_정량검증_실패분석_20260708.md` | 설계 근거가 되는 P32 검증 리포트 | [[../P32_CoRB/P32_정량검증_실패분석_20260708]] |
| (repo) `26_p33_design.md` | 원안 CG-MoD (perimage-viz 워크트리 브랜치) | `git show fcf3857:.claude_logs/26_p33_design.md` |

## 상태

- 원안(CG-MoD) 설계: ✅ (2026-07-07, repo doc 26)
- P32 4축 독립 검증: ✅ PASSED with corrections (2026-07-08 멀티에이전트)
- **P33-v2 개정안: ✅ 작성 완료 (2026-07-08) — 미승인/구현 대기**
- 다음 게이트: **M0 진단 3종(무학습)** → SOTA per-class test 삼각측량 결과에 따라 M1 타깃 확정
- Global escape: P33.2 후 test <55.5 또는 val <65.5 → 카드 A(DINOv3-RBMA) 전환 ([[../material/brainstorm_next_arch_20260708]])

## 관련

- [[../P32_CoRB/00_P32_CoRB_index]] · [[../00_MOC_26_MultimodalSeg]] · [[../PROJECT_TRACKING_26_MultimodalSeg]]
