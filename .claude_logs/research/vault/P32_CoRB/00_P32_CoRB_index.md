---
title: "P32_CoRB — 폴더 인덱스 / 로그"
tags: [P32, CoRB, index, log, folder-readme]
created: 2026-07-06
updated: 2026-07-06
---

# P32_CoRB — 폴더 인덱스

> **이 폴더는 뭔가**: MemorySAM 계열 **P32-B (CoRB = Corroboration-Biased Memory Attention)** 관련 리포트·그림·PDF를 모은 곳.
> 이전에는 `P32_CoRB_조사보고서.md/.pdf`가 볼트 루트에 그냥 떠 있었는데, 다른 세션도 찾기 쉽게 이 폴더로 묶었다(2026-07-06).
> **P32-B 핵심**: P28(RBMA)의 신뢰도 신호를 self-entropy → cross-modal corroboration으로 교체. LiDAR 신뢰도 AUROC 0.22→0.81 (무학습 검증).

## 파일 목록

| 파일 | 무엇 | 비고 |
|------|------|------|
| `00_P32_CoRB_index.md` | **이 파일** — 폴더 인덱스/로그 | 다른 세션 진입점 |
| `P32_CoRB_리포트.md` | **메인 리포트 (그림판)** — 6개 figure를 서사에 녹인 최신 버전 | 발표/공유용. **여기부터 읽어라** |
| `P32_CoRB_조사보고서.md` | 텍스트 상세판 (2026-07-05 작성) | 그림 없는 원본 서술. 리포트가 이걸 대체·요약 |
| `P32_CoRB_조사보고서.pdf` | 위 텍스트판 PDF 내보내기 | |
| `P32_CoRB_novelty_risk_register.md` | **위험 누락 정리 (양 축 통합)** — RBMA-mech/signal + CoRB 위협 마스터 표, 위험순 정렬 | 투고 전 반드시 확인 |
| `../relatedworks/49_corb_novelty_defense.md` | **CoRB 노벨티 방어 노트** — 4-pillar 주장, 최근접 선행연구 표(RSGMamba/MAGIC++), 5개 차별점 | [[49_corb_novelty_defense]] |
| `assets/fig0_storyboard.png` | 1장 요약 (문제→수리→veto→선택 2×2) | |
| `assets/fig1_p28_selfentropy_anticalibrated.png` | 문제: P28 self-entropy가 event/lidar에서 random 아래 | |
| `assets/fig2_corroboration_repair.png` | 수리: corroboration이 event +0.25 / lidar +0.59 | |
| `assets/fig3_veto_protects_workhorse.png` | veto 필요성: pure corr가 P31 depth .90→.28, veto가 .71 복구 | |
| `assets/fig4_signal_form_selection.png` | 신호형 선택: worst-modality AUROC 기준 corr_veto 채택 | |
| `assets/fig5_dead_modality_symptom.png` | 진단↔증상: anti-cal AUROC → drop-modality ΔmIoU≈0 (Mode C) | |

## 데이터·산출물 출처 (재현용)
- **그림 원본 + 재생성 스크립트**: `/mnt/HDD2/src/logs/P32_reliability_figs_20260706/`
  (스크립트: `plot_p28_p32_reliability.py`, 숫자 하드코딩 — 아래 로그에서 그대로 인용)
- **AUROC 수치**: `<repo>/.claude_logs/24_p32_phase0_results.md` (Phase-0 결과 A/B/C 표)
- **Mode C drop-modality ΔmIoU**: `<repo>/.claude_logs/16_failure_analysis_P28_P29.md` line 190
- **로드맵**: `<repo>/.claude_logs/23_seg_arch_proposals_P32.md`
- `<repo>` = `/mnt/HDD1/Workspace/src/Project/Drone24/detection/drone-MemorySAM`

## 노벨티 상태
- **CoRB = NOVEL as conjunction** (4-pillar: training-free × posterior-BC × N≥3 LOO consensus × additive pre-softmax into SAM2 memory, + unique-info veto). 단독 pillar는 어느 것도 새롭지 않음.
- **MUST-CITE: RSGMamba(2604.12319, #1) + MAGIC++(2412.16876, sleeper #2).** 최강 discriminator = **posterior-space Bhattacharyya** (feature-diff/cosine 아님). 상세: [[49_corb_novelty_defense]] · [[P32_CoRB_novelty_risk_register]].

## 상태
- Phase 0 (무학습 신호 진단): ✅ PASSED
- P32-B 구현: ✅ 완료 (`LoRA_Sam_P32`, config-gated, GPU smoke PASS)
- 학습: 🟡 B200 진행중 (~22h/200ep). 판정 게이트 #2 = Test mIoU vs P31 54.75 / P28 55.27, 공식목표 val≥66.51/test≥56.71
- ⚠️ 그림은 **신호 품질(AUROC) 수리**를 증명한 것 — 최종 mIoU 개선은 학습 결과 대기

## 관련
- [[P32_CoRB_리포트]] · [[P32_CoRB_조사보고서]] · [[00_MOC_26_MultimodalSeg]] · [[PROJECT_TRACKING_26_MultimodalSeg]]
