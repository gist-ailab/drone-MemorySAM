---
title: "P34_ReliaDINO — 폴더 인덱스 / 로그"
tags: [P34, P35, P36, ReliaDINO, DINOv3, MUSES, index, log, folder-readme]
created: 2026-07-16
updated: 2026-07-16
---

# P34_ReliaDINO — 인덱스

> **이 인덱스는 뭔가**: MemorySAM 계보의 **ReliaDINO 세대(P34 / P35 / P36)** 와 **MUSES 벤치마크 진출**에 대한 **리서치 해석**의 진입점. (MOC 규약대로 전용 폴더 없이 타입 폴더 + P-prefix 파일명, 진입점은 이 루트 인덱스.)
> **수치·실험 로그의 canonical은 repo `.claude_logs/`** (헤르메스 프로토콜 §1). 여기엔 *해석과 논문 서사*만 둔다.
> **P34 핵심**: 백본을 SAM2 → **DINOv3 ViT-L/16 frozen** 으로 교체 + per-modality LoRA. P28~P33이 못 뚫던 val ~60 천장을 단번에 돌파.

## 파일 목록

| 파일 | 내용 |
|---|---|
| [[P34_ReliaDINO_노벨티정산_20260716\|products/P34_ReliaDINO_노벨티정산_20260716]] | **가장 중요** — 제안 모듈 전수 판정(무엇이 작동했고 무엇이 0인가) + 논문 서사 재정비 제안 |

## 계보 한눈에 (legal = val-best ckpt 기준, DELIVER)

| 모델 | 구성 | val | test |
|---|---|---|---|
| **P34** | gate/veto/calib + attn_bias + consistency + **PhysAug** | **68.19** | **56.62** ← 계보 test 최고 |
| P35 (fair) | P34 − attn_bias − consistency − **PhysAug** | 67.61 | 55.52 |
| P36_router | P35 + **per-class router**(P31 포트) | 67.74 | 55.62 |
| P36_physaug | P36_router + **PhysAug** | **68.76** ← val 최고 | 54.18 |

**레퍼런스**: DGFusion val 66.51 / **test 56.71** · CAFuser 68.12 / 55.80 · CAFuser-CAA **68.79** / 55.38 · CMNeXt 66.30 / 53.0.
→ **아무도 test-SOTA를 못 넘었다.** 최선 P34가 **−0.09**.

## MUSES (2026-07-15 진출)

- **공식 val 80.86** / **벤치마크 test 78.979**(Codabench 14005 제출 완료).
- 🔴 **SOTA 재정정**: ETH 벤치마크 2025-12-31 종료 → Codabench 이관. **현 test SOTA = 82.39 (GtA, camera-only)**, DGFusion 79.49는 **4위**. **우리 격차 −3.41**.
- 🔴 **서사적 함의**: **1위가 카메라 단독**이고 4모달 융합(DGFusion/CAFuser/GeminiFusion/CMNeXt)이 전부 그 아래 → **MUSES에서 멀티센서 융합이 이기고 있지 않다.** 우리 자체 ablation 결과와 방향이 일치한다(아래 노트 참조).

## 관련

- 상세 수치·진단·회수 경로 = repo `.claude_logs/experiments/monitor-log.md`
- 표준분석 산출물 = `/nas_jm/analysis_logs/{P35,P36}_eval_20260715/`
- ckpt·제출물 = `/nas_jm/drone_ckpts/{P34_final_20260713, P36_physaug_20260715, MUSES_P34_20260715}/`
- 선행 세대 = [[00_P32_CoRB_index]] · [[00_P33_CGMoD_index]]
