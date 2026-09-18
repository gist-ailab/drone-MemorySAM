---
legacy_id: 22
legacy_file: 22_supervisor_taskboard.md
moved: 2026-07-08
---

# 🎯 Supervisor 태스크보드 (세션별 목표 태스크 할당)

> 🔴 **STALE (2026-08-08 판정)**: 이 보드는 2026-07-03 이후 미갱신 — 전 항목이 P29~P31/B200 시대라 현재와 불일치(예: Det 목표는 실제로는 0.8501@2026-07-04 달성, 현재 0.9321). **현재 상태의 단일 출처 = status/current.md.** Supervisor 세션이 재작성하기 전까지 이 보드의 태스크 할당을 따르지 말 것.

> 관리: **Supervisor 세션(목표 확인 및 방향 설정봇)**. 최종 갱신: 2026-07-03 v2 — 공식 목표(논문 publish / 국가 R&D) 반영 (세션 a1282180)
> **사용 규칙**:
> 1. 각 세션은 **세션 시작 시 + 큰 작업 단위 완료 시** 이 문서에서 자기 역할 섹션을 읽는다.
> 2. 태스크 완료/진행 시 해당 행의 상태를 직접 갱신한다 (`☐ → 🔄 → ✅`, 완료일 기입).
> 3. 새 태스크 추가·우선순위 변경은 Supervisor 세션이 담당. 다른 세션은 "제안" 칸에 append만.
> 4. 세션 식별: 사용자가 역할 키워드로 호출 (예: "코딩봇" → Det 트랙 섹션).

---

## 0. 전체 목표 (North Star — 공식, 2026-07-03 사용자 설정)

**모든 세션의 역할 = ① 연구논문 publish (Segmentation 트랙) + ② 국가연구개발과제 R&D 목표 달성 (Detection 트랙)을 위한 액션이다. 모든 실험 수치는 반드시 아래 기준 수치와 비교해 보고한다.**

### ① Seg — 논문 publish 목표
| 벤치마크 | 목표 수치 | 현재 우리 최고 | 갭 |
|---|---|---|---|
| **DELIVER** (all modalities) | **val ≥66.51 / test ≥56.71** | P29 val 63.20 / test 54.34 (P31 학습 중: ep32 60.71/52.29 ↑) | val −3.31 / test −2.37 |
| **MUSES** | **mIoU val 79.72 / test 79.49 (SOTA 수준)** | **미측정** — 우리 스택 MUSES 벤치 셋업/러닝 필요 | − |
| **MULTIAQUA** | 실행 예정 (기존 최고 M-score 82.10, P9/P22) | 82.10 | − |

### ② Det — 국가 R&D 과제 목표
| 항목 | 목표 수치 | 현재 | 갭 |
|---|---|---|---|
| poongsan indoor **mAP50** | **0.85** | 우리 스택 0.446 (P29-Det ep9, v2) · P31.1-Det ep4 0.4619 (clip-v3 — v2와 직접 비교 불가) · YOLO RGB-only 기준점 0.864 (label-v3) | ~0.40 |

**핵심 판정(2026-07-03)**: YOLO RGB-only가 label-v3에서 0.864로 목표 돌파 → **데이터 무죄, 우리 스택 유죄 확정** (doc 19 E1.1b/c). Det 전략 = head는 빌리고(fusion만 우리 것), 주장은 "동일 head에서 RGB-only 대비 악조건 delta".

---

## 1. 코딩봇 — Det 트랙 (세션 90ef43c0 계열)

현재: jarvis GPU1-4에서 P31.1-Det(det_P31_v3clip) 학습 중. ep4 mAP50 0.4619.

| # | 태스크 | 우선순위 | 상태 |
|---|--------|---------|------|
| D1 | P31.1-Det clip-v3 학습 완주 → best ckpt 확보 | 🔴 | 🔄 학습 중 (jarvis, ~28min/ep) |
| D2 | **split 비교성 확보**: P31.1-Det best ckpt를 **v2 split test로도 eval** — clip-v3 수치는 v2와 직접 비교 불가 (doc 19 §7 낙관편향 경고). P29 0.446과의 우열은 v2 기준으로만 판정 | 🔴 | ☐ |
| D3 | E2.1: 우리 파이프라인 `MODALS=['img']` RGB-only vs 3-modal — fusion 득실 분리 (doc 19 Phase 2) | 🟠 | ☐ |
| D4 | E2.2: 같은 FCOS head에서 mean vs ReliabilityAnchoredRouter — router 단독 기여 측정 | 🟠 | ☐ |
| D5 | B2: `REQUIRE_ALL_MODALITIES` 폐지 + missing-modality dropout → train 5,862→13,712장 복원 (egofill lidar 활용 가능, doc 21) | 🟠 | ☐ |
| D6 | B1: COCO-pretrained head 이식 검토 (RT-DETR/Deformable-DETR) — D2~D4 결과 본 후 착수 | 🟡 | ☐ |
| D7 | recipe 수리: best-ckpt 기준을 mAP→**mAP50**으로, ep9-피크 후 하락(LR 스케줄) 원인 수정 | 🟡 | ☐ |

## 1.5 코딩봇 — Seg 트랙 (세션 81910b16 "p31 model implementation" 계열) — 논문 publish 직결

| # | 태스크 | 우선순위 | 상태 |
|---|--------|---------|------|
| S1 | P31 seg 학습 완주(B200 RUN-9) → best ckpt로 DELIVER val/test 최종 측정, **목표 66.51/56.71 대비 갭 보고** | 🔴 | 🔄 학습 중 |
| S2 | **MUSES 벤치마크 셋업 + 우리 스택(P29/P31 best) 학습·평가** — SOTA 79.72/79.49 대비 위치 확인. 논문 벤치 테이블 필수 (수치 canonical = research_vault/relatedworks/09) | 🔴 | ☐ |
| S3 | MULTIAQUA에서 P31 실행 (기존 최고 M-score 82.10 P9/P22 대비) | 🟠 | ☐ |
| S4 | P31이 P29 미달/근접 시: doc 20 잔여 레버(consistency bias, complementary assignment 등) 중 다음 1개 선택·구현 | 🟡 | ☐ |

## 2. 실험모니터링봇 (세션 4e9bdc6f 계열, 2h 주기)

| # | 태스크 | 우선순위 | 상태 |
|---|--------|---------|------|
| M1 | B200 P31 seg (RUN-9) + jarvis det_P31_v3clip (RUN-10) 2h 주기 추적, doc 15 append | 🔴 | 🔄 진행 중 |
| M2 | **트리거**: P31 seg Day-Val이 P29(63.20) 추월 시 → doc 01 스냅샷 갱신 + Supervisor/사용자에게 보고 | 🔴 | ☐ (ep32 60.71, -2.5) |
| M3 | **트리거**: det ep9 eval 수치 확인 — ep4 0.4619 대비 상승/하락 판정 (P29는 ep9 피크 후 하락했음 — 동일 병리 감시) | 🔴 | ☐ |
| M4 | P31 신규 로깅 `p31/*` (per-modal reliability AUROC, router-w) 정상 기록 여부 1회 검증 — event/LiDAR AUROC>0.5 회복이 P31 calibration loss의 성패 지표 | 🟠 | ☐ |

## 3. 그림봇 (세션 40153b30 계열)

| # | 태스크 | 우선순위 | 상태 |
|---|--------|---------|------|
| F1 | **worktree `p28-figures` 미커밋 산출물 커밋/push** (figures/p28_rbma_figure.py + 08 문서 §15) — 유실 방지 | 🔴 | ☐ |
| F2 | doc 08이 P26까지만 커버 — P29/P30/P31(seg) 아키 피규어 추가 (P31이 P29 추월 확정되면 P31 우선) | 🟡 | ☐ |
| F3 | Det 트랙 피규어: RBMA-fused backbone + det head 파이프라인 (논문 det 섹션용, D6 방향 확정 후) | 🟡 | ☐ |

## 4. 리팩토링/코드분석봇 (세션 e5dff7a7 계열)

| # | 태스크 | 우선순위 | 상태 |
|---|--------|---------|------|
| R1 | **`.wandb_key` 시크릿 rotate + git history purge** (origin에 이미 커밋됨 — 보안) | 🔴 | 🔄 부분 (2026-07-08 재구조화 PR에서 추적중단+gitignore — **key rotate와 history purge는 여전히 필요**) |
| R2 | doc 19가 경고한 stale 경로 정리: repo 루트 `objdet/`·`configs/det/det_P9_base.yaml`(구버전 stride) — worktree(p30-det/p29-det) 최신본과의 관계 명시 or 아카이브 | 🟠 | 🔄 부분 (2026-07-08 configs 데드군 27개 archive/ 이동; det_P9_base.yaml·objdet 구본 관계 명시는 잔여) |
| R3 | 리팩토링 후속: 각 서버(B200/jarvis)에서 develop pull 시 학습 스크립트 무결성 확인 (진행 중 학습은 pull 금지) | 🟠 | ☐ (**재구조화 PR 병합 후 필수** — 구경로 shim·symlink 있으나 pull은 학습 종료 후) |
| R4 | 데드 outputs 아카이브 이동 | 🟠 | ✅ 2026-07-08 **전량 완료** — 18개 디렉토리 ~176G → `/drone_nas/home/jemo_archive/MemorySAM_dead_outputs_20260708/` (HDD2 경유분 포함 재이동, HDD1 ~170G 회수, HDD2 쓰기 정상화. `outputs/ARCHIVE_MANIFEST.md`) |

## 5. Supervisor (이 문서 관리 — 세션 a1282180 계열)

| # | 태스크 | 상태 |
|---|--------|------|
| SUP1 | 태스크보드 생성 + 00_INDEX 등록 | ✅ 2026-07-03 |
| SUP2 | 주기적으로 각 세션 transcript 감사 → 방향 이탈/중복 작업 감지 시 보드 갱신 | ☐ |

---

## 제안 (다른 세션이 append)

- (비어 있음)
