# 🧾 실험 진행 원장 (Experiment Ledger) — 무조건 먼저 읽기

> **이 문서의 역할**: 모든 실험을 **시도순(=시간순)** 한 줄로 훑는 **목차/타임라인**. "지금까지 뭘 어떤 순서로 시도했고 결과가 어땠는지"를 여기서 한눈에 보고, 상세는 링크로 들어간다.
> **읽기 순서**: `CLAUDE.md` → `00_INDEX.md`(카테고리 front door) → **이 원장(시도순 타임라인)** → 개별 실험 문서.
> **역할 분담** (중복 금지):
> - 이 원장 = **시도순 목차 + 상태**(one-liner + 링크). 숫자·상세는 여기서 단정하지 않는다.
> - [03_experiment_log.md](03_experiment_log.md) = **결과 canonical**(수치·config·진단 상세).
> - [02_model_arch.md](02_model_arch.md) = **아키텍처 canonical**(모듈/forward/한계).
> - [01_project_status.md](01_project_status.md) 상단 = **현재 상태 스냅샷**(지금 돌아가는 것).
> - `experiments/` 폴더 = **실험별 분석/설계 문서**(날짜 프리픽스, 시간순 정렬). 여기 새 md를 만들면 **반드시 이 원장에 한 줄 추가**한다.
>
> **갱신 규칙**: 새 실험을 돌리거나 분석/설계 문서를 만들면 → (1) `experiments/`에 `YYYY-MM-DD_<모델>_<종류>.md`로 저장, (2) 이 원장 해당 트랙 표에 한 줄 추가(날짜·상태·링크), (3) 결과 수치는 03에 기록.

---

## 범례
상태: 🏁완료 · 🟢진행중 · ⏹중단 · 🔴실패/사망 · 🧊취소 · 📐설계만 · ⏳대기
데이터셋: **MULTIAQUA**(지표 M-score=0.75·valmIoU+0.25·testmIoU) · **DELIVER**(지표 mIoU, 25cls)

---

## 트랙 A — Seg / MULTIAQUA (M-score era, P8~P25)
> 챌린지 최선 고정: **P9 ep131 & P22 ep120 공동 1위 M=82.10** (Val 93.3 / Test 70.9). 상세·전체 M-score 표 = [03 "전체 결과 요약"](03_experiment_log.md).

| 시도 | 모델 / 핵심 변경 | 결과(요약) | 상태 | 상세 |
|---|---|---|---|---|
| P8 | ConfidenceHeadV2 + sigmoid UAMM | M 78.45 | 🏁 | [03 §P8](03_experiment_log.md) |
| **P9** | CrossModalFusionHead + max-norm UAMM | **M 81.98→82.10**(hardaug8/physaug) **최선** | 🏁 | [03 §P9](03_experiment_log.md) · [05 심층분석](05_result_analysis_P9_P12.md) |
| P10 | CrossModalFusionHeadV2 + ModalAuxHead + oracle KL | M 79.27 (test 하락) | 🧊 | [03 §P10](03_experiment_log.md) |
| P11 | P10 + MI routing loss | M 77.09 | 🧊 | [03 §P11](03_experiment_log.md) |
| P12 | Input-Conditioned Soft MoE LoRA | — | 📐 | [03 §P12](03_experiment_log.md) |
| P13~P17 | Dynamic Fusion 계열 | 실패(gate 상수수렴) | 🔴 | [03 §P14~P17 종합](03_experiment_log.md) · [06](06_result_analysis_P13.md)/[07](07_result_analysis_P14.md) |
| P19~P22 | DeBA-FP / Multi-Scale DeBA-FP | P22 M=82.10(P9 공동최선) | 🏁 | [03 §P21/P22](03_experiment_log.md) |
| P24/P25 | Quality-aware Memory Gating / Spatial Quality Gating | 진단 위주 | ⏹ | [03 §P24/P25 진단](03_experiment_log.md) |
| — | Aug/TTA/I2I/FDA/CV-enhance 실험 (I~VI) | 대부분 실패/취소 | 🔴🧊 | [03 §실험 I~VI](03_experiment_log.md) |

**교훈**: P10~P27 adaptive fusion은 전부 gate 상수수렴 병목 → P9 미돌파. **이 진단이 RBMA(트랙 B) 동기.**

---

## 트랙 B — Seg / DELIVER (RBMA era, P26~P33)
> 연구 정체성 = **RBMA (Reliability-Biased Memory Attention)**: SAM2/3 memory cross-attn **logit에 training-free reliability를 additive bias**. canonical = [12_novelty_and_related_work.md](12_novelty_and_related_work.md).
> DELIVER 기준선: CMNeXt-B2 test 53.0 · SOTA DGFusion test 56.71 / 공식목표 val 66.51·test 56.71.

| 시도 | 날짜 | 모델 / 핵심 변경 | 결과(Val / Test) | 상태 | 상세·분석 |
|---|---|---|---|---|---|
| P26 | 2026-03~04 | Per-Modality SQG + Triple-Duty 해소 + UAMM softmax | 메모리/프로브 | 🏁 | [03 §P26](03_experiment_log.md) |
| P28 | 2026-06-24 | RBMA(self-entropy) 순수 | (초기 사망 ep16) 이후 장기런 test~55.27 | 🔴→🏁 | [16 실패분석](16_failure_analysis_P28_P29.md) |
| P29 | 2026-06-30 | + SDC 조건 라우팅 | **63.20 / 54.34** | 🏁 | [02 §P29](02_model_arch.md) · [16](16_failure_analysis_P28_P29.md) |
| P30 | 2026-07-02 | class-token decoder + reliability-anchored router | 49.76 / 44.10 (붕괴) | 🔴 | [02 §P30](02_model_arch.md) |
| P31 | 2026-07-03 | Calibrated dual-reliability + MS-HR class-token decoder | **63.20 / 54.75** | 🏁 | [20 설계](20_p31_design_proposal.md) · [02 §P31](02_model_arch.md) |
| **P32** | 2026-07-05~ | **CoRB** — self-entropy→cross-modal **corroboration** bias | **64.12@ep98 / 54.79@ep108**(학습 진행중, P31 추월) | 🟢 | **[분석 experiments/2026-07-07_P32_perimage_analysis](experiments/2026-07-07_P32_perimage_analysis.md)** · Phase0=doc24(다른브랜치) |
| **P33** | 2026-07-07 | **CG-MoD** — competence-gated hard fusion + modality dropout + calibration 복원 + thin-class 강건화 | — | 📐 | **[설계 experiments/2026-07-07_P33_design](experiments/2026-07-07_P33_design.md)** |

**최신 인사이트(P32 per-image 분석 @best ep108)**: misalloc 51.6%(융합이 competence 무시)·corroboration "신호는 맞고 라우팅 실패"(flip 0.046%)·test 죽은클래스는 도메인 전이 붕괴 → **P33 설계 근거**. 산출물 `/mnt/HDD2/src/logs/P32_perimage_20260707/ep108/`.

---

## 트랙 C — Detection (Jarvis/hinton, P29-Det~P30-Det)
> 목표 mAP50 **0.85**. 지표 = mAP/mAP50/mAP75.

| 시도 | 날짜 | 모델 / 핵심 변경 | 결과(mAP50) | 상태 | 상세 |
|---|---|---|---|---|---|
| P29-Det | 2026-07 | RBMA backbone + FPN/FCOS | **0.446@ep9**(best-overall) | 🏁 | [19 진단계획](19_det_diagnosis_plan.md) · [17 데이터수정](17_p29det_data_fix.md) |
| P30-Det | 2026-07-02 | seg backbone(router+query decoder)→det | 0.2562 (small-obj collapse) | 🏁 | [19](19_det_diagnosis_plan.md) |
| egofill | 2026-07-03 | lidar egofill 데이터셋 v20260703 재학습 | — | 🟢 | [21 egofill](21_egofill_dataset.md) |

---

## 관련연구 / 인프라 (실험 아님, 포인터만)
- 노벨티·차별표: [12](12_novelty_and_related_work.md) · 리서치 다이제스트: [18](18_research_digest.md) · 볼트: [research_vault/](research_vault/)
- 서버·원격실행: [13](13_servers_and_launch.md) · 환경·경로: [14](14_environment_and_infra.md) · 이슈: [04](04_issues_and_fixes.md)
- 분석·eval 산출물 정규 위치: `/mnt/HDD2/src/logs/<model>_<kind>_<date>/`
