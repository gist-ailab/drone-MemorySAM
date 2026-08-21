# 📁 .claude_logs 인덱스 (Master Index)

> 최종 업데이트: 2026-08-08 (MOC 전면 재등록 — 미등록 40건 해소; 폴더 재구조화 자체는 2026-07-08)
> 이 폴더의 **front door**. 새 세션은 `CLAUDE.md` → **이 인덱스** → `status/current.md` → 작업 폴더 순으로 진입.
> 각 폴더의 `00_MOC.md`가 그 폴더의 문서 목록/역할을 안내한다.

---

## 🧭 새 세션 권장 읽기 순서

1. `../CLAUDE.md` — 세션 규칙 + 프로젝트 개요 (역할 키워드 시 [meta/bot-roles.md](meta/bot-roles.md) 먼저)
2. **이 인덱스(00_INDEX)** — 어디에 뭐가 있는지
3. [status/current.md](status/current.md) — **현재 상태 스냅샷 (단일 출처)**: 지금 무엇을 하는 중인지
4. [meta/taskboard.md](meta/taskboard.md) — 내 세션에 할당된 태스크 확인
5. 작업 폴더의 `00_MOC.md` → 해당 문서로 이동

> ✋ **파일 생성·코드 추가·브랜치 생성 전**: [meta/conventions.md](meta/conventions.md) — 구조 유지 규칙(git develop 기준, 문서 배치, MODEL_REGISTRY, configs 명명) 필수 확인.

---

## 🗂 폴더 구조

| 폴더 | 역할 | 핵심 문서 |
|------|------|-----------|
| [status/](status/00_MOC.md) | **현재 상태 + 진행 이력** (구 01 분할) | [current.md](status/current.md) = 스냅샷 단일 출처 · history-2026H2/H1 |
| [models/](models/00_MOC.md) | 모델 아키텍처 | [arch-evolution.md](models/arch-evolution.md)(canonical) · figures-ascii · explain/(버전별 노트) · [p44-bmr-implementation.md](models/p44-bmr-implementation.md) |
| [experiments/](experiments/00_MOC.md) | 실험 기록 | [registry.md](experiments/registry.md)(허브) · [log.md](experiments/log.md)(canonical) · monitor-log · analysis/ · [benchmark_roadmap.md](experiments/benchmark_roadmap.md)(벤치마크·모달리티 확장 로드맵) · [plan.md](experiments/plan.md)(GPU 잡기 전 필독) · [launch-runbook.md](experiments/launch-runbook.md) |
| [det/](det/00_MOC.md) | Detection 트랙 진단 | [diagnosis-plan.md](det/diagnosis-plan.md)(det 작업 전 필독) · p29det-data-fix · [det-cert-D1-realtime.md](det/det-cert-D1-realtime.md)·[det-cert-D1-vitsp-handoff.md](det/det-cert-D1-vitsp-handoff.md)(D1 인증 트랙) · [det-architecture-map.md](det/det-architecture-map.md) |
| [datasets/](datasets/00_MOC.md) | 데이터셋 구축/수리 | [lidar-egofill.md](datasets/lidar-egofill.md) · [muses-dataset.md](datasets/muses-dataset.md) |
| [research/](research/00_MOC.md) | 관련연구·노벨티 | [novelty-and-related-work.md](research/novelty-and-related-work.md)(canonical) · vault-digest · vault/ · related-work-raw |
| [decisions/](decisions/00_MOC.md) | 설계 제안·감사 (날짜 prefix) | P39~P49 설계 제안 계보 (최신: 2026-08-08 condexpert-adapter-probe) — 목록은 00_MOC |
| [infra/](infra/00_MOC.md) | 서버·환경 | [servers-and-launch.md](infra/servers-and-launch.md)(원격 실행 시 먼저) · environment |
| [issues/](issues/00_MOC.md) | 이슈 트래킹 | [issues-and-fixes.md](issues/issues-and-fixes.md)(코딩 전 상단 상태표 확인) |
| [meta/](meta/00_MOC.md) | 세션 운영 | **[conventions.md](meta/conventions.md)(구조 유지 규칙)** · bot-roles · taskboard |
| [archive/](archive/00_MOC.md) | 🗄 동결 문서 | P9~P14 분석 · P13 설계 가이드 |

---

## 🔢 구번호 → 새경로 매핑표 (기존 "doc N" 참조 해석용)

산문 속 "doc 19", "01 스냅샷" 같은 옛 참조는 이 표로 해석한다. 각 파일의 frontmatter `legacy_id`/`legacy_file`로도 역추적 가능.

| 구번호/구파일 | 새 경로 |
|------|---------|
| 01_project_status.md | **분할**: 상단 스냅샷 → [status/current.md](status/current.md) / history → [status/history-2026H2.md](status/history-2026H2.md)(2026-07-01~) + [status/history-2026H1.md](status/history-2026H1.md)(~2026-06-30) |
| 02_model_arch.md | [models/arch-evolution.md](models/arch-evolution.md) |
| 03_experiment_log.md | [experiments/log.md](experiments/log.md) |
| 04_issues_and_fixes.md | [issues/issues-and-fixes.md](issues/issues-and-fixes.md) |
| 05_result_analysis_P9_P12.md | [archive/result-analysis-p9-p12.md](archive/result-analysis-p9-p12.md) |
| 06_result_analysis_P13.md | [archive/result-analysis-p13.md](archive/result-analysis-p13.md) |
| 07_result_analysis_P14.md | [archive/result-analysis-p14.md](archive/result-analysis-p14.md) |
| 08_architecture_figures.md | [models/figures-ascii.md](models/figures-ascii.md) |
| 09_bot_roles_guide.md | [meta/bot-roles.md](meta/bot-roles.md) |
| 10_related_work.md | [research/related-work-raw.md](research/related-work-raw.md) |
| 11_sam3_rbma_plan.md | [decisions/2026-06-16-sam3-porting-plan.md](decisions/2026-06-16-sam3-porting-plan.md) |
| 12_novelty_and_related_work.md | [research/novelty-and-related-work.md](research/novelty-and-related-work.md) |
| 13_servers_and_launch.md | [infra/servers-and-launch.md](infra/servers-and-launch.md) |
| 14_environment_and_infra.md | [infra/environment.md](infra/environment.md) |
| 15_training_monitor_log.md | [experiments/monitor-log.md](experiments/monitor-log.md) |
| 16_failure_analysis_P28_P29.md | [experiments/analysis/2026-06-30-p28-p29-failure-analysis.md](experiments/analysis/2026-06-30-p28-p29-failure-analysis.md) |
| 17_p29det_data_fix.md | [det/p29det-data-fix.md](det/p29det-data-fix.md) |
| 18_research_digest.md | [research/vault-digest.md](research/vault-digest.md) |
| 19_det_diagnosis_plan.md | [det/diagnosis-plan.md](det/diagnosis-plan.md) |
| 20_p31_design_proposal.md | [decisions/2026-07-02-p31-redesign-proposal.md](decisions/2026-07-02-p31-redesign-proposal.md) |
| 20_train_eval_optimization_audit.md | [decisions/2026-07-03-train-eval-optimization-audit.md](decisions/2026-07-03-train-eval-optimization-audit.md) |
| 21_egofill_dataset.md | [datasets/lidar-egofill.md](datasets/lidar-egofill.md) |
| 22_supervisor_taskboard.md | [meta/taskboard.md](meta/taskboard.md) |
| P13_design_guide.md | [archive/p13-design-guide.md](archive/p13-design-guide.md) |
| research_vault/ | [research/vault/](research/vault/) |
| (repo) outputs_model_explain/*.md | [models/explain/](models/00_MOC.md) (kebab-case 사본) |

> ⚠️ "doc 20"은 두 문서(20_p31_design_proposal / 20_train_eval_optimization_audit)를 가리킬 수 있음 — 문맥으로 구별 (P31 설계 얘기면 전자).

---

## 🗂 유지보수 규칙 (기존 규칙 승계, 경로만 갱신)

- **현재 상태**는 [status/current.md](status/current.md) 한 곳만 덮어쓰기 갱신. 진행 이력은 [status/history-2026H2.md](status/history-2026H2.md) 최상단에 append (2026H2 이후 반기별 파일 추가).
- **새 실험 launch/상태 변화**는 [experiments/registry.md](experiments/registry.md) 행 갱신 + 상세는 [experiments/log.md](experiments/log.md)·[experiments/monitor-log.md](experiments/monitor-log.md).
- **새 이슈**는 [issues/issues-and-fixes.md](issues/issues-and-fixes.md) 상단 인덱스 표 + 본문 양쪽 갱신.
- **새 선행연구/노벨티**는 [research/novelty-and-related-work.md](research/novelty-and-related-work.md)(canonical) 먼저, 원시 로그는 [research/related-work-raw.md](research/related-work-raw.md). `research/vault/` 내부는 직접 수정 금지(NAS 원본에서 재동기화).
- **환경/인프라 변경**(GPU·경로·파이프라인)은 [infra/servers-and-launch.md](infra/servers-and-launch.md) 또는 [infra/environment.md](infra/environment.md)에 기록.
- **설계 제안/감사**는 `decisions/YYYY-MM-DD-<slug>.md`로 신설 (시점 고정 문서).
- 새 문서 추가 시 **해당 폴더 `00_MOC.md` + 이 인덱스** 갱신. 파일명은 kebab-case.
- 이동/개명 시 frontmatter에 `legacy_id`/`legacy_file` 유지 — 옛 참조 역추적용.
