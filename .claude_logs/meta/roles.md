<!-- LABSTANDARD v1.2 -->
# 역할 배정 — drone-MemorySAM (기존 구조 상위 호환 선언)

이 레포는 2026-09-18 감사로 정리된 자체 구조를 쓴다. 기록의 흐름과 위치는 전역 작업 사이클 계약(`~/lab-status/SPEC-cycle.md`)·LAB STANDARD·역할 카드가 정하고, 이 파일은 **역할 배정·범위·기존 구조 대응표·프로젝트 예외**만 정한다(2026-10-16 개정: 기존 파일 우선 선언 삭제). `meta/bot-roles.md`는 2026-10-16부터 보관 문서이며 역할 판별에 쓰지 않는다. `meta/analysis-session-protocol.md`는 analysis 역할의 측정 절차서로 계속 쓴다.
세션 이름 `MMSAM | <역할 부분>`의 역할 부분으로 세션 시작 훅이 역할을 찾는다.

| 역할 | 세션 이름의 역할 부분(쉼표 구분) | 모델 | 이 프로젝트에서의 범위 | 전용 문서 |
|---|---|---|---|---|
| survey | trend research, trand research | opus + sonnet 서브에이전트 | 관련 연구·SOTA 동향 조사, 볼트 `26_MultimodalSeg` 기록 | 볼트 `/nas_jm/Research/26_MultimodalSeg` |
| method | 생각정리 | fable | 판정·설계 담당. 판정은 이 세션만 `lab-plan verdict <id> --analysis <근거 문서>`로 남긴다 | `.claude_logs/tasks/<항목 id>/analysis.md`(판정 근거), `experiments/protocol.md`(판정 규칙), `experiments/judgment-ledger.md`(기존 행 = 근거 서술) |
| manager | learning status monitoring, monitoring | sonnet(판단은 opus) | 기동·감시·재채점 집행. **세션 수명 예외**: `/loop` 장기 실행을 허용한다. 단 7일 만료 전(권장 3~5일마다) 또는 컨텍스트가 커지면 새로 열고, 인계는 `lab-inbox`·`status/current.md`·`status/handoff-<날짜>-manager.md`로 한다(대화 맥락이나 인계 문서 하나에 기대지 않는다) | `infra/servers-and-launch.md`, `experiments/plan.md`(GPU 예약·일시 회피·사고 기록), `meta/monitoring-session-handover.md`(2026-09-08 감시 명세, 참고용) |
| analysis | analysis, 분석 | opus(측정은 sonnet) | 새 ckpt 측정·보고(판정 안 함). 보고는 `lab-inbox send --to method --type analysis_report --ref <항목 id>` | `meta/analysis-session-protocol.md` |
| notion | Notion | sonnet(서술 요약 opus) | 노션 논문 페이지 절 단위 동기화. 빌더 실행은 이 세션만 한다(`notion_sync` 요청을 받아서) | 페이지 `Drone Object Detection for RGB-IR Fusion`(`33d05310-a165-408a-b0b8-ec4427d1fe2c`), 빌더 `.claude/skills/notion-experiment-log/paper_page_builder.py` |
| specialist-baseline | dgfusion deliver training | opus | 기준선(DGFusion) 재학습·실패 분석 보강 | unknown |

## 기존 구조 대응표
| LAB STANDARD / SPEC-cycle 파일 | 이 레포의 대응 |
|---|---|
| status/current.md (PI 블록) | `status/current.md` ①~④ 블록(헤드라인 표는 `experiments/headline.yaml` 생성물). PI 블록 = frontmatter 바로 아래(2026-09-26 추가, 값은 ①·③·④에서 옮김). **"실행 중" 표는 current.md 한 곳**(정본 = `lab-status`) |
| experiments/plan.md | 대기열 = `lab:plan-queue` 생성 블록(lab-plan, 2026-09-26 이관). GPU 예약·일시 회피·사고 기록 = plan.md(사람). plan.md의 "실행 중" 표는 2026-10-16 동결 사본(더 갱신하지 않는다) |
| experiments/protocol.md | `experiments/protocol.md`(판정 규약 정본 = 카드 문서 §0~§4) |
| experiments/registry.md | `experiments/registry.md`(사람이 관리) + 파일 끝 `lab:registry` 생성 블록(2026-09-26 추가, 파이프라인 `pipeline/results.jsonl`) |
| experiments/judgment-ledger.md | 기존 표(2026-10-16까지) = 판정 근거 서술(동결). 이후 판정 = 파일 끝 `lab:verdicts` 생성 블록(`lab-plan verdict`에서 생성) |
| research/hypothesis-ledger.md | 있음. H 표 = 2026-10-16까지의 가설 이력(동결, 근거). 열린 문제 = 파일 끝 `lab:problems` 생성 블록(PRB, `lab-plan problem`에서 생성) |
| 작업 폴더 tasks/<항목 id>/ | `.claude_logs/tasks/<항목 id>/`(2026-10-16부터 새 작업). 기존 분석 문서 `experiments/analysis/*.md`는 보존하고 해당 항목 `docs`에 경로를 적는다 |
| 판정 기록 | `lab-plan verdict` 한 곳. 카드 문서 §5·`decisions/cards/2026-Www-verdicts.md` = 2026-10-16 이전 판정의 근거 보관(새 판정을 append하지 않는다) |
| 세션 간 기록 전달 | `lab-inbox`(`.claude_logs/inbox/`). SendMessage는 실시간 대화용, 기록 아님 |
| 헤드라인 수치 | `experiments/headline.yaml`(정본) |
| 학습 종료→자동 평가 | `pipeline.toml`(DELIVER legal v2 자동, MUSES 알림만) |

## 프로젝트 추가 기록 (SPEC-cycle 밖, 이 프로젝트에만 있는 것)
| 언제 | 무엇을 | 누가 |
|---|---|---|
| 모델 아키텍처 변경(새 PNN 세대) | `models/arch-evolution.md`에 버전·변경 이유 | 변경한 세션(method 또는 구현 세션) |
| 새 선행연구·노벨티 논의 갱신 | `research/novelty-and-related-work.md`(비교표), 원시 로그 `research/related-work-raw.md` | survey / method |
| 실험 launch·상태 변화, 분석 행 | `experiments/registry.md` 사람 표 행(분석 행 형식은 analysis-session-protocol §4-4) | manager / analysis |
| 분석 문서 등록 | `experiments/00_MOC.md` "analysis/ 전체 목록" 표 | analysis |
