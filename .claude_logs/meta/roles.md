<!-- LABSTANDARD v1.1 -->
# 역할 배정 — drone-MemorySAM (기존 구조 상위 호환 선언)

이 레포는 2026-09-18 감사로 정리된 자체 구조를 쓴다. LAB STANDARD v1.1과는 아래 대응표로 연결한다. 기존 파일·세션 규칙(CLAUDE.md Step 0~1, `meta/bot-roles.md`, `meta/analysis-session-protocol.md`)이 우선이다.
세션 이름 `MMSAM | <역할 부분>`의 역할 부분으로 세션 시작 훅이 역할을 찾는다.

| 역할 | 세션 이름의 역할 부분(쉼표 구분) | 모델 | 이 프로젝트에서의 범위 | 전용 문서 |
|---|---|---|---|---|
| survey | trend research, trand research | opus + sonnet 서브에이전트 | 관련 연구·SOTA 동향 조사, 볼트 `26_MultimodalSeg` 기록 | 볼트 `/nas_jm/Research/26_MultimodalSeg` |
| method | 생각정리 | fable | 판정·설계 담당(판정은 이 세션만, CLAUDE.md Step 1) | `experiments/judgment-ledger.md`, `experiments/protocol.md` |
| manager | learning status monitoring, monitoring | sonnet(판단은 opus) | 기동·감시·재채점·기록 | `experiments/plan.md`, `infra/servers-and-launch.md`, `meta/monitoring-session-handover.md` |
| analysis | analysis, 분석 | opus(측정은 sonnet) | 새 ckpt 측정·보고(판정 안 함) | `meta/analysis-session-protocol.md` |
| notion | Notion | sonnet(서술 요약 opus) | 노션 논문 페이지 절 단위 동기화 | 페이지 `Drone Object Detection for RGB-IR Fusion`(`33d05310-a165-408a-b0b8-ec4427d1fe2c`), 빌더 `.claude/skills/notion-experiment-log/paper_page_builder.py` |
| specialist-baseline | dgfusion deliver training | opus | 기준선(DGFusion) 재학습·실패 분석 보강 | unknown |

## 기존 구조 대응표
| LAB STANDARD 파일 | 이 레포의 대응 |
|---|---|
| status/current.md (PI 블록) | `status/current.md` ①~④ 블록(헤드라인 표는 `experiments/headline.yaml` 생성물). PI 블록 = frontmatter 바로 아래(2026-09-26 추가, 값은 ①·③·④에서 옮김) |
| experiments/plan.md | 대기열·판정 = lab-plan(2026-09-26 이관), 실행 중·GPU 현황·사고 기록 = plan.md(사람) |
| experiments/protocol.md | `experiments/protocol.md` |
| experiments/registry.md | `experiments/registry.md`(사람이 관리) + 파일 끝 `lab:registry` 생성 블록(2026-09-26 추가, 파이프라인 `pipeline/results.jsonl`) |
| experiments/judgment-ledger.md | `experiments/judgment-ledger.md` |
| research/hypothesis-ledger.md | 없음(카드·판정 문서가 대신함) |
| 헤드라인 수치 | `experiments/headline.yaml`(정본) |
| 학습 종료→자동 평가 | `pipeline.toml`(DELIVER legal v2 자동, MUSES 알림만) |
