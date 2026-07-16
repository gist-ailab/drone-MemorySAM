---
created: 2026-07-08
---

# 📐 리포 컨벤션 (2026-07-08 재구조화 이후 — 모든 세션 필독·준수)

> **이 문서가 구조 유지의 단일 출처다.** 2026-07-08 재구조화(브랜치 `restructure/ia-taxonomy`)로 확립된 형식을 모든 세션이 유지한다.
> 새 파일을 만들거나 코드를 추가하기 전에 이 규칙에 맞는 위치·이름인지 확인할 것. 규칙 변경은 사용자 승인 후 이 문서를 갱신.

---

## 1. Git 규칙

- **모든 브랜치는 `develop` 기준으로 분기한다** (main 아님, 다른 feature 브랜치 아님). worktree 생성 시: `git worktree add <path> -b <branch> develop`.
- 브랜치 이름: `<유형>/<주제>` (예: `feat/p33-seg`, `fix/issue-023`, `restructure/...`, `jobs/...`).
- **병합은 develop으로 직접**: PR 없이 `git push origin HEAD:develop` (사용자 선호). `main`은 건드리지 않는다.
- 병합 후 **로컬 허브 체크아웃**(이 리포의 메인 체크아웃 — 원격 서버들이 여기서 pull)도 pull로 최신화. 미커밋분은 wip 커밋으로 보존 후 pull.
- **원격 서버(B200/jarvis/bengio 등)는 진행 중 학습이 있는 동안 pull 금지.** 학습 종료 후 pull → 스크립트 무결성 확인(taskboard R3).
- 시크릿(.wandb_key 등) 커밋 금지 — .gitignore에 이미 등록됨.

## 2. 문서 규칙 (`.claude_logs/`)

**폴더 택소노미 (여기에만 새 문서 생성):**

| 폴더 | 무엇을 넣나 |
|---|---|
| `status/` | `current.md`(현재 상태 스냅샷 — 덮어쓰기만) + `history-<YYYY>H<1\|2>.md`(진행 이력 — 최상단 append) |
| `models/` | 모델 버전 아키텍처 문서(`arch-evolution.md`), 피규어, `explain/pNN-*.md` |
| `experiments/` | `registry.md`(실험 허브표 — **새 실험 launch 시 행 추가**), `log.md`, `monitor-log.md`, `analysis/YYYY-MM-DD-*.md` |
| `det/` `datasets/` | det 트랙 진단·계획 / 데이터셋 구축 노트 |
| `research/` | 노벨티·관련연구. `vault/`는 **손편집 금지**(아래 §5) |
| `decisions/` | 설계 제안·ADR — `YYYY-MM-DD-<제목>.md` |
| `infra/` `issues/` | 서버·환경 / 이슈(상단 인덱스 표 + 본문 양쪽 갱신) |
| `meta/` | 봇 역할, 태스크보드, 이 컨벤션 |
| `archive/` | 동결 문서 (헤더에 🗄 ARCHIVED 명시 후 이동) |

- **명명**: kebab-case 영문. 날짜성 문서는 `YYYY-MM-DD-` 접두, 모델 문서는 `pNN-` 접두. **번호 프리픽스 신규 부여 금지** (구번호는 frontmatter `legacy_id`로만 유지).
- **새 문서 추가 시**: 해당 폴더 `00_MOC.md`에 1행 등록 + 필요하면 `00_INDEX.md` 갱신. 문서 간 링크는 상대 md-link.
- **append-only 로그 롤오버**: `monitor-log.md`·history가 100KB를 넘으면 반기/월 단위 파일로 분할하고 MOC에 등록.
- 구번호 참조("doc 19" 등)를 만나면 `00_INDEX.md`의 매핑표로 해석.

## 3. 코드 규칙

- **새 모델 버전(P33, P34, …)**: `semseg/models/sam2/sam2/lora_sam/pNN.py`에 클래스 작성 + `lora_sam/__init__.py`의 `MODEL_REGISTRY`에 등록. **메가파일 부활 금지** — `sam_lora_image_encoder_seg.py`는 shim이므로 여기에 클래스를 추가하지 말 것.
- **공통 모듈**(MoE/LoRA adapter, fusion head, reliability/confidence)은 `semseg/models/sam2/sam2/modules/{moe,fusion,reliability,common}.py`에. sam3 쪽에 같은 개념을 재구현하지 말고 modules를 import.
- **신규 코드는 shim 경유 import 금지** — `lora_sam`/`modules`에서 직접 import. (`sam_lora_image_encoder_seg.py`, `sam_lola_utils.py`, `*_bkup.py` shim은 하위호환용.)
- 모델 클래스 선택은 `get_model(name)` (eval() 금지).
- 일회성 분석 스크립트는 `tools/`(재사용 가치 있음) 또는 `_archive/oneoff/`(완료된 일회성)에. 리포 루트에 새 스크립트를 늘리지 말 것.
- 폐기 판정된 버전 클래스는 `lora_sam/legacy.py`로 이동(삭제 금지 — config 재현성).

## 4. 실험/Configs 규칙

- **신규 config 명명**: `<dataset>_<modal>_<version>_<aug>.yaml` — **서버명 접두어 금지**. 서버별 경로 차이는 `configs/profiles/<server>.yaml`(참조 문서)와 `scripts/servers.conf`로.
- 위치: 학습 config는 `configs/<dataset>/`, 평가 config는 `configs/eval/` (학습 config와 같은 이름 + `MODEL_PATH`). 데드 실험 config는 `configs/archive/`로 이동.
- **새 실험 launch 시**: `experiments/registry.md`에 행 추가 (config 경로·서버·상태), 종료 시 상태·수치 갱신 + `experiments/log.md` 상세 기록.
- 기존 config 파일명은 변경 금지 (output 디렉토리 매핑 보존).
- 실험 결과·시각화 산출물은 `/mnt/HDD2/src/logs/<model>_eval_<date>/` (쓰기 전 touch 테스트 — ISSUE-023 참조).

## 5. Obsidian 볼트 규칙

- **원본 = NAS `/nas_jm/Research/26_MultimodalSeg/`** (canonical). repo의 `research/vault/`는 사본 — **손편집 금지**, 갱신은 `bash scripts/sync_research_vault.sh`로만.
- 볼트 원본 편집 시: 번호 충돌 금지, 변경은 `VAULT_CHANGELOG_*.md`에 기록, bare 개념링크는 대상 노트 frontmatter `aliases`로 해소.

## 6. 세션 시작/종료

- 시작: `CLAUDE.md` → `.claude_logs/00_INDEX.md` → `status/current.md` → 작업 폴더 `00_MOC.md` (CLAUDE.md 지침 준수).
- 의미 있는 작업 완료 시: `status/current.md` 덮어쓰기 + `status/history-<현재반기>.md` 최상단 append (CLAUDE.md §3 자동 업데이트 규칙).

## 7. 🔴 모델 위임 규칙 (user 지정 2026-07-16)

**이 리포에서 작업하는 모든 세션·서브에이전트 공통.** 상세는 `CLAUDE.md` §1.6.

- **sonnet**: 학습 기동(`remote_exp.sh run`/torchrun) · **tmux 제어** · 상태 조회(`nvidia-smi`/`ps`/로그) · kill · rsync 회수 · **git**(pull/push/fetch/commit) · 기계적 파일 정리
- **해당 세션의 opus 또는 fable**: **코드를 만지는 일**(패치·config 설계·스크립트) · **에러 검증/진단** · 로그 판독 · 수치 판정
- 위임은 `Agent` tool `model: "sonnet"`. **sonnet은 집행, 판정은 상위 모델.**
- 위임 금지: cherry-pick 선별 · 충돌 해결 · 브랜치 전략 · 실패 원인 규명 · 실험 판정이 담긴 커밋 메시지
