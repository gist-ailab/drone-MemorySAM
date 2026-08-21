# 학습 워치독 + 서버 launch 정책 (2026-08-08 도입)

> **목적**: 원격 학습 run의 launch 이후 감시(기동검증·주기점검·사망/완주 감지)를 LLM 세션 폴링에서 스크립트로 이전.
> 세션은 launch 판단만 하고, 감시는 cron이 한다. "지금 어때?"는 `status` 한 번으로 답한다.
> 배경 결정: 세션 구조를 "역할"이 아닌 "컨텍스트 수명"으로 재편 — 운영(C)은 세션이 아니라 스크립트.

## 구성요소

| 파일 | 역할 |
|---|---|
| `scripts/servers.conf` 6번째 `policy` 필드 | launch 정책. `ban:1,2`(GPU 금지) / `off`(서버 금지) / 없음·`-`(무제한). **GPU 예약은 메모리가 아니라 여기 기록** (예: jarvis `ban:0`) |
| `scripts/remote_exp.sh` | `run` 시 정책 강제(off 서버 거부, auto-pick banned 제외, 명시적 지정도 banned 겹치면 거부) + launch 성공 시 허브에 `.watchdog/runs/<run_id>/manifest.json` 자동 등록 (로그 파일명 타임스탬프는 허브에서 생성 → 결정론적) |
| `scripts/watchdog.sh` | cron 5분 주기 scan. 상태 머신 + 알림. `bash scripts/watchdog.sh <subcmd>` |
| `tests/test_watchdog.sh` | 오프라인 테스트 119개 (ssh는 PATH stub, 실서버 접근 0) |

## watchdog.sh 서브커맨드

```bash
bash scripts/watchdog.sh status          # 활성 run 표 (세션이 "지금 어때?"에 답할 때 이거 하나)
bash scripts/watchdog.sh scan            # 1회 점검 (cron이 5분마다 호출)
bash scripts/watchdog.sh register <server> <config> <gpus_csv> <원격log경로> [--repo <path>]
                                         # remote_exp.sh가 안 통하는 서버(B200류) 수동 등록
bash scripts/watchdog.sh close <run_id> [사유]   # 수동 종결
bash scripts/watchdog.sh install-cron    # */5 cron 등록 (flock 중복방지, 태그로 멱등) / uninstall-cron
```

## 상태 머신 (CLAUDE.md §1.6 기동검증 기준의 기계화)

- `launching → running`: 스캔 간 iter 전진 확인 (관측 2회 — 07-16 NCCL 데드락 오보 사고의 교훈).
- `launching → failed_startup` 🔴: 15분(`WATCHDOG_STARTUP_MIN`) 내 전진 없음 / Traceback / **RANDOM INIT**(hpca100 백본 미로드 함정 — iter가 돌아도 즉시 사형).
- `running → stalled` 🔴: 30분(`WATCHDOG_STALL_MIN`) iter 불변 **AND** util ≤20%(`WATCHDOG_STALL_UTIL`) — eval 구간 오탐 방지 이중 조건. 회복 시 🟢 recovered.
- `running → completed` 🟢: 프로세스 종료 + 종료 배너(`Total Training Time`/`Training complete`) 또는 최종 epoch·iter 도달(sam3 트레이너는 배너 없음). **알림에 "슬롯 비었음 — experiments/plan.md 대기열 확인" 포함 (GPU-never-idle 규칙 자동화).**
- `running → dead` 🔴: 프로세스 종료 + 배너 없음. 마지막 로그 50줄을 run 디렉토리에 보존.
- ssh 실패 = `unreachable`(비종결), **2연속 실패 시에만** 알림 (hpca100 MTU 플래핑 오탐 방지).

트레이너별 iter/epoch 로그 포맷은 코드에서 실측해 내장 (sam2/reliadino/det/sam3 4종). override: `WATCHDOG_ITER_REGEX`.

## 알림

`.watchdog/alerts.log` append + `notify-send`(cron 환경에서는 gnome-shell environ에서 DISPLAY/DBUS 추출 — OPERATING.md 규약). 훅: `WATCHDOG_ONALERT_CMD "<run_id> <state> <manifest>"` (기본 비활성 — 추후 headless 진단 연결용).

## 운영 규칙

- **GPU 예약/서버 제한 지시를 받으면 servers.conf policy 필드를 고친다** (세션 메모리에만 두지 말 것 — 어느 세션이 launch하든 기계 강제되는 곳은 여기뿐).
- `.watchdog/`는 머신 로컬 상태라 gitignore됨. cron은 허브 체크아웃(`/mnt/HDD1/.../drone-MemorySAM`) 경로로 설치 — worktree에서 설치 금지(삭제 시 cron 파손).
- 활성 학습 감시 중 스크립트 수정 시 `bash tests/test_watchdog.sh` 통과 후 반영.
