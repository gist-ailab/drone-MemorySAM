---
created: 2026-09-08
author: 이 세션(worktree `.claude/worktrees/p30-det`, background job — EnterWorktree 아님, 작업폴더가 잡 생성 시 고정된 세션이라 ExitWorktree로 못 나감)
status: ✅ 인계 완료 (2026-09-08) — 감시 주체는 `MMSAM | learning status monitoring` 세션 하나다.
revised: 2026-09-08 — §1 감시 명세를 개정판으로 교체(진행 판정 방식 변경, 정체 감지 추가, PING 가드), §1-5 크론 생성 반영
---

# 학습 감시 인계 문서 (2026-09-08)

> 새 세션은 이 대화 맥락을 물려받지 못한다. `CLAUDE.md` 진입 순서대로 `.claude_logs`를 읽고 시작하므로,
> **이 문서가 유일한 연결 고리**다. 아래 수치는 전부 2026-09-08 02:00~02:05 UTC(hpca100 기준, +9h=KST 11:00~11:05)
> 사이에 ssh로 직접 실측한 값이다 — 기억이나 추정이 아니다.

## 0. 왜 이관하는가

이 세션의 작업폴더가 `.claude/worktrees/p30-det`로 잡혀 있는데, 지금 진행 중인 작업(P52 캠페인)의
성격상 리포 루트 세션이 맞다고 user가 지적했다. `ExitWorktree(action:"keep")`을 시도했으나
**no-op**이었다 — 이 세션은 EnterWorktree로 들어온 게 아니라 잡 생성 시점에 작업폴더가 고정된
background 세션이라 이 도구의 적용 대상이 아니다(`pwd` 재확인으로도 확인됨). 대화 중에는
작업폴더를 바꿀 방법이 없어서, 새 세션을 리포 루트에서 만들어 역할을 넘기는 방식으로 이관한다.

원래는 "지금 도는 런들이 다 끝난 뒤(감시 재설치 공백 최소화)"로 미뤘으나, 활성 감시가
8건이 아니라 실제로는 **2건뿐**임을 확인(§3)한 뒤 지금 바로 이관하기로 했다 — 재설치할 게
둘뿐이면 새 세션이 그 둘을 켜고 정상 동작을 확인한 다음에 이 세션을 끄면 감시 공백이 0이다.

## 1. 활성 감시 4건 — 재설치 명세

> 🔴 **2026-09-08 개정판.** 초판 명세는 `tail -c 2000~3000`으로 잘라낸 조각에서 `[Val]` 줄을 찾았는데,
> tqdm 진행 줄이 초당 여러 번 쌓여 그 창을 가득 채우기 때문에 평상시에는 `[Val]` 줄이 밀려나 진행
> 이벤트가 거의 출력되지 않았다(재설치 직후 네 건 모두 출력 0바이트였던 이유가 이것이다). 창을 키우는
> 것으로는 부족하다 — 학습이 빨라지면 다시 밀린다. 그래서 **진행 판정과 에러 판정을 분리**했다.
> 아울러 초판에는 감지 사각지대와 오판 경로가 각각 하나씩 있어 함께 고쳤다(아래 ③④). **재설치할 때는
> 반드시 이 개정판을 쓰고, 옛 방식으로 되돌리지 마라.**

### 1-0. 네 감시가 공유하는 판정 로직

원격 조회는 대상 하나당 주기마다 `ssh` **한 번**만 쓴다. 그 한 번으로 아래 다섯 신호를 모두 받아 온다.

| 신호 | 원격에서 얻는 방법 | 쓰임 |
|---|---|---|
| `PING` | 무조건 `echo PING` | 원격 조회 성공 여부 |
| `ALIVE` | `tmux has-session -t <세션> 2>/dev/null && echo ALIVE` | 세션 생존 |
| `AGE` | `$(date +%s) - $(stat -c %Y <로그>)` | 로그 무갱신 경과 초 |
| `VAL` | `grep -a '\[Val\]' <로그> \| tail -1` | 최신 평가 결과 |
| `ERR` | `tail -c 4000 <로그> \| tr '\r' '\n' \| grep -aE '<에러패턴>' \| tail -2` | 크래시 흔적 |

① **진행 판정은 로그 전체를 훑는다.** `tail`로 자른 조각이 아니라 `grep -a '\[Val\]' <로그> | tail -1`로
마지막 한 줄만 뽑는다. tqdm이 아무리 쌓여도 밀리지 않고, 전송량은 한 줄뿐이라 초판보다 오히려 가볍다.
`-a` 옵션은 로그에 섞인 제어문자 때문에 grep이 파일을 바이너리로 판단하는 것을 막는다.

② **에러 판정은 초판 그대로 꼬리 4KB에서 찾는다.** 크래시가 나면 Traceback이 로그 맨 끝에 오므로
이 방식이 맞다. 패턴은 `Traceback|CUDA out of memory|Killed|RuntimeError`.

③ **정체 판정을 두 층으로 새로 넣었다.** 프로세스는 살아 있는데 진전이 없는 경우가 초판의 사각지대였다.
`tmux has-session`은 `ALIVE`를 반환하고, 로그 끝에 Traceback도 없으므로 어느 판정에도 걸리지 않는다.
이 프로젝트에는 실제 사고 기록이 있다 — `CLAUDE.md` §1.6의 2026-07-16 NCCL 데드락 건으로, 기동됐다는
사실만 보고 살아났다고 보고했으나 실제로는 `0/187`에서 13분간 정지해 있었다.

- **`STALLED_LOG` (1차, 민감)**: `AGE >= 900`(15분)이면 알린다. tqdm이 초당 여러 번 쓰고 평가 구간에서도
  `eval: NN%|...` 줄이 계속 쌓이므로(2026-09-08 실측으로 확인), 15분 무갱신은 정상 학습에서 나오지 않는다.
  네 감시 모두 900초로 같다. 이것이 실질적인 1차 방어선이다.
- **`STALLED_VAL` (2차, 보수적)**: `[Val]` 줄이 `CYCLES` 주기 연속으로 그대로면 알린다. 임계는 런마다
  다르다 — `[Val]` 갱신 간격은 `EVAL_INTERVAL × epoch당 소요 시간`이고, 평가 구간에서 epoch이 잠시
  멈추므로 그 간격의 **2.5배 이상**이 되도록 잡았다(§1-1 표의 `CYCLES` 열, 산출 근거 포함).

④ **`PING` 가드로 원격 조회 실패와 세션 종료를 구분한다.** 초판은 `ssh`가 일시적으로 실패해 출력이 비면
그것을 `SESSION_ENDED`(=완주)로 오판했고, 단일 대상 감시는 거기서 `break`로 **감시 자체를 끝내 버렸다**.
개정판은 응답에 `PING`이 없으면 조회 실패로 보고 아무 판정도 하지 않으며, 3주기 연속 실패했을 때만
`SSH_FAIL`을 알리고 감시는 계속 유지한다. `SESSION_ENDED`는 `PING`이 있고 `ALIVE`가 없을 때만 낸다.

⑤ **알림 폭주를 막는다.** `PROGRESS`는 `[Val]` 값이 **바뀐 주기에만** 출력한다(평가가 한 번 끝날 때마다
정확히 한 번). `STALLED_LOG`·`STALLED_VAL`은 플래그를 세워 한 번만 알리고, 진전이 재개되면 플래그를
푼다. 첫 주기에는 `WATCH_START`를 한 번 내보내 **설치 직후 정상 동작을 즉시 확인할 수 있게 한다**
(초판은 설치 후 몇 시간 동안 아무 출력이 없어 살아 있는지 확인할 방법이 없었다).

### 1-1. 감시 넷의 대상과 임계

`persistent: true`로 걸고, `SESSION_ENDED`가 뜨면 단일 대상 감시는 루프를 끝낸다(1-a는 두 런이 모두
끝났을 때 끝낸다).

| 항목 | 서버 | tmux 세션 | 폴링 | `CYCLES` (산출 근거) |
|---|---|---|---|---|
| 1-a | `yeon` | `p52_deliver_s1`, `p52_deliver_s2` (각각 독립, window `main`) | 1800초 | 7 (=3.5시간. `[Val]` 갱신 간격 약 80분 = EVAL_INTERVAL 2 × 40분/ep 의 2.6배) |
| 1-b | `yeon` | `elora_a_r16` | 1500초 | 8 (=3.3시간. 갱신 간격 약 81분 = 2 × 40.4분/ep 의 2.5배) |
| 1-c | `hpca100` | `hpca100_E2` | 1500초 | 14 (=5.8시간. 갱신 간격 약 139분 = EVAL_INTERVAL 5 × 27.8분/ep 의 2.5배) |
| 1-d | `hpca100` | `hpca100_E7c` | 1500초 | 6 (=2.5시간. 갱신 간격 약 58분 = 5 × 11.6분/ep 의 2.6배) |

감시 로그 절대경로:

- 1-a: `/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs/p52_deliver_s1_launch.log` 및 `..._s2_launch.log`
- 1-b: `/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs/elora_a_r16_launch.log`
- 1-c: `/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM/logs/hpca100_E2_launch.log`
- 1-d: `/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM/logs/hpca100_E7c_launch.log`

완주 시 대응: **1-c 완주 시 hpca100 GPU1,3이 비고 1-d 완주 시 GPU2가 빈다. `gpu-never-idle` 원칙에 따라
즉시 다음 배치를 정하라.**

### 1-2. 재설치 스크립트 (단일 대상판 — 1-b·1-c·1-d 공통)

`SRV`·`S`·`log`·`CYCLES`·`POLL`만 위 표대로 바꿔서 쓴다.

```bash
SRV=yeon; S=elora_a_r16
log=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs/elora_a_r16_launch.log
STALE=900; CYCLES=8; POLL=1500
PREV=""; CYC=0; FS=0; FV=0; FAIL=0
while true; do
  out=$(ssh -o ConnectTimeout=10 $SRV "
echo PING
tmux has-session -t $S 2>/dev/null && echo ALIVE
echo \"AGE:\$(( \$(date +%s) - \$(stat -c %Y $log 2>/dev/null || echo 0) ))\"
echo \"VAL:\$(grep -a '\[Val\]' $log 2>/dev/null | tail -1 | tr -d '\r')\"
echo ERRSTART
tail -c 4000 $log 2>/dev/null | tr '\r' '\n' | grep -aE 'Traceback|CUDA out of memory|Killed|RuntimeError' | tail -2
" 2>/dev/null)
  if [ "$(echo "$out" | grep -c '^PING$')" = "0" ]; then
    FAIL=$(( FAIL + 1 ))
    [ "$FAIL" = "3" ] && echo "[$S] SSH_FAIL — 원격 조회가 3주기 연속 실패했습니다. 세션 종료로 단정하지 않고 감시를 유지합니다."
    sleep $POLL; continue
  fi
  FAIL=0
  alive=$(echo "$out" | grep -c '^ALIVE$')
  age=$(echo "$out" | sed -n 's/^AGE://p' | head -1)
  case "$age" in ''|*[!0-9]*) age=0 ;; esac
  val=$(echo "$out" | sed -n 's/^VAL://p' | head -1)
  err=$(echo "$out" | sed -n '/^ERRSTART$/,$p' | tail -n +2)
  if [ "$alive" = "0" ]; then
    echo "[$S] SESSION_ENDED — tmux 세션이 사라졌습니다(완주 또는 사망). last=${val:-unknown}"; break
  fi
  if [ -n "$err" ]; then
    echo "[$S] ERROR_DETECTED (세션은 아직 살아 있음) last=${val:-unknown}: $err"
  elif [ "$age" -ge "$STALE" ]; then
    if [ "$FS" = "0" ]; then
      echo "[$S] STALLED_LOG — 로그가 ${age}초 동안 갱신되지 않았습니다(임계 ${STALE}초). tmux 세션은 살아 있으므로 데드락이나 정지를 의심하십시오. last=${val:-unknown}"; FS=1
    fi
  else
    FS=0
    if [ -z "$PREV" ]; then
      echo "[$S] WATCH_START — 감시를 시작했습니다. last=${val:-없음(첫 평가 전)}"; PREV="$val"; CYC=0; FV=0
    elif [ "$val" = "$PREV" ]; then
      CYC=$(( CYC + 1 ))
      if [ "$CYC" -ge "$CYCLES" ] && [ "$FV" = "0" ]; then
        echo "[$S] STALLED_VAL — [Val] 값이 ${CYC}주기(약 $(( CYC * POLL / 60 ))분) 동안 그대로입니다. 로그는 갱신되고 있으니 학습 지연이나 평가 구간 장기화를 확인하십시오. last=${val:-unknown}"; FV=1
      fi
    else
      echo "[$S] PROGRESS: $val"; PREV="$val"; CYC=0; FV=0
    fi
  fi
  sleep $POLL
done
```

### 1-3. 재설치 스크립트 (다중 대상판 — 1-a 전용)

1-a는 한 감시가 두 런(`p52_deliver_s1`·`p52_deliver_s2`)을 함께 본다. 위 스크립트의 상태 변수를
연관배열로 바꾸고, 대상별로 종료 여부(`DONE`)를 따로 들고 있다가 **둘 다 끝났을 때만** 루프를 끝낸다.
한쪽만 끝났으면 그 대상은 건너뛰고 다른 쪽 감시를 계속한다.

```bash
SRV=yeon
LOGDIR=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs
STALE=900; CYCLES=7; POLL=1800
TARGETS="p52_deliver_s1 p52_deliver_s2"
declare -A PREV CYC FS FV FAIL DONE
for s in $TARGETS; do PREV[$s]=""; CYC[$s]=0; FS[$s]=0; FV[$s]=0; FAIL[$s]=0; DONE[$s]=0; done
while true; do
  for s in $TARGETS; do
    [ "${DONE[$s]}" = "1" ] && continue
    log="$LOGDIR/${s}_launch.log"
    # ... 원격 조회와 판정은 1-2 와 동일, 상태 변수만 ${PREV[$s]} 형태로 바꾼다 ...
    # SESSION_ENDED 시 break 대신 DONE[$s]=1; continue
  done
  allover=1
  for s in $TARGETS; do [ "${DONE[$s]}" = "0" ] && allover=0; done
  [ "$allover" = "1" ] && { echo "P52 seed1·seed2 두 런 모두 종료되어 감시를 마칩니다."; break; }
  sleep $POLL
done
```

### 1-4. 재설치할 때 지킬 것

- **먼저 기존 감시를 `TaskStop`으로 정리하고 새로 걸어라.** 정리하지 않고 겹쳐 걸면 같은 대상에 감시가
  여러 개 쌓인다. 2026-09-08에 좀비 감시 6건(§3)이 나온 원인이 바로 이 누적이었다.
- **설치 직후 `WATCH_START` 네 줄이 오는지 확인하라.** 오지 않으면 `ssh` 연결이나 로그 경로를 의심한다.
- 이 감시들은 세션에 종속된다. **세션이 중단·재기동되면 감시와 크론이 함께 사라지므로 둘 다 다시 만들어야
  한다**(§1-5의 크론 포함).
- 🔴 **표시용 이름을 바꾸려고 세션을 재기동하지 마라.** 2026-09-08 인계 때 세션 이름이 임시값
  (`MMSAM | monitoring (인계중)`)으로 남았는데, `claude` CLI 에는 이름을 바꾸는 명령이 없고 `respawn` 은
  이름 옵션을 받지 않은 채 재시작만 한다. 즉 이름을 고치려는 재기동은 **방금 건 감시 넷과 크론을 전부
  날리고 아무것도 바꾸지 못한다.** 이름은 기능에 영향이 없으므로 그대로 두고, 잡 상태 파일을 손으로
  고치는 비공식 조작도 하지 마라(데몬이 도는 중이라 잡 관리가 꼬이면 복구가 번거롭다).

### 1-5. 세션 내부 정기 실행 — 3시간 주기 정기 진행보고 크론 (2026-09-08 생성 완료)

✅ **2026-09-08 생성 완료** — 인계받은 세션이 `CronCreate`로 만들었다(잡 `c0aba5c9`, `23 */3 * * *`). 사용자가 3시간 주기로 다시 원한다고 "mmsam session merge" 경유로 확인한 데 따른 것이다. 이 밖에 `/loop` 반복 같은 정기 실행 장치는 없다. "MMSAM | 생각정리" 세션은 bengio 일일 카드 런을 보는 Monitor `b23rrk2ii` 하나만 갖고 있으며, 그것은 인계 대상이 아니다.

아래가 그 크론의 사양이며, 재생성할 때도 이대로 만든다.

- 주기: **3시간마다**, cron 표현식 `23 */3 * * *` 권장(정각 실행 회피, 00:23·03:23·06:23... 식으로 분산).
- 보고 형식: auto-memory `progress-report-format` + `CLAUDE.md` §1.6 말미 규정 그대로 — ①서버별 현황 표(실험별 데이터셋 열·SOTA 대비 델타·내부최고 대비 델타·ETA) ②남은 런의 서버별 배치 계획, 두 블록 필수 + 벤치 baseline 표(SOTA/우리최고/격차) 포함.
- 보고 범위: **bengio도 포함해야 한다**(일일 카드 E1·E3·E4, B0 재채점이 그쪽에서 돎 — 감시는 "MMSAM | 생각정리"의 `b23rrk2ii`가 맡지만, 정기보고의 서버별 표는 전체를 포괄해야 하므로 새 세션이 bengio도 직접 조회할 것).
- 크론 본문에는 조회를 sonnet에 위임하라는 지시(`CLAUDE.md` §1.6)와, 보고를 마친 뒤 감시 넷과 이 크론의 생존을 스스로 점검해 유실된 것을 즉시 재설치하라는 자가 점검 절차를 함께 담았다.
- ⚠️ **세션 안에서 만든 크론은 세션이 재기동되면 함께 사라지고, 그러지 않더라도 7일 뒤 자동 만료된다.** 과거 정기보고 크론이 유실된 원인으로 추정되는 지점이다 — 세션이 중단·재기동될 때마다, 위 1-a~1-d 감시를 다시 거는 것과 마찬가지로 **이 크론도 다시 만들어야 한다.**

## 2. 현재 활성 런 현황 (2026-09-08 02:00~02:05 UTC 실측)

| 서버 | 실험 | 데이터셋 | 진행 | 최근 val | 내부최고 델타 | ETA |
|---|---|---|---|---|---|---|
| yeon GPU2,3 (tmux `p52_deliver_s1`) | P52 RxDINO DELIVER seed20260901 | DELIVER 4모달 | ep32/200 | 65.51 (mIoU) | best 65.51@ep32(신기록 갱신 직후), 게이트 G1 기준 54.65(Phase1 legal test 54.95−0.3) — **주의: 이 val은 legal test 아님, 최종 판정은 완주 후 val.py 공식 재채점** | ~4.7일 (39.9분/ep 실측, 09-13 새벽경) |
| yeon GPU4,5 (tmux `p52_deliver_s2`) | P52 RxDINO DELIVER seed20260902 | DELIVER 4모달 | ep34/200 | 65.61 | best 66.19@ep28 | ~4.5일 (39.1분/ep, 09-12 저녁경) |
| yeon GPU6,7 (tmux `elora_a_r16`) | E-LoRA arm A 재런(per-modal r16) | DELIVER 4모달 | ep34/200 | 65.88 | best 66.64@ep26 — 이 값을 B/C(각 r16 기준 shared/shared+resid)와 비교할 새 arm A 기준치 | ~4.66일 (40.4분/ep, 09-13 오전경) |
| hpca100 GPU1,3 (tmux `hpca100_E2`) | 일일카드 E2(LoRA 전 선형층 QKVO+MLP, r32) | DELIVER 4모달 | ep25/40 | 65.25 | B0(현행 레시피, 변경 0) 대비 — ep20 시점 62.40으로 −1.5(음성 방향)였으나 ep25에서 65.25로 반등, **정본 비교용으로 완주까지 중단 안 함**(discussion 세션 지시) | ~6.9시간 (27.8분/ep, 09-08 08:20 UTC≈17:20 KST) |
| hpca100 GPU2 (tmux `hpca100_E7c`) | 일일카드 E7c(MUSES PhysAug-on 대조군, PYTHONUNBUFFERED=1) | MUSES 3모달(img/lidar/event) | ep4/40 (70%) | 아직 첫 eval 전(EVAL_INTERVAL 5) | E7(PhysAug-off, 완료, 트레이너 val 80.29@ep40 — **legal test 아님**)과 페어 비교 예정 | ~6.8시간 (11.6분/ep, 09-08 08:50 UTC≈17:50 KST) |

**갱신 실측 (2026-09-08 13:30 KST = 04:30 UTC, 인계받은 세션이 ssh로 직접 측정).** 위 표는 02:00 UTC 시점 값이므로 약 11시간 낡았다. 아래가 최신이며, 전부 트레이너가 찍는 val 이라 legal test 가 아니다.

| 런 | 진행 | 최신 `[Val]` | best |
|---|---|---|---|
| `p52_deliver_s1` | ep38 진행 중 | 66.25@ep36 | 66.25@ep36 (내부 신기록 갱신) |
| `p52_deliver_s2` | ep38 | 65.23 | 66.19@ep28 |
| `elora_a_r16` | ep38 | 65.55 | 66.64@ep26 |
| `hpca100_E2` | ep30/40 | 66.91@ep30 | 66.91@ep30 |
| `hpca100_E7c` | ep13/40 | 75.52@ep10 | 75.52@ep10 |

E2 는 표의 ep25 시점 65.25 에서 ep30 66.91 로 계속 올라가고 있어, ep20 의 −1.5 하락은 일시적 흔들림이었던 것으로 보인다. 다만 채택·폐기 판정은 완주 후 공식 재채점으로만 한다.

체크포인트 경로(전부 `.../outputs/ReliaDINO/<SAVE_DIR명>/`):
- `yeon:/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/outputs/ReliaDINO/yeon_deliver_rgbdel_P52_seed20260901/`
- `yeon:/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/outputs/ReliaDINO/yeon_deliver_rgbdel_P52_seed20260902/`
- `yeon:/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/outputs/ReliaDINO/yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260821_elora_permodal_r16/`
- `hpca100:/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM/outputs/ReliaDINO/hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E2/`
- `hpca100:/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM/outputs/ReliaDINO/hpca100_muses_rgbel_P39_1_seed2_physaugon_screen40_E7c/`

벤치마크 SOTA 격차(고정 기준표, 위 런들은 전부 완주 전이라 아직 적용 안 됨):

| 벤치 | 우리 최고 | SOTA | 격차 |
|---|---|---|---|
| DELIVER val | 67.74 (P34/P36) | 68.79 (CAFuser-CAA) | −1.05 |
| DELIVER test | 56.62 (P34/P36) | 56.71 (DGFusion) | −0.09 |
| MUSES test(공식) | 79.025 (P38-m2f) | 82.39 (GtA, 카메라단독) | −3.365 |

**참고**: MCubeS P52 seed20260901(yeon GPU6,7 자리, 이전에 elora_a_r16이 그 자리를 이어받음)은 2026-09-08 04:41 KST 완주(val 57.96, best 58.18@ep174) — legal eval은 discussion 세션의 P52.1 재설계 정리 후 진행 예정, 아직 안 함.

## 3. 오늘 정리한 좀비 감시 6건 (데몬 기록엔 남아 있었으나 대상이 이미 종료됨 — 재설치 불필요)

| task_id | 감시 대상 | 정리 근거(실측) |
|---|---|---|
| `bwkl490kz` | hpca100 P50-EXT 채택게이트 파인튠(`p50ext_gate`) | 이 세션이 2026-09-07 GPU1,3을 E2에 내주려고 직접 종료시킴 |
| `buf26dpv4` | P47-2 공식 MUSES 평가(yeon) | 2026-09-03 완료(mIoU 81.72 산출, G2=81.42 확정에 사용됨) |
| `bhdope8s2` | N7 크래시 감시(yeon) | N7이 2026-09-06 200/200 정상 완주 |
| `bmov8vpd5` | hpca100 워치독 릴레이(`watchdog.log`) | **서버 전체 안전망 아님** — `watchdog.sh`가 `p50ext_deliver_main/run3.log` 딱 하나만 보고 있었음(직접 확인). 그 대상(P50-EXT Phase2 사전학습)도 2026-09-05 완주. 정리해도 안전망 구멍 없음 |
| `bln2vbvyu` | P47-2 복구런 크래시 감시(yeon) | 2026-09-03 ep300/300 완주 |
| `b1ybcf2cx` | hpca100 백업 진행 감시 | 백업 전체 완료 + 회수분 20개 디렉토리 삭제까지 이미 끝남(2026-09-02~03) |

6건 전부 `TaskStop` 호출 시 성공 응답을 받음 — 즉 단순 stale 메타데이터가 아니라 실제로 살아있는 프로세스였고(대상이 끝났는데도 안 죽고 도는 좀비), 전부 목적을 다한 뒤였다. **"감시가 죽은 줄 모르고 지나가는" 위험한 케이스는 없었다.**

## 4. 인계받는 쪽이 알아야 할 함정

- **hpca100 origin = GitHub(`gist-ailab/drone-MemorySAM`), 로컬 허브(`local` remote) 없음.** `CLAUDE.md` §1.7이 전제하는 "서버는 로컬 허브를 pull"이 hpca100엔 적용 안 됨 — develop 갱신 후 hpca100은 `git fetch origin develop && git merge origin/develop --ff-only`로 직접 동기화해야 한다(이 세션이 매번 이렇게 했음). yeon도 마찬가지로 origin 직접 사용 확인됨.
- **jarvis가 `2f14dae`로 크게 뒤처져 있다**(다른 세션 확인 사항, 전달만 함) — P52 관련 config가 아예 없다. jarvis에서 뭔가 돌리려면 먼저 동기화 필요.
- **bengio·lecun도 오랫동안 뒤처져 있었다**(이 세션이 2026-09-05경 직접 확인: bengio 436커밋, lecun 190커밋 차이 + 각각 다른 세션의 미커밋 WIP 존재). 다만 discussion 세션이 최근(2026-09-07) bengio에 daily-cards 체크아웃(`/SSDe/jemo_maeng/src/drone-MemorySAM-daily`, GitHub 직접 클론)을 새로 만들어 B0·E0·E1·E9 등을 그쪽에서 돌리고 있으므로, **bengio는 기존 체크아웃과 daily-cards 체크아웃 두 개가 공존**한다 — 혼동 주의.
- **hpca100 시스템 시계는 UTC**, yeon/jarvis/bengio/lecun은 KST(UTC+9)다. 로그 타임스탬프를 서버 간 직접 비교하면 9시간 오차가 난다(메모리 `hpca100-utc-clock` 참고). 위 §2 표의 hpca100 두 행(E2·E7c)은 UTC 기준, 나머지 세 행(yeon)은 KST 기준으로 각각 실측한 것이니 섞어서 계산하지 말 것.
- **registry.md/current.md 편집 시 병합 충돌 주의** — discussion 세션("MMSAM | 생각정리")도 같은 파일을 자주 고친다. 편집 전 반드시 `git pull`(fast-forward 시도), 안 되면 `git merge`로 받아서 양쪽 내용 다 보존하는 방식으로 충돌 해결(오늘 실제로 한 번 겪음, 커밋 `cddc319` 참고).
- **P52 MUSES seed1/seed2(hpca100)는 보류 중, 체크포인트 보존됨** — `hpca100_muses_rgbelr_P52_seed20260901`(last_checkpoint epoch54, best val 79.63@ep54) / `_seed20260902`(last_checkpoint epoch34, best val 78.81@ep30). AUTO_RESUME:true라 GPU 여유 생기면 그대로 이어 돌릴 수 있음. 재개 여부는 discussion 세션의 P52.1 재설계 정리 후 판단.
- **discussion 세션("MMSAM | 생각정리")과 "mmsam session merge" 세션이 실질적인 판단·설계 주체다.** 이 세션(구 p30-det)은 그쪽 지시를 실행·검증·기동하는 역할이었다 — 새 세션도 그 관계를 그대로 이어받으면 된다. `ListAgents`로 두 세션의 현재 이름/상태를 재확인할 것(이름이 바뀔 수 있음).
- **세션과 무관하게 도는 사용자 크론탭 3건이 있다**(이 세션 소관 아님, `CronList`에도 안 잡힘 — OS 크론탭이라 그런 것으로 추정, "mmsam session merge"가 전달): 5분마다 `watchdog.sh scan`, 10분마다 `gpu_slot_watch.sh scan`, 매주 월요일 05:17 체크포인트 백업. 🔴 **이 크론들의 로그 출력 경로가 워크트리 `logs-reorg-0808/.watchdog/` 아래로 하드코딩돼 있다** — 그 워크트리를 지우면 로그 출력만 조용히 깨진다(스크립트 실행 자체는 리포 루트 경로를 우선 참조해서 안전). **워크트리 정리 전 이 경로를 먼저 확인·이관할 것.**

## 5. 다음에 판단해야 할 것 (런 종료 시점 기준)

- **P52 DELIVER seed1·seed2 완주 시(§2 ETA 참고, ~4.5~4.7일 후)**: val-best ckpt를 `val.py` 하네스가드+공식 재채점 → G1 게이트(≥54.65) 통과 여부 + `[C3-ADPT]`/`[UB-ADPT]` 로그로 G4(RailTrack λ_c↑ 창발) 확인. 단, discussion 세션의 P52.1 재설계(게이트 재등록·시드 3페어 규약)가 그 전에 정리되면 그 새 기준을 따를 것 — 지금 게이트(G1=54.65)는 구버전 P52 개정 문서 기준이라 재설계 결과로 바뀔 수 있음.
- **E-LoRA arm A(r16) 완주 시**: B(shared r16)·C(shared8+resid8)와의 3-way 비교가 목적이었는데, B·C는 아직 미착수(bengio/lecun 동기화 문제로 보류, 2026-09-05 기준) — arm A 완주 결과가 나오면 B·C 착수 여부·장소를 다시 판단해야 함.
- **E2(전 선형층 LoRA) 완주 시(~6.9시간 후)**: B0(현행 레시피) 대비 legal test ≥ +1.0이면 채택 방향(구조 사다리 S2), <+0.5면 폐기. 트레이너 val이 ep20 −1.5 → ep25 +반등한 궤적이 계속 이어지는지가 관건 — 완주 후 반드시 공식 재채점으로 최종 판정할 것(트레이너 val로 조기 판정하지 말 것, 이 프로젝트 반복 교훈).
- **E7c 완주 시(~6.8시간 후)**: E7(PhysAug-off)과 페어로 `tools/eval_muses_official.py` 공식 재채점 → PhysAug 효과 크기 확정, MUSES 헤드라인 교체 여부 결정(3.5-1 지적 반영).
- **위 넷 다 끝나면 hpca100 GPU1,2,3 + yeon GPU2~7이 전부 해방** — 그 시점에 discussion 세션과 다시 상의해서 다음 배치(P52.1 본런 3벤치×3페어, 축2 A2, E-LoRA B·C 등)를 정할 것. GPU 유휴 금지 원칙 적용.

관련 문서: [decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md](../decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md) · [decisions/2026-09-07-daily-cycle-experiment-cards.md](../decisions/2026-09-07-daily-cycle-experiment-cards.md) · [decisions/2026-08-31-p52-rxdino-adaptive-amendment.md](../decisions/2026-08-31-p52-rxdino-adaptive-amendment.md) · [experiments/registry.md](../experiments/registry.md)
