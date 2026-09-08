---
created: 2026-09-08
author: 이 세션(worktree `.claude/worktrees/p30-det`, background job — EnterWorktree 아님, 작업폴더가 잡 생성 시 고정된 세션이라 ExitWorktree로 못 나감)
status: 🟡 인계 대기 — 새 세션이 아래 감시 2건을 재설치하고 정상 동작을 확인하면 이 세션 종료
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

## 1. 활성 감시 2건 — 재설치 명세

### 1-a. `bx6x2n7x2` — yeon P52 DELIVER seed1·seed2 크래시/진행 감지

- 대상 서버: `yeon` (ssh alias)
- 대상 tmux 세션(각각 독립): `p52_deliver_s1`, `p52_deliver_s2` (window `main`)
- 감시 로그 절대경로:
  - `/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs/p52_deliver_s1_launch.log`
  - `/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs/p52_deliver_s2_launch.log`
- 크래시 판정: `tmux has-session -t <세션>`이 실패(세션 없음) → `SESSION_ENDED`(완주 또는 죽음, 로그 tail로 구분). 로그에 `Traceback|CUDA out of memory|Killed|RuntimeError` 매칭 → `ERROR_DETECTED`(세션은 아직 살아있는 상태에서 잡힐 수 있음).
- 진행 판정: 로그에서 `\[Val\] epoch:[0-9]+  mIoU: [0-9.]+  Best: [0-9.]+ \(ep[0-9]+\)` 패턴의 최신 줄.
- 폴링 주기: 1800초.
- 재설치 스크립트 골격(Monitor 도구, `persistent:true`):
  ```bash
  while true; do
    for s in p52_deliver_s1 p52_deliver_s2; do
      alive=$(ssh -o ConnectTimeout=10 yeon "tmux has-session -t $s 2>/dev/null && echo ALIVE" 2>/dev/null)
      tail_out=$(ssh -o ConnectTimeout=10 yeon "tail -c 2000 /SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs/${s}_launch.log 2>/dev/null | tr '\r' '\n'" 2>/dev/null)
      last_val=$(echo "$tail_out" | grep -oE '\[Val\] epoch:[0-9]+  mIoU: [0-9.]+  Best: [0-9.]+ \(ep[0-9]+\)' | tail -1)
      err=$(echo "$tail_out" | grep -E 'Traceback|CUDA out of memory|Killed|RuntimeError' | tail -2)
      case "$alive" in
        *ALIVE*) [ -n "$err" ] && echo "[$s] ERROR_DETECTED last=${last_val:-unknown}: $err" || { [ -n "$last_val" ] && echo "[$s] PROGRESS: $last_val"; } ;;
        *) echo "[$s] SESSION_ENDED last=${last_val:-unknown}"; echo "$tail_out" | tail -10 ;;
      esac
    done
    sleep 1800
  done
  ```

### 1-b. `bsl6xtt9w` — yeon E-LoRA arm A(r16) 크래시/진행 감지

- 대상 서버: `yeon`
- 대상 tmux 세션: `elora_a_r16` (window `main`)
- 감시 로그 절대경로: `/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs/elora_a_r16_launch.log`
- 크래시/진행 판정: 1-a와 동일 패턴(단일 세션 버전 — `SESSION_ENDED` 시 `break`로 루프 종료).
- 폴링 주기: 1500초.
- 🔴 **2026-09-08 버그 수정 이력**: 최초 설치판은 `tmux has-session ... 2>&1`로 원격 stderr를 stdout에 합류시킨 뒤 `[ -z "$alive" ]`로 판정했다 — 세션이 죽으면 `alive`가 `can't find session: ...` 에러 문구를 담아 **빈 문자열이 아니게 되므로 SESSION_ENDED가 영원히 안 찍히는 치명적 버그**였다("mmsam session merge" 세션이 발견, `bsl6xtt9w` 실측으로 확인됨 — arm A는 최대 며칠간 크래시 무방비 상태였을 수 있음). 2026-09-08 `2>/dev/null` + `case ... *ALIVE*)` 매칭으로 재설치·검증 완료. 아래는 **수정된 버전**이다 — 재설치 시 반드시 이 버전을 쓸 것.
- 재설치 스크립트 골격(Monitor 도구, `persistent:true`):
  ```bash
  while true; do
    alive=$(ssh -o ConnectTimeout=10 yeon "tmux has-session -t elora_a_r16 2>/dev/null && echo ALIVE" 2>/dev/null)
    tail_out=$(ssh -o ConnectTimeout=10 yeon "tail -c 3000 /SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs/elora_a_r16_launch.log 2>/dev/null | tr '\r' '\n'" 2>/dev/null)
    last_val=$(echo "$tail_out" | grep -oE '\[Val\] epoch:[0-9]+  mIoU: [0-9.]+  Best: [0-9.]+ \(ep[0-9]+\)' | tail -1)
    err=$(echo "$tail_out" | grep -E 'Traceback|CUDA out of memory|Killed|RuntimeError' | tail -3)
    case "$alive" in
      *ALIVE*)
        if [ -n "$err" ]; then echo "ERROR_DETECTED (still alive) last=${last_val:-unknown}: $err";
        elif [ -n "$last_val" ]; then echo "PROGRESS: $last_val"; fi
        ;;
      *)
        echo "SESSION_ENDED last=${last_val:-unknown}"; echo "$tail_out" | tail -15; break
        ;;
    esac
    sleep 1500
  done
  ```

### 1-c. `b3ed3wvps` — hpca100 E2 크래시/완주 감지 (2026-09-08 신설)

- 대상 서버: `hpca100`, tmux 세션 `hpca100_E2`
- 감시 로그: `/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM/logs/hpca100_E2_launch.log`
- 판정 로직은 1-b 수정판과 동일(`case ... *ALIVE*)`), 폴링 1500초.
- 완주 예정 2026-09-08 ~08:20 UTC(§2) — **완주 감지 시 GPU1,3이 비니 즉시 다음 배치를 정할 것**(`gpu-never-idle` 원칙).

### 1-d. `bmhf79on7` — hpca100 E7c 크래시/완주 감지 (2026-09-08 신설)

- 대상 서버: `hpca100`, tmux 세션 `hpca100_E7c`
- 감시 로그: `/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM/logs/hpca100_E7c_launch.log`
- 판정 로직·폴링 동일. 완주 예정 2026-09-08 ~08:50 UTC(§2) — **완주 감지 시 GPU2가 비니 즉시 다음 배치를 정할 것.**

이 두 감시는 원래 §1 끝에 "안 걸려 있다"고만 적었으나(초판 인계 시점 판단 보류), discussion 세션("mmsam session merge")이 "완주가 몇 시간 내인데 인계 타이밍이 사용자 조작에 달려 불확실하다"고 지적해 **이 세션이 직접 지금 걸었다** — 인계 대상이 2건에서 4건으로 늘었다. 새 세션은 1-a~1-d 넷 다 재설치 대상으로 볼 것.

### 1-e. 세션 내부 정기 실행 — 없음 확인(2026-09-08)

`CronCreate`로 만든 크론, `/loop` 반복 등 이 세션 안에서 돈 정기 실행 장치는 `CronList` 조회 결과 **없다**("No scheduled jobs."). 세션이 죽어도 같이 사라질 숨은 루틴은 없다는 뜻 — 위 1-a~1-d 넷이 이 세션이 책임지는 것의 전부다.

## 2. 현재 활성 런 현황 (2026-09-08 02:00~02:05 UTC 실측)

| 서버 | 실험 | 데이터셋 | 진행 | 최근 val | 내부최고 델타 | ETA |
|---|---|---|---|---|---|---|
| yeon GPU2,3 (tmux `p52_deliver_s1`) | P52 RxDINO DELIVER seed20260901 | DELIVER 4모달 | ep32/200 | 65.51 (mIoU) | best 65.51@ep32(신기록 갱신 직후), 게이트 G1 기준 54.65(Phase1 legal test 54.95−0.3) — **주의: 이 val은 legal test 아님, 최종 판정은 완주 후 val.py 공식 재채점** | ~4.7일 (39.9분/ep 실측, 09-13 새벽경) |
| yeon GPU4,5 (tmux `p52_deliver_s2`) | P52 RxDINO DELIVER seed20260902 | DELIVER 4모달 | ep34/200 | 65.61 | best 66.19@ep28 | ~4.5일 (39.1분/ep, 09-12 저녁경) |
| yeon GPU6,7 (tmux `elora_a_r16`) | E-LoRA arm A 재런(per-modal r16) | DELIVER 4모달 | ep34/200 | 65.88 | best 66.64@ep26 — 이 값을 B/C(각 r16 기준 shared/shared+resid)와 비교할 새 arm A 기준치 | ~4.66일 (40.4분/ep, 09-13 오전경) |
| hpca100 GPU1,3 (tmux `hpca100_E2`) | 일일카드 E2(LoRA 전 선형층 QKVO+MLP, r32) | DELIVER 4모달 | ep25/40 | 65.25 | B0(현행 레시피, 변경 0) 대비 — ep20 시점 62.40으로 −1.5(음성 방향)였으나 ep25에서 65.25로 반등, **정본 비교용으로 완주까지 중단 안 함**(discussion 세션 지시) | ~6.9시간 (27.8분/ep, 09-08 08:20 UTC≈17:20 KST) |
| hpca100 GPU2 (tmux `hpca100_E7c`) | 일일카드 E7c(MUSES PhysAug-on 대조군, PYTHONUNBUFFERED=1) | MUSES 3모달(img/lidar/event) | ep4/40 (70%) | 아직 첫 eval 전(EVAL_INTERVAL 5) | E7(PhysAug-off, 완료, 트레이너 val 80.29@ep40 — **legal test 아님**)과 페어 비교 예정 | ~6.8시간 (11.6분/ep, 09-08 08:50 UTC≈17:50 KST) |

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
