---
created: 2026-09-08
author: 이 세션(worktree `.claude/worktrees/p30-det`, background job — EnterWorktree 아님, 작업폴더가 잡 생성 시 고정된 세션이라 ExitWorktree로 못 나감)
status: ✅ 인계 완료 (2026-09-08) — 🔴 감시 주체는 세션 **이름**이 아니라 아래 감시 task id 로 특정한다. 같은 이름을 다른 세션이 가질 수 있어 이름으로 지목하면 혼선이 난다(2026-09-08 실제로 발생: 구 p30-det 세션이 그 이름을 쥔 채 구식 로직 감시 4건을 겹쳐 걸었다가 정리됨). 현재 감시(2026-09-08 18:30 KST 기준) = bra3thfme(yeon P52 seed1·seed2) · b6kocwysa(yeon E-LoRA arm A r16) · b1g0vla8o(hpca100 E7c, 정확일치판) · bbr0mchq0(hpca100 E1M) · byyvs2cj4(hpca100 E2 legal 재채점), 크론 = c0aba5c9. 구 bwsc1xofh(E2 학습)은 E2 완주로 역할을 다해 정지했고, 구 bz6goimur(E7c)은 ⑥ 의 `=` 정확일치판으로 교체했다. 세션이 재기동되면 id 가 바뀌므로 이 줄도 함께 갱신할 것.
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

## 1. 활성 감시 열두 건 — 재설치 명세

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
| `ALIVE` | `tmux has-session -t =<세션> 2>/dev/null && echo ALIVE` (🔴 `=` 필수, 아래 ⑥) | 세션 생존 |
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

⑥ 🔴 **`tmux has-session -t` 앞에 `=` 를 반드시 붙인다(2026-09-08 실제 사고).** `tmux` 의 대상 세션 해석은
**접두어 매칭**이라, `-t hpca100_E2` 는 `hpca100_E2_eval` 같은 **다른 세션에도 걸린다.** 그래서 감시 대상이
끝났는데도 이름이 겹치는 새 세션이 있으면 계속 `ALIVE` 로 판정되어 **`SESSION_ENDED` 가 영영 안 나온다.**

2026-09-08 에 그대로 재현됐다. E2 학습이 08:35 UTC 에 끝났는데 재채점을 위해 만든 `hpca100_E2_eval` 세션이
접두어에 걸려 감시가 완주를 놓쳤고, 40분 뒤 로그 무갱신 임계에 걸려 **"데드락 의심"이라는 엉뚱한 알림**이 왔다.
별도 감지기를 따로 걸어 둔 덕에 완주는 13분 만에 알았지, 감시에만 의존했다면 GPU 가 그만큼 더 놀았다.

서버에서 직접 검증한 결과다.

| 명령 | 결과 |
|---|---|
| `tmux has-session -t hpca100_E2` | MATCH (그 이름의 세션은 이미 없는데도) |
| `tmux has-session -t =hpca100_E2` | NOMATCH (정확) |
| `tmux has-session -t =hpca100_E2_eval` | MATCH |

**완주한 실험의 자리에 재채점·후속 세션을 만들 때 이름이 겹치기 쉬우므로**(`<실험>_eval` 이 자연스러운 작명이다)
이 함정은 반복된다. `=` 를 붙이거나, 후속 세션 이름을 접두어가 겹치지 않게 짓는다. 둘 다 하는 편이 안전하다.

⑦ 🔴 **첫 평가 전에 감시를 걸 때의 오탐 두 가지(2026-09-08 실제 발생).** 기동 직후, 즉 로그에 `[Val]` 줄이
아직 하나도 없는 런에 감시를 걸면 `val` 이 빈 문자열이 된다. 그 상태에서 초판 로직은 두 곳에서 잘못 동작한다.

- **`WATCH_START` 무한 반복**: 시작 알림 여부를 `[ -z "$PREV" ]` 로 판단하는데, 빈 `val` 을 `PREV` 에 넣어도
  `PREV` 가 여전히 비어 있어 조건이 계속 참이다. 매 주기 시작 알림이 반복된다. → **`STARTED` 플래그를 따로 둔다.**
- **`STALLED_VAL` 오탐**: 빈 `val` 이 매 주기 `PREV` 와 같다고 판정되어 정체 카운터가 올라간다. 첫 평가가
  `CYCLES × POLL` 보다 늦게 오는 런이면 **아직 정상인데 정체 알림이 뜬다.** → **`val` 이 비어 있지 않을 때만 센다.**

2026-09-08 E1M 기동 때 첫 증상이 났다(첫 평가 ep5 = 약 78분 뒤, 임계 6주기 = 150분이라 두 번째 증상은
아슬아슬하게 비껴갔다). `EVAL_INTERVAL` 이 크거나 epoch 이 느린 런이면 정체 오탐까지 났을 것이다.
**이미 `[Val]` 이 쌓인 런에 거는 경우에는 두 증상 모두 드러나지 않으므로**, 기동 직후에 감시를 거는 자리에서만
문제가 된다 — 그래서 늦게 발견됐다. §1-2·§1-3 스크립트에는 수정이 반영돼 있다.

⑧ 🔴 **원격 명령 안에서 `awk '{print $N}'` 같은 인용을 겹치지 마라(2026-09-08).** 감시 스크립트는
`ssh 서버 "..."` 안에 다시 `$(...)` 와 작은따옴표가 들어가는 3중 인용 구조다. 여기서 `awk` 의 `$4` 를
살리려면 역슬래시를 여러 겹 붙여야 하는데, 한 겹만 어긋나도 **오류 없이 빈 문자열이 나온다.** 실제로
`/tmp` 여유 공간을 찍으려다 `여유=` 로 값이 비어 나왔다.

**해법은 원격에서 가공하지 않고 원문을 그대로 받아 로컬에서 파싱하는 것이다.** 인용 계층이 하나 줄어
깨질 여지가 없어진다.

```bash
# 나쁨 — 원격에서 awk 로 뽑으려다 인용이 깨진다
echo \"DISK:\$(df -h /tmp | tail -1 | awk '{print \\\$4}')\"

# 좋음 — 원문을 구분자로 감싸 보내고 로컬에서 자른다
echo DFSTART
df -h /tmp | tail -1
echo ERRSTART
# ... 로컬에서:
disk=$(echo "$out" | sed -n '/^DFSTART$/,/^ERRSTART$/p' | sed -n '2p' | awk '{print $4}')
```

이 결함도 ①~⑦ 과 마찬가지로 **조용히 실패한다** — 값이 비어도 감시는 계속 돌기 때문에 알림 문구를
읽어 보지 않으면 모른다. 새 필드를 추가할 때는 **첫 알림의 값이 실제로 채워졌는지 반드시 눈으로 확인하라.**

⑨ **감시 대상 서버의 디스크 여유를 알림에 함께 찍어라(2026-09-08 사고 반영).** hpca100 의
`/home/jovyan/SSDb` 가 100% 차서 E1M 학습이 ep18 에서 **Traceback 없이** 죽고 새 tmux 세션 생성까지
실패한 일이 있었다. 그때 감시는 원인을 모른 채 `STALLED_LOG`("데드락 의심")만 냈다. 에러 패턴에
`No space left` 를 넣고 알림마다 `df` 결과를 붙이면 같은 상황에서 원인이 즉시 드러난다.

⑩ 🔴 **체크포인트를 자동 회수·정리할 때 `test_` 접두어를 반드시 제외하라(2026-09-09 실사고).**
학습 코드는 val 기준과 test 기준 체크포인트를 **같은 `*_top1_checkpoint.pth` 이름**으로 저장한다.

```
epoch5_61.04_top1_checkpoint.pth        ← val-best (판정 대상)
test_epoch5_52.28_top1_checkpoint.pth   ← test-best (메모리 seg-report-sota-gap 이 사용 금지)
```

`/tmp` 휘발 대책으로 만든 회수 스크립트가 `*_top1_checkpoint.pth` 로만 찾은 탓에 **test-best 까지 받아 갔고**,
거기에 "같은 런의 이전 회수본은 지운다"는 규칙이 겹쳐 **먼저 받아 둔 val-best 를 밀어냈다.** 결과적으로
NAS 에는 쓰면 안 되는 파일만 남고 정작 필요한 것이 사라졌다(원본이 `/tmp` 에 있어 실손실은 없었다).

**고침**: 찾을 때 `! -name 'test_*'` 를 붙이고, 이전본 정리 대상에서도 `test_*` 를 뺀다.

이 함정은 **이미 알고 있던 것을 다른 스크립트에 옮기지 못해** 생겼다 — E2 재채점 때 ckpt 목록을 뽑으며
`grep -E '^epoch.*top1_checkpoint'` 로 `test_` 를 걸러낸 적이 있는데, 회수 장치를 새로 만들 때 그 지식을
적용하지 않았다. **체크포인트를 이름으로 다루는 코드를 새로 쓸 때마다 이 두 계열을 먼저 떠올려라.**

### 1-1. 감시 대상과 임계 (2026-09-10 12:00 KST 실측 — 열둘)

> 🔴 **09-10 04:00 판을 대체한다.** 바뀐 것: **E13 확정 런 200ep 기동**(legal test 56.30·24클래스 +1.79 로 카드 최고),
> **E4b 가 Adam ComplexFloat 에러로 사망**해 ep35 val-best 재채점으로 전환.

`CYCLES` 는 **실측 `[Val]` 간격 × 2.5 ÷ 폴링**. 🔴 간격은 학습 속도가 아니라 **`[Val]` 줄 두 개의 타임스탬프 차이**로 재라.

| 항목 | 서버 | tmux 세션 | 실측 ep 시간 | `[Val]` 간격 | 폴링 | `CYCLES` |
|---|---|---|---|---|---|---|
| 1-a | `yeon` | `p52_deliver_s1`, `p52_deliver_s2` | 40.0 / 39.0분 | 2ep ≈ 79분 | 1800초 | 7 |
| 1-b | `yeon` | `elora_a_r16` (arm A) | 40.5분 | 2ep ≈ 81분 | 1500초 | 8 |
| 1-c | `yeon` | `elora_c_shres` (arm C) | 41.5분 | 2ep ≈ 83분 | 1500초 | 8 |
| 1-d | `jarvis` | `elora_b_shared` (arm B) | 18.8분 | 2ep ≈ 38분 | 1500초 | 4 |
| 1-e | `jarvis` | `e1_confirm200` (E1 확정 런, GPU2·4) | 19.0분 | 2ep ≈ 38분 | 1500초 | 6 |
| 1-f | `jarvis` | **`e13_confirm200`** (E13 확정 런, GPU1 **단독**) | **30.2분** | **5ep ≈ 151분** | 1500초 | **15** |
| 1-g | `jarvis` | `e1s2_resume` | 30.4분 | 5ep ≈ 152분 | 1500초 | 20 |
| 1-h | `jarvis` | **`e4b_legal`** (재채점, 완료 시 종료) | — | — | 600초 | — |
| 1-i | `hpca100` | `hpca100_E12` | 56.2분 | 5ep ≈ 281분 | 1500초 | 31 |
| 1-j | `hpca100` | `hpca100_E2s2` | 55.2분 | 5ep ≈ 276분 | 1500초 | 30 |
| 1-k | `hpca100` | `hpca100_B0s2` | 40.4분 | 5ep ≈ 202분 | 1500초 | 25 |
| 1-l | `hpca100` | (세션 없음 — `/tmp` val-best NAS 자동 회수) | — | — | 회수 주기 | — |
| 1-m | `lecun` | CAFuser (감시 미설치, 조회로만 추적) | — | iter 기반 | — | — |

로그: yeon `<yeon-p38>/logs/<세션>_launch.log` · jarvis `/SSDb/jemo_maeng/src/drone-MemorySAM/logs/<세션>_launch.log`
· hpca100 `/tmp/jemo_scratch/logs/<세션>_launch.log`(B0s2 만 `_resume.log`)

⚠️ **두 확정 런의 조건이 다르다.** E1 은 GPU 2장·`EVAL_INTERVAL 2`(19.0분/ep), E13 은 **GPU 1장·`EVAL_INTERVAL 5`**(30.2분/ep)다.
E13 이 1.6배 느려 완주가 **09-14** 로 E1(09-12)보다 이틀 늦다. **24클래스 기준으로는 E13(+1.79)이 E1(+0.57)의 세 배인데
자원은 절반**이므로, 자리가 나면 E13 에 GPU 를 더 주는 것을 검토하라(다만 DDP 재기동은 진행분 손실이 있다).

⚠️ **hpca100 은 공유 GPU 다.** E3s2 가 평가 구간 38.2GB(A100 의 93%)를 쓰다가 타 사용자가 34GB 를 잡자 OOM 으로 죽었다.

**완주 임박(09-10)**: E2s2 13:10 → E12 13:30 → E1s2 13:50 → e4b_legal 14:00 → B0s2 15:30.
그 뒤 arm B 09-12 03시 · E1 확정 09-12 18시 · P52 s2 19시 · P52 s1 21시 · arm A 09-13 03시 · lecun 09-14 03시 · E13 확정 09-14 09시 · arm C 09-14 22시.

🔴 **`PYTORCH_CUDA_ALLOC_CONF` 에 `expandable_segments:True` 와 `max_split_size_mb` 를 함께 주지 마라 (2026-09-10 실증).**
두 옵션은 호환되지 않아 할당자 내부에서 터진다:
```
RuntimeError: !block->expandable_segment_ INTERNAL ASSERT FAILED
  at "../c10/cuda/CUDACachingAllocator.cpp":2549, please report a bug to PyTorch.
```
E3s2 의 OOM 재발을 막으려고 `expandable_segments:True,max_split_size_mb:128` 을 넣었다가 **E13s2 가 기동 직후
이 assert 로 죽었다.** 같은 설정을 받은 E3s2 는 ep13 까지 살아 있었는데, assert 가 특정 할당 패턴에서만
발동하기 때문이지 안전해서가 아니다 — 두 런 모두에서 `max_split_size_mb` 를 제거했다.
**단편화 완화가 목적이면 `expandable_segments:True` 단독으로 충분하다.**


### 1-2. 재설치 스크립트 (단일 대상판 — 1-b·1-c·1-d 공통)

`SRV`·`S`·`log`·`CYCLES`·`POLL`만 위 표대로 바꿔서 쓴다.

```bash
SRV=yeon; S=elora_a_r16
log=/SSDb/jemo_maeng/src/Project/Drone/detection/drone-MemorySAM-p38/logs/elora_a_r16_launch.log
STALE=900; CYCLES=8; POLL=1500
PREV=""; CYC=0; FS=0; FV=0; FAIL=0; STARTED=0
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
    if [ "$STARTED" = "0" ]; then
      echo "[$S] WATCH_START — 감시를 시작했습니다. last=${val:-없음(첫 평가 전)}"; STARTED=1; PREV="$val"; CYC=0; FV=0
    elif [ "$val" = "$PREV" ]; then
      CYC=$(( CYC + 1 ))
      if [ -n "$val" ] && [ "$CYC" -ge "$CYCLES" ] && [ "$FV" = "0" ]; then
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

- 🔴 **감시 생존 확인을 `TaskList` 로 하지 마라.** 감시는 `Monitor` 로 걸리고 `TaskList` 는 별개의 작업
  목록이라, 감시가 열 건 다 살아 있어도 `TaskList` 는 `No tasks found` 를 돌려준다(2026-09-09 실측).
  살아 있는지는 **감시 이벤트가 계속 도착하는가**로 판단하고, 개별 감시를 끄거나 갈아 끼울 때만
  `TaskStop <task-id>` 를 쓴다. 정기보고 크론 지시문에 "TaskList 로 확인하라"고 적혀 있다면 그 문구가
  틀린 것이다.
- **정기보고 크론 지시문의 조회 대상도 낡는다.** 크론은 만들 당시의 tmux 세션 이름을 문자열로 담고 있어
  런이 완주하면 없는 세션을 조회하게 된다. 보고를 쓸 때는 지시문을 그대로 따르지 말고 **`tmux ls` 로
  현재 세션을 먼저 확인**하라(2026-09-09 에 크론이 이미 완주한 `hpca100_E2`·`hpca100_E7c` 와 양도된
  bengio 를 조회 대상으로 지목하고 있었다).

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

- 🔴 **감시를 걸지 않은 런이 죽으면 GPU를 잃는다 — 추정이 아니라 실증(2026-09-09).** jarvis DGFusion이 12:32에 발산으로 죽었는데 감시가 없어 14:20까지 몰랐고, 그 1시간 45분 사이에 다른 사용자가 **GPU5·7 두 장을 가져갔다**. 되찾을 방법은 없다. 그러므로 **런을 기동하면 같은 턴에서 감시를 설치한다**. 감시가 `break` 로 끝났으면 그 자리에서 재설치한다 — "나중에"는 없다. 이 규칙은 학습뿐 아니라 10분짜리 평가에도 적용한다(빈 GPU는 몇 분 만에 채워진다, 메모리 `gpu-never-idle`).
- **`tmux kill-session` 은 조용히 실패한다.** `tmux kill-session -t =X 2>/dev/null && echo KILLED` 로 결과를 반드시 확인하라. 2026-09-09에 잘못된 설정으로 돌던 평가를 죽이려다 실패했는데 에러가 묻혀서, 그 런이 11분을 더 돌고 쓸모없는 결과를 냈다. 죽인 뒤에는 `tmux has-session -t =X` 로 부재를 확인하고 나서 다음 단계로 넘어간다.
- **DGFusion 평가에서 스플릿을 바꿀 때는 `DATASETS.TEST` 가 아니라 `DATASETS.TEST_SEMANTIC` 을 덮어쓴다.** `DATASETS.TEST` 만 바꾸면 오버라이드가 config 에는 찍히지만 세만틱 평가는 그대로 이전 스플릿을 쓴다. DELIVER 는 val 2005장 / test 1897장이므로 **로그의 `Inference done N/M` 장수로 어느 스플릿인지 판별**할 수 있다.

## 5. 다음에 판단해야 할 것 (런 종료 시점 기준)

- **P52 DELIVER seed1·seed2 완주 시(§2 ETA 참고, ~4.5~4.7일 후)**: val-best ckpt를 `val.py` 하네스가드+공식 재채점 → G1 게이트(≥54.65) 통과 여부 + `[C3-ADPT]`/`[UB-ADPT]` 로그로 G4(RailTrack λ_c↑ 창발) 확인. 단, discussion 세션의 P52.1 재설계(게이트 재등록·시드 3페어 규약)가 그 전에 정리되면 그 새 기준을 따를 것 — 지금 게이트(G1=54.65)는 구버전 P52 개정 문서 기준이라 재설계 결과로 바뀔 수 있음.
- **E-LoRA arm A(r16) 완주 시**: B(shared r16)·C(shared8+resid8)와의 3-way 비교가 목적이었는데, B·C는 아직 미착수(bengio/lecun 동기화 문제로 보류, 2026-09-05 기준) — arm A 완주 결과가 나오면 B·C 착수 여부·장소를 다시 판단해야 함.
- **E2(전 선형층 LoRA) 완주 시(~6.9시간 후)**: B0(현행 레시피) 대비 legal test ≥ +1.0이면 채택 방향(구조 사다리 S2), <+0.5면 폐기. 트레이너 val이 ep20 −1.5 → ep25 +반등한 궤적이 계속 이어지는지가 관건 — 완주 후 반드시 공식 재채점으로 최종 판정할 것(트레이너 val로 조기 판정하지 말 것, 이 프로젝트 반복 교훈).
- **E7c 완주 시(~6.8시간 후)**: E7(PhysAug-off)과 페어로 `tools/eval_muses_official.py` 공식 재채점 → PhysAug 효과 크기 확정, MUSES 헤드라인 교체 여부 결정(3.5-1 지적 반영).
- **위 넷 다 끝나면 hpca100 GPU1,2,3 + yeon GPU2~7이 전부 해방** — 그 시점에 discussion 세션과 다시 상의해서 다음 배치(P52.1 본런 3벤치×3페어, 축2 A2, E-LoRA B·C 등)를 정할 것. GPU 유휴 금지 원칙 적용.

관련 문서: [decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md](../decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md) · [decisions/2026-09-07-daily-cycle-experiment-cards.md](../decisions/2026-09-07-daily-cycle-experiment-cards.md) · [decisions/2026-08-31-p52-rxdino-adaptive-amendment.md](../decisions/2026-08-31-p52-rxdino-adaptive-amendment.md) · [experiments/registry.md](../experiments/registry.md)
