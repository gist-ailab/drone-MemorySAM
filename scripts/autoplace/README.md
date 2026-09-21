# autoplace — 빈 GPU 자동 배치 파이프라인

연구실 서버들을 주기적으로 조사해 빈 GPU 를 찾고, 대기열(`queue.tsv`)의 다음 작업을
tmux 세션으로 자동 기동한 뒤, 몇 분 뒤 살아 있는지 검증까지 한 번에 돌리는 도구 모음.
서버 목록과 정책(배치 금지 서버·금지 GPU)은 `scripts/servers.conf` 가 단일 출처다.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `probe_free_gpus.sh` | 서버 한 대의 빈 GPU 인덱스를 조사해 공백 구분 한 줄로 출력 |
| `queue.tsv` | 배치 대기열(탭 9열). 사람이 여기에 행을 추가한다 |
| `hostenv.tsv` | 서버별 기동 환경(conda.sh 경로, pylibs, 포트 기준값) |
| `place.py` | 대기열을 빈 GPU 에 배정. `--launch` 면 ssh+tmux 기동 후 state 기록 |
| `verify.py` | 최근 기동 항목의 로그·GPU 점유를 검증 (PASS/FAIL) |
| `run_cycle.sh` | 배치 → 240초 대기 → 검증 한 사이클. `--dry` 는 계획만 |
| `state/launched.tsv` | place.py 가 append 하는 기동 기록 (자동 생성) |

## 사용 예시

```bash
# 1) 대기열 확인/추가: queue.tsv 에 행 추가(주석 예시 2줄 참고)

# 2) 무엇이 어디에 배정될지 미리 보기 (기동 없음)
python3 scripts/autoplace/place.py

# 3) 실제 배치 + 검증까지 한 사이클
bash scripts/autoplace/run_cycle.sh

# 4) 계획만 다시 보기
bash scripts/autoplace/run_cycle.sh --dry

# 5) 최근 60분 안에 올린 작업만 다시 검증
python3 scripts/autoplace/verify.py --since-min 60

# 6) 특정 세션만 검증
python3 scripts/autoplace/verify.py --repo-log-name muses_p52_e1_s902
```

## queue.tsv 열 의미 (탭 구분 9열)

| 열 | 의미 |
|----|------|
| `id` | 고유 문자열(중복 금지). state 에 이미 있으면 건너뛴다 |
| `priority` | 정수, 작을수록 먼저 배정 |
| `hosts` | 배치 허용 서버. `any` 또는 쉼표 목록(예: `bengio,yeon`) |
| `ngpu` | 1 또는 2 |
| `repo` | 그 서버에서의 저장소 절대경로 |
| `config` | 그 서버에서의 config 절대경로 |
| `session` | tmux 세션 이름. 로그 파일명(`logs/<session>_launch.log`)도 이것을 쓴다 |
| `epochs` | 참고용 정수 |
| `note` | 자유 문자열 |

## hostenv.tsv 열 의미 (탭 구분 4열)

| 열 | 의미 |
|----|------|
| `host` | `scripts/servers.conf` 의 alias |
| `conda_sh` | `source <conda.sh>` 에 쓸 conda.sh 절대경로 |
| `pylibs` | `PYTHONPATH` 접두에 붙는 외부 라이브러리 경로. 없으면 `-` |
| `master_port_base` | torchrun 마스터 포트 기준값. 실제 포트 = base + 배정 GPU 첫 인덱스 |

값을 모르면 `FILL_ME` 로 둔다. `FILL_ME` 가 하나라도 있는 서버에는 **배치하지 않는다**
(place.py 가 강제). 초기 상태에서 거의 모든 서버가 `FILL_ME` 이므로, 운용 전에 이 파일부터
채워야 한다. `jarvis` 의 conda_sh 는 servers.conf 기록("conda at /home/jemo_maeng/miniconda3")에서
표준 레이아웃 경로를 유추해 넣어 둔 값이니 사용 전 확인하라.

기동 명령의 `conda activate <env>` 대상 환경명은 servers.conf 의 `conda_env` 열에서 읽는다.

## state(launched.tsv) 열 (탭 구분 5열)

`id | host | gpus | ISO시각 | session` — place.py 가 기동 성공 건만 append 한다.
기동에 실패한 작업은 기록되지 않으므로 다음 회차에 자동 재시도된다.

## verify.py 판정 항목

최근 N 분(기본 30) 안에 기록된 항목마다 아래를 검사한다. FAIL 이 하나라도 있으면 종료코드 1.

1. **파라미터 마커** — 로그에 `total_trainable` 또는 `lora_params_total` 줄이 있는가
2. **치명 오류 없음** — 로그에 `Traceback|OutOfMemory|Error` 가 없는가
3. **학습 전진** — `Epoch [` 진행 줄이 있고 반복 인덱스가 0 이 아닌가
4. **GPU 점유** — 배정 GPU 각각의 memory.used 가 3000MiB 이상인가

로그 경로는 queue.tsv 의 같은 id 행 `repo`, 없으면 servers.conf 의 해당 서버 `repo_path` 로
찾는다. `--repo-log-name <세션>` 을 주면 해당 세션 이름의 항목만 검사한다.

## 안전 규칙

- 빈 GPU 판정을 통과하지 못한 GPU 에는 절대 배치하지 않는다. 판정 기준은
  `memory.used <= ${GPU_MAXMEM:-2000}MiB` 그리고 `utilization.gpu <= ${GPU_MAXUTIL:-10}%`.
- 같은 id 를 두 번 띄우지 않는다(state 파일에 있으면 건너뛴다).
- servers.conf `policy=off` 서버는 probe 조차 하지 않는다(예: lecun). `ban:` GPU 는 결과에서 제외(예: jarvis GPU0).
- sudo, pip install, git 명령을 쓰지 않는다. 파일을 지우지 않는다(state 는 append 만).
- `--launch` 없이 실행하면 배정 계획만 출력하고 아무것도 실행하지 않는다.

## 운용 주의

- **중복 배치 창**: 기동 직후 수 분은 GPU 가 아직 비어 보일 수 있다. `run_cycle.sh` 의
  240초 대기는 검증을 위한 것이므로, 그보다 짧은 간격으로 사이클을 돌리면 같은 GPU 에
  다른 작업이 겹칠 수 있다. 사이클 간격은 10분 이상을 권장한다.
- **hpca100 비호환**: hpca100 은 conda 가 없는 venv 서버라 이 파이프라인의
  `source conda.sh && conda activate` 템플릿으로 기동할 수 없다. hostenv 를 FILL_ME 로
  유지해 배치되지 않게 한다.
- **tmux 세션명 충돌**: 같은 서버에 같은 이름의 세션이 이미 있으면 기동이 실패하고
  state 에 기록되지 않는다(의도된 동작).
- place.py 는 배정 시 probe 순서(메모리 오름차순)로 앞의 GPU 부터 고르고,
  같은 회차 안에서 한 번 배정된 GPU 는 다른 작업에 재사용하지 않는다.
