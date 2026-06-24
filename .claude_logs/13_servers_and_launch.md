# 13. 서버 레지스트리 & 원격 실험 자동 실행 (Servers & Remote Launch)

> 이 문서는 "X 실험을 <서버>에서 돌려줘" 류의 지시를 AI가 자동 처리하기 위한 **운영 매뉴얼**이다.
> 서버 메타데이터의 단일 출처(single source of truth)는 **`scripts/servers.conf`** 이고,
> 실제 실행/추적은 **`scripts/remote_exp.sh`** 로 한다.

---

## 0. AI 운영 규칙 (반드시 준수)

사용자가 "<config> 실험을 <서버>에서 돌려줘"라고 하면:

1. **레지스트리 확인**: `bash scripts/remote_exp.sh servers` 로 해당 서버의 repo_path / env / default_gpus 확인.
   - `repo_path` 또는 `gpus` 가 `FILL_ME` 면 멈추고 사용자에게 값을 물어본다 (추측 금지).
2. **GPU 여유 확인 (필수)**: `bash scripts/remote_exp.sh status <서버>` 로 빈 GPU와 기존 `jemo` 세션 창 확인.
   - **반드시 비어 있는 GPU에만 배치**한다 (사용 중 GPU에 얹지 않는다 → OOM/타인 작업 방해).
   - GPU를 직접 안 정했으면 `run ... auto:N` 으로 **원격의 빈 GPU N장을 자동 배정**(`auto`=1장)하거나, `status`로 보고 골라 확정받는다.
3. **실행**: `bash scripts/remote_exp.sh run <서버> <config> <gpus|auto:N>` 실행. 출력의 `LOG=...` 경로를 기록한다.
   - 빈 GPU 판정: `memory.used ≤ 2000MiB && util ≤ 10%` (메모리 적은 순). 빈 GPU 부족하면 런처가 실행을 거부한다.
4. **추적**: 수 분 뒤 `bash scripts/remote_exp.sh log <서버> <cfg_name>` 로 초기 로그(에러/Started epoch 등) 확인.
   초기 몇 스텝이 도는 걸 본 뒤에야 "정상 시작됨"이라고 보고한다.
5. **기록**: 의미 있는 학습을 시작했으면 `03_experiment_log.md`(+ 필요시 `01_project_status.md`)에
   서버·config·GPU·로그경로·시작시각을 남긴다.

원격 학습은 `nohup`이 아니라 **tmux 세션 `jemo`의 새 window** 안에서 돌아간다 → 접속이 끊겨도 살아 있고,
사용자가 직접 `tmux attach -t jemo` 로 들어가 볼 수 있다.

---

## 1. 서버 레지스트리 (`scripts/servers.conf`)

파이프(`|`) 구분, 한 줄에 한 서버: `alias | repo_path | conda_env | default_gpus | notes`.
모르는 값은 리터럴 `FILL_ME` 로 둔다 (런처가 실행을 거부함). 새 서버/경로 변경 시 이 파일만 고치면 된다.

현재 상태 (2026-06-24):

| alias  | repo_path                                                       | env       | GPU 박스            | 비고 |
|--------|-----------------------------------------------------------------|-----------|---------------------|------|
| gyuri  | **FILL_ME**                                                     | MMSS_SAM  | ?                   | port 100 |
| lecun  | `/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM`| MMSS_SAM  | ?                   | port 300 |
| bengio | `/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-MemorySAM`| MMSS_SAM  | RTX 3090 ×8         | port 400, P9 hardaug6 여기서 학습 |
| levine | `/SSDe/jemo_maeng/src/Project/Drone/drone-MemorySAM`            | MMSS_SAM  | ?                   | port 500, 현재 최선 P9 config가 `levine-` 프리픽스 (경로 `SSDe`, 짧음) |
| yeon   | `/SSDb/jemo_maeng/src/Project/**Drone**/detection/drone-MemorySAM`| MMSS_SAM | ?                  | port 600, 경로가 `Drone24`가 아니라 `Drone` |
| B200   | `/NHNHOME/ailab/Workspaces/jemo_maeng/src/drone-MemorySAM`      | MMSS_SAM  | **B200 180GB ×8** (SHARED) | default_gpus=`FILL_ME`(명시 강제) → **`run B200 <cfg> auto:N`** 권장. 프로세스는 unix user `gm_huis`로 뜸. P28 DELIVER 학습 중(2026-06-24) |
| hinton | (미등록)                                                         | MMSS_SAM  | -                   | port 200 **UNREACHABLE**(timeout) — 복구되면 `ssh-copy-id hinton` 후 등록 |

- 무비밀번호 SSH: gyuri/lecun/bengio/levine/yeon 완료. hinton은 포트 200 미도달.
- `MMSS_SAM` conda env는 각 서버에 실재함 (`~/anaconda3/envs` 또는 `~/miniconda3/envs`).
- repo_path는 `git remote = gist-ailab/drone-MemorySAM` 으로 검증됨.

---

## 2. 런처 (`scripts/remote_exp.sh`)

```bash
scripts/remote_exp.sh run    <server> <config.yaml> [gpus] [nproc] [entry]
scripts/remote_exp.sh log    <server> <config|cfg_name> [follow]
scripts/remote_exp.sh status <server>          # jemo 세션 창 목록 + nvidia-smi
scripts/remote_exp.sh list   <server>          # jemo 세션 창 목록만
scripts/remote_exp.sh servers                  # 레지스트리 출력
```

### run 동작
- 서버에 tmux 세션 `jemo` 없으면 생성, 있으면 재사용.
- config 이름으로 된 **새 window**를 만들고 그 안에서 학습 실행 → 동시에 여러 실험을 창 단위로 격리.
- `gpus` 생략 시 레지스트리 `default_gpus` 사용. `nproc` 생략 시 GPU 개수로 자동.
- `entry=auto`(기본): config 이름에 `SAM3/RBMA` 포함 → `train_sam3_rbma.py`(+ `PYTHONPATH=semseg/models/sam3`, `HF_HUB_OFFLINE=1`), 아니면 `train_sam2_lora_paper.py`.
- 실행 셸에서 `conda activate <env>` → `CUDA_VISIBLE_DEVICES` / `OMP_NUM_THREADS=1` / `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정 → `torchrun --nproc_per_node=<nproc> --master_port=<랜덤> <entry> --cfg <config>`.
- 로그: `logs/<cfg_name>/<cfg_name>_<timestamp>.log` (`tee`). master_port는 21600~21899 랜덤(충돌 회피).

### log 동작
- `logs/<cfg_name>/` 안에서 가장 최신 `.log` 를 골라 `tail -n 80`. `follow` 인자를 주면 `tail -f`(블로킹).

### 예시
```bash
# bengio에서 빈 GPU 4장으로 P9 hardaug6 학습
scripts/remote_exp.sh run bengio configs/bengio-multiaqua_rgbtl_P9_hardaug6.yaml 0,1,2,3
# 진행 로그 확인
scripts/remote_exp.sh log bengio bengio-multiaqua_rgbtl_P9_hardaug6
# 서버 상태(창 + GPU)
scripts/remote_exp.sh status bengio
```

---

## 3. 주의사항 / 트러블슈팅
- **GPU 점유 충돌**: bengio는 8장이지만 다른 작업이 점유 중일 수 있다. `run` 전 항상 `status`로 빈 GPU 확인.
- **config 프리픽스**: config 파일명 앞단(`levine-`, `bengio-`, `b200-`)은 보통 학습 서버를 암시한다. 다른 서버에서 돌릴 땐 데이터 경로가 그 서버에 맞는지 config를 한 번 확인.
- **DDP 실패 'marked ready twice'**: SAM3 trainer는 `static_graph=True` 필요(기록상 해결됨).
- **로그가 안 보임**: 학습이 아직 첫 출력 전이거나 즉시 죽었을 수 있음 → `status`로 해당 window가 살아있는지 확인.
- **hinton**: 포트 200 복구 시 `ssh-copy-id hinton` 후 `servers.conf`의 hinton 줄 주석 해제 + repo_path 입력.
