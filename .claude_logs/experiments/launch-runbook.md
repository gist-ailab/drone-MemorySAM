# 학습 런치 런북 — 서버별 레시피 & 함정 (2026-07-19 작성)

새 실험(P39 등)을 **즉시 기동**하기 위한 단일 참조. 여기 적힌 함정들은 전부 실제로 한 번씩 데인 것들이다.

## 🔴 전 서버 공통 불변식 (어기면 조용히 망함)

| 규칙 | 이유 |
|---|---|
| **`GRADIENT_CHECKPOINT: false` 항상** | `encoder.py`가 공유 mutable `active_modality`를 쓰는데 checkpoint 재실행이 **마지막 모달 값**으로 backward → **마지막 외 전 모달이 노이즈로 학습**. forward loss는 bit-identical이라 **로그로 안 보임**. 24GB에서 메모리 부족하면 ckpt 켜지 말고 **BATCH_SIZE로 조절**. |
| **YAML LR은 `0.0001` (지수표기 금지)** | PyYAML이 `1e-4`를 **str**로 파싱 → `base_lr * mult`에서 TypeError. 값 말고 표기 문제. |
| **코드는 운용 전 develop에 있어야** | 단일출처 규칙. det 계보는 `worktree-p38-det`(RF-DETR/CEFR/classtoken/M2F-det 통합본). |
| **기동 검증 4종** | ①rank0 util>0 + 메모리가 가중치 수준 이상 ②로그에 iteration **실제 전진**(loss 유한) ③Traceback/CUDA/OOM 없음 ④cfg가 의도한 것으로 로드. ※"프로세스 살아있음"만으로 판정 금지(NCCL 데드락 전례). |
| eff-batch | **seg**: 트레이너가 `accum=ceil(16/(BS×world))`로 자동보정 → GPU 수 바뀌어도 대략 유지. **det**: config의 `GRAD_ACCUM_STEPS` 고정 → **GPU 수가 eff-batch를 바꾼다**(비교 실험은 GPU 수 고정할 것). |

## hpca100 — A100 40GB × 4 (대용량, 1024²·BS2 가능)

- 접속: `ssh hpca100`. **불통이면 MTU 호스트라우트 소실** → `sudo ip route add 210.125.69.5 via 172.27.183.254 dev enp6s0 mtu 1200` (user 상시 위임됨).
- repo: `/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM` · venv: `source /home/jovyan/SSDb/jemo_maeng/venv/p34/bin/activate` (🔴 공유 `~/.venv`에 pip install 금지)
- 필수 env: `WANDB_MODE=disabled`(wandb sentry 도달불가 → rank0 futex 블록 → **NCCL 데드락**), `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 🔴 **cuDNN — 최초 기동뿐 아니라 "재기동·resume 때도 반드시"** (2026-07-21 실사고: 이 줄을 알고도 resume 명령에서 빠뜨려 ep112 첫 backward에서 크래시, GPU 장시간 유휴):
  ```bash
  export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
  ```
  시스템 `/usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8`(8.9.0)이 venv의 8.9.2.26을 가려 `undefined symbol` → `RuntimeError: GET was unable to find an engine to execute this computation`. **`~/.bashrc`가 시스템 cudnn을 앞세우므로 tmux 새 shell마다 재발한다.**
  **검증법**: 같은 환경에서 `python -c "import torch; print(torch.backends.cudnn.version())"` → **8902**여야 함(8900이면 처방 미적용).
- 데이터: MUSES 처리본(`dset/MUSES`, projected_to_rgb 포함), DELIVER.

## jarvis — RTX 24GB × 8

- repo: `/home/jemo_maeng/src/drone-MemorySAM-develop` (develop = P38 seg 코드 포함). 구 `drone-MemorySAM`은 `p37a-jarvis` 브랜치라 m2f 없음.
- 🔴 **`PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:.` 필수** — 없으면 `could not build backbone 'vit_large_patch16_dinov3'`(mainline timm에 DINOv3 미등록). **상대경로 금지, 절대경로로.**
- python: `/home/jemo_maeng/miniconda3/envs/MMSS_SAM/bin/torchrun`
- 메모리: **1024²는 BS1**(BS2는 22.5GB에서 OOM). 768²면 BS2 가능.
- 데이터: DELIVER(`/SSDb/jemo_maeng/dset/DELIVER`), **MUSES 처리본(`/SSDb/jemo_maeng/dset/MUSES`, 2026-07-19 회수, 8.14GB, radar 제외)**.

## yeon — RTX 24GB × 8 (detection 박스)

- 워크트리: `drone-MemorySAM-p37b`(P37b-det), `drone-MemorySAM-p38`(worktree-p38-det = det 계보 통합). conda env **openmmlab**.
- 🔴 **`DET_GRAD_CLIP=0.1` env 필수** — `train_det.py`가 env 기본값 10.0으로 **config의 `GRAD_CLIP`을 덮어씀**. 안 주면 사실상 무clip(붕괴 전례).
- 데이터: `poongsan_v2` (`_final_ann/instances_train_egofill.json` / `instances_test_common.json`).
- 순차 체인 관례: tmux `jemo`에 wrapper window로 `while kill -0 <PID>; do sleep 300; done; <launch>` (세션 독립). ⚠️ wrapper는 실행 시점에 메모리로 파싱되므로 **파일만 고쳐도 반영 안 됨 → 재무장 필요**.

## 산출물 저장 (필수)

완주/평가 산출은 전부 **DRONE-NAS** `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/{ckpts,analysis_logs,train_logs}/`로 회수(로컬 접근 가능해야 함). 단일 출처 = 메모리 `eval-logs-stats-location`.
