# MemorySAM: Multimodal Segmentation via SAM2 Memory Attention

## System Instructions

너는 이 프로젝트의 AI 연구 보조 및 엔지니어이다.
항상 세션 간의 문맥(Context)을 유지하기 위해 아래 규칙을 엄격하게 따른다.

### 1. 세션 시작 시 (Initialization)

새로운 대화나 작업 지시를 받으면, 코드 수정을 시작하기 전에 **반드시** 아래 순서대로 `.claude_logs` 폴더 내의 파일들을 읽어라:

#### Step 0 — 역할 판별 (최우선)
- **가장 먼저** `meta/bot-roles.md`를 읽어라.
- 사용자의 첫 메시지에 역할 키워드("코드분석봇", "코딩봇", "실험분석봇", "그림봇")가 포함되어 있으면, 해당 역할의 지침을 이번 세션 전체에 적용한다.
- 역할이 지정되지 않으면 기본 모드(AI 연구 보조 및 엔지니어)로 동작한다.

#### Step 1 — 프로젝트 상태 파악
- **`00_INDEX.md`를 먼저 읽어라** — 주제 폴더 구조(status/models/experiments/det/datasets/research/decisions/infra/issues/meta/archive)의 front door + 구번호("doc N")→새경로 매핑표. 어떤 문서를 볼지 여기서 결정한다. 각 폴더의 `00_MOC.md`가 폴더 내 문서를 안내한다.
- `status/current.md`: **현재 상태 스냅샷 — 현재 상태의 단일 출처**. 전체 진행 상황·현재 최선 모델·남은 과제. (진행 이력은 `status/history-2026H2.md`·`history-2026H1.md`)
- `models/arch-evolution.md`: P8~P31 + SAM3-RBMA 모델 아키텍처 상세, 변천 과정, 각 버전의 한계점
- `experiments/log.md`: 모든 실험 결과, 체크포인트 경로, 챌린지 제출 결과 (한눈표는 `experiments/registry.md`, 실시간 모니터는 `experiments/monitor-log.md`)
- `issues/issues-and-fixes.md`: 알려진 이슈, 해결 기록, 코딩 시 주의사항 — **상단 "이슈 상태 인덱스 표" 먼저** (**코드 작성 전 반드시 확인**)
- `research/novelty-and-related-work.md`: **RBMA 노벨티 & 관련연구(canonical)** — 우리 모델 한눈에, 선행연구 vs RBMA 구조 차별표, 리뷰 방어 포인트, lit-check TODO. **연구 방향·논문 포지셔닝 논의 전 반드시 확인.** (원시 deep-research 로그는 `research/related-work-raw.md`)
- `infra/servers-and-launch.md`: **서버 레지스트리 & 원격 실험 자동 실행** — "X 실험을 <서버>에서 돌려줘" 류 지시를 받으면 **반드시 먼저 읽어라.** 서버 메타데이터 단일 출처는 `scripts/servers.conf`, 실행/추적은 `scripts/remote_exp.sh`.
- `infra/environment.md`: 실행 환경/명령, 데이터·가중치 경로, 체크포인트 포맷, DDP, B200 파이프라인 튜닝.

> `.claude_logs` 진입 순서: **00_INDEX(front door)** → `status/current.md`(현재 스냅샷) → 작업 폴더 `00_MOC.md`. 관련연구/노벨티는 **research/novelty-and-related-work.md**, 원격 학습 지시는 **infra/servers-and-launch.md**, det 작업은 **det/diagnosis-plan.md**, 세션 태스크는 **meta/taskboard.md**를 읽어라. (archive/ = 🗄 동결 문서)

### 1.5 구조 유지 규칙 (Conventions — 파일 생성·코드 추가·브랜치 생성 전 필수)

**`.claude_logs/meta/conventions.md`가 리포 구조 유지의 단일 출처다.** 핵심만 요약하면:
- **Git**: 모든 브랜치는 **`develop` 기준**으로 분기하고, 병합도 PR 없이 `git push origin HEAD:develop` 직접 병합. `main` 금지. 병합 후 로컬 허브 체크아웃 pull 유지. **진행 중 학습이 있는 원격 서버는 pull 금지.**
- **문서**: 새 문서는 `.claude_logs/` 주제 폴더에 kebab-case로 생성하고 해당 폴더 `00_MOC.md`에 등록. 번호 프리픽스 신규 부여 금지.
- **코드**: 새 모델 버전은 `lora_sam/pNN.py` + `MODEL_REGISTRY` 등록 (메가파일·shim에 클래스 추가 금지), 공통 모듈은 `modules/`에, 신규 코드는 shim 경유 import 금지.
- **Configs**: `<dataset>_<modal>_<version>_<aug>.yaml` (서버접두어 금지), 학습=`configs/<dataset>/`, 평가=`configs/eval/`, 신규 실험은 `experiments/registry.md`에 행 추가.

#### ⚠️ 프로젝트 로그 vs 리서치 콘텐츠 — 원본 위치 규칙 (2026-07-09 정합화)

- **프로젝트 로그(상태·아키 evolution·실험 로그/분석·이슈·결정·인프라)의 원본 = repo `.claude_logs/` 주제 폴더** (git 추적). NAS로 이관하거나 심링크로 대체하지 말 것 — 2026-07-08 심링크 이관 시도는 sshfs `.fuse_hidden` 파손 사고 + 원격 서버(NAS 미마운트) dangling으로 **철회**됐다 (구경로들은 리다이렉트 스텁).
- **리서치 콘텐츠(논문 노트·소스·아이디어·볼트 실험노트 `P<N>_<이름>/`)의 원본 = NAS Obsidian 볼트** `/nas_jm/Research/26_MultimodalSeg`. repo의 `.claude_logs/research/vault/`는 **동기화 사본(손편집 금지)** — 갱신은 `bash scripts/sync_research_vault.sh`. 볼트 배치 규약·에이전트 규칙 = `/nas_jm/Research/00_AGENT_PROTOCOL_HERMES.md` + `research/vault/README.md`.
- 볼트에 `architecture/ experiments/ issues/ synthesis/` 등 repo-로그 미러 폴더를 만들지 말 것 (위 사고 잔재 폴더는 격리됨). 위치가 애매하면 사용자에게 묻는다.

### 1.6 🔴 모델 위임 규칙 (모든 세션·에이전트 공통 — user 지정 2026-07-16)

**이 리포에서 작업하는 모든 세션과 서브에이전트에 동일하게 적용한다.**

| 작업 | 어느 모델로 |
|------|------------|
| **학습 기동** (`remote_exp.sh run` / torchrun), **tmux 제어**, 상태 조회(`nvidia-smi`/`ps`/로그 tail·grep), 프로세스 kill, rsync 회수 | **sonnet** |
| **git** (pull/push/fetch/commit), 기계적 파일 이동·동기화·정리 | **sonnet** |
| **코드를 만지는 일** — 패치·config 설계·스크립트 작성 | **해당 세션의 opus 또는 fable** |
| **에러 검증/진단**, 로그 판독, 수치 해석·판정 | **해당 세션의 opus 또는 fable** |

- 위임은 `Agent` tool에 **`model: "sonnet"`** 을 명시해서 한다.
- **sonnet은 데이터를 물어오고 명령을 집행하되, 판정은 상위 모델이 한다.** "이게 붕괴인가 노이즈인가", "왜 죽었나"는 위임하지 마라.
- **위임 금지(판단이 섞인 것)**: cherry-pick 대상 선별, 충돌 해결, 브랜치 전략, 커밋 메시지에 실험 판정을 담는 경우, 실패 원인 규명.
- ⚠️ **기동 "검증"의 기준은 상위 모델이 정의**해 주고 결과를 검토한다. 판정 기준 = **iteration이 실제 전진하는가**(예: `73/187` → 25초 뒤 `92/187`) · **rank0 GPU util > 0인가**(0%면 collective 이탈=데드락) · **메모리가 가중치 수준(3~4GiB)이 아니라 실제 활성화 수준인가** · **첫 eval 통과**. 2026-07-16에 "기동됨"만 보고 살아났다고 오보했다가 실제론 NCCL 데드락(`0/187`에서 13분 정지)이었던 사고가 있다.

**Why**: 반복 잡무·기계적 원격 조작에 상위 모델을 쓰는 건 비용 낭비. 상위 모델은 **판단·진단·코드**에만 쓴다.

### 1.7 🔴 코드 단일출처 규칙 (모든 세션·에이전트 공통 — user 지정 2026-07-17)

**멀티 세션이 중복 구현하지 않도록, 모든 코드는 운용(학습/평가 기동) 전에 반드시:**

1. **`develop` 브랜치에 병합**돼 있어야 한다. feature 브랜치·worktree·서버 로컬에만 있는 코드로 학습을 돌리지 마라. (모델 코드·config·스크립트 전부.)
2. **로컬 허브(`jemo@172.27.183.150` = 이 박스, `.../drone-MemorySAM`)에서 접근 가능**해야 한다. 원격 서버들은 GitHub이 아니라 **이 허브를 `local` remote로 pull**한다(jarvis 등 확인됨). 즉 `develop`에 push + 허브가 그 커밋을 보유해야 다른 세션·서버가 받을 수 있다.

**절차 (새 모델/코드를 서버에서 돌리기 전)**:
- 코드 작성 → **`develop`에 직접 병합**(`git push origin HEAD:develop`, PR 없음 — [[git-direct-merge-develop]]) → **로컬 허브 pull로 최신화** → 서버가 `git fetch local && git checkout/merge develop`.
- config도 코드다. 서버 전용 튜닝(경로·GPU·batch)이라도 **develop에 커밋**해 다른 세션이 볼 수 있게 하라. 서버 로컬에만 둔 미커밋 config는 그 세션이 죽으면 소실된다(2026-07-16 bengio HW 사망으로 P37 미커밋 config가 서버에 갇힌 사례).

**왜**: 세션 A가 만든 모델을 세션 B가 모르면 재구현한다. develop+허브가 유일한 "다른 세션이 볼 수 있는 곳"이다. 서버 로컬 브랜치·worktree는 **그 세션만의 것**이다.

⚠️ **P37 병합 대기 (2026-07-17, user 결정)**: P37a-CEFR/P37b-ClassToken 코드는 `worktree-p33-impl`(9c5e2cc)에만 있고 develop엔 없다. **통짜 머지 금지** — develop은 이미 P34~P36 + 분석훅(P36 router-off 토글 등, p33-impl엔 없음)을 갖고 있어 reliadino/*.py가 양쪽 독립 진화(충돌). P37 순증분(classtoken.py +135 / fusion.py CEFRHead +153 / model.py CEFR +129 / train_reliadino +66 / P37 configs / ColorAugSSD)만 얹어야 함. **jarvis P37a 완주·검증 후 opus가 수동 이식+재검증하여 병합 예정.** 그 전엔 인계 시 `worktree-p33-impl` 브랜치를 직접 참조. 성급히 머지하지 말 것.

### 2. 실험 및 코드 변경 시 (Execution)

- 모델 아키텍처를 수정하거나 실험 Config를 생성하면, 작업 후 반드시 `models/arch-evolution.md` 또는 `experiments/log.md`를 업데이트하여 기록을 남겨라 (새 실험 launch/상태 변화는 `experiments/registry.md` 행도 갱신).
- 버전(P8, P9, P10 등)을 명시하고, 왜 변경했는지(이전 실험 결과 기반) 타당한 이유를 적어라.
- 실험 결과 파일 경로는 프로젝트 기준 상대 경로로 기록해라.
- 새 선행연구를 조사했거나 RBMA 노벨티/차별점 논의가 갱신되면 `research/novelty-and-related-work.md`(canonical 비교표·판정)를 업데이트하고, 원시 조사 로그는 `research/related-work-raw.md`에 추가해라.

### 3. 구현/작업 완료 시 자동 업데이트 (Auto-update)

- 새 모델 버전 구현, config 생성, 학습/평가 스크립트 수정 등 **의미 있는 작업이 완료되면** 사용자 요청 없이도 자동으로 `.claude_logs/status/current.md`(스냅샷 덮어쓰기)를 업데이트하고, 진행 이력은 `.claude_logs/status/history-2026H2.md` 최상단에 append해라.
  - 상태 변경 (예: "설계 완료 (구현 대기)" → "구현 완료 (학습 대기)")
  - 변경 파일 목록 및 핵심 내용 기록
  - 디자인 가이드 대비 의도적 차이가 있으면 사유 기록
- 모델 아키텍처 변경이 있었으면 `models/arch-evolution.md`도 함께 업데이트해라.

### 4. 세션 종료 시 (Wrap-up)

- 사용자가 "작업 끝", "기록해줘" 등의 말을 하면, 이번 세션에서 변경된 사항을 `.claude_logs/` 내 파일들에 요약 추가해라.

---

## 프로젝트 개요

**목표**: MACVi MULTIAQUA Challenge — 드론 촬영 야간 수상 환경에서 RGB + LiDAR + Thermal 멀티모달 세그멘테이션

**핵심 아이디어**: SAM2의 시간축 메모리 어텐션을 모달리티 축으로 전용하여, 멀티모달 Cross-Modal Fusion을 수행. 각 모달리티를 별도 "프레임"으로 인코딩 후, SAM2의 memory attention으로 상호 참조.

**데이터셋**: MULTIAQUA
- 클래스: Static(0), Dynamic(1), Water(2), Sky(3), ignore(255)
- Val = 주간 145장, Test = 야간만 (challenge server 평가)
- 모달리티: RGB (`img`), LiDAR (`lidar`), Thermal (`thermal`)
- 경로: `/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night`

**평가 지표**: M-score = 0.75 × val_mIoU + 0.25 × test_mIoU (MACVi Challenge)

---

## 환경 설정

```bash
# Conda 환경
conda activate MMSS_SAM
# 또는 직접 경로: /home/jemo/anaconda3/envs/MMSS_SAM/bin/python

# 학습
python train_sam2_lora_paper.py --cfg configs/<config>.yaml

# 평가 (val)
python val_multiaqua.py --cfg configs/eval/<config>.yaml --mode val --model_path <checkpoint_path>

# 평가 (test + challenge 제출)
python val_multiaqua.py --cfg configs/eval/<config>.yaml --mode test --model_path <checkpoint_path> --macvi

# P9 전용 시각화 평가 (MoE routing 분석 포함)
python val_multiaqua_P9.py --cfg configs/eval/levine-multiaqua_rgbtl_P9_hardaug4.yaml --mode val
python val_multiaqua_P9.py --cfg configs/eval/levine-multiaqua_rgbtl_P9_hardaug4.yaml --mode test
```

### 원격 서버에서 실험 실행 (tmux 세션 `jemo`)

"X 실험을 <서버>에서 돌려줘" → 아래 런처 사용. 상세는 `.claude_logs/infra/servers-and-launch.md`, 서버 목록은 `scripts/servers.conf`.

```bash
# 서버 레지스트리 확인 (repo_path / env / default_gpus)
bash scripts/remote_exp.sh servers
# 서버 상태(빈 GPU + jemo 세션 창)
bash scripts/remote_exp.sh status bengio
# 실행: ssh -> tmux 세션 'jemo' 새 window -> torchrun -> logs/<cfg>/<cfg>_<ts>.log
bash scripts/remote_exp.sh run bengio configs/multiaqua/bengio-multiaqua_rgbtl_P9_hardaug6.yaml 0,1,2,3
# 진행 로그 추적
bash scripts/remote_exp.sh log bengio bengio-multiaqua_rgbtl_P9_hardaug6
```

### 📊 평가/분석 산출물 저장 위치 (모든 세션 공유)

**🔴 웨이트·로그·분석 산출물의 단일 정규 루트 (2026-07-17 재확정, 모든 세션):**
`/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/` 하위 — `ckpts/`(웨이트 .pth, `<run>_<YYYYMMDD>/`) · `analysis_logs/`(eval·분석·시각화, `<model>_eval_<YYYYMMDD>/` = report/+viz/+perdomain/) · `train_logs/`(학습 런 로그). **모든 세션(P38 등 신규 포함)은 새 학습/평가/분석 산출을 여기 저장**하고 새 전략 전 이 루트를 먼저 확인한다. 원격(hpca100/jarvis/yeon)은 `rsync`로 회수·누적. 경로 변천: `/mnt/HDD2/src/logs/`(ISSUE-023) → `/drone_nas/drone/analysis_logs/`(flat) → 위 nested(07-17). 단일 출처 = 메모리 `eval-logs-stats-location`.
- 재사용 도구(repo `tools/`): `eval_per_domain.py`(per-condition 러너) · `analyze_per_domain.py`(per-class 분류) · `viz_features.py`(feature/RBMA 패널) · `module_diagnostics.py`(모듈 정량). 모델 무관(`--cfg`/`--model_path`만 교체).
- lecun/yeon에서 SAM2 코드 실행 시 `sam2` editable 미설치면 `PYTHONPATH=<repo>/semseg/models/sam2` 지정.

---

## 핵심 코드 구조

```
drone-MemorySAM/
├── CLAUDE.md                          # 이 파일
├── .claude_logs/                      # AI 세션 로그 (front door = 00_INDEX.md)
│   ├── 00_INDEX.md                    # 폴더 구조 안내 + 구번호→새경로 매핑표
│   ├── status/                        # current.md(현재 스냅샷) + history-2026H1/H2
│   ├── models/                        # arch-evolution.md, figures-ascii.md, explain/
│   ├── experiments/                   # registry.md, log.md, monitor-log.md, analysis/
│   ├── det/  datasets/  research/     # det 진단 · 데이터셋 · 관련연구(vault 포함)
│   ├── decisions/  infra/  issues/    # 설계 제안 · 서버/환경 · 이슈
│   └── meta/  archive/                # 봇 역할·태스크보드 · 동결 문서
├── train_sam2_lora_paper.py           # 메인 학습 스크립트
├── val_multiaqua.py                   # 범용 평가 스크립트 (P8~P12)
├── val_multiaqua_P9.py                # P9 전용 시각화 + MoE routing 분석
├── diagnose_moe_gate.py               # MoE gate 진단 스크립트
├── configs/
│   ├── deliver/ · multiaqua/ · det/   # 학습 configs (분류 기준: configs/README.md)
│   └── eval/                          # 평가 configs (MODEL_PATH 포함, 구 eval_config/)
├── semseg/
│   └── models/sam2/sam2/
│       ├── sam_lora_image_encoder_seg.py  # LoRA_Sam_P8~P12 모델 정의
│       ├── sam_lola_utils.py              # SoftMoE_LoRA_Layer 등 유틸리티
│       └── checkpoints/
│           └── sam2.1_hiera_base_plus.pt  # SAM2 pretrained weight
└── outputs/
    ├── MMSamP8/   # P8 실험 결과들
    ├── MMSamP9/   # P9 실험 결과 (현재 최선)
    ├── MMSamP10/  # P10 실험 결과 (취소됨)
    └── MMSamP11/  # P11 실험 결과 (취소됨)
```

---

## 모델 버전 요약

| 버전 | 핵심 변경 | 최선 M-score | 상태 |
|------|----------|-------------|------|
| P8 | ConfidenceHeadV2 + sigmoid UAMM | 78.45 | hardaug 기반실험 완료 |
| **P9** | CrossModalFusionHead + max-norm UAMM | **81.98** (hardaug8 ep131) | **현재 최선** |
| P10 | CrossModalFusionHeadV2 + ModalAuxHead + oracle KL | 79.27 | 취소 (test 성능 하락) |
| P11 | P10 + MI routing loss | 77.09 | 취소 (MoE gate 진단 우선) |
| P12 | Input-Conditioned Soft MoE LoRA (cond_dim) | - | 설계만 완료 |
| P24 | P9 + SpatialQualityGating (scalar UAMM/AMF + CE teacher) | - | 학습 중 |
| P25 | Unified Spatial Quality Fusion (spatial UAMM/AMF, no CrossModalFusionHead) | - | 구현 완료 (학습 대기) |

**현재 최선 모델: P9 hardaug8_physaug ep131** — `outputs/MMSamP9/levine_multiaqua_rgbtl_P9_hardaug8_physaug/MULTIAQUA_CMNeXt-B2_ilt/epoch131_94.41_top1_checkpoint.pth`

---

## 주의사항

1. **Checkpoint 포맷 차이**: `.pth` = raw state_dict, `_checkpoint.pth` = `{'model_state_dict': ..., 'optimizer_state_dict': ..., ...}` 형태. `val_multiaqua.py`는 `_checkpoint.pth`를 기대하고, `val_multiaqua_P9.py`는 `.pth`를 직접 로드.
2. **Val vs Test 갭**: Val mIoU ~93-94% (주간) vs Test mIoU 58-70% (야간). 모든 모델이 이 갭을 보임.
3. **MoE Gate "Uniform" 문제**: 공간 평균(`_gate_callback`) 결과 uniform으로 보이지만, per-token 분석 시 실제로는 분화되어 있음 (entropy_ratio=0.55, max_weight=0.72). 측정 artifact임.
4. **NIGHT_AUG**: 야간 시뮬레이션 증강. hardaug4가 최종 튜닝 버전. `BRIGHTNESS_SAMPLING: dark_biased`로 극저조도 편향.
5. **DDP 학습**: `TRAIN.DDP: True`로 멀티GPU 학습. 단일 GPU 시 `train_sam2_lora_paper_singlegpu.py` 사용.
6. **🔴 GPU 가용성 확인 (모든 학습 실행 전 필수)**: 어떤 실험이든 돌리기 **전에 반드시 해당 서버의 빈 GPU를 확인하고, 비어 있는 GPU에만** 배치한다(사용 중 GPU에 얹지 않는다 → OOM/타인 작업 방해).
   - **로컬 런처**(`run_sam.sh` / `run_sam3_train.sh` / `run_sam3_rbma.sh`): `CUDA_VISIBLE_DEVICES`를 직접 주지 않으면 **`scripts/pick_free_gpus.sh`로 빈 GPU를 자동 선택**한다. 개수는 `NGPU=` (SAM2/3 train) 또는 `NPROC=` (rbma)로 지정. 빈 GPU가 부족하면 실행을 거부한다.
     - 예: `NGPU=4 bash run_sam.sh` · `NGPU=1 bash run_sam3_train.sh` · `CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash run_sam3_rbma.sh <cfg>`(직접 지정은 그대로 존중).
   - **원격 런처**(`scripts/remote_exp.sh`): 먼저 `status <server>`로 확인하고, `run <server> <cfg> auto:N`으로 **원격의 빈 GPU N장을 자동 배정**한다(`auto`=1장). 빈 GPU가 없으면 거부.
   - 판정 기준: GPU가 `memory.used ≤ 2000MiB && util ≤ 10%`이면 "빈 GPU"(환경변수 `GPU_MAXMEM`/`GPU_MAXUTIL`로 조정). 헬퍼/`auto`는 메모리 적은 순으로 고른다.
