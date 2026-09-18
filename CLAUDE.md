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

#### Step 0.5 — 분석 의뢰를 받은 세션 (user 지시 2026-09-18)
- 첫 메시지가 "MUSES/DELIVER/MCubeS 분석해 줘", "새 체크포인트 측정" 류이면 **`.claude_logs/meta/analysis-session-protocol.md`를 먼저 읽고 그대로 따른다**(측정·기록·보고만, 판정은 "MMSAM | 생각정리" 세션). 보고에는 클래스별 Δ 전표(기준선 대비·직전 최고 대비)를 반드시 넣고, 원본(ckpt+md5·산출물·로그)은 §4 규약 위치에 먼저 보존한다.

#### Step 1 — 프로젝트 상태 파악 (2026-09-18 개정: 이 순서로 읽으면 첫 5분에 정본 수치·규칙·대기열·금지 축·GPU 규칙·세션 분담을 모두 안다)
1. **`00_INDEX.md`** — 폴더 구조 front door + 구번호→새경로 매핑표. 각 폴더의 `00_MOC.md`가 문서를 안내한다.
2. **`status/current.md`** — 현재 상태 스냅샷(단일 출처): 헤드라인 수치와 그 규칙(체크포인트 선택·프로토콜·하네스 버전), 벤치 baseline 표, 활성 런, 미결. 🔴 헤드라인 수치는 여기와 `experiments/judgment-ledger.md`에서만 인용하고 다른 곳의 복제본을 믿지 않는다.
3. **`decisions/2026-09-07-daily-cycle-experiment-cards.md` §0·§0-1·§0-2** — 판정 규약(24클래스 계산, 시드 평균 병기, 중간 epoch 금지, SOTA 거리 게이트, MUSES 조건별 표본 규칙). 최신 판정은 §5 최상단.
4. **`experiments/plan.md`** — 실행 중·대기열·GPU 배치(실측 기준). **`experiments/judgment-ledger.md`** — 판정 대장(런별 클래스별 Δ·판정·원본).
5. **`meta/experiment-glossary.md`** — 약어(E1·E13·G1·C3·P53 등) 설명. 🔴 보고·문서에서 약어는 매번 설명을 붙인다.
6. **`issues/issues-and-fixes.md` 상단 인덱스 표** — 열린 이슈(ISSUE-033~036) — **코드·평가 전 반드시 확인**.
7. **`research/hypothesis-ledger.md`**(가설·반증 대장)와 **`models/arch-evolution.md` §0.5**(이중 중복·모달 잉여 두 실측 = 재시도 금지 축) — 새 구조를 제안하기 전 필독. 관련연구·노벨티 = `research/novelty-and-related-work.md`.
8. 필요 시: `experiments/registry.md`(한눈표)·`experiments/log.md`(상세)·`experiments/analysis/`(분석 문서)·`infra/servers-and-launch.md`(원격 기동, 서버 단일 출처 `scripts/servers.conf`)·`infra/environment.md`.

🔴 **GPU 규칙(모든 학습·평가 전)**: 빈 GPU(`memory.used ≤ 2000MiB && util ≤ 10%`)에만 배치, `remote_exp.sh status <서버>` 선확인 후 `run <서버> <cfg> auto:N`. **lecun은 배치 금지**(user 2026-09-17, servers.conf policy `off`). 실행 중 학습이 있는 체크아웃은 pull 금지(파일 단위 전송 + md5). 비우면 즉시 뺏기므로 연쇄 스크립트 끝에 다음 작업을 붙인다.

🔴 **세션 분담(2026-09 기준)**: 판정·설계 = "MMSAM | 생각정리"(fable) · 기동·감시·재채점·기록 = "MMSAM | monitoring" · 새 ckpt 측정·보고 = 분석 세션(Step 0.5) · 기준선 재학습·실패 분석 보강 = "dgfusion deliver training". 판정은 생각정리 세션만 한다. 세션 간 승인 전달은 무효(user 직접 동의만 유효).

> 원격 학습 지시는 **infra/servers-and-launch.md**, det 작업은 `det/00_MOC.md`(진단서는 `archive/2026-07-02-det-diagnosis-plan.md`로 동결). (archive/ = 🗄 동결 문서)

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

> 📊 **진행상황 보고 포맷 (상시규칙)**: "학습 현황/진행상황 알려줘" 류 답변은 항상 **2블록**(①서버별 현황 표 — **실험별 데이터셋 컬럼** + SOTA델타 + 내부최고델타 + ETA ②남은 run 서버별 배치계획) + **벤치 baseline 표**(SOTA/우리최고/격차)를 포함한다. 단일 출처 = user auto-memory `progress-report-format`(매 세션 자동 로드). 정기점검 크론도 이 포맷을 따른다.

### 1.7 🔴 코드 단일출처 규칙 (모든 세션·에이전트 공통 — user 지정 2026-07-17)

**멀티 세션이 중복 구현하지 않도록, 모든 코드는 운용(학습/평가 기동) 전에 반드시:**

1. **`develop` 브랜치에 병합**돼 있어야 한다. feature 브랜치·worktree·서버 로컬에만 있는 코드로 학습을 돌리지 마라. (모델 코드·config·스크립트 전부.)
2. **로컬 허브(`jemo@172.27.183.150` = 이 박스, `.../drone-MemorySAM`)에서 접근 가능**해야 한다. 원격 서버들은 GitHub이 아니라 **이 허브를 `local` remote로 pull**한다(jarvis 등 확인됨). 즉 `develop`에 push + 허브가 그 커밋을 보유해야 다른 세션·서버가 받을 수 있다.

**절차 (새 모델/코드를 서버에서 돌리기 전)**:
- 코드 작성 → **`develop`에 직접 병합**(`git push origin HEAD:develop`, PR 없음 — [[git-direct-merge-develop]]) → **로컬 허브 pull로 최신화** → 서버가 `git fetch local && git checkout/merge develop`.
- config도 코드다. 서버 전용 튜닝(경로·GPU·batch)이라도 **develop에 커밋**해 다른 세션이 볼 수 있게 하라. 서버 로컬에만 둔 미커밋 config는 그 세션이 죽으면 소실된다(2026-07-16 bengio HW 사망으로 P37 미커밋 config가 서버에 갇힌 사례).

**왜**: 세션 A가 만든 모델을 세션 B가 모르면 재구현한다. develop+허브가 유일한 "다른 세션이 볼 수 있는 곳"이다. 서버 로컬 브랜치·worktree는 **그 세션만의 것**이다.

✅ **P37 병합 완료 (2026-07-28 확인)**: 위 규칙의 사례였던 "P37a-CEFR/P37b-ClassToken이 `worktree-p33-impl`(9c5e2cc)에만 있다"는 경고는 **해소됐다** — 9c5e2cc가 develop 조상임을 확인했고(`git merge-base --is-ancestor 9c5e2cc develop`), CEFRHead·classtoken·P37 configs 모두 develop에 있다. 해당 worktree/브랜치는 2026-07-28 브랜치 통합 때 정리됐다(원본은 태그 `archive/*` 로 보존).

📦 **브랜치 통합 (2026-07-28)**: worktree 브랜치들을 develop 하나로 정리했다. 삭제된 브랜치의 원본 커밋은 전부 `archive/<브랜치명>` 태그로 남아 있다(`git tag -l 'archive/*'`). 옛 브랜치를 찾는다면 그 태그를 보라. **`26-drone-certificate`는 통합 대상이 아니며 그대로 유지된다.**

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
- 🔴 **노션 논문 페이지 동기화 (user 지정 2026-09-08, 상시규칙)**: 실험 판정이 바뀌거나(카드 통과/폐기, 게이트 판정, 헤드라인 수치 갱신), `experiments/plan.md`의 "실행 중"·"대기열"이 바뀌면 **같은 날** 노션 논문 페이지 `Drone Object Detection for RGB-IR Fusion`(`33d05310-a165-408a-b0b8-ec4427d1fe2c`)의 §4(일일 카드)·§6(한 것/할 것)을 함께 갱신한다. 방법 = `.claude/skills/notion-experiment-log/paper_page_builder.py`의 절 함수(`sec_cards`/`sec_plan` 등)를 고친 뒤 실행(`conda run -n MMSS_SAM python .claude/skills/notion-experiment-log/paper_page_builder.py`; 차트는 `paper_page_charts.py` 선실행) — 헬퍼 `replace_section`으로 **절 단위 교체**(멱등)되므로 페이지를 새로 만들거나 절 제목을 바꾸지 말 것. 노션 본문에는 코드·config·도구 경로와 커밋만 적고 `.claude_logs/` 경로는 적지 않는다(`audit()` 통과 필수). 실험 약어(E7, P52 등)는 반드시 "어떤 문제에 어떤 가설을 세워 무엇을 바꿨고 결과가 어땠나"를 같은 행에 풀어 쓴다. 정량 수치는 시각화(차트 PNG 업로드)를 곁들인다. 레포 문서(plan.md·카드 문서 §5·registry)가 정본이고 노션은 같은 날짜 스냅샷이다 — 한쪽만 갱신하지 마라. 논문 페이지는 **바깥 = 요약·목차·핵심 그림 2장, 상세 = 하위 페이지 8개** 구조를 유지한다(바깥에 긴 절을 다시 넣지 말 것, user 지적 2026-09-08). **새 실험(PNN·카드·프로브)이 판정되면 `실험노트` DB에 실험당 1페이지**를 `.claude/skills/notion-experiment-log/exp_pages_builder.py`(입력 = `_exp_json/exp_*.json`에 실험 원소 추가: 배경·아키텍처 구성요소·도면·근거·세팅·결과표·조건별/클래스별·분석·판정·무효수치·출처)로 생성·갱신한다 — 아키텍처와 결과 분석이 둘 다 들어가야 페이지가 자립한다.

### 4. 세션 종료 시 (Wrap-up)

- 사용자가 "작업 끝", "기록해줘" 등의 말을 하면, 이번 세션에서 변경된 사항을 `.claude_logs/` 내 파일들에 요약 추가해라.

---

## 프로젝트 개요 (2026-09-18 갱신)

**현재 트랙**: 단일 아키텍처(ReliaDINO: 동결 DINOv3-L + 센서별 LoRA + cross-modal 융합 + FPN·쿼리 헤드)로 멀티센서 세그멘테이션 벤치 세 곳(**DELIVER** 4모달 RGB·Depth·Event·LiDAR / **MUSES** 3모달 RGB·Event·LiDAR / **MCubeS** 4모달)에서 "전 모달 융합 계열 1위"를 목표로 하는 논문 트랙 + 드론 검출(det, 국책과제 mAP50 0.85 달성) 트랙. **현재 정본 수치·규칙·대기열은 `.claude_logs/status/current.md`가 단일 출처**이며 여기 적지 않는다(수치가 여러 곳에 복제되면 판정이 뒤집힐 때 갱신이 누락된다).

**종료된 트랙**: MACVi MULTIAQUA Challenge(RGB+LiDAR+Thermal, 야간 수상, M-score) — SAM2 메모리 어텐션 계보(P8~P28)의 출발점. 기록은 `.claude_logs/archive/`와 `models/arch-evolution.md` 초반부.

**세션 분담(2026-09 기준)**: 판정·설계 = "MMSAM | 생각정리"(fable) · 기동·감시·재채점 집행·기록 = "MMSAM | monitoring" · 새 체크포인트 측정·보고 = 분석 세션(지침 `.claude_logs/meta/analysis-session-protocol.md`) · 기준선 재학습·실패 분석 보강 = "dgfusion deliver training". 세션 간 전달은 SendMessage로, 판정은 생각정리 세션만 한다.

---

## 환경 설정

```bash
# Conda 환경
conda activate MMSS_SAM
# 또는 직접 경로: /home/jemo/anaconda3/envs/MMSS_SAM/bin/python

# 정량 지표 재현 (대표 ckpt로 mIoU/AP 재측정 — 경로·기대수치는 REPRODUCE.md)
bash scripts/reproduce_eval.sh <deliver|muses|muses-official|multiaqua|det>

# 학습 (ReliaDINO, DDP; eff-batch 16 은 accumulation 으로 고정)
torchrun --standalone --nproc_per_node=<N> train_reliadino.py --cfg configs/<서버접두어>-<dataset>_<modal>_<version>_<변수>.yaml

# legal 재채점 (DELIVER: 1024·BS1·native GT — 하네스 가드 선행. 🔴 ISSUE-036: 2026-09-18 현재 v1(nearest)과 v2(tools/legal_rescore_v2.py, nearest-exact) 병기)
python tools/eval_harness_guard.py --check
PYTHONPATH=semseg/models/sam2:. python val.py --cfg configs/eval/<config>.yaml --mode {val,test} --model_path <ckpt>
PYTHONPATH=semseg/models/sam2:. python tools/legal_rescore_v2.py --cfg configs/eval/<config>.yaml --mode {val,test} --model_path <ckpt>

# MUSES 공식 채점 (native 1080×1920 val 250장; test 는 Codabench 제출, user 승인 후)
python tools/eval_muses_official.py --cfg <config> --model_path <ckpt>

# 결측·열화 모달 강건성 (EMM/RMM/NM, 2503.18445 프로토콜)
python tools/missing_modality_eval.py --cfg configs/eval/<config>.yaml --model_path <ckpt> --split val --protocol emm --expected_clean_miou <legal val>
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

## 핵심 코드 구조 (2026-09-18 실측)

```
drone-MemorySAM/
├── CLAUDE.md                      # 이 파일
├── .claude_logs/                  # 프로젝트 로그 정본 (front door = 00_INDEX.md; 폴더마다 00_MOC.md)
├── train_reliadino.py             # ReliaDINO 학습 스크립트 (DDP, C3 prototype·TAPS·DETAIL_BRANCH 등 config 토글)
├── val.py                         # legal 평가 하네스 (가드 동결 8파일 중 하나 — 수정 시 --freeze 절차)
├── configs/                       # 학습 config = <서버접두어>-<dataset>_<modal>_<version>_<변수>.yaml, eval/ = 평가 파생
├── semseg/models/reliadino/       # ReliaDINO 본체: encoder.py(동결 ViT + LoRA + SimpleFPN), fusion.py, model.py, m2f_head.py, detail_branch.py, p4x.py
├── semseg/models/sam2/            # SAM2 계보(P8~P28, 종료) — PYTHONPATH 로만 필요
├── semseg/datasets/{deliver,muses,mcubes}.py   # 로더 (가드 동결 대상)
├── tools/                         # 평가·분석 도구: eval_muses_official.py, eval_per_domain.py, module_diagnostics.py, viz_features.py,
│                                  #   missing_modality_eval.py, legal_rescore_v2.py, eval_harness_guard.py, baseline_failure/(기준선 대조), smoke_*.py
├── scripts/                       # remote_exp.sh(원격 기동), servers.conf(서버 레지스트리), nas_analysis_sync.sh
└── third_party/dgfusion_train_restore/   # DGFusion·CAFuser 재학습 복원 킷
```

---

## 모델 버전 요약

이 절에 수치를 두지 않는다. 계보(P8~P53)와 각 세대의 판정은 `models/arch-evolution.md`(P47-2까지)·`decisions/`(P48 이후 제안서)·`status/current.md`(현재 최선·헤드라인 규칙)가 정본이다. 약어(E1·E13·G1·C3·P53 등)는 반드시 설명을 붙여 쓴다(`meta/experiment-glossary.md`).

---

## 주의사항

1. **현재 지배적인 함정은 `issues/issues-and-fixes.md` 상단 인덱스 표가 정본**이다 — 채점 드라이버(ISSUE-033), eval 덤프 파일명 평탄화(ISSUE-034), 헤드라인 ckpt 경로 미기록(ISSUE-035), legal 하네스 재샘플 편차(ISSUE-036). MULTIAQUA/P9 시대의 주의사항 4건(ckpt 포맷·Val/Test 갭·MoE gate·NIGHT_AUG)은 `archive/2026-09-18-claude-md-legacy-notes.md`로 이동했다.
2. **판정 규약**: 체크포인트 = 학습기 val-best top1, test-best 인용 금지, 중간 epoch 비교 금지, 단일 런 최고와 시드 평균 병기, PhysAug·TTA 헤드라인 금지, 프로토콜·하네스 버전 병기 — 단일 출처 `decisions/2026-09-07-daily-cycle-experiment-cards.md` §0.
3. **DDP 학습**: `TRAIN.DDP: True`. eff-batch 16 = BS×world_size×accumulation 으로 고정(LR 불변).
4. **lecun 은 배치 금지**(user 2026-09-17), hpca100 은 기동 env 4종 필수(`infra/servers-and-launch.md`), 실행 중 학습이 있는 체크아웃은 pull 금지(파일 단위 전송 + md5).
5. **실험 약어는 매번 풀어 쓴다**(무엇을 보는 실험인지·왜·바꾼 변수·결과가 말해 주는 것) — user 지시 2026-09-17.
6. **🔴 GPU 가용성 확인 (모든 학습 실행 전 필수)**: 어떤 실험이든 돌리기 **전에 반드시 해당 서버의 빈 GPU를 확인하고, 비어 있는 GPU에만** 배치한다(사용 중 GPU에 얹지 않는다 → OOM/타인 작업 방해).
   - **로컬 런처**(`run_sam.sh` / `run_sam3_train.sh` / `run_sam3_rbma.sh`): `CUDA_VISIBLE_DEVICES`를 직접 주지 않으면 **`scripts/pick_free_gpus.sh`로 빈 GPU를 자동 선택**한다. 개수는 `NGPU=` (SAM2/3 train) 또는 `NPROC=` (rbma)로 지정. 빈 GPU가 부족하면 실행을 거부한다.
     - 예: `NGPU=4 bash run_sam.sh` · `NGPU=1 bash run_sam3_train.sh` · `CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash run_sam3_rbma.sh <cfg>`(직접 지정은 그대로 존중).
   - **원격 런처**(`scripts/remote_exp.sh`): 먼저 `status <server>`로 확인하고, `run <server> <cfg> auto:N`으로 **원격의 빈 GPU N장을 자동 배정**한다(`auto`=1장). 빈 GPU가 없으면 거부.
   - 판정 기준: GPU가 `memory.used ≤ 2000MiB && util ≤ 10%`이면 "빈 GPU"(환경변수 `GPU_MAXMEM`/`GPU_MAXUTIL`로 조정). 헬퍼/`auto`는 메모리 적은 순으로 고른다.
