---
created: 2026-09-18
author: opus 서브에이전트(판정 세션 "MMSAM | 생각정리" 의뢰) — 출처는 seg-analysis 스킬·tools/README_seg_analysis.md·카드 §0/§0-1/§0-2/§5-31b/§5-34·status/current.md·infra/artifact-locations.md·각 도구 docstring
status: 상시 규약. 분석 세션은 세션 시작 직후 이 문서 하나를 끝까지 읽고 그대로 따른다.
---

# 분석 세션 프로토콜 — 새 체크포인트를 받았을 때 무엇을 재고, 어떻게 보고하는가

> **이 문서를 읽는 사람**: user 가 "MUSES 분석해 줘 / DELIVER 분석해 줘 / 이 ckpt 판독해 줘"라고 말해 새로 열린 Claude 세션.
> **왜 있나**: 세션마다 프로토콜(채점 해상도·체크포인트 선택 규칙·24클래스 계산)이 달라 수치가 섞였고, 판정 세션이 "이전 모델 대비 어느 클래스가 몇 점 움직였는지"를 매번 다시 캐내야 했다. 이 문서는 그 두 가지를 없앤다.
> **선행 문서**: 도구 매핑은 `.claude/skills/seg-analysis/SKILL.md` 와 `tools/README_seg_analysis.md`, 판정 규약은 `decisions/2026-09-07-daily-cycle-experiment-cards.md` §0·§0-1·§0-2, 용어는 `meta/experiment-glossary.md`. 이 문서와 그 문서들이 어긋나면 **이 문서의 §7 "충돌·미확인"** 을 먼저 보라.

---

## 0. 역할과 금지

### 0-1. 분석 세션이 하는 일 / 하지 않는 일

| 분석 세션이 한다 | 분석 세션이 하지 않는다 |
|---|---|
| 측정(재채점·조건별·클래스별·모듈 진단·결측 모달) | **판정**(카드 통과/폐기, 게이트 통과 선언, 헤드라인 교체, SOTA 돌파 주장) |
| 집계(Δ 표 작성, 24클래스 재계산, 시드 평균) | 정본 문서의 수치 변경 — `status/current.md` 헤드라인 절, 카드 문서 §5 판정 행 |
| 기록(`experiments/analysis/` 문서 작성, MOC 등록, registry 행 갱신) | 노션 실험노트·논문 페이지 갱신(판정 세션 승인 후) |
| 보고(판정 세션에 고정 형식으로 전달) | 다음 실험 설계·모델 제안(`model-proposal` 스킬, 판정 세션 소관) |

**판정은 판정 세션("MMSAM | 생각정리")이 한다.** 분석 세션은 "이 수치가 게이트 X를 넘었는지"를 계산해 보여 줄 수 있지만, "그러므로 통과/폐기"라고 쓰지 않는다. 판정 세션의 회신을 받은 뒤에 그 문장을 그대로 분석 문서의 "판정" 절에 옮겨 적는다(§5-3).

**결정이 필요한데 판정 세션의 회신을 기다릴 수 없으면, 가장 보수적인 선택으로 진행하고 그 선택과 근거를 보고의 "막힘 / 판단 필요"에 남긴다.** 보수적인 선택의 예: 두 프로토콜 중 어느 쪽을 쓸지 모르면 둘 다 돌려 병기한다. 체크포인트 선택 규칙이 애매하면 학습기 val-best(top1)를 쓰고 다른 후보는 각주로 적는다.

### 0-2. 절대 금지 (어기면 그 수치는 전부 무효다)

1. **test-best 체크포인트 인용 금지.** 파일명 `test_` 접두어(`test_epoch178_55.27_top1_checkpoint.pth`)나 로그의 `Best Test ...@ep20` 은 test 를 보고 고른 값이라 논문·registry·보고 어디에도 쓰지 않는다. 철회 사고 2회(P34 57.60, P46 57.05). 쓸 것은 `epoch<N>_<val>_top1_checkpoint.pth`.
2. **중간 epoch 으로 시드 페어·카드 간 비교 금지.** 판정은 완주 시점 val-best 에서만. 근거: B0 대 B0s2 의 ep5/10/15 Δ가 +1.00/−2.36/+1.36 으로 부호가 두 번 뒤집혔고 진폭(3.72)이 카드 간 차이(0.7)보다 크다(카드 §0 보강·§5-3).
3. **PhysAug·TTA 를 헤드라인 수치로 쓰지 않는다.** MUSES 레시피는 PhysAug-off 로 통일됐다(E7 대 E7c 페어에서 PhysAug 가 −0.13 으로 손해). 현행 MUSES 헤드라인(79.29±0.71)은 PhysAug-on 레시피라 공정선 밖이며 그 사실을 반드시 각주로 단다. TTA(`val.py --tta`, `--test_gamma_tta`)는 진단용이며 헤드라인에 쓰지 않는다.
4. **실험 약어를 설명 없이 쓰지 않는다.** `E1`·`E13`·`G1`·`C3`·`P46` 같은 꼬리표가 나올 때마다 같은 문장·같은 표 행에 ①무엇을 보는 실험인지 ②왜 ③바꾼 변수 하나 ④결과가 말해 주는 것을 붙인다. 용어 정본은 `meta/experiment-glossary.md` 이지만, 그 문서가 있다는 사실이 설명을 생략할 근거는 아니다. 예: "E1(백본 블록 6/12/18/24 중간층 4탭을 FPN 에 함께 읽는 카드)".
5. **모달 수 표기 의무.** 모든 표·문장·registry 행에 3모달(MUSES img/lidar/event)/4모달(DELIVER img/depth/event/lidar)을 명시한다. 같은 모델도 모달 수가 다르면 다른 실험이다(P34 3모달 test 78.979 대 4모달 78.256).

### 0-3. 서버·GPU 규칙

- **빈 GPU 에만 배치한다.** 판정 기준 = `memory.used <= 2000MiB && util <= 10%`. 헬퍼 `bash scripts/pick_free_gpus.sh [N]` 이 빈 GPU 인덱스를 콤마로 출력하고, 부족하면 실패한다. 원격은 `bash scripts/remote_exp.sh status <server>` 로 먼저 보고 `auto:N` 으로 배정한다.
- 🔴 **lecun 에는 어떤 작업도 배치하지 않는다**(user 2026-09-17, 후배에게 양도). lecun 은 파일 복사만 한다. 덧붙여 lecun·levine 은 공유 conda 의 timm 0.4.12 에 `vit_large_patch16_dinov3` 가 미등록이라 **ReliaDINO 계열 실행 자체가 불가**하다.
- **학습이 도는 서버의 체크아웃을 `git pull` 하지 않는다.** 분석에 코드가 필요하면 그 서버의 격리 worktree(`/SSDb/jemo_maeng/src/dm_analysis` 전례)에서 `origin/develop` detach 로 돌리거나, 파일 단위로 전송하고 md5 로 대조한다.
- 학습과 GPU 를 공유할 때는 메모리를 감시하고 임계를 넘으면 **분석만** kill 한다(학습 보호).
- 기계적 실행·전송·회수는 sonnet 에 위임할 수 있다(`Agent` tool, `model: "sonnet"`). **수치 해석과 원인 규명은 위임하지 않는다**(CLAUDE.md §1.6).

---

## 1. 입력 확정 — 측정을 시작하기 전에 이 표를 채운다

**이 표가 비어 있으면 측정하지 않는다.** 비어 있는 채로 낸 수치는 나중에 재현할 수 없고, 실제로 ISSUE-035(헤드라인 56.99 런의 경로가 기록되지 않아 재평가 불가)가 그렇게 났다.

### 1-1. 분석 대상 (필수 11항목)

| 항목 | 어떻게 채우나 | 비어 있으면 |
|---|---|---|
| 런 이름 + 한 줄 설명 | config 파일명 + "무엇을 보는 실험인가" | user·판정 세션에 질의. 추측 금지 |
| config 경로 | 학습 config(`configs/<dataset>/...`)와 평가 config(`configs/eval/...`) 둘 다 | 학습 config 에서 EVAL/TEST `IMAGE_SIZE: [1024,1024]`·`EVAL.BATCH_SIZE: 1` 만 바꿔 파생시킨다 |
| 체크포인트 경로 | 서버 절대 경로 + NAS 정본 경로 | §4-1 로 먼저 NAS 에 보존한 뒤 측정 |
| 체크포인트 md5 | `md5sum <ckpt>` | 필수. 같은 이름의 다른 파일이 서버마다 있다 |
| 선택 규칙 | **학습기 val-best(top1)** 이 정본 규칙(2026-09-16 고정). 다른 규칙(N6 = 저장된 ckpt 를 legal val 로 재선택)을 썼으면 규칙 이름을 반드시 병기 | 규칙 이름 없이 "5시드 평균"이라고 쓰면 53.83(top1)인지 54.39(N6)인지 구분 불가 |
| 시드 | config 의 `TRAIN.SEED`. 키가 없으면 코드 기본 3407 | MUSES E7·E1M·E13M 초판은 SEED 키가 없어 전부 3407. 파일명의 `seed2` 는 시드가 아니라 **P39.1 레시피 이름**이다 |
| 학습 길이 | 스크린 40ep / 확정 200ep / 그 외 실제 epoch 수 + `EVAL_INTERVAL` | 평가 간격이 다르면 val-best 후보 밀도가 달라진다(E1 확정 interval 2 = 후보 100지점, E13 확정 interval 5 = 40지점 → 차이가 ±0.5 안이면 동급으로 읽는다, 카드 §0-1) |
| 데이터셋·모달 | DELIVER 4모달 / MUSES 3모달 / MCubeS 4모달 + `DATASET.MODALS` 원문 | 모달 라벨 오표기의 원인 |
| PhysAug on/off | config `PHYSAUG` 블록 | MUSES 공정선은 off. on 이면 헤드라인 후보 아님 |
| 학습 해상도 / 평가 해상도 | 예: 768² 학습 · 1024² 평가 | mismatch 는 논문에 명시 의무 |
| 코드 커밋 | 그 체크포인트를 만든 체크아웃의 `git rev-parse HEAD` 와 develop 조상 여부 | `MODEL.TAPS` 도입 커밋 7d83c11 이전 체크아웃은 키를 조용히 무시한다 → `total_trainable` 대조 필수 |

### 1-2. 비교 대상 확정 (네 축을 모두 채운다)

측정값 하나만 보고하면 판정 세션이 다시 캐내야 한다. **아래 넷을 미리 확정하고, 각각이 어느 프로토콜·하네스 버전으로 잰 값인지 함께 적는다.**

| 축 | 무엇 | DELIVER 현재값(2026-09-18) | MUSES 현재값 |
|---|---|---|---|
| (a) 같은 시드·같은 코드의 기준선 | 페어 판정의 분모 | 확정 런 분모 = P46 C3-only 시드 20260821 ep90(`epoch90_67.3_top1`) legal test **53.57** / 24클래스 **54.68** / 얇은 객체 4클래스 **43.25** · 시드2(902) ep140 53.46 / 54.79 · 시드3(903) 54.41 / 54.54 · 스크린 분모 B0 시드1 53.78 / 24클래스 54.69 | E7(PhysAug-off 기준선) 공식 native val 3페어 평균 **79.9856**(시드별 80.0756 / 79.6057 / 80.2754) |
| (b) 현재 정본 헤드라인 | `status/current.md` 벤치 표 | test **56.99**(P46 C3-only 본 런 ep70) — 🔴 채점 프로토콜 확정 중, §2-1-3 참조 · legal val 최고 67.89(E1 확정 시드3) | 공식 test **79.29 ± 0.71**(2시드 평균), best 단일 런 **79.788**(P39.1-rank seed2 3모달 ep208, PhysAug-on) |
| (c) 직전 세대 최고 | 같은 계보의 이전 최선 | E1 확정 3쌍 평균 test 54.93 / 24클래스 55.49 · E13 확정 3쌍 평균 55.14 / 55.32 · P34 56.62(구기록) | P38-m2f 79.025(7월) · P43 79.351 · 시드 20260825 78.786 |
| (d) 공개 SOTA — 모달리티 정합 | **1차 = 같은 모달 집합, 2차 = 적은 모달**(카드 §0-2) | 1차: DGFusion 4모달 test 56.71 / val 66.51, CAFuser 4모달 55.6 · 2차: MM SAM-adapter RGB+D test 57.35 / val 69.60, CAFuser-CAA val 68.79 | 1차: DGFusion 융합 79.5 · 2차: MM SAM-adapter RGB+L 81.07, GtA 카메라 단독 82.39 |
| (d') 우리가 직접 재학습한 기준선 | 공개 수치가 아니라 같은 환경에서 잰 값 | DGFusion Swin-T 재학습: final-iter val 65.62 / test 55.56 · val-best(80k) 66.54 / 55.68 · CAFuser Swin-T 재학습 test 55.3761 | (없음 — §7-2 미확인 6) |
| | MCubeS | 우리 3시드 58.07±0.49, published 최고 Mul-VMamba 54.65 | |

🔴 **(a)와 (b)의 분모 이름을 반드시 적는다.** DELIVER 두 분모의 24클래스 값이 54.69(스크린 B0)와 54.68(확정 seed821)로 0.01 차이라 "분모 54.7"이라고만 쓰면 되짚을 수 없다.

🔴 **프로토콜·하네스 버전을 값마다 붙인다.** 같은 체크포인트가 채점 경로에 따라 1.81 갈린 실측이 있다(§2-1-3). 표기 예: `55.18 (legal v1 native, 하네스 가드 OK)` / `56.99 (학습기 복제 경로, 1024-resized GT)` / `55.88 (legal v1 native, 덤프 재현)`.

### 1-3. 착수 전 확인 3줄

```bash
# 1) 하네스 동결 확인 — legal 수치를 내는 모든 실행 전 필수. FAIL 이면 그 수치는 legal 인용 금지
python tools/eval_harness_guard.py --check          # 기대: [guard] OK — 8 files match frozen manifest.

# 2) 체크포인트 md5 (보고표에 그대로 적는다)
md5sum <ckpt>

# 3) 빈 GPU 확인
bash scripts/pick_free_gpus.sh 1                    # 또는 원격: bash scripts/remote_exp.sh status <server>
```

---

## 2. 벤치별 표준 절차 — 명령 그대로

### 2-0. 실행 환경 (서버별, 하나라도 빠지면 다른 얼굴로 실패한다)

| 서버 | 파이썬 | 필수 env |
|---|---|---|
| 허브(이 박스) | `conda activate MMSS_SAM` | `PYTHONPATH=<repo>/semseg/models/sam2:$PYTHONPATH` |
| jarvis | conda `MMSS_SAM` | `PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:<repo>/semseg/models/sam2:.`(timm 1.0.24 = DINOv3 지원) + `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` |
| yeon | conda `MMSS_SAM` | `PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:<repo>/semseg/models/sam2` |
| hpca100 | venv `/home/jovyan/SSDb/jemo_maeng/venv/p34` (conda 없음) | 넷 다 필요: `PYTHONPATH=<repo>/semseg/models/sam2` · `HF_HOME=/home/jovyan/.cache/huggingface`(셸 기본값은 **타 사용자의 빈 캐시**다) · `HF_HUB_OFFLINE=1` · `LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH` |
| lecun·levine | — | 🔴 ReliaDINO 실행 불가(timm registry gap). 배치 금지 서버이기도 하다 |

- `ModuleNotFoundError: No module named 'sam2'` 는 **평가(`val.py`)에서 주로 걸린다** — 학습은 이 변수 없이도 뜬다.
- **ckpt 기반 평가는 `HF_HUB_OFFLINE=1` 이어도 안전**하다(백본 가중치가 ckpt 에서 복원되므로 RANDOM INIT 위험 없음). fresh 학습과 다르다.
- 로그의 `Loading pretrained weights from Hugging Face hub` 문구는 offline 이어도 그대로 찍힌다 — **문구로 백본 로드를 판별하지 마라.**

---

### 2-1. DELIVER (4모달 img/depth/event/lidar)

#### 2-1-1. legal 재채점 (모든 DELIVER 수치의 1차 산출)

정본 정의: `val.py` · 평가 해상도 1024² · `EVAL.BATCH_SIZE 1` · **native GT**(1042 격자) · 하네스 가드 통과.

```bash
# 사전: 하네스 가드
python tools/eval_harness_guard.py --check

# test (1897장)
PYTHONPATH=<repo>/semseg/models/sam2:. python val.py \
  --cfg configs/eval/<server>-deliver_rgbdel_<run>_eval1024.yaml \
  --mode test --model_path <val-best.pth>

# val (2005장)
PYTHONPATH=<repo>/semseg/models/sam2:. python val.py \
  --cfg configs/eval/<server>-deliver_rgbdel_<run>_eval1024.yaml \
  --mode val --model_path <val-best.pth>
```

🔴 **진행 표시줄로 BS 를 검증한다.** test 는 1897/1897, val 은 2005/2005 가 그대로 돌아야 한다. `475/501` 로 뜨면 BATCH_SIZE 가 4 인 것이므로 **즉시 중단**한다(두 번 실증된 사고, 평가 config 헤더 주석).

평가 config 는 학습 config 에서 `EVAL.IMAGE_SIZE`·`TEST.IMAGE_SIZE` 를 `[1024,1024]` 로, `EVAL.BATCH_SIZE` 를 `1` 로 바꾼 파생본이다. 새로 만들면 `configs/eval/` 에 `<server>-<dataset>_<modal>_<version>_eval1024[_<카드>].yaml` 로 커밋한다.

#### 2-1-2. 🔴 2026-09-18 현재 — 하네스 v1·v2 를 **둘 다** 돌려 병기한다

ISSUE-036: `val.py:_unpad_resize_to_orig` 가 1024 예측을 native 1042 로 올릴 때 `F.interpolate(mode="nearest")`(torch 의 floor 정렬)를 쓴다. 이 정렬이 반 픽셀 계통 편차를 만들어 **우리 legal 수치를 약 −1.3 낮게**(얇은 클래스는 −3~−6) 재고 있다. GPU 없이 한 실측(DGFusion 80k test 예측 RLE 1024, 317장 표본): `nearest` 52.34 / `nearest-exact` 53.68 / PIL NEAREST 53.68, 얇은 객체 4클래스 50.25 대 53.38(Pole 51.3 대 57.1, Pedestrian 79.1 대 83.5).

v2 채택은 **판정 세션이 아직 정하지 않았다**(카드 §5-34 (2-정정)). 그때까지 분석 세션은 둘 다 돌리고 병기한다.

```bash
# v2 = val.py 무수정, _unpad_resize_to_orig 만 nearest-exact 로 몽키패치. 인자는 val.py 와 동일
PYTHONPATH=<repo>/semseg/models/sam2:. python tools/legal_rescore_v2.py \
  --cfg configs/eval/<...>.yaml --mode test --model_path <val-best.pth>
```

- v2 는 하네스 가드를 깨지 않는다(val.py 파일을 건드리지 않으므로). 가드 `--check` 는 그대로 먼저 돌린다.
- **카드 간 상대 판정(같은 하네스끼리 비교)은 이 편차의 영향을 받지 않는다.** 영향을 받는 것은 (i) 공개·재학습 기준선과의 절대 비교 (ii) "얇은 객체 병목"의 크기 주장이다.
- 보고표에는 `legal v1(nearest)` / `legal v2(nearest-exact)` 두 열을 만든다.

#### 2-1-3. 헤드라인 56.99 의 현재 상태 (인용할 때 반드시 각주)

같은 체크포인트(md5 `d340e3fe9824bd922fc7fe6eff7a8b26`)·같은 config 로 두 경로를 돌린 실측:
- legal(`val.py`, 가드 통과, native 1042 GT): **test 55.18 / val 66.88**
- 학습기 복제 경로(2026-08-14 Job1 스크립트, `Resize(1024)` 로 이미지·라벨을 함께 줄여 그 격자에서 채점): **56.99**

2026-09-18 12:00 판정: 1.81 의 원인은 프로토콜 차이가 아니라 **legal 하네스의 재샘플 정렬 편차**(§2-1-2)로 보인다 → **"헤드라인 legal native 통일"과 "DGFusion 대비 +0.28 철회"는 보류**, `current.md` 헤드라인은 "하네스 재표본 방식 확정 중"으로 두고 수치를 바꾸지 않는다. 분석 세션은 이 상태를 그대로 옮겨 적고, 56.99 를 인용할 때 위 두 값을 함께 적는다.

#### 2-1-4. 클래스별 · 24클래스 · 얇은 객체 · 큰 영역

`val.py` 가 25클래스 per-class IoU 표를 찍는다. 그 **원자료**에서 매번 다음을 재계산한다(다른 문서에서 옮겨 적지 않는다 — §5-6 초판의 분모 오기가 E13·E4b 판정에 전파된 사고가 있다).

| 지표 | 계산 | 왜 |
|---|---|---|
| 25클래스 mean | val.py 출력 그대로 | 기본 |
| **24클래스 mean** | `(25 × mean − RailTrack) / 24` — **매번 원자료에서** | RailTrack 이 한 런 안에서 26.96~72.03 으로 요동하고 시드 간 43점(E2 시드1 47.75 대 시드2 4.36)까지 갈린다. 25클래스 Δ는 이 한 클래스에 휘둘린다 |
| 얇은 객체 4클래스 평균 | Pole · Pedestrian · Static · TrafficLight | 분모(확정 seed821) 값 43.25. 게이트 G2(얇은 객체) = E1 대비 Δ ≥ +2.0 |
| 큰 영역 3클래스 | RailTrack · Wall · Water | 요동·붕괴 클래스 묶음. 전이 실패 서사의 주 무대 |

#### 2-1-5. 조건별 (5조건, 조건당 379~380장)

```bash
python tools/eval_per_domain.py \
  --cfg configs/deliver/<train-cfg>.yaml \
  --ckpt <라벨>=<val-best.pth> \
  --dataset-root <DELIVER 루트> \
  --conditions cloud,fog,night,rain,sun \
  --split test --batch 1 --gpu <free> --out-dir <out>

python tools/analyze_per_domain.py --logs-dir <out> \
  --label <라벨>=<로그 prefix> --out <out>/per_domain_analysis.md
```

- `--batch 1` 을 반드시 준다(기본값 2). `--split` 기본은 test.
- `eval_per_domain.py` 는 `--macvi` 로 val.py 를 부른다(4모달 시각화 패널이 크래시하므로). per-class IoU 표는 그대로 나온다.
- **DELIVER 조건별에는 §5-31b 의 표본 규칙(3시드 반복 시에만 실재)을 적용하지 않는다** — 조건당 379~380장으로 MUSES(25~34장)보다 10배 이상 크다.
- 케이스별(센서 고장): scene 폴더명 접미사 `_motionblur` / `_overexposure` / `_underexposure` / `_lidarjitter` / `_eventlowres`, 없으면 `none`. 케이스 단위 집계는 `tools/baseline_failure/failure_mining.py` 경로를 쓴다.

#### 2-1-6. 결측·열화 모달 (EMM / RMM / NM)

**언제 돌리나**: 헤드라인 후보 체크포인트일 때, 또는 융합·게이팅 구조를 바꾼 카드일 때. 스크린 카드마다 돌리지 않는다(비용).

```bash
python tools/missing_modality_eval.py \
  --cfg configs/eval/<...>.yaml --model_path <val-best.pth> \
  --split val --protocol all \
  --expected_clean_miou <legal val 값> --tol 0.05 \
  --subset_every 1 --batch 1 \
  --out /drone_nas/.../analysis_logs/<run>_missing_modality_<YYYYMMDD>
```

- 프로토콜(벤치마크 arXiv 2503.18445): **EMM** = 결측 모달을 정규화 후 0 으로 채운 15조합(전부-결측 제외) · **RMM** = 같은 15조합을 픽셀×채널 독립 드롭(비율 `--rmm_ratios`, 기본 0.25/0.5/0.75) · **NM** = 결측 없이 4모달 전부에 salt-and-pepper(`--nm_density`, 기본 0.05/0.1/0.2). Gaussian 은 릴리스에서 주석 처리돼 `--nm_gaussian` 으로만.
- `--expected_clean_miou` 를 주면 clean 값이 legal 과 어긋날 때 멈춘다 — **반드시 준다**.
- `--presence_renorm` 은 기본 꺼짐. 켜고 끈 쌍을 같은 부분집합에서 재는 것이 다음 측정 우선순위에 있다(카드 §5-34 (3)).
- `--subset_every N` 은 N장당 1장만 본다(첫 실측은 5장당 1장 = 401장). **부분집합 수치는 벤치마크 비교용이 아니다** — 문서에 그 사실을 적는다.
- EMM 평균과 RMM 평균은 **다른 수치 계열이다. 섞어 쓰지 마라.**

#### 2-1-7. 모듈 진단 · 특징 시각화 (구조를 바꾼 카드일 때)

```bash
# 모듈 A/B 토글 + 모달 제거 + 라우터 가중 + UAMM 배분
python tools/module_diagnostics.py --cfg <cfg> --model_path <ckpt> \
  --dataset-root <DELIVER> --conditions cloud,fog,night,rain,sun \
  --split test --max-imgs 120 --ablate-n 20 --gpu <free> --out <prefix>

# 한 번에: 8스테이지 파이프라인 (조건당 약 10분, 전체 80~120분)
python tools/seg_analysis_pipeline.py --cfg <cfg> --model_path <ckpt> \
  --dataset-root <DELIVER> --out-dir <out> --gpu <free> \
  --stages D1,D2,D2N,D3,D3B,D4,D5

# 특징 패널 (per-modal 인코더 / fused / reliability / UAMM)
python tools/viz_features.py --cfg <cfg> --model_path <ckpt> \
  --dataset-root <DELIVER> --case night --contains RailTrack --num 2 \
  --gpu <free> --out-dir <out>/viz
```

🔴 **토글이 목록에 안 뜨는 것과 no-op 은 다르다.** `module_ablation.py::make_toggles()` 는 모듈이 **실제로 결선돼 있을 때만** 토글을 등록한다(`p38_m2f_off` 는 arbiter 부재 시에만, `p39_*` 는 arbiter 존재 시에만, `p37_cefr_off` 는 `fusion.cefr` 존재 시에만). 실행 로그의 `available=[...] skipped=[...]` 줄을 **반드시 보고에 남기고**, skip 사유는 "모듈 부재"로 적는다.

판정 보조선(판정 자체는 판정 세션): `miou_delta_when_off` 가 **+ 면 모듈 기여** · `|Δ|<0.5 & pred_agreement>0.99` 면 **no-op 후보** · `Δ <= −0.5` 면 **유해 후보** · `Δ >= +20 & agreement<0.8` 이면 기여가 아니라 **co-adaptation 의존**(단일 실패점).

---

### 2-2. MUSES (3모달 img/lidar/event)

#### 2-2-1. 공식 native 채점 (val 250장, 1080×1920)

MUSES 는 **트레이너 내부 지표(레터박스 1024²)와 공식 지표가 다르다.** 리더보드는 native 1080×1920 `*_gt_labelTrainIds.png` 로 채점한다. 보고에는 **공식 native 값**을 쓰고 레터박스 값은 각주로만 둔다.

```bash
python tools/eval_muses_official.py \
  --cfg configs/<server>-muses_rgbel_<run>.yaml \
  --ckpt <val-best_checkpoint.pth> \
  --gpu <free> \
  --out <out>/muses_official_<run>
# 산출: report.json + hist_*.npy(원시 혼동행렬). hist_1024(트레이너 내부)와 hist_full(공식)을
#       같은 forward 에서 동시에 누적하므로 두 값이 엄격히 사과-대-사과다.
```

- 조건별(weather × time-of-day) 혼동행렬도 같은 패스에서 누적된다.
- `--dataset-root` 는 config 의 `DATASET.ROOT` 만 덮어쓴다. `--limit N` 은 스모크용.
- native GT 는 데이터셋이 `meta['orig_label']` 로 돌려준다(`return_meta=True`) — 이 도구가 PNG 를 직접 다시 읽지 않는다.

#### 2-2-2. 조건별 (8셀) — 🔴 표본 규칙

공식 조건 축: weather 4(clear/fog/rain/snow) × tod 2(day/night) = 8셀. **val 은 조건당 25~34장, test 는 75~150장이다.**

🔴 **MUSES 조건별 수치는 한두 시드에서 나온 특정 조건의 손실을 실재로 읽지 않는다. 같은 조건이 3시드 이상에서 반복될 때만 실재로 본다**(카드 §5-31b). 근거: E13M 의 rain/night 손실 −1.79 가 두 시드에서 보였는데 시드3 에서 +0.30 으로 뒤집히고 대신 rain/day 가 −0.80 이 됐다. 조건이 고정된 것이 아니라 **손실 조건이 시드마다 옮겨 다닌다.**

예외로 "시드에 무관하게 재현되는 것"으로 이미 확인된 셀은 그대로 보고한다: fog_night 69~70 고정 · snow_day < snow_night 역전 4회 재현 · night truck 붕괴.

#### 2-2-3. 클래스별 19개

Cityscapes trainID 0~18: road / sidewalk / building / wall / fence / pole / traffic light / traffic sign / vegetation / terrain / sky / person / rider / car / truck / bus / train / motorcycle / bicycle. `report.json` 에서 뽑아 전표로 낸다.

#### 2-2-4. test 제출 (Codabench comp 14005) — **user 승인 필수**

MUSES test GT 는 서버 비공개다(그래서 test-best 훔쳐보기가 구조적으로 불가하고, 전 제출이 val-best 단일 선택이라 선택편향이 없다).

- **제출 게이트**: `status/current.md` 의 기록 = "공식 val ≥ 82.62 일 때만 1회". 이 게이트와 무관하게 **제출은 user·판정 세션 승인 사항**이다. 분석 세션이 임의로 제출하지 않는다.
- 제출물 생성:

```bash
python tools/predict_muses_test.py \
  --cfg configs/<...>.yaml --ckpt <val-best_checkpoint.pth> \
  --gpu <free> --out <out>/labelTrainIds
# 기하는 eval_muses_official.py 와 동일: 레터박스 1024 forward -> 패딩 crop ->
# logits 를 native 1080x1920 로 bilinear 업샘플 -> argmax -> uint8 trainID PNG 1920x1080

python tools/verify_submission.py <out>/labelTrainIds <MUSES test RGB 디렉터리>
# 검사: labelTrainIds/ 디렉터리 · zip 200MB 이하 · 파일명 {sequence}_frame_{frame:0>6}*.png
#       · test 이미지당 정확히 1장(750장) · 1920x1080 · 라벨 0..18(255 금지)
```

- 제출 zip 과 그 zip 을 만든 체크포인트를 **짝으로** `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/muses/` 에 보관하고, zip md5 와 ckpt md5 를 분석 문서에 적는다(전례: `muses_P39_1_seed20260825_3modal_ep168_submission.zip`, md5 `e4f02f3f…`).
- Codabench 는 **최종 수치만 반환**한다(원시 혼동행렬 없음). 필요하면 제출 zip 의 예측 PNG 750장으로 로컬 재집계한다(P38 전례).
- 결과 수신 후 `experiments/analysis/MUSES_TEST_RESULTS_INDEX.md`(MUSES test 수치의 단일 진입점)에 행을 추가한다.

---

### 2-3. MCubeS (4모달)

- 🔴 **final-epoch 기준이 정본이다.** 로더가 `split='val'` 일 때 `list_folder/test.txt` 를 읽기 때문에(`semseg/datasets/mcubes.py:140`) 학습 중 val = 커뮤니티 표준 test split(102장)이고, 따라서 **val-best = test-best 가 되어 선택편향이 생긴다.** 그래서 게이트는 final-epoch 로 재고 val-best 는 병기만 한다.
- 게이트: 3시드 평균 Δ ≥ +0.5.
- 🔴 **척도 열 필수**: 카드 값이 final 인지 val-best 인지, 기준선 값이 어느 쪽인지를 표의 열로 적는다. E13Mc 를 카드 final 대 기준선 val-best 로 비교해 −0.62(실제 +0.40)로 잘못 보고한 사고가 이 규칙의 계기다(카드 §0-1).
- 현재 정본: 통일 레시피(C3-off) 3시드 **58.07 ± 0.49** {57.93, 57.67, 58.62}.

---

### 2-4. 공통 — 필요할 때만

| 무엇 | 도구 | 언제 |
|---|---|---|
| 모듈 정량(모달 제거·라우터 가중·탭 기여) | `tools/module_diagnostics.py` | 구조를 바꾼 카드 |
| 특징 통계(dead-ch / eff-rank / CKA / PCA) | `tools/feature_stats.py`(파이프라인 D2N) | 병목의 물리적 실체를 물을 때 |
| 특징 패널 시각화 | `tools/viz_features.py` | 정성 자료가 필요할 때 |
| 어댑터 건강(정적 dW, CPU만·데이터 불필요) | `tools/adapter_health.py --ckpt <pth> --out health.json` | 어느 ckpt든 즉시 |
| 다모델 비교(STRUCTURAL / DESIGN-GAP / DOMAIN-GAP / SOLVED 자동 분류) | `tools/compare_models.py --run A=<dir> --run B=<dir> --out compare.md` | D1 산출이 2개 이상일 때 |
| 실패 사례 채굴(이미지별 지표·혼동·객체 크기·센서 고장) | `tools/baseline_failure/`(D1 덤프 → D2 per_image_metrics → D3 failure_mining → D5 viz_panels) | 기준선 대비 실패 구조를 물을 때. 설계서 = `analysis/2026-09-17-baseline-failure-analysis-plan.md` |

**분석 코드를 새로 짜지 않는다.** 필요한 게 없으면 기존 도구를 **확장**하고 `tools/README_seg_analysis.md` 와 `.claude/skills/seg-analysis/SKILL.md` 를 같은 턴에 갱신한다.

---

## 3. 판독 — 무엇을 보고 무엇을 말하나

### 3-1. 반드시 만드는 표

1. **클래스별 Δ 전표** — DELIVER 25클래스 / MUSES 19클래스 **전부**. 열은 최소 셋: 절대값 · (a)기준선 대비 Δ · (c)직전 최고 대비 Δ. 표 아래에 **오른 것 상위 5개와 내린 것 상위 5개**를 문장으로 요약한다.
2. **묶음 Δ** — DELIVER: 24클래스 · 얇은 객체 4클래스 · 큰 영역 3클래스. MUSES: 얇은 클래스(pole·rider·motorcycle·bicycle·traffic light)와 넓은 stuff(wall·fence·terrain)를 나눠 본다(시드 20260825 판독에서 이 축이 손실의 소재지를 갈랐다).
3. **조건별 Δ** — DELIVER 5조건, MUSES 8셀(+ weather 4 · tod 2 집계). MUSES 는 §2-2-2 표본 규칙을 표 아래에 명시.
4. **(Δval, Δtest) 쌍** — 🔴 **전이율(Δtest/Δval)을 단독으로 쓰지 않는다.** 항상 쌍으로 적는다. Δval 이 작으면 전이율이 폭발한다(E3 는 val +0.89 · test +1.40 으로 "157%"가 되지만 그 숫자 자체는 의미가 얇다).

### 3-2. 크기를 시드 분산과 비교한다

| 벤치 | 알려진 분산 | 함의 |
|---|---|---|
| DELIVER 스크린 페어 | 페어 편차 약 ±0.6, 검정력 기준 ±1.0 | Δ < 1.0 은 단일 페어로 판정 불가 |
| DELIVER RailTrack 단일 클래스 | 시드 간 43점(E2 47.75 대 4.36), 같은 길이 런에서 26.96~72.03 | **단일 시드 RailTrack 값은 어떤 판정에도 쓰지 않는다** |
| DELIVER test (P46 5시드) | std 2.21 | |
| MUSES 공식 test | 2시드 mean 79.287 · std 0.709(첫 실측) | 우리 제출 간 아키텍처 격차(0.4~0.9)가 **전부 이 분산 폭 안**이다 |
| MUSES 공식 val | 2점 mean 81.80 · std 0.47 | val spread 0.66 이 test 에서 1.00 으로 증폭됐다 |
| MCubeS 3시드 | ±0.49 | |

### 3-3. SOTA 거리 (§0-2 게이트, 두 항목 필수)

확정 런을 보고할 때는 Δ 게이트에 더해 **둘 다** 판정표에 적는다.
1. **우리 최고 단일 런(같은 선택 규칙) 대비 ≥ 0 인가**
2. **모달리티 정합 SOTA 대비 거리** — 1차(같은 모달 집합) 먼저, 2차(적은 모달)를 병기

미달이면 그 카드는 통과라도 **헤드라인 후보가 아니라 "기준선 대비 이득 카드"** 로만 기록된다. 예: E1 확정 3쌍 평균 test 54.93 은 56.99 대비 −2.06, DGFusion 56.71 대비 −1.78 → 기준선 대비 이득 카드.

또한 **단일 런 최고와 시드 평균을 반드시 함께 적는다.** DELIVER 4모달 1위 56.99 는 단일 런이고, 같은 레시피 5시드 평균(학습기 top1 규칙) 53.83 은 DGFusion 미달이다.

### 3-4. 🔴 무효 판정 조건 — 하나라도 걸리면 그 수치는 버린다

| 조건 | 어떻게 확인 |
|---|---|
| 백본 RANDOM INIT | 학습 로그에 `grep -iE 'dinov3\|RANDOM INIT\|falling back\|safetensors'`. "RANDOM INIT" 이 뜨면 그 런 전체가 무효. 원인은 서버마다 다르다(hpca100 = HF_HOME 오지정, jarvis = timm 구버전) |
| 장수 불일치 | DELIVER test 1897 / val 2005, MUSES 공식 val 250 / test 750, MCubeS 102. 진행 표시줄이 `475/501` 이면 BS=4 → 즉시 중단 |
| 하네스 가드 FAIL | `python tools/eval_harness_guard.py --check` 가 exit 1. 이 상태로 만든 수치는 legal 인용 금지 |
| test-best ckpt | 파일명 `test_` 접두어, 로그의 `Best Test ...@epN` |
| 중간 epoch 비교 | 완주 전 예비 재채점은 "예비 조회"일 뿐이다. 완주 시점에 val-best 가 그 ckpt 그대로인지 대조하고, 갱신됐으면 예비 수치를 버리고 다시 돌린다 |
| 1페어짜리 확정 판정 | 확정 게이트는 3페어 전제. 한 페어는 **"1페어 예비"** 로만 적고 통과·미달로 판정하지 않는다 |
| 24클래스 값을 다른 문서에서 옮겨 적음 | 반드시 클래스별 원자료에서 `(25×mean − RailTrack)/24` 재계산 |
| 모듈 토글 미등록을 no-op 으로 읽음 | `available=[...] skipped=[...]` 로그 확인 |
| 부분집합 수치를 벤치마크 비교에 씀 | `--subset_every`·`--limit`·`--max-imgs` 를 쓴 값에는 그 사실을 명시 |

### 3-5. 이미 반증된 것 (재확인 불필요, 새 모델에서 재발만 체크)

RBMA/CoRB attn-bias(4세대 무효) · reliability gate/calib/veto(3세대 no-op, 일부 조건에선 유해) · CEFR per-class 라우팅(미분화) · 무감독 threshold 마스크 게이트(P37b 영구 random) · zero-init 잔차 결선 일반(β·σ(a) 모두 "열리다 만" 고착) · 추론 경로 안의 재가중 전반 · radar(MUSES) · gradient 균형화. 상세 = `analysis/2026-07-20-failure-keys-p38-deliver-p37a-muses.md`.

---

## 4. 저장 — 지우면 재현할 수 없는 것들

### 4-1. 체크포인트 (측정 **전에** 보존한다)

```
/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/<런>_<YYYYMMDD>/
```

- 이동(복사 + 원본 삭제) 전에 반드시 **NAS 사본이 bit-identical 인지**(파일 수 + md5) 확인한다. 크기 대조만으로는 부족하다.
- 헤드라인 수치를 만든 체크포인트는 `infra/artifact-locations.md` §1b 표에 **행을 추가**한다(수치 · 런 · NAS 경로 · md5 · 남은 사본 · 평가 config). 이 표에 없는 헤드라인 수치는 재평가할 수 없다.
- 🔴 **장시간 전송을 `nohup` 으로 분리하지 마라.** 세션이 완료로 처리한 뒤에도 프로세스가 살아 있어 겹쳐쓰기가 난 사고가 있다(크기 대조로는 안 잡힌다). 세션이 추적하는 방식으로 띄우고, 의심되면 `tar tf` 항목 수 + payload md5 까지 본다.
- 서버에서 ckpt 가 안 보이면 **지워진 것이 아니라 이관된 것**이다. 재학습 전에 `infra/artifact-locations.md` 를 먼저 본다.

### 4-2. 측정 산출물

```
/drone_nas/.../drone-MemorySAM/analysis_logs/<run>_eval_<YYYYMMDD>/
    report/     채점 로그 원문, report.json, per-class 표, 원시 혼동행렬(npy/npz)
    viz/        패널 PNG, 차트 PNG + 그 차트를 만든 파싱 스크립트(재생성 가능하게)
    perdomain/  조건별 로그 + per_domain_analysis.md
```

- **원시 혼동행렬(`hist_*.npy` / `.npz`)을 반드시 남긴다.** Codabench 는 최종 수치만 돌려주므로 로컬 산출이 유일한 원자료다. 전례: `ckpts/MUSES_P34_20260715/official_eval/` 에 `hist_per_condition.npz`·`hist_full.npy`·`hist_1024.npy` 보존.
- 그림은 dataviz 팔레트(#2a78d6 / #1baf7a / #eda100 / #008300 / #4a3aa7) + `Noto Sans CJK JP`.
- 서버 로컬(`outputs/`, `/tmp`)은 **작업 사본이지 정본이 아니다.** `/tmp` 는 컨테이너 재시작 시 소실되므로 완주·완료 즉시 회수한다.

### 4-3. 제출물 · 공유 분석

| 종류 | 위치 |
|---|---|
| 제출(submission) 코드·zip | `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/{code,muses}/` |
| 여러 서버가 함께 읽고 쓰는 분석 산출물 | `/ailab_mat2/.../drone-memorysam/analysis/<run_id>/` — 하위 `raw/` `preds/` `metrics/` `mining/` `reports/` `code/` + `MANIFEST.tsv`. 도구 `scripts/nas_analysis_sync.sh`(`init`/`push`/`pull`/`verify`/`manifest`/`ls`/`mounts`) |

- `/ailab_mat2` 마운트: hub·yeon·lecun 있음, **hpca100 없음**(hub 경유).
- NFS 가 그룹 변경을 거부하므로 전송은 `rsync -rt --no-perms --no-owner --no-group`.
- `/nas_jm` 에는 가중치·대용량을 두지 않는다(논문·리서치 문서 전용).

### 4-4. 문서 (repo — 여기가 정본이다)

**파일명**: `.claude_logs/experiments/analysis/YYYY-MM-DD-<벤치>-<런>-<주제>.md`
(예: `2026-09-18-muses-official-test-p39_1-seed20260825.md`, `2026-09-18-baseline-failure-d3-test-findings.md`)

**절 구성**(기존 분석 문서와 같게):

```
---
created: YYYY-MM-DD
type: <한 줄 — 무엇을 잰 분석인가>
---

# <제목 — 런 이름(풀어 쓴 설명) -> 핵심 수치와 Δ>

> 실험 이름 풀이: <약어 -> 무엇을 보는 실험인지·왜·바꾼 변수 하나>
> 레시피 동일성 근거: <config diff 가 무엇 한 줄인지>

## 1. 입력 확정        (§1-1 표 + §1-2 비교 대상)
## 2. 실행 명령·환경   (실제로 돌린 명령 그대로, env 포함, 하네스 가드 결과)
## 3. 헤드라인 표      (우리 정본 / 같은 시드 기준선 / 직전 최고 / SOTA 대비, val·test)
## 4. 클래스별 Δ 전표  (전 클래스 + 오른 것·내린 것 상위 5)
## 5. 조건별 Δ         (표본 수 명시)
## 6. 판독             (무엇이 움직였고 무엇이 안 움직였나 — 해석이지 판정이 아니다)
## 7. 무효·미확인 수치 (버린 값과 버린 이유, 재지 못한 항목)
## 8. 원본 위치        (ckpt+md5 / 산출물 NAS 경로 / 로그 / 제출 zip+md5)
## 9. 판정             (판정 세션 회신을 받은 뒤 그대로 옮겨 적는다. 그 전에는 "판정 대기")
```

**등록**(같은 턴에 한다):

1. `.claude_logs/experiments/00_MOC.md` 의 "analysis/ 전체 목록" 표 **맨 위**에 행 추가(날짜 역순). 한 줄 설명은 결론을 담는다 — 제목 반복 금지. (analysis 폴더에는 별도 `00_MOC.md` 가 없다. 등록처는 `experiments/00_MOC.md` 다.)
2. `.claude_logs/experiments/registry.md` 의 해당 벤치 표에 행 추가 또는 갱신. 열 = `실험ID(config명) | 트랙 | 데이터셋 | 모델버전 | config 경로 | 서버 | ckpt 경로 | 상태 | 핵심 수치 | 관련 문서`. 분석 행은 트랙에 `seg-analysis` 를 쓰고 실험ID 를 `**ANALYSIS: <주제> (<서버>)**` 형식으로 적는다. 상태 = 🟢 best / 🟡 active / 🔴 dead / ✅ 완료. **데이터셋 칸에 3모달/4모달 명시.**
3. 서사가 필요하면 `.claude_logs/experiments/log.md` 에 추가(결과 canonical).
4. `status/current.md` 는 **판정 세션 승인 후에만** 건드린다.
5. 노션(실험노트 DB `8ec54838-…`, 논문 페이지 `33d05310-…`)도 **판정 세션 승인 후**. 방법은 `.claude/skills/notion-experiment-log/SKILL.md` — 페이지를 새로 만들지 말고 절 함수를 고쳐 빌더를 돌린다(멱등).

---

## 5. 보고 — 판정 세션에 무엇을 어떻게 보내나

### 5-1. 전달 방법

`SendMessage` 로 판정 세션 **"MMSAM | 생각정리"** 에 보낸다. 판정 세션이 그대로 supervisor 에게 넘길 수 있는 형태로 쓴다.

### 5-2. 고정 형식

**첫 줄**: 한 문장 요약 = 런 설명(약어 풀어서) + 핵심 Δ.

> 예: "E1(백본 중간층 4탭을 FPN 에 함께 읽는 카드) 확정 런 시드3 를 legal v1·v2 로 재채점했다 — v1 test 54.41, v2 55.7x, 같은 시드 기준선 대비 Δ +0.40(25클래스) / +0.81(24클래스), 얇은 객체 +2.66."

이어서 **표 네 개를 고정 순서로** 붙인다.

**① 입력 확정**

| 항목 | 값 |
|---|---|
| 런 / 설명 | |
| 학습 config / 평가 config | |
| ckpt 경로 + md5 | |
| 선택 규칙 | 학습기 val-best(top1) / N6 legal-val 재선택 / 기타 |
| 시드 · 길이 · EVAL_INTERVAL | |
| 데이터셋 · 모달 수 · PhysAug | |
| 프로토콜 · 하네스 버전 | legal v1(nearest) / legal v2(nearest-exact) / 공식 native / 학습기 복제 |
| 하네스 가드 | OK / FAIL |
| 코드 커밋 | |
| 장수 확인 | test 1897/1897 등 |

**② 헤드라인 표**

| 대상 | val | test | 척도(val-best·final / legal·트레이너) | Δ vs 이번 런 |
|---|---|---|---|---|
| **이번 런** | | | | — |
| 같은 시드 기준선 | | | | |
| 직전 최고(우리) | | | | |
| 정본 헤드라인(current.md) | | | | |
| SOTA 1차(같은 모달) | | | | |
| SOTA 2차(적은 모달) | | | | |

+ §3-3 두 항목(우리 최고 단일 런 대비 / 모달리티 정합 SOTA 대비)을 숫자로 적는다.

**③ 클래스별 Δ 전표** — 전 클래스(DELIVER 25 / MUSES 19). 열 = 클래스 · 이번 런 절대값 · **기준선 대비 Δ** · **직전 최고 대비 Δ**. 표 아래 한 줄로 오른 것·내린 것 상위 5개. DELIVER 는 24클래스·얇은 객체 4클래스·큰 영역 3클래스 묶음 Δ 를 같이.

**④ 조건별 Δ** — DELIVER 5조건(조건당 379~380장) / MUSES 8셀(표본 수 병기 + §5-31b 규칙 명시).

**마지막에**:

- 무효로 버린 수치와 그 이유
- 재지 못한 항목("미확인"으로 명시)
- 원본 경로(ckpt · NAS 산출물 · 로그 · 제출 zip)
- **판정 요청을 질문 형태로** 쓴다. 예: "E1 확정 시드3 은 확정 게이트(3쌍 평균 Δtest ≥ +1.0 그리고 Δ24 ≥ +0.5)의 세 번째 쌍입니다. 3쌍 평균이 +1.12 / +0.82 가 되는데 통과로 판정하시겠습니까? 그리고 하네스 v2 값을 판정 분모로 쓸지 v1 로 둘지 결정이 필요합니다."

### 5-3. 회신을 받으면

1. 판정 문장을 분석 문서 §9 "판정" 절에 **그대로** 옮겨 적는다(요약·윤색 금지).
2. `experiments/registry.md` 의 상태 셀을 바꾼다(🟡 active → ✅ 완료 / 🔴 dead 등).
3. 판정이 헤드라인·정본 수치를 바꾸는 것이면 `status/current.md` 갱신과 노션 동기화를 **같은 날** 한다(CLAUDE.md §3 상시규칙).

---

## 6. 예시 — 실제 명령 시퀀스

### 6-1. MUSES 새 체크포인트 하나를 받았을 때 (hpca100 예시)

```bash
# 0) 환경 — 넷 다 필요. 하나만 빠져도 다른 얼굴로 실패한다
ssh hpca100
REPO=/home/jovyan/SSDb/jemo_maeng/src/drone-MemorySAM
export PYTHONPATH=$REPO/semseg/models/sam2:$REPO
export HF_HOME=/home/jovyan/.cache/huggingface
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/home/jovyan/SSDb/jemo_maeng/venv/p34/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
PY=/home/jovyan/SSDb/jemo_maeng/venv/p34/bin/python

# 1) 입력 확정 — md5, 그리고 학습 로그에서 백본 로드·시드·길이 확인
md5sum <CKPT>
grep -iE 'dinov3|RANDOM INIT|falling back|safetensors|fix_seeds|SEED|EPOCHS' <학습로그>   # RANDOM INIT 있으면 중단

# 2) 하네스 가드 + 빈 GPU
cd $REPO && $PY tools/eval_harness_guard.py --check
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 3) 공식 native 채점 (val 250장; hist_full=공식, hist_1024=트레이너 내부를 한 패스에서 동시 누적)
$PY tools/eval_muses_official.py --cfg configs/<...>.yaml --ckpt <CKPT> \
    --gpu <FREE> --out /tmp/jemo_scratch/muses_official_<RUN>

# 4) 전체·조건별 8셀·클래스별 19개를 report.json 에서 뽑는다
$PY -c "import json;print(json.dumps(json.load(open('/tmp/jemo_scratch/muses_official_<RUN>/report.json')),indent=1,ensure_ascii=False))" | head -100

# 5) 산출물 회수 (원시 혼동행렬 포함) — 허브에서 실행
rsync -rt --no-perms --no-owner --no-group hpca100:/tmp/jemo_scratch/muses_official_<RUN>/ \
  /drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/<RUN>_eval_<YYYYMMDD>/report/

# 6) (test 제출은 user·판정 세션 승인 후에만) 예측 PNG 생성 -> 규격 검사 -> zip 보관
$PY tools/predict_muses_test.py --cfg configs/<...>.yaml --ckpt <CKPT> --gpu <FREE> --out <OUT>/labelTrainIds
$PY tools/verify_submission.py <OUT>/labelTrainIds <MUSES_TEST_RGB_DIR>

# 7) 문서 작성 + 등록 (허브, repo 안에서)
#    analysis/YYYY-MM-DD-muses-official-<run>.md -> experiments/00_MOC.md 행 -> registry.md 행
#    -> (test 제출했으면) MUSES_TEST_RESULTS_INDEX.md 행

# 8) 판정 세션에 §5-2 네 표로 보고 (SendMessage)
```

### 6-2. DELIVER 새 체크포인트 하나를 받았을 때 (jarvis 예시)

```bash
# 0) 환경
ssh jarvis
REPO=/home/jemo_maeng/src/drone-MemorySAM-develop
export PYTHONPATH=/SSDb/jemo_maeng/pylibs_p34:$REPO/semseg/models/sam2:$REPO
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
conda activate MMSS_SAM && cd $REPO

# 1) 입력 확정 + 하네스 가드 + 빈 GPU
md5sum <CKPT>
python tools/eval_harness_guard.py --check
bash scripts/pick_free_gpus.sh 1

# 2) legal v1 재채점 — test / val. 장수(1897 / 2005)를 진행 표시줄로 검증
CUDA_VISIBLE_DEVICES=<FREE> python val.py --cfg configs/eval/<EVALCFG>.yaml --mode test --model_path <CKPT> 2>&1 | tee logs/<RUN>_legal_test.log
CUDA_VISIBLE_DEVICES=<FREE> python val.py --cfg configs/eval/<EVALCFG>.yaml --mode val  --model_path <CKPT> 2>&1 | tee logs/<RUN>_legal_val.log

# 3) legal v2 재채점 (ISSUE-036 병기용 — v2 채택 판정 전까지 둘 다)
CUDA_VISIBLE_DEVICES=<FREE> python tools/legal_rescore_v2.py --cfg configs/eval/<EVALCFG>.yaml --mode test --model_path <CKPT> 2>&1 | tee logs/<RUN>_legalv2_test.log

# 4) 24클래스·얇은 객체·큰 영역을 클래스별 원자료에서 재계산 (다른 문서에서 옮겨 적지 않는다)
#    24클래스 = (25 x mean - RailTrack) / 24
#    얇은 객체 = mean(Pole, Pedestrian, Static, TrafficLight) / 큰 영역 = mean(RailTrack, Wall, Water)

# 5) 조건별 5조건 (조건당 379~380장) -> per-class x per-domain 표
python tools/eval_per_domain.py --cfg configs/deliver/<TRAINCFG>.yaml --ckpt <RUN>=<CKPT> \
  --dataset-root /SSDb/jemo_maeng/dset/DELIVER --conditions cloud,fog,night,rain,sun \
  --split test --batch 1 --gpu <FREE> --out-dir <OUT>/perdomain
python tools/analyze_per_domain.py --logs-dir <OUT>/perdomain --label <RUN>=<로그prefix> --out <OUT>/perdomain/analysis.md

# 6) (헤드라인 후보·구조 변경 카드일 때만) 결측·열화 모달
python tools/missing_modality_eval.py --cfg configs/eval/<EVALCFG>.yaml --model_path <CKPT> \
  --split val --protocol all --expected_clean_miou <legal val> --batch 1 --subset_every 1 \
  --out <OUT>/missing_modality

# 7) (구조를 바꾼 카드일 때만) 모듈 토글 — available/skipped 로그를 반드시 남긴다
python tools/module_diagnostics.py --cfg configs/deliver/<TRAINCFG>.yaml --model_path <CKPT> \
  --dataset-root /SSDb/jemo_maeng/dset/DELIVER --conditions cloud,fog,night,rain,sun \
  --split test --max-imgs 120 --ablate-n 20 --gpu <FREE> --out <OUT>/modulediag

# 8) 산출물·ckpt NAS 회수(md5 대조) -> 문서 작성 -> 00_MOC·registry 등록 -> 판정 세션 보고
```

---

## 7. 충돌 · 미확인

### 7-1. 기존 문서와 어긋나는 규칙 (이 문서가 어느 쪽을 택했는지)

| 항목 | 어긋남 | 이 문서의 처리 |
|---|---|---|
| 분석 산출물 저장 위치 | `tools/README_seg_analysis.md` 본문 상단은 `/mnt/HDD2/src/logs/<model>_eval_<YYYYMMDD>/` 라고 적고, 같은 문서의 2026-07-21 갱신 절과 `infra/artifact-locations.md` 는 `/drone_nas/.../analysis_logs/` 라고 적는다 | **`/drone_nas/.../analysis_logs/` 가 정본**(HDD2 는 ISSUE-023 으로 폐기). README 상단 문구는 잔재다 |
| 체크포인트 선택 규칙 | seg-analysis SKILL.md §2 는 "논문 수치는 val-best만"이라고만 적고 N6(legal-val 재선택) 규칙을 모른다 | 정본 = 학습기 val-best(top1) 고정(2026-09-16). N6 는 **규칙 이름을 붙여 보조로만** 병기 |
| legal 하네스 재샘플 | seg-analysis SKILL.md·README 는 ISSUE-036(nearest floor 정렬 편차)을 아직 모른다 | §2-1-2 대로 v1·v2 둘 다 돌린다 |
| 24클래스·얇은 객체 묶음 | 두 도구 문서 어디에도 없다(카드 문서에만 있다) | §2-1-4 에 계산식을 옮겨 적었다. 도구가 자동 계산하지 않으므로 **손으로 재계산**해야 한다 |
| G1 의 뜻이 세 가지 | 카드 §0(분모 = 기준선) / P53 제안·전략탐색(분모 = E1 확정 시드1 24클래스 55.26) / P52 게이트(DELIVER ≥ 54.95−0.3) | 보고할 때 **어느 체계의 G1 인지 같은 행에 적는다.** 신규 카드의 G1 분모는 E1 확정 시드1(24클래스 55.26)로 통일하기로 결정됐다(카드 §5-34 (1)) |
| MUSES 조건별 표본 규칙의 적용 범위 | §5-31b 는 MUSES 전용이고 DELIVER 에는 적용하지 않는다 | 표 아래에 매번 명시 |
| 스크린 파이프라인의 조건 문자열 | seg-analysis SKILL.md 는 MUSES 를 `--conditions clear,fog,rain,snow,day,night`(6조건 축)로 돌리라고 하고, 공식 채점기는 8조합 셀을 한 패스에서 낸다 | 공식 수치는 `eval_muses_official.py` 8셀이 1차. 파이프라인 6조건은 진단용 |

### 7-2. 미확인 (문서·코드에서 확인하지 못했다 — 추측으로 채우지 않았다)

1. **Codabench 제출 횟수 제한**("1일 1회" 등)은 리포 문서에 없다. `status/current.md` 에 있는 것은 "MUSES 제출 게이트 = 공식 val ≥ 82.62 일 때만 1회"라는 **내부 게이트**이며, 플랫폼의 일일 한도는 미확인이다. 제출 전에 Codabench comp 14005 페이지에서 직접 확인할 것.
2. **`module_diagnostics.py`·`viz_features.py`·`missing_modality_eval.py` 의 실측 소요 시간과 GPU 메모리**는 문서에 없다. 아는 값은 `seg_analysis_pipeline.py` 8스테이지 = 조건당 약 10분 · 전체 80~120분(SKILL.md §1)뿐이다.
3. **`val.py` legal 재채점 1회의 소요 시간**은 로그 한 건에서 역산한 값만 있다: bengio E9 test 1897장 · 3.05 s/it(BS1). 서버·GPU 별 실측은 미확인.
4. **MCubeS 평가의 정확한 실행 명령**(어떤 config·스크립트로 102장을 채점하는지)은 확인하지 못했다. 규약(final-epoch 기준 · 로더가 `list_folder/test.txt` 를 읽음 · 3시드 평균 Δ ≥ +0.5)만 확인했다.
5. **MUSES 제출 zip 을 실제로 묶는 명령**(`labelTrainIds/` 구조로 zip 하는 절차)은 `verify_submission.py` 의 검사 항목에서 역추적했고, 전용 스크립트가 있는지는 미확인이다.
6. **우리가 직접 재학습한 MUSES 기준선은 없다**(DELIVER 만 DGFusion·CAFuser 재학습본이 있다). MUSES 의 SOTA 비교는 전부 공개 수치 대비다.
7. **`eval_per_domain.py` 의 MUSES 조합 셀 지원**은 SKILL.md 가 `fog_night` 식 지정이 가능하다고 적고 있으나(`semseg/datasets/muses.py` CASE 조합, dee524f), 코드에서 직접 확인하지 않았다. MUSES 조건별은 `eval_muses_official.py` 가 한 패스에서 누적하므로 그쪽을 1차로 쓴다.
8. **DELIVER 조건당 379~380장**은 `meta/experiment-glossary.md` §3 G3 의 기재를 옮긴 것이고, 데이터셋에서 직접 세지는 않았다.
9. **DELIVER 얇은 객체 4클래스 게이트(G2, E1 대비 Δ ≥ +2.0)** 는 P53 제안 문서 기준이다. 다른 벤치의 대응 게이트는 정의돼 있지 않다.

---

> 관련: `.claude/skills/seg-analysis/SKILL.md`(도구 매핑·피쳐 특성화) · `tools/README_seg_analysis.md`(도구 사용법) · `decisions/2026-09-07-daily-cycle-experiment-cards.md` §0·§0-1·§0-2·§5-31b·§5-34(판정 규약) · `meta/experiment-glossary.md`(약어) · `infra/artifact-locations.md`(원본 위치) · `infra/environment.md`(실행 환경) · `.claude/skills/notion-experiment-log/SKILL.md`(노션 반영)
