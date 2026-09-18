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
