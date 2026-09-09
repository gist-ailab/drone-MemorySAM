---
created: 2026-09-07
author: fable (background 세션 "MMSAM | 생각정리") — user 요청: "빈 GPU에 하루 안에 결과를 보고 바로 다음 실험을 할 수 있게, 실험 계획을 미리 짜고 코드를 각각 구현해 돌리자"
status: 🟡 설계 완료 — 코드 구현 워커 선택·GPU 배정 대기
depends: decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md §3.5·§3.6 (구조 사다리 S0~S6, 클래스 혼동 진단)
---

# 일일 사이클 실험 카드 — DELIVER 클래스 혼동 + MUSES 공통 한 단계 상승 (2026-09-07)

> **user의 문제 정의**: DELIVER는 분할 형상은 정성적으로 괜찮은데 **클래스를 헷갈려서** 점수를 잃는다. 센서마다 클래스 근거가 다르므로 어댑터가 그것을 배우게 하면 오를 것이다. MUSES는 압도적 SOTA가 아니다. 두 벤치에 공통으로 한 단계 올리는 방법이 필요하다.
> **설계 원칙**: 카드 하나 = 변수 하나 = 하루. 스크린은 40 epoch, 확정은 200 epoch 3페어. 라우팅·게이트 계열(H1·H3·H16 반증)은 카드에 넣지 않는다. 클래스 근거의 센서 차이는 **학습 손실과 특징 접점**에서 다룬다.

---

## 0. 공통 스크린 규약

| 항목 | 값 | 근거 |
|---|---|---|
| 벤치·레시피 | DELIVER 4센서(img/depth/event/lidar), P46 C3-only λ0.1, PhysAug off, DGFUSION_AUG on, 768² 학습 | 현 최선 고정팔 |
| 시드 | 20260821 매칭(모든 카드 동일) | 페어 편차 ~0.6 |
| 길이 | **40 epoch**, EVAL_INTERVAL 5 | P50 val-best ep30 전례. yeon 4090×2 ≈ 30분/ep → 20h, hpca100 A100×4 ≈ 9분/ep → 6h |
| 판정 수치 | 40ep 이내 legal-val 최고 ckpt를 `val.py` native-GT BS1(하네스 가드)로 test 재채점 | ISSUE-033 규약 |
| 기준선 | **B0 = 같은 규약의 현행 레시피 40ep 스크린**(1회 실행 후 고정) | 카드 간 공통 대조 |
| 스크린 통과 | Δtest(카드 − B0) ≥ **+1.0** and 악조건(night·fog·rain) 어느 것도 −0.5 미만 아님 | 페어 설계 검정력(±1.0) |
| 확정 | 통과 카드만 200ep × 3페어, 게이트 = 3페어 mean ≥ +1.0 | 축 0 규약 |
| 조기 kill | ep20 legal-val이 B0 ep20 −1.5 미만 | 비용 절감 |

카드마다 구현은 **토글 하나**(config 키)로 켜지고, 기본값 off에서 forward가 byte-동일해야 한다(스모크: state_dict 키 + |Δ|max=0). 코드 검수 파이프라인(conventions.md)을 따른다.

---

## 1. 카드 목록 (우선순위·일정 순)

### 1일차 — 학습 0 또는 config만

| 카드 | 가설 | 변경(1개) | 구현 | 시간 | 통과 기준 |
|---|---|---|---|---|---|
| **B0** 기준선 스크린 | — | 없음 | config만(EPOCHS 40, EVAL_INTERVAL 5) | 6~20h | — (대조군) |
| **E0** 특징 정보 프로브 (S0) | 클래스·센서 정보가 DINOv3 원 특징에는 있는데 어댑터·헤드가 버린다 | 학습 = 선형 헤드만 | `tools/probe_feature_info.py`: (i) 블록 6/12/18/24 원 특징 concat vs 어댑터 후 per-modal 특징 vs fused 특징 위 **클래스 선형 프로브**(val·test per-class recall, 특히 RailTrack·Wall·Water·Bridge) (ii) 같은 특징 위 **오라클 센서 선택 프로브**(H16 정답 부분집합) | 반나절 | (i) 원 특징의 붕괴 클래스 recall − fused 특징 recall ≥ +10%p → "어댑터가 버린다" 확정, E1·E2·E5 정당화. (ii) 원 특징에서도 우연 수준 → 학습형 라우팅 영구 폐쇄(H26 ✗) |
| **E9** 검증셋 사전 로짓 보정 | 혼동은 클래스 사전의 도메인 이동 | 추론 시 logit − τ·log(p_val) | `val.py` 옵션, 학습 0 | 1h | Δtest ≥ +0.5면 무료 이득으로 채택 |
| **E7** MUSES PhysAug-off 기준선 | 공정선 정합(user 지적) | PHYSAUG off | config만 | 하루(MUSES 40ep 스크린) + 200ep 3페어는 확정 단계 | 헤드라인 교체용, 게이트 없음 |

### 2일차 — 특징 접점 (구조 사다리 S1·S2)

| 카드 | 가설 | 변경 | 구현 | 통과 |
|---|---|---|---|---|
| **E1** 4탭 읽기 (S1) | 기하 센서(depth·lidar)의 클래스 근거는 중간층에 있다 | 블록 6/12/18/24 출력을 SimpleFPN 레벨별로 투입 | `model.py`: P43 lateral을 M2F 헤드와 **분리**해 단독 토글(`MODEL.TAPS: [6,12,18,24]`), 투영은 non-zero init | Δtest ≥ +1.0. 예측: depth·lidar 의존 클래스(Wall·Water)에서 이득 집중 |
| **E2** 전 선형층 LoRA (S2) | Q/V만으로는 센서별 클래스 근거를 담을 용량이 없다 | LoRA를 Q/K/V/O + MLP fc1/fc2, r32, α64 | `encoder.py`: `MultiModalLoRAQKV` 확장 + `MultiModalLoRALinear`(O·fc1·fc2), `LORA_TARGETS: [qkv, proj, fc1, fc2]` | Δtest ≥ +1.0. ep20 RGB-only −0.5면 kill |

### 3일차 — 클래스 혼동 직격 (user 가설의 손실 구현)

| 카드 | 가설 | 변경 | 구현 | 통과 |
|---|---|---|---|---|
| **E3** 센서별 클래스 prototype (C3-M) | 클래스 정체성을 **센서마다 독립으로** 지지하게 하면, RGB가 헷갈리는 클래스를 lidar·depth의 근거가 붙잡는다 | prototype bank를 센서별로 분리(`_last_per_modal_feats` 위, K×D×M) + 센서 간 prototype 일치 항 λ_agree | `p46.py`: `P46.C3_PROTO.SRC: permodal`, `AGREE_LAMBDA` | Δtest ≥ +1.0. 예측: RailTrack 유지 + Wall/Water 회복. **이것이 user 가설의 직접 검증** — 어댑터가 센서별 클래스 근거를 배우는지 |
| **E4** 혼동 쌍 margin 손실 | 붕괴는 특정 쌍(RailTrack→Sky/Static, Wall→Building 등)으로 흡수되는 형태 | 시작 시 검증셋 혼동행렬에서 top-k 쌍 고정 → 쌍별 prototype/로짓 margin | `p46.py`: `CONFUSION_PAIRS: auto_k5`, margin m | Δtest ≥ +1.0, 표적 쌍 IoU 회복 |
| **E4b** 혼동 쌍 margin — 명시 쌍판 | E4 auto 쌍(ep6 검증 200장 top-5: Ground→SideWalk, Water→Terrain, Other→Fence, Other→SideWalk, Static→Building)에 **RailTrack이 빠졌다**. RailTrack은 DGFusion 대비 진짜 격차 클래스(Wall·Water·Bridge는 DGFusion도 test 0~4 공통 붕괴 — [2026-07-28 §68](2026-07-28-p46-classtransfer-recovery-proposal.md)) | PAIRS 명시 [RailTrack→Sky, RailTrack→Static, RailTrack→Terrain, Wall→Building, Water→Terrain], 나머지 E4 동일 | `configs/bengio-…_screen40_E4b.yaml` (2026-09-08 추가) | Δtest ≥ +1.0 **and** legal test RailTrack IoU > B0 31.98 |
| **E8** 클래스 표적 copy-paste 증강 | 붕괴 클래스의 학습 노출 부족 | RailTrack·Wall·Water·Bridge 인스턴스를 4센서 동시 copy-paste | 로더 `deliver.py` 옵션 | Δtest ≥ +1.0 (C1 RCS 실패와 구분: 샘플링이 아니라 합성) |

### 4일차 이후 — 구조 (구현 2~5일, 스크린은 하루)

| 카드 | 가설 | 변경 | 구현 | 통과 |
|---|---|---|---|---|
| **E5** 변형 가능 다중스케일 픽셀 디코더 (RF-DETR·Mask2Former 픽셀 디코더 유형) | 4탭을 SimpleFPN 합산이 아니라 **deformable attention**으로 읽어야 얇은 클래스가 산다 | SimpleFPN → MSDeformAttn 픽셀 디코더(3층), 출력은 여전히 dense 픽셀 헤드(질의 경로 없음) | 신규 모듈 `pixel_decoder.py`, E1 위에서만 | Δtest ≥ E1 + 0.5 |
| **E6** 매 블록 센서 간 교환 어댑터 (S3) | 후기 융합이 병목. 인코딩 중 센서 간 저랭크 교환이 필요 | 각 블록 attn 뒤·FFN 뒤에 StitchFusion-MoA식 양방향 저랭크 MLP, 백본 frozen | `encoder.py` 후크. **P51-CMLC(−0.82)와의 차이**(블록 출력 교환 vs LoRA 코드 결합, non-zero init, 양방향)를 제안서에 선행 기술 | Δtest ≥ +1.0, fog·night 비악화 |
| **E10** 상위 절반 블록 부분 FT (S4) | P49 실패는 전체 FT 때문 | 블록 13~24만 unfreeze, LLRD 0.9, top lr 1e-5 | 옵티마이저 그룹 | Δtest ≥ E2 + 0.5, RGB-only −0.5면 kill |

RF-DETR에 대한 판단: RF-DETR의 이득은 DINOv2 백본 + **다중 스케일 deformable 디코더**에서 나왔고, 객체 질의 자체는 우리 세그 스택에서 세 번(P30·P38·P43) 무효였다. 세그로 옮길 수 있는 부분은 E5(deformable 픽셀 디코더)이며, 질의 기반 클래스 디코딩은 카드에 넣지 않는다.

---

## 2. 카드가 답하는 질문의 구조

```
E0 원 특징에 정보가 있는가?
 ├─ 있고 어댑터가 버림 → E1(어느 층) · E2(어느 가중치) · E5(어떻게 읽나) · E6(언제 섞나)
 └─ 원 특징에도 없음 → 구조 카드 중단, 손실·데이터 카드(E3·E4·E8)와 분석 논문
E3/E4 클래스 혼동은 센서별 근거·쌍 분리로 풀리는가?  ← user 가설의 직접 검증
E7 MUSES 정직한 기준선 (공통 상승분 측정의 출발점)
```

MUSES 공통 상승: DELIVER 스크린 통과 카드를 MUSES 3센서(PhysAug off, E7 기준선)에 같은 규약으로 옮겨 검증한다. 두 벤치 모두 +1.0이면 "공통 한 단계"로 인정.

---

## 3. 실행 절차 (매일)

1. 아침: 전날 스크린 결과를 `val.py`로 재채점 → 카드 표에 Δtest 기입 → 통과/폐기.
2. 통과 카드는 확정 큐(200ep 3페어)로, 폐기 카드는 원장에 1줄.
3. 빈 GPU에 다음 카드 기동(sonnet: `remote_exp.sh status` → `run ... auto:N`, 기동 검증 5항목). **환경변수 `PYTHONUNBUFFERED=1` 필수** — tee 파이프 시 파이썬 stdout 블록 버퍼링으로 `[E2] lora_targets=…` 같은 1회성 기동 로그가 flush 전까지 로그 파일에 안 보인다(2026-09-07 hpca100 E2·bengio E0 실측). `remote_exp.sh`가 git을 부르는 환경(worktree 격리 세션)에서는 `ssh <host>` 직접 실행.
   1일차 실측 속도: DELIVER 40ep — bengio 3090×4 ≈12.5분/ep(8.3h), 3090×2 ≈26분/ep(17.5h), hpca100 A100×2 ≈22분/ep(14.8h) · MUSES 40ep — A100×1 ≈14.5분/ep(9.7h).
4. 구현 워커: 카드별 지시문을 이 문서에서 복사해 `labcode -p` / `glmcode -p`로 위임, 이 세션이 diff 검수 + 스모크(기본 off byte-동일).
5. **속도 실측(2026-09-08, yeon 4090 24GB, DELIVER 768² 4센서 BS1)**: 학습 1 epoch ≈26분, **학습 중 평가(Val 2,005장 + Test 1,897장, BS1) ≈28분/2 epoch → 벽시계의 ≈35%가 평가**. **BS2는 첫 backward에서 즉시 OOM(22.3GiB 할당)** → 4090에서는 배치를 못 키운다(학습 중 3GiB로 보이는 순간은 평가 구간). 처방: 새 config는 `EVAL.BATCH_SIZE 4`(BS 불변성 ISSUE-033에서 확인됨)·`EVAL_INTERVAL 5`·**학습 중 Test 평가 끄기**(선택에 안 쓰는 test-peeking, 정본은 오프라인 재채점)로 평가 비중을 10% 미만으로. 진행 중 런은 재시작 비용(수십 시간)이 커서 그대로 둔다. A100 40GB(hpca100)에서는 BS2 가능성 있음(미실측).

## 4. 등재
- plan.md 대기열에 카드 행 추가(B0·E0·E7·E1·E2·E3·E4·E8 순), 각 행 EPOCHS 40 명시.
- 카드 결과는 `experiments/analysis/2026-09-XX-daily-cards-<E>.md` 1건씩.

- 🔴 **`--gpu` 함정(2026-09-08 두 번째 사고)**: `tools/eval_muses_official.py`(49행)와 `tools/probe_feature_info.py`는 `--gpu N`으로 `CUDA_VISIBLE_DEVICES`를 통째로 덮어쓴다. 셸에서 `CUDA_VISIBLE_DEVICES=2`를 주고 `--gpu 0`을 겹쳐 주면 **절대 인덱스 0(타 사용자 GPU)**에 올라간다. 둘 중 하나만, 그리고 `--gpu`에는 절대 인덱스를 준다.

### 3-6. 2026-09-08 저녁 배치 계획 (분담: bengio = 이 세션, hpca100 = 감시 세션)

| 시각(KST) | 서버·GPU | 작업 | config |
|---|---|---|---|
| ~~지금~~ | ~~bengio GPU0~~ | ~~조기 legal test 재채점~~ **취소** — 16:3x 다른 사용자가 GPU0 점유(빈 자리 즉시 상실, gpu-never-idle 실증) | — |
| ✅ 20:5x 기동 | bengio GPU6,7 | **B0s2** 기동 검증 통과(peak 13.9GiB, 1.34it/s, RANDOM INIT 없음) | `bengio-…seed20260902_screen40_B0s2.yaml` |
| E3 완주(≈00:00) | bengio GPU4,5 | **E1s2**(bengio판) | `bengio-…seed20260902_screen40_E1s2.yaml` |
| ✅ 20:5x 기동 | bengio GPU1-3 | **E4b** 기동 검증 통과(`[E4] 명시 pairs=[RailTrack→Sky, RailTrack→Static, RailTrack→Terrain, Wall→Building, Water→Terrain]` 확인) | `bengio-…seed20260821_screen40_E4b.yaml` |
| 17:26 | hpca100 GPU1 / GPU3 | E2 legal 재채점 / **E1M**(MUSES 4탭) | `hpca100-muses_…_taps_screen40_E1M.yaml` |
| 20:26 | hpca100 GPU2 | E7·E7c 공식 재채점 → 이후 E1s2(hpca100판) | `hpca100-…seed20260902_screen40_E1s2.yaml` |
| 20:35 (E2 판정 후) | hpca100 GPU1 | **E12**(E1+E2 결합) — E2 회색지대(+0.72)지만 val +3.41·E1 val 신기록이라 축 가산성 확인이 최우선 | `hpca100-…seed20260821_screen40_E12.yaml` |
| MUSES 페어 재채점 후(≈22:00) | hpca100 GPU2 | **E2s2**(E2 시드2 페어, 회색지대 판별) — E1s2(hpca100판)는 yeon으로 이관 | `hpca100-…seed20260902_screen40_E2s2.yaml` |
| ~~지금~~ | ~~yeon GPU7~~ | ~~E1s2(yeon판)~~ **취소** — 기동 직전 타 사용자 프로세스(openpi, 17.2GiB)가 점유. config는 develop에 유지, 빈 자리 생기면 재시도 | `yeon-…seed20260902_screen40_E1s2.yaml` |
| ✅ 20:5x 기동 | bengio GPU0 | E1 ep35 test→val→E4 ep15 test→val 순차 legal 재채점(분할당 ≈1h35, E1 test ≈22:15, 전체 ≈03:00) | `configs/eval/bengio-…_eval1024_{E1,E3,E4}.yaml` |

- hpca100은 E7c 완주(20:26)까지 pull 금지(§1.5). 코드가 이미 develop과 동일(cddc319)함을 감시 세션이 md5로 확인했고, config 4벌은 파일 복사로 배치됨.
- 재채점 규약: 각 카드의 **학습 config를 기반**으로 EVAL/TEST 네 항목(IMAGE_SIZE 1024, BATCH 1, TEST.FILE)만 바꾼 eval config를 쓴다(MODEL 블록 동일 → strict load). B0 블록용 `hpca100-deliver_rgbdel_P46_eval1024_legal.yaml`은 B0 계열에만.

## 5. 결과 기록

| 카드 | 결과 | 판정 | 근거 |
|---|---|---|---|
| E9 τ=0 (재채점) | test 53.57 = 기존 정본(구 53.57) 일치 | 하네스 재동결 완료(636e490) | bengio `logs/e9_la_tau0_*.log` |
| **E9 검증셋 사전 로짓 보정** | τ=0 53.57 / τ=0.5 53.45 / τ=1.0 51.89 (mAcc 63.8→67.1→69.2) | **폐기** — mIoU 단조 하락, 게이트 +0.5 미달. 혼동은 클래스 사전 이동이 아니라 재현율↔정밀도 교환으로만 반응 | bengio `logs/e9_la_tau{0.5,1.0}_*.log` |
| E0 특징 정보 프로브 | raw 27.8 < adapted 35.6 < fused 45.1 (test 선형 mIoU); depth 중간층 탭에 Water 47.6·RailTrack 23.9 잔존(fused 16.6·0.0) | 게이트(i) 미달(−8.2%p) → "어댑터가 버린다" 기각. **E5·E6 하향, E1·E3 유지**. 오라클 프로브 미실행 | [analysis/2026-09-08-daily-cards-E0-feature-probe.md](../experiments/analysis/2026-09-08-daily-cards-E0-feature-probe.md) |
| B0 기준선 40ep | legal val **64.97**(mAcc 74.73) / test **53.78**(mAcc 63.87), ckpt `epoch40_65.4_top1` | 통과선 test **54.78**, 폐기선 54.28 확정. 트레이너 val(65.4)이 공식 val보다 **+0.43** 높다(보정값). 클래스별: val→test 낙차 11.19 중 61%가 Wall(71.7→12.2)·Bridge(58.4→0.0)·Water(59.6→6.5) — 단, 이 셋은 DGFusion도 test 0~4로 공통 붕괴(2026-07-28 §68)라 SOTA 격차 원인이 아님. 진짜 격차 = RailTrack(19.4→32.0, DGFusion보다 낮음) | bengio `logs/b0_eval_{val,test}_20260908_*.log` (감시 세션 재채점) |
| **E1 4탭 읽기 (완주·legal test 확정)** | 트레이너 val ep5~40 = 60.98/63.56/66.30/66.57/66.43/66.88/**67.25**/66.74 vs B0 …/65.4. val-best **ep35 67.25**. **legal test 54.85**(mAcc 65.42, 1897장·1024·BS1), legal val 재채점 진행 중 | ✅ **스크린 통과 — Δtest +1.07 ≥ +1.0(첫 통과 카드)**. 클래스별 표·val 수치 확인 후 확정 기록. 확정 런(200ep×3페어)은 E12(E1+E2) 결과(09-10 아침) 본 뒤 대상(E1 단독 vs E12) 확정 후 착수; 그 사이 E1s2(bengio GPU4,5)로 시드 판별 보강 | bengio `logs/e1_taps_screen40_20260907_223704.log`, `logs/e1_eval_{test,val}_ep35_*.log` |
| E3 센서별 prototype (완주, 재채점 대기) | 트레이너 val ep5~40 = 59.35/63.72/62.23/65.44/**65.97**/65.51/65.61/65.69 vs B0 …/65.4. val-best **ep25 65.97**, 카드 넷 중 최저(E2 68.65 > E1 67.25 > E3 65.97 ≈ E4 65.92) | 🟡 보류(트레이너 val +0.6 이내, 후반 평평). legal 재채점(ep25 ckpt) GPU0 대기열 | bengio `logs/e3_permodal_screen40_*.log` |
| **E4 혼동 쌍 margin auto (완주, 재채점 중)** | 트레이너 val ep5~40 = 59.77/63.17/**65.92**/65.70/65.79/65.62/65.70/**64.99**(B0 ep40 65.4). val-best **ep15 65.92**, 이후 정체 뒤 ep40에서 하락 | 🟡 보류(트레이너 val 기준 이득 0.5 이내, 후반 하락 — margin 항이 후기 학습을 방해했을 가능성, 판정은 ep15 ckpt legal test로). **auto 쌍에 RailTrack 누락** → 명시 쌍판 E4b를 같은 GPU1-3에 기동 | bengio `logs/e4_confmargin_screen40_20260908_085306.log`, `logs/e4_eval_{test,val}_ep15_*.log` |
| **E2 전 선형층 LoRA (완주·재채점 완료)** | 트레이너 val ep40 **68.65**(궤적 62.54/63.44/60.09/62.40/65.25/66.91/67.85/68.65). **legal val 68.38 / test 54.50** (ckpt `epoch40_68.65_top1`, 1024·BS1). Δ vs B0: val **+3.41**, test **+0.72**. 클래스별 test: RailTrack 31.98→**47.75**(+15.77), TrafficLight 30.23→**42.72**(+12.49), TrafficSign +2.89, Truck +2.68 / Wall 12.17→5.77(−6.40), Water 6.46→0.99(−5.47), Static 28.63→23.19(−5.44) | 🟡 **회색지대(+0.5 ≤ Δ < +1.0) → 판정 보류, 시드2 페어(E2s2)로 판별**. 해석: 용량 확장이 진짜 격차 클래스 RailTrack을 실제로 풀지만(DGFusion 64.47과의 격차 60→17), 이득 전부가 RailTrack·TrafficLight 두 클래스(+1.13 mIoU 기여)에서 나오고 Wall·Water·Static(−0.69 기여)이 절반을 되받아간 상쇄 결과. val→test 전이율 21%. 트레이너 val − 공식 val = 0.27(B0 0.43과 같은 방향, 보정값 안정). 후속: E12(E1+E2 결합)로 축 가산성 확인, E2s2로 시드 판별 | hpca100 `logs/hpca100_E2_eval_{test,val}_20260908_083859.log` |
| E7 vs E7c MUSES PhysAug 대조 (둘 다 완주) | 트레이너 val ep5~40 페어 차이(켬−끔) = −0.66/+0.13/−1.09/+0.27/+0.38/−0.32/+0.27/**−0.11**(ep40: E7 80.29 vs E7c 80.18) | ✅ **확정: PhysAug 효과 없음(근소 손해)** — 공식 native(250장, 1080×1920) E7 **80.0756** vs E7c **79.9417**(−0.13), letterbox 1024 80.29 vs 80.18(−0.11), 트레이너 val −0.11로 세 축 일치. 클래스별: E7c가 train +0.31·bicycle +1.30·terrain +0.70에서 앞서고 rider −1.53·motorcycle −2.10·pole −0.75·person −0.63·traffic light −0.53에서 뒤짐(작은 객체에서 오히려 손해). → **MUSES 레시피 PhysAug-off 통일**(공정성 문제 §3.5-1 해소, 성능 손실 없음). 각주: 조건별로는 snow/night만 E7c 73.89 > E7 72.93(+0.96) — 야간·악천후 증강이 가장 어려운 구간에서만 작동하나 전체 결론을 뒤집을 크기 아님 | hpca100 `~/SSDb/jemo_maeng/muses_official_eval_20260908/{E7,E7c}`, `logs/muses_official_{E7,E7c}.log` |
| E7 MUSES PhysAug-off (완주) | 트레이너 val **80.29**@ep40 | 🔵 E7c(PhysAug-on, ep23/40) 완주(20:26 KST) 후 두 val-best ckpt를 `tools/eval_muses_official.py`로 페어 재채점해야 PhysAug 효과 확정 | hpca100 `logs/hpca100_E7_launch.log` |
| **E1M MUSES 4탭 (재개·완주)** | 트레이너 val ep5~40 = 71.73/75.55/77.57/78.21/79.56/80.08/**80.64**/80.49(val-best ep35). `/tmp` 우회 후 재개해 완주. **공식 재채점 80.416** vs E7(PhysAug-off 기준선) 80.0756 → **Δ +0.34** | ❌ **폐기 — 스크린 통과선 +1.0 미달**. DELIVER에서 +1.07로 통과한 E1(4탭)이 MUSES에서는 이득의 3분의 1에 그쳤다. 4탭 읽기의 이득이 데이터셋에 의존한다는 뜻이므로, 확정 런 대상 선정 때 DELIVER 결과를 MUSES로 옮겨 적지 말 것 | hpca100 `/tmp/jemo_scratch/logs/hpca100_E1M_resume.log`, 공식 `muses_official_E1M.log`, 리포트 `~/SSDb/jemo_maeng/muses_official_eval_20260908/E1M/report.json` |
| E4b 혼동 쌍 명시(진행) | 트레이너 val ep5 59.66 (B0 58.71, E4 auto 59.77) | 🔵 진행 | bengio `logs/e4b_confmargin_explicit_screen40_*.log` |
| hpca100 디스크 우회(2026-09-09 00시) | SSDb 2.0T 중 우리 몫 538G(dset 327G·src 143G·cache 56G), 타 사용자 1.45T. `/` 오버레이(/tmp) 362G 여유 발견 → SAVE_DIR·로그를 `/tmp/jemo_scratch/`로 돌려 **E12(GPU1)·E2s2(GPU2) 기동 검증 통과**(45.6분/ep 단일 GPU, 완주 09-10 07~08시) | ✅ 삭제 없이 복구. 위험: /tmp는 컨테이너 재시작 시 소실 → 5ep마다 val-best ckpt 허브로 회수 지시. 정본 ckpt(E2·E7·E7c val-best) NAS 회수 지시. 중간 ckpt 삭제는 사용자 승인 대기 | 감시 세션 |
| MCubeS P52 seed1 (완주) | val-best 58.18@ep174 / final 57.96; 3시드 58.07±0.49 | P46과 동률 — P52 컨트롤러 이득 없음(감사 결론 재확인) | yeon |

> 🔗 **노션 동기화(2026-09-08)**: 이 표는 노션 논문 페이지(`Drone Object Detection for RGB-IR Fusion`, 33d05310) §4.2 카드 표와 같은 내용이다. 판정이 바뀌면 둘을 같은 날 갱신한다(CLAUDE.md §3, 빌더 `.claude/skills/notion-experiment-log/paper_page_builder.py`).
