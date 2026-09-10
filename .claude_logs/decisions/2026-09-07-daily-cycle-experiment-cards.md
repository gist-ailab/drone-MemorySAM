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
| 🔴 보강(2026-09-09) 부기준 | **RailTrack 제외 24클래스 mIoU Δ ≥ +0.5**를 통과의 부기준으로 병기 | 40ep B0의 RailTrack test 31.98은 200ep 기준선(67.7~72.0, analysis/2026-08-06)의 절반 = 스크린 구간은 RailTrack 학습 곡선의 급경사라 "RailTrack을 빨리 배우는" 카드가 전부 +1로 보인다(§5-3). **추가 실측(09-10)**: 같은 레시피 E2 시드1 RailTrack 47.75 vs 시드2 4.36 — 시드 분산 43점, 단일 시드 RailTrack 값은 어떤 판정에도 쓸 수 없음 |
| 🔴 보강(2026-09-09) 중간 epoch 비교 금지 | 판정·시드 페어 비교는 **40ep 완주(val-best) 지점에서만**. 중간 epoch 이득 폭은 인용 금지 | B0 vs B0s2 ep5/10/15 = +1.00/−2.36/+1.36, 진폭 3.72 > 카드 간 차이(0.7) → 중간 지점 이득은 잡음(§5-3) |
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

**2026-09-09 재개 배치(사용자 "재개해줘" 이후)**

| 시각(KST) | 서버·GPU | 작업 | config / 근거 |
|---|---|---|---|
| 지금 | jarvis GPU5 → GPU5,7 | DGFusion 80k test 평가(8분) → **E3s2**(E3 시드2, DDP 2) | `jarvis-…seed20260902_screen40_E3s2.yaml` |
| 지금 | hpca100 GPU3 | **B0s2 재개**(bengio ep5 ckpt를 /tmp로 복사, AUTO_RESUME, 단일 A100 ≈26h) — 시드2 페어 공통 기준선 | bengio B0s2 config의 SAVE_DIR·ROOT 사본 |
| 20:00 E3b 완주 | jarvis GPU1,2,4 | **E13**(E1+E3 결합, 시드 821) + E3b val-best legal 재채점 병행 | `jarvis-…seed20260821_screen40_E13.yaml` |
| 09-10 07~08시 | hpca100 GPU1,2 | E12·E2s2 완주 → legal 재채점 → 확정 런 대상 선정 | — |
| 다음 빈 자리 | — | E1s2 재개(bengio ep5) → E4b 재개(bengio ep10) 순 | plan.md 「bengio 중단 런」 |

**2026-09-10 새벽~낮 배치(09-09 23시 결정)**

| 시각(KST) | 자리 | 작업 | 근거 |
|---|---|---|---|
| 02:30 E13 완주 | jarvis GPU1 / GPU2,4 | E13 legal 재채점(test→val) / **E1 확정 런 200ep 시드 20260821**(`jarvis-…seed20260821_E1_confirm200.yaml`, 기존 200ep seed821 기준선과 매칭) | §5-3: E3 확정은 보류(RailTrack 천장), E1이 24클래스 유일 양수 |
| 08:30 E4b 완주 | jarvis GPU5 | E4b legal 재채점 → 이후 E1 확정 런 3장으로 확장 여부 | — |
| 10:45~14:30 | jarvis 7, hpca100 0·3·2·1 | E1s2·E3s2·B0s2·E2s2·E12 완주 → 각 legal 재채점(40ep 완주 지점 페어) | 아침에 재배치 |

**2026-09-10 낮 결정**

| 시각 | 자리 | 작업 |
|---|---|---|
| ✅ 14:0x | jarvis GPU1+5 | E13 확정 런 2장 DDP 재기동 완료 — `Resumed (epoch 20)`, 30.2→**13분/ep**, 완주 **09-12 오전**(1장이면 09-14). E1 확정 런(GPU2,4) 유지 |
| 13:10~15:20 | hpca100 GPU1·2·3 | E2s2·E12·E1s2·B0s2 완주 → 각 legal 재채점(시드2 페어 판정은 B0s2까지 끝난 뒤 한 표로) |
| 재채점 후 | hpca100 3장 | ① E3s2 재개(ep10, 학습 중 test 평가 off) ② **E13s2**(E13 시드2 40ep) ③ **E4c**(E4b에서 쌍을 RailTrack→Sky/Static/Terrain 3개로, MARGIN 0.25) |
| — | E4b | 35ep 재채점으로 종결(재개 안 함). ep36~40 Adam `ComplexFloat` 에러 = margin 항 후기 불안정 → E4c에서 margin 완화 |

## 4.5 체크포인트 보존 대장 (2026-09-09~10, user 승인: "보존 완료된 체크포인트는 지워도 괜찮아")

규칙: 정본 = **val-best `*_top1_checkpoint.pth` 하나**. NAS 사본 md5 일치 후에만 서버 원본 정리. **서버에서 ckpt가 안 보이면 지워진 것이 아니라 NAS로 이관된 것** — 위치 단일 출처 = [infra/artifact-locations.md](../infra/artifact-locations.md)(develop 798ec44). NAS 루트 `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/ckpts/`.

| 런 | 정본 ckpt | 지금 위치 | md5 | 상태 |
|---|---|---|---|---|
| E2·E7·E7c (hpca100) | ep40 top1 각 1 | `daily_cards_20260908/{E2,E7,E7c}/` ✅ + `hpca100_archive_20260909/*.tar`(디렉터리 전체) | ✅ 3/3 | ✅ 서버 원본 이관·삭제 완료(09-09 16:22, 다른 세션) — 추가 삭제 대상 없음 |
| E1M (hpca100) | `epoch35_80.64_top1` | `/tmp/jemo_scratch`(휘발) + `daily_cards_20260908/_tmp_volatile/E1M/`(자동 회수); SSDb 잔여분 `hpca100_archive_20260909/…E1M.tar` | ep35 사본 대조 확인 필요 | 🟡 |
| B0 / E1 / E3 (bengio, 양도됨) | `epoch40_65.4` / `epoch35_67.25` / `epoch25_65.97` | `ckpts/bengio_…_screen40_{B0,E1,E3}/` | ✅ `cbaa692f…` / `e69d7deb…` / `123f94e4…` (09-10 회수) | ✅ 카드 판정 근거 정본 셋 보존. bengio 원본은 손대지 않음 |
| E4 (bengio) | `epoch15_65.92` | bengio만 | — | 폐기 카드, 회수 생략 |
| DGFusion 80k (jarvis) | `model_0079999.pth` | `ckpts/dgfusion_swin_tiny_bs8_200k_deliver_clde/` | ✅ `4bb241fa…` | ✅ 재현 정본 보존 |
| E3s2 (hpca100, OOM 중단) | `epoch10_63.33_top1` | `/tmp/jemo_scratch` + NAS 자동 회수 | ✅ | 재개 대기 |
| 진행 중(E12·E2s2·B0s2 hpca100 /tmp, E13·E1 확정·E1s2 jarvis) | — | 서버 작업 사본 + hpca100은 NAS `_tmp_volatile/<런>/` 30분 회수 | 회수 시 | 완주 후 val-best만 정본화 |
| hpca100 이전 런(P52·P50-EXT·P46 lam02·P38/P39 import) | 각 val-best | `hpca100_archive_20260909/*.tar`(14건) | artifact-locations.md §2.2 | ✅ 이관 완료 |

- 결론: hpca100·bengio·jarvis 어디에도 **추가 삭제 대상 없음**. 남은 확인 = E1M ep35 NAS 사본 md5.

## 5. 결과 기록

| 카드 | 결과 | 판정 | 근거 |
|---|---|---|---|
| E9 τ=0 (재채점) | test 53.57 = 기존 정본(구 53.57) 일치 | 하네스 재동결 완료(636e490) | bengio `logs/e9_la_tau0_*.log` |
| **E9 검증셋 사전 로짓 보정** | τ=0 53.57 / τ=0.5 53.45 / τ=1.0 51.89 (mAcc 63.8→67.1→69.2) | **폐기** — mIoU 단조 하락, 게이트 +0.5 미달. 혼동은 클래스 사전 이동이 아니라 재현율↔정밀도 교환으로만 반응 | bengio `logs/e9_la_tau{0.5,1.0}_*.log` |
| E0 특징 정보 프로브 | raw 27.8 < adapted 35.6 < fused 45.1 (test 선형 mIoU); depth 중간층 탭에 Water 47.6·RailTrack 23.9 잔존(fused 16.6·0.0) | 게이트(i) 미달(−8.2%p) → "어댑터가 버린다" 기각. **E5·E6 하향, E1·E3 유지**. 오라클 프로브 미실행 | [analysis/2026-09-08-daily-cards-E0-feature-probe.md](../experiments/analysis/2026-09-08-daily-cards-E0-feature-probe.md) |
| B0 기준선 40ep | legal val **64.97**(mAcc 74.73) / test **53.78**(mAcc 63.87), ckpt `epoch40_65.4_top1` | 통과선 test **54.78**, 폐기선 54.28 확정. 트레이너 val(65.4)이 공식 val보다 **+0.43** 높다(보정값). 클래스별: val→test 낙차 11.19 중 61%가 Wall(71.7→12.2)·Bridge(58.4→0.0)·Water(59.6→6.5) — 단, 이 셋은 DGFusion도 test 0~4로 공통 붕괴(2026-07-28 §68)라 SOTA 격차 원인이 아님. 진짜 격차 = RailTrack(19.4→32.0, DGFusion보다 낮음) | bengio `logs/b0_eval_{val,test}_20260908_*.log` (감시 세션 재채점) |
| **E1 4탭 읽기 (완주·legal test 확정)** | 트레이너 val ep5~40 = 60.98/63.56/66.30/66.57/66.43/66.88/**67.25**/66.74 vs B0 …/65.4. val-best **ep35 67.25**. **legal val 66.90 / test 54.85**(mAcc 76.55/65.42, 1024·BS1). Δ vs B0: val +1.93, test **+1.07**(전이율 55%). 클래스별 test: RailTrack 31.98→45.11(+13.13), TrafficLight +5.01, SideWalk +3.09, Truck +1.82 / Wall +0.26·Static +0.25(보존), Water −1.71 | ✅ **스크린 통과(Δtest +1.07, 순이득 구조 — E2와 달리 붕괴 클래스를 잃지 않음)**. 확정 런(200ep×3페어)은 E12(E1+E2) 결과(09-10 아침) 본 뒤 대상(E1 단독 vs E12) 확정 후 착수; 그 사이 E1s2(bengio GPU4,5)로 시드 판별 보강 | bengio `logs/e1_taps_screen40_20260907_223704.log`, `logs/e1_eval_{test,val}_ep35_*.log` |
| **E3 센서별 class prototype (완주·legal test 확정)** | 트레이너 val ep5~40 = 59.35/63.72/62.23/65.44/**65.97**/65.51/65.61/65.69 (val-best ep25, 트레이너 val은 카드 중 최저). **legal test 55.18**(mAcc 64.97, ckpt `epoch25_65.97_top1`), legal val 미측정(bengio 양도로 대기열 소멸) | ✅ **스크린 통과 — Δtest +1.40, 카드 넷 중 최고**. 트레이너 val +0.57뿐인데 test +1.40 = **유일하게 val보다 test에서 더 버는 카드(증폭 2.5배)**. user 가설(센서별 클래스 근거)의 직접 검증 카드가 test 1위. 후속: E3s2(jarvis GPU5,7)로 시드 판별, E1+E3 결합(E13) 기동, E3b(일치 항 λ0.1) jarvis 진행 중 | bengio `logs/e3_permodal_screen40_*.log` |
| **E4 혼동 쌍 margin auto (완주, 재채점 중)** | 트레이너 val ep5~40 = 59.77/63.17/**65.92**/65.70/65.79/65.62/65.70/**64.99**(B0 ep40 65.4). val-best **ep15 65.92**, 이후 정체 뒤 ep40에서 하락 **legal val 65.86 / test 53.75**(ckpt `epoch15_65.92_top1`). Δ vs B0: val +0.89, test **−0.03**. 클래스별 test: RailTrack 45.71(+13.73)·Ground +10.88·Water +5.86·Static +2.59 / Pole 36.52(−8.87)·TwoWheeler −5.10·GroundRail −4.51·RoadLine −3.74·Fence −2.64 | ❌ **폐기(Δtest −0.03 < +0.5)**. margin 항이 겨냥 쌍 주변(RailTrack·Water)은 올리지만 대가를 다른 클래스에서 무차별로 치름 — val에서 벌고 test에서 전부 잃는 형태(E2의 극단판). 명시 쌍판 E4b(bengio ep10 중단, 재개 대기)가 부작용 축소를 검증 | bengio `logs/e4_confmargin_screen40_20260908_085306.log`, `logs/e4_eval_{test,val}_ep15_*.log` |
| **E2 전 선형층 LoRA (완주·재채점 완료)** | 트레이너 val ep40 **68.65**(궤적 62.54/63.44/60.09/62.40/65.25/66.91/67.85/68.65). **legal val 68.38 / test 54.50** (ckpt `epoch40_68.65_top1`, 1024·BS1). Δ vs B0: val **+3.41**, test **+0.72**. 클래스별 test: RailTrack 31.98→**47.75**(+15.77), TrafficLight 30.23→**42.72**(+12.49), TrafficSign +2.89, Truck +2.68 / Wall 12.17→5.77(−6.40), Water 6.46→0.99(−5.47), Static 28.63→23.19(−5.44) | 🟡 **회색지대(+0.5 ≤ Δ < +1.0) → 판정 보류, 시드2 페어(E2s2)로 판별**. 해석: 용량 확장이 진짜 격차 클래스 RailTrack을 실제로 풀지만(DGFusion 64.47과의 격차 60→17), 이득 전부가 RailTrack·TrafficLight 두 클래스(+1.13 mIoU 기여)에서 나오고 Wall·Water·Static(−0.69 기여)이 절반을 되받아간 상쇄 결과. val→test 전이율 21%. 트레이너 val − 공식 val = 0.27(B0 0.43과 같은 방향, 보정값 안정). 후속: E12(E1+E2 결합)로 축 가산성 확인, E2s2로 시드 판별 | hpca100 `logs/hpca100_E2_eval_{test,val}_20260908_083859.log` |
| E7 vs E7c MUSES PhysAug 대조 (둘 다 완주) | 트레이너 val ep5~40 페어 차이(켬−끔) = −0.66/+0.13/−1.09/+0.27/+0.38/−0.32/+0.27/**−0.11**(ep40: E7 80.29 vs E7c 80.18) | ✅ **확정: PhysAug 효과 없음(근소 손해)** — 공식 native(250장, 1080×1920) E7 **80.0756** vs E7c **79.9417**(−0.13), letterbox 1024 80.29 vs 80.18(−0.11), 트레이너 val −0.11로 세 축 일치. 클래스별: E7c가 train +0.31·bicycle +1.30·terrain +0.70에서 앞서고 rider −1.53·motorcycle −2.10·pole −0.75·person −0.63·traffic light −0.53에서 뒤짐(작은 객체에서 오히려 손해). → **MUSES 레시피 PhysAug-off 통일**(공정성 문제 §3.5-1 해소, 성능 손실 없음). 각주: 조건별로는 snow/night만 E7c 73.89 > E7 72.93(+0.96) — 야간·악천후 증강이 가장 어려운 구간에서만 작동하나 전체 결론을 뒤집을 크기 아님 | hpca100 `~/SSDb/jemo_maeng/muses_official_eval_20260908/{E7,E7c}`, `logs/muses_official_{E7,E7c}.log` |
| E7 MUSES PhysAug-off (완주) | 트레이너 val **80.29**@ep40 | 🔵 E7c(PhysAug-on, ep23/40) 완주(20:26 KST) 후 두 val-best ckpt를 `tools/eval_muses_official.py`로 페어 재채점해야 PhysAug 효과 확정 | hpca100 `logs/hpca100_E7_launch.log` |
| **E1M MUSES 4탭 (ep18 사망 → /tmp 복사 재개 → 완주·공식 재채점)** | 트레이너 val ep5~40 = 71.73/75.55/77.57/78.21/79.56/80.08/**80.64**/80.49 vs E7 73.84/75.39/77.06/77.41/79.16/79.84/79.69/80.29 (8지점 중 7지점 양수, 폭 +0.16~+0.95). **공식 native val 80.416 vs E7 80.076 = +0.340**, letterbox +0.349, 트레이너 val-best +0.35(세 축 일치). 조건별(E1M): fog/day 85.43·snow/day 78.06·clear/day 75.90·fog/night 76.13·snow/night 74.29·rain/night 70.65·clear/night 68.46·rain/day 67.24 | 🟡 **게이트(공식 Δ ≥ +1.0) 문면상 미달이나 폐기 아님** — 방향 일관(PhysAug의 4:4 진동과 다름), 같은 축이 DELIVER에서 통과, 단일 아키텍처 원칙상 MUSES는 '손해 없음'이면 충분 → **채택 저지 사유 없음, 중립~미소 양성**. 원인 메모: ep18 사망은 ❌ **hpca100 디스크 100%(SSDb 2.0T/2.0T)로 ckpt·로그 쓰기 실패 추정**. 재개·E12·E2s2 기동 전부 불가(tee 파일조차 못 엶). 삭제는 사용자 승인 사항 → 승인 요청(21:xx KST 푸시). 승인 전 조치: 다른 마운트 여분 확인, 완주 런 정본 ckpt(E2 ep40·E7 ep40·E7c ep40) NAS 보존, 용량 조사 | 감시 세션 보고 |
| **E3b 센서 간 일치항 (완주·legal test 확정)** | 트레이너 val-best **63.69@ep35**(E3 65.97@25 대비 −2.28, 8지점 중 7지점 열세). **legal test 54.10**(mAcc 64.40, 1897장·1024·BS1) — B0 53.78 대비 **Δ +0.32**, E3 55.18 대비 **−1.08** | ❌ **폐기 확정** — 통과선 +1.0·폐기선 +0.5 둘 다 미달. 🔬 **손실이 무차별적이지 않고 E3 의 이득 원천에 집중됐다**: RailTrack 70.37→**56.99(−13.38)** · Water 10.75→3.51(−7.24) · Wall 14.41→9.11(−5.30) · TrafficLight 32.72→27.46(−5.26). 나머지 21클래스는 ±2 안에서 상쇄(SideWalk +2.13 · Dynamic +2.16 · Ground +1.97 vs Pedestrian −1.63 · Building −1.15). RailTrack 이득이 B0 대비 +38.39 → +25.01 로 3분의 1 넘게 줄었다. **→ 결론: 센서별 prototype 은 서로 달라야 한다. 이득의 원천이 센서마다 다른 표현을 갖는 데 있으므로 `AGREE_LAMBDA` 로 당기면 그 원천이 사라진다.** user 가설의 세부 = 「센서 간 일치는 강제하지 마라」 | jarvis `logs/e3b_agree_launch.log`(학습, 6:58:57) · `logs/e3b_legal_launch.log`(재채점). ckpt `epoch35_63.69_top1`(로그의 `Best Test 53.46@ep20` 은 test-best 라 무효). legal val 재채점은 진행 중 |
| hpca100 디스크 우회(2026-09-09 00시) | SSDb 2.0T 중 우리 몫 538G(dset 327G·src 143G·cache 56G), 타 사용자 1.45T. `/` 오버레이(/tmp) 362G 여유 발견 → SAVE_DIR·로그를 `/tmp/jemo_scratch/`로 돌려 **E12(GPU1)·E2s2(GPU2) 기동 검증 통과**(45.6분/ep 단일 GPU, 완주 09-10 07~08시) | ✅ 삭제 없이 복구. 위험: /tmp는 컨테이너 재시작 시 소실 → 5ep마다 val-best ckpt 허브로 회수 지시. 정본 ckpt(E2·E7·E7c val-best) NAS 회수 지시. 중간 ckpt 삭제는 사용자 승인 대기 | 감시 세션 |
| DGFusion 공식 설정 재현 (jarvis, 200k 중 80k) | **80k ckpt val 66.5427**(fwIoU 93.73, mACC 75.16) vs 공개 val 66.51. 학습은 iter 86,572 / 87,370 / 87,842에서 3회 `FloatingPointError: Loss became infinite or NaN`(loss_ce·loss_contrastive NaN, mask·dice 0, condition·depth 정상)으로 발산 — 같은 구간·같은 형태 | ✅ **재현 성립(40% 지점에서 공개값 +0.03)**. 발산은 확률 사건이 아니라 87k 부근 수치 불안정 → 공식 설정 변경 없이는 완주 불가 → **200k 완주 포기, 80k ckpt를 재현 정본으로 기록**, test 평가(8분) 지시. 재현 노트: "공식 설정으로 87k 부근 발산 3회" | jarvis `/SSDb/jemo_maeng/dgfusion_train/dgfusion_eval80k.log`, `model_0079999.pth` |
| bengio 양도(2026-09-09 04:18) | user 지시로 bengio를 후배에게 양도 → E4b(ep14, 재개점 ep10 62.83)·B0s2(ep10, 재개점 ep5 59.71)·E1s2(ep6, 재개점 ep5 61.75) SIGTERM 중단, ckpt는 `/SSDe/…/outputs/`에 잔존(지우면 재시작) | ⏸ 재개 대기(plan.md 「bengio 중단 런」 절). B0s2는 hpca100 GPU3에서 /tmp 복사 재개 지시, E1s2·E4b는 다음 빈 자리 | 감시 세션 |
| jarvis 리포 구축 + E3b·E-LoRA arm B 기동(2026-09-09) | jarvis에 develop(b99ce18) 클론(기존엔 체크아웃 없어 7장 유휴). **E3b**(E3 + AGREE_LAMBDA 0.1) GPU1,2,4 완주 09-09 20시; **E-LoRA arm B**(완전공유 r16) GPU0,3 → A(yeon 0,1)·C(yeon 6,7)와 3-way 성립. ep2: A 52.28 / B 49.79 / C 55.77(파라미터 100/25/62.5%) — 중간 지점 C가 최고, H22 정합. ep12: C 64.49 vs A 63.87(6지점 중 5지점 C 우세, 게이트 C ≥ A−0.3 충족) | 🔵 진행 | 감시 세션(`brr1mnl3y`, `b6pwg1rxw`) |
| P52 DELIVER seed2 (yeon, 진행) | ep58 66.76 최고 갱신(ep28 66.19 이후 30ep 만) | 정체는 붕괴 아님 | yeon |
| E3b 센서 간 prototype 일치 항(AGREE_LAMBDA 0.1, jarvis, 완주) | 트레이너 val ep5~40 = 59.66/55.88/62.00/63.17/63.59/62.26/**63.69**/63.60 vs E3 …/**65.97**… — 8지점 중 7지점 음수, val-best −2.28 | 🟡 방향은 "일치 강제는 해롭다"(센서별 prototype은 서로 달라야 한다는 해석). 판정은 legal test(jarvis GPU5 재채점 중, ep35 ckpt) | jarvis 감시 세션 |
| **E13 E1+E3 결합 (완주·legal 확정)** | 트레이너 val-best **65.50@ep20**(카드 중 최저, 8지점 모두 E1보다 낮음). **legal val 65.42 / test 56.30** — 25클래스 **+2.52**, **24클래스 +1.79**(E1 +0.57의 3배). 이득 분산: TrafficLight +11.57·Static +9.25·Fence +6.67·SideWalk +4.46·Water +3.71(RailTrack 64.05). 손실 Ground −2.25·Pedestrian −1.98·Wall −1.29·Pole −1.11 | ✅ **통과·확정 런 1순위** — 우리 최고 P34 56.62에 −0.32, DGFusion 56.71에 −0.41(40ep). E13 확정 런 200ep jarvis 기동(04:36, 14:00에 GPU5 합류해 2장 DDP). 상세 §5-4 | jarvis |
| 시드2 스크린 완주(2026-09-10 오후, 트레이너 val-best) | **E12**(E1+E2) 68.52@ep40 · **E2s2** 68.00@ep40 · **E1s2** 67.01@ep35 · B0s2 65.82@ep30(ep37 진행, 15:20 완주) | 🔵 legal 재채점 진행(E2s2 hpca100 GPU2, E1s2 jarvis GPU7; E12·B0s2 대기). 판정은 넷 다 끝난 뒤 시드2 페어 한 표로(25·24클래스) | 감시 세션 |
| E3s2 재개 (hpca100 GPU0) | ep10 ckpt에서 재개, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`; 학습 중 test 평가 off 플래그는 코드에 없음(train_reliadino.py 254~258·1299행 무조건 평가) → 플래그 추가는 OOM 재발 시 | 🔵 진행 | — |
| E2s2 legal test (시드2, 완주) | test **53.70**(mAcc 62.51), val-best ep40 68.00. 25클래스 Δ +0.40, **24클래스 Δ +1.57**(E13 다음). **RailTrack 4.36**(시드1 E2 47.75, B0 31.98) — 같은 레시피에서 43점 갈림. 24클래스 안: TrafficLight **+18.57**(카드 중 최대)·TrafficSign +3.63·GroundRail +3.35·TwoWheeler +3.19·SideWalk +2.71 / Static −5.88·Wall −3.34·Ground −2.03 | 🟡 E2 축은 25클래스 회색지대(+0.72/+0.40)이나 24클래스는 시드1 +0.09 vs 시드2 +1.57로 편차 커 시드 3 없이 확정 불가. TrafficLight는 E2·E13·E4b 공통 이득 → 얇은 객체 병목과 직결 | hpca100 |
| 함정(09-10) `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True`와 `max_split_size_mb`는 호환되지 않음 → E13s2 기동 직후 `!block->expandable_segment_ INTERNAL ASSERT FAILED`. `max_split_size_mb` 제거 | 감시 세션 명세 반영(4553b66) | — |
| MCubeS P52 seed1 (완주) | val-best 58.18@ep174 / final 57.96; 3시드 58.07±0.49 | P46과 동률 — P52 컨트롤러 이득 없음(감사 결론 재확인) | yeon |

### 5-1. 카드 넷 최종 정리 (2026-09-09, legal 기준)

| 카드 | 축 | 트레이너 val-best | legal val | legal test | Δtest | Δval | 전이율 | 판정 |
|---|---|---|---|---|---|---|---|---|
| **E3** | 센서별 class prototype | 65.97 | 미측정 | **55.18** | **+1.40** | (+0.57 트레이너) | 증폭 ≈2.5× | ✅ 통과·최고 |
| **E1** | 중간층 4탭 읽기 | 67.25 | 66.90 | 54.85 | **+1.07** | +1.93 | 55% | ✅ 통과 |
| E2 | 전 선형층 LoRA r32 | 68.65 | 68.38 | 54.50 | +0.72 | +3.41 | 21% | 🟡 회색지대(E2s2 판별) |
| E4 | 혼동 쌍 margin(auto k5) | 65.92 | 65.86 | 53.75 | −0.03 | +0.89 | 소멸 | ❌ 폐기 |
| B0 | 기준선 | 65.40 | 64.97 | 53.78 | — | — | — | — |
| **E13** | E1+E3 결합 | 65.50 | 65.42 | **56.30** | **+2.52** | +0.45 | 증폭 5.6× | ✅ 통과·확정 1순위(24클래스 +1.79) |
| **E4b** | 혼동 쌍 margin(명시 5쌍) | 66.07@ep35 | 65.00 | 55.14 | +1.36 | +0.03 | 증폭 | ✅ 통과 상당(35ep 기준, 24클래스 +0.58; ep36~40 Adam ComplexFloat로 미완주) |
| E3b | E3 + 일치 항 λ0.1 | 63.69 | 64.29 | 54.10 | +0.32 | −0.68 | — | ❌ 폐기(일치 강제는 해로움) |

- 🔴 **트레이너 val 순위(E2>E1>E3>E4)와 test 순위(E3>E1>E2>E4)가 뒤집혔다.** val로만 봤으면 E3를 버렸을 것. **val 이득이 클수록 전이율이 낮다**(E2 21% / E1 55% / E3 증폭) — val 이득이 도메인 특화 과적합의 지표일 수 있음. → **스크린 판정표에 '전이율' 열을 상시 포함**하고, 확정 대상 선정에서 Δtest와 전이율을 함께 본다(§0 규약 보강).
- 트레이너 val − 공식 val 보정값 4사례: B0 0.43 / E1 0.35 / E2 0.27 / E4 0.06 → 평균 0.28, 편차 큼. 트레이너 val은 위치 가늠용일 뿐 판정에 쓰지 않는다.
- 클래스 구조: E1·E2·E4 모두 RailTrack에서 +13~+16(진짜 격차 클래스가 세 경로 모두에서 풀림). 갈리는 곳은 붕괴 클래스(Wall·Water·Static): E1 보존, E2 −5~−6, E4는 다른 클래스로 손해 전이. E3 클래스별 표는 미확보(추가 필요).
- **확정 런 결정(2026-09-10)**: 1순위 **E13**(24클래스 +1.79, 이득 분산) — jarvis 200ep 진행, 2순위 **E1**(24클래스 +0.57, 두 벤치 일관) — jarvis 200ep 병행. E3 단독은 24클래스 −0.14라 보류. E12는 완주 후 재채점으로 "결합이 일반적으로 나은가"를 확인.

### 5-2. E3 클래스별 분해 (2026-09-09 오후) — 이득은 RailTrack 한 클래스, 세 축은 직교하지 않는다

| 클래스 | B0 | E3 | Δ | E1 Δ | E2 Δ | DGFusion 80k |
|---|---|---|---|---|---|---|
| **RailTrack** | 31.98 | **70.37** | **+38.39** | +13.13 | +15.77 | 50.38 |
| Water | 6.46 | 10.75 | +4.29 | −1.71 | −5.47 | 4.06 |
| TrafficLight | 30.23 | 32.72 | +2.49 | +5.01 | +12.49 | 37.47 |
| Wall | 12.17 | 14.41 | +2.24 | +0.26 | −6.40 | 2.21 |
| Static | 28.63 | 24.93 | −3.70 | +0.25 | −5.44 | 32.61 |
| Pole | 45.39 | 45.68 | +0.29 | — | +4.29 | 58.99 |
| Pedestrian | 73.22 | 74.65 | +1.43 | — | +1.37 | 83.11 |

- **E3 +1.40의 산술**: RailTrack 단독 기여 +38.39/25 = **+1.54** → 나머지 24클래스 합은 소폭 음수. E1(+13.13)·E2(+15.77)도 RailTrack이 최대 축. **세 카드가 같은 클래스를 서로 다른 경로(특징 접점·용량·손실)로 풀고 있으므로 결합(E12·E13)에서 두 번 벌 수 없다.** E12 ep20이 E1 단독 대비 −1.24인 것이 같은 구조일 수 있음(단, E2는 ep15 바닥→ep40 최고 전례라 완주 전 판정 금지). E13은 "포화 여부"를 직접 재는 실험으로 성격이 바뀜 — 그대로 진행.
- **RailTrack은 이제 격차가 아니다**: E3 70.37 > DGFusion 50.38(+20). 센서별 prototype이 lidar/depth 기하 근거로 RailTrack 정체성을 붙잡는다는 user 가설의 가장 강한 증거(RailTrack Acc 82.32).
- 🔴 **다음 병목 = 얇고 작은 객체**: E3 vs DGFusion 80k에서 뒤지는 곳은 **Pole −13.31 · Pedestrian −8.46 · Static −7.68 · TrafficLight −4.75**(우리가 앞서는 곳은 RailTrack·Wall +12.20·Water +6.69 = 희소·경계 모호 클래스). 전체 −0.50은 이 상쇄의 결과. 얇은 객체는 1/16 패치·마지막 블록 SimpleFPN의 공간 분해능 문제로 읽힌다. ⚠️ **입력 해상도 1024는 이미 반증**: registry의 P46 C3 @1024 200ep 두 시드가 val 70.58/70.73인데 legal test 54.85/54.55로 768 학습(54.39±0.76)과 차이 없음 → 해상도만으로는 test의 얇은 객체가 안 풀린다. 남는 후보 = 고해상도 레벨 탭(E1의 level-0 강화) · 디코더 교체(구조 사다리 S3~S4, 구현 며칠) · 얇은 객체 표적 손실(경계·소객체 가중). **다음 카드 설계는 이 클래스 묶음을 통과 기준에 명시**(예: Pole·Pedestrian·TrafficLight 합 Δ ≥ +10).
- 논문 서사 재료: "희소·모호 클래스에서 크게 이기고 얇은 객체에서 진다"는 대조가 DGFusion 실측으로 확보됨.
- E12·E2s2 ep20: 65.33 / 65.14(둘 다 자기 최고). E2s2는 E2 같은 지점(62.40) +2.74이나 시드2 기준선 B0s2(ep10)가 8시간 뒤라 대조 유보.

### 🔴 중간 epoch 으로 시드 페어를 판정하지 마라 (2026-09-09 실측)

시드2 기준선 B0s2 가 나오면서, **기준선 자체가 epoch 마다 크게 흔들린다**는 것이 드러났다.

| ep | B0 (시드 821) | B0s2 (시드 902) | 차이 |
|---|---|---|---|
| 5 | 58.71 | 59.71 | +1.00 |
| 10 | 63.17 | 60.81 | **−2.36** |
| 15 | 62.18 | 63.54 | **+1.36** |

**부호가 두 번 바뀌고 진폭이 3.72 다.** 기존 규정은 "ep5 수치로 축 비교 금지(시드 흔들림 1.41)"였는데,
1.41 은 ep5 한 지점만 본 값이고 실제 진폭은 그 2.6 배였다. 따라서 **금지 범위를 모든 중간 epoch 으로 넓힌다.**

**왜 위험한가**: 기준선 진폭 3.72 가 판정 대상인 카드 간 차이보다 훨씬 크다(E2 의 legal test Δ 는 +0.72).
신호가 잡음에 묻히므로, 어느 중간 epoch 을 고르느냐에 따라 같은 카드가 통과로도 폐기로도 보인다. 실제로
ep10 으로 계산하면 E1s2 이득이 +4.00(시드1 +0.39 의 열 배)으로 나오지만, 그것은 B0s2 ep10 이 우연히
저점이었기 때문이다. 같은 카드를 ep15 로 맞추면 E2 계열이 두 시드 모두 기준선보다 **낮다**(시드1 −2.09,
시드2 −1.63).

**규칙**: 시드 페어 판정은 **40ep 완주 지점(또는 val-best ckpt 의 legal 재채점) 한 곳에서만** 한다.
중간 epoch 수치는 크래시 감지와 진행 확인에만 쓰고, 채택·폐기 근거로 인용하지 마라.

### 5-6. 🔴 시드2 페어 판정 — 기준선의 RailTrack 이 시드 간 30.49 차이난다 (2026-09-10)

**기준선 둘**

| | test | **RailTrack** | 24클래스 |
|---|---|---|---|
| B0 (시드 20260821) | 53.78 | **31.98** | 54.19 |
| B0s2 (시드 20260902) | **55.07** | **62.47** | 54.76 |

**아무 축도 얹지 않은 기준선끼리 RailTrack 이 30.49 차이난다.** 시드2는 기준선만으로 62.47 을 배웠고
시드1은 31.98 에 그쳤다. 25클래스 test 차이 +1.29 의 대부분이 여기서 나오며, 24클래스로 좁히면
+0.57 로 줄어든다. §5-3 의 "스크린 구간 = RailTrack 학습 곡선의 급경사"가 **기준선 수준에서** 재확인된다.

**같은 시드끼리 맞춘 Δ (올바른 계산)**

| 시드 | 카드 | 25클래스 Δ | **24클래스 Δ** |
|---|---|---|---|
| 1 | E1 | +1.07 | +0.57 |
| **2** | **E1s2** | +0.82 | **+0.78** |
| 1 | E2 | +0.72 | +0.09 |
| **2** | **E2s2** | **−1.37** | **+1.00** |

- **E1 축은 두 시드에서 +0.57 / +0.78 로 일관**된다. **E2 축은 +0.09 / +1.00 으로 편차가 크다.**
  시드 안정성은 E1 이 낫고 이득 폭은 E2 가 크지만, E2 는 RailTrack 을 파괴한다.
- 🔴 **E2s2 의 25클래스 −1.37 은 24클래스로 보면 +1.00 이다.** 기준선이 RailTrack 62.47 을 배웠는데
  E2s2 가 4.36 으로 무너뜨린 결과다. **25클래스 기준을 썼다면 시드2 카드 중 최고인 축을 폐기했을 것이다.**
  §0 의 24클래스 부기준이 없었다면 E2 축을 잘못 버렸다.

⚠️ **감시 세션이 중간 보고에서 E1s2 +1.35 · E2s2 +1.57 로 적었던 것은 시드1 기준선(B0)으로 계산한
과대평가였다**(각각 +0.57 부풀림). B0s2 재채점 전이라 어쩔 수 없었으나, **페어 판정은 같은 시드의
기준선이 나온 뒤에만 하라**는 원칙을 다시 확인시켜 준다.

**B0s2 클래스별 특징**: RailTrack 62.47 외에 Wall 15.92(시드1 +3.75)가 높고 Pole 40.21(−7.78) ·
TrafficLight 26.46(−3.77)이 낮다. **기준선끼리도 얇은 객체에서 8 포인트 가까이 흔들린다.**

출처: hpca100 `logs/B0s2_legal_test_20260910_071257.log`(ckpt `epoch40_66.21_top1`) ·
`logs/E2s2_legal_test_*.log`(`epoch40_68.0_top1`) · jarvis `logs/e1s2_legal_test_*.log`(`epoch35_67.01_top1`).

### 5-5. 🔴 E4b(명시 쌍) legal test 55.14 — 폐기에서 통과로 뒤집혔다 (2026-09-10, **35ep 기준**)

카드 설계 당시의 가설("E4 auto 쌍에 RailTrack 이 누락돼 무차별 손해를 본다")이 검증됐다.

| 카드 | legal test | 25클래스 Δ | **24클래스 Δ** |
|---|---|---|---|
| B0 | 53.78 | — | — |
| **E4 (auto k5)** | 53.75 | −0.03 | **−0.60** (폐기) |
| **E4b (명시 5쌍)** | **55.14** | **+1.36** | **+0.58** (통과, E1 +0.57 과 동급) |

**RailTrack 이 45.71 → 64.07(+18.36)** 로 뛰었다. 명시 쌍에 `RailTrack→Sky/Static/Terrain` 을 넣은 것이 직접 작용했다.

**B0 대비 클래스별(24클래스 안)**

| 번 것 | Δ | | 잃은 것 | Δ |
|---|---|---|---|---|
| TrafficLight | **+7.47** | | Pole | −3.21 |
| SideWalk | +5.24 | | Wall | −2.90 |
| Terrain | +3.84 | | TrafficSign | −2.01 |
| GroundRail | +2.38 | | Water | −1.24 |
| TwoWheeler | +2.02 | | Building | −1.12 |
| Fence | +1.63 | | Pedestrian | −0.91 |

**margin 항이 여전히 얇은 객체 일부를 해친다**(Pole −3.21 · Wall −2.90 · TrafficSign −2.01). 쌍을 좁혀
무차별 손해를 줄였을 뿐 없애지는 못했다. §5-3 의 「다음 병목 = 얇고 작은 객체」와 맞물리는 지점이다.

⚠️ **이 결과는 35ep 기준이다.** E4b 는 ep36~40 구간에서 Adam `_multi_tensor_adam` 의
`result type ComplexFloat can't be cast to Float` 로 죽었다(수치 불안정으로 `exp_avg_sq` 가 음수 →
sqrt 에서 복소수 추정). **E4(auto)는 완주했는데 E4b 만 죽었으므로 confusion margin 항의 불안정이
의심된다.** val-best 가 이미 ep35 였고 E4 도 val-best 가 ep15(ep40 은 64.99 로 더 낮음)여서 ep35 로
재채점했다. 다른 카드와 나란히 놓을 때 이 차이를 명시할 것.

출처: jarvis `logs/e4b_legal_test_20260910_115525.log`, ckpt `epoch35_66.07_top1`(val-best). legal val 재채점 진행 중.

### 5-4. 🔴 E13(E1+E3 결합) legal test 56.30 — 트레이너 val 과 순위가 완전히 역전 (2026-09-10)

| 카드 | 트레이너 val-best | legal test | 25클래스 Δ | **24클래스 Δ** |
|---|---|---|---|---|
| B0 | 65.40 | 53.78 | — | — |
| E2 | **68.65** | 54.50 | +0.72 | +0.09 |
| E1 | 67.25 | 54.85 | +1.07 | +0.57 |
| E3 | 65.97 | 55.18 | +1.40 | −0.14 |
| **E13** | **65.50 (카드 최저)** | **56.30** | **+2.52** | **+1.79** |

**E13 은 트레이너 val 이 카드 중 가장 낮은데 legal test 는 가장 높다.** 완주 전 여덟 평가 지점 전부에서
E1 단독보다 낮았고(ep20 −1.07 · ep30 −2.19 · ep35 −1.82 · ep40 −1.36), 감시 세션은 그것을 근거로
"결합이 단독 축을 넘지 못한다"고 읽었으나 **틀렸다.** §0 의 「중간 epoch·트레이너 val 로 판정하지 마라」가
왜 필요한지 보여주는 가장 강한 사례다.

**이득이 RailTrack 이 아니라 다른 클래스에서 나온다** — 그래서 24클래스 기준으로도 살아남는다.

| 클래스 | B0 | E13 | Δ |
|---|---|---|---|
| **TrafficLight** | 30.23 | 41.80 | **+11.57** |
| **Static** | 28.63 | 37.88 | **+9.25** |
| Fence | 45.30 | 51.97 | +6.67 |
| SideWalk | 71.86 | 76.32 | +4.46 |
| Water | 6.46 | 10.17 | +3.71 |
| Other | 2.33 | 5.38 | +3.05 |
| (RailTrack) | 31.98 | 64.05 | (+32.07) |

잃은 것은 Ground −2.25 · Pedestrian −1.98 · Wall −1.29 · Pole −1.11 · Building −0.99 로 작다.
**E3 가 RailTrack 한 클래스에 이득이 몰려 24클래스에서 −0.14 였던 것(§5-2)과 정반대 구조다.**

🔬 **TrafficLight +11.57 · Static +9.25 는 §5-3 의 「다음 병목 = 얇고 작은 객체」 진단에 정확히 해당한다.**
DGFusion 대비 우리가 뒤졌던 클래스가 Pole −13.31 · Pedestrian −8.46 · Static −7.68 · TrafficLight −4.75
였는데 E13 이 그중 둘을 크게 회복했다. E13 test 56.30 은 **우리 최고(P34 56.62)에 −0.32**,
**DGFusion 공개값(56.71)에 −0.41** 이다. 40ep 스크린인데도 그렇다.

**확정 런 대상 재검토가 필요하다.** 09-10 02:38 에 E1 확정 런(200ep)을 jarvis GPU2,4 에 기동했으나,
24클래스 기준으로 E13(+1.79)이 E1(+0.57)의 세 배다. 생각정리 세션에 재검토를 요청했다.

출처: jarvis `logs/e13_legal_test_20260910_023814.log`, ckpt `epoch20_65.5_top1`(val-best). legal val 재채점 진행 중.

### 5-3. 🔴 스크린 규약의 교란 발견 (2026-09-09 밤) — 40ep 이득 = RailTrack 학습 가속

| 카드 | 25클래스 test | RailTrack | **RailTrack 제외 24클래스 평균** | Δ24 vs B0 |
|---|---|---|---|---|
| B0 | 53.78 | 31.98 | 54.69 | — |
| E1 | 54.85 | 45.11 | **55.26** | **+0.57** |
| E2 | 54.50 | 47.75 | 54.78 | +0.09 |
| E3 | 55.18 | 70.37 | 54.55 | −0.14 |
| E4 | 53.75 | 45.71 | 54.09 | −0.60 |
| E3b | 54.10 | 56.99 | 53.98 | −0.71 |

- **사실**: 200ep 정본 기준선(P46 C3-only)은 RailTrack test **67.69~72.03**(analysis/2026-08-06-p46-c3only-fair-eval-final.md)에 이미 도달한다. 40ep B0(31.98)는 그 절반 = 스크린 40ep는 RailTrack 학습 곡선의 급경사 구간이다. 따라서 E1·E2·E3의 "+1" 대부분은 **200ep가 어차피 도달하는 RailTrack을 더 빨리 배운 것**이고, E3의 +1.40은 24클래스에서 −0.14다.
- **함의**: (1) E3 200ep 확정 런은 RailTrack 천장(≈70)이 같아 이득이 사라질 가능성이 큼 → 착수 보류. (2) 24클래스에서 유일하게 양수인 **E1(+0.57, 약 1σ)** + 붕괴 클래스 보존 + MUSES +0.34의 일관성 → **E1을 첫 확정 런 대상으로**(기존 200ep seed821 기준선과 매칭 페어라 B0 재학습 불필요). (3) 스크린 통과 부기준으로 24클래스 Δ ≥ +0.5를 병기(§0 보강). (4) 시드 페어 판정은 40ep 완주 지점에서만(B0 vs B0s2 중간 진폭 3.72).
- **다음 카드 방향**은 §5-2대로 얇고 작은 객체(Pole·Pedestrian·Static·TrafficLight)이며, 24클래스 기준으로 측정한다.

> 🔗 **노션 동기화(2026-09-08)**: 이 표는 노션 논문 페이지(`Drone Object Detection for RGB-IR Fusion`, 33d05310) §4.2 카드 표와 같은 내용이다. 판정이 바뀌면 둘을 같은 날 갱신한다(CLAUDE.md §3, 빌더 `.claude/skills/notion-experiment-log/paper_page_builder.py`).
