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

## 4. 등재
- plan.md 대기열에 카드 행 추가(B0·E0·E7·E1·E2·E3·E4·E8 순), 각 행 EPOCHS 40 명시.
- 카드 결과는 `experiments/analysis/2026-09-XX-daily-cards-<E>.md` 1건씩.

## 5. 결과 기록

| 카드 | 결과 | 판정 | 근거 |
|---|---|---|---|
| E9 τ=0 (재채점) | test 53.57 = 기존 정본(구 53.57) 일치 | 하네스 재동결 완료(636e490) | bengio `logs/e9_la_tau0_*.log` |
| E0 특징 정보 프로브 | raw 27.8 < adapted 35.6 < fused 45.1 (test 선형 mIoU); depth 중간층 탭에 Water 47.6·RailTrack 23.9 잔존(fused 16.6·0.0) | 게이트(i) 미달(−8.2%p) → "어댑터가 버린다" 기각. **E5·E6 하향, E1·E3 유지**. 오라클 프로브 미실행 | [analysis/2026-09-08-daily-cards-E0-feature-probe.md](../experiments/analysis/2026-09-08-daily-cards-E0-feature-probe.md) |
