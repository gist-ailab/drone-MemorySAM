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
| 🔴 보강(2026-09-10) 계산 규칙 | 24클래스 값은 **매번 클래스별 원자료에서 `(25×mean − RailTrack)/24`로 재계산**하고 문서의 다른 절에서 옮겨 적지 않는다 | §5-6 초판의 분모 오기(54.19, 정답 54.69)가 E13·E4b 판정에 전파된 사고(09-10, §5-8) |
| 🔴 보강(2026-09-09) 부기준 | **RailTrack 제외 24클래스 mIoU Δ ≥ +0.5**를 통과의 부기준으로 병기 | 40ep B0의 RailTrack test 31.98은 200ep 기준선(67.7~72.0, analysis/2026-08-06)의 절반 = 스크린 구간은 RailTrack 학습 곡선의 급경사라 "RailTrack을 빨리 배우는" 카드가 전부 +1로 보인다(§5-3). **추가 실측(09-10)**: 같은 레시피 E2 시드1 RailTrack 47.75 vs 시드2 4.36 — 시드 분산 43점, 단일 시드 RailTrack 값은 어떤 판정에도 쓸 수 없음 |
| 🔴 보강(2026-09-09) 중간 epoch 비교 금지 | 판정·시드 페어 비교는 **40ep 완주(val-best) 지점에서만**. 중간 epoch 이득 폭은 인용 금지 | B0 vs B0s2 ep5/10/15 = +1.00/−2.36/+1.36, 진폭 3.72 > 카드 간 차이(0.7) → 중간 지점 이득은 잡음(§5-3) |
| 조기 kill | ep20 legal-val이 B0 ep20 −1.5 미만 | 비용 절감 |


### 0-2. 🔴 SOTA 거리 게이트 + 모달리티 정합 비교 (2026-09-17 신설 — user 지적 「1순위 목표는 단일 구조로 세 벤치 SOTA」)

**누락 인정**: §0의 통과·확정 게이트는 전부 「같은 조건 기준선 대비 Δ」였고 SOTA까지의 거리는 게이트에 없었다. 그래서 기준선 +1로 통과한 레시피(E1·E13 확정 test 54.6~55.9)가 우리 최고 단일 런(56.99)보다 낮은데도 확정 후보로 올라왔다. 이 절이 그 빈칸을 채운다.

| 항목 | 규칙 |
|---|---|
| SOTA 거리 게이트 | 확정 런은 Δ 게이트에 더해 **「우리 최고 단일 런(같은 선택 규칙) 대비 ≥ 0」과 「모달리티 정합 SOTA 대비 거리」**를 판정표에 필수 병기한다. 미달이면 통과라도 헤드라인 후보가 아니라 「기준선 대비 이득 카드」로만 기록 |
| 모달리티 정합 비교(1차) | 우리는 벤치가 주는 모달리티를 전부 쓰므로, 1차 비교 상대는 **같은 모달리티 집합을 쓰는 방법**(DELIVER 4모달: DGFusion 56.71·CAFuser 55.6 / MUSES 융합: DGFusion 79.5 / MCubeS 4모달: StitchFusion 55.9). 적은 모달을 쓰는 방법(MM SAM-adapter 2모달 57.35·81.07, GtA 카메라 단독 82.39)은 2차로 병기하고 「더 많은 모달을 쓰고도 뒤진다」는 반론에 답을 준비한다 |
| 시드 평균 병기 | 단일 런 최고와 시드 평균을 반드시 함께 적는다. DELIVER 4모달 1위(56.99)는 단일 런이고 같은 레시피 5시드 평균(학습기 top1 규칙) 53.83은 DGFusion 미달이다 |
| 단일 레시피 | 세 벤치 최고치의 손실 설정이 다르면(DELIVER C3 on / MUSES·MCubeS C3 off) 「같은 구조」는 맞아도 「같은 레시피」는 아니다 — 논문 표에서 손실 설정 열을 숨기지 않는다 |

**2026-09-17 현황(모달리티 정합 1차 비교, 단일 런 최고)**: DELIVER 4모달 56.99 vs DGFusion 56.71(+0.28) · MUSES 3모달 공식 test 79.788 vs DGFusion 4모달 79.5(+0.29) · MCubeS 4모달 58.07(3시드) vs 55.9(+2.17) → 「전 모달 융합 계열 1위」는 세 벤치 모두 성립. 2차: MM SAM-adapter 2모달에 DELIVER −0.36·MUSES −1.28, GtA 1모달에 MUSES −2.60. E1·E13 확정 런은 우리 최고 단일 런보다 DELIVER에서 1~2점 낮아 **SOTA 거리 게이트 미달**(기준선 대비 이득 카드).

### 0-1. 🔴 확정 런(200ep) 판정 규약 — 결과보다 먼저 등록한다 (2026-09-11)

스크린을 통과한 카드는 200 epoch 확정 런으로 넘어간다. 그 판정에 쓸 분모와 게이트를 **수치가 나오기
전에** 여기 고정한다.

| 항목 | 값 |
|---|---|
| 매칭 페어 분모 | P46 C3-only 시드 20260821 200ep val-best `epoch90_67.3_top1` 의 legal test = **53.57** |
| 분모 출처 | 2026-09-08 E9 τ=0 재채점(`val.py` native 1024·BS1, bengio E0 eval config). 원본 ckpt = jarvis, 보존본 = NAS `ckpts/p46_c3only_seed20260821_200ep_20260918/epoch90_67.3_top1_checkpoint.pth`(md5 854a2c9d…; 2026-09-18 확인 결과 옛 기록 경로 `ckpts/…p46_c3only_seed20260821/` 은 NAS 에 없었고 bengio·yeon 평가 스테이징 사본에서 회수 — ISSUE-035 유형) |
| 참고 분모 | 5시드 평균 **54.39 ± 0.76** — 매칭 페어가 아니므로 **독립 비교에만** 쓴다 |
| 확정 게이트 | **3페어 mean Δtest ≥ +1.0** **그리고** **24클래스 mean Δ ≥ +0.5** |
| 24클래스 계산 | §0 보강 규칙대로 매번 클래스별 원자료에서 `(25 × mean − RailTrack) / 24` 로 재계산한다 |
| **분모의 24클래스 값** | **54.68** = (합계 − RailTrack 26.96) / 24. 클래스별 원자료 직접 합산(25클래스 평균 53.5748, 로그 표기 53.57) |
| 분모의 얇은 객체 4클래스 | **43.25** (Pole 44.46 · Pedestrian 73.81 · Static 30.10 · TrafficLight 24.64) |
| 분모 로그 | bengio `/SSDe/jemo_maeng/src/drone-MemorySAM-daily/logs/e9_la_tau0_20260907_222205.log` (1897장·3.05s/it = BS1 확인) |

🔴 **기준선 척도 열 필수(2026-09-15, 생각정리 지시).** 모든 판정표에 **기준선 값이 val-best 인지 final 인지, legal 재채점인지 트레이너 값인지**를
열로 적어라. MCubeS E13Mc 를 카드 final 대 기준선 val-best 로 비교해 −0.62(실제 +0.40)로 잘못 보고한 사고가 계기다.

🔴 **두 분모의 24클래스 값이 0.01 차이로 붙어 있다 — 판정표에 어느 것을 썼는지 반드시 적어라.**

| 분모 | 쓰는 곳 | 25클래스 | RailTrack | 24클래스 |
|---|---|---|---|---|
| B0 시드1 (40ep 스크린) | 카드 스크린 Δ | 53.78 | 31.98 | **54.69** |
| seed821 200ep val-best `epoch90_67.3` | 확정 런 Δ | 53.57 | 26.96 | **54.68** |

25클래스는 0.21 벌어져 구분되는데, RailTrack 차이(5.02)가 상쇄해 24클래스에서 사실상 같아진다.
"분모 54.7" 이라고만 적으면 스크린용인지 확정용인지 되짚을 수 없으니, **분모 이름을 함께 적어라.**

⚠️ 부수적 관찰: 이 200ep 분모의 RailTrack 26.96 은 40ep B0 의 31.98 보다 **낮다.** §5-3 이 인용한
다른 200ep 기준선(67.7~72.0, analysis/2026-08-06)과 비교하면 같은 길이의 런에서 RailTrack 이
**26.96 ~ 72.0 으로 요동친다**는 뜻이다. §5-6 의 시드 간 30.49 차이, E2 시드1 47.75 대 시드2 4.36
(43점)과 같은 현상이며, **RailTrack 을 뺀 24클래스를 부기준으로 둔 판단이 확정 런에서도 옳다**는
근거가 하나 더 늘었다. 반대로 말하면 이 분모의 25클래스 Δ 는 RailTrack 한 클래스에 휘둘린다.

🔴 **최고 갱신 여부를 완주 전에 단정하지 마라(2026-09-11 실사고).** E1 확정은 ep80 최고(67.88)
이후 **58 epoch 동안 갱신이 없다가 ep138 에서 68.83 으로 급등**하고 ep140 68.90 으로 연속 갱신했다.
"무갱신 구간이 길면 수렴한 것"이라는 전제로 완주 전 재채점을 걸었다가 무효가 되어 중단시켰다.
전이율(§5-14)·트레이너 val 순위(§5-4)와 같은 계열의 교훈이다 — **중간 지점의 안정은 최종 결과를
예고하지 않는다.** 완주 전 재채점은 GPU 공백을 메우는 용도로만 하고, val-best 대조 전에는 §5 에
쓰지 않는다.

🔴 **페어 수를 표기에 반드시 드러낸다.** 게이트는 3페어를 전제하는데 현재 확보된 것은 시드 20260821
한 페어뿐이다. 한 페어짜리 수치는 **"1페어 예비"** 로만 적고, 게이트 통과·미달로 판정하지 마라.
§5 에는 세 페어가 모두 차거나 완주가 확정된 뒤에만 판정을 쓴다.

⚠️ **평가 간격이 달라 val-best 후보 밀도가 다르다(각주 필수).** E1 확정은 `EVAL_INTERVAL 2` 라
200ep 안에 후보가 100 지점이고, E13 확정은 `5` 라 40 지점이다. 후보가 촘촘한 쪽이 우연히 더 높은
지점을 집을 여지가 크므로, **두 카드의 차이가 ±0.5 안이면 동급으로 읽는다.**

⚠️ **완주 전 예비 재채점의 취급**(2026-09-11 적용): GPU 공백을 메우려고 완주 전 val-best 로 미리
재채점할 수 있다. 다만 **완주 시점에 val-best 가 그 ckpt 그대로인지 대조**해야 하고, 갱신됐으면 예비
수치를 버리고 다시 돌린다. 대조 전까지 그 수치는 예비 조회일 뿐이며 §5 에 적지 않는다.

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

### MUSES 이식 카드 (2026-09-11 등재 — 그동안 config 헤더에만 있었다)

DELIVER 에서 통과한 카드가 **두 벤치에서 공통으로 오르는지**를 40ep 스크린으로 먼저 확인하는 계열이다.
단일 아키텍처 원칙상 3페어 확정에 GPU 6일치를 쓰기 전에 이쪽 정보량이 더 크다.

| 카드 | 원본 | 변경(1개) | config | 게이트(사전 등록) |
|---|---|---|---|---|
| **E7** MUSES PhysAug-off 기준선 | — | PHYSAUG off | — | 게이트 없음(헤드라인 교체용 대조군). **공식 val 80.0756** |
| **E1M** 4탭 읽기 MUSES 이식 | E7 | `MODEL.TAPS` on `[6,12,18,24]`·per_modal | — | E7 대비 상승. **공식 val 80.416**(달성) |
| **E13M** 4탭 + 센서별 prototype | E1M | `MODEL.P46.C3_PROTO` on(SRC permodal·λ0.1·τ0.1·EMA 0.999·PIXELS 4096·WARMUP_EP 5) | `configs/hpca100-muses_rgbel_P39_1_seed2_physaugoff_taps_c3permodal_screen40_E13M.yaml` | 아래 셋 **모두** |

🔴 **E13M 게이트 셋** — `tools/eval_muses_official.py` 의 **공식 val**(native 1080×1920) 기준이다.
트레이너가 찍는 레터박스 1024² 내부 지표로 판정하지 마라.

1. **vs E7(80.0756) ≥ +0.5**
2. **vs E1M(80.416) ≥ 0** — 4탭 위에 prototype 을 얹어 해롭지 않을 것
3. **조건별로 night·fog 에서 −0.5 미만이 없을 것**

셋 중 하나라도 미달이면 **"E13 의 이득은 DELIVER 특화"**로 기록하고 MUSES 이식을 닫는다.

세 카드는 시드가 같아 **매칭 페어**다. 재채점 명령 형태는 다음과 같다(별도 eval config 가 없고 학습
config 를 그대로 쓴다. 도구는 hpca100 레포에 있고 로컬과 md5 동일 `0b71f4e2…` 를 확인했다).

```bash
python tools/eval_muses_official.py --cfg <학습 config> --ckpt <val-best ckpt> --gpu <N> --out <출력 디렉터리>
```

⚠️ **C3_PROTO 의 warmup 은 `WARMUP_EP 5` 이고 epoch 이 0-index 라, prototype 손실은 로그상 ep6 부터
작동한다**(`train_reliadino.py:564` 주석). ep5 이전 수치로 "효과 없음"을 판정하지 마라.

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

6. 🔴 **메모리 제약(2026-09-15 실측, 생각정리 결정)**: **TAPS 를 켠 MCubeS 런(E1Mc 계열)은 24GB 카드에 들어가지 않는다.** A100 에서 24.2GB 를 쓰고, lecun 24GB 카드에서는 첫 반복에 CUDA OOM(23.46/23.56GiB)으로 죽었다. 그래디언트 체크포인팅은 ISSUE-027(멀티모달 LoRA 기울기 오염) 때문에 쓸 수 없다. → **E1 레시피 MCubeS 런은 hpca100(40GB) 전용**이다. 같은 설정의 기준선 B0Mc(TAPS 없음)도 24GB 카드에서는 23.8GB 로 한계에 붙고 hpca100 대비 약 6배 느리다(4.66 s/it 대 1.30 it/s). 과거 yeon 3090 에서 E1Mc 가 돌았던 것은 옛 체크아웃에 TAPS 가 빠져 있었기 때문이다(§5-25). **TAPS 없는 B0Mc 도 lecun 24GB 에서 ep1 도중 backward OOM(13:24, 23.56GiB 중 179MiB 남음)으로 죽었다 — 이 MCubeS 레시피 전체가 24GB 카드에서 안정적이지 않다.**

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

**MCubeS 이식 예약(2026-09-12, user 지시 "확정되는대로 mcubes에도 돌려보자")**

| 트리거 | 기동 | config | 게이트(사전 등록) |
|---|---|---|---|
| DELIVER E13 확정 3페어 통과 | E13Mc 3시드(3407·20260827·20260828, 200ep) | `configs/{hpca100,yeon}-mcubes_rgbadn_P39_1_rank_E13Mc_seed*.yaml` | 3시드 평균 Δ vs 매칭 C3-off ≥ +0.5, final-epoch 기준 |
| E13 미달·E1 통과 | E1Mc 3시드 | `…_E1Mc_seed*.yaml`(기준선과 TAPS 하나만 다름, diff 확인) | 동일 |
| 둘 다 미달 | 기동 안 함 | — | — |

- 매칭 기준선 = MCubeS 통일 레시피 C3-off 3시드 {57.93, 57.67, 58.62} = 58.07±0.49(registry N4). E13Mc는 C3-on(N4b) config 파생.
  🔴 **척도 주의(09-15 정정)**: 위 {57.93, 57.67, 58.62} 는 **val-best** 다. 게이트 척도인 **final(ep200)** 은 **{56.97, 56.61, 57.58} = 57.05** — 판정은 final 대 final 로만 한다(§5-21 정정).
- MCubeS 로더는 test split을 'val'로 읽으므로 트레이너 val-best 선택이 test-best와 동치 → **final-epoch 값을 판정 기준**, val-best는 병기만.
- 기동 검증: TAPS 덤프 `[6,12,18,24] per_modal`, (E13Mc) `SRC: permodal`와 `[C3-M]` 로그에 네 센서(image·aolp·dolp·nir) 손실이 모두 찍히는지, `fix_seeds(<시드>)`, RANDOM INIT 없음, 진행 표시줄.

- 🔴 **새 판정 등록 규칙(2026-09-18, §5 주 단위 분할과 함께 신설)**: 2026-09-18 이후 새 판정은 번호를 부여하지 않고 해당 주 파일(`cards/2026-Www-verdicts.md`)에 `### YYYY-MM-DD <주제> — <결론 한 줄>` 제목으로 **최하단 append** 한다. 주 파일이 없으면 만들고 `decisions/00_MOC.md` 에 등록한다. 상위 문서에는 '최신 판정 3개' 링크만 갱신한다.

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
| **hpca100 2026-09-15 삭제 (사용자 승인 방식 B)** | 각 디렉터리 val top1 + last_checkpoint 만 남김 | hpca100 서버(NAS 사본 없음) | — | ✅ 공유 볼륨 가득 참(Errno 28, 09-15 09:24~09:40 KST) 복구 — 판정 끝난 12개 디렉터리(DELIVER E14·E15·E13s3, MCubeS E13Mc 3시드·E1Mc 3407, MUSES E13M 시드1·2, E1M 시드2, E7 시드2·3)에서 test_* 와 val top2~5 63개(약 108GB) 삭제. **보존 없이 삭제**라 기존 승인("보존 완료된…") 범위 밖 — 생각정리 세션이 사용자에게 직접 받은 승인으로 집행. 여유 0 → 112G |

- 결론: hpca100·bengio·jarvis 어디에도 **추가 삭제 대상 없음**. 남은 확인 = E1M ep35 NAS 사본 md5.
- 🔴 **09-15 추가**: hpca100 방식 B 삭제(위 행). 이후 새로 기동하는 MCubeS 런은 `TRAIN.SAVE_TOPK: 1`·`SAVE_TEST_CKPT: false`(40843ed)로 val top1 + last 만 저장한다.

## 5. 결과 기록

> 🔀 **2026-09-18 분할**: 소절 본문을 주 단위 파일(`cards/`)로 옮겼다(구조 감사 [meta/2026-09-18-docs-structure-audit.md](../meta/2026-09-18-docs-structure-audit.md) R1·권고 #12). 옛 번호 앵커(`#5-NN`)는 주 파일 안에 그대로 보존되므로 기존 "카드 §5-NN" 인용은 그대로 유효하다. 판정 규약 정본은 §0~§4, 새 판정 등록 규칙은 §4 말미.

### 참조 리다이렉트 (옛 §5-NN → 주 파일 앵커)

| 옛 번호 | 새 위치 | 날짜 | 한 줄 요약 |
|---|---|---|---|
| §5-0 (무제 표) | [cards/2026-W37-verdicts.md#5-0](cards/2026-W37-verdicts.md#5-0) | 2026-09-08~10 | 스크린 러닝 로그 표 — §5-0 으로 인용되는 원본 표 |
| §5-1 | [cards/2026-W37-verdicts.md#5-1](cards/2026-W37-verdicts.md#5-1) | 2026-09-09 | 카드 넷 최종 정리(legal 기준) — val·test 순위 역전, 전이율 열 상시 포함 |
| §5-2 | [cards/2026-W37-verdicts.md#5-2](cards/2026-W37-verdicts.md#5-2) | 2026-09-09 | E3 클래스별 분해 — 이득은 RailTrack 한 클래스, 다음 병목 = 얇은 객체 |
| §5 (무번호) | [cards/2026-W37-verdicts.md#5-x-mid-epoch](cards/2026-W37-verdicts.md#5-x-mid-epoch) | 2026-09-09 | 중간 epoch 으로 시드 페어를 판정하지 마라 — 기준선 진폭 3.72 |
| §5-3 | [cards/2026-W37-verdicts.md#5-3](cards/2026-W37-verdicts.md#5-3) | 2026-09-09(밤) | 스크린 규약의 교란 발견 — 40ep 이득의 정체는 RailTrack(09-11 정정 포함) |
| §5-4 | [cards/2026-W37-verdicts.md#5-4](cards/2026-W37-verdicts.md#5-4) | 2026-09-10 | E13(E1+E3 결합) legal test 56.30 — 트레이너 val 과 순위 완전 역전 |
| §5-5 | [cards/2026-W37-verdicts.md#5-5](cards/2026-W37-verdicts.md#5-5) | 2026-09-10 | E4b(명시 쌍) legal test 55.14 — 폐기에서 통과로 뒤집힘(35ep 기준) |
| §5-6 | [cards/2026-W37-verdicts.md#5-6](cards/2026-W37-verdicts.md#5-6) | 2026-09-10 | 시드2 페어 판정 — 기준선 RailTrack 이 시드 간 30.49 차이 |
| §5-7 | [cards/2026-W37-verdicts.md#5-7](cards/2026-W37-verdicts.md#5-7) | 2026-09-10 | Δval 은 시드 노이즈가 크고 Δtest 가 더 안정적이다 |
| §5-8 | [cards/2026-W37-verdicts.md#5-8](cards/2026-W37-verdicts.md#5-8) | 2026-09-10 | 24클래스 Δ 전수 재검산 — E13·E4b 두 판정이 틀려 있었다 |
| §5-9 | [cards/2026-W37-verdicts.md#5-9](cards/2026-W37-verdicts.md#5-9) | 2026-09-10 | E12(E1+E2 결합) legal test 54.39 — 가산 가설 기각, E2 축 종결 |
| §5-10 | [cards/2026-W37-verdicts.md#5-10](cards/2026-W37-verdicts.md#5-10) | 2026-09-11 | 시드 편차는 초반에 가장 크고 ep20 에서 수렴한다(E13 세 시드) |
| §5-11 | [cards/2026-W37-verdicts.md#5-11](cards/2026-W37-verdicts.md#5-11) | 2026-09-11 | E3 축 두 시드 — 24클래스 일관 음수로 단독 폐기, E13 구성요소로 생존 |
| §5-12 | [cards/2026-W37-verdicts.md#5-12](cards/2026-W37-verdicts.md#5-12) | 2026-09-11 | E4 계열 종결 — margin 은 RailTrack 만 올리고 24클래스로 넘어오지 않는다 |
| §5-13 | [cards/2026-W37-verdicts.md#5-13](cards/2026-W37-verdicts.md#5-13) | 2026-09-11 | E13 시드2 통과 — 24클래스 안정성 네 번째 확증 |
| §5-14 | [cards/2026-W37-verdicts.md#5-14](cards/2026-W37-verdicts.md#5-14) | 2026-09-11 | 전이율은 부호만으로 읽으면 정반대로 묶는다 — Δval·Δtest 병기 규칙 |
| §5-15 | [cards/2026-W37-verdicts.md#5-15](cards/2026-W37-verdicts.md#5-15) | 2026-09-11 | E15(탭 투영 어블레이션) 해석 한계 — mean 은 두 요인을 묶는다 |
| §5-16 | [cards/2026-W37-verdicts.md#5-16](cards/2026-W37-verdicts.md#5-16) | 2026-09-12 | E13 확정 런 시드1 완주 — 스크린 클래스 프로필의 비보존 |
| §5-17 | [cards/2026-W37-verdicts.md#5-17](cards/2026-W37-verdicts.md#5-17) | 2026-09-12 | E13M(MUSES 이식) 공식 재채점 — 게이트 ①② 통과 |
| §5-18 | [cards/2026-W37-verdicts.md#5-18](cards/2026-W37-verdicts.md#5-18) | 2026-09-13 | 밤사이 판정 넷 — E13 세 시드 확정·E14 미달·E1 우위·E13M 야간 손실 |
| §5-19 | [cards/2026-W37-verdicts.md#5-19](cards/2026-W37-verdicts.md#5-19) | 2026-09-13 | 판정 셋 — E15 미달·E-LoRA 공유판 우위·P52 기준선 미달 |
| §5-20 | [cards/2026-W38-verdicts.md#5-20](cards/2026-W38-verdicts.md#5-20) | 2026-09-14 | E13 확정 런 시드2 완주 — 24클래스 시드 간 0.12, 분모 공백으로 판정 보류 |
| §5-21 | [cards/2026-W38-verdicts.md#5-21](cards/2026-W38-verdicts.md#5-21) | 2026-09-14 | E13Mc(MCubeS 이식) 3시드 — 게이트 미달(같은 코드 기준선 −0.63 재계산 포함) |
| §5-22 | [cards/2026-W38-verdicts.md#5-22](cards/2026-W38-verdicts.md#5-22) | 2026-09-14 | E13 확정 런 3시드 완결 — 게이트 통과 여부가 분모 선택에 달렸다 |
| §5-23 | [cards/2026-W38-verdicts.md#5-23](cards/2026-W38-verdicts.md#5-23) | 2026-09-14 결정 / 09-15 기록 | 생각정리 결정 집행 기록 — 분모 (c) · MUSES 페어 보강 · E-LoRA 3자 · 종결 넷 |
| §5-24 | [cards/2026-W38-verdicts.md#5-24](cards/2026-W38-verdicts.md#5-24) | 2026-09-15 | (b) 5시드 기준선 24클래스 잠정값 — 같은 규칙이면 E13 이 두 조건을 넘는다 |
| §5-25 | [cards/2026-W38-verdicts.md#5-25](cards/2026-W38-verdicts.md#5-25) | 2026-09-15 | E1Mc 철회 — yeon 두 시드는 TAPS 없이 돌았다, MCubeS 판별 불가 |
| §5-26 | [cards/2026-W38-verdicts.md#5-26](cards/2026-W38-verdicts.md#5-26) | 2026-09-15 | E1 확정 런 시드2 완주 — legal test 55.35 / 24클래스 55.85 |
| §5-27 | [cards/2026-W38-verdicts.md#5-27](cards/2026-W38-verdicts.md#5-27) | 2026-09-15 | DELIVER 조건별 평가 — E1·E13 확정 시드1 악조건 규칙 통과 |
| §5-28 | [cards/2026-W38-verdicts.md#5-28](cards/2026-W38-verdicts.md#5-28) | 2026-09-15 | MUSES 시드 페어 보강 — E1M 소폭 양성 · E13M 게이트 ①② 통과 |
| §5-29 | [cards/2026-W38-verdicts.md#5-29](cards/2026-W38-verdicts.md#5-29) | 2026-09-16 | E1 확정 런 시드3 완주 — legal val 67.89 내부 최고, E1 3시드 완결 |
| §5-30 | [cards/2026-W38-verdicts.md#5-30](cards/2026-W38-verdicts.md#5-30) | 2026-09-16 | 조건별 평가 3시드 확장 + MCubeS 같은 코드 쌍 — MCubeS 게이트 미달 종결 |
| §5-31 | [cards/2026-W38-verdicts.md#5-31](cards/2026-W38-verdicts.md#5-31) | 2026-09-16 | MUSES 확정 — E13M 세 게이트 3페어 통과, MUSES 에서는 E13 > E1 |
| §5-31b | [cards/2026-W38-verdicts.md#5-31b](cards/2026-W38-verdicts.md#5-31b) | 2026-09-16 | §0-1 규칙 추가 — MUSES 조건별 수치의 표본 한계(25~34장) |
| §5-32 | [cards/2026-W38-verdicts.md#5-32](cards/2026-W38-verdicts.md#5-32) | 2026-09-16 | E13M 200ep 풀 런 기동 — MUSES test 제출 후보 확보 |
| §5-33 | [cards/2026-W38-verdicts.md#5-33](cards/2026-W38-verdicts.md#5-33) | 2026-09-16 | 벤치 세 곳의 E1·E13 우열 요약 — MUSES 만 E13 우위 |
| §5-34 | [cards/2026-W38-verdicts.md#5-34](cards/2026-W38-verdicts.md#5-34) | 2026-09-18 | DELIVER 확정 3쌍 판정 + 헤드라인 56.99 프로토콜 보류 + 결측 모달 첫 실측 |

### 최신 판정 3개

- [2026-09-18 legal v2 채택 — DELIVER 헤드라인 56.39(DGFusion −0.32, SOTA 미달), 56.99 제외](cards/2026-W38-verdicts.md#2026-09-18-legal-v2) (2026-09-18) — 같은 ckpt v1 55.18→v2 56.39, E1 시드1 54.63→55.97, 하락 클래스 0; E1 시드1 v2 val 69.39 는 단일 시드라 잠정.
- [§5-34 — DELIVER 확정 3쌍 판정 + 🔴 헤드라인 56.99 채점 프로토콜 보류 + 결측 모달 첫 실측](cards/2026-W38-verdicts.md#5-34) (2026-09-18) — E1(중간층 4탭)을 DELIVER·MCubeS 공통 레시피로 확정, 56.99 는 legal 하네스 재샘플 정렬 편차(ISSUE-036)로 v2 재채점 전까지 수치 동결.
- [§5-33 — 벤치 세 곳의 E1·E13 우열 요약](cards/2026-W38-verdicts.md#5-33) (2026-09-16) — MUSES 만 E13 우위, DELIVER·MCubeS 는 E1 ≥ E13 이라 단일 구조 원칙에서 E1 이 안전.

### 분할 파일

- [cards/2026-W37-verdicts.md](cards/2026-W37-verdicts.md) — 2026-09-07~09-13 판정 20소절 + §5-0 러닝 로그 표.
- [cards/2026-W38-verdicts.md](cards/2026-W38-verdicts.md) — 2026-09-14~09-20 판정 16소절.
