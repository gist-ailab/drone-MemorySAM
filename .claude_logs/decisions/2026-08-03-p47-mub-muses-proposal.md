# P47-MUB — MUSES Uni-modal Balance & Projection Density: 제안 (2026-08-03)

> model-proposal 스킬 산출. **fable 딥리서치 3축**(기제·노벨티·물리/벤치) 병렬 조사 + 기존 분석 자산 교차. 판정·설계 = 이 세션(opus).

> 🔴 **전장 정의 (user 확정 2026-08-03)**: 우리 메인 벤치 구성은 **4모달**이고 3모달은 ablation이다. DELIVER는 이미 4모달(img/depth/event/lidar, 현 SOTA test 57.05도 4모달)이나 **MUSES만 3모달(img/lidar/event)로 굳어져 있었다** — radar 4모달이 3모달보다 낮았기 때문(seed2 4모달 82.35 < 3모달 82.62, drop-radar +0.13). 본 제안은 그 결과를 **"radar 무익"이 아니라 "학습이 추가 모달을 살리지 못함"**으로 재해석하고, **4모달(img/lidar/event/radar)을 기준 구성으로** D-1·D-2를 검증한다. 3모달은 대조군.

## 0. 선행조건 충족 (§0.5)

- **분석 존재**: MUSES 표준분석 11종(`experiments/analysis/2026-07-15~2026-07-30`), 실패-키(`2026-07-20-failure-keys-*`), 제출 인덱스(`MUSES_TEST_RESULTS_INDEX.md`), drop-radar ablation(07-30).
- **수치 유효성**: ISSUE-025(radar 디코딩) 픽스 후 재확정된 drop-radar(+0.13)와 3-seed plateau 사용. ISSUE-026(ColorAugSSD)는 DELIVER 한정으로 MUSES 무영향.
- **신규 실측**: Codabench 리더보드 API 원본(2026-08-03) — 전 엔트리 per-condition 공개 확인.

## 1. 🔴 진단 재정의 — 병목은 야간이 아니라 clear/day

기존 프레임("주야 격차 5.14가 주 병목")은 **내부 상대비교**였고, SOTA 추월 관점에선 틀렸다. Codabench 원본 대조:

| 조건 | GtA(1위) | 우리 | 격차 |
|---|---|---|---|
| **clear** | — | — | **−5.85** 🔴 |
| **day** | 84.62 | 80.25 | **−4.37** 🔴 |
| rain | — | — | −3.50 |
| **night** | 77.81 | 75.12 | **−2.69** (최소) |
| **fog** | 72.64 | **77.50** | **+4.86 (전체 1위)** ✅ |

- 우리 주야격차 5.14는 **4모달 SOTA 대역 상단**(DGFusion 3.57 / CAFuser 5.12 / camera-only 9.6~11.7) — 이미 좋은 편. 잔여 헤드룸 ~1.6pt(전체 환산 +0.63).
- **야간 개선은 전체 mIoU에 0.4배만 반영**(day:night=6:4). 격차 전소거해도 상한 +2.06.
- 🔴 **모달 수 ↑ = 순위 ↓ 역상관 실재**: camera-only 82.39 > C+L 81.07 > 4모달 79.49. 우리 radar 무익(+0.13)과 정합.

**기제 판정**: **modality laziness / greedy joint learning** — 융합 학습이 RGB uni-modal feature를 under-optimize시킨다. 이론 증명(2203.12221), 실증(1905.12681·2202.05306·2305.01233), 리더보드 역상관, 우리 radar/event 무기여가 모두 한 방향.

### 1.5 🔴 자체 실측 확증 — C3 실패의 조건별 분해 (2026-08-03)

MUSES-C3(λ0.2, val 81.65@ep136 완주)와 base(P39.1-seed2)의 **조건별 차이**가 §1 진단을 직접 뒷받침한다. Δ = C3 − seed2:

| 조건 | Δ | | 조건 | Δ |
|---|---|---|---|---|
| **clear** | **−1.72** 🔴 | | fog | **+0.16** ✅ |
| **day** | **−1.29** 🔴 | | rain | **+0.21** ✅ |
| snow | −1.06 | | rain_night | **+1.03** ✅ |
| snow_night | −3.98 | | fog_day | **+0.52** ✅ |
| night | −0.62 | | clear_night | −1.29 |

**판정**: C3의 손해가 **clear/day(RGB 주도 조건)에 집중**되고 악조건(fog/rain)에선 오히려 이득이다. 즉 MUSES에서 C3가 base를 못 넘은 원인은 "prototype 기제가 무효"가 아니라 **RGB 본류 표현력을 깎았기 때문**이며, 이는 §1의 modality laziness 진단(clear/day −4.4~−5.9가 SOTA 격차의 실체)과 **동일 지점을 가리킨다**. → **D-2(uni-modal balance)의 표적이 문헌뿐 아니라 우리 실측으로도 확인됨.**

- 원시: `analysis_logs/P46_c3_muses_eval_20260803/val_eval_summary.md`(14조건 + 19클래스 + seed2 대조).
- C3 재-eval overall 81.20(학습로그 81.65 대비 −0.45, 프로토콜 차이).
- 부수: C3 zip `muses_P46_c3only_lam02_3modal_ep136_submission.zip`(검증 통과 750장) 보관, **미제출**(base 미달이라 제출 슬롯 보존).

## 2. 진단 ↔ 문헌 대응

| 우리 실측 | 문헌 기제 | arXiv | 함의 |
|---|---|---|---|
| clear/day −4.4~−5.9, 모달↑=순위↓ | **modality laziness**(joint 학습이 uni-modal feature 조기 포화 방치) | **2305.01233**(UMT) · 1905.12681(Grad-Blending) · 2202.05306(greedy) · 2203.12221(경쟁 이론증명) | RGB 본류를 살리는 **학습시** 개입 |
| lidar 유효 6.7%(SDK 기본 (2,2)) | DGFusion 저자들이 (7,7)+MC로 밀도화 → night·fog·원거리 이득 보고 | 2509.09828 / 2410.10791 | **밀도화 = 비용 0 처방** |
| night truck 76.43→44.40 | 대면적 저텍스처 암부 소실; **lidar가 대형 연속객체 정보원**(night lidar +8.6 PQ vs event +2.8) | 2401.12761 Table 3/10 · ACDC 2104.13395(night truck 8.3) | truck 복구 경로 = lidar 기하 |
| drop-event ≈0, event~lidar CKA 0.79~0.92 | lidar 공존 시 event 순증분 문헌 상한 **+0.4~2.8 PQ** | 2410.10791 Table IX · 2401.12761 | **구현 결함 아님** — 설정의 예측된 결과 |
| val→test 전이율 4% | shift 축 상이(accuracy-on-the-line 이탈) + val 250장 노이즈 + MUSES **지리적 엄격 분리** | 2107.04649 · 2401.12761 | val 최적화 신뢰 금지 |

## 3. 제안 — P47-MUB (2 모듈, 전부 학습시·내부신호·추론 불변)

🔴 **base = P39.1-rank seed2 4모달(val 82.35, hpca100 완주)** — 3모달 seed2(val 82.62/test 79.788)는 대조군으로만 참조. D-1·D-2 모두 **4모달 구성에 적용**한다. **추론 경로 불변 ⇒ DELIVER 훼손 경로가 구조적으로 없다.**

### D-1 · LiDAR 투영 밀도화 (데이터 레시피, 최우선)
- `projected_to_rgb`(SDK 기본 (2,2), 유효 6.7%) → **`projected_to_rgb_dgf`**((7,7)+motion compensation, **32.6% = 4.99×**)로 교체 학습.
- 🎯 **데이터가 이미 존재**: `/ailab_mat2/dataset/MUSES/projected_to_rgb_dgf/`(2026-07-15 생성, 7500 PNG). DGFusion 공개 PIXEL_MEAN 대비 −0.1%/+2.4%/−1.0%로 **오라클 검증 완료**(기존 것은 −81%로 전혀 다른 물건이었음). motion comp 실측 이동 중앙값 7.9px(86.7%가 >1px) — 기존 lidar는 RGB 노출시점과 misregistered였다.
- ⚠️ `muses.py:165`가 `'projected_to_rgb'` 하드코딩 → **config knob 필요**(구현 항목).
- 근거: 우리 drop-lidar 야간 1.75×, MUSES night lidar +8.6 PQ, DGFusion 저자 이행. **기대 +0.5~2.0 night(중심 ~+1), 비용 ≈0.**

### D-2 · Uni-modal Balance (UMT-style, 학습시 gradient)
- **각 모달 인코더 출력에 uni-modal aux head + 자기 손실**을 달아, 융합 손실만으로 학습될 때 생기는 under-optimization을 막는다(2305.01233 UMT / 1905.12681 Gradient-Blending 계열).
- 구현: per-modal LoRA 출력 → 경량 linear head → CE(주 손실과 별도 가중 λ_u). **추론 시 aux head 미사용**(P46 C3와 동일한 학습전용 계약).
- 선택 확장(토글): OGM-GE(2203.15332)식 on-the-fly gradient modulation — 모달별 학습속도 불균형 보정.
- 근거: 리더보드 모달↑=순위↓ 역상관 + 우리 radar/event 무기여 + 이론(2203.12221). **RGB 본류 표현력(clear/day −4.4)을 직접 겨냥.**

### 3.1 🔴 구현 중 발견 — base에 이미 per-modal aux CE가 있다 (2026-08-04, 코드검수)

**발견**: `FUSION.AUX_CE_WEIGHT`(4모달 seed2 config에서 **0.5**)로 이미 모달별 aux decoder + CE가 돌고 있다 (`fusion.py:363` `self.aux_decoders`, `:551` `aux_logits = [self.aux_decoders[i](feats[i]) ...]`, `train_reliadino.py:178`). D-2 원안의 전제("uni-modal 감독이 없다")는 **부분적으로 틀렸다.**

**그럼에도 P47-2가 별도 모듈로 성립하는 이유 — 코드로 확인한 3가지**:

| # | 근거 | 확인 위치 |
|---|---|---|
| 1 | 🔴 **기존 aux head는 추론 경로에 있다.** P36 router가 `sum(w_route[i] * aux_logits[i])`를 **예측에 더한다**(주석: "train AND eval — the routed residual is part of the prediction"). 4모달 seed2는 `ROUTER.ENABLE: true`. ⇒ 그 head는 *uni-modal 표현력*이 아니라 *라우팅용 로짓 품질*도 동시에 최적화 중이며, **가중치를 올리면 추론 예측 자체가 바뀐다.** | `fusion.py:600-603` |
| 2 | 기존 aux logits는 reliability 신호(`rel_cal`/`corr_veto`/`b_cons`)와 calibration loss의 **입력**이기도 하다 ⇒ 목적 3중 결합. | `fusion.py:558, 496, 658` |
| 3 | 기존은 **모달 평균 고정**(`AUX_CE_WEIGHT/m`) ⇒ 4모달에서 모달당 0.125 균등. **"RGB에만 더 주기"가 표현 불가능**한데, 우리 진단은 정확히 RGB 편중을 요구한다. | `fusion.py` aux 합산부 |

⇒ **P47-2 = 추론 불변(학습 전용) + 목적 단일(uni-modal 정확도만) + 모달별 가중 가능**한 별도 head. "λ만 올린 것"이 아니다.

**대조군 설계 수정**: 당초 "`AUX_CE_WEIGHT` 0.5→1.0만 올린 대조군"을 두려 했으나, #1·#2 때문에 그것은 **깨끗한 대조군이 아니다**(router 잔차·reliability 신호가 함께 변함 = 교란). "순진한 경로가 충분한가"를 보는 값은 남으므로 **우선순위 3(여유 GPU 시)** 으로 격하하고, 해석 시 교란을 명시한다.

**λ 캘리브레이션 주의**: `REDUCE: mean`이라 모달당 실효 가중 = `λ_u/4`. λ_u=0.4 → **0.1/모달**(기존 0.125보다 작다). 즉 `MODALS: all`·λ_u 0.4는 per-modal 압력을 0.125→0.225(**+80%**)로 올리는 **보수적** 설정이다. 진단(RGB 편중)을 **직접** 때리는 설정은 `MODALS: ['img']`이며, 이때 λ_u 전량이 RGB에 실려 0.4 = 기존 대비 **3.2×**.

**실행 arm 우선순위(수정)**: ① `MODALS:['img']` λ_u 0.4 — 진단의 직접 검증(clear/day 게이트와 1:1 대응) ② `MODALS:all` λ_u 0.4 — 문헌 정합(balance) ③ `AUX_CE_WEIGHT` 1.0 대조군(교란 있음, 여유 시).

**검수 결과(2026-08-04, opus)**: conventions 준수(`p47.py` 신규 360줄 + 스모크 365줄, 결선만 model.py/train_reliadino.py) · 추론 게이팅 `self.training and gt_mask is not None`(`model.py:1067`) · **추가 forward 없음**(같은 forward의 feats 재사용 ⇒ ISSUE-028 무관) · off면 `self.p47_2 is None`(DELIVER 무영향) · config 1-변수 확인(base 대비 `SAVE_DIR`+`P47_2` 블록만) · 메모리 +51.7MiB/step. **판정: 병합 가능.** 등가성 `|Δ|max=0`·키1 grad 도달은 labcode 스모크 자체보고 → ep30 즉검의 **per-modal acc 분화**로 재확인한다.

## 4. 게이트 사전등록 (🔴 4모달 기준 재설정, 2026-08-03)

| 항목 | 기준 |
|---|---|
| **Primary(4모달)** | MUSES **4모달** val ≥ **82.62** — 즉 **4모달이 3모달 기록을 넘는 것**이 1차 목표(현재 4모달 82.35 < 3모달 82.62 역전 상태 해소) |
| **Stretch** | val ≥ 83.0(3모달 대비 명확한 우위) |
| **Secondary(공식)** | Codabench test ≥ **79.788**(우리 최고, 3모달 기록) — 4모달로 이를 넘으면 "4모달 우선 SOTA" 경로 성립 |
| **D-1 falsifiable** | 밀도화 lidar에서 drop-lidar dMIoU가 **주간에도** 상승(현 day 4.24 → ≥6) |
| **D-2 falsifiable(핵심)** | 🔴 **modality balance 적용 시 4모달 > 3모달로 역전**(현재 82.35 < 82.62). 이것이 modality laziness 가설의 직접 검증 — 실패 시 "radar는 실제로 정보가 없다"로 확정하고 3모달 회귀 |
| **부가** | drop-radar dMIoU가 유의하게 상승(현 +0.13 → ≥+0.5) = radar가 실제로 쓰이기 시작했는가 |
| ep30 조기 kill | 4모달 base(seed2 4modal 동일 ep) 대비 −1.0 이하 |
| 🔴 DELIVER 보존 | 변경 없음(추론 불변·MUSES 전용 데이터) |
| ablation | D-1 단독 / D-2 단독 / D-1+D-2 (전부 4모달, 3모달은 대조군) |

## 5. 노벨티 포지셔닝 (정직)

🔴 **단일 기법 단위 "first" 전무**:
- per-modal LoRA on frozen VFM: **MoE-LoRA-SAM(2412.04220)이 DELIVER·MUSES 동일 벤치에서 점유** — **인용 없이 내면 데스크리젝트급**.
- frozen VFM 멀티모달 adapter: 2509.10408 외 다수. modality balance: 2305.01233 등. 투영 밀도화: DGFusion이 이미 이행.
- prototype consistency(우리 DELIVER 기제): **MemorySAM SPMM(2503.06700)**과 근접 → 차별화 문장 필수(도메인불변 EMA bank·학습전용).

**미점유 조합 축(견고한 순)**:
1. **감독 원천** — 조건 인지를 학습·추론 **전 과정 내부신호만**으로. CAFuser/DGFusion은 조건 메타데이터+언어 지도(+depth 지도) 필수. 이 셀은 실측으로 비어 있음.
2. **frozen 반례**(성능 조건부) — MM-SAM-Adapter가 "frozen은 −1.8, fine-tune 필수"를 자체 ablation으로 주장. frozen DINOv3로 그 수치권 도달 시 직접 반박.
3. **벤치별 지배 실패요인 이질성**(DELIVER=class 전이붕괴→prototype 유효 / MUSES=RGB under-optimization→동일 기법 무효)을 단일 아키로 실증. 단 content/style 갭 이분법(2103.15467)의 재발견으로 읽힐 수 있어 선제 인용 필요. n=2.

**리뷰어가 깰 지점 3(선제 대응)**:
1. "MoE-LoRA-SAM의 백본 교체 아닌가" → 라우팅 단위(모달 MoE vs per-class)·reliability·수치 실측으로 방어 + **반드시 인용**.
2. "조건 라벨 제거의 가치가 작다" → CAFuser condition loss 기여가 **+0.4 PQ뿐**이라 역인용됨. head-to-head ablation 필수.
3. "camera-only GtA 82.39가 1위인데 융합이 왜 필요한가" → ① GtA는 **익명·논문 없음**(fact sheet 공란) → published SOTA는 MM-SAM-Adapter 81.07 ② **fog에서만 융합이 이김**(우리 77.5 = 전체 1위) = 멀티센서 잔존가치의 정량 근거.

## 6. 실행 계획 (🔴 4모달 우선 순서로 재정렬, 2026-08-03)

1. **선행(학습 0, 완료)**: `muses.py` 투영 경로 config knob 추가(commit 2879667) + `projected_to_rgb_dgf` 무결성 확인(7500장).
2. **D-1 · 4모달 먼저**(비용 0, 기대값 최고): 4모달 seed2 config(val 82.35)에서 데이터 경로만 `projected_to_rgb_dgf`로 교체 → 300ep. ep30 즉검(drop-lidar day ≥6). **게이트 = 4모달 val ≥82.62(3모달 역전).**
3. **D-1 · 3모달 대조군**: 동일 처치를 3모달 seed2에도 적용해 밀도화 효과가 모달 수와 무관한 공통 이득인지 분리 확인(2번과 병렬 가능, GPU 여유 시).
4. **D-2 구현·검수**: conventions 코드검수 파이프라인(fresh-eyes 7렌즈 + 스모크 grad/등가 assert + 추론 등가성 |Δ|=0). labcode 위임. **4모달 seed2 base에 우선 적용.**
5. **D-1+D-2** 4모달 합본 → 완주 → val 게이트(≥82.62, stretch 83.0) → 통과 시 Codabench 제출 1회. 3모달 D-1+D-2는 대조군으로만 참조(제출 대상 아님, 4모달이 역전 실패할 때의 폴백).

## 7. 제약 준수 체크 (§2)

- ✅ **반증경로 재시도 없음**: attn-bias·gate/calib/veto·CEFR·fusion rank(P41)·radar·prototype(MUSES)·zero-init 잔차 전부 미포함. IAF-Net/UP-Fuse류 추론 재가중도 **동형이라 배제**.
- ✅ 키1: D-2는 aux CE로 주손실과 직접 경쟁(zero-init 아님). D-1은 데이터 레시피.
- ✅ 내부신호만(조건 라벨·CLIP text·GT-depth 무). 단일 아키 유지.
- ✅ **DELIVER 무영향**(추론 불변 + MUSES 전용 데이터 + 토글).
- ⚠️ 공정성: D-1은 **데이터 전처리 변경**이라 논문에 명시 필요(DGFusion과 동일 파라미터가 되므로 오히려 비교 공정성 ↑).

---
**Sources**: modality laziness 2305.01233 · Gradient-Blending 1905.12681 · greedy 2202.05306 · 경쟁이론 2203.12221 · OGM-GE 2203.15332 · MUSES 2401.12761 · DGFusion 2509.09828 · CAFuser 2410.10791 · MM-SAM-Adapter 2509.10408 · MoE-LoRA-SAM 2412.04220 · MemorySAM 2503.06700 · ACDC 2104.13395 · accuracy-on-the-line 2107.04649 · Codabench comp 14005 API(2026-08-03)
