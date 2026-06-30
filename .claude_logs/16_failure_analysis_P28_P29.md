# P28(RBMA) · P29(SDC) 체계적 실패분석 → P30 커버리지 판정

> 작성: 2026-06-29 (백그라운드 분석 세션). 데이터 출처 = B200 `outputs/MMSamP28|P29/.../train.log` 파싱 (재평가 없이 로그만).
> 대상: `LoRA_Sam_P28`(RBMA) / `LoRA_Sam_P29`(SDC routing) / 판정 = `LoRA_Sam_P30`(c0351a4, worktree `.claude/worktrees/wandb-logging`).
> 데이터셋 = **DELIVER** (25 cls, 4 modal `img/depth/event/lidar`). MULTIAQUA P28 config는 **미실행**(train.log 0 byte).

---

## 0. 학습 현황 요약 (로그 사실)

| 모델 | 상태 | Day-Val best | Test best | 최종(ep) |
|------|------|-------------|-----------|----------|
| **P28** (RBMA, AMF=uniform) | ✅ 완료 200ep (19:57:30) | **63.40 @ep100** | **55.27 @ep178** | ep200: Val 59.85 / Test 54.80 |
| **P29** (P28 + SDC FiLM gate) | 🔄 학습 중 (~ep102) | 62.71 @ep54 (63.20 @ep100) | 53.85 @ep102 | 진행 중 |
| P28 MULTIAQUA | ❌ 미실행 (train.log 0 byte, Jun15) | — | — | — |

**핵심 관찰**: P29(SDC)는 P28 대비 **Test가 더 낮다(53.85 < 55.27)**. 조건-라우팅이 Test mIoU를 못 올림 → 라우팅 재설계가 ceiling을 못 뚫음(아래 Mode D / ISSUE-008 정합).

---

## 1. 실패모드 분류 (원인가설 + 근거수치)

### Mode A — Rare/thin-class collapse (지배적 실패, mIoU 갭의 거의 전부)
P28 **Test** per-class (대표 ep196, mIoU 55.10):

| 클래스 | Test IoU | Day-Val IoU | 비고 |
|--------|---------:|------------:|------|
| Bridge | **0.00** | 15~30 | Test 완전 사망 |
| Water | **0.08** | 1~7 | 양쪽 사실상 0 |
| Wall | **2.1** | 46~62 | day→test 붕괴(−60) |
| Other | 3.4 | 0.00 | 양쪽 사망 |
| Dynamic | 7.1 | 23~31 | 큰 갭 |
| Ground | 7.3 | 1~5 | 양쪽 저조 |
| Static | 23.8 | 23~37 | 저조 |
| TrafficLight | 13~42(불안정) | 80~82 | day→test 붕괴(−51) |

vs Road 97 / Sky 98 / Cars 94 / Truck 95 / Bus 94. **~7개 thin/rare class(<10)가 mIoU를 끌어내림 = 63→실효 갭의 거의 전부.**
**원인가설**: ① per-pixel argmax가 지배 class에 밀림(thin class가 자기 영역을 능동적으로 주장 못 함). ② frozen SAM2 backbone feature에 rare class가 애초에 약하게만 인코딩(ISSUE-008 ceiling). ③ 융합 `m_feat=Σ q_uamm_norm[i]·f_i`가 **class-agnostic per-pixel scalar**(`q_uamm_norm` (B,1,H,W))라 "Water엔 LiDAR" 같은 class별 모달 선택 불가.
**근거**: `analyze_failures.py`(P30 docstring 인용) per-class 양극화 — Water 0.00/Bridge 0.00/Wall 0.035/Other 0.054/Dynamic 0.083/Ground 0.097/TrafficLight 0.137. 위 Test 로그와 일치.

### Mode B — Day-Val vs Test 일반화 갭 (**weather가 아니라 class-transfer**)
글로벌 갭 ~6–8 mIoU(Val ~62 / Test ~55). **그러나 per-condition mIoU는 타이트**(night 0.526 / rain 0.561 — `analyze_failures.py`). → 갭은 야간·악천후 도메인 시프트가 **아니라** 특정 class의 day→test 전이 실패.
- Wall 61.8→2.1(−60), TrafficLight 81.3→30.2(−51, 13~42 불안정), Bridge 29.2→0.0(−29), Dynamic 30.8→7.1(−24).
- 역전(test>day): RailTrack 24.7→60.1(+35), Pedestrian 76→81. = day/test split의 class 분포·외형 차이.
**원인가설**: day에서 학습된 Wall/TrafficLight/Bridge 표현이 test split 외형으로 전이 안 됨(over-specialization). 조명 문제 아님 → night-aug류로 해결 안 됨.
**P29 교차검증**: SDC가 **Day-Val rare class는 개선**(Water 1~7→16~23, Bridge 29→39~53) 했으나 **Test는 그대로 사망**(Water 0.07~0.97, Bridge 0.00~0.04, Wall 1.99~3.27). → SDC가 day 과적합만 키워 **day→test 갭을 오히려 벌림**, 그 결과 Test 55.27→53.85. **Mode B는 라우팅/조건화로 안 풀린다는 직접 증거.**

### Mode C — Dead modalities (event·LiDAR 미사용)
drop-modality ablation(cloud, `analyze_failures.py`): **drop-depth Δ−0.224, drop-RGB Δ−0.097, drop-event Δ−0.000, drop-LiDAR Δ+0.001**. → 융합이 **RGB+Depth 2-모달로 퇴화**, event/LiDAR는 빼도 성능 불변 = 사실상 미사용.
**원인가설**: ① 고정 UAMM scalar 융합이 class-agnostic이라 geometry 모달을 특정 class에서만 살릴 표현력 없음. ② RBMA reliability(=1−H, training-free)가 event/LiDAR per-modal decoder 미적합 시 항상 高엔트로피→低신뢰→memory bias가 down-weight. RBMA는 **memory-attn logit에만** 가산(융합 가중엔 직접 미반영, AMF=uniform) → 출력 융합에서 event/LiDAR 부활 경로 없음.

### Mode D — 라우팅/게이트 행동 (collapse 재발 여부)
두 층위 구분:
- **출력 융합 층(UAMM scalar)**: P9~P27의 'gate 상수수렴'(ISSUE-002/015) 계보. P28/P29도 **고정 UAMM scalar** → Mode C의 2-모달 퇴화가 그 증상.
- **MoE-LoRA expert 층**: ISSUE-002 Block9 argmax E1≈0~10%(img)/~0.5%(lidar) = **E1 dead expert**, soft-MoE가 사실상 평균 단일 LoRA. P29 SDC가 per-condition 조건화를 추가했으나 **Test 무이득** → expert 특화가 mIoU로 전환 안 됨(viz는 spatial-mean artifact, per-token은 분화 entropy_ratio≈0.55 — 측정 축 문제는 CLAUDE.md #3).
**판정**: 융합·expert 양쪽에서 "유효 다양성 부족". P29로도 미해소.

---

## 2. P30 커버리지 판정 (코드 근거)

P30 코드 = `sam_lora_image_encoder_seg.py:8066 LoRA_Sam_P30(LoRA_Sam_P29)` + `sam_lola_utils.py:810 ReliabilityAnchoredRouter` / `:847 ClassTokenDecoder`. config `b200-deliver_rgbdel_P30_physaug.yaml`(두 기구 ON). trainer: `output[0]=cls_logits`가 main OHEM loss로 학습됨(`train_sam2_lora_paper.py:945 loss_orig=loss_fn(output,lbl)`, P30.forward가 `(cls_logits,m_feat,...)` 반환) → **class-token decoder는 end-to-end 지도학습 확인**.

| Mode | 판정 | 코드 근거 / 한계 |
|------|------|------------------|
| **A** Rare-class collapse | 🟡 **부분** | ① `ClassTokenDecoder`(C개 class query가 self+cross-attn으로 `m_feat` 질의 → per-class mask (B,C,h,w), `:869-881`)가 per-pixel argmax 우회 메커니즘 직격. output[0]로 OHEM 지도(확인). SAM3-RBMA 동형(val 8.49→16.27). **한계**: (i) 근사 구현(경량 transformer, 실제 sam_mask_decoder 수술 아님 — docstring 명시), (ii) **frozen-backbone ceiling 미해소**(ISSUE-008): rare class가 frozen feat에 없으면 query로도 한계, (iii) m_feat@feat_ch=32 단일 저해상도만 질의 → thin-class 경계 muffle 위험. → 메커니즘은 ✅, 효과는 GPU 미검증 |
| **B** Day→Test class-transfer 갭 | ❌ **미해결** (🟡 위험) | P30에 전이/도메인 일반화 기구 **없음**. SDC(P29 상속)가 가장 근접하나 **P29 Test 무이득(실증)**. 오히려 class-token decoder가 day rare-class에 더 적합되면 P29처럼 **갭 확대** 가능. Wall 62→2 / TrafficLight 81→13 의 transfer 붕괴는 P30 두 기구 어느 것도 직접 안 건드림 |
| **C** Dead event/LiDAR | ✅ **메커니즘** (효과 GPU 미검증) | ② `ReliabilityAnchoredRouter(per_class=True)`: `w=softmax_m(learned_logits(feat_i)+λ·reliability_i)`, per-class (B,C,h,w) → "class가 자기를 보는 모달에 라우팅" 표현력 획득(`_fuse_outputs` override `:8120-8141`, class-agnostic scalar UAMM 대체). **모달별 독립 conv head**(각자 bias = per-modality global prior 학습 가능) + zero-init(`sam_lola_utils.py:832-834`)→초기 reliability-구동(붕괴 방지). **설계 건전**: ⚠️ 처음엔 "anchor가 절대신뢰라 RGB/depth 편향"을 의심했으나 — **softmax(over modality)는 per-pixel 상수 shift에 불변**이라 anchor의 절대 레벨은 모달 softmax를 못 바꾸고 **상대 차이만 작용**(centering은 수학적 no-op, §4 스모크로 Δ=0.0000 실증). 즉 P30 router는 이미 상대 신뢰도를 옳게 사용. 효과는 reliability 신호 자체가 event/LiDAR에 유의미한지에 달림(GPU ablation 필요) |
| **D** 라우팅 collapse | 🟡 **부분** | 출력 융합 collapse: anchored router + diversity reg(`REG_LAMBDA=0.01`, modality-mixing entropy 보상 `:843`)로 상수수렴 방지 → ✅. **MoE expert E1-dead(ISSUE-002)는 P30 미터치**(P29 SDC gate 상속, load-balance/z-loss 없음). expert층 collapse는 P29에 의존 = Test 무이득 = 미해소 |

**한 줄 판정**: P30은 **Mode A·C·D의 "메커니즘"을 정확히 겨냥**(class-token decoder + per-class anchored router)하나, ① rare-class는 frozen-backbone ceiling, ② **Mode B(day→test 전이)는 구조적으로 미커버**, ③ Mode C anchor가 절대-신뢰 편향이라 event/LiDAR revival 불확실, ④ MoE expert collapse 미해소. **헤드라인 효과는 GPU 학습 전 미검증**(P30 자체 docstring: full forward 미실행).

---

## 3. 미해결 갭 → 개선안

| 갭 | 개선안 | P30 대비 |
|----|--------|----------|
| **A-ceiling** 저해상도 단일 질의 | class-token decoder의 dynamic-kernel dot-product를 **학습형 upsample(×up)** 고해상도 pixel embed에서 수행(SAM3-RBMA `use_high_res_features` 후속). bilinear 후처리 대신 학습 upsample로 thin-class 경계 회복 | **프로토타입 구현(P31)** — §4 |
| **B** day→test 전이 | **데이터/학습 측**: transfer-fragile class(Wall/TrafficLight) 타깃 강증강, 또는 backbone 마지막 stage unfreeze(ISSUE-008 ceiling 직격). 라우팅으로 불가(P29 실증) | P30 범위 밖 |
| **C-효과** event/LiDAR reliability 신호 | reliability anchor centering은 **no-op으로 기각**(§4). 진짜 지렛대 = per-modal decoder가 event/LiDAR에서 의미 신호를 내도록(decoder 용량↑ 또는 modal-specific). P30 학습 후 ablation으로 신호 유무 먼저 확인 | 미구현(진단 우선) |
| **D-expert** E1 dead | SoftMoE에 **load-balance/router-z loss** 추가(P30 직교) | 미구현 |

> **기각된 아이디어 (근거 보존)**: "router reliability anchor를 모달 간 centering" — 초기 가설은 P30 anchor가 절대신뢰라 event/LiDAR를 억압한다는 것이었으나, **softmax(over modality)가 per-pixel 상수 shift에 불변**이므로 centering은 출력에 영향 0. CPU 스모크로 Δ=0.0000 실측 → 폐기. 결론: P30 router는 이미 상대 신뢰도만 사용(설계 건전), Mode C는 메커니즘상 ✅.

---

## 4. 프로토타입: P31 = high-res class-token decoder (Mode A thin-class)

**동기**: P30 `ClassTokenDecoder`는 dynamic-kernel mask를 fused-memory 해상도(~h/4, feat_ch=32)에서 디코딩 후 caller가 **bilinear upsample** → Water/Wall/Bridge/Pole 등 thin-class 경계가 뭉개짐(02_model_arch P30 risk (1)/(iii)). SAM3-RBMA `use_high_res_features` 후속을 SAM2에 이식.

**구현**(worktree `.claude/worktrees/wandb-logging`, config-gated, 기본 OFF → P28/P29/P30 **byte-identical**, git diff = +80 insertions / 0 deletions 확인):
- `sam_lola_utils.py` append `ClassTokenDecoderHR(ClassTokenDecoder)`: query 경로(self+cross-attn) 동일, pixel-embed 분기에 **학습형 `ConvTranspose2d`(×up)** 추가 → per-class mask를 `up`×해상도로 생성. drop-in(동일 `(feat)` 시그니처).
- `sam_lora_image_encoder_seg.py` append `LoRA_Sam_P31(LoRA_Sam_P30)`: 인자 `ctd_high_res=False, ctd_up=2`. False→`self.class_decoder`=base(=P30 그대로). True→`__init__`에서만 `self.class_decoder`를 `ClassTokenDecoderHR`로 교체(P30.forward의 `class_decoder(m_feat)`+interpolate가 고해상도 mask를 투명 처리, forward 미수정).
- `train_sam2_lora_paper.py`: `QUALITY_GATE_MODELS`에 `'LoRA_Sam_P31'` 추가 + `if 'ctd_high_res' in sig.parameters` 가드(P30은 해당 파라미터 없음 → 무영향).
- config `configs/b200-deliver_rgbdel_P31_physaug.yaml`: P30 복제 + `CLASS_TOKEN_DECODER.{HIGH_RES: true, UP: 2}`.

**검증**: `py_compile` 3파일 PASS + CPU 더미 스모크 PASS — HR mask shape(up=2→2×, up=4→4×), no-NaN, grad가 feat+upsampler에 도달, base decoder shape 불변. 파라미터 +262K(학습 upsampler). **full SAM2 forward는 GPU 미검증**(P30 자체와 동일 단서 — track_step 내부 상호작용은 1-GPU sanity 권장).

**실행 큐 (GPU 비면 — 지금은 금지)**: GPU 0,1=타 연구원 / 2,3=P28(ep200 완료, 비우는 중). P30 먼저, 그다음 P31:
```
# GPU 2,3 free 확인 후
bash scripts/remote_exp.sh run B200 configs/b200-deliver_rgbdel_P30_physaug.yaml auto:2   # 먼저 P30 baseline
bash scripts/remote_exp.sh run B200 configs/b200-deliver_rgbdel_P31_physaug.yaml auto:2   # 그다음 HR ablation
# 성공 기준(P31 vs P30): Water/Wall/Bridge/Pole Test IoU 상승(thin-class 경계 회복)
```

---

## 5. 추천 다음 액션 (우선순위)
1. **P30 먼저 학습**(config ON, c0351a4) — Mode A·C 메커니즘 효과 실측: Water/Wall/Bridge>0?, event/LiDAR drop-ablation Δ<0(부활)?
2. **P31(high-res class-token decoder)** 로 Mode A thin-class 경계 보강 — P30 대비 Water/Wall/Bridge/Pole Test IoU ablation.
3. **Mode B는 별도 트랙**: transfer-fragile class(Wall/TrafficLight) 강증강 / backbone 부분 unfreeze (라우팅으로 불가 — P29 실증).
4. **Mode C-효과**: P30 학습 후 per-modal decoder의 event/LiDAR 신호 유무 진단 → 약하면 decoder 용량/modal-specific 개선(centering은 기각).
5. MoE expert collapse(Mode D) → SoftMoE load-balance/z-loss(P30 직교, 후순위).
</content>

---

## 6. 실측 per-domain × per-class 재평가 (jarvis, 2026-06-29)

> 출처: **체크포인트 직접 재평가**(로그파싱 아님). 서버 `jarvis`(172.27.183.201, 8×24GB), env MMSS_SAM 신규 빌드(torch 2.7.0+cu128), DELIVER `/ailab_mat2/dataset/DELIVER` **test split**, per-condition(`DATASET.CASE`), PHYSAUG **off**(clean eval), `val.py --mode test --macvi`(per-image viz가 4-modal에서 width mismatch로 crash → metric만 뽑는 macvi 경로 사용). 체크포인트 2종: **ep178**(test-best, pooled 55.27) / **ep100**(val-best, Day-Val 63.40).

### per-domain mIoU (test split, condition별)
| domain | ep178 (test-best) | ep100 (val-best) |
|---|---|---|
| cloud | 53.99 | 52.96 |
| fog   | 54.11 | 50.51 |
| night | **51.56** | **50.13** |
| rain  | 54.25 | 53.75 |
| sun   | 52.13 | 51.66 |
| **spread** | **2.69** | 3.62 |

- **ep178 ≥ ep100 모든 도메인** → test 운용점은 ep178이 일관 우위(test-best 지위 확인).
- **per-domain spread 2.7~3.6에 불과** → Val(~63)→Test(~55) 갭은 **weather/condition 도메인 시프트가 아님**. **Mode B(특정 class의 day→test 전이 실패) 재확인**(로그파싱 §1-B를 실측으로 입증). night가 최약 도메인(thin class 다수가 야간에 추가 하락).

### per-class 실패 유형 (ep178, test split)
**(a) 도메인-불변 사망**(모든 condition에서 collapse = 구조적 ceiling, Mode A):
- **Bridge 0.0**(전 도메인), **Water 0.1~0.2**, **Wall 0.3~6.1**, **Other 2.1~4.8**. → condition 무관 → 증강·도메인기법으로 불가, class-token decoder/backbone unfreeze 영역.

**(b) 도메인-민감 실패**(condition별 큰 편차 = 진짜 per-domain failure case):
- **RailTrack** spread **43.6** (cloud/fog 68~70 ↔ **sun 26.8**; ep100은 **fog 5.8** 붕괴, spread 51.6).
- **TwoWheeler** spread 32.5 (cloud 42.6/night 45.3 ↔ rain 75.1/sun 69.4).
- **Fence** 16.9 (night 33.1 ↔ sun 50.0), **TrafficSign** 14.1 (night 32.1 최약), **Ground** 13.3 (fog/rain ~4 ↔ cloud 17).

**(c) 도메인-강건**(low spread·high IoU): Building/Road/Sky/Cars/Vegetation/GroundRail/Truck — condition 불변 강건.

**night 특이**: Fence·TrafficSign·Terrain·Pedestrian·Truck 모두 night에서 최약 → 야간은 thin/rare class를 추가로 깎음(단 RailTrack은 sun에서 더 붕괴 → "야간만의 문제"는 아님).

### 함의(개선 우선순위 갱신)
1. **(a) 도메인-불변 사망(Bridge/Water/Wall/Other)** = 최대 갭 기여, condition 무관 → P31 high-res class-token decoder + (필요시) backbone 마지막 stage unfreeze로 직격(증강 무효).
2. **(b) 도메인-민감(RailTrack-sun / -fog, TwoWheeler-cloud/night)** = 타깃 증강 효과 기대 가능 구간(전이 취약 class 한정 강증강).
3. ep178을 challenge/제출 운용점으로 고정(전 도메인 우위). raw 로그: `jarvis:~/eval_P28_out/{ep178_test55.27,ep100_val63.4}__{cloud,fog,night,rain,sun}.log`.

### 6.1 재사용 평가/분석 도구 (P28/P29/P30 공용, 2026-06-29)

failure 평가를 모델 무관하게 재사용하도록 2개 스크립트로 모듈화(repo `tools/`):

- **`tools/eval_per_domain.py`** — per-domain(condition) eval 러너. 모델의 *자기 config*를 받아 MODEL/QUALITY_GATE 블록은 그대로 두고(→ 어떤 `LoRA_Sam_PXX`도 빌드됨) eval 전용 필드만 override(ROOT, `DATASET.CASE`, RESUME off, PHYSAUG off, batch). 체크포인트별×condition별로 `val.py --mode test --macvi` 호출.
  - ⚠️ `--macvi` 사용 이유: `val.py`의 per-image viz 패널이 4-modal에서 row width mismatch로 crash(`np.concatenate` @ val.py:1468). `--macvi`는 그 viz를 건너뛰고 GT 기반 per-class IoU 표는 그대로 출력(DELIVER는 항상 GT 보유).
  - 예: `python tools/eval_per_domain.py --cfg configs/b200-deliver_rgbdel_P30_physaug.yaml --ckpt best=<P30.pth> --dataset-root /ailab_mat2/dataset/DELIVER --gpu 1 --out-dir ~/eval_P30_out`
- **`tools/analyze_per_domain.py`** — 로그 파서/분류기. condition별 로그에서 per-class IoU 표를 읽어 per-domain×per-class 매트릭스 + 자동 분류(**도메인-불변 사망** max IoU<dead_thresh / **도메인-민감** spread>spread_thresh / **강건**) + per-domain mIoU spread(작으면 "도메인시프트 아님") → 마크다운 산출.
  - 예: `python tools/analyze_per_domain.py --logs-dir ~/eval_P30_out --label best=best --out analysis_P30.md`

**검증(jarvis)**: §6 P28 10런이 이 도구 출력과 일치. 분석기는 기존 로그 재파싱으로 동일 매트릭스 재현 확인. P29/P30은 학습된 체크포인트만 있으면 위 두 줄로 동일 분석 가능.

- **`tools/viz_features.py`** ✅ (2026-06-29 검증) — feature-level 진단 뷰어. 1회 eval forward로 모델의 `_last_*` 캡처(`_last_per_modal_feats/_outputs/_uamm_spatial`) + `(m_output, m_feat)` 반환만 사용 → **모델 코드 변경 0**. 패널 5행: R1 입력4모달+GT+Pred+**Error(vs GT)** / R2 모달별 encoder feature PCA+**fused PCA** / R3 모달별 **reliability(1−H)**(=RBMA 신뢰도, per-modal logit 엔트로피) / R4 모달별 decoder argmax / R5 UAMM 융합가중치. P28 forward=P27 상속이라 P26/P27/P28/P29/P30 **동일 `_last_*` → 그대로 동작**. 선택: `--case`/`--indices`/`--contains <ClassA,ClassB>`(GT에 해당 class 포함 이미지 자동선별).
  - 예: `python tools/viz_features.py --cfg <model.yaml> --model_path <ckpt> --case sun --contains RailTrack --num 2 --gpu N --out-dir <dir>`
  - prior-art: DINO/DINOv2 feature-PCA, CMX·CMNeXt 모달 기여 viz, Trusted-Multi-view(TMC) per-modality confidence map과 동류.

> ⚠️ **실행 인프라 주의(2026-06-29)**: **jarvis(`/ailab_mat2/dataset/DELIVER`)는 sshfs(FUSE-over-ssh) 마운트라 간헐적으로 I/O hang**(D-state 프로세스, "20분 모델로드"처럼 보이는 것은 실은 데이터 read 지연/행). eval/viz는 **데이터가 로컬인 B200(`/NHNHOME/.../dset/DELIVER`)에서 돌리는 게 안정적**. B200 conda는 `conda activate` 불가 → env python 직접 호출: `/NHNHOME/ailab/anaconda3/envs/MMSS_SAM/bin/python` + `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. 위 P28 패널(RailTrack-sun, Water-night)도 B200 GPU에서 생성.

---

## 7. 통계적 모듈 진단 — 왜 우리 모듈이 작동 안 하나 (P28 vs P29, 2026-06-30)

> 도구 `tools/module_diagnostics.py` (P26~P30 공용). DELIVER test 5조건 × 100 img(+ablation 15)·256 해상도 상대분석. P28=test_epoch178(55.27), P29=test_epoch122(54.21, **학습중 ep122/200**). 헤드라인 IoU는 §6/eval_per_domain(GT해상도) 사용; 여기 수치는 모듈 상대분석용.

### per-domain mIoU (test): **P29(SDC) ≤ P28 전 도메인**
| | cloud | fog | night | rain | sun | mean |
|---|---|---|---|---|---|---|
| P28 ep178 | 53.99 | 54.11 | 51.56 | 54.25 | 52.13 | **53.21** |
| P29 ep122 | 53.24 | 52.02 | 50.89 | 53.82 | 50.70 | **52.13** (−1.08) |

### 모듈 신호 (조건 평균, [img,depth,event,lidar])
| 신호 | P28 | P29 | 해석 |
|---|---|---|---|
| **reliability AUROC** (1−H가 per-modal 정답을 맞히나) | [0.77, 0.62, **0.30, 0.22**] | [0.80, 0.63, **0.28, 0.26**] | RGB·depth는 정보성↑, **event·lidar는 0.5 미만=anti-calibrated**(틀린 데서 더 "확신") |
| **UAMM 평균가중치** | [0.27, 0.28, **0.23, 0.23**] | [0.28, 0.28, 0.22, 0.22] | **거의 uniform** → 죽은 event/lidar에 ~45% 낭비, 정작 workhorse depth는 28%만 |
| **drop-modality ΔmIoU** (제거 시 하락) | [8.4, **23.5**, 0.02, 0.01] | [5.7, **19.3**, 0.05, 0.05] | **depth가 압도적 기여, event/lidar≈0(완전 사망)**. P29는 RGB의존도까지 낮춤(8.4→5.7) |

### WHY — 모듈별 실패 원인 (정량)
1. **RBMA reliability 모듈: event/lidar에서 신호 자체가 깨짐.** AUROC 0.30/0.22 (<0.5) = 신뢰도가 정답을 *역상관*으로 예측 → memory-attention bias에 들어가는 신호가 무의미. RGB/depth(0.77/0.62)는 정상이라, **RBMA는 "이미 잘 되는 모달"에서만 작동**하고 정작 살려야 할 geometry 모달엔 무용.
2. **UAMM 융합: reliability-비례가 아니라 거의 uniform → 오배분.** event/lidar에 각 ~22% 주는데 기여 ΔmIoU≈0(45% 낭비), depth는 과소가중. 결과로 **특정 모달이 "할 수 있는" class를 못 살림**:
   - `modal_competence`(각 모달 단독예측 per-class recall, P28): **TrafficLight=[img .24, depth .62, event .64, lidar .18]** — event/depth가 유능한데도 IoU 41·**misalloc 0.37**. **Ground**=[.,.,event .37,.] event 최강인데 misalloc 0.34. **Wall** RGB .30/ misalloc 0.27. **RailTrack** 전모달 ~.6/ **misalloc 0.40**. → **정보는 있는데 라우팅(reliability)이 깨져 못 씀.**
3. **구조적 바닥(융합 무관):** Bridge/Other **competence [0,0,0,0]**(어느 모달도 예측 0), Water [.07,0,0,0]→Sky 혼동. = frozen-backbone ceiling → P31/unfreeze 영역, 융합 개선으로 불가.

### P29(SDC) 판정: 미개선, 일부 악화
per-class(조건평균) **P29 손실**: **TrafficLight 41.3→9.6(−31.7!)**, RailTrack 39.4→32.8(−6.7), TrafficSign −2.1, TwoWheeler −2.0, Pole −1.9. **P29 이득**: Static +9.5, Wall +2.8, Water +2.2(여전히 ~2), Building +1.5. → SDC는 reliability miscalibration을 **안 건드림**; 모달 의존만 분산(dropMIoU img 8.4→5.7)시켜 **event/depth가 유능하던 TrafficLight를 붕괴**시킴. 소수 rare-class 미세이득 < TrafficLight 대붕괴 = **net −1.1**. **Mode B(라우팅/조건화로 test 못 올림) 재확인.**

### 처방 (우선순위)
1. **reliability 재보정이 진짜 지렛대**: event/lidar의 per-modal decoder가 *틀린 곳에서 과확신* → reliability(1−H)가 무용. per-modal decoder 용량↑ 또는 calibration(temp/conf penalty)로 AUROC>0.5 만들면 RBMA가 비로소 geometry를 라우팅. (지금은 RBMA가 RGB/depth에서만 동작)
2. **uniform 탈피**: AMF_MODE=uniform → reliability-비례 출력융합으로(단 신뢰도가 보정된 후). 미보정 상태에서 비례주면 더 악화 위험.
3. **구조적 사망(Bridge/Water/Other)**: 융합 아님 → P31 high-res class-token decoder + backbone 마지막 stage unfreeze.
4. **SDC 경로는 보류**: TrafficLight 붕괴 리스크. 도구 재현: `tools/module_diagnostics.py --cfg <m.yaml> --model_path <ckpt> --max-imgs 100 --ablate-n 15`.

