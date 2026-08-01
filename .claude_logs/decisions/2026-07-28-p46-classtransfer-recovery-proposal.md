# P46-CTR — Class-Transfer Recovery: DELIVER SOTA 공략 제안 (2026-07-28)

> model-proposal 스킬 산출. 딥리서치는 subagent spawn 한도(200/200) 도달로 fable 대신 opus가 WebSearch로 수행(§1.6 설계·수치판정은 opus 담당). fable 재실행 원하면 `CLAUDE_CODE_MAX_SUBAGENTS_PER_SESSION` 상향 후 재지시.

## 0. 근거 확인 (§0.5 선행조건 충족)
- **분석 존재**: `experiments/analysis/p32-verification-p33v2.md`, `2026-06-30-p28-p29-failure-analysis.md`(Mode B), `2026-07-19-p38-m2f-standard-analysis.md`, `2026-07-07-p32-perimage-analysis.md` — class-transfer가 지배 원인임을 다수 확립.
- **val/test 수치**: DELIVER P34/P36 fair **67.74 / 56.62**, P39.1-rank DELIVER **67.60 / 55.56**(registry·log 기록, 2026-07-28 완주). SOTA = val 68.79(CAFuser-CAA) / test 56.71(DGFusion).
- **유효성**: ISSUE-026(ColorAugSSD aug 버그) 픽스 후 P39.1이 첫 클린 DELIVER 런 → 근거 유효(오염 없음).

## 1. 진단 (우리 실측)
DELIVER 성능 저하의 **지배 원인 = per-class 도메인 전이 붕괴**(모달 부족도, 단순 도메인시프트도 아님):
- **thin/rare 클래스가 train/val(주간) 생존 → OOD-test(야간·adverse) 붕괴**: Wall **62→2**, TrafficLight **81→13**, Water **33→0**, Bridge **46→0**.
- per-domain spread 작음(P38 2.58) = 도메인시프트가 주범 아님. **모달 융합은 이미 천장**(drop-lidar −0.78, depth와 잉여 — P39.1이 val·test 모두 baseline 못 넘음).
- 복구 상한 **+7.9pt**(class-평균, 분석 실측).
- 두 하위원인: (a) **rare-class 늦은 학습/under-learning**, (b) **도메인 간 class 표현 붕괴**(주간 외형에 과적합).

## 2. 진단 ↔ 문헌 대응
| 우리 실측 | 문헌 기제 | arXiv | 함의 |
|---|---|---|---|
| rare-class 늦은학습·test 붕괴 | Rare Class Sampling(RCS) | DAFormer **2111.14887** | collapse 클래스 우선 샘플로 조기·충분 학습 |
| 야간 외형 이동 → 국소외형 의존 thin 붕괴 | Masked Image Consistency(context 추론) | MIC **2212.01322** (CVPR23) | 국소 외형 대신 **context**로 추론하게 강건화 |
| 도메인 간 class 표현 불안정 | class-prototype 정렬/consistency | dual-prototypical **2309.14282** · SCSD **2412.12050**(AAAI25) | 도메인불변 per-class prototype |
| (참고)frozen-VFM DG 아키텍처 | Rein(frozen DINOv2+token+M2F) | **2312.04265**(CVPR24)·Rein++ **2508.01667** | 우리 P39.1과 **최근접 아키** = 노벨티 아님 |

## 3. 제안 — P46-CTR (내부신호만, P39.1 base 위, 전 항목 토글)
P39.1 base(gated_mlp trunk + VICReg + P36 router + M2F) **유지**. class-transfer 3종 추가 — **전부 학습-시·내부신호·주손실 직결**(키1 준수, zero-init 잔차 아님):

- **C-1 · Adaptive Rare-Class Sampling** [근거: DAFormer RCS 2111.14887 + 우리 관측 클래스 특화]
  collapse-prone 클래스(자기추정: 현재 per-class 학습 손실 상위 + 등장빈도 하위)를 포함하는 이미지를 우선 샘플. 난이도 신호 = **내부**(런타임 per-class loss·pixel 빈도). 외부 라벨/조건 무. → rare-class를 조기·충분히 학습.

- **C-2 · Masked-Context Consistency (source-only DG 변형)** [근거: MIC 2212.01322]
  학습 이미지 patch 랜덤 마스킹 + **EMA-teacher의 전체이미지 예측에 consistency**. thin 클래스를 국소 외형이 아니라 **주변 context로 추론**하게 → 야간 외형 이동에 강건. ⚠️ MIC는 UDA(target 필요)이나 **우리는 target 없이 source에 regularizer로**(내부신호 DG 변형) — 이게 우리 setting 적응.

- **C-3 · Domain-Invariant Class-Prototype Consistency** [근거: 2309.14282 / SCSD 2412.12050]
  per-class EMA prototype 유지 + **ColorAugSSD 스타일 변주 view 간** 각 클래스 feature를 자기 prototype로 당김(도메인불변화). 스타일 다양성 = **내부**(ColorAugSSD, ISSUE-026 픽스본; PhysAug 금지). → class 표현이 주간 외형에 과적합 안 되고 test로 전이.

## 4. 게이트 사전등록
- **현행 최선 대비**: P36 fair **val 67.74 / test 56.62**. 목표 = **test ≥56.71(DGFusion SOTA 돌파) & val ≥68**(CAFuser 68.79 접근).
- 🔴 **핵심 falsifiable 예측(class-transfer 가설의 직접 검증)**: collapse 클래스 **test IoU 회복** — Wall 2→**≥13**, TrafficLight 13→**≥40**, Water 0→**≥9**, Bridge 0→**≥20**, RailTrack **≥62 유지**. 이 회복이 없으면 **가설 반증 → 설계 폐기**.
- **ep30 조기 kill**: collapse 클래스(Wall/Water/TL) test IoU 합이 P39.1 대비 **하락 또는 무변화**면 즉시 kill.
- **ablation**: C-1/C-2/C-3 각 토글로 기여 분해(어느 게 실제로 collapse를 살리나).

## 5. 노벨티 포지셔닝 (정직 — 리뷰 방어)
- **최근접 선행**: RCS(DAFormer), MIC(masked consistency), Rein(frozen VFM DG), SCSD·dual-prototypical(prototype DG). **개별 기제는 전부 선행 존재 → 노벨티 아님.**
- **미점유 조합 축(우리 차별)**: ① **multimodal frozen-VFM**(Rein/MIC/DAFormer/SCSD 전부 단일 RGB 모달) ② **내부신호만**(CLIP-text·GT-depth·조건라벨 배제 — 과거 P33-v2 CLIP-text는 이 방침으로 폐기) ③ **단일 아키가 DELIVER(멀티모달)+MUSES 공용** ④ **진단주도**(관측된 per-class val→test 붕괴를 표적).
- **"first X" 주장 불가.** 논문 노벨티는 **"multimodal frozen-VFM seg에서 내부신호만으로 per-class 전이붕괴를 복구"라는 진단→처방 구조**에서 나온다(성능 원천 = 백본+LoRA는 관용, 노벨티 주장 금지).

## 6. 실행 계획
1. **선행(학습 0)**: P39.1-rank DELIVER ckpt로 **per-class val↔test 붕괴 맵 확정**(어느 클래스가 얼마나·어느 조건에서 무너지나) — seg-analysis 스킬(D1 class×domain)로 분석 세션 요청. C-1 타깃 클래스·C-3 prototype 대상 확정.
2. **구현**: C-1/C-2/C-3 각 모듈 + `meta/conventions.md` 코드검수 파이프라인(fresh-eyes 7종 + 스모크 grad/등가 assert + 로더 실측 + ep30 토글 즉검). EMA-teacher·prototype은 학습 전용(추론 불변).
3. **스모크 → 슬롯**: jarvis P44-DELIVER 완주 슬롯 또는 4-modal 완주분. EPOCHS 200(ep30 조기게이트).

## 7. 제약 준수 체크 (§2)
- ✅ 반증경로 재시도 없음(attn-bias/gate/calib/CEFR/무감독 threshold/CLIP-text 무).
- ✅ 키1: C-1/C-2/C-3 전부 주손실 직결(RCS=샘플링, consistency·prototype=aux loss로 gradient 직접).
- ✅ 공정성: PhysAug off · ColorAugSSD(ISSUE-026 픽스) · val-best ckpt · TTA 금지.
- ✅ 내부신호만 · 단일 아키(DELIVER+MUSES) · 성능원천 백본+LoRA 노벨티 주장 안 함.

## 8. 대기 (루프백)
결과 나오면 실패-키 문서 갱신 + 1로 복귀. MUSES에도 같은 아키로 적용 가능(class-transfer는 MUSES night truck −32 등에도 존재 — 부수 이득 기대).

---
**Sources (WebSearch)**: MIC [2212.01322](https://arxiv.org/abs/2212.01322) · DAFormer/RCS [2111.14887](https://arxiv.org/pdf/2111.14887) · Rein [2312.04265](https://arxiv.org/pdf/2312.04265) · Rein++ [2508.01667](https://arxiv.org/html/2508.01667) · dual-prototypical DG [2309.14282](https://arxiv.org/pdf/2309.14282) · SCSD [2412.12050](https://arxiv.org/abs/2412.12050) · Balancing Logit Variation [2306.02061](https://arxiv.org/pdf/2306.02061)


## §9 재타깃 (2026-07-29, R1024+DGFusion montage 근거) — 원안 §3 타깃 수정
**근거(신규 실측)**: DGFusion(DELIVER test-SOTA) 재현 + P39.1 per-class 대조 + R1024(768→1024) + 정성 montage.
- Wall/Water/Bridge: **DGFusion도 test IoU 0~4로 동반 붕괴**(우리 8/6/0), montage 확증(둘 다 그 픽셀만 놓침), 해상도 무효(Wall+0.36/Water−1.15) → **복구불가·SOTA도 못 넘음 → 1차 타깃 제외**. 원안 게이트(Wall≥13/Water≥9/Bridge≥20) **폐기**.
- **RailTrack = 진짜 격차**: 우리 test 4.02(@1024), DGFusion 64.47, 전 조건·해상도 무관. montage: 우리 0.00 / DGFusion ~1.00. MAP_10 test 상단 대영역을 Sky/Static으로 **오분류(class confusion — 픽셀기하 정상, 클래스할당만 틀림)**. val조차 23.92(under-learned).
- 해상도: 768→1024 test +1.21(thin), native-res eval로 흡수(무료).
**재설정 게이트**: 🔴 primary falsifiable **RailTrack test 4→≥40**(DGFusion ≥64 실증). overall DELIVER test ≥56.62 + **@1024 병기**. ep30 kill=RailTrack 무변화. ablation=C1/C2/C3 토글.
## §10 구현 스펙(labcode)
Base=P39.1-rank(reliadino). 3모듈 전부 config 토글·주손실/aux 직결(키1):
- **C-1 RCS**(DAFormer): train per-class 픽셀빈도 사전계산 → class prob ∝ freq^{-T}(T~0.01) → step마다 class c 샘플 후 c포함 이미지 샘플, 런타임 per-class EMA loss로 blend. 데이터로더 레벨. RailTrack auto up-weight.
- **C-2 MCC**(MIC, source-only DG): EMA-teacher. student=패치마스킹(ratio0.5/patch64), teacher=원본, masked영역 consistency(CE/KL, λ2).
- **C-3 Proto**: per-class prototype bank(25×feat_dim, EMA), 클래스 픽셀feature를 자기proto로 당김+타proto서 밀기, ColorAugSSD 2-view간 동일클래스→동일proto(λ3).
- 토글 `MODEL.P46.C1_RCS/C2_MCC/C3_PROTO`, EMA-teacher·proto=학습전용(추론불변). 공정성: PhysAug off·ColorAugSSD(ISSUE-026)·val-best·TTA금지. config `configs/<server>-deliver_rgbdel_P46_ctr.yaml`, EPOCHS200(ep30게이트).
