# SYNTHESIS — 전 지식 종합: 우리가 너무 좁게 보고 있는가?

> 작성: 2026-07-08, 리드 오케스트레이터 세션.
> 입력: 4개 disjoint 소스 병렬 추출 — (a) relatedworks 로그(doc 10/12/18), (b) 실험/시도 로그(doc 03/05-07/16/17/19/21), (c) Obsidian research_vault(~94노트), (d) 모델구조 문서(doc 02/04/08/11/20).
> 용도: **이후 설계 확장의 입력**. 즉각 수정 지시 아님. 수치 canonical은 여전히 doc 03.

---

## 1. "이미 해본 것" 지도 — 다시 제안하면 안 되는 것들

### 1.1 학습형 fusion/gating 계보 (P8→P26) — **전원 P9(고정 상수) 미달**

| 시도 | 결과 (M-score / mIoU) | 사망 원인 (기록된 인과) |
|---|---|---|
| P8 sigmoid UAMM | 78.45 | sigmoid 포화 → 전부 ~1.0 → AMF uniform 퇴화 |
| **P9 CrossModalFusionHead + max-norm** | **81.98/82.10 (불변의 최고)** | 동작하지만 가중치 사실상 상수(thermal 1.0/lidar .96/img .74 반복); GAP이 공간정보 살해(ISSUE-003). **핵심: P9의 이득은 메커니즘이 아니라 학습파라미터 증가에서 옴** (UAMM 스칼라곱은 Pre-Norm+Residual에 의해 근사-상쇄, MoE 라우팅 entropy_ratio 0.95≈uniform) |
| P10 oracle-KL gate | 79.27 | oracle이 주간에 과적합; 멀쩡한 gate에 감독 얹어 net 해악 |
| P11 +MI loss | 77.09 | "uniform gate"는 공간평균 측정 artifact — 이미 분화된 gate를 제약 |
| P12 input-conditioned MoE | 80.80 | cond_proj zero-init 무기여; expert collapse 악화(Block0 lidar 단일 expert) |
| P13 energy fusion + init fix | 81.21 | energy=확신≠정답 → LiDAR UAMM 1.0 고정 "confidently wrong"(ISSUE-009); kaiming init은 resume이 무효화 |
| P14 per-modal aux decoder ×3 | 74.27 | frozen backbone 위 aux mask 부정확(ISSUE-008); Sky 붕괴 36.47 |
| P15 spatial energy | 71.05 | **공간 증폭**: 부정확 신호를 per-pixel로 확대 (Sky 16.66) |
| P16 4-fix calibrated entropy | 68.42 (최악) | thermal 저-entropy 지배 0.923 → Sky 3.17 |
| P17 multi-scale aux | 73.23 | Sky 3→33 회복하나 aux-품질 천장 그대로 |
| P19 spatial head 직접학습 | 69.63 | train-night 과적합, LiDAR 지배 0.992 |
| P21/P22 DeBA-FP | 81.77 / **82.10 (P9 동률)** | 구조 refinement는 진짜 유효(Dynamic +15 / Static·Sky +2.5) — 그러나 fusion 상수수렴은 미해결, P9 미돌파 |
| P24 SQG distill | 미제출 | ISSUE-013 sigmoid teacher ep40에 1.0 포화 |
| P25 unified spatial quality | 27ep 중단 | predictor 용량 부족(spatial std 0.05 vs target 0.40); **lidar/thermal 단독 mIoU 18%/15%라 RGB-first가 오히려 합리적** |
| P29 SDC 조건 라우팅 | Test 53.85 < P28 55.27 (전 도메인 하락) | **결정적 증거**: Mode B(day→night class-transfer)는 라우팅/컨디셔닝으로 해결 불가. TrafficLight 41→9.6 |
| P30 router+CTD | 49.76/44.10 (−13.4/−10.2) | ISSUE-022: router hook이 dead code로 200ep 방치 + 경량 CTD가 최종출력 대체한 게 붕괴 주범 (→ P31.1이 CTD aux-only로 강등) |

**교훈(재제안 금지 목록)**: ① 학습형 gate에 감독/정규화 추가(oracle-KL, MI, teacher-distill) ② scalar→spatial 승격(신호가 부정확하면 증폭만 됨) ③ energy/confidence류 training-free 신호를 정답 proxy로 사용 ④ 조건 라우팅으로 night gap 공략 ⑤ router anchor centering(softmax shift-invariance로 수학적 no-op 판명, Δ=0.0000).

### 1.2 도메인 브리징 계보 — **픽셀 레벨은 양방향 모두 실패 (정보이론적 한계)**

| 시도 | 결과 | 원인 |
|---|---|---|
| Gamma TTA soft-voting | Test −10.78 | γ>1.0은 학습분포 밖 → confident-wrong; memory attention이 RGB 오염을 타 모달로 전파 |
| I2I night→day (추론) | Test −5.25 | 정보 없는 픽셀에 hallucination; RGB만 변환돼 cross-modal 불일치 |
| I2I day→night (학습확장) | Test 53.18, Sky −49.6 | 합성야간≠실야간 (픽셀만 어둡고 정보량은 주간) |
| FDA | 학습 전 취소 | day/night 저주파 진폭 수십 배 차이 → 구조 파괴 |
| CV 휘도 보정(56frame) | −2.39 | 멀쩡한 프레임까지 "교정" |
| CRM/ZERO 증강 | ISSUE-007 | exact-zero artifact를 44% 샘플에 주입 → "zero면 RGB 무시" 지름길 학습, night-val proxy까지 오염 |
| hardaug 튜닝 전반 | 포화 | no-aug→basic +26.6pp(이득의 80%), 이후 튜닝 +1.4pp뿐. **증강은 이미 짜낸 레몬** |

### 1.3 Det 트랙 — **"스택 문제"의 진범은 데이터였음 (완결된 스토리)**

- AP≈0 → 라벨 버그(빈 프레임 52%) → v2 0.4455 → **egofill 데이터 복원만으로 0.8501 (동일 스택·동일 레시피)** → 목표 0.85 달성.
- YOLO 브래킷: RGB-only 0.821→(라벨 +904장)→0.864/0.866 — **모델 세대 무관하게 0.86 동일 = 천장은 라벨/데이터가 결정**.
- E0.1: P30-Det 하락의 범인은 router가 아니라 single-s16 query head (aux FCOS 0.431 ≈ P29 0.446).
- event≈egofill-lidar (0.8427 vs 0.8501) — 3rd modality ablation 완료.
- 남은 것: 멀티모달 robustness delta 주장 (Y1 bar: RGB low-light delta −0.070을 이겨야 함) — M1/M2/M3 **미측정**.

### 1.4 포지셔닝에서 이미 소각된 주장들

- "first additive attention bias" — **금지** (PRIMED=learned log-odds bias, SAE=training-free attention-entropy bias가 셀 점유).
- "SAM2 memory fusion은 MemorySAM뿐" — 반박됨 (SAM4D MCMA).
- "VFM 멀티모달 SOTA" 헤드라인 — 거짓 (MM-SAM-adapter 2-modal이 DELIVER test 57.35 / MUSES 81.07).
- clean-val mIoU 전장 자체 — 지는 싸움 (StitchFusion 68.2~70.3, OmniSegmentor 68.0, EQUISeg 67.9 > MemorySAM 65.38).
- 생존 novelty = **4축 교집합**: training-free × per-modal predictive entropy × additive pre-softmax × SAM2 memory cross-attn (RGB-X semseg). PRIMED/SAE 전문 정독이 제출 전 BLOCKING.

---

## 2. "왜 좁아 보이는가" — 지금 접근이 갇혀 있는 가정 3개

### 가정 ① "Fusion 메커니즘을 고치면 night/test gap이 닫힌다"
15개 버전(P10~P30)이 전부 fusion/gating/routing 변주였으나 **단 하나도 P9의 '고정 상수 fusion'을 못 이겼다**. 한편 자체 진단은 반대 방향을 가리킨 지 오래다:
- DELIVER gap은 날씨가 아니라 **class-transfer** (per-condition spread 2.7~3.6뿐; Wall 62→2, TrafficLight 81→13) — P29가 "라우팅으로 해결 불가"를 실험으로 증명.
- MULTIAQUA에서 P9 대비 유일한 동률(P22)은 fusion이 아닌 **구조 refinement**(DeBA)에서 나옴.
- P9 자체 분석: UAMM/MoE 기여 ≈ 0 (상수/uniform).
**갇힌 이유**: "RBMA가 노벨티"라는 논문 전략이 실험 프로그램을 fusion 축에 묶어둠. 노벨티 방어(ablation)와 성능 확보(다른 축)는 분리 가능한데 현재는 동일시되고 있다.

### 가정 ② "Frozen backbone + LoRA r4는 불가침이다"
P8→P31 전 버전의 불변식. 그러나:
- ISSUE-008(frozen-feature 천장)이 aux/energy/entropy/rare-class 실패의 공통 근인으로 반복 지목됨. Bridge/Other는 modal_competence [0,0,0,0] — **어떤 fusion도 못 살리는 구조적 사망**을 자체 문서가 인정.
- 경쟁자들이 이기는 축은 전부 representation: OmniSegmentor(멀티모달 pretraining, 68.0), MM-SAM-adapter(side-encoder, 2-modal로 SOTA), StitchFusion(encoder간 adapter weaving), TUNI(RGB-T pretraining).
- 유일한 탈출구로 문서 스스로 지목한 unfreeze(마지막 3블록, LR×0.1)는 **config로 만들어놓고 한 번도 안 돌림**.
**갇힌 이유**: "frozen = training-free 노벨티의 일부"라는 암묵적 결합. 실제로는 RBMA(추론시 bias)와 backbone 적응(학습시)은 독립 — frozen은 노벨티 조건이 아니라 습관이다.

### 가정 ③ "성능은 모델에서 나온다 — 데이터·평가축은 부차적이다"
- Det 트랙이 방금 반증: 모델 고정, **데이터만 복원해서 0.446→0.850**. 라벨 +904장이 아키텍처 3세대 차이보다 컸다.
- Seg에서 같은 종류의 미실행 레버가 쌓여 있음: RGB-zeroed double-forward(MULTIAQUA 논문 자체 레시피), modality-dropout + REQUIRE_ALL_MODALITIES 폐지, class-targeted 증강 — 전부 제안만 되고 미실행.
- 평가축도 마찬가지: 이길 수 있는 전장(per-condition table, EMM/RMM/NM robustness protocol, **MUSES AUPQ uncertainty track — B_i map의 천연 서식지인데 어떤 fusion 논문도 선점 안 함**)은 미측정이고, 지는 전장(clean-val mIoU)에서 계속 비교당함. model-selection proxy 부재(day-val도 night-val도 오염)는 미해결인 채 checkpoint 선택이 반복됨.
**갇힌 이유**: 세션 구조가 "새 P버전 구현→학습→미달→다음 P버전" 루프로 조직돼 있어, 버전 번호가 안 올라가는 작업(데이터, 평가 프로토콜, 벤치 재정렬)이 구조적으로 뒷순위가 된다.

*(부수 불변식 — 3대 가정보다 작지만 한 번도 의심 안 됨: modality-as-frame 고정 순서 img→lidar→thermal 미ablation, 단일 fusion point/단일 weight source, IMG_SIZE 1024 고정, 모달리티 순차 encoder forward로 인한 만성 VRAM 압박.)*

---

## 3. 아직 안 건드린 유망 방향 후보 (근거 = §related_works)

우선순위 = (논문 목표 직결도 × 비용 대비 확실성). **A~D는 서로 배타적이지 않고 전부 P31 위에 얹을 수 있다.**

### A. 🔴 Injection-point ablation 패키지 — "논문의 존재 증명"이 아직 미실행
같은 B_i 신호를 ① UNO식 output-average ② HyperDUM식 feature-multiply ③ RBMA logit-bias에 주입 비교 + additive vs multiplicative(SAM2Long key-scaling, DAMM (1−U), ModalPatch post-softmax) + 함수형(PRIMED log-odds vs linear vs log(r+ε)) + B_i calibration(ECE, graded-corruption curve — UDML 비판 방어).
**근거**: vault가 "novelty 방어의 핵심 증거"로 반복 처방, 전부 미실행. 대부분 추론-시간 실험이라 저비용. **이것 없이는 RBMA 주장 자체가 리뷰에서 무너진다.**

### B. 🔴 Training-time 항-편향 레버 3종 — 전부 "제안됨, 미실행", 전부 Mode B 직격
1. **Zheng functional-entropy regularization (2505.06635)**: parameter-free drop-in loss. relatedworks 로그가 **"#1 leverage"로 랭크해놓고 한 번도 실행 안 한** 항목. RBMA(추론)와 정확히 상보(학습).
2. **RGB-zeroed double-forward loss**: MULTIAQUA 논문 자체 레시피 (L_full + L_rgb-zero). aux 모달이 야간 의미를 스스로 지게 강제 — "missing RGB"는 이걸로, "noisy RGB"는 RBMA로 역할분담하면 스토리도 깔끔.
3. **Modality dropout + REQUIRE_ALL_MODALITIES 폐지**: det에서 데이터 복원이 승부를 갈랐던 것의 seg 동형(5,862→13,712장).

### C. 🟠 Backbone last-stage unfreeze — 이미 구현돼 있고 스위치만 안 켰음
UNFREEZE_LAST_N_BLOCKS=3, LR×0.1 — 구조적 사망 클래스(Bridge/Wall/Water)의 유일한 레버로 자체 문서가 지목. 노벨티 훼손 없음(가정② 참조). 1회 학습이면 판정 가능.

### D. 🟠 전장 재선정: MUSES AUPQ + per-condition/robustness 프로토콜
- MUSES uncertainty-aware panoptic 트랙: RBMA의 B_i reliability map을 그대로 제출물로 쓸 수 있는 **무주공산** (vault 확인). DGFusion의 최약 컬럼이 Night 58.97/Fog 58.86 = 우리가 이겨야 할 바로 그 칸.
- DELIVER per-condition 10-case table + EMM/RMM/NM sensor-failure protocol 채택 + 단일 프로토콜로 CMNeXt/MemorySAM 재평가(cluster-mixing 리뷰 리스크 제거).

### E. 🟡 Representation 축 합성 (중기): OmniSegmentor ImageNeXt pretraining을 RBMA 아래 스택 — vault가 "composable, 이득 가산 기대"로 명시. clean-val 65.38→68.0 갭의 정공법.

### F. 🟡 미개척 소축 (저비용 순)
- **modality-as-frame 순서 ablation** — 전 버전 미검증 pitfall #5, 추론만으로 가능.
- **DGFusion robust log-L1** — P10 noise 실패의 지목된 해독제, 재방문 안 됨.
- **token-efficiency/registers** (Fast SAM2, Expedit-SAM, ViT-registers) — memory-engine 접점이 구조적으로 일치; 야간 dense feature 개선 가능성은 open question.
- **P9+P13 ensemble** (Dynamic↔Sky 상보) — 수차례 제안, 미실행.
- **Det: RBMA-in-head inference-only λ·B 주입** — vault 조사 기준 detection에서 additive pre-softmax reliability bias는 **first-mover 빈 셀**; Y1 bar(−0.070) 대비 M1/M2/M3 robustness 측정과 함께.

### 재제안 금지 재확인 (§1과 중복이지만 설계 확장 시 유혹이 큰 것들)
학습형 gate 감독 강화 · 조건 라우팅 부활 · spatial 신호 승격 · 픽셀 도메인 변환(I2I/FDA/TTA-γ) · hardaug 추가 튜닝 · "first" 계열 무한정 주장.

---

## 4. 모순 레지스터 (설계 확장 시 반드시 해소하고 시작할 것)

| # | 모순 | 상태 |
|---|---|---|
| C1 | "UAMM/MoE 기여≈0" 진단 vs P12~P26 전체가 그걸 adaptive하게 만드는 프로그램이었음 | §2 가정①로 흡수 — fusion 축 이탈 필요 |
| C2 | night-val ckpt는 test에 좋고 M-score에 나쁨 / day-val은 그 반대 — 신뢰 가능한 selection proxy 부재 | 미해결. per-condition eval(D)이 부분 해독제 |
| C3 | 초기 노벨티 프레이밍("first logit bias") vs threat-watch 반박(PRIMED/SAE) — 프로젝트 문서 일부에 옛 프레이밍 잔존 | 4축 교집합으로 전면 통일 필요, PRIMED/SAE 정독 BLOCKING |
| C4 | SOTA 야망(leaderboard mIoU) vs "clean-val은 지는 전장" 판정 | §3-D 전장 재선정으로 해소 |
| C5 | P30 붕괴 원인이 CTD 때문인지 router-dead-code(ISSUE-022) 때문인지 완전 분리 안 됨 | P31이 CTD aux-only로 우회했으나 router 단독 기여는 여전히 미측정 |
| C6 | egofill이 gap-fill(>53ms)에서 동적객체 LiDAR 신호를 훼손(71%→fail) vs 그대로 0.85 달성 | gap-fill-excluded ablation(~10k) 미실행 — robustness 주장 전에 확인 |

---

*생성 로그: 4개 서브에이전트 구조화 추출(JSON) → 리드 1회 병합. 원본 추출물은 세션 transcript에 보존.*
