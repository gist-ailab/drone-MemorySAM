---
created: 2026-08-21
author: fable (discussion 세션, model-proposal 절차 — hybrid 결정 user 2026-08-21)
status: 🟡 제안 — 하이브리드 승인(user). 최소 N2 프로브 먼저(동일백본 baseline 대비) → 이득이면 A 확장 / 무이득이면 B 재프레이밍
---

# P51 제안 — Cross-modal LoRA Coupling (CMLC): 인코딩-시간 결합 (2026-08-21)

> **한 줄**: 선택(라우팅/게이팅)은 오라클+실현성 배터리로 **종결(H16)**됐다. 남은 열린 축 = **인코딩-시간 결합** — per-modal LoRA를 고립(현행)·평균(MLE-SAM)이 아니라 **저랭크 어댑터 부분공간 안에서 cross-modal 결합**하고, 학습 시 지배모달 마스킹으로 소수 신호를 **강제** 보존한다. frozen DINOv3 무변경. **SOTA가 목표가 아니라 동일 백본에서 baseline 대비 통제 이득**(하이브리드 1단계).

## 1. 진단 ↔ 문헌 대응 (우리 실측이 설계를 구속)

| 우리 실측/근거 | 함의 → 설계 규칙 | 출처 |
|---|---|---|
| 오라클 여지 +8.5(공간응집)는 **anti-consensus AND anti-confidence** — 추론 선택으로 도달 불가 | 선택/라우팅/재가중 금지(H1·H3·H16 종결). **특징을 *생성*(인코딩 결합)해야지 *선택*하면 안 됨** | [analysis/2026-08-20-spatial-axis-closure-h16.md](../experiments/analysis/2026-08-20-spatial-axis-closure-h16.md) |
| P49-AIR 실패: **강한 RGB가 주입 압력 삼킴**(RGB-centric, zero-init) | 결합은 **대칭**(RGB-중심 금지) + 학습 시 **지배모달 마스킹 강제**(옵티마이저 RGB-shortcut 능동 차단) | 원장 H13·H14, MoBaNet MCRM(2603.17705) |
| MLE-SAM(frozen SAM + per-modal LoRA)이 모달을 **산술 평균** → 우리 최근접 선행이자 진입로 | LoRA를 **평균 말고 결합** = 이 논문 대비 정확한 delta | 2412.04220 |
| CrossWeaver(2604.02948, 4개월 선행) = **학습 SegFormer-B0의 feature-space 신뢰도 MIB** | 우리 = **frozen VFM + LoRA-부분공간 결합** = 스택·결합위치 둘 다 상이 → 차별화 축(§4) | 검증됨 2026-08-21 |
| 소수 보존은 아키텍처 대칭이 아니라 **학습 신호**로 온다 | 결합 게이트 + **dominant-mask + 소수 hard-pixel aux** 없이는 결합만으로 안 됨 | 딥리서치 축2, MoBaNet |
| 신뢰도 가중이 **실제 예측을 안 바꾸는** 경우 흔함 | 이득 나오면 **leakage-safe 결정영향**으로 "소수가 실제로 살아났음" 증명 | 2606.26473 |

## 2. 설계 — CMLC (전 항목 토글, ablation 분해 보장)

**베이스 = P39.1-rank 추론 그래프**(frozen DINOv3-L + per-modal LoRA + gated-MLP trunk + FPN + M2F-lite). 여기에:

- **[C2-CMLC] Cross-modal LoRA Coupling** (핵심, 신규): LoRA-적용 레이어 ℓ에서 모달 m의 저랭크 코드 z_m = A_m^ℓ x_m (r차원, r=8~16). **결합**: z̃_m = z_m + Σ_{k≠m} γ_{m←k} · (C_{m←k} z_k), C는 r×r 소행렬, γ는 **gradient-흐름 스칼라 게이트(LayerScale init 1.0, zero-init 금지 — 원장 키1)**. 출력 h_m = W₀x_m + B_m z̃_m. **cross-modal 정보가 r차원 병목(어댑터 부분공간)으로만 흐른다** — full-feature attention 아님. 결합 레이어 = 사전 config `COUPLE_LAYERS`(1~4단계).
- **[F] Dominant-mask forcing** (학습 전용, P49 해독): 학습 스텝의 p(=0.3) 확률로 지배모달(RGB) 입력을 마스킹 + 소수모달 경로에 hard-pixel aux CE(주손실 직결, 키1). 추론 무변경.
- **[R] (2단계) 신뢰도-게이팅**: γ를 **training-free 불일치도**(모달 예측 disagreement)로 변조. ⚠️ **1단계 프로브엔 미포함** — 결합 자체의 순이득을 먼저 격리(R은 H2/H3 인접이라 신중). N2 성공 시에만 추가.

**토글**: `MODEL.CMLC.ENABLE` 기본 off = 기존 전 모델과 byte-동일(가드 필수). C2-CMLC / F / R 각각 독립 토글.

## 3. 게이트 (사전 등록) — 하이브리드 분기

**1단계 최소 프로브**(cheap, 동일 백본 DINOv3-L, C2-CMLC+F on vs off, **매칭 시드 2점**):

| Δ = mean(CMLC on) − mean(CMLC off), legal DELIVER test | 판정 |
|---|---|
| **≥ +1.5** (단일시드 노이즈 std 2.2 초과 대역) | **인코딩-결합 기제 순이득 실증 → A 확장**(다중시드+MUSES 통일+동일백본 경쟁자 재구현) |
| 0 ~ +1.5 | 경계 — 시드 추가로 재판정 |
| **≤ 0** | **기제 무이득 → B 재프레이밍**(진단-구동 프레임워크 + anti-consensus 분석 헤드라인) |

- **MUSES 비회귀 필수**: 같은 config로 MUSES val ≥ base(−0.5 이내). DELIVER만 오르고 MUSES 깨지면 = C3와 같은 축-특이성 = 통일 실패로 간주.
- **조기 kill(ep30)**: base 궤적 대비 −1.0 이하 or γ 전부 0 수렴(결합 미사용).
- **falsifiable**: on/off 유일 변수 = 결합 → 이득 귀속 자명. dominant-mask 없이는 결합이 RGB-swallow로 죽을 것(P49 예측) → F 유무 A/B로 검증.

## 4. 노벨티 포지셔닝 (정직, CrossWeaver 선점 반영)

- **카테고리 "신뢰도 인코더 cross-modal"은 CrossWeaver(2604.02948)가 선점** — 주장 금지.
- **방어 가능한 delta 3축**(전부 CrossWeaver·MLE-SAM 미점유):
  1. **결합 위치 = LoRA/PEFT 부분공간**(feature-space 아님). 정독 선행 중 PEFT 부분공간 안 cross-modal 결합 = 0.
  2. **frozen 대형 VFM**(저들 학습 SegFormer-B0). 스택 상이.
  3. **소거-증명 동기**: 선택 축을 오라클+실현성 배터리로 닫고 결합을 도출 = 우리 고유 분석 자산.
- **최근접 선행 인용 의무**: MLE-SAM 2412.04220 · CrossWeaver 2604.02948 · CHARM 2508.03060 · MoBaNet 2603.17705 · CoLA 2604.03314(inter-modal LoRA pathway — **가장 가까운 위협, 단 VL/AV 이종·2인코더, dense 시각 seg 아님**).

## 5. 공정성 (필수 — 없으면 기각)

- **동일 백본 ablation(SpectraDINO Table 8 방식)**: CMLC on/off를 **동일 frozen DINOv3-L**에서 param-matched. + **경쟁자 재구현**: MLE-SAM 평균융합 · CrossWeaver MIB를 **우리 DINOv3-L 위에** 이식해 동일 조건 대결(≥1개 필수). → "큰 백본이 이긴다"를 "같은 백본, 더 나은 기제"로 전환.
- **소백본 이전**: CMLC를 DINOv3-S+에서도(ProbeA2 자산) — 이득 지속 시 "백본-무관 모듈"(Defense C) 성립.

## 6. 실행 계획

1. **구현 위임**(labcode 8/22 복귀 or GLM): `modules/cmlc.py`(CMLC 레이어) + `reliadino/`에 토글 배선 + config `configs/deliver/deliver_rgbdel_P51_cmlc.yaml`(base c3only에서 CMLC on만 차이) + MUSES config. 스모크(off=byte-동일 가드 / on=γ grad 흐름 / dominant-mask 손실 유한 / eval 결정론 / 파라미터 증가분). **코드 검수 파이프라인 의무**(fresh-eyes 7렌즈 + 등가 assert + 로더 실측 + ep30 토글 즉검).
2. **1단계 프로브**: DELIVER CMLC on/off × 2시드(매칭) + MUSES 1런. GPU = P50/시드 완주 후 해방분 or 신규 유휴.
3. **판정**(이 세션): §3 게이트 → A 확장 or B 재프레이밍.
4. C3 축-특이성은 **진단-구동 활성화**(class-transfer 붕괴 자동 탐지)로 재설계 — A·B 공통, 병행 설계.

관련: 원장 [research/hypothesis-ledger.md](../research/hypothesis-ledger.md)(H16 종결이 이 방향의 동기) · [decisions/2026-08-17-p50-map-modal-alignment-pretraining-proposal.md](2026-08-17-p50-map-modal-alignment-pretraining-proposal.md)(P50=초기화 개선, 직교·병행) · [analysis/2026-08-20-fusion-mechanism-double-negative.md](../experiments/analysis/2026-08-20-fusion-mechanism-double-negative.md)(믹서 비병목 → 결합은 믹서 아니라 인코딩)
