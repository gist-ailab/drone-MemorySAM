---
created: 2026-07-16
scope: P36(및 ReliaDINO 계보) 노벨티 비판적 판정 — 실측 전거 기반
inputs: 표준분석(P29~P36, NAS analysis_logs) · G0c full-res toggle · A-1 probe · MUSES 공식 test · doc 18 novelty 스윕 · legal 수치(val-best ckpt 규칙)
---

# P36 노벨티 비판적 리뷰 — "무엇이 살아남고, 무엇을 내려놓아야 하나"

## 0. 전제가 되는 정직한 수치 (legal = val-best ckpt 기준)

| 모델 | val | test | vs test-SOTA(DGFusion 56.71) | 비고 |
|---|---|---|---|---|
| P34 | 68.19 | **56.62** | **−0.09** | PhysAug ON = 공정선 밖(unfair-ours) |
| P35 (공정 레시피) | 67.61 | 55.52 | −1.19 | **우리의 진짜 공정 위치** |
| P36 (P35+router) | 67.74 | 55.62 | −1.09 | router +0.10(legal)/+0.76(D1 mean) |
| P36+physaug (ep64 중단) | **68.76** | 54.18 | −2.53 | val 신기록이나 test 미성숙 |
| val-SOTA | CAFuser-CAA 68.79 | — | — | 68.76도 −0.03 미달 (게다가 physaug) |
| MUSES test | — | **78.979** | DGFusion −0.52 / CAFuser **+0.48(모달 −1)** | 유일한 대외 방어 가능 결과 |

**사용자 전제 검증**: ① "P34가 SOTA보다 높지 않다" — **맞음** (test −0.09·val −0.60, 그것도 physaug 포함 시). ② "RBMA류 모듈이 작동하지 않는다" — **bias 계열은 맞음**(아래), 단 router와 calibration 신호는 구분해야 함.

## 1. 모듈별 노벨티 후보 → 실측 판정

| 후보 | 실측 | 노벨티 판정 |
|---|---|---|
| **RBMA-v2 pre-softmax additive bias** (λ1·B_cal+λ2·B_cons) | P31-eval·P32(CoRB p=4.5e-22 순손해)·P34/P35/P36 toggle **전부 Δ≈0**, G0c full-res에서도 strip 효과 0 | **사망. 주기여로 못 씀.** 셀 자체도 PRIMED/SAE 인접(“first additive bias” 금지, doc 18). 살릴 길 = **negative finding**: "soft pre-softmax bias는 4세대·2백본에서 결정을 바꾸지 못했다"는 체계적 반증 — ablation 절 재료로는 오히려 강함 |
| **Competence gate + calibrated self-entropy + veto** | G0c: gate+calib off 시 test −0.26 (미세 실기여), veto ≈0. 단 **신호 품질은 계보 최초 4모달 균형 AUROC [.85,.78,.87,.70] + 수렴 후 유지** | **성능 기여 경로 약함**. "신호는 옳은데 이득이 작다" — 신호 자체(training-free calibrated reliability)는 router의 anchor 재료로서만 가치 실증 |
| **Per-class reliability-anchored router (P36 핵심)** | **유일한 대형 유효 모듈**: P35→P36 +0.76(D1)/+0.10(legal), off 시 −38~42(지배 경로), Wall 6.0→13.3·Water 5.3→9.5·RailTrack →62.5 부활, **SAM2 계보(P31 +10.7~13.8)와 DINOv3 계보에서 이중 재현**, 공간 증거(차선·thin-class 패널) | **살아남는 유일한 모듈 novelty** — 단 아래 §2의 세 가지 공격 지점 방어 필요 |
| **DINOv3 frozen + per-modal LoRA 프레임** | 성능의 실제 원천(+2.6~4.1). 그러나 "frozen VFM+adapter" 자체는 관용 기법 | 모듈 novelty 아님. **분석 기여로 전환**: SAM 계열 MMSS 정체의 피쳐-레벨 진단(rank 1.26 붕괴·CKA 0.1 비정렬) + A-1 통제 probe(+11.6) = "백본이 지배 변수"의 실증 연구 |

## 2. Router novelty의 공격 지점과 방어 (비판적)

1. **"MoE gating의 변형 아니냐"** — per-class/per-pixel 모달 라우팅 자체는 기존 있음(BiXFormer query-MMSS, CMNeXt hub, MoE 계열). 우리 셀 = "**training-free calibrated reliability를 anchor로 한 zero-init per-class router + frozen-VFM fusion head에 residual 주입(collapse-safe)**" — doc 18 스윕 기준 미점유이나 정확히는 **unfalsified**(조합 novelty, 얇음). 방어 자료: zero-init anchor의 collapse 방지 이력(P10-P27 gate 상수수렴 계보), 이중 백본 재현, thin-class 공간 증거.
2. **"off Δ+40 = 기여"로 쓰면 안 됨** — 그것은 *의존도*(residual alpha가 주 경로화)이지 *증분 가치*가 아님. 증분 가치는 **+0.76/+0.10**뿐. 논문에 +40을 기여처럼 쓰면 자멸.
3. **"제안 모듈 < 증강 하나"** — router(+0.76)가 physaug(+1.12)보다 작다는 비교가 표에서 그대로 보임. 방어 경로 = **router×physaug 독립 축 실증**: P36+physaug가 val에서 시너지(68.76, 계보 최고 + P34의 ep120 도달치를 ep20에 도달 = 수렴 6×)를 이미 보였고 test는 ep64 중단으로 미완 — **완주가 이 공격의 유일한 방어**.

## 3. 종합 판정 — 논문 포지셔닝 재구성 (권고)

**현 상태로 "새 모듈 제안 + SOTA" 논문은 성립하지 않는다** (모듈 대부분 무효 + test 미달). 생존 가능한 3가지:

- **A. 실증 연구 중심(추천)**: "VFM 백본이 MMSS의 지배 변수다" — SAM 계열 피쳐 붕괴 진단(rank/CKA) + A-1 통제 probe + **additive-bias 계열의 체계적 무효 실증(negative)** + 유일 생존 기전 router의 이중 재현. 우리가 세계 최고 수준의 분석 데이터를 보유한 축.
- **B. Modality-efficient robustness**: **MUSES를 주 결과로**(3모달로 4모달 CAFuser +0.48, night 하락 −3.45) + DELIVER는 보조. 모듈 서사는 "calibrated-reliability router" 하나로 좁히고 bias는 ablation의 negative로.
- **C. (보강 후) on-par SOTA**: P36+physaug 완주(ckpt 구조됨, resume 가능) 또는 TTA로 test 56.71 돌파 시에만. TTA는 CAFuser/DGFusion의 TTA 사용 여부 정합 필수(공정성).

**내려놓아야 할 것**: "RBMA" 브랜딩의 성능 주장 — 실측과 정면 모순. bias 기전은 negative finding으로 전환하고, router는 RBMA와 분리된 기전으로 명명·제시할 것. val 68.76을 헤드라인으로 쓰는 것 — physaug(공정선 밖)+test 꼴찌(54.18)라 표에서 자멸.

## 4. 액션 (test를 올릴 실제 레버, 타 세션 제안과 정합)

1. TTA/MSF (학습 0, NAS ckpt로 즉시) — 단 baseline TTA 사용 여부 정합 후 두 행 병기.
2. **P36+physaug resume 완주** (lecun/bengio; test 궤적 55.26→55.39→55.60 상승 중 중단) — §2-3 방어의 핵심.
3. class-transfer 트랙 (복구 상한 +7.9pt) — 0.09 싸움이 아닌 구조적 레버.
4. val-SOTA 기준 정합(CAFuser 68.79 vs 68.6) — 논문 전 필수.
