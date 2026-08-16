---
created: 2026-08-10
author: fable (MMSAM discussion 세션, model-proposal 스킬 절차)
status: 🔴 **계열 종결(2026-08-16, §6 규칙 집행)** — DELIVER 미달 + MUSES 공식 val −0.97 + G-4M 실패. 통일 아키텍처 = ReliaDINO 계보 + 진단-짝 손실. 회수 자산 = H13·H14·INJECT-off 실증·radar fog/day 특이 기여
---

# P49-AIR 제안 — Asymmetric Injection with RGB-primary (2026-08-10)

> **한 줄**: 대칭 융합 트렁크를 버리고, **온전한 RGB 주경로(DINOv3-L 부분 fine-tune) + 인코더-내부 비대칭 보조 주입(zero-init 게이트)** 구조로 전환한다. 근거는 발명이 아니라 해부 — 현행 SOTA(MM SAM-adapter)가 우리와 같은 입력으로 이기는 이유를 그들의 ablation에서 역산했고, 그 병리("RGB 오염")는 우리 원장에도 3회 독립 기록돼 있다.
> 딥리서치 3축(해부/노벨티/벤치, 2026-08-10) + 가설 원장 H1~H12′ 기반. 절차 = model-proposal 스킬 6단계.

## 1. 진단 ↔ 문헌 대응 표

| 우리 실측 (원장/분석) | 문헌 근거 (arXiv) | 함의 |
|---|---|---|
| 매칭 구성 열세: RGB-L MUSES −1.50, RGB-D DELIVER test 54.14 | MM-SA 2509.10408 Tab.4: **frozen+LoRA 49.77 = 최약 구성**(RGB-only 53.32보다 낮음), full FT 57.14 | **frozen+LoRA가 구조적 열세 원인** — 백본을 풀어야 함 (layer-wise LR decay 0.9) |
| "RGB 건드리면 clear/day 손해" 3회: C3 MUSES 이식 −0.765(clear/day 집중), ProbeA2 H+ clear_day −3.06, P44 fog_night 파국 | MM-SA: 비대칭 설계로 RGB-hard **+7.53** / RGB-easy +0.7 | 대칭 융합이 easy 픽셀에서 RGB를 오염 — **RGB 주경로를 init-identity로 보존**하고 보조는 편향만 주입 |
| 융합 고도화 5세대 저수익 (H1~H4, CEA oracle +0.2) | MM-SA Tab.9 분해: 융합 기여 **+1.16뿐**, 백본+adapter가 +5.3 | 융합 경쟁은 원리적 저수익 — 자원을 백본·주입 구조에 배분 (원장과 문헌 합치) |
| lidar rank 붕괴 → VICReg 복원 (H8, P39.1) | sparse 모달에 conv prior 유리 (MM-SA ConvNeXt-S aux) | 보조 모달 인코더를 ViT+LoRA → **CNN(ConvNeXt-S)** 전환 |
| query 경로 = dense 복제 (H9, 순기여 −0.09) | — | P49에서 **query 경로 제거**, 단일 head — 중복 제거로 "얹으면 흡수" 병리 원천 차단 |
| fog에서 융합이 camera-only 우위 (우리 drop-lidar fog_night 7.2) | Codabench 실측: fog GtA 72.64 < MM-SA 74.12·DGFusion 73.61 | fog 조건이 융합의 존재 증명 — 게이트에 fog 방어선 포함 |

## 2. 변경 목록 (전 항목 토글 — ablation 분해 보장)

| # | 변경 | 근거 키 | 토글 |
|---|---|---|---|
| A1 | RGB 주경로 = **DINOv3-L 부분 fine-tune** (layer-wise LR decay 0.9; LoRA 폐지) | 진단표 1행 | `P49.RGB_FT` (off=현행 frozen+LoRA) |
| A2 | 보조 모달(depth/lidar/event) = **ConvNeXt-S 인코더** (모달별) | 진단표 4행 | `P49.AUX_CNN` |
| A3 | **인코더-내부 비대칭 주입**: N=4 지점 injector–extractor(멀티스케일 deformable cross-attn), **zero-init γ** — RGB 경로 init-identity | 진단표 2행 | `P49.INJECT` |
| A4 | head = 멀티스케일 피라미드 → 기존 M2F-lite 유지, **query 경쟁 경로 제거** | 진단표 5행 | `P49.MS_HEAD` |
| A5 | 손실 = OHEM CE(채택) + **C3 prototype 유지**(클래스축) + VICReg(보조 인코더) | H8 실증 | 기존 토글 유지 |

**⚠️ 키1(zero-init 잔차 4연속 사망)과의 충돌 논증 — 필수 숙지**: 과거 사망 사례는 *이미 중복된 시스템(dense∥query)에 모듈을 얹은* 경우로, 대체 경로가 gradient를 흡수했다. A3의 zero-init은 **보조 정보가 모델에 들어오는 유일한 경로**(A4로 중복도 제거됨)라 흡수 불가 — gradient 압력이 구조적으로 보장된다. MM-SA에서 동일 구조가 실증됨(RGB-hard +7.5). **검증 장치**: γ 노름을 매 eval 로깅(`p49/gamma_*`) — ep30에 γ≈0 정체면 키1 재발로 판정, 즉시 중단.

## 3. 노벨티 포지셔닝 (과장 금지)

- **최근접 선행 = MM SAM-adapter(2509.10408)** — 구조 계열이 같음을 명시하고 시작한다. 차별 축 4개:
  1. **백본 비교의 최초 제공**: 그들은 SAM v1 단일 백본(백본 ablation 부재). 우리는 동일 주입 구조에서 **DINOv3(self-distilled) vs SAM(SA-1B mask-centric)** 을 직접 비교 — "mask-centric 사전학습이 필수인가"라는 미답 질문에 답한다.
  2. **학습전용손실 스위트**: 그들은 OHEM뿐. 우리는 진단-짝 손실(클래스축↔C3 prototype, 조건축↔C2 masked consistency[측정 대기]) — 노벨티축 조사 결과 "지도 멀티모달 + 축진단 짝짓기" 프레임은 미점유 (MIC 2212.01322·SePiCo 2204.08808은 RGB UDA).
  3. **진단 체계**: RGB-easy/hard 렌즈(그들 방법론 채택·인용) + 우리 축분리·drop-modal 인과 분석 — "왜 작동하는가"의 깊이.
  4. (opt, 후순위) MUSES event **3초 스트림** 시간 인코딩 — 벤치 합법 유일 시간 신호, 미점유. 단 현행 정적 event 기여 +0.217이라 천장 캘리브레이션 선행(별도 프로브 게이트 없이는 본학습 금지).
- 제약 준수: DGFusion/CAFuser 유사 구조 아님(조건 토큰·융합 게이팅 없음), 외부 신호 불사용, 단일 아키텍처(DELIVER·MUSES 공용).
- **공정성 트레이드오프 정직 기록 (구현 후 정정 2026-08-10)**: 학습 파라미터 ≈**0.5B**(백본 304M + ConvNeXt-S×3 ≈150M + 주입/헤드) — §3 초안의 "≈300M"은 백본만 센 수치였다. 2모달 매칭 구성에서는 MM-SA(백본 308M+aux 1개)와 대등, 4모달은 aux 3개라 더 큼 — 논문에 구성별 파라미터 표 공개. "frozen foundation" 효율 서사는 포기, H12′ 유지.
- **구현 확정 사항 (검수 세션 판정 2026-08-10)**: ① **γ 게이트 2종**(injector + pyramid) — extractor→헤드 경로도 보조 의존이라 identity 보존에 2종 필수(스모크 C 검증). γ=0 step-0에서 보조 인코더 gradient 출구는 VICReg(기본 on). ② **HEAD_MODE=pixel 확정** — 픽셀 헤드가 주 출력, M2F-lite는 독립 보조손실(deep supervision 실증 자산과 정합, 추론 결정론); query 직접출력 팔은 ablation 토글로 유지. ③ DEFORM 미구현은 명시적 raise(조용한 폴백 금지), vanilla attn + KV_GRID 64 예산. ④ **INJECT와 grad-ckpt 상호배제**(hook 오염) → 1024² full-FT는 40GB 기준 설계 — **24GB 실측이 Phase 2 선행 관문**(OOM 시 KV_GRID 32/부분 FT 검토).

## 4. 게이트 (사전 등록)

| 시점 | 게이트 | 미달 시 |
|---|---|---|
| ep30 (조기) | ① γ 노름 성장(≈0 정체 아님) ② **RGB-clean 무손실**(base 대비 ≥−0.3) ③ RGB-degraded 개선 방향(+) | ①흡수 재발 → 중단·키1 재판정 ②③ RGB 오염 재발 → 중단 |
| DELIVER 완주 | **legal test ≥ 57.35**(SOTA, primary) · val ≥ 69.60(보조) · thin-class 유지(RailTrack ≥62) | 미달 시 A1~A4 토글 ablation으로 기여 분해 후 부분 채택 판단 |
| MUSES 완주 | 3모달 79.788 초과 + **published 1위(MM-SA 81.07) 사정권 ≥80.5** · **fog ≥74**(융합 우위 조건 방어) | fog 붕괴 시 P44 재림으로 판정 |
| **4모달 복원 (G-4M, user 목표 정합)** | **P49 4모달 ≥ P49 3모달** (radar 추가 무손실) — 비대칭 주입의 "유해 모달이 RGB를 오염 못 함" 예측의 직접 검증 = **H11 재검증**. 통과 시 "모달을 늘려도 깨지지 않는 융합"이 논문 주장으로 승격 + **4모달 구성이 양 벤치 공식 대표**가 됨 | 미달 시(4<3 지속) radar 유해는 구조 무관으로 H11 ✗ 재확정, 대표 구성은 벤치별 최적(DELIVER 4모달/MUSES 3모달)으로 정직 보고 |
| falsifiable | 비대칭 주입(A3) on/off A/B: hard 조건 ≥+2.0 & easy ≥−0.3 — "비대칭이 오염을 막는다"의 직접 검증 | A/B 무차이면 구조 전환 무효, MM-SA 우위는 백본(SAM) 요인으로 귀속 |

기대치 캘리브레이션(벤치축): DELIVER 격차 −0.36은 백본급 노이즈 안쪽 — A1+A3의 문헌 이득(+1.8~7.4 대역)이면 충분히 사정권. MUSES −1.28(vs MM-SA)은 구조 전환 몫. GtA(−2.6)는 익명·미발표라 **논문 비교군에서 제외하고 "published methods 중 1위" 스코프**로 주장(fog에선 우리 축이 이김).

## 5. 실행 계획

1. **Phase 0 (학습 0, 즉시)**: ⚠️ **분할 정의 교체(2026-08-10 판정)** — MM-SA의 easy/hard는 수동·시각검사 분할로 미공개라 재현 불가(그들 repo에 산출물 없음, 1797/100 전수 수동 분할). 대체 = **메타데이터 기반 재현 가능 분할**: `RGB-degraded` = RGB 손상 case(Motion-Blur/Over-Exposure/Under-Exposure) ∪ night 조건, `RGB-clean` = 나머지. **proxy임을 명시**하고 MM-SA 수치(57.75/45.46)와의 직접 수치 비교는 하지 않는다(분할 상이) — 패턴 비교만. 부수 이득: 그들 분할의 비재현성 비판 + 재현 가능 대안 제공이 논문 방법론 기여가 됨. 현행 P46 ckpt의 clean/degraded 분해 측정, 유휴 GPU 1장 ~2h.
2. **Phase 1 구현**: labcode 위임(대규모 — A1~A4). conventions 코드 검수 파이프라인 의무(fresh-eyes 7렌즈 + 스모크 grad/등가 + 로더 실측). 신규 `semseg/models/reliadino/p49.py` + MODEL_REGISTRY 등록.
3. **Phase 2 본런**: DELIVER 먼저. ⚠️ **해상도 확정(2026-08-10 실측)**: @1024 full-FT는 24GB에서 forward OOM(no-go, KV_GRID 32도 무효 — ViT-L 활성화 지배·grad-ckpt는 INJECT와 상호배제) → **@768² 학습 + 완주 후 @1024 평가**로 확정. 이는 검증된 최선 프로토콜(P46 최고 69.44/56.99 = @768 학습+@1024 평가)과 동일해 공정 비교 정합. config `deliver_rgbdel_P49_air_768.yaml`. jarvis 1,2,5(3×4090), eff-batch 16 accum. 완주·게이트 판정 후 MUSES.
4. C2 측정(A100 확보 시)·7B 프로브(진행 중)는 독립 병행 — 결과에 따라 §3-2 손실 스위트·백본 선택 보강.

관련: 원장 [research/hypothesis-ledger.md](../research/hypothesis-ledger.md) · 딥리서치 원문 = 본 문서 §1 인용(해부/노벨티/벤치 3축, 2026-08-10) · [2026-08-08-condexpert-adapter-probe-proposal.md](2026-08-08-condexpert-adapter-probe-proposal.md)(적응 계열 폐쇄) · MM-SA 2509.10408 · MIC 2212.01322 · SePiCo 2204.08808 · MUSES 2401.12761

## 6. 단일-아키텍처 통일 규칙 (사전 등록, 2026-08-15 — user 원칙 재확인에 따라)

**원칙(user)**: "모델 제안을 데이터셋별로 할 수는 없다 — 두 데이터셋에서 하나의 아키텍처로 문제를 푼다." 현재 두 계보 병렬(DELIVER=P46-CTR, MUSES=P49-AIR)은 위반이 아니라 **최종 1개를 고르기 위한 후보 선발전**이며, 아래 규칙으로 종결한다:

| MUSES P49.1 3모달 판정 | 통일 결과 |
|---|---|
| **승리** (공식 test > 79.788) | P49-AIR 승격 → **DELIVER 재도전 의무**: P49 + C3 토글(@768)로 P46-CTR 대체 검증. 통과해야 단일 아키텍처 = P49-AIR. DELIVER 재도전 실패 시 아래 행으로 폴백 |
| **패배** (현 추세) | P49 계열 종결(원장 기록, N-E 발견으로 회수) → **통일 아키텍처 = ReliaDINO 계보(P39.1-rank 추론 그래프) + 진단-짝 학습손실 스위트** |

**중요한 사실**: 패배 시나리오의 통일은 재작업이 아니라 **이미 성립해 있다** — 양 벤치 현재 최고(DELIVER P46-CTR 56.99 / MUSES P39.1-rank 79.788)는 **추론 아키텍처가 동일**하다(C1/C2/C3는 전부 학습 전용 — RCS=샘플러, MCC=EMA teacher, prototype=손실; 추론 그래프 = P39.1-rank 그대로). 즉 skill §2의 단일-모델 원칙은 추론 그래프 기준으로 현 최고 조합이 이미 충족하며, 벤치별 차이는 "축 진단에 따른 학습손실 선택"뿐이고 이것이 논문의 방법론(진단-짝 프레임) 그 자체다. 논문 기재 시 이 구조를 명시(감추지 않음).

G-4M은 P49 전용 게이트이므로 패배 시 소멸 — 4모달 서사는 "radar 유해 실측 → 3모달 최적" 정직 보고로 대체.
