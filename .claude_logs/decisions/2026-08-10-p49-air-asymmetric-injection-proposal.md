---
created: 2026-08-10
author: fable (MMSAM discussion 세션, model-proposal 스킬 절차)
status: 🟢 승인(2026-08-10 user) — Phase 0 측정 + Phase 1 구현(labcode) 동시 진행 중
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
- **공정성 트레이드오프 정직 기록**: A1로 학습 파라미터가 ~300M이 됨 — MM-SA(SAM ViT-L FT ~308M)와 대등해져 매칭 비교는 깔끔해지나, "frozen foundation + 소량 어댑터" 효율 서사는 포기. H12′(용량 정합 방어 불가)는 유지 — S+급 스케일링 행 별도 보고.

## 4. 게이트 (사전 등록)

| 시점 | 게이트 | 미달 시 |
|---|---|---|
| ep30 (조기) | ① γ 노름 성장(≈0 정체 아님) ② **RGB-easy 무손실**(base 대비 ≥−0.3) ③ RGB-hard 개선 방향(+) | ①흡수 재발 → 중단·키1 재판정 ②③ RGB 오염 재발 → 중단 |
| DELIVER 완주 | **legal test ≥ 57.35**(SOTA, primary) · val ≥ 69.60(보조) · thin-class 유지(RailTrack ≥62) | 미달 시 A1~A4 토글 ablation으로 기여 분해 후 부분 채택 판단 |
| MUSES 완주 | 3모달 79.788 초과 + **published 1위(MM-SA 81.07) 사정권 ≥80.5** · **fog ≥74**(융합 우위 조건 방어) | fog 붕괴 시 P44 재림으로 판정 |
| **4모달 복원 (G-4M, user 목표 정합)** | **P49 4모달 ≥ P49 3모달** (radar 추가 무손실) — 비대칭 주입의 "유해 모달이 RGB를 오염 못 함" 예측의 직접 검증 = **H11 재검증**. 통과 시 "모달을 늘려도 깨지지 않는 융합"이 논문 주장으로 승격 + **4모달 구성이 양 벤치 공식 대표**가 됨 | 미달 시(4<3 지속) radar 유해는 구조 무관으로 H11 ✗ 재확정, 대표 구성은 벤치별 최적(DELIVER 4모달/MUSES 3모달)으로 정직 보고 |
| falsifiable | 비대칭 주입(A3) on/off A/B: hard 조건 ≥+2.0 & easy ≥−0.3 — "비대칭이 오염을 막는다"의 직접 검증 | A/B 무차이면 구조 전환 무효, MM-SA 우위는 백본(SAM) 요인으로 귀속 |

기대치 캘리브레이션(벤치축): DELIVER 격차 −0.36은 백본급 노이즈 안쪽 — A1+A3의 문헌 이득(+1.8~7.4 대역)이면 충분히 사정권. MUSES −1.28(vs MM-SA)은 구조 전환 몫. GtA(−2.6)는 익명·미발표라 **논문 비교군에서 제외하고 "published methods 중 1위" 스코프**로 주장(fog에선 우리 축이 이김).

## 5. 실행 계획

1. **Phase 0 (학습 0, 즉시)**: DELIVER RGB-easy/hard 분할 재현 + 현행 P46 최고 ckpt의 easy/hard 분해 측정 — "대칭 융합의 easy 오염" 병리를 우리 수치로 확인(제안의 motivation 그림). 유휴 GPU 1장, ~2h.
2. **Phase 1 구현**: labcode 위임(대규모 — A1~A4). conventions 코드 검수 파이프라인 의무(fresh-eyes 7렌즈 + 스모크 grad/등가 + 로더 실측). 신규 `semseg/models/reliadino/p49.py` + MODEL_REGISTRY 등록.
3. **Phase 2 본런**: DELIVER 먼저(jarvis/yeon — MM-SA가 2×3090으로 학습했음이 실증, 자원 블록 아님. grad-ckpt+accum으로 eff-batch 16 유지). 완주·게이트 판정 후 MUSES.
4. C2 측정(A100 확보 시)·7B 프로브(진행 중)는 독립 병행 — 결과에 따라 §3-2 손실 스위트·백본 선택 보강.

관련: 원장 [research/hypothesis-ledger.md](../research/hypothesis-ledger.md) · 딥리서치 원문 = 본 문서 §1 인용(해부/노벨티/벤치 3축, 2026-08-10) · [2026-08-08-condexpert-adapter-probe-proposal.md](2026-08-08-condexpert-adapter-probe-proposal.md)(적응 계열 폐쇄) · MM-SA 2509.10408 · MIC 2212.01322 · SePiCo 2204.08808 · MUSES 2401.12761
