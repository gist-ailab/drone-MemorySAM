---
created: 2026-08-18
author: fable (discussion 세션, model-proposal 절차 — user 재개방 질의 "SoftMoE-LoRA를 DINOv3에서 되게, spatial modality routing")
status: 🟡 제안 — 대기열 등재(user 승인 2026-08-18). 구현·기동은 슬롯 확보 후
---

# Spatial-Modality Oracle Probe 제안 — SoftMoE 재개방의 천장 측정 (2026-08-18)

> **한 줄**: user가 "SoftMoE-LoRA 실패는 SAM2 백본/frozen/공간정보 훼손 탓일 수 있으니 DINOv3에서 spatial modality routing으로 되게 해보자"를 제기. 라우터를 **만들기 전에** CEA(H4)와 동형 논리로 **spatial 축의 상계를 먼저 잰다.** 학습 0(순수 추론), 4090 1장 하루. 결과가 축을 결정적으로 열거나(→실제 라우터 설계 정당화) 닫는다(→H1의 DINOv3 확장 완결).

## 1. 왜 지금, 왜 오라클 먼저 (진단 ↔ 근거)

| 우리 실측 | 함의 | 출처 |
|---|---|---|
| SoftMoE 게이트는 per-token 분석 시 **공간적으로 분화돼 있었다**(entropy_ratio 0.55, max_weight 0.72) — uniform은 공간평균 artifact | user 가설 "게이트가 공간을 못 봐서 실패"는 **이미 반증**. 잘 갈라 라우팅하고도 무이득 = 실패 위치가 라우터 품질이 아니라 "나눌 정보"에 있음 | CLAUDE.md 주의 #3, [status/history-2026H1.md](../status/history-2026H1.md) |
| drop-modality: 추론 시 event/lidar/depth 한계기여 ≈ 0(RGB 주면 잉여) | 추론 시점 모달 게이팅(spatial 포함)은 모달 합집합 정보 상한에 묶임 — 한계정보 0 위 가중은 뽑을 게 없음 | [experiments/analysis/2026-07-30-muses-drop-radar-ablation.md](../experiments/analysis/2026-07-30-muses-drop-radar-ablation.md), RGB-D fair-eval |
| H1(학습 게이트: UAMM·SoftMoE·quality gating) ✗ 반증, 단 **증거가 P10~P27 = SAM2 계보** | user 지적 타당: **spatial×modality 축의 DINOv3 재검증은 없다.** H4(CEA) oracle은 **조건 축**이라 spatial 축에 자동 이월 안 됨 | [research/hypothesis-ledger.md](../research/hypothesis-ledger.md) H1 |

→ **user가 연 축은 진짜로 미측정 축**이다(그 점에서 정당). 다만 위 두 실측이 "Δ는 작을 것"이라는 강한 사전확률을 준다. 프로브의 가치 = 이 사전확률을 **spatial 축 전용 측정 상계**로 전환해, 재개방 질문을 유추가 아니라 수치로 닫거나/여는 것.

## 2. 설계 — spatial-modality oracle (하드 부분집합 선택의 상계)

대상 ckpt = 확정 대표 DELIVER **P46 C3-only 67.79@ep70**(jarvis), M=4 modals = {rgb, depth, event, lidar}. **학습 없음.**

1. **부분집합 예측 맵**: 각 비공집합 S ⊆ modals(최대 2⁴−1 = 15종)에 대해, S 밖 모달을 zero-fill(현행 drop-modality 기계의 keep-subset 일반화)한 뒤 전체 세그 로짓맵 `P_S(x)` 산출. `P_full` = 전 모달.
2. **오라클 spatial 라우터**: 각 픽셀 x에서 `S*(x) = argmax_S [P_S(x)==GT(x)]` (어떤 부분집합이라도 x를 맞히면 맞은 것으로 채택). 합성맵 `O(x)=P_{S*(x)}(x)`. → **임의의 하드 픽셀별 모달-부분집합 라우터의 상계.**
3. **비교**: `Δ = mIoU(O) − mIoU(P_full)`, val·test 양쪽. per-class·per-condition Δ도 기록(어느 클래스/조건에서 spatial 선택 여지가 있는지 = user "모달별 spatial 정보 상이" 직관의 직접 검증).
4. **부가 진단**: 각 픽셀의 채택 `S*` 분포 — 대다수가 `full`/`rgb`면 "라우팅 여지 없음"이 그 자체로 결론. 단일모달이 이기는 픽셀 비율·공간 분포 시각화.

## 3. 게이트 (사전 등록)

| Δ = mIoU(oracle) − mIoU(full) | 판정 |
|---|---|
| **< 0.3** | **spatial-modality routing 축 폐쇄** — 완벽한 하드 라우터도 무의미. H1의 DINOv3 확장 완결(H16 반증). SoftMoE 재구현 재제안 금지 |
| 0.3 ~ 1.0 | 경계 — 여지 미미, 비용 대비 불리. 라우터 미착수, 원장 기록만 |
| **≥ 1.0** | **여지 실재** → 실제 라우터 설계 정당화(§5 레버). full run 별도 제안 |

- **falsifiable**: 오라클은 GT 치팅이므로 상계가 자명. Δ가 작으면 원인 귀속(라우팅으로 못 넘음)이 반박 불가.
- **사전확률**(§1) = Δ 작음. 이 프로브는 그 예측의 검정이다 — 예측이 맞으면 재개방 질문을 정량 종결, 틀리면(Δ≥1) 우리 진단 체계에 구멍이 있다는 뜻이라 그것대로 중대.

### 정직한 한계 (결과 해석 시 명기)
1. **하드 부분집합 선택의 상계**다. soft 혼합은 연속 공간이라 초과 가능 — Δ<0.3이 soft 라우팅까지 절대 배제하진 않음. 완화: 부가로 per-pixel best-of{단일모달·full·모달평균} richer set도 리포트.
2. **출력 레벨 측정**(SoftMoE는 encoder-token 레벨 라우팅). 토큰 라우팅 후 디코딩이 재조합할 여지는 있으나, 출력 레벨은 **정보 상한의 최청정 프록시** — 토큰 라우터가 모달 피쳐에서 복원 불가한 출력 정보를 만들 수는 없음.

## 4. 노벨티/원장 위치

- **신규 가설 H16**(잠정): "spatial×modality routing은 DINOv3에서 SAM2-SoftMoE가 못 잡은 여지를 가진다." 프로브가 확인/반증.
- H4(CEA, 조건 oracle)의 **spatial 축 자매편** — 같은 방법론(치팅 상계)으로 다른 축을 닫거나 연다. 음성이면 논문의 "적응 기제 전 축 소거" 절에 **spatial 축까지 명시적으로** 추가(서사 강화).
- 재제안 금지 저촉 없음: H1은 구현 반증, 이건 **새 측정**(oracle spatial ceiling, 미실시).

## 5. Δ≥1일 때의 라우터 레버 (조건부, 참고용 — 이번 스코프 아님)

상계가 열리면 그때 설계: load-balancing/importance loss(Shazeer 2017) · expert-choice routing(Zhou 2022, collapse 구조 회피) · **게이트 gradient-흐름 init**(원장 키1, zero-init 금지) · router를 scratch 학습 말고 per-modal confidence/entropy 신호에서 유도 · router LR 분리 · noisy top-k. **단 §1 상한 논거상 성공해도 천장 낮음 — Δ 측정 없이 착수 금지.**

## 6. 실행

- **구현**(위임 코딩 + 검수): `tools/oracle_spatial_modality.py` = `eval_reliadino_ckpt.py` drop-modality(zero-fill)를 **keep-subset 마스크**로 일반화 + 픽셀별 오라클 선택 + Δ/분포 리포트. 스모크(합성 미니, 2모달 4부분집합, 오라클≥full 단조성 assert). 새 학습 코드 0.
- **비용**: 순수 추론. DELIVER val(~500)+test 15 forward-config × 2 split, 4090 1장 수 시간.
- **슬롯**: P50 프로브(yeon)·시드런(jarvis) 무간섭 — 학습 0이라 **아무 빈 GPU 1장**이면 됨(시드런 완주로 jarvis 비면 거기, 아니면 다른 유휴 1장). plan.md 등재.
- MUSES(M=3~4)로도 동일 프로브 확장 가능(2차) — DELIVER 먼저.

관련: [research/hypothesis-ledger.md](../research/hypothesis-ledger.md) H1·H4 · [decisions/2026-08-08-condexpert-adapter-probe-proposal.md](2026-08-08-condexpert-adapter-probe-proposal.md)(CEA — 방법론 원형) · [decisions/2026-08-17-p50-map-modal-alignment-pretraining-proposal.md](2026-08-17-p50-map-modal-alignment-pretraining-proposal.md)
