---
created: 2026-08-31
author: fable (discussion 세션) — user 지적("벤치별 토글 불가, 모델이 자연스럽게 학습하거나 일괄 적용") 수용 개정, user 승인
status: 🟢 개정 확정 — 컨트롤러 구현 위임 중. 본런은 UniBal 고정런 완주(~2일) 후
---

# P52 개정 — RxDINO: 단일-config 자기-적응 처방 (2026-08-31)

> **개정 사유**: 원 P52("오프라인 진단이 벤치별 C3 on/off를 결정")는 사람이 하는 per-dataset 튜닝과 구분 불가 — 단일 메서드로 방어 불가(user 지적, 타당). **개정 = 진단의 온라인화**: 학습 중 계산되는 병리 지표가 처방 강도를 연속 조절, **3벤치 완전 동일 config**. 벤치별 차이는 창발로만.

## 1. 개정 정의

**RxDINO (P52)** = frozen DINOv3-L + 모달별 LoRA + gated-MLP trunk(+VICReg) + FPN/M2F-lite (추론 그래프, 3벤치 동일 — 불변)
- **+ P50 정렬-사전학습 init** (H22 ✓; MCubeS는 Phase2 EXT init 완성 시 적용, 그 전 런은 무-init으로 명기)
- **+ C3-adaptive**: per-class λ_c = f(온라인 붕괴 지표) — 학습 배치 혼동 EMA에서 클래스별 "흡수도" s_c(= (1−recall_c) × top-1 confuser 집중도)를 계산, λ_c = λ_max·clamp(s_c/τ, 0, 1). 붕괴 클래스만 prototype 당김을 받음.
- **+ UniBal-adaptive**: per-modal λ_u,m = f(온라인 laziness 지표) — 모달별 unimodal-head 손실의 상대 갭(L_m/mean−1, clamp)으로 게으른 모달만 보조 CE 강화. (선례: OGM-GE CVPR'22 — 학습-내 모달 기여 측정→변조)
- 두 컨트롤러 모두 **학습 전용·train-배치 통계만 사용(val 불사용)**·EMA 평활·warmup·λ 궤적 로깅(창발 증거 = 논문 그림).

## 2. 게이트 (사전 등록)

| 게이트 | 기준 |
|---|---|
| **G1 DELIVER** | adaptive 단일-config legal test ≥ 최선 고정팔(P50init+C3on = 54.95, N8 최종치로 갱신) − 0.3 |
| **G2 MUSES** | ≥ 최선 고정팔(UniBal 고정런 판정 후 확정; 현행 82.62 val 계열) − 0.3 |
| **G3 MCubeS** | ≥ 최선 고정팔(C3-off 58.07±0.49) − 0.3 |
| **G4 창발 (정성, 필수)** | λ 궤적이 벤치별 병리를 스스로 재현: DELIVER=RailTrack λ↑ / MUSES=전 클래스 λ≈0 & radar λ_u↑ / MCubeS=rubber λ↑ — **config는 동일한데 궤적이 갈라짐**이 "자기-진단" 주장의 직접 증거 |
| 실패 조건 | 어느 벤치든 고정팔 대비 −0.3 초과 열세 → 컨트롤러 지표/시정수 1회 수정 재시도, 재실패 시 개정 철회(고정팔 결과로 정직 보고 + limitation) |

## 3. 기존 실험의 지위 (버려지는 것 없음)

- C3 고정 on/off ×3벤치(H20 dose-response) = **컨트롤러 설계 근거 + adaptive의 채점 기준**(각 벤치 최선 고정팔).
- UniBal 고정런(진행 중) = MUSES 대조점 + λ_u 지표 캘리브레이션 재료.
- P50/N6/N7/소거체인 = 그대로 (init·선택 프로토콜·미니멀 융합 정당화).

## 4. 실행

1. 컨트롤러 구현 = GLM 위임 + 이 세션 검수(기본 off byte-동일 가드·합성 스모크: 인위 붕괴/게으름 주입 시 λ 반응·eval 불변) — 즉시.
2. UniBal 고정런 완주·판정(~2일) → G2 기준 확정.
3. **P52 본런 = 3벤치 × 단일 config × (시드 2 이상)** — 슬롯: yeon/hpca100 해방분.

관련: plan.md P52 행 · [research/hypothesis-ledger.md](../research/hypothesis-ledger.md) H20/H22 · [2026-08-21-p51-crossmodal-lora-coupling-proposal.md](2026-08-21-p51-crossmodal-lora-coupling-proposal.md)(§4 하이브리드 계보)
