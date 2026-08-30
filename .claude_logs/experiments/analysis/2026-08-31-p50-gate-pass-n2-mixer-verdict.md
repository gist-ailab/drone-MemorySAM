---
created: 2026-08-31
type: P50 게이트 통과 + N2 믹서 판정 (H21·H22) — P52 구성 확정 재료
---

# P50 게이트 통과(+0.74) · N2 "평균 ≥ 우리 트렁크"(H21) — 2026-08-31

> 실측 = 모니터링 세션(yeon, 하네스 가드 PASS 상태, val.py native-GT BS1). 판정 = discussion 세션(fable).

## 1. P50-MAP finetune — 게이트 통과 (H22 ✓)

| 비교 | 값 |
|---|---|
| P50-pretrained finetune (seed821 매칭, ep30 val-best) legal test | **54.95** |
| 무-사전학습 seed821 (재선택 54.21 / 구 53.57) | Δ **+0.74 / +1.38** |
| 게이트 (사전등록: ≥ +0.5) | ✅ **통과** (양 기준 모두) |

- **정렬 사전학습(Places365 200k pseudo-모달, 어댑터만)이 실제 이득** — 캠페인에서 "정보 축"이 처음으로 양성. 유일차 = 초기화(시드·레시피 매칭).
- ⚠️ 특성: 이득이 **초반 수렴에 집중**(val-best ep30) — 이후 trainer 궤적 지속 하락(ep200 58.55, 과적합/망각 추정). 사전학습 init은 "더 좋은 출발점"이지 "더 좋은 종착점" 아님 → 실무 처방 = 조기 정지와 결합. 후속(값쌈): P50 런 자체의 legal-val 재선택(대칭 프로토콜)로 최종치 확정.
- 함의: **P52 편입 확정** + 제안서의 "통과 시 확장"(코퍼스/에폭 스케일업) 후보로 등록. **논문의 정량 설계-기여 한 줄 확보**: "frozen 백본 위 어댑터 정렬 사전학습 +0.74~+1.38 (매칭 통제)".

## 2. N2 mean-fusion — "우리 gated-MLP 트렁크, 평균 대비 우위 없음" (H21 반증)

| 믹서 (동일 백본·레시피, DELIVER legal test) | 값 |
|---|---|
| **산술 평균** (MLE-SAM 방식, seed821, +VICReg 유지) | **55.45** |
| gated-MLP (우리, seed821 재선택 54.21 · 5-seed 54.39±0.76 · 최고 단일 55.40) | 평균이 5-seed 분포 상단(+1.4σ)·최고 단일런들과 동급 |
| cross-attn (H17) | 명백 열세 |

- **믹서 3점 스윕 완성: mean ≈ gated-MLP > xattn.** 우리 학습형 트렁크가 무파라미터 평균 대비 **우위 없음**(n=1이라 "평균이 낫다"까지는 단정 보류, "우리가 낫다"는 기각).
- 🔴 **철회**: "우리 트렁크가 MLE-SAM 평균을 동일조건에서 +X 상회" 기여 주장(어제 계획) — **사용 불가**. 컴포넌트 기여표에서 트렁크-타입 행 삭제.
- 재귀속: P39.1의 MUSES +0.76은 트렁크+VICReg 묶음이었음 — 이번 결과로 **기여 후보는 VICReg(rank 복원)** 쪽으로 좁혀짐(N2도 VICReg 유지라 트렁크-타입만 격리된 실험). VICReg 단독 격리는 미측정(원하면 토글 1런).
- **논문 서사는 오히려 강화**: "connector는 negligible"(MM1)이 자기 컴포넌트에도 성립함을 정직하게 보고 — 소거 논지("믹서 무관, 학습신호·초기화가 전부")의 완결. 단 P52 트렁크는 gated-MLP 유지(현직·MUSES 검증·n=1 근거로 교체 안 함, 논문에 등가성 명기).

## 3. P52 구성 확정 (판정 전부 도착)

**P52 = frozen DINOv3-L + 모달별 LoRA + gated-MLP trunk(+VICReg) + FPN/M2F-lite** (추론 그래프, 3벤치 동일)
- **+ P50 정렬-사전학습 init** (H22 ✓, +0.74)
- **+ 진단-구동 학습손실**: C3를 붕괴 강도에 따라 (DELIVER on / MUSES off / MCubeS 중간) — H20 ✓
- **+ (대기) UniBal** — MUSES 4모달, 학습 중(~2.9일)
- **+ 정본 선택 프로토콜**: legal-val 재선택(N6) + 하네스 가드 + BS1 + mean±std 보고

정량 설계-기여표(논문 ablation 골격): C3 (+1.4 DELIVER / rubber +9.76 MCubeS) · P50 init (+0.74~+1.38) · UniBal (TBD) · [VICReg — 격리 미측정 명기 or 토글 1런]. 트렁크-타입 행은 등가 보고.

## 4. 원장

- **H21 신설 ✗**: "학습형 gated-MLP 트렁크가 산술평균 융합 대비 우위" — 반증(동일조건 55.45 vs 54.2~55.4).
- **H22 신설 ✓**: "frozen 백본 위 어댑터 정렬 사전학습이 타깃 이득" — 확인(+0.74, 매칭 통제, 이득은 조기 수렴 국면 집중).

관련: [decisions/2026-08-17-p50-map-modal-alignment-pretraining-proposal.md](../../decisions/2026-08-17-p50-map-modal-alignment-pretraining-proposal.md) · [2026-08-20-fusion-mechanism-double-negative.md](2026-08-20-fusion-mechanism-double-negative.md)(믹서 스윕 1·2점) · [research/hypothesis-ledger.md](../../research/hypothesis-ledger.md)
