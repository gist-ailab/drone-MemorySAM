---
created: 2026-07-22
scope: P41 — fusion-단계 스펙트럼 붕괴 "판별 우선 + 조건부 개입". P38-MUSES 피쳐 특성화(2026-07-22)·실패-키 키3(2026-07-20) 기반, fusion-rank 딥리서치(기제 12·rank↔성능 양방향·노벨티) 교차.
gates: Phase0(학습0) 판별 게이트 → Phase1(조건부) MUSES val ≥82.22(P38) & LDA-rank∧mIoU 동시상승
supersedes_direction: P39.1 per-modal rank 작업(무이득) — 이 제안은 **fusion locus**로 이동
---

# P41 — Fusion Spectral Collapse: 판별 우선, 조건부 개입

## 0. 진단 ↔ 문헌 (딥리서치 교차)

| 우리 실측 (P38-MUSES 피쳐 특성화) | 문헌 근거 | 함의 |
|---|---|---|
| per-modal 건강(eff_rank 25, idim90 ~220) → **FUSED 붕괴**(eff_rank ~9, idim90 **~21**, kurt 10~12), 전 조건 보편 | attention rank collapse(2103.03404, 깊은 attention은 rank-1 수렴) · EBR(2505.22483, fusion rank bottleneck→모달 억압) | fusion operator가 압축 주체 (키3 재확인) |
| **idim90 21 ≈ 19클래스 최소분리차원(~18)+2** · 야간 rank 최저(7.45)인데 mIoU 정상(77.6) | neural collapse = 양성 저rank(2402.03991) · low-rank simplicity bias 잘 일반화(2103.10427) | **저rank가 양성 압축일 수 있음** — rank↑=성능↑ 반례 실재 |
| P39.1 per-modal VICReg **무이득** | 개입 locus가 per-modal(이미 건강) — 2505.22483/DAGR도 per-modal locus | **fusion-level 개입은 미시도** (유효 신규 방향) |
| **fused rank 인과조작→dense seg 성능** 실증 | **문헌 전무** (근접: DBP 2510.14657 사전학습 decorrelation→ADE20K +6.1%; RTF 2511.06450 비디오 행동분할 +3.74%) | "rank 개입이 이긴다"는 **미확립 가설** → 판별 필수 |

## 1. 🔴 Phase 0 — 판별 우선 (학습 0, 기존 P38 ckpt)

**전제 거부**: "fusion rank 올리면 P38 넘는다"를 가정하지 않는다. 먼저 **붕괴가 양성(neural-collapse)인가 유해(EBR 모달 억압)인가**를 학습 없이 판별한다.

| # | 분석 | 판별 논리 | 도구 |
|---|---|---|---|
| **P0-A** modality-ablated fused 스펙트럼 | 각 모달을 fusion 입력에서 제거 → fused eff_rank/idim 변화. **거의 불변**이면 그 모달 정보가 이미 fused에 안 실림 = **EBR 억압(유해)**. 모달마다 rank가 유의하게 바뀌면 = fusion이 다 쓰는 중(양성) | `feature_stats` + modality-drop 훅(기존 `_maybe_drop`/입력 zero 재사용) |
| **P0-B** task-aware rank | fused feature의 **LDA-rank**(GT 클래스 판별에 실제 쓰이는 차원, MUSES val GT down-sample) vs eff_rank. LDA-rank << idim이면 판별정보가 더 좁음(개선 여지) / LDA-rank ≈ C−1이면 이미 클래스구조 정렬(양성) | `feature_stats` 확장(LDA-rank 함수 추가 — 새 도구 금지, §7) |
| **P0-C** 조건 대비 | 양성압축이면 조건 무관 rank가 task-set. fog(최약 62.7)에서 fused LDA-rank가 clear보다 **낮으면** fog 병목이 fusion-판별력 부족일 가능성(키4 미설명 fog와 연결 시도) | 위 도구 per-condition |

**Phase 0 게이트(사전등록)**:
- **유해 판정**(→ Phase 1 진행): P0-A에서 modality-drop 시 fused eff_rank Δ 작음(특정 모달이 이미 억압) **또는** P0-B에서 LDA-rank << idim90(판별정보가 스펙트럼보다 좁음).
- **양성 판정**(→ **STOP**, fusion-rank 폐기·피벗): 모달별 rank가 고루 기여 + LDA-rank ≈ idim ≈ C−1. 이 경우 rank 개입은 무이득(야간 반례·2402.03991 정합) → **fog 원인규명 등 다른 병목으로 전환**.

> Phase 0가 이 제안의 핵심 기여다. "rank↑=성능↑"를 검증 없이 태우면 P39.1 재판(rank 건드렸으나 무이득). 학습 0으로 방향을 먼저 falsify한다.

## 2. Phase 1 — fusion-단계 개입 (Phase 0 유해 판정 시만)

전 항목 토글 가능. 키1 준수(주 손실 직접 수신 또는 in-path 경쟁 — **zero-init 잔차 금지**).

| # | 변경 | 근거(arXiv) | 형태 |
|---|---|---|---|
| **C-1 (주 변수)** | fused 출력에 **in-path decorrelation** 층(Shuffled-DBN 2105.00470 / ContraNorm 2303.06562) | attention rank collapse를 in-path norm이 저지(2103.03404·2303.06562); DBP decorrelation→seg +6.1%(2510.14657) | 주 손실 관통, 잔차 아님 |
| **C-2 (병행)** | fused에 **supervised var-cov aux**(VCReg 2306.13292) 또는 log-det coding-rate(MCR² 2006.08558) | 주손실 병행 규제 실증(2306.13292); MUSES seg aux 규제 대폭 이득 선례(함수엔트로피 2505.06635 +13.94) | aux 손실(주손실과 합산) |
| **C-3 (baseline 비교)** | RTF 채널 블렌딩(2511.06450)을 fused에 이식 | 최근접 선행 — 리뷰어 비교 요구 대비 | 아키텍처 모듈 |

⚠️ **eff_rank 단독 KPI 금지**(LiDAR 2312.04000: 무정보 차원 오염) → **LDA-rank 병용**. 개입이 eff_rank만 올리고 LDA-rank·mIoU 불변이면 무효.

## 3. 게이트 사전등록 (falsifiable)

| 단계 | 게이트 | falsify 예측 |
|---|---|---|
| Phase 0 | 위 §1 유해/양성 이분 | 양성이면 제안 자체 폐기(정직) |
| Phase 1 ep30 조기 | MUSES val 궤적 ≥ P38 동에폭 & **module_ablation로 C-1 off-Δ>0** | no-op이면 조기 kill |
| Phase 1 완주 | MUSES val ≥ **82.22**(P38) & 개입이 **LDA-rank ∧ mIoU 동시 상승** | rank↑인데 mIoU 불변 = 양성압축 확정·가설기각 |
| 공정성 | physaug 정합(ISSUE-026 픽스본)·val-best ckpt·radar 미포함·TTA 금지 | |

## 4. 노벨티 포지셔닝 (정직)

- **미점유 조합**(검색 범위 내): {post-fusion 텐서} × {스펙트럼/rank 개입} × {dense semantic seg} × {frozen VFM per-modal 인코더}. + **fused 붕괴의 조건별(주/야) 양성-vs-유해 판별 프로토콜** 자체가 선행 부재.
- **주장 가능**: "first to characterize and intervene on fusion-stage spectral collapse in frozen-VFM multimodal semantic segmentation" (스코프 한정).
- **주장 금지**: rank-targeted fusion(RTF 2511.06450 선점), rank↔modality-collapse 연결(EBR 2505.22483), regularized multimodal seg(2505.06635).
- 🔴 **야간 반례를 기여로**: "언제 fusion 저rank가 양성인가"(NC vs EBR 판별)를 분석 기여로 전환 — 숨기면 neural-collapse 문헌(2402.03991·2103.10427) 아는 리뷰어에 역공.

## 5. 실행 순서

1. **Phase 0 (지금·학습 0)**: `feature_stats` 확장(LDA-rank + modality-ablation 스펙트럼) → hpca100 유휴 A100에서 P38 ckpt로 판별. 코드검수 파이프라인(§conventions). **← 유휴 A100 2장의 즉시 용처**.
2. Phase 0 **유해** → Phase 1 C-1 구현·투입(ep30 토글 즉검). **양성** → 제안 폐기·fog 병목 피벗(별도).
3. ablation: C-1/C-2/C-3 개별 토글 + P38 baseline + (있으면)per-modal VICReg 대조(P39.1 재사용).

**미결/선행**: LDA-rank 도구 확장(§7 확장), P0-A modality-drop 훅 재사용 확인. Phase 0는 학습 0이라 리스크 최소 — 먼저 돌려 방향부터 확정.

**근거 arXiv**: 2511.06450 · 2505.22483 · 2505.06635 · 2105.00470 · 2303.06562 · 2510.14657 · 2306.13292 · 2006.08558 · 2103.03404 · 2402.03991 · 2103.10427 · 2312.04000 · 2210.02885
