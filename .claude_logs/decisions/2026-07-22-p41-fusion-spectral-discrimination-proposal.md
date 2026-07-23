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

---

## Phase 0 결과 + 판정 (2026-07-22, hpca100 A100, P38 ckpt, MUSES val clear/fog/night)

원시: NAS `analysis_logs/p41_phase0_20260722/` (full + drop_img/lidar/event).

**P0-B (η² = tr(Sb)/tr(St))**: FUSED_pf η² = **0.32~0.35**(전 조건), FUSED(decode) η² = **0.63**.
→ 저rank(9)인데 η²가 →1이 **아님** ⇒ **neural-collapse 양성 압축 가설 반증**("rank-9가 task-최적이라 개입 무의미"는 틀림). 단 η²가 조건 불변(night≈clear)인데 성능은 변동 ⇒ **η²도 성능 직접예측 못 함**(rank와 동일 한계) — "fusion 고치면 성능↑"는 여전히 미증명. decode가 η²를 0.35→0.63으로 회복(head가 fusion 미달분 보정).

**P0-A (모달 드롭 시 FUSED_pf eff_rank Δ)**:
| | full | drop img | drop lidar | drop event |
|---|---|---|---|---|
| clear | 9.21 | **14.53** | 10.38 | 7.95 |
| night | 7.02 | **9.48** | 9.59 | 6.38 |
| fog | 8.75 | **6.91** | 10.07 | 7.85 |
→ **🔴 img가 fusion을 과지배·압축**: clear/night에서 img 제거 시 fused rank **상승**(+5.3/+2.5) — 가장 정보 풍부한 모달(rank 29~35)이 joint 표현을 짓누름 = **dominant-modality collapse**(고전 EBR 억압 아님). **단 fog는 반대**(img 제거 시 rank↓) — fog는 lidar가 죽어 img가 진짜 캐리어(키4 fog 최약과 연결 실마리). event 드롭 Δ≈−1(약기여).

**게이트 판정**: 깨끗한 양성(STOP)도 유해(확신 진행)도 아님. **정제된 결론**: 붕괴는 양성 아님(η² 0.35) + **img 과지배**라는 개입점 확보. 성능 링크는 미증명 → **ep30 게이트로 결판**.

**🟢 결정 (2026-07-22, opus): Phase 1 진행(A안).** 근거: (1) "무의미(양성)" 반증됨, (2) img-지배라는 구체 타깃, (3) ep30 falsify가 저비용.

## Phase 1 (확정 설계) — FCR: Fused Class-alignment Regularizer

img-지배로 저-task-정렬된 fused를 **주 손실 레벨**에서 교정(frozen 백본이라 loss-lever만 유효 — 딥리서치 R1). **키1 준수**(aux 손실 = 주손실 경로, zero-init 잔차 아님).

| # | 변경 | 근거 | 형태 |
|---|---|---|---|
| **F-1 (주 변수)** | fused(T3)에 **supervised between-class 분산 규제**: `L_fcr = −λ·η²(fused, gt_mask)` 또는 등가 class-center 분리 손실. 측정된 deficit(η² 0.35)를 직접 최적화 ⇒ img-지배 완화·클래스 정렬 | VCReg supervised var-cov(2306.13292), MUSES seg aux 규제 대폭이득 선례(2505.06635) | aux 손실(warmup, λ 소), 토글 `P41.FCR` |
| F-2 (대안) | in-path decorrelation(ContraNorm 2303.06562)을 fused 뒤 — F-1 무효 시 | attention rank collapse 저지(2103.03404) | in-path 층 |

**구현**: model.forward의 fused(L536~) + gt_mask로 `aux['fcr']` 계산(vicreg/cefr_reg 패턴), trainer 합산. eval 시 Phase-0 도구로 η² 재측정.

**ep30 게이트(사전등록, falsifiable)**: ① **fused η² 상승** — Phase-0 도구(`feature_stats --lda-rank`)로 P41 ep30 ckpt vs P38(η²=0.35) 재측정 (FCR은 **학습-전용 aux**라 eval-time `module_ablation` 토글은 무의미 — 대조는 P41-FCR vs 기존 P38 두 학습) ② val mIoU ≥ P38 동에폭. **판정**: η²↑ ∧ mIoU↑ ⇒ fusion이 레버 확정 / **η²↑인데 mIoU 불변 ⇒ fusion은 병목 아님 확정 → fog 피벗**(정직한 종결). 공정성: physaug 정합·val-best·radar 미포함.

**실행**: hpca100 A100(Phase-0 종료로 유휴) 첫 슬롯. 코드검수 파이프라인 + ep30 토글 즉검.

---

## 🔴 Phase 1 결과 = 게이트 부정 (2026-07-23, airtight falsification)

FCR 학습(hpca100 A100, ep90+까지) 후 게이트 두 축 실측:

**① mIoU (vs P38 同에폭)**: P41 ≈ P38, 순평균 근소 열세. 초기(ep8-12) FCR +1.2~+1.5 앞섰으나 중후반 P38이 따라잡음. ep42=79.83 vs P38 79.94, ep50대 80.4~81.2 = P38 수준. **개선 없음.**

**② fused η² 재측정** (P41 ep86, feature_stats --lda-rank, MUSES val): clear **0.9482** / fog **0.9339** / night **0.9381** vs **P38 0.35** → **2.7배 상승**.

**판정**: **η²↑(2.7×) AND mIoU 불변** = 사전등록 falsification 케이스 정확히 실현.
→ **fusion rank/η²는 MUSES 성능 레버가 아니다 (definitively 기각).** fused를 거의 완전 클래스-정렬(η² 0.94)시켜도 seg 성능 무이득. 기제: decode가 이미 클래스정보를 추출(P38 decode η² 0.63)하므로 fusion 사전정렬은 head와 중복. Phase-0 hedge(야간 rank최저·성능정상 / η²조건불변·성능변동)가 옳았음. → **P41 중단, A100 회수.**

**의의**: framework의 첫 완결 가설검증 — 추측으로 완주 후 실패발견(P39.1/P40)이 아니라 **학습0 Phase-0 판별 → 사전등록 게이트 → 조기 확정**. negative지만 방법론 성공.

## 다음 = fog 병목 (방향 전환 확정)

fog 분석(P38, 2026-07-23) D1 per-domain: **clear 75.85 · fog 62.67 · night 78.05** → **fog −13pt = MUSES 진짜 병목**(키4 재확증, night은 최약 아님). fusion-rank 종결, **fog 원인규명(F-A~F-D) → fog 타깃 제안**으로 이동. 상세 분석 문서 = `experiments/analysis/2026-07-23-fog-*`(진행 중).
