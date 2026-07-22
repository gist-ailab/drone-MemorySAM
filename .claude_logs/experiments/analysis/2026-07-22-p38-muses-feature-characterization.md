# P38-m2f MUSES 피쳐 특성화 (§0.5 tap×method, 2026-07-22)

> **대상**: P38-m2f MUSES val-best `epoch156_82.22_top1_checkpoint.pth` (내부최고, 3모달 img/lidar/event)
> **도구**: `tools/feature_stats.py` (§0.5 확장본, develop 23f0f51) — tap T0(encoder)/T3(FUSED_pf)/T5(PREHEAD)/decode(FUSED) × activation·PCA·eff_rank·CKA·stage-CKA
> **원시 산출물**: NAS `analysis_logs/p38_featchar_20260722/` (json/md/pca.png + config 사본). MUSES val 6조건 각 n≈58~60.
> **관련**: [실패-키 2026-07-20](2026-07-20-failure-keys-p38-deliver-p37a-muses.md) 키3/4/5 · [P39.1/P40 제안](../../decisions/2026-07-21-p39_1-p40-rank-rca-proposal.md)

## 수치 요지 (6조건 범위)

| tap | eff_rank | idim90 | kurtosis | 해석 |
|---|---|---|---|---|
| **img** (T0) | 28~36 | 190~287 | 1.3~1.5 | 인코더 피쳐 **건강**(고rank·고차원·dense, dead 1/1024) |
| **lidar** (T0) | 24~27 | 189~230 | 0.7~0.9 | 〃 |
| **event** (T0) | 23~24 | 182~240 | 0.3 | 〃 (셋 중 최저이나 여전히 건강) |
| **FUSED_pf** (T3) | **7.5~9.5** | **21~24** | **10~12** | 🔴 **fusion 직후 급붕괴**(rank 3×·idim 10× 압축, 스파이크) |
| **PREHEAD** (T5) | 7.5~9.5 | 21~24 | 10~12 | = FUSED_pf와 **완전 동일**(stage_cka=1.0) |
| **FUSED** (decode) | 8.7~11.8 | 21~26 | 23~31 | 256ch decode 표현 |

**cross-modal CKA**: img~lidar 0.71~0.78 · img~event 0.69~0.73 · **lidar~event 0.89~0.92**(거의 중복)
**stage CKA(→PREHEAD)**: img 0.29~0.41 · lidar 0.18~0.22 · **event 0.15~0.19**(fused 기여 최저) · **FUSED_pf~PREHEAD = 1.0**

## 판정

1. **🔴 fusion 정보 병목 재확인(키3 정밀화)**: per-modal(rank ~25, idim ~220)이 fusion 직후 rank ~9, idim ~21로 붕괴. 키3(2026-07-20, P38-DELIVER·P37a-MUSES에서 rank 6.8~8.0)이 **P38-MUSES에서도 재현** + 신규 지표(idim90 21, kurt 10~12). 개입 지점 = **융합 단계 용량/구조**(로짓 근처 아님).
2. **`FUSED_pf~PREHEAD = 1.0`**: P38엔 fused-level 모듈 없음(CEFR/trunk_exp 부재, m2f는 logit-level). 즉 이 rank 붕괴는 **memory-attention 융합 자체**의 산물. (도구 정합성 검증도 겸함 — 모듈 없으면 1.0.)
3. **event = 기하적 잉여이나 과제기여는 유효(키5 준수)**: lidar~event CKA 0.90 + event stage_cka 0.15(최저)로 **기하적으론 lidar와 중복·최소 기여**. 단 키5는 MUSES event Δacc **+0.24~0.29(강력)**로 실증 → **MUSES에서 event 제거 금지**(잉여는 기하일 뿐 과제엔 기여). event 제거는 DELIVER 한정.
4. **⚠️ 야간 붕괴 ≠ 야간 문제(과잉해석 차단)**: night에서 FUSED_pf rank 7.45(전 조건 최저)·img 기여 0.29(최저)로 **피쳐는 가장 붕괴**. 그러나 키4는 night mIoU 77.6(정상), 최약은 **fog 62.7**. 즉 **저rank가 성능저하와 동행하지 않음**(night이 반례) → "fusion rank 올리면 성능 오른다"는 **미증명**. fog 최약은 이 피쳐 데이터로 설명 안 됨(fog rank 8.75=중간).

## 제안으로의 함의

- **유효 신규 방향** = **fusion-단계 rank/용량 개입**(키3). P39.1의 rank 작업(VICReg on **per-modal**)은 P38의 건강한 per-modal이 아니라 **fusion 붕괴를 안 건드림** → fusion-level은 미시도.
- **단 가설로 세울 것**: night 반례 때문에 "rank↑→성능↑" 주장 금지. 제안은 **fusion 용량이 병목인지 falsifiable하게 검증**하는 형태 + 주 손실 직접 수신(키1, zero-init 잔차 금지).
- 차기 제안 = `decisions/2026-07-22-*` (딥리서치 진행 중: fusion-rank 기제·rank↔성능 인과·노벨티).
