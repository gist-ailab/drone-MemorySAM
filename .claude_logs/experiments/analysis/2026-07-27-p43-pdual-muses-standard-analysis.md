# P43-PanopticDual MUSES 표준분석 (2026-07-27, val 3모달)

**대상**: P43-pdual ep156 (val 82.51 / test 79.788 제출은 미정). ReliaDINO ViT-L + per-modal LoRA + P36-router + P43(M2F Head B co-train + LATERAL). SEM_SOURCE=pixel. **P39 trunk_exp/VICReg OFF.**
**원시 산출**: NAS `analysis_logs/P43_pdual_eval_20260727/` (699파일). ⚠️ 수치는 분석 파이프라인 자체 val-subset eval — 공식 val 82.51과 직접 비교 금지, 상대·구조용.

## ① adapter 적응도 — lidar adapter 강하게 작동
- 정적 dW: mean 10.74, dead 0/48, per-modal img 11.30/lidar 11.40/event 9.53.
- 동적(D3B on/off): lidar feat_cos~0.40, **Δacc +0.33~+0.45**(전 조건). adapter 살아있음.

## ② 모달별 피쳐 — lidar eff-rank 회복 + 융합 병목 잔존
- **lidar eff-rank 23.5~28.0**(P39-DPC 4.7 붕괴 대비 완전 회복, 게이트≥15 상회). img 18.5~31.6/event 29.4~36.3, dead 1/1024.
- 🔴 **P43는 VICReg OFF인데도 rank 건강** → rank 붕괴는 P39-DPC V1 선형 트렁크 자초, P43는 그 트렁크 미사용으로 회피. VICReg 순효과는 seed2 분석(별건)이 답함.
- 🔴 **융합 병목**: per-modal ~25~35 → FUSED_pf 5.5~11.3 급압축(fog_night 5.54 최저). P38 정보병목과 동일.

## ③ 모듈 전후 — LATERAL 유효, router 지배
- p43_lateral_off Δ +0.3~+1.9(clear/day +1.86, snow_night +1.28, fog_night +0.34), feat_cos~0.75 → no-op 아님, 실질 기여.
- p36_router_off Δ +4.7~+11.3(최대 기여).
- p34 gate/veto/calib Δ=0(off, 미사용). p43_m2f_off/p39_* = skip(sem_source=pixel·P39 off, 미결선).

## ④ 클래스×도메인 — drop-modality dMIoU(4모달 근거)
- **drop-lidar**: day 0.64/clear 0.64 → night 2.26/snow_night 2.73/rain_night 4.99/**fog_night 7.19**. lidar는 야간·adverse에서 인과 기여(P39.1-DELIVER −0.78과 정반대).
- drop-event ≈0~음수 + CKA(event~lidar)0.79~0.85 → event 잉여/사망.
- drop-img 14.8~25.6(지배). 최악 셀 fog_night.

## 종합
1. P43 건강(adapter·rank·LATERAL·router 정상), 융합 병목만 잔존.
2. **🎯 4모달 실증 근거**: drop-lidar fog_night +7.19·야간 +2.26 = 비RGB가 adverse-night에서 이미 인과 기여. event 잉여 → 4번째는 event 대체 아닌 **radar 추가**(fog에서 lidar 산란 보완). radar-fix 재실험 최우선.
3. rank 붕괴 = P39-DPC 트렁크 자초(P43 회피). seed2 VICReg 순효과는 별건 분석 중.
