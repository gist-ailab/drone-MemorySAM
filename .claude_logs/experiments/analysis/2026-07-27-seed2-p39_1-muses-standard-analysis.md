# P39.1-seed2 MUSES 표준분석 (2026-07-27, val 3모달) — 우리 최고 모델
**대상**: P39.1-rank seed2 ep208, **val 82.62 / test 79.788**(프로젝트 최고, SOTA GtA 82.39 −2.60). ReliaDINO ViT-L + per-modal LoRA + P39.1(gated_mlp trunk R-1 + VICReg R-2) + router + M2F. 3모달(img/lidar/event). 원시=NAS `analysis_logs/seed2_P39_1_eval_20260727/`.
⚠️ 수치는 파이프라인 자체 val-subset eval(공식 82.62와 직접비교 금지, 상대·구조용).

## ① adapter 적응도 — 강함(특히 lidar)
dW mean 14.19, per-modal img 13.97/**lidar 15.03**/event 13.59(lidar 최대), dead 0/48. D3B lidar Δacc **+0.39~+0.55**(P43 +0.33~0.41보다 큼).

## ② 모달별 피쳐 — VICReg가 lidar rank 대폭 확장
🔴 **lidar eff-rank 78.5~100.3**(VICReg on) = P43(VICReg off) 23.5~28.0의 **3~4×**. → R-2(VICReg)가 lidar 표현 rank를 실제로 크게 확장(기제 실증). dead 1/1024. 융합 stage CKA FUSED_pf~PREHEAD 0.636~0.710(P43는 1.0) → **trunk가 pre-head 피쳐를 실제로 바꿈**(trunk 작동 증거).

## ③ 모듈 전후 (부호 = base−toggled, + = 기여; 소스 module_ablation.py:280 확인)
- **p39_trunkexp_off +2.05~+6.78**(clear/day: base 75.63 > toggled 69.11) = **R-1 trunk 강한 순기여** ✅
- p36_router_off +0.5~+4.5 = router 순기여
- **p39_query_off 일부 야간 음수**(rain −0.26/night −0.37/clear_night −0.29) = arbiter query가 야간 조건서 미세 유해(소규모, 향후 조건부 off 검토)
- p34 gate/veto/calib +0.00 = off(미사용). p43_* skip(P43 모듈 부재).

## ④ 클래스×도메인 — drop-modality dMIoU
drop-lidar: day 4.24 → **fog_night 7.39 / snow_night 7.6 / rain_night 7.57**(야간·adverse 인과 기여). drop-event ≈0(잉여, event~lidar 중복). drop-img 지배.

## 종합
1. **우리 최고 모델(82.62/79.788)의 기제 전부 검증**: VICReg(lidar rank 78~100)·trunk(+2~7)·router(+0.5~4.5) 전부 순기여, drop-lidar 야간 인과. P39.1의 R-1·R-2 둘 다 유효 실증.
2. 흠: arbiter query 야간 미세 유해(향후 조건부 off).
3. **4모달 근거**: drop-lidar 야간 기여 → radar 추가로 fog(lidar 산란) 보완 = 4-modal 실험(yeon 0,1,5) 착수 근거.
4. vs P43: seed2(VICReg on) lidar rank 78~100 ≫ P43(off) 23~28; drop-lidar 야간 기여는 유사; 순기여 모듈은 seed2=trunk(+2~7)·P43=LATERAL(+1.86)로 상이.
