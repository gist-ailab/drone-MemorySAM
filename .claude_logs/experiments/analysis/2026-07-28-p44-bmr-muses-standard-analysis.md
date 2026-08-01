# P44-BMR MUSES 표준분석 (2026-07-28, val 3모달)
**대상**: P44-BMR ep156 (val 80.71, hpca100 완주분). P39.1 base + BMR(B-1 MMPareto + B-2 상호증류 + B-3 국소마스킹). 3모달. 원시=NAS `analysis_logs/P44_muses_eval_20260728/`. ⚠️ 파이프라인 자체 val-subset eval.

## 핵심 판정: BMR이 비RGB 사용을 P39.1/seed2 대비 못 늘림
- **drop-lidar dMIoU**: day **−0.42** / fog_night **6.71** — **seed2(day 4.24/fog_night 7.39)보다 낮음.** lidar 사용이 **야간 편중, 주간엔 잉여~미세유해**. BMR의 목표(비RGB 사용 증대)가 val에선 달성 안 됨.
- 기제는 활성: p39_trunkexp_off +3.67~+11.16(trunk 강한 기여), lidar eff-rank 46.7~67.55(3모달 최고), adapter dW lidar 16.39(최고), D3B lidar Δacc +0.43~+0.62(최고).

## 종합
1. **BMR은 MUSES val 이득 없음** — val 80.71 < seed2 82.62(−1.91), drop-lidar도 seed2 미만. BMR 균형 기제가 이미 lidar를 쓰는 P39.1을 더 개선 못함.
2. lidar 사용의 **야간 편중**(fog_night 6.71 vs day −0.42)이 BMR의 유일한 특징적 변화 → **test(adverse-night 80%)에서 val↔test 전이가 seed2보다 나을 가능성**은 남음 → P44-MUSES test zip(staged) 제출로만 판단 가능.
3. DELIVER에서도 P44-BMR(66.31) < P39.1-rank(67.60) 정체 → BMR이 두 벤치 모두 P39.1 대비 우위 없음(현재까지).
