# P46 모듈 기여도 A/B — 쿼리 경로 순기여 0 (dense 복제) (2026-08-05)

## 대상
- 체크포인트: `epoch92_67.02_top1_checkpoint`(yeon λ0.15 중단 런)
- 데이터: DELIVER **test** split, 조건당 60장
- 도구: `tools/module_ablation.py`

## 부호 규약
`miou_delta_when_off = base − off`(`module_ablation.py:280`) → **양수 = 해당 모듈이 기여**(끄면 성능이 떨어짐), 음수 = 해당 모듈이 손해(끄면 오히려 오른다).

## 결과 표 (조건 × 토글, mIoU)

| 조건 | base | p39_query_off | p36_router_off | p39_trunkexp_off |
|---|---|---|---|---|
| night | 42.16 | −0.06 | +0.78 | +1.11 |
| fog | 39.61 | +0.16 | +0.20 | +0.91 |
| cloud | 44.95 | −0.34 | +0.48 | +1.83 |
| rain | 44.50 | −0.19 | +0.88 | +1.68 |
| sun | 43.41 | −0.02 | +0.28 | +1.79 |
| **평균** | | **−0.09** | **+0.52** | **+1.46** |

## 보조 지표
- query·router 토글: `feat_cos=1.0, feat_shift=0.0` — logits 단계 개입이라 정상(융합 피쳐 자체는 불변).
- trunkexp 토글만 `feat_cos≈0.91, feat_shift≈0.44` — **융합 피쳐를 실제로 바꾸는 유일 모듈**.
- `pred_agreement`(base vs off, 픽셀 단위): query 0.9883~0.9903 / router 0.9895~0.9915 / trunkexp 0.9629~0.9727.

## 판정

1. **쿼리 경로 = 순기여 0, 그러나 죽은 모듈이 아니라 dense의 복제.** 꺼도 픽셀의 98.9%가 base와 동일 예측 — 모듈이 뭔가 "하고는" 있지만 dense head와 거의 같은 해에 수렴해 있다. P39-V5의 path dropout이 강제한 과제가 semantic이라, 쿼리 경로의 최적해가 "dense를 베끼는 것"이 되어버렸다는 뜻. → P48 인스턴스 감독의 근거를 **"쿼리를 살리자"**가 아니라 **"dense가 원리적으로 풀 수 없는 과제(인스턴스 분리)를 줘서 두 경로의 중복을 깨자"**로 정밀화해야 한다. 기존 PQ 분해(SQ 79.51 살아있음 / RQ 44.51 붕괴)와도 정합적 — semantic recognition quality는 문제없고 recognition-at-instance-level만 죽어 있다.

2. **P36 router 기여(+0.52 평균)의 대부분이 RailTrack 한 클래스로 설명된다.** router_off top-class Δ: RailTrack +16.3(night) / +11.2(cloud) / +14.8(rain). night의 전체 기여 +0.78 중 약 +0.65가 RailTrack 하나에서 나온다. ⚠️ 단 fog +2.7 / sun +1.2로 조건별 크기가 비일관 — **방향(router가 RailTrack에 특화 기여)은 신뢰하되 크기는 신뢰하지 말 것** (RailTrack 자체가 희소 클래스이고 조건당 n=60이라 노이즈에 취약).

3. **trunk_exp가 유일한 실질 기여자(평균 +1.46)이나, Wall 클래스를 일관되게 해친다.** trunkexp_off 시 Wall Δ: night −9.1, fog −13.4 (즉 trunk_exp를 켠 상태에서 Wall이 그만큼 나쁘다). Wall은 thin-class 게이트(threshold Wall≥13) 대상 클래스 — 전반적 이득(+1.46 평균) 뒤에 국소적이지만 뚜렷한 손해가 가려져 있다.

## 유보 사항
- 조건당 60장은 표본이 작아 per-class(특히 희소 클래스: RailTrack, Wall 등)는 노이즈가 크다.
- 이 체크포인트는 **중단된 λ0.15 런**이며, DELIVER 최고 구성인 λ0.05 런으로는 아직 재확인하지 않았다.
- 이번 측정은 **semantic mIoU 기여**를 보는 것이지 마스크 품질(경계·인스턴스 분리) 자체를 재는 것은 아니다.

## 원시 산출물
`yeon:/SSDb/jemo_maeng/analysis/P46_lam015_deliver_analysis_20260805/ablation/`(`P46_lam015.json`, `P46_lam015.md`)

관련: decisions/2026-08-03-p47-mub-muses-proposal.md · tools/README_seg_analysis.md · [[seg-analysis-pipeline]]
