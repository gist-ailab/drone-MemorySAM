# P29-Det vs P30-Det 비교 (v2 test, hinton eval)

- **평가**: hinton GPU0/1, `objdet/tools/diag_det.py` (동일 파이프라인, letterbox, ROOT=/ailab_mat2), test = poongsan_v2 `det_test_v2.json` (캡처 holdout 115206+114808, kept=1772장/모달 3개 필수).
- **P29-Det** = `det_P29_v2_bundle` best **epoch 9** (AP 0.269, 학습 종료·최종). mean-fusion + FCOS.
- **P30-Det** = `det_P30_v2` best **epoch 24** (AP 0.108, **학습 진행중 스냅샷**). reliability-router 융합 + object-query decoder + FCOS aux.
- 날짜: 2026-07-02.

## 전체 지표
| metric | P29-Det (ep9) | P30-Det (ep24) | Δ(P30−P29) |
|---|---|---|---|
| AP@[.50:.95] | **0.269** | 0.108 | −0.161 |
| **AP50** | **0.4455** | 0.2285 | −0.217 |
| AP75 | **0.283** | 0.088 | −0.195 |
| AP small | **0.120** | 0.006 | −0.114 |
| AP medium | **0.168** | 0.026 | −0.142 |
| AP large | **0.348** | 0.174 | −0.174 |

## 클래스별 AP50 (크기순, 대→소)
| 클래스 | 크기 | P29 AP50 | P30 AP50 | Δ | #GT |
|---|---|---|---|---|---|
| Obstacles | 대 | 0.397 | **0.401** | +0.004 | 722 |
| Allies | 대 | 0.498 | 0.476 | −0.022 | 928 |
| Enemies | 대 | 0.416 | 0.332 | −0.084 | 872 |
| Casualties | 대 | 0.560 | 0.472 | −0.088 | 1194 |
| Doors | 중 | 0.345 | 0.163 | −0.182 | 459 |
| Windows | 중/소 | 0.414 | 0.216 | −0.198 | 1684 |
| Fire Extinguishers | 소 | 0.394 | 0.025 | −0.369 | 1056 |
| Landing Markers | 소 | 0.526 | 0.154 | −0.372 | 933 |
| Emergency Exits | 소 | 0.452 | 0.028 | −0.424 | 1100 |
| Lighting | 소 | 0.455 | 0.019 | −0.436 | 341 |
| **OVERALL** | | **0.4455** | **0.2285** | **−0.217** | |

## 클래스별 AP@[.50:.95] (참고)
| 클래스 | P29 AP | P30 AP |
|---|---|---|
| Casualties | 0.375 | 0.221 |
| Allies | 0.371 | 0.260 |
| Landing Markers | 0.352 | 0.051 |
| Enemies | 0.272 | 0.157 |
| Emergency Exits | 0.261 | 0.008 |
| Windows | 0.234 | 0.095 |
| Obstacles | 0.228 | 0.197 |
| Fire Extinguishers | 0.209 | 0.007 |
| Lighting | 0.202 | 0.004 |
| Doors | 0.187 | 0.080 |

## 핵심 결론
1. **P30는 균일하게 열세가 아님 — 대형 객체는 P29와 대등, 소형에서 붕괴.**
   - 대형(사람·장애물): Obstacles +0.004, Allies −0.02, Enemies/Casualties −0.08~0.09 → 근접. P30의 reliability-router 융합은 대형에서 정상 작동.
   - 소형(소화기·마커·비상구·조명): AP50 0.02~0.15로 P29 대비 3~24배 붕괴. area별 P30 AP_small 0.006 vs P29 0.120.
2. **P30 전체 열세(AP50 −0.217)의 대부분이 소형 객체에서 발생.** 원인: object-query decoder의 제한된 query 수·저해상도 샘플링(DETR류 소형객체 약점) + ep24 미수렴.
3. **현 최선 = P29-Det ep9 (AP50 0.446).** P30는 아직 학습 중이므로 수렴 후 재비교 필요.
4. **P30 개선 방향**: query 수↑ / 고해상도(deformable) feature 샘플링 / FCOS-aux가 소형 담당. 대형은 이미 경쟁력 있음.

## 원본 산출물
- hinton: `~/src/dm_eval/out_p29/diag_summary.json`, `out_p30/diag_summary.json`, `diag_p29.log`, `diag_p30.log`
