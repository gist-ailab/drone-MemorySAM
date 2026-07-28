# P30-Det 통계 · 저성능 분석 · 모달리티/모듈 피쳐 분석 (ep39 final)

> 이관: `.claude_logs/det_eval/p30_feature_probe/P30_STATISTICS_AND_FEATURE_ANALYSIS.md` (브랜치 `det-p29-p30-analysis`, 2026-07-28 통합). 원본 커밋은 태그 `archive/det-p29-p30-analysis` 에 보존.
>
> 원본 panel PNG 24장은 git에 포함하지 않음 — 태그 `archive/det-p29-p30-analysis` 의 `.claude_logs/det_eval/p30_feature_probe/` 에서 복원 가능.

- test = poongsan_v2 `det_test_v2.json` (캡처 holdout 115206+114808, kept 1772). hinton, [`objdet/tools/diag_det.py`](../../objdet/tools/diag_det.py)(통계) + [`tools/probe_det_features.py`](../../tools/probe_det_features.py)(피쳐). ROOT=/ailab_mat2.
- P30-Det = `det_P30_v2` **ep39 (최종 best, 학습완료)**. 비교 = P29-Det ep9 (AP50 0.446).
- 원본: `p30_feature_probe/`(probe_stats.json + panel_*.png), `P30_ep39_diag_summary.json`. 날짜 2026-07-02.

## 1. 통계 (mAP/mAP50) + P29 비교
| metric | P29-Det ep9 | **P30-Det ep39** | Δ |
|---|---|---|---|
| mAP@[.50:.95] | 0.269 | **0.129** | −0.140 |
| **mAP50** | 0.446 | **0.256** | −0.190 |
| mAP75 | 0.283 | 0.116 | −0.167 |
| AP small | 0.120 | 0.015 | −0.105 |
| AP medium | 0.168 | 0.030 | −0.138 |
| AP large | 0.348 | 0.226 | −0.122 |
| AR@100 | 0.328 | 0.208 | −0.120 |
| AR small | — | 0.041 | 소형 재현율 4% |

## 2. 클래스별 AP50 + 저성능 구간
| 클래스 | 크기 | P29 | **P30 ep39** | Δ | #GT |
|---|---|---|---|---|---|
| Casualties | 대 | 0.560 | 0.519 | −0.04 | 1194 |
| Allies | 대 | 0.498 | 0.464 | −0.03 | 928 |
| Obstacles | 대 | 0.397 | 0.375 | −0.02 | 722 |
| Enemies | 대 | 0.416 | 0.351 | −0.07 | 872 |
| Windows | 중/소 | 0.414 | 0.241 | −0.17 | 1684 |
| Doors | 중 | 0.345 | 0.219 | −0.13 | 459 |
| Landing Markers | 소 | 0.526 | 0.262 | −0.26 | 933 |
| **Emergency Exits** | 소 | 0.452 | **0.054** | **−0.40** | 1100 |
| **Fire Extinguishers** | 소 | 0.394 | **0.044** | **−0.35** | 1056 |
| **Lighting** | 소 | 0.455 | **0.033** | **−0.42** | 341 |
| OVERALL | | 0.446 | 0.256 | −0.19 | |

**저성능 구간 = 소형·얇은 객체.** 대형(사람·장애물)은 P29에 근접(−0.02~0.07)하나, 소형(조명/소화기/비상구)은 AP50 0.03~0.05로 붕괴. AR_small 0.041 → 작은 객체의 96%를 아예 놓침. 전체 −0.19 격차의 대부분이 소형에서 발생(ep34 대비 대형은 개선, 소형은 여전).

## 3. 이미지별 모달리티 encoding feature + PCA (probe, 6장, 소형객체 포함)
| 지표 | img(RGB) | lidar(depth) | thermal |
|---|---|---|---|
| fpn0 L2 norm | 1.24 | **2.92** | 1.57 |
| mem L2 norm | 0.33 | 0.63 | 0.68 |
| **mem PCA top-1 설명분산** | **0.18** (고차원·풍부) | **0.62** (저차원·퇴화) | 0.44 |
| fpn0 active_frac | 1.00 | 1.00 | 1.00 |

- **LiDAR feature는 크기(norm)는 최대인데 mem이 저차원(PCA top1=0.62)으로 퇴화** — 정보량이 1개 방향에 몰림. 입력 depth map이 희소/거의 비어있음(panel의 lidar 입력=검정)이 원인.
- **RGB mem은 고차원(top1=0.18)으로 가장 풍부.** thermal 중간.

## 4. 제안 모듈 전후 피쳐 비교 (RBMA · Memory Attention)
| 모듈 | 측정 | 결과 | 해석 |
|---|---|---|---|
| **Memory Attention** | before/after cos | **≈ 0.04** (거의 직교) | **큰 효과** — SAM2 memory-attention이 인코더 피쳐를 대폭 변환(실제 작동) |
| **RBMA (reliability bias)** | λ-on vs λ-off mem cos | **1.000** ( \|Δmem\|≈0 ) | **무효과** — λ=0.997인데도 기여 0 |

**RBMA가 무효과인 이유**: per-modal reliability(`1−H/logC`)가 포화(img/thermal≈1.0)되고 모달리티 간 zero-mean 센터링을 거치면 bias≈0 → memory-attention logit에 더해지는 값이 사실상 0. **제안한 신뢰도 bias 메커니즘이 학습 후 inert.**

## 5. Fusion (Reliability-Anchored Router) 분석
P5(coarse/mem) 레벨 융합 가중치 (6장 평균): **img 0.879 · lidar 8.7e-9(≈0) · thermal 0.121**
- **LiDAR를 완전히 0으로 드롭**, RGB 88% / thermal 12%로 고정. reliability가 포화라 anchor가 무의미 → **router는 신뢰도 기반이 아니라 학습된 고정 RGB-우위 가중치**로 수렴.
- LiDAR depth가 희소/퇴화(§3)라 드롭 자체는 타당하나, "reliability-anchored" 노벨티가 의도대로 동작하지 않음.

## 6. 종합 결론
- **P30의 노벨티 모듈이 의도만큼 기여하지 못함**: RBMA=inert, reliability-router=RGB-우위 고정융합으로 collapse, LiDAR=사실상 미사용. 실제로 일하는 건 SAM2 memory-attention(§4)뿐.
- **병목 = object-query decoder(DETR류)의 소형객체 미검출**(AR_small 0.041). 이게 P30(0.256) < P29(0.446)의 직접 원인.
- **개선 레버**: ① 소형객체 — query 수↑/고해상도(deformable) 샘플링/FCOS-aux가 소형 담당 ② RBMA — reliability 포화 해소(온도/센터링 제거) 또는 제거 ③ LiDAR — 희소 depth 전처리(densify) 또는 modality-dropout 학습.
- **현 최선 모델 = P29-Det ep9 (AP50 0.446).**

## 6b. 24장 확대 검증 + reliability 포화 (근본 원인)
위 §3~5는 **24장(spread + 소형객체) 재실행**으로 모두 재현됨(`p30_feature_probe_full/summary.png`, `probe_stats.json`, `raw/*.npz`):
- router P5: img 0.88 · **lidar ≈0** · thermal 0.12 (24장 일관). mem PCA top-1: img 0.19 · **lidar 0.57(퇴화, 광범위)** · thermal 0.42. memattn cos ≈0.03 / **RBMA cos =1.00**.
- **핵심: per-modal reliability(`1−H/logC`)가 img/lidar/thermal 모두 ≈0.99999로 완전 포화** (summary 우하단, 스케일 `1e-5+9.999e-1`). per-modal seg decoder가 어디서나 과확신(저엔트로피) → **신뢰도가 모달리티를 구분하지 못함.**
- ⇒ **RBMA(신뢰도 zero-mean 센터링 → bias≈0)와 reliability-anchored router(anchor 균일 → 무의미)가 동시에 inert.** 융합은 결국 router의 학습된 conv head가 만든 **고정 RGB-우위 가중치**. 제안 노벨티의 "reliability" 축이 작동하지 않는 정량적 증거.
- 처방(정정): RBMA/router를 살리려면 **per-modal decoder의 confidence calibration**(temperature/entropy penalty)으로 reliability 포화부터 해소해야 함.

## 6c. Router 융합은 **non-adaptive** (입력 적응 안 함) — 정량
24장에 걸친 router 가중치 통계 (mean±std, 이미지 간 변동):

| FPN level | img | lidar | thermal | 변동성 |
|---|---|---|---|---|
| P3 (fine, s4) | 0.000±0.000 | **0.992±0.002** | 0.008±0.002 | 사실상 상수 |
| P4 (mid, s8) | 0.219±0.021 | 0.008±0.001 | **0.773±0.020** | 사실상 상수 |
| **P5 (coarse, query-decoder 입력)** | **0.879±0.027** | 5.8e-8±1e-7 | 0.121±0.027 | **CV 3.0%, span 0.10** |

- **각 레벨 가중치가 입력 이미지와 무관하게 거의 상수**(P5 img CV=3%, lidar는 24장 전부 <1e-6). reliability가 전모달 포화(§6b)라 anchor가 무의미 → router의 학습 conv head가 **레벨별 고정 모달 배정**으로 수렴: **P3=LiDAR(0.99) / P4=thermal(0.77) / P5=RGB(0.88)**.
- 즉 제안한 "reliability-anchored **adaptive** fusion"이 **적응성을 상실**하고 정적 배정으로 붕괴. object-query decoder가 쓰는 P5에서 LiDAR≈0·RGB우위 고정 → 소형객체(thermal/lidar 유리) 정보가 주 검출경로에 반영 안 됨 = 소형객체 붕괴(§2)와 인과적으로 연결.
- **처방**: reliability 포화 해소(per-modal decoder calibration) 없이는 router가 adaptive해질 수 없음. 또는 router에 입력 의존 gating(스칼라 아닌 공간적/조건적) 도입.

**이 수치들이 기록된 위치**:
- per-image 원시 수치: `p30_feature_probe_full/probe_stats.json` → 각 항목 `router_weights.level{0,1,2}` = `[img,lidar,thermal]` (24장).
- per-image raw 텐서: `p30_feature_probe_full/raw/<image_id>.npz` → `router_level{0,1,2}`, `mem_*`, `rel_*`, `det_*`.
- 집계 시각화: `p30_feature_probe_full/summary.png` (좌상단 "router P5 weight" boxplot).
- (canonical) `/mnt/HDD2/src/logs/P29_vs_P30_v2_20260702/` · (구 git 경로) `.claude_logs/det_eval/p30_feature_probe/` — **현재 develop에 없음**, 태그 `archive/det-p29-p30-analysis` 에서 복원 · (hinton) `~/src/dm_eval/out_probe_p30_ep39_full/`.

## 7. 시각화 · raw 산출물

> 원본 panel PNG 24장은 git에 포함하지 않음 — 태그 `archive/det-p29-p30-analysis` 의 `.claude_logs/det_eval/p30_feature_probe/` 에서 복원 가능. (19MB, 미디어 규약상 NAS/태그 보관. per-class 수치 요약만 [`assets/p29-p30-perclass-compare.csv`](assets/p29-p30-perclass-compare.csv) 로 회수됨.)

- **panel 24장** `p30_feature_probe_full/panel_*.png` (검출뷰: **GT=빨강 점선, pred=녹색**).
- **summary.png**: 24장 aggregate(router/mem-rank/cosine/norm/det수/reliability).
- **raw/`<id>`.npz** (24개, /mnt/HDD2 + hinton `~/src/dm_eval/out_probe_p30_ep39_full/raw/`, 총 150MB): 다른 에이전트/파이썬 비교분석용 — `mem_{img,lidar,thermal}`(256,64,64 fp16), `fused_p5`, `rel_{...}`, `router_L{0,1,2}`, `det_boxes/scores/labels`, `gt_labels`. 로드: `d=np.load('<id>.npz'); d['mem_img']`.
- 재현 스크립트는 develop의 [`tools/probe_det_features.py`](../../tools/probe_det_features.py) (2026-07-28 회수 시 `.claude_logs/det_eval/p30_feature_probe/` → `tools/` 로 이동). raw/panel은 용량상 /mnt/HDD2·hinton·`archive/det-p29-p30-analysis` 태그 보관.
- `p30_feature_probe/panel_<id>.png` (6장): 행=[입력 3모달+FUSED PCA / fpn0 PCA / mem PCA(RBMA on)+router bar / reliability맵+검출결과]. RBMA |Δmem|=검정(무효과)·router LiDAR=0 시각 확인.
- `out_p30_ep39/` (hinton): diag_det 검출 bbox 시각화 16장.
