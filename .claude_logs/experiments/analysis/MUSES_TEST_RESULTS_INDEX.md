# MUSES 공식 test 결과 — 통합 인덱스 (2026-07-20 기준)

Codabench comp **14005** 제출분 전체. **이 파일이 MUSES test 수치의 단일 진입점**이고, 상세·원시 데이터는 아래 "출처" 열을 따라간다.
⚠️ MUSES test GT는 서버 비공개 → **test-best 훔쳐보기가 구조적으로 불가**. 전 제출이 val-best ckpt 단일 선택이라 선택편향 없음.

## 제출 전체 (시간순)

| # | 모델 / ckpt | 모달 | 내부 val | 공식 val | **공식 test** | 제출일 |
|---|---|---|---|---|---|---|
| 1 | P34-ReliaDINO ep276 | 3모달 CLE | 81.02 | 80.86 | **78.979** | 07-15 |
| 2 | P34-ReliaDINO ep182 | **4모달 CLRE**(+radar) | 80.76 | 80.77 | **78.256** | 07-17 |
| 3 | **P38-m2f ep156** | 3모달 CLE | 82.22 | — | **79.025** ★최고 | 07-20 |
| — | P37a-CEFR ep190 | 3모달 CLE | 81.57 | — | *미제출* | zip 보관 |
| — | P39-DPC ep146 | 3모달 CLE | 81.52 | — | *미제출* | zip 보관 |
| 4 | **P46-C3only λ0.2 ep136** | 3모달 CLE | 81.65 | — | **79.023** | 08-03 |

**SOTA**: GtA **82.39**(camera-only) → 우리 최고 79.025와 **−3.365**. 순위: GtA 82.39 > MM-SAM-adapter 81.07 > DGFusion 79.5 > **P38 79.025** > P34 78.979 > CAFuser-CAA 78.5 > CAFuser-CA² 78.2 > CMNeXt 72.1.
→ **비교군은 전원 4모달(CLRE), 우리만 3모달** — CAFuser를 모달 하나 덜 쓰고 상회.

## 핵심 판정 3가지

1. **radar는 손해**: 동일 P34에서 3모달 78.979 → 4모달 **78.256** (**−0.72**). val도 −0.09. → 3모달 유지가 맞음.
2. **val 개선의 test 전이율이 낮다**: P34→P38 내부 val **+1.20** → test **+0.046**(전이 ~4%). val→test 낙차는 ~2~3pt로 일관. **내부 val 개선을 test 개선으로 읽지 말 것.**
3. **snow_day < snow_night 역전이 2회 재현**: P34-4모달 69.711<73.994, P38 70.584<74.867. 다른 날씨는 전부 day>night. **주간 설상 고반사/과노출 가설, 미검증.**

## 조건별 대조 (제출 2·3)

| condition | P34 4모달 | P38 3모달 | Δ(P38−P34₄) |
|---|---|---|---|
| **Full (750)** | 78.256 | **79.025** | +0.77 |
| Clear (225) | 77.693 | 78.218 | +0.53 |
| **Fog (175)** | **70.884** ← 최약 | **77.524** | **+6.64** ⭐ |
| Rain (175) | 77.536 | 78.096 | +0.56 |
| Snow (175) | 76.394 | 78.329 | +1.94 |
| Day (450) | 79.225 | 80.253 | +1.03 |
| Night (300) | 74.786 | 75.118 | +0.33 |
| 주야 격차 | 4.44 | **5.14** | 악화 |
| clear_day / clear_night | 78.978 / 73.461 | 80.222 / 71.877 | |
| fog_day / fog_night | 69.622 / **64.451** | 76.747 / 74.728 | **+7.1 / +10.3** ⭐ |
| rain_day / rain_night | 77.367 / 73.180 | 78.512 / 73.510 | |
| snow_day / snow_night | 69.711 / 73.994 | 70.584 / 74.867 | 둘 다 역전 |

> ⭐ **P38의 이득은 압도적으로 fog에서 나왔다** (+6.64, fog_night +10.3). 나머지 조건은 +0.3~1.9. P34-4모달은 fog에서 **train IoU 0.00 완전사멸** + motorcycle 2.23 + rider 42.65였는데, P38 fog는 train **100.00** · motorcycle 21.95 · rider 38.05. → **M2F 쿼리헤드가 안개에서 대형 구조물 사멸을 막았다**는 해석이 가능(단 motorcycle/rider는 여전히 낮음).

## per-class (Full test)

| class | P34 4모달 | P38 3모달 | Δ |
|---|---|---|---|
| road | 97.15 | 97.18 | +0.03 |
| sidewalk | 86.81 | 87.51 | +0.70 |
| building | 92.82 | 92.97 | +0.15 |
| wall | 80.48 | 80.43 | −0.05 |
| fence | 64.43 | 66.61 | +2.18 |
| pole | 59.32 | 61.46 | +2.14 |
| traffic light | 67.71 | 71.08 | +3.37 |
| traffic sign | 71.05 | 73.93 | +2.88 |
| vegetation | 89.28 | 89.07 | −0.21 |
| terrain | 78.02 | 79.31 | +1.29 |
| sky | 96.55 | 96.24 | −0.31 |
| person | 66.87 | 70.88 | +4.01 |
| rider | 58.03 | 57.68 | −0.35 |
| car | 93.21 | 93.73 | +0.52 |
| truck | 72.47 | 73.89 | +1.42 |
| bus | 94.62 | 94.24 | −0.38 |
| train | 93.34 | 93.47 | +0.13 |
| motorcycle | 58.57 | 55.07 | **−3.50** |
| bicycle | 66.16 | 66.72 | +0.56 |

**전역 약클래스(P38 기준)**: motorcycle 55.07 · rider 57.68 · pole 61.46 · fence 66.61 · bicycle 66.72 — 전부 얇거나 작은 구조물. **motorcycle은 P34-4모달보다 오히려 −3.50 퇴보.**
**야간 붕괴(P38)**: truck 76.43(day)→**44.40**(night, −32.03) · bus −15.27 · bicycle −18.44. 역전: traffic sign **+7.59**(반사재 추정).

## 출처 (상세·원시 데이터)

| 제출 | 상세 문서 | 원시 데이터 |
|---|---|---|
| P34 3모달 | `.claude_logs/experiments/analysis/2026-07-15-p34-muses-test-official.md` (75줄, Codabench sub 850776) | **NAS** `ckpts/MUSES_P34_20260715/official_eval/` — `hist_per_condition.npz`·`hist_full.npy`·`hist_1024.npy`·`REPORT.md`·`report.json`·`viz/` ⭐**원시 혼동행렬 있음** |
| P34 4모달 | `.claude_logs/experiments/monitor-log.md` §2026-07-17 (per-condition·per-class 전문) | **NAS** `ckpts/MUSES_P34_4modal_20260717/official_eval/` — `hist_full.npy`·`hist_1024.npy`·`report.json` |
| P38-m2f | `.claude_logs/experiments/analysis/2026-07-20-muses-official-test-P38-m2f-ep156.md` (102줄) + ailab_mat2 사본 | *(Codabench가 최종 수치만 반환 — 원시 hist 없음. 필요 시 예측 PNG 750장으로 로컬 재집계 가능)* |
| P46-C3only λ0.2 | `.claude_logs/experiments/analysis/2026-08-03-muses-official-test-P46-c3only-lam02.md` | *(Codabench 최종 수치만 반환)* |

**제출 zip 아카이브(정본)**: `/ailab_mat2/personal/jemo_maeng/src/Project/Drone/drone-memorysam/submission/muses/` (zip 5종 + P38 결과문서)
**제출 절차·규격**: 같은 폴더 `../code/README.md`

## 분석 표적 (이 데이터가 가리키는 것)

1. **motorcycle/rider/pole = 전 제출 공통 최약** — 얇고 작은 객체. P38에서 motorcycle은 오히려 퇴보(−3.50).
2. **주야 격차가 주 축**(P38 5.14, P34-4모달 4.44) — 날씨 spread는 P38에서 0.8로 평평해짐. **야간 truck −32.03**이 최대 손실원.
3. **snow_day 역전**(2회 재현) — 원인 미규명, 단일 조사 대상.
4. **fog는 P38이 해결에 가까움**(+6.64) — 무엇이 작동했는지 규명하면 다른 조건에 이식 가능.
