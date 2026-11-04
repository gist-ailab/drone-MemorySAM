판정은 이 파일이 아니라 `lab-plan verdict DRN-261022-01 --analysis .claude_logs/tasks/DRN-261022-01/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-10-27. 요청 inbox `20261022-184018-request-7d5e45f3`(method). 학습 없음.
진행: **(a) DELIVER 완료(10-27), (a) MUSES 완료(11-04), (b) 선형 프로브 완료(11-04)**.

## 방법 — 객체 클래스 오라클
- 스크립트 `.claude_logs/experiments/analysis/DRN-261022-01/object_class_oracle.py`(합성 자기검증 `oracle_selftest.py` 통과).
- 예측 맵의 클래스별 8연결 성분 = 예측 객체. 성분 안 유효 GT(255 제외) 다수 클래스로 라벨을 바꾸고 마스크는 그대로 둔다.
- 채점: `tools/baseline_failure/common.py` 규약(hist[gt,pred], 부재 클래스 IoU 0 포함 25클래스 평균). 원래 mIoU 가 등록값과 소수점 넷째 자리까지 같다(Q2 57.7272/55.9818/56.0698, A2 58.6138/57.0919/57.5244).
- 두 가지 정의:
  - **전체판**: 모든 성분을 바꾼다. stuff 영역(SideWalk·Terrain·Ground 등)의 영역 단위 혼동까지 포함한다.
  - **사물판**(`--restrict`): 예측 클래스와 GT 다수 클래스가 **둘 다 사물 11클래스**(Pedestrian, Pole, TrafficLight, TrafficSign, Static, Dynamic, Cars, Bus, Truck, TwoWheeler, Other)일 때만 바꾼다. "마스크는 맞는데 사물 클래스가 틀림" 에 해당한다.
- 입력: DELIVER test 1897장, legal v2 1024 덤프. Q2 = yeon `dump_preds/q2s{821,902,903}_v2`, A2 = `analysis_logs/a2_test1024_dump_20261016/s{821,902}` + `a2s903_20261021/s903_test`.
- 원본: NAS `analysis_logs/object_oracle_deliver_20261027/{all,things}/`(summary.json·혼동행렬 npy).

## 사전 기준 대비 사실
H-X1 문턱: DELIVER A2 오라클 여유 ≥ +1.5(그리고 MUSES 1536 ≥ +1.0).

### (a) DELIVER 오라클 여유(오라클 mIoU − 원래 mIoU)
| 모델 | 원래 | 전체판 여유 | **사물판 여유** |
| --- | ---: | ---: | ---: |
| Q2 시드821 | 57.73 | +14.49 | +4.62 |
| Q2 시드902 | 55.98 | +15.47 | +4.05 |
| Q2 시드903 | 56.07 | +16.24 | +5.67 |
| A2 시드821 | 58.61 | +14.43 | +3.91 |
| A2 시드902 | 57.09 | +16.14 | +5.00 |
| A2 시드903 | 57.52 | +15.13 | +3.74 |
| A2 3시드 평균 | 57.74 | +15.23 | **+4.22** |
| Q2 3시드 평균 | 56.59 | +15.40 | +4.78 |

- 사물판 기준으로도 A2 여유 +3.74~+5.00 으로 문턱 +1.5 를 넘는다. 전체판은 stuff 영역 혼동이 절반 이상(+8.4~9.8)이라 "객체 클래스 여유" 로 인용하지 않는다.

### 사물판 클래스별 IoU 여유(A2 3시드 평균, 괄호 = mIoU 기여)
TrafficLight +36.70(+1.47) · Other +22.81(+0.91) · TrafficSign +16.85(+0.67) · Dynamic +10.60(+0.42) · Static +9.20(+0.37) · TwoWheeler +2.74 · Pole +2.68 · Truck +1.75 · Pedestrian +0.92 · Bus +0.68 · Cars +0.44.
- 상위 3클래스(TrafficLight·Other·TrafficSign)가 사물판 여유 4.22 의 72%(3.05)를 차지한다.

### 사물판에서 바뀐 픽셀 상위 혼동쌍(예측 → GT 다수, 백만 px; A2 시드821 / 902 / 903)
| 예측 → 정답 | s821 | s902 | s903 |
| --- | ---: | ---: | ---: |
| Static → Other | 1.26 | 3.21 | 0.76 |
| TrafficLight → Other | 1.18 | 1.46 | 1.05 |
| Pole → Other | 0.77 | 1.04 | 0.70 |
| TrafficLight → TrafficSign | 0.48 | 0.53 | 0.50 |
| Static → Dynamic | 0.17 | 0.17 | 0.24 |
| Pole → TrafficSign | 0.15 | 0.15 | 0.12 |
| Truck → Dynamic | 0.10 | 0.15 | 0.22 |

- 이 결과는 A2 실패-키 분석(DRN-261012-02)의 TrafficLight FP 출처(GT Other·TrafficSign)와 같은 방향이다.

### (a) MUSES 오라클 여유(2026-11-04 측정)
- 모델: E1M(MUSES 3센서 RGB·Event·LiDAR, PhysAug-off) 3시드 824·825·826, jarvis `/SSDb/jemo_maeng/ckpt/muses_physaugoff/`. A2 의 MUSES 판(4모달, DRN-261019-02)은 아직 없어 E1M 으로 측정했다(사전 계획과 같음).
- 덤프: 새 도구 `tools/muses_dump_preds.py`(develop 7a9ffe8, `eval_muses_official.py` 동결 파일 미수정·같은 경로), 공식 val 250장, native 1080×1920, 1024·1536 config(`configs/eval/muses_physaugoff_rank_eval{1024,1536}.yaml`). 재현: 공식 val mIoU 824 82.4631/83.1175 · 825 82.1859/82.6253 · 826 82.2560/82.8473(1024/1536) — DRN-261016-02 기록과 일치(825 1024 만 82.1857 대비 +0.0002).
- 오라클: 같은 스크립트에 `--dataset muses`(19클래스, 공식 채점 = GT 에 있는 클래스만 평균). 오라클 입력의 원래 mIoU 가 공식 값과 소수 넷째 자리까지 같다. 사물판 집합 = pole·traffic light·traffic sign·person·rider·car·truck·bus·train·motorcycle·bicycle(DELIVER 사물 11클래스에 대응).
- 원본: NAS `analysis_logs/muses_dump_e1m_20261104/`(report.json·조건별 혼동행렬·예측 PNG) · `analysis_logs/object_oracle_muses_20261104/{all,things}/`.

| 오라클 여유(3시드: 824 / 825 / 826, 평균) | 1024 | 1536 |
| --- | --- | --- |
| 사물판 | +1.82 / +1.99 / +1.68, **+1.83** | +1.96 / +2.39 / +2.18, **+2.18** |
| 전체판 | +4.11 / +4.55 / +3.83, +4.16 | +4.49 / +5.08 / +4.52, +4.70 |

- 사물판 1536 클래스별 IoU 여유(3시드 평균, 괄호 = mIoU 기여): rider +15.77(+0.83) · truck +5.40(+0.28) · bicycle +4.01(+0.21) · motorcycle +3.98(+0.21) · person +2.99(+0.16) · bus +2.82(+0.15) · traffic light +2.23(+0.12) · traffic sign +2.10(+0.11).
- rider 한 클래스가 사물판 여유 2.18 의 38% 다. rider 는 MUSES 에서 화소가 매우 적은 클래스(시드당 TP 약 0.06M px, DRN-261016-03)라 몇 장면에 좌우된다. rider 를 빼면 사물판 여유는 약 +1.35 다.
- 사물판에서 바뀐 픽셀 상위 쌍(1536, 3시드 공통): car → truck(0.05~0.08M) · pole ↔ traffic sign · traffic light → pole · pole → car. DELIVER(쌍당 0.5~3M px)보다 바뀐 픽셀이 1~2 자릿수 적다.
- 전체판의 큰 몫은 stuff 혼동(terrain → fence, vegetation ↔ sky, wall → fence, road → sidewalk)이라 DELIVER 와 같이 "객체 클래스 여유" 로 인용하지 않는다.

### H-X1 사전 기준 대비
- DELIVER A2 사물판 여유 +4.22(≥ +1.5 충족), MUSES 1536 사물판 여유 +2.18(≥ +1.0 충족, rider 제외 시 약 +1.35 도 충족). 반증 조건(MUSES < +0.5) 해당 없음.
- 두 벤치 모두 객체 클래스 여유 ≥ +1.0 이다(사물판 정의 기준).

### (b) 선형 프로브(H-X2, 2026-11-04 측정)
- 도구 `tools/probe_object_pooling.py`(develop 67a7c50, 워커 codex2 작성·이 세션 검수). A2 s821, DELIVER **val** 2005장, jarvis GPU7(프로세스 0개 확인). GT 클래스별 8연결 성분(면적 ≥ 64 px)을 객체로 보고, 마스크를 특징 격자로 면적 평균한 가중치로 두 특징을 풀링: fused = `model.fusion` 출력(stride 16, 1024ch), hires_s4 = `model._hires_feats[0]`(stride 4, 256ch). 표준화 + 로지스틱 회귀(L2, C=1, class_weight balanced), **이미지 단위 GroupKFold 5-fold**.
- 객체 수: TrafficLight 4,426 · TrafficSign 437 · Other 287(총 5,150).

| 쌍 | 특징 | AUROC(풀링) | fold 평균 ± std |
| --- | --- | ---: | ---: |
| TrafficLight : Other | fused | 1.0000 | 1.0000 ± 0.0000 |
| TrafficLight : Other | hires_s4 | 1.0000 | 1.0000 ± 0.0000 |
| TrafficLight : TrafficSign | fused | 1.0000 | 1.0000 ± 0.0000 |
| TrafficLight : TrafficSign | hires_s4 | 1.0000 | 1.0000 ± 0.0000 |

- 검산(버그 배제): 라벨을 섞은 대조군 AUROC 0.47~0.54(우연 수준), 면적 단독 0.52, 특징 벡터 5,150개 모두 서로 다름·NaN 없음. AUROC 1.0 은 누설이 아니라 실제 분리다.
- **H-X2 사전 기준 대비**: "fused < 0.8, 고해상 ≥ 0.9" 의 앞부분이 성립하지 않는다(fused 도 1.0). 반증 조건 "고해상과 fused 차이 < 0.05" 에 해당(차이 0) → '고해상 질감이 클래스를 가른다' 가설은 기각 조건에 든다.
- [미검증] 해석: GT 객체 마스크 전체로 풀링하면 의미 특징만으로 쌍이 완벽히 갈린다. 따라서 TrafficLight ↔ Other/TrafficSign 혼동은 객체 수준 의미 부족이 아니라 **픽셀 단위 결정**(객체 일부·경계, 64 px 미만 소형 객체 — 이번 프로브에서 제외)에서 생긴다고 보는 것이 맞다. 같은 날 DRN-261020-01 에서 고해상 경로가 TL 예측 전체(TP·FP)를 담당함을 확인.
- 한계: val 기준(test 아님), 64 px 미만 객체 제외, 단일 시드.
- 원본: NAS `analysis_logs/a2_hires_ablation_20261104/probe_s821_val/`(objects.npz·csv·probe_auroc.csv·run.json) + 검산 스크립트 `probe_sanity.py`.

## 원인 해석 (검증 여부 표시)
- [사실] DELIVER 에서 "마스크는 그렸는데 사물 클래스를 틀린" 손실은 mIoU 약 4 점이다. 그 대부분은 Other·TrafficSign 과 다른 사물(TrafficLight·Static·Pole) 사이의 혼동이다.
- [사실] A2(고해상 경로)는 Q2 보다 사물판 여유가 평균 0.56 작다. A2 가 일부 클래스 혼동을 이미 줄였지만, TrafficLight 여유는 오히려 크다(A2 +36.7 vs Q2 +29.7).
- [미검증] DELIVER 의 Other 는 "기타 물체" 범주라 정의 자체가 넓다. 여유 중 Other 몫(+0.91)은 라벨 정의 문제일 수 있다.

## 판정 제안
판정은 method 가 한다.

## 다음 결정
- (b) 완료. 소형 객체(64 px 미만)·픽셀 단위 결정을 보려면 객체 일부(경계 띠) 풀링 변형을 추가할 수 있다.
- MUSES 측 A2(4모달) 가 나오면 같은 오라클로 다시 잴 수 있다(덤프 도구·스크립트 준비됨).
