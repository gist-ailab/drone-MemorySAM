판정은 이 파일이 아니라 `lab-plan verdict DRN-261022-01 --analysis .claude_logs/tasks/DRN-261022-01/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-10-27. 요청 inbox `20261022-184018-request-7d5e45f3`(method). 학습 없음.
진행: **(a) DELIVER 완료**, (a) MUSES 는 도구 준비 완료·GPU 대기, (b) 선형 프로브 미착수.

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

### (a) MUSES — 미측정(GPU 대기)
- E1M(MUSES 3센서 PhysAug-off) 3시드 824·825·826 의 val 예측 덤프가 없어 새 도구 `tools/muses_dump_preds.py`(develop 7a9ffe8)를 만들었다. `eval_muses_official.py`(하네스 동결)와 같은 추론·채점 경로이고 동결 파일은 수정하지 않았다(sha256 매니페스트 일치). 1024 config `configs/eval/muses_physaugoff_rank_eval1024.yaml`(d0e1ca5).
- 체크포인트는 hpca100 `/tmp/jemo_maeng/muckpt/`(md5 jarvis 원본과 일치). 2026-10-27 17:30 기준 hpca100 GPU 4장과 연구실 서버 모두 사용 중이라 기동하지 못했다.

### (b) 선형 프로브 — 미착수
- GT 객체별 fused 토큰 평균과 고해상 s4 특징 평균을 뽑는 훅 코드가 필요하다(새 코드, GPU 필요).

## 원인 해석 (검증 여부 표시)
- [사실] DELIVER 에서 "마스크는 그렸는데 사물 클래스를 틀린" 손실은 mIoU 약 4 점이다. 그 대부분은 Other·TrafficSign 과 다른 사물(TrafficLight·Static·Pole) 사이의 혼동이다.
- [사실] A2(고해상 경로)는 Q2 보다 사물판 여유가 평균 0.56 작다. A2 가 일부 클래스 혼동을 이미 줄였지만, TrafficLight 여유는 오히려 크다(A2 +36.7 vs Q2 +29.7).
- [미검증] DELIVER 의 Other 는 "기타 물체" 범주라 정의 자체가 넓다. 여유 중 Other 몫(+0.91)은 라벨 정의 문제일 수 있다.

## 판정 제안
판정은 method 가 한다.

## 다음 결정
- MUSES (a): 빈 GPU 가 나면 `tools/muses_dump_preds.py` 로 3시드 × 1024·1536 덤프 후 같은 오라클(MUSES 클래스 매핑) 실행.
- (b): 특징 추출 훅 코드 작성(워커 위임) 후 실행.
