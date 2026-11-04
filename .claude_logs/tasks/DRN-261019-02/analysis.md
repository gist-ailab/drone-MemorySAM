판정은 이 파일이 아니라 `lab-plan verdict DRN-261019-02 --analysis .claude_logs/tasks/DRN-261019-02/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-11-04. 요청 inbox `20261104-205009-request-550b6a5f`(method). 학습 없음.

## 측정 방법
- 모델: MUSES 4모달(img·lidar·event·radar) A2 = E1 구조 + Q2 레시피(열화 두 패스 + E1M4 교사 증류) + 고해상 경로, 40ep 스크린, 768 크롭 학습.
  - s2 `epoch40_80.56_top1`(md5 019b118fd3bafbb282f901c7c000183c), s3 `epoch40_80.18_top1`(md5 b57f14b8b29832407a382c7493b142d7). yeon `/SSDe/.../drone-MemorySAM/outputs/ReliaDINO/yeon_muses_rgbelr_P39_1_physaugoff_taps_screen40_A2_s{2,3}_768/`. 학습기 top1(80.56·80.18)은 768 내부 레터박스 val 값이다.
- 도구: `tools/eval_muses_official.py`(공식 val 250장, native 1080×1920 채점, 등장 클래스 평균). 평가 config = `configs/eval/muses_a2m4_768_eval{1024,1536}_yeon.yaml`(학습 config 에서 EVAL/TEST.IMAGE_SIZE·교사 경로만 변경, TRAIN.IMAGE_SIZE 768 유지 → pos-embed 는 768 기준으로 만들고 평가 입력만 키움).
- 실행: yeon GPU0(s2)·GPU1(s3), 기동 직전 두 GPU 프로세스 0 확인, GPU2~5(L4 학습)·6(고장) 미접촉. 코드 = develop archive `dms-analysis-develop`(학습 중 체크아웃 미접촉). ckpt 둘 다 `loaded clean`.
- 원본: yeon `/SSDe/jemo_maeng/analysis_logs/muses_a2m4_official_val_20261104/`(시드×해상도 report.json·hist, 로그, ckpt_md5.txt) → NAS 같은 이름 폴더로 회수.

## 사실 — 공식 val mIoU (native 1080×1920, 250장)
| 시드 | 1024 | 1536 | Δ(1536−1024) |
| --- | ---: | ---: | ---: |
| s2 | 81.491 | 81.758 | +0.268 |
| s3 | 81.344 | 81.311 | −0.033 |
| **2시드 평균** | **81.418** | **81.535** | **+0.117** |
- 19클래스 모두 GT 에 등장(부재 없음) → all19 와 등장 클래스 평균이 같다.
- **해상도 선택 규칙(사전): 2시드 평균이 높은 해상도 = 1536.** 다만 차이 +0.12 는 시드 간 차이(1024 에서 0.15, 1536 에서 0.45)보다 작고 s3 는 1536 에서 오히려 −0.03 이다. 비교: E1M(3모달 clean, 3시드) 은 1024→1536 이 +0.56(82.30→82.86, DRN-261016-03) 이었다 — 4모달 A2 의 해상도 이득은 그보다 작다.

### 조건별(2시드 평균, 조건당 25~50장 — 참고만)
| 조건 | n | 1024 | 1536 | Δ |
| --- | ---: | ---: | ---: | ---: |
| clear/day | 50 | 78.71 | 77.64 | −1.07 |
| clear/night | 25 | 69.49 | 70.85 | +1.36 |
| fog/day | 33 | 86.83 | 87.65 | +0.82 |
| fog/night | 25 | 73.41 | 70.03 | −3.38 |
| rain/day | 34 | 67.04 | 68.21 | +1.17 |
| rain/night | 25 | 68.29 | 66.83 | −1.46 |
| snow/day | 33 | 80.49 | 81.43 | +0.95 |
| snow/night | 25 | 74.34 | 72.73 | −1.60 |
- 시드별 조건 값은 원본 report.json `per_condition_official`. fog/night 은 시드 간 차이가 1024 에서 5.5점(76.18/70.64)이라 조건 단위 Δ 는 해석하지 않는다.
- [사실] 열화+야간 3조건(fog·rain·snow night)이 1536 에서 떨어지고 낮 조건·clear/night 이 오른다. E1M 3시드에서 본 패턴(DRN-261016-03 Q3, 같은 3조건 하락)과 방향이 같다.

### 클래스별(2시드 평균) 1536−1024
- 오름: pole +1.85 · traffic light +1.74 · traffic sign +1.24 · bicycle +1.11 · person +0.82 · bus +0.77.
- 내림: motorcycle −2.48 · rider −2.23 · fence −1.44 · terrain −0.48. rider·motorcycle 은 GT 화소가 적은 클래스라 변동이 크다.

## 정성 은행 덤프
- s2 공식 val 250장 전체 예측 덤프 완료: NAS `analysis_logs/muses_a2m4_official_val_20261104/dump_s2_{1024,1536}/pred/{pred,gt}/<weather>/<tod>/<stem>.png`(각 250+250장, 재현 mIoU 1024 81.4908 · 1536 81.7583 = 공식 평가와 동일) + hist_cond. 선택 해상도 1536 쪽을 은행 정본으로 쓰도록 paper 에 통지(2026-11-04 22:1x).

## 판정 제안
판정은 method 가 한다. test 추론 해상도는 사전 규칙상 1536 이나, 차이가 시드 잡음 안이라는 점을 함께 적는다.

## 다음 결정
- test 추론·zip(DRN-261029-09)은 1536 으로 진행할지 method 확인 후(요청상 별도 승인 불필요로 적혀 있으나, 차이가 작아 1024 와 1536 중 하나로 확정하는 문장을 판정에 남기는 것을 권한다). 제출은 user.
