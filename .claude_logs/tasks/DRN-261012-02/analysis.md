판정은 이 파일이 아니라 `lab-plan verdict DRN-261012-02 --analysis .claude_logs/tasks/DRN-261012-02/analysis.md`로 기록한다.

## 측정
작성: MMSAM | analysis, 2026-10-18. 요청 inbox `20261017-102640-analysis_request-54d6e33e`(method). 새 추론 없음, 기존 덤프 재집계.

입력:
| 모델 | 덤프 | 재채점 mIoU(25클래스) | ckpt |
| --- | --- | ---: | --- |
| Q2 시드821 | yeon `/SSDb/jemo_maeng/dump_preds/q2s821_v2/test` | 57.73 | Q2_s821_epoch40_67.0.pth |
| Q2 시드902 | yeon `/SSDb/jemo_maeng/dump_preds/q2s902_v2/test` | 55.98 | Q2_s902_epoch40_66.06.pth |
| A2 시드821 | NAS `analysis_logs/a2_test1024_dump_20261016/s821/test` | 58.61 | yeon …A2hires/…/epoch40_68.93_top1_checkpoint.pth |
| A2 시드902 | NAS `analysis_logs/a2_test1024_dump_20261016/s902/test`(yeon 사본 md5 일치) | 57.09 | bengio …A2hires/…/epoch40_70.09_top1_checkpoint.pth |
모두 DELIVER test 1897장, legal v2, 평가 1024. Q2 재채점값은 등록값(57.73/55.98)과 일치한다.

### (1) mIoU·클래스별 Δ (A2 − Q2, 같은 시드)
| 지표 | 시드821 | 시드902 | 2시드 평균 |
| --- | ---: | ---: | ---: |
| 25클래스 Δ | +0.89 | +1.11 | +1.00(A2 57.85 vs Q2 56.85) |
| 24클래스(RailTrack 제외) Δ | +0.99 | +1.18 | **+1.08**(A2 58.41 vs Q2 57.33) |
| 얇은4 평균 Δ | +2.17 | +0.62 | **+1.39** |

얇은4 클래스별 Δ(시드821 / 시드902 / 평균): Pole −1.06 / +4.77 / +1.86 · TrafficLight −2.68 / **−10.51** / **−6.60** · Pedestrian +3.69 / +3.55 / +3.62 · Static +8.73 / +4.66 / +6.69.
그 밖의 큰 변화(평균): Water +5.48 · TwoWheeler +2.29 · SideWalk +2.02 · TrafficSign +1.99 · Dynamic +1.75 · GroundRail +1.39 · Ground −2.10 · RailTrack −1.06.
전표: `.claude_logs/experiments/analysis/DRN-261012-02/a2_vs_q2_perclass.csv`.

### (2) 두께 구간별 찾음률·"찾았지만 못 그림" — 실행 중
yeon tmux `a2thin`, 출력 `/SSDe/jemo_maeng/analysis_logs/a2_thin_anatomy_20261018/`(--model_input_px 1024 --patch_px 16, 4모델). 완료 후 이 절에 덧붙인다.

### (3) Pole 가짜 기둥(GT Building/Other→Pole px) — (2)와 같은 실행의 fp_sources.md 에서 추출 예정

## 사전 기준 대비 사실
- 24클래스 Δ ≥ +0.5: **+1.08, 충족**(2시드 모두 +0.99 이상).
- 얇은4 평균 Δ 문턱: 사전 문구는 "A1 이 같은 해상도 대비 얻은 얇은4 이득의 절반 — A1 미완 시 +1.5"다. A1(학습 스케일 상향)은 완료됐지만 얇은4가 −1.01(손실)이어서 "이득의 절반"이 정의되지 않는다. +1.5 를 적용하면 +1.39 로 미달이고, 반증 문턱(+0.5)은 넘는다. 어느 문턱을 쓸지는 method 가 정한다.
- 두께 0.25~0.5 패치 "못 그림" ≤ .35: (2) 결과 대기.

## 원인 해석 (검증 여부 표시)
- [미검증] TrafficLight 하락(−6.60)은 시드902 에 몰려 있다(−10.51). DELIVER Q2 의 1536·2048 평가에서도 TrafficLight 는 FP 증가로 떨어졌다(thin_anatomy run_all). 그래서 고해상 특징이 신호등 주변 FP 를 늘리는 같은 현상일 수 있다. (2)의 precision_recall 로 FP/FN 을 확인할 예정이다.

## 판정 제안
판정은 method 가 한다.

## 다음 결정
- (2)(3) 결과 수신 후 이 파일 갱신 → `lab-plan status DRN-261012-02 judging`.
