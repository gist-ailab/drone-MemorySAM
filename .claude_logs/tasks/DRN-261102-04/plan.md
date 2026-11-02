착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# A2 3시드 NM 벤치 정의 측정(학습 0) — Liao et al. 2025(arXiv 2503.18445) 식 7: S&P(D)+Gaussian(σ, Event 제외), Low/Mid/High = (0.05,0.1)/(0.1,0.2)/(0.2,0.5), val·test

- id: DRN-261102-04
- kind: experiment
- 문제(PRB): PRB-DRN-261016-01
- 질문: 원고 Table 4 의 NM 을 S&P 단독 자체 정의 대신 인용 벤치마크 정의로 재서 Table 11(MiT-B2 5종) 바로 아래에 붙일 수 있는가
- 가설: A2 는 벤치 NM 에서도 벤치 보고 최고(High: StitchFusion 9.25, Mid: MAGIC++ 18.31, Low: CMNeXt 35.23)를 넘는다
- 바꾼 변수(change): 평가 노이즈 정의만 교체: tools/mm_eval_v2.py --protocol nm --nm_mode bench(기본 levels 0.05:0.1,0.1:0.2,0.2:0.5), 모델·체크포인트·하네스(legal v2 1024) 불변
- 기준선: 벤치 Table 11 mIoU_NM(val, MiT-B2): CMNeXt 35.23/16.37/2.31 · GeminiFusion 5.52/4.14/3.30 · MAGIC 24.03/13.31/3.98 · MAGIC++ 33.12/18.31/8.70 · StitchFusion 24.91/17.11/9.25; 같은 체크포인트의 기존 S&P 단독 test 53.11/51.36/47.85
- 성공 기준: 측정 항목(학습 0): 3시드×(val,test)×(Low,Mid,High) 18값 + clean 이 기존 legal v2 값(val 69.59/71.05/69.86, test 58.61/57.09/57.52)과 ±0.05 내 일치하면 유효 → Table 4(a) 에 NM 열 추가, 원고의 'S&P 단독' 서술·S&P 열 삭제(user 2026-11-02)
- 반증 조건: clean 이 기존 값과 0.05 넘게 어긋나면 측정 무효(하네스·config 점검). 벤치 보고 최고보다 낮아도 결과를 숨기지 않고 그대로 게재
- configs: configs/eval/a2s903_eval1024_hpca100.yaml, configs/eval/a2s821_eval1024_hpca100_probe.yaml, configs/eval/a2s902_eval1024_hpca100_probe.yaml

## 정정
