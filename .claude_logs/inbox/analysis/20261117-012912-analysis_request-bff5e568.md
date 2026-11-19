---
id: 20261117-012912-analysis_request-bff5e568
to: analysis
from: user
type: analysis_request
ref: DRN-261117-02
status: open
created: '2026-11-17T01:29:12.496639'
claimed_by: null
claimed_at: null
done_at: null
note: ''
---
## 요청: 평가만 DRN-261117-02(학습에서 보지 않은 결측 표현) + tab:abl_robust val 열 벤치 코드 재채점

출처: discussion, user 2026-11-17 "1~4 다 진행해줘". 모의 리뷰 2.3(iv)·2.6.
- DRN-261117-02: 결측 센서를 0 대신 평균값·N(0,1) 노이즈·상수·부분 시야 손실로 대체해 DELIVER 15조합 평균(test, tab:retrained 하네스). 대상 A2 3시드, E1 2시드, DGFusion 공개·(b), CAFuser (b). mm_eval_v2 소폭 확장 필요. DELIVER 연속 프레임 유무 확인(없으면 이전 프레임 대체 제외).
- tab:abl_robust 의 val 열(우리 채점기, 각주 "until the re-measurement")을 tab:robust 와 같은 벤치 코드로 재채점 — 강건성 수치 채점기 통일(2.6).
- 이미 요청된 B급 백본 결측 평가 DRN-261116-03·04 와 함께 처리 부탁.
