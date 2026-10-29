---
id: 20261029-174230-analysis_request-a27dd4d7
to: analysis
from: user
type: analysis_request
ref: DRN-261027-01
status: open
created: '2026-10-29T17:42:30.105974'
claimed_by: null
claimed_at: null
done_at: null
note: ''
---
## 재요청 + 추가: 20261028-183826-analysis_request-253e6bd6(paper 의 DRN-261027-01 표 측정 요청) 미처리 — 우선 처리 부탁

출처: 세션 "MMSAM | discussion", user 직접 지시 2026-10-29 "둘 다 보내줘 테이블이 있어야 논문 작성을 먼저 하는거 아니야?". 판정 불필요, 측정·집계만.

기존 요청 1~4(조건별 clean 표 · A2 센서 하나만 · ablation EMM 열 · 조건별 의존도 test 통일)는 그대로 유효하다. 여기에 두 가지를 더한다.

5. **E1 조건별 센서 하나 제거(특히 RGB 제거)를 test 로 측정** — 요청 4(oracle_cond_test)의 행에 E1(clean 학습만 한 같은 구조) 포함. 이유: 논문 기여 후보 "조건 감독 없이 학습되는 적응형 융합"의 근거(ledger:102, Q2 s821 val: 야간 RGB 제거 저하 −8.62→−5.15, DGFusion 은 −9.6→−9.0)가 학습 덕인지 구조 덕인지 가르려면 E1 의 같은 표가 필요하다. 행 = DGFusion 공식 가중치(md5 0f40a641…, ledger:130) · E1 · Q2 · A2, 열 = 조건 6종 × {−RGB, −depth, −event, −LiDAR}, 전부 test 1897장 같은 축.
6. **A2 3시드 val 평균 기록**(legal v2 val@1024). headline.yaml:39 메모상 시드903 69.86 만 있고 평균은 미확인. 원고 Table 1 val 칸.

주의: 원고 Table 4(b) "DGFusion (official)" 행 값 55.68/31.53 은 재현본 (a)(ledger:117)이다. 공식 가중치는 clean 56.73 · EMM 31.14 · RGB만 7.50(ledger:130). 새 측정의 DGFusion 행은 **공식 가중치**로 하고 행 이름에 재현본/공식을 구분해 달라.
