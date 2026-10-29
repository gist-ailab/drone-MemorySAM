착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# MUSES 4모달 A2 test 제출본(zip) 생성 — 공식 val 로 평가 해상도 선택 후 test 추론·포맷 검증, 제출은 user

- id: DRN-261029-09
- kind: build
- 문제(PRB): PRB-DRN-261016-01
- 질문: 논문 모델 A2(4모달, 768 크롭 학습)의 MUSES test 제출본을 공식 포맷으로 만들 수 있는가(3모달 제출본은 금지, user 2026-10-18)
- 가설: A2 s2·s3 완주 후 공식 val(native 1080×1920, tools/eval_muses_official.py)에서 1024/1536 중 높은 해상도를 고르고 그 설정으로 test 추론 → Codabench 포맷 zip
- 바꾼 변수(change): 학습 없음. 입력 = DRN-261019-02 의 val-best ckpt(시드별), 평가 해상도 선택 규칙 = val 최고(2026-10-16 규칙), 출력 = /ailab_mat2 … submission/muses/(허브 마운트 불가 시 drone_nas submission/muses/ 에 두고 README 로 안내) + md5 + 포맷 검증 로그
- 기준선: 기존 4모달 최고 P39.1-rank 시드2 공식 test 79.571
- 성공 기준: zip 2개(s2·s3) + 공식 val 수치·선택 해상도·md5 기록, 포맷 검증 통과. 제출·결과 기록은 user
- 반증 조건: 해당 없음
- configs: -

## 정정
