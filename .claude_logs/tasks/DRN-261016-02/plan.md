착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# MUSES 평가 해상도 반응 — PhysAug-off 3시드 공식 val @1024/1536/2048 (레터박스 0.53배 → 기준선 0.95배 정렬)

- id: DRN-261016-02
- kind: analysis
- 문제(PRB): PRB-DRN-261016-01
- 질문: MUSES 에서 우리 추론이 기준선 절반 해상도였던 것을 맞추면 공식 val 이 오르는가, 오르면 그 해상도로 test 제출본
- 가설: -
- 바꾼 변수(change): -
- 기준선: -
- 성공 기준: 3시드 공식 val 표; val 선택 해상도 결정; 1024 1개 시드 재현 ±0.05
- 반증 조건: -
- configs: tools/eval_muses_official.py + EVAL.IMAGE_SIZE 파생 config

## 정정
