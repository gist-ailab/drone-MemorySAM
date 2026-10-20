착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# Ablation L1 · A2 − 센서별 LoRA(→ 완전공유 LoRA 1개, r 동일)

- id: DRN-261020-09
- kind: experiment
- 문제(PRB): PRB-DRN-261020-01
- 질문: 센서별 LoRA 가 A2 의 clean·강건성에 얼마나 기여하는가
- 가설: E-LoRA(09-21) 재현: clean 24클래스 −0.5~−1.0, depth 열화 내성은 공유 쪽이 더 큼 → 두 축이 갈림
- 바꾼 변수(change): LoRA 를 센서 공유 1개로(E1shared config 의 키를 A2 config 에 적용), 그 외 A2 동일
- 기준선: A2 같은 시드(s821·s902), DELIVER 40ep 스크린, legal v2 test@1024 + 모달 zero-out + EMM15(val·test)
- 성공 기준: 표 행 완성(24클래스·thin4·depth 제거 Δ·EMM15 val)
- 반증 조건: 해당 없음(ablation 행)
- configs: -

## 정정
