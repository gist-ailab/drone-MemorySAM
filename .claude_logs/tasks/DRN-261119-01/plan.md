착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# 센서별 합성 열화의 효과 축 찾기(학습 0): A2(센서 제거 + 센서별 열화) vs L7(센서 제거만)을 clean·통째 결측 밖의 사전 지정 3축에서 비교

- id: DRN-261119-01
- kind: analysis
- 문제(PRB): -
- 질문: 센서별 합성 열화(블러·노출·depth 구멍·LiDAR 빔 누락·event 해상도 저하)는 clean·EMM15 test 에서 제거만 쓴 것과 차이가 없었다(DRN-261029-01 reject). 열화가 실제로 듣는 축(부분 결측·노이즈·DELIVER 내장 센서 고장 시퀀스)이 있는가 — 선행 조사(2026-11-19)에서 멀티센서 증류에 센서별 열화를 넣은 것이 문헌상 비어 있는 유일한 차별점이라, 측정된 근거가 있는지 가른다
- 가설: 열화는 센서가 통째로 빠지는 상황이 아니라 센서가 '나빠지는' 상황에서 듣는다: DELIVER test 의 내장 센서 고장 케이스(motion blur·over/under exposure·LiDAR jitter·event low-resolution)와 부분 결측 RMM 에서 A2 > L7
- 바꾼 변수(change): 측정만. 비교 변수 = 학생 입력 교란(제거+열화 vs 제거만), 같은 시드 짝 821·902
- 기준선: A2 s821·s902 의 test RMM15·NM 벤치 test 는 DRN-261111-02 에서 이미 측정(재사용). L7 s821·s902 ckpt = jarvis eval_fill_20261105. 참고 행 L5(손상 패스 없음)
- 성공 기준: 사전 지정 3축(이 밖은 보지 않는다): ① test RMM 15조합 평균 r=.25/.5/.75 ② NM 벤치 정의 test Low/Mid/High(학습에 없던 노이즈) ③ DELIVER test 내장 센서 고장 케이스별 mIoU(전 센서 입력, 케이스: motion blur·overexposure·underexposure·LiDAR jitter·event low-res 및 고장 없는 clean 케이스) — 고장 케이스 5개 평균과 케이스별. 한 축에서 '열화 기여 인정' = A2 − L7 의 2시드 평균 ≥ +1.0 이고 두 시드 부호 일치. 인정된 축만 원고 주장에 쓰고 본 축 수(3)와 미충족 축을 모두 적는다
- 반증 조건: 3축 모두 미충족이면 센서별 합성 열화의 측정된 효과 없음으로 확정 → 원고에서 열화를 차별점으로 쓰지 않고 '제거만으로 같은 결과'로 서술
- configs: -

## 정정
