리포트 요약
소수점 값의 의미
범위: 0.0 ~ 1.0 (정규화된 그레이스케일 밝기)
0.0 = 검은색
1.0 = 흰색
0.5 = 중간 회색
형식: 평균값 ± 표준편차
평균: 해당 조건의 전체 이미지 평균 밝기
± 표준편차: 이미지 간 밝기 분산 (클수록 다양함)
주요 발견사항
RGB 밝기 (환경 조건에 가장 민감)
가장 밝음: OVEREXPOSURE (0.87) > SUN (0.57)
가장 어두움: UNDEREXPOSURE (0.04) > NIGHT (0.11)
평균 밝기 순: OVEREXPOSURE > SUN > FOG > MOTIONBLUR > RAIN > EVENTLOWRES > LIDARJITTER > CLOUD > NIGHT > UNDEREXPOSURE
Depth 밝기 (환경 조건과 무관)
모든 조건에서 약 0.34 (거의 일정)
Depth 센서는 물리적 거리를 측정하므로 조명 조건에 영향받지 않음
LiDAR 밝기 (매우 낮음)
모든 조건에서 약 0.007 (매우 어두움)
점군 데이터의 특성상 대부분 배경(검은색), 포인트만 밝음
Event 밝기 (조건별 차이)
NIGHT (0.033) > 다른 조건들 (0.023~0.028) > FOG (0.004)
데이터 불균형
RAIN: 3,983개 (가장 많음, 약 40%)
UNDEREXPOSURE, LIDARJITTER: 각 199개 (가장 적음)
학습 권장사항
RGB 이미지 정규화/균등화
극단적 조건(UNDER/OVEREXPOSURE)에 높은 가중치
조건별 샘플 수 불균형 해소
LiDAR 데이터 스케일링 고려