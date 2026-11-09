# 원고 빈 칸 전수조사 (2026-11-09 오후, develop d8e7228 이후)

원고에서 `\todo`·`\td`·`\phbox` 를 전부 뽑아 lab-plan 항목·서버 상태와 대조했다. 서버 상태(lab-status 13:11): yeon·hpca100 연결 안 됨, 빈 GPU = levine 2·4·5·6 · bengio 1·3·4·6·7 · lecun 2 · drone-demo 0 · amphibious-demo 0.

판정: ✅ 오늘 채움 · 🟢 재료 있음(이 세션이 바로 채움 가능) · 🟡 측정만 남음(학습 0, 1일 안) · 🔴 학습·구현 필요(11-10 동결 전 불가) · ⚪ user 결정

## A. 숫자 칸 (표·그림)

| # | 원고 칸 | 필요한 것 | lab-plan 항목 | 상태 | 판정 |
|---|---|---|---|---|---|
| 1 | Table 3 MUSES Ours **test** | 4모달 A2 test 추론 → zip → user 공식 서버 제출 | DRN-261029-09 queued | zip 미생성. ckpt s2·s3 는 yeon(연결 안 됨) — NAS 백업 여부 미확인 | 🟡 zip(manager) → ⚪ 제출(user) |
| 2 | Table 5 (b) DGFusion 공개 | 벤치 공개 코드에 detectron2 모델 어댑터 | **없음** | 미측정(monitoring 11-09 "어댑터 필요") | 🟡 어댑터 0.5일 + 측정 |
| 3 | Table 5 (b) DGFusion·CAFuser 재학습 | 위 어댑터 + 재학습 가중치 | **없음** | CAFuser 재학습 ckpt NAS `cafuser_b_resume_20261003/` 있음, DGFusion 재학습 ckpt 는 서버(접근 불가 기록 10-30) | 🟡/🔴 |
| 4 | 서론 "released DGFusion weights lose …" 값 | #2 와 같은 측정 | 없음 | — | #2 에 종속 |
| 5 | Table 6 조건별 4행×6조건 | 조건별 test mIoU | **없음**(inbox 20261028-183826 item 1 미착수) | 정성 은행에 A2 s821·s902·DGFusion 공개·CAFuser 재학습의 이미지별 혼동행렬이 있음 → 조건별 합산 가능. A2 s903·DGFusion 재학습은 은행에 없음 | 🟢 3행(우리는 2시드) · 🟡 나머지 |
| 6 | Table 7 L5 행(열화 패스 제거) | L5 s821 legal v2 test + EMM15 | DRN-261020-13 evaluating | ckpt 는 jarvis 전송본 있음(md5 일치), 평가 체인이 `&&` 구문 오류로 정지(11-05) | 🟡 약 1.5시간 |
| 7 | Table 7 교차주의 제거 행 EMM | L2 s821 EMM15 | **없음** | ckpt 위치 미확인 | 🟡 |
| 8 | Table 8 (a) 첫 행(clean 학습·고해상 없음) | E1 40ep 3시드 test·val·EMM·RMM | **없음**(inbox 20261102-180010·20261104-163300 미착수) | s821·s902 ckpt NAS 있음(오늘 벤치 NM 측정에 사용), s903 은 yeon 에만 | 🟡 2시드 |
| 9 | ~~Fig.2(b) Ours 막대~~ | A2 단일 센서 3시드 | — | **✅ 11-09 채움**(기존 test EMM15 emm.csv 의 단일 센서 행) | ✅ |
| 10 | Fig.3 센서 고장 정성 그림 | CMNeXt·CAFuser·DGFusion·우리의 결측/노이즈 입력 예측 덤프 4설정 | **없음**(20261027-182054 는 옛 열 구성) | 미착수 | 🟡 1일(기준선 3종 × 4설정) |
| 11 | Fig.5 MUSES 정성 그림 | 기준선 MUSES val 덤프(우리 s2 는 있음) | **없음** | DGFusion MUSES 공개 가중치 NAS `baseline_dgfusion_muses_official/` 있음, CAFuser MUSES 는 미확인, CMNeXt 는 MUSES 공개 가중치 없음 | 🟡 / ⚪ 열 구성 |
| 12 | Fig.6 depth 의존도 | 최종 모델·clean 모델 test 조건별 depth 제거 | **없음**(inbox 20261104-163300 미착수) | A2 test EMM15 의 `present=img+event+lidar` 행은 전체 평균만 있고 조건별 없음 | 🟡 |
| 13 | Fig.8 TL 오검출 그림 | NAS `a2_tl_fp_scenes_20261022` 패널 조립 | 해당 없음(paper 작업) | 재료 있음 | 🟢 |
| 14 | 비용표 DINOv3-B 행 | A2-B DELIVER legal v2 + 비용 측정 | DRN-261105-01 evaluating | 학습기 54.63(legal 전) | 🟡 |
| 15 | Table 4 MCubeS DINOv3-B 행(FuseAdapt 동급 백본) | C2D2-B 2시드 | DRN-261102-02 running | 진행 상황 미확인(amphibious) | 🔴/🟡 |
| 16 | 부록 MUSES·MCubeS·결측 장면 | #10·#11 재료 | 없음 | — | #10·#11 에 종속 |
| 17 | Fig.7 확대 패널 | Q2 덤프(yeon) | 없음 | yeon 연결 안 됨 | ⚪ 선택 |

## B. 서술 칸 (실험 판정에 걸린 문장)

| # | 원고 위치 | 걸린 것 | 상태 | 판정 |
|---|---|---|---|---|
| 18 | 4절 라우터 역할, 서론 기여 | L8 판정(DRN-261029-07: s821 56.48 / s902 57.88, 시드 간 반대) + 라우터 추론 끄기 DRN-261029-06 queued | 판정 대기 | 🟡 06 은 학습 0 |
| 19 | 4절·서론·관련연구 "각 부분의 기여" | L5·L6·L7 판정(FU-03) | #6 측정 뒤 판정 | 🟡 |
| 20 | 6절·결론 "구조 의존성" | DRN-261029-02·03(DGFusion·CAFuser 에 두 패스 학습) queued, 미착수 | 재학습 2회 | 🔴 동결 전 불가 → 문장 축소 결정 필요 |
| 21 | 6절 TL·Pole recall 수치 확인 | analysis 확인 | 미착수 | 🟡 |
| 22 | 6절 장면 오류 영역 서술(Fig.4·9) | paper 가 그림을 읽고 서술 | 재료 있음 | 🟢 |
| 23 | 서론 코드·체크포인트 공개 문장 | user 결정 | — | ⚪ |
| 24 | Fig.1 최종 다시 그리기 | paper | — | 🟢 |

## C. 판정 대기로 예비값을 넣은 칸
MUSES val 81.54(DRN-261019-02 judging) · MCubeS 59.78(DRN-261017-02 judging) · MCubeS w/o C2D2 59.89(DRN-261104-03 evaluating, 재채점 요청 20261105-115201) · L6 EMM 49.64(pane 로그 값).

## D. 지금 돌 수 있는 것(학습 0)과 배치 제안
- jarvis 는 빈 GPU 없음, yeon·hpca100 은 연결 안 됨 → 평가는 bengio(1·3·4·6·7)·levine(2·4·5·6)·drone-demo·amphibious 에 배치.
- 순서 제안: #6 L5 → #7 L2 EMM → #8 E1 2시드 → #12 depth 재측정 → #14 A2-B legal → #5 나머지 → #2·#3 어댑터 → #10 덤프.
