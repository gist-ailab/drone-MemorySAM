# 원고 남은 todo 전수조사 + 계획 (2026-11-09 저녁, develop 1908f5c)

오늘 채운 것: Table 3 MUSES test 80.01 · Table 5 Ours · Table 6 조건별 · Table 10 재구성 · Fig.2(a)(b) · Fig.3 센서 고장 정성 · Fig.5 MUSES 정성 · Fig.6 test 재측정 · Fig.8 TL 오검출 · 부록 B(Table 11) · 초록·서론·결론 수치.

## A. 숫자 칸 (`\td`) — 남은 것
| # | 위치 | 필요한 측정 | 담당 | 가능 시점 |
|---|---|---|---|---|
| A1 | Table 7 L5 행(열화 패스 제거) clean·EMM | L5 s821 legal v2 test + EMM15 (jarvis 체인 정지 11-05) | manager→analysis | 학습 0, 약 1.5시간 |
| A2 | Table 7 교차주의 제거 행 EMM | L2 s821 EMM15 test | analysis | 학습 0, 약 1시간 |
| A3 | Table 8 (a) 첫 행(clean 학습, 고해상 없음) | E1 test 3시드(현재 s821 55.94·s902 56.86 있음, s903 은 yeon) + val·EMM·RMM | analysis | 학습 0, 반나절 |
| A4 | 비용표 DINOv3-B 행 + 본문 todo | A2-B DELIVER(DRN-261105-01) legal v2 + 비용 측정 | analysis | 학습 0, 2시간 |
| A5 | Table 4 MCubeS DINOv3-B 행(FuseAdapt 동급 백본) | C2D2-B(DRN-261102-02) 진행 확인 | manager | 미확인 |

## B. 판정에 걸린 서술
| # | 위치 | 필요 | 담당 |
|---|---|---|---|
| B1 | 서론·관련연구·방법 "각 부분의 기여"(3곳) | L5·L6·L7 판정(FU-03) — A1 뒤 | method |
| B2 | 방법 §IV-C 라우터 역할 | L8 판정(s821 56.48 / s902 57.88, 시드 간 반대) + 라우터 추론 끄기(DRN-261029-06) | method |
| B3 | 6절·결론 구조 의존성(FU-04) | DGFusion·CAFuser 에 C2D2(DRN-261029-02·03) — 미착수, 동결 전 불가 | **user 결정**: 문장 축소(future work) 권장 |
| B4 | 6절 고해상 특징 크기 그림 | DRN-261104-02 판정 | method → paper 그림 조립 |
| B5 | 6절 TL·Pole recall 확인(2곳) | A2 3시드 P/R 재확인 | analysis |
| B6 | 6절 재학습 DGFusion depth −4.7 문장의 "벤치 코드 재측정" todo | Table 5 (b) 삭제로 재측정 계획 없음 | paper: 문장을 "test, our evaluation code" 로 확정(재측정 todo 삭제) |

## C. paper 세션이 바로 할 수 있는 것
| # | 위치 | 재료 |
|---|---|---|
| C1 | Fig.4·Fig.9 오류 영역 서술(2곳) | 그림 판독 |
| C2 | 부록 MUSES·MCubeS 장면, DELIVER 결측 장면 | MUSES 은행(3모델) · deliver_s1~s4 은행 · mcubes_rescore 덤프 |
| C3 | Fig.7 얇은 물체 확대 패널(선택) | 은행 Q2s821·A2s821 |
| C4 | Table 5 각주 "가장 낮은 시드도 모든 열에서 벤치 최고 초과" | analysis 검산 20261109-161420 |
| C5 | Fig.1 최종 다시 그리기 | notes/figure_guide.md |

## D. user 정보·결정
저자·소속·이메일·IEEE 회원·연구비·교신저자·약력·사진(root.tex 8곳), 코드·체크포인트 공개 문장(서론), B3.

## E. 진행 중
MUSES 2센서 보충자료(DRN-261109-01·02, yeon): 학생 완료 11-10 15시 / 11-11 오전 → test zip → user 제출 → 보충자료 표.

## F. 그림·표 배치와 논문 흐름 (점검 결과)
현재 순서: I 서론 → II 관련연구 → **III 센서·벤치마크·평가 프로토콜(Table 1 벤치마크)** → IV 방법(Fig.1) → V 실험(T2~T9, F2~F5) → VI 분석(F6~F9, T10) → VII 결론 → 부록.
문제:
1. 첫 표가 벤치마크 소개(Table 1, 3쪽)이고 개요도(Fig.1)가 그 뒤(4쪽)에 온다.
2. 방법보다 앞에 벤치마크·평가 프로토콜 절(III)이 있다 — 보통은 실험 절 첫머리.
3. 부유물 표류: Table 3 이 Table 2 보다 먼저 배치(6쪽), Fig.2 가 Table 5·7·8 뒤(7쪽), Fig.3·4·5 가 분석 절 시작(8쪽) 뒤 8~10쪽, Fig.9 가 결론 머리 뒤(13쪽), 14쪽 빈 쪽.
제안:
- Fig.1 을 첫 부유물로(2쪽 상단).
- III 절 해체: "센서와 고장 양상"은 방법 첫 절(문제 설정)로, "벤치마크·평가 프로토콜"은 실험 V-A 로 이동. Table 1 은 실험 절의 작은 표로 옮기거나 부록으로.
- 실험 절 부유물 순서를 본문 참조 순서로: V-B 비교(T DELIVER→MUSES→MCubeS) → V-C 강건성(T5, Fig.2, Fig.3) → V-D 정성(Fig.4, Fig.5) → V-E 조건별(T6) → V-F ablation(T7, T8) → V-G 비용(T9).
- 절 끝마다 `\FloatBarrier`(placeins)로 부유물이 다음 절로 넘어가지 않게 하고, 빈 쪽 제거.
