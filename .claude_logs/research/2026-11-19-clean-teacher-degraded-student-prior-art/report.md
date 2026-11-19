# 손상 입력 + clean 교사 증류 구조의 선행연구 조사 (2026-11-19)

작성: method(MMSAM | 생각정리). 계기: 2026-11-18 미팅에서 교수님 피드백 "이런 게 기존에 없었나, 본 것 같다". 조사 = 서브에이전트 4갈래(원시 노트 `notes/`, 영어, 항목마다 검증 수준 표기). 이 문서는 그 요약이며 **method 가 원문을 재확인하지 않았다**(불확실 등급). 논문에 인용하기 전 원문 표·식 번호를 다시 본다.

## 결론
- "입력을 손상(센서 제거·열화)시키고 clean 입력을 보는 교사에서 증류한다"는 **우리가 처음이 아니다.** 단일 모달에서는 2016년부터, 멀티센서 세그멘테이션에서는 이미 기준선이다.
- 여섯 축(A clean 교사 증류 · B 원본+손상 두 패스 · C 멀티센서 · D 세그멘테이션 · E 센서별 합성 열화 · F 동결 기반 모델+어댑터)을 모두 채운 단일 논문은 찾지 못했다. 비어 보이는 것은 "E 를 A+C+D 에 넣은 것"과 "여섯 축 결합"뿐이고, 이 판단의 신뢰도는 중간 이하다(인용 그래프 미순회).
- 비어 보이는 E 와 A 는 우리 실험에서 효과가 확인되지 않았다(L7 물리 열화 reject, 증류 5축 DRN-261111-02 reject). 선행에 비어 있다는 것과 기여가 있다는 것은 별개다.

## 가장 가까운 선행 (A~F 일치)
| 순위 | 논문 | id · 게재 | A | B | C | D | E | F | 남는 차이 | 확인 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FuseAdapt | arXiv 2610.21779 · NeurIPS 2026 | 부분 | 부분 | ✓ | ✓ | ✗ | ✓ | 교사 = 어댑터 EMA 사본, 주 손실 = 가린 모달 잠재 예측, 센서 제거만. 로짓 KL("Output KD")은 대조군 행(MCubeS 49.84 대 52.89) | 본문 |
| 2 | RMMSS(v1 제목에 RobustSeg) | arXiv 2505.12861 · v2 에 AAAI 2026 표기 | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | 기본 틀이 CE + λ·KL(식 3, 표 4 "Basic" EMM 43.45/RMM 51.69/Full 57.50). 관련연구에 "modality masking and noisy inputs" 를 기존 관행으로 서술 | 본문 |
| 3 | AnySeg | arXiv 2411.17141 | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | 동결 교사 + 모달 무작위 제거 학생 + 예측·특징 KL(식 6·8) | 본문 |
| 4 | CRM | arXiv 2303.17386 · ICRA 2024 | 부분 | ✓ | ✓ | ✓ | 부분 | ✗ | 같은 네트워크 자기 증류(별도 교사 없음), 패치 마스킹, 로짓 L1 | 본문 |
| 5 | RobustSAM | arXiv 2406.09627 · CVPR 2024 | ✓ | 부분 | ✗ | ✓ | ✓ | ✓ | 단일 카메라, 합성 손상 15종, 특징·토큰 일관성 | 본문 |
| 6 | md4all | arXiv 2308.09711 · ICCV 2023 | ✓ | ✗ | ✗ | 깊이 | 부분 | ✗ | 동결 주간 교사 → 야간·우천 변환 입력 학생 | 본문 |
| 7 | Robo3D 밀도 불변 학습 | arXiv 2303.17597 · ICCV 2023 | 부분 | ✓ | ✗ | LiDAR 분할 | 부분 | ✗ | 교사 동결 아님, 점 마스킹 | 본문 |
| 8 | M3L | arXiv 2304.10756 | 부분 | 미확인 | ✓ | ✓ | ✗ | ✗ | EMA 전 모달 교사 → 모달 마스킹 학생, pseudo-label | 본문 |
| 9 | MMANet | arXiv 2304.08028 · CVPR 2023 | 부분 | ✗ | ✓ | 부분 | ✗ | ✗ | 관계 증류 | 본문 |
| 10 | Compass | arXiv 2608.03559 · ACM MM 2026 | 부분 | 부분 | ✓ | ✓ | 부분 | ✗ | "degradation simulation stream" + 상호 증류, RGB-depth 균열 분할 | 초록 |
| 11 | VG-SAF | arXiv 2608.24366 | 부분 | ✓ | ✓ | ✗(주행) | ✓ | ✗ | 물리 기반 고장 증강 noisy 분기를 stop-grad clean 분기로 끌어당김 | 일부 |

## 단일 모달 계보 (구조의 연식)
- Su & Maji, "Adapting Models to Signal Degradation using Distillation"(arXiv 1604.00433, Cross Quality Distillation): `L1(g(T(x)), y) + λ·L2(g(T(x)), f(x))` — 지도 손실 + clean 교사 출력 증류, 교사 선학습 후 고정. 우리 손실과 같은 형태. (본문)
- ARD(arXiv 1905.09747): `α·t²·KL(S(x+δ), T(x)) + (1−α)·CE` — 교사 clean, 학생 적대적 입력. (본문)
- 조밀 예측: Sakaridis 외(1708.07819, 맑은 날 모델 예측으로 합성 안개 지도), md4all, MIC(2212.01322).
- 저해상 얼굴 인식 계열: Ge 외(1811.09998), **A-SKD(arXiv 2209.14498, ECCV 2022)**. 검출: **CD-FKD(arXiv 2603.16439, 2026; 고정 교사 원본 · 학생 ImageNet-C 계열 손상 15종)**. 🔴 조사 에이전트는 A-SKD·CD-FKD 저자 소속이 GIST 이규빈 교수 그룹이라고 보고했다(method 미확인) — 교수님이 떠올린 것이 이쪽일 수 있다.
- 2026 에도 과제만 바꿔 계속 나온다: 2604.22529(동결 DINO 교사, "asymmetric KD"), DARA 2607.16644("original-to-corrupted distillation"). 통일된 이름 없음.
- 교사 없는 계열(AugMix JSD, mean teacher, Noisy Student)은 학습 중인 모델이 목표를 만든다는 점에서 구분.

## 기존 감사 문서(2026-10-20) 정정
- "RobustSeg(CVPR 2026)"는 RMMSS(2505.12861) v1 제목의 이름이다. 같은 저자의 CVPR 2026 포스터가 다른 제목으로 따로 있으며 본문은 미확인.
- FuseAdapt 는 "동결 교사 + 로짓 증류"가 아니라 "EMA 교사 + 잠재 표현 예측". 로짓 KL 은 대조군.
- MAGIC·MM SAM-adapter 에는 교사·증류가 없다. GeomPrompt = arXiv 2604.11585(CVPR 2026 워크숍), UniMRSeg = arXiv 2509.16170(NeurIPS 2025).

## 방어 가능한 주장 범위
- 불가: "열화 + clean 교사 증류를 제안한다", "강건 멀티센서 세그멘테이션의 첫 자기 증류", "교사를 둔 첫 동결 VFM + LoRA 멀티센서 모델".
- 문헌상 가능: "센서별 열화 모사와 clean 교사 로짓 증류를 동결 기반 모델 위 멀티센서 세그멘테이션에 결합(우리가 아는 한 처음)". 단 우리 ablation 이 열화·증류의 효과를 받쳐 주지 못하므로 기여로 세우지 않는 것을 권고.
- 예상 공격: "RMMSS Basic + RobustSAM식 열화 증강 + FuseAdapt식 백본의 조합". 필요한 대조 행: (a) 센서 제거만 쓴 CE+KL(= L7, 있음) (b) 교사 없는 열화 학습(= L6, 있음; 단일 패스 L9 진행) (c) FuseAdapt(같은 벤치 수치 인용).
- 관련연구 절에 반드시 인용: Su & Maji 2016, ARD, AnySeg, RMMSS, FuseAdapt, CRM, M3L, MMANet, RobustSAM, md4all, A-SKD·CD-FKD.

## 조사 한계
- 갈래당 도구 호출 25~44회, Semantic Scholar 인용 그래프 미순회. MLE-SAM·MAGIC++·DGFusion·CAFuser 학습 절, CVPR 2026 RobustSeg 본문, SDSTrack·MetaBEV·RoboFusion·UniBEV 미확인. 학회명 일부는 기억에 의존(노트에 표시).

## 추가(같은 날): 상세판
- `full-report.md` = 보고서 작성 에이전트가 노트 4개를 종합한 상세판(16편 일치표, 주장마다 검증 수준·링크). 이 요약과 결론은 같다.
- 상세판이 더한 것: PFS(arXiv 2603.05623, 센서별 독립 손상 + 커리큘럼, 증류 없음, 요약만 확인) · 공백을 깨뜨릴 최대 위험 = MLE-SAM·FPT 가 학습 때 잡음을 주입하는지 미확인(센서 고장 벤치 표 1 에 NM 대응 방법으로 표시) · SDSTrack(RMMSS 가 "noisy inputs" 근거로 인용) 미확인.
- "to our knowledge, the first …" 문구를 쓰려면 MLE-SAM·FPT·SDSTrack·Compass·CVPR 2026 RobustSeg 본문과 인용 그래프 확인이 선행돼야 한다.
