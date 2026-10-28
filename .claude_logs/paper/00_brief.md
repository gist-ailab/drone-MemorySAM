# 논문 집필 브리프 — IEEE Sensors Journal 투고 (2026-10-27, method "MMSAM | 생각정리" 작성, user 결정)

## 결정 사항(user 2026-10-27)
- 목표 저널: **IEEE Sensors Journal**(1순위). 기각 시 **IEEE Access** 재투고(동시 투고 금지, 순차). 두 저널 모두 IEEE 2단 템플릿(IEEEtran 계열)이며 Access 는 자체 템플릿 변형을 쓴다 → 본문·그림은 그대로, 머리말·저자 약력 절만 바꾸면 된다(저자 약력은 Access 필수).
- 분량: Sensors J 는 8쪽(2단) 기준, 초과분은 쪽당 과금($175, 채택 시 부과; 미확인 출처: ieee-sensors.org for-authors 검색 결과 2026-10-27). **본문 8쪽 + 보충자료**로 설계한다.
- 논문 틀: **분석형 + 성능**. 구조 노벨티(S2)는 이번 투고에 넣지 않는다. 단 S2 가 초고 동결 전에 게이트를 통과하면 "방법" 절에 추가한다(병행 트랙, §6).
- 집필 세션: `MMSAM | paper`(역할 specialist-paper). 이 브리프가 전용 문서.

## 논문 모델
A2 하나. 정의·그림: artifact https://claude.ai/artifact/Am5i5hF7CuPQCRuJWibNwc (2026-10-25 판), 코드 `semseg/models/reliadino/`, config `configs/*screen40_A2hires.yaml`.
- 몸통 E1: 동결 DINOv3-L + 센서별 LoRA r8 + 4탭 + 교차 모달 attention(2층, 대칭, 게이트 없음) + SimpleFPN + dense 헤드 + Mask2Former-lite 질의 헤드 + arbiter.
- 학습 Q2: clean 패스 + 합성 열화 패스(센서별 p0.5, 결측 0.2, 커리큘럼) + 동결 E1 교사 로짓 증류(0.5, T2). 라벨·게이트 없음.
- 고해상 경로: DINOv3 공식 어댑터 결선, FPN 합산. 공학적 보완으로 서술(노벨티 아님).

## 기여 문장(초안)
1. 발견: 전 모달 정확도 1위 모델들의 센서 과의존은 구조·감독·백본이 아니라 clean 학습의 산물이다(같은 백본 통제 비교 + 기준선 2종 같은 도구).
2. 설계: 라벨·게이트 없는 두 패스 학습 + 교사 증류로 과의존을 2/3 줄이면서 clean 정확도를 올린다. 여기에 고해상 경로로 얇은 객체를 회복해 DELIVER 전 모달 1위.
3. 분석: 센서별 분업·조건별 의존도·결측/부분 열화/노이즈·평가 해상도·얇은 객체·실패 사례(신호등)·비용. 세 벤치(DELIVER·MUSES 4모달·MCubeS) 단일 모델.

## 절 구성(8쪽 목표)
1. Introduction (0.8쪽): 문제(센서 하나가 망가지면), 두 트랙(정확도 vs 결측 강건성)이 갈라져 있음, 기여 3개.
2. Related Work (0.6쪽): 전 모달 융합(CMNeXt·StitchFusion·CAFuser·DGFusion), 결측 강건성(MAGIC·AnySeg·RMMSS·RobustSeg·FuseAdapt), 동결 파운데이션 적응(MLE-SAM·MM SAM-adapter·FuseAdapt). 비교표는 `research/novelty-and-related-work.md`, 감사 `research/2026-10-20-structural-novelty-audit.md`.
3. Sensors and Benchmarks (0.5쪽): 각 센서의 물리·고장 양상, DELIVER/MUSES/MCubeS 구성, 평가 프로토콜(legal v2 원 축, 공식 채점 병기).
4. Method (1.5쪽): 구조 그림 1장(artifact 도식 재작성), 두 패스 학습, 고해상 경로.
5. Experiments (2.5쪽): Table 1 DELIVER(전 모달, 조합별) · Table 2 MUSES 4모달 · Table 3 MCubeS · Table 4 결측·열화(EMM/RMM/NM, **val** 로 벤치 계열과 맞춤, test 병기) · Table 5 센서 제거 Δ 3모델+우리 조건별 · Table 6 ablation(사다리 + 하나씩 빼기) · Table 7 비용.
6. Analysis (1.5쪽): 해상도 이득의 출처(서브패치), 얇은 객체 해부, 실패 사례(포스터→신호등, 그림), 한계.
7. Conclusion (0.3쪽).
보충자료: 조건별·클래스별 전체 표, 시드별 수치, 정성 그림 세트, 프로토콜 상세.

## 수치 정본과 인용 규칙
- 헤드라인·판정 수치는 `status/current.md`(PI·① 표)와 `experiments/judgment-ledger.md`, `experiments/headline.yaml` 에서만 가져온다. 다른 문서의 복제본·대화 수치는 쓰지 않는다.
- 3모달 MUSES 수치(79.788 등)는 쓰지 않는다(user 2026-09-29). MUSES 는 4모달 A2 만.
- 결측 벤치 비교는 계열을 섞지 않는다: 벤치마크 계열(2503.18445, 정규화 후 0 채움, val) vs 건너뜀 계열(AnySeg·RMMSS·RobustSeg 1a·FuseAdapt). 표에 계열·백본·split 을 병기.
- 기준선 수치는 원문 표·쪽을 각주로. 재채점(원 축) 값은 "공식 채점" 과 병기.
- 실험 약어(E1·Q2·A2·L1~L7)는 논문에서 쓰지 않는다. 구성요소 이름으로 쓴다.
- 주장하지 않는 것: 구조 노벨티, Q2 가 "새 방법", FuseAdapt 대비 절대 우위(백본 다름), 3벤치 1위(A2 로 다 채우기 전까지).

## 표별 데이터 출처·상태(2026-10-27)
| 표 | 있음 | 없음(의존) |
|---|---|---|
| T1 DELIVER | A2 2시드 57.85, E1 3시드, Q2, 기준선 공식+원 축 | A2 시드903 legal(DRN-261018-01) |
| T2 MUSES 4모달 | 교사 E1M4 학습 완료(학습기 val 82.17) | 공식 val, A2 2시드(DRN-261019-02), user 제출 |
| T3 MCubeS | P39.1 58.07(이전 세대, 참고) | A2 2시드(DRN-261017-02) |
| T4 강건성 | Q2 test EMM 48.68, A2 test EMM 49.12/48.57(jarvis robust_a2_20261019), RMM·NM | A2 **val** EMM/RMM(벤치 비교용) |
| T5 센서 제거 | DGFusion·CAFuser·E1·Q2 (judgment-ledger 09-19~30) | A2 분(robust 폴더 rmm 4종으로 대체 가능) |
| T6 ablation | RGB 단독 55.14, E1 55.94, Q2 57.73, A2 58.61(s821) | 단순 융합(DRN-261020-07 학습 중), L1(학습 완료·평가 대기), L2·L4(대기열), L5·L6(학습 중) |
| T7 비용 | 파라미터 수 | FLOPs·지연·메모리 측정(manager 에 요청 필요) |
| 그림 | 구조 도식(artifact), 실패 장면 4장(NAS analysis_logs/a2_tl_fp_scenes_20261022) | 조건별 정성 4~6장면, 결측 시나리오 1장 |

## 집필 순서(바뀌지 않을 것부터)
1. §3 센서·벤치·프로토콜 → 2. §4 방법 → 3. §6 분석(해상도·얇은 객체·실패) → 4. §1·§2 → 5. §5 표는 수치가 들어오는 대로 채움 → 6. 초고 동결 목표 **2026-11-10**.

## 세션 규칙
- 역할 specialist-paper: 집필·그림·표 조판만. 학습 기동·평가·판정은 하지 않는다(요청은 lab-inbox 로 manager/analysis/method).
- LaTeX 작업 위치: 볼트 `/nas_jm/Research/26_MultimodalSeg/` 아래 `_paper_submission/`(RA-L 초안 v1 이 있던 곳; 없으면 `reports/` 에서 찾아 새 폴더 `ieee-sensors-2026/` 생성). repo 에는 LaTeX 를 두지 않는다(리서치 콘텐츠 = 볼트).
- 템플릿은 IEEE 공식 사이트에서 받은 현재 IEEEtran 을 쓴다(스킬 venue-templates 의 통화 규칙: 출처 URL·확인 날짜를 이 문서에 적는다).
- 수치를 옮길 때 출처 파일·행을 각 표 아래 주석으로 남긴다.

## 작업 위치·템플릿 기록(2026-10-27, specialist-paper)
- 원고 위치: `/nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-sensors-2026/`(`latex/root.tex` + `sections/0_abstract … 7_conclusion.tex`, 메모 `notes/`). `_paper_submission/`은 볼트 `_archive/`로 옮겨져 있었고, RA-L 초안 v1 은 `reports/manuscript/latex/`에 있다(`references.bib`·`IEEEabrv.bib`를 여기서 복사).
- **템플릿(정정 2026-10-27)**: Sensors Journal 은 전용 템플릿 `IEEE-JSEN-LaTeX-template-202405`(`ieeecolor.cls` md5 `a08917c6a409f0693d9ac3f1833054b8` + `jsen.sty` md5 `f631e3cbadef91d3bbac8bb0024e0448`)를 쓴다. user 가 IEEE Template Selector(`https://template-selector.ieee.org`)에서 받아 줬다(`~/Downloads/sensors_templete.zip` md5 `bc250c1f7ffad4e852feedcea5090af6`, 받은 날 2026-10-27). 원고 `latex/`는 이 템플릿으로 옮겼고, 원본은 `ieee-sensors-2026/templates/jsen-202405/`에 둔다.
- IEEE Access 재투고용 템플릿: `ACCESS_latex_template_20260513`(`ieeeaccess.cls` md5 `6e473d2a8652127c24e6e8af3d7a34f5`, zip md5 `14c40fa0696749bcbd51e010b9f40151`, user 제공 2026-10-27) → `templates/access-20260513/`.
- 처음 쓴 CTAN `IEEEtran.cls` V1.8b(md5 `5b2e4fa1…`)는 쓰지 않는다(`templates/ctan-ieeetran-1.8b/`에 보관).
- JSEN 템플릿이 요구하는 것: **그래픽 초록**(초록 상자 오른쪽 3in 그림, 원고에 자리 있음), 초록 한 문단 150~250 단어(약어·참고문헌·수식·표 금지), 키워드 알파벳 순, 저자 약력(사진 포함, 템플릿 끝 `IEEEbiography`).
- 이 머신 TeX 에는 `algorithmic.sty`가 없다(템플릿 예제가 부르지만 원고에는 필요 없음). 러닝헤드가 대문자로 바뀌므로 `\textcolor{red}`가 `RED`가 되는 문제를 `root.tex`에서 색 별칭으로 막았다.
- 빌드: `latexmk -pdf root.tex`. 무에러 확인 2026-10-27(3쪽, 표 7개·그림 5개 자리). 참고문헌은 본문에 `\cite`가 생기면 `root.tex` 끝 주석을 푼다.

## 결정 추가(user 2026-10-28)
- **공유 LoRA 교체 조건부 결정**: ablation L1(센서별 LoRA → 완전공유 r16)이 clean test 58.84 로 A2 58.61 을 넘었음(DRN-261020-09). analysis 의 L1 강건성 측정(inbox 20261028-162006)이 A2 이상(test EMM15 Δ ≥ 0 그리고 val RMM 비악화)이면 **논문 모델을 공유 LoRA 판(A2-shared)** 으로 바꾼다: A2-shared 시드 902·903 추가 학습, MUSES 4모달 A2 s2 는 공유판으로 재시작, MCubeS A2 는 처음부터 공유판. 미달이면 공유 LoRA 는 ablation 행으로만 둔다.
- **토큰 적응 LoRA 제어**는 P56-C 기각이 신호원(블록 6)에 한정된 것이므로, 고해상 특징 신호 프로브 DRN-261028-01 통과 시에만 1시드 스크린을 사전등록한다. 이번 투고의 노벨티 줄에는 프로브·스크린 결과가 나오기 전까지 넣지 않는다.
