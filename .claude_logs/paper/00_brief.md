# 논문 집필 브리프 — 1차 투고 IEEE Access (2026-10-27 작성, 2026-10-29 전면 정정: 저널·범위·기여 문장, method "MMSAM | 생각정리")

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
- **LaTeX 작업 위치(user 결정 2026-10-28로 변경)**: repo `_paper/ieee-access-2026/` 가 원고 원본이며 git 으로 버전을 관리한다(구조·빌드는 그 폴더 `README.md`). 볼트 `reports/manuscript/ieee-sensors-2026/`는 2026-10-27~28 작업본이며 더 갱신하지 않는다(`MOVED.md` 안내). 사용자에게 보내는 PDF 는 `bash build.sh snapshot` 으로 `drafts/` 에 만들어 커밋한다. 이전 규칙("repo 에는 LaTeX 를 두지 않는다")은 폐기한다. NAS 경로에 git 저장소를 둘 수 없으므로(STORAGE.md) 버전 관리는 repo 에서만 가능하다.
- 템플릿은 IEEE 공식 사이트에서 받은 현재 IEEEtran 을 쓴다(스킬 venue-templates 의 통화 규칙: 출처 URL·확인 날짜를 이 문서에 적는다).
- 수치를 옮길 때 출처 파일·행을 각 표 아래 주석으로 남긴다.

## 작업 위치·템플릿 기록(2026-10-27, specialist-paper)
- 원고 위치(2026-10-28 이전, 지금은 repo `_paper/ieee-access-2026/`): `/nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-sensors-2026/`(`latex/root.tex` + `sections/0_abstract … 7_conclusion.tex`, 메모 `notes/`). `_paper_submission/`은 볼트 `_archive/`로 옮겨져 있었고, RA-L 초안 v1 은 `reports/manuscript/latex/`에 있다(`references.bib`·`IEEEabrv.bib`를 여기서 복사).
- **템플릿(정정 2026-10-27)**: Sensors Journal 은 전용 템플릿 `IEEE-JSEN-LaTeX-template-202405`(`ieeecolor.cls` md5 `a08917c6a409f0693d9ac3f1833054b8` + `jsen.sty` md5 `f631e3cbadef91d3bbac8bb0024e0448`)를 쓴다. user 가 IEEE Template Selector(`https://template-selector.ieee.org`)에서 받아 줬다(`~/Downloads/sensors_templete.zip` md5 `bc250c1f7ffad4e852feedcea5090af6`, 받은 날 2026-10-27). 원고 `latex/`는 이 템플릿으로 옮겼고, 원본은 `ieee-sensors-2026/templates/jsen-202405/`에 둔다.
- IEEE Access 재투고용 템플릿: `ACCESS_latex_template_20260513`(`ieeeaccess.cls` md5 `6e473d2a8652127c24e6e8af3d7a34f5`, zip md5 `14c40fa0696749bcbd51e010b9f40151`, user 제공 2026-10-27) → `templates/access-20260513/`.
- 처음 쓴 CTAN `IEEEtran.cls` V1.8b(md5 `5b2e4fa1…`)는 쓰지 않는다(`templates/ctan-ieeetran-1.8b/`에 보관).
- JSEN 템플릿이 요구하는 것: **그래픽 초록**(초록 상자 오른쪽 3in 그림, 원고에 자리 있음), 초록 한 문단 150~250 단어(약어·참고문헌·수식·표 금지), 키워드 알파벳 순, 저자 약력(사진 포함, 템플릿 끝 `IEEEbiography`).
- 이 머신 TeX 에는 `algorithmic.sty`가 없다(템플릿 예제가 부르지만 원고에는 필요 없음). 러닝헤드가 대문자로 바뀌므로 `\textcolor{red}`가 `RED`가 되는 문제를 `root.tex`에서 색 별칭으로 막았다.
- 빌드: `latexmk -pdf root.tex`. 무에러 확인 2026-10-27(3쪽, 표 7개·그림 5개 자리). 참고문헌은 본문에 `\cite`가 생기면 `root.tex` 끝 주석을 푼다.

## 결정 추가(user 2026-10-28)
- **공유 LoRA 교체 조건부 결정**: ablation L1(센서별 LoRA → 완전공유 r16)이 clean test 58.84 로 A2 58.61 을 넘었음(DRN-261020-09). analysis 의 L1 강건성 측정(inbox 20261028-162006)이 A2 이상(test EMM15 Δ ≥ 0 그리고 val RMM 비악화)이면 **논문 모델을 공유 LoRA 판(A2-shared)** 으로 바꾼다: A2-shared 시드 902·903 추가 학습, MUSES 4모달 A2 s2 는 공유판으로 재시작, MCubeS A2 는 처음부터 공유판. 미달이면 공유 LoRA 는 ablation 행으로만 둔다.
- **토큰 적응 LoRA 제어**는 P56-C 기각이 신호원(블록 6)에 한정된 것이므로, 고해상 특징 신호 프로브 DRN-261028-01 통과 시에만 1시드 스크린을 사전등록한다. 이번 투고의 노벨티 줄에는 프로브·스크린 결과가 나오기 전까지 넣지 않는다.
- **분량 10쪽(user 2026-10-28)**: 본문 8쪽 상한을 넘겨 쪽당 초과 비용을 내기로 했다(10쪽이면 2쪽 × $175 = $350, 저자 안내 문구 기준이며 현재 금액은 투고 전 공식 페이지에서 재확인). 목표 = 참고문헌 포함 10쪽 이내, 그림·표를 촘촘히.
- **본문 그림·표 배치(specialist-paper 2026-10-28, 상세 = 볼트 `ieee-sensors-2026/notes/figure_guide.md` §2)**: 그림 7(개요 · 강건성 그래프 · 센서 고장 정성 · 악조건 정성 DELIVER+MUSES · 조건별 의존도 그래프 · 얇은 객체 클래스별 이득 · 실패 사례) + 그래픽 초록, 표 7(DELIVER · MUSES · MCubeS · 결측/노이즈 · **조건별 mIoU(신설)** · ablation · 비용). 조건별 의존도 표는 그래프로 대체하고 보충자료 표 S1 로 옮겼다(`sections/supp_tables.tex`). 데이터 그래프는 `figures/src/make_data_figs.py` 가 정본 수치로 다시 그린다.
- 측정 요청: analysis inbox `20261027-182054`(정성 장면 선별) · `20261028-183826`(조건별 표, 센서 하나만 남은 A2 값, ablation EMM, 의존도 split 통일).

## 결정 추가(user 2026-10-29) — 목표 저널 변경
- **목표 저널 = IEEE Access**(IEEE Sensors Journal 에서 변경). 사유(user): 리뷰가 빠르고 채택 가능성이 높을 것, SCI급. 같은 벤치(DELIVER·MUSES)의 직접 비교 논문 MM SAM-adapter 가 IEEE Access 2025 게재(DOI 10.1109/ACCESS.2025.3609640, CrossRef 확인)라 선례가 있다. IEEE Access 의 심사 기간·채택률·게재료는 2차 출처 수치만 확인했으므로(첫 결정 약 4주, 채택률 27~45%, 게재료 약 $1,850~2,160) 투고 전 공식 페이지에서 확인한다.
- 원고 폴더: `_paper/ieee-access-2026/`(옛 이름 `ieee-sensors-2026`, git mv). 템플릿: `ACCESS_latex_template_20260513`(`ieeeaccess.cls` md5 `6e473d2a8652127c24e6e8af3d7a34f5`). 그래픽 초록은 필요 없고 저자 약력(사진)은 필요하다. Access 클래스는 TikZ 와 충돌하므로 Fig.1 은 별도 PDF 로 넣는다(`build.sh`).
- 분량: 10쪽 목표를 유지한다(Access 쪽수 규정은 미확인; JSEN 의 쪽당 초과 비용 규정은 더 적용되지 않는다).
- **후속 RA-L 원고(조건부)**: 토큰(질의) 적응 LoRA 구조가 사전등록 게이트(고해상 특징 신호 프로브 DRN-261028-01 → 1시드 스크린 → 3시드 확정)를 통과하면, 그 구조를 방법 기여로 하는 별도 원고를 RA-L 에 낸다. 조건: Access 원고와 주장·핵심 실험이 겹치지 않을 것(Access 원고는 인용), Access 원고에는 토큰 적응 LoRA 를 넣지 않을 것. RA-L 은 6쪽 + 유료 2쪽 = 최대 8쪽, 최종 결정까지 평균 4개월(RA-L 공식 FAQ).

## 출처 규칙(user 2026-10-29)
- **표 수치**: 모든 표 수치의 출처(파일·줄·행 제목)는 `notes/table_sources.md` 에 적는다. 표를 고치면 같은 커밋에서 이 문서를 고치고 `python figures/src/check_table_sources.py` 로 빠진 숫자가 없는지 확인한다.
- **그림**: 모든 그림은 만든 코드와 입력이 이 폴더에 있어야 한다. 데이터 그래프의 수치는 `figures/data/*.csv`(source_file·source_locator 열)에 두고 `figures/src/make_data_figs.py` 가 읽어서 그린다(스크립트에 수치를 적지 않는다). 그림별 코드·입력·방법 목록은 `notes/figure_sources.md`.

## 결정 추가(user 2026-10-29) — 원고 위치
- **원고 원본 = repo `_paper/ieee-access-2026/`**(`.claude_logs/paper/ieee-access-2026/` 에서 git mv). 이 브리프와 작업 기록은 `.claude_logs/` 에 남는다. `_paper/` 바로 아래의 CVPR 원고(2026-03, `main.tex`·`rebuttal.*`, git 미추적)는 별개이며 건드리지 않는다.
- repo `.gitignore` 가 `data/`·`*.json`·`*.png` 를 무시하므로, 그래프 수치는 `figures/source_data/` 에 두고 원고 폴더 `.gitignore` 에서 `*.json`·`*.png` 를 다시 포함한다. (2026-10-29 이전 커밋 ad30385 에는 그래프 CSV 3개와 사이드카 JSON 3개가 이 무시 규칙 때문에 빠져 있었고, 이번 이동 커밋에서 포함했다.)
- (2026-10-29) 토큰 적응 LoRA 축 종결: 프로브 DRN-261028-01 미통과(고해상 통계가 블록 6 보다 낫지 않음). 이번 투고·다음 논문 모두에서 이 축을 쓰지 않는다. 노벨티 줄은 발견 + 학습 설계(+공유 LoRA 조건부) + 단일 모델 3벤치로 고정.

## 논문 분리 결정(user 2026-10-29)
- **1차 투고(IEEE Sensors Journal, 기각 시 IEEE Access)**: 논문 모델 = A2(L1 강건성 통과 시 공유 LoRA 판). 구조 노벨티 주장 없음 — 발견 + 학습 설계 + 단일 모델 3벤치 + 분석. 토큰 적응 LoRA·S2 등 구조 보완은 **쓰지 않는다**.
- **2차 논문(RA-L 목표)**: 1차 모델 위에 구조 보완(내용 기반 토큰 MoE-LoRA DRN 등록, S2 센서별 마스크 헤드 등)을 더해 별도 투고. 1차 투고 뒤 착수, GPU 는 1차 메인 표가 우선.

## 🔴 2026-10-29 정정(정본은 이 절, 위 절들과 어긋나면 이 절이 우선)
- **저널**: 1차 투고 = **IEEE Access**(user 결정 10-29, paper 세션). Sensors Journal 은 폐기. 분량 **10쪽**(user 10-28 결정). 원고 위치 = repo `_paper/ieee-access-2026/`(git 추적, user 10-29), 빌드 `build/`, 스냅샷 `drafts/`. 2차 논문 = RA-L(구조 보완: MoE-LoRA·S2).
- **논문 범위(user 지적 10-29, discussion 세션)**: 목표 = 단일 아키텍처 A2 로 DELIVER·MUSES·MCubeS 전 모달 융합 계열 1위(clean 정확도). 결측 모달 강건성은 "그쪽 벤치에서도 통한다" 를 보이는 **보조 축**. 초록·기여 1 을 결측 중심으로 쓰지 않는다.
- **기여 문장(갱신)**: ① 제안 방법 = 두 패스 학습(clean 패스 + 센서별 합성 열화 패스 + 동결 clean 교사 로짓 증류) — A2 의 clean 이득 최대 기여(s821: +1.8, 고해상 경로 +0.9). 주장 형태 = "이 학습은 우리 구조(동결 DINOv3 + LoRA)와 결합할 때 clean 을 올린다"(구조 의존성 실험 DRN-261029-02·03 둘 다 채택 시 "우리 구조에서만"), 선행 RobustSeg basic distillation 과의 차이는 L7(DRN-261029-01)로 보임. ② 단일 모델 3벤치 전 모달 1위 + 결측·열화 강건성 보조. ③ 분석(센서 분업·조건별·해상도·얇은 객체·실패 사례·비용). 위 "주장하지 않는 것: Q2 가 새 방법" 줄은 **철회**: 두 패스 학습을 제안 방법으로 쓰되, L7·구조 의존성 결과 전까지 "선행과의 차이" 문장은 보류.
- **표 결정**: T1 은 전 모달만(조합별 행 없음, 조합은 T4 결측으로 대신). T4(b) "DGFusion (official)" 행은 실제로 재현본 (a) 80k 이므로 **"DGFusion (reproduced)" 와 "DGFusion (official weights, 같은 프로토콜 clean 56.73·EMM 31.14, ledger:130)" 두 행 병기**. 같은 프로토콜 격차 = +1.01(발표치 기준 +1.03 병기).
- **모델 서술**: A2 config 에 MODEL.ROUTER(P36 클래스별 고정 모달 라우터, 0 초기화 잔차)가 켜져 있음 → 추론 시 off Δ 를 analysis 가 측정해 |Δ| < 0.1 이면 "이전 판 잔재, 효과 없음(보충 표)" 로 적고, 크면 ablation 행 추가. 보조 손실 3종(fusion 보조 CE 0.5·C3 prototype 0.1·VICReg)은 보충자료로. NM 정의는 analysis 확인 후 확정(현재 본문: S&P 단독). 모델 이름 = ReliaDINO 유지.
- **T1 A2 val**: DRN-261019-03 의 mm_eval_v2 clean val(legal v2 도구) 3시드 평균 70.17 을 "legal v2 val" 로 인용 가능(같은 채점기). T4 Q2 depth 부분 결측 두 출처 불일치(ledger 10-01 3시드 vs DRN-261019-03 2시드)는 analysis 정합 확인 전까지 열 제외. T5 센서 제거는 전 행 test 로 통일(analysis 재측정 요청). ablation 행에 EMM 열 추가(analysis).
