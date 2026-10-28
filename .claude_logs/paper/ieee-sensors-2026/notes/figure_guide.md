# 그림 안내서: IEEE Sensors Journal 원고 (2026-10-27, specialist-paper)

원고 위치: `../latex/` (빌드 `latexmk -pdf root.tex`). 디자인 출처: `carbon@11.84.0` (`~/labdesign/build/py/carbon_palette.py`).

## 1. 공통 규칙 (모든 그림)

| 항목 | 규칙 |
|---|---|
| 크기 | 단 폭 그림 = 8.9 cm(3.5 in), 두 단 그림 = 18.2 cm(7.16 in). 그린 크기 그대로 넣는다(축소 배율 1.0). |
| 글꼴 | 산세리프(Helvetica 계열) 한 가지. 본문 라벨 6.5 pt, 보조 라벨 5.6 pt, 패널 제목 6.8 pt 굵게, 패널 기호 (a)(b) 8 pt 굵게. 7 pt 미만은 인쇄 시 읽히는 한계이므로 보조 라벨에만 쓴다. |
| 여백 | 글자는 상자 테두리에서 1.5 mm 이상 떨어뜨린다. 글자가 선·화살표를 가로지르지 않는다. 상자끼리 간격 3 mm 이상. |
| 선 | 일반 0.45 pt 회색(#525252), 우리가 학습하는 부품 0.7 pt 파랑(#0F62FE), 동결 부품은 점선 0.45 pt + 옅은 회색 바탕(#F4F4F4). 화살촉은 작게(길이 1.6 mm). |
| 색 | 센서 4색 고정: RGB #6929C4(보라), Depth #1192E8(하늘), Event #005D5D(청록), LiDAR #9F1853(자주). 모든 그림에서 같은 센서는 같은 색. 분할 결과는 DELIVER 공식 팔레트(`semseg/datasets/deliver.py:28`). |
| 하지 않는 것 | 그라데이션, 그림자, 3D 효과, 아이콘 클립아트, 둥근 알약 모양 상자, 이모지, 무지개색 배치, 눈에 띄는 큰 둥근 모서리(반경은 0.5 mm 이하). 이런 요소가 "AI가 만든 그림" 인상을 준다. |
| 정성 그림 | 실제 덤프에서만 가져온다(경로·md5를 그림 원본 메모에 남긴다). 장면 고르는 규칙을 캡션에 쓴다(예: "DGFusion 대비 mIoU 차이 상위 장면 중 조건이 겹치지 않게"). 오류 화소는 같은 색 하나(빨강 #DA1E28)로 표시한다. |

## 2. 그림 목록과 상태

분량: **본문 10쪽**(user 2026-10-28, 8쪽 초과분은 쪽당 비용 지불). 2026-10-28 빌드에서 그림·표가 약 3.2쪽을 차지한다.

| 번호 | 무엇을 보여 주나(기여) | 형태 | 재료 | 상태 |
|---|---|---|---|---|
| GA | 그래픽 초록: 센서 하나가 망가져도 우리 모델은 유지된다 | 3 in 폭, 초록 상자 오른쪽 | Fig.1 축약 또는 Fig.3 한 장면(결정 대기) | todo |
| Fig.1 | 모델 개요(방법) | 두 단, (a) 추론 (b) 학습 | `latex/figures/fig_arch.tex`(TikZ 초안), 썸네일 `latex/figures/fig1_thumbs/` | **초안 완료**, 다시 그리기 대기. 논문 모델이 공유 LoRA 판으로 바뀌면 LoRA 칸을 네 센서 공통 한 색으로 바꾼다 |
| Fig.2 | 노이즈·결측에서 우리는 버티고 공식 기준선은 무너진다(기여 2) | 단 폭, (a) 노이즈 곡선 (b) 센서 하나만 남았을 때 막대 | `figures/src/make_data_figs.py`(정본 수치 내장) | **초안 완료**, (b) 우리 모델 값 측정 대기 |
| Fig.3 | 센서가 고장 났을 때 기준선은 무너지고 우리는 버틴다(핵심 정성 그림) | 두 단, 4행 × 6열 | 기준선·우리 모델의 결측·노이즈 조건 장면별 예측 | analysis 요청(장면별 예측 없음) |
| Fig.4 | 악조건(DELIVER 야간·안개·비·저노출, MUSES 실측 야간·눈·안개)에서 기준선이 놓치는 객체를 우리는 잡는다 | 두 단 | DELIVER 덤프 있음, MUSES 4모달 우리 모델은 학습 중 | analysis 요청(장면 선별) |
| Fig.5 | 조건별 센서 의존도: clean 학습 모델은 depth 에 크게 기대고 두 패스 학습은 이를 줄인다(기여 1) | 단 폭, 선 그래프 | `make_data_figs.py`; 전체 표 = 보충 표 S1 | **초안 완료**, split 통일·나머지 행 측정 대기 |
| Fig.6 | 얇은 객체 클래스별 고해상 경로 이득(시드 짝 비교) | 단 폭, 막대 + 시드 점 | `make_data_figs.py` + 클래스별 CSV | **초안 완료**, 확대 장면 추가 todo |
| Fig.7 | 한계: 포스터·간판을 신호등으로 오인 | 단 폭 | NAS `analysis_logs/a2_tl_fp_scenes_20261022/` 4장면 | 재료 있음 |

표: I DELIVER · II MUSES · III MCubeS · IV 결측/노이즈 · **V 조건별 mIoU(신설)** · VI ablation · VII 비용. 보충자료: S1 조건별 depth 의존도 표(`latex/sections/supp_tables.tex`), 클래스별 전체 표, 시드별 수치, EMM 15조합 전체.

데이터 그래프 공통: Python(matplotlib), 글꼴 TeX Gyre Heros(Helvetica 계열, TikZ 그림과 같은 모양), 단 폭 3.5 in, 글자 5.8 pt 이상. 기준선은 회색 계열(선 모양으로 구분), 우리 모델은 파랑 단계(가장 진한 것이 최종 모델). 측정 전 값은 빨간 점선 빈 막대로 "pending" 표시하고, 보간하지 않는다. 내보낸 뒤 겹침 검사(`audit_figure_collisions.py`)와 글자 크기 검사를 통과해야 한다(2026-10-28 세 그림 모두 PASS). 빗금(hatch) 패턴은 겹침 검사에서 거짓 양성을 내므로 쓰지 않는다.

## 3. Fig.1 다시 그리기 안내

초안 PDF: `latex/root.pdf` 2쪽. 원본 TikZ: `latex/figures/fig_arch.tex`(좌표는 cm, 전체 폭 18 cm). 다른 도구(Illustrator·Inkscape·PowerPoint·Figma)로 다시 그릴 때는 아래 구조를 지킨다.

### (a) 추론, 왼쪽에서 오른쪽으로 다섯 칸
1. **센서 입력 4개**(세로로 쌓음): 실제 DELIVER 장면 `sun/test/MAP_10_point21/002500` 의 RGB·Depth·Event·LiDAR 썸네일, 각 테두리를 센서 색 0.9 pt 로. 센서 이름은 썸네일 왼쪽에 세로로.
2. **동결 DINOv3-L**(점선 회색 큰 상자 하나): 안에 센서마다 한 줄. 줄마다 [센서 색 LoRA 칸] → [회색 블록 4개 = 6블록씩] 이고, 각 블록 끝의 센서 색 점이 "탭"(블록 6·12·18·24)이다. **모든 줄의 회색 블록이 똑같아 보여야** "가중치를 공유한다"가 전달된다. 센서마다 다른 것은 LoRA 색뿐이다.
3. **교차 모달 attention**(파란 실선 상자): 네 센서 색 점을 정사각형 꼭짓점에 두고 모든 쌍을 양방향 화살표로 잇는다. "주인 센서 없음, 대칭, 게이트 없음"이 이 그림 한 장으로 보여야 한다. 아래에 "2 layers, no gate".
4. **SimpleFPN**(옅은 테두리 상자): 크기가 다른 정사각형 4개(1/32·1/16·1/8·1/4). 아래쪽 ⊕ 로 고해상 경로가 각 단계에 더해진다.
5. **헤드**(파란 실선): Dense head 와 Query head(100 queries) → arbiter ⊕ → 예측 썸네일(같은 장면의 실제 A2 예측, `latex/figures/fig1_thumbs/pred.png`).
- **아래 띠 = 고해상 경로**(회색 실선, 선행 기법이라 파랑을 쓰지 않는다): 원 해상도 입력 → conv stem → deformable extractor(×4, 백본 탭 토큰을 점선 화살표로 읽음) → FPN 의 ⊕.

### (b) 학습
- clean 썸네일과 degraded 썸네일(뒤에 빠진 센서를 사선 친 빈 칸으로 겹침) → 같은 모델 상자 하나(두 입력이 **한 상자**로 들어가야 "가중치 공유"가 보인다) → 각각 L_CE.
- 교사(점선 회색, frozen, clean-trained)는 **clean 입력**을 받는다. 교사 출력이 degraded 패스의 L_KD 로 간다(파란 점선, T=2). 근거 코드: `train_reliadino.py:1012`(교사 입력 = clean `sample`), `:1036`(KD 대상 = 열화 패스 로짓).

### 다시 그릴 때 바꿔도 되는 것 / 안 되는 것
- 바꿔도 됨: 상자 비율, 화살표 경로, 썸네일 장면(다른 장면이면 `figures/src/make_arch_thumbs.py` 로 다시 만들 것).
- 바꾸면 안 됨: 센서 색, 파랑=학습·점선=동결 규칙, 교사 입력이 clean 이라는 점, 고해상 경로를 "우리 기여" 색으로 칠하지 않는 것(노벨티 아님, 브리프 인용 규칙).

## 4. 재료 위치

| 재료 | 경로 |
|---|---|
| Fig.1 썸네일 생성 | `figures/src/make_arch_thumbs.py <장면 루트> <출력 폴더> [A2 예측 png]` |
| 장면 원본(5장) | jarvis `/SSDb/jemo_maeng/dset/DELIVER/{img,depth,event,lidar,semantic}/sun/test/MAP_10_point21/002500_*` |
| A2 예측 덤프(test 1024) | NAS `analysis_logs/a2_test1024_dump_20261016/{s821,s902}/test/pred/` |
| DGFusion·CAFuser(재학습) 예측 | NAS `analysis_logs/native_axis_baseline_20261016/{dgfusion,cafuserb}_{test,val}{1024,1280,1536}/official_inference/` |
| 신호등 오탐 장면 | NAS `analysis_logs/a2_tl_fp_scenes_20261022/`(README 에 선택 규칙) |
| 얇은 객체 해부 | NAS `analysis_logs/a2_thin_anatomy_20261018/` |
