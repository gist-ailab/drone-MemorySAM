# 그림 출처·생성 방법 (IEEE Access 원고)

규칙(user 2026-10-29): 원고의 **모든 그림**은 "어떤 수치(또는 이미지)를 읽어서, 어떤 코드로, 어떻게 그렸는지"가 이 폴더 안의 파일로 남아 있어야 한다.
- 데이터 그래프: 수치는 `figures/source_data/*.csv`(행마다 source_file·source_locator 열) 또는 정본 CSV 에서 **읽는다**. 스크립트 안에 수치를 적지 않는다.
- 그림을 다시 만들 때: `python figures/src/make_data_figs.py` (데이터 그래프 3개) → `bash build.sh` (Fig.1 과 원고). 각 그래프 옆 `<이름>.json` 에 입력 파일 경로와 md5 가 남는다.
- 품질 검사: 겹침 검사 `~/.claude/skills/nature-figure/scripts/audit_figure_collisions.py`(PyMuPDF 필요), 글자 크기 `audit_pdf_text.py`(5 pt 하한).

| 원고 그림 | 파일(원고가 읽는 것) | 만드는 코드 | 읽는 입력 | 그리는 방법 | 상태 |
|---|---|---|---|---|---|
| Fig.1 개요(`fig:arch`) | `latex/figures/fig_arch.pdf` | `figures/src/fig_arch_standalone.tex` + `latex/figures/fig_arch.tex`(TikZ, `build.sh` 가 컴파일) | 썸네일 `latex/figures/fig1_thumbs/*.png` | 좌표 직접 배치(cm), 색 = carbon 토큰. 수치 데이터 없음 | 초안, 다시 그리기 대기(`notes/figure_guide.md` §3) |
| 〃 썸네일 | `latex/figures/fig1_thumbs/{rgb,depth,event,lidar,gt,pred,rgb_degraded}.png` | `figures/src/make_arch_thumbs.py <장면 루트> <출력> <A2 예측 png>` | DELIVER `sun/test/MAP_10_point21/002500`(jarvis `/SSDb/jemo_maeng/dset/DELIVER`), A2 시드821 예측 NAS `analysis_logs/a2_test1024_dump_20261016/s821/test/pred/img/sun/test/MAP_10_point21/002500_rgb_front.png` | depth·LiDAR = carbon 순차 파랑, event = 흰 바탕 극성 2색, GT·예측 = DELIVER 팔레트, 열화 RGB = 블러+대비 저하+가우시안(설명용, 학습 샘플러 아님) | 완료 |
| Fig.2 강건성(`fig:robust`) | `latex/figures/fig_robust.pdf` | `figures/src/make_data_figs.py` → `fig_robust()` | `figures/source_data/fig_robust_noise.csv`, `figures/source_data/fig_robust_single.csv` | (a) 모델별 선(S&P 0/.05/.10/.20), (b) 남은 센서별 묶음 막대. 빈 값 = 빨간 점선 "pending" | 초안. (b) 우리 모델 값 대기 |
| Fig.3 센서 고장 정성(`fig:failure_scenarios`) | 없음(자리 상자) | 아직 없음 | analysis 요청 inbox 20261027-182054 #2 | — | 대기 |
| Fig.4 악조건 정성(`fig:adverse`) | 없음(자리 상자) | 아직 없음 | analysis 요청 inbox 20261027-182054 #1·#3 | — | 대기 |
| Fig.5 조건별 의존도(`fig:dependence`) | `latex/figures/fig_dependence.pdf` | `figures/src/make_data_figs.py` → `fig_dependence()` | `figures/source_data/fig_dependence.csv` | 모델별 선(6개 조건), 빈 조건은 끊김 | 초안. split 통일·나머지 행 대기 |
| Fig.6 얇은 객체(`fig:thin`) | `latex/figures/fig_thin.pdf` | `figures/src/make_data_figs.py` → `fig_thin()` | `.claude_logs/experiments/analysis/DRN-261018-01/a2_vs_q2_perclass_3seed.csv`(열 d821·d902·d903) | 클래스별 막대 = 3시드 차이의 평균, 점 = 시드별 차이, 평균순 정렬, 얇은 4클래스 굵게 | 초안. 확대 장면 추가 todo |
| Fig.7 실패 사례(`fig:failure`) | 없음(자리 상자) | 아직 없음 | NAS `analysis_logs/a2_tl_fp_scenes_20261022/scene_0{1..4}.png`(선택 규칙 = 같은 폴더 README) | — | 재료 있음, 조판 대기 |

새 그림을 만들 때: (1) 수치를 `figures/source_data/<그림>.csv` 로 먼저 적고 source 열을 채운다 → (2) `make_data_figs.py` 에 함수를 추가해 그 CSV 를 읽게 한다 → (3) 이 표에 행을 추가한다. 정성 그림(장면)은 장면 목록 CSV(장면 id·선택 규칙·덤프 경로·md5)를 `figures/source_data/` 에 두고, 패널을 붙이는 스크립트를 `figures/src/` 에 둔다.
