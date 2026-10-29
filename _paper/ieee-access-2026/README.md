# IEEE Access 원고 (DRN-261027-01)

원고 원본은 이 폴더(repo `_paper/ieee-access-2026/`)이고 git 으로 버전을 관리한다(user 결정 2026-10-28, 위치는 2026-10-29 user 지시로 `.claude_logs/paper/` 에서 `_paper/` 로 이동). 집필 브리프(결정·규칙 기록)는 repo `.claude_logs/paper/00_brief.md` 에 남는다.
같은 `_paper/` 바로 아래의 `main.tex`·`cvpr.sty`·`rebuttal.*`·`sec/`·`fig/` 는 2026-03 의 CVPR 원고(별개, git 미추적)이며 이 원고와 무관하다. 볼트의 옛 위치(`/nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-sensors-2026/`)는 2026-10-28 이후 갱신하지 않는다.

| 경로 | 내용 |
|---|---|
| `latex/` | 원고 소스: `root.tex`, `sections/*.tex`, `figures/`(TikZ·PDF 그림, 썸네일), `references.bib`, Access 템플릿 파일(클래스·글꼴) |
| `figures/src/` | 그림·점검 스크립트: `make_arch_thumbs.py`(Fig.1 썸네일), `make_data_figs.py`(데이터 그래프, `figures/source_data/` CSV 를 읽어 그림), `fig_arch_standalone.tex`(Fig.1 단독 컴파일), `check_table_sources.py`(표 숫자 출처 대조), `verify_bib.py`(참고문헌 대조) |
| `figures/source_data/` | 데이터 그래프 수치 CSV(행마다 source_file·source_locator). 폴더 이름이 `data/` 가 아닌 이유: repo `.gitignore` 가 `data/` 를 전부 무시한다 |
| `notes/` | `table_sources.md`(표 수치 출처), `figure_sources.md`(그림별 코드·입력·방법), `figure_guide.md`(디자인 규칙·다시 그리기 안내), `reference_check.md`(참고문헌 점검 결과) |
| `templates/` | 받은 템플릿 원본: `access-20260513/`(사용 중, 2026-10-29~), `jsen-202405/`(IEEE Sensors Journal, 2026-10-29 목표 변경으로 미사용), `ctan-ieeetran-1.8b/`(미사용) |
| `drafts/` | 날짜별 PDF 스냅샷(커밋 대상). 사용자에게 보낼 PDF 는 여기서 가져간다 |
| `build/` | 빌드 산출물(git 제외) |

빌드: `bash build.sh` → `build/root.pdf`. 스냅샷: `bash build.sh snapshot` → `drafts/IEEE-Access-draft_<날짜>.pdf` 를 커밋한다.
데이터 그래프 갱신: `python figures/src/make_data_figs.py` (입력 = `figures/source_data/*.csv` 와 repo `.claude_logs/experiments/analysis/DRN-261018-01/a2_vs_q2_perclass_3seed.csv`, 출력 = `latex/figures/`).

## 출처 규칙(user 2026-10-29)
- **표 수치**: 모든 표 수치의 출처(파일·줄·행 제목)는 `notes/table_sources.md` 에 적는다. 표를 고치면 같은 커밋에서 이 문서를 고치고 `python figures/src/check_table_sources.py` 로 빠진 숫자가 없는지 확인한다.
- **그림**: 모든 그림은 만든 코드와 입력이 이 폴더에 있어야 한다. 데이터 그래프의 수치는 `figures/source_data/*.csv`(source_file·source_locator 열)에 두고 `figures/src/make_data_figs.py` 가 읽어서 그린다(스크립트에 수치를 적지 않는다). 그림별 코드·입력·방법 목록은 `notes/figure_sources.md`.
