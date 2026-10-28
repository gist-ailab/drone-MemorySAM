# IEEE Access 원고 (DRN-261027-01)

원고 원본은 이 폴더이고 git 으로 버전을 관리한다(user 결정 2026-10-28). 집필 브리프는 `../00_brief.md`.
볼트의 옛 위치(`/nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-access-2026/`)는 2026-10-28 이후 갱신하지 않는다.

| 경로 | 내용 |
|---|---|
| `latex/` | 원고 소스: `root.tex`, `sections/*.tex`, `figures/`(TikZ·PDF 그림, 썸네일), `references.bib`, Access 템플릿 파일(클래스·글꼴) |
| `figures/src/` | 그림·점검 스크립트: `make_arch_thumbs.py`(Fig.1 썸네일), `make_data_figs.py`(데이터 그래프, 정본 수치 내장), `verify_bib.py`(참고문헌 대조) |
| `notes/` | `figure_guide.md`(그림 목록·디자인 규칙·다시 그리기 안내), `reference_check.md`(참고문헌 점검 결과) |
| `templates/` | 받은 템플릿 원본: `access-20260513/`(사용 중, 2026-10-29~), `jsen-202405/`(IEEE Sensors Journal, 2026-10-29 목표 변경으로 미사용), `ctan-ieeetran-1.8b/`(미사용) |
| `drafts/` | 날짜별 PDF 스냅샷(커밋 대상). 사용자에게 보낼 PDF 는 여기서 가져간다 |
| `build/` | 빌드 산출물(git 제외) |

빌드: `bash build.sh` → `build/root.pdf`. 스냅샷: `bash build.sh snapshot` → `drafts/IEEE-Access-draft_<날짜>.pdf` 를 커밋한다.
데이터 그래프 갱신: `python figures/src/make_data_figs.py latex/figures ../../experiments/analysis/DRN-261018-01/a2_vs_q2_perclass_3seed.csv`.
