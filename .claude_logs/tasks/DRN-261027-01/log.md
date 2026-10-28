append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-10-27 | 원고 뼈대 생성(IEEEtran V1.8b journal, 절 8개 자리), 무에러 빌드 1쪽 | /nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-sensors-2026/latex/root.pdf | 템플릿 출처·미확인 사항은 paper/00_brief.md "작업 위치·템플릿 기록" |
| 2026-10-27 | 표 7개(T1~T7)·그림 5개 자리 조판, 정본 수치만 채우고 빈 칸은 붉은 todo, 출처는 표 위 LaTeX 주석 | /nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-sensors-2026/latex/sections/5_experiments.tex | 3쪽 빌드 오류 0. 빈 칸·불일치 6건 method 에 질문(inbox 20261027-173029-question-c7e5c8b0) |
| 2026-10-27 | 원고를 JSEN 전용 템플릿(ieeecolor+jsen, 2024-05판)으로 이전, 그래픽 초록 자리 추가, Access 템플릿 보관 | /nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-sensors-2026/latex/root.pdf | 빌드 오류 0, 3쪽. 템플릿 md5 는 paper/00_brief.md |
| 2026-10-27 | Fig.1 개요도 TikZ 초안(실제 DELIVER 장면·A2 예측 썸네일, carbon 색), 다시 그리기 안내서, 표 7개 글자 크기 통일(resizebox 제거), 그림 6개 계획, 정성 장면 선별 analysis 요청 | /nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-sensors-2026/notes/figure_guide.md · latex/figures/fig_arch.tex | 빌드 오류 0, 표·그림 overfull 0(남은 경고는 템플릿 머리말과 저자 todo). inbox 20261027-182054 |
| 2026-10-28 | 10쪽 결정 반영: 데이터 그래프 3개(강건성·조건별 의존도·얇은 객체) Python 초안, 겹침·글자 크기 검사 통과, 조건별 mIoU 표 신설, 의존도 표 보충자료 이동, ablation 확정 4행 채움 | /nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-sensors-2026/latex/root.pdf · figures/src/make_data_figs.py | 빌드 오류 0, 4쪽(본문 글 미작성). analysis inbox 20261028-183826 |
| 2026-10-28 | 참고문헌 점검(CrossRef·arXiv·DBLP 자동 대조 + 수동 판정): 40개 중 36 확인, 오류 5건 수정(HyperDUM 저자 오류 포함), 거짓 경보 3, 미확인 1(READ), 브리프 필요 문헌 4건 추가 | /nas_jm/Research/26_MultimodalSeg/reports/manuscript/ieee-sensors-2026/notes/reference_check.md | 도구 figures/src/verify_bib.py. PDF 사본 ~/Downloads/IEEE-Sensors-draft_2026-10-28.pdf |
| 2026-10-28 | 원고 원본을 repo `.claude_logs/paper/ieee-sensors-2026/`로 이동(user 결정: git 버전 관리), build.sh(빌드는 build/, 스냅샷은 drafts/), 볼트에 MOVED.md | .claude_logs/paper/ieee-sensors-2026/drafts/IEEE-Sensors-draft_2026-10-28.pdf | repo 안 빌드 오류 0. Downloads 사본 삭제 |
| 2026-10-29 | 목표 저널 IEEE Access 로 변경(user): 폴더 ieee-access-2026 으로 git mv, Access 템플릿 이전(TikZ 충돌 → Fig.1 별도 PDF), 데이터 그래프 단 폭 3.36 in 로 재생성(겹침·글자 검사 통과), 브리프 결정 절·lab-plan 제목·method inbox | .claude_logs/paper/ieee-access-2026/drafts/IEEE-Access-draft_2026-10-29.pdf | 빌드 오류 0, 표·그림 넘침 0. inbox 20261029-015504 |
