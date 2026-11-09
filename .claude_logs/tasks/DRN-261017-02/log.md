append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-04 | (MMSAM \| analysis) test 102장 재채점 5 ckpt(C2D2 s827·s828, w/o C2D2 s827, E1Mc s827·s828): 하네스 = qualbank_dump.py(val.py 에 MCubeS 분기 없음, 가드 OK), 5개 모두 학습기 top1 과 Δ0.00. C2D2 2시드 59.78 · w/o C2D2 60.27 · E1Mc 2시드 58.78 + 정성 은행 덤프 | NAS analysis_logs/mcubes_rescore_20261104/ | hpca100 GPU1~3 |
| 2026-11-09 | (MMSAM \| analysis) w/o C2D2 768 s827·s828 재채점(bengio, qualbank_dump, guard OK): 59.81·59.96 학습기 Δ0.00 → 같은 768 크롭 C2D2 − w/o = −0.11(2시드) + 정성 은행 덤프 | NAS analysis_logs/mcubes_rescore_wo768_20261109/ | hpca100 단절로 bengio |
