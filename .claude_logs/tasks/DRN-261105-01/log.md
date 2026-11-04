append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-05 | method 판독(1단계 E1-B 완주, manager 보고) | jarvis outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1B/ · 교사 복사본 /SSDb/jemo_maeng/ckpt/E1B_s821_valbest.pth(md5 일치) | 학습기 val-best 63.15(ep35) / test 53.94(ep40). 같은 시드 DINOv3-L E1 은 legal test 55.94 → B 교사는 약 −2.0(학습기 test 기준, legal 재채점 전). 2단계 A2-B 는 queue 등록, jarvis 4장 확보 시 자동 기동. 판정 기준은 A2-B − E1-B(같은 하네스) ≥ +1.0·EMM Δ ≥ +5. |
