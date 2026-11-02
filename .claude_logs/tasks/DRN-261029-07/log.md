append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-02 | amphibious-demo(ailab@172.27.190.240, RTX5090 32GB) GPU0에 시드821 1런 기동 | amphibious-demo:`~/workspace/jemo_maeng/src/Project/drone/drone-MemorySAM/outputs/ReliaDINO/amphibious_deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L8norouter/` | config `configs/amphibious-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2L8norouter.yaml`(develop 2324d6b 커밋, 허브 동기화 후 기동 — user 지시 2026-11-02). MODEL.ROUTER.ENABLE true→false 한 변수만 변경, TRAIN.QAF.TEACHER_CFG=E1.yaml로 교사 아키텍처 오염 방지(L1/L2/L4와 동일 패턴). 1차 기동 시도는 TEACHER_CFG 누락으로 RuntimeError(unexpected router 키) — 수정 후 재기동 성공. 검증: trainable 73,699,641(router 제외분), teacher missing=0 unexpected=0, GPU 21.4GB/98%, loss 22.98→18.94 안정 감소(ep1 114/3983). yeon 단절로 MUSES A2(교사 E1M4 체크포인트가 yeon 전용)는 보류, 체크포인트 독립적인 이 ablation으로 대체. 통합 감시자는 amphibious-demo의 tmux가 PATH 밖 경로라 자동 등록 불가 — 수동 주기 확인으로 대체. |
