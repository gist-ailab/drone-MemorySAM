append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-04 | amphibious-demo GPU0에 시드20260827 1런 기동(paper 세션 집행) | amphibious-demo:`~/workspace/jemo_maeng/src/Project/drone/drone-MemorySAM/logs/chain_e1mcb.status`, `.status.20261104_fail`(실패 기록 보존) | L8 종료 후 chain_e1mcb_amphibious.sh 첫 기동(04:41)은 TRAIN.DDP true인데 torchrun 없이 python 직행이라 "Default process group has not been initialized" 즉시 실패 — 재시도 2회 후 연쇄 정지(GPU 04:45~13:24 유휴). 스크립트를 torchrun --standalone --nproc_per_node=1로 수정(develop 828aed5, amphibious 파일 md5 6ca1a3d5) 후 13:24 재기동 성공. 검증: total_trainable 39,476,904(스모크와 일치), RANDOM INIT 없음, iteration 3.87it/s, GPU 13.2GB·89%. ETA: 시드20260827 약 5시간(오늘 18~19시경 완주), 이어 시드20260828 약 5시간(내일 00~01시경). 각 런 뒤 qualbank_dump.py로 test 102장 덤프. amphibious-demo tmux가 PATH 밖이라 통합 감시자 등록 불가 — 수동 주기 확인으로 대체(기존 L8과 동일 패턴).
