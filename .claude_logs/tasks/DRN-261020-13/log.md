append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-05 | jarvis 유휴 GPU0,1 채움(cross-session jarvis-eval-fill 요청, ref DRN-261029-01) — L5(s821/s902) legal v2 clean test 재채점 + EMM15 기동(yeon 원본을 jarvis로 scp -3 중개 전송 후) | ckpt 전송본: `/home/jemo_maeng/eval_fill_20261105/ckpts/L5_s821_epoch40_70.27_top1_checkpoint.pth`(md5 79f992c0a97c2566aeaf1e37c9abe0dd, yeon 원본과 일치)·`L5_s902_epoch40_69.78_top1_checkpoint.pth`(md5 d31eaf41920c7f4b5eb6cf5c609dfd09, yeon 원본과 일치) — 결과: 옆에 clean v2 txt + `/home/jemo_maeng/eval_fill_20261105/out/L5_s821/{clean_v2_test.log,test_emm/}`·`L5_s902/{clean_v2_test.log,test_emm/}` | tmux jemo 창 gpu0_L7s821(GPU0, L7_s821 뒤에 체인)·gpu1_L7s902(GPU1, L7_s902 뒤에 체인). scp 2개 파일 모두 전송 전후 md5 완전 일치 확인(yeon 소스 md5sum 직접 계산 → jarvis 수신본 md5sum 대조). 05:34 기준 L7 선행 작업이 74% 진행 중이라 L5 작업은 대기열에 있고 아직 미시작. 수치 해석·판정 없음 |
