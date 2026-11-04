append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-05 | jarvis 유휴 GPU3,7 채움(cross-session jarvis-eval-fill 요청, ref DRN-261029-01) — L6(s821/s902) legal v2 clean test 재채점 + EMM15 기동(yeon 원본을 jarvis로 scp -3 중개 전송 후), L8 레인 뒤에 체인 | ckpt 전송본: `/home/jemo_maeng/eval_fill_20261105/ckpts/L6_s821_epoch40_69.39_top1_checkpoint.pth`(md5 95d97cc394ec9aba76c17182309baabb, yeon 원본과 일치)·`L6_s902_epoch30_69.24_top1_checkpoint.pth`(md5 fa8239baa7a69fbf7543701d94348676, yeon 원본과 일치, 주의: s902는 top1 라벨이 epoch30, epoch40 top1 파일 없음 — yeon 디렉터리 전체 목록으로 확인) — 결과: 옆에 clean v2 txt + `/home/jemo_maeng/eval_fill_20261105/out/L6_s821/{clean_v2_test.log,test_emm/}`·`L6_s902/{clean_v2_test.log,test_emm/}` | tmux jemo 창 gpu3_L8s902(GPU3, L8_s902 체인 뒤)·gpu7_L8s821(GPU7, L8_s821 체인 뒤). scp 2개 파일 모두 전송 전후 md5 완전 일치 확인. 05:34 기준 L8 선행 작업이 GPU3는 clean test 73%, GPU7은 방금 시작(1%) 단계라 L6 작업은 대기열, 아직 미시작. 수치 해석·판정 없음 |
