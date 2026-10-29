append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-10-29 | L1(공유 LoRA r16) s821 강건성 측정 기동(MMSAM \| analysis, 요청 inbox 20261028-162006). ckpt yeon `…A2L1shared/…/epoch35_69.5_top1_checkpoint.pth` md5 ff7015ea… → NAS `ckpts/a2L1shared_s821_20261021/`·drone-demo 백업·amphibious-demo(4곳 md5 일치). A2 s903 과 같은 프로토콜(mm_eval_v2 batch 8·seed 0·전수) 8종 + clean legal 덤프(기대 58.84) | amphibious-demo `~/workspace/jemo_maeng/robust_l1_s821_20261029/`(tmux l1_robust_a·l1_robust_b), config `configs/eval/a2L1shared_s821_eval1024_amphibious.yaml`(97c6207) | 비교 = A2 s821 jarvis robust_a2_20261019/s821(batch 4) |
