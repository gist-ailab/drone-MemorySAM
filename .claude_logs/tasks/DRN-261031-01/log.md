append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-01 | AUTO_RESUME 재개 런 완주 확인 | hpca100:`outputs/ReliaDINO/hpca100_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827_A2hires/MCubeS_ReliaDINO-ViTL16_iadn/{epoch180_60.27_top1_checkpoint.pth, last_checkpoint.pth, train.log}` | ep128→ep200 정상 완주(train.log "Total Training Time 06:55:51" 확인, Traceback 없음). 학습기 top1 val mIoU 60.27(ep180). epoch180 ckpt md5 `f3509574f394287b93deccc4b32bccef`(hpca100). NAS 아카이브·legal 평가는 analysis에 inbox 통지 예정. |
