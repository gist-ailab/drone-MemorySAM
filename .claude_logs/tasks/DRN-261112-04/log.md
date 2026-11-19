append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-16 12:20 | 유효 배치 확인(method 요청): s821 jarvis GPU0·s902 GPU2·s903 GPU6 모두 BS1·world 1(진행바 3983 it/epoch)·accum 16 → 유효 16, LR 스케줄 248 step/epoch = A2(4GPU: BS1×4×4) 와 같음. TRAIN.DDP false(NVML 고장). 계산식 train_reliadino.py:401-404 — accum=ceil(16/(BS×world)), 스케줄러 스텝=updates_per_epoch=3983//(BS×world×accum)=248(A2 4GPU 와 동일). 근거=config TRAIN.BATCH_SIZE 1 + 로그 진행바 epoch 당 iteration(3983=world1, 1991=world2) | manager |
| 2026-11-18 | (MMSAM \| analysis) 고해상 있음·C2D2 없음 legal v2 test clean: s821 57.94 · s902 57.64 · s903 56.72(평균 57.43). qualbank test 덤프 3시드(trainer식 58.81/58.55/57.60). lecun 겹침(11-17 19:13 A2noDepth 학습과 같은 GPU)으로 지연, manager 배정 jarvis GPU4 로 이전 | jarvis /SSDb/jemo_maeng/analysis_logs/hnc_20261117/ , 덤프 lecun /SSDb/jemo_maeng/analysis_logs/hnc_20261117/hnc_s*_test_dump | val EMM15 진행, val RMM 은 GPU 대기 |
| 2026-11-19 | (MMSAM \| analysis) val EMM15 3시드: val clean 71.03/70.48/70.41(평균 70.64), EMM15 48.87/47.96/47.83(평균 48.22). Fig.8 패널은 DRN-261027-01 에 기록 | jarvis /SSDb/jemo_maeng/analysis_logs/hnc_20261117/hnc_s*_val_emm/ | val RMM 21/42, test EMM GPU4 |
| 2026-11-19 | (MMSAM \| analysis) val RMM 3시드 완료: .75/.50/.25 = 57.62/61.13/64.34(시드별 README). 행 전체 paper 송부(위치 목록 포함). test EMM15 는 GPU 를 DRN-261119-01 에 넘기며 중단 | NAS analysis_logs/hnc_A2hiresNoC2D2_20261117/ | 완료(표 밖 test EMM 미측정) |
