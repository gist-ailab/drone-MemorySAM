append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-16 12:20 | 유효 배치 확인(method 요청): s821 jarvis GPU4·s902 GPU7 BS1·world 1(3983 it/epoch)·accum 16 → 유효 16, 248 step/epoch = A2 와 같음. TRAIN.DDP false(NVML 고장). 계산식 train_reliadino.py:401-404 — accum=ceil(16/(BS×world)), 스케줄러 스텝=updates_per_epoch=3983//(BS×world×accum)=248(A2 4GPU 와 동일). 근거=config TRAIN.BATCH_SIZE 1 + 로그 진행바 epoch 당 iteration(3983=world1, 1991=world2) | manager |
