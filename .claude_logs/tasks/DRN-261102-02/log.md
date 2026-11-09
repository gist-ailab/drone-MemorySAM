append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-09 | MCubeS C2D2-B 2시드 착수(user "FuseAdapt DINOv2-B 와 백본 맞춰 실험") | s828: bengio `/SSDf/jemo_maeng/c2d2b_20261109/logs/c2d2b_s828.log`(tmux c2d2b, GPU0·5 DDP2) · s827: amphibious `~/workspace/jemo_maeng/c2d2b_20261109/logs/c2d2b_s827.log`(대기 스크립트가 E1 NM 측정 종료 후 GPU0 엄격 빈 GPU 확인 뒤 기동) | config develop 69d72cd(L→B 백본 3키·탭 [3,6,9,12]·교사 E1Mc-B 같은 시드). 코드 = develop 69d72cd git archive(두 서버 c2d2b_20261109/repo). bengio 로 MCubeS(4,504파일, split 목록 md5 일치)·DINOv3-B 가중치(sha256 일치)·교사 s828(md5 84da62a6…) 전송. s828 기동 검증: trainable 52.6M, [QAF-T] 교사 missing=0 unexpected=0, 두 패스 on, 1.33 s/it·151 it/epoch(200ep 약 12~13h 추정), GPU 11.8GB. 운영 실수: 처음 GPU6·7 에 기동했으나 그 직전 다른 사용자 작업이 들어와 있어 즉시 내림(우리 프로세스 잔존 없음 확인) → 엄격 확인 스크립트로 GPU0·5 재기동. |
