append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-02 | 등록(user 직접 지시) · hpca100 GPU2(val)·GPU3(test) 기동 스크립트 `hpca_nmbench.sh`(tools/mm_eval_v2.py --protocol nm --nm_mode bench, batch 4, seed 0, 전수, 3시드 순차) | hpca100 `~/SSDb/jemo_maeng/nmbench_20261102/` | 기동 검증은 sonnet 위임 중; 원고 S&P 단독 삭제 결정은 inbox(specialist-paper)로 전달 |
| 2026-11-02 | 기동 검증(sonnet 관찰, 판정=메인): 17:52 KST 기동, tmux nmbench_val(GPU2)·nmbench_test(GPU3), 배치 카운터 6→14 전진(약 20.5 s/배치), Traceback 없음. ckpt md5 s821 f05b54d4…, s902 0aab4cae…, s903 63ac54a0…(기록값 일치). 시드당 val 502·test 475 배치 ≈ 2.8 h → 3시드 순차 약 8.5 h(등록 시 예상 6 h보다 김), 완료 예상 11-03 02:30 KST 전후 | hpca100 `~/SSDb/jemo_maeng/nmbench_20261102/logs/chain.log` | clean mIoU 는 각 시드 끝에 기록됨 |
