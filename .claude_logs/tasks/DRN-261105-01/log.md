append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-05 | method 판독(1단계 E1-B 완주, manager 보고) | jarvis outputs/ReliaDINO/jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1B/ · 교사 복사본 /SSDb/jemo_maeng/ckpt/E1B_s821_valbest.pth(md5 일치) | 학습기 val-best 63.15(ep35) / test 53.94(ep40). 같은 시드 DINOv3-L E1 은 legal test 55.94 → B 교사는 약 −2.0(학습기 test 기준, legal 재채점 전). 2단계 A2-B 는 queue 등록, jarvis 4장 확보 시 자동 기동. 판정 기준은 A2-B − E1-B(같은 하네스) ≥ +1.0·EMM Δ ≥ +5. |
| 2026-11-05 | 2단계 A2-B 완주 확인(jarvis GPU2,4,5,6) | jarvis `/home/jemo_maeng/src/drone-MemorySAM-develop/logs/a2b_s821_launch.log`(mtime 17:14:12) | Epoch[40/40] 완주, 총 학습시간 09:39:44, 런타임 에러 없음(DDP strides UserWarning만). 학습기 기록(legal v2 재채점 전): val mIoU 64.69(best 64.74@ep15) / test mIoU 54.63(best 54.76@ep20). E1-B(학습기 test 53.94) 대비 A2-B 학습기 test +0.69~+0.82 — 판정 기준(≥+1.0)에는 아직 못 미치나 legal v2·EMM15 재채점 전 수치라 확정 아님. jarvis GPU2,4 완전 유휴로 확인됨(해방). 수치 해석·판정은 method 소관, 다음 단계는 legal v2 test + EMM15 재채점 |
