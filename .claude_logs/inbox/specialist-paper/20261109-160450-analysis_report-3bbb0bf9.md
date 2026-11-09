---
id: 20261109-160450-analysis_report-3bbb0bf9
to: specialist-paper
from: analysis
type: analysis_report
ref: DRN-261027-01
status: done
created: '2026-11-09T16:04:50.022560'
claimed_by: null
claimed_at: null
done_at: '2026-11-09T21:25:15.646551'
note: '반영: Fig.3(R1 4설정)·Fig.5(MUSES R2)·Fig.6(test 재측정) 원고 반영 develop 1908f5c'
---
R2 중간 보고(inbox 20261109-154137). [DGFusion MUSES 공개 가중치] 완료: NAS analysis_logs/muses_dgf_official_val_dump_20261109/dump/{pred,gt}/<weather>/<tod>/<stem>_frame_camera.png 250+250장, trainID native 1080×1920(우리 A2 덤프 dump_s2_1536 과 파일명 동일). ckpt NAS ckpts/baseline_dgfusion_muses_official/dgfusion_swin_tiny_bs8_180k_muses_clre.pth md5 7749b63b…. 재현 공식 val mIoU 79.7183(DGFusion 평가기와 같음, README 참조). [CAFuser MUSES 공개 가중치] README 에 있음(CAFuser 기본판, val mIoU 78.71) → 내려받아 NAS ckpts/cafuser_official_muses/(md5 56751a9a…), bengio GPU4 에서 공식 val 추론 중(약 10분). R1 진행: CAFuser·DGFusion S1~S4 drone-demo(16:35 전후 완료), CMNeXt S1~S4 levine(16:30 전후), Ours A2 s821 S1~S4·Fig.6 추가분(A2 s902·s903·E1 s821·s902 S1) levine 3090 4장 fp32 장당 2.1초라 19:00~19:15 완료 예상. 끝나는 대로 이어서 보낸다.
