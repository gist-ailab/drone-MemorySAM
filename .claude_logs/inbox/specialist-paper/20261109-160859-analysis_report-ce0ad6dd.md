---
id: 20261109-160859-analysis_report-ce0ad6dd
to: specialist-paper
from: analysis
type: analysis_report
ref: DRN-261027-01
status: done
created: '2026-11-09T16:08:59.628859'
claimed_by: null
claimed_at: null
done_at: '2026-11-09T21:25:15.788723'
note: '반영: Fig.3(R1 4설정)·Fig.5(MUSES R2)·Fig.6(test 재측정) 원고 반영 develop 1908f5c'
---
R2 완료(inbox 20261109-154137). MUSES 공식 val 250장, trainID native 1080×1920, 파일명 <stem>_frame_camera.png(우리 A2 덤프 dump_s2_1536/pred/{pred,gt}/ 와 동일). ① DGFusion 공개: NAS analysis_logs/muses_dgf_official_val_dump_20261109/dump/{pred,gt}/ 250+250장, ckpt md5 7749b63b…, 재현 79.7183. ② CAFuser 공개(README MUSES 기본판, 이번에 내려받음 NAS ckpts/cafuser_official_muses/ md5 56751a9a…): NAS analysis_logs/muses_caf_official_val_20261109/dump/{pred,gt}/ 250+250장, 재현 78.7106(README val 78.71). ③ Ours: analysis_logs/muses_a2m4_official_val_20261104/dump_s2_1536/pred/{pred,gt}/ (81.7583). 각 폴더 README 에 명령·조건별 mIoU. R1 은 CMNeXt·CAFuser·DGFusion 16:30 전후, Ours·Fig.6 추가분 19:00~19:15 예상.
