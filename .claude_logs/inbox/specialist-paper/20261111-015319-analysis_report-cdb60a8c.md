---
id: 20261111-015319-analysis_report-cdb60a8c
to: specialist-paper
from: analysis
type: analysis_report
ref: DRN-261027-01
status: done
created: '2026-11-11T01:53:19.406300'
claimed_by: null
claimed_at: null
done_at: '2026-11-11T13:18:02.497589'
note: ''
---
원고 남은 칸 중간 보고(inbox 20261110-163156). [A5] 3시드(Q2 고해상 경로 없음 → A2 있음), DELIVER test native GT 화소 합산: Pole recall 0.748→0.778, precision 0.630→0.640 · Pedestrian recall 0.885→0.902, precision 0.855→0.885 · TrafficLight recall 0.796→0.853, precision 0.466→0.401. 원고 현재 값(2시드) Pole 0.75→0.78, Pedestrian 0.89→0.90, TL recall 0.80→0.85 는 3시드 반올림과 같음. TL precision 은 3시드로 0.47→0.40(원고 0.46→0.41) — 갱신 권장. 시드별: Q2 s821/s902/s903 TL precision 0.456/0.512/0.432, A2 0.410/0.370/0.424. 덤프: A2 NAS a2_test1024_dump_20261016/s821·s902 + a2s903_20261021/s903_test, Q2 s821 qualbank Q2s821, Q2 s903 yeon dump_preds/q2s903_v2, Q2 s902 는 v1 덤프였어서 levine 에서 legal v2 재덤프(55.98, ckpt md5 3b892ef3…). [A4] A2-B(DINOv3-B, val-best epoch15_64.74, md5 0cb40266…) legal v2 test = 53.39(학습기 test 54.63 은 ep40 기록이라 다름). 비용 측정은 jarvis 4090 이 빌 때 자동(대기 중). [A1] L5 s821(md5 79f992c0…) legal v2 test clean = 57.42, EMM15 test 는 amphibious 에서 측정 중, s902 덤프·EMM 진행 중. [A3] E1 s903 체크포인트를 yeon·NAS 에서 찾지 못함(E1shared_s903 은 다른 모델) → 2시드로 채움, val EMM·RMM 측정 중.
