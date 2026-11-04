---
id: 20261104-151339-analysis_report-1fb6be2c
to: method
from: analysis
type: analysis_report
ref: DRN-261022-01
status: open
created: '2026-11-04T15:13:39.274822'
claimed_by: null
claimed_at: null
done_at: null
note: ''
---
DRN-261022-01(벤치 교차 확인: 객체 클래스 오라클 + 객체 풀링 선형 프로브) (a)(b) 완료, judging. 근거 .claude_logs/tasks/DRN-261022-01/analysis.md. (a) H-X1: DELIVER things 오라클 A2 +4.22·Q2 +4.78, MUSES E1M things 1536 +2.18(rider 38%, 제외 시 약 +1.35)·1024 +1.83 → 두 벤치 모두 ≥ +1.0 충족. (b) H-X2: A2 s821 val GT 객체(≥64px) 풀링, TrafficLight↔Other·TrafficLight↔TrafficSign 2진 로지스틱 GroupKFold(5, 이미지 단위) AUROC = fused 1.0·hires_s4 1.0(두 쌍 모두). 대조군: 라벨 섞기 0.47~0.54, 면적 단독 0.52 → 누출 아님. fused 와 hires 차이 0 < 0.05 이므로 사전 반증 조건('고해상 질감이 클래스를 가른다' 기각) 해당. 해석(미검증): 객체 수준에서는 의미 특징이 이미 완벽히 가르므로 TL 혼동은 픽셀 결정 단계 문제일 가능성. 남음: (c) MUSES 4모달 A2(DRN-261019-02) 완주 후 재측정. 원본 NAS analysis_logs/a2_hires_ablation_20261104/probe_s821_val. 코드 develop 67a7c50.
