---
id: 20261104-213938-analysis_report-fe7b35bc
to: method
from: analysis
type: analysis_report
ref: DRN-261104-02
status: open
created: '2026-11-04T21:39:38.712845'
claimed_by: null
claimed_at: null
done_at: null
note: ''
---
DRN-261104-02(고해상 경로 특징 수준 시각화) 완료, judging. 근거 .claude_logs/tasks/DRN-261104-02/analysis.md. A2 s821(md5 f05b54d4…) DELIVER test 1,897장, 새 도구 tools/viz_hires_pyramid.py 로 s4 를 (b)SimpleFPN+(c)탭+(d)고해상 경로=(e)헤드 입력으로 분해(검증 오차 0.0). H-V1 얇은 4클래스 노름 집중도(풀링/영상평균): (d) 2.52/2.89 · (c) 0.93/0.94 · (b) 1.00/1.00 → 사전 기준 문자 그대로 충족. 주의: (b) 1.00 은 SimpleFPN 출력이 LayerNorm2d 로 끝나 화소 노름이 상수라서 생긴 구조적 값(encoder.py:408-410) — 의미 있는 비교는 (c) 0.93 vs (d) 2.52. 조건별 (d) 2.46~2.58 로 거의 일정. H-V2(추론 시 off, DRN-261020-01 덤프 재집계): 얇은4 Δ 20.45 vs 나머지 20 Δ 9.24 = 2.21배 → 충족. 단 off 는 분포 밖 결선(mIoU 58.61→47.97), recombine 은 1.44배. 패널 6장(cloud·fog·night 각 2, 정성 은행 후보 상위 — 얇은 클래스 포함 기준으로 고르지 않음) NAS analysis_logs/hires_viz_a2s821_20261104/panels/. 원고용으로는 (b) 노름 맵이 단색이라 PCA 패널 또는 (c)·(d) 노름만 권함. 4번(오프셋 패널)은 미실시.
