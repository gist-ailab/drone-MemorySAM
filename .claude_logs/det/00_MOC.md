# 🗺 det/ — MOC (Map of Content)

> 폴더 역할: **Detection 트랙(poongsan indoor, 국가 R&D mAP50 0.85 목표)** 전용 진단·데이터 수리 문서.

| 파일 | 한줄설명 | legacy_id |
|------|----------|-----------|
| [diagnosis-plan.md](diagnosis-plan.md) | P29/P30-Det 성능 진단 계획(2026-07-02) — Phase 0~3 실험 체크리스트. **det 작업 전 필독** | 19 |
| [p29det-data-fix.md](p29det-data-fix.md) | P29-Det 학습실패 진단 → 깨끗한 라벨셋 재학습 (poongsan v2/v3 split) | 17 |
| [p29-vs-p30-comparison.md](p29-vs-p30-comparison.md) | P29-Det ep9 vs P30-Det ep24 전체/클래스별 AP 비교(v2 test 1772장) — P30은 대형에서 대등, **소형에서 붕괴**(AP_small 0.006 vs 0.120) | — (2026-07-28 회수, 구 `det_eval/COMPARE_P29_vs_P30.md`) |
| [p30-statistics-and-feature-analysis.md](p30-statistics-and-feature-analysis.md) | P30-Det ep39 통계 + **모듈 정량 판정**: RBMA=inert(cos 1.000)·reliability-router=non-adaptive 고정배정(P3=LiDAR/P4=thermal/P5=RGB)·reliability 전모달 포화(≈0.99999). 재현=[`tools/probe_det_features.py`](../../tools/probe_det_features.py). panel PNG 24장은 git 미포함(태그 `archive/det-p29-p30-analysis`) | — (2026-07-28 회수) |
| [assets/p29-p30-perclass-compare.csv](assets/p29-p30-perclass-compare.csv) | 위 두 문서의 per-class AP50/AP 원시 수치(CSV) | — (2026-07-28 회수) |
| [det-cert-D1-realtime.md](det-cert-D1-realtime.md) | Det 공인인증(2026-07-23) — D1 ViT-S/S+/B/L 백본 스윕, RTX 5080 FPS 추정(fleet에 실기 없음), 인증 모델 확정(ViT-S+) + 재현 패키지 위치 | — |
| [det-cert-D1-vitsp-handoff.md](det-cert-D1-vitsp-handoff.md) | det-cert-D1-vitsp-handoff.md — D1 ViT-S+ 인증 웨이트·코드·정보 핸드오프(다른 세션용) | — |
