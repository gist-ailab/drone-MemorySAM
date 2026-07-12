# 🗺 experiments/ — MOC (Map of Content)

> 폴더 역할: **실험 기록** — 결과 로그(canonical), 실시간 학습 모니터, 실험 레지스트리, 심층 분석.

| 파일 | 한줄설명 | legacy_id |
|------|----------|-----------|
| [registry.md](registry.md) | **실험 레지스트리 허브** — 핵심 실험 ID/config/서버/ckpt/상태/수치 한눈표 (신설 2026-07-08) | — (신규) |
| [log.md](log.md) | 전체 결과 M-score 표 + 버전별 상세 + 진단 — **실험 canonical** | 03 |
| [monitor-log.md](monitor-log.md) | 진행 중 학습 실시간 모니터 로그 (RUN-N 단위, `/loop` 세션이 append) | 15 |
| [analysis/2026-06-30-p28-p29-failure-analysis.md](analysis/2026-06-30-p28-p29-failure-analysis.md) | P28(RBMA)·P29(SDC) 체계적 실패분석 + P30 커버리지 판정 + P31 프로토타입 | 16 |
| [analysis/2026-07-12-p29-p34-standard-analysis.md](analysis/2026-07-12-p29-p34-standard-analysis.md) | **P29·P31·P32·P34 표준분석 종합(동일 프로토콜)** — P34 전도메인 1위·Water 부활, SAM2 피쳐 rank-1 붕괴 vs DINOv3 정렬, additive-bias 3세대 no-op, P31 router +10~13 기여. 산출물=NAS `/drone_nas/drone/analysis_logs/` | — (2026-07-12) |
| (repo) [tools/README_seg_analysis.md](../../tools/README_seg_analysis.md) | **표준 분석항목 1–4 ↔ 도구 매핑 (canonical)** — 모델 분석 지시를 받으면 **먼저 읽기**. adapter 적응도(D3B)·피쳐 통계(D2N)·모듈 A/B(D5)·멀티모델 비교(compare_models) 전부 model-agnostic, P31/32/33/34+ 재사용. **새 모델 분석 코드를 새로 짜지 말 것** | — (2026-07-12) |
