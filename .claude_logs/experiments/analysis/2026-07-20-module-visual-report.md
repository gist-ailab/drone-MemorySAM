# 모듈·제안영역 시각 리포트 포인터 (2026-07-20)

**본문+그림 8장 = NAS** `/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/module_report_20260720/` (`report.md` + `figs/fig1~8`).

차기 모델 개발 참고용 — [2026-07-20-failure-keys-p38-deliver-p37a-muses.md](2026-07-20-failure-keys-p38-deliver-p37a-muses.md)(실패-키, canonical 텍스트 판정)의 수치를 그림·표로 고정:

| 그림 | 내용 |
|---|---|
| fig1 | 신규 모듈 순기여 세대별(RBMA≈0→CEFR .13→m2f .07→**P39 V1 +1.6/V5 +0.74**) + router 의존 해소(+40→+0.4~2.1) |
| fig2 | 키3 rank 병목 (per-modal 16~33 vs FUSED 7/256, 양 데이터셋) |
| fig3 | 키5 모달 반전 (event: DELIVER dead vs MUSES +0.27) |
| fig4 | 키2/④ thin-class 궤적 (P35↓→P36 router 회복→P38 m2f 되잃음) |
| fig5 | 키4 조건 프로파일 (DELIVER spread 2.6 vs MUSES 14.9·fog 병목) |
| fig6 | P39 조기 즉검 토글×조건 (V1 전 조건 기여 · V5 DELIVER night/rain 음수) |
| fig7·8 | 피쳐 PCA 패널 (DELIVER P38 / MUSES P37a) |

말미 "모듈별 최종 판정 표"(🔴폐기 6종 / 🟢유지·채택 4종 / 🟡검증 대기 2종)가 차기 설계의 시작점. 그림 재생성 = json 파싱 스크립트(리포트 헤더 참조), 팔레트 = dataviz validator 통과본(backbone_report_20260713과 동일).
