append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |

## 2026-10-16 (MMSAM | 생각정리) — val 원본 GT 축 3해상도 완료(감시 세션 보고, summary.json 원본 기준)
- 결과 파일: lecun `/SSDb/jemo_maeng/analysis_logs/native_axis_baseline_20261016/{dgfusion,cafuserb}_val{1024,1280,1536}/summary.json` (`native_axis_miou`)
- DGFusion 공식: 1024 66.4733 · 1280 66.4683 · 1536 67.3444 → val 선택 1536
- CAFuser(b): 1024 62.8732 · 1280 63.0023 · 1536 63.2299 → val 선택 1536
- 체인 버그: ssh 출력 노이즈로 json 파싱 실패 → 최고 해상도가 1024 로 고정돼 test@1536 자동 단계가 건너뛰어짐. 감시 세션이 test@1024 완료 후 test@1536 두 건을 수동 실행 예정.
- 1024 해상도 재현: DGFusion val 원본 GT 66.47 vs 공식 하네스 66.51(차 0.04)
