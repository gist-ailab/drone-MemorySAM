append-only, 최신이 아래. 수치는 결과 파일 경로로.

| 날짜 | 작업·실행 | 결과 파일 경로 | 메모 |
| --- | --- | --- | --- |
| 2026-11-03 | 1차 측정(MMSAM \| analysis): jarvis GPU2(RTX 4090), torch 2.7.0+cu128, bf16, DELIVER test 첫 장 4모달 1024², 도구 `tools/cost_table_ours.py`(develop 958ca9e). **유효**: A2 s821 총 377.05M · 학습 73.98M · 최대 메모리 2.47 GB / E1 s1(= Q2 와 같은 구조) 총 361.87M · 학습 58.79M · 2.18 GB(A2−E1 = 15.18M = 고해상 경로 파라미터, 학습 로그 HIRES params 15,184,512 와 일치). **무효**: 지연(A2 377.9 ms·E1 335.1 ms) — 측정 중 같은 GPU 에 다른 사용자 프로세스(pid 236096, 9.5 GB) 동시 실행, std 67~77 ms. GFLOPs 미측정(fvcore AssertionError: 3) → 도구에 FlopCounterMode 대체·latency_valid 표시 추가. DGFusion·CAFuser 미측정 | NAS `analysis_logs/cost_table_20261103/{A2_s821,E1_s1}.json` | 빈 GPU(독점)에서 지연·FLOPs 재측정 필요 |
