판정은 이 파일이 아니라 `lab-plan verdict DRN-261119-01 --analysis .claude_logs/tasks/DRN-261119-01/analysis.md`로 기록한다.

작성: MMSAM | analysis, 2026-11-19 18:3x(③ 완료, ①② 진행 중). 요청 inbox `20261119-143558`(method, user 지시). 학습 없음. 판정·해석 문장 없음(축별 값·Δ·부호 일치만).

## 대상·공통
- A2 s821 `epoch40_68.93`(f05b54d4) · s902 `epoch40_70.09`(0aab4cae).
- L7(센서 제거만, 센서별 물리 열화 없음) s821 `epoch40_69.73`(b9bb488f) · s902 `epoch40_69.84`(7c84ba24) — NAS `ckpts_backup/jarvis/ReliaDINO/jarvis_deliver_rgbdel_P46_c3only_seed*_screen40_A2L7basicdist/`(jarvis 원 학습 폴더의 pth 는 삭제돼 있었다).
- L5 s821 · s902 는 참고.
- L7 의 모델 블록은 A2 평가 config 와 같다(yaml 대조). 평가 config `configs/eval/a2_eval1024_jarvis.yaml`, jarvis GPU4~7.
- 산출물 NAS `analysis_logs/degr_axes_20261119/`(케이스 json·집계 스크립트·레인 스크립트), 원본 jarvis `/SSDb/jemo_maeng/analysis_logs/degr_axes_20261119/`.

## ③ DELIVER test 내장 센서 고장 케이스별 mIoU (전 센서 입력, legal v2) — 완료
- 케이스 = 장면 폴더 이름 접미사(`tools/baseline_failure/condition_delta.py` 와 같은 규칙), 케이스별 mIoU = 그 케이스 이미지들의 혼동행렬 합에서 클래스 IoU 평균. 집계 스크립트 `case_miou.py`(NAS 같은 폴더).
- 입력과 검산:
  - A2 = paper_qualbank `deliver_test/scores/A2s821·A2s902.npz`(이미지별 혼동행렬). 합계 58.6138·57.0919 = 등록 legal v2 와 일치.
  - L7 = jarvis 새 legal v2 test 덤프(`tools/baseline_failure/dump_preds_ours.py`). 전체 58.17·57.25 = 등록값과 일치.
  - L5 = levine `fill_20261111/L5_s*_test`(legal v2 덤프). 전체 57.42·56.63.

| 케이스 | 장수 | A2 s821 | A2 s902 | L7 s821 | L7 s902 | Δ s821 | Δ s902 | Δ 2시드 평균 | 부호 일치 | 참고 L5 s821 / s902 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 전체 | 1897 | 58.61 | 57.09 | 58.17 | 57.25 | +0.44 | -0.16 | **+0.14** | 아니오 | 57.42 / 56.63 |
| 고장 없음 | 1198 | 58.48 | 57.02 | 58.28 | 57.13 | +0.20 | -0.11 | **+0.05** | 아니오 | 57.42 / 56.42 |
| motion blur | 300 | 56.10 | 54.14 | 53.41 | 54.45 | +2.68 | -0.30 | **+1.19** | 아니오 | 54.72 / 54.33 |
| overexposure | 100 | 56.23 | 55.80 | 55.70 | 56.15 | +0.53 | -0.35 | **+0.09** | 아니오 | 54.52 / 55.35 |
| underexposure | 100 | 54.78 | 54.13 | 53.12 | 54.18 | +1.67 | -0.05 | **+0.81** | 아니오 | 53.66 / 53.90 |
| LiDAR jitter | 99 | 58.20 | 57.52 | 58.13 | 58.65 | +0.06 | -1.13 | **-0.53** | 아니오 | 57.39 / 57.44 |
| event low-res | 100 | 59.37 | 58.09 | 59.08 | 58.58 | +0.29 | -0.50 | **-0.10** | 아니오 | 57.13 / 57.54 |

## ② NM 벤치 정의 test (진행 중)
- A2 값 = DRN-261111-02 ②(drone-demo batch 8). L7 = 같은 도구 `tools/mm_eval_v2.py --protocol nm --nm_mode bench`, batch 8, seed 0, 전수(원본 벤치와 비트 대조 통과한 정의, develop c2ddd22 기준 동일).

| | Low | Mid | High |
| --- | ---: | ---: | ---: |
| A2 s821 | 50.40 | 46.47 | 38.36 |
| L7 s821 | 49.76 | 45.84 | 38.40 |
| Δ s821 | +0.64 | +0.63 | −0.04 |
| A2 s902 | 50.81 | 47.37 | 39.67 |
| L7 s902 | 진행 중 | | |

## ① test RMM 15조합 (진행 중)
- A2 값 = DRN-261111-02 ①: s821 r=.75/.5/.25 52.68/53.95/55.07 · s902 52.21/53.71/54.81.
- L7 = 부분집합 14 레인 × 2시드(A2 와 같은 분할 방식, MM_PRESENT_ONLY, seed 0), 진행 중.
