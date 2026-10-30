# 표 수치 출처 (IEEE Access 원고)

규칙(user 2026-10-29): 원고의 **모든 표 수치**는 이 문서에 출처를 남긴다. 표를 고치면 같은 커밋에서 이 문서를 고친다.
- 경로는 repo 루트 기준. 줄 번호는 **커밋 `ae26766` 기준**이다(대장·문서에 행이 추가되면 줄이 밀릴 수 있으므로 날짜·행 제목·항목 id 를 같이 적는다).
- "약어"는 내부 실험 이름이다: E1 = clean 만으로 학습한 같은 구조(논문의 "Ours, clean training" / "w/o two-pass"), Q2 = 두 패스 학습 + 교사 증류, 고해상 경로 없음(논문의 "Ours w/o high-res"), A2 = Q2 + 고해상 경로(논문의 "Ours"). 원고 본문에는 이 약어를 쓰지 않는다.
- 상태: ✅ 정본에서 옮김 · ⚠️ 정본에 있으나 단서 있음 · ❌ 비어 있음(todo) · 📄 원 논문 표에서 옮겨야 함(아직 미확인).

## 표 0. Benchmarks (`tab:benchmarks`, `sections/3_sensors.tex`)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| DELIVER train/val/test | 3983 / 2005 / 1897 | CMNeXt 원논문(Zhang et al., CVPR 2023); repo 사본 `.claude_logs/research/novelty-and-related-work.md:78` | ⚠️ 원 논문 쪽 번호 미기재 |
| DELIVER classes 25 | 25 | `semseg/datasets/deliver.py:20-22` CLASSES | ✅ |
| MUSES train/val/test, 19 classes | 1500 / 250 / 750, 19 | `semseg/datasets/muses.py` docstring l.18, l.33 | ✅ |
| MCubeS test, 20 classes | 102, 20 | test 102: `.claude_logs/experiments/headline.yaml` mcubes.ours[0].task(“test split 102장”); 20: `semseg/datasets/mcubes.py:21` | ✅ |
| MCubeS train / val | 302 / 96 | MCubeS 원 논문(Liang et al., CVPR 2022, `mcubes2022`) 7쪽 “Training Details”: “Training, validation, and test sets contain 302, 96 and 102 images” (CVF open access PDF, 2026-10-30 확인). 우리 서버 `list_folder/*.txt` 줄 수는 yeon·lecun 연결 불가로 미확인. 우리 로더는 평가에 test.txt 를 쓰므로 96장 val 은 학습·선택에 쓰이지 않음(`semseg/datasets/mcubes.py:142`) | ✅ (체크포인트는 관행대로 test 102 로 선택, user 결정 2026-10-30, 원고 §3·표 III 캡션에 명시) |

## 표 I. DELIVER (`tab:deliver`, `sections/5_experiments.tex`)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| CAFuser-CAA val | 68.79 | `.claude_logs/experiments/headline.yaml:17` (deliver.sota_val) | ✅ (📄 원 논문 표 쪽 번호 필요) |
| DGFusion test | 56.71 | `headline.yaml:15` (deliver.sota) | ✅ (📄) |
| DGFusion backbone | Swin-T | `headline.yaml:15` note | ✅ |
| MM SAM-adapter val / test | 69.60 / 57.35 | `headline.yaml:18` / `:16` (sota2_val / sota2) | ✅ (📄) |
| Ours (3-seed mean) test | 57.74 | `headline.yaml:22` (deliver.ours[0].value); 시드별 = `.claude_logs/tasks/DRN-261018-01/analysis.md` 3시드 표(“A2 25클래스” 행) | ✅ |
| Ours (best seed) test | 58.61 | `headline.yaml:23` (best, 시드 821) | ✅ |
| Ours val (3-seed mean / best seed) | 70.17 / 69.59 | `tasks/DRN-261019-03/analysis.md:23` (val clean 69.59 / 71.05 / 69.86, 평균 70.17; best seed = test 기준 시드 821 → 69.59). method 답변(inbox 20261029-181016 #3): mm_eval_v2 clean val 을 legal v2 val 로 인용 가능 | ✅ |
| CMNeXt, StitchFusion, GeminiFusion, CAFuser 전 칸 | 비어 있음 | 후보값만 repo `research/novelty-and-related-work.md:83`(2차 사본) | ❌ 📄 |

## 표 II. MUSES (`tab:muses`)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| DGFusion test | 79.5 | `headline.yaml:84` (muses.sota) | ✅ (📄) |
| GtA test | 82.39 | `headline.yaml:85` (muses.sota2) | ⚠️ 원 논문 미확인(Codabench 로그인 장벽, `research/novelty-and-related-work.md` 2026-09-08 메모) |
| MM SAM-adapter test | 81.07 | `headline.yaml:99` (gap_2nd 문구) | ⚠️ 원 논문 표 미확인 |
| Ours val / test | 비어 있음 | MUSES 4모달 A2(DRN-261019-02) 학습 중 | ❌ |
| CAFuser 전 칸, 기준선 val | 비어 있음 | — | ❌ 📄 |

## 표 III. MCubeS (`tab:mcubes`)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| StitchFusion test | 55.9 | `headline.yaml:109` (mcubes.sota) | ✅ (📄) |
| Mul-VMamba test | 54.65 | `headline.yaml:122` (gap_2nd 문구 “published 최고 Mul-VMamba 54.65”) | ⚠️ 원 논문 미확보(bib 없음) |
| Ours test | 비어 있음 | MCubeS A2(DRN-261017-02) 대기. headline 의 58.07 은 이전 세대(P39.1)라 쓰지 않음 | ❌ |

## 표 IV. Robustness (`tab:robust`)
(a) 공개 벤치 정의, val
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| MAGIC EMM / RMM .25 .50 .75 | 44.97 / 51.61 48.19 45.30 | arXiv 2503.18445(벤치마크 논문) 표 7(mIoU_EMM^Avg)·표 10/9/8(mIoU_RMM^Avg, r = .25/.5/.75), MiT-B2, val. 원문 6쪽에서 직접 확인(alphaXiv, 2026-10-29) | ✅ |
| MAGIC++ EMM / RMM .25 .50 .75 | 44.85 / 53.92 49.31 47.06 | 같은 표 | ✅ |
| (제외) RMMSS 49.93 | — | 결측 입력을 건너뛰는 계열이라 0 채움 표에서 제외(FU-06, inbox 20261029-174230) | — |
| Ours EMM | 57.85 | `tasks/DRN-261019-03/analysis.md:24` (val EMM15 평균, A2 3시드) | ✅ |
| Ours RMM .25/.50/.75 | 65.53 / 63.35 / 61.28 | 같은 문서 `:25-27` | ✅ |

(b) 같은 프로토콜, test
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| DGFusion (official weights) clean / EMM / S&P .05 .10 .20 | 56.73 / 31.14 / 39.12 16.85 4.53 | `experiments/judgment-ledger.md:130` (2026-10-01 DRN-261001-03, 공식 가중치 md5 0f40a641…). S&P 칸은 **계산값** = clean 56.73 − 대장의 “NM S&P 저하 17.61/39.88/52.20” | ✅ (S&P ⚠️ 계산값) |
| DGFusion (reproduced) clean / EMM / S&P | 55.68 / 31.53 / 37.97 16.71 5.52 | `experiments/judgment-ledger.md:117` (2026-10-01 행 “기준선 DGFusion (a)·(b) 강건 프로토콜”, (a) 재현 학습 80k val-best, summary.json md5 e6b5ab1f). 10-29 이전 원고는 이 행을 “official”로 잘못 표기(FU-06 정정) | ✅ |
| DGFusion (retrained) | 54.77 / 45.52 / 48.89 40.51 23.34 | 같은 행, (b) (md5 21f14917) | ✅ |
| CAFuser (retrained) clean / EMM | 55.79 / 46.50 | `status/current.md:63` (“CAFuser (b) 완료(10-04)”) | ✅ |
| CAFuser (retrained) S&P | 비어 있음 | — | ❌ |
| Ours w/o C2D2 EMM | 40.01 | `judgment-ledger.md:117` 같은 행 “E1 스크린 3시드: EMM 40.01” | ✅ |
| Ours w/o C2D2 clean, S&P | 비어 있음 | 같은 행에는 저하폭(3.09·5.73·11.22)만 있음 | ❌ |
| Ours w/o high-res clean / EMM | 56.59 / 48.06 | `tasks/DRN-261019-03/analysis.md:18-19` (Q2 3시드 평균) | ✅ |
| Ours w/o high-res S&P .05 .10 .20 | 53.25 / 51.64 / 48.79 | 같은 문서 `:20-22` | ⚠️ 2시드 값(시드821 값 54.10 은 NAS 원본 없음, 같은 문서 `:10`) |
| Ours clean / EMM / S&P | 57.74 / 48.66 / 53.11 51.36 47.85 | 같은 문서 `:18-22` (A2 3시드 평균) | ✅ |
| 노이즈 측정 모드 | S&P 단독(본문 서술) | `tools/missing_modality_eval.py` docstring l.26-29 기본 모드 | ⚠️ 실제 실행 모드 확인 요청(method inbox 20261029-113217 #3) |

## 표 V. Per-condition (`tab:condition`)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| “All” 열 5개 | 56.73 / 54.77 / 55.79 / 56.59 / 57.74 | 표 IV(b) clean 열과 같은 출처(첫 행 = 공식 가중치, ledger:130) | ✅ |
| 조건별 6열 | 비어 있음 | analysis 요청 inbox 20261028-183826 #1 | ❌ |

## 표 VI. Ablation (`tab:ablation`) — 시드 821, 40ep, legal v2 test@1024
| 행 | Clean / EMM | 출처 | 상태 |
|---|---|---|---|
| RGB only | 55.14 / – | `judgment-ledger.md:177` (2026-10-28, DRN-261020-08 adopt) | ✅ |
| + mean fusion | 비어 있음 | DRN-261020-07 학습 중 | ❌ |
| + cross-modal attention, query head | 55.94 / 비어 있음 | `status/current.md:55,60` (“4모달 E1 스크린821 … test 55.94”) | ✅ / ❌ |
| + C2D2 (two-pass, distillation) | 57.73 / 48.68 | `tasks/DRN-261019-03/analysis.md:18-19` (Q2 시드821) | ✅ |
| + high-res path (full) | 58.61 / 49.12 | 같은 문서 `:18-19` (A2 시드821) | ✅ |
| − sensor-specific LoRA | 58.84 / 비어 있음 | `judgment-ledger.md:176` (DRN-261020-09) | ✅ / ❌ |
| − cross-modal attention | 57.76 / 비어 있음 | `judgment-ledger.md:175` (DRN-261020-10; 트렁크 확장도 함께 빠짐) | ✅ / ❌ |
| − query head and arbiter | 57.89 / 비어 있음 | `judgment-ledger.md:174` (DRN-261020-12) | ✅ / ❌ |
| − degradation pass / − distillation | 비어 있음 | DRN-261020-13 / -14 평가 대기 | ❌ |

## 표 VII. Cost (`tab:cost`)
전 칸 비어 있음 — DRN-261027-02(비용표 측정) 대기. ❌

## 보충 표 S1. Depth dependence (`sections/supp_tables.tex`)
그림 Fig. dependence 와 같은 수치. 출처는 `figures/source_data/fig_dependence.csv` 의 source 열(`judgment-ledger.md:112`, 2026-09-30 행).
| 행 | Sun · Cloud · Rain · Fog · Night · Low exp. | 출처 | 상태 |
|---|---|---|---|
| DGFusion (reproduced) | 23.1 · 24.9 · 27.6 · 25.9 · 32.6 · 37.1 | `judgment-ledger.md:112` “DGFusion (a) test 이미지 평균 축 … −23.1 · −24.9 · −27.6 · −25.9 · −32.6 · −37.1” ((a) = 재현본) | ⚠️ test split(다른 행은 val) |
| Ours w/o C2D2 | 23.5 · 23.9 · 23.7 · 26.8 · 28.7 · 38.7 | 같은 행 “E1 확정 −depth” | ✅ |
| Ours w/o high-res | 3.9 · – · – · – · 10.1 · 13.4 | 같은 행 “Q2 시드821 −depth” | ✅ (cloud/rain/fog 없음) |
| DGFusion·CAFuser (retrained), Ours | 비어 있음 | analysis 요청 inbox 20261028-183826 #4 | ❌ |

## 본문·초록에 직접 쓴 수치
| 수치 | 위치 | 출처 |
|---|---|---|
| 57.74 / 58.61, 발표치 DGFusion 대비 +1.03 / +1.90, MM SAM-adapter 대비 +0.39 | §5 | `headline.yaml:22-23,31-32` |
| 같은 프로토콜 격차 +1.01(57.74 − 56.73), 초록 “one point above … four-sensor”, 31.1(공식 EMM), 46.5(재학습 기준선 최대 = CAFuser) | 초록·§1·§5 | `judgment-ledger.md:130`, `status/current.md:63` |
| val 70.17, CAFuser-CAA 대비 +1.4 | §5 | `tasks/DRN-261019-03/analysis.md:23`, `headline.yaml:17` |
| val EMM 57.85, MAGIC 대비 +12.9 | §5 | `tasks/DRN-261019-03/analysis.md:24`, arXiv 2503.18445 표 7 |
| depth 제거 하락(조건별 우리 w/o C2D2 23.5~38.7, DGFusion 재현본 23.1~37.1), C2D2 3.9 / 10.1 / 13.4 | §6 | `judgment-ledger.md:112` |
| depth 통째 결측: 재현본 −33.3 → 재학습 −4.7, 공식 가중치 −33.4 | §6 | `judgment-ledger.md:117`, `:130` |
| 한 센서만: 공식 가중치 7.5 / 43.3 / 1.1 / 1.1, 재학습 49.0 …, Q2 53.9 | §5 | `judgment-ledger.md:130`, `:117` |
| 고해상 경로 +1.15(56.59→57.74), 얇은 4클래스 +3.04, 클래스별 Static +10.35 · Pedestrian +3.80 · TrafficSign +3.08 · Pole +2.14 · TrafficLight −4.14 · RailTrack −7.16, RailTrack Q2 58.46/32.63/45.13 | §6 | `tasks/DRN-261019-03/analysis.md:18`, `tasks/DRN-261018-01/analysis.md`(얇은4 +3.04), `.claude_logs/experiments/analysis/DRN-261018-01/a2_vs_q2_perclass_3seed.csv`(d_mean3·q2 열) |
| 재현율·정밀도(기둥 0.75→0.78, 보행자 0.89→0.90, 신호등 0.80→0.85, 정밀도 0.46→0.41) | §6 | NAS `analysis_logs/a2_thin_anatomy_20261018/summary.md` (2시드) — ⚠️ 정본 대장에 없음, 원고에 확인 todo |
| 신호등 오탐 장면 화소 수 | §6 주석 | NAS `analysis_logs/a2_tl_fp_scenes_20261022/README.md` |
| 라벨 리샘플링 차이 1.2~1.3점 | §3 | `.claude_logs/issues/issues-and-fixes.md:25` (ISSUE-036) |
| 학습 설정(40ep, AdamW 6e-4, 768 crop, p 0.5, 결측 0.2, KD 0.5/T 2, 커리큘럼 0.3/0.6/1.0) | §4·§5 | `configs/lecun-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2hires.yaml` |
