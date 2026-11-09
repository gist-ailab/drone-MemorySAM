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
표 I~III 의 선행 방법 값은 모두 원 논문 표에서 옮겼다(원문 PDF 쪽 순번, 원문 확인 = notes/sota_survey.md(2026-11-02 조사, 메인 세션이 CAFuser·DGFusion·StitchFusion·FuseAdapt 표를 paper2md 변환본으로 재확인)). 쪽 = PDF 쪽 순번.

| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| (a) CMNeXt / StitchFusion(MiT-B2) / GeminiFusion val·test | 66.3·53.0 / 68.2·53.4 / 66.9·54.5 | CAFuser(arXiv 2410.10791) Table III, p6 — test 는 CAFuser 저자 측정(표 각주 a). DGFusion Table III p6 test 같은 값 | ✅ 원문 재확인 |
| (a) CAFuser-CAA val·test | 68.6 · 55.2 | CAFuser Table III, p6 | ✅ 원문 재확인. 옛 초안 68.79 는 출처 없음 → 삭제 |
| (a) CAFuser val·test | 67.8 · 55.6 | CAFuser Table III, p6 | ✅ 원문 재확인 |
| (a) DGFusion test | 56.7 | DGFusion(arXiv 2509.09828) Table III, p6 (val 미보고) | ✅ 원문 재확인. 옛 56.71 은 headline.yaml 의 재측정값 |
| (a) DGFusion, official weights test | 56.73 | judgment-ledger 2026-10-01 DRN-261001-03 행(표 IV(b) 와 같은 값) | ✅ |
| (b) MemorySAM / MLE-SAM val | 65.38 / 64.08 | MemorySAM(2503.06700) Table 1 p6 / MLE-SAM(2412.04220) Table I p7; split 미표기(각주 c) | ✅ 조사 파일 |
| (b) OmniSegmentor MiT-B2 / DFormer-L | 67.5 / 68.0 | OmniSegmentor(2509.15096) Table 1(f), p6; split 미표기 | ✅ 조사 파일 |
| (b) HyperDUM | 67.59 | HyperDUM(2503.20011) Table 4, p6, 본문 "validation set" | ✅ 조사 파일 |
| (b) MAGIC | 67.66 | MAGIC(2407.11344) Table 1, p10 | ✅ 조사 파일 |
| (b) Any2Seg | 68.25 | Any2Seg(2407.11351) Table 1, p9 | ✅ 조사 파일 |
| (b) FuseAdapt DINOv2-B | 68.33 | FuseAdapt(2610.21779) Table 3, p7; split 미표기 | ✅ 조사 파일 |
| (b) StitchFusion Swin-T | 70.3 | StitchFusion(2408.01343) Table 1, p6 (Table 3 p7 은 70.34); split 미표기 | ✅ 원문 재확인 |
| (c) MM SAM-adapter RGB-E / RGB-L / RGB-D val·test | 60.74·55.70 / 61.89·57.14 / 69.60·57.35 | MM SAM-adapter(2509.10408) Table 7 p11(val), Table 4 p10(test) | ✅ 조사 파일 |
| Ours test (표에는 이 3시드 평균만 싣는다, user 2026-11-02) | 57.74 | `headline.yaml:22`; 시드별 = `.claude_logs/tasks/DRN-261018-01/analysis.md` | ✅ |
| (기록만, 원고 미게재) Ours best seed test | 58.61 | `headline.yaml:23` (시드 821). user 2026-11-02 결정: 최고 시드는 원고에 넣지 않고 3시드 평균을 기본으로 한다. Table 7·8 의 58.61 은 "시드 821 단일 런" ablation 행으로 따로 쓰인다 | 기록 |
| Ours val (3시드 평균 70.17 만 게재; best seed 69.59 는 기록만) | 70.17 / 69.59 | `tasks/DRN-261019-03/analysis.md:23` | ✅ |
| 본문 격차 1.0 / 1.01 / 1.6 | — | 57.74−56.7, 57.74−56.73, 70.17−68.6 (최고 시드 격차 1.9 는 원고에서 뺐다) | ✅ 계산 |

## 표 II. MUSES (`tab:muses`)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| (a) Mask2Former Swin-T / SegFormer B2 / B5 / Mask2Former Swin-L test | 70.7 / 72.5 / 74.7 / 77.1 | MUSES(arXiv 2401.12761) Table 4, p13 | ✅ 조사 파일 |
| (b) CMNeXt / GeminiFusion test | 72.4 / 75.3 | CAFuser Table II, p5 (CAFuser 저자 측정; DGFusion Table II p6 은 CMNeXt 72.1) | ✅ 원문 재확인(DGFusion 표) |
| (b) CAFuser / CAFuser-CAA / DGFusion test | 78.2 / 78.5 / 79.5 | CAFuser Table II p5; DGFusion Table II p6 | ✅ 원문 재확인 |
| (c) FuseAdapt DINOv2-S RGB-L-E val | 77.28 | FuseAdapt Table 2, p7 (부록 p17 best-validation) | ✅ 원문 재확인 |
| (c) MM SAM-adapter RGB-E / RGB-L test | 79.92 / 81.07 | MM SAM-adapter Table 6·8, p11(서버 제출 명시) | ✅ 조사 파일 |
| GtA 82.39 | 삭제 | 원 논문을 찾지 못함(조사 파일 0절) | ❌ 표에서 뺌 |
| Ours val / test | 비어 있음 | MUSES 4모달 A2(DRN-261019-02) 학습 중 | ❌ |

## 표 III. MCubeS (`tab:mcubes`)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| MCubeSNet | 42.86 | MCubeSNet(CVPR 2022 CVF) Table 1(a), p6; 백본 ResNet-101(본문 p6) | ✅ 조사 파일 |
| ShareCMP (RGB-A-D) | 50.99 | ShareCMP(2312.03430) Table 4, p9 | ✅ 조사 파일 |
| MLE-SAM / CMNeXt / U3M / Any2Seg | 51.02 / 51.54 / 51.69 / 51.91 | MLE-SAM Table III p8 / CMNeXt Table 1(e) p6 / U3M Table I p6 / Any2Seg Table 2 p9 | ✅ 조사 파일 |
| Cao et al. (BPCF) / StitchFusion Swin-T / MemorySAM / MAGIC / MMSFormer | 52.22 / 52.3 / 52.88 / 53.01 / 53.11 | 2610.17800 Table 8 p19 / StitchFusion Table 1 p6 / MemorySAM Table 2 p6 / MAGIC Table 1 p10 / MMSFormer Table 1 p6 | ✅ 조사 파일(StitchFusion 원문 재확인) |
| StitchFusion MiT-B4 / Swin-L | 53.92 / 55.9 | StitchFusion Table 7 p8 / Table 1 p6 | ✅ 원문 재확인 |
| FuseAdapt DINOv2-B | 57.62 | FuseAdapt Table 1 p6, 본문 p8 | ✅ 원문 재확인 |
| Mul-VMamba 54.65 | 삭제 | 원문 미확보(조사 파일 0절) | ❌ 표에서 뺌 |
| Ours / Ours w/o C2D2 | 비어 있음 | DRN-261017-02(C2D2 학습 중) / DRN-261031-01(정식 평가 대기) | ❌ |

## 표 V. Robustness (`tab:robust`) — 2026-11-02 arXiv 2503.18445 벤치 정의로 교체(user), S&P 단독 수치 삭제
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| (a) CMNeXt / GeminiFusion / MAGIC / MAGIC++ / StitchFusion clean | 66.33 / 66.92 / 66.10 / 67.34 / 68.20 | arXiv 2503.18445 Table 3, RDEL 열(MiT-B2, DELIVER val), PDF 텍스트 변환본(paper2md)에서 메인 세션 확인 | ✅ |
| (a) EMM Avg / E(.2) / E(.1) / E(.05) | CMNeXt 37.90/54.46/60.41/63.38 · GeminiFusion 37.07/54.33/60.62/63.77 · MAGIC 44.97/58.66/62.68/64.47 · MAGIC++ 44.85/59.18/63.52/65.50 · StitchFusion 41.98/58.02/63.29/65.80 | 같은 논문 Table 7 | ✅ |
| (a) RMM Avg r=.75 / .50 / .25 | CMNeXt 42.17/47.49/53.61 · GeminiFusion 39.78/42.41/49.74 · MAGIC 45.30/48.19/51.61 · MAGIC++ 47.06/49.31/53.92 · StitchFusion 45.16/48.33/53.10 | Table 8 / 9 / 10 (mIoU_RMM^Avg) | ✅ |
| (a) NM Low / Mid / High | CMNeXt 35.23/16.37/2.31 · GeminiFusion 5.52/4.14/3.30 · MAGIC 24.03/13.31/3.98 · MAGIC++ 33.12/18.31/8.70 · StitchFusion 24.91/17.11/9.25 | Table 11, mIoU_NM 행 | ✅ |
| (삭제 2026-11-04: 공개 코드 미확인, user "열려 있는 것만") (b) EQUISeg clean / EMM Avg / E(.10) / RMM r=.50 Avg / NM Low / Mid | 67.90 / 48.22 / 65.75 / 50.96 / 34.87 / 19.13 | arXiv 2509.24505 Table 1 p4 (DeLiVER), 메인 세션이 변환본에서 재확인. 기준선 행이 2503.18445 Table 9(r=0.5)와 같아 RMM 은 r=.50 열. E(.20)·E(.05)·r=.75/.25·NM High 미보고(--) | ✅ |
| (b) 각주 RMMSS EMM / RMM | 49.93 / 51.90 | arXiv 2505.12861 Table 2 p6 (MiT-B0, r 미표기) — 변환본 378행 "Ours B0 49.93 51.90" 메인 세션 확인 | ✅ |
| 본문 9.6 | — | 57.85 − 48.22 (EQUISeg) | ✅ 계산 |
| (c) DGFusion 공식·재학습 (b)·CAFuser (b) | 비어 있음 | analysis inbox 20261102-180010(val 벤치 프로토콜 측정 요청) | ❌ |
| (원고에서 todo 로 내림 2026-11-04, user: 벤치 공개 코드와 같은 방식만) Ours clean / EMM Avg / RMM .75 .50 .25 — 자체 채점기(1042 native GT, flip 없음) | 70.17 / 57.85 / 61.28 63.35 65.53 | `.claude_logs/tasks/DRN-261019-03/analysis.md` val 행(A2 3시드 평균) | ✅ |
| (원고에서 todo 로 내림, 같은 이유) Ours NM Low / Mid / High — 자체 채점기, flip 없음 | 58.34 / 51.92 / 39.99 | DRN-261102-04 A2 3시드 val, nm_mode bench(2503.18445 식 7). hpca100 ~/SSDb/jemo_maeng/nmbench_20261102/s{821,902,903}_val/val/missing_modality/summary.json (md5 f98fb583·46f6fa86·153b8e3b), 시드별 57.84/59.29/57.88 · 51.52/52.79/51.45 · 39.63/40.21/40.12 평균. 메인 세션 직접 판독 2026-11-04(analysis 보고 전) | ✅ 원본 |
| Ours E(p) | 비어 있음 | inbox 20261102-180010(시드903 만 66.10/68.20/69.09) | ❌ |
| 본문 58.3/51.9/40.0 vs 35.2/19.1/9.3 | — | 위 Ours NM; 다른 행 최고 = Low CMNeXt 35.23, Mid EQUISeg 19.13, High StitchFusion 9.25 | ✅ 계산 |
| 본문 수치 12.9 / 61.3~65.5 / 47.1~53.9 / 12.3 / 21~30 | — | 57.85−44.97; ours RMM; 벤치 최고 RMM(.75 47.06, .25 53.92); 70.17−57.85; 벤치 모델 clean−EMM 21.13~29.85 | ✅ 계산 |
| 삭제(2026-11-02) | test S&P .05/.10/.20, DGFusion test EMM 31.14 등 | judgment-ledger 에 보존 | — |

## 표 IX. C2D2·고해상 강건성 (`tab:abl_robust`) — val 벤치 정의
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| ✓C2D2 ✗high-res (Q2 3시드) clean / EMM / RMM .75 .50 .25 | 67.91 / 56.87 / 60.69 62.46 64.08 | DRN-261019-03/analysis.md val 행 | ✅ |
| ✓ ✓ (A2) | 70.17 / 57.85 / 61.28 63.35 65.53 | 같은 문서 | ✅ |
| ✗ ✗ (E1 40ep) | 비어 있음 | inbox 20261102-180010 | ❌ |

## 그림 robust (a) — `figures/source_data/fig_robust_rmm.csv` (위 표 V 값과 같은 출처, 행마다 source 열)

## 표 V. Per-condition (`tab:condition`)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| “All” 열 5개 | 56.73 / 54.77 / 55.79 / 56.59 / 57.74 | 표 IV(b) clean 열과 같은 출처(첫 행 = 공식 가중치, ledger:130) | ✅ |
| 조건별 6열 | 비어 있음 | analysis 요청 inbox 20261028-183826 #1 | ❌ |

## 표 VI. Ablation (`tab:ablation`) — 시드 821, 40ep, legal v2 test@1024 — **원고에서 삭제(2026-11-04 user: C2D2·고해상 행이 tab:abl_robust 와 중복). 55.14 만 5절 본문에 남음**
| 행 | Clean / EMM | 출처 | 상태 |
|---|---|---|---|
| RGB only | 55.14 / – | `judgment-ledger.md:177` (2026-10-28, DRN-261020-08 adopt) | ✅ |
| + mean fusion | 비어 있음 | DRN-261020-07 학습 중 | ❌ |
| + cross-modal attention, query head | 55.94 / 비어 있음 | `status/current.md:55,60` (“4모달 E1 스크린821 … test 55.94”) | ✅ / ❌ |
| + C2D2 (two-pass, distillation) | 57.73 / 48.68 | `tasks/DRN-261019-03/analysis.md:18-19` (Q2 시드821) | ✅ |
| + high-res path (full) | 58.61 / 49.12 | 같은 문서 `:18-19` (A2 시드821) | ✅ |
| (삭제 2026-11-02, user 결정 inbox 20261102-191138) − sensor-specific LoRA(L1 공유 LoRA) | — | Access 원고에 싣지 않음. RA-L 별개 논문 재료(DRN-261020-09, 판정 대장 176행) | — |
| − cross-modal attention | 57.76 / 비어 있음 | `judgment-ledger.md:175` (DRN-261020-10; 트렁크 확장도 함께 빠짐) | ✅ / ❌ |
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

## 표 X. 추론 비용 (`tab:cost`) — DRN-261027-02 adopt(2026-11-03)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| DGFusion params / 학습 / GFLOPs / ms / GB | 78.88 / 78.88 / 695.6 / 88.9 / 1.83 | `.claude_logs/tasks/DRN-261027-02/analysis.md:18` | ✅ |
| CAFuser | 75.01 / 75.01 / 670.8 / 83.1 / 1.57 | 같은 문서 :19 | ✅ |
| Ours (A2) | 377.05 / 73.98 / 10,135 / 254.5 / 2.47 | 같은 문서 :16 | ✅ |
| 본문 74.0 / 75.0 / 78.9 / 14.6배 / 2.9배 / 약 4 img/s / 15.2M / 약 2% / 13% | — | 위 값; 고해상 = A2 − Q2(:17, 361.87M·9,922·224.8 ms); 1000/254.5 = 3.9 img/s. 같은 계열(동결 VFM) 공개 비용은 없음: FuseAdapt Table 15(arXiv 2610.21779 부록 D)는 MiT-B0 판만, MM SAM-adapter 는 미보고 | ✅ 계산 |

### 2026-11-04 벤치 공개 코드 대조 (user: 열린 것만, 같은 평가 방식)
- 공개 저장소 Chenfei-Liao/Multi-Modal-Semantic-Segmentation-Robustness-Benchmark: val_mm_EMM.py·val_mm_RMM.py 는 `preds = model(images)` 단일 forward(sliding=False), val_mm_NM.py 는 `sliding_predict(..., flip=True)`; 셋 다 `get_val_augmentation(IMAGE_SIZE)` 로 라벨까지 1024 Resize 후 Metrics 채점, split 'val'.
- 우리 자체 채점기(tools/missing_modality_eval.py)는 1042 native GT 채점·flip 없음 → 같은 방식 아님. 주입 위치(정규화 후)·S&P(텐서 max/min)·Gaussian(Event 제외)·RMM(원소 독립 r)·정규화(RGB ImageNet, 나머지 /255)는 같음.
- 공개 코드: EQUISeg 미확인(검색), RMMSS 공식 코드 미확인(후속 RobustSeg 저장소 표기만) → 표에서 제외.

## 본문 수치 — 서론 Contributions (`sections/1_intro.tex`, user 지시 2026-11-04, lab-inbox 20261104-145821-request-e754448d)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| 우리 DELIVER test (3시드 평균) | 57.74 | `.claude_logs/experiments/headline.yaml` benchmarks.deliver.ours[0] (A2, legal v2, test@1024) | ✅ |
| DGFusion DELIVER test | 56.7 → 격차 1.0 | DGFusion 원 논문 Table III p6 (notes/sota_survey.md) | ✅ |
| 우리 val EMM15 (3시드 평균) | 57.85 | `.claude_logs/tasks/DRN-261019-03/analysis.md` (자체 채점기: native GT, flip 없음) | ⚠️ 원고에는 todo 로만 표기 — 강건성 수치는 벤치 공개 코드 값만 쓴다(user 2026-11-04, 감사 #19·#26). 재측정 후 숫자로 교체 |
| MAGIC val EMM | 44.97 → 격차 12.9 | arXiv 2503.18445 (원고 Table robust (a)) | ✅ (우리 값 재측정 후 격차 재계산) |

## 표 abl_robust Test 열 추가 (2026-11-04, tab:ablation 삭제에 따름)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| C2D2 tr. ✓, high-res ✗, test 3시드 | 56.59 | `tasks/DRN-261019-03/analysis.md` R03:17 (Q2 3시드 legal v2 test@1024) | ✅ |
| C2D2 tr. ✓, high-res ✓, test 3시드 | 57.74 | `headline.yaml` benchmarks.deliver.ours[0] | ✅ |
| C2D2 ✗, high-res ✗, test 3시드 | 비어 있음 | E1 40ep 3시드 test 평균 정본 위치 미확인 | ❌ |

## 표 thin (`tab:thin`, `sections/6_analysis.tex`, 2026-11-04 user) — DELIVER test, legal v2, 3시드 평균
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| Ours w/o high-res: Pole·TrafficLight·Pedestrian·Static·Thin·mIoU | 51.94 · 41.67 · 76.94 · 33.36 · 50.98 · 56.59 | `.claude_logs/experiments/analysis/DRN-261018-01/a2_vs_q2_perclass_3seed.csv` q2s821/q2s902/q2s903 열 평균(25클래스 평균 56.59 = headline 과 일치) | ✅ |
| Ours: 같은 열 | 54.08 · 37.52 · 80.74 · 43.71 · 54.02 · 57.74 | 같은 CSV a2s821/a2s902/a2s903 열 평균(25클래스 평균 57.74 일치) | ✅ |
| CMNeXt·CAFuser·DGFusion (공개 가중치) | 비어 있음 | 전체 예측 덤프에서 클래스별 IoU 계산 요청(analysis) | ❌ |


## 2026-11-09 갱신 — 실험 결과 반영(원고 칸별)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| Table 5 (tab:robust) Ours: clean · EMM Avg · E(.20/.10/.05) · RMM r=.75/.50/.25 · NM Low/Mid/High | 71.34 · 58.52 · 67.15/69.46/70.45 · 61.92/64.05/66.28 · 58.63/52.00/39.87 | 벤치 공개 코드 그대로(`third_party/mmss_robust_bench`, 53c7a6e), DELIVER val 2005장, A2 s821/s902/s903 평균. NAS `analysis_logs/mmssb_table5_20261109/`, `.claude_logs/tasks/DRN-261027-01/log.md` 2026-11-09 행. 설정 검증: CMNeXt 공개 가중치 66.33/37.90/54.46/60.41/63.38 = 벤치 원문 | ✅ |
| Table 5 본문 격차 | 13.6 (58.52−44.97) · 낙폭 12.8 (71.34−58.52) vs 벤치 모델 21.1~28.4 | 같은 표 값 | ✅ |
| Table 3 (tab:muses) Ours val | 81.54 | `.claude_logs/tasks/DRN-261019-02/analysis.md` 공식 val 1536, s2 81.758 / s3 81.311 평균 81.535 (method 판정 전, 사전 규칙상 선택 해상도 1536) | ⚠️ 판정 대기 |
| Table 4 (tab:mcubes) Ours | 59.78 | `.claude_logs/tasks/DRN-261017-02/analysis.md` C2D2 768 s827 59.80 / s828 59.76, 하네스 재채점 Δ0.00 (판정 대기, test 로 체크포인트 선택은 §3 에 공개) | ⚠️ 판정 대기 |
| Table 8 (tab:abl_robust) (b) MCubeS w/o C2D2 / C2D2 | 59.89 / 59.78 | DRN-261104-03 lab-plan 이력(s827 59.81, s828 59.96, 학습기 top1 = test 102) / 위 행 | ⚠️ w/o 행은 재채점 요청 중(inbox 20261105-115201) |
| Table 7 (tab:ablation_loo) L6 (teacher KD 제거) s821 | 58.77 / 49.64 | `.claude_logs/tasks/DRN-261020-14/log.md` 2026-11-05 (EMM 은 pane 로그 49.6407, summary.json 직접 미확인) | ⚠️ |
| Table 7 L7 (물리 열화 제거) s821 | 58.17 / 48.86 | `.claude_logs/tasks/DRN-261029-01/log.md` 2026-11-05 (EMM 48.8631) | ✅ |
| Table 7 본문 둘째 시드 | L6 56.32 · L7 57.25 · full 57.09 (s902) | 같은 log.md · A2 s902 test 57.09 (DRN-261019-03 R03:17) | ✅ |
| Table 10 (tab:thin) 기준선 (Pole·TL·Ped·Static·Thin·mIoU) | CMNeXt 48.39·31.96·76.64·31.26·47.06·52.34 · CAFuser 60.64·46.21·82.93·33.91·55.92·55.78 · DGFusion 55.65·41.14·83.06·38.36·54.55·56.71 | NAS `analysis_logs/baseline_official_dump_20261104/class_iou_official.csv`, analysis report 20261104-185440 (1024 예측 → native GT 최근접, 전역 confusion) | ✅ |
| Fig.2(a) Ours 곡선 | 71.34 · 66.28 · 64.05 · 61.92 | Table 5 Ours 와 같은 측정. w/o high-res 곡선은 공개 코드 값이 없어 삭제 | ✅ |
| 초록·서론·결론 MCubeS | 59.8 / 59.78, 격차 2.2 (59.78−57.62) | Table 4 | ⚠️ 판정 대기 |
| 정성 그림 본문 격차(5절·6절) | top1 3.9~13.0 · 18장 3.4~13.0 · reverse1 8.5~12.9 (DGFusion 4, CAFuser 2) | `figures/source_data/qualbank/deliver_candidates.csv`(비교군 CMNeXt·CAFofficial·DGFofficial, 2026-11-09 재계산) | ✅ |

## 표 perclass (`tab:thin`, 2026-11-09 user: 25클래스 전체, Δ 순 정렬) — 행별 값

출처: 기준선 = NAS `analysis_logs/baseline_official_dump_20261104/class_iou_official.csv`(CMNeXt·CAFofficial·DGFofficial), Ours w/o high-res·Ours = `.claude_logs/experiments/analysis/DRN-261018-01/a2_vs_q2_perclass_3seed.csv` q2*/a2* 3시드 평균, Δ = Ours − 기준선 최고. 생성 스크립트 = paper 세션 perclass_table.py(행 정렬·굵게 수작업 표기).

| 행 (CMNeXt · CAFuser · DGFusion · Ours w/o high-res · Ours · Δ) |
|---|
| Wall · 3.72 · 2.82 · 4.03 · 13.20 · 15.31 · +11.28 |
| TrafficSign · 38.37 · 45.57 · 45.14 · 50.46 · 53.54 · +7.97 |
| Water · 2.02 · 0.17 · 0.00 · 4.79 · 8.80 · +6.77 |
| Truck · 77.43 · 89.35 · 89.15 · 93.73 · 94.90 · +5.54 |
| Static · 31.26 · 33.91 · 38.36 · 33.36 · 43.71 · +5.36 |
| TwoWheeler · 61.83 · 68.95 · 65.96 · 69.02 · 71.69 · +2.75 |
| Dynamic · 6.78 · 7.28 · 8.55 · 9.77 · 10.69 · +2.14 |
| Terrain · 59.75 · 68.96 · 63.59 · 69.05 · 70.89 · +1.94 |
| Building · 87.45 · 86.89 · 87.55 · 89.36 · 89.34 · +1.79 |
| Road · 96.47 · 96.67 · 96.52 · 97.21 · 97.67 · +0.99 |
| RoadLine · 81.42 · 77.08 · 77.50 · 82.06 · 82.36 · +0.94 |
| GroundRail · 77.23 · 79.56 · 80.59 · 80.23 · 81.49 · +0.90 |
| Bus · 94.34 · 93.69 · 94.18 · 94.35 · 95.08 · +0.75 |
| Vegetation · 82.46 · 83.75 · 83.56 · 83.86 · 84.49 · +0.74 |
| SideWalk§ · 67.64 · 78.49 · 78.06 · 76.11 · 78.85 · +0.36 |
| Cars · 93.74 · 94.40 · 94.61 · 93.89 · 94.80 · +0.19 |
| Bridge · 0.03 · 0.00 · 0.00 · 0.02 · 0.01 · -0.02 |
| Sky · 97.15 · 98.20 · 97.99 · 97.97 · 98.15 · -0.05 |
| Ground · 4.38 · 5.68 · 7.24 · 7.64 · 6.78 · -0.46 |
| Pedestrian · 76.64 · 82.93 · 83.06 · 76.94 · 80.74 · -2.31 |
| Fence · 48.04 · 52.59 · 53.59 · 49.69 · 50.46 · -3.13 |
| Other · 5.76 · 7.65 · 7.22 · 3.11 · 3.99 · -3.67 |
| Pole · 48.39 · 60.64 · 55.65 · 51.94 · 54.08 · -6.56 |
| TrafficLight · 31.96 · 46.21 · 41.14 · 41.67 · 37.52 · -8.68 |
| RailTrack · 34.22 · 33.00 · 64.46 · 45.41 · 38.25 · -26.21 |
| mIoU · 52.34 · 55.78 · 56.71 · 56.59 · 57.74 · +1.04 |

## 6절 클래스별 구조 해석 문장 (2026-11-09 user)
| 칸 | 값 | 출처 | 상태 |
|---|---|---|---|
| 고해상 경로 없이도 3시드 모두 앞선 클래스와 격차 | Wall 9.2 · TrafficSign 4.9 · Truck 4.4 · Water 2.8 · Building 1.8 · Dynamic 1.2 (Ours w/o high-res 3시드 평균 − 기준선 최고) | tab:thin 열 계산(paper 세션 decomp.py, 기준선 CSV + DRN-261018-01 CSV) | ✅ |
| 고해상 경로가 앞서게 만든 클래스 | Static·SideWalk·Cars·GroundRail (w/o 는 3시드 모두 기준선 아래, 경로 추가 후 평균이 기준선 위), Static +10.36 | 같은 계산 | ✅ |
| 특징 크기 집중도 · 끄기 비율 | (d) 고해상 2.52 vs (c) 탭 0.93 · 얇은4 Δ20.45 / 나머지 9.24 = 2.21 | `.claude_logs/tasks/DRN-261104-02/analysis.md` H-V1·H-V2 (A2 s821, test 1897) | ⚠️ 판정 대기(method) |

