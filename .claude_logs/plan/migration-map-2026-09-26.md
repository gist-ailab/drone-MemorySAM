# lab-plan 이관 대조표 (2026-09-26)

원문 위치 → lab-plan 항목. 원문 문서는 지우지 않았다. 이 표는 이관 누락 점검용이며 다시 생성하지 않는다(정본은 plan.yaml).

## 1. plan.md 대기열·완료 절(1차 이관)

| 원문 키 | DRN |
|---|---|
| E1conf | DRN-260926-01 |
| MUSESfull | DRN-260926-02 |
| ELoRA | DRN-260926-03 |
| E17 | DRN-260926-04 |
| NE18 | DRN-260926-05 |
| NE19 | DRN-260926-06 |
| ND0 | DRN-260926-07 |
| NE23 | DRN-260926-08 |
| NUSER | DRN-260926-09 |
| NMC | DRN-260926-10 |
| N1 | DRN-260926-11 |
| N2 | DRN-260926-12 |
| N3 | DRN-260926-13 |
| N4 | DRN-260926-14 |
| N4b | DRN-260926-15 |
| N6 | DRN-260926-16 |
| N7 | DRN-260926-17 |
| N8 | DRN-260926-18 |
| N9 | DRN-260926-19 |
| P52audit | DRN-260926-20 |
| P50rep | DRN-260926-21 |
| EBBA1 | DRN-260926-22 |
| MUphys | DRN-260926-23 |
| DLVpc | DRN-260926-24 |
| S0 | DRN-260926-25 |
| Router | DRN-260926-26 |
| DAILY | DRN-260926-27 |
| S1 | DRN-260926-28 |
| S3 | DRN-260926-29 |
| P50diag | DRN-260926-30 |
| P52 | DRN-260926-31 |
| P50EXT | DRN-260926-32 |
| N5 | DRN-260926-33 |
| A100u | DRN-260926-34 |
| A100p51 | DRN-260926-35 |
| E4b | DRN-260926-36 |
| B0s2 | DRN-260926-37 |
| E1s2 | DRN-260926-38 |
| P40 | DRN-260926-39 |
| P39rad | DRN-260926-40 |
| P47D1 | DRN-260926-41 |
| P49 | DRN-260926-42 |
| P49a | DRN-260926-43 |
| Pr7B | DRN-260926-44 |
| AB | DRN-260926-45 |
| TTA | DRN-260926-46 |
| Proj | DRN-260926-47 |
| ModAbl | DRN-260926-48 |
| DetDiag | DRN-260926-49 |
| P37 | DRN-260926-50 |
| MUs825 | DRN-260926-51 |
| P43 | DRN-260926-52 |
| MU2m | DRN-260926-53 |

## 2. 판정 대장 추가 실험·카드·제안서(2차)

| 원문 | 제목 | DRN |
|---|---|---|
| extra | Q0 · E1 확정 강건성 전 프로토콜(EMM 15조합·RMM 4부분집합×3비율·NM) | DRN-260926-54 |
| extra | Q2→Q3 강건성 전 프로토콜(EMM15·RMM4×3·NM3, Q2 먼저) | DRN-260926-55 |
| extra | Q2noDeg · 원천분리: DEGRADE_P=0, KD 유지(증류만의 순기여) | DRN-260926-56 |
| extra | E1-shared · E1 레시피 + 전 센서 공유 LoRA r16 40ep 스크린 3시드 | DRN-260926-57 |
| extra | CAFuser (b) · 열화 커리큘럼 재학습(detectron2) — 재개 대기 | DRN-260926-58 |
| extra | D4/D6 · 기준선 실패 분석(DGFusion·CAFuser·우리 모델 모달 zero-out·조건별 Δ·깊이 구간·셀 지도·원거리 원인) | DRN-260926-59 |
| extra | Q1 · 품질 헤드 실현성 프로브(Q1 5ep·Q1b-1 15ep·Q1b-2 연산자별 분해) | DRN-260926-60 |
| extra | R1 · depth 경계 prior refinement 40ep 스크린 3시드 | DRN-260926-61 |
| extra | R2 · 연결 성분 soft-IoU 손실 40ep 스크린 | DRN-260926-62 |
| extra | E1 40ep 스크린 3시드 짝 기준선 | DRN-260926-63 |
| extra | E3b · E3 + 센서 간 prototype 일치 항(AGREE_LAMBDA 0.1) | DRN-260926-64 |
| extra | E4c · 혼동 쌍 margin(RailTrack 3쌍·MARGIN 0.25) | DRN-260926-65 |
| extra | E5 · 변형 가능 다중스케일 픽셀 디코더(MSDeformAttn 3층) | DRN-260926-66 |
| extra | E6 · 매 블록 센서 간 교환 어댑터(S3) | DRN-260926-67 |
| extra | E8 · 클래스 표적 copy-paste 증강 | DRN-260926-68 |
| extra | E10 · 상위 절반 블록 부분 FT(S4) | DRN-260926-69 |
| extra | E17x · E17 통과 시 ViT-Adapter extractor 결합 | DRN-260926-70 |
| extra | E20 · 공유 A + 모달 B + 공유 LoRA (D0-i ≥ 0.9일 때만) | DRN-260926-71 |
| extra | E21 · 확률적 분할-B LoRA + 병합 | DRN-260926-72 |
| extra | E22 · 라벨 없는 조건 게이트 | DRN-260926-73 |
| extra | 마지막 2~4블록 MLP LoRA / 부분 해동 | DRN-260926-74 |
| extra | Q3+R · QAF 위에 depth 경계 refinement + 연결 성분 손실 결합 카드 | DRN-260926-75 |
| extra | Q4 · depth-anchor 비대칭 D (Q3 통과 후) | DRN-260926-76 |
| extra | Q5 · 품질 조건부 잔차 LoRA M (E-LoRA 판정 + Q3 통과 후) | DRN-260926-77 |
| extra | P49 · AIR 비대칭 주입 | DRN-260926-78 |
| extra | P50 · MAP 모달 정렬 사전학습 | DRN-260926-79 |
| extra | P51 · CMLC 교차모달 LoRA 결합 | DRN-260926-80 |
| extra | condexpert 어댑터 프로브 | DRN-260926-81 |
| extra | H10 재판정 실험 요청 | DRN-260926-82 |
| extra | 공간 모달 오라클 프로브 | DRN-260926-83 |
| extra | P36 노벨티 비판 검토 | DRN-260926-84 |
| extra | [대기 트리거] P37b-DELIVER 완주 시 → P34 per-class 비교 분석 | DRN-260926-85 |
| extra | [대기 트리거] yeon P37b-det 완주 → P38-det 자동 기동 | DRN-260926-86 |

## 3. registry.md 전 행(run 단위)

| 원문 줄 | 제목 | DRN | 상태 |
|---|---|---|---|
| registry.md:18 | [registry] levine_multiaqua_rgbtl_P9_hardaug8_physaug (ep131) | DRN-260926-87 | judging |
| registry.md:19 | [registry] levine_multiaqua_rgbtl_P22_hardaug8_physaug (ep120) | DRN-260926-88 | judging |
| registry.md:20 | [registry] levine_multiaqua_rgbtl_P21_hardaug8_physaug (ep94) | DRN-260926-89 | judging |
| registry.md:26 | [registry] dgfusion_swin_tiny_bs8_200k_deliver_clde | DRN-260926-90 | judging |
| registry.md:27 | [registry] dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade | DRN-260926-91 | judging |
| registry.md:28 | [registry] cafuser_swin_tiny_bs6_267k_deliver_clde_lecun | DRN-260926-92 | judging |
| registry.md:29 | [registry] b200_deliver_rgbdel_P31_physaug | DRN-260926-93 | judging |
| registry.md:30 | [registry] ANALYSIS: P29·P31·P32·P34 표준분석 (lecun) | DRN-260926-94 | judging |
| registry.md:31 | [registry] b200_deliver_rgbdel_P29_physaug | DRN-260926-95 | judging |
| registry.md:32 | [registry] b200_deliver_rgbdel_P30_physaug | DRN-260926-96 | judging |
| registry.md:33 | [registry] P28 RBMA seg (B200 RUN-1) | DRN-260926-97 | withdrawn |
| registry.md:34 | [registry] ANALYSIS: P37a-CEFR MUSES 출력분석 (yeon) | DRN-260926-98 | judging |
| registry.md:35 | [registry] ANALYSIS: P37a MUSES 표준분석 + 종합 실패-키 (yeon) | DRN-260926-99 | judging |
| registry.md:36 | [registry] hpca100_deliver_rgbdel_P39_dpc | DRN-260926-100 | judging |
| registry.md:37 | [registry] jarvis_muses_rgbel_P39_dpc | DRN-260926-101 | judging |
| registry.md:38 | [registry] ANALYSIS: P39 조기 즉검 + DELIVER 3시점 + 모듈 시각리포트 (yeon) | DRN-260926-102 | judging |
| registry.md:39 | [registry] ANALYSIS: P39-MUSES 3모달 표준분석 (yeon) | DRN-260926-103 | judging |
| registry.md:40 | [registry] SAM3-RBMA (DELIVER 25cls) | DRN-260926-104 | judging |
| registry.md:41 | [registry] hpca100_deliver_rgbdel_P38_m2f | DRN-260926-105 | judging |
| registry.md:42 | [registry] ANALYSIS: P38-m2f 표준분석 (yeon) | DRN-260926-106 | judging |
| registry.md:43 | [registry] jarvis_muses_rgbel_P39_1_rank_seed2 | DRN-260926-107 | judging |
| registry.md:44 | [registry] hpca100_muses_rgbelr_P39_1_rank_4modal_seed2 | DRN-260926-108 | judging |
| registry.md:45 | [registry] hpca100_muses_P47_d1_dgfproj_4modal | DRN-260926-109 | judging |
| registry.md:46 | [registry] yeon_muses_rgbel_P39_1_seed3 | DRN-260926-110 | judging |
| registry.md:47 | [registry] hpca100_muses_rgbel_P43_pdual | DRN-260926-111 | judging |
| registry.md:48 | [registry] jarvis_muses_rgbel_P44_bmr | DRN-260926-112 | idea |
| registry.md:49 | [registry] hpca100_muses_rgbel_P44_bmr | DRN-260926-113 | judging |
| registry.md:50 | [registry] jarvis_deliver_rgbdel_P46_ctr | DRN-260926-114 | withdrawn |
| registry.md:51 | [registry] jarvis_deliver_rgbdel_P46_ctr_c1c3 | DRN-260926-115 | judging |
| registry.md:52 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only 🔴 헤드라인 ckpt 정본 = NAS ckpts/P46_c3only_base_ep70_test5699_20260730/epoch70_67.79_top1_checkpoint.pth(md5 d340e3fe…, 2026-09-17 이관 — ISSUE-035) | DRN-260926-116 | judging |
| registry.md:53 | [registry] hpca100_deliver_rgbdel_P46_ctr_c2c3 | DRN-260926-117 | withdrawn |
| registry.md:54 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_seed2 | DRN-260926-118 | judging |
| registry.md:55 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_res1024_seedB | DRN-260926-119 | judging |
| registry.md:56 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_res1024_seedC | DRN-260926-120 | judging |
| registry.md:57 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_768_seed15/16 | DRN-260926-121 | judging |
| registry.md:58 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam015 | DRN-260926-122 | withdrawn |
| registry.md:59 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam02 | DRN-260926-123 | judging |
| registry.md:60 | [registry] hpca100_muses_rgbel_P46_c3only_lam02 | DRN-260926-124 | judging |
| registry.md:61 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam03 | DRN-260926-125 | judging |
| registry.md:62 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam02_seed2 | DRN-260926-126 | judging |
| registry.md:63 | [registry] hpca100_muses_rgbelr_P47_2_unibal_4modal | DRN-260926-127 | judging |
| registry.md:64 | [registry] eliceb200_deliver_rgbdel_P46_c3only_1024 | DRN-260926-128 | judging |
| registry.md:65 | [registry] jarvis_deliver_rgbdel_P46_c3only_1024_gpu67 | DRN-260926-129 | judging |
| registry.md:66 | [registry] jarvis_deliver_rgbdel_P39_1_rank_1024_ctrl | DRN-260926-130 | judging |
| registry.md:67 | [registry] jarvis_muses_rgbl_P39_1_rank_2modal | DRN-260926-131 | judging |
| registry.md:68 | [registry] jarvis_deliver_P46_c3only_xattn_trunk | DRN-260926-132 | judging |
| registry.md:69 | [registry] yeon_mcubes_rgbadn_P39_1_rank | DRN-260926-133 | judging |
| registry.md:70 | [registry] jarvis_muses_P39_1_rank_seed20260824 | DRN-260926-134 | judging |
| registry.md:71 | [registry] yeon_muses_P39_1_rank_seed20260825_testsubmit | DRN-260926-135 | judging |
| registry.md:72 | [registry] yeon_p50_align_pretrain | DRN-260926-136 | judging |
| registry.md:73 | [registry] yeon_p50_pseudomodal_gen | DRN-260926-137 | judging |
| registry.md:74 | [registry] yeon_muses_rgbelr_P49_1_air_4modal_g01 | DRN-260926-138 | judging |
| registry.md:75 | [registry] yeon_muses_rgbel_P49_1_air_g01 | DRN-260926-139 | judging |
| registry.md:76 | [registry] hpca100_deliver_rgbdel_P46_ctr_c2c3 (재기동) | DRN-260926-140 | judging |
| registry.md:77 | [registry] yeon_deliver_rgbdel_P49_1_air_768_g01 | DRN-260926-141 | judging |
| registry.md:78 | [registry] yeon_deliver_rgbdel_P49_air_768 | DRN-260926-142 | judging |
| registry.md:79 | [registry] yeon_deliver_rgbd_P46_c3only_lam005_2modal_eval1024 | DRN-260926-143 | judging |
| registry.md:80 | [registry] P48 인스턴스 감독 제안 | DRN-260926-144 | judging |
| registry.md:81 | [registry] jarvis_muses_rgbl_P39_1_rank_2modal | DRN-260926-145 | judging |
| registry.md:82 | [registry] yeon_deliver_rgbd_P46_c3only_lam005_2modal_eval1024 | DRN-260926-146 | judging |
| registry.md:83 | [registry] hpca100_deliver_rgbdel_P46_c3only_p50ext_seed821 | DRN-260926-147 | judging |
| registry.md:84 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260821_elora_permodal_r16 | DRN-260926-148 | judging |
| registry.md:85 | [registry] hpca100_muses_rgbelr_P52_seed20260901 | DRN-260926-149 | idea |
| registry.md:86 | [registry] yeon_deliver_rgbdel_P52_seed20260901 | DRN-260926-150 | judging |
| registry.md:87 | [registry] yeon_deliver_rgbdel_P52_seed20260902 | DRN-260926-151 | judging |
| registry.md:88 | [registry] yeon_mcubes_rgbadn_P52_seed20260901 | DRN-260926-152 | judging |
| registry.md:89 | [registry] hpca100_muses_rgbelr_P52_seed20260902 | DRN-260926-153 | idea |
| registry.md:90 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260821_vicregoff | DRN-260926-154 | judging |
| registry.md:91 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_B0 | DRN-260926-155 | judging |
| registry.md:92 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_E1 | DRN-260926-156 | judging |
| registry.md:93 | [registry] E0_probe_feature_info_deliver_seed821 | DRN-260926-157 | judging |
| registry.md:94 | [registry] E9_logit_adjust_deliver_seed821 | DRN-260926-158 | judging |
| registry.md:95 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E2 | DRN-260926-159 | judging |
| registry.md:96 | [registry] hpca100_muses_rgbel_P39_1_seed2_physaugoff_screen40_E7 | DRN-260926-160 | judging |
| registry.md:97 | [registry] hpca100_muses_rgbel_P39_1_seed2_physaugon_screen40_E7c | DRN-260926-161 | judging |
| registry.md:98 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_E3 | DRN-260926-162 | judging |
| registry.md:99 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_E4 | DRN-260926-163 | judging |
| registry.md:100 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260821_screen40_E4b | DRN-260926-164 | judging |
| registry.md:101 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E12 | DRN-260926-165 | judging |
| registry.md:102 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_E13_confirm200 | DRN-260926-166 | judging |
| registry.md:103 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260902_E13_confirm200_s2 | DRN-260926-167 | judging |
| registry.md:104 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260903_E13_confirm200_s3 | DRN-260926-168 | judging |
| registry.md:105 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_E1_confirm200 | DRN-260926-169 | judging |
| registry.md:106 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260902_E1_confirm200_s2 | DRN-260926-170 | judging |
| registry.md:107 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260903_E1_confirm200_s3 | DRN-260926-171 | judging |
| registry.md:108 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_E13s2 | DRN-260926-172 | judging |
| registry.md:109 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260903_screen40_E13s3 | DRN-260926-173 | judging |
| registry.md:110 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E14 | DRN-260926-174 | judging |
| registry.md:111 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E15 | DRN-260926-175 | judging |
| registry.md:112 | [registry] hpca100_muses_rgbel_P39_1_seed2_physaugoff_taps_screen40_E1M | DRN-260926-176 | judging |
| registry.md:113 | [registry] hpca100_muses_rgbel_P39_1_seed2_physaugoff_taps_c3permodal_screen40_E13M | DRN-260926-177 | judging |
| registry.md:114 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_c3permodal_screen40_E13M_s2 | DRN-260926-178 | judging |
| registry.md:115 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_screen40_E1M_s2 | DRN-260926-179 | judging |
| registry.md:116 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E13Mc_seed3407 | DRN-260926-180 | judging |
| registry.md:117 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E13Mc_seed20260827 | DRN-260926-181 | judging |
| registry.md:118 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E13Mc_seed20260828 | DRN-260926-182 | judging |
| registry.md:119 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E1Mc_seed3407 | DRN-260926-183 | judging |
| registry.md:120 | [registry] yeon_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827 | DRN-260926-184 | withdrawn |
| registry.md:121 | [registry] yeon_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260828 | DRN-260926-185 | withdrawn |
| registry.md:122 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260821_elora_shared | DRN-260926-186 | judging |
| registry.md:123 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260821_elora_sharedresidual | DRN-260926-187 | judging |
| registry.md:124 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260902 | DRN-260926-188 | judging |
| registry.md:125 | [registry] jarvis_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260903 | DRN-260926-189 | judging |
| registry.md:126 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_screen40_E7_s2 | DRN-260926-190 | judging |
| registry.md:127 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_screen40_E7_s3 | DRN-260926-191 | judging |
| registry.md:128 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_screen40_E1M_s3 | DRN-260926-192 | judging |
| registry.md:129 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_full200_E1M | DRN-260926-193 | judging |
| registry.md:130 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_c3permodal_full200_E13M | DRN-260926-194 | judging |
| registry.md:131 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_c3permodal_full200_E13M_s2 | DRN-260926-195 | judging |
| registry.md:132 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_full200_E1M_s2 | DRN-260926-196 | judging |
| registry.md:133 | [registry] jarvis_muses_rgbel_P39_1_physaugoff_full200_E7 | DRN-260926-197 | judging |
| registry.md:134 | [registry] hpca100_muses_rgbel_P39_1_physaugoff_taps_c3permodal_screen40_E13M_s3 | DRN-260926-198 | judging |
| registry.md:135 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_B0Mc_seed3407 | DRN-260926-199 | judging |
| registry.md:136 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_B0Mc_seed20260827 | DRN-260926-200 | judging |
| registry.md:137 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_B0Mc_seed20260828 | DRN-260926-201 | judging |
| registry.md:138 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827 | DRN-260926-202 | judging |
| registry.md:139 | [registry] hpca100_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260828 | DRN-260926-203 | judging |
| registry.md:140 | [registry] lecun_mcubes_rgbadn_P39_1_rank_B0Mc_seed20260827 | DRN-260926-204 | withdrawn |
| registry.md:141 | [registry] lecun_mcubes_rgbadn_P39_1_rank_E1Mc_seed20260827 | DRN-260926-205 | withdrawn |
| registry.md:142 | [registry] bengio_deliver_…_seed20260902_elora_permodal_r16 | DRN-260926-206 | judging |
| registry.md:143 | [registry] bengio_deliver_…_seed20260902_elora_shared | DRN-260926-207 | judging |
| registry.md:144 | [registry] bengio_deliver_…_seed20260903_elora_permodal_r16 | DRN-260926-208 | judging |
| registry.md:145 | [registry] bengio_deliver_…_seed20260903_elora_shared | DRN-260926-209 | judging |
| registry.md:146 | [registry] yeon_deliver_rgbdel_P46_c3only_seed20260904_E1_confirm200 | DRN-260926-210 | withdrawn |
| registry.md:147 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260904 | DRN-260926-211 | withdrawn |
| registry.md:148 | [registry] yeon_deliver_rgbdel_P46_c3only_seed20260904_E1_confirm200_v2 | DRN-260926-212 | judging |
| registry.md:149 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260904_v2 | DRN-260926-213 | judging |
| registry.md:150 | [registry] yeon_deliver_rgbdel_P46_c3only_seed20260904_E13_confirm200 | DRN-260926-214 | judging |
| registry.md:151 | [registry] yeon_deliver_rgbdel_P46_ctr_c3only_lam01_seed20260902_elora_sharedresidual | DRN-260926-215 | judging |
| registry.md:152 | [registry] jarvis_…_seed20260903_elora_sharedresidual | DRN-260926-216 | judging |
| registry.md:153 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260902_screen40_E2s2 | DRN-260926-217 | judging |
| registry.md:154 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260902_screen40_E3s2 | DRN-260926-218 | idea |
| registry.md:155 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_E13 | DRN-260926-219 | idea |
| registry.md:156 | [registry] E3_legal_rescore_seed821 | DRN-260926-220 | judging |
| registry.md:157 | [registry] E1_legal_rescore_seed821 | DRN-260926-221 | judging |
| registry.md:158 | [registry] E4_legal_rescore_seed821 | DRN-260926-222 | judging |
| registry.md:159 | [registry] yeon_deliver_rgbdel_P46_c3only_seed20260902_screen40_E1s2 | DRN-260926-223 | withdrawn |
| registry.md:160 | [registry] E2_legal_rescore_seed821 | DRN-260926-224 | idea |
| registry.md:161 | [registry] bengio_deliver_rgbdel_P46_c3only_seed20260902_screen40_B0s2 / E1s2 | DRN-260926-225 | idea |
| registry.md:162 | [registry] B0_legal_rescore_seed821 | DRN-260926-226 | judging |
| registry.md:163 | [registry] daily_cards_early_eval_bengio_gpu0 | DRN-260926-227 | withdrawn |
| registry.md:164 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_screen40_E17 | DRN-260926-228 | judging |
| registry.md:165 | [registry] hpca100_deliver_rgbdel_P46_c3only_seed20260821_E17_confirm200 | DRN-260926-229 | withdrawn |
| registry.md:166 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260902_E17_confirm200_s2 | DRN-260926-230 | withdrawn |
| registry.md:167 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_Q2 | DRN-260926-231 | judging |
| registry.md:168 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_Q3 | DRN-260926-232 | judging |
| registry.md:169 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed2026090{2,3}_screen40_Q2 | DRN-260926-233 | running |
| registry.md:170 | [registry] jarvis_deliver_rgbdel_P46_c3only_seed20260821_screen40_Q2noKD / Q2noDeg | DRN-260926-234 | idea |
| registry.md:176 | [registry] det_P29_egofill_bengio | DRN-260926-235 | judging |
| registry.md:177 | [registry] det_P29_event_bengio | DRN-260926-236 | judging |
| registry.md:178 | [registry] det_P29_final_full | DRN-260926-237 | judging |
| registry.md:179 | [registry] det_P29_v2 (재학습) | DRN-260926-238 | judging |
| registry.md:180 | [registry] det_P31_v3clip_jarvis | DRN-260926-239 | judging |
| registry.md:181 | [registry] det_P30_v2 | DRN-260926-240 | judging |
| registry.md:182 | [registry] YOLO11m RGB-only 기준점 (E1.1b/c) | DRN-260926-241 | judging |
| registry.md:183 | [registry] jarvis_muses_probea2_backbone_scaling | DRN-260926-242 | judging |

## 4. 모델 세대(arch-evolution.md)

| 원문 줄 | 제목 | DRN | 상태 |
|---|---|---|---|
| arch-evolution.md:11,3459 | [세대] P47-2: UniBal — Uni-modal Balance (구 D-2, 2026-08-04) | DRN-260926-243 | judging |
| arch-evolution.md:48,3363 | [세대] P46: CTR — Class-Transfer Recovery (RCS + MCC + Prototype) (2026-07-29 ~ 2026-08-05) | DRN-260926-244 | judging |
| arch-evolution.md:78,3284 | [세대] P43: PanopticDual — 독립 주손실 mask-classification 헤드 (2026-07-25) | DRN-260926-245 | judging |
| arch-evolution.md:110,3315 | [세대] P44: BMR — Balanced Multimodal Reliability (+ P45 FogStyle) (2026-07-25) | DRN-260926-246 | judging |
| arch-evolution.md:126,3065 | [세대] P38: MaskQueryLite — Mask2Former-lite Query Head (2026-07-18 launch) | DRN-260926-247 | judging |
| arch-evolution.md:149,3104 | [세대] P39: DPC — Dual-Path Compete (2026-07-20) | DRN-260926-248 | judging |
| arch-evolution.md:185,3145 | [세대] P39.1: Rank 수리 — gated_mlp trunk + VICReg (2026-07-21) ★ 현행 기준선 | DRN-260926-249 | judging |
| arch-evolution.md:208,3187 | [세대] P40: RCA-Fusion — Reliability-Conditioned Attenuation (2026-07-21) | DRN-260926-250 | judging |
| arch-evolution.md:232,2888 | [세대] P32: CoRB — Corroboration-Biased Memory Attention (2026-07-06) | DRN-260926-251 | judging |
| arch-evolution.md:255,2863 | [세대] P31: Calibrated Dual-Reliability RBMA + Multi-scale HR Class-Token Decoding (2026-07-02) | DRN-260926-252 | judging |
| arch-evolution.md:283 | [세대] P30-Det — P30 백본 detection 확장: Reliability-router 융합 + Object-Query decoder + FCOS aux (2026-06-30) | DRN-260926-253 | judging |
| arch-evolution.md:306,2839 | [세대] P30: Class-token decoder + Reliability-anchored learned router (2026-06-28) | DRN-260926-254 | judging |
| arch-evolution.md:326,2816 | [세대] P29: SDC — Self-Derived Condition 라우팅 (2026-06-27) | DRN-260926-255 | judging |
| arch-evolution.md:357,2790 | [세대] P28: RBMA — Reliability-Biased Memory Attention (2026-06-15) | DRN-260926-256 | judging |
| arch-evolution.md:443 | [세대] P8: ConfidenceHeadV2 + Sigmoid UAMM | DRN-260926-257 | judging |
| arch-evolution.md:481 | [세대] P9: CrossModalFusionHead + Max-Norm UAMM (현재 최선) | DRN-260926-258 | judging |
| arch-evolution.md:538 | [세대] P10: CrossModalFusionHeadV2 + ModalAuxHead + Oracle KL (취소됨) | DRN-260926-259 | withdrawn |
| arch-evolution.md:601 | [세대] P11: P10 + MI Routing Loss (취소됨) | DRN-260926-260 | withdrawn |
| arch-evolution.md:642 | [세대] P12: Input-Conditioned Soft MoE LoRA | DRN-260926-261 | judging |
| arch-evolution.md:668 | [세대] P13: Energy Score Fusion + Expert Collapse Fix | DRN-260926-262 | judging |
| arch-evolution.md:745 | [세대] P14: Per-Modality Separate Aux Decoders | DRN-260926-263 | judging |
| arch-evolution.md:773 | [세대] P15: Calibrated Spatial Entropy Fusion (설계 단계) | DRN-260926-264 | idea |
| arch-evolution.md:961 | [세대] P16: Calibrated Spatial Entropy Fusion (P15 설계의 구현 버전) | DRN-260926-265 | judging |
| arch-evolution.md:1052 | [세대] P17: Multi-Scale FPN Aux Decoder + Calibrated Spatial Entropy Fusion | DRN-260926-266 | judging |
| arch-evolution.md:1135 | [세대] P18: Trainable ResNet-18 Aux Backbone + Configurable Fusion | DRN-260926-267 | judging |
| arch-evolution.md:1195 | [세대] P19: Learned Spatial Cross-Modal Fusion (SpatialCrossModalFusionHead) | DRN-260926-268 | judging |
| arch-evolution.md:1252 | [세대] P20: Shared MLP Gate + Higher Rank MoE (실험 J-A) | DRN-260926-269 | judging |
| arch-evolution.md:1365 | [세대] P21: DeBA-FP (Deformable Bottleneck Adapter for Feature Pyramid) (실험 K) | DRN-260926-270 | judging |
| arch-evolution.md:1468 | [세대] P22: Multi-Scale DeBA-FP (all FPN levels, Phase 1) (실험 L) | DRN-260926-271 | judging |
| arch-evolution.md:1570 | [세대] P23: MoE DeBA-BB (구현 완료, 학습 대기) (실험 M) | DRN-260926-272 | idea |
| arch-evolution.md:1693 | [세대] P24: P9 + Quality-aware Memory Gating via Per-Modality Decoder Distillation (실험 N) | DRN-260926-273 | judging |
| arch-evolution.md:1833 | [세대] P25: Unified Spatial Quality Fusion — Quality Map으로 UAMM + AMF + Memory 통합 (설계 중) | DRN-260926-274 | idea |
| arch-evolution.md:2051 | [세대] P26: Per-Modality SQG + Multi-Scale + Per-Modality Decoder + Modal-Cond MoE + UAMM Softmax (설계 v5, 2026-03-23) | DRN-260926-275 | idea |
| arch-evolution.md:2762 | [세대] P27: Additive Attention Bias on Cross-Modal Memory Attention (RBMA 전구체, 2026-04-14) | DRN-260926-276 | judging |
| arch-evolution.md:2919 | [세대] P33: CG-MoD — Competence-Gated Hard Fusion + Modality Dropout (2026-07-07) | DRN-260926-277 | judging |
| arch-evolution.md:2946 | [세대] P34: ReliaDINO — DINOv3-L frozen + per-modal LoRA (계보 전환점, 2026-07-13 완주) | DRN-260926-278 | judging |
| arch-evolution.md:2980 | [세대] P35: 공정 레시피 동결 (P34 − ATTN_BIAS − CONSISTENCY − PhysAug, 2026-07-15) | DRN-260926-279 | judging |
| arch-evolution.md:2999 | [세대] P36: Per-Class Reliability-Anchored Router (= P35 + router, 2026-07-15 완주) | DRN-260926-280 | judging |
| arch-evolution.md:3028 | [세대] P37a / P37b: CEFR-Head · ClassToken-lite-Learned (2026-07-17~18) | DRN-260926-281 | judging |
| arch-evolution.md:3213 | [세대] P41: FCR — Fusion Spectral Collapse / Fused Class-alignment Regularizer (2026-07-22~23) | DRN-260926-282 | judging |
| arch-evolution.md:3251 | [세대] P42: lidar-강제 — 조건부 균형 img 마스킹 (2026-07-23) | DRN-260926-283 | judging |
| arch-evolution.md:3348 | [세대] P45: FogStyle (2026-07-25) — 미실행 | DRN-260926-284 | idea |
| arch-evolution.md:3429 | [세대] P47-1: LiDAR 투영 밀도화 (구 D-1, 2026-08-03) | DRN-260926-285 | judging |
| arch-evolution.md:3510 | [세대] P48: 쿼리 경로 인스턴스 감독 (2026-08-05) — 제안 단계 | DRN-260926-286 | judging |

## 5. judgment-ledger.md 행 → 항목

| 대장 줄 | DRN |
|---|---|
| 13 | DRN-260926-01 |
| 14 | DRN-260926-01 |
| 15 | DRN-260926-135 |
| 16 | DRN-260926-116 |
| 17 | DRN-260926-116 |
| 18 | DRN-260926-01 |
| 19 | DRN-260926-02 |
| 20 | DRN-260926-01 |
| 21 | DRN-260926-03 |
| 22 | DRN-260926-168 |
| 23 | DRN-260926-01 |
| 24 | DRN-260926-04 |
| 25 | DRN-260926-188,DRN-260926-189 |
| 26 | DRN-260926-01 |
| 27 | DRN-260926-04 |
| 28 | DRN-260926-59 |
| 29 | DRN-260926-59 |
| 30 | DRN-260926-59 |
| 31 | DRN-260926-59 |
| 32 | DRN-260926-59 |
| 33 | DRN-260926-59 |
| 34 | DRN-260926-01 |
| 35 | DRN-260926-04 |
| 36 | DRN-260926-02 |
| 37 | DRN-260926-59 |
| 38 | DRN-260926-59 |
| 39 | DRN-260926-214 |
| 40 | DRN-260926-91 |
| 41 | DRN-260926-04 |
| 42 | DRN-260926-04 |
| 43 | DRN-260926-60 |
| 44 | DRN-260926-03 |
| 45 | DRN-260926-03 |
| 46 | DRN-260926-03 |
| 47 | DRN-260926-04 |
| 48 | DRN-260926-03 |
| 49 | DRN-260926-03 |
| 50 | DRN-260926-60 |
| 51 | DRN-260926-232 |
| 52 | DRN-260926-61 |
| 53 | DRN-260926-232 |
| 54 | DRN-260926-61 |
| 55 | DRN-260926-62 |
| 56 | DRN-260926-57 |
| 57 | DRN-260926-61 |
| 58 | DRN-260926-91 |
| 59 | DRN-260926-232 |
| 60 | DRN-260926-61 |
| 61 | DRN-260926-60 |
| 62 | (실험 아님: ISSUE-038 워킹트리 점검 기록 — 항목 없음) |
| 63 | DRN-260926-91 |
| 64 | DRN-260926-63 |
| 65 | DRN-260926-61 |
| 66 | DRN-260926-57 |
| 67 | DRN-260926-62 |
| 68 | DRN-260926-57 |
| 69 | DRN-260926-57 |
| 70 | DRN-260926-54 |
| 71 | DRN-260926-231 |
| 72 | DRN-260926-62 |
| 73 | DRN-260926-232 |
| 74 | DRN-260926-23 |
| 75 | DRN-260926-23 |
| 76 | DRN-260926-23 |
| 77 | DRN-260926-57 |
