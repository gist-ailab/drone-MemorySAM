# 비교 표용 발표 수치 조사 (DELIVER · MUSES · MCubeS)

작성 2026-11-02. 원고 `latex/sections/5_experiments.tex`의 `tab:deliver`·`tab:muses`·`tab:mcubes`를 키우기 위한 원문 대조 기록이다.

## 읽는 법

- **모든 수치는 원문 PDF 표에서 직접 확인했다.** PDF는 `/home/jemo/.claude/jobs/e0fe6210/tmp/papers/sota_survey/<arXiv id>.pdf`(arXiv 최신판, 2026-11-02 내려받음)와 `.../sota_survey_cvf/mcubesnet_cvpr22.pdf`(CVF open access)이다. 텍스트 변환본은 `paper2md`(pdftotext 엔진) 출력 `~/.cache/paper2md/*-<id>.md`.
- **쪽 번호는 PDF 쪽 순번**(1쪽 = PDF 첫 장)이다. 학회 게재본의 인쇄 쪽 번호가 아니다.
- "구분" 열: **자기** = 그 방법의 저자가 자기 논문에서 보고한 값 / **재현** = 다른 논문 저자가 다시 학습·평가해 보고한 값 / **재인용** = 다른 논문이 표에 옮겨 적은 값(출처 논문의 값과 같은지 대조 결과를 비고에 적음).
- "미확인" = 원문을 확보하지 못했거나 원문에 그 값이 없다. 추정해 채운 칸은 없다.
- 센서 표기: DELIVER R=RGB, D=depth, E=event, L=LiDAR(DGFusion·CAFuser 표의 C=camera와 같다). MUSES C/F=frame camera, L=LiDAR, R=radar, E=event. MCubeS R(I)=RGB, A=AoLP, D=DoLP, N=NIR.
- 학회 표기 근거: (헤더) = PDF 머리말/서식에서 확인, (자기표기) = 저자가 자기 표에 적은 학회, (타 논문 표기) = 다른 논문의 표에만 있음, (bib) = 기존 references.bib 주석에서 이미 검증된 것.

---

## 0. 원고 표에 이미 있던 값 대조 (가장 먼저 볼 것)

| 표 | 행 | 원고 값 | 원문 값 | 판정 | 출처 |
|---|---|---|---|---|---|
| tab:deliver | CAFuser-CAA val | **68.79** | **68.6** | ❌ 불일치. 68.79는 CAFuser 원문(최신판 v2, arXiv 2410.10791)에 없다. v1에는 DELIVER 표 자체가 없다. 출처 불명 | CAFuser Table III, p6 |
| tab:deliver | DGFusion test | **56.71** | **56.7** | ⚠️ 원문은 소수 1자리. 56.71은 원문에 없는 자릿수(우리가 측정한 공식 가중치 값 56.73과도 다름). 원문 인용이면 56.7로 적어야 한다 | DGFusion Table III, p6 (v1·최신판 동일) |
| tab:deliver | DGFusion backbone | Swin-T | Swin-T | ✅ | DGFusion Table III, p6 |
| tab:deliver | MM SAM-adapter (RGB-D) val / test | 69.60 / 57.35 | 69.60 / 57.35 | ✅ | MM SAM-adapter Table 7, p11 (arXiv 2509.10408) |
| tab:muses | DGFusion test | 79.5 | 79.5 (ablation 표에는 79.50) | ✅ | DGFusion Table II, p6; Table IV, p7 |
| tab:muses | GtA (RGB) test | 82.39 | — | ❓ **미확인**. 이 이름의 논문·arXiv를 찾지 못했다(웹 검색 2회, alphaXiv 검색). 리더보드 항목뿐이면 원문 표 인용이 불가능하다 | — |
| tab:muses | MM SAM-adapter (RGB-L) test | 81.07 | 81.07 | ✅ 공식 온라인 벤치마크 제출값이라고 본문에 명시 | MM SAM-adapter Table 6·8, p11; 본문 p11 "submitted to the MUSES online benchmark" |
| tab:mcubes | Mul-VMamba | 54.65 | — | ❓ **미확인**. arXiv판을 찾지 못했다(KBS 게재로 알려졌으나 원문 미확보, 웹 검색으로도 논문 확인 실패) | — |
| tab:mcubes | StitchFusion | 55.9 | 55.9 (Table 1) / 55.96 (Table 3) | ✅ 백본 Swin-Large-22k. 같은 논문 Table 3에서는 RGB-A-D(3센서)가 56.94로 4센서보다 높다 | StitchFusion Table 1, p6; Table 3, p7 |

---

## 1. DELIVER

DELIVER는 split 표기가 논문마다 다르다. CMNeXt·GeminiFusion·MAGIC·Any2Seg·StitchFusion 계열 수치는 **val(2,005장)** 이고(GeminiFusion은 본문에 "3983 training and 2005 testing"이라고 써서 val을 test라 부른다, p7), CAFuser·DGFusion·MM SAM-adapter는 **test(1,897장)** 를 따로 적는다. CAFuser Table III가 같은 모델의 val/test를 나란히 적어(예: CMNeXt 66.3 / 53.0) 두 군의 차이가 split 때문임을 보여 준다. **val 열과 test 열을 섞지 않는다.**

### 1-a. 4센서(R-D-E-L) 행

| 방법 | 학회·연도 | arXiv / DOI | 백본 | 센서 | val | test | 출처 | 구분 | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| CMNeXt | CVPR 2023 (bib) | 2303.01480 | MiT-B2 | R-D-E-L | 66.30 | | CMNeXt Table 1(a), p6; Table 2, p7 | 자기 | 표에 split 미표기. 동일 값이 CAFuser·StitchFusion에서 "val"로 재인용됨 |
| CMNeXt | | | MiT-B2 | R-D-E-L | 66.3 | 53.0 | CAFuser Table III, p6 | val=재인용, test=CAFuser 측정 | test 53.0은 DGFusion Table III(p6)에도 같은 값 |
| CMNeXt | | | MiT-B2 | R-D-E-L | 66.30 | 53.00 | MM SAM-adapter Table 7, p11 | 재현(공식 가중치 평가) | 본문 p10: "official weights" |
| CMNeXt | | | MiT-B2 | R-D-E-L | 66.33 | | 2503.18445 Table 3, p5 (RDEL 열) | 재현(벤치마크 재평가, val) | |
| CMNeXt | | | MiT-B0 | R-D-E-L | 59.18 | | MemorySAM Table 1, p6; MLE-SAM Table I, p7; EGFormer Table 2, p7 | 재현(B0) | CrossWeaver Table 1/5(p6, p13)는 같은 B0 CMNeXt를 59.84로 적음(MemorySAM 표의 R-D-E 값과 같아 열이 밀린 것으로 보임) |
| HRFuser | ITSC 2023 (타 논문 표기) | 2206.15157 | HRFormer-T | R-D-E-L | 52.97 | | CMNeXt Table 1(a), p6 | 재현(CMNeXt 저자) | HRFuser 원 논문은 2D 검출 논문이라 DELIVER 결과가 없다 |
| TokenFusion | CVPR 2022 (bib) | 2204.08721 | MiT-B2 | R-D-E-L | 63.5 | | GeminiFusion Table 1, p4 (†=재현) | 재현(GeminiFusion 저자) | TokenFusion 원 논문에는 DELIVER 없음(원문 미대조, 출간 시기상 불가능). StitchFusion Table 1(p6)도 63.5 재인용 |
| GeminiFusion | ICML 2024 (bib, 미검증 주석) | 2406.01210 | MiT-B2 | R-D-E-L | 66.9 | | GeminiFusion Table 1, p4; Table 5, p7 | 자기 | |
| GeminiFusion | | | MiT-B2 | R-D-E-L | 66.9 | 54.5 | CAFuser Table III, p6 | val=재인용, test=CAFuser 측정 | DGFusion Table III(p6) test 54.5 |
| GeminiFusion | | | MiT-B2 | R-D-E-L | 66.90 | 54.46 | MM SAM-adapter Table 7, p11 | 재현(공식 가중치) | test 54.46 vs CAFuser 54.5 |
| GeminiFusion | | | MiT-B2 | R-D-E-L | 66.92 | | 2503.18445 Table 3, p5 | 재현(벤치마크, val) | |
| StitchFusion | ACM MM 2025 (자기표기) | 2408.01343 | MiT-B2 | R-D-E-L | 68.18 (Table 7) / 68.2 (Table 1) | | StitchFusion Table 1, p6; Table 3, p7; Table 7, p8 | 자기 | 표에 split 미표기 |
| StitchFusion | | | Swin-Tiny-1k | R-D-E-L | 70.34 (Table 3) / 70.3 (Table 1) | | StitchFusion Table 1, p6; Table 3, p7 | 자기 | DELIVER 최고 val 자기보고. split 미표기, CAFuser의 같은 군(val) 표기로 val로 본다 |
| StitchFusion | | | MiT-B2 | R-D-E-L | 68.2 | 53.4 | CAFuser Table III, p6 | val=재인용, test=CAFuser 측정 | DGFusion Table III(p6) test 53.4 |
| StitchFusion | | | MiT-B2 | R-D-E-L | 68.20 | | 2503.18445 Table 3, p5; EQUISeg Table 1, p4 | 재현(벤치마크, val) | |
| MAGIC | ECCV 2024 (bib 미검증, LNCS 서식) | 2407.11344 | SegFormer-B2 | R-D-E-L | 67.66 | | MAGIC Table 1, p10 | 자기 | CMNeXt 66.30 대비 +1.36 |
| MAGIC | | | MiT-B2 | R-D-E-L | 66.10 | | 2503.18445 Table 3, p5; EQUISeg Table 1, p4 | 재현(벤치마크) | 자기보고 67.66과 1.56 차 |
| MAGIC++ | arXiv 2024 | 2412.16876 | MiT-B2 | R-D-E-L | 67.34 | | 2503.18445 Table 3, p5; EQUISeg Table 1, p4 | 재현(벤치마크) | MAGIC++ 자기 논문의 B2 전모달 값은 이번에 대조하지 않음(미확인) |
| Any2Seg | ECCV 2024 (타 논문 표기, LNCS 서식) | 2407.11351 | MiT-B2 | R-D-E-L | 68.25 | | Any2Seg Table 1, p9 | 자기 | Table 4(p11)의 조건별 표 "Mean" 67.19는 조건 평균이라 다른 값 |
| OmniSegmentor | NeurIPS 2025 (bib·타 논문 표기, 자기 PDF에서 미확인) | 2509.15096 | MiT-B2 | R-D-E-L | 67.5 | | OmniSegmentor Table 1(f), p6 | 자기 | DELIVER split 3983/2005/1897 명시(Table 5, p12), 평가 split은 표에 미표기 |
| OmniSegmentor | | | DFormer-L | R-D-E-L | 68.0 | | OmniSegmentor Table 1(f), p6 | 자기 | |
| HyperDUM | CVPR 2025 (bib) | 2503.20011 | CMNeXt(MiT-B2) + UQ | R-D-E-L | 67.59 | | HyperDUM Table 4, p6 | 자기 | 본문 p6 "DeLiVER validation set". 표의 "Mean" 행인데, 같은 표 CMNeXt Mean이 66.30(=전체 mIoU)이라 전체 val mIoU로 읽힌다 |
| CAFuser-CAA | RA-L 2025 (헤더) | 2410.10791 | Swin-T | R-D-E-L | 68.6 | 55.2 | CAFuser Table III, p6 | 자기 | DGFusion Table III(p6) test 55.2 재인용 일치. **원고의 68.79와 다름** |
| CAFuser (CA²) | RA-L 2025 (헤더) | 2410.10791 | Swin-T | R-D-E-L | 67.8 | 55.6 | CAFuser Table III, p6 | 자기 | DGFusion 55.6, MM SAM-adapter 67.80/55.60 재인용 일치. CAFuser 본문 p5: "selecting the best checkpoint on the validation set" |
| DGFusion | RA-L 2026 (헤더) | 2509.09828 | Swin-T | R-D-E-L | | 56.7 | DGFusion Table III, p6 | 자기 | val 값은 논문에 없음. CLE(depth 입력 없이) 51.6 |
| EQUISeg | arXiv 2025 | 2509.24505 | MiT-B2 | R-D-E-L | 67.90 | | EQUISeg Table 1, p4 | 자기 | split 미표기, 같은 표 다른 행이 2503.18445 val 재평가값과 같음. **references.bib 키 `robustseg2025`가 이 arXiv를 가리키는데 논문 자신은 방법을 "EQUISeg"라 부른다** |
| BPCF (Cao et al.) | arXiv 2026 | 2610.17800 | MiT-B2 | R-D-E-L | 67.4 (split 불명) | | 2610.17800 Table 6, p18 | 자기 | 데이터 표(p15)는 Train/Test 3893/1897로 적었으나 비교 대상은 val 군 값(CMNeXt 66.3)이다. 본문 "half the original resolution", 60 epoch. 열 배정 보류 |
| FuseAdapt | arXiv 2026 (bib) | 2610.21779 | DINOv2-B(동결) | R-D-E-L | 68.33 (split 미표기) | | FuseAdapt Table 3, p7 | 자기 | val 군 값과 같은 표에서 비교. 체크포인트 = best validation |
| MemorySAM | arXiv 2025 | 2503.06700 | SAM2 Hiera-B+ (LoRA 1) | R-D-E-L | 65.38 (split 미표기) | | MemorySAM Table 1, p6 | 자기 | 비교 대상이 CMNeXt-**B0** 59.18 |
| MLE-SAM | arXiv 2024 | 2412.04220 | SAM2 Hiera-B+ (MoE-LoRA 4) | R-D-E-L | 64.08 (split 미표기) | | MLE-SAM Table I, p7 | 자기 | MemorySAM Table 1(p6)에 같은 값 재인용 |
| EGFormer | arXiv 2025 | 2505.14014 | MiT-B0 | R-D-E-L | 59.53 (split 미표기) | | EGFormer Table 2, p7 | 자기 | B0 군 |
| CrossWeaver | arXiv 2026 | 2604.02948 | MiT-B0 | R-D-E-L | 64.69 (split 미표기) | | CrossWeaver Table 1, p6; Table 5, p13 | 자기 | B0 군. 같은 표 CHARM 64.03은 CHARM의 "Top-1"(조합 중 최고) 값을 옮긴 것이라 전모달 값이 아님 |

### 1-b. 센서가 4개보다 적은 행 (2차 비교)

| 방법 | 학회·연도 | arXiv / DOI | 백본 | 센서 | val | test | 출처 | 구분 | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| MM SAM-adapter | IEEE Access 2025 (bib, DOI 10.1109/ACCESS.2025.3609640) | 2509.10408 | SAM ViT-L + ConvNeXt-S | R-D | 69.60 | 57.35 | Table 7, p11 (test는 Table 4, p10에도 57.35) | 자기 | 1024×1024. RGB-hard 부분집합 test 53.35 |
| MM SAM-adapter | | | 같음 | R-L | 61.89 | 57.14 | Table 7, p11; Table 4, p10 | 자기 | |
| MM SAM-adapter | | | 같음 | R-E | 60.74 | 55.70 | Table 7, p11; Table 4, p10 | 자기 | |
| CMNeXt (2센서 재평가) | | | MiT-B2 | R-D / R-L / R-E | | 53.87 / 51.32 / 50.81 | MM SAM-adapter Table 4, p10 | 재현(공식 가중치) | |
| GeminiFusion (2센서) | | | MiT-B2 | R-D / R-L / R-E | | 54.98 / 50.57 / 51.71 | MM SAM-adapter Table 4, p10 | 재현 | |
| RoadFormer+ | (타 논문 표기 TIV'24) | 미확인 | ConvNeXt-L | R-D / R-L / R-E | | 55.95 / 54.56 / 54.10 | MM SAM-adapter Table 4, p10 | 재현(재학습) | 원 논문 미대조 |
| CAFuser-CAA / CAFuser / DGFusion (C-L-E, depth 없음) | | | Swin-T | R-L-E | | 51.2 / 51.3 / 51.6 | DGFusion Table III, p6 | 재인용·자기 | |
| CMNeXt / StitchFusion / GeminiFusion (C-L-E) | | | MiT-B2 | R-L-E | | 50.3 / 50.8 / 50.5 | DGFusion Table III, p6 | DGFusion 측정 | |
| CMNeXt | | | MiT-B2 | R-D / R-E / R-L / R-D-E / R-D-L | 63.58 / 57.48 / 58.04 / 64.44 / 65.50 | | CMNeXt Table 1(a), p6 | 자기 | |
| CMX | T-ITS 2023 (bib) | 2203.04838 | MiT-B2 | R-D / R-E / R-L | 62.67 / 56.52 / 56.37 | | CMNeXt Table 1(a), p6 | 재현(CMNeXt 저자) | CMX 원 논문에는 DELIVER 없음(확인). Any2Seg Table 1(p9)은 R-E를 56.62로 적음(0.10 차) |
| TokenFusion | | | MiT-B2 | R-D / R-E / R-L | 60.25 / 45.63 / 53.01 | | CMNeXt Table 1(a), p6 | 재현(CMNeXt 저자) | GeminiFusion Table 1(p4)은 63.7 / 55.7 / 55.5로 다르게 재현 |
| GeminiFusion | | | MiT-B2 | R-D / R-E / R-L | 66.4 / 58.5 / 58.6 | | GeminiFusion Table 1, p4 | 자기 | |
| StitchFusion | | | MiT-B2 | R-D / R-D-E / R-D-L | 65.75 / 66.03 (Table 7) · 66.65 (Table 3) / 66.03 (Table 3) | | StitchFusion Table 3, p7; Table 7, p8 | 자기 | Table 3의 R-D-L 66.03·R-D-E 66.65와 Table 7의 RGB-DE 66.03이 서로 엇갈림(같은 논문 내 불일치) |
| StitchFusion | | | Swin-Tiny-1k | R-D | 68.75 | | StitchFusion Table 3, p7 | 자기 | |
| Any2Seg | | | MiT-B2 | R-D / R-E / R-L / R-D-E / R-D-L | 67.12 / 57.83 / 57.92 / 66.75 / 67.20 | | Any2Seg Table 1, p9 | 자기 | |
| MemorySAM | | | Hiera-B+ | R-D | 63.48 | | MemorySAM Table 1, p6 | 자기 | |
| OmniSegmentor | | | DFormer-L | R-D / R-E / R-L | 64.7 / 58.6 / 59.2 | | OmniSegmentor Table 1(f), p6 | 자기 | |
| SegRGB-X | arXiv 2026 | 2603.28023 | (범용 모델) | R-E | 51.83 | | SegRGB-X Table(I), p6 | 자기 | 5개 데이터셋 공동 학습한 범용 모델이라 본표에 부적합 |

---

## 2. MUSES

MUSES는 test 라벨이 공개되지 않아 **test 값은 공식 평가 서버 제출로만** 나온다(MUSES 원 논문 p9; MM SAM-adapter p6). 아래에서 test 열은 서버 제출값, val 열은 로컬 평가값이다. MUSES 원 논문은 "Performance is always reported on the respective test sets"(p11)라고 적었다.

| 방법 | 학회·연도 | arXiv / DOI | 백본 | 센서 | val | test | 출처 | 구분 | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| Mask2Former | (기준선) | | Swin-T | C | | 70.7 | MUSES Table 4, p13 | MUSES 저자 측정 | CAFuser Table II(p5)·DGFusion Table II(p6) 재인용 일치 |
| Mask2Former | | | Swin-L | C | | 77.1 | MUSES Table 4, p13 | MUSES 저자 측정 | 원 논문의 카메라 단독 최고 |
| SegFormer | | | MiT-B2 / MiT-B5 | C | | 72.5 / 74.7 | MUSES Table 4, p13 | MUSES 저자 측정 | |
| DeepLabv3+ / OCRNet / SETR | | | R101-D8 / HRNetV2p-W48 / ViT-L | C | | 70.5 / 71.9 / 71.1 | MUSES Table 4, p13 | MUSES 저자 측정 | |
| OneFormer | | | Swin-T | C | | 72.8 | CAFuser Table II, p5; DGFusion Table II, p6 | CAFuser 측정 | |
| MUSES 기준선 융합 | ECCV 2024 (bib) | 2401.12761 | 4×Swin-T (Mask2Former) | C-L-R-E | | mIoU 미보고 (PQ 53.6) | MUSES Table 3, p12 | 자기 | 다중센서 mIoU는 원 논문에 없음 |
| CMNeXt | | | MiT-B2 | C-L-R-E | | 72.4 | CAFuser Table II, p5 | CAFuser 측정 | **같은 방법 다른 값**: DGFusion Table II(p6) 72.1, CAFuser v1 Table II(p5) 72.5, MM SAM-adapter Table 8(p11) 72.40 |
| CMNeXt | | | MiT-B2 | C-L / C-E | | 72.36 / 70.49 | MM SAM-adapter Table 6, p11 | 재현(재학습, 서버 제출) | |
| CMNeXt | | | MiT-B0 | F-E-L | 46.66 | | AnySeg Table 1, p5; MLE-SAM Table II, p8 | 재현(val, 결측 학습 계열) | |
| GeminiFusion | | | MiT-B2 | C-L-R-E | | 75.3 | CAFuser Table II, p5 | CAFuser 측정 | GeminiFusion 원 논문에는 MUSES 없음. DGFusion·MM SAM-adapter 75.3 재인용 |
| GeminiFusion | | | MiT-B2 | C-L / C-E | | 74.22 / 68.62 | MM SAM-adapter Table 6, p11 | 재현(서버 제출) | |
| RoadFormer+ | | 미확인 | ConvNeXt-L | C-L / C-E | | 80.38 / 77.70 | MM SAM-adapter Table 6, p11 | 재현(서버 제출) | 원 논문 미대조 |
| CAFuser-CAA | RA-L 2025 | 2410.10791 | Swin-T | C-L-R-E | | 78.5 | CAFuser Table II, p5 | 자기 | DGFusion Table II 78.5 일치 |
| CAFuser (CA²) | RA-L 2025 | 2410.10791 | Swin-T | C-L-R-E | | 78.2 | CAFuser Table II, p5 | 자기 | MM SAM-adapter Table 8(p11)은 78.18로 적음. PQ 59.7(Table I) |
| DGFusion | RA-L 2026 | 2509.09828 | Swin-T | C-L-R-E | | 79.5 | DGFusion Table II, p6 | 자기 | ablation Table IV·V(p7) 79.50, PQ 61.03 |
| MM SAM-adapter | IEEE Access 2025 | 2509.10408 | SAM ViT-L + ConvNeXt-S | C-L | | 81.07 | Table 6, p11; Table 8, p11 | 자기(서버 제출 명시) | 레이더 제외(본문 p6). 조건별 Day 83.34 / Night 74.97 / Fog 74.12 |
| MM SAM-adapter | | | 같음 | C-E | | 79.92 | Table 6·8, p11 | 자기 | |
| MLE-SAM | arXiv 2024 | 2412.04220 | Hiera-B+ | F-E-L | 74.8 (split 미표기) | | MLE-SAM Table II, p8 | 자기 | 같은 표 CMNeXt-B0 46.66이 다른 논문의 val 값과 같아 val로 보임. 서버 제출 언급 없음 |
| FuseAdapt | arXiv 2026 | 2610.21779 | DINOv2-S(동결) | F-E-L | 77.28 | | FuseAdapt Table 2, p7 | 자기 | 부록 p17: "best-validation mIoU", DINOv2-B로는 80.16(본문 수치, 표 없음) |
| EQUISeg | arXiv 2025 | 2509.24505 | MiT-B0 | F-E-L(로 추정) | 50.26 | | EQUISeg Table 1, p4 | 자기 | 센서 조합 표기 없음, split 미표기 |
| MAGIC / Any2Seg / AnySeg / MAGIC++ | | | MiT-B0 | F-E-L | 49.02 / 50.00 / 51.14 / 40.58 | | AnySeg Table 1, p5; MAGIC++ Table I, p6 | val, 결측 강건 학습 계열 | 전모달 최고값이 아니라 결측 대비 학습 모델의 FEL 칸. 본표 부적합 |
| GtA | — | — | — | C | | 82.39 | — | **미확인** | 논문을 찾지 못함 |

---

## 3. MCubeS

MCubeS split은 302/96/102(MCubeSNet 원 논문 p7). 아래 논문들은 대부분 표에 split을 적지 않았고, 관행상 test 102장 값으로 본다. MCubeS 평가 해상도는 CMNeXt 부록 p15("512×512 during training and 1024×1024 during evaluation", MCubeSNet 설정을 따른다고 명시)에서 1024×1024.

| 방법 | 학회·연도 | arXiv / DOI | 백본 | 센서 | test | 출처 | 구분 | 비고 |
|---|---|---|---|---|---|---|---|---|
| MCubeSNet | CVPR 2022 (bib, CVF) | CVF open access | DeepLabv3+ ResNet-101 | R-A-D-N | 42.86 | MCubeSNet Table 1(a), p6 | 자기 | 백본은 본문 p6 "We chose DeepLab v3+ as the backbone ... Resnet-101". **StitchFusion Table 1(p6)·FuseAdapt Table 1(p6)은 백본을 MiT-B4로 잘못 적음**. StitchFusion은 42.9, MMSFormer Table 1(p6)은 **42.46**(같은 논문 Table 6, p9는 42.86) |
| CMNeXt | CVPR 2023 | 2303.01480 | MiT-B2 | R-A-D-N | 51.54 | CMNeXt Table 1(e), p6 | 자기 | MAGIC·Any2Seg·MMSFormer·U3M 재인용 일치. StitchFusion Table 1은 "MiT-B4(MiT-B2)" 51.5 |
| CMNeXt | | | MiT-B0 | R-A-D-N | 36.16 | MemorySAM Table 2, p6; MLE-SAM Table III, p8; EGFormer Table 3, p7 | 재현(B0) | CrossWeaver Table 1(p6)은 40.94(열 밀림 추정) |
| MMSFormer | arXiv 2023 (IEEE OJSP로 타 논문 표기) | 2309.04001 | MiT(변형 미표기; StitchFusion은 B4로 표기) | R-A-D-N | 53.11 | MMSFormer Table 1, p6; Table 6, p9 | 자기 | StitchFusion Table 1 53.1, Table 7 53.11 재인용 일치 |
| MMSFormer | | | MiT-B0 | R-A-D-N | 45.98 | CrossWeaver Table 5, p13 | 재현(B0) | |
| ShareCMP | arXiv 2023 (TCSVT'25로 타 논문 표기) | 2312.03430 | MiT-B2 | R-A-D (NIR 없음) | 50.99 | ShareCMP Table 4, p9; 초록 p1 | 자기 | 3센서. StitchFusion Table 1의 "ShareCMP 50.3"은 RGB-A 50.34 값 |
| MAGIC | ECCV 2024 | 2407.11344 | SegFormer-B2 | R-A-D-N | 53.01 | MAGIC Table 1, p10 | 자기 | |
| Any2Seg | ECCV 2024 | 2407.11351 | MiT-B2 | R-A-D-N | 51.91 | Any2Seg Table 2, p9 | 자기 | 표에 MCubeSNet 4센서 행이 "I-A-D"로 오기 |
| U3M | arXiv 2024 (IEEE TMM 서식 헤더 / BPCF 표는 PR'25로 표기, 불일치) | 2405.15365 | SegFormer(동결, 변형 미표기) | R-A-D-N | 51.69 | U3M Table I, p6; Table VI, p8 | 자기 | |
| StitchFusion | ACM MM 2025 | 2408.01343 | MiT-B4 (+FFMs) | R-A-D-N | 53.92 (Table 7) / 53.9 (Table 1) | StitchFusion Table 1, p6; Table 3, p7; Table 7, p8 | 자기 | FFMs 없이 51.74(Table 3) |
| StitchFusion | | | Swin-Tiny | R-A-D-N | 52.3 | StitchFusion Table 1, p6 | 자기 | |
| StitchFusion | | | Swin-Large-22k | R-A-D-N | 55.9 (Table 1) / 55.96 (Table 3) | StitchFusion Table 1, p6; Table 3, p7 | 자기 | 같은 백본 R-A-D 56.94, R-A 56.07(Table 3) |
| StitchFusion | | | MiT-B0 | R-A-D-N | 46.99 | CrossWeaver Table 5, p13 | 재현(B0) | |
| MemorySAM | arXiv 2025 | 2503.06700 | Hiera-B+ (LoRA 1) | R-A-D-N | 52.88 | MemorySAM Table 2, p6 | 자기 | |
| MLE-SAM | arXiv 2024 | 2412.04220 | Hiera-B+ (MoE-LoRA 4) | R-A-D-N | 51.02 | MLE-SAM Table III, p8 | 자기 | |
| BPCF (Cao et al.) | arXiv 2026 | 2610.17800 | MiT-B4 | R-A-D-N | 52.22 | 2610.17800 Table 8, p19 | 자기 | 데이터 표 p15: train/test 302/102 |
| FuseAdapt | arXiv 2026 | 2610.21779 | DINOv2-B(동결) | R-A-D-N | 57.62 | FuseAdapt Table 1, p6; Table 11, p18 | 자기 | 동결 VFM + LoRA. 우리 구성(동결 DINOv3-L)과 가장 가까운 비교군 |
| EGFormer | arXiv 2025 | 2505.14014 | MiT-B0 | R-A-D-N | 43.40 | EGFormer Table 3, p7 | 자기 | B0 군. 표의 R-A 행이 "R-D"로 오기 |
| CrossWeaver | arXiv 2026 | 2604.02948 | MiT-B0 | R-A-D-N | 48.76 | CrossWeaver Table 1, p6 | 자기 | B0 군 |
| Mul-VMamba | KBS 2025(로 알려짐) | 미확인 | VMamba(로 알려짐) | R-A-D-N | 54.65 | — | **미확인** | 원문 미확보 |

---

## 4. 같은 방법이 논문마다 다른 값으로 보고된 경우

| 방법 | 벤치·설정 | 값들 (출처) | 원인 판단 |
|---|---|---|---|
| CMNeXt | DELIVER R-D-E-L MiT-B2 | val 66.30 (자기) · 66.33 (2503.18445 재평가) · test 53.0 (CAFuser·DGFusion) · 53.00 (MM SAM-adapter) | val/test 차이. 66.30↔66.33은 재평가 오차 |
| CMNeXt | DELIVER MiT-B0 | 59.18 (MemorySAM·MLE-SAM·EGFormer) · 59.84 (CrossWeaver) | CrossWeaver 열 밀림으로 보임 |
| CMNeXt | MUSES C-L-R-E test | 72.4 (CAFuser 최신판) · 72.5 (CAFuser v1) · 72.1 (DGFusion) · 72.40 (MM SAM-adapter) | 같은 저자군 내에서도 판마다 다름. 원인 미확인 |
| CMNeXt | MCubeS B0 | 36.16 (MemorySAM 등) · 40.94 (CrossWeaver) | CrossWeaver 열 밀림으로 보임 |
| GeminiFusion | DELIVER R-D-E-L MiT-B2 | val 66.9 (자기) · 66.92 (2503.18445) · test 54.5 (CAFuser·DGFusion) · 54.46 (MM SAM-adapter) | 반올림 차 |
| TokenFusion | DELIVER 2센서 | R-D 60.25 / R-E 45.63 / R-L 53.01 (CMNeXt 재현) · 63.7 / 55.7 / 55.5 (GeminiFusion 재현) | 재현자 차이 |
| MAGIC | DELIVER R-D-E-L B2 | 67.66 (자기) · 66.10 (2503.18445 재평가) | 재평가에서 1.56 낮음 |
| StitchFusion | DELIVER MiT-B2 | 68.18 (자기) · 68.2 (CAFuser 재인용, test 53.4) · 68.20 (2503.18445) | 일치 |
| StitchFusion | MCubeS Swin-L | 55.9 (Table 1) · 55.96 (Table 3) | 반올림 |
| CAFuser | MUSES CA² | 78.2 (자기) · 78.18 (MM SAM-adapter) | 반올림 |
| MCubeSNet | MCubeS | 42.86 (자기·CMNeXt·Any2Seg) · 42.9 (StitchFusion) · 42.46 (MMSFormer Table 1) | MMSFormer 오기(같은 논문 Table 6은 42.86) |
| ShareCMP | MCubeS | 50.99 RGB-A-D (자기) · 50.3 (StitchFusion, = RGB-A 50.34) | StitchFusion이 다른 센서 조합 값을 옮김 |
| CMX | DELIVER R-E | 56.52 (CMNeXt) · 56.62 (Any2Seg) | 재인용 오기 |

---

## 5. 본표에서 뺀 것 (프로토콜이 다름)

- **결측 강건 학습(modality-agnostic, MaSS) 계열의 15조합 평균**: AnySeg(2411.17141, B0, DELIVER 평균 46.64 / RDEL 칸 60.26, Table 2 p6), MAGIC++(2412.16876, B0, 평균 47.74 / RDEL 61.67, Table II p7; 본문 p10에 "MAGIC's 63.40" 명시), FunEntropy(2505.06635, `zheng2025entropy`, 평균 48.29 / RDEL 51.65 Table 3 p6, MCubeS IADN 42.73 Table 4 p7, MUSES FEL 47.89 Table 2 p6), CHARM(2508.03060, Average/Top-1/Last-1만 보고, 전모달 값 없음, Table 1 p6), RMMSS(2505.12861, AnySeg 지표 기준, 이번에 칸 대조 안 함). 모두 결측 대비 학습 모델이라 전모달 정확도 비교에 쓰면 안 된다.
- **2503.18445 벤치마크의 EMM/RMM/NM 값**은 원고 표 IV에서 이미 쓰고 있어 여기서는 clean RDEL 열만 옮겼다.
- **MARS**(2606.30355): MCubeS를 결측 조건(DeepLabv3+ 기반)으로만 평가, 전모달 단일 값 없음.
- **URVIS 2026 챌린지 리포트**(2604.16984): MUSES 판옵틱(PQ)만 보고, mIoU 없음.
- **Sigma**(2404.04256), **CMX**(2203.04838), **HRFuser**(2206.15157): 원 논문에 세 벤치 결과가 없다(본문 검색으로 확인).

## 6. 원고 표에 넣을 때 주의

1. `tab:deliver`는 test 열 기준이므로 test 값이 있는 행만 그대로 쓸 수 있다: CMNeXt 53.0, StitchFusion 53.4, GeminiFusion 54.5, CAFuser-CAA 55.2, CAFuser 55.6, DGFusion 56.7, MM SAM-adapter(R-D) 57.35. CMNeXt·StitchFusion·GeminiFusion test 값은 **CAFuser 저자가 측정한 값**(CAFuser Table III)이라 원 논문 자기보고가 아니다. 인용은 CAFuser로 단다.
2. val 열을 채우려면 CAFuser Table III 값(66.3 / 68.2 / 66.9 / 68.6 / 67.8)을 쓰면 한 표에서 val/test가 같은 출처로 맞는다. **CAFuser-CAA val은 68.6이다(68.79 아님).**
3. DGFusion은 val 값을 보고하지 않았다. DGFusion test는 원문 56.7로 적고, 본문 격차 계산(57.74−56.71)도 56.7 기준으로 다시 계산해야 한다(+1.04 / best +1.91).
4. MUSES `tab:muses`의 CAFuser 행: CAFuser(CA²) test 78.2, CAFuser-CAA test 78.5, 백본 Swin-T, 센서 C-L-R-E(= RGB-L-R-E). DGFusion 백본 Swin-T, 센서 C-L-R-E.
5. MCubeS `tab:mcubes`: Mul-VMamba 54.65는 원문 미확인이므로 빼거나 "미확인" 처리가 필요하다. 원문 확인된 상위 값은 FuseAdapt 57.62(DINOv2-B 동결), StitchFusion 55.9(Swin-L), StitchFusion 53.92(MiT-B4), MMSFormer 53.11, MAGIC 53.01, MemorySAM 52.88.

---

## 7. references.bib에 없는 논문 — BibTeX 제안

저자·제목은 각 PDF 1쪽에서 확인했다. 학회는 원문 헤더에서 확인되지 않은 경우 arXiv로만 적는다. (기존 키 재사용: cmnext2023, muses2024, mcubes2022, memorysam2025, dgfusion2026, cafuser2025, stitchfusion2024, geminifusion2024, magic2024, magicpp2024, anyseg2024, omnisegmentor2025, mmsamadapter2025, mlesam, cmx2023, tokenfusion2022, hyperdum2025, mmssbench2025, rmmss2025, fuseadapt2026, robustseg2025(=EQUISeg 2509.24505), zheng2025entropy(=FunEntropy 2505.06635))

```bibtex
@article{any2seg2024,
  author  = {Zheng, Xu and Lyu, Yuanhuiyi and Wang, Lin},
  title   = {Learning Modality-agnostic Representation for Semantic Segmentation from Any Modalities},
  journal = {arXiv preprint arXiv:2407.11351},
  year    = {2024}
} % ECCV 2024로 알려짐(LNCS 서식, 타 논문 표기) — 게재 정보 미확인

@article{mmsformer2023,
  author  = {Reza, Md Kaykobad and Prater-Bennette, Ashley and Asif, M. Salman},
  title   = {{MMSFormer}: Multimodal Transformer for Material and Semantic Segmentation},
  journal = {arXiv preprint arXiv:2309.04001},
  year    = {2023}
} % IEEE OJSP 게재로 타 논문 표기 — 미확인

@article{u3m2024,
  author  = {Li, Bingyu and Zhang, Da and Zhao, Zhiyuan and Gao, Junyu and Li, Xuelong},
  title   = {{U3M}: Unbiased Multiscale Modal Fusion Model for Multimodal Semantic Segmentation},
  journal = {arXiv preprint arXiv:2405.15365},
  year    = {2024}
} % PDF 머리말은 IEEE TMM 서식, 2610.17800은 PR'25로 표기 — 게재지 불일치, 미확인

@article{egformer2025,
  author  = {Zhang, Zelin and Zhang, Tao and Li, Kedi and Zheng, Xu},
  title   = {{EGFormer}: Towards Efficient and Generalizable Multimodal Semantic Segmentation},
  journal = {arXiv preprint arXiv:2505.14014},
  year    = {2025}
} % PDF에 "KediLI"로 붙어 표기됨 — 이름 철자 확인 필요

@article{crossweaver2026,
  author  = {Zhang, Zelin and Li, Kedi and Liang, Huiqi and Xu, Chuanzhi and Zhang, Tao and Cai, Weidong},
  title   = {{CrossWeaver}: Cross-modal Weaving for Arbitrary-Modality Semantic Segmentation},
  journal = {arXiv preprint arXiv:2604.02948},
  year    = {2026}
} % 저자 순서는 PDF 1쪽 2열 배치에서 읽은 것 — 확인 필요

@article{charm2025,
  author  = {Wen, Lekang and Xiao, Jing and Liao, Liang and Chen, Jiajun and Wang, Mi and others},
  title   = {{CHARM}: Collaborative Harmonization across Arbitrary Modalities for Modality-agnostic Semantic Segmentation},
  journal = {arXiv preprint arXiv:2508.03060},
  year    = {2025}
} % 1쪽 저자 줄이 쉼표로 끝나 뒤 저자가 더 있을 수 있음 — 전체 저자 확인 필요

@article{sharecmp2023,
  author  = {Liu, Zhuoyan and Wang, Bo and Wang, Lizhi and Mao, Chenyu and Li, Ye},
  title   = {{ShareCMP}: Polarization-Aware {RGB-P} Semantic Segmentation},
  journal = {arXiv preprint arXiv:2312.03430},
  year    = {2023}
} % TCSVT 2025로 타 논문 표기 — 미확인

@article{bpcf2026,
  author  = {Cao, Hu and Cheng, Yifeng and He, Mu and Li, Mengyu and Liu, Yinlong and Knoll, Alois},
  title   = {Multimodal Fusion of {RGB} and Complementary Modalities for Semantic Segmentation},
  journal = {arXiv preprint arXiv:2610.17800},
  year    = {2026}
} % 방법 이름(BPCF)은 초록의 모듈 이름에서 따온 임시 표기

@article{hrfuser2023,
  author  = {Br{\"o}dermann, Tim and Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
  title   = {{HRFuser}: A Multi-resolution Sensor Fusion Architecture for {2D} Object Detection},
  journal = {arXiv preprint arXiv:2206.15157},
  year    = {2022}
} % ITSC 2023 게재로 타 논문 표기 — 미확인. HRFuser 수치는 CMNeXt·CAFuser가 측정한 값

@article{segrgbx2026,
  title   = {{SegRGB-X}: General {RGB-X} Semantic Segmentation Model},
  journal = {arXiv preprint arXiv:2603.28023},
  year    = {2026}
} % 저자 미확인(변환본 1쪽에서 저자 줄을 읽지 못함). 본표에는 쓰지 않을 것을 권함
```

미확인 상태로 남은 인용 대상: GtA(MUSES 82.39), Mul-VMamba(MCubeS 54.65), RoadFormer+(MM SAM-adapter 표에서만 확인).
