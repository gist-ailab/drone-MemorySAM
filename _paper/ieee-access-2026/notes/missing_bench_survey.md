# 결측·잡음 모달 강건성 벤치(arXiv 2503.18445) 기준 최신 논문 수치 조사

작성 2026-11-02. 모든 수치는 원문 PDF 표에서 직접 확인했다(`paper2md` 변환본 grep 후, 핵심 표는 PDF 쪽 이미지로 재확인).
PDF 위치: `/home/jemo/.claude/jobs/e0fe6210/tmp/papers/sota_survey/`(기존), `/home/jemo/.claude/jobs/e0fe6210/tmp/papers/missing_bench/`(이번에 받음).
"미확인" = 원문에서 찾지 못했거나 보고되지 않은 값. 쪽 번호는 PDF 쪽 번호다.

## §0. 2503.18445 원문 표 전사 (MiT-B2, DELIVER val, 1024×1024)

출처: Liao et al., "Benchmarking Multi-modal Semantic Segmentation under Sensor Failures: Missing and Noisy Modality Robustness", arXiv:2503.18445v3, CVPR 2025 Workshop(TMM-OpenWorld) 채택·최우수 논문상(저장소 README 기준), CVPR 2025 proceedings(workshop) pp. 1576–1586.
표 3~6은 PDF p5, 표 7~11은 PDF p6. 저장소 README 표와도 전 값 일치를 확인했다.

### 정의(원문 §3.2, p4~p6에서 확인)
- 학습: 전 모달(R-D-E-L)로 학습한 가중치를 그대로 평가한다. 평가 이미지 1024×1024, batch 4, A800 1장. "validation mIoU"로 정의(= DELIVER val).
- EMM: 결측 모달 데이터를 **0으로 채운다**(입력을 빼지 않는다). 남은 모달 조합 15개(R, D, E, L, RD, …, DEL, **RDEL 포함**)의 단순 평균이 mIoU_EMM^Avg다. 표 3의 15열 평균을 재계산하면 표 7 Avg와 소수 둘째 자리까지 일치한다(CMNeXt 37.90, GeminiFusion 37.07, MAGIC 44.97, MAGIC++ 44.85, StitchFusion 41.98). 즉 "15개 결측 조합"은 정확히는 "전체 조합을 포함한 15개 부분집합"이다.
- E(p): 각 모달이 확률 p로 독립 고장(Bernoulli)일 때 조합 확률 p^k(1-p)^(n-k)로 가중한 기대 mIoU, p = 0.2/0.1/0.05. 원문 p6: "Expected values are additionally normalized at the end"(모든 모달이 결측인 경우를 뺀 확률 합으로 정규화한 것으로 읽힌다; 정규화 식 자체는 원문에 없음).
- RMM: 각 모달 데이터의 비율 r을 무작위로 0으로 만든다(r = 0.75/0.5/0.25). Avg와 E(p)는 EMM과 같은 방식, p도 같은 값. 픽셀 단위인지 블록 단위인지는 원문에 명시가 없다.
- NM: 가우시안 N_G(μ=0, σ) + salt-and-pepper N_SP(D)를 동시에 더한다. Low(D=0.05, σ=0.1) / Mid(D=0.1, σ=0.2) / High(D=0.2, σ=0.5). event에는 N_G를 넣지 않고, 비 RGB 모달의 salt/pepper 값은 그 모달의 최대·최소값이다. 전 모달 입력 R-D-E-L에서 측정.

### 표 7 EMM (mIoU_EMM^Avg, E(p))
| 방법 | Avg | E p=0.2 | E p=0.1 | E p=0.05 |
|---|---|---|---|---|
| CMNeXt | 37.90 | 54.46 | 60.41 | 63.38 |
| GeminiFusion | 37.07 | 54.33 | 60.62 | 63.77 |
| MAGIC | 44.97 | 58.66 | 62.68 | 64.47 |
| MAGIC++ | 44.85 | 59.18 | 63.52 | 65.50 |
| StitchFusion | 41.98 | 58.02 | 63.29 | 65.80 |

### 표 8/9/10 RMM (Avg, E p=0.2 / 0.1 / 0.05)
| 방법 | r=0.75 Avg | E .2/.1/.05 | r=0.5 Avg | E .2/.1/.05 | r=0.25 Avg | E .2/.1/.05 |
|---|---|---|---|---|---|---|
| CMNeXt | 42.17 | 56.66 / 61.60 / 63.99 | 47.49 | 58.85 / 62.68 / 64.53 | 53.61 | 61.28 / 63.86 / 65.11 |
| GeminiFusion | 39.78 | 55.88 / 61.47 / 64.22 | 42.41 | 57.30 / 62.25 / 64.62 | 49.74 | 60.55 / 63.91 / 65.46 |
| MAGIC | 45.30 | 58.77 / 62.72 / 64.49 | 48.19 | 59.53 / 63.01 / 64.61 | 51.61 | 60.46 / 60.37 / 64.76 |
| MAGIC++ | 47.06 | 59.81 / 63.78 / 65.62 | 49.31 | 60.50 / 64.07 / 65.75 | 53.92 | 62.07 / 64.78 / 66.08 |
| StitchFusion | 45.16 | 59.44 / 64.02 / 66.17 | 48.33 | 60.58 / 64.53 / 66.41 | 53.10 | 62.34 / 65.39 / 66.85 |

주의: MAGIC r=0.25의 E(p=0.1)=60.37 < E(p=0.2)=60.46은 원문 그대로다(단조성이 깨져 원문 오기 가능성이 있다; 고치지 않고 옮겼다).

### 표 11 NM (mIoU_NM, R-D-E-L 입력)
| 방법 | Low | Mid | High |
|---|---|---|---|
| CMNeXt | 35.23 | 16.37 | 2.31 |
| GeminiFusion | 5.52 | 4.14 | 3.30 |
| MAGIC | 24.03 | 13.31 | 3.98 |
| MAGIC++ | 33.12 | 18.31 | 8.70 |
| StitchFusion | 24.91 | 17.11 | 9.25 |

참고(표 3 RDEL 열 = 깨끗한 전 모달 mIoU): CMNeXt 66.33, GeminiFusion 66.92, MAGIC 66.10, MAGIC++ 67.34, StitchFusion 68.20.
메인 세션이 적어 준 값(표 7 Avg, 표 10 Avg, 표 11 전부)은 원문과 모두 일치한다.

## §1. 같은 정의(EMM/RMM/NM, 2503.18445 인용)로 점수를 보고한 최신 논문

본표 후보는 2편이다(EQUISeg, RMMSS). 두 논문 모두 같은 그룹(BUPT·HKUST(GZ), Xu Zheng 공저)이고, 기준선 행은 2503.18445 표 7·9·11 값을 그대로 옮겨 왔다(값 일치로 확인).

### §1-a. DELIVER
| 방법 | 학회·연도 | arXiv | 백본 | split | 깨끗한 mIoU | EMM Avg | EMM E(p=.2/.1/.05) | RMM Avg (r) | RMM E(p=.2/.1/.05) | NM Low/Mid/High | 출처 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EQUISeg | 미확인(5쪽 IEEE 학회 형식, 채택 학회 표기 없음) 2025 | 2509.24505 | MiT-B2 | 미확인(본문 미명시; 벤치 정의상 val) | 67.90 | **48.22** | 미보고 / **65.75** / 미보고 | **50.96 (r=0.5)** | 미보고 / **64.64** / 미보고 | **34.87 / 19.13** / 미보고 | Table 1, PDF p4, 자기 보고 |
| RMMSS | 미확인(AAAI 템플릿 ©2026, 채택 여부 미확인) 2025 | 2505.12861 v2 | **MiT-B0** | 미확인 | 61.85(RDEL, 표 1) | **49.93** | 미보고 | **51.90 (r 미표기, 아래 주)** | 미보고 | 미보고 | Table 2, PDF p6, 자기 보고 |

정의 일치 여부:
- EQUISeg: 2503.18445를 [19]로 인용하고 EMM/RMM/NM을 그 벤치로 평가한다고 쓴다(p3~p4). 기준선 4개 행(CMNeXt·MAGIC·MAGIC++·StitchFusion)이 원문 표 7(Avg, p=0.1), 표 9(r=0.5의 Avg, p=0.1), 표 11(Low, Mid)과 정확히 같으므로 **RMM은 r=0.5**로 판단한다. 0 채움·15 조합·1024 평가 여부는 본문에 따로 적혀 있지 않다(학습 crop 1024만 명시). 표 1 캡션은 EMM/RMM을 "random removal, block-wise removal"이라고 설명하는데, 이는 원문 정의(EMM = 모달 전체 0, RMM = 비율 r 무작위 0)와 표현이 다르다. 기준선이 원문 값 재인용이라 우리 쪽 비교 시 같은 표에 놓을 수 있다. 저자 "Mean"(robustness score 50.21)은 mIoU·EMM·RMM·NM 6개 열을 평균한 저자 고유 지표이므로 옮기지 않는다.
- RMMSS: 2503.18445를 "sensor failure benchmark (Liao et al. 2025)"로 인용하고 EMM·RMM·NM으로 시험했다고 쓰지만 "Detailed metrics are provided in Appendix C"의 부록은 arXiv v2 PDF(9쪽)에 없다. 표 2의 CMNeXt·MAGIC·MAGIC++ DELIVER 행은 원문 표 7 Avg와 표 8(r=0.75) Avg와 같으므로 RMM 열은 **r=0.75로 추정**한다(본문 미표기). 그러나 같은 표의 AnySeg 행(46.64)은 AnySeg 원 논문(2411.17141, Table 2 p5)의 "Anymodal Evaluation" 평균과 같고, RMMSS 자신의 EMM 49.93도 자기 표 1의 AnySeg 방식 15조합 평균과 같다. 즉 **EMM 열은 사실상 15조합 평균(=EMM Avg와 같은 산술 정의)이지만, 결측을 0으로 채웠는지는 본문에 없다.** 또 우리 방법은 B0, 비교 기준선은 B2로 백본이 섞여 있다. NM 수치는 표에 없다.

### §1-b. MCubeS·MUSES (같은 정의 표기, 참고용)
| 방법 | 데이터 | 백본 | EMM Avg | EMM E(p=0.1) | RMM Avg | RMM E(p=0.1) | NM Low/Mid | 출처 |
|---|---|---|---|---|---|---|---|---|
| EQUISeg | MUSES | 본문 충돌: §3.2 "All models use MIT-B2" vs 구현 상세 "MUSES … with MIT-B0" | 35.63 | 45.06 | 38.61 (r 미표기) | 47.63 | 20.47 / 12.62 | Table 1, p4 |
| EQUISeg 표의 기준선(MUSES) | MUSES | 같음 | CMNeXt 20.80 · MAGIC 33.34 · MAGIC++ 34.52 · StitchFusion 26.59 | 31.48 / 42.37 / 43.84 / 40.11 | 26.53 / 37.02 / 37.34 / 35.66 | 42.17 / 46.24 / 46.90 / 45.38 | CMNeXt 18.32/10.52 · MAGIC 13.34/8.46 · MAGIC++ 15.43/9.15 · StitchFusion 12.44/10.28 | Table 1, p4, 출처 미표기(자체 측정으로 보임, 미확인) |
| RMMSS | MCubeS | B0 | 43.22 | 미보고 | 44.18 | 미보고 | 미보고 | Table 2, p6 |
| RMMSS | MUSES | B0 | 41.12 | 미보고 | 42.78 | 미보고 | 미보고 | Table 2, p6 |
| RMMSS 표의 기준선 | MCubeS / MUSES | CMNeXt·MAGIC·MAGIC++ B2, AnySeg B0 | MCubeS 40.67 / 42.61 / 42.53 / 39.33 · MUSES 26.43 / 33.34 / 33.17 / 40.23 | 미보고 | MCubeS 41.25 / 42.25 / 44.61 / 42.50 · MUSES 33.85 / 42.49 / 43.24 / 42.66 | 미보고 | 미보고 | Table 2, p6 |

주: 같은 그룹 두 논문의 MUSES CMNeXt EMM 기준선이 서로 다르다(EQUISeg 20.80 vs RMMSS 26.43). MUSES용 공식 EMM 기준선은 2503.18445에 없으므로(원문은 DELIVER만) MUSES 비교는 각 논문 자체 측정에 기대며 서로 호환되지 않는다.

## §2. 정의가 다른 논문(본표 제외 후보)

공통 차이: 아래는 "전 모달로 학습 후 모달 부분집합으로 평가"하는 modality-agnostic 계열(MAGIC·Any2Seg·AnySeg 프로토콜)이다. 대부분 2503.18445를 인용하지 않거나 참고문헌에만 두고, E(p)·RMM·NM을 보고하지 않는다. 15조합 평균이라는 산술은 EMM Avg와 같아도 **결측 처리(0 채움 vs 입력 제거·마스킹)와 기준선 수치의 출처가 다르다.** 특히 이 계열이 쓰는 CMNeXt DELIVER 평균 20.77(B0, MAGIC 논문 유래)은 2503.18445의 CMNeXt-B2 EMM Avg 37.90과 크게 다르다.

| 방법 | 학회·연도 | arXiv/DOI | 백본 | 데이터 | 보고 지표·값(DELIVER, 15조합 평균) | 2503.18445와 다른 점 | 출처 |
|---|---|---|---|---|---|---|---|
| CrossWeaver | 미확인(arXiv v2 2026-06) | 2604.02948 | MiT-B0 | MCubeS, DELIVER | 40.52 (RDEL 64.69); MCubeS 32.68 | 결측 모달을 **0 텐서로 대체**(p6), 15조합 평균이라 산술은 EMM Avg와 같다. 다만 2503.18445 인용 없음, E(p)·RMM·NM 없음, B0. CMNeXt 20.77·StitchFusion 36.72는 자체 재현(B0). 정의상 가장 가깝지만 백본이 B0이라 원문 B2 표와 바로 같은 열에 놓기 어렵다 | Table 2, PDF p7 |
| FuseAdapt | Preprint 2026 | 2610.21779 | 동결 DINOv2-B(ViT-B) + 어댑터 | MCubeS, MUSES, DELIVER | 56.22 (RDEL 68.33); MUSES 61.30(ViT-S); MCubeS 52.89 | 결측 센서를 **마스킹**(입력 제거, Appendix p21 "absent sensors are masked"). 2503.18445 미인용. 그런데 표 6의 CMNeXt·GeminiFusion·MAGIC·MAGIC++·StitchFusion 조합별 값은 2503.18445 표 3 값과 같다. 평균 열은 CMNeXt 41.31, GeminiFusion 42.35, MAGIC++ 44.78, StitchFusion 43.91로 표 3의 15열 평균(37.90/37.07/44.85/41.98)과 다르다(MAGIC 44.97만 일치) → 평균 계산 오류로 보인다. 인용 시 주의 | Table 6, PDF p9 |
| FunEntropy(Zheng et al.) | ICCV 2025(DOI 10.1109/ICCV51701.2025.01966, Semantic Scholar 기준) | 2505.06635 | 본문 미명시(FuseAdapt 표 6은 MiT-B2로 표기) | DELIVER, MUSES, MCubeS | 48.29 (RDEL 51.65); MUSES 47.80 | 2503.18445는 참고문헌 [62]에만 있고 평가에 쓰지 않는다. "Training with All Modalities and Evaluation with Different Combinations" 평균, 결측 처리 방식 본문 미명시. 전 모달 mIoU가 51.65로 낮다 | Table 3, PDF p6 |
| CHARM | 미확인(AAAI 형식, arXiv v2 2025-08) | 2508.03060 | MiT-B0 / MiT-B2 | DELIVER, MCubeS, MUSES | B2: Average 54.96, Top-1 68.43, Last-1 28.76; B0: 50.97 | 2503.18445 미인용. 조합 평균·최고·최저(Average/Top-1/Last-1)라는 별도 지표, 결측 처리 방식 미명시 | Table 1, PDF p6 |
| BiXFormer | IEEE TMM 2026(DOI 10.1109/TMM.2026.3703333, Semantic Scholar 기준) | 2506.03675 | ResNet-34 | DELIVER, MUSES, NYUv2 | 43.24 | 2503.18445는 참고문헌에만 있다. 조합 평균만 보고, 결측 처리 미확인 | Table II, PDF p6 |
| EGFormer | 미확인(arXiv 2025-05) | 2505.14014 | MiT-B0 | DELIVER, MCubeS, MUSES | 31.60 | 2503.18445는 관련연구 참고문헌에만 있다. "Validation with arbitrary modality combinations" 평균 | Table 5, PDF p8 |
| AnySeg | 2024-11(벤치 이전) | 2411.17141 | SegFormer-B0 | DELIVER, MUSES | 46.64 (RDEL 60.26); MUSES 40.23 | 벤치보다 먼저 나온 논문. "Anymodal Evaluation" 평균. RMMSS 표 2가 이 값을 EMM 열에 그대로 넣었다 | Table 2, PDF p5 |
| MARS | 미확인 2026 | 2606.30355 | DeepLabv3+ | MCubeS(분류 데이터셋 3종 포함) | 해당 없음 | 결측 모달 학습(IML) 설정이며 DELIVER 없음, 2503.18445 미인용 | Table 2 |

조사했으나 해당 없음: BPCF(2610.17800, 결측 모달 평가 없음), GeomPrompt(2604.11585, RGB-D SUN RGB-D), Condition Dropout(2607.20326, RGB-D NYUv2·SUN), UMSS(2607.12372, 결측 평가 키워드 없음), 2605.20372(원격탐사 데이터).
확인 못 함: "Reliability-Aware Fusion for Semantic Segmentation under Sensor Degradation and Failures"(IEEE IV 2026, DOI 10.1109/IV66570.2026.11623893)는 Semantic Scholar 인용 목록에 2503.18445 인용 논문으로 나오지만 arXiv 판이 없어 PDF·수치를 확인하지 못했다. BiSPRNet(DOI 10.1117/12.3136414)은 RGB-T라 범위 밖으로 보았으나 원문은 보지 않았다.

검색 경로: Semantic Scholar 인용 API(2503.18445를 인용한 30편), alphaXiv 검색("EMM RMM DELIVER missing modality"), 웹 검색. 2026-11-02 기준.

## §3. references.bib에 없는 논문 BibTeX 제안

`latex/references.bib`에 이미 있는 것: mmssbench2025(2503.18445), rmmss2025, fuseadapt2026, robustseg2025(=EQUISeg 2509.24505), anyseg2024, zheng2025entropy(저자 TODO).
저자·제목은 각 PDF 1쪽에서 확인했다. 학회는 확인된 것만 넣고 나머지는 arXiv로 둔다.

```bibtex
% 2503.18445 venue 보강 제안(기존 mmssbench2025 대체 시): CVPR 2025 Workshop(TMM-OpenWorld), pp. 1576-1586 (저장소 README·EQUISeg 참고문헌 [19] 기준)

% zheng2025entropy의 TODO 해소용 저자 목록 (2505.06635 p1)
%   author = {Zheng, Xu and Lyu, Yuanhuiyi and Jiang, Lutao and Paudel, Danda Pani and Van Gool, Luc and Hu, Xuming}

@article{crossweaver2026,
  author  = {Zhang, Zelin and Li, Kedi and Liang, Huiqi and Xu, Chuanzhi and Zhang, Tao and Cai, Weidong},
  title   = {{CrossWeaver}: Cross-modal Weaving for Arbitrary-Modality Semantic Segmentation},
  journal = {arXiv preprint arXiv:2604.02948},
  year    = {2026}
}

@article{charm2025,
  author  = {Wen, Lekang and Xiao, Jing and Liao, Liang and Chen, Jiajun and Wang, Mi},
  title   = {{CHARM}: Collaborative Harmonization across Arbitrary Modalities for Modality-agnostic Semantic Segmentation},
  journal = {arXiv preprint arXiv:2508.03060},
  year    = {2025}
}
% CHARM 1쪽 저자 줄이 "Mi Wang1 ," 뒤에서 잘려 추가 저자가 있을 수 있다(미확인).

@article{bixformer2025,
  author  = {Chen, Jialei and Zheng, Xu and Paudel, Danda Pani and Van Gool, Luc and Murase, Hiroshi and Deguchi, Daisuke},
  title   = {{BiXFormer}: A Robust Framework for Maximizing Modality Effectiveness in Multi-Modal Semantic Segmentation},
  journal = {arXiv preprint arXiv:2506.03675},
  year    = {2025}
}
% Semantic Scholar 기준 IEEE TMM 게재(DOI 10.1109/TMM.2026.3703333) — 1차 출처 미확인

@article{egformer2025,
  author  = {Zhang, Zelin and Zhang, Tao and Li, Kedi and Zheng, Xu},
  title   = {{EGFormer}: Towards Efficient and Generalizable Multimodal Semantic Segmentation},
  journal = {arXiv preprint arXiv:2505.14014},
  year    = {2025}
}
```

## 작업 중 사고 기록
- 조사 중 `paper2md`에 PDF 여러 개를 한 번에 넘겨 두 번째 인자가 출력 경로로 해석되면서 `sota_survey/2505.12861.pdf`가 2503.18445 변환본으로 덮어써졌다. arXiv에서 v2를 다시 받아 같은 경로에 복구했고, 바이트 크기 2,751,628이 원래 목록의 크기와 같음을 확인했다(내용 해시는 원본과 비교하지 못했다).
