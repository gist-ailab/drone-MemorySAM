# 초록·서론 교체 초안 (확정된 사실만, discussion 세션 2026-10-29)

paper 세션이 `latex/sections/0_abstract.tex`·`1_intro.tex` 에 옮길 영문 초안이다. 규칙은 `notes/followups.yaml`(FU-03·04·05·06·07·08)과 브리프 `.claude_logs/paper/00_brief.md` "2026-10-29 정정" 절을 따른다.
- 범위: 전 모달 정확도가 주, 결측 강건성은 보조. 4모달 필요성은 방어하지 않는다.
- 기여 ① = C2D2(두 패스 학습). "선행과의 차이"(FU-03)·"우리 구조에서만"(FU-04)·"조건별 적응"(FU-05) 문장은 결과 전까지 넣지 않는다 → 아래 `\todo` 자리.
- "clean training(only)" 표현 금지: 기준선도 데이터셋의 악천후·야간 장면을 학습했다. 차이는 합성 센서 열화의 유무다.
- 수치 출처: 57.74 = headline.yaml deliver.ours[0](A2 3시드, legal v2 test@1024) · 56.73·31.14 = judgment-ledger 2026-10-01 DRN-261001-03(DGFusion 공식 가중치, 같은 프로토콜) · 48.66 = tasks/DRN-261019-03/analysis.md(A2 3시드 test EMM15). 격차 57.74 − 56.73 = +1.01.
- 초록 규칙(Access): 한 문단 150–250단어, 약어·각주·인용·수식 금지 → C2D2 는 방법 이름이라 풀이와 함께 한 번만 쓴다(편집 규정상 문제 되면 풀이만 남김). 아래 초안 약 200단어.

## Abstract

```latex
\begin{abstract}
Multimodal semantic segmentation combines a camera with depth, event, and laser range sensors so that driving scenes are understood across weather, lighting, and sensor failures. The most accurate fusion models handle adverse conditions by guiding the fusion with an inferred scene condition or with depth supervision, yet they still lose much of their accuracy when a sensor fails at test time. We propose Clean-to-Degraded Two-Pass Distillation (C2D2), a training scheme that uses every batch twice: once with clean inputs and once with sensor-specific synthetic degradation, where the degraded pass is distilled from a frozen teacher trained on clean inputs. C2D2 needs no condition or degradation labels and adds no inference cost. We train with it a fusion model built on a frozen vision foundation backbone shared by all sensors, with sensor-specific low-rank adapters, symmetric cross-sensor attention, and a high-resolution decoding path. On the DELIVER benchmark, the model reaches 57.7 mean intersection over union on the test set, averaged over three training runs, one point above the best published fusion model evaluated under the same protocol. When sensors are removed, its score averaged over all sensor subsets is 48.7, compared with 31.1 for the official weights of that model. We further analyse sensor dependence across conditions, thin objects, and failure cases \todo{one sentence on MUSES and MCubeS once measured (FU-08)}.
\end{abstract}

\begin{keywords}
% IEEE Access template: three or four keywords, alphabetical order
Knowledge distillation, multimodal semantic segmentation, robustness, sensor fusion
\end{keywords}
```

## Introduction

```latex
\section{Introduction}\label{sec:intro}
\PARstart{A}{utonomous} vehicles and mobile robots carry several sensors because each one fails in different conditions. A camera loses contrast at night and in fog, a depth estimate develops holes on reflective surfaces, an event camera goes silent on static scenes, and LiDAR returns thin out in rain and with distance. Multimodal semantic segmentation fuses these sensors so that the scene is still understood when one of them degrades.

The fusion models that lead the full-sensor benchmarks~\cite{cmnext2023,cmx2023,stitchfusion2024,geminifusion2024,cafuser2025,dgfusion2026} handle adverse conditions by telling the fusion what the condition is. CAFuser~\cite{cafuser2025} infers a condition token from the camera image and supervises it with a textual description of the weather and lighting, and DGFusion~\cite{dgfusion2026} adds local depth tokens and depth supervision. These models see the adverse scenes contained in their training data, but they are not trained with degraded or missing sensors, and they lose much of their accuracy when a sensor actually fails: under one evaluation protocol on DELIVER, the official DGFusion weights fall from 56.7 to 31.1 mIoU when averaged over all subsets of removed sensors. Methods designed for missing sensors~\cite{magic2024,anyseg2024,rmmss2025,fuseadapt2026} train with modality dropout and distillation, but the two lines of work rarely evaluate the same models under the same protocol~\cite{mmssbench2025}. Frozen vision foundation models offer strong features for every sensor, yet their multimodal adaptations mostly keep RGB as the primary stream and inject one auxiliary sensor~\cite{mmsamadapter2025}.

We ask whether a change of the training signal, instead of additional condition supervision, can give an accurate fusion model the ability to hold up when its sensors degrade. We propose Clean-to-Degraded Two-Pass Distillation (C2D2). Every training batch is used twice, once with clean inputs and once with sensor-specific synthetic degradation, such as blur and exposure changes for the camera, holes for depth, dropped beams for LiDAR, and lost resolution for events, together with entire sensor removal. The degraded pass is supervised by the labels and distilled from a frozen teacher that sees the clean input. C2D2 needs no condition or degradation labels and adds no computation at inference. We apply it to a fusion model that reads every sensor with one frozen DINOv3 backbone~\cite{dinov3_2025} through sensor-specific low-rank adapters, fuses the sensors with symmetric cross-modal attention, and recovers thin structures with a high-resolution decoding path.

On DELIVER~\cite{cmnext2023}, the model reaches 57.74 mIoU on the test set (mean of three training runs), 1.0 point above the official DGFusion weights evaluated under the same protocol, and it keeps 48.7 mIoU when averaged over all subsets of removed sensors, compared with 31.1 for DGFusion \todo{MUSES and MCubeS results (FU-08)}. We also analyse how the dependence on each sensor changes across conditions, where the high-resolution path helps, and where the model still fails.

Our contributions are as follows.
\begin{itemize}
\item We propose C2D2, a label-free two-pass training scheme that combines sensor-specific synthetic degradation with distillation from a clean-input teacher and adds no inference cost. \todo{after FU-03 (L7) and FU-04 (DGFusion/CAFuser transfer): one sentence on how it differs from missing-modality distillation and whether its gain depends on the frozen foundation backbone}
\item We build a single fusion model on a frozen vision foundation backbone that, trained with C2D2, improves full-sensor accuracy on DELIVER \todo{and MUSES and MCubeS (FU-08)} while remaining robust under the public missing-sensor protocol.
\item We analyse sensor dependence across conditions, missing, partial, and noisy sensors, thin objects, and failure cases, and report where the model still fails. \todo{state the code and checkpoint release}
\end{itemize}
```

## 기존 원고에서 빠지는 것(어디로 가나)
- "We investigate why"·"over-reliance is learned from clean training data rather than … architecture": 분석 절(6_analysis) 주장으로만, FU-05·FU-04 결과에 맞춰 수위 조정.
- depth 제거 23–39점: 분석 절, test 통일 재측정 뒤(FU-05).
- val EMM 57.85 "+7.9 vs best published": 실험 절 강건성 표, 계열 분리 표기 뒤(FU-06).
- "reducing depth over-reliance to a third or less": Q2 측정값이라 A2 재측정 뒤 분석 절(FU-05).
