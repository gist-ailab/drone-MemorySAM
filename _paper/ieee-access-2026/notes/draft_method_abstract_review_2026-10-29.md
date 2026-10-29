# 방법 절 교체 초안 + 초안 초록 논리 리뷰 (확정 사실만, discussion 세션 2026-10-29)

paper 세션이 `latex/sections/4_method.tex` 에 옮길 영문 초안과, `notes/draft_abstract_intro_2026-10-29.md` 초록의 논리 리뷰다.

## 0. 근거(코드에서 직접 읽은 것)
config `configs/lecun-deliver_rgbdel_P46_c3only_seed20260821_screen40_A2hires.yaml`, `train_reliadino.py`, `semseg/models/reliadino/{encoder,fusion,model,hires_adapter,m2f_head}.py`, `semseg/datasets/degrade.py`.
| 서술 | 코드 위치 |
|---|---|
| LoRA rank 8, Q·V 에만, 센서별 | `encoder.py:35` MultiModalLoRAQKV, config LORA_R 8 |
| 블록 6·12·18·24 탭, 센서별 | config TAPS |
| 교차 attention 2층, query = 센서 s, key/value = 나머지 센서 토큰, 가중치 센서 공유 | `fusion.py:229-231, 953-961` |
| 융합 출력 = 센서별 결과 평균 + 센서별 MLP 잔차 × tanh(γ_s), γ_s 초기값 0.1 | `fusion.py:980`, `model.py:692-707, 1411-1416` |
| router: w = softmax_s(h_s(T_s) + λ·r_s), r_s = 1 − 엔트로피/log K(detach), h_s 마지막 층 0 초기화, 결과를 α(0 초기화)로 최종 로짓에 더함, 정규화 0.01, CE 0.4 | `fusion.py:63-100, 700-728, 990-993`, `model.py:1526, 1961-1968`, config ROUTER |
| 고해상 경로: 공유 stem + 센서별 1×1, extractor 탭당 1개 + 마지막 탭 2개 = 6개, injector 없음, FPN 에 게이트 없이 합산 | `hires_adapter.py:9, 158, 302-335`, `model.py:1368-1388` |
| query head: 100 queries, 6층, memory = 센서별 토큰 + 센서 임베딩 | `m2f_head.py:173-185`, config M2F |
| arbiter: ℓ = D + softplus(λ_k)·Q, λ 0 초기화(→ 시작 가중 0.69), 학습 시 path dropout 25% dense만 / 25% query만 / 50% 둘 다 | `model.py:1926-1953` |
| 열화 연산자: RGB = patch drop·blur·gamma/gain·colour shift, depth = patch drop·blur·gamma/gain·hole, LiDAR = patch drop·blur·beam drop, event = patch drop·resolution loss·blur | `degrade.py:159-164` |
| 센서별 열화 확률 0.5, 그중 0.2 완전 결측(정규화 후 0), 강도 상한 ep 0/10/20 에 0.3/0.6/1.0, 노이즈 학습 금지 | config TRAIN.QAF, `degrade.py` docstring |
| 열화 패스 손실 = OHEM CE + 0.5·T²·KL(교사‖학생), T 2, 픽셀 평균 / 보조 손실은 clean 패스에만 | `train_reliadino.py:283-293, 1085-1110` |
| clean 패스 손실 = OHEM CE + M2F 0.5 + 센서 보조 CE 0.5 + router CE 0.4 + router 정규화 0.01 + prototype 0.1 + VICReg(센서별 토큰, LiDAR 1.0·기타 0.25) | `train_reliadino.py:438, 893-914`, config |
| 교사 = 고해상 경로 없는 같은 구조, clean 입력, 더 긴 스케줄, 동결 | config TEACHER_CKPT(E1) |

쓰지 않는 것(결과 대기): "no gate", "no reliability signal", "Its parts are known … not a new loss"(FU-03), router 역할(FU-01), 게이트 값 해석(FU-02), "우리 구조에서만"(FU-04), "적응"(FU-05).

## 1. Method (LaTeX 초안)

```latex
\section{Method}\label{sec:method}
% Facts verified against the code on 2026-10-29: see notes/draft_method_abstract_review_2026-10-29.md §0.

We propose Clean-to-Degraded Two-Pass Distillation (C2D2), a training scheme for multi-sensor segmentation, and apply it to a fusion model built on a frozen vision foundation backbone (Fig.~\ref{fig:arch}). We first describe the model (Sections~\ref{sec:m_backbone}--\ref{sec:m_heads}) and then C2D2 (Section~\ref{sec:m_c2d2}).

% Figure block kept from the previous 4_method.tex (caption to be updated with the router and C2D2 naming).
\begin{figure*}[t]
\centering
% Source: figures/src/fig_arch_standalone.tex + latex/figures/fig_arch.tex (TikZ), built by build.sh.
\includegraphics[width=\textwidth]{figures/fig_arch.pdf}
\caption{Overview of \ours{} trained with C2D2. (a) Each sensor passes through the same frozen DINOv3-L with its own LoRA adapter; tokens from four depths are fused by cross-modal attention, decoded by a SimpleFPN augmented with a high-resolution path, and classified by a dense head and a query head combined by an arbiter. (b) C2D2: every batch is used twice, a clean pass and a synthetically degraded pass, the latter also distilled from a frozen teacher trained on clean inputs. Blue outlines mark trained components; dashed boxes are frozen. \todo{add the sensor router to the drawing or caption (FU-01); redraw per notes/figure\_guide.md}}
\label{fig:arch}
\end{figure*}

\subsection{Frozen Backbone with Sensor-Specific Adapters}\label{sec:m_backbone}
Let $x_s$ denote the input of sensor $s\in\{1,\dots,S\}$, converted to a three-channel image (Section~\ref{sec:exp}). Every $x_s$ is processed by the same DINOv3 ViT-L/16 backbone~\cite{dinov3_2025}, whose weights are frozen. Each sensor has its own low-rank adapters (LoRA)~\cite{lora2022} of rank 8 on the query and value projections of every attention block, so that each sensor can adapt the shared features to its own statistics. We take the token maps after blocks 6, 12, 18, and 24 and project each with a sensor-specific $1\times1$ layer, which gives a stride-16 token map $T_s$ per sensor.

\subsection{Cross-Modal Fusion}\label{sec:m_fusion}
Each sensor attends to all other sensors. In each of two layers, the tokens of sensor $s$ form the queries of a multi-head attention whose keys and values are the tokens of the other sensors,
\begin{equation}
\hat{T}_s \leftarrow \hat{T}_s + \mathrm{MHA}\big(\hat{T}_s,\ [\hat{T}_{s'}]_{s'\neq s}\big), \qquad \hat{T}_s^{(0)} = T_s,
\end{equation}
followed by a feed-forward layer. The attention weights are shared across sensors, so every sensor is treated by the same operation. The fused map is the average of the attended maps plus a sensor-specific residual,
\begin{equation}
F = \frac{1}{S}\sum_{s}\hat{T}_s + \sum_{s}\tanh(\gamma_s)\, m_s(T_s),
\end{equation}
where $m_s$ is a small MLP and $\gamma_s$ is a learned scalar per sensor, initialized at 0.1.

\subsection{Sensor Router}\label{sec:m_router}
Each sensor also has an auxiliary head that predicts class logits $A_s$ from $T_s$ and is trained with cross-entropy. A per-class router mixes these predictions pixel by pixel. Its weights are
\begin{equation}
w_{s,k}(p) = \operatorname{softmax}_{s}\big( h_{s,k}(T_s)(p) + \lambda\, r_s(p) \big),
\end{equation}
where $r_s = 1 - H(\sigma(A_s))/\log K$ is the normalized confidence of the auxiliary prediction of sensor $s$ (with gradients stopped), $K$ is the number of classes, and $h_s$ is a small network whose last layer is initialized at zero, so that routing starts from confidence alone. The routed logits $R_k=\sum_s w_{s,k} A_{s,k}$ are added to the final prediction with a learned scale $\alpha$ initialized at zero (Section~\ref{sec:m_heads}). The router is supervised with cross-entropy on $R$ and with a regularizer that favours decisive per-pixel weights. \todo{role of the router after FU-01 (DRN-261029-06/07/08)}

\subsection{Decoder with a High-Resolution Path}\label{sec:m_decoder}
A SimpleFPN~\cite{vitadapter2023} turns $F$ into a feature pyramid at strides 4, 8, 16, and 32. Because every level is interpolated from stride-16 tokens, structures thinner than a token are poorly represented. We therefore add a high-resolution path that follows the adapter of the official DINOv3 segmentation code~\cite{dinov3_2025,vitadapter2023}. A convolutional stem shared across sensors, followed by sensor-specific $1\times1$ projections, extracts multi-scale features from the raw inputs. Deformable-attention extractors, one for each tapped block and two more after the last, let these convolutional features sample semantic evidence from the tapped token maps. Unlike the original adapter, nothing is injected back into the backbone, which stays untouched. The output of the path is added to the FPN levels.

\subsection{Dense and Query Heads}\label{sec:m_heads}
A dense convolutional head classifies every pixel at stride 4 and gives logits $D$. A Mask2Former-style query head~\cite{mask2former2022} with 100 queries and six decoder layers attends to the token maps of all sensors, tagged with a sensor embedding, and its class and mask predictions are converted into per-pixel class scores $Q$. The final logits are
\begin{equation}
\ell_k(p) = D_k(p) + \operatorname{softplus}(\lambda_k)\, Q_k(p) + \alpha\, R_k(p),
\end{equation}
with a learned per-class weight $\lambda_k$. During training, each step uses the dense head alone, the query head alone, or both, with probabilities 0.25, 0.25, and 0.5, so that neither head can rely on the other; at inference both are used.

\subsection{Clean-to-Degraded Two-Pass Distillation}\label{sec:m_c2d2}
C2D2 uses every training batch $(x, y)$ twice.

\emph{Clean pass.} The model sees the clean inputs $x$ and is trained with the segmentation loss $\mathcal{L}_{\mathrm{clean}}$, which combines online hard-example cross-entropy~\cite{ohem2016} on $\ell$ with the losses of the query head, the auxiliary sensor heads, and the router, and two regularizers listed in Section~\ref{sec:exp}.

\emph{Degraded pass.} The model sees a degraded copy $\tilde{x}$. Each sensor of each sample is corrupted independently with probability 0.5. One fifth of the corrupted sensors are removed entirely by setting their normalized input to zero; the others receive one operator drawn from a sensor-specific set: patch dropout, blur, and gain and gamma changes for all image-like sensors, a colour shift for RGB, holes for depth, dropped beams for LiDAR, and resolution loss for events. The corruption strength is capped at 0.3, 0.6, and 1.0 of its maximum from epochs 0, 10, and 20 onward. Gaussian and salt-and-pepper noise are never used in training, so that the noise evaluation tests corruption that the model has not seen. The degraded pass is supervised by the labels and by a frozen teacher $g$, a model of the same architecture without the high-resolution path that was trained on clean inputs with a longer schedule. The teacher sees the clean input:
\begin{equation}
\mathcal{L}_{\mathrm{deg}} = \mathcal{L}_{\mathrm{ce}}\big(\ell(\tilde{x}), y\big) + \beta\, T^2\, \mathrm{KL}\!\left(\sigma\!\left(\tfrac{g(x)}{T}\right) \Big\|\, \sigma\!\left(\tfrac{\ell(\tilde{x})}{T}\right)\right),
\end{equation}
with $\beta=0.5$ and $T=2$, where $\sigma$ is the softmax over classes and the Kullback--Leibler divergence is averaged over pixels. The total loss is $\mathcal{L}_{\mathrm{clean}} + \mathcal{L}_{\mathrm{deg}}$.

C2D2 needs no condition labels and no degradation labels: the degradation is sampled on the fly, and the teacher only sees clean inputs. It changes training only; the model at inference is the same network run once. \todo{after FU-03 (L7, L5, L6): one paragraph on what each part contributes and how C2D2 differs from distillation with missing modalities~\cite{anyseg2024,rmmss2025}}
```

메모: 원래 원고의 식 기호 $\alpha$(KD 가중)는 router 스케일과 겹쳐 $\beta$ 로 바꿨다. `\label{sec:m_*}` 를 새로 붙였으므로 다른 절에서 참조할 때 이 이름을 쓴다. Fig.1 캡션의 "Blue outlines … dashed boxes" 설명은 router·arbiter 블록 표기가 그림에 있는지 확인 필요(그림 미확인).

## 2. 초안 초록의 논리 리뷰 (notes/draft_abstract_intro_2026-10-29.md)

판정: **뼈대(문제 → 방법 → 근거)는 유지하되, 아래 세 군데는 논리상 고쳐야 한다.**

1. **"best published fusion model" 이 틀릴 수 있다.** 같은 test 프로토콜의 공개 수치 중 MM SAM-adapter(RGB+depth 2센서)가 57.35 로 DGFusion(56.73, 4센서)보다 높다. 우리 57.74 는 MM SAM-adapter 보다 +0.39 뿐이다. → "**the best published four-sensor fusion model**" 로 한정한다. (출처: 원고 Table 1 MM SAM-adapter 57.35, ledger:130 DGFusion 56.73)
2. **강건성 비교가 불공정해 보인다.** 48.7 대 31.1 은 열화 학습을 하지 않은 공식 가중치와의 비교다. 같은 열화로 재학습한 DGFusion(45.52)·CAFuser(46.50)와 비교해도 우리가 높으므로(+2.2 / +3.1) 그것을 쓰는 편이 리뷰어 반론("증강하면 당연")을 막는다. → "48.7, compared with 31.1 for the official weights and at most 46.5 for the same baselines retrained with our degradation". 재학습 기준선은 단일 런이라 표에 그 사실을 병기. (출처: ledger:117, current.md CAFuser (b))
3. **문제 제기와 근거가 어긋난다.** 문제 문장은 "기준선은 조건 감독으로 악조건을 다룬다"인데, 초록에 우리가 **조건 감독 없이도 악조건에서 잘한다**는 근거가 없다(조건별 표 Table 5 미측정, FU-05). 지금 상태로는 조건 감독 대비는 "C2D2 needs no condition labels" 라는 비용 장점으로만 읽히고 성능 근거가 없다. → 두 가지 중 하나:
   - (a) Table 5 가 나올 때까지 문제 문장에서 "guiding the fusion with an inferred scene condition or with depth supervision" 을 빼고 "yet they lose much of their accuracy when a sensor fails" 만 남긴다(권장, 지금 쓸 수 있음).
   - (b) Table 5 결과로 "without condition supervision, it matches or exceeds them in every condition" 을 넣는다(FU-05 뒤).
4. (유지) C2D2 가 clean 이득의 원인이라는 문장은 초록에 쓰지 않는다. ablation 근거(+1.8)가 단일 최고 시드이고 L5·L6 대기다(FU-03·FU-09). 지금 초록은 이 인과를 명시하지 않으므로 논리적으로는 안전하다.
5. (유지) 범위(정확도 주·강건성 보조)와 순서는 맞다: 문제 → 방법 → 정확도 → 강건성 → 분석.

### 1·2·3(a) 반영 초록 (약 210단어)
```latex
\begin{abstract}
Multimodal semantic segmentation combines a camera with depth, event, and laser range sensors so that driving scenes are understood across weather, lighting, and sensor failures. The most accurate fusion models, however, lose much of their accuracy when a sensor fails at test time. We propose Clean-to-Degraded Two-Pass Distillation (C2D2), a training scheme that uses every batch twice: once with clean inputs and once with sensor-specific synthetic degradation, where the degraded pass is distilled from a frozen teacher trained on clean inputs. C2D2 needs no condition or degradation labels and adds no inference cost. We train with it a fusion model built on a frozen vision foundation backbone shared by all sensors, with sensor-specific low-rank adapters, cross-sensor attention, and a high-resolution decoding path. On the DELIVER benchmark, the model reaches 57.7 mean intersection over union on the test set, averaged over three training runs, one point above the best published four-sensor fusion model evaluated under the same protocol. When sensors are removed, its score averaged over all sensor subsets is 48.7, compared with 31.1 for the official weights of that model and at most 46.5 for strong fusion models retrained with the same degradation. We further analyse sensor dependence across conditions, thin objects, and failure cases \todo{one sentence on MUSES and MCubeS once measured (FU-08)}.
\end{abstract}
```
변경점: "symmetric" 삭제(정적 게이트 γ_s 값 확인 전, FU-02) · 조건 감독 구절 삭제(3a) · "four-sensor" 한정(1) · 재학습 기준선 병기(2).
