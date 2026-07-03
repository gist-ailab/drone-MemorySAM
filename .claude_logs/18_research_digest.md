# 리서치 다이제스트 — 옵시디언 볼트 동기화 (2026-07-02)

> **출처/스코프**: Obsidian 볼트 `/nas_jm/Research/26_MultimodalSeg/`의 2026-07-02 사본(`research_vault/`) 전체 + 병렬 딥리서치 8트랙(2-skeptic adversarial 검증 포함) 결과를 아이디어 회의·구현용으로 압축한 작업 문서.
> **Canonical 규칙 (한 줄)**: 벤치마크 숫자는 `research_vault/relatedworks/09_benchmark_tables_deliver_muses_mcubes.md`, RBMA/프로젝트 결정은 `12_novelty_and_related_work.md`가 canonical — 충돌 시 프로젝트 결정은 doc 12, 외부 논문 수치는 볼트 09 우선.
> 상세 파일 맵: `research_vault/README.md`.

**결정적 사실 5줄 (회의 스타터)**:
1. DELIVER "two-cluster"는 **split×backbone**으로 해소 (66.30=B2·val / 53.0=B2·test / 59.18=B0·val) — P28 val~55의 올바른 비교선은 **CAFuser val 67.8** (test 55.6 아님).
2. **MM-SAM-adapter(2509.10408)가 2모달로 DELIVER test 57.35·MUSES test 81.07** — "VFM 멀티모달 SOTA" 헤드라인은 스코프 한정 없이는 불가.
3. RBMA logit-bias 셀은 **PRIMED(2605.07154)·SAE(2603.16558) 발견으로 축소** — 생존 claim = training-free × predictive-entropy × pre-softmax additive × SAM2 **memory** attention × dense seg (fenced, §2).
4. P29 셀(무감독 condition latent→FiLM-on-gate)은 **미점유 confirmed** — 단 MLE-SAM(2412.04220)·DAMP(2512.20251) 선제 인용 + scoped wording 의무 (§3.1).
5. 조건-적응 스토리의 타깃 컬럼 확보: DGFusion 최약 = MUSES **Night 58.97 / Fog 58.86 PQ**; MUSES **AUPQ 트랙은 아무도 선점 안 함** (§1.6, §6-6).

---

## 1. 벤치마크 정량 (인용 가능한 숫자만)

모든 행: method / backbone / modality / split 태그 / 수치 / 출처 표. 출처 노트: `research_vault/relatedworks/09` (§1–8 + 2026-07-02 §U1–U9 확정판).

### 1.1 DELIVER "two-cluster" 문제 — 해소 (실제로는 3클러스터) [skeptic-confirmed ×2]

| 보고된 CMNeXt 수치 | 실제 config | 근거 | 태그 |
|---|---|---|---|
| **66.30** | MiT-**B2** · **val** · RGB-D-E-L (1024×1024 crop) | CMNeXt Tab.1(a); DELIVER repo README; CAFuser Tab.III "mIoU-val 66.3" | [VERIFIED-PDF]+[REPO] [val] |
| **53.0** | MiT-**B2** · **test** · CLDE | CAFuser Tab.III; DGFusion Tab.III | [VERIFIED-PDF]×2 [test] |
| **59.18** | MiT-**B0** · **val** · RGB-D-E-L | MemorySAM Tab.1, MAGIC, AnySeg Tab.2, EGFormer Tab.2, MLE-SAM Tab.I | [VERIFIED-PDF]×4 [val] |

- 원인 = **split(val/test) × backbone(B0/B2)**. 해상도는 무관 (동일 모델 컨트롤: CAFuser Tab.III에서 같은 CMNeXt가 66.3 val / 53.0 test, 같은 CAFuser-CA²가 67.8 val / 55.6 test → ~12–13pt 갭 = split).
- 코드 증거: DELIVER repo `tools/val_mm.py` L141 `'val'` 하드코딩; MemorySAM repo `val_mm_sam.py` L146도 `split='val'` 하드코딩 → **MemorySAM 65.38 = [val, code-inferred]** (논문에 split 미표기).
- Split 규모: train 3,983 / val 2,005 / test 1,897, 25클래스.
- ⚠️ **우리 P28 비교 규칙 (doc 12 §2.5)**: **P28 val~55는 CAFuser val 67.8과 비교해야 함** (test 55.6과 비교 금지). DGFusion 56.7 / CAFuser 55.6은 [test] → P28 **test** mIoU로만 비교. **65.38(val, Hiera)과 56.7(test, Swin-T)을 한 열에 절대 놓지 말 것.**

### 1.2 DELIVER SOTA — split-tagged 확정판 (09 §U3)

**(a) [val] · B2/Swin 클러스터 (高):**

| Method | arXiv/venue | Backbone | mIoU | Split | 출처/태그 |
|---|---|---|---:|---|---|
| StitchFusion | 2408.01343, ACM MM'25 | Swin-Tiny-1k | **70.34** | [val]* caption 미표기, CAFuser Tab.III 교차추론 | [VERIFIED-PDF]* |
| Mul-VMamba | KBS 334:115119 | VMamba 55.33M | 68.98 | [unknown, likely val] | [ABSTRACT-ONLY] |
| CAFuser-CAA | 2410.10791, RA-L'25 | Swin-T | 68.6 | [val] | CAFuser Tab.III [VERIFIED-PDF] |
| StitchFusion | 2408.01343 | MiT-B2 | 68.18 | [val] | StitchFusion Tab.7 [VERIFIED-PDF] |
| OmniSegmentor | 2509.15096, NeurIPS'25 | DFormer-L | 68.0 | [val] | Tab.1(f) [VERIFIED-PDF] |
| EQUISeg | 2509.24505 | n/s | 67.90 | [unknown; val-cluster 수치] | [ABSTRACT-ONLY] |
| CAFuser-CA² | 2410.10791 | Swin-T | 67.8 | [val] | CAFuser Tab.III [VERIFIED-PDF] |
| MAGIC | 2407.11344, ECCV'24 | SegFormer-B2 | 67.66 | [val, cluster-inferred] | [VERIFIED-PDF] |
| HyperDUM | 2503.20011, CVPR'25 | CMNeXt-B2+UQ | 67.59 (10-case mean) | [val] | HyperDUM Tab.4 [VERIFIED-PDF] |
| GeminiFusion | 2406.01210, ICML'24 | MiT-B2 | 66.9 | [val, val-as-test] | [VERIFIED-PDF] |
| CMNeXt | 2303.01480, CVPR'23 | MiT-B2 | 66.30 | [val] | CMNeXt Tab.1(a) [VERIFIED-PDF] |

**(b) [val] · B0/SAM 클러스터 (MemorySAM 비교 universe):**

| Method | arXiv | Backbone | RDEL mIoU | Anymodal mean | Split | 출처 |
|---|---|---|---:|---:|---|---|
| MemorySAM | 2503.06700 | SAM2 Hiera-B+ (1 LoRA) | **65.38** | — | [val, code-inferred] | MemorySAM Tab.1 |
| MLE-SAM | 2412.04220 | Hiera-B+ MoE-LoRA | 64.08 | — | [val, 동일 프로토콜] | MLE-SAM Tab.I |
| MAGIC | 2407.11344 | SegFormer-B0 | 63.40 | 40.49 | [val] | MAGIC++ Tab.II |
| AnySeg | 2411.17141 | SegFormer-B0 | 60.26 | **46.64** | [val] | AnySeg Tab.2 |
| RobustSeg/RMMSS | 2505.12861 | MiT-B0 | 60.16 | 49.89 (robustness mean) | [val] | [VERIFIED-PDF] |
| EGFormer | 2505.14014 | SegFormer-B0 | 59.53 | — | [val] | [VERIFIED-PDF] |
| CMNeXt | — | SegFormer-B0 | 59.18 | 20.77 | [val] | MemorySAM Tab.1 |
| MAGIC++ | 2412.16876 | SegFormer-B0 | 47.74 (anymodal-trained) | 48.67 MaSS | [val] | MAGIC++ Tab.II |
| FunEntropy-Reg | 2505.06635 | (B0 line) | — | 48.29 | [val] | 그 논문 Tab.3 |

**(b′) MemorySAM Tab.1/Tab.2 — 모달리티 추가에 따른 진행 (우리 P28 모달 ablation의 직접 비교선, 전부 [val, code-inferred]):**

| Dataset | Method | R(GB) | +D | +E | +L(전체) | 출처 |
|---|---|---:|---:|---:|---:|---|
| DELIVER | CMNeXt (MiT-B0) | 51.29 | 59.61 | 59.84 | 59.18 | MemorySAM Tab.1 |
| DELIVER | SAM-LoRA (ViT-B, 1 LoRA) | 51.84 | 60.25 | 60.08 | 59.54 | MemorySAM Tab.1 |
| DELIVER | MLE-SAM (Hiera-B+, 4 LoRA) | 55.23 | 63.57 | 62.69 | 64.08 | MemorySAM Tab.1 |
| DELIVER | **MemorySAM** (Hiera-B+, 1 LoRA) | 53.22 | 63.48 | 62.42 | **65.38** | MemorySAM Tab.1 |
| MCubeS | CMNeXt (MiT-B0) | R-A 37.21 | R-A-D 38.72 | — | R-A-D-N 36.16 | MemorySAM Tab.2 |
| MCubeS | **MemorySAM** | R-A 51.20 | R-A-D 52.20 | — | R-A-D-N **52.88** | MemorySAM Tab.2 |

- 관찰: MemorySAM은 RGB 단독에서 MLE-SAM에 뒤지고(53.22 vs 55.23) 4모달에서 역전(65.38 vs 64.08) — memory-attention 융합의 이득이 모달 수와 함께 커짐. P28 모달 ablation 보고 시 같은 형식 사용.

**(c) [test] (CAFuser/DGFusion 프로토콜):**

| Method | Backbone | CLE test | CLDE test | 출처 |
|---|---|---:|---:|---|
| CMNeXt | MiT-B2 | 50.3 | 53.0 | DGFusion Tab.III |
| StitchFusion | MiT-B2 | 50.8 | 53.4 | DGFusion Tab.III |
| GeminiFusion | MiT-B2 | 50.5 | 54.5 | DGFusion Tab.III |
| CAFuser-CAA | Swin-T | 51.2 | 55.2 | DGFusion Tab.III |
| CAFuser | Swin-T | 51.3 | 55.6 | DGFusion Tab.III |
| DGFusion | Swin-T | **51.6** | **56.7** | DGFusion Tab.III |
| **MM-SAM-adapter** (2509.10408) | SAM ViT-L + ConvNeXt-S side-adapter, 1024² | — | **RGB-D 57.35 / RGB-L 57.14 / RGB-E 55.70** (RGB-L Hard 45.46) | [VERIFIED-PDF] [test] |

⚠️ **SCOOP ALERT [skeptic-confirmed]**: MM-SAM-adapter는 **2모달만으로** 4모달 DGFusion을 DELIVER test에서 이기고(57.35), MUSES test에서도 **81.07 (RGB-L) > 79.5** — "VFM 기반 멀티모달 SOTA" 헤드라인 주장은 MUSES test 81.07을 넘지 못하면 거짓. 주장 스코프를 arbitrary-modal / robustness / condition-adaptive로 한정할 것.

### 1.3 MUSES (09 §3, §U4)

| Method | Modality | Backbone | PQ [test] | mIoU [test] | 출처 |
|---|---|---|---:|---:|---|
| Mask2Former | C | Swin-T | 46.89 | 70.7 | DGFusion Tab.I/II |
| OneFormer | C | Swin-T | 55.21 | 72.8 | DGFusion Tab.I/II |
| MUSES baseline | CLRE | 4xSwin-T | 53.60 | — | DGFusion Tab.I |
| CMNeXt | CLRE | MiT-B2 | — | 72.1 (DGFusion) / 72.4 (CAFuser) — 출처별 유지 | DGFusion/CAFuser Tab.II |
| GeminiFusion | CLRE | MiT-B2 | — | 75.3 (CAFuser의 서버 run — GeminiFusion 본문엔 MUSES 실험 없음) | CAFuser Tab.II |
| CAFuser-CAA | CLRE | Swin-T | 59.38 | 78.5 | DGFusion Tab.I/II |
| CAFuser(-CA²) | CLRE | Swin-T | 59.70 | 78.2 | DGFusion Tab.I/II |
| **DGFusion** | CLRE | Swin-T | **61.03** | **79.5** | DGFusion Tab.I/II |
| **MM-SAM-adapter** | RGB-L / RGB-E | SAM ViT-L | — | **81.07 / 79.92** | Tab.6/8 [VERIFIED-PDF] [test] |

- Dataset card [2401.12761, ECCV'24]: 1,500 train / 250 val / 750 test; **test 라벨 비공개 — 공식 eval 서버** (semantic / panoptic / **uncertainty-aware panoptic AUPQ** × RGB-only·multimodal). 19 Cityscapes 클래스.
- Per-condition PQ [test]: **DGFusion 최약 = Night 58.97, Fog 58.86** (vs Rain 61.26, Snow 59.77, Clear 62.16) — 우리 condition-adaptive gain 스토리의 타깃 컬럼.
- B0/anymodal (F-E-L): MAGIC 33.34 → MAGIC++ 35.53 → AnySeg 40.23 [모두 val]; MLE-SAM F-E-L 74.8 (⚠ split 미표기, 서버 필요하므로 추정 val/local).

### 1.4 MCubeS (09 §U5) — 단일 프로토콜 (전원 **test**, 102장; 302/96/102, 20 material 클래스)

StitchFusion Swin-L-22k **55.9** > StitchFusion B4 53.92 > MMSFormer B4 53.11 > **MemorySAM 52.88** (R-A-D-N, Hiera-B+ 1 LoRA, Tab.2) > U3M 51.69 > CMNeXt-B2 51.54 > MLE-SAM 51.02; B0 라인: EGFormer 43.40 vs CMNeXt-B0 36.16. Mul-VMamba 54.65 [ABSTRACT-ONLY]. ⚠ 해상도 표기 불일치(1224×1024 vs MemorySAM HTML "1920×1080") 재확인 필요.

### 1.5 MULTIAQUA (arXiv 2512.17450, 공개) [VERIFIED-PDF]

3,293 frames; RGB+thermal LWIR+NIR+polarization+LiDAR+radar; 4클래스; **day train/val · night-only test** (MaCVi @ CVPR 2026 챌린지 사용). Table III (val-day/test-night mIoU): CMNeXt-DH **93.58/74.25**, StitchFusion-D 89.81/74.23, CMNeXt-D 92.95/72.24. 그들의 robustness는 training-time(RGB-zeroed double forward + modality-specific heads) — 우리는 inference-time → day→night reliability-shift showcase로 최적.

### 1.6 Per-condition & robustness 프로토콜 (09 §U7–U8)

- DELIVER per-condition 분해는 확인된 모든 논문에서 **val split에만** 존재 (CMNeXt Tab.2, HyperDUM Tab.4, EQUISeg). **DGFusion은 DELIVER per-condition 표 없음** (aggregate test만).
- 헤드라인 val per-condition: CMNeXt-B2 Night 62.46 → HyperDUM 64.21 (+1.75, 최대 이득); Cloudy 68.70→69.76; mean 66.30→67.59. ⚠ 센서 failure-case 5셀(MB/OE/UE/LJ/EL)은 추출 정렬 불일치 — **LaTeX 전 시각 재확인 필수**.
- Robustness 프로토콜 채택안: "Benchmarking MMSS under Sensor Failures" (2503.18445, MemorySAM 그룹): EMM(15조합)/RMM(r∈{0.25,0.5,0.75})/NM(noise ×3). 주요: MAGIC++ 44.85 EMM-avg, MAGIC 44.97, StitchFusion 41.98, CMNeXt 37.90 (고노이즈에서 2.31%로 붕괴).
- **우리 보고 프로토콜 (09 §U8)**: DELIVER dual-column val+test (CAFuser Tab.III 포맷) + backbone 병기; MUSES 공식 서버 제출 (semantic+panoptic+**AUPQ**); MCubeS test; MULTIAQUA day→night. 목표선: val ceiling 70.34 / test CLDE 57.35 / MUSES test 81.07 mIoU · 61.03 PQ.

---

## 2. 경쟁자 메커니즘 taxonomy (신호 × 주입 위치)

출처: `relatedworks/40`(§C taxonomy)·`41`·`42`(near-miss 랭킹+fenced claim)·`43`·`02`·`07`·`50`·`51` + 2026 스텁/gap-fill. 메커니즘 클래스 어휘: feature-multiply | learned-gate | output-scale | loss-level | condition-token | attn-multiplicative | **logit-additive-bias**.

| Method | arXiv/venue | 신뢰도/조건 신호 | Training-free? | 주입 위치 | RBMA와의 차이 (한 줄) |
|---|---|---|---|---|---|
| **RBMA (ours, P28)** | — | per-modality decoder softmax **predictive entropy** B_i | **YES** | **additive PRE-softmax bias, SAM2 memory cross-attn** | — (기준점) |
| MemorySAM | 2503.06700 | 없음 (등가 memory 융합) | — | memory attention 융합 | 우리의 토대 — reliability 부재, RBMA는 직접 확장 |
| CAFuser-CA² | 2410.10791, RA-L'25 | learned CT (**text-supervised**, verbo-visual contrastive) | no | **query set에 token concat** (attention 후 제거; Tab.VI query 59.7 > K/V 59.1 PQ) | 토큰 추가 ≠ logit 가산; text/조건 라벨 필요 [confirmed ×2] |
| CAFuser-CAA | 동일 | learned CT → FC+softmax | no | per-modality **scalar × features** | learned-gate/feature-multiply [confirmed] |
| DGFusion | 2509.09828, RA-L'26 | learned depth tokens (**LiDAR=입력+depth-GT**, robust log-depth loss) + global CT | no | `F_q=[F_rgb,t_c,t_d]` **query concat**, 표준 softmax | 최근접 경쟁자 — depth 필수·learned vs entropy training-free·logit 가산 아님 [confirmed] |
| HyperDUM | 2503.20011, CVPR'25 | HDC prototype 거리 (**labeled prototypes + FT**) | no | learned **feature reweighting** | feature-multiply, 학습형 [confirmed] |
| UTFNet | GRSL'23 | learned evidential (Dirichlet) head + DST | no | evidence-guided weighting | 학습 evidential vs training-free [ABSTRACT-ONLY, paywall] |
| TMC/ETMC | ICLR'21/TPAMI'22 | learned Dirichlet evidence | no | output-scale (DS opinion) | 분류, late fusion |
| UNO | 1911.05611, ICRA'20 | softmax entropy (변형 중 training-free 존재) + learned TempNet | partially | **output-logit multiplicative** temp scaling + noisy-or | ⚠ 인용 의무 — output-level·sim-only·2모달; "pre-softmax" 단독 표현 금지 사유 |
| ReliFusion | 2502.01856 (3D det) | learned confidence (CMCL) | no | confidence × attention **OUTPUT** | output-scale (⚠ 수식 재검증 요) |
| ModalPatch | 2603.02481 (3D det) | learned variance (NLL) | no | `W̃=W·[1−softmax(U)]` **POST-softmax multiplicative** | post-softmax·multiplicative·learned [VERIFIED-PDF] |
| SAM2Long | 2410.16268 | SAM2 occlusion score | **yes** | **multiplicative key scaling, SAM2 memory attn** | SAM2-memory 최근접 이웃(리뷰어 1순위 반례 후보) — mult vs additive, unimodal video |
| **SAE** | 2603.16558 | attention-distribution entropy (predictive 아님) | **YES** | **additive PRE-softmax** (`S̃=S+λ·SAE·C`, Eq.7) | 유일한 training-free entropy→additive-logit 선행 — 단 LVLM hallucination, 단일 이미지 모달 |
| **PRIMED** | 2605.07154 (RAVS) | modality-reliance prior P (**Qwen3-omni distill, learned**) | no | `softmax(QK^T/√d + b_M)V`, `b_M=γ_p·log(P/(1−P))` — **additive pre-softmax** | near-miss #0 — 셀("multimodal dense pred + additive pre-softmax modality bias") 점유; learned·reliance≠reliability·SAM2 memory 아님. **인용 없이는 scoop-call 위험** |
| Not-All-Pixels-Are-Equal | 2505.02161 | learned confidence | no | `A=QK^T+B` additive pre-softmax | learned B, 단일 RGB feature matching — "first additive confidence bias" 금지 사유 |
| READ | ICLR'24 | confidence-aware objective | no | Q,K,V projection W/B의 **gradient TTA** (loss-level) | parameter adaptation vs closed-form logit bias; AV 분류 |
| RSGMamba | 2604.12319 | learned uncertainty gate + consistency gate | no | SSM(C-matrix) readout **learned-gate** | 학습형·Mamba·SAM2 memory 아님 (P30 router 최근접 2026 learned-gate) |
| EQUISeg | 2509.24505 | supervised class prototypes (SGM, teacher/student KL) | no | training-time balancing (loss-level) | 학습시 균형 vs 추론시 per-frame bias |
| AW-MoE | 2603.16261 (3D det) | **supervised weather classifier** (CE, ~99% acc) | no | MoE full-branch top-1 select | 라벨 필요 vs P29 무감독; 검출 |
| MLE-SAM (MoE-LoRA SAM) | 2412.04220 | 무감독 pooled per-modality feature stats | 입력은 yes, gate learned | LoRA(Q,V) gate INPUT, softmax top-k | modality-routing (condition 아님); 같은 SAM2·DELIVER 공간 — 실험 비교 필수 |
| M⁴-SAM | 2605.11760 | modality identity (dispatcher) | — | conv-LoRA modality dispatch (SAM2 encoder) | "MoE-LoRA in SAM2" 선점 — memory는 init-only라 RBMA 비위협 |
| Missing-modality masks (UMSE 등) | 2305.02504 | availability (binary) | trivially | −∞ hard logit mask | binary vs continuous |
| ALiBi | — | positional distance | yes | additive logit penalty | 형식적 템플릿으로 인용 (unimodal LM) |
| FunEntropy-Reg | 2505.06635, ICCV'25 | functional entropy / Fisher | — | **loss-level** regularization | 같은 entropy, loss 레벨 |
| MAGIC++ / AnySeg / RMMSS | 2412.16876 / 2411.17141 / 2505.12861 | 선택/distillation | — | modality selection / distill | anymodal 축, attention scoring 불변 |
| OmniSegmentor | 2509.15096, NeurIPS'25 | 없음 | — | feature-add + enhancement (**pretraining 축**) | 완전 직교 — composable |

**빈 칸 (RBMA claim)**: *(i) training-free × (ii) per-modality **predictive** entropy × (iii) **additive pre-softmax** × (iv) cross-modal **memory** attention × dense multimodal seg* 의 4축 조합 셀은 **미점유 (unfalsified, not proven — 2-skeptic 판정 "uncertain")**. "first additive attention bias" 류 메커니즘-단독 first 주장 **금지** (PRIMED·SAE·2505.02161 반례).

**논문용 fenced claim 최종본 (42 Track-8 addendum, verbatim)**:
> To our knowledge, no published method (as of mid-2026) injects a continuous, **training-free**, per-modality **reliability** signal as an additive pre-softmax bias into cross-modal **memory** attention for dense multimodal segmentation. The closest mechanisms are: a *learned* modality-reliance prior added to pre-softmax cross-attention logits for referring audio-visual segmentation (PRIMED); entropy-driven additive logit modulation for LVLM hallucination within a single image modality (SAE); a learned additive confidence bias on attention logits for single-modality feature matching; training-free multiplicative key scaling in SAM2 memory attention for unimodal video (SAM2Long); and learned multiplicative reweighting of post-softmax attention weights or outputs in 3D detection (ModalPatch, ReliFusion). Condition-token approaches (CAFuser, DGFusion) enlarge the query set rather than biasing scores.

**필수 ablation (42/43 합본)**: post-softmax scaling vs pre-softmax bias / feature-level scaling(HyperDUM-style) / output-level weighting(UNO-style) / multiplicative key scaling(SAM2Long-style) / post-softmax mult(ModalPatch-style) / attention-entropy(SAE) vs predictive-entropy 신호 / log-odds 변환 `b=λ·log(B_i/(1−B_i))`(PRIMED 함수형) vs centered-linear vs log(r+ε) / global vs per-patch B_i / ECE·uncertainty-error correlation.

---

## 3. 구현 참고 자산 (P29/P30/P31 직결)

### 3.1 P29 — MoE-LoRA 조건 라우팅 (SDC) — 출처: `relatedworks/50`, `20`, `22`, `23`, `48`

- **PEFT 기본 선택 (20/23)**: LoRA(attention Q/V projection)가 default; MemorySAM=Hiera-B+ 1 LoRA, SAMed가 LoRA-on-SAM 원조 레시피. 순위: ① MemorySAM+RBMA ② MoE-LoRA SAM ③ StitchFusion MultiAdapter ④ ViT-Adapter ⑤ SAMed/SAM-Adapter ⑥ VPT(약함).
- **MLE-SAM / MoE-LoRA SAM (2412.04220) 세부**: per-modality LoRA experts on SAM2 **Q,V** projections (Eq.5 `Q′ᵐ=Qᵐ+ΔQᵐ`); router Eq.8–9 `wᵢᵐ = σ(Wᵢ·fᵢᵐ + bᵢ)` — **spatially-averaged per-modality embedding에 linear+sigmoid**, softmax+top-k over modalities; load-balancing 미보고. DELIVER 64.08 [val 추정] / MCubeS 51.02. **같은 SAM2+DELIVER 공간 → 실험 비교 대상 + 선제 인용 필수 (차별화: modality stat vs condition prototype / gate-input vs FiLM-on-gate / per-modality vs per-condition).**
- **최근접 3대 (Track 6 판정: P29 셀 UNOCCUPIED, 단 scoped wording만 허용)**:
  1. MoCLE (2312.12379, ICLR'24): **텍스트** instruction k-means 클러스터 → cluster embedding이 gate INPUT. 시각 입력은 라우팅에 무관.
  2. AW-MoE (2603.16261): **supervised weather classifier** router (CE, ~99% acc), full-branch experts, hard top-1. K-Radar Total AP_3D 78.0→**83.9**, Light Snow 78.9→**90.2** [test].
  3. MoFME (2312.16610, AAAI'24): **FiLM이 expert를 인스턴스화** (`FFN{Σᵢ rᵢ(x)·[γ⁽ⁱ⁾∘x+β⁽ⁱ⁾]}`) — P29는 FiLM 화살표를 **gate로 역전**. 리뷰어 혼동 1순위 → inversion 명시.
- **Near-occupant 2건 (선제 인용)**: DAMP (2512.20251) — 6개 training-free hand-crafted degradation stats를 learned gate에 **concat** (restoration); LFB Loss-Free Balancing (2408.15664) — expert-**load** 통계의 gradient-free **additive gate-logit bias** (pre-top-K). → P30 router는 "same injection family, input-derived reliability anchor (correctness vs balance)"로 1문장 차별화.
- **Gate-collapse 교훈**: MoE router는 dominant modality/expert로 붕괴 (우리 P10–P27 직접 증거, ISSUE-002/015: 조건이 라우터에 부재 + zero-init 가산 bias + 무감독 soft-softmax → E1 dead/상수수렴). 대책 후보: Mod-Squad式 condition↔expert MI regularizer / load-balancing loss / LFB-style bias — **P29 ablation 3종으로 비교**.
- **허용 클레임 (50 §Application, universal negative 금지)**: (a) "no prior work FiLM-modulates an MoE gate with a condition latent" [confirmed ×2], (b) "unsupervised image-only condition prototype from global feature stats → LoRA-expert routing **in multimodal dense prediction**" [scoped], (c) "training-free input-derived reliability as **additive gate-logit bias**" [scoped]. "no prior work uses training-free input statistics for expert routing"은 **FALSE** (DAMP).
- **P29 ablation 설계**: (i) FiLM-on-gate vs cluster-embedding-gate-input(MoCLE-style) vs stat-concat-gate-input(DAMP/MLE-SAM-style) — injection-point ablation이 novelty의 실증; (ii) M⁴-SAM의 conv-LoRA expert 변형 포함 고려.

### 3.2 P30 — class-token decoder + reliability-anchored router — 출처: `relatedworks/51`, `31`, `32`, `34`

- **Track 7 판정**: broad 셀("query decoder on fused multimodal features")은 **점유** (CAFuser/DGFusion=OneFormer head post-fusion, BiXFormer 2506.03675=per-modality queries+UMM, DF2RQ, RoadFormer+); 정확 셀("SAM2 **memory** features 위 mask-classification + reliability-anchored routing")은 외부 미점유. ⚠ **Gray zone**: MemorySAM 자체가 SAM2 stock mask decoder(내부 learnable output token)로 memory feature를 디코드 → "first tokens attending to memory features" 표현 금지; **mask-classification framework + reliability routing 수준에서 차별화** + stock decoder vs class-token decoder ablation(d)로 실증.
- **Head 선택지 (31/32)**: Mask2Former = masked cross-attn + mask classification (semantic/instance/panoptic 통합); OneFormer = +task token 조건화 (CAFuser/DGFusion가 실제 사용하는 head). Attachment: 융합 feature → pixel decoder.
- **Reliability 주입 후보 지점 (31 표)**: ① pixel decoder 앞 feature bias ② masked cross-attention 내부(logit) ③ mask logits 직접 — "reliability를 pixel decoder 전에 넣을지, masked cross-attn 안에 넣을지, mask logit에 넣을지"가 열린 설계 질문 → P30 ablation 축.
- **Rare-class 근거 (P28 실패모드 대응)**: Frequency-based Matcher (2406.03917, TMM) — one-to-one Hungarian matching이 tail class 기아 유발 (ADE20K-Full: Mask2Former 18.8/4.8 rare → 20.3/8.3); GOOSE-M2F (2606.15937) — <50px/crop rare class "zero gradient", training-only aux per-pixel CE head가 "+5–8% rare (qualitative)" / 격리 ablation **+3.4% composite**. → **설계: fixed per-class token (Hungarian 회피) + auxiliary per-pixel head** (우리 추론임을 명시).
- **Ablation 세트 (51)**: (a) BiXFormer-style per-modality queries+UMM vs queries-on-fused-memory; (b) CAFuser-style CT-append vs RBMA logit bias(동일 attention); (c) post- vs pre-softmax; (d) stock SAM2 decoder(MemorySAM) vs class-token decoder(동일 RBMA feature).
- **Must-cite**: BiXFormer(TMM'26/2506.03675 — "first mask-classification × multi-sensor" 문구 소유, 단 그 주장도 CAFuser에 선행됨), SAM-DAQ(2511.09870 — queries가 SAM2 memory bank를 **대체**, binary saliency), SHIFNet(2503.02581 — frozen text embedding dot-product, memory 미사용), EoMT(2503.19108, CVPR'25 Highlight), MemorySAM.
- ⚠ **BiXFormer 프로토콜 경고 (51)**: BiXFormer DELIVER Tab.II RDEL **58.29** (Mean 43.24; E 1.03/L 1.49 — modality-dropout 학습 프로토콜) — MemorySAM 65.38/CMNeXt 66.30과 한 SOTA 표에 절대 혼입 금지 (프로토콜 태그 필수).
- **참고 수치 (51 verbatim)**: SHIFNet PST900/FMB/MFNet **89.8 / 67.8 / 59.2** mIoU (SAM2-L, 32.27M trainable); OpenWorldSAM ADE20K-857/VOC-20/ScanNet-40 **60.4 / 73.7 / 55.6** mIoU (4.5M trainable); Freq-based Matcher ADE20K-Full 18.8/4.8 rare → 20.3/8.3.

### 3.3 P31 / 효율 축

- **ClustViT** (2510.01948) [ABSTRACT-ONLY]: 학습형 Cluster module이 seg mask pseudo-cluster로 토큰 병합 + Regenerator로 dense 복원 — **2.18× GFLOPs 감소, 1.64× 추론 가속, 정확도 유지**. 토큰 효율 축 후보.
- **EoMT** (2503.19108): plain ViT 최종 블록에 query 주입, decoder-free — "up to 4× faster with ViT-L". SAM3(단일 스케일) fallback 경로: EoMT-style query 주입 or ViTDet simple-FPN (SAM3-RBMA ~24 plateau = multi-scale 부재와 정합).
- **FS-SAM2** (2509.12105) [ABSTRACT-ONLY]: SAM2 video 능력을 few-shot에 repurpose + 원 모듈에 LoRA meta-train — "SAM2 video 메커니즘 재활용 + LoRA" 선례로 인용 가치.

---

## 4. 2026 신규 위협 워치 (스텁 출발 — 원문 정독 필요)

> 섹션 명명은 스텁(abstract-only) 기준이나, 일부는 이후 gap-fill 노트로 [VERIFIED-PDF] 승격됨 — 검증 열 참조. 정량 인용은 반드시 해당 노트에서.

| Paper | arXiv | 한 줄 요약 | 위협도 (대상) | 근거 | 검증 |
|---|---|---|---|---|---|
| RSGMamba | 2604.12319 | learned uncertainty+consistency 이중 게이트를 SSM readout에 넣은 RGB-D/T seg (NYUv2 58.8, MFNet 61.1) | **MED** (P30) | learned-gate 최근접 2026 이웃; 단 학습형·Mamba·DELIVER 아님 — "training-free anchor" 구분 유지 | [VERIFIED-PDF] → `61` |
| EQUISeg | 2509.24505 | supervised class-prototype teacher/student로 modality 균형; DELIVER/MUSES EMM/RMM/NM | **MED** (벤치마크·P29) | DELIVER 67.90 [unknown, val-cluster] 경쟁 행; prototype이 supervised → P29는 "unsupervised **condition** prototype"으로 스코프 | [VERIFIED-PDF] → `62` |
| GeomPrompt | 2604.11585 | frozen RGB-D 모델용 4번째 채널 geometric prompt 학습 (missing/degraded depth; 7.8ms) | **LOW** (보완재) | RGB-D 특화 워크숍(CVPR'26 URVIS); RBMA와 상호보완 — §6 결합 실험 후보 | [VERIFIED-PDF] → `63` |
| ModalPatch | 2603.02481 | modality-drop 보상 + uncertainty-guided cross-modal fusion, plug-and-play 3D det | **MED** (RBMA 수사) | "uncertainty-guided fusion" 문구 충돌; 단 **post-softmax multiplicative + learned variance** → 메커니즘 상이 [VERIFIED-PDF via `46_attention_*`] | 노트 검증 |
| AW-MoE | 2603.16261 | supervised weather classifier가 full-branch expert를 top-1 라우팅 (K-Radar 83.9 AP_3D) | **MED** (P29) | "weather-conditioned routing" 셀 점유 — 단 라벨 필요·검출·full branch → P29 3축(무감독/LoRA/soft-FiLM) 차별화 명확 | [VERIFIED-PDF] → `50` |
| M⁴-SAM | 2605.11760 | conv-LoRA MoE + modality dispatcher를 SAM2 encoder에; memory는 pseudo-mask init 전용 (RGB-D VSOD) | **MED-HIGH** (P29 아키텍처 주장) | "MoE-LoRA in SAM2" 선점 (계보: MLE-SAM 2412.04220 먼저 인용) → P29 셀은 **routing signal**로 축소; RBMA는 무관(memory init-only) | [ABSTRACT-ONLY] → `48` |
| OmniSegmentor | 2509.15096 | ImageNeXt 5-모달 pretraining + 단순 additive fusion, NeurIPS'25 | **HIGH** (리더보드) / LOW (메커니즘) | DELIVER **68.0** [val] = MemorySAM 65.38 +2.6 — clean-val 상한 경쟁; condition-적응 전무 → "pretraining 축 vs fusion-mechanism 축, composable" 프레이밍 | [VERIFIED-PDF] → `54` |
| FS-SAM2 | 2509.12105 | SAM2 video 능력 few-shot repurpose + LoRA meta-training | **LOW** | 벤치마크/메커니즘 비충돌; SAM2-repurpose 선례 인용 가치 | [ABSTRACT-ONLY] 스텁 |

**스텁 목록 외 필독 위협 (Track 8 결론)**: **PRIMED** (2605.07154 — logit-bias 셀 점유자, near-miss #0, full read = blocking), **SAE** (2603.16558 — training-free entropy→additive logit, LVLM), **MM-SAM-adapter** (2509.10408 — DELIVER test 57.35·MUSES 81.07 SCOOP), SAM4D (2506.21547 — "Motion-aware Cross-modal Memory Attention" 모듈명 distinguish 인용), ICRCV'25 underwater condition-aware MMSS (P29 blocking read), BiXFormer (2506.03675), SAMCM-SR ("first multimodal SAM3" 주장 선점).

---

## 5. 논문 작성 자산

### 5.1 Six-cluster related-work map (`relatedworks/90` §1 압축)

| 클러스터 | 대표 노트 | 논문 내 역할 | RBMA gap |
|---|---|---|---|
| 직접 MMSS | 01/02/04/05/07/08 | 본 related-work + baseline | 융합은 강하나 reliability가 암묵/proxy — SAM2 memory-attn logit에 없음 |
| 멀티모달 검출 | 10–14 | BEV/query fusion 설계 교훈 (baseline 아님) | dense seg는 pixel-level + 지역적 reliability 필요 |
| Adapter/LoRA/VFM 적응 | 20–23 | PEFT 문단 + ablation 설계 | PEFT는 표현 적응만 — "얼마나 신뢰할지"는 미결 |
| Seg/Det heads | 30–34 | head 선택·task-scope 정당화 | head는 reliability 메커니즘이 아님 (직교 유지) |
| Uncertainty/novelty | 40/41/42(+43) | novelty 방어·rebuttal·ablation 계획 | 기존은 feature/output 가중 — RBMA는 pre-softmax 경쟁 |
| 벤치마크/데이터셋 | 06/09 | 실험 문단 + 비교표 | source-table-backed 숫자만 보고 |

핵심 문장 (90 §2.5): "**RBMA changes the mathematical location of reliability control**" — feature 곱/출력 평균이 아니라 softmax 전 logit에 prior를 더해 modality-memory 토큰의 경쟁 자체를 바꾼다.

### 5.2 English paragraph candidates A–D (`relatedworks/90` §5 — VERBATIM, paper-ready)

**Paragraph A — direct multimodal segmentation**

Recent multimodal semantic segmentation methods improve robustness by fusing complementary sensors through feature rectification, token fusion, adapter exchange, condition-aware modulation, modality selection, or distillation. CMX and TokenFusion represent feature/token-level RGB-X fusion, while MAGIC++ and AnySeg address arbitrary or missing modality settings through hierarchical selection and unimodal/cross-modal distillation. CAFuser and DGFusion further show that environmental condition tokens and depth-guided local tokens can improve driving-scene perception. Most closely related, MemorySAM maps modalities into a SAM2 memory-style formulation. These works establish strong baselines for multimodal segmentation, but they typically handle reliability implicitly or through proxy conditioning rather than explicitly biasing memory attention according to calibrated modality trust.

**Paragraph B — foundation-model adaptation**

Foundation segmentation models require task and sensor adaptation before they can serve as reliable dense-prediction backbones. LoRA, AdaptFormer, visual prompt tuning, and ViT-Adapter demonstrate parameter-efficient strategies for adapting transformer representations, while SAM-Adapter, MedSAM, SAMed, MemorySAM, MoE-LoRA SAM, and SAM-FuseNet show that SAM-family models can be specialized to medical, SAR, RGB-thermal, and multimodal semantic segmentation domains. These methods motivate a modular design in which LoRA/adapters learn modality-specific representations, but they do not by themselves determine which modality should dominate fusion under corruption. Reliability-aware attention is therefore complementary to PEFT rather than a replacement for it.

**Paragraph C — detection analogy**

Multimodal object detection provides useful architectural lessons for segmentation. BEVFusion demonstrates that a shared BEV representation can preserve camera semantics and LiDAR geometry better than sparse point-level association. TransFusion and FUTR3D use query-based transformer fusion to gather evidence from heterogeneous sensors, and DeepInteraction argues for preserving modality-specific streams instead of collapsing them prematurely. These detection methods support the broader principle that robust multimodal perception should use learned, flexible cross-modal association. However, dense semantic segmentation still requires pixel-level class prediction and spatially local reliability decisions, motivating a segmentation-specific attention-bias mechanism.

**Paragraph D — novelty statement**

The proposed RBMA mechanism targets a gap left by current multimodal segmentation, detection-fusion, adapter, and uncertainty-fusion literature. Existing methods may scale features, aggregate evidential outputs, select modalities, add condition tokens, regularize unimodal bias, or adapt foundation encoders with LoRA/adapters. RBMA instead changes the location of reliability control: it adds a predicted reliability prior to the attention logits before softmax in SAM2-style multimodal memory attention. This pre-softmax intervention directly changes the competition among modality-memory tokens, allowing corrupted or uninformative sensors to be down-weighted during fusion while preserving the representation benefits of foundation-model adaptation.

> ⚠️ 사용 주의: A–D는 2026-06-25 작성분 — 이후 발견된 PRIMED/SAE/UNO/MM-SAM-adapter를 반영하려면 D 뒤에 §2의 fenced claim(42 최종본)과 43의 "Uncertainty-driven multimodal fusion" 문단, 40 §E의 갱신 문단을 병용할 것. 최신 대체 후보 문단: `40` §E, `43` 말미, `50`·`51`·`48`·`54`의 "Related-work paragraph candidate".

### 5.3 권장 related-work 섹션 outline (90 §6 기반 + 갱신)

1. **§1 Multimodal semantic segmentation**: CMX, TokenFusion, MAGIC++, CAFuser, DGFusion, StitchFusion, AnySeg, Reducing Unimodal Bias, MemorySAM (+ OmniSegmentor pretraining 축, BiXFormer mask-classification 축).
2. **§2 Foundation-model adaptation**: LoRA, AdaptFormer, VPT, ViT-Adapter, SAM-Adapter, SAMed/MedSAM, MoE-LoRA SAM(2412.04220), M⁴-SAM, SAM-FuseNet, MM-SAM-adapter.
3. **§3 Reliability & uncertainty**: UNO, Blum'18, TMC/ETMC, UTFNet, HyperDUM, Seeing-Through-Fog, READ, RSGMamba + near-miss fence(PRIMED, SAE, SAM2Long, ModalPatch, ReliFusion) — fenced claim으로 마무리.
4. **§4 Detection heads & transfer**: BEVFusion, TransFusion, DeepInteraction, FUTR3D — 지지 문맥으로만.
5. **Experiments**: DELIVER/MUSES/MCubeS(/MULTIAQUA), 볼트 09의 source-table-backed 숫자만, [val]/[test] 태그 유지.

---

## 6. 아이디어 회의 어젠다 후보 (연구정보 → 실험/구현 제안)

1. **Per-condition 평가표 구축** — DGFusion/CAFuser는 MUSES per-condition을 보고하는데 우리는 안 함. DGFusion 최약 = Night 58.97 / Fog 58.86 PQ [test]; DELIVER per-condition은 val에만 존재(HyperDUM Tab.4: Night 62.46→64.21). *근거*: 09 §U4/U7. *예상 실험*: P28의 DELIVER val 10-case 분해표 + MUSES 서버 제출로 per-condition PQ — "adverse에서 이긴다" 스토리의 필수 데이터.
2. **CMNeXt(+MemorySAM) 단일 프로토콜 재평가** — 남의 표 숫자 혼용 시 cluster 불일치 리뷰 지적 확실. *근거*: doc 12 §2.5 TODO, 09 §U1/U2. *예상 실험*: 우리 split·해상도로 CMNeXt-B0/B2, 가능하면 MemorySAM 공식 코드 재실행 → 한 표에 [val]/[test] 나란히.
3. **Reliability-anchored router vs RSGMamba식 self-gating 차별화 실험** — learned gate(무anchor)는 우리 P10–P27에서 붕괴 실증. *근거*: `61`, `50`(LFB), doc 12 §2.8. *예상 실험*: 동일 백본에서 (a) learned gate (b) RBMA-anchored gate (c) LFB-style load bias (d) RSGMamba식 consistency 게이트 추가(`logits + λ_e·B_entropy + λ_c·B_consistency`) 4-way 비교.
4. **GeomPrompt-style depth prompt × RBMA 결합** — missing-depth에서 "합성 geometry를 넣을지(RGB-D 보완)"와 "신뢰도를 낮출지(RBMA)"는 직교. *근거*: `63`. *예상 실험*: RGB+degraded D / RGB+GeomPrompt / RGB+degraded D+GeomPrompt+RBMA 3단 ablation.
5. **MULTIAQUA↔DELIVER 이중 데이터셋 일반화 스토리** — day-train/night-test 구조가 inference-time reliability shift의 이상적 쇼케이스; 그들의 방어는 training-time(RGB-zero). *근거*: `64`, 09 §U6. *예상 실험*: P28을 MULTIAQUA에 그대로(재학습 최소) 이식, CMNeXt-DH 74.25 [test-night] 대비 + B_i가 night에 RGB→thermal/LiDAR로 이동하는 시각화.
6. **MUSES AUPQ(uncertainty-aware panoptic) 트랙 선점** — RBMA의 B_i 맵이 자연 적합; fusion-cluster 논문 중 이 트랙을 전면에 쓴 곳 없음. *근거*: 09 §U4/U8. *예상 실험*: P30 panoptic head + B_i 기반 per-pixel uncertainty 제출 (semantic/panoptic/AUPQ 3트랙).
7. **동일 B_i 신호의 injection-point 3-way ablation** — "왜 attention logit인가"를 실증으로 전환. *근거*: `43` §Improvement 1. *예상 실험*: (i) UNO-style output-average (ii) HyperDUM-style feature-multiply (iii) RBMA logit-bias — 같은 신호, 주입만 변경; + graded corruption 하 B_i calibration curve(ECE)로 UDML 비판 선제 방어.
8. **MM-SAM-adapter scoop 대응 스코프 결정** — 2모달로 DELIVER test 57.35·MUSES 81.07. *근거*: 09 §U3(c). *논의*: 헤드라인을 "condition-adaptive/robustness/anymodal"로 한정할지, MUSES test 81.07 초과를 노릴지; P28 Hiera-L(67 본명 경로)의 목표선 재설정.

---

## 7. 남은 구멍 (병렬 리서치 8트랙 — `research_vault/sources/07` 참조)

8트랙은 2026-07-02 전부 1차 완료 (07 §완료 기록). 미해결 잔여:

- **MemorySAM 65.38의 split 원저자 확정** — 코드 추론(val)만 존재 (Track 3/8 잔여; 09 §U2).
- **PRIMED(2605.07154)·SAE(2603.16558) full read** — dense-prediction/sensor-fusion 실험 존재 시 우리 claim 스코프 재축소 (Track 8 blocking; `42`).
- **ICRCV'25 underwater condition-aware MMSS 정독** — 전까지 P29 셀 점유 주장 금지 (Track 8; `48`).
- **CMNeXt Tab.2 / HyperDUM Tab.4 failure-case 행 시각 재확인** — LaTeX 전 필수 (Track 3; 09 §U7).
- **StitchFusion 70.34의 split** 자체 caption 미확인(repo 404); EQUISeg 67.90 표 미렌더; Mul-VMamba paywall (Track 3; 09 §U9).
- **Night-TTA CAR 수식 / UTFNet 주입 locus(paywall) / "prior bias matrices" 원논문 / 2505.06635 dataset명** (Track 4; `43` §6).
- **WM-MoE PDF**(weather cluster의 라벨 사용 여부) / **DF2RQ PDF**(IEEE paywall) / LER-YOLO·SkillMoV venue watch (Track 6/7; `50`·`51`).
- **MLE-SAM DELIVER 64.08·MUSES 74.8 split 미확정** (Track 3 인계; `50`).
- **sources/08_threat_watch_2026H2.md 미동기화** — 여러 노트가 참조하나 이 사본에 없음 → NAS에서 추가 동기화 필요.
- 전칭 부정(novelty 셀 미점유)은 **매 투고 직전 arXiv 최신 6개월 재스윕** (Track 8 상시).
