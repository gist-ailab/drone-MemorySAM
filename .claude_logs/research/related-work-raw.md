---
legacy_id: 10
legacy_file: 10_related_work.md
moved: 2026-07-08
---

# 관련 연구 (Related Work) — 멀티모달 융합 & 어텐션 동역학

> **[2026-07-08 append] 차세대 아키텍처 deep-research 원시 로그**: 2트랙(VFM 후보 / adaptive fusion) 병렬 조사 수행 — 전체 결과·문헌 인덱스·미확인 표기는 `research_vault/material/brainstorm_next_arch_20260708.md` §5에 통합 기록(본 문서엔 중복 미전개). 핵심: DINOv3(2508.10104) frozen ADE20K 63.0/COCO 66.1, C-RADIOv4(2601.17237) SigLIP2+DINOv3+SAM3 3교사, SAM3 가중치 gated, entropy-신뢰도 비판 계열(MG-MTTA 2604.24602) 대두, det-head additive-bias 셀 빈 것 재확인.

> 최초 작성: 2026-06-12
> 출처: NotebookLM 큐레이션 (Research Bibliography: Diffusion Transformers and Multi-Modal Fusion)
> 목적: MemorySAM 추가 연구·실험을 위한 외부 소스 정리 및 프로젝트 매핑

---

## 0. 마스터 레퍼런스 테이블

| # | 제목 | 주저자 | 연도 | 출처 | 프로젝트 적합도 |
|---|------|--------|------|------|----------------|
| 1 | Reducing Unimodal Bias in Multi-Modal Semantic Segmentation (Multi-Scale Functional Entropy Regularization) | Xu Zheng et al. | 2024 (Suppl., ECCV) | Supplementary Material | 🟢 최상 |
| 2 | DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception | Tim Brödermann et al. | 2026 | IEEE RA-L / arXiv | 🟢 상 |
| 3 | StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation | Bingyu Li et al. | 2024 | arXiv | 🟡 중 |
| 4 | Lidar-Camera Fusion 3D Object Detection (GitHub) | Azitt (Azam Kowalczyk) | 2023 | GitHub Repo | 🟡 중 (교훈) |
| 5 | Attention Sinks in Diffusion Transformers: A Causal Analysis | Fangzheng Wu, Brian Summa | 2026 | ICML / arXiv | 🔴 하 (간접) |

---

## 1. Zheng et al. — Multi-Scale Functional Entropy Regularization 🟢

**문제의식**: 표준 융합 모델은 가장 "학습하기 쉬운"/정보량 많은 모달리티(보통 RGB)에 과의존(unimodal bias)하여,
해당 센서가 모션블러·조도실패로 손상되면 보조 센서가 멀쩡해도 성능 급락.

**핵심 기법**: 학습 중 **functional entropy를 최대화** → 모델이 모든 모달리티에 의존을 강제로 분산.
(정보이론적으로 high entropy = "의존/어텐션"의 균등 분포)

**검증**: 성능 표준편차 22.36% → 7.91% (vs MAGIC) 로 변동성 대폭 감소, DELIVER/MCubeS에서 mIoU 우위.

**MemorySAM 매핑**:
- 우리의 핵심 난제 **val(주간 93–94%) vs test(야간 58–70%) 갭**과 정확히 일치 — 본질은 주간 학습 중 RGB 과의존 고착.
- UAMM/AMF(공간·모달 가중)와 같은 목표를 *최적화 단(loss)* 에서 거는 보완재. P24의 CE teacher와 결이 맞음.
- **실험 후보**: P9/P25 위에 multi-scale entropy reg를 auxiliary loss로 추가 → NIGHT_AUG와 결합 시 야간 변동성 안정화 기대.
- 연결: [[P25 Unified Spatial Quality Fusion]], [[val-test 주야간 갭]]

---

## 2. DGFusion — Depth-Guided Sensor Fusion 🟢

**핵심 아이디어**: 멀티모달 세그멘테이션을 멀티태스크로 처리.
- **전역 Condition Tokens (CT)**: 장면 전역 맥락.
- **로컬 Depth Tokens (DT)**: 공간별 센서 신뢰도에 따라 센서 기여도를 적응적 가중.
- LiDAR를 입력뿐 아니라 **auxiliary depth head의 GT supervision** 으로 "공짜로" 활용.
- **outlier-robust L1 loss** 로 안개·비 환경의 노이즈 LiDAR 리턴 완화.

**MemorySAM 매핑**:
- P24/P25의 **SpatialQualityGating** 과 거의 동일 철학 (spatial UAMM/AMF = 로컬 신뢰도 가중).
- P10에서 ModalAuxHead 시도 후 취소했는데, DGFusion의 robust L1 loss가 당시 노이즈 처리 문제의 해법일 수 있음.
- **실험 후보**: P25에 LiDAR depth aux head + outlier-robust L1 추가, depth token으로 per-region 가중.
- 연결: [[P24 SpatialQualityGating]], [[P25 Unified Spatial Quality Fusion]], [[P10 ModalAuxHead 취소]]

---

## 3. StitchFusion — MultiAdapter 조기 융합 🟡

**핵심**: 전용 Feature Fusion Module(FFM) 대신 **MultiAdapter** 모듈로 사전학습 인코더들 사이의
멀티스케일 정보를 **인코딩 단계에서 동기화**("weaving"). 사전학습 인코더의 고유 모델링 능력 활용.
최소 추가 파라미터(+0.14M ~ +0.71M, DeLiVER)로 SOTA.

**MemorySAM 매핑**:
- 우리의 LoRA 어댑터 기반 SAM2 적응과 호환. MemorySAM은 이미 memory attention으로 모달리티를 "프레임"으로 엮지만,
  StitchFusion은 더 이른(인코더 단) 멀티스케일 융합이라 상보적. 우선순위 중간.

---

## 4. Lidar-Camera Fusion GitHub (Azitt 2023) 🟡

**교훈 위주**: Early fusion(LiDAR→이미지 평면 투영, YOLOv8) vs Late fusion(YOLOv8 + PV-RCNN++ confidence matching).
- **Early fusion의 catastrophic failure**: 카메라 검출이 실패하면 LiDAR 점이 명확히 존재해도 작은 자전거·원거리 차량을 통째로 놓침.
- → 우리 **야간 RGB 실패 시나리오**와 동일한 경고. 새 기법보다 "왜 적응적 가중/entropy reg가 필요한가"의 근거 자료.

---

## 5. Attention Sinks in Diffusion Transformers (Wu & Summa 2026) 🔴

**핵심 결과**: SD3/SDXL에서 attention sink는 AR 모델과 달리 **의미정렬에 필수적이지 않음**.
- 동적 sink 정의(per-head, per-timestep), training-free 개입(score-path 마스킹 / value-path 치환).
- Sink가 index-0에 없고(겹침 <0.2%) positional drift 큼 → diffusion에서 sink는 transient·phase-dependent.
- 주 sink 제거(k=1)해도 CLIP-T/ImageReward/HPS-v2 무손상. 단 perceptual drift(LPIPS/FID)는 랜덤 대비 ~6배.
- **인사이트**: "높은 incoming attention mass ≠ 기능적 필요성".

**MemorySAM 매핑**:
- 우리는 SAM2(비-diffusion) 기반이라 직접 연관 약함.
- 단 인사이트는 우리 **MoE gate "uniform" 진단**(측정 artifact였던 이슈)이나 memory attention 희소화 고민 시 간접 참고.

---

## 종합 — 일관된 그림

> **아키텍처(DGFusion/StitchFusion: 적응적 공간 가중) + 최적화(Zheng: entropy로 의존 분산)**
> = 야간 단일모달 실패에 강건한 "Anymodal" 시스템.

**val/test 갭에 대한 레버리지 우선순위**:
1. Zheng entropy regularization → P9/P25에 loss 추가 (구현 가벼움, 갭에 직격)
2. DGFusion depth-aux + robust L1 → P25 spatial gating 강화 (P10 실패 복구 가능성)
3. StitchFusion adapter는 그 다음.

---

## 원문 추출 상태
- [x] arXiv 원문 방법론·loss 수식 정밀 추출 완료 (deep-research, 2026-06-12, 102 agents / 20 sources / 24 claims 검증, 23건 3-0 만장일치)

---

# 원문 정밀 추출 (deep-research 검증 결과, 2026-06-12)

> 모든 수식은 arXiv abstract+HTML+PDF (+CVF camera-ready / 공식 GitHub) 교차 검증, 3-0 만장일치 통과분만 기록.
> ⚠️ 한계: mIoU 정량 테이블(DELIVER/MCubeS/MUSES, vs MAGIC/CAFuser)과 전체 하이퍼파라미터(optimizer/lr/epoch/batch, λ_p·λ_f 수치)는 이번 패스에서 미확보. 필요 시 추가 추출.

## A. Reducing Unimodal Bias (Zheng et al.) — **arXiv:2505.06635, ICCV 2025**
> ⚠️ NotebookLM은 "2024 ECCV Suppl."로 적었으나 실제는 **ICCV 2025**. 정정.

**문제**: 표준 융합은 RGB 과의존(unimodal dominance) → 보조 센서 보유에도 RGB 손상 시 급락.

**핵심**: **plug-and-play, 파라미터/모듈 0개** functional-entropy 정규화. log-Sobolev 부등식으로 functional entropy를 functional-Fisher-information으로 bound. 각 모달리티 기여 정보를 최대화 → unimodal dominance 완화.

**수식**:
- Eq.2 (log-Sobolev bound): `Ent_μ(f) ≤ (1/2) ∫ ||∇f(x')||² / f(x') dμ(x')`  — 적분항 `||∇f||²/f` = functional Fisher information
- Eq.3 (per-modality 분해): `Ent_μ(f^x) ≤ Σ_i Ent_{μ_i}(f(x_i))`
- Eq.4 (base regularizer): `R = λ Σ_{i=1}^n ( ∫ ||∇_{x_i} CE(p_v(·|x_i), p_v(·|x))||² / CE(p_v(·|x_i), p_v(·|x)) dμ_i(x_i) )^{-1}`
  - **핵심 트릭**: Fisher 비율의 **역수(^-1)**를 최소화 → 모달리티별 정보 기여를 최대화 → 균등 분산
- Eq.6 (prediction-level): `R_p = λ_p Σ_i ( ∫ ||∇_{x_i} CE(p, gt)||² / CE(p, gt) dμ_i )^{-1}`
- Eq.7 (feature-level): `R_f = λ_f Σ_{j=1}^4 Σ_i ( ∫ ||∇_{f_i} CE({f_r^j, f_d^j}, f_m^j)||² / CE(...) dμ_i )^{-1}`  — SegFormer 4개 transformer-block 스케일(j) 합산, 스케일 간 smoothness/balance 강제
- Eq.8: `L_sup = CE(p, gt)`
- Eq.9 (총 목적): `L = L_sup + R_p + R_f`

**융합 구조**: **별도 융합 모듈 없음**. 최종 예측 = 모달리티 예측 평균 `p = Mean(p_r, p_d)`. → **임의 멀티모달 백본에 drop-in auxiliary loss**.
**데이터셋**: DELIVER, MCubeS. baseline: MAGIC.

## B. DGFusion (Brödermann et al.) — **arXiv:2509.09828, IEEE RA-L 2026** / 코드 `github.com/timbroed/DGFusion`

**문제·관점**: 멀티모달 세그멘테이션을 **멀티태스크**로. LiDAR를 입력이자 depth GT로 동시 사용. 센서 신뢰도의 **공간적 변동**에 융합을 적응.

**아키텍처 (cross-attention query 구성)**:
- 전역 Condition Token: `t_c = Transformer(Flatten(F_rgb^4))` (최상위 RGB feature → 경량 2-enc/2-dec Transformer; verbo-visual contrastive condition loss로 감독), `t_cl = FC(t_c)` — 이미지당 1개, 날씨/조도 인코딩
- 로컬 Depth Token: `t_dl,i = Pool_mean(Conv(d_li))` — window별 공간 로컬
- 융합 쿼리: `F_ql,i = [F_rgbl,i, t_cl, t_dl,i]` (RGB 토큰 + CT + DT concat → cross-attention query)

**Outlier-robust depth loss (핵심 재사용 포인트)**:
- Eq.8 (per-pixel 잔차): `r_p = |log(D̂_p) − log(D_p)|`  (D̂=예측 depth, D=LiDAR 기준)
- Eq.9 (τ-quantile 필터): `P_τ = {p ∈ P_l : r_p ≤ Quantile_τ({r_q})}`  — 기본 **τ=0.8** (오차 상위 20% 노이즈 픽셀 마스킹)
- Eq.10: `L_logL1 = (1/|P_τ|) Σ_{p∈P_τ} r_p`
- Eq.16: `L_depth = λ_L1 L_logL1 + λ_es L_es + λ_pes L_pes`  (λ_L1=0.9, λ_es=0.05, λ_pes=0.05; es=edge-aware, pes=panoptic-edge-aware smoothness)
- Eq.17 (총): `L_total = λ_seg L_seg + λ_cond L_cond + λ_depth L_depth`  (λ_depth=1; λ_seg·λ_cond은 OneFormer/CAFuser 기본값 상속)

## C. StitchFusion (Li et al.) — **arXiv:2408.01343** / 코드 `github.com/LiBingyu01/StitchFusion`

**핵심**: 전용 post-encoder 융합 모듈 대신 **인코딩 단계**에서 사전학습(frozen SegFormer) 인코더들 사이로 멀티스케일 정보를 전파하는 **MultiAdapter**.
**구조**: 경량 선형 모듈 (down d→r, GELU+dropout, up r→d). transformer block당 2개 삽입 — self-attention 뒤 `F_Ada1`, MLP/FFN 뒤 `F_Ada2`. 양방향 weaving: `z_j = z_j + DropPath(F_Ada1(LN1(x_i)))` (i≠j).
**데이터셋**: DeLiVER.

## (보너스) AnySeg (Zheng et al.) — arXiv:2411.17141 / `github.com/zhengxuJosh/AnySeg`
> 요청 외 발견. anti-unimodal-bias의 **distillation 계열** 대안. 멀티모달 teacher → anymodal student.
> `L_total = L_sup + λ_mad·L_mad + α·L_umd + β·L_cmd` (modality-agnostic + unimodal + cross-modal distillation).
> ⚠️ cross-modal distillation의 cosine-similarity KL 세부 수식은 검증에서 **반증(0-3)** → 정확 공식 미확정, 인용 주의.

## D. Attention Sinks in DiT (Wu & Summa 2026)
- 이번 패스에서 **검증된 주장 0건**. SAM2 세그멘테이션과 가장 거리 멀어 우선순위 최하였고 원문 미확보. 필요 시 별도 조회.

---

## 미해결 질문 (후속 추출 후보)
1. Paper A의 DELIVER/MCubeS mIoU 정량치 및 vs MAGIC 마진, λ_p/λ_f 실제 값
2. Paper B의 MUSES/DeLiVER mIoU vs CAFuser, 전체 학습 하이퍼파라미터
3. Paper D 실체 (SAM2 memory attention으로 sink 인사이트 전이 여부)
4. AnySeg L_cmd 정확 공식

---

# 토큰 프루닝 / 머징 / 어텐션 최적화 (ViT & DiT) — deep-research, 2026-06-13

> 조사 질의: task에 유의미한 토큰을 프루닝/머징하거나 attention을 가해 효율·성능을 올리는 ViT/DiT 연구.
> 108 agents / 26 sources / 25 claims 검증, 23건 3-0 통과. MemorySAM = SAM2 memory-attention 기반 RGB+LiDAR+Thermal 세그멘테이션 → 모달리티별 "프레임" 토큰 수가 곧 비용.

## 핵심 구분: training-free vs trained
- **Training-free (off-the-shelf)**: ToMe, ToMeSD, PPT, PiToMe, ToMA, StructSAM, Fast SAM2(추론 시점) — 기존 학습된 모델에 그대로 삽입.
- **Trained (학습 필요)**: DynamicViT, DTEM.

## 두 전략
- **Token Pruning**: 저중요 토큰을 버림 (DynamicViT).
- **Token Merging**: 중복 토큰을 합침 (ToMe). 하이브리드(PPT), 학습형(DTEM)이 중간.

---

### 🟢 MemorySAM 직접 적용 후보 (SAM/SAM2 + dense prediction)

**Fast SAM2** — Text-Driven Token Pruning, arXiv:2512.21333 (2025-12, preprint)
- **위치**: 인코더 직후 / **memory engine 직전** 토큰 프루닝 → 우리 구조에 가장 정확히 대응.
- 3개 신호 융합: (1) training-free CLIP text→visual 최소제곱 투영(semantic align), (2) 인코더 layer 3-5의 Monte Carlo Dropout 불확실성, (3) 경량 2-layer MLP로 softmax 정규화 top-k 유지.
- **training-free at segmentation time** (투영은 closed-form, MLP만 오프라인 학습; SAM2 encoder/CLIP/decoder/memory는 frozen).
- 결과: 추론 **최대 42.5% 빠름, GPU 메모리 37.4% 감소**, J&F 경쟁력 유지. ⚠️ "up to" 수치, 미검증 preprint.

**StructSAM** — resolution-preserving merge-unmerge for SAM, arXiv:2603.07307 (2026-03, preprint)
- 1차 feature gradient 기반 **token-energy score**, grid flatness screening으로 **경계/프롬프트 영역 보호**, flat 영역만 low-energy 목적지로 머징 후 **명시적 unmerge로 원해상도 복원**. training-free, 추론 전용.
- ⚠️ **벤치마크 수치(25-30% FLOPs↓, vs ToMe/PiToMe 우위 등)는 적대적 검증 실패(1-2)**, 코드 공개 미확인. 메커니즘만 신뢰.
- → 경계 보호 + 해상도 복원은 dense segmentation 품질 유지에 정확히 필요한 속성.

### 🟢 에너지/세일런시 인지 머징 (정보 토큰 보호 = 세그멘테이션 적합)

**PiToMe** — Protect Informative Tokens before Merging, NeurIPS 2024, arXiv:2405.16148, 코드 `github.com/hchautran/PiToMe`
- spectral graph energy score: 큰 중복 클러스터=high energy(머징), 작고 distinct한 전경 영역=low energy(**보호**). 이후 Bipartite Soft Matching.
- training-free(파인튜닝 선택), **효율+정확도 동시**. **40-60% FLOPs 절감 @ 0.3-0.5% drop** (일부 task는 향상).
- → "전경/정보 토큰 보호" 설계가 우리 세그멘테이션 품질 유지에 직결.

### 🟢 범용 머징/프루닝 (training-free, 검증된 baseline)

**ToMe** — Token Merging, ICLR 2023, arXiv:2210.09461, 코드 `github.com/facebookresearch/ToMe`
- key-vector 코사인 유사도 기반 **bipartite soft matching**으로 유사 토큰 점진 병합. training-free(학습 중 적용도 가능). **효율+정확도**.
- 결과: ViT-L@512 / ViT-H@518 **~2x throughput @ 0.2-0.3% drop**, video 2.2x, audio 2x.

**PPT** — Token Pruning + Pooling 하이브리드, arXiv:2310.01812, 코드 `github.com/xjwu1024/PPT`
- 층마다 pruning(inattentive 토큰 제거) + pooling(중복 토큰 병합) 적응적 결합. **추가 파라미터 0, training-free**.
- DeiT-S/ImageNet: **>37% FLOPs↓, throughput >45%↑, 정확도 손실 0**.

**ToMA** — Token Merge with Attention, ICML 2025, arXiv:2509.10918, 코드 `github.com/wenboluu/ToMA`
- Facility Location 서브모듈러로 목적지 토큰 선택 → 목적지=query, 전체=key/value의 SDPA로 soft merge 할당. **GPU 친화 재설계**, training-free.

### 🟡 학습형

**DynamicViT** — NeurIPS 2021, arXiv:2106.02034, 코드 `github.com/raoyongming/DynamicViT`
- 여러 층에 경량 prediction module로 토큰 중요도 추정 → attention masking으로 미분가능 프루닝. **end-to-end 학습 필요(training-free 아님)**.
- 66% 토큰 프루닝: **31-37% FLOPs↓, throughput >40%↑ @ <0.5% drop**.

**DTEM** — Decoupled Token Embedding for Merging, NeurIPS 2024, arXiv:2412.10569, 코드 `github.com/movinghoon/dtem`
- ViT forward와 **분리된** 경량 embedding 모듈로 머징 전용 feature 추출, 미분가능 relaxation으로 학습. (중간 feature 의존 한계 극복)

### 🟡 Diffusion (참고, 우리는 비-diffusion)

**ToMeSD** — Token Merging for Fast Stable Diffusion, arXiv:2303.17604, 코드 `github.com/dbolya/tomesd`
- SD transformer 블록 내부 중복 토큰 머징. training-free, out-of-the-box. 50% 머징 @ SD1.5 512²: **1.87x 속도, 3.83x 메모리 절감, FID 거의 불변**.

---

## ⚠️ 이번 패스 미커버 (검증 주장 0건 → 후속 조사 필요)
요청했으나 검증된 주장이 안 나온 항목:
- ViT pruning: **A-ViT, EViT([CLS]-attn 재정렬), SP-ViT, Evo-ViT, AdaViT**
- merging: **ToFu, BAT, standalone token pooling, segmentation 전용 dynamic merging**
- **DiT 어텐션 캐싱 전체**: DeepCache, Delta-DiT, FORA, AT-EDM, sparse/linear attention for DiT
- **attention sink / register tokens / 어텐션 재가중** (ViT registers, attention sinks)
→ 별도 deep-research 패스로 보강 가능.

## MemorySAM 적용 시 핵심 미해결 질문
1. 멀티모달 토큰 축소를 **융합 전 모달리티별로** vs **모달리티 프레임 간 공동으로** 할지 — Fast SAM2(memory engine 직전 프루닝)/StructSAM(merge-unmerge)이 memory attention이 요구하는 cross-modal 대응을 보존하는지.
2. 에너지/세일런시 보호 기법(PiToMe/StructSAM/Fast SAM2)이 야간·저조도 dense data에서 mIoU를 **유지를 넘어 향상**시키는지.
3. StructSAM 코드 공개 및 (검증 실패한) 벤치마크 재현 여부.

---

# 보강 조사 1: attention/CLS 기반 ViT 프루닝 + 세그멘테이션 전용 (deep-research, 2026-06-13)

> 105 agents / 23 sources / 25 claims 검증, **25건 전부 3-0 통과 (0 killed)**.
> ⭐ 핵심 교훈: 요청한 Family 1(A-ViT/EViT 등)은 **전부 학습 필요 + ImageNet 분류 전용**. 정작 MemorySAM에 값진 건 패밀리 밖에서 나온 **세그멘테이션 전용/training-free 3종**.

## ⭐ MemorySAM 최우선 — 세그멘테이션 dense-prediction 전용 (Family 밖 발견)

**Expedit** — NeurIPS 2022, arXiv:2210.01035, 코드 `Expedit-LargeScale-Vision-Transformer` (+ **Expedit-SAM** 변종 존재!)
- **완전 training-free, 비파라미터**. 두 연산자: token **clustering** layer(공간 인접 토큰 군집화로 토큰 수↓) + token **reconstruction** layer(고해상도 복원). CLS top-k 프루닝 아님.
- **5개 dense-prediction 태스크 검증**: semantic/panoptic/instance segmentation, detection, depth. GFLOPs↓/FPS↑ @ 약간의 성능 저하.
- → **Expedit-SAM 변종이 이미 존재** = MemorySAM에 가장 직접적인 출발점.

**DToP** — ICCV 2023, arXiv:2308.01045, "Dynamic Token Pruning ... for Semantic Segmentation"
- **세그멘테이션 네이티브**. per-token classification **confidence**(보조 head의 max prob p)로 점수 → "쉬운" 토큰(p ≥ p0≈0.95) 조기 종료(early-exit), 클래스별 top-k=5 토큰은 문맥용 보존.
- 핵심 논거: "CLS-attention inattentive 토큰 제거는 모든 패치에 dense 예측이 필요한 세그멘테이션에 직접 확장 불가" → 우리가 분류용 프루너(EViT 등)를 그대로 못 쓰는 이유의 근거.
- 두 모드: **@Direct = training-free** (~20% FLOPs↓, ~1.8% mIoU 하락); @Finetune (seg head만 40k iter) = 20-35% FLOPs↓ 무손실 (SETR ViT-B ADE20K 25.2%↓ @ mIoU 47.0 불변).

**Token Transforming** — 2025-06, arXiv:2506.05709, "Unified and Training-Free Token Compression"
- 모든 토큰 축소(프루닝+머징)를 **many-to-many 행렬 변환**으로 통일 — 기존 방법들이 그 특수형. **training-free** (사후 재학습 불필요).
- DeiT-S: ~40% FLOPs↓ / x1.5 @ ~0.1% drop. **dense prediction 확장 명시**: ADE20K seg 30% 무손실 압축, Cityscapes, depth, detection.

## Family 1 — attention/CLS 기반 ViT 프루닝 (전부 학습 필요, 분류 전용)

| 방법 | 점수 기준 | training-free | 결과(ImageNet) | dense? |
|------|----------|:---:|------|:---:|
| **EViT** (arXiv:2108.01390) | **[CLS] attention** — attentive 보존, inattentive **융합** | ❌ 학습 통합 | DeiT-S +50% 속도 @ -0.3% (또는 동일 compute +1%) | ❌ |
| **Evo-ViT** (arXiv:2202.07800) | global class attention, **slow-fast** 경로로 정보/비정보 토큰 별도 갱신(공간구조 보존) | ❌ 처음부터 공동학습 | DeiT-S >60% throughput @ ~0.4% drop | ❌ |
| **A-ViT** (arXiv:2112.07658, NVIDIA, CVPR'22 Oral) | 학습된 **halting score**(ACT), 추가 파라미터 ~0(γ,β 2개) | ❌ 학습(100ep) | DeiT-Ti +62%, DeiT-S +38% throughput @ 0.3% drop | ❌ |
| **AdaViT** (arXiv:2111.15668, Meng et al.) | 3축(patch/head/block) 결정망 + Gumbel-Softmax | ❌ end-to-end | ~2x 효율 @ ~0.8% drop | ❌ |
| **ATS** (arXiv:2111.15667, ECCV'22) | **비파라미터** 미분가능 adaptive sampling (이미지별 토큰수 가변) | ✅ plug-and-play | ~50% GFLOPs↓ (ImageNet/Kinetics) | ❌ |
| **SPViT** (arXiv:2111.11802, TPAMI'24) | self-attn→conv weight-sharing 단일경로 탐색 | ❌ 탐색+학습 | 분류 전용 | ❌ |

⚠️ **이름 충돌 주의**: (1) "AdaViT" 둘 — NVIDIA A-ViT(2112.07658, halting) vs Meng AdaViT(2111.15668, 3축 결정망). (2) "SPViT" 둘 — ziplab attn→conv(TPAMI'24) vs ECCV'22 latency-aware soft pruning. 둘 다 학습 필요.

> **결론**: 분류용 CLS-attention 프루너(EViT/Evo-ViT/A-ViT)는 MemorySAM에 부적합. **training-free + 세그멘테이션 검증된 Expedit / Token Transforming / DToP@Direct**를 우선.

---

# 보강 조사 2: Family 2 (DiT 캐싱) + Family 3 (register/sink) — 직접 WebFetch, 2026-06-13

> deep-research 검증 패스에서 F2/F3가 top-25에 못 들어, 확보한 arXiv ID로 원문 직접 fetch. (arxiv abstract/HTML 기준)

## Family 2 — DiT/Diffusion 어텐션·feature 캐싱 (timestep 간 재사용; 우리는 비-diffusion → 개념 참고용 🟡)

| 방법 | 무엇을 캐시/재사용 | training-free | 결과 | 코드 |
|------|------------------|:---:|------|------|
| **DeepCache** (arXiv:2312.00858) | U-Net **upsampling 고수준 feature**를 N-1 step 재사용(skip의 저수준만 갱신) | ✅ | SD v1.5 2.3x @ CLIP -0.05; LDM-4-G 4.1x @ FID +0.22 | `horseee/DeepCache` |
| **FORA** (arXiv:2407.01425) | **attention + MLP layer 출력**을 interval N으로 정적 캐시·재사용 | ✅ | DiT-XL/2 2.8x@FID2.82(N=3), 5.73x@FID9.80(N=7); PixArt-α 1.5-1.9x | `prathebaselva/FORA` |
| **PAB** (arXiv:2408.12588) | attention 차이의 **U자 패턴** 이용, attention 출력을 **pyramid 방식 broadcast**(분산별 전략) | ✅ | video DiT 최대 **10.5x**, 720p 실시간 | (공개) |
| **L2C** (arXiv:2406.01733) | transformer **layer 단위** 캐시; input-invariant·timestep-variant **router를 학습**해 skip 결정 | ❌ 학습 필요(base는 freeze) | U-ViT-H/2 ~47% compute @ FID <0.01 | (공개) |
| Delta-DiT (arXiv:2406.01125) | DiT 블록 feature **offset(delta)** 캐시 | (미정밀) | — | — |

⚠️ **DeepCache는 U-Net skip 구조 의존 → DiT/ViT 직접 적용 불가**. FORA/PAB/L2C는 transformer(DiT) 대상이라 *개념적으로* SAM2에 더 가까움.
**핵심 인사이트(우리 적용)**: FORA의 "attention+MLP 출력을 인접 step 간 재사용"과 L2C의 "router로 layer skip 학습"은, MemorySAM이 **모달리티 프레임 축**으로 옮기면 — *유사 모달리티 간 memory-attention 출력 재사용/skip* — 비용 절감 아이디어로 전용 가능. (단, diffusion의 timestep ≈ 우리의 modality라는 유추는 검증 필요.)

## Family 3 — Register tokens & Attention sinks (🟢 dense feature 품질 → 세그멘테이션 직결)

**Vision Transformers Need Registers** (Darcet et al., ICLR'24 Oral, arXiv:2309.16588)
- **현상**: 추론 시 저정보 **배경 영역**에 **high-norm artifact 토큰**(attention sink) 발생 → ViT가 그 토큰을 내부 연산 임시저장소로 "납치". feature/attention map을 오염.
- **해법**: 입력 시퀀스에 **추가 register 토큰** 몇 개를 붙여 그 연산 역할을 흡수 → 출력 시 버림.
- **⚠️ retraining 필요** (register 토큰과 함께 재학습; training-free 아님).
- **dense prediction 직접 개선**: "self-supervised 모델의 **dense visual prediction SOTA** 갱신", "**더 매끄러운 feature/attention map**", object discovery 향상.
- → **우리에게 가장 관련 깊은 발견**: artifact 토큰이 세그멘테이션 feature 품질을 떨어뜨리며, register로 교정 가능. SAM2/우리 인코더에 register 토큰 도입 → 야간 dense feature 정제 후보.

**연결**: NotebookLM의 "Attention Sinks in DiT"(Paper D)와 같은 sink 현상이지만, 이쪽은 **ViT dense feature 품질** 관점이라 우리에게 훨씬 실용적. StreamingLLM(arXiv:2309.17453)은 LLM long-context용 sink 보존이라 우리와 거리 멈.

---

## 최종 종합 — MemorySAM 토큰 최적화 우선순위 (전 조사 통합)

1. **🥇 Expedit (+Expedit-SAM)** — training-free, 세그멘테이션 검증, SAM 변종 존재. 즉시 실험 가능.
2. **🥇 Fast SAM2** — memory engine 직전 프루닝, 우리 구조와 정확 대응 (보강 전 1차 조사).
3. **🥈 PiToMe / Token Transforming** — training-free, 정보 토큰 보호 / 세그멘테이션 확장.
4. **🥈 ViT Registers** — 효율보다 **dense feature 품질 개선**(야간 갭에 기여 가능), 단 재학습 필요.
5. **🥉 DToP@Direct** — 세그멘테이션 네이티브 confidence 기준 (보조 head 필요).
6. **참고** — DiT 캐싱(FORA/L2C)은 modality-axis 전용 아이디어로만.

---

# 신규성 조사: A(uncertainty fusion) vs B(reliability-biased attention) — deep-research, 2026-06-15

> 102 agents / 20 sources, 24/25 claim 3-0. 목적: SAM2 memory-attention 기반 멀티모달 세그에서 A/B의 노벨티 + 선행연구 포지셔닝. (coverage mask는 MULTIAQUA 특수성 의존 → 일반화 안 됨 → 핵심 기여서 제외 결정.)

## 0. 베이스라인 — 원조 MemorySAM (arXiv:2503.06700, Liao/Zheng et al., 2025-03, 코드 `Chenfei-Liao/MemorySAM`)
- modality를 "프레임 시퀀스"로 → LoRA-tuned SAM2 encoder + **SAM2 native memory attention**으로 융합. + 학습 전용 **SPMM(Semantic Prototype Memory Module)** + prototypical adaptation loss.
- **최종 융합 = 단순 등가중 평균** `Mask = (1/M) Σ Mask^i` (Eq.6).
- **명시적으로 안 하는 것**: uncertainty/confidence 가중 ✗, reliability mask ✗, **attention-logit bias ✗**, per-modality decoder ✗ (단일 mask decoder).
- → 우리 P-series(UAMM/AMF/SQG)는 이미 base 너머로 "가중"을 추가했으나 **정적 붕괴**. **A·B 둘 다 base 대비 미개척.**
- 성능: DeLiVER 65.38, MCubeS 52.88 mIoU.

## 경쟁자들이 reliability를 다루는 방식 (전부 A·B와 다름)
| 방법 | reliability 처리 | 방식 |
|------|-----------------|------|
| Any2Seg (2407.11351) | cosine-sim 상관맵을 feature에 **곱셈** | feature-multiply |
| MAGIC++ (2412.16876) | cosine-sim로 feature **랭킹** | feature-level |
| RMMSS (2505.12861) | 학습 FSM(depthwise-conv+sigmoid) **max-select** | learned gating |
| EQUISeg (2509.24505) | **랜덤** teacher/student + KL distill (품질선택 회피) | loss-level |
| AG-Fusion (2510.23151) | sigmoid gate를 cross-attn **출력**에 곱 | output-scale |
| ReliFusion (2502.01856) | 학습 Reliability모듈 → confidence를 cross-attn **출력**에 곱 (3D det) | output-scale |
| READ (ICLR'24) | confidence-weighted **loss** (TTA) | loss-level |

→ **decoder 예측 불확실성(entropy/conf, GT-free)으로 가중하는 것 + SAM2 memory 결합한 곳 0건. attention LOGIT에 additive reliability bias 넣는 곳 0건.**

## 판정 (deep-research verdict)
- **🥇 B (reliability를 attention LOGIT에 additive bias) = 더 신규·덜 포화.** 모든 선행연구는 feature-multiply / attention-output-scale / loss-level. **logit-additive bias는 SAM/SAM2 memory attention에서 전례 0.** + MemorySAM의 *핵심 메커니즘*을 개조 → 서사 강함.
- **🥈 A (decoder predictive uncertainty 가중) = 날카롭게 프레이밍해야만 신규.** "학습형 quality head 없이, GT-free·training-free decoder entropy/conf로 SAM2 modality fusion 가중". 단 evidential/TMC/Dirichlet 계열(이번 조사 범위 밖)과 겹칠 수 있어 **추가 확인 필요.**

## ⚠️ 정확한 차별화 포인트 (리뷰 방어용)
- ReliFusion CW-MCA·READ가 "confidence×attention"의 최근접 — 그러나 **출력 스케일/loss**지 **logit bias 아님**. B 프레이밍은 반드시 **"softmax logit 내부 additive bias" vs "출력 재가중"**을 명시 구분.
- READ의 self-adaptive attention을 B 선행으로 보는 주장은 **반증(1-2)** — 그래도 주의.
- base MemorySAM은 단일 decoder+평균 → **A의 per-modality 불확실성은 per-modality decode 필요.** ★우리 P25엔 이미 `_teacher_decode_single`(per-modality decoder)이 있음 → A의 신호원으로 추론 시 재사용 가능 (GT 불필요, entropy/conf만).

## 추천 결합 프레이밍 (최종 후보)
**RBMA — Reliability-Biased Memory Attention**: per-region reliability를 **per-modality decoder 예측 불확실성(GT-free)**에서 뽑아(A의 신호), **SAM2 memory-attention logit에 additive bias로 주입(B의 기구)**.
- 신규 기구(B, 전례0) + 원리적 신호(A, B-2가 지목한 frozen-feature 병목 회피) + **데이터셋 무관 일반화**(uncertainty는 보편, coverage mask 불필요) + 우리가 진단한 **정적 RGB-붕괴를 직접 해결**(입력별 적응 + floor 누수 없음).
- 미확인 1건: A의 신호(decoder uncertainty)가 evidential/TMC dense-seg에서 이미 쓰였는지 → 별도 조사 필요.

## A 신호 신규성 확정 (deep-research 2차, 2026-06-15, 100 agents / 25/25 claim 3-0)
**판정**: "uncertainty가 멀티모달 융합을 이끈다"는 큰 개념은 **포화**. 그러나 **"training-free·GT-free·per-modality PREDICTIVE-softmax/entropy uncertainty로 DENSE 멀티모달 세그 융합"** 정확한 칸은 **미점유.**

반드시 인용·차별화할 최근접 3편:
| 논문 | dense seg? | per-modality unc.→fusion? | 신호 | training-free? |
|------|:---:|:---:|------|:---:|
| **UTFNet** (GRSL'23, RGB-T) | ✅ | ✅ | **evidential/Dirichlet+DST** | ❌ 학습 evidential head+loss |
| **HyperDUM** (CVPR'25, DeLiVER) | ✅ | ✅ (feature 가중) | **학습 hyperdimensional** deterministic | ❌ 학습 prototype |
| **TMC/ETMC** (ICLR'21/TPAMI) | ❌ 분류만 | ✅ | evidential SL | ❌ |
| (의료) Huang'24 | ✅(의료) | ✅ | evidential DS | ❌ |
| DMS (2025) | ❌ MLLM/VQA | ✅ softmax-entropy+MC-dropout | softmax/MC | ❌ 학습 scheduler |
| CMX/TokenFusion | ✅ | ❌ (학습 feature attn) | — | — |

**핵심 차별점 (방어 논리)**:
1. **진짜 차별자 = "TRAINING-FREE"** (evidential head·uncertainty loss 없음 + raw predictive softmax). ⚠️ "GT-free at inference"는 evidential도 만족하므로 **단독 차별점으로 쓰지 말 것.**
2. **A 신호 단독 노벨티는 약함** (UTFNet/HyperDUM가 "per-modality uncertainty for dense seg fusion" 점유). → A는 **신호원**으로만, **B(attention-LOGIT bias)를 전면 노벨티 축**으로.
3. deep-research openQ: **"가중을 attention-LOGIT 레벨에 거는 것 자체가 uncertainty-신호 축과 독립적인 방어가능 노벨티 축"** — RBMA의 핵심 주장.

## ✅ 최종 결정: RBMA 진행
- **헤드라인 = B**: SAM2 memory-attention **logit에 additive reliability bias** (전례 0, feature-multiply/output-scale/loss와 명확 구분).
- **신호 = A(경량 버전)**: per-modality decoder의 **training-free predictive uncertainty**(entropy/conf) — UTFNet/HyperDUM의 *학습 evidential/HD head* 대비 "학습 불필요"가 차별점.
- 차별화 대상: UTFNet, HyperDUM, TMC/ETMC (+ ReliFusion/READ는 trained-head/loss 대조군).

---

# SAM3 이식성 (RBMA → SAM3) — deep-research 2026-06-16, 99 agents / 23·25 claim

## SAM3 사실 (검증)
- **"SAM 3: Segment Anything with Concepts"**, arXiv:2511.16719, Meta, 2025-11 공개. 코드 `github.com/facebookresearch/sam3`, 가중치 `huggingface.co/facebook/sam3` (SAM License), 2026-03 SAM 3.1 업데이트.
- 구조: **detector + tracker 분리**(백본 공유). 백본 = **Perception Encoder(PE, 비전-언어 ViT)** — **Hiera 아님**.
- 헤드라인: **Promptable Concept Segmentation**(text/exemplar 프롬프트, DETR 검출기). RBMA엔 불필요.
- **memory 메커니즘 유지** — 단 **tracker 내부에만** ("tracker는 SAM2처럼 학습"). 그러나 SAM2 `memory_attention` 모듈이 아니라 **memory를 prompt token으로 concat하는 encoder-only transformer**로 재구현(RoPE attention은 내부 존재, call site 다름).
- decoder: two-way transformer 유지하나 **3-mask+confidence + binary prompt-match logit** (multi-class softmax 아님).

## RBMA 이식 판정: 원리적 가능, drop-in 아님 (4곳 재작업)
| RBMA hook | SAM3 | 작업 |
|---|---|---|
| 모달리티-as-프레임 memory attn | ✅ memory bank 유지(tracker) | 위치 이동 |
| **(A)** LoRA 인코더 | PE (Hiera 아님) | LoRA 타겟 재지정 |
| **(B)** attn LOGIT bias 주입점 | `memory_attention` 없음, prompt-token concat | P27 `_sdpa_with_optional_bias` 패턴을 tracker encoder attn에 재이식. memory가 prompt token이라 "모달 memory token 컬럼에 reliability bias 가산"으로 자연 매핑 |
| **(C)** per-modality decoder 불확실성 | binary logit/3-mask conf | ★ 우리 프로젝트는 이미 decoder를 **semantic multi-class로 개조** → SAM3도 동일 개조 시 **softmax entropy 복원**(비차단). 미개조 시 3-mask disagreement surrogate |
| **(D)** DETR concept 검출기 | 신규 | 우리는 tracker만 활용 → 무시 가능 |

## ✅ 코드 레벨 확인 완료 (2026-06-16, GitHub `facebookresearch/sam3` main 직접 확인)

**메모리 융합 경로**:
- `sam3_tracker_base.py` `_prepare_memory_conditioned_features()`: maskmem + object pointer를 `prompt = torch.cat(to_cat_prompt)`로 모아 `self.transformer.encoder(src=current_vision_feats, prompt=prompt, prompt_pos=..., prompt_key_padding_mask=..., num_obj_ptr_tokens=...)` 호출. `num_maskmem=7`, `assert transformer.decoder is None`(encoder-only).
- 그 encoder = **`TransformerEncoderFusion`** (`sam3/model/encoder.py`), 레이어 = `TransformerEncoderLayer`(self-attn → cross-attn-to-memory → MLP). **RoPE 아님** (additive pos enc).

**★ bias 주입점 (RBMA 핵심)** — `TransformerEncoderLayer.forward_pre/forward_post`:
```python
tgt2 = self.cross_attn_image(
    query=tgt + query_pos ...,
    key=memory + pos ...,
    value=memory,
    attn_mask=memory_mask,            # ← 이미 first-class 인자! RBMA bias = 여기
    key_padding_mask=memory_key_padding_mask,
)[0]
```
- `attn_mask=memory_mask`가 **이미 배선**돼 있음. `cross_attn_image`는 주입식 `nn.Module`(시그니처상 `nn.MultiheadAttention` 계열, `[0]` 인덱싱 + float attn_mask = pre-softmax 가산).
- → **SAM2처럼 SDPA를 손패치(`_sdpa_with_optional_bias`)할 필요 없음.** `memory_mask`에 per-modality reliability bias를 넣기만 하면 됨. **SAM2보다 오히려 깔끔.**
- 레이아웃: **seq-first (L,B,C)**. memory(=key) 시퀀스에 모달리티별 memory 블록이 concat → 해당 블록 컬럼에 reliability_i 가산.

**decoder** (`_forward_sam_heads`): `low_res_multimasks [B,M,H,W] (M=3) + ious + object_score_logits`. multi-class softmax 아님 → 우리 semantic 개조 시(MemorySAM 방식) softmax entropy 복원, 미개조 시 ious/3-mask disagreement surrogate.

## SAM3 포팅 결론 (코드 확정)
**이식 가능 + bias 주입은 SAM2보다 쉬움.** 작업 3곳:
1. **(A) LoRA → PE 백본** (`sam3/model/encoder.py`의 PE / 인코더 블록 타겟; Hiera QKV 패턴 대체).
2. **(B) bias** = `TransformerEncoderLayer.cross_attn_image`의 `memory_mask`에 per-modality reliability 가산 (이미 인자 존재, 모달 memory 블록 컬럼에 broadcast). modality-as-frame은 tracker에 각 모달을 frame으로 투입.
3. **(C) 불확실성 신호** = decoder를 semantic multi-class로 개조해 softmax entropy 복원(권장) 또는 ious/object_score surrogate.
- 확인 잔여: `cross_attn_image` 구체 클래스가 `nn.MultiheadAttention`인지(float attn_mask 가산 의미 확정), tracker의 memory 토큰 컬럼↔모달리티 매핑 순서.

## 2026-07-17 — 경쟁 논문 Modality Ablation 조사 (radar 재맥락화)

우리 MUSES-P34 radar 결과(val −0.09/test −0.72)를 경쟁 논문 ablation과 대조. 근거 원문 표 인용.

**문헌 modality ablation 실태:**
| 논문 | ablation 방식 | metric | 수치 |
|---|---|---|---|
| CAFuser (2410.10791 v2) Table IX | 누적 RGB→+L→+R→+E | **PQ** (test) | 55.7 → +L **+3.0** → +R **+0.6** → +E +0.4 |
| MUSES 원논문 (2401.12761 v4) Table 3 | 카메라 대비 개별 | **PQ** (test) | +E +2.6 / **+R +4.4** / +L +5.8. night: cam 39.4→+R 44.8(**+5.4**), snow 42.2→46.1(**+3.9**) |
| CMNeXt (2303.01480) | 누적 | **mIoU** | RGB 57.20→+D **+6.38**→+E +0.86→+L +1.86 (DELIVER, radar 없음) |
| DGFusion (2509.09828 v3) | **per-sensor ablation 부재** | — | 아키텍처/loss만(Table IV/V). DELIVER CLE 51.6→CLDE(+depth) **+5.1 mIoU** |

**핵심 판정:**
1. **radar 기여는 "lidar 유무"에 조건부.** 카메라 대비 radar 단독 = +4.4 PQ(강함)이나, **lidar 위에 얹은 radar = CAFuser +0.6 PQ(≈0)** — radar/lidar가 range 계열로 중복. **우리 P34는 lidar 사용 → 기대 천장 ≈0.** 우리 val −0.09는 이 문헌 잉여성과 정합. **"우리 융합의 radar 실패"로 단정 불가.**
2. **radar는 lidar-열화 조건(night/snow)에서 대체재로 이득**(MUSES night +5.4/snow +3.9 PQ). **aggregate(−0.72)가 조건별 이득을 가릴 수 있음** → per-condition 재검토 필요.
3. **mIoU 기준 modality ablation은 문헌 전무**(MUSES/CAFuser=PQ only, DGFusion=per-sensor 부재, CMNeXt=mIoU지만 DELIVER radar 없음). **우리가 mIoU radar ablation을 내면 문헌 공백을 메움 = 기여.**

**논문 포지셔닝(RBMA 노벨티 방어):** "radar는 MUSES에서 lidar와 중복이라 SOTA(CAFuser)조차 lidar 위 radar는 +0.6 PQ — 우리 −0.72는 radar 무능이 아니라 잉여성. 단 우리 신뢰도 라우팅은 radar를 'lidar 보조'가 아니라 **lidar-degraded 조건의 대체 range 신호**로 라우팅할 수 있다." → 이 주장은 **누적 ablation(RGB→+L→+L+R→+L+R+E, mIoU) + per-condition 수치**로만 방어됨(실험계획 규약 = experiments/plan.md).

**확인 불가**: DGFusion per-sensor(radar) 수치 없음. CMNeXt/CAFuser 원표 PDF는 CVF 403 → arXiv HTML+abstract 교차확인.
출처: DGFusion 2509.09828 v3 / CAFuser 2410.10791 v2 Table IX / MUSES 2401.12761 v4 Table 3 / CMNeXt 2303.01480.


---

## 2026-09-07 — 멀티모달 세그 구조 지형 조사 2축 (user 요청: "다른 벤치·Mamba 계열 등이 어떻게 접근했나") — 요지

> 원시 표(방법 × 백본/frozen × 읽는 층 × 비RGB 경로 × 융합 위치 × 신뢰도 출처 × ablation 수치 × 레시피)는 세션 transcript. 인용은 전부 arXiv 1차 출처. 판정은 [decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md](../decisions/2026-09-07-p52-validity-audit-and-bottleneck-program.md) §3.6.

### 규칙성 5개
1. **5개 벤치(DELIVER·MUSES·MCubeS·NYUDv2·FMB) 1위 어느 것도 "Q/V LoRA + 마지막 층만 읽기 + 후기 융합"이 아니다.** 전부 (i) 매 블록/스테이지 주입·교환 어댑터, (ii) 4레벨 다중스케일 헤드, (iii) 모달별 파라미터·별도 경량 인코더·멀티모달 사전학습 중 둘 이상. 같은 SAM2에서 Q/V LoRA만 쓴 SARTM(2505.01950)은 매블록 어댑터 SHIFNet(2503.02581) 대비 FMB −6.2.
2. **어댑터 깊이·접점의 문헌 이득**: Q/V→attn+MLP LoRA +1.5(SoMA 2412.04077 T8) · Q/V LoRA→주입형 어댑터 +1.4(MM SAM-adapter 2509.10408 T9) · 마지막 층→4탭: frozen 백본에서 +2~4(SHIFNet T7, DINOv2 2304.07193 T11 depth), 백본 전체 FT면 0(ViTDet 2203.16527) · 1스테이지만→전 스테이지 주입 +0.5~1.3(DPLNet 2312.00360, StitchFusion 2408.01343, CrossWeaver 2604.02948, GeminiFusion 2406.01210) · 상위 절반 블록 부분 FT +1.4~1.8(SpectraDINO 2605.02258 T8: full FT는 −9.4 붕괴) · 비RGB 별도 경량 인코더: RGB-hard +7~11, 전체 +1~2(MM-SA T9/T12).
3. **가장 큰 레버는 여전히 백본·사전학습**: GeminiFusion B3→Swin-L +3.4, StitchFusion B4→Swin-L +2.0, OmniSegmentor(2509.15096) 정렬 사전학습 +2.4~5.1, DFormer(2309.09668) depth 사전학습 +15.2(depth 단독).
4. **자기파생 신뢰도의 clean 이득은 백본 ≤30M에서 ≤1.5(RSGMamba 2604.12319 +1.5, MAGIC 2407.11344, CrossWeaver), Swin-T급 이상에서 ≈0~0.4(CAFuser 텍스트 지도 없으면 0, 함수엔트로피 +0.3).** 외부 신호(깊이 GT·텍스트)가 붙어야 +0.4~1.3(DGFusion·CAFuser). 우리 30세대 실패와 정합.
5. **Mamba는 답이 아니다**: 동일 백본 융합 이득 0.6~1.1(CM-SSM 2506.17869, MambaSeg 2512.24243), 순수 Cross-Mamba ≈ Add(RSGMamba); 대형 백본 벤치는 attention/어댑터 계열이 1위. 이득의 실체는 FLOPs.

### 레시피
조사한 논문 전부 표준 증강(scale 0.5~2, flip, jitter, blur)뿐. **물리 열화 증강(PhysAug류) 사용 사례 0** — MUSES PhysAug-on은 공정선 밖(user 지적 확인). CAFuser만 모달 20% 랜덤 드롭.

### 우리 계보와의 대조 (주의)
- StitchFusion-MoA/CrossWeaver-MIB식 "인코딩 중 모달 간 교환"은 우리 P51-CMLC(LoRA 부분공간 대칭 결합, −0.82)와 정신이 같다. 재시도하려면 차이(블록마다 attn 뒤·FFN 뒤 양방향 저랭크 MLP 교환 vs LoRA 코드 결합)를 먼저 써야 한다.
- MM-SA식 "별도 경량 인코더 + 비대칭 주입"은 P49/P49.1에서 실패(−0.8/−0.97). 단 P49는 **DINOv3 전체 FT(RGB_FT)**를 동반했고, SpectraDINO T8은 소량 데이터 전체 FT가 −9.4 붕괴·상위 절반만 FT가 최선임을 보인다 → P49 실패의 원인이 주입이 아니라 전체 FT였을 가능성(재해석 후보).
- 미확인: MUSES 리더보드 GtA 82.39(camera-only)는 두 조사 모두 웹에서 찾지 못함(Codabench 로그인 페이지). 세션 메모리 값 — 제출 전 재확인 필요.

---

## 2026-09-08 — 최신 지형 전수 스윕 (user 요청: "기존 기록 이외의 신규 접근 조사" — 웹 병렬 3축: 세그 / 검출 / SAM·메모리 경쟁축)

> 조사 방법: 병렬 웹 조사 에이전트 3개(WebSearch+arXiv 원문 대조) + 기존 기록(canonical 2문서 + 볼트 사본, arXiv ID 191건) 전수 대조. 아래는 **기존 기록에 없던 신규 항목 중심**이며, 기존 항목은 갱신된 사실(venue 확정 등)만 적는다. 미확인 수치는 표기대로 미확인.

### A. 노벨티 판정에 직접 닿는 발견 (최우선)

1. **MemorySAM(arXiv 2503.06700) 게재 상태 = 여전히 preprint.** GitHub bibtex(`@misc`)·저자 홈페이지 모두 학회 표기 없음. 일부 검색 요약의 "ICCV 2025 accepted"는 1차 출처에서 확인 불가 → **미확인 판정**. 코드·DELIVER/MCubeS 가중치는 공개돼 있음(스타 45). 구현 재확인: 모달을 프레임 시퀀스로 넣고 2번째 모달부터 memory attention으로 이전 모달 메모리에 cross-attend, 인코더는 동결+Q/V LoRA(rank4, 모달 공유), memory attention·memory encoder·mask decoder는 풀튜닝, SPMM(학습시 전용)의 prototypical adaptation loss. **신뢰도/가중 개념은 원문에 전무** — 모달 마스크 균등 평균의 "memory residual"뿐. ablation: 메모리 제거 −3.62, SPMM 제거 −1.07 (RGB-D).
2. **"training-free 신뢰도 → memory-attention pre-softmax additive bias" 셀은 이번 스윕에서도 점유자 미발견.** 단, 이 판정은 웹 스윕 한정이며 **기존 기록의 PRIMED(2605.07154, learned modality-prior additive pre-softmax bias, RAVS)·SAE(2603.16558, training-free entropy additive bias, LVLM)가 여전히 최근접 위협**으로 유효 — 전문 정독 TODO는 그대로 blocking.
3. 🔴 **신규 must-cite: DFormerv2 (CVPR 2025, arXiv 2504.04701, 난카이 VCIP, 코드 공개).** depth를 별도 인코더로 태우지 않고 **depth에서 뽑은 기하 prior로 self-attention 가중 배분을 변조**하는 Geometry Self-Attention(GSA). "보조 모달을 attention 변조로 주입"이라는 점에서 RBMA와 개념적으로 가장 가까운 CVPR급 이웃. 차별축: GSA의 신호는 **기하(거리)**이지 신뢰도가 아니고, 무대도 SAM memory attention이 아님. 원문에서 additive인지 multiplicative인지 정독 필요(→ lit-check TODO 7).
4. **RSGMamba(2604.12319) 상세 확보**: Reliability-aware Self-Gated Mamba Block이 모달 신뢰도를 명시 모델링해 cross-modal 상호작용 강도를 self-gating(곱셈적)으로 조절. MFNet 61.1/PST900 88.9/NYUDv2 58.8/SUN 54.0, 48.6M. RBMA 방어 문구: "곱셈적 게이팅 vs 가산적 logit bias, 학습형 vs training-free".
5. **검출 쪽에서도 2025~26년에 신뢰도-가중 융합이 급증** (JFRDet 조도 게이팅, MDQF 열화-모달 배제, SAMFusion 날씨별 센서 적응). 전부 게이팅/선택이며 attention-logit bias는 아님 — det-additive-bias 빈 셀 재확인.

### B. 세그멘테이션 — 신규 파악 항목

- **GeminiFusion** (arXiv 2406.01210, ICML 2024로 알려짐·venue 미확인, 코드 공개): 백본 블록의 attn과 MLP 사이 픽셀단위 cross-attention 교환, 선형 비용. 단 CVPRW 2025 강건성 벤치마크에서 "과도한 모달 교환이 노이즈를 전파해 노이즈 조건 급락" 판정 — 교환량과 강건성의 트레이드오프 실증 사례.
- **센서 실패 강건성 벤치마크** (arXiv 2503.18445, **CVPRW 2025 Best Paper**, MemorySAM 1저자 동일 인물, 코드 공개): DELIVER 기반 Entire/Random Missing + noisy-modality 프로토콜로 CMNeXt·MAGIC(++)·GeminiFusion·StitchFusion 재평가. MAGIC++ 최상위, GeminiFusion 급락. **이후 DELIVER 신작은 거의 전부 missing 수치를 병기** — 우리 평가 프로토콜에도 결손·노이즈 축 추가 검토 근거.
- **EGFormer** (2505.14014): 모달별 중요도 점수(ASM) + 스테이지마다 저정보 모달을 실제 탈락(MDM). "융합 대신 버리기"의 효율 노선.
- **MMSFormer** (2309.04001, IEEE OJSP 2024): MCubeS 53.11로 상위권 기준점(RGB+AoLP+DoLP+NIR).
- **Sigma** (2404.04256, WACV 2025, 코드 공개): 최초 SSM 멀티모달 세그(샴 VMamba + Cross/Concat Mamba Block). 수치 출처 간 불일치(통상 MFNet 61.3/PST900 88.6) — 인용 시 원문 재확인.
- **AlignMamba** (2412.00833, CVPR 2025): 세그 아님(융합 일반). OT 국소 정렬+MMD 전역 정렬로 Mamba 순차 스캔의 cross-modal 한계 보완 — Mamba 융합 참고용.
- **EIFNet** (2507.21971): event-image 융합 세그(DDD17/DSEC-Semantic SOTA 주장, 수치 미확인).
- **KAN-SAM** (2504.05878, ICME 2025): thermal을 KAN 어댑터로 SAM2에 프롬프트 주입 + 상호배타 랜덤 마스킹(RGB 의존 저감). RGB-T SOD.
- **CRISP-SAM2** (2506.23121): 언어-영상 cross-modal 프롬프트를 SAM2 메모리 파이프라인에 태우는 의료 세그 — 겹침 하.
- **MedSAM2 계열** (2504.03600 등): 시간축 메모리를 **z축(볼륨 슬라이스)으로 전용** — "메모리 축 재해석" 계보의 선례로 관련연구 절 인용 가치.
- **SAM3-Adapter** (2511.19425): SAM3 첫 어댑터 프레임워크(camouflage/shadow/의료). **SAM3의 멀티모달 센서 적응은 여전히 공백 — 선점 여지.** SAM3 구조 재확인: detector-tracker 분리로 memory attention은 트래커 쪽에 잔존 → 모달 축 전용하려면 트래커 브랜치를 떼어 써야 함(기존 10_related_work §SAM3 분석과 합치).
- venue 확정: **StitchFusion = ACM MM 2025**, **MM SAM-adapter = IEEE 게재**(Xplore 11162503), SHIFNet = IROS 2025(기존 기록대로).
- 스텁(미정독): HAPNet(2404.03527), BIMII-Net(2503.19303), Mul-VMamba(KBS, paywall 유지), SARTM 수치 미확인 유지.
- 벤치 포화 신호: MFNet 61~62 정체, DELIVER 68대(StitchFusion), MUSES val 79~80 — 신작들은 절대치보다 강건성·효율·일반화로 차별화 중. MUSES GtA 82.39(camera-only)는 이번에도 웹 재확인 실패(Codabench 로그인 장벽) — 제출 전 재확인 의무 유지.

### C. 검출 — 신규 파악 항목 (기존 기록이 얇던 축, 대부분 신규)

**미스얼라인 3세대 진화**: 암묵 적응 → 명시적 offset/affine + deformable 정렬 → 쿼리·출력 공간 분리.
- **OAFA** (CVPR 2024): CSOM으로 모달 공통 부분공간에서 feature-level 오프셋 명시 추정 + ODAF deformable 정렬 — weak-misalignment 기준점.
- **CoDAF** (2506.16737): offset-유도 정렬(OSA) + 동적 attention 융합(DAFM), DroneVehicle 78.6 mAP.
- **JFRDet** (2608.10680): 대변위 affine을 이미지 레벨에서 추정해 feature 워핑 + **조도 기반 모달 신뢰도 가중(IGCF)** + **정렬 신뢰도로 검출 supervision 강도 게이팅(AQCG)**. DVMA(DroneVehicle-Misaligned) 신규 벤치 제안, 69.7 mAP50.
- **DPDETR** (2408.06123): 객체를 (카테고리, visible 좌표, IR 좌표)로 분해해 **모달별 박스를 둘 다 예측** — 정렬 문제의 출력-공간 해법(가장 급진적).
- **MDQF** (2601.08458, 기존 스텁 → 상세 확보): 모달별 독립 DETR 브랜치 + 디코더 단계마다 고품질 쿼리 선별 교환, 열화 모달 융합 배제 가능, **unpaired 학습 가능**.
- **MS-DETR** (2302.00290, TITS 2024, 코드 공개): loosely-coupled 디코더 융합 + instance-aware modality-balanced loss — 보행자 DETR 표준.
- **DAMSDet** (ECCV 2024, 기존 스텁 → 상세 확보): Modality Competitive Query Selection + multispectral deformable cross-attention(모달별 독립 sparse 샘플링으로 정합 불요).

**주파수 분해가 지배적 도구화** (융합·증류 공통):
- **F2Net** (Sensors 2026): RGB 고주파+thermal 저주파 분리 가중, M3FD 89.6 mAP50. **DRPFNet** (ICME 2026, 2608.03370): 주파수+공간 dual-domain 점진 융합, LLVIP 97.8 mAP50. **IC-Fusion** (2505.15137, 코드 공개): wavelet 분석으로 IR 우위 논증 → **IR 중심 비대칭 설계**(RGB 백본 소형화).
- **FreqKD** (2606.11572): RGB-IR 특징 발산이 고주파에서 저주파의 2.4×라는 정량 관찰 → 대역별 비대칭 증류. **uniform feature-matching/cosine/response KD 전부 베이스라인 미달**이라는 부정 결과 병기 — cross-modal KD 설계 시 필독.

**융합 회의론의 실증화** (우리 poongsan "RGB-only ≥ 3-modal" 관찰과 수렴):
- **M²D-LIF** (2503.11780, 코드 공개): linear probing으로 joint 멀티모달 학습이 단일모달 표현을 부실화시키는 **"Fusion Degradation"** 정량 규명 → Mono-Modality Distillation + 경량 조도 융합으로 해소.
- **Words-to-Wavelengths** (2512.15971): GroundingDINO/YOLO-World를 멀티스펙트럴로 개조 — few-shot에선 **VLM prior가 전용 융합 모델을 압도**.
- 문헌 전반: "융합 자체"에서 "모달별 표현 품질 확보 후 선택적 융합"으로 무게중심 이동(M²D-LIF·MDQF·JFRDet 공통).

**foundation→thermal 이식**:
- **Thermal-Det** (**CVPR 2026**, 2605.10130, Patel 그룹): 열화상 캡션 100만+ 합성 + frozen RGB OV teacher 증류 + full FT, IR 7벤치 zero-shot +2~4 AP — OV 검출기의 thermal 이식 레시피 첫 체계화.
- **UniRGB-IR** (2404.17360, ACM MM 2025): frozen RGB 파운데이션에 어댑터로 IR 주입, 검출·세그 통일.
- **AMFD** (2405.12944, TMM 2025, 코드 공개): 멀티모달 teacher→단일 student 증류에서 융합-후 특징이 아니라 **융합-전 원 모달 특징**을 증류(MEA) — GISTOLO와 직접 비교 대상.

**기타**: YOLOv11-RGBT(2506.14696, 6융합모드 통일 프레임워크+MCF, 코드 공개 — 베이스라인 인프라), **SFEDet**(2606.30215, **ECCV 2026**): 희소 RoI에만 cross-modal 융합(융합 연산 희소화), E2E-MFD(NeurIPS 2024 Oral, 2403.09323: fusion image 생성과 검출의 gradient-수준 동기 joint), FAOD(2412.04149: event 주모달+Time-Shift 일관성으로 비동기 강건), MCFNet(2508.10704: optical-flow 이벤트 동기화+Cross-Modal Mamba), COMO(2412.18076: cross-Mamba+고레벨만 교환), CAGT(Inf. Fusion 2024: RoI 단위 TSR 분해 정렬), MM-DETR(TCSVT 2025), Mixture-of-Scale-Experts(2410.12143, alignment-free RGBT VOD), UAV-CB(2603.17492, anti-UAV RGB-T 데이터셋).

### D. 후속 TODO (이번 스윕이 새로 남긴 것)

1. DFormerv2 GSA 수식 정독 — attention 변조가 additive/multiplicative/거리감쇠 중 무엇인지 확정 후 novelty §2 비교표에 행 추가.
2. SFEDet(ECCV 2026)·Thermal-Det(CVPR 2026) 카메라레디 공개 시 수치 확정.
3. 검출 P-Det 스토리에 M²D-LIF의 Fusion Degradation 진단(linear probing)을 우리 poongsan ablation에 적용해 볼 것 — "RGB-only ≥ 3-modal"의 원인 규명 도구로 적합.
4. MemorySAM venue 추적 유지(현 preprint) — 학회 게재 확정 시 인용 갱신.
5. 결손·노이즈 강건성 프로토콜(2503.18445) 채택 여부 결정 — DELIVER 신작 관행이 됐으므로 리뷰어 요구 가능성 높음.

---

## 2026-09-08 (추가) — missing/noisy-modality 평가 프로토콜 상세 (user 질의: "우리도 결측 실험 하려면 세팅 파악" — 논문 HTML + 코드 repo 확인)

> 배경 판단(세션): "training-free reliability → memory-attn additive bias" 셀이 비어 있는 이유의 절반은 **clean-set에서 자기파생 신뢰도 이득이 작다는 것을 분야가 경험적으로 알기 때문**일 수 있다(09-07 규칙성 4와 정합). RBMA 주장 형태는 "clean 동급 + 결측·열화에서 training-free 강건"이어야 하며, 아래 프로토콜이 그 입증 무대다.

### 프로토콜 정의 (벤치마크 = arXiv 2503.18445, CVPRW'25 Best Paper; 코드 `Chenfei-Liao/Multi-Modal-Semantic-Segmentation-Robustness-Benchmark`)

- **EMM(Entire-Missing)**: 4모달에서 결측 0~3개의 **15개 조합을 열거**(전부-결측 제외, 전부-존재 포함). 결측 모달은 **정규화 후 배치 텐서에 `images[index].zero_()`** (브랜치는 그대로 통과). 지표 = ① 15조합 단순평균 `mIoU^Avg_EMM` ② 모달별 독립 Bernoulli 고장확률 p∈{0.2,0.1,0.05}로 조합 가중한 기대값 `mIoU^E_EMM`(`p^k(1-p)^(n-k)` 가중, 총확률 정규화).
- **RMM(Random-Missing)**: 같은 15조합 루프에서 대상 모달에 **픽셀×채널 독립** `torch.rand(shape) < r` 마스크로 0 주입. ⚠️ README는 "retention ratio"라 부르지만 **코드상 r = 드롭 비율**(r∈{0.25,0.5,0.75}; 릴리스 스크립트는 0.25만 하드코딩). 집계는 EMM과 동일 2종.
- **NM(Noisy)**: 결측 없이 노이즈 3단계 — salt-and-pepper density {0.05,0.1,0.2}(min/max 픽셀 치환, 4모달 전부) + Gaussian σ {0.1,0.2,0.5}(Event 제외). 🔴 **릴리스 코드에서 Gaussian 호출부가 주석 처리돼 있어 그대로 돌리면 S&P만 적용** — 논문 식(Eq.7)은 합산 명시. 우리 보고 시 어느 정의인지 반드시 명시.
- **MAGIC 계열(modality-agnostic) 프로토콜은 별개**: 15조합 전수 평가 + Mean은 같지만 결측 브랜치를 **아예 건너뜀**(weight-shared 인코더, zero-fill 아님). MAGIC repo는 **코드 미공개**(README뿐).
- **CAFuser 학습 시 드롭**: 샘플 단위·모달별 독립 p=0.2(RGB 포함), **정규화 전 raw 이미지를 `np.zeros`로 대체**. 최소-1-모달 보장 로직은 코드에서 미발견. test 시 `missing_mod` 리스트로 EMM식 평가 내장.
- **DELIVER 내장 corner case와의 관계**: MB/OE/UE(RGB)·LJ(LiDAR)·EL(Event) 5개 파티션은 시뮬레이션 단계에서 구워진 "저품질로 존재"이고, EMM/RMM/NM은 사후 주입 "부재/합성열화" — 보완 관계.

### 대표 수치 (벤치마크 README, DELIVER 4모달)

- EMM 15조합 평균: CMNeXt 37.90 · GeminiFusion 37.07 · MAGIC 44.97 · **MAGIC++ 44.85**(E(p=0.2) 59.18로 최고) · StitchFusion 41.98.
- RMM 평균(MAGIC++): 53.92(r=.25)/49.31(r=.5)/47.06(r=.75). NM: 전 모델 붕괴(High에서 CMNeXt 2.31, MAGIC++ 8.70).
- 🔴 **수치 계열 혼용 금지**: 같은 CMNeXt가 벤치마크 EMM 평균 37.90 vs MAGIC계 논문 15조합 Mean 25.25로 크게 다름(원인 미확인 — zero-fill 위치·해상도·ckpt 차이 추정). Any2Seg "+19.79"는 후자 계열(CMNeXt 25.25 대비). 비교는 같은 프로토콜 계열 안에서만.

### 우리(SAM2 4모달) 재현 체크리스트

1. val 루프에 15조합 열거 + **정규화 후 `zero_()`**(raw-zero 아님) → 조합별 mIoU + 평균 + Bernoulli 기대값 3종.
2. RMM: 픽셀×채널 rand 마스크 r 3단계. NM: S&P 3단계(+Gaussian 여부 명시).
3. **우리 고유 세팅 2종 보고 가능**: (a) zero 프레임을 memory attention에 그대로 통과(벤치마크 정합) vs (b) 결측 모달 프레임을 memory에서 제외(MAGIC식) — (b)는 memory 구조의 강점 어필 + RBMA가 (a)에서 zero-프레임 기여를 자동 억제하는지가 핵심 가설.
4. 학습 강건화 시: CAFuser식 p=0.2 샘플 단위 드롭 또는 AnySeg식 최소-1-모달 마스킹 — 학습·평가의 zero 주입 위치(정규화 전/후)를 통일할 것.
5. 검증 순서: 벤치마크 `val_mm_*.py`를 DELIVER 코드베이스에 얹어 CMNeXt 37.90 근처 재현 확인 → 우리 모델 적용.

미확인: MAGIC 실제 코드, 벤치마크의 MAGIC 재평가 방법, NM 논문 수치의 Gaussian 포함 여부, 37.90 vs 25.25 괴리 원인, AnySeg 드롭 확률값.

---

## 2026-09-08 (추가 2) — CVPR 2026 본회의 + ECCV 2026 채택작 표적 전수 스캔 (user 질의: "2026 CVPR/ECCV에는 관련 연구 없었나")

> 조사 방법 = 리스트 전수 스캔이라 근거 수준이 높다: **CVPR 2026** = openaccess.thecvf.com 본회의 4,042편 제목 전수 + 후보 ~30편 abstract 확인(워크숍은 PBVS·URVIS·MaCVi·DriveX 등 9개만) / **ECCV 2026** = 공식 예비 accepted 리스트(eccv.ecva.net, ~2,864건) 통째 다운로드 후 키워드 전수 검색(1차 출처; SFEDet 존재로 교차검증).

### 🔴 노벨티 판정 (두 학회 공통)

- **RBMA 직접 충돌 논문 = 0건.** CVPR 본회의 SAM 계열 16편 전수 확인: "SAM2 memory attention의 모달리티 축 전용" 없음, "신뢰도→attention logit additive bias" 없음. ECCV 리스트에서 "memory attention" 문자열 포함 제목 0건. → **전칭부정("셀 미점유") 주장의 2026 양대 학회 커버리지 확보**(제목 기반 스캔이므로 제목에 단서 없는 논문 누락 가능성은 잔존).
- 단, **"신뢰도/품질로 불량 모달을 억제"라는 상위 스토리는 이미 혼잡한 채택 클러스터가 됐다**: RAF·InfraNet·RA-SOD·GIML(ECCV) + CoRiM·UMFNet·ACR(CVPR). 전부 학습형 gating/weighting/quality-map이며 logit bias·training-free 아님 → 논문에서는 스토리가 아니라 **메커니즘(pre-softmax logit-additive, training-free, SAM2 memory 무대)** 차별화를 전면에.

### 구분 인용 필수 이웃 (신규 must-cite 후보)

- **M4-SAM** (CVPR 본회의): SAM2+멀티모달(RGB-D VSOD)+memory 키워드 전부 겹침. 실제로는 융합=인코더 Modality-Aware MoE-LoRA, memory attention=시간축 유지(pseudo-mask로 bank 초기화). 기존 기록의 arXiv 2605.11760(M⁴-SAM)이 CVPR 2026 본회의로 확정된 것.
- **RobustSeg** (CVPR 본회의) = RMMSS(2505.12861)의 확장판 게재 확정. **DELIVER 명시 사용 유일 본회의 세그 논문**: teacher-student + Hybrid Prototype Distillation로 missing-modality +2.40%, full-modality −0.1%. 백본 AnySeg·CMNeXt. 코드 github.com/RobustSeg/RobustSeg. **우리 결측 실험(EMM/RMM)의 직접 비교 대상.**
- **SENTRY** (ECCV, 2606.24449): **training-free plug-and-play로 SAM2 메모리를 조작하는 유일한 채택작** — 쓰기 전 시간 일관성 검증(refine-before-write) 게이트. 시간축·단일 RGB라 충돌 아님, "training-free SAM2 memory 개입" 선례로 인용.
- **RAF** (ECCV, 2607.04587, KAIST 윤국진랩, 코드 공개): per-pixel **학습형** reliability map으로 악천후 3D 검출에서 불신 카메라 특징 억제(+6.5 AP_BEV). / **InfraNet** (ECCV, 2607.03795): QualGate 품질 게이트로 저품질 RGB 가이던스 억제(LLVIP·FLIR·M³FD·DroneVehicle). / **GIML** (ECCV, 2607.06943): 모달 통째 결손과 부분 열화를 연속 품질 저하로 통합, noise-aware quality estimator — UAMM 문제의식과 최근접.
- **CoRiM** (CVPR 본회의): per-sample 동적 융합을 Modality Conflict Risk 최소화(Frank-Wolfe)로 정식화, **scalar confidence 가중의 이론적 한계 지적** — 신뢰도 가중 계열 포지셔닝 시 이론 인용. / **UMFNet** (CVPR 본회의): 픽셀별 Gaussian 불확실성→confidence map으로 feature 변조(unaligned RGB-T SOD).

### CVPR 2026 그 외 (주제별)

- 세그: MM-OVSeg(optical+SAR OV, 코드), SkySense-VITA(원격탐사 파운데이션), REL-SF4PASS(2601.16788, 파노라마 RGB-D), MARSS/RAVEN(radar 세그). MUSES·MCubeS·FMB 명시 본회의 논문 없음. **URVIS 워크숍에 MUSES 기반 멀티모달 판옵틱 챌린지 리포트(2604.16984)** — MUSES 동향 필독.
- 검출: **Distribution-Aligned Multimodal Fusion**(융합 특징을 frozen 검출기의 사전학습 분포에 정렬 — 미지 열화 일반화 우월 주장), DyFCLT(RGBT tiny, 주파수 분리), **DSERT-RoLL**(2604.03685, KAIST: stereo event+RGB+thermal+4D radar+dual LiDAR 주행 데이터셋), **MMVIP**(최초 대규모 해양 VIS-IR 페어 128K — MULTIAQUA 관련연구 인용 후보), SEATrack(Oral, RGB-T/D/E 통합 트래커, LoRA+계층 MoE). radar-camera 2D 검출 없음(전부 3D).
- missing/강건성: Missing No More(2603.08018, 공유 dictionary 계수 도메인에서 VIS→pseudo-IR 추론 후 융합, 코드), ACR(2603.02200, 융합 confidence가 단일모달보다 낮아지는 confidence degradation 벌점화), SCDT·CodeAlign·DCTrack(워크숍).
- SAM: 비RGB 센서 적응 본회의 0건, SAM3 downstream은 워크숍 벤치 1편뿐. evidential 센서 융합 본회의 0건.
- InfoCalib(URVIS 워크숍): 정적 주파수 attention의 분포이동 붕괴를 정보이론 진단 + 융합 가중 온라인 재보정.

### ECCV 2026 그 외 (주제별)

- 세그: **SegFly**(2603.17920, 항공 RGB-T — 2D→3D→2D 라벨 리프팅으로 대규모 pseudo GT, 데이터셋 20.6K RGB+15K 정렬 RGB-T 공개; 드론 도메인이라 우리와 인접), **UMSS**(2607.12372, 비지도 멀티모달 세그 최초 정식화, DINOv3 기반, fusion degradation 해결), HyperRadar(멀티뷰 radar), Dark-Scenes dense depth(LiDAR+event+RGB, 프리프린트 미공개). **DELIVER/MUSES 정면 supervised 경쟁작 부재.**
- 검출: SFEDet(재확인), **"DETR is Secretly a Multispectral Detector"**(zero-parameter adaptation — training-free 계열, 프리프린트 공개 시 추적 필수), RA-SOD(신뢰도 RGB-T SOD, 프리프린트 미공개), 회전등변 multispectral.
- SAM/파운데이션: SAM+D(2607.29033, depth-routed LoRA로 2D→3D 승격, 의료), **REALM**(2605.00271, Toronto STARS: 이벤트 스트림을 frozen RGB 파운데이션 latent에 LoRA 투영해 RGB 디코더 zero-shot 재사용 — "비RGB를 RGB 파운데이션에 태운다" 철학 유사, 코드 예정), X2SAM(RGB 전용). **비RGB 센서 SAM 적응은 ECCV에도 0건 — 선점 여지 유지.**
- missing: MARS(2606.30355, 완전-불완전 표현 residual로 MoE expert를 결손 패턴별 특화), BIP(프롬프팅).
- venue 부정 확인: EQUISeg·MM SAM-adapter·EGFormer·CBC-SLP·Frequency-Guided RGB-T(2605.26273=CVPR PBVS 워크숍이 맞음)는 **ECCV 2026 리스트에 없음**.

### 후속 TODO

1. 프리프린트 미공개 채택작 추적: RA-SOD, DETR-multispectral(zero-parameter), BIP, Dark-Scenes depth, MMVIP, CoRiM, DyFCLT.
2. RobustSeg 코드로 우리 EMM/RMM 결과와 직접 비교(같은 DELIVER·같은 프로토콜 계열인지 확인 후).
3. M4-SAM venue를 기존 기록(arXiv 2605.11760)에 CVPR 2026 본회의로 반영 — 완료(이 절). MemorySAM은 여전히 양대 학회 리스트에 없음(preprint 유지).

---

## 2026-09-20 딥리서치 B — 품질 조건부 MoE-LoRA·어댑터(에이전트 원문, 35편)

## 한 일
- WebSearch 14회·WebFetch 40여 회로 2023~2026 논문 35편의 초록·본문을 직접 확인하고 아래 표 4개를 작성했다.
- 본문(HTML/PDF)까지 읽은 논문은 "확인" 표시, 초록만 읽은 논문은 "초록만"으로 구분했다. 빈 셀은 §5에 모아 두었다.
- 앞선 조사(2026-09-17)의 SMEAR·MoCLE·ST-MoE 수치는 이번에 재확인하지 않았으므로 아래에 다시 인용하지 않는다.

## 결과

### 표 1. 모달 품질·열화 조건부 LoRA/어댑터 라우팅 (라우터 입력이 "내용 토큰"이 아닌 설계)

| 논문 (arXiv/학회) | 과제 | 라우터 입력 | 라우터 감독 | 전문가 수·형태 | clean 이득 | 열화·결측 이득 | 붕괴 방지 기제 | 확인 수준 |
|---|---|---|---|---|---|---|---|---|
| LoRA-IR (2410.15385) | 올인원 복원 | CLIP 열화 표현(다운샘플 이미지 + 슬라이딩 패치, MLP 결합) | **열화 라벨 CE 감독**(1단계 20분 사전학습, 2단계 PEFT) | n = 열화 유형 수, LoRA 전문가, top-k 후 재softmax | 표 10(Snow100K): 전체 32.28 vs LoRA 전문가 제거 32.01 PSNR | 14 과제·29 벤치 SOTA 주장(혼합 열화에서 이득 크다고 서술) | 로드밸런스 손실 언급 없음(감독 라우터로 대체) | 본문 확인 |
| MoR-DASR (2511.16024, AAAI 2026) | 실세계 SR | CLIP 이미지 임베딩과 긍/부정 텍스트 쌍의 코사인 → 열화 점수(무감독) | 라벨 없음 | 40 rank = 공유 8 고정 + 라우팅 32, top-8, zero-expert 4 슬롯 | LoRA 0.670 → LoRA+MoE 0.689 → MoR 0.717 CLIPIQA(TRES 78.81→79.36→81.78) | 위 수치가 열화 입력 기준(SR) | 열화 강도에 따라 활성 전문가 수를 바꾸는 로드밸런스 손실; zero-expert 4가 0·8보다 우수(표 9) | 본문 확인 |
| DAME-Net / CD-MoE (2604.09313) | UAV 복합 열화 복원 | 요인별 열화 존재 확률(multi-label) | **multi-label BCE + 라벨 유사도 소프트 정렬** | 전역 3 + 공간 5 = 8 전문가 + base branch(전문가 0개여도 용량 유지) | 미보고 | 열화 43 조합; 의미 임베딩 제거 −1.46 dB(4중 열화 −3.31), 하드 마스크 제거 −0.56, 분리 MoE→표준 MoE −0.70(4중 −1.50) | **부재 열화는 하드 key-padding 마스크로 배제**(소프트 다운웨이트 아님) | 본문 확인 |
| DA-CLIP (2310.01018) | 복원 | 동결 CLIP + 컨트롤러 | 합성 캡션(열화 유형)으로 학습, 열화 분류기 겸용 | 전문가 없음(cross-attention 조건화) | 미보고 | 열화별·통합 복원 SOTA 주장 | 해당 없음 | 초록만 |
| WM-MoE (2303.13739) | 블라인드 날씨 제거 | [내용 토큰; 날씨 토큰] 채널 결합 → 비선형 어댑터 | 날씨 라벨로 **패치 단위 대조학습(WGF-CL)**; 추론은 블라인드 | 16 전문가(4 그룹, DWConv 1/3/5/7) | 미보고 | MAW-Sim: ViT 29.51 → 단순 MoE 29.69~29.84 → WEAR 29.80~29.99 → +멀티스케일 30.18~30.33 PSNR | 미기술 | 본문 확인 |
| AW-MoE (2603.16261) | 전천후 멀티모달 3D 검출 | 이미지 특징 → 날씨 분류기(≈99% 정확) | **날씨 라벨 CE 감독** | 날씨별 전문가 7, top-K=1 (K=2와 동일 성능) | 미보고(normal 조건 수치 표에 있으나 추출 실패) | Fog 88.6→95.3, 경설 78.9→90.2, 폭설 51.9→64.0 AP3D(IoU .3) | 로드밸런스 없음; 확신 가중 MoE 손실 | 본문 확인 |
| **RobuMTL** (2601.10921, WACV 2026) | 날씨 섭동 하 멀티태스크 밀집 예측 | 원본 이미지 → 경량 CNN(SE) 섭동 분류기(rain/snow/noise/blur/fog/clean) | **섭동 라벨로 별도 지도학습** | 섭동별 계층 LoRA 전문가 ≈7(clean 포함), top-1(단일)·top-6(혼합); 백본 동결 | 미보고 | PASCAL Δm: clean LoRA −12.78 → +타 전문가 −7.25 → +분류기 −5.71 → +MEPF +1.5 → +Squad +2.8(단일 섭동 상대 +2.8%, 혼합 최대 +44.4%); NYUD +9.7% | 감독 분류기 | 본문 확인 |
| CAFuser (2410.10791, RA-L 2025) | MUSES/DELIVER 융합 | RGB 최상위 특징 → 2enc/2dec Transformer → Condition Token | MUSES 조건 라벨을 문장으로 바꿔 **언어-시각 대조 손실** | 전문가 없음; CT가 모달별 가중치(CAA) 또는 cross-attn 쿼리(CA2) 생성 | MUSES PQ: 공유 백본 58.4 → +어댑터 59.3 → +CAF 59.7; 조건 손실 제거 시 59.3(−0.4) | 미분리 보고 | 해당 없음 | 본문 확인 |
| **MaMOL** (2511.11460) | RS 결측 모달 분류(HS+LiDAR 등) | [토큰 z ; ψ(결측 지시자 m∈{0,1}^M)] → TopK softmax | 라벨 없음(가용성 지시자만) | 정적: 공유 1 + 모달별 M(가용 시 활성) / 동적 K_d=2, top-1; 마지막 6층 | Houston2013 50% 결측: 98.40 OA vs 최고 기준선 97.60 | 위와 동일(결측률별 일관 개선) | top-K 희소화만 기술; 로드밸런스 없음 | 본문 확인 |
| QA-MoE (2604.05704) | 멀티모달 감성 | 모달별 가우시안 분산(우연 불확실성) → 품질 r_m | 자기지도(heteroscedastic 회귀), 라벨 없음 | GLU 전문가 8, top-3 + 학습 가능한 prior fallback | MOSI ACC7 53.6 vs 46.9 | 노이즈 λ=0.7 ACC2 87.7 vs 87.3; 결측 70% ACC7 30.5 vs 26.4; 품질 게이트 제거 시 MAE +0.359 | prior fallback로 보간 | 본문 확인 |
| MISS / FPT (2401.16923) | DELIVER 모달 결손 세그 | 라우터 없음; 단일 프롬프트 집합 + 주파수 정보 | 결측 비트 무작위 스위치(MMS) | 프롬프트 1.07M(1.1%) | DELIVER 전모달 57.81 vs AdaptFormer 55.84 | RGB 결측 29.36 vs 26.54; Depth 결측 50.73 vs 47.89 | 해당 없음 | 본문 확인 |
| MMP (2303.03369, CVPR 2023) | VL 인식 결측 | 결측 케이스별 프롬프트 선택 | 결측 케이스 | 프롬프트 <1% 파라미터 | 미확인 | 결측 케이스 전반 개선 주장 | 해당 없음 | 초록만 |
| MCULoRA (2507.11202) | 결측 감정 인식 | 모달 조합별 LoRA(공유+조합 특이 분리) | 조합 라벨(가용성) | 조합별 LoRA + 공유 | 미확인 | 기준선 대비 우위 주장(수치 미추출) | 조합별 학습 비율 동적 조정 | 초록만 |
| Deweather-MoE (2312.16610) | 날씨 제거 | 불확실성 인지 라우터 | 미확인 | **단일 공유 전문가 + FiLM 활성 변조로 가상 전문가** | 미보고 | MoE 대비 +0.1~0.2 dB, 파라미터 −72%, 추론 −39% | 가중치 공유로 붕괴 문제 회피 | 초록만 |
| MAGIC++ (2412.16876) | 임의 모달 세그 | 특징-평균 코사인 순위(내용 기반, 라벨 없음) | 없음 | 전문가 없음(계층적 모달 선택) | DELIVER RDEL 61.67 (MAGIC 63.40 보다 **낮음**) | 임의 조합 평균 47.74 vs MAGIC 40.49 | 해당 없음 | 본문 확인 |
| EQUISeg (2509.24505) | DELIVER/MUSES 강건 세그 | **무작위 교사-학생 쌍** | 없음 | 없음 | DELIVER 67.90(StitchFusion 68.20) | EMM 평균 48.22 vs MAGIC++ ≈45.66; 코사인 유사도 쌍으로 바꾸면 Event/LiDAR 22.66/24.91로 하락 | 해당 없음 | 본문 확인 |

표 1이 말해 주는 것(근거 뒤 의견): 열화 조건부 라우팅이 성공한 사례는 예외 없이 **라우터에 라벨(열화 유형·날씨·결측 지시자)이 직접 들어간다**. AW-MoE 표 VII의 무감독 포인트 특징 라우팅(Fog AP 11.4) vs 감독 이미지 라우팅(95.3), EQUISeg의 코사인 쌍 < 무작위 쌍, MAGIC++의 내용 기반 선택이 clean에서 MAGIC보다 낮은 것은 우리 계보의 실측("no-GT 내용 라우팅 전부 음수")과 방향이 같다. 반면 **clean 이득은 어느 논문도 라우팅 자체에서 유의하게 얻지 못했다**(CAFuser 조건 손실 +0.4 PQ, RMMSS "−0.1 mIoU", MoR는 LoRA+MoE 대비 +0.028 CLIPIQA).

### 표 2. 공유 전문가 + 잔차 구조와 학습 일정

| 논문 | 구조 | 전문가 켜는 시점·일정 | 초기화(upcycling) | 라우터 온도·LR | 결과 수치 | 확인 수준 |
|---|---|---|---|---|---|---|
| DeepSeekMoE (2401.06066) | 공유 전문가 항상 활성(라우팅 없음) + 라우팅 전문가 | 처음부터 | 처음부터 학습 | expert-level balance α=0.001 | 공유 1개 제거 시 Pile loss 1.808→2.414; 공유 1/2/4개 = 1.808/1.806/1.811 | 본문 확인 |
| HydraLoRA (2404.19245) | **공유 A + B 헤드 N개**, 토큰 softmax 라우터 | 처음부터 | 없음 | 미기술 | LLaMA2-7B: LoRA r32(0.248%) MMLU 46.59 vs Hydra r8(0.124%) 47.22; BBH: LoRA 36.8 / LoRA-MoE(48/48) 40.3 / Hydra(1/10) 41.5; B 헤드 2~4는 둔감 | 본문 확인 |
| ASE (2510.00570) | 공유 전문가에 **라우터 게이트를 주고 희소 전문가와 함께 정규화** | 처음부터, 40 epoch | 없음 | Mod-Squad 손실 | PASCAL-Context SemSeg: LoRA-MoE(16/4/0/4) 73.7, Δm +6.06 → ASE(16/3/1/4) 74.0, Δm +7.49; **고정(DeepSeek식) 공유 전문가는 불균형·그래디언트 충돌로 성능 하락**(그림 2) | 본문 확인 |
| MTLoRA (2403.20320) | 태스크 공유 LoRA(스테이지 내부) + 태스크 특이 LoRA(스테이지 마지막 블록, r=4) | 처음부터 | 없음 | 해당 없음 | PASCAL-Context Δm: Full FT +2.23(30.06M) / LoRA −2.17(2.87M) / VPT-deep −10.85 / MTLoRA r32 +2.16(6.08M), r64 +2.55(8.34M) | 본문 확인 |
| C-LoRAE (2505.06303, IJCAI 2025) | 범용 전문가(r) + 태스크 전문가(r/N), 토큰 게이트 h=g1·U+g2·D | 처음부터, 일정 없음 | 없음 | 상호정보 최대화 | 범용만 −27.6pt; 게이트 대신 단순 합은 "실질적 열화" | 본문 확인 |
| MaMOL (2511.11460) | 정적(공유 1 + 모달별 M) + 동적 2, top-1 | 처음부터; **선형 warm-up 10% 후 선형 감쇠**, LR 2e-3 | 없음 | 미기술 | 표 1 참조 | 본문 확인 |
| MoR-DASR (2511.16024) | 공유 rank 8 고정 + 라우팅 rank 32 + zero 4 | 처음부터 | 없음 | 열화 인지 로드밸런스 | 표 1 참조 | 본문 확인 |
| LoRA-IR (2410.15385) | **2단계**: 열화 라우터 사전학습(20분, 고정 LR 2e-4) → LoRA 전문가 PEFT(1e-4, 100k iter 후 1e-5) | 라우터 먼저 고정, 전문가 나중 | 없음 | 상동 | 표 1 참조 | 본문 확인 |
| Sparse Upcycling (2212.05055) | 밀집 MLP 복제로 전문가 초기화, 라우터 무작위 | 밀집 사전학습 후 전환 | **복제 초기화가 필수, 무작위 초기화는 큰 손실** | 32 전문가가 안정 | ViT-B/16 10-shot +1% 달성에 밀집 +58% 시간 vs upcycled +13%(4.5×) | 요약 페이지 확인(PDF 파싱 실패) |
| Drop-Upcycling (2502.19261) | 복제 + 중간 차원 r 비율 재초기화 | 밀집 후 전환 | r=0.5 최적 | 미기술 | 8×1.5B: 단순 upcycling 37.4(전문가 2개만 항상 선택되는 붕괴) / 노이즈 39.1 / BTX 38.6 / Drop 40.3 / scratch 34.5 | 본문 확인 |
| When Does Sparse MoE Help in Vision (2605.15484) | 라우터 온도 스케줄 | — | — | **선형 감쇠 τmin=0.1은 피크 +0.92%p 후 후반 붕괴; sigmoid 스케줄은 붕괴 없음(+0.03%p); τmin=0.3이면 선형도 붕괴 없음**; ImageNet head-only(라우팅 FLOPs<1%) −1.29~−1.90%p, 백본 MoE ρ≈38% k=2 +1.17 / k=1 −2.08 | 본문 확인(검색 요약과 상충했으나 원문 인용으로 확정) |
| LoRAMoE (2312.09979) | LoRA 전문가 + 국소 균형 제약(LBC) | 처음부터 | 없음 | LBC | 전문가 붕괴 방지 주장(수치 미추출) | 초록만 |
| MULTIAQUA (2512.17450) | 일정 아닌 학습 전략: **이중 forward(RGB 0 치환)** + 모달별 디코더 헤드 | 매 스텝 | — | — | 야간 mIoU CMNeXt 50.52→72.24, StitchFusion 41.86→74.23; 헤드 단독은 +0.5~2; 품질 감지 모듈은 "향후 과제"로 명시 | 본문 확인 |

표 2가 말해 주는 것: (a) "공유 고정 + 잔차 라우팅"에서 공유 전문가를 **라우터 밖에 고정하면** ASE가 성능 하락을 보고했고, DeepSeekMoE는 LLM 사전학습에서 필수라고 보고했다(도메인이 다르므로 우리 세팅에서 어느 쪽인지는 실험 필요). (b) 전문가를 중반부터 켜는 일정에 대한 직접 ablation은 LoRA 문헌에서 찾지 못했다. 가장 가까운 것은 LoRA-IR의 "라우터 먼저 고정" 2단계와 upcycling 계열(복제 초기화 필수, 단 동일 복제는 붕괴 → 부분 재초기화 r=0.5). (c) 온도 스케줄은 2605.15484 하나만 정량 근거가 있으며, 우리 "학습 게이트 상수 수렴" 증상과 결이 같은 것은 τmin이 너무 낮을 때의 후반 붕괴다.

### 표 3. 동결 ViT 위 어댑터 설계 대안 (멀티모달·밀집 예측 수치)

| 논문 | 어댑터 축 | 백본 | 학습 파라미터 | clean 수치 | 비교 대상 | 확인 수준 |
|---|---|---|---|---|---|---|
| Rein (2312.04265) | 인스턴스 연결 토큰으로 층별 특징 정제(측면 토큰) | 동결 DINOv2-L | 2.99M | GTAV→Cityscapes/BDD/Mapillary 평균 **64.3** | Full FT 61.7(304M) / Freeze 61.1 / LoRA 62.7(0.79M) / AdaptFormer 62.7(3.17M) / VPT 63.3(3.69M) | 본문 확인 |
| Mona (2408.08345, CVPR 2025) | 다중 필터 conv 어댑터 + 스케일 정규화 | Swin-L | 5.08M(2.56%) | ADE20K **51.36** | Full FT 51.18 / LoRA 50.34(4.57M) / AdaptFormer 50.83 / Adapter 50.78; COCO box 53.4 vs Full 52.4 | 본문 확인 |
| MTLoRA (2403.20320) | 공유+태스크 LoRA (Q/K/V·proj·MLP 전부) | Swin | 6.08~8.34M | 표 2 참조; **LoRA 단독 Δm −2.17, VPT −10.85** | Polyhistor +2.34(8.96M) | 본문 확인 |
| DPLNet (2312.00360, IROS 2024) | 동결 RGB MiT-B5 + 멀티모달 프롬프트 생성기(MPG) + 특징 어댑터(MFA) | MiT-B5 동결 | 백본측 3.88M(4.4%) + 디코더 3.27M | NYUDv2 58.3(ms 59.3) / SUN 52.1 / MFNet 59.3 / PST900 86.7 | Full FT NYUDv2 58.1(88.58M); MPG 또는 MFA 제거 −0.9 | 본문 확인 |
| MISS/FPT (2401.16923) | AdaptFormer 병목에 Fourier 프롬프트 | 동결 | 1.07M(1.1%) | DELIVER 57.81 | AdaptFormer 55.84(1.19M) | 본문 확인 |
| MLE-SAM (2412.04220) | **모달별 LoRA 전문가(r=32, Q/V) + 모달 평균 임베딩 softmax 라우터(top-k)** | 동결 SAM | ≈20.8M | DELIVER 64.08 / MUSES 74.80 / MCubeS 51.02 (자체 프로토콜, CMNeXt 대비 +4.90/+28.14/+14.86) | 가중 특징만 58.35 / 통합만 61.87 / 둘+보조 헤드 64.08; RGB-only 50.28, Event-only 0.74 | 본문 확인 |
| CAFuser (2410.10791) | 공유 백본(비동결) + 모달·레벨별 2층 MLP 어댑터 16개 + α 잔차 | ImageNet 초기화, 전체 학습 | 149.0M→77.7M(−54%) | MUSES PQ 59.7 / DELIVER test 55.6 | 어댑터 없음 58.4 → 있음 59.3 | 본문 확인 |
| DINOv3 (2508.10104) | injector 제거 ViT-Adapter + M2F | 동결 ViT-7B | 디코더 927M | ADE20K 63.0 | 미동결 비교 없음 | 검색 요약 |
| ViPT (2303.10826) | 모달 보완 프롬프트 | 동결 RGB 트래커 | <1% | RGB-D/T/E에서 Full FT 이상 주장 | 수치 미추출 | 초록만 |
| MM-SAM (2408.09085) | 모달별 패치 임베딩 + LoRA | 동결 SAM | 미확인 | 미추출 | — | 초록만 |
| MoBaNet (2603.17705) | 공유 프롬프트 주입 병목 어댑터 + 모달 조건 무작위 마스킹 | 대부분 동결 VFM | 미추출 | Vaihingen/Potsdam SOTA 주장 | — | 초록만 |
| CLFSam / SSCL (ISPRS 2026) | **모달 특이 + 모달 공유 LoRA 병렬 가지(역할 비대칭)** | SAM2-Hiera | 미확인 | 미추출(유료) | — | 초록만 |
| META (2502.01962, ICLR 2025) | 메모리 효율 ViT-Adapter | — | — | 수치 미추출 | — | 초록만 |

표 3이 말해 주는 것: 동결 VFM 밀집 예측에서 **단순 LoRA는 프롬프트·측면 토큰(Rein, VPT)이나 conv 어댑터(Mona)에 뒤지는 사례가 반복된다**(Rein 표: LoRA 62.7 < VPT 63.3 < Rein 64.3; MTLoRA 표: LoRA Δm −2.17). "공유 + 특이 LoRA"는 MTLoRA·SSCL·HydraLoRA·MaMOL·C-LoRAE 다섯 곳에서 이미 쓰였으므로 그 자체는 노벨티가 아니다. 노벨티 여지는 (i) 잔차 가지를 **품질·열화 상태에 조건화**하는 것(표 1에서 세그멘테이션·동결 VFM·멀티센서에 이를 결합한 사례 없음), (ii) 잔차의 형태를 LoRA가 아니라 Rein식 토큰이나 FiLM 변조로 두는 것(Deweather-MoE)이다.

### 표 4. 노벨티 판정: "공유 전문가 고정 + 열화 상태로 켜지는 모달별 잔차 전문가 + 합성 열화 라벨 감독 라우터"

| 순위 | 가장 가까운 선행 | 같은 점 | 다른 점(우리 제안이 새로 갖는 것) |
|---|---|---|---|
| 1 | **RobuMTL** (2601.10921, WACV 2026) | 동결 백본, 섭동 유형 분류기를 **섭동 라벨로 별도 지도학습**, 섭동별 LoRA 전문가 + clean 전문가, top-k 집계 | RGB 단일 모달·멀티태스크(세그 포함)이고 **모달별 잔차가 없다**; 공유 전문가 고정 개념 없음(clean LoRA가 사실상 그 역할); 전문가가 백본 전 층 계층 LoRA(r 16~512)로 크다 |
| 2 | **MaMOL** (2511.11460) | 공유 정적 전문가 + 모달별 정적 전문가 + 동적 전문가, 결측 지시자로 조건화, 마지막 6층에만 삽입 | 조건이 **이진 가용성**이지 품질·열화 강도가 아니고 라벨 감독이 없다; RS 분류(세그 아님); 동적 라우터 입력에 토큰 내용이 섞여 있다 |
| 3 | **LoRA-IR** (2410.15385) / DAME-Net (2604.09313) | 열화 라벨 CE(또는 multi-label BCE)로 감독한 라우터 → LoRA/전문가; 부재 열화 하드 마스크 | 복원 과제, 단일 모달, 동결 VFM 아님, 모달별 잔차 없음 |
| 보조 | AW-MoE (2603.16261) | 멀티모달(카메라+LiDAR) 3D 검출, 날씨 분류기 감독 라우팅, 무감독 라우팅 대비 압도적 우위 실측 | 전문가가 검출 헤드 수준이고 LoRA 아님; 공유 전문가 없음 |
| 보조 | CAFuser (2410.10791) | 조건 토큰(라벨 대조 감독)이 모달 가중치를 조절, MUSES/DELIVER | 어댑터가 아니라 **융합 가중치**를 조절; 백본 비동결; 조건은 실제 라벨(합성 아님) |
| 보조 | MoR-DASR (2511.16024) | 공유 rank 고정 + 라우팅 rank + zero-expert | 무감독 CLIP 열화 점수, SR 과제 |

first 주장 가능성: "동결 DINOv3 멀티센서 세그멘테이션에서 **모달별 잔차 LoRA를 합성 열화 라벨로 감독된 품질 라우터로 켜는 최초**"라는 좁은 주장은 이번 조사 범위(35편)에서 반례를 찾지 못했다. 다만 구성 요소 셋(감독 라우터=RobuMTL/LoRA-IR/AW-MoE, 공유+모달 잔차=MaMOL/MTLoRA/SSCL, 부재 마스크·zero-expert=DAME-Net/MoR)은 각각 선행이 있으므로 리뷰어가 "조합"으로 볼 위험이 크다. 주장 문구는 "first"보다 "라우터 감독 신호를 외부 라벨 없이 합성 열화로 대체해 세그멘테이션 clean 성능을 잃지 않으면서(RMMSS −0.1 수준) 열화 조건 mIoU를 올린다"는 **실측 주장**으로 두는 편이 안전하다. 미확인 위험: MAIL(OpenReview caMnGCyONx, "Mixture of LoRA Experts for Adaptive Incomplete Multimodal Learning")은 접근 차단으로 내용을 못 읽었다. 제목만으로는 순위 2에 근접할 수 있다.

### 빈 셀 목록 (못 채운 것)
- AW-MoE normal(clean) 조건 AP, WM-MoE·LoRA-IR·RobuMTL·MaMOL의 clean 대비 손실 여부, MaMOL LoRA rank·파라미터 수.
- Sparse Upcycling 전환 직후 성능 하락 폭·회복 시간(PDF 파싱 실패, 요약 페이지 기준).
- MAIL(OpenReview) 전체, SSCL/CLFSam(ISPRS 유료)·MFESeg(Springer) 수치, MM-SAM·ViPT·META·MoBaNet 수치, MCULoRA 결측률별 수치, DA-CLIP 수치.
- "전문가를 중반부터 켜는" 일정의 직접 ablation: 어느 논문에서도 못 찾음.
- 앞선 조사의 SMEAR/MoCLE/ST-MoE 수치는 이번에 재검증하지 않음.

### 우리 제약(동결 DINOv3-L, 24~40GB, 외부 신호 불사용)에서 실행 가능한 설계 2개

**설계 A: 합성 열화 라벨 감독 라우터 + 하드 마스크 잔차 LoRA (RobuMTL×MaMOL×DAME-Net 결합).** 지금 완주 직전인 E-LoRA 3갈래 중 "공유 r8 + 센서별 잔차 r8" 가지를 그대로 공유 전문가(고정)로 두고, 센서별 잔차 r8을 열화 상태별(예: clean / 노이즈 / 블러·저조도 / 결측 = 3~4개)로 복제해 잔차 전문가 은행을 만든다(4 모달 × 3 상태 × r8이면 LoRA 파라미터는 현재의 3배 수준으로 24GB에서 문제 없다). 라우터는 각 모달의 동결 DINOv3 CLS/평균 토큰만 입력으로 받는 2층 MLP이며, 학습 중 우리가 넣는 합성 열화(2503.18445의 NM/EMM 프로토콜 그대로)의 **유형 라벨로 CE 감독**하고 결측 모달은 DAME-Net식 하드 마스크로 전문가를 배제한다(zero-expert 슬롯 1개 포함). 추론은 라벨 없이 라우터 예측만 쓴다. 기대 근거: 감독 라우터가 무감독 대비 압도적이라는 실측(AW-MoE 표 VII 11.4→95.3 AP; 우리 계보의 no-GT 라우팅 전부 음수와 일치), 동결 백본 + 섭동별 LoRA로 Δm −12.78→+2.8(RobuMTL), 결측 지시자 조건화로 +0.8 OA(MaMOL). 예상: clean 이득은 0에 가깝고(문헌 전부 ±0.4 이내), 열화 조건(NM·RMM) mIoU에서 이득을 보는 설계다. 사전 등록할 게이트는 "clean 3시드 평균 −0.2 이내 + NM 평균 +1.0 이상"이 문헌상 현실적이다.

**설계 B: 조건 토큰이 잔차 LoRA의 스케일을 연속 변조(CAFuser×Deweather-MoE, 이산 라우팅 없음).** 전문가를 늘리지 않고 센서별 잔차 r8을 하나만 두되, 잔차 출력의 스케일 α_m(모달별 스칼라 또는 rank별 벡터)을 모달별 조건 토큰에서 FiLM 방식으로 생성한다. 조건 토큰은 동결 DINOv3 특징에서 작은 Transformer(CAFuser는 2enc/2dec)로 뽑고, 합성 열화 라벨(유형×강도 2~3단계)로 CE 또는 CAFuser식 대조 손실을 건다. 이산 라우터가 없으므로 우리 계보의 "게이트 상수 수렴"·후반 붕괴(2605.15484)가 구조적으로 생기지 않고, 메모리 추가는 조건 헤드 수 M 개뿐이다. 기대 근거: CAFuser 조건 손실 +0.4 PQ(MUSES), Deweather-MoE가 FiLM 가상 전문가로 MoE 대비 +0.1~0.2 dB·파라미터 −72%, QA-MoE의 품질 게이트 제거 시 MAE +0.359. 예상 이득은 A보다 작지만(문헌 +0.4 PQ 수준) 실패 위험과 구현 비용이 가장 낮아, A의 대조군으로 같은 시드에서 함께 돌리는 것이 판정에 유리하다.

## 막힘 / 판단 필요
- MAIL(OpenReview)과 SSCL(ISPRS)은 접근이 막혀 노벨티 순위 1~3에 끼어들 가능성을 배제하지 못했다. 필요하면 브라우저 도구나 도서관 접근으로 재확인이 필요하다.
- 설계 A/B 모두 문헌상 clean 이득 근거는 없다. "MoE-LoRA에서 노벨티"를 요구하는 user 조건은 열화·결측 축에서만 충족 가능하다는 것이 이번 조사의 결론이다.

## 다음
- E-LoRA 3갈래 3시드 결과가 나오면, "공유 r8 + 잔차 r8" 가지가 clean에서 손해가 없을 때만 설계 A/B를 그 위에 얹는 순서가 문헌(RMMSS −0.1, ASE의 고정 공유 전문가 하락)과 정합한다.

주요 출처: 2410.15385, 2511.16024, 2604.09313, 2310.01018, 2303.13739, 2603.16261, 2601.10921, 2410.10791, 2511.11460, 2604.05704, 2401.16923, 2303.03369, 2507.11202, 2312.16610, 2412.16876, 2509.24505, 2401.06066, 2404.19245, 2510.00570, 2403.20320, 2505.06303, 2212.05055, 2502.19261, 2605.15484, 2312.09979, 2512.17450, 2312.04265, 2408.08345, 2312.00360, 2412.04220, 2508.10104, 2303.10826, 2408.09085, 2603.17705, 2502.01962, 2503.18445, 2505.12861, 2605.20372, 2603.02505.

---

## 2026-09-20 딥리서치 A — 품질 인지·오염 방지 융합 attention + 강건 벤치 SOTA(에이전트 원문, 30여 편)

## 한 일
- 관련연구 조사를 WebSearch·WebFetch·alphaXiv PDF 조회로 수행하고 1차 출처(arXiv 원문·벤치마크 README·CVPR 페이퍼노트)에서 수치를 대조했다(총 30여 편, 2023~2026).
- 기존 리포 기록(`.claude_logs/research/related-work-raw.md` 2026-09-07·09-08 절)과 중복되는 항목은 갱신된 사실만 적고, 새 항목(EQUISeg 벤치 표 전체·RAF·InfraNet·GIML·UDML·UMQ·IAF·RollingQ·CrossWeaver·CHARM·A2DINOv3·UniM2·MoGA·품질신호 진단 2606.26473)을 추가했다.
- 표 3개 + 빈 셀 목록 + 우리 제약(동결 DINOv3-L·LoRA·외부 신호 불사용)에서 실행 가능한 설계 2개를 작성했다.

## 결과

### 표 1. 품질 인지·오염 방지 융합 attention 설계 (기제 · 품질 신호 출처 · clean 이득 · 열화/결측 이득 · 우리 적합성)

수치 표기: DELIVER 는 별도 표기가 없으면 MiT-B2. "벤치" = arXiv 2503.18445 프로토콜(정규화 후 zero-fill, EMM 15조합 평균/RMM 픽셀 드롭 r/NM = S&P D + Gaussian σ). "MAGIC계" = 결측 브랜치 건너뜀 프로토콜(수치 혼용 금지).

| 논문 (ID·venue) | 기제 1줄 | 품질 신호 출처 | clean 이득 | 열화/결측 이득 | 우리 제약 적합성 |
|---|---|---|---|---|---|
| CMNeXt (2303.01480, CVPR'23) | RGB 주 브랜치 + Self-Query Hub 가 보조 모달에서 정보량 큰 feature 를 max-select | 무감독(내용 기반) | clean 66.33 | 벤치 EMM 평균 37.90, RGB 결측 시 −15.5(DE/DL/DEL), NM 저/중/고 35.23/16.37/2.31 (2503.18445 표3·7·11) | RGB 의존 구조라 "RGB 열화 오염" 문제의 반례 |
| GeminiFusion (2406.01210, ICML'24) | 픽셀 단위 intra+inter attention, layer-adaptive noise | 무감독 | clean 66.92 | EMM 37.07, NM 5.52/4.14/3.30 — 벤치 저자 해석: 과도한 모달 간 교환이 노이즈를 전파 (2503.18445 §4.2.3) | "무제한 대칭 교환 = 오염 경로" 증거. 우리 대칭 cross-attn 이 열화 모달에 −6~−13 나는 것과 같은 계열 |
| MAGIC / MAGIC++ (2407.11344 ECCV'24 / 2412.16876) | 모달 feature 와 평균 feature 의 cosine 유사도로 순위 → 상위/하위 선택(MASM), 계층적 | 무감독(유사도 랭킹) | clean 66.10 / 67.33 | 벤치 EMM 44.97/44.85, RMM(r=.5) 48.19/49.31, NM 24.03·13.31·3.98 / 33.12·18.31·8.70. MAGIC계 B0: MAGIC RDEL 63.40·Mean 40.49, MAGIC++ 61.67·47.74 | 결함 지적(EQUISeg 2509.24505 §1): 고품질 모달(RGB·Depth)은 열화돼도 유사도가 높게 유지돼 선택이 오작동 → 내용 기반 신호의 한계(우리 계보 "no-GT 신호 전부 음수"와 정합) |
| StitchFusion (2408.01343, ACM MM'25) | 모달 동등 인코딩 + 밀도별 Modality Adapter 교환 | 없음 | clean 68.20(벤치 최고) | EMM 41.98, RMM(r=.5) 48.33, NM 24.91/17.11/9.25 | 품질 처리 없음 → clean 1위·열화 중위 |
| EQUISeg (2509.24505, 2025-09) | CMTB: 주 모달 self-attn → SQ-Hub 로 보조 통합 → 주 모달 query, 보조 key/value cross-attn + residual. SGM: 프로토타입을 랜덤 teacher/student 로 KL 증류(품질 선택 자체를 회피) | 무감독(랜덤 배정) | clean 67.90 (StitchFusion −0.30) | 벤치 EMM 48.22 / E(p=.1) 65.75, RMM 50.96 / 64.64 (r 미명시; 베이스라인 값이 벤치 표9 r=0.5 와 일치하므로 r=0.5 추정), NM 저/중 34.87/19.13, 강건성 점수 50.21 (MAGIC++ +2.56). ablation(B0): cross-attn 제거 clean −1.34 / EMM −5.41, SQ-Hub→mean EMM −1.94, cosine 페어링으로 바꾸면 EMM −1.66 | self-attn(모달 내)→cross-attn(모달 간) 순서의 근거 수치. SGM 은 우리 gated-MLP 트렁크에 손실만 추가하면 이식 가능. 단 B2 전체 학습이라 동결 백본 이식 이득은 미확인 |
| CrossWeaver (2604.02948, 2026-04) | MIB: 모달 수준 softmax 신뢰도 + 토큰 수준 soft top-p mask + cosine 일관성 필터 + 토큰 mixer | 무감독(학습형 임계) | B0 clean 64.69 | MAGIC계 Mean 40.52 (StitchFusion 36.72, CMNeXt 20.77). NM 미보고 | "토큰 수준 key 선택" 선례. 감독 없음·NM 없음이 차이 |
| CHARM (2508.03060, 2025-08) | 윈도우 cross-modal MPU + fragile-modality-biased sampling(SoftMin) | 무감독 | B2 top-1 조합 68.43 | MAGIC계 Mean 54.96 (Any2Seg 45.04 +9.92), last-1 28.76 | 약한 모달 편향 샘플링은 우리 학습 레시피에 이식 가능 |
| Any2Seg (2407.11351, ECCV'24) | cosine 상관맵을 feature 에 곱 + max-sim 선택, LanguageBind 증류 | 무감독 + 외부 VLM | B2 RDEL 68.25 (CMNeXt +1.95) | MAGIC계 Mean 45.04 (CMNeXt 25.25) | 외부 텍스트 교사 → 우리 제약 밖 |
| RMMSS→RobustSeg (2505.12861 / CVPR'26) | 완전모달 teacher + 프로토타입 증류(HPD), FSM = depthwise-conv+sigmoid 로 두 teacher 출력 max-select, 피드백 루프(비주 모달 인코더 lr 6e-6 미세조정) | 무감독(학습형 점수) | B0 full 61.85 (AnySeg 59.41 +2.44); 자체 teacher 대비 −0.1 | MAGIC계 Mean 49.91(CVPR본, 페이퍼노트) / 49.93(arXiv), +2.40; EL 33.01 | 학습 시에만 작동하는 증류라 추론 구조 불변 → 동결 백본에도 적용 가능. DELIVER 결측 직접 비교 대상 |
| AnySeg (2411.17141) | uni/cross-modal 증류 + 최소-1-모달 랜덤 마스킹 | 없음 | B0 RDEL 60.26 | Mean 46.64 (MAGIC +6.15); 드롭 확률 미명시 | 학습 레시피 참고 |
| FunEntropy (2505.06635, ICCV'25) | 함수 엔트로피 정규화(파라미터 0) | 없음 | — | B0 Mean 48.29 (MAGIC 40.49) | plug-and-play 손실, 이식 가능 |
| BiXFormer (2506.03675) | 모달 무관 쿼리 매칭 + 보완 매칭, VAE 정렬 | 없음 | B0 RDEL 58.29 | Mean 43.24 (+2.75 vs MAGIC) | Mask2Former 계열 헤드 참고 |
| CAFuser (2410.10791, RA-L'25) | RGB 최상위 feature → Transformer 로 Condition Token; CAA = softmax 모달 가중, **CA² = CT 를 RGB 토큰에 concat 한 조건부 query 로 cross-attn** | **감독**: MUSES 조건 라벨(날씨·조도 등)을 텍스트 프롬프트로 만든 verbo-visual contrastive | MUSES 78.2 mIoU(CA²) | 모달 드롭 p=0.2 학습; 결측 ablation PQ 55.7(RGB)→59.7(4모달); 열화 평가 없음 | "query 조건화" 선례. 외부 텍스트·데이터셋 조건 라벨 사용이라 우리 제약 밖 — 조건 라벨을 합성 열화 라벨로 바꾸면 우리 설계 B |
| IAF (2608.26879, 2026-08) | 지배 모달은 cross-attn 을 통째로 우회(pure path), 약한 모달만 지배 모달을 anchor 로 attend | 없음(고정 비대칭) | MultiHuSE 78.06 (단일모달 74.86 +3.20), UR-FUNNY 70.72 | 대칭 융합이 지배 모달 경로를 74.9→56.4(−18.5pp) 붕괴시킴("strong-modality collapse"); 분류 과제 | 우리 실측 "depth 제거 −17~−33 = depth 지배"와 결합하면 depth-anchor 비대칭 설계 근거 |
| RollingQ (2506.11465, ICML'25) | 모달 간 key 분포 격차가 attention 동적성을 죽이는 자기강화 → query 회전 | 없음 | 분류 과제 | — | 대칭 attention 의 "한 모달 편중" 진단 도구 |
| AMBER (2512.11331) | 결측 모달 key 를 modality-aware mask 로 attention 에서 제외, 랜덤 불완전 입력으로 학습 | 결측 여부(관측 가능) | 빔 예측 과제 | 결측만, 열화 미처리 | KV 마스킹 선례(결측 한정) |
| Ma et al. (2204.05454, CVPR'22) | 최적 융합 전략이 데이터셋 의존 → 탐색 | — | 3개 벤치 | 결측 | "결측 강건 융합 전략은 보편해가 없다" 근거 |
| Lee et al. missing-aware prompts (CVPR'23) | 동결 ViLT 에 결측 패턴별 프롬프트, <1% 파라미터 | 결측 여부 | — | 결측 | 동결 백본 선례(분류) |
| RAF (2607.04587, ECCV'26) | 픽셀별 reliability R 로 카메라 feature 를 곱셈 게이팅 F⊙(α+(1−α)R), CALM | **약감독**: 이미지 수준 clean/noisy/mixed 라벨 → 동결 LiDAR-Radar 특징과의 교차모달 유사도를 픽셀 pseudo-label 로 | K-Radar L4DR clean 80.4 (베이스 77.0) | mixed 82.1 (78.4), noisy 59.0 (=59.0), 총 +4.5 AP_BEV; 3D-LRF +6.5. clean 에서 R≈1(항등) 관찰 | "존재하되 열화" + 감독 + 항등 출현의 최근접(3D 검출). 픽셀 라벨을 다른 센서와의 일치로 만든 점이 차이 |
| InfraNet QualGate (2607.03795, ECCV'26) | q=σ(MLP(GAP(RGB feat))) 로 RGB 가이던스 억제(w=q)·IR 증폭 α(q)=clip(1.5−q) | **과제 손실만**(품질 라벨 없음, 저자가 "지각 품질 추정기 아님" 명시) | LLVIP 70.5 (IR-only 68.8) | 최대 안개에서 97.1 vs naive(q≡1) 92.6 AP50 | 과제 손실만으로 학습한 스칼라 게이트가 열화에서 작동한 사례 — 단 우리 계보의 "학습 게이트 상수 수렴"과 충돌하므로 감독 여부가 갈림길 |
| ReliFusion (2502.01856) | 모달별 스칼라 confidence(sigmoid) 를 cross-attn **출력**에 곱(CW-MCA) | 대조학습(clean=positive, corrupted=negative) | nuScenes 70.6 mAP | 제한 FOV 52.4 (BEVFusion 46.4) | 출력 스케일 ≠ key 마스크 |
| GIML (2607.06943, ECCV'26) | Noise-aware Quality Estimator: 주입 노이즈 강도 η 를 MSE 로 직접 회귀, 역분산 가중 w∈[0,1] (clean w=1, 결측 w=0 으로 연속화), Noise-Semantic Decoupled(가우시안 μ/σ) | **감독: 합성 열화 강도 라벨** | CREMA-D clean 73.94 (T2DR 67.93) | 심한 열화(0.3,0.7) 62.93 (59.89); 50% clean+50% corrupted 학습 | "합성 열화 라벨 감독 품질 추정 + clean 항등" 의 최근접. 모달별 스칼라·분류 과제·가중합(attention 아님) |
| UDML (2603.19681) | 분산항에서 노이즈 강도 회귀(2단계, 인코더 grad 차단) + 모달 의존도 재보정(dual suppression 방지) | 감독(주입 σ) | CREMA-D clean 70.02 (68.56) | 노이즈 67.53 (60.32) | 동일 계열; "불확실성 가중이 약한 모달을 이중 억제" 경고는 우리 event/LiDAR 에 해당 |
| UMQ (2603.02695) | sigmoid 품질 추정기 → MQ-MoE 라우팅 + 품질 향상기 | 감독: Gaussian 노이즈 인스턴스=0, 저손실 단일모달>0.95, 중간은 순위 제약 | MOSI 90.1 (88.5) | 결측 74.8 / 노이즈 87.3 평균 | 순위 기반 라벨링은 우리 합성 라벨 설계에 차용 가능 |
| QA-MoE (2604.05704) | 자기지도 aleatoric 불확실성으로 expert 라우팅, 결측·열화를 연속 스펙트럼으로 | 무감독 | 감정 분석 | — | — |
| CoReFuse-Med (2609.10261, 2026-09) | F-CNR(병변/배경 대비) 로 손상 정량, 얕은 층 게이팅 + 깊은 층 대칭 cross-attn(공유 consensus) | 무감독 | 의료 DSC +3~4% | 해상도 열화 | 의료 |
| MDQF (2601.08458) | 모달별 DETR 브랜치, proposal confidence top-k 로 저품질 쿼리 배제, ViT-tiny 동결 | 검출 confidence | FLIR 43.8 mAP | RGB 소실 시 −18.2 | 동결 백본 선례(검출) |
| 품질신호 진단 (2606.26473, 2026-06) | 테스트 샘플 간 reliability 점수를 **치환**해도 성능이 안 떨어지면 모델이 신호를 안 쓰는 것 | — | StressID·MOSEI 에서 치환 무손실 관측 | — | 리뷰어 방어용 필수 실험: 우리 품질 토큰이 실제로 쓰이는지 |
| SAM2 memory 계열: MemorySAM (2503.06700) / MA-SAM2 (MICCAI'25) / DAM4SAM (2411.17576) / SENTRY (2606.24449, ECCV'26) / MoGA (2605.12006, CVPR'26) / Efficient-SAM2 (2602.08224, ICLR'26) | MemorySAM = 모달을 프레임으로, 신뢰도 없음(벤치 표1: RMM✗ EMM✗ NM✗). MA-SAM2 = 마스크 품질로 memory 선택. DAM4SAM = introspection 갱신. SENTRY = 쓰기 전 검증. MoGA = 시간가변 합성 손상으로 학습한 memory-object 조건 gated-rank 적응. | 마스크 품질/시간 일관성(모두 시간축) | — | MoGA: YouTube-VOS-C 78.7→79.9 | "첫 입력(RGB) 열화가 memory 를 오염" 을 모달 축에서 다룬 논문 = 0건(검색 범위). 우리 현 구조는 SAM2 memory 를 쓰지 않으므로 인용용 |
| 동결 DINOv3 계열: UniM2 (2607.12372, ECCV'26) / A2DINOv3 (2608.21099) / RobustMedSAM (2604.09814) | UniM2: 동결 DINOv3 로 비지도 멀티모달 세그, CMH 가 비신뢰 보조 관계 감독 억제. A2DINOv3: 공유 DINOv3 + 저랭크 zero-init 통신 경로, 품질 추정 없음(동결 40.27 vs 전체 FT 43.05 mAP). RobustMedSAM: 인코더 동결, 디코더만 12종 손상으로 학습 | 무감독 / 없음 / 합성 손상 학습 | UniM2 NYU +6.4, MFNet +9.8 | RobustMedSAM Dice 0.613→0.719 | "동결 파운데이션 + 합성 손상 학습" 이 열화에 큰 이득(+0.106 Dice)을 낸 선례 |

### 표 2. 결측·열화 강건 세그 벤치 SOTA (DELIVER)

(a) 벤치마크 프로토콜 (2503.18445, MiT-B2, zero-fill) — 출처: 2503.18445 표3·7~11, EQUISeg 2509.24505 표1

| 방법 | clean RDEL | EMM 평균 | EMM E(p=.1) | RMM 평균 r=.75 / .5 / .25 | NM 저/중/고 | 열화 이득 기제 | clean 손실 |
|---|---|---|---|---|---|---|---|
| CMNeXt | 66.33 | 37.90 | 60.41 | 42.17 / 47.49 / 53.61 | **35.23** / 16.37 / 2.31 | 없음(RGB 주도) | — |
| GeminiFusion | 66.92 | 37.07 | 60.62 | 39.78 / 42.41 / 49.74 | 5.52 / 4.14 / 3.30 | 없음 | — |
| MAGIC | 66.10 | 44.97 | 62.68 | 45.30 / 48.19 / 51.61 | 24.03 / 13.31 / 3.98 | 유사도 랭킹 선택 | StitchFusion −2.10 |
| MAGIC++ | 67.33 | 44.85 | 63.52 | 47.06 / 49.31 / 53.92 | 33.12 / 18.31 / **8.70**(고 노이즈 2위) | 계층 선택 | −0.87 |
| StitchFusion | **68.20** | 41.98 | 63.29 | 45.16 / 48.33 / 53.10 | 24.91 / 17.11 / **9.25** | 없음 | 0 |
| EQUISeg (2509.24505) | 67.90 | **48.22** | **65.75** | 50.96 (r 미명시, r=.5 추정) | 34.87 / **19.13** / 미보고 | CMTB cross-attn(EMM −5.41 ablation) + 랜덤 teacher SGM(EMM +3.23) | −0.30 |
| 우리(ReliaDINO E1, 참고) | val 69.51(3시드 v2) | 미측정 | — | 픽셀 50% 드롭 시 −6~−13(사용자 실측) | 미측정 | — | — |

관찰: ① 벤치 프로토콜에서 EMM 평균 최고는 48.22(EQUISeg), RMM r=.5 최고 ≈50.96, **NM 고 노이즈는 전 모델 ≤9.25 로 붕괴** — 2025-09 이후 NM 개선을 보고한 DELIVER 논문은 EQUISeg(중 19.13) 뿐. ② depth 단독 50.6~58.4 vs RGB 단독 15.9~42.7 (표3) → DELIVER 는 depth 지배 데이터셋이며 우리 "depth 제거 −17~−33" 과 정합. ③ Event·LiDAR 는 벤치 저자도 "잉여"로 결론(§4.3) → 우리 ±0.5 관찰과 정합.

(b) MAGIC계 프로토콜 (결측 브랜치 건너뜀, 15조합 Mean) — 혼용 금지

| 방법 | 백본 | RDEL | Mean(15) | 출처 |
|---|---|---|---|---|
| CMNeXt | B0 | 39.07(MAGIC++ 표) / 59.18(AnySeg 표) — 재학습 차이, 미확인 | 20.77 | 2412.16876, 2411.17141 |
| CMNeXt | B2 | 66.30 | 25.25 | 2407.11351 |
| MAGIC (2407.11344) | B0 | 63.40 | 40.49 | 2412.16876 |
| MAGIC++ (2412.16876) | B0 | 61.67 | 47.74 | 원문 |
| Any2Seg (2407.11351) | B2 | 68.25 | 45.04 | 원문 |
| AnySeg (2411.17141) | B0 | 60.26 | 46.64 | 원문 |
| FunEntropy (2505.06635) | B0 | — | 48.29 | 원문 |
| BiXFormer (2506.03675) | B0 | 58.29 | 43.24 | 원문 |
| RMMSS/RobustSeg (2505.12861/CVPR'26) | B0 | 61.85 | 49.91~49.93 | 원문·페이퍼노트 |
| CHARM (2508.03060) | B2 | top-1 68.43 | **54.96** | 원문 |
| CrossWeaver (2604.02948) | B0 | 64.69 | 40.52 | 원문 |
| StitchFusion | B0 | — | 36.72 | CrossWeaver 표 |

ShaSpec (2307.14126, CVPR'23)·MMANet (2304.08028, CVPR'23): DELIVER 수치 **미발견**(원문은 BraTS·오디오비주얼·RGB-D 분류/세그) — 어떤 DELIVER 논문도 이 둘을 재평가하지 않았다(검색 범위). MLE-SAM (2412.04220) 은 벤치 표1에 EMM✓·NM✓ 로 기재돼 있으나 수치는 이번에 대조하지 않았다.

### 표 3. 노벨티 판정 — "합성 열화로 품질 라벨 → 모달별 품질 토큰 감독 → cross-attention key 마스킹/가중, clean 항등 제약"

| 최근접 선행 | 같은 점 | 다른 점 |
|---|---|---|
| GIML (2607.06943, ECCV'26) | 합성 노이즈 강도 η 를 라벨로 품질 추정기 직접 감독(MSE); w=1(clean)→0(결측) 연속화로 clean 항등이 구조적으로 성립; 50/50 clean·corrupted 학습 | 모달별 **스칼라**, **가중합**(attention 아님), 분류 과제(CREMA-D 등), ResNet/BERT 부분 동결. 세그·픽셀 단위·key 마스크 없음 |
| RAF (2607.04587, ECCV'26) | "존재하되 열화" 를 픽셀별 reliability 로 게이팅, 감독 학습, clean 에서 R≈1 항등이 **관찰**됨(α 보존항으로 완전 억제 방지) | 라벨이 합성 열화가 아니라 이미지 수준 clean/noisy 메타라벨 + 타 센서 유사도 pseudo-label; feature 곱셈 게이트(attention key 아님); 3D 검출 |
| UMQ (2603.02695) | Gaussian 노이즈 인스턴스=0 등 합성 품질 라벨 + 순위 제약으로 sigmoid 품질 추정기 감독 | MoE 라우팅·표현 향상기; 감정 분석; 세그·attention 없음 |
| (보조) ReliFusion 2502.01856 / UDML 2603.19681 / CAFuser 2410.10791 / AMBER 2512.11331 | 각각: corrupted 대조 감독 스칼라 → attn 출력 곱 / 주입 σ 회귀 / 감독된 조건 토큰을 query 에 concat / 결측 key 마스크 | 모두 세그 아님 또는 열화 아님 또는 key 마스크 아님 |

"first X" 판정(정직):
- **불가**: "합성 열화 라벨로 품질 추정기를 감독한 최초"(GIML·UDML·UMQ·ReliFusion 2025~26), "비신뢰 모달을 attention 에서 마스킹한 최초"(AMBER, Ma et al. 2204.05454), "clean 항등 제약 최초"(GIML 구조적, RAF 관찰).
- **가능(좁힌 형태, 2026-09-20 웹 조사 범위)**: "멀티모달 **시맨틱 세그멘테이션**(DELIVER/MUSES/MCubeS)에서 (a) 합성 열화 라벨로 감독한 **모달별·토큰별 품질 토큰**을 (b) cross-modal attention 의 **key 측 pre-softmax 마스크/가중**으로 쓰고 (c) clean 항등을 손실로 명시 제약하며 (d) 2503.18445 의 EMM/RMM/NM 세 축을 모두 보고한 첫 사례". DELIVER 강건 경쟁작(EQUISeg·RobustSeg·MAGIC++·CrossWeaver)은 전부 무감독 또는 랜덤·증류 기반이며 NM 을 다루는 것은 EQUISeg 뿐이다. 벤치 표1(2503.18445)도 인용 근거가 된다.
- 주의: 이 판정은 제목·초록·검색 기반이므로 제목에 단서 없는 논문 누락 가능성이 남는다. 또한 2606.26473 의 치환 진단(reliability 점수를 샘플 간 섞어도 성능 불변 → 신호 미사용)을 리뷰어가 요구할 가능성이 높으므로 같은 실험을 선제 포함해야 한다.

### 빈 셀 목록 (아무도 안 한 조합, 검색 범위 내)
1. 세그멘테이션 + 합성 열화 감독 품질 토큰 + attention key 마스크 (표3 전부 분류/검출).
2. 벤치 NM(노이즈) 축에서 CMNeXt 저 노이즈 35.23 을 넘는 방법 없음; 고 노이즈 ≥10 인 방법 없음; 노이즈 증강으로 학습한 뒤 NM 을 보고한 DELIVER 논문 없음.
3. RMM r=0.75(심한 픽셀 드롭) 를 보고한 2025-09 이후 방법 없음(EQUISeg 는 단일 r).
4. 동결 파운데이션 백본(DINOv2/v3·SAM2) + LoRA 를 EMM/RMM/NM 세 축으로 평가한 세그 논문 없음(MemorySAM 은 벤치 표1에 ✗✗✗).
5. "존재하되 열화"(우리 −6~−13) 를 clean 항등 제약과 함께 다룬 세그 논문 없음(RAF 는 3D 검출).
6. 토큰(공간) 단위 품질을 감독 학습해 attention 에 넣은 세그 논문 없음(CrossWeaver 토큰 soft top-p 는 무감독·NM 미보고).
7. 2606.26473 치환 진단을 세그 융합에 적용한 사례 없음.
8. 결측을 "브랜치 제외"와 "zero-fill" 두 프로토콜로 동시에 보고한 논문 없음(우리 모델은 결측 무해이므로 양쪽 보고 가능).

### 우리 제약에서 실행 가능한 설계 2개

**설계 A — 감독 품질 토큰 key-mask cross-attention (QK-mask).** 각 모달 LoRA 출력 토큰에서 학습 가능한 품질 query 1개가 모달 토큰을 attend 해 모달별 스칼라 η̂_m∈[0,1] 과 토큰별 η̂_{m,t} 를 낸다(파라미터 ≈ MLP 2층, 백본 동결 유지). 학습 시 배치의 50% 에 합성 열화(정규화 후 zero_ 전체 결측, 픽셀×채널 드롭 r∈{.25,.5,.75}, S&P D∈{.05,.1,.2}, Gaussian σ∈{.1,.2,.5}; 이벤트는 Gaussian 제외 — 벤치 정의와 동일)를 주입하고 라벨 η 를 MSE 로 회귀(GIML 방식) + UMQ 식 순위 제약(라벨 절대값 불확실 구간). 융합에서는 모달 m 의 key 로짓에 log(1−η̂_{m,t}) 를 더하고(η̂=0 이면 0 = 항등, η̂→1 이면 −∞ 마스크), clean 배치에는 L_id = mean(η̂) 로 항등을 명시 제약한다. 산술 평균 단계도 (1−η̂_m) 정규화 가중으로 바꾼다. 우리 계보에서 반증된 것과의 차이: 과거 게이트는 과제 손실만으로 학습(InfraNet 형)돼 상수로 수렴했고, 무학습 additive bias 는 내용 기반 신호였다 — 여기서는 신호가 **합성 라벨 감독**이고 clean 에서는 손실로 0 에 고정되므로 clean 성능에 영향이 없도록 설계된다. 예상 이득 근거: RAF 는 감독 reliability 게이트로 noisy 장면을 베이스라인 동률(59.0)로 유지하며 clean +3.4(80.4 vs 77.0), GIML 심한 열화 +3.0, RobustMedSAM 12종 손상 학습으로 Dice +0.106. 벤치 B2 모델들은 RMM r=.5 에서 clean 대비 −17~−19, NM 고에서 −57~−61 이고 우리 모델은 픽셀 50% 드롭에 −6~−13 이므로, 보수적 목표 = RMM r=.5 손실 절반(−3~−6), NM 고 노이즈 ≥ 15(현 SOTA 9.25 의 1.6배). 게이트 붕괴·항등 검증: 2606.26473 치환 진단 + η̂ 의 라벨 상관(r>0.8) 을 사전 등록 성공 기준으로 둔다. 위험: UDML 이 지적한 "약한 모달 이중 억제" — event/LiDAR 는 clean 에서 ±0.5 라 손실 위험이 작다.

**설계 B — depth-anchor 비대칭 융합 + 열화 조건부 query (IAF × CAFuser-CA², 조건 라벨을 합성 열화 라벨로 대체).** DELIVER 에서 depth 는 단독 50.6~58.4 로 RGB 단독 15.9~42.7 을 압도하고(2503.18445 표3) 우리 실측도 depth 제거 −17~−33 이므로 depth 가 지배 모달이다. IAF 는 대칭 cross-attn 이 지배 모달 경로를 −18.5pp 붕괴시키고 지배 모달을 우회(pure path)시키면 단일모달 상한 +3.2 를 얻었다(2608.26879). 설계: depth 토큰은 cross-attn 을 우회해 트렁크로 직행하고, RGB·event·LiDAR 만 depth 를 key/value 로 attend 한다(가중치 공유 2층 유지). RGB query 에는 설계 A 의 품질 토큰(합성 열화 라벨 감독)을 concat 해 조건부 query 로 만든다(CAFuser CA² 의 텍스트 조건 라벨을 자체 합성 라벨로 대체 → 외부 신호 불사용 제약 충족). 예상 이득 근거: EQUISeg ablation 에서 cross-attn 이 EMM +5.41 을 담당했고, IAF 는 지배 모달 보존으로 +3.2, CAFuser 는 조건 토큰 CA² 로 MUSES 78.2 를 얻었다. 단 우리 계보에서 "cross-attention 트렁크가 gated-MLP·산술평균을 못 이김"이 실측이므로 clean 기대치는 ±0.3 이내이고, 주된 목표는 RGB 열화(NM·RMM-R) 시 depth 경로 오염 차단이다 — 벤치 표11 에서 RGB 의존 모델(CMNeXt)이 고 노이즈 2.31 로 붕괴한 것이 방지 대상. 설계 A 보다 우선순위가 낮으며, A 의 η̂ 가 실제로 사용된다는 치환 진단이 통과된 뒤에만 착수한다.

공통 전제: 학습 레시피에 CAFuser/AnySeg 식 모달 드롭(p=0.2, 최소 1모달 보장) + 벤치 정의 픽셀 드롭·노이즈 주입을 정규화 후에 넣고, 평가는 2503.18445 의 `val_mm_EMM/RMM/NM.py` 정의(zero-fill, r 3단계, NM 은 Gaussian 포함 여부 명시)로 한다. RobustSeg 코드(github.com/RobustSeg/RobustSeg)와 같은 프로토콜인지 확인 후 비교한다.

불확실 — EQUISeg RMM 의 r 값(원문 미명시, r=0.5 추정); CMNeXt B0 RDEL 이 논문마다 39.07/59.18 로 갈리는 원인(재학습 차이 추정); RobustSeg CVPR 본 수치(49.91)는 페이퍼노트 2차 출처; MAGIC++ HTML 표는 열이 밀려 있어 BiXFormer·AnySeg 표와 교차해 복원; MLE-SAM·OmniSegmentor 의 결측 수치는 이번에 대조하지 않음; "first" 판정은 제목·검색 기반이라 누락 가능.

## 막힘 / 판단 필요
- 설계 A 의 성공 기준(RMM r=.5 손실 절반, NM 고 ≥15, 치환 진단 통과, clean −0.3 이내)을 카드 사전 등록 규약(§0)으로 확정할지는 생각정리 세션 판단.
- 이 조사 결과를 `.claude_logs/research/related-work-raw.md` 에 append 하고 `novelty-and-related-work.md` 비교표를 갱신할지 — 지시가 "조사"였으므로 파일은 수정하지 않았다.

## 다음
- 없음(조사 완료). 파일 반영 여부만 결정이 필요하다.

주요 출처: arXiv 2503.18445(벤치, 표3·7~11), 2509.24505(EQUISeg 표1·2·3), 2412.16876, 2407.11344, 2407.11351, 2411.17141, 2505.12861, 2505.06635, 2506.03675, 2508.03060, 2604.02948, 2410.10791, 2608.26879, 2506.11465, 2512.11331, 2204.05454, 2607.04587, 2607.03795, 2502.01856, 2607.06943, 2603.19681, 2603.02695, 2604.05704, 2609.10261, 2601.08458, 2606.26473, 2503.06700, 2411.17576, 2605.12006, 2602.08224, 2607.12372, 2608.21099, 2604.09814, 2307.14126, 2304.08028, github.com/Chenfei-Liao/Multi-Modal-Semantic-Segmentation-Robustness-Benchmark, en.papernotes.org CVPR2026 RobustSeg 노트.

---

## 2026-09-20 딥리서치 C — 모달 품질 감독 학습·증류·게이트 붕괴·평가 규약(에이전트 원문)

## 한 일
- 조사 질문 4개를 병렬 하위 조사(각각 60~70회 WebFetch/WebSearch, arXiv abs/html 본문 대조)로 수행하고, 핵심 경쟁 논문(2503.18445 벤치, RobustSeg/RMMSS 2505.12861, EQUISeg 2509.24505, MAGIC 2407.11344, LIMoE 2206.02770, AnySeg 2411.17141)은 직접 재확인함.
- 레포 실측 근거를 대조함: 게이트 상수 수렴 H1(`.claude_logs/research/hypothesis-ledger.md:16`), MIC 계열 −1.67 H15(같은 파일 :33), P50-MAP +0.74 H22(:40, 09-07 "1페어라 판정 불가 대역"으로 강등), 결측·열화 첫 실측(`.claude_logs/decisions/cards/2026-W38-verdicts.md:510~515`: RMM r=0.5 depth −12.68·RGB −3.12·event −0.04·LiDAR +0.01, 단일 depth 57.17).
- 표기: ✔ = 논문 본문에서 직접 읽은 수치·문장, 미확인 = 본문에서 못 찾음. arXiv ID 정정 2건: MISS/FPT = **2401.16923**(2401.10723 아님), ModDrop++ = **2203.04959**.

## 결과

### 표 1. 모달 품질 감독 학습 (학습 시 합성 열화 + 열화 라벨로 품질 예측기 감독)

| 논문 | 태스크·모달 | 감독 형태 | 학습 열화·일정 | clean 이득 | 열화 이득 | 예측기→융합 | 비고 |
|---|---|---|---|---|---|---|---|
| GIML/NQE 2607.06943 (arXiv 2026-07) | 감정·감성 인식(V/A/T) | 주입한 mask 비율 η를 MSE 회귀 ✔ | mask 비율 0.0~1.0(0.1 간격) 무작위, 배치 50% clean/50% 열화 ✔; 가우시안은 테스트 전용(미학습 열화) ✔; curriculum 없음 | MOSI 80.82 vs 79.07(+1.75), CREMA-D 73.94 vs 66.99 ✔ | MOSI 혼합열화 67.55 vs 65.36, CREMA-D 62.93 vs 57.23 ✔ | 예측 강도로 가중 ω^(v) ∝ Σ_{u≠v} η̂_u² ✔ | 정의에 가장 정확히 맞음. 감독손실 단독 ablation 없음 ✔ |
| VG-SAF 2608.24366 (arXiv 2026-08) | E2E 주행, 카메라+LiDAR(CARLA) | 픽셀별 열화 마스크 M으로 log-분산 감독 `logV_tgt = sg(logV_clean)+αM`, SmoothL1 ✔ | 카메라 8종·LiDAR 5종, 1~100%; **4단계 curriculum**, 2단계부터 BN 동결 ✔ | 미확인 | DS 카메라오염 42.6 vs 32.1, LiDAR오염 44.9 vs 36.4, 양쪽 38.2 vs 24.5 ✔ | 국소 게이트 exp(−ηV̄) × 모달 간 trust softmax → 특징 곱셈 ✔ | curriculum 있는 유일 사례 |
| MoME 2503.19776 (CVPR 2025) | 3D det, LiDAR+카메라 | 드롭 one-hot 라벨로 쿼리 라우터 CE ✔ | 카메라/LiDAR/무드롭 각 1/3 ✔; 노이즈 없음 | 71.2/73.6 vs BEVFusion 68.5/71.4 ✔ | Limited FOV 50.6 vs 0.2 mAP ✔; LiDAR 드롭 NDS 48.2 < 55.4(기준선보다 나쁨) ✔ | 하드 라우팅 → 전문가 디코더 3개 ✔ | 이산 라벨만 → 부분 열화 일반화 없음 |
| 카메라 신뢰도 모니터 2605.05439 (arXiv 2026-05) | 단일 카메라(융합 아님) | presence multi-hot BCE + severity SmoothL1(존재 조건부) + health SmoothL1 + 공간 마스크 BCE; 라벨 = 열화 연산자 파라미터에서 해석적 도출 ✔ | 12종, s∈[0,1] 연속, curriculum 없음 ✔ | — | 열화분류 mAP 0.891, health MAE 0.064, YOLOv8 mAP와 상관 0.95 ✔ | 미연결(경보용) | **감독 품질 헤드 설계 정본** |
| CAFuser 2410.10791 (RA-L 2025) | seg, MUSES/DELIVER | 조건 토큰을 MUSES 조건 메타라벨로 CLIP식 대조 감독 ✔ | 모달 20% 드롭(MUSES) ✔; 합성 열화 없음 | MUSES 78.2 vs CMNeXt 72.4; 조건손실 ablation PQ 59.3→59.7(+0.4) ✔ | 조건별 미확인 | 토큰→FC→softmax 모달 가중(CAA) ✔ | **외부 조건 라벨 의존** = 우리 제약 위반 |
| DGFusion 2509.09828 (RA-L 2026) | seg, MUSES/DELIVER | depth 헤드를 LiDAR 투영 depth로 감독(τ-분위 L1+edge-aware) ✔; 품질 예측기 없음 | CAFuser 프로토콜 승계 ✔ | MUSES 78.2→79.5, DELIVER CLDE test 55.6→56.7 ✔ | snow +2.57, night +1.63 PQ ✔ | Depth Token이 cross-attn 조건화 ✔ | 기하 감독(우리가 이미 허용한 +α 축) |
| AV-RelScore 2303.08536 (CVPR 2023) | AVSR | 신뢰도 점수는 task loss만(열화 라벨 없음) ✔ | 패치가림 30~50% p=0.8, 블러 p=0.3, 노이즈 p=0.3, 음성 −5~20dB ✔ | LRS3 2.8 vs 3.2 WER ✔ | LRS2 −5dB 13.36 vs 17.58 ✔ | f' = f + f⊙s ✔ | 열화 주입 + 비감독 게이트로 clean·열화 동시 이득 |
| QA-MoE 2604.05704 (arXiv 2026-04) | 감성(T/A/V) | 이분산 손실로 σ² 자기감독 ✔ | λ~U(0,1) 가우시안, η~U(0,1) 결측 연속 샘플링 ✔ | MOSI ACC7 53.6 vs 46.9 ✔ | 결측 70% 30.5 vs 26.4 ✔ | r=1/(1+σ²) 곱셈 + (1−r) fallback ✔ | one-checkpoint-for-all |
| ReliFusion 2502.01856 | 3D det | 대조(원본 vs 오염) + confidence loss; 타깃 정의 미확인 | 미확인 | +1.4 mAP ✔ | LiDAR 50% drop 53.1 vs 50.3 ✔ | C_LiDAR로 cross-attn 출력 스케일 ✔ | 감독 상세 불투명 |
| QMF 2306.02050 (ICML 2023) / PDF 2406.04802 (ICML 2024) | 분류(NYU **장면분류**, Food101 등) | QMF 로짓 에너지 비감독; PDF 정답확률 회귀 MSE ✔ | 학습 노이즈 없음 ✔ | PDF NYU 71.37 vs 69.54 ✔ | Food101 ε=5 76.03 vs 68.49(late) ✔ | late-fusion 가중 | seg 아님 |
| "When Does Quality-Aware Fusion Matter?" 2606.26473 | 진단 연구 | permutation 검정 ✔ | — | — | 자연 품질점수 셔플 Δ −0.002(효과 없음); 오염 정렬 +0.071, 정답 정렬 +0.346 ✔ | — | **품질 점수는 정답성과 정렬(감독)돼야 융합에 효과** |
| 2503.18445 (CVPRW 2025) | seg, DELIVER | 벤치만, 학습법 없음 ✔ | NM: Gaussian σ{0.1,0.2,0.5}+S&P D{0.05,0.1,0.2}, event엔 Gaussian 미적용 ✔ | — | NM high: MAGIC++ 8.70, CMNeXt 2.31 ✔ | — | NM에서 전 모델 붕괴 |
| CMNeXt 2303.01480 | seg, DELIVER | 품질 감독 없음(SQ-Hub 비감독) ✔ | **train split에 코너케이스 포함**(MB 600·OE 200·UE 199·LJ 199·EL 200) ✔ | — | — | — | 열화 라벨은 파일명에 있으나 미사용 |
| 제외(패턴 참고): SelectFusion 1912.13077(게이트는 pose loss만), DQSD 2008.04159(데이터 내부 불일치로 의사 GT), DA-CLIP 2310.01018(열화 10종 분류 99.2%), MetaBEV·RoboFusion·SAFER-DEIM 2606.01173(수작업 서술자 라우팅, retention 87.9→95.0 ✔) | | | | | | | |

요점: (1) **DELIVER/MUSES seg 계열에 감독 품질 헤드 사례는 없다**. 정의에 맞는 것은 GIML(연속 강도 회귀)·VG-SAF(픽셀 마스크→분산)·MoME(이산 드롭 라벨) 셋뿐. (2) clean 손실 없이 열화 이득: GIML·AV-RelScore·ReliFusion·VG-SAF — 공통점 = 학습 시 열화 주입 + **곱셈 가중**. (3) 연속 강도 라벨(GIML·2605.05439)이 이산 라벨(MoME, LiDAR-drop NDS 열세)보다 부분 열화에 일반화. (4) 2606.26473: 비감독 품질 점수는 셔플해도 Δ≈0 → 우리 "엔트로피·일치도 무효" 실측과 일치. curriculum 유무 ablation은 어느 논문에도 없음(미확인).

### 표 2. clean 교사 → 열화 학생 증류 / 모달 드롭아웃 / 일관성

| 논문 | 벤치 | 학생 입력 | 교사 | 증류 수준 | 드롭 확률·일정 | clean 풀모달 Δ(무드롭 대비) | 결측/열화 이득 | 비고 |
|---|---|---|---|---|---|---|---|---|
| AnySeg 2411.17141 | DELIVER/MUSES B0 | 결측(≥1 모달 보존) | 동결 풀모달 교사 | 특징 4스케일(KL+코사인)+예측 KL | 샘플별 무작위, 확률 미확인 | **−3.14**(60.26 vs MAGIC 63.40), RMMSS 표 기준 −2.51 ✔ | 15조합 평균 46.64 vs 40.49 ✔ | 동결 교사가 있어도 clean 손실 |
| RMMSS/RobustSeg 2505.12861 | DELIVER B0/MUSES/MCubeS | 결측 + NM 평가 | 1단계 동결 풀모달 → 2단계 교사 2개(강건+풀모달) + FSM | 프로토타입+특징 KL+로짓 KL | 미확인 | **61.92 → 1단계 60.45(−1.47) → 2단계 61.85(−0.07)** ✔ | EMM 39.36→49.86(+10.5) ✔ | 풀모달 교사를 로짓 수준에서 재주입해 회복 |
| MMANet 2304.08028 (CVPR 2023) | NYUv2 RGB-D | 결측(Bernoulli) | 풀모달 사전학습 교사 | 특징 관계행렬(불확실도 가중) | 미확인 | 전용 49.18 / **단순 드롭 47.23(−1.95)** / MMANet **49.62(+0.44)** ✔ | 평균 42.77→45.58 ✔ | 단순 드롭 손실을 증류로 역전 |
| DMRNet 2407.04458 (ECCV 2024) | NYUv2 | 결측 | 없음(확률 임베딩+hard-combination 정규화) | — | 미확인 | 49.27(+2.04 vs 드롭, +0.09 vs 전용) ✔ | 평균 45.08 ✔ | 증류 없이 회복 |
| MMP 2410.03010 | NYUDv2/MCubeS(CMNeXt) | 결측(반복별 무작위 부분집합) | 없음 | — | 미확인 | **56.30 → 단순드롭 51.12(−5.18) → MMP 53.81**; MCubeS 51.54→48.56→48.95 ✔ | D-only 6.01→41.08 ✔ | RGB 중심 융합에 단순 드롭 = clean 대손실 |
| CHARM 2508.03060 | DELIVER B2 | 결측(취약모달 편향 SoftMin 샘플링) | 없음; 이중 경로(CoL 전체모달 + InE 부분집합) 둘 다 CE | — | 절대 확률 미확인 | **68.43(+0.18 vs Any2Seg)** ✔ | 평균 54.96 vs 45.04, Last-1 28.76 vs 0.31 ✔ | **전체모달 경로 손실 상시 유지**가 clean 보존 재료 |
| MISS/FPT 2401.16923 (IV 2024) | DELIVER RGB-D(동결 MultiMAE+프롬프트) | 결측(MMS 모달별 독립 스위치) | 없음 | — | 비율 사전지정 없음 | AdaptFormer 55.84→55.74(−0.10), FPT 57.81→57.38(−0.43), Cityscapes +0.07/+0.31 ✔ | RGB 결측 26.54→39.33 ✔ | **동결 백본+소수 파라미터면 드롭이 clean을 거의 안 흔듦** |
| MAGIC 2407.11344 / Any2Seg 2407.11351 | DELIVER B2 | 드롭 없음(유사도 순위 선택 / VLM 교사) | 없음 / 동결 VLM | 특징 일관성 | 없음 | +1.36 / +1.95 vs CMNeXt ✔ | 평균 44.66 / 45.04 vs 25.25 ✔; E·L 단독 ≈0 ✔ | clean 보존하되 취약모달 회복 포기 |
| MAGIC++ 2412.16876 | DELIVER B0 | 드롭 없음 | 없음 | 계층 유사도 정렬 | 없음 | B0 −1.73 vs MAGIC-B0 ✔ (B2 재현 67.33) | E 19.03·L 18.67 회복 ✔ | |
| 함수엔트로피 정규화 2505.06635 | DELIVER/MUSES/MCubeS | 결측 + 예측 평균 융합 | 없음 | Fisher 정보 정규화 | 미확인 | RDEL 51.65 < R-only 55.56 ✔(구조 차이 혼재) | 평균 48.29 ✔ | 드롭+평균융합 = clean 대손실 |
| MetaBEV 2304.09801 (ICCV 2023) | nuScenes | 결측(모달 스위치) | 없음 | — | **1/3·1/3·1/3 샘플별** ✔ | det NDS 71.0→71.5(+0.5); **BEV seg 70.4→68.5(−1.9)** ✔ | LiDAR 결측 NDS 9.8→42.6 ✔ | 같은 일정이 det는 이득·dense seg는 손실 |
| M3L 2304.10756 (CVPR 2023) | Stanford Indoor 반지도 RGB-D | 결측(RGB/D/둘 다 1/3) | **EMA 교사**(풀모달, hard pseudo-label) | 예측 | 1/3 ✔ | 0.2% 라벨: 48.54→49.05 ✔; 완전지도 결과 없음 | MM-robust 35.29→45.46 ✔ | EMA+모달마스킹의 유일 사례, 반지도 한정 |
| ModDrop 1501.00102 / AV-HuBERT 2201.02184 | 제스처 / LRS3 | 결측 | 없음 / 자기 | — | 모달별 독립 / p_m=0.5,p_a=0.5 사전학습만 ✔ | 96.77→96.81(+0.04) ✔ / 단일모달 평가 | 모캡결측 38.41→92.82 ✔ / 시각 WER 55.2→46.8 ✔ | 늦은 융합·사전학습 단계 드롭은 무손실 |
| BraTS 계열: ShaSpec 2307.14126, mmFormer 2206.02425, MetaKD 2405.07155, M3AE 2303.05302, ModDrop++ 2203.04959, CCSD 2511.14599(점진 드롭 curriculum 명시, 수치 미확인) | | 결측 | 자기/없음 | 특징 | 대부분 미확인 | 무드롭 전용 대비 미보고 | | 결측 전용, 존재-열화 없음 |
| MIC 2212.01322 (CVPR 2023) | UDA 단일모달 | 패치 마스킹 | EMA | 의사라벨 일관성 | — | GTA→CS +2.1 ✔ | — | 전제 = 타깃 라벨 없음; 지도 멀티모달 유해 보고는 미확인 |
| 무드롭 계열: MemorySAM 2503.06700, StitchFusion 2408.01343, GeminiFusion 2406.01210, Sigma, DPLNet | | 드롭 없음 ✔ | | | | RDEL 65.38 / 68.18(B2) / 66.9 ✔ | 2503.18445에서 결측·노이즈 붕괴 | |

(A) clean 무손실 조건: ① 전체모달 CE 경로를 학습 내내 유지(CHARM +0.18, ModDrop +0.04, MetaKD 평균 채움) ② 동결 백본+소수 파라미터(MISS ±0.5) ③ 풀모달 **동결** 교사를 로짓 수준에서 재주입(RMMSS 2단계 −0.07, MMANet +0.44) ④ 드롭 대신 결정론적 선택(MAGIC·Any2Seg, 단 E/L≈0).
(B) clean 손실 조건: 샘플별 무작위 마스킹 + 단일 학생 + RGB 중심/토큰 결합 융합(MMP −5.18, AnySeg −3.14, RMMSS 1단계 −1.47, MetaBEV seg −1.9). 동결 교사가 있어도 학생이 마스킹 입력만 받으면 못 막음(AnySeg).
MIC와의 차이: MIC는 (i) 타깃 라벨 없음 (ii) EMA 교사 (iii) 공간 패치 마스킹. clean 회복 사례는 전부 **고정 풀모달 교사 + GT 유지 + 모달 단위 열화**였고, EMA+모달마스킹은 M3L(반지도) 한 편뿐. EMA 교사는 학생과 같은 궤적을 따라 드롭 열화가 교사에 전파되어 clean 기준점이 고정되지 않는다(추론; 직접 보고 미확인). 우리 H15(−1.67)는 이 조합(EMA+마스킹+GT 존재)이었다. "존재-열화(노이즈) 모달을 학습에 넣은 dense seg 논문"은 **하나도 못 찾음**(미확인) = 공백.

### 표 3. 게이트 붕괴 원인과 처방

| 논문 | 게이트 | 붕괴 원인(논문 근거) | 처방 | 실증 수치 | 입력 의존 증거 | 동결+LoRA 적용 |
|---|---|---|---|---|---|---|
| ReMix 2603.10160 | Mixture-of-LoRAs 소프트맥스 | Theorem 1: 가우시안 초기화 라우터 ESS≈1, 학습 중 ESS→1 급락 ✔ | 학습 소프트맥스 가중치 폐기 + 상수 ω + RL 선택 | GSM8K 62.47→65.66 ✔ | ESS 곡선 | **가장 직접(LoRA 혼합)** |
| ConfSMoE 2505.19525 | 소프트맥스 top-k | 야코비안: 기울기 소수 전문가 집중; 부하균형손실이 반대 방향 기울기로 충돌 → 진동 ✔ | 전문가별 confidence net을 GT MSE로 **감독** | MIMIC-IV F1 +1.37~4.12 ✔ | 라우팅 시각화 | 감독 게이트 = seg 보조 헤드로 대체 가능 |
| GS-MoE 2609.18688 | 암묵 라우터 | 무작위 초기화 시 균형손실 있어도 93.6% 단일 전문가 ✔ | 사전학습 ckpt 복제 초기화 | 93.6 → 41.5/34.5/24.0 ✔ | 사용비율 | LoRA 전문가 동일 ckpt 복제 |
| DiT-MoE 2605.19378 | 토큰 MoE | 소프트맥스 포화·자기강화·**bf16 ULP 아래 갱신 절단**·얕은/깊은 층 데드락 ✔ | dense 초기화, 마스터 fp32, cross-attn 라우터 | 데드락 층 30%→<10% ✔ | 점유율<10%, 코사인>0.99 | **게이트만 fp32 유지 즉시 적용** |
| ARGate 1901.10610 | NetGated 소프트 | Fig 6: 센서 손상돼도 가중치 ~0.4 봉우리(중요도 미반영) ✔ | 모달별 보조 단일모달 손실을 가중치 목표로 + lattice 단조 제약 | KITTI det +4.81, HAR 최악손상 +7.57 ✔ | 손상 시 가중치 0.06~0.1 히스토그램 ✔ | 센서별 LoRA 가지에 보조 헤드 |
| AECF 2505.15417 | 소프트맥스 p∈Δ | 결측 학습 시 게이트 엔트로피 0.12 nats로 붕괴 ✔ | 인스턴스 적응 엔트로피 벌점 + curriculum 마스킹 | COCO 0.598→0.628, 50% 드롭 0.228→0.440; 항 제거 시 −1.7pp ✔ | 엔트로피-신뢰도 단조 | 경량 융합층 |
| MoME 2503.19776 | 하드 쿼리 라우터 | 동기만(진단 아님) | 합성 드롭 라벨 CE, 2단계 | nuScenes-R LiDAR 드롭 +4.2 mAP ✔ | **Table 5 조건별 라우팅 비율**(clean 94% 융합, LiDAR 드롭 92% 카메라) ✔ | 디코더 앞 → 백본 무관 |
| Flex-MoE 2410.08245 (NeurIPS 2024) | 소프트맥스 top-k | 결측 조합별 특화 안 됨 | 2단계: G-router(균형손실) → S-router(관측 조합→목표 전문가 CE) | ADNI 66.11 vs LIMoE 55.18; 특화 제거 62.75 ✔ | 전문가 분포 | 라벨 = 모달 조합 |
| LIMoE 2206.02770 | 토큰 MoE | importance loss는 소수 모달 전부 버려도 최솟값 0.5% 이내 ✔ | local 엔트로피 + global 엔트로피(τ=log S 하한) | 텍스트 53.3→56.2; 이미지 local은 56.2→53.5 **해로움** ✔ | 전문가별 모달 특화 통계 | 토큰 MoE 전용 |
| Switch 2101.03961 / ST-MoE 2202.08906 | 하드 top-1 | ST-MoE: 로짓 폭주 | 균형손실 α=1e-2 / z-loss c_z=1e-3 | 안정 런 4/6→3/3 ✔ | — | z-loss는 안정성 처방, 입력의존 처방 아님. 모달 3~4개면 균형손실이 품질 신호를 지움 |
| CNN-seg MoE 2604.13761 | 패치 top-2 | 균형손실 종류·백본 크기에 따라 붕괴 ✔ | switch/importance/entropy 비교 | Cityscapes +1.84~3.90; NRE switch ≥0.966 vs importance 0.58~0.81; entropy loss는 정확도 최고지만 붕괴↑ ✔ | NRE·TEC | **밀집 예측 실증**; 진단 그대로 계산 가능 |
| DynMM 2204.00102 | Gumbel 하드 | 진단 없음 | 온도 감쇠 + 자원 손실 | NYU seg 50.5→51.0 ✔ | λ별 경로 분포 ✔ | 자원 손실 없이는 입력의존 보장 없음 |
| MLE-SAM 2412.04220 | 동결 SAM+LoRA, 공간평균→선형→소프트맥스 | 절제: 가중 특징만 57.99~58.35 < 균등 61.87 < 둘 다 64.08 ✔ | 없음(균등과 병렬) | DELIVER 64.08 ✔ | 라우터 분포 분석 없음 | **우리와 가장 유사**: 학습 라우터 단독 < 균등 평균 |
| R2-T2 2502.20395 (ICML 2025) | 멀티모달 MoE 소프트 | 오라클 라우팅 92.1 vs 기본 79.3 ✔ | 테스트 시 kNN 재라우팅 | 79.3→85.2 ✔ | 오라클 격차 | 동결 모델 |
| QMF/PDF | 게이트 미학습(불확실성→가중치 고정 함수) | 붕괴 우회 | — | 잡음 하 1~3pp ✔ | — | 모달별 보조 헤드 필요 |
| 균형손실 계열(게이트 없음): OGM-GE 2203.15332(CREMA-D 51.7→61.9 ✔), Greedy 2202.05306(u(RGB|depth)=0.01 ✔), UMT 2305.01233, MLA 2311.10707, 2407.09705, MMPareto 2405.17730, 2309.06255 | | 모달 경쟁으로 약한 모달 인코더가 덜 학습 | 기울기 변조·교대 학습·Pareto | 분류만 | | seg 실증 없음 |

실증된 원인: ① 소프트맥스 라우터의 기울기 집중·포화(ReMix·ConfSMoE·DiT) ② 균형손실은 못 막거나 충돌(GS-MoE·ConfSMoE·LIMoE) ③ **학습 데이터에 품질 변동이 없으면 배울 신호가 없음**(ARGate Fig 6; MoME·Flex-MoE는 합성 라벨로 감독해야 조건별 라우팅 발생) ④ bf16+소형 게이트 파라미터 절단(DiT). 우리 H1(P10~P27 12세대 상수 수렴, std≈0.0000)은 ③+①과 부합하며, 처방 중 조건별 라우팅 분포를 실제로 바꾼 것은 **감독형 게이트**뿐이다. 권장 진단: 검증셋 게이트 벡터 모달별 σ·평균 엔트로피(AECF 0.12 nats=붕괴), NRE/TEC(2604.13761), **한 모달 드롭/노이즈 시 게이트 이동 여부 표**(MoME Table 5), 게이트를 데이터셋 평균 상수로 치환한 재평가(Δ≈0이면 상수 등가; MLE-SAM 절제 형태).

### 표 4. 평가 규약

| 항목 | 내용 | 근거 |
|---|---|---|
| 2503.18445 정의 | EMM: 완전모달 학습 가중치로 15조합 0-채움 평가. RMM: 비율 r∈{0.75,0.5,0.25} 무작위 0. NM: Gaussian σ{0.1,0.2,0.5}+S&P D{0.05,0.1,0.2} 동시, event엔 Gaussian 미적용 | §3.2.1~3.2.3 ✔ |
| 평가 방법·학습 조건 | CMNeXt·GeminiFusion·MAGIC·MAGIC++·StitchFusion(MiT-B2), **DELIVER만**(MUSES·MCubeS 없음). "완전 모달 학습 가중치" 사용, 재학습·공식 ckpt 여부 미기재(MAGIC clean 66.10 vs 원논문 70.86 불일치) | §3.1·§4.1 ✔ / 미확인 |
| 명시된 공정성 규칙 | "For a fair comparison, MiT-B2 is selected as the backbone" 하나뿐. 학습 증강 규칙 없음 | §4.1 ✔ |
| DELIVER 수치 | clean 66.33/66.92/66.10/67.34/68.20 → EMM 평균 37.90/37.07/44.97/44.85/41.98 → RMM r=0.75 평균 42.17/–/45.30/47.06/45.16 → NM low/mid/high CMNeXt 35.23/16.37/2.31, MAGIC++ 33.12/18.31/8.70, StitchFusion 24.91/17.11/9.25 | 표 ✔ |
| 경쟁 방법 학습 증강 | **드롭 없음**: CMNeXt·CMX·MAGIC·MAGIC++·Any2Seg·StitchFusion·GeminiFusion·Sigma·MemorySAM·DPLNet(기하·광도만: resize 0.5–2, flip, color jitter, gaussian blur, crop 1024). **드롭 있음**: CAFuser(MUSES 20%, DELIVER 적용 여부 미확인), DGFusion(CAFuser 승계, 명시 문장 없음), MUSES 기준선(20%), MISS/FPT(MMS), AnySeg, RMMSS, CHARM, EQUISeg(드롭 없음, 무작위 교사-학생 프로토타입 KL) | 각 논문 구현 세부 ✔ |
| 같은 표 혼재 사례 | CAFuser Table III·DGFusion 표는 드롭 학습 자기 모델을 clean 학습 CMNeXt·StitchFusion과 한 표에 두고 차이 미언급. MAGIC Table 2만 "All methods are trained with four modalities" 캡션 명시 | ✔ |
| DELIVER 특수성 | train split에 MB/OE/UE/LJ/EL 코너케이스 포함(CMNeXt 부록 Table 5) → 조건별 열은 "미학습 부식"이 아니라 분포 내 조건 성능. 2503.18445의 zero-fill·Gaussian·S&P가 진짜 미학습 열화 | ✔ |
| 인접 규약 | ImageNet-C 1903.12261: "networks should not be trained on these images", Clean Error + mCE 병기, held-out 부식 4종 ✔. RoboBEV 2304.06719: NDS(clean)+mCE+mRR 병기, 공식 ckpt 우선 ✔. 3D-det 2303.11040: AP_clean·AP_cor·RCE 병기, 증강 효과는 부록 별 실험(PointCutMix 등 "hardly improve") ✔. MUSES 2401.12761: 조건별 열 + All, 기준선 20% 드롭 ✔. ACDC 2104.13395: UDA vs supervised 별 표 ✔ |
| 드롭 학습의 clean 대가 | MMP −3~−8, RMMSS 1단계 −1.47, AnySeg −3.14 vs 예외 MISS +0.31·ConD 2607.20326("preserving full-modality accuracy") ✔; 2102.11273: 학습 증강과 테스트 부식의 지각적 유사도가 점수를 부풀림 ✔ |

권고 규약: clean 헤드라인(전 모달, 기하·광도 증강만) / 강건 헤드라인(2503.18445 프로토콜: EMM 평균·RMM r=0.75·NM low/mid/high + mRR식 상대 지표) **별 표**, 두 표 모두 "학습 시 모달 드롭/노이즈 증강 여부" 플래그 열, 드롭 변형은 같은 구조의 clean 학습 행 병기(Δclean, Δrobust), NM 벤치 열화(Gaussian+S&P)를 학습에 넣지 않고 다른 계열로 학습(ImageNet-C 규칙), DELIVER 조건별 열은 "train 포함 조건" 각주.

## 실행 가능한 학습 레시피 2개 (동결 DINOv3-L + 센서별 LoRA, 외부 신호·조건 라벨 없음)

공통 전제: 시작점 = 현재 E1 확정 ckpt(4탭). 학습 열화는 라벨이 아니라 **우리가 주입한 연산자 파라미터**에서 나오므로 제약 위반 없음(2605.05439 방식). 벤치 열화(Gaussian+S&P)는 학습에서 제외하고 held-out으로 둔다.

**레시피 A: 감독 품질 헤드 + 고정 단조 곱셈 가중 (2단계)** — 근거: GIML 2607.06943(강도 회귀, clean +1.75/열화 +2.2), VG-SAF 2608.24366(픽셀 마스크 감독, 4단계 curriculum), 2605.05439(presence+severity 다중 헤드), 2606.26473(감독 정렬 필수), ARGate(감독 없는 게이트는 손상에 무반응), QMF(게이트를 고정 함수로 두면 붕괴 우회), MLE-SAM(학습 라우터 단독 < 균등).
- 1단계(품질 헤드 선학습, 백본·LoRA·융합 전부 동결, ~5 epoch): 모달별 LoRA 가지 출력 토큰(융합 전) 위에 경량 헤드 h_m(2층 conv, 패치 단위)를 붙여 (a) presence BCE(모달 0-채움 여부) (b) severity s∈[0,1] SmoothL1(존재 조건부) (c) 패치별 열화 마스크 BCE를 감독. 열화 주입: 샘플당 각 모달 독립 p=0.5로 열화(GIML 50/50), 종류는 {패치 드롭 r~U(0,0.75), 가우시안 블러 σ~U(0,3), 감마 노출 γ∈[0.3,3], 스펙클/컬러시프트(RGB), depth 홀·스케일 노이즈, LiDAR 빔 드롭·jitter, event 저해상 다운샘플} 균등 샘플링; severity 라벨 = 연산자 파라미터의 정규화값. 합격 판정(학습 전에 고정): 열화 vs clean 패치 AUROC>0.9, severity MAE<0.1, **held-out 열화(Gaussian+S&P)에서도 AUROC>0.8**.
- 2단계(융합 미세조정, LoRA+융합+헤드 학습, 30~50 epoch, LR 기존의 0.5×): 가중치 w_m = 1 − β·ŝ_m(β=0.8 고정, 학습 파라미터 아님; QMF식 고정 함수) 를 모달 토큰에 **곱셈**(f' = f⊙w, 잔차 아님), 4모달 정규화 없음(결측 시 RGB+depth만으로 가는 현재 거동 유지). 손실 = CE(clean 패스) + CE(열화 패스, 같은 샘플) + λ_q(0.1)·품질손실 + λ_kd(0.5)·KL(동결 E1 로짓 ‖ 열화 패스 로짓). **clean 패스 CE를 매 배치 유지**(CHARM). curriculum: severity 상한을 0.3→0.6→1.0으로 10 epoch마다(VG-SAF 4단계 축약). 게이트 파라미터 fp32 유지(DiT 2605.19378).
- 예상 이득 근거: RMM r=0.5 depth −12.68(우리 실측)은 "존재-열화"이며 GIML은 같은 형태(mask 비율)에서 +2~6, RMMSS는 EMM +10. clean은 MISS(동결 백본 ±0.5)·CHARM(+0.18) 근거로 ±0.5 대역 목표. 판정은 3시드 필수(depth 의존 σ 3.8).

**레시피 B: 동결 풀모달 교사 → 존재-열화 학생 증류, 품질 헤드 없음, 이중 경로** — 근거: RMMSS 2단계(풀모달 교사 로짓 재주입으로 −1.47→−0.07), MMANet(특징 관계 증류로 −1.95→+0.44), CHARM(전체모달 경로 상시 유지 +0.18), MISS(동결 백본에서 드롭 무손실), AnySeg 반례(학생이 마스킹만 받으면 −3.14), M3L/MIC 반례(EMA 교사 회피).
- 교사 = E1 확정 ckpt **동결**(EMA 아님). 학생 = 같은 구조, E1에서 초기화, LoRA+융합만 학습.
- 매 배치 두 패스: (i) clean 패스 CE (ii) 열화 패스 = 모달별 독립 p=0.3으로 열화(MUSES 관행 20%~MetaBEV 1/3 사이), 열화 종류 레시피 A와 동일 계열, 단 **완전 결측은 열화의 20%만**(우리 실측상 결측은 이미 무해, 표적은 존재-열화). 손실 = CE_clean + CE_deg + λ_logit(1.0)·KL(교사 clean 로짓 ‖ 학생 열화 로짓, T=2) + λ_feat(0.1)·MMANet식 샘플 간 관계행렬 L1(융합 특징). 열화 패스에서 clean 모달 토큰에는 증류 손실을 걸지 않는다.
- 일정: 40 epoch, LR 0.5×, 첫 5 epoch는 λ_logit만(warm-up), 이후 λ_feat 추가. P50-MAP 정렬 사전학습을 쓴 ckpt에서 시작하면 H22 +0.74와 합산 가능(단 H22는 1페어라 재현 대기 중).
- 예상 이득 근거: RMMSS EMM +10.5·clean −0.07, MMANet 평균 +2.8. 표적 지표 = RMM r=0.5 depth −12.68 → 절반 이하, NM low에서 CMNeXt 35.23·MAGIC++ 33.12 대비 우위. clean 판정 게이트 = 3시드 평균 Δ ≥ −0.3.

두 레시피의 공통 판정 규약: clean 헤드라인과 강건 헤드라인 별 표, "학습 시 열화 증강: 있음" 플래그, 같은 구조의 무열화 E1 행 병기, held-out 열화(Gaussian+S&P) 별도 열, 게이트 진단(모달별 σ·조건별 이동표·상수 치환 재평가)을 결과와 함께 보고.

## 위험 목록
- **clean 손실**: 샘플별 마스킹+단일 학생 구조에서 −1.5~−5(MMP·AnySeg·RMMSS 1단계·MetaBEV seg −1.9). 완화 = clean 패스 CE 상시 유지 + 동결 교사(EMA 금지) + 동결 백본. 그래도 시드 σ가 크므로 1페어 판정 금지.
- **학습 열화 과적합**: 2102.11273(학습 증강과 테스트 부식의 유사도가 점수를 부풀림). 벤치 NM 열화(Gaussian+S&P)를 학습에 넣으면 ImageNet-C 규칙 위반 → held-out으로 두고 별도 열 보고.
- **품질 헤드의 자명해**: clean 샘플이 항상 severity 0이면 헤드가 "RGB 통계"만 배울 수 있음. 완화 = 1단계 합격 판정에 held-out 열화 AUROC 포함, 게이트 셔플 검정(2606.26473) 실시.
- **게이트 재붕괴**: 곱셈 가중을 학습 파라미터로 풀면 H1 재발 가능(MLE-SAM 절제). 완화 = 가중치를 severity의 고정 함수로 두고 β만 sweep, 게이트 fp32.
- **공정성**: CAFuser·DGFusion처럼 드롭 학습 모델을 clean 학습 기준선과 같은 표에 두면 리뷰어 지적 대상. 플래그 열 필수. 반대로 MUSES 리더보드는 이미 20% 드롭이 관행이므로 우리 무드롭 수치가 불리할 수 있음(DGFusion 드롭 여부 미확인).
- **DELIVER train에 코너케이스 포함**: 조건별 열 이득을 "강건성"으로 주장할 수 없음. 강건성 주장은 2503.18445 프로토콜(zero-fill·노이즈)로만.
- **event·LiDAR 잉여**: 우리 실측상 event·LiDAR 결측 무해(−0.04/+0.01)이므로 이 두 모달에 대한 품질 가중은 clean에서 무의미하고 존재-열화(D3 케이스 EL·LJ)에서만 검증 가능; 표본 19~20장이라 3시드 필수.
- **미확인 항목**: 존재-열화 모달을 학습에 쓴 dense seg 선례 없음(레시피 A·B 모두 문헌 외삽), curriculum ablation 없음, RMMSS·AnySeg·MMANet 드롭 확률 미기재, 2503.18445의 ckpt 출처 미기재, CAFuser DELIVER 드롭 적용 여부·DGFusion 드롭 여부 미확인.

## 막힘 / 판단 필요
- 레시피의 수치(p=0.5/0.3, β=0.8, λ, epoch)는 문헌 값(GIML 50/50, MetaBEV 1/3, MUSES 20%, VG-SAF 4단계)에서 옮긴 설계값이며 우리 세팅에서 검증된 것이 아니다. 사전 등록 게이트(clean 3시드 Δ ≥ −0.3, RMM r=0.5 depth 손실 절반 이하)를 판정 세션이 확정해야 한다.
- 레시피 A와 B 중 하나만 먼저 돌린다면 B(품질 헤드 없음, 재료 전부 dense seg 선례 있음)가 문헌 위험이 낮고, A는 "감독 품질 헤드"라는 노벨티 축이지만 seg 선례가 없다.
