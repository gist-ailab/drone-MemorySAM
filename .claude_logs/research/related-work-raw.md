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
