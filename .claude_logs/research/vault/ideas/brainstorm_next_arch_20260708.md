---
title: 차세대 멀티모달 인식 아키텍처 브레인스토밍 (VFM encoder × adaptive fusion)
tags: [brainstorm, next-arch, vfm, dinov3, sam3, radio, rbma, adaptive-fusion, deep-research]
created: 2026-07-08
source: 내부 문서 전수(doc 01/02/12/16/18/19/20 + research_vault 122노트) + 신규 deep research 2트랙(2026-07-08, WebSearch/arXiv 원문 검증)
status: proposal-draft (미승인)
---

# 차세대 멀티모달 인식 아키텍처 브레인스토밍 — 2026-07-08

> **평가 기준(고정)**: ① Seg — DELIVER val ≥66.51 / test ≥56.71, MUSES SOTA(79.72/79.49) 경쟁 ② Det — mAP50 0.85 (v2 split) ③ 병목 3종 공략: (a) 주간→야간/test 도메인 갭, (b) small-object 붕괴, (c) adaptive fusion 한계.
> 벤치마크 숫자 canonical = `relatedworks/09`, RBMA 포지셔닝 canonical = `../../12_novelty_and_related_work.md`. 신규 문헌은 본 문서가 1차 기록.

---

## 1. 현황 요약 (읽은 문서 기반)

**현재 구조**: frozen SAM2 Hiera-B+ + SoftMoE-LoRA(인코더 적응) + SAM2 시간축 memory attention을 모달리티 축으로 전용한 cross-modal fusion + **RBMA**(training-free 예측 엔트로피 reliability를 memory-attn pre-softmax logit에 additive bias) + SAM mask decoder repurpose(+P30~ class-token decoder).

**강점 (유지해야 할 자산)**
- RBMA의 노벨티 셀(*training-free × predictive-entropy × additive pre-softmax × SAM2 **memory** attention × dense multimodal seg*)은 2026-07 기준 **여전히 미점유** (doc 18 §2 fenced claim; 이번 조사에서도 반례 미발견 — §G 참조). 오히려 이론적 지지가 추가됨: attention=Entropic-OT 관점에서 additive logit bias = log-prior ("Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport", arXiv:2508.08369; PriorSoftmax, arXiv:2601.15380), pre-softmax 신호가 post-softmax보다 정보 보존적(RCStat, arXiv:2506.19549).
- Det 트랙은 데이터 수리(egofill 2.01×)만으로 **mAP50 0.8501 달성**(doc 19 E2.5) — det의 남은 스토리는 absolute mAP가 아니라 저조도 robustness delta(Y1 RGB −0.070 vs 멀티모달).
- MULTIAQUA 챌린지 M=82.10 고정 자산. 재사용 가능한 진단 도구(`tools/eval_per_domain.py`, `module_diagnostics.py` 등) 완비.

**한계 (실측 실패모드, doc 16 §7 정량)**
1. **Frozen-backbone ceiling**: Bridge/Other/Water는 4모달 전부 competence 0 — 융합 개선으로 불가(ISSUE-008). P32 최종 val 64.12/test 55.00 → 목표까지 **val −2.39 / test −1.71**, P28 이후 4세대(P28→P32)가 융합·디코더 개선으로 +1~2pt에 정체 = **인코더 표현력이 병목**이라는 강한 신호.
2. **도메인 갭의 실체 = class-transfer**: DELIVER per-condition spread 2.7~3.6뿐, Wall 62→2 / TrafficLight 81→30 등 특정 class 전이 실패. 라우팅/조건화로 불가(P29 SDC net −1.1 실증). MULTIAQUA만 진짜 day→night RGB 과의존(93→58-70).
3. **Reliability anti-calibration**: RBMA 신호(1−H)의 AUROC = img .77 / depth .62 / **event .30 / lidar .22** — 약한 모달일수록 신호가 역상관. RBMA가 "이미 잘 되는 모달"에서만 작동.
4. **Dead modality**: drop-Δ event 0.02 / lidar 0.01 — 융합이 RGB+depth 2모달로 퇴화. UAMM은 uniform 수렴(질량 45% 낭비).
5. **Small-object 붕괴 (det)**: P30의 단일 s16 query decoder에서 AP_small 0.014 vs FCOS 3-scale 0.111 (E0.1) — **stride-4/8 보존이 관건**, 이는 SAM3-RBMA seg ~24 plateau(single-scale ViT)와 동일 병리.

**시사점**: 다음 세대는 (i) 인코더 천장을 올리고(더 강한 frozen VFM), (ii) RBMA는 **기구(additive pre-softmax bias)를 계승하되 신호를 교체·보강**(entropy 단독 → consistency/교정 신호), (iii) fusion이 stride-4~32 멀티스케일을 통과시키도록 설계해야 함. 이번 deep research의 문헌 수렴점도 동일: 신뢰도는 "단일 디코더 자기 softmax"가 아니라 **cross-modal 합의/프로토타입**에서 얻어야 하며(MG-MTTA, CLoE, Multi-QuAD, AECF — §후보 카드 참조), 검출은 **모든 FPN 레벨에서 융합**해야 한다(DAMSDet, ATM-Net).

---

## 2. 후보 아키텍처 카드

### 카드 A — **DINOv3-RBMA "ReliaDINO"** (본명 후보)

| 축 | 설계 |
|---|---|
| **Encoder** | **DINOv3 ViT-L/16 frozen** (LVD-1689M, Gram anchoring) + **per-modality LoRA**(Q/V, r8~16, 모달별 독립 — MLE-SAM 계보) 또는 thermal/LiDAR는 AnyThermal식 경량 student 증류. 멀티스케일은 **ViTDet simple-FPN**(stride {4,8,16,32}, SAM3-UNet 레시피 검증됨) 또는 ConvNeXt-L 증류판(native hierarchical) |
| **Fusion** | modality-as-token-set **cross-modal memory-style attention 블록을 자체 구현**(SAM2 memory attention의 일반화 — 각 모달 feature가 서로의 K/V가 됨) + **RBMA v2**: `softmax(QKᵀ/√d + λ₁·B_cal + λ₂·B_cons)V` — B_cal = 교정된(per-modal temperature) 예측 엔트로피, B_cons = cross-modal 합의도(training-free 2차 신호, doc 20 Seg-B 승계) |
| **Seg head** | Mask2Former-lite(고정 per-class token, Hungarian 회피 — Frequency-Matcher 근거) + training-only aux per-pixel CE @H/4 (GOOSE-M2F) |
| **Det head** | COCO-pretrained **Deformable-DETR/DINO head**를 FPN {4,8,16} 위에 이식 + **RBMA-in-head**(deformable cross-attn pre-softmax logit에 λ·B 주입 — det 쪽 빈 셀, §F 재확인) |

- **병목 공략**: (a) 도메인 갭 — DINOv3 frozen feature는 현존 최강 dense 표현(frozen ADE20K 63.0 mIoU / COCO 66.1 mAP, 최초의 frozen-backbone 경쟁 검출)으로 Bridge/Water류 "frozen ceiling" 직격 + 저조도 간접 근거(DINOv2 BEV 악천후, arXiv:2501.08118). (b) small object — simple-FPN stride-4 + 멀티스케일 head. (c) adaptive fusion — RBMA v2 dual-signal(아래 노벨티).
- **RBMA 관계**: **계승+확장**. 기구(additive pre-softmax bias)는 유지·이론화(EOT log-prior 인용), 신호는 entropy 단독→dual(교정 entropy + consistency)로 교체. 노벨티 강도 **상**: "reliability-biased cross-modal attention을 SAM2 종속성에서 해방해 VFM-agnostic 프레임워크로 일반화 + backbone ablation(SAM2/SAM3/DINOv3)" — vault 52가 이미 "RBMA는 cross-attention fusion이면 어디든 이식 가능" 문장을 권고. DINOv3 기반 멀티모달 semantic seg mIoU 경쟁자는 현재 **부재**(MMMS는 interactive NoC뿐) → 선점 기회.
- **실현 가능성**: 가중치 HF/timm 공개(ViT-S 21M~7B + ConvNeXt 증류판), **DINOv3 License = 상업 허용**(표기 의무). ViT-L 300M frozen + LoRA는 B200은 물론 3090에서도 가능. 코드 재사용 **중**: 학습 루프/eval/진단 도구 재사용, 인코더-fusion 래퍼는 신규(SAM2 track_step 의존 제거). 예상 공수: fusion 블록+FPN 2주, det head 이식은 doc 20 Det-1과 공유.
- **리스크**: memory attention "토대(MemorySAM)" 서사를 버리고 자체 fusion 블록으로 가면 "그냥 cross-attention fusion + bias" 로 보일 수 있음 → 논문 프레이밍은 "RBMA 프레임워크의 백본 일반화"로, MemorySAM/SAM2 버전을 ablation 축으로 유지.
- **근거 문헌**: "DINOv3" (arXiv:2508.10104, 2025) · DINOv3 GitHub/License · "SAM3-UNet" (arXiv:2512.01789) · "AnyThermal" (arXiv:2602.06203, 2026) · "MLE-SAM" (arXiv:2412.04220) · "Exploiting the Potential of Vision Foundation Models for BEV..." (arXiv:2501.08118) · PriorSoftmax (arXiv:2601.15380, 2026) · RCStat (arXiv:2506.19549).

### 카드 B — **SAM2-RBMA v2 "Calibrated Dual-Reliability"** (진화형·최저 리스크)

| 축 | 설계 |
|---|---|
| **Encoder** | 현행 SAM2 Hiera-B+ (→**Hiera-L 승격**, doc 12 "67 본명 경로") + LoRA. **마지막 stage unfreeze(LR×0.1)** — frozen ceiling의 유일한 인코더 레버 |
| **Fusion** | 기존 memory attention + **RBMA v2 신호 교체**: ① per-modal decoder 용량↑ + temperature/correctness-contrastive **calibration loss**로 event/LiDAR AUROC>0.5 수리(P31 Seg-A 승계) ② **B_cons(cross-modal 합의) 2차 additive 항** ③ AECF식 **subset-monotone calibration**(모달 부분집합 간 신뢰도 단조성 제약 — gate 상수수렴의 문헌적 anchor) ④ 교정 후 AMF를 uniform→reliability-비례 전환 |
| **Seg head** | P31 MS class-token decoder(ClassTokenDecoderMS) 유지 + CHARM식 **dual-path**(융합 경로와 modality-specific 경로 분리로 fragile modality 보호 — dead modality 직격) |
| **Det head** | P29-Det(FCOS 3-scale) 유지 + doc 20 Det-1(Deformable-DETR 이식) 병행 |

- **병목 공략**: (a) 도메인 갭 — class-transfer는 타깃 강증강 + unfreeze + MULTIAQUA는 RGB-zero dual-loss(2512.17450)로. (b) small object — FCOS 3-scale 유지(이미 검증) + P2 aux head 검토. (c) adaptive fusion — 신호 수리가 본질(문헌 만장일치: 신뢰도는 자기 softmax가 아니라 합의/프로토타입에서 — CLoE arXiv:2603.09316, Multi-QuAD arXiv:2412.14489, AECF arXiv:2505.15417, MG-MTTA arXiv:2604.24602).
- **RBMA 관계**: **직접 계승·강화**. 노벨티 강도 **중상**: "single-signal logit bias → **dual-axis training-free reliability field**(entropy+consistency)" + "학습 gate를 training-free reliability로 anchor"(P30 계보). 단 MG-MTTA(2604.24602)가 "entropy 기반 적응은 약한 모달 지배 시 실패"를 이론화했으므로 **선제 인용 필수** — 우리 AUROC .22/.30 실측이 그 이론의 seg-측 증거가 되어 서사가 오히려 강해짐.
- **실현 가능성**: 코드 재사용 **최상**(P31/P32 diff 수준, 2~4주 내 학습 착수). Hiera-L은 VRAM 여유 필요하나 B200 확보됨. 가중치·라이선스 이슈 없음(SAM2 Apache-2.0).
- **리스크**: 인코더 천장 자체는 Hiera-L+unfreeze로도 한계 가능(P28-L 미학습이라 미검증). val 66.51 도달이 안 되면 카드 A로 전환할 탈출 기준 필요.
- **근거 문헌**: doc 16/20 내부 진단 · "AECF: Robust Multimodal Learning via Entropy-Gated Contrastive Fusion" (arXiv:2505.15417) · "CLoE(Expert Consistency Learning)" (arXiv:2603.09316) · "Multi-QuAD" (arXiv:2412.14489) · "Majorization-Guided Test-Time Adaptation ... under Modality-Specific Shift" (arXiv:2604.24602, VLM 문맥 — 스코프 주의) · "CHARM: Collaborative Harmonization across Arbitrary Modalities" (arXiv:2508.03060) · MULTIAQUA (arXiv:2512.17450).

### 카드 C — **SAM3-RBMA 2.0** (PE encoder + ViTDet neck + tracker memory 계승)

- **Encoder**: SAM3의 **Perception Encoder ViT(~446M, 1008², patch14, windowed+global attn)** frozen + per-modality LoRA. plateau(~24) 원인이던 single-scale은 **Sam3DualViTDetNeck/simple-FPN**으로 해소(SAM3-UNet·DART가 메커니즘 선례 — 우리는 "multimodal semantic seg 적용"만 주장).
- **Fusion**: SAM3 tracker가 SAM2-style memory bank/propagation을 계승하므로 **RBMA 이식이 구조적으로 가장 자연스러움**(기존 SAM3-RBMA 트랙의 정통 후계). RBMA v2 신호 교체는 카드 B와 동일.
- **Heads**: SAM3 내장 DETR-style detection head(200 queries) + presence head를 det에 재활용 가능 — det/seg 통합 스토리.
- **병목 공략**: (a) PE는 1008 고해상 + 검출 SOTA 계열이라 소물체 유리, (b) FPN neck으로 멀티스케일, (c) fusion은 B와 동일.
- **RBMA 관계**: 직접 계승. 노벨티 **중**: "SAM3 최초 멀티모달 semantic seg"는 SAMCM-SR이 선점 주장(doc 18 §4) — 확인 필요. 백본 ablation 축으로서의 가치는 확실.
- **실현 가능성**: **최대 리스크 = 가중치 접근성**. HF `facebook/sam3`는 gated(학술 거절 사례 보고), 라이선스는 커스텀 SAM License(재배포 제약). 840M 규모. 기존 sam3_lora_rbma.py 코드 재사용 **상**.
- **근거 문헌**: "SAM 3: Segment Anything with Concepts" (arXiv:2511.16719, 2025) · "SAM3-UNet" (arXiv:2512.01789) · "Detect Anything in Real Time" (arXiv:2603.11441) · "Perception Encoder" (arXiv:2504.13181) · vault 57.

### 카드 D — **C-RADIOv4 백본 + RBMA fusion** (agglomerative 절충안)

- **Encoder**: **C-RADIOv4-SO400M(412M)/H(631M)** frozen — SigLIP2+**DINOv3**+**SAM3** 3교사 증류 학생. any-resolution + **ViTDet 윈도우 모드**(고해상 효율) + teacher adaptor head로 SAM3/DINOv3 feature 공간 출력 가능(SAM3 디코더 결합 공식 데모 존재). per-modality LoRA는 A와 동일.
- **Fusion/Heads**: 카드 A와 동일(RBMA v2 cross-modal attention + Mask2Former-lite + Deformable-DETR).
- **병목 공략**: DINOv3의 dense 품질 + SAM3의 seg 특성을 한 백본에서 — dense 벤치에서 DINOv3-7B와 경쟁(10× 작은 크기, 세부 수치 미확인). any-resolution이 드론 비정방 입력에 유리.
- **RBMA 관계**: 계승(A와 동일). 노벨티 **중상**: "RADIO 계열의 멀티모달(RGB-X) 적응 선례 전무" — 첫 진입 자체가 마이너 기여. 단 백본 서사가 "남의 증류물" 위라 임팩트는 A보다 낮을 수 있음.
- **실현 가능성**: HF `nvidia/C-RADIOv4-*` 게이트 없음, **NVIDIA Open Model License(상업 허용)**. 412M frozen + LoRA는 B200 무난. 코드 재사용 A와 동일 수준. 2026-01 공개라 커뮤니티 검증 짧음(리스크).
- **근거 문헌**: "C-RADIOv4 (Tech Report)" (arXiv:2601.17237, 2026-01-24 — 원문 확인) · "AM-RADIO" (arXiv:2312.06709) · "RADIOv2.5" (arXiv:2412.07679) · NVlabs/RADIO GitHub(라이선스 이원화).

### 카드 E — **Det 특화: 멀티스케일 deformable 신뢰도 융합 헤드** (국책과제 마무리 트랙)

- **구성**: (백본 무관 — P29-Det egofill 스택 또는 카드 A/B 백본) FPN {4,8,16} 각 레벨에서 융합하되 레벨별 분리 유지(ATM-Net "extraction→fusion→separation" 참조) + **DAMSDet식 Modality-Competitive Query Selection + multispectral deformable cross-attention** + **RBMA-in-head**: deformable attention pre-softmax logit에 per-modality reliability bias(1차는 inference-time-only, doc 20 Det-2).
- **병목 공략**: small object 직격(모든 스케일 융합 + P2 검토 — LAF-YOLOv10류 P2 aux head가 소물체 표준 처방), dead modality는 MDQF식 query 교환(저품질 모달 배제)으로.
- **RBMA 관계**: 확장. **det에서 "per-modality reliability additive pre-softmax bias" 셀은 이번 조사에서도 빈 것으로 재확인**(near-miss: DAMSDet=query 선택, Uncertainty-aware DETR 2507.14855=box Gaussian loss, 2505.02161=multiplicative pruning) → seg·det 공통 프레임워크 주장의 두 번째 기둥. 노벨티 **중상**(det 단독으론 상, 단 mAP 0.85는 이미 데이터로 달성했으므로 주장은 저조도 delta로).
- **실현 가능성**: det worktree 재사용 상. egofill 데이터·YOLO 기준점·final split 저조도 프로토콜 완비(doc 19 §9).
- **근거 문헌**: "DAMSDet" (ECCV 2024) · "ATM-Net" (MDPI Drones 10(1):067, 2026 — DroneVehicle 83.7 mAP/4.83M) · "MDQF: Modality-Decoupled RGB-T Detector via Query Fusion" (arXiv:2601.08458, 2026) · "Scarf-DETR" (arXiv:2511.06406) · "WS-DETR" (arXiv:2504.07441, 수상 USV — MULTIAQUA 인접 도메인) · "Uncertainty-aware DETR" (arXiv:2507.14855).

### (보류) 카드 F — V-JEPA 2 / 순수 Mamba 백본

- **V-JEPA 2/2.1**: dense seg가 개선됐어도(ADE20K linear 47.9, arXiv:2603.14482) DINOv3 frozen(63.0)과 격차 큼, 검출 transfer 근거 부재 → **백본 후보 탈락**. 단 "cross-modal latent prediction"을 **보조 loss**(약한 모달 feature를 강한 모달로부터 예측 → dead modality 완화)로 쓰는 아이디어는 카드 A/B에 접목 가능(vault 90 제안과 정합).
- **순수 Vision Mamba/VMamba**: 대규모 SSL pretrain 부재(ImageNet-1K 수준)로 day→night 전이에 불리, 2026 기준 VFM 대체재 아님. 단 **MFuser**(arXiv:2504.03193, CVPR'25 Highlight — Mamba를 frozen VFM 간 co-adapter/fusion으로, token 선형 복잡도)의 패턴은 카드 A의 fusion 블록 대안(고해상 4모달 토큰이 부담일 때)으로 보관.

---

## 3. 비교표 + 추천 순위

| 카드 | 인코더 천장(병목1) | Small obj(병목2) | Fusion 수리(병목3) | RBMA 노벨티 | 가중치/라이선스 | 코드 재사용 | 목표 달성 기대 |
|---|---|---|---|---|---|---|---|
| **A. DINOv3-RBMA** | ◎ (frozen 최강 dense) | ○ (simple-FPN 신규) | ◎ (v2 dual-signal) | **상** (VFM-agnostic 일반화+선점) | ◎ 공개/상업가 | 중 | **val 66.5+ 가장 유력** |
| **B. SAM2-RBMA v2** | △ (Hiera-L+unfreeze) | ○ (FCOS 검증됨) | ◎ (신호 교체) | 중상 (dual-axis) | ◎ | **최상** | val 65~66 (미달 리스크) |
| C. SAM3-RBMA 2.0 | ○ (PE+FPN) | ○ | ◎ (B와 동일) | 중 (선점 경쟁) | **△ gated** | 상 | 중 (접근성 변수) |
| D. C-RADIOv4 | ◎? (검증 짧음) | ○ (ViTDet 모드) | ◎ (A와 동일) | 중상 (첫 RADIO-RGBX) | ◎ | 중 | 중상 |
| E. Det-deformable | — | ◎ | ○ | 중상 (det 빈 셀) | ◎ | 상 | det delta용 |

**추천 top-2**

1. **카드 A (DINOv3-RBMA)** — 근거: ① 병목 1(frozen ceiling)은 P28→P32 4세대가 융합 개선으로 못 뚫은 것이 실증됐고, 인코더 교체만이 남은 1차 레버. DINOv3는 frozen 조건에서 유일하게 SOTA급 dense 증거(ADE20K 63.0/COCO 66.1)를 가짐 — "frozen+PEFT" 전제와 정확히 합치. ② DELIVER에 DINO 계열 mIoU 경쟁자 부재 = 리더보드+노벨티 동시 선점 기회. ③ RBMA를 백본-무관 프레임워크로 승격시켜 "SAM2 부속품" 리스크(SAM2 계열 스쿠프 다발: MLE-SAM/M⁴-SAM/MM-SAM-adapter)에서 탈출. ④ 라이선스/접근성 문제 없음.
2. **카드 B (SAM2-RBMA v2)** — 근거: ① 즉시 착수 가능(코드 diff 수준)한 안전판으로 A와 **병행**: A의 fusion v2(신호 교체)를 B에서 먼저 검증하면 A로 그대로 이식됨(fusion 설계가 공유됨). ② 신호 수리(AUROC .22→>.5)는 어느 카드로 가든 선행 게이트. ③ P31/P32 자산과 MULTIAQUA/DELIVER 제출 파이프라인 무중단.

> 권고 실행 구도: **B(2~4주, fusion v2 검증) → A(본명, fusion v2 이식 + DINOv3 인코더)**, C는 가중치 승인이 나면 백본 ablation 축으로, E는 det 국책과제 보고서용 병행 트랙으로. D는 A가 막힐 때의 대체 백본.

---

## 4. 리스크와 검증 실험 제안 (top-2)

### 카드 A (DINOv3-RBMA) — 싸게 가설 검증

| # | 실험 | 비용 | 기각/확정되는 가설 |
|---|---|---|---|
| A-1 | **Frozen feature probe**: DINOv3 ViT-L frozen + linear/경량 head를 DELIVER RGB 단독으로 학습(≤1 GPU-day, 기존 eval 인프라 재사용), 같은 조건의 frozen Hiera-B+ probe와 per-class 비교. **판정: Bridge/Water/Other가 0을 벗어나는가** (frozen ceiling이 인코더 문제라는 핵심 가설의 직접 검증) | 1 GPU-day | 안 벗어나면 카드 A의 근거 절반 소멸 → B의 unfreeze 레버로 회귀 |
| A-2 | **모달리티 LoRA 적합성 probe**: DINOv3 frozen + LoRA(r8)를 LiDAR-projection/Thermal 단독 입력으로 짧게 학습(DELIVER/MULTIAQUA 서브셋) → per-modal decoder AUROC 측정(module_diagnostics 재사용). **판정: DINOv3 feature 위에서 약한 모달의 reliability가 교정 가능한가(AUROC>0.5)** | 1~2 GPU-day | RGB 특화 SSL이 LiDAR/thermal에 부적합하면 AnyThermal식 증류 경로로 전환 |
| A-3 | **미니 fusion ablation**: 2-layer cross-modal attention 블록(256 해상도, 2모달)에서 (i) bias 없음 (ii) RBMA v1(entropy) (iii) RBMA v2(entropy+consistency) 3-way — DELIVER 서브셋 20ep. **판정: v2 신호가 v1의 anti-calibration 문제를 실제로 완화하는가** | 2 GPU-day | v2 무이득이면 신호 교체 서사 재설계(CLoE식 합의 신호로) |

### 카드 B (SAM2-RBMA v2)

| # | 실험 | 비용 | 기각/확정되는 가설 |
|---|---|---|---|
| B-1 | **학습 없는 신호 스왑**: 기존 P32 ckpt에서 B_cons(cross-modal 예측 합의도)를 오프라인 계산 → entropy 대비 AUROC 비교(module_diagnostics 확장, **학습 0회**). **판정: consistency 신호가 event/LiDAR에서 AUROC>0.5인가** — 카드 A/B 공통의 최우선 게이트 | GPU 수 시간 | <0.5면 training-free 신호로는 불가 → 교정 loss(학습형) 필수로 확정 |
| B-2 | **inference-time AMF 전환**: P32 ckpt에 교정(temperature per modality, val서 피팅) 후 AMF uniform→reliability-비례를 **추론만으로** 적용, test mIoU delta 측정 | GPU 수 시간 | 델타 ≤0이면 "교정 후 비례 융합" 가설 기각 |
| B-3 | **per-modal decoder 용량/교정 단축 학습**: P32 구성에서 per-modal decoder 2배 + temperature+correctness-contrastive loss만 추가, 40ep 단축 런 → AUROC와 drop-Δ(event/LiDAR 부활 여부) | 2~3 GPU-day (B200 P32 종료 후) | AUROC 수리돼도 mIoU 무이득이면 "신호 수리=성능" 연결고리 기각 → dead modality는 CHARM식 구조 보호로 |

**공통 리스크**: ① MG-MTTA(2604.24602)·UDML류 "entropy 신뢰도 비판" 계열은 리뷰 단계에서 반드시 만남 — B-1/A-3 결과를 그대로 방어 데이터로 전환. ② DELIVER 프로토콜 혼입 금지 규칙(doc 18 §1) 유지 — DINOv3 probe도 val/test 태그 병기. ③ 2026 신규 문헌(2601~2606 arXiv ID) 다수가 image-only PDF로 세부 수치 미확인 — 인용 전 원문 정독 필요(특히 MG-MTTA, "Before Fusion, Ask What to Keep" 2606.02679, UGDDL 2605.09600).

---

## 5. 신규 문헌 인덱스 (이번 조사에서 vault에 없던 것만; 정독 우선순위 ★)

**VFM/백본**: DINOv3 (arXiv:2508.10104) ★★★ · C-RADIOv4 (arXiv:2601.17237, 원문 abs 확인) ★★ · SAM 3 (arXiv:2511.16719) ★★ · SAM3.1 (Meta 블로그) · SAM2-to-SAM3 Gap (arXiv:2512.06032) · V-JEPA 2.1 dense (arXiv:2603.14482) · Franca (arXiv:2507.14137) · MambaVision (arXiv:2407.08083) · MFuser (arXiv:2504.03193) ★ · AnyThermal (arXiv:2602.06203) ★★ · RangeSAM (arXiv:2509.15886) ★ · Generative Event Pretraining w/ FM Alignment (arXiv:2603.23032) ★ · Mid-level SSL Probing (arXiv:2411.17474) · 3D Awareness of VFMs (arXiv:2404.08636) · DPLNet (arXiv:2312.00360) · IVGF (arXiv:2409.00973) · SEAR (arXiv:2603.18774).

**Fusion/신뢰도**: MG-MTTA (arXiv:2604.24602, "Majorization-Guided Test-Time Adaptation for Vision-Language Models under Modality-Specific Shift" — 제목 원문 확인, VLM 스코프) ★★★ · AECF (arXiv:2505.15417) ★★ · CLoE (arXiv:2603.09316) ★★ · Multi-QuAD (arXiv:2412.14489) ★ · CAL (arXiv:2510.26289) · Before-Fusion contextual calibration (arXiv:2606.02679) ★ · CHARM (arXiv:2508.03060) ★ · CMPT (arXiv:2501.17823) · PCDF+ATR-UMOD (arXiv:2510.13620, UAV RGB-IR 조건 라우팅+데이터셋) ★★ · Similarity-Preserving Load-Balancing (arXiv:2506.14038) · PriorSoftmax (arXiv:2601.15380) ★ · Attention as One-Sided EOT (arXiv:2508.08369) ★ · RCStat (arXiv:2506.19549) · UGDDL (arXiv:2605.09600, uncertainty→attn-logit bias, unimodal — near-miss 워치) ★.

**Det(멀티스케일/소물체)**: DAMSDet (ECCV 2024) ★★ · ATM-Net (Drones 10(1):067, 2026) ★★ · MDQF (arXiv:2601.08458) ★ · Scarf-DETR (arXiv:2511.06406) · WS-DETR (arXiv:2504.07441) ★ · Uncertainty-aware DETR (arXiv:2507.14855) · MGDFIS (arXiv:2506.12697) · LAF-YOLOv10 (arXiv:2602.13378).

> 미확인 표기: C-RADIOv4 dense 세부 수치(테크리포트 표 미독), StitchFusion Swin 70.3(기각 유지), V-JEPA2.1 가중치 공개 여부, MG-MTTA/2606.02679/2605.09600/2505.02161 내부 수치(이미지 PDF), Web-SSL arXiv ID(2504.01017 간접), ATM-Net 저널 상세(비-arXiv). 투고 전 6개월 재스윕 규칙(doc 18 §7) 적용 대상.
