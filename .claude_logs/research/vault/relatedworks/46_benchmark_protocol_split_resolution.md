---
title: DELIVER/MUSES/MCubeS Benchmark Protocol Resolution — split×backbone clusters, MemorySAM split forensics, reporting rules
tags: [benchmark-protocol, deliver, muses, mcubes, multiaqua, sota, two-cluster, verified-draft]
created: 2026-07-02
source: parallel deep-research Track 3 (sources/07_parallel_research_prompts_2026-07-02.md) + 2× adversarial verification passes (all critical claims confirmed)
status: verified-draft
---

# Benchmark Protocol Resolution — DELIVER / MUSES / MCubeS (+ MULTIAQUA)

Companion synthesis note to [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] (which holds the full verbatim SOTA tables, updated 2026-07-02). This note is the *protocol argument*: why the literature's numbers disagree, what the evidence is, and how we must report.

## Problem setting

DELIVER(RGB-D-E-L, 25 classes, 1042×1042; splits 3,983 train / 2,005 val / 1,897 test)에서 동일 모델 CMNeXt R-D-E-L이 논문마다 **66.30 / 59.18 / 53.0**으로 보고되는 "two-cluster" 불일치가 있었음. 이 불일치를 해소하지 않으면 우리 논문의 비교 표가 리뷰어(특히 CAFuser/DGFusion 저자군)에게 즉시 반박당함. MUSES(1,500/250/750, test 라벨 비공개 — 공식 평가 서버)와 MCubeS(302/96/102, 전원 test 보고)는 상대적으로 단일 프로토콜.

## Novelty (of this finding)

이 노트의 기여는 방법론이 아니라 **프로토콜 포렌식**: (1) two-cluster가 실제로는 **three-cluster (split × backbone)**임을 원문 표 + 공식 코드 라인 수준에서 확정, (2) MemorySAM 65.38의 split을 코드 레벨에서 val로 판정, (3) 2-모달 SAM 기반 MM-SAM-adapter가 4-모달 SOTA를 이미 넘었다는 스쿠프 경보. 세 항목 모두 2개 독립 skeptic 검증 통과.

## Method (how the resolution was established)

**Cluster decomposition** — same-model control이 결정적 증거:

| CMNeXt number | Config | Evidence | Tag |
|---|---|---|---|
| 66.30 | MiT-B2, **val**, RDEL | CMNeXt Tab.1(a) + DELIVER repo README | [VERIFIED-PDF]+[REPO] [val] |
| 53.0 | MiT-B2, **test**, CLDE | CAFuser Tab.III (same row: "66.3 val / 53.0 test") + DGFusion Tab.III | [VERIFIED-PDF]×2 [test] |
| 59.18 | MiT-**B0**, val, RDEL | MemorySAM Tab.1, AnySeg Tab.2, EGFormer Tab.2, MLE-SAM Tab.I | [VERIFIED-PDF]×4 [val] |

- **Resolution은 요인이 아님** — 단, 검증 결과의 정밀 수정: CAFuser/DGFusion은 DELIVER 입력 해상도를 명시하지 않으므로 "모두 1024×1024라서"가 아니라, CAFuser Tab.III가 **같은 모델**을 val 66.3 / test 53.0 (CA²는 67.8/55.6)으로 병기한다는 same-model control이 split 원인을 증명함.
- 공식 코드 증거 [REPO]: DELIVER repo `tools/val_mm.py` L141 = `'val'` 하드코딩(test 라인 주석 처리), split 플래그 부재; README "Please check tools/val_mm.py to modify the dataset for validation and test sets" → README 수치 = val. GeminiFusion 논문은 DELIVER를 "3983 training and 2005 testing"으로 기술 — val을 test처럼 사용.

**MemorySAM 65.38 = val (code-inferred, NOT author-stated)** — 논문·README 모두 split 미기재, repo 이슈 0건, 저자 확인 없음. 판정 근거: repo `val_mm_sam.py` **L146**이 `split='val'` 하드코딩(L148 test 주석 처리); `deliver.py` `__main__`도 `split='val'`. Skeptic 정밀 수정: 하드코딩 위치는 `deliver.py` 데이터셋 클래스(train/val/test 모두 허용, default 'train')가 아니라 **`val_mm_sam.py`**. Table-1 baseline들(CMNeXt MiT-B0 59.18)이 전부 val/B0 클러스터인 것과 정합. 인식론적 라벨: **[val, code-inferred]** — 논문에 인용할 때 이 단서를 유지할 것.

## Quantitative results (headline numbers; full tables in [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] §U3–U7)

- DELIVER **val** ceiling: StitchFusion Swin-Tiny **70.34** [VERIFIED-PDF, val*-inferred: caption에 split 미기재, CAFuser Tab.III의 B2 68.2=val 교차 확인으로 추론]; CAFuser-CAA 68.6 [val], OmniSegmentor DFormer-L 68.0 [val], Mul-VMamba 68.98 [ABSTRACT-ONLY, unknown].
- DELIVER **test** CLDE: DGFusion **56.7** (4-mod) [VERIFIED-PDF, test]; **MM-SAM-adapter RGB-D 57.35 / RGB-L 57.14** (2-mod!) [VERIFIED-PDF, test].
- MUSES **test**: DGFusion 79.5 mIoU / **61.03 PQ** (CLRE) [VERIFIED-PDF]; **MM-SAM-adapter RGB-L 81.07** [VERIFIED-PDF]. GeminiFusion 75.3은 CAFuser Tab.II에만 존재(자체 논문에 MUSES 없음).
- MCubeS **test**: StitchFusion Swin-L 55.9 > MemorySAM 52.88 > CMNeXt-B2 51.54 [VERIFIED-PDF].
- Per-condition: DELIVER 10-case는 **val에서만 존재**(CMNeXt Tab.2, HyperDUM Tab.4: Night 62.46→64.21 +1.75; EQUISeg도 per-condition 보유 — split 미기재·val 클러스터 값); DGFusion은 DELIVER per-condition 표 없음. MUSES test per-condition PQ에서 DGFusion 최약점 = **Night 58.97 / Fog 58.86**. [skeptic-confirmed; "val에서만"은 확인한 논문들에 한정된 universal negative — test 라벨이 공개라 미확인 논문이 원리상 보고 가능]

### ⚠ Scoop alert (skeptic-confirmed, 원 발견보다 오히려 과소평가)

MM-SAM-adapter (2509.10408, SAM ViT-L + ConvNeXt-S side-adapter, 1024², pairwise 2-modality)가 DELIVER test(57.35 RGB-D)와 MUSES test(81.07 RGB-L) **모두에서 4-모달 DGFusion(56.7/79.5)을 상회**. 둘은 동시기(2025-09) 논문으로 상호 인용 없음. → 우리의 "VFM-based multimodal SOTA" 헤드라인은 **MUSES test 81.07을 넘지 못하는 한 거짓**; 대안은 arbitrary-modal / robustness(EMM·RMM·NM) / condition-adaptive 축으로 클레임 한정.

## Limitations

- StitchFusion 70.34의 split은 자체 캡션 미확인(추론); EQUISeg 67.90 표 렌더 실패; Mul-VMamba paywall.
- CMNeXt Tab.2 vs HyperDUM Tab.4의 sensor-failure 5개 케이스(MB/OE/UE/LJ/EL) 행 정렬이 자동 추출 간 불일치 — **LaTeX 인용 전 시각 PDF 재확인 필수**.
- MLE-SAM MUSES 74.8 split 미기재(서버 test일 가능성 낮음 → val/local 추정); MCubeS 해상도(1224×1024 vs MemorySAM 기재 "1920×1080") 미해결; MUSES 리더보드 로그인 게이트 — 미관측 서버 엔트리 가능.
- MAGIC/GeminiFusion/MemorySAM 캡션은 "val"을 문자 그대로 인쇄하지 않음 — 배정은 number-cluster 정합성 + repo 코드에 근거.

## Improvement directions

1. **Dual-split reporting을 표준화**: CAFuser Tab.III의 "mIoU-val / mIoU-test" 이중 컬럼이 유일하게 정직한 포맷 — 우리 메인 표의 템플릿으로 채택.
2. **Backbone 컬럼 의무화**: 59.18(B0) vs 66.30(B2) 혼동은 backbone 컬럼 부재가 원인; 모든 row에 backbone+파라미터 병기.
3. **Test-split per-condition 공백 = 기회**: DELIVER test 라벨은 공개인데 아무도 per-condition test 표를 내지 않음 — 우리가 최초로 내면 protocol-hygiene 기여.
4. **MUSES AUPQ(uncertainty-aware PQ) 트랙 공백**: fusion 클러스터 어느 논문도 이 트랙을 주공략하지 않음 — RBMA의 B_i 신뢰도 맵과 자연 결합.
5. Robustness 부록은 2503.18445의 EMM/RMM/NM 프로토콜 채택(CMNeXt 고노이즈 2.31% 붕괴가 motivating figure 소재).

## Comparison to RBMA-P29-P30 (mechanism-class)

이 노트는 프로토콜 노트이므로 mechanism 비교는 경쟁자 기준으로 요약 (전체 taxonomy는 [[relatedworks/42_attention_logit_bias_novelty_defense]]):

| Competitor | Mechanism-class | RBMA cell 점유 여부 |
|---|---|---|
| CAFuser (CA²/CAA) | **condition-token** (CLIP-grounded verbal token → fusion modulation; global, 라벨/텍스트 필요) | ✗ (per-pixel logit-bias 아님) |
| DGFusion | condition-token + local depth tokens (**depth-GT supervision 필요**, 학습형) | ✗ |
| HyperDUM | **feature-multiply** (hyperdimensional UQ → learnable weight Ω가 feature에 곱) | ✗ |
| MM-SAM-adapter | injector-extractor cross-attn side-tuning (RGB-primary, pairwise, reliability 신호 없음) | ✗ |
| MAGIC/MAGIC++/AnySeg/EGFormer | **learned-gate** / distillation (training-time, inference-time 신호 없음) | ✗ |
| MemorySAM | fusion 가중 없음 (modalities-as-frames, memory attention 무편향) | ✗ (우리의 base) |

**logit-additive-bias (pre-softmax memory-attention) 셀은 이 벤치마크 클러스터 내에서 여전히 미점유.** RBMA = training-free per-modality entropy B_i를 SAM2 memory cross-attention logit에 additive 주입 — DGFusion의 "spatially-varying reliability"를 depth-GT 없이, HyperDUM의 uncertainty-weighting을 학습 없이 달성하는 위치.

## Application to ours (RBMA/P29/P30 적용방향)

1. **표 구성**: DELIVER 메인 표 = val+test 이중 컬럼, CLDE+CLE, backbone 명기, MemorySAM row에 "[val, code-inferred]" 각주. 65.38과 56.7을 같은 컬럼에 두면 즉시 리젝 사유.
2. **넘어야 할 숫자**: val 70.34(StitchFusion Swin-T) / test CLDE **57.35**(MM-SAM-adapter RGB-D — 57.14가 아님) / MUSES test **81.07** mIoU·61.03 PQ / MCubeS 55.9. 4-모달 프로토콜 한정 시 DGFusion 56.7/79.5가 기준.
3. **조건-적응 스토리의 타깃 컬럼**: MUSES per-condition PQ에서 DGFusion 최약 Night 58.97/Fog 58.86 — RBMA의 night/fog 이득이 여기서 보여야 함. DELIVER는 10-case val 표(HyperDUM 대비) + 우리가 최초의 per-condition **test** 표 제출.
4. **P29 SDC**: CAFuser의 CLIP condition token의 label-free 대응물로 포지셔닝 (가장 가까운 condition-token 선행). **P30**: MUSES AUPQ 트랙에 B_i 기반 uncertainty 출력으로 참전 — 미점유 트랙.
5. **MULTIAQUA**: day-train/night-test가 reliability-shift 시연 무대; inference-time(우리) vs training-time(CMNeXt-DH 74.25) 대비.

## Related-work paragraph candidate (English)

> Benchmark protocols for multimodal segmentation on DELIVER are notoriously inconsistent: the same CMNeXt model is cited at 66.30, 59.18, and 53.0 mIoU across the literature. We trace this to a split-and-backbone confound rather than any implementation difference: 66.30 is the MiT-B2 model on the 2,005-image validation split — the protocol hardwired in the official evaluation script — 59.18 is a MiT-B0 reproduction on the same split used by the modality-agnostic line of work, and 53.0 is the MiT-B2 model on the held-out 1,897-image test split adopted by CAFuser and DGFusion, whose side-by-side val/test columns confirm the ~13-point gap is attributable to the split alone. Notably, several strong recent results, including SAM2-based MemorySAM (65.38), do not state their evaluation split, which we could only attribute to the validation protocol via released code. To avoid such ambiguity, we report both splits with matched backbones throughout, and additionally provide per-condition results on the test split, which prior work reports only on validation.

## Cross-links

- Full verbatim tables: [[relatedworks/09_benchmark_tables_deliver_muses_mcubes]] (§U1–U9, 2026-07-02 update)
- Dataset cards: [[relatedworks/06_deliver_muses_mcubes_dataset_note]]
- Mechanism taxonomy / novelty defense: [[relatedworks/42_attention_logit_bias_novelty_defense]], [[relatedworks/44_hyperdum_uncertainty_fusion_relatedwork]]
- Competitors: [[relatedworks/01_memorysam_relatedwork]], [[relatedworks/02_dgfusion_relatedwork]], [[relatedworks/07_cmx_tokenfusion_magic_cafuser_baselines]]
