# 🗺 models/ — MOC (Map of Content)

> 폴더 역할: **모델 아키텍처 문서** — 버전 변천(canonical), 논문/발표용 피규어, 버전별 설명 노트.

| 파일 | 한줄설명 | legacy_id |
|------|----------|-----------|
| [arch-evolution.md](arch-evolution.md) | P8~P31 + SAM3-RBMA 모델 상세(forward/모듈/한계/결과) — **아키텍처 canonical** | 02 |
| [figures-ascii.md](figures-ascii.md) | 논문/발표용 ASCII 아키텍처 피규어. ⚠️ P26까지만 (P27+ 미작성) | 08 |

## explain/ — 버전별 설명 노트 (구 `outputs_model_explain/`, 사본)

| 파일 | 한줄설명 |
|------|----------|
| [explain/p08-confidence-head-v2.md](explain/p08-confidence-head-v2.md) | P8: ConfidenceHeadV2 + Sigmoid UAMM |
| [explain/p09-cross-modal-fusion-head.md](explain/p09-cross-modal-fusion-head.md) | P9: CrossModalFusionHead + Max-Norm UAMM (MULTIAQUA 최선) |
| [explain/p10-cross-modal-fusion-head-v2.md](explain/p10-cross-modal-fusion-head-v2.md) | P10: CrossModalFusionHeadV2 + ModalAuxHead + Oracle KL (취소) |
| [explain/p11-mi-routing-loss.md](explain/p11-mi-routing-loss.md) | P11: P10 + MI Routing Loss (취소) |
| [explain/p12-input-conditioned-moe.md](explain/p12-input-conditioned-moe.md) | P12: Input-Conditioned Soft MoE LoRA |
| [explain/p13-energy-score.md](explain/p13-energy-score.md) | P13: Energy Score Fusion + Expert Collapse Fix |
| [explain/p14-separate-aux-decoders.md](explain/p14-separate-aux-decoders.md) | P14: Per-Modality Separate Aux Decoders |
| [explain/p15-spatial-energy.md](explain/p15-spatial-energy.md) | P15: Spatial Energy Fusion (역대 최악) |
| [explain/p16-calibrated-entropy.md](explain/p16-calibrated-entropy.md) | P16: Calibrated Spatial Entropy Fusion |
| [explain/p17-multi-scale-aux.md](explain/p17-multi-scale-aux.md) | P17: Multi-Scale FPN Aux Decoder + Calibrated Spatial Entropy |
| [explain/p19-spatial-fusion-head.md](explain/p19-spatial-fusion-head.md) | P19: Learned Spatial Cross-Modal Fusion |
| [explain/p20-shared-mlp-gate.md](explain/p20-shared-mlp-gate.md) | P20: Shared MLP Gate + Higher Rank MoE |
| [explain/p21-deba-fp.md](explain/p21-deba-fp.md) | P21: DeBA-FP (Deformable Bottleneck Adapter for Feature Pyramid) |
