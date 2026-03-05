# Architecture Figures — P9, P17, P19

> 최종 업데이트: 2026-02-28
>
> 이 문서는 논문/발표용 피규어 가이드입니다.
> 모듈별 I/O 텐서 shape, 내부 구조, 버전 간 차이를 시각화합니다.

---

## 공통 차원 참조표

| 기호 | 값 | 출처 |
|------|-----|------|
| B | batch size | - |
| m | 3 (RGB, Thermal, LiDAR) | num_modalities |
| C_cls | 4 (Static, Dynamic, Water, Sky) | num_classes |
| d_model | 256 | SAM2 config |
| rank | 4 | LoRA rank |
| num_experts | 3 | num_modalities (NOT default 4) |
| fpn[0] | (B, 32, 256, 256) | conv_s0: 256→32 |
| fpn[1] | (B, 64, 128, 128) | conv_s1: 256→64 |
| fpn[2] | (B, 256, 64, 64) | d_model 그대로 |
| vision_feats[0] | (65536, B, 32) | fpn[0] flatten (HW,B,C) |
| vision_feats[1] | (16384, B, 64) | fpn[1] flatten (HW,B,C) |
| vision_feats[2] | (4096, B, 256) | fpn[2] flatten (HW,B,C) |
| Hiera block 내부 x | (B, H, W, C) | 4D tensor (NOT 3D) |
| maskmem | (B, 64, 64, 64) | memory encoder out |

---

# 1. P9 (CrossModalFusionHead + Max-Norm UAMM)

코드: `sam_lora_image_encoder_seg.py:1355` (LoRA_Sam_P9)

## 1.1 P9 Overview Figure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MemorySAM P9 — Full Pipeline                        │
│                                                                         │
│  Input: 3 modalities, each (B, 3, 1024, 1024)                          │
│                                                                         │
│  ══════════════════════════════════════════════════════════════════════  │
│  ║  PHASE 1: Independent Encoding (forward_image × 3)                ║  │
│  ║  ★ Memory 없음, Cross-Modal 상호작용 없음 ★                       ║  │
│  ══════════════════════════════════════════════════════════════════════  │
│                                                                         │
│   RGB (B,3,1024²)      Thermal (B,3,1024²)    LiDAR (B,3,1024²)       │
│        │                     │                      │                   │
│        ▼                     ▼                      ▼                   │
│  ┌───────────┐         ┌───────────┐          ┌───────────┐            │
│  │ Hiera-B+  │         │ Hiera-B+  │          │ Hiera-B+  │            │
│  │+MoE-LoRA  │         │+MoE-LoRA  │          │+MoE-LoRA  │  weight   │
│  │(24 blocks)│         │(24 blocks)│          │(24 blocks)│  공유     │
│  └─────┬─────┘         └─────┬─────┘          └─────┬─────┘            │
│        ▼                     ▼                      ▼                   │
│  ┌───────────┐         ┌───────────┐          ┌───────────┐            │
│  │FPN + conv │         │FPN + conv │          │FPN + conv │  weight   │
│  │ s0/s1     │         │ s0/s1     │          │ s0/s1     │  공유     │
│  └─────┬─────┘         └─────┬─────┘          └─────┬─────┘            │
│        │                     │                      │                   │
│        ▼                     ▼                      ▼                   │
│   backbone_fpn[0]      backbone_fpn[0]        backbone_fpn[0]          │
│   (B,32,256²)          (B,32,256²)            (B,32,256²)              │
│   + vision_feats       + vision_feats         + vision_feats           │
│   (3 levels)           (3 levels)             (3 levels)               │
│        │                     │                      │                   │
│  ══════╪═════════════════════╪══════════════════════╪═══════════════    │
│  ║  PHASE 2: Cross-Modal Weight 산출 (scalar)                      ║   │
│  ══════╪═════════════════════╪══════════════════════╪═══════════════    │
│        │                     │                      │                   │
│        └─────────┬───────────┴──────────┬───────────┘                   │
│                  ▼                      │                                │
│        ┌──────────────────┐             │                                │
│        │CrossModalFusion  │             │                                │
│        │Head (fpn[0]×3)   │             │                                │
│        │→ (B, 3) softmax  │             │                                │
│        └────────┬─────────┘             │                                │
│                 │                       │                                │
│          cross_weights (B,3)            │                                │
│           ┌─────┴─────┐                 │                                │
│           ▼           ▼                 │                                │
│     UAMM용:      AMF용 (보관):          │                                │
│     max-norm     그대로                  │                                │
│     (B, 3)       (B, 3)                │                                │
│           │                             │                                │
│  ═════════╪═════════════════════════════╪═══════════════════════════    │
│  ║  PHASE 3: UAMM + Sequential Tracking                           ║   │
│  ║  ★ Memory Attention은 여기서만 발생 ★                            ║   │
│  ═════════╪═════════════════════════════╪═══════════════════════════    │
│           │                             │                                │
│    ┌──────▼────────────────────────────────────────────────────┐       │
│    │ frame 0 (RGB): is_init=True                               │       │
│    │   vision_feats × score[0]=1.0 → track_step               │       │
│    │   _prepare_memory: NO memory (첫 프레임)                   │       │
│    │   → SAM Decoder → output[0] → maskmem → output_dict      │       │
│    ├───────────────────────────────────────────────────────────┤       │
│    │ frame 1 (Thermal): is_init=False                          │       │
│    │   vision_feats × score[1]≤1.0 → track_step               │       │
│    │   _prepare_memory: KV=RGB maskmem ★ Memory Attn ★         │       │
│    │   → SAM Decoder → output[1] → maskmem → output_dict      │       │
│    ├───────────────────────────────────────────────────────────┤       │
│    │ frame 2 (LiDAR): is_init=False                            │       │
│    │   vision_feats × score[2]≤1.0 → track_step               │       │
│    │   _prepare_memory: KV=RGB+Thermal maskmem ★ Memory Attn ★ │       │
│    │   → SAM Decoder → output[2]                               │       │
│    └──────┬────────────────────────────────────────────────────┘       │
│           │                                                             │
│  ═════════╪═════════════════════════════════════════════════════════    │
│  ║  PHASE 4: AMF (Adaptive Modality Fusion)                       ║   │
│  ═════════╪═════════════════════════════════════════════════════════    │
│           ▼                                                             │
│    m_output = w₀·output[0] + w₁·output[1] + w₂·output[2]             │
│    (B, C_cls, 1024, 1024)                                              │
│                                                                         │
│    w₀,w₁,w₂ = cross_weights (softmax, scalar broadcast)               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1.2 P9 — SoftMoE-LoRA Layer

```
┌────────────────────────────────────────────────────────────────────┐
│  SoftMoE-LoRA Layer (×48 = Q 24개 + V 24개)                        │
│  코드: sam_lola_utils.py:629 (SoftMoE_LoRA_Layer)                  │
│                                                                    │
│  입력: x (B, H, W, C)    ← 단일 모달리티, Hiera backbone 4D       │
│        C ∈ {112, 224, 448, 896} (stage별 상이)                     │
│        H×W: 256²→128²→64²→32² (stage별 축소)                      │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │ Gate: Linear(C → 3)                                      │     │
│  │ → softmax(dim=-1)                                        │     │
│  │ → gate_weights (B, H, W, 3)     per-token routing        │     │
│  │                                                          │     │
│  │ ★ Linear는 마지막 dim에만 적용 → spatial 구조 유지         │     │
│  └────────────┬─────────────────────────────────────────────┘     │
│               │                                                    │
│  ┌────────────▼─────────────────────────────────────────────┐     │
│  │ 3 Expert LoRA Paths (rank=4)                              │     │
│  │                                                           │     │
│  │  Expert 0: A₀(C→4) → B₀(4→C)                            │     │
│  │  Expert 1: A₁(C→4) → B₁(4→C)                            │     │
│  │  Expert 2: A₂(C→4) → B₂(4→C)                            │     │
│  │                                                           │     │
│  │  ★ 3 experts = num_modalities (config null → auto 3)     │     │
│  │  ★ A 초기화: kaiming_uniform (전 버전 공통)               │     │
│  │  ★ B 초기화:                                              │     │
│  │    P9:     zeros  (reset_parameters 기본값)               │     │
│  │    P17/19: kaiming_uniform × 0.01 (expert collapse fix)  │     │
│  └────────────┬─────────────────────────────────────────────┘     │
│               │                                                    │
│  ┌────────────▼─────────────────────────────────────────────┐     │
│  │ Weighted Sum (for loop, sam_lola_utils.py:724-729):       │     │
│  │                                                           │     │
│  │  for i in range(3):                                       │     │
│  │    weight = gate_weights[..., i].unsqueeze(-1)  (B,H,W,1)│     │
│  │    expert_out = Bᵢ(Aᵢ(x))                      (B,H,W,C)│     │
│  │    output += weight × expert_out                          │     │
│  └────────────┬─────────────────────────────────────────────┘     │
│               ▼                                                    │
│  출력: Δ (B, H, W, C)    → Q 또는 V에 additive                    │
│                                                                    │
│  Hiera Block 내 적용 (_SoftMoE_LoRA_qkv, sam_lola_utils.py:749):  │
│    qkv = original_qkv(x)              (B, H*W, 3, nHead, C_head) │
│    ↓ reshape 후 → (B, H, W, 3*C)                                  │
│    qkv[:,:,:, :C]  += MoE_Q(x)        Q에 LoRA 적용               │
│    qkv[:,:,:, C:2C]                    K 변경 없음                 │
│    qkv[:,:,:, -C:]  += MoE_V(x)       V에 LoRA 적용               │
└────────────────────────────────────────────────────────────────────┘
```

## 1.3 P9 — CrossModalFusionHead

```
┌───────────────────────────────────────────────────────────────┐
│  CrossModalFusionHead                                         │
│  코드: sam_lola_utils.py:119                                  │
│                                                               │
│  입력: 3× backbone_fpn[0]                                     │
│    feat_rgb     (B, 32, 256, 256)                             │
│    feat_thermal (B, 32, 256, 256)                             │
│    feat_lidar   (B, 32, 256, 256)                             │
│                                                               │
│  ┌────────────────────────────────────────────────────┐      │
│  │ Compress (공유 weight, 3회 호출)                    │      │
│  │                                                    │      │
│  │  feat → GAP(spatial) → (B, 32)                     │      │
│  │       → Flatten                                    │      │
│  │       → Linear(32 → 64) → ReLU                    │      │
│  │                                                    │      │
│  │  z_rgb (B,64), z_thermal (B,64), z_lidar (B,64)   │      │
│  └────────────┬───────────────────────────────────────┘      │
│               │                                               │
│  ┌────────────▼───────────────────────────────────────┐      │
│  │ Concat: z_cat = [z_rgb; z_thermal; z_lidar]        │      │
│  │                  (B, 192)                           │      │
│  └────────────┬───────────────────────────────────────┘      │
│               │                                               │
│  ┌────────────▼───────────────────────────────────────┐      │
│  │ Compare (MLP)                                      │      │
│  │  Linear(192→64) → ReLU → Linear(64→3, zero-init)  │      │
│  └────────────┬───────────────────────────────────────┘      │
│               │                                               │
│  ┌────────────▼───────────────────────────────────────┐      │
│  │ softmax(logits / τ, dim=1)                         │      │
│  └────────────┬───────────────────────────────────────┘      │
│               ▼                                               │
│  출력: cross_weights (B, 3)    ← scalar per-image             │
│        cross_logits  (B, 3)                                   │
│                                                               │
│  ★ 단일 softmax 1회 호출. UAMM/AMF 모두 이 결과에서 파생      │
│  ★ GAP로 공간정보 소실 → (B, m) 스칼라 가중치만 가능           │
└───────────────────────────────────────────────────────────────┘
```

## 1.4 P9 — UAMM + AMF (동일 softmax에서 분기)

```
┌───────────────────────────────────────────────────────────────┐
│  UAMM + AMF: 동일 cross_weights에서 분기                      │
│                                                               │
│  CrossModalFusionHead                                         │
│       │                                                       │
│       ▼                                                       │
│  cross_weights (B, 3) = softmax 출력                          │
│  예: [0.45, 0.30, 0.25]                                      │
│       │                                                       │
│       ├──────────────────────┐                                │
│       │                      │                                │
│       ▼                      ▼                                │
│  ┌──────────────┐     ┌──────────────┐                       │
│  │ UAMM         │     │ AMF          │                       │
│  │ (Phase 3)    │     │ (Phase 4)    │                       │
│  │              │     │              │                       │
│  │ max-norm:    │     │ 그대로 사용:  │                       │
│  │ w / max(w)   │     │ cross_weights│                       │
│  │              │     │              │                       │
│  │ [1.00,       │     │ [0.45,       │                       │
│  │  0.67,       │     │  0.30,       │                       │
│  │  0.56]       │     │  0.25]       │                       │
│  │              │     │              │                       │
│  │ 적용:        │     │ 적용:        │                       │
│  │ vision_feats │     │ track_step   │                       │
│  │ × scalar     │     │ output ×     │                       │
│  │ broadcast    │     │ scalar       │                       │
│  │ (track 전)   │     │ broadcast    │                       │
│  │              │     │ (track 후)   │                       │
│  └──────────────┘     └──────────────┘                       │
│                                                               │
│  ┌─────────────┬──────────────┬───────────────┐              │
│  │             │ UAMM         │ AMF           │              │
│  ├─────────────┼──────────────┼───────────────┤              │
│  │ 변환        │ max-norm     │ identity      │              │
│  │ shape       │ (B, 3)      │ (B, 3)        │              │
│  │ 값 범위     │ (0,1], max=1 │ (0,1), 합=1  │              │
│  │ 적용 시점   │ track 전     │ track 후      │              │
│  │ 적용 대상   │ vision_feats │ output masks  │              │
│  │ 효과        │ 약한 모달 억제│ 출력 가중합   │              │
│  └─────────────┴──────────────┴───────────────┘              │
└───────────────────────────────────────────────────────────────┘
```

## 1.5 P9 — Memory Attention (track_step 내부)

```
┌───────────────────────────────────────────────────────────────┐
│  Memory Attention — track_step 내부에서만 발생                  │
│  코드: sam2_base.py:497 (_prepare_memory_conditioned_features) │
│                                                               │
│  ★ forward_image()에는 memory 없음 (순수 인코딩)               │
│  ★ track_step() 내부에서만 호출됨                              │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ frame_idx=0 (RGB), is_init=True                     │     │
│  │                                                     │     │
│  │  modulated_vision_feats[-1] (4096,B,256)            │     │
│  │       │                                             │     │
│  │       ▼                                             │     │
│  │  _prepare_memory_conditioned_features()             │     │
│  │  → is_init=True → Memory Attn 스킵                  │     │
│  │  → pix_feat = vision_feats[-1] 그대로               │     │
│  │       │                                             │     │
│  │       ▼                                             │     │
│  │  _forward_sam_heads() → multimasks (B,C,1024²)     │     │
│  │       │                                             │     │
│  │       ▼                                             │     │
│  │  _encode_memory_in_output()                         │     │
│  │  → maskmem (B,64,64,64) + obj_ptr (B,256)          │     │
│  │  → output_dict["cond"][0]에 저장                    │     │
│  └─────────────────────────────────────────────────────┘     │
│                          │                                    │
│                          ▼ output_dict 누적                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ frame_idx=1 (Thermal), is_init=False                │     │
│  │                                                     │     │
│  │  modulated_vision_feats[-1] (4096,B,256)            │     │
│  │       │                                             │     │
│  │       ▼                                             │     │
│  │  _prepare_memory_conditioned_features()             │     │
│  │  ┌──────────────────────────────────────────┐      │     │
│  │  │ ★ Memory Attention 수행 ★                 │      │     │
│  │  │                                          │      │     │
│  │  │ Q: vision_feats[-1] (4096,B,256)         │      │     │
│  │  │ KV: RGB maskmem (4096,B,64) + obj_ptr    │      │     │
│  │  │                                          │      │     │
│  │  │ 4 layers of:                             │      │     │
│  │  │   Self-Attn(Q,Q) with RoPE               │      │     │
│  │  │   Cross-Attn(Q,KV) with RoPE             │      │     │
│  │  │                                          │      │     │
│  │  │ → pix_feat (B, 256, 64, 64)             │      │     │
│  │  └──────────────────────────────────────────┘      │     │
│  │       │                                             │     │
│  │       ▼                                             │     │
│  │  SAM Decoder → multimasks → maskmem 저장            │     │
│  └─────────────────────────────────────────────────────┘     │
│                          │                                    │
│                          ▼ output_dict 누적                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ frame_idx=2 (LiDAR), is_init=False                  │     │
│  │                                                     │     │
│  │  _prepare_memory_conditioned_features()             │     │
│  │  ┌──────────────────────────────────────────┐      │     │
│  │  │ ★ Memory Attention 수행 ★                 │      │     │
│  │  │                                          │      │     │
│  │  │ Q: vision_feats[-1] (4096,B,256)         │      │     │
│  │  │ KV: RGB maskmem + Thermal maskmem (누적)  │      │     │
│  │  │     + 양쪽 obj_ptr                        │      │     │
│  │  └──────────────────────────────────────────┘      │     │
│  │       │                                             │     │
│  │       ▼                                             │     │
│  │  SAM Decoder → multimasks                           │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  Memory 정보 흐름 요약:                                        │
│    RGB ──maskmem──→ Thermal KV                                │
│    RGB ──maskmem──┐                                           │
│    Thermal─maskmem┴→ LiDAR KV                                │
└───────────────────────────────────────────────────────────────┘
```

---

# 2. P17 (MultiScale Aux Decoder + Calibrated Spatial Entropy)

코드: `sam_lora_image_encoder_seg.py:3881` (LoRA_Sam_P17)

## 2.1 P17 Overview Figure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MemorySAM P17 — Full Pipeline                       │
│                                                                         │
│  Input: 3 modalities, each (B, 3, 1024, 1024)                          │
│                                                                         │
│  ══════════════════════════════════════════════════════════════════════  │
│  ║  PHASE 1: Independent Encoding (P9과 동일)                        ║  │
│  ══════════════════════════════════════════════════════════════════════  │
│                                                                         │
│   RGB              Thermal           LiDAR                              │
│    │                  │                 │                                │
│    ▼                  ▼                 ▼                                │
│  Hiera-B++MoE-LoRA (weight 공유, kaiming×0.01 init)                    │
│    │                  │                 │                                │
│    ▼                  ▼                 ▼                                │
│  FPN + conv_s0/s1                                                       │
│    │                  │                 │                                │
│    ▼                  ▼                 ▼                                │
│  fpn[0] (B,32,256²)  fpn[0]            fpn[0]                          │
│  fpn[1] (B,64,128²)  fpn[1]            fpn[1]    ← P9 대비 추가 활용   │
│  fpn[2] (B,256,64²)  fpn[2]            fpn[2]    ← P9 대비 추가 활용   │
│  + vision_feats      + vision_feats    + vision_feats                   │
│    │                  │                 │                                │
│  ══╪══════════════════╪═════════════════╪═══════════════════════════    │
│  ║ PHASE 2: Multi-Scale Aux + Calibrated Spatial Entropy           ║   │
│  ║ ★ P9과 완전히 다른 Phase 2 ★                                    ║   │
│  ══╪══════════════════╪═════════════════╪═══════════════════════════    │
│    │                  │                 │                                │
│    ▼                  ▼                 ▼                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                      │
│  │MultiScale   │ │MultiScale   │ │MultiScale   │                      │
│  │ModalAux     │ │ModalAux     │ │ModalAux     │  독립 3개            │
│  │Decoder[0]   │ │Decoder[1]   │ │Decoder[2]   │  (weight 비공유)     │
│  │             │ │             │ │             │                      │
│  │ 입력:       │ │ 입력:       │ │ 입력:       │                      │
│  │ fpn[0,1,2]  │ │ fpn[0,1,2]  │ │ fpn[0,1,2]  │                      │
│  │             │ │             │ │             │                      │
│  │ 출력:       │ │ 출력:       │ │ 출력:       │                      │
│  │ aux_logits  │ │ aux_logits  │ │ aux_logits  │                      │
│  │(B,4,256²)  │ │(B,4,256²)  │ │(B,4,256²)  │                      │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                      │
│         │               │               │                               │
│         └───────┬───────┴───────┬───────┘                               │
│                 ▼               │                                        │
│                                 │                                        │
│    ┌────────[Warmup 분기]────────────────────────────────────┐          │
│    │                                                         │          │
│    │ epoch < 10:  uniform (1/3, 1/3, 1/3)                   │          │
│    │ epoch 10~14: (1-ramp)×uniform + ramp×entropy            │          │
│    │ epoch ≥ 15:  full entropy                               │          │
│    │                                                         │          │
│    │ [Fix 1] .detach() → aux logits에서 gradient 차단         │          │
│    │ [Fix 2] compute_spatial_entropy_confidence()             │          │
│    │         probs → entropy → 1-entropy/max_ent → softmax   │          │
│    └──────────────────┬──────────────────────────────────────┘          │
│                       ▼                                                  │
│              cross_weights (B, m, 256, 256)  ← ★ SPATIAL ★             │
│                       │                                                  │
│                ┌──────┴──────┐                                           │
│                ▼             ▼                                           │
│          UAMM용:       AMF용 (보관):                                     │
│          spatial       spatial                                           │
│          max-norm      그대로                                            │
│          (B,m,256²)    (B,m,256²)                                       │
│                │                                                         │
│  ══════════════╪══════════════════════════════════════════════════════   │
│  ║  PHASE 3: Spatial UAMM + Sequential Tracking                    ║   │
│  ║  ★ UAMM이 spatial → vision_feats level별 interpolate ★          ║   │
│  ══════════════╪══════════════════════════════════════════════════════   │
│                │                                                         │
│    for frame_idx in [0,1,2]:                                            │
│      spatial_score = uamm_scores[:, frame_idx]    (B, 256, 256)         │
│      for each level:                                                    │
│        score_resized = interpolate(score, (h,w))  (B,1,h,w)            │
│        modulated_feat = feat × score_flat          per-pixel            │
│      track_step(modulated_feats, memory)                                │
│                                                                         │
│    (Memory Attention: P9과 동일한 메커니즘)                               │
│                │                                                         │
│  ══════════════╪══════════════════════════════════════════════════════   │
│  ║  PHASE 4: Spatial AMF                                           ║   │
│  ══════════════╪══════════════════════════════════════════════════════   │
│                ▼                                                         │
│    for i in [0,1,2]:                                                    │
│      wi = interpolate(amf[:, i:i+1], output_size)  (B,1,1024²)         │
│      m_output += output[i] × wi                                        │
│                                                                         │
│    m_output (B, C_cls, 1024, 1024)                                      │
│                                                                         │
│  학습 시 반환: (m_output, m_feat, aux_logits_list)                       │
│  추론 시 반환: (m_output, m_feat)                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.2 P17 — MultiScaleModalAuxDecoder

```
┌───────────────────────────────────────────────────────────────┐
│  MultiScaleModalAuxDecoder (×3 독립, weight 비공유)            │
│  코드: sam_lora_image_encoder_seg.py:2490                     │
│  파라미터: ~53K/modality                                      │
│                                                               │
│  입력: fpn_feats = [fpn[0], fpn[1], fpn[2]]                  │
│    fpn[0]: (B, 32, 256, 256)   high-res spatial detail        │
│    fpn[1]: (B, 64, 128, 128)   mid-level features             │
│    fpn[2]: (B, 256, 64, 64)    semantic context               │
│                                                               │
│  ┌────────────────────────────────────────────────────┐      │
│  │ Projection (각 FPN level → 32ch)                   │      │
│  │                                                    │      │
│  │  fpn[0] → Conv1×1(32→32)+BN+ReLU → (B,32,256²)   │      │
│  │  fpn[1] → Conv1×1(64→32)+BN+ReLU → ×2 upsample   │      │
│  │                                    → (B,32,256²)   │      │
│  │  fpn[2] → Conv1×1(256→32)+BN+ReLU → ×4 upsample  │      │
│  │                                     → (B,32,256²)  │      │
│  └────────────┬───────────────────────────────────────┘      │
│               │                                               │
│  ┌────────────▼───────────────────────────────────────┐      │
│  │ Concat: (B, 96, 256, 256)                          │      │
│  └────────────┬───────────────────────────────────────┘      │
│               │                                               │
│  ┌────────────▼───────────────────────────────────────┐      │
│  │ Decoder                                            │      │
│  │  Conv 3×3(96→48) + BN + ReLU                      │      │
│  │  Conv 1×1(48→4)                                    │      │
│  └────────────┬───────────────────────────────────────┘      │
│               ▼                                               │
│  출력: aux_logits (B, 4, 256, 256)                            │
│                                                               │
│  ★ 학습: aux CE loss로 직접 학습                               │
│  ★ 융합: .detach() 후 entropy 계산 → fusion weight 생성       │
└───────────────────────────────────────────────────────────────┘
```

## 2.3 P17 — Calibrated Spatial Entropy Confidence

```
┌───────────────────────────────────────────────────────────────┐
│  compute_spatial_entropy_confidence()                          │
│  코드: sam_lora_image_encoder_seg.py 내 함수                   │
│                                                               │
│  입력: [aux_logits_0, aux_logits_1, aux_logits_2]             │
│        각 (B, 4, 256, 256)   ← .detach() 적용                │
│                                                               │
│  for each modality i:                                         │
│  ┌────────────────────────────────────────────────────┐      │
│  │ z = aux_logits_i                  (B, 4, H, W)     │      │
│  │                                                    │      │
│  │ probs = softmax(z/T, dim=1)       (B, 4, H, W)     │      │
│  │ log_probs = log_softmax(z/T)      (B, 4, H, W)     │      │
│  │ entropy = -Σ(probs × log_probs)   (B, H, W)        │      │
│  │                                                    │      │
│  │ max_entropy = log(4) ≈ 1.386                       │      │
│  │ confidence = 1 - entropy/max_ent  (B, H, W)        │      │
│  │   0 = 완전 불확실 (4클래스 균등)                     │      │
│  │   1 = 완전 확신 (단일 클래스 100%)                   │      │
│  └────────────┬───────────────────────────────────────┘      │
│               │                                               │
│  ┌────────────▼───────────────────────────────────────┐      │
│  │ stack: (B, 3, H, W)                                │      │
│  │ → softmax(stacked/T, dim=1)                        │      │
│  │ → weights (B, 3, H, W)                             │      │
│  │                                                    │      │
│  │ 의미: 각 위치에서 어떤 모달리티가 가장               │      │
│  │       확신있게 예측하는지를 나타내는 spatial map      │      │
│  └────────────┬───────────────────────────────────────┘      │
│               ▼                                               │
│  출력: weights (B, 3, 256, 256)                               │
│                                                               │
│  ★ Energy Score와의 차이:                                     │
│  │  Energy: logit 크기 기반 → "자신있게 틀려도" 높은 점수      │
│  │  Entropy: 분포 불확실성 → 불확실하면 낮은 confidence       │
│  ★ Gradient 격리: .detach()로 aux→main loss 역전파 차단       │
└───────────────────────────────────────────────────────────────┘
```

## 2.4 P17 — Spatial UAMM + AMF (P9과의 차이)

```
┌───────────────────────────────────────────────────────────────┐
│  Spatial UAMM + AMF (P17/P19 공통 패턴)                       │
│                                                               │
│  cross_weights (B, m, H, W)                                   │
│       │                                                       │
│       ├──────────────────────┐                                │
│       ▼                      ▼                                │
│  ┌──────────────┐     ┌──────────────┐                       │
│  │ Spatial UAMM │     │ Spatial AMF  │                       │
│  │              │     │              │                       │
│  │ max_w =      │     │ for i:       │                       │
│  │  max(dim=1)  │     │   wi = inter-│                       │
│  │  (B,1,H,W)  │     │   polate(    │                       │
│  │              │     │   amf[:,i:   │                       │
│  │ uamm =      │     │   i+1],      │                       │
│  │  w/max_w     │     │   (1024²))   │                       │
│  │  (B,m,H,W)  │     │              │                       │
│  │              │     │ m_output +=  │                       │
│  │ for level:   │     │  output[i]   │                       │
│  │  resize to   │     │  × wi        │                       │
│  │  (h,w)       │     │              │                       │
│  │  flatten     │     │ ★ per-pixel  │                       │
│  │  → (hw,B,1)  │     │   가중합     │                       │
│  │  feat × score│     │              │                       │
│  └──────────────┘     └──────────────┘                       │
│                                                               │
│  P9 vs P17 UAMM/AMF 차이:                                    │
│  ┌──────────┬──────────────────┬──────────────────────┐      │
│  │          │ P9               │ P17 / P19            │      │
│  ├──────────┼──────────────────┼──────────────────────┤      │
│  │ shape    │ (B, m) scalar    │ (B, m, H, W) spatial │      │
│  │ UAMM    │ broadcast × feat │ interpolate per-level │      │
│  │ AMF     │ .view(-1,1,1,1)  │ interpolate to 1024² │      │
│  │ 정보량   │ 이미지당 3값     │ 이미지당 3×256²=196K │      │
│  └──────────┴──────────────────┴──────────────────────┘      │
└───────────────────────────────────────────────────────────────┘
```

---

# 3. P19 (Learned Spatial Cross-Modal Fusion)

코드: `sam_lora_image_encoder_seg.py:4591` (LoRA_Sam_P19)

## 3.1 P19 Overview Figure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MemorySAM P19 — Full Pipeline                       │
│                                                                         │
│  Input: 3 modalities, each (B, 3, 1024, 1024)                          │
│                                                                         │
│  ══════════════════════════════════════════════════════════════════════  │
│  ║  PHASE 1: Independent Encoding (P9/P17과 동일)                    ║  │
│  ══════════════════════════════════════════════════════════════════════  │
│                                                                         │
│   RGB              Thermal           LiDAR                              │
│    │                  │                 │                                │
│    ▼                  ▼                 ▼                                │
│  Hiera-B++MoE-LoRA (weight 공유, kaiming×0.01 init)                    │
│    │                  │                 │                                │
│    ▼                  ▼                 ▼                                │
│  FPN + conv_s0/s1                                                       │
│    │                  │                 │                                │
│    ▼                  ▼                 ▼                                │
│  fpn[0] (B,32,256²)  fpn[0]            fpn[0]                          │
│  fpn[1] (B,64,128²)  fpn[1]            fpn[1]                          │
│  fpn[2] (B,256,64²)  fpn[2]            fpn[2]                          │
│  + vision_feats      + vision_feats    + vision_feats                   │
│    │                  │                 │                                │
│  ══╪══════════════════╪═════════════════╪═══════════════════════════    │
│  ║ PHASE 2: Learned Spatial Fusion Weights                         ║   │
│  ║ ★ Aux decoder 없음, 직접 학습 ★                                 ║   │
│  ══╪══════════════════╪═════════════════╪═══════════════════════════    │
│    │                  │                 │                                │
│    ▼                  ▼                 ▼                                │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │ SpatialCrossModalFusionHead                              │          │
│  │                                                          │          │
│  │  입력: all_fpn_feats[m][3]                               │          │
│  │    (3 modalities × 3 FPN levels)                         │          │
│  │                                                          │          │
│  │  Phase A: Multi-scale Projection (공유)                   │          │
│  │  Phase B: Per-Modal Spatial Context (공유)                │          │
│  │  Phase C: Cross-Modal Comparison                         │          │
│  │                                                          │          │
│  │  출력: cross_weights (B, 3, 256, 256)  ← SPATIAL         │          │
│  └──────────────────────────┬───────────────────────────────┘          │
│                             │                                           │
│                      ┌──────┴──────┐                                    │
│                      ▼             ▼                                    │
│                UAMM용:       AMF용:                                     │
│                spatial       spatial                                    │
│                max-norm      그대로                                     │
│                (B,m,256²)    (B,m,256²)                                │
│                      │                                                  │
│  ════════════════════╪═══════════════════════════════════════════════   │
│  ║  PHASE 3: Spatial UAMM + Sequential Tracking (P17과 동일 패턴) ║   │
│  ════════════════════╪═══════════════════════════════════════════════   │
│                      │                                                  │
│    for frame_idx in [0,1,2]:                                           │
│      spatial_score → interpolate per level → feat × score              │
│      track_step(modulated_feats, memory)                               │
│                      │                                                  │
│  ════════════════════╪═══════════════════════════════════════════════   │
│  ║  PHASE 4: Spatial AMF (P17과 동일 패턴)                        ║   │
│  ════════════════════╪═══════════════════════════════════════════════   │
│                      ▼                                                  │
│    m_output = Σ interpolate(amf[:,i]) × output[i]                      │
│    (B, C_cls, 1024, 1024)                                              │
│                                                                         │
│  반환: (m_output, m_feat)   ← aux 없으므로 항상 2-tuple                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3.2 P19 — SpatialCrossModalFusionHead (핵심 모듈)

```
┌───────────────────────────────────────────────────────────────────────┐
│  SpatialCrossModalFusionHead (~23K params)                            │
│  코드: sam_lola_utils.py:190                                         │
│                                                                       │
│  입력: all_fpn_feats[modal_idx][fpn_level]                            │
│    modal: 3개, fpn_level: 3개 → 총 9개 feature map                    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │ Phase A: Multi-Scale FPN Projection (weight 공유 across m)  │     │
│  │                                                             │     │
│  │ for each modality i:                                        │     │
│  │   fpn[0] (B,32,256²)  → Conv1×1(32→32)+BN+ReLU   ────────→│     │
│  │   fpn[1] (B,64,128²)  → Conv1×1(64→32)+BN+ReLU   →×2 ↑──→│     │
│  │   fpn[2] (B,256,64²)  → Conv1×1(256→32)+BN+ReLU  →×4 ↑──→│     │
│  │                                                             │     │
│  │   concat → fused_i (B, 96, 256, 256)                       │     │
│  └──────────────────────────┬──────────────────────────────────┘     │
│                             │ × 3 modalities                          │
│  ┌──────────────────────────▼──────────────────────────────────┐     │
│  │ Phase B: Per-Modality Spatial Context (weight 공유 across m)│     │
│  │                                                             │     │
│  │ for each modality i:                                        │     │
│  │   fused_i (B, 96, 256²)                                    │     │
│  │   → DWConv 3×3(96, groups=96) + BN + ReLU                  │     │
│  │     (local spatial context: density, edge, illumination)    │     │
│  │   → Conv1×1(96→32) + BN + ReLU                             │     │
│  │   → spatial_i (B, 32, 256, 256)                             │     │
│  └──────────────────────────┬──────────────────────────────────┘     │
│                             │                                         │
│  ┌──────────────────────────▼──────────────────────────────────┐     │
│  │ Phase C: Cross-Modal Spatial Comparison                     │     │
│  │                                                             │     │
│  │ concat([spatial_0, spatial_1, spatial_2], dim=1)             │     │
│  │ → (B, 96, 256, 256)                                        │     │
│  │                                                             │     │
│  │ → Conv1×1(96→64) + BN + ReLU                               │     │
│  │ → DWConv 3×3(64, groups=64) + BN + ReLU                    │     │
│  │   (spatial coherence: 인접 위치 일관성)                      │     │
│  │ → Conv1×1(64→3)  [★ zero-init ★]                           │     │
│  │                                                             │     │
│  │ → softmax(logits/T, dim=1)                                  │     │
│  └──────────────────────────┬──────────────────────────────────┘     │
│                             ▼                                         │
│  출력: weights (B, 3, 256, 256)                                       │
│        logits  (B, 3, 256, 256)                                       │
│                                                                       │
│  ★ zero-init → 학습 초기에 uniform (1/3) per-pixel                    │
│  ★ DWConv → 공간 정보 보존 (GAP와 대비)                               │
│  ★ 모든 proj/context layer가 modality 간 공유                         │
│  ★ aux decoder 불필요 → end-to-end main loss만으로 학습               │
└───────────────────────────────────────────────────────────────────────┘
```

---

# 4. 버전 간 비교 피규어

## 4.1 Phase 2 비교 (핵심 차이)

```
┌───────────────────────────────────────────────────────────────────────┐
│  Phase 2 비교: P9 vs P17 vs P19                                      │
│  (Phase 1, 3, 4는 spatial/scalar 차이 외 동일 구조)                   │
│                                                                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐        │
│  │      P9          │ │      P17         │ │      P19         │        │
│  ├─────────────────┤ ├─────────────────┤ ├─────────────────┤        │
│  │                  │ │                  │ │                  │        │
│  │ fpn[0] only      │ │ fpn[0,1,2]       │ │ fpn[0,1,2]       │        │
│  │ (32ch)           │ │ (32+64+256ch)    │ │ (32+64+256ch)    │        │
│  │    │             │ │    │             │ │    │             │        │
│  │    ▼             │ │    ▼             │ │    ▼             │        │
│  │ CrossModal       │ │ MultiScaleModal  │ │ SpatialCrossModal│        │
│  │ FusionHead       │ │ AuxDecoder ×3    │ │ FusionHead       │        │
│  │                  │ │ (독립)           │ │                  │        │
│  │ GAP → MLP        │ │    │             │ │ Proj → DWConv    │        │
│  │ → softmax        │ │    ▼             │ │ → Compare        │        │
│  │                  │ │ aux_logits ×3    │ │ → softmax        │        │
│  │                  │ │ .detach()        │ │                  │        │
│  │                  │ │    │             │ │                  │        │
│  │                  │ │    ▼             │ │                  │        │
│  │                  │ │ Entropy →        │ │                  │        │
│  │                  │ │ Warmup →         │ │                  │        │
│  │                  │ │ softmax          │ │                  │        │
│  │    │             │ │    │             │ │    │             │        │
│  │    ▼             │ │    ▼             │ │    ▼             │        │
│  │ (B, 3)           │ │ (B, 3, 256²)    │ │ (B, 3, 256²)    │        │
│  │ scalar           │ │ spatial          │ │ spatial          │        │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘        │
│                                                                       │
│  ┌────────────┬──────────────┬───────────────┬──────────────┐        │
│  │            │ P9            │ P17            │ P19           │        │
│  ├────────────┼──────────────┼───────────────┼──────────────┤        │
│  │ FPN 입력   │ fpn[0] only  │ fpn[0,1,2]   │ fpn[0,1,2]   │        │
│  │ 학습 방식  │ end-to-end   │ aux CE(det)  │ end-to-end   │        │
│  │ weight형태 │ scalar (B,m) │ spatial(B,m,H,W)│spatial(B,m,H,W)│    │
│  │ Aux decoder│ 없음         │ 있음 (×3)    │ 없음          │        │
│  │ 추가 loss  │ 없음         │ aux CE (0.3) │ 없음          │        │
│  │ Warmup     │ 없음         │ 10ep+5ep     │ 없음          │        │
│  │ Grad 격리  │ N/A          │ .detach()    │ N/A           │        │
│  │ 파라미터   │ ~15K         │ ~159K(aux)   │ ~23K          │        │
│  │ 반환 형태  │ 2-tuple      │ 3-tuple(train)│ 2-tuple      │        │
│  └────────────┴──────────────┴───────────────┴──────────────┘        │
└───────────────────────────────────────────────────────────────────────┘
```

## 4.2 전체 아키텍처 구성요소 비교

```
┌───────────────────────────────────────────────────────────────────────┐
│  아키텍처 구성요소 상세 비교                                           │
│                                                                       │
│  ┌────────────────────┬───────────┬───────────┬───────────┐          │
│  │ 구성요소            │ P9        │ P17       │ P19       │          │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ Backbone           │ Hiera-B+  │ Hiera-B+  │ Hiera-B+  │          │
│  │  (frozen)          │ (동일)    │ (동일)    │ (동일)    │          │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ SoftMoE-LoRA       │ 3 exp     │ 3 exp     │ 3 exp     │          │
│  │  experts_b init    │ zeros     │ zeros→    │ zeros→    │          │
│  │                    │(기본값)   │kaiming    │kaiming    │          │
│  │                    │           │×0.01 재초기화│×0.01 재초기화│       │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ Fusion Head        │ CrossModal│ (없음)    │ Spatial   │          │
│  │                    │ FusionHead│           │ CrossModal│          │
│  │                    │ GAP+MLP   │           │ FusionHead│          │
│  │                    │           │           │ Conv+DWConv│         │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ Aux Decoder        │ 없음      │ MultiScale│ 없음      │          │
│  │                    │           │ ModalAux  │           │          │
│  │                    │           │ Decoder×3 │           │          │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ Confidence 산출    │ softmax   │ calibrated│ softmax   │          │
│  │                    │ (학습)    │ entropy   │ (학습)    │          │
│  │                    │           │ (계산)    │           │          │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ Weight shape       │ (B,m)     │ (B,m,H,W) │ (B,m,H,W) │          │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ UAMM               │ scalar    │ spatial   │ spatial   │          │
│  │                    │ max-norm  │ max-norm  │ max-norm  │          │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ AMF                │ scalar    │ spatial   │ spatial   │          │
│  │                    │ broadcast │ interp    │ interp    │          │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ 학습 loss          │ main only │ main+aux  │ main only │          │
│  │                    │           │ CE(0.3)   │           │          │
│  ├────────────────────┼───────────┼───────────┼───────────┤          │
│  │ Trainable params   │ ~8.5M     │ ~8.7M     │ ~8.5M     │          │
│  │ (MoE+Head+Decoder) │(MoE+Head) │(MoE+Aux) │(MoE+Head) │          │
│  └────────────────────┴───────────┴───────────┴───────────┘          │
│                                                                       │
│  설계 철학 비교:                                                       │
│  P9:  학습된 scalar fusion (단순, 안정, 공간정보 없음)                 │
│  P17: 계산된 spatial fusion (aux entropy 기반, gradient 격리, warmup)  │
│  P19: 학습된 spatial fusion (end-to-end, aux 없음, 가장 간결)          │
└───────────────────────────────────────────────────────────────────────┘
```

## 4.3 정보 흐름 비교 (한눈에 보기)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

P9 정보 흐름:
                                 scalar (B,3)
  3×img → Hiera+MoE → FPN →─┬─→ CrossModalFH ──→ UAMM(scalar)
                             │                     → track_step
                             │                     → AMF(scalar)
                             └─→ vision_feats ────────┘

P17 정보 흐름:
                                 spatial (B,3,H,W)
  3×img → Hiera+MoE → FPN →─┬─→ AuxDecoder×3 → .detach()
                             │   → entropy_conf ──→ UAMM(spatial)
                             │     (warmup 포함)    → track_step
                             │                      → AMF(spatial)
                             └─→ vision_feats ─────────┘
                                  aux CE loss ←──┘

P19 정보 흐름:
                                 spatial (B,3,H,W)
  3×img → Hiera+MoE → FPN →─┬─→ SpatialCrossFH ─→ UAMM(spatial)
                             │   (fpn[0,1,2]×3)    → track_step
                             │                      → AMF(spatial)
                             └─→ vision_feats ─────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
