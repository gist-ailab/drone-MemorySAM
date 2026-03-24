# Architecture Figures — P9, P17, P19, P20, P21, P22, P23, P24, P25, P26

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

---

## 7. P20: Per-Layer Independent MLP Gate + Higher Rank MoE

### 7.1 P20 Overview Figure

P9와 동일한 4-Phase 구조. 유일한 차이: SoftMoE-LoRA **V2** (per-layer MLP gate, rank=8).

```
┌─────────────────────────────────────────────────────────────────────┐
│ LoRA_Sam_P20 — Per-Layer Independent MLP Gate + Higher Rank MoE    │
│                                                                     │
│  ┌─── Phase 1: Independent Encoding (NO memory) ──────────────┐    │
│  │                                                             │    │
│  │  RGB ─┐                                                     │    │
│  │  THR ─┤─→ forward_image() ─→ _prepare_backbone_features()  │    │
│  │  LID ─┘   [Hiera + SoftMoE-LoRA V2]   [FPN conv_s0/s1]    │    │
│  │            (rank=8, MLP gate)                               │    │
│  │                                                             │    │
│  │  출력 per modality:                                         │    │
│  │    image_embedding['backbone_fpn'][0]: (B, 32, 256, 256)    │    │
│  │    vision_feats[3]: [(1,B,32), (HW,B,64), (HW,B,256)]      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│  ┌─── Phase 2: Cross-Modal Weight (P9 동일) ──────────────────┐    │
│  │  fpn[0]×3 → CrossModalFusionHead → softmax → (B, 3)        │    │
│  │            scalar 가중치                                     │    │
│  │  UAMM: max-norm → best=1.0                                  │    │
│  │  AMF:  raw softmax                                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│  ┌─── Phase 3: UAMM + Sequential Tracking (P9 동일) ─────────┐    │
│  │  vision_feats × uamm_score → track_step()                   │    │
│  │  [frame 0: no memory]  [frame 1,2: memory attention]        │    │
│  │  → high_res_multimasks per modality                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│  ┌─── Phase 4: AMF Output Fusion (P9 동일) ───────────────────┐    │
│  │  m_output = Σ output[i] × amf_w[i]                          │    │
│  │  m_feat   = Σ fpn[0][i] × amf_w[i]                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                      │
│  Output: (m_output, m_feat)                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 SoftMoE-LoRA V2 Module (P20 신규)

P9 V1과의 차이: gate가 `Linear(C→E)` → `MLP(C→C//4→E)`, per-layer 독립.

```
┌──────────────────────────────────────────────────────────────┐
│ SoftMoE_LoRA_Layer_V2 (per-layer 독립 인스턴스)              │
│                                                              │
│  Input: x (B, H, W, C)   [Hiera 4D format]                  │
│         C ∈ {112, 224, 448, 896} (stage별)                   │
│                                                              │
│  ┌─── MLP Gate (per-layer 독립) ─────────┐                   │
│  │  x → Linear(C → C//4) → ReLU          │                   │
│  │    → Linear(C//4 → 3) → softmax       │                   │
│  │    → gate_weights (B,H,W,3)            │                   │
│  └────────────────────────────────────────┘                   │
│         │                                                     │
│  ┌─── Expert 0 ──────────┐  ┌─── Expert 1 ──────────┐       │
│  │  a: Linear(C→8)       │  │  a: Linear(C→8)       │       │
│  │  b: Linear(8→C)       │  │  b: Linear(8→C)       │       │
│  │  out = b(a(x))        │  │  out = b(a(x))        │       │
│  └──────────┬────────────┘  └──────────┬────────────┘       │
│             │                           │                     │
│  ┌─── Expert 2 ──────────┐              │                    │
│  │  a: Linear(C→8)       │              │                    │
│  │  b: Linear(8→C)       │              │                    │
│  └──────────┬────────────┘              │                    │
│             │                           │                     │
│  Σ gate_weights[...,i] × expert_out[i]                       │
│  = final_output (B, H, W, C)                                 │
│                                                              │
│  Injection: qkv[:,:,:,:C]  += V2_q(x)  (Q에 적용)           │
│             qkv[:,:,:,-C:] += V2_v(x)  (V에 적용)           │
└──────────────────────────────────────────────────────────────┘

V1 vs V2 비교:
┌──────────────────────┬──────────────────────┐
│ SoftMoE-LoRA V1 (P9) │ SoftMoE-LoRA V2 (P20)│
├──────────────────────┼──────────────────────┤
│ gate = Linear(C→E)   │ gate = MLP(C→C//4→E) │
│ rank = 4             │ rank = 8             │
│ gate 공유 가능        │ per-layer 독립 gate  │
│ 3 experts            │ 3 experts            │
│ experts_b init: 0    │ experts_b init: 0    │
└──────────────────────┴──────────────────────┘
```

---

## 8. P21: P9 + DeBA-FP (fpn[0] only, Phase 2)

### 8.1 P21 Overview Figure

P9 base + **DeBA-FP** 모듈 추가. fpn[0]만 Phase 2에서 deformable conv로 refine.

```
┌─────────────────────────────────────────────────────────────────────┐
│ LoRA_Sam_P21 — P9 + DeBA-FP (Deformable Bottleneck Adapter)       │
│                                                                     │
│  ┌─── Phase 1: Independent Encoding (P9 동일) ───────────────┐     │
│  │  RGB/THR/LID → forward_image() [Hiera + SoftMoE-LoRA V1]  │     │
│  │             → _prepare_backbone_features() [FPN]           │     │
│  │  출력: fpn[0] (B,32,256,256), vision_feats, etc.           │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 2: DeBA-FP → Cross-Modal Weight ─────────────────┐     │
│  │                                                             │     │
│  │  fpn[0]_RGB ─→ DeBAFP(mod_idx=0) ─→ refined_fpn[0]_RGB    │     │
│  │  fpn[0]_THR ─→ DeBAFP(mod_idx=1) ─→ refined_fpn[0]_THR ──┤     │
│  │  fpn[0]_LID ─→ DeBAFP(mod_idx=2) ─→ refined_fpn[0]_LID   │     │
│  │                   [공유 DCM/W_d/W_u/LN, α만 per-modal]    │     │
│  │                                                             │     │
│  │  refined_fpn[0]×3 → CrossModalFusionHead → softmax (B,3)   │     │
│  │  UAMM: max-norm, AMF: raw softmax                          │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 3: UAMM + Tracking (P9 동일) ────────────────────┐     │
│  │  vision_feats × uamm_score → track_step()                   │     │
│  │  ※ vision_feats는 원본 (DeBA-FP 미적용)                    │     │
│  │  ※ DeBA-FP는 fpn[0]만, CrossModalFusionHead 입력만 영향    │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 4: AMF Output Fusion ────────────────────────────┐      │
│  │  m_output = Σ output[i] × amf_w[i]                         │     │
│  │  m_feat   = Σ refined_fpn[0][i] × amf_w[i]                 │     │
│  │  ※ m_feat는 DeBA-FP refined된 fpn[0] 사용                  │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Output: (m_output, m_feat)                                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.2 DeBA-FP Module (P21)

```
┌──────────────────────────────────────────────────────────────┐
│ DeBAFP — Deformable Bottleneck Adapter for Feature Pyramid   │
│                                                              │
│  Input: feat (B, C, H, W)    C=32 (fpn[0])                  │
│         modality_idx: 0/1/2                                  │
│                                                              │
│  ┌─── Shared (cross-modal weight sharing) ───────────┐       │
│  │                                                    │       │
│  │  feat ─→ W_d: Conv2d(32→64, 1×1) ─→ h (B,64,H,W) │       │
│  │                                                    │       │
│  │  h ─→ offset_mask_conv(64→27, 3×3) ─→ om          │       │
│  │       offset = om[:,:18]  (B,2K²,H,W)             │       │
│  │       mask   = om[:,18:].sigmoid()  (B,K²,H,W)    │       │
│  │       K=3, K²=9, 3K²=27                           │       │
│  │                                                    │       │
│  │  h ─→ deform_conv2d(h, offset, dcm_weight, mask)  │       │
│  │       dcm_weight: (64,64,3,3)  ─→ h (B,64,H,W)   │       │
│  │                                                    │       │
│  │  h ─→ permute(0,2,3,1) → LN(64) → GELU           │       │
│  │    → permute(0,3,1,2) ─→ h (B,64,H,W)            │       │
│  │                                                    │       │
│  │  h ─→ W_u: Conv2d(64→32, 1×1) ─→ h (B,32,H,W)   │       │
│  │                                                    │       │
│  └────────────────────────────────────────────────────┘       │
│                                                              │
│  Output = feat + α[modality_idx] × h                         │
│           α init=0 → identity at start                       │
│                                                              │
│  공유 구조:                                                   │
│  ┌────────────┬──────────────┬───────────────┐               │
│  │ Shared     │ Per-modality │ 비고          │               │
│  ├────────────┼──────────────┼───────────────┤               │
│  │ W_d, W_u   │ α (scalar)   │ α init=0     │               │
│  │ DCM weight │              │ offset init=0│               │
│  │ offset_conv│              │               │               │
│  │ LayerNorm  │              │               │               │
│  └────────────┴──────────────┴───────────────┘               │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. P22: P9 + DeBA-FP MultiScale (all FPN, Phase 1)

### 9.1 P22 Overview Figure

P21과 동일 원리이나 **적용 범위** 다름: 모든 FPN 레벨, Phase 1에서 적용.

```
┌─────────────────────────────────────────────────────────────────────┐
│ LoRA_Sam_P22 — P9 + DeBA-FP MultiScale (all FPN levels, Phase 1)  │
│                                                                     │
│  ┌─── Phase 1: Encoding + DeBA-FP (ALL FPN levels) ──────────┐     │
│  │                                                             │     │
│  │  for each modality i:                                       │     │
│  │    img_emb = forward_image(input[i])                        │     │
│  │    [Hiera + SoftMoE-LoRA V1]                                │     │
│  │                                                             │     │
│  │    ┌── DeBA-FP refine ALL levels ──────────────────────┐    │     │
│  │    │ fpn[0](B,32,256,256) → DeBAFP_MS(mod=i,lv=0)  ──┤    │     │
│  │    │ fpn[1](B,64,128,128) → DeBAFP_MS(mod=i,lv=1)  ──┤    │     │
│  │    │ fpn[2](B,256,64,64)  → DeBAFP_MS(mod=i,lv=2)  ──┤    │     │
│  │    └───────────────────────────────────────────────────┘    │     │
│  │    img_emb['backbone_fpn'] updated in-place                 │     │
│  │                                                             │     │
│  │    _prepare_backbone_features(img_emb)                      │     │
│  │    → vision_feats = REFINED features ★                      │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 2: Cross-Modal Weight (P9 동일) ──────────────────┐     │
│  │  refined_fpn[0]×3 → CrossModalFusionHead → softmax (B,3)   │     │
│  │  ※ fpn[0]는 이미 Phase 1에서 DeBA-FP로 refined             │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 3: UAMM + Tracking ──────────────────────────────┐     │
│  │  REFINED_vision_feats × uamm_score → track_step()          │     │
│  │  ★ 핵심 차이: tracking에도 refined features 전파            │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 4: AMF Output Fusion ────────────────────────────┐      │
│  │  m_output = Σ output[i] × amf_w[i]                         │     │
│  │  m_feat   = Σ refined_fpn[0][i] × amf_w[i]                 │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Output: (m_output, m_feat)                                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 DeBAFP_MultiScale Module (P22)

```
┌──────────────────────────────────────────────────────────────┐
│ DeBAFP_MultiScale — All FPN levels with cross-layer sharing  │
│                                                              │
│  Input: feat (B, C_l, H, W)                                 │
│         modality_idx: 0/1/2                                  │
│         level_idx: 0/1/2                                     │
│                                                              │
│  C_l = {32, 64, 256}  (fpn[0], fpn[1], fpn[2])              │
│                                                              │
│  ┌─── Per-level W_d ─────────────────────────────────┐       │
│  │  W_d_list[level_idx]: Conv2d(C_l→64, 1×1)        │       │
│  │  → h (B, 64, H, W)                                │       │
│  └───────────────────────────────────────────────────┘       │
│                              │                               │
│  ┌─── Shared DCM (cross-level + cross-modal) ────────┐       │
│  │  offset_mask_conv(64→27, 3×3) → offset, mask      │       │
│  │  deform_conv2d(h, offset, dcm_weight, mask)        │       │
│  │  → h (B, 64, H, W)                                │       │
│  └───────────────────────────────────────────────────┘       │
│                              │                               │
│  ┌─── Shared LN + GELU ─────────────────────────────┐       │
│  │  LN(64) → GELU → h (B, 64, H, W)                 │       │
│  └───────────────────────────────────────────────────┘       │
│                              │                               │
│  ┌─── Per-level W_u ─────────────────────────────────┐       │
│  │  W_u_list[level_idx]: Conv2d(64→C_l, 1×1)        │       │
│  │  → h (B, C_l, H, W)                               │       │
│  └───────────────────────────────────────────────────┘       │
│                                                              │
│  Output = feat + α[modality_idx] × h                         │
│                                                              │
│  공유 구조:                                                   │
│  ┌──────────────────────┬──────────────┬────────────────┐    │
│  │ Shared (all lv+mod)  │ Per-level    │ Per-modality   │    │
│  ├──────────────────────┼──────────────┼────────────────┤    │
│  │ DCM (offset+weight)  │ W_d (C_l→64)│ α (scalar)     │    │
│  │ LayerNorm(64)        │ W_u (64→C_l)│                │    │
│  └──────────────────────┴──────────────┴────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 9.3 P21 vs P22 비교

```
┌──────────────────────────────┬──────────────────────────────┐
│ P21: DeBA-FP                 │ P22: DeBA-FP MultiScale      │
├──────────────────────────────┼──────────────────────────────┤
│ fpn[0] only                  │ fpn[0], fpn[1], fpn[2] 전부  │
│ Phase 2에서 적용              │ Phase 1에서 적용              │
│ CrossModalFusionHead +       │ 전체 파이프라인에 refined     │
│ m_feat에만 영향               │ features 전파 ★              │
│ W_d/W_u: Conv2d(32↔64)      │ Per-level W_d/W_u            │
│ 1개 DCM                      │ 1개 shared DCM               │
│ per-modal α                  │ per-modal α                  │
└──────────────────────────────┴──────────────────────────────┘

정보 흐름 차이:

P21:
  forward_image → fpn[0/1/2] → _prepare_backbone  → vision_feats(원본)
                   fpn[0] → DeBAFP → refined_fpn[0] → CrossModalFH
                                                     → m_feat(refined)

P22:
  forward_image → fpn[0/1/2] → DeBAFP_MS(all) → refined_fpn[0/1/2]
                  → _prepare_backbone → vision_feats(REFINED) ★
                  → CrossModalFH(refined fpn[0])
                  → tracking(REFINED vision_feats) ★
                  → m_feat(refined fpn[0])
```

---

## 10. P23: MoE DeBA-BB (LoRA 완전 교체)

### 10.1 P23 Overview Figure

SoftMoE-LoRA를 **MoE_DeBA_BB**로 완전 교체. Backbone 내부 adapter 변경.

```
┌─────────────────────────────────────────────────────────────────────┐
│ LoRA_Sam_P23 — MoE DeBA-BB (Deformable Bottleneck, replaces LoRA) │
│                                                                     │
│  ┌─── Phase 1: Encoding (MoE-DeBA-BB inside Hiera) ──────────┐     │
│  │                                                             │     │
│  │  for each modality i:                                       │     │
│  │    deba_bb.set_modality(i)  ← α[i] 선택                    │     │
│  │    img_emb = forward_image(input[i])                        │     │
│  │    [Hiera + MoE_DeBA_BB]  ← LoRA 없음, DeBA로 교체         │     │
│  │    → _prepare_backbone_features()                           │     │
│  │                                                             │     │
│  │  MoE_DeBA_BB 구조 (모든 Hiera block에 inject):              │     │
│  │    original_qkv(x) → qkv                                   │     │
│  │    delta = deba_bb(x, stage_idx) → (B,H,W,C)               │     │
│  │    qkv[:,:,:,:C]  += delta  (Q에 적용)                      │     │
│  │    qkv[:,:,:,-C:] += delta  (V에 적용)                      │     │
│  │    ※ delta는 Q, V에 동일하게 적용 (DeBA paper 원칙)        │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 2: Cross-Modal Weight (P9 동일) ──────────────────┐     │
│  │  fpn[0]×3 → CrossModalFusionHead → softmax (B,3)           │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 3: UAMM + Tracking (P9 동일) ────────────────────┐     │
│  │  vision_feats × uamm_score → track_step()                   │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 4: AMF (P9 동일) ────────────────────────────────┐      │
│  │  m_output = Σ output[i] × amf_w[i]                         │     │
│  │  m_feat   = Σ fpn[0][i] × amf_w[i]                         │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Output: (m_output, m_feat)                                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 10.2 MoE_DeBA_BB Module (P23 핵심)

```
┌──────────────────────────────────────────────────────────────────┐
│ MoE_DeBA_BB — Mixture-of-Experts Deformable Bottleneck Adapter  │
│ (1개 공유 인스턴스, 모든 Hiera block에서 참조)                   │
│                                                                  │
│  Input: x (B, H, W, C)   C ∈ {112, 224, 448, 896} (stage별)    │
│         stage_idx: 0/1/2/3                                       │
│                                                                  │
│  ┌─── GAP Gate (per-stage) ──────────────────────────────┐       │
│  │  x → mean(H,W) → gap (B, C)                           │       │
│  │    → gates[stage_idx]: Linear(C→2) → (+noise if train) │       │
│  │    → softmax → gate_weights (B, 2)                     │       │
│  └────────────────────────────────────────────────────────┘       │
│         │                                                         │
│  ┌─── Shared Down-projection (per-stage) ────────────────┐       │
│  │  W_d_list[stage_idx]: Linear(C→64) → h (B,H,W,64)     │       │
│  │  → permute(0,3,1,2) → h_4d (B,64,H,W)                │       │
│  └────────────────────────────────────────────────────────┘       │
│         │                                                         │
│  ┌─── Expert 0 (×1 scale) ──┐  ┌─── Expert 1 (×2 scale) ──┐    │
│  │  h_4d → DCM_0             │  │  h_4d → upsample(×2)      │    │
│  │  offset_mask_conv_0(64→27)│  │    → DCM_1                 │    │
│  │  deform_conv2d             │  │    offset_mask_conv_1(→27) │    │
│  │  → e0 (B,64,H,W)         │  │    deform_conv2d            │    │
│  │                           │  │    → downsample(→H,W)      │    │
│  │                           │  │    → e1 (B,64,H,W)         │    │
│  └──────────┬────────────────┘  └──────────┬─────────────────┘    │
│             │                               │                     │
│  combined = Σ gate_w[:,i] × expert_out[i]   (B,64,H,W)           │
│                                                                   │
│  ┌─── Shared Norm + GELU + Up-projection ────────────────┐       │
│  │  → permute(0,2,3,1) → (B,H,W,64)                     │       │
│  │  → LN(64) → GELU                                      │       │
│  │  → W_u_list[stage_idx]: Linear(64→C)                  │       │
│  │  → out (B,H,W,C)                                      │       │
│  └────────────────────────────────────────────────────────┘       │
│                                                                   │
│  Output = α[_modality_idx] × out   (B,H,W,C)                     │
│           α init=0 → identity at start                            │
│                                                                   │
│  Cross-layer weight sharing 구조:                                 │
│  ┌───────────────────┬─────────────────┬──────────────────┐       │
│  │ Shared (all layer)│ Per-stage       │ Per-modality     │       │
│  ├───────────────────┼─────────────────┼──────────────────┤       │
│  │ LayerNorm(64)     │ W_d(C→64)      │ α (scalar)       │       │
│  │ DCM_0 (offset+wt) │ W_u(64→C)      │                  │       │
│  │ DCM_1 (offset+wt) │ gate(C→2)      │                  │       │
│  └───────────────────┴─────────────────┴──────────────────┘       │
│                                                                   │
│  Hiera-B+ block→stage mapping:                                    │
│  Block 0-2: stage 0 (C=112) │ Block 3-5: stage 1 (C=224)        │
│  Block 6-21: stage 2 (C=448)│ Block 22-23: stage 3 (C=896)      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 11. P24: P9 + Spatial Quality Gating (Teacher-Student)

### 11.1 P24 Overview Figure

P9 base + **SpatialQualityGating** → memory modulation. 학습/추론 차이 있음.

```
┌─────────────────────────────────────────────────────────────────────┐
│ LoRA_Sam_P24 — Quality-aware Memory Gating via Decoder Distill.   │
│                                                                     │
│  ┌─── Phase 1: Encoding (P9 동일) ───────────────────────────┐     │
│  │  RGB/THR/LID → forward_image() [Hiera + SoftMoE-LoRA V1]  │     │
│  │             → _prepare_backbone_features() [FPN]           │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 2: Cross-Modal Weight + Quality Map ─────────────┐     │
│  │                                                             │     │
│  │  fpn[0]×3 → CrossModalFusionHead → softmax (B,3)           │     │
│  │  UAMM: max-norm, AMF: raw softmax                          │     │
│  │                                                             │     │
│  │  ┌─ SpatialQualityGating (per modality) ─────────────┐     │     │
│  │  │  fpn[0]_RGB → SQG → q_logit₀ → sigmoid+scale     │     │     │
│  │  │  fpn[0]_THR → SQG → q_logit₁ → quality_map₁      │     │     │
│  │  │  fpn[0]_LID → SQG → q_logit₂ → quality_map₂      │     │     │
│  │  │  quality ∈ [min_quality=0.1, 1.0]                  │     │     │
│  │  └────────────────────────────────────────────────────┘     │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 2.5 (Training ONLY): Teacher → Quality Target ───┐     │
│  │  for each modality i:                                       │     │
│  │    with torch.no_grad():                                    │     │
│  │      teacher_logits = _teacher_decode_single(               │     │
│  │          vision_feats[i], ...)    ← SAM2 decoder, no memory │     │
│  │      CE = cross_entropy(teacher_logits, gt_safe)            │     │
│  │      quality_target = exp(-CE) ∈ (0, 1]                     │     │
│  │      ※ ignore(255) regions → CE=0 → target=1.0             │     │
│  │                                                             │     │
│  │  gate_loss_data = {predicted: q_logits,                     │     │
│  │                    target: quality_targets,                  │     │
│  │                    ignore_mask: (B,1,H,W)}                  │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 3: UAMM + Tracking + Memory Modulation ─────────┐      │
│  │  for each modality frame_idx:                               │     │
│  │    vision_feats × uamm_score → track_step()                 │     │
│  │                                                             │     │
│  │    ┌── Memory Modulation (P24 핵심) ──────────────────┐     │     │
│  │    │  maskmem = track_step_output["maskmem_features"]  │     │     │
│  │    │  q_map = quality_maps[frame_idx]                  │     │     │
│  │    │  q_map_resized = interpolate(q_map, maskmem.size) │     │     │
│  │    │  maskmem_features *= q_map_resized  ★             │     │     │
│  │    │  → 열화 영역 memory 기여 ↓                        │     │     │
│  │    │  → 잘 예측하는 영역 memory 기여 ↑                 │     │     │
│  │    └───────────────────────────────────────────────────┘     │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌─── Phase 4: AMF Output Fusion (P9 동일) ───────────────────┐     │
│  │  m_output = Σ output[i] × amf_w[i]                          │     │
│  │  m_feat   = Σ fpn[0][i] × amf_w[i]                          │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  Output:                                                             │
│    Training:  (m_output, m_feat, gate_loss_data)                     │
│    Inference: (m_output, m_feat)                                     │
└──────────────────────────────────────────────────────────────────────┘
```

### 11.2 SpatialQualityGating Module (P24)

```
┌──────────────────────────────────────────────────────────────┐
│ SpatialQualityGating — Per-pixel Quality Prediction          │
│                                                              │
│  Input: feat (B, C, H, W)  C=32 (fpn[0]), H=W=256           │
│                                                              │
│  ┌─── Conv Head ─────────────────────────────────────┐       │
│  │  Conv2d(32→64, 3×3, pad=1) → ReLU                │       │
│  │  Conv2d(64→64, 3×3, pad=1) → ReLU                │       │
│  │  Conv2d(64→1,  1×1)        → raw logits           │       │
│  │  (bias init=+1.0 → sigmoid≈0.73, optimistic start)│       │
│  └───────────────────────────────────────────────────┘       │
│                              │                               │
│  forward() → raw logits (B, 1, H, W)                         │
│                                                              │
│  logits_to_quality():                                        │
│    quality = sigmoid(logits) × (1-0.1) + 0.1                 │
│    → quality ∈ [0.1, 1.0]   (B, 1, H, W)                    │
│                                                              │
│  Memory Modulation 적용:                                     │
│    maskmem_features *= interpolate(quality, maskmem.size)     │
│    → 고품질 영역: ≈1.0 (memory 유지)                         │
│    → 저품질 영역: ≈0.1 (memory 억제, 완전 제거 방지)         │
└──────────────────────────────────────────────────────────────┘
```

### 11.3 Teacher-Student Distillation Flow (P24 Training)

```
┌──────────────────────────────────────────────────────────────┐
│ P24 Teacher-Student Quality Distillation (Training Only)     │
│                                                              │
│  ┌─── Teacher (frozen, no grad) ─────────────────────┐       │
│  │                                                    │       │
│  │  vision_feats[i] → _forward_sam_heads()            │       │
│  │    (single-frame, no memory, no prompt)             │       │
│  │    → teacher_logits (B, C_cls, H_img, W_img)       │       │
│  │                                                    │       │
│  │  CE = cross_entropy(teacher_logits, gt_mask)        │       │
│  │    (per-pixel, ignore=255 → CE=0)                   │       │
│  │                                                    │       │
│  │  quality_target = exp(-CE)  ∈ (0, 1]                │       │
│  │    (low CE → high quality, ignore → 1.0)            │       │
│  │                                                    │       │
│  │  downsample to (B, 1, fpn_h, fpn_w)                │       │
│  └────────────────────────────────────────────────────┘       │
│                              │                               │
│                     quality_target.detach()                   │
│                              │                               │
│  ┌─── Student ───────────────────────────────────────┐       │
│  │                                                    │       │
│  │  fpn[0] → SpatialQualityGating → q_logit           │       │
│  │                                                    │       │
│  │  Loss = BCE_with_logits(q_logit, quality_target)   │       │
│  │         masked by ignore_mask_fpn                   │       │
│  └────────────────────────────────────────────────────┘       │
│                                                              │
│  Inference: Teacher 불필요, SQG만 실행 → quality_map          │
└──────────────────────────────────────────────────────────────┘
```

---

## 12. P20-P24 비교

### 12.1 정보 흐름 비교

```
P20 (= P9 + V2 MoE-LoRA):
  3×img → Hiera+MoE_V2(MLP gate, rank=8) → FPN
       → CrossModalFH(scalar) → UAMM → track_step → AMF
  (P9와 완전 동일 흐름, adapter 내부만 변경)

P21 (P9 + DeBA-FP on fpn[0]):
  3×img → Hiera+MoE_V1 → FPN → fpn[0] → DeBAFP → refined_fpn[0]
       → CrossModalFH(refined) → UAMM → track_step(원본) → AMF

P22 (P9 + DeBA-FP MultiScale):
  3×img → Hiera+MoE_V1 → FPN → DeBAFP_MS(all) → refined FPN
       → _prepare_backbone → refined vision_feats
       → CrossModalFH(refined) → UAMM → track_step(REFINED★) → AMF

P23 (MoE-DeBA-BB replaces LoRA):
  3×img → Hiera+MoE_DeBA_BB(GAP gate, 2 experts) → FPN
       → CrossModalFH(scalar) → UAMM → track_step → AMF
  (P9와 동일 흐름, backbone adapter 완전 교체)

P24 (P9 + SpatialQualityGating):
  3×img → Hiera+MoE_V1 → FPN
       → CrossModalFH(scalar) + SQG(quality_map)
       → UAMM → track_step + memory_modulation(quality★) → AMF
  (학습: + teacher distillation loss)
```

### 12.2 Component 비교표

```
┌────────┬────────────────┬──────────────┬────────────────┬──────────────┐
│ 모델   │ Backbone       │ FPN Adapter  │ Fusion Head    │ Memory       │
│        │ Adapter        │              │                │ Modulation   │
├────────┼────────────────┼──────────────┼────────────────┼──────────────┤
│ P9     │ SoftMoE-LoRA   │ ─            │ CrossModalFH   │ UAMM(scalar) │
│        │ V1, rank=4     │              │ (scalar)       │              │
├────────┼────────────────┼──────────────┼────────────────┼──────────────┤
│ P20    │ SoftMoE-LoRA   │ ─            │ CrossModalFH   │ UAMM(scalar) │
│        │ V2, rank=8     │              │ (scalar)       │              │
│        │ MLP gate       │              │                │              │
├────────┼────────────────┼──────────────┼────────────────┼──────────────┤
│ P21    │ SoftMoE-LoRA   │ DeBAFP       │ CrossModalFH   │ UAMM(scalar) │
│        │ V1, rank=4     │ fpn[0] only  │ (scalar)       │              │
│        │                │ Phase 2      │                │              │
├────────┼────────────────┼──────────────┼────────────────┼──────────────┤
│ P22    │ SoftMoE-LoRA   │ DeBAFP_MS    │ CrossModalFH   │ UAMM(scalar) │
│        │ V1, rank=4     │ all FPN      │ (scalar)       │              │
│        │                │ Phase 1 ★    │                │              │
├────────┼────────────────┼──────────────┼────────────────┼──────────────┤
│ P23    │ MoE_DeBA_BB    │ ─            │ CrossModalFH   │ UAMM(scalar) │
│        │ 2 experts      │              │ (scalar)       │              │
│        │ GAP gate       │              │                │              │
├────────┼────────────────┼──────────────┼────────────────┼──────────────┤
│ P24    │ SoftMoE-LoRA   │ ─            │ CrossModalFH   │ UAMM(scalar) │
│        │ V1, rank=4     │              │ (scalar)       │ + quality    │
│        │                │              │ + SQG          │ gating ★     │
├────────┼────────────────┼──────────────┼────────────────┼──────────────┤
│ P25    │ SoftMoE-LoRA   │ ─            │ ❌ FH 제거     │ UAMM(spatial)│
│        │ V1, rank=4     │              │ SQG only ★     │ + AMF(spatial)│
│        │                │              │ (triple-duty)  │ + quality ★  │
└────────┴────────────────┴──────────────┴────────────────┴──────────────┘
```

---

## 13. P25: Unified Spatial Quality Fusion (CrossModalFusionHead 제거)

### 13.1 P25 Overview Figure

P24에서 CrossModalFusionHead를 **완전 제거**. SpatialQualityGating 하나로 UAMM + AMF + Memory를 통합 제어.

```
┌───────────────────────────────────────────────────────────────────────┐
│ LoRA_Sam_P25 — Unified Spatial Quality Fusion                        │
│                                                                       │
│  ┌─── Phase 1: Encoding (P9 동일) ──────────────────────────────┐    │
│  │  RGB/THR/LID → forward_image() [Hiera + SoftMoE-LoRA V1]     │    │
│  │             → _prepare_backbone_features() [FPN]              │    │
│  │  출력 per modality:                                           │    │
│  │    fpn[0]: (B, 32, 256, 256)                                  │    │
│  │    vision_feats: [(1,B,32), (HW,B,64), (HW,B,256)]           │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│  ┌─── Phase 2: Spatial Quality Map ──────────────────────────────┐   │
│  │  ★ CrossModalFusionHead 없음 (P24까지의 상수 수렴 문제 해소)   │   │
│  │                                                                │   │
│  │  fpn[0]_RGB → SQG → q_logit₀ → logits_to_quality             │   │
│  │  fpn[0]_THR → SQG → q_logit₁ → quality_map₁                  │   │
│  │  fpn[0]_LID → SQG → q_logit₂ → quality_map₂                  │   │
│  │  quality_maps[i]: (B, 1, 256, 256)  ∈ [0.1, 1.0]             │   │
│  │  (SQG는 P24와 동일 모듈, 가중치 공유)                         │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                              │                                        │
│  ┌─── Phase 2.5 (Training ONLY): Teacher Quality Target ─────────┐  │
│  │  P24와 동일:                                                    │  │
│  │  vision_feats[i] → SAM2 decoder (no memory, no_grad)            │  │
│  │  → CE(teacher_logits, gt) → quality_target = exp(-CE)           │  │
│  │  gate_loss_data = {predicted: q_logits, target: q_targets}      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│  ┌─── Phase 3: Spatial UAMM + Tracking + Memory Modulation ──────┐  │
│  │                                                                 │  │
│  │  ┌── Spatial Max-Norm (P25 핵심 ①) ─────────────────────┐      │  │
│  │  │  q_stack = stack(quality_maps, dim=1)                  │      │  │
│  │  │           → (B, m, 1, H_fpn, W_fpn)                   │      │  │
│  │  │  q_max = q_stack.max(dim=1)  → (B, 1, 1, H, W)       │      │  │
│  │  │  q_uamm_norm = q_stack / q_max  (pixel별 max-norm)    │      │  │
│  │  │  → 각 pixel에서 best modality의 weight = 1.0           │      │  │
│  │  └────────────────────────────────────────────────────────┘      │  │
│  │                                                                 │  │
│  │  for each modality frame_idx:                                   │  │
│  │    q_uamm_i = q_uamm_norm[:, frame_idx]  (B,1,H_fpn,W_fpn)    │  │
│  │                                                                 │  │
│  │    ┌── Per-level Spatial Modulation ─────────────────────┐      │  │
│  │    │  for each level_feat in vision_feats[frame_idx]:     │      │  │
│  │    │    level_feat: (HW, B, C)  [SAM2 format]             │      │  │
│  │    │    q_resized = interpolate(q_uamm_i, (h_l, w_l))    │      │  │
│  │    │    q_flat = q_resized.flatten(2).permute(2,0,1)      │      │  │
│  │    │            → (HW, B, 1)                              │      │  │
│  │    │    modulated = level_feat × q_flat                    │      │  │
│  │    └──────────────────────────────────────────────────────┘      │  │
│  │                                                                 │  │
│  │    → track_step(modulated_vision_feats, ...)                    │  │
│  │                                                                 │  │
│  │    ┌── Memory Modulation (P24 동일) ─────────────────────┐      │  │
│  │    │  maskmem_features *= interpolate(quality_map, size)   │      │  │
│  │    └──────────────────────────────────────────────────────┘      │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│  ┌─── Phase 4: Spatial AMF Output Fusion (P25 핵심 ②) ────────────┐ │
│  │                                                                  │ │
│  │  ┌── Output Fusion ──────────────────────────────────────┐      │ │
│  │  │  q_amf[i] = interpolate(quality_maps[i], (H_out,W_out))│      │ │
│  │  │  q_stack = stack(q_amf, dim=0)  → (m, B, 1, H, W)    │      │ │
│  │  │  q_norm = q_stack / q_stack.sum(dim=0)                 │      │ │
│  │  │  ★ sum-normalize (NOT softmax) across modalities       │      │ │
│  │  │  m_output = Σ q_norm[i] × output[i]                   │      │ │
│  │  └────────────────────────────────────────────────────────┘      │ │
│  │                                                                  │ │
│  │  ┌── Feature Fusion (proto loss용) ───────────────────────┐     │ │
│  │  │  q_feat[i] = interpolate(quality_maps[i], (H_fpn,W_fpn))│     │ │
│  │  │  q_feat_norm = q_feat_stack / sum                       │     │ │
│  │  │  m_feat = Σ q_feat_norm[i] × fpn[0][i]                 │     │ │
│  │  └─────────────────────────────────────────────────────────┘     │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Output:                                                              │
│    Training:  (m_output, m_feat, gate_loss_data)                      │
│    Inference: (m_output, m_feat)                                      │
└───────────────────────────────────────────────────────────────────────┘
```

### 13.2 Quality Map의 Triple-Duty 활용 (P25 핵심 개념)

```
┌──────────────────────────────────────────────────────────────────┐
│ SpatialQualityGating — 하나의 Quality Map, 세 가지 역할          │
│                                                                  │
│  fpn[0]_i (B,32,256,256)                                         │
│      │                                                           │
│      ▼                                                           │
│  ┌─── SQG ───────────────────────────────────────────────┐       │
│  │  Conv2d(32→64, 3×3) → ReLU                           │       │
│  │  Conv2d(64→64, 3×3) → ReLU                           │       │
│  │  Conv2d(64→1,  1×1)  (bias init=+1.0)                │       │
│  │  → q_logit (B, 1, 256, 256)                           │       │
│  │  → sigmoid × 0.9 + 0.1 = quality_map ∈ [0.1, 1.0]    │       │
│  └────────────────────────────────────────────────────────┘       │
│      │                                                           │
│      ├─── ① Spatial UAMM ──────────────────────────────────┐    │
│      │    stack 3 modalities → pixel별 max-norm              │    │
│      │    → vision_feats × q_uamm_norm (per level, per HW)  │    │
│      │    "이 pixel에서 가장 신뢰할 수 있는 modality = 1.0"  │    │
│      │    ★ P9의 scalar UAMM 대비: pixel마다 다른 가중치     │    │
│      │                                                       │    │
│      ├─── ② Spatial AMF ───────────────────────────────────┐│    │
│      │    output resolution으로 interpolate                  ││    │
│      │    → sum-normalize across modalities                  ││    │
│      │    → output × q_norm (pixel별 fusion weight)          ││    │
│      │    ★ P9의 scalar AMF 대비: pixel마다 다른 fusion 비율  ││    │
│      │                                                       ││    │
│      └─── ③ Memory Modulation (P24 동일) ──────────────────┘│    │
│           maskmem_features × quality_map                      │    │
│           → 열화 영역 memory 기여 ↓                           │    │
│           → 고품질 영역 memory 기여 ↑                         │    │
│           ★ 추가 모듈 없이 quality_map 재활용                 │    │
│                                                               │    │
│  학습: Teacher CE → exp(-CE) → quality_target → BCE loss      │    │
│  추론: SQG만 실행 (teacher 불필요)                             │    │
└──────────────────────────────────────────────────────────────────┘
```

### 13.3 Spatial UAMM 상세 — Vision Feats Modulation

```
┌──────────────────────────────────────────────────────────────────┐
│ Spatial UAMM — Per-level, Per-pixel Quality Modulation           │
│                                                                  │
│  Input:                                                          │
│    quality_maps: [q₀, q₁, q₂]  각 (B, 1, 256, 256)             │
│    vision_feats[i]: 3 levels                                     │
│      level 0: (1, B, 32)        feat_size = (1, 1)               │
│      level 1: (HW, B, 64)      feat_size = (h₁, w₁)            │
│      level 2: (HW, B, 256)     feat_size = (h₂, w₂)            │
│                                                                  │
│  Step 1: Pixel-wise Max-Norm                                     │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  q_stack = stack([q₀, q₁, q₂], dim=1)                 │      │
│  │           → (B, 3, 1, 256, 256)                        │      │
│  │                                                        │      │
│  │  q_max = q_stack.max(dim=1)                            │      │
│  │         → (B, 1, 1, 256, 256)                          │      │
│  │                                                        │      │
│  │  q_uamm_norm = q_stack / q_max                         │      │
│  │               → (B, 3, 1, 256, 256)                    │      │
│  │               각 pixel에서 best modality = 1.0          │      │
│  │               나머지 = quality 비율만큼 감쇠             │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  Step 2: Per-level Interpolation + Modulation                    │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  for frame_idx (0=RGB, 1=THR, 2=LID):                  │      │
│  │    q_i = q_uamm_norm[:, frame_idx]  → (B, 1, 256, 256) │      │
│  │                                                        │      │
│  │    for each level_feat:                                 │      │
│  │      (HW, B, C) — SAM2 내부 포맷                        │      │
│  │      h_l, w_l = feat_sizes에서 HW 매칭으로 유추          │      │
│  │                                                        │      │
│  │      q_resized = interpolate(q_i, (h_l, w_l))          │      │
│  │                → (B, 1, h_l, w_l)                       │      │
│  │                                                        │      │
│  │      q_flat = flatten(2).permute(2,0,1)                │      │
│  │             → (HW, B, 1)                                │      │
│  │                                                        │      │
│  │      modulated = level_feat × q_flat                    │      │
│  │                → (HW, B, C)  spatial quality 곱          │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  P9 Scalar UAMM 대비:                                           │
│  ┌────────────────────────────┬────────────────────────────┐     │
│  │ P9: scalar (B, m)          │ P25: spatial (B,m,1,H,W)   │     │
│  │ score = 단일 스칼라         │ score = pixel별 quality     │     │
│  │ 모든 pixel 동일 scale       │ pixel마다 다른 scale ★      │     │
│  │ broadcast (1→HW)           │ interpolate → match level  │     │
│  └────────────────────────────┴────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

### 13.4 Spatial AMF 상세 — Output Fusion

```
┌──────────────────────────────────────────────────────────────────┐
│ Spatial AMF — Per-pixel Output Fusion                            │
│                                                                  │
│  Input:                                                          │
│    quality_maps: [q₀, q₁, q₂]  각 (B, 1, 256, 256)             │
│    output: [out₀, out₁, out₂]  각 (B, C_cls, H_out, W_out)     │
│    all_backbone_feats: [f₀, f₁, f₂]  각 (B, 32, 256, 256)      │
│                                                                  │
│  Step 1: Output Resolution Matching                              │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  for i in range(3):                                     │      │
│  │    q_amf[i] = interpolate(quality_maps[i],              │      │
│  │                           (H_out, W_out))               │      │
│  │             → (B, 1, H_out, W_out)                      │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  Step 2: Sum-Normalize (NOT softmax)                             │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  q_stack = stack(q_amf, dim=0)  → (3, B, 1, H, W)      │      │
│  │  q_norm = q_stack / q_stack.sum(dim=0)                  │      │
│  │          → pixel별 비율 (합=1)                           │      │
│  │                                                        │      │
│  │  ★ softmax가 아닌 sum-normalize                         │      │
│  │    quality가 이미 [0.1, 1.0] 범위이므로                  │      │
│  │    비율로 직접 나눠도 의미 있음                           │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  Step 3: Weighted Fusion                                         │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  m_output = q_norm[0]×out₀ + q_norm[1]×out₁            │      │
│  │           + q_norm[2]×out₂                              │      │
│  │  → (B, C_cls, H_out, W_out)                            │      │
│  │                                                        │      │
│  │  ── Feature Fusion (proto loss용) ──                    │      │
│  │  q_feat[i] = interpolate(quality_maps[i], (256,256))    │      │
│  │  q_feat_norm = q_feat_stack / sum                       │      │
│  │  m_feat = Σ q_feat_norm[i] × fpn[0][i]                 │      │
│  │  → (B, 32, 256, 256)                                   │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  P9 Scalar AMF 대비:                                             │
│  ┌────────────────────────────┬────────────────────────────┐     │
│  │ P9: scalar (B, m)          │ P25: spatial (B,1,H,W)     │     │
│  │ w = softmax(CrossModalFH)  │ w = sum-norm(quality_map)  │     │
│  │ 모든 pixel 동일 weight      │ pixel마다 다른 weight ★    │     │
│  │ broadcast → 곱             │ interpolate → 곱           │     │
│  │ CrossModalFH 필요          │ CrossModalFH 불필요 ★      │     │
│  └────────────────────────────┴────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

### 13.5 P24 vs P25 비교

```
┌───────────────────────────────────┬───────────────────────────────────┐
│ P24                               │ P25                               │
├───────────────────────────────────┼───────────────────────────────────┤
│ CrossModalFusionHead ✅            │ CrossModalFusionHead ❌ 제거       │
│ → scalar (B, m) 상수 수렴         │                                   │
├───────────────────────────────────┼───────────────────────────────────┤
│ SQG → Memory만                    │ SQG → UAMM + AMF + Memory ★      │
│       (1가지 역할)                │       (3가지 역할 = triple-duty)  │
├───────────────────────────────────┼───────────────────────────────────┤
│ UAMM: scalar max-norm (B, m)     │ UAMM: spatial max-norm            │
│ score_expanded → broadcast        │ q_uamm → interpolate per level    │
│ vision_feats × scalar             │ vision_feats × spatial (HW,B,1) ★ │
├───────────────────────────────────┼───────────────────────────────────┤
│ AMF: scalar softmax (B, m)        │ AMF: spatial sum-norm              │
│ w.view(-1,1,1,1) → broadcast     │ q_amf → interpolate → per-pixel ★ │
├───────────────────────────────────┼───────────────────────────────────┤
│ Memory Mod: quality_map × maskmem │ Memory Mod: quality_map × maskmem │
│ (동일)                            │ (동일)                            │
├───────────────────────────────────┼───────────────────────────────────┤
│ 추가 파라미터: CrossModalFH ~3K   │ 추가 파라미터: 0 (FH 제거)        │
│              + SQG ~12.5K         │              + SQG ~12.5K         │
│              = ~15.5K             │              = ~12.5K (↓3K)       │
├───────────────────────────────────┼───────────────────────────────────┤
│ Teacher: CE → exp(-CE)            │ Teacher: CE → exp(-CE) (동일)     │
├───────────────────────────────────┼───────────────────────────────────┤
│ 반환값: (o, f, gate_data)         │ 반환값: (o, f, gate_data) (동일)  │
├───────────────────────────────────┼───────────────────────────────────┤
│ Scoring 근거: FH=상수, Memory만   │ Scoring 근거: 모든 경로가         │
│ adaptive                          │ spatial quality로 adaptive ★      │
└───────────────────────────────────┴───────────────────────────────────┘

정보 흐름 비교:

P24:
  3×img → Hiera+MoE → FPN
       → CrossModalFH(scalar) ─→ UAMM(scalar) → track_step → AMF(scalar)
       → SQG(spatial) ──────────────────→ memory modulation만
  (2개 경로: FH=상수, SQG=spatial — 역할 분리, FH가 상수 수렴)

P25:
  3×img → Hiera+MoE → FPN
       → SQG(spatial) ─┬→ UAMM(spatial) → track_step → AMF(spatial)
                        ├→ memory modulation
                        └→ (teacher loss, training only)
  (1개 경로: SQG 하나가 모든 fusion을 spatial하게 통합 제어 ★)
```

---

## 14. P26: Per-Modality SQG + Multi-Scale + Per-Modal Decoder + Modal-Cond MoE

P25의 6가지 구조적 문제를 해결. 8가지 변경으로 모든 모듈을 모달리티별로 특화.

### 14.1 P26 Overview Figure

```
┌───────────────────────────────────────────────────────────────────────────┐
│ LoRA_Sam_P26 — Full v5: 8 Changes from P25                              │
│                                                                           │
│  ┌─── Phase 1: Encoding ★⑧ Modal-Cond MoE + ⑦ Per-Modal Decoder ────┐  │
│  │                                                                     │  │
│  │  for each modality i (RGB=0, THR=1, LID=2):                        │  │
│  │    ⑧ modal_cond = modal_embed(i)  → (cond_dim=8,)                  │  │
│  │       → set_condition(modal_cond) to all MoE layers                 │  │
│  │    ⑦ _swap_decoder(i) → sam.sam_mask_decoder = per_modal_decoders[i]│  │
│  │                                                                     │  │
│  │    _encode_single_modality(img, i):                                 │  │
│  │      image_encoder(img) → raw_fpn[0,1,2] (all 256ch)               │  │
│  │      ⑦ conv_s0/s1 from per-modal decoder → fpn[0](32ch), fpn[1](64ch)│ │
│  │      → backbone_fpn, vision_pos_enc, raw_fpn(⑥용)                  │  │
│  │                                                                     │  │
│  │    _prepare_backbone_features() → vision_feats, feat_sizes          │  │
│  │                                                                     │  │
│  │  (gradient checkpointing: per-modality 단위)                        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│  ┌─── Phase 2: ①⑥ Per-Modal Multi-Scale SQG ────────────────────────┐   │
│  │                                                                     │  │
│  │  for each modality i:                                               │  │
│  │    ⑥ _fuse_fpn_multiscale(raw_fpn[i]):                             │  │
│  │      raw_fpn[0](256ch) → fpn_proj0 → 32ch ──┐                      │  │
│  │      raw_fpn[1](256ch) → fpn_proj1 → 32ch → resize ─┤ cat          │  │
│  │      raw_fpn[2](256ch) → fpn_proj2 → 32ch → resize ─┘              │  │
│  │      → sqg_input (B, 96, 256, 256)                                  │  │
│  │                                                                     │  │
│  │    ① quality_gatings[i](sqg_input) → q_logit_i  (B, 1, 256, 256)   │  │
│  │       logits_to_quality(q_logit_i) → quality_map_i ∈ [0.3, 1.0]    │  │
│  │       ★ min_quality=0.3 (P25: 0.1) — 연쇄 약화 방지                │  │
│  │       ★ 모달리티별 독립 SQG (P25: 공유 1개)                         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│  ┌─── Phase 2.5 (Training): ③ Relative Quality Teacher + KL Loss ────┐  │
│  │                                                                     │  │
│  │  for each modality i:                                               │  │
│  │    ⑦ _swap_decoder(i) → per-modal decoder로 teacher decoding       │  │
│  │    teacher_logits = _teacher_decode_single(vision_feats[i])         │  │
│  │    CE_i = cross_entropy(teacher_logits, gt_safe)  (B, H, W)         │  │
│  │                                                                     │  │
│  │  ③ ce_stack = stack([CE₀, CE₁, CE₂])  → (3, B, 1, H_fpn, W_fpn)   │  │
│  │     quality_target_dist = softmax(-ce_stack / τ_teacher, dim=0)     │  │
│  │     ★ 절대 quality가 아닌 모달리티 간 상대적 비교                   │  │
│  │     ★ τ_teacher=0.5 (sharpness 조절)                                │  │
│  │                                                                     │  │
│  │  Loss = KL(log_softmax(q_logit_stack / τ_uamm), target_dist)       │  │
│  │  ★ P25의 BCE → KL divergence (분포 간 학습)                         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│  ┌─── Phase 3: ② UAMM Softmax + ⑦ Per-Modal Decoder + ⑤ No MemMod ─┐  │
│  │                                                                     │  │
│  │  ② q_uamm = softmax(q_logit_stack / τ_uamm, dim=modality)          │  │
│  │     → (m, B, 1, H_fpn, W_fpn)   pixel별 softmax (합=1)             │  │
│  │     ★ P25의 max-norm → softmax 교체 (불연속 제거)                   │  │
│  │                                                                     │  │
│  │  for each modality frame_idx:                                       │  │
│  │    ⑦ _swap_decoder(frame_idx)                                       │  │
│  │    per-level interpolation + modulation (P25와 동일 방식):           │  │
│  │      vision_feats × q_uamm_i  → modulated_vision_feats              │  │
│  │                                                                     │  │
│  │    → track_step(modulated_vision_feats, per_modal_decoder[i])       │  │
│  │      [frame 0: no memory]  [frame 1,2: memory attention]            │  │
│  │                                                                     │  │
│  │    ⑤ maskmem 그대로 저장 (Memory Modulation 제거)                   │  │
│  │       ★ UAMM에서 이미 quality-aware → 이중 페널티 방지              │  │
│  │       (config memory_mod=true로 복원 가능)                           │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│  ┌─── Phase 4: ④ AMF Output Entropy (SQG와 독립) ───────────────────┐   │
│  │                                                                     │  │
│  │  ④ for each modality i:                                             │  │
│  │    prob = softmax(output[i], dim=class)  → (B, 4, H_out, W_out)     │  │
│  │    entropy = -Σ(prob × log(prob))  → (B, 1, H_out, W_out)           │  │
│  │    confidence = 1 - entropy / log(4)  ∈ [0, 1]                      │  │
│  │                                                                     │  │
│  │  amf_norm = confidence_stack / sum  → per-pixel fusion weight        │  │
│  │  m_output = Σ amf_norm[i] × output[i]                               │  │
│  │                                                                     │  │
│  │  ★ AMF가 SQG와 독립 — triple-duty 해소                              │  │
│  │  ★ 학습 불필요: output 자체 확신도로 판단                            │  │
│  │  (config amf_mode='quality_map'으로 P25 방식 복원 가능)              │  │
│  │                                                                     │  │
│  │  Feature fusion: q_feat = softmax(q_logit/τ, dim=0)                 │  │
│  │  m_feat = Σ q_feat[i] × fpn[0][i]                                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  Output:                                                                  │
│    Training:  (m_output, m_feat, gate_loss_data)                          │
│    Inference: (m_output, m_feat)                                          │
└───────────────────────────────────────────────────────────────────────────┘
```

### 14.2 P26 8가지 변경 모듈 상세

#### ① Per-Modality SQG + ⑥ Multi-Scale FPN Input

```
┌──────────────────────────────────────────────────────────────────┐
│ Multi-Scale FPN → Per-Modality SQG (변경 ① + ⑥)                │
│                                                                  │
│  Input: raw_fpn (conv_s0/s1 적용 전, 모두 256ch)                │
│                                                                  │
│  ┌─── _fuse_fpn_multiscale(raw_fpn) ─────────────────────┐      │
│  │                                                        │      │
│  │  raw_fpn[0] (B,256,H0,W0) → fpn_proj0: Conv2d(256→32) │      │
│  │                             → f0 (B, 32, H0, W0)      │      │
│  │                                                        │      │
│  │  raw_fpn[1] (B,256,H1,W1) → fpn_proj1: Conv2d(256→32) │      │
│  │                             → interpolate(H0,W0)       │      │
│  │                             → f1 (B, 32, H0, W0)      │      │
│  │                                                        │      │
│  │  raw_fpn[2] (B,256,H2,W2) → fpn_proj2: Conv2d(256→32) │      │
│  │                             → interpolate(H0,W0)       │      │
│  │                             → f2 (B, 32, H0, W0)      │      │
│  │                                                        │      │
│  │  cat([f0, f1, f2], dim=1) → (B, 96, H0, W0)           │      │
│  └────────────────────────────────────────────────────────┘      │
│                              │                                   │
│  ┌─── Per-Modality SQG (3개 독립) ───────────────────────┐      │
│  │  quality_gatings[0](sqg_input) → q_logit_RGB           │      │
│  │  quality_gatings[1](sqg_input) → q_logit_THR           │      │
│  │  quality_gatings[2](sqg_input) → q_logit_LID           │      │
│  │                                                        │      │
│  │  각 SQG: Conv2d(96→64,3×3) → ReLU                     │      │
│  │          Conv2d(64→64,3×3) → ReLU                     │      │
│  │          Conv2d(64→1,1×1)  → logit (B,1,H,W)          │      │
│  │          sigmoid × 0.7 + 0.3 = quality ∈ [0.3, 1.0]   │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
│  P25 vs P26 비교:                                                │
│  ┌────────────────────────┬────────────────────────────┐        │
│  │ P25                    │ P26                        │        │
│  ├────────────────────────┼────────────────────────────┤        │
│  │ SQG 1개 공유           │ SQG 3개 독립 (per-modal)   │        │
│  │ fpn[0] only (32ch)     │ fpn[0,1,2] concat (96ch)   │        │
│  │ min_quality=0.1        │ min_quality=0.3             │        │
│  │ triple-duty (3곳 공유) │ UAMM 전용 (AMF 분리)       │        │
│  │ ~12.5K params          │ ~42K + ~19K proj ≈ 61K     │        │
│  └────────────────────────┴────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

#### ⑧ Modality-Conditioned MoE LoRA Gate

```
┌──────────────────────────────────────────────────────────────────┐
│ Modal-Conditioned SoftMoE-LoRA Gate (변경 ⑧)                    │
│                                                                  │
│  기존 SoftMoE_LoRA_Layer (P9/P25):                               │
│    gate_logits = gate(x)  ← token feature만, 모달리티 무관       │
│    → 상수 수렴 (모든 모달리티에서 동일 routing)                  │
│                                                                  │
│  P26: modality identity embedding 추가                           │
│                                                                  │
│  ┌─── Initialization ────────────────────────────────────┐       │
│  │  modal_embed = nn.Embedding(3, cond_dim=8)             │       │
│  │    RGB=0, THR=1, LID=2                                 │       │
│  │                                                        │       │
│  │  SoftMoE_LoRA_Layer(dim, rank=4, num_experts=3,        │       │
│  │                     cond_dim=8)  ← P12 인프라 재사용   │       │
│  │    gate: Linear(C→3)                                   │       │
│  │    cond_proj: Linear(8→3)  (zero-init)                 │       │
│  └────────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌─── Forward (per modality) ────────────────────────────┐       │
│  │                                                        │       │
│  │  modal_cond = modal_embed(i)  → (8,)                   │       │
│  │  → expand(B, 8) → set_condition() to all MoE layers    │       │
│  │                                                        │       │
│  │  Inside SoftMoE_LoRA_Layer.forward():                  │       │
│  │    gate_logits = gate(x)           → (B,H,W,3)        │       │
│  │    cond_bias = cond_proj(_condition) → (B, 3)          │       │
│  │    gate_logits += cond_bias[...,None,None]  (broadcast) │       │
│  │    gate_weights = softmax(gate_logits)                  │       │
│  │                                                        │       │
│  │  → RGB 처리 시: cond_bias가 RGB 전용 expert 선호 유도  │       │
│  │  → THR 처리 시: cond_bias가 THR 전용 expert 선호 유도  │       │
│  │  → zero-init이므로 초기에는 P25와 동일 동작             │       │
│  └────────────────────────────────────────────────────────┘       │
│                                                                  │
│  핵심: quality가 아닌 modality IDENTITY로 conditioning            │
│  비용: Embedding(3,8) + cond_proj(8,3)×48 ≈ 수백 파라미터       │
└──────────────────────────────────────────────────────────────────┘
```

#### ② UAMM Softmax + ③ Relative Quality Teacher + KL Loss

```
┌──────────────────────────────────────────────────────────────────┐
│ UAMM Softmax 정규화 + Relative Quality Teacher (변경 ②③)       │
│                                                                  │
│  ┌─── ② UAMM: softmax 정규화 ────────────────────────────┐     │
│  │  q_logit_stack: (m, B, 1, H, W) — 3개 SQG raw logits   │     │
│  │                                                         │     │
│  │  q_uamm = softmax(q_logit_stack / τ_uamm, dim=0)       │     │
│  │         → (m, B, 1, H, W)   합=1 보장                   │     │
│  │                                                         │     │
│  │  P25 max-norm:  max=1.0, 나머지=비율 (불연속 전환점)    │     │
│  │  P26 softmax:   합=1.0, smooth 전환 (연속 미분 가능) ★  │     │
│  │                                                         │     │
│  │  τ_uamm=1.0: 기본 (sharp/soft 조절 가능)               │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌─── ③ Relative Quality Teacher (Training) ──────────────┐     │
│  │                                                         │     │
│  │  Teacher per modality:                                  │     │
│  │    CE_i = cross_entropy(teacher_logits_i, gt)           │     │
│  │    → (B, H, W) per-pixel CE loss                        │     │
│  │                                                         │     │
│  │  ce_stack = stack([CE₀, CE₁, CE₂])  → (3, B, 1, H, W)  │     │
│  │  target_dist = softmax(-ce_stack / τ_teacher, dim=0)    │     │
│  │              → (3, B, 1, H, W)  모달리티 간 경쟁 분포    │     │
│  │                                                         │     │
│  │  ★ P25: exp(-CE_i) → 각각 독립, 대부분 ~1.0 (차별 없음) │     │
│  │  ★ P26: softmax(-CE/τ) → 상대적 비교 (차이 증폭)        │     │
│  │                                                         │     │
│  │  예시 (τ=0.5):                                          │     │
│  │    쉬운 pixel: CE=[0.01, 0.02, 0.01] → ~uniform [.34,.32,.34]│  │
│  │    어려운 pixel: CE=[2.0, 0.3, 1.5] → sharp [.01,.97,.02]    │  │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  ┌─── Loss: KL Divergence ────────────────────────────────┐     │
│  │  pred_dist = log_softmax(q_logit_stack / τ_uamm, dim=0)│     │
│  │  loss = KL(pred_dist, target_dist.detach())             │     │
│  │                                                         │     │
│  │  ★ P25 BCE: 각 모달리티 독립 학습 → 관계 무시           │     │
│  │  ★ P26 KL:  분포 간 비교 학습 → 상대적 순서 학습        │     │
│  └─────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

#### ④ AMF Output Entropy (SQG와 독립)

```
┌──────────────────────────────────────────────────────────────────┐
│ AMF Output Entropy — Decoding Confidence 기반 Fusion (변경 ④)   │
│                                                                  │
│  Input: output[i] (B, 4, H_out, W_out) — per-modality 4-class   │
│                                           segmentation logits   │
│                                                                  │
│  ┌─── Per-modality Confidence ────────────────────────────┐      │
│  │  for each modality i:                                   │      │
│  │    prob = softmax(output[i], dim=1)  (B, 4, H, W)       │      │
│  │    entropy = -Σ_c (prob_c × log(prob_c))  (B, 1, H, W)  │      │
│  │    confidence = 1 - entropy / log(4)                     │      │
│  │              ∈ [0, 1]  (high=certain, low=uncertain)     │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                    │
│  ┌─── Sum-Normalize + Fusion ─────────────────────────────┐      │
│  │  amf_stack = stack(confidence, dim=0) → (3, B, 1, H, W) │      │
│  │  amf_norm = amf_stack / sum(dim=0)                       │      │
│  │  m_output = Σ amf_norm[i] × output[i]                    │      │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                  │
│  Triple-duty 해소 핵심:                                          │
│  ┌────────────────────────────────────────────────────────┐      │
│  │ UAMM: SQG quality map → encoding quality (학습)        │      │
│  │ AMF:  output entropy → decoding confidence (학습 불필요)│      │
│  │ Memory: 제거됨 (⑤)                                     │      │
│  │ → 각각 독립적 신호 사용, optimization conflict 없음 ★   │      │
│  └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

#### ⑦ Per-Modality Decoder

```
┌──────────────────────────────────────────────────────────────────┐
│ Per-Modality Decoder — 모달리티별 독립 SAM2 Mask Decoder (변경 ⑦)│
│                                                                  │
│  ┌─── Initialization ────────────────────────────────────┐       │
│  │  per_modal_decoders = ModuleList([                     │       │
│  │    deepcopy(sam_mask_decoder),  # decoder_RGB  ~4M     │       │
│  │    deepcopy(sam_mask_decoder),  # decoder_THR  ~4M     │       │
│  │    deepcopy(sam_mask_decoder),  # decoder_LID  ~4M     │       │
│  │  ])                                                    │       │
│  │  ★ 동일한 pretrained weight에서 시작, 학습하며 분화     │       │
│  └────────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌─── _swap_decoder(modal_idx) ──────────────────────────┐       │
│  │  sam.sam_mask_decoder = per_modal_decoders[modal_idx]   │       │
│  │  ★ track_step 호출 전 swap → 내부적으로 올바른 decoder  │       │
│  └────────────────────────────────────────────────────────┘       │
│                                                                  │
│  적용 위치:                                                       │
│  Phase 1: _encode_single_modality → conv_s0/s1 from decoder[i]   │
│  Phase 2.5: _teacher_decode_single → decoder[i]로 teacher 생성   │
│  Phase 3: track_step → decoder[i]로 decode                       │
│                                                                  │
│  분리/공유 원칙:                                                  │
│  ┌────────────────────┬──────────┬───────────────────────┐       │
│  │ 모듈               │ P26      │ 이유                  │       │
│  ├────────────────────┼──────────┼───────────────────────┤       │
│  │ Mask Decoder       │ 분리 ×3  │ feat→mask 특화        │       │
│  │ Memory Attention   │ 공유 ×1  │ cross-modal 상호참조  │       │
│  │ Memory Encoder     │ 공유 ×1  │ memory bank 포맷 통일 │       │
│  │ Image Encoder      │ 공유 ×1  │ MoE로 이미 특화       │       │
│  └────────────────────┴──────────┴───────────────────────┘       │
│                                                                  │
│  추가 비용: ~8M params (decoder ×2), ~0.13GB VRAM               │
│  Activation: 이미 3회 forward하므로 동일                         │
└──────────────────────────────────────────────────────────────────┘
```

### 14.3 P25 vs P26 비교

```
┌────────────────────────────────┬────────────────────────────────────┐
│ P25                            │ P26                                │
├────────────────────────────────┼────────────────────────────────────┤
│ SQG 1개 공유, fpn[0] 32ch     │ SQG 3개 독립, fpn[0,1,2] 96ch ①⑥ │
│ quality triple-duty (3곳 공유) │ UAMM 전용, AMF 분리 ④            │
│ UAMM max-norm (불연속)        │ UAMM softmax (연속, smooth) ②     │
│ Teacher: exp(-CE) 절대 quality │ Teacher: softmax(-CE/τ) 상대적 ③  │
│ Loss: BCE (독립)               │ Loss: KL divergence (분포 비교) ③ │
│ AMF: SQG quality sum-norm     │ AMF: output entropy confidence ④   │
│ Memory Mod: maskmem × quality  │ Memory Mod: 제거 (이중 페널티) ⑤  │
│ Decoder: 1개 공유              │ Decoder: 3개 독립 (per-modal) ⑦   │
│ MoE gate: input-only           │ MoE gate: + modal embedding ⑧    │
│ min_quality: 0.1               │ min_quality: 0.3                   │
│ 추가 params: ~12.5K            │ 추가 params: ~8M + 61K + ~수백   │
│ VRAM: baseline                 │ VRAM: +~0.13GB                    │
├────────────────────────────────┼────────────────────────────────────┤
│ 1개 경로 (SQG → 3곳)          │ 2개 독립 경로 ★                   │
│                                │ UAMM: SQG → quality (학습)        │
│                                │ AMF: output → entropy (학습 불필요)│
└────────────────────────────────┴────────────────────────────────────┘
```

### 14.4 P26 정보 흐름

```
P26 정보 흐름:

  for each modal i:
    ⑧ modal_embed(i) → set MoE condition
    ⑦ swap decoder[i]

  3×img → Hiera+MoE(⑧modal-cond) → raw_fpn(256ch×3)
       │                             │
       │                             ├→ ⑥ proj+resize+cat → (96ch)
       │                             │      → ① SQG_i (독립×3)
       │                             │        → q_logit_i
       │                             │
       ├→ ⑦ conv_s0/s1 (per-modal)   │
       │   → fpn(32,64,256)          │
       │   → vision_feats            │
       │                             │
       ├─────────────────────────────┘
       │
       ├→ ② UAMM softmax(q_logits/τ)
       │   → vision_feats × q_uamm
       │   → ⑦ track_step(per_modal_decoder[i])
       │      [no memory mod ⑤]
       │   → output[i]
       │
       └→ ④ AMF: entropy(output[i]) → confidence
          → sum-norm → m_output

  Training: ③ Teacher(per_modal_decoder[i]) → CE → softmax(-CE/τ)
            → KL(pred_dist, target_dist)
```
