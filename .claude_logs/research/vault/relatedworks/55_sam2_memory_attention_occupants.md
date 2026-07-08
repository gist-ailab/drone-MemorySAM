---
title: SAM2 Memory-Attention Cross-Modality Occupants — SAM4D, M4-SAM, OmniSAM (RBMA base-claim check)
tags: [related-work, novelty-defense, sam2, memory-attention, sam4d, m4-sam, omnisam, rbma]
created: 2026-07-02
source: arXiv:2506.21547, 2605.11760 (html v1 verified), 2503.07098, 2507.09577 + adversarial verification skeptic1/skeptic2
status: verified-draft
---

# SAM2 memory attention을 modality fusion에 쓰는가 — occupant 전수 점검

Track 1 Q2 (RBMA base-claim direct threat check), 2026-07-02. **결론이 adversarial verification에서 한 번 뒤집혔음** — 원래 "MemorySAM 유일" 주장은 REFUTED. 이 노트가 정본.

## Problem setting

RBMA의 전제: "SAM2 memory attention을 cross-modality fusion에 쓰는 것은 MemorySAM(우리 base)뿐이고, 그 attention logit에 reliability bias를 더한 사례는 없다." 이 중 앞부분의 사실 확인.

## 판정 (2-layer로 분리)

| Cell | Occupants | 상태 |
|---|---|---|
| SAM2-스타일 memory attention을 **cross-modality** fusion에 사용 (base mechanism) | **MemorySAM + SAM4D** | ❌ unoccupied 아님 (2편) |
| 그중 **multimodal semantic segmentation** (DELIVER-류) | **MemorySAM 단독** | ✅ 여전히 단독 — 단, universal negative (search-coverage 한계, skeptic2 confirmed with caveat) |
| 그 memory-attention logits에 reliability bias 주입 | 미발견 | [[relatedworks/52_vfm_multimodal_landscape_synthesis]] Q2b 참조 (DAMM-Diffusion/SAM2Long이 인접 셀 점유) |

## SAM4D (ICCV 2025, arXiv:2506.21547) — 반례, 필수 인용

- "Segment Anything in Camera and LiDAR Streams."
- **MCMA (Motion-aware Cross-modal Memory Attention)**: SAM2-스타일 memory attention이 **카메라와 LiDAR 모달리티를 가로질러** attend. Unified multi-modal positional encoding으로 두 스트림을 정렬. [skeptic1 확인]
- 단 **promptable** (class-agnostic) segmentation — DELIVER-류 semantic segmentation 아님, reliability 신호 없음.
- 위협도: **HIGH watch** — SAM4D 후속이 semantic seg로 확장되면 우리의 narrow cell도 침식. 스쿠프 경보 목록(Track 8) 등재 필요.

## M⁴-SAM (arXiv:2605.11760, 2026-05) [VERIFIED-PDF html v1]

- RGB-D **video** salient object detection (semantic seg 아님).
- Modality-Aware MoE-LoRA가 **인코더에서** 모달리티 fusion (conv experts: RGB/depth/fusion 그룹; Modality Dispatcher D(·)가 입력 모달리티 기준 routing; top-K=2, r=4).
- Memory: Pseudo-Guided Initialization (pseudo mask → 2 linear projections, Eq. 8) + memory cross-attention (Eq. 6) — **temporal context 전용**.
- **핵심 negative (skeptic2 재확인): modality fusion은 encoder MoE-LoRA에서 일어나고 memory attention 내부가 아님; attention logit bias 없음.** MemorySAM 미인용.
- 수치: DViSal E-m 0.925, RDVS 0.927, ViDSOD-100 0.936 / MAE 0.016 [VERIFIED-PDF]. Code: https://github.com/HankLiu2020/M4-SAM

## OmniSAM (ICCV'25, arXiv:2503.07098) [ABSTRACT-ONLY]

- SAM2 memory를 **panorama patch sequence(sub-window) 간** 사용 — cross-FoV(공간), 단일 모달리티. Cross-sensor 아님. [skeptic2 확인]

## 기타 near-miss (전부 non-occupant, skeptic2 adversarial search 재확인)

- **Memory-Augmented SAM2 (2507.09577)**: surgical video, temporal memory only. [ABSTRACT-ONLY]
- **SHIFNet (2503.02581, IROS'25)**: SAM2 기반 RGB-T이나 fusion은 text-guided SACF 모듈 (memory attention 아님) → [[relatedworks/56_sam_family_multimodal_periphery]].
- **SAM-DAQ (2511.09870)**: depth-guided decoder query, memory는 temporal. [ABSTRACT-ONLY]
- **CRISP-SAM2**: text-visual medical — sensor fusion 아님.
- MemorySAM 인용 논문 13편 (S2, 2026-07-02 pull): **memory attention을 수정한 논문 없음.**

## Limitations (of this check)

- Universal negative — S2 인용망 + targeted search 커버리지 내에서만 유효. Google Scholar 쪽 인용 sweep 미완 (Track 8).
- SHIFNet·OmniSAM·SAM-DAQ는 abstract-level 확인 — negative 인용 전 PDF 재확인 권장.

## Improvement directions

- 제출 전 SAM4D 인용 논문 재스캔 (semantic화 후속 여부).
- M⁴-SAM journal version에서 memory 사용이 바뀌는지 감시.

## Comparison to RBMA-P29-P30 (mechanism-class)

- SAM4D: memory cross-attention (mechanism-class 동일 계열) but promptable + no reliability → RBMA와 task·signal 양축에서 구분.
- M⁴-SAM: learned-gate (MoE routing) + temporal memory → RBMA의 logit-additive-bias와 클래스 상이. P29 SDC와는 "modality-ID routing vs condition-latent routing"으로 구분.
- OmniSAM: 공간축 memory — 무관.

## Application to ours (RBMA/P29/P30 적용방향)

1. 논문 novelty 문장에서 "first/only memory-attention modality fusion" 계열 표현 전면 금지 — "among multimodal *semantic* segmentation methods" 한정사 필수 + SAM4D 인용.
2. Related work의 memory-attention 클래스 단락에 MemorySAM → SAM4D → ours 계보로 서술 (아래 후보 단락).
3. M⁴-SAM 문장은 findings 원안 유지 가능: "even recent SAM2 multimodal extensions reserve memory attention for temporal context and fuse modalities in the encoder."

## Related-work paragraph candidate (English)

> A small but growing line of work reinterprets SAM2's memory attention as a fusion mechanism beyond time. MemorySAM treats sensor modalities as frames of the same scene and fuses them through memory attention for multimodal semantic segmentation, while SAM4D introduces motion-aware cross-modal memory attention over camera and LiDAR streams for promptable 4D segmentation. Other SAM2 extensions keep memory strictly temporal: M4-SAM fuses RGB and depth with modality-aware MoE-LoRA experts in the encoder and uses its memory bank only for temporal context in video salient object detection, and OmniSAM shares memory across panoramic sub-windows of a single modality. None of these methods conditions the memory-attention computation itself on sensor reliability; the attention logits remain purely content-based, which is the gap our reliability-biased memory attention addresses.
