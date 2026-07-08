---
title: "P32-B — Novelty Risk Register (위험 누락 정리, 양 축 통합)"
tags: [P32, CoRB, RBMA, novelty-defense, risk-register, related-work, threat-watch]
created: 2026-07-06
status: verified-draft
source: "[[relatedworks/49_corb_novelty_defense]]; [[relatedworks/42_attention_logit_bias_novelty_defense]]"
---

# P32-B — Novelty Risk Register (양 축 통합)

> One master table of every prior-art threat across **both** novelty axes of our method: the **RBMA mechanism** (additive pre-softmax bias into SAM2 memory attention), the **RBMA signal** (predictive-entropy reliability), and **CoRB** (cross-modal corroboration + veto). Ranked most-dangerous first. Purpose: a defense that survives review, not a cheerleading doc — so every row names the reviewer's actual objection and what, honestly, still separates us.

## 판정 (읽기 전 3줄 요약)

- **RBMA-signal (predictive entropy) is NOT novel standalone.** Raw per-modal predictive entropy as a reliability score is well-trodden (UTFNet/HyperDUM cluster, [[relatedworks/43_a_signal_entropy_priorart]]). **Do NOT headline the signal by itself.**
- **RBMA-mechanism is NEAR-MISS.** PRIMED occupies the additive-pre-softmax cell; SAE occupies training-free-entropy-additive; SAM2Long occupies the SAM2-memory site. **Claim only the 4-axis conjunction**, never a mechanism-level "first."
- **CoRB is NOVEL as a conjunction** — but **MUST cite RSGMamba (#1) and MAGIC++ (#2 sleeper)**. The cleanest single discriminator is **posterior-space Bhattacharyya** vs everyone else's feature-space agreement.

## Master threat table (most dangerous first)

| # | Component | Threat paper (arXiv/id) | Level | Reviewer's one-line objection | Our surviving differentiator | Citation status | Action |
|---|---|---|---|---|---|---|---|
| 1 | RBMA-mech | **PRIMED** 2605.07154 | **NEAR** | "You add a bias to pre-softmax cross-attention logits — PRIMED already does `Softmax(QKᵀ/√d + b_M)V`." | PRIMED's `b_M=γ·log(P̂/(1−P̂))` is **learned/distilled** (Qwen3-omni teacher), a modality-level **reliance** scalar, at a generic cross-attn site — not training-free, not per-pixel predictive entropy, not SAM2 memory. | MUST-CITE · already-noted-in-42 & 60 | Cite pre-emptively; claim conjunction only |
| 2 | **CoRB** | **RSGMamba** 2604.12319 | **NEAR** | "Reliability-aware fusion with an uncertainty gate + a cross-modal consistency gate already exists." | Their gates are **learned**; consistency = **feature abs-diff MLP** (not posterior BC); **pairwise** RGB+X (not N≥3); injection **multiplicative** into SSM C-matrices (not additive pre-softmax). CoRB beats it on **all 4 axes**. | MUST-CITE · already-noted-in-61 & 70 | Cite + contrast on 4 axes; see [[relatedworks/61_rsgmamba_reliability_self_gated_mamba]] |
| 3 | **CoRB** | **MAGIC++** 2412.16876 | **NEAR (sleeper)** | "Training-free mean-consensus over N modalities that keeps the fragile one — that's your corroboration + veto." | **Feature cosine**, not posterior Bhattacharyya; **hard top/bottom ranking + selection**, not additive pre-softmax; "keep fragile" = missing-modality robustness, not inference-time protect-the-dissenter. | MUST-CITE · already-noted-in-70 | Cite as closest training-free-consensus; distinguish on posterior-BC + injection + veto rationale |
| 4 | RBMA-mech | **SAE** 2603.16558 | **NEAR** | "Training-free additive pre-softmax entropy bias already exists (`S̃=S+λ·SAE·C`)." | Entropy is over the **attention distribution**, not a **predictive decoder softmax**; **unimodal** LVLM self-attention (hallucination), not cross-modal fusion. | MUST-CITE · already-noted-in-42 & 45 | Cite; separate signal axis (attn-entropy vs predictive-entropy) |
| 5 | RBMA-mech | **SAM2Long** 2410.16268 | **NEAR** | "Reliability into SAM2 memory attention is done — this is the SAM2-memory counter-example." | **Multiplicative** key scaling (`M̃=w·M`, w∈[0.95,1.05] from occlusion), not additive pre-softmax; **single-modality temporal**, signal = occlusion score. | MUST-CITE · already-noted-in-42 & 46 | Cite as #1 SAM2-memory counter-example; ablate additive vs multiplicative |
| 6 | **CoRB** | **SCRNet** (PR 162:111398, 2025) | **ADJACENT** | "Resolving semantic conflicts in RGB-T seg already framed the conflict problem." | **Learned** Semantic Rectification Module (ViT global-context), pairwise RGB-T; no posterior-BC, no veto — **on abstract reading only**. | UNVERIFIED (paywalled) | ⚠ Pull PDF before submission; re-rank if it uses posteriors |
| 7 | RBMA-mech | **"Not All Pixels Are Equal"** 2505.02161 | **ADJACENT** | "Additive confidence bias on attention logits (`A=QKᵀ+B`) already exists." | Bias **B is LEARNED** (`B=α(Q⊙W₁)Kᵀ`); single-RGB feature matching, not fusion/seg. **⚠ title/id mismatch on fetch** (returned "Focus What Matters: Matchability-Based Reweighting…"). | UNVERIFIED (id pairing) | ⚠ Re-verify id↔title before citing; keeps mechanism-first claim fenced |
| 8 | RBMA-signal | **UTFNet / HyperDUM** ([[relatedworks/44_hyperdum_uncertainty_fusion_relatedwork]], [[relatedworks/43_a_signal_entropy_priorart]]) | **DIRECT (on signal only)** | "Per-modal predictive uncertainty for reliability is not new." | Correct — **conceded**. Signal is not headlined standalone; it is one pillar of a conjunction. RBMA/CoRB novelty is the mechanism + corroboration, not the raw entropy. | already-noted-in-40/43/44 | Do NOT claim signal novelty; cite as prior for the signal axis |
| 9 | **CoRB** | **EQUISeg** 2509.24505 / **Any2Seg** 2407.11351 | **ADJACENT** | "Cross-modal agreement / balanced-modality gating already exists." | Feature-space, **learned** (SGM mutual gating; KD + correlation reweight). CoRB is training-free, posterior-space, additive-bias. | already-noted-in-62/70 | Cite as learned cross-modal agreement |
| 10 | RBMA-mech | **SISA** 2606.02332 / **ALiBi** 2108.12409 | **NONE (context)** | (none — textbook additive logit bias) | Not reliability, not-a-threat; cite only to show additive-logit-bias is textbook (positional/importance). | context-only | Cite as mechanism template, not a threat |

*(Downgraded — kept for the record: **SAC²-Net** 2606.25542 was earlier drafted as a veto competitor; it is a **micro-expression** paper (flow + motion-mag, 2 modalities), learned CCF → **NOT-A-THREAT**. Do not spend rebuttal space on it.)*

## Tool / concept citations (not threats, but MUST appear)

- **Bhattacharyya coefficient in segmentation** — arXiv:2206.00947 (Drees et al., random-walker weights): cite as **tool provenance**; do not claim the coefficient.
- **PID / "unique information"** — Williams & Beer 2010; arXiv:2512.22102: cite as **concept origin** for the veto; state explicitly the veto is a training-free heuristic, not a PID computation.

## Open verification debts

- **SCRNet PDF paywalled (ScienceDirect 403).** "No posterior-BC, no veto" rests on abstract only → row #6 stays UNVERIFIED until the PDF is pulled.
- **2505.07154? — no; 2505.02161 title/id pairing** (row #7): fetched title was "Focus What Matters: Matchability-Based Reweighting for Local Feature Matching" — the id↔title pairing is UNVERIFIED. Re-verify before citing as a mechanism near-miss.
- **Universal-negative re-sweep.** Both the mechanism-conjunction and the CoRB-conjunction are universal negatives ("no work combines all axes"). Re-sweep arXiv last-6-months at **every** submission; RSGMamba (Apr-2026) and MAGIC++ (late surface) show the cell is actively filling.

## Links

- [[relatedworks/49_corb_novelty_defense]]
- [[relatedworks/42_attention_logit_bias_novelty_defense]]
- [[relatedworks/61_rsgmamba_reliability_self_gated_mamba]]
- [[relatedworks/60_primed_attention_logit_bias_threat]]
- [[relatedworks/45_sae_additive_logit_entropy_lvlm_nearmiss]]
- [[P32_CoRB_리포트]] · [[00_P32_CoRB_index]]
