---
title: CoRB (Corroboration-Biased Memory Attention) Novelty Defense
tags: [related-work, novelty-defense, corroboration, reliability, bhattacharyya, veto, rbma, sam2, multimodal-segmentation, P32]
created: 2026-07-06
source: [[relatedworks/61_rsgmamba_reliability_self_gated_mamba]]; [[relatedworks/60_primed_attention_logit_bias_threat]]; [[relatedworks/45_sae_additive_logit_entropy_lvlm_nearmiss]]; [[relatedworks/42_attention_logit_bias_novelty_defense]]; [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]]; [[products/P32_CoRB_리포트]]
status: verified-draft
---

# CoRB (Corroboration-Biased Memory Attention) Novelty Defense

> Companion to [[relatedworks/42_attention_logit_bias_novelty_defense]]. That note defends the **RBMA mechanism** (additive pre-softmax bias into SAM2 memory attention). This note defends the **CoRB *signal*** — the P32-B headline — where reliability is redefined from per-modal self-entropy to **cross-modal corroboration** plus a **unique-info veto**. Both are needed: the mechanism note and this note together fence the full P32-B claim.
>
> Skeptical framing up front: neither the corroboration idea nor the Bhattacharyya coefficient nor "protect the fragile modality" is individually new. The claim is a **conjunction**, and it survives only because no single verified work combines all four pillars. If a reviewer collapses any one pillar, the defense degrades to NEAR-MISS. Write accordingly.

## Claim to defend (CoRB, 4 pillars)

**CoRB is novel as the conjunction of:**

1. **Training-free, closed-form** reliability — no learned gate, no trained uncertainty head, no distillation teacher. Reliability is computed at inference from per-modal decoder posteriors.
2. **Posterior-space Bhattacharyya** agreement — reliability = $\mathrm{BC}(p_i, \bar{p}_{-i}) = \sum_c \sqrt{p_i(c)\,\bar{p}_{-i}(c)}$, a similarity between **class-posterior distributions**, not a feature-space cosine or a feature abs-diff.
3. **Leave-one-out consensus over N≥3** modalities — $\bar{p}_{-i} = \mathrm{mean}_{j\neq i}\,p_j$ — a joint consensus, not a pairwise RGB+X comparison.
4. **Additive pre-softmax bias into SAM2 memory cross-attention** repurposed across modalities — $\mathrm{softmax}(QK^\top/\sqrt{d} + \lambda B)V$ — not a multiplicative SSM gate and not hard modality ranking/selection.

Plus a fifth, **standalone** differentiator that no verified work matches:

5. **Unique-info veto** — a training-free, threshold-free "protect-the-dissenter" term $g_i=\mathrm{clamp}(\text{selfent}_i-\max_{j\neq i}\text{selfent}_j,0,1)$, $\;\mathrm{corr\_veto}_i = g_i\cdot\text{selfent}_i+(1-g_i)\cdot\mathrm{corr}_i$, that prevents a confidently-correct minority modality from being punished by consensus.

## Minimal formulation

Per-modal posteriors and leave-one-out consensus (all at inference, `torch.no_grad()`):

$$
p_i = \mathrm{softmax}(D_i(f_i)), \qquad
\bar{p}_{-i}(x) = \frac{1}{N-1}\sum_{j\neq i} p_j(x),
$$
$$
\mathrm{corr}_i(x) = \sum_c \sqrt{p_i(c,x)\,\bar{p}_{-i}(c,x)} \;\in[0,1]
\quad\text{(Bhattacharyya coefficient)}.
$$

Veto (protect a confidently-dissenting modality when others are uninformative):

$$
g_i(x) = \mathrm{clamp}\!\big(\text{selfent}_i(x) - \max_{j\neq i}\text{selfent}_j(x),\,0,\,1\big),
\qquad
B_i(x) = g_i\,\text{selfent}_i + (1-g_i)\,\mathrm{corr}_i .
$$

Injection point is unchanged from RBMA — the bias $B$ is added to memory cross-attention logits before softmax. The **only** learned parameter remains the scalar $\lambda$.

## Closest prior art (ranked, most dangerous first)

| # | Work / id | What it does (verified) | Why it is NOT CoRB | Threat |
|---|---|---|---|---|
| 1 | **RSGMamba** arXiv:2604.12319 → [[relatedworks/61_rsgmamba_reliability_self_gated_mamba]] | Reliability-aware self-gated Mamba for RGB-D/RGB-T seg. Two **learned** gates. | Uncertainty gate **learned** ($g_u=\sigma(\mathrm{MLP}(f))$); consistency gate is **feature-space abs-diff through a learned MLP** ($g_c=\sigma(\mathcal{G}_c([f_{rgb},f_x,|f_{rgb}-f_x|]))$), **not** posterior BC; **pairwise** (RGB+X, never joint N≥3); injection **multiplicative** into SSM C-matrices ($C^{eff}_{rgb}=g_u^{rgb}\cdot(1-g_c)\cdot C_{rgb}$). We beat it on **all four** checkable axes. | **NEAR-MISS — #1 CoRB threat, MUST-CITE** |
| 2 | **MAGIC++** arXiv:2412.16876 | **Training-free** mean-consensus over N modalities ($f_m=\mathrm{Mean}(f_r,f_d,f_l,f_e)$), then $\mathrm{Rank}(\mathrm{Cos}\{\cdot\},f_m)$; no trainable fusion params; deliberately **keeps** the bottom-ranked "fragile" modality. | **Feature cosine**, not posterior Bhattacharyya; **hard top/bottom ranking + selection**, not additive pre-softmax bias; "keep fragile" = missing-modality robustness during training, **not** protecting a confidently-correct dissenter (our veto's rationale). Closest to our training-free-consensus idea. | **NEAR-MISS — SLEEPER #2, MUST-CITE** |
| 3 | **SCRNet** — Pattern Recognition 162:111398 (2025), Zhao/Jin et al., "Resolving semantic conflicts in RGB-T semantic segmentation" | **Learned** Semantic Rectification Module (ViT global-context), pairwise RGB-T. Shares the "conflict" framing only. | Learned, pairwise; no posterior-BC, no veto (**on abstract reading only**). | **ADJACENT — ⚠ UNVERIFIED (paywalled)** |
| 4 | **SAC²-Net** arXiv:2606.25542 | Micro-expression recognition (flow + motion-magnification, 2 modalities), learned CCF. | Wrong domain, wrong modality count, learned; **downgraded** from an earlier draft that mistook it for a veto competitor. | **NOT-A-THREAT (downgraded)** |
| 5 | **EQUISeg** arXiv:2509.24505 → [[relatedworks/62_equiseg_balanced_modality_contributions]] | Learned SGM mutual gating for balanced modality contribution. | Feature-space, **learned**. Cite as learned cross-modal agreement. | NOT-A-THREAT |
| 6 | **Any2Seg** arXiv:2407.11351 | Learned KD + correlation reweighting, modality-agnostic representation. | Feature-space, **learned**; missing-modality robustness. | NOT-A-THREAT |
| 7 | **Bhattacharyya coefficient in segmentation** arXiv:2206.00947 (Drees et al., random-walker weights) | Uses BC as an edge/affinity weight. | Cite as **TOOL provenance**. Do **NOT** claim the coefficient itself. | TOOL-CITE |
| 8 | **PID / "unique information"** — Williams & Beer 2010; arXiv:2512.22102 | Partial information decomposition; the concept of a modality carrying unique information. | Cite as **CONCEPT origin** for the veto. Our veto is a training-free heuristic, **not** a PID computation — say so explicitly. | CONCEPT-CITE |

### Verbatim RSGMamba mechanism (for the rebuttal)

- Uncertainty self-gate: $g_u=\sigma(\mathrm{MLP}(f))$ — **learned**.
- Consistency gate: $g_c=\sigma\!\big(\mathcal{G}_c([f_{rgb},\,f_x,\,|f_{rgb}-f_x|])\big)$ — **feature-space abs-diff, learned MLP**, pairwise.
- Injection: $C^{eff}_{rgb}=g_u^{rgb}\cdot(1-g_c)\cdot C_{rgb}$ — **multiplicative** into the Mamba SSM C-matrices.

These three quotes are the anchor of the CoRB rebuttal: they show RSGMamba is learned, feature-space, pairwise, and multiplicative — CoRB inverts every one of those.

## Surviving differentiators (all checkable at review)

| | Differentiator | Beats |
|---|---|---|
| (a) | **Training-free closed-form** vs learned gate | RSGMamba, SCRNet, EQUISeg, Any2Seg |
| (b) | **Posterior-space Bhattacharyya** vs feature abs-diff / feature cosine | RSGMamba **and** MAGIC++ — **← the single strongest, cleanest discriminator** |
| (c) | **Leave-one-out consensus over N≥3** vs pairwise RGB+X | RSGMamba, SCRNet |
| (d) | **Additive pre-softmax into SAM2 memory attention** vs multiplicative SSM gate / hard ranking | RSGMamba **and** MAGIC++ |
| (e) | **Unique-info veto = training-free protect-the-dissenter** | Unmatched by any verified work |

**Differentiator (b) is the load-bearing one.** RSGMamba's consistency and MAGIC++'s ranking both live in **feature** space; CoRB agreement lives in **class-posterior** space. This is the cleanest single sentence that separates CoRB from both #1 and #2 simultaneously, and it is trivially checkable from the equations. Lead with it.

## Forbidden claims (each falsified — do NOT write)

- "First to use cross-modal consensus / corroboration for reliability" — **MAGIC++** (training-free mean-consensus) and **RSGMamba** (learned consistency gate) precede us.
- "First reliability-aware multimodal fusion" — a whole cluster precedes us ([[relatedworks/40_uncertainty_reliability_fusion_relatedwork]]).
- "First to protect the weaker/fragile modality" — **MAGIC++** already deliberately keeps the bottom-ranked modality (different rationale, but a reviewer will cite it).
- "We invented the Bhattacharyya coefficient / a novel divergence" — it is a textbook coefficient (arXiv:2206.00947 uses it in segmentation).
- "We compute unique information / PID" — the veto is a heuristic, not a PID computation.
- Any mechanism-level "first additive attention bias" — see the forbidden list in [[relatedworks/42_attention_logit_bias_novelty_defense]] (PRIMED/SAE/SAM2Long).

Defensible wording is always the **conjunction**: *training-free × posterior-space Bhattacharyya × leave-one-out N≥3 consensus × additive pre-softmax into SAM2 memory attention × unique-info veto* — "to our knowledge, as of mid-2026."

## Mandatory citations

Must appear in the CoRB related-work / rebuttal:

1. **RSGMamba** arXiv:2604.12319 — #1 threat; cite and contrast on all 4 axes.
2. **MAGIC++** arXiv:2412.16876 — sleeper #2; cite the training-free-consensus + keep-fragile overlap and distinguish on posterior-BC + injection + veto rationale.
3. **SCRNet** (PR 162:111398, 2025) — adjacent "conflict" framing; cite with the paywall caveat below.
4. **EQUISeg** arXiv:2509.24505, **Any2Seg** arXiv:2407.11351 — learned cross-modal agreement.
5. **Bhattacharyya-in-seg** arXiv:2206.00947 — tool provenance.
6. **PID / unique information** — Williams & Beer 2010; arXiv:2512.22102 — concept origin for the veto.
7. Mechanism-axis threats inherited from [[relatedworks/42_attention_logit_bias_novelty_defense]]: **PRIMED** (2605.07154), **SAE** (2603.16558), **SAM2Long** (2410.16268).

## Open verification debts (TODO before submission)

- [ ] **SCRNet full text is paywalled (ScienceDirect 403).** The "no posterior-BC, no veto" reading rests on the **abstract only**. Pull the PDF and re-read the Semantic Rectification Module before final submission; if it turns out to operate on posteriors, re-rank it upward. **UNVERIFIED until then.**
- [ ] **Re-sweep arXiv (last 6 months) at each submission.** Training-free consensus + reliability is an active area (RSGMamba is Apr-2026, MAGIC++ surfaced late). The conjunction claim is a universal negative and can only be *maintained*, never *proved*.
- [ ] Verify MAGIC++ keeps the fragile modality at **train** time (robustness) vs **inference** — our veto is an inference-time protect-the-dissenter; confirm the distinction holds against the PDF.
- [ ] Confirm RSGMamba never forms a joint N≥3 consensus anywhere in the paper (checked: pairwise RGB+X in the verified PDF; re-confirm on any v-bump).

## Verdict

**NOVEL as a conjunction.** No single verified work combines the 4 pillars, and the unique-info veto (pillar 5) is unmatched. But the margin is *thin and axis-by-axis*: RSGMamba matches 0/4 pillars yet occupies the same "reliability-aware fusion" slot, and MAGIC++ matches the training-free-consensus pillar outright. **Do not headline "corroboration" or "consensus" as the novelty** — headline the **conjunction**, and lead the discriminator argument with **posterior-space Bhattacharyya** (differentiator b). Cite RSGMamba and MAGIC++ pre-emptively; a rebuttal that omits either invites a scoop-call.

## Links

- [[relatedworks/60_primed_attention_logit_bias_threat]]
- [[relatedworks/61_rsgmamba_reliability_self_gated_mamba]]
- [[relatedworks/45_sae_additive_logit_entropy_lvlm_nearmiss]]
- [[relatedworks/42_attention_logit_bias_novelty_defense]]
- [[relatedworks/40_uncertainty_reliability_fusion_relatedwork]]
- [[relatedworks/62_equiseg_balanced_modality_contributions]]
- [[products/P32_CoRB_리포트]] · [[00_P32_CoRB_index]] · [[synthesis/P32_CoRB_novelty_risk_register]]
