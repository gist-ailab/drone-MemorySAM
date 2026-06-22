"""
LoRA_Sam3_RBMA — RBMA (Reliability-Biased Memory Attention) ported to SAM3.

Plan/progress: .claude_logs/11_sam3_rbma_plan.md

Design (decided 2026-06-16):
  - Encoder adaptation = PLAIN LoRA (MemorySAM/`LoRA_Sam` style), injected into SAM3's
    ViT backbone blocks (`*.attn.qkv`). SoftMoE/SQG NOT ported (TODO: input/modality-
    conditioned expert LoRA).
  - RBMA = inject a per-modality reliability bias into the memory cross-attention
    pre-softmax logits via `cross_attn_image(attn_mask=memory_mask)` in
    `sam3/model/encoder.py TransformerEncoderLayer` (the attn_mask arg already exists).
  - Reliability = 1 - normalized predictive entropy of each modality's standalone decode.

This file is built INCREMENTALLY (see plan). Phase 1.1 = _LoRA_qkv + plain-LoRA injector.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.1 — plain LoRA on SAM3 ViT qkv
# ─────────────────────────────────────────────────────────────────────────────
class _LoRA_qkv(nn.Module):
    """Wrap a ViT attention's `qkv` Linear with LoRA on the Q and V projections.

    Shape-agnostic: SAM3's `Attention.forward` calls `self.qkv(x)` then reshapes,
    where x may be (B, L, C) or (B, H, W, C). We only touch the LAST dim (size 3*C),
    so `...` indexing works for both. Mirrors SAM2 `_LoRA_qkv` but rank-agnostic to x.dim().
    """

    def __init__(self, qkv: nn.Module,
                 linear_a_q: nn.Module, linear_b_q: nn.Module,
                 linear_a_v: nn.Module, linear_b_v: nn.Module):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)                              # (..., 3*dim) = [q | k | v]
        new_q = self.linear_b_q(self.linear_a_q(x))    # (..., dim)
        new_v = self.linear_b_v(self.linear_a_v(x))    # (..., dim)
        # non-in-place additive delta (autograd-safe): [Δq | 0 | Δv]
        zero_k = new_q.new_zeros(new_q.shape)
        delta = torch.cat([new_q, zero_k, new_v], dim=-1)
        return qkv + delta


def inject_plain_lora(root: nn.Module, r: int,
                      name_filter=("backbone", "visual", "trunk"),
                      verbose: bool = True):
    """Inject plain LoRA into every ViT-style attention qkv under `root`.

    Targets modules `m` that have `m.qkv: nn.Linear` with out_features == 3*in_features
    (the SAM3 `vitdet.Attention` signature). Restricts to names containing any token in
    `name_filter` so we hit the image ViT backbone and not the mask-decoder / memory-
    fusion attentions (which are MultiheadAttention-style, no 3x-qkv Linear).

    Returns (w_As, w_Bs, n_injected). Freezes `root` params first; LoRA params are new
    and trainable. Caller should keep references (e.g. nn.ModuleList) for state_dict.
    """
    for p in root.parameters():
        p.requires_grad = False

    w_As, w_Bs = [], []
    targets = []
    for name, m in root.named_modules():
        qkv = getattr(m, "qkv", None)
        if not isinstance(qkv, nn.Linear):
            continue
        if qkv.out_features != 3 * qkv.in_features:
            continue
        if name_filter and not any(tok in name for tok in name_filter):
            continue
        targets.append((name, m, qkv))

    for name, m, qkv in targets:
        dim = qkv.in_features
        a_q = nn.Linear(dim, r, bias=False)
        b_q = nn.Linear(r, dim, bias=False)
        a_v = nn.Linear(dim, r, bias=False)
        b_v = nn.Linear(r, dim, bias=False)
        # LoRA init: A ~ kaiming, B = 0  → delta starts at 0 (identity at init)
        nn.init.kaiming_uniform_(a_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(b_q.weight)
        nn.init.zeros_(b_v.weight)
        m.qkv = _LoRA_qkv(qkv, a_q, b_q, a_v, b_v)
        w_As += [a_q, a_v]
        w_Bs += [b_q, b_v]

    if verbose:
        print(f"[inject_plain_lora] injected LoRA(r={r}) into {len(targets)} qkv modules"
              f" (filter={name_filter})")
        for name, _, qkv in targets[:3]:
            print(f"    e.g. {name}.qkv  dim={qkv.in_features}")
    return w_As, w_Bs, len(targets)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.2/1.3 — build SAM3 tracker + inject plain LoRA
# ─────────────────────────────────────────────────────────────────────────────
class SemanticHead(nn.Module):
    """Lightweight conv head: backbone feature (B,C,h,w) -> class logits (B,num_classes,h,w).

    SAM3's mask decoder emits SAM mask tokens (M=1/3), not semantic classes, so — like
    MemorySAM's conv seg heads on SAM2 — we attach a semantic head. Shared across
    modalities. Used for (a) the per-modality predictive-uncertainty reliability signal
    (entropy, standalone features) and (b) the fused semantic output (later).
    """
    def __init__(self, in_channels=256, hidden=128, num_classes=4):
        super().__init__()
        gn = 32 if hidden % 32 == 0 else (16 if hidden % 16 == 0 else 8)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(gn, hidden),   # batch-independent, train==eval (see MultiScaleSemanticHead note)
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, x):
        return self.head(x)


class MultiScaleSemanticHead(nn.Module):
    """FPN-style semantic head fusing 3 feature scales.

    The 1-conv head on the 72x72 (stride-14) feature was the main mIoU bottleneck
    (vs SAM2 Hiera-L, whose hierarchical multi-scale features feed a strong decoder).
    Here we fuse the high-res backbone detail (fpn0 @288, fpn1 @144 — per-modality)
    with the low-res LOW feature (@72) which, at output time, is the MEMORY-CONDITIONED
    feature (so cross-modal fusion + RBMA bias still flow to the prediction). Decodes at
    fpn0 resolution (288) instead of 72, then the caller upsamples to input size.

    forward(f_hi, f_mid, f_low):  f_hi (B,Chi,288,288), f_mid (B,Cmid,144,144),
                                  f_low (B,Clow,72,72) -> logits (B,num_classes,288,288)
    """
    def __init__(self, in_hi=32, in_mid=64, in_low=256, hidden=256, num_classes=25):
        super().__init__()
        self.in_hi, self.in_mid, self.in_low = in_hi, in_mid, in_low
        self.l_hi = nn.Conv2d(in_hi, hidden, 1)
        self.l_mid = nn.Conv2d(in_mid, hidden, 1)
        self.l_low = nn.Conv2d(in_low, hidden, 1)
        # GroupNorm (NOT BatchNorm): the head runs on multiple distinct feature
        # distributions per forward (standalone-reliability vs memory-conditioned-output,
        # x4 modalities). BatchNorm is broken either way: running-stats get polluted (eval
        # collapses) and track_running_stats=False makes eval batch-dependent (proven: SAME
        # ckpt gave val 8.5 in-trainer vs 1.28 standalone). GN has no batch coupling / no
        # running stats → eval deterministic, train==eval, trustworthy.
        gn = 32 if hidden % 32 == 0 else (16 if hidden % 16 == 0 else 8)
        def block(n):                      # n stacked 3x3 conv-GN-ReLU
            layers = []
            for _ in range(n):
                layers += [nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
                           nn.GroupNorm(gn, hidden), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)
        # wider (128->256) + deeper (extra blocks at the high-res level) so small classes
        # survive — the 1-conv@72 head collapsed to the 3 biggest classes (Road/Sky/Building).
        self.smooth_mid = block(2)
        self.smooth_hi = block(3)
        self.classifier = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, f_hi, f_mid, f_low):
        assert f_hi.shape[1] == self.in_hi and f_mid.shape[1] == self.in_mid \
            and f_low.shape[1] == self.in_low, (
            f"MultiScaleSemanticHead channel mismatch: got "
            f"({f_hi.shape[1]},{f_mid.shape[1]},{f_low.shape[1]}) "
            f"expected ({self.in_hi},{self.in_mid},{self.in_low})")
        x = self.l_low(f_low)                                                   # @72
        x = F.interpolate(x, size=f_mid.shape[-2:], mode="bilinear",
                          align_corners=False) + self.l_mid(f_mid)             # @144
        x = self.smooth_mid(x)
        x = F.interpolate(x, size=f_hi.shape[-2:], mode="bilinear",
                          align_corners=False) + self.l_hi(f_hi)               # @288
        x = self.smooth_hi(x)
        return self.classifier(x)                                              # (B,C,288,288)


def build_sam3_tracker(checkpoint_path=None, load_from_HF=False,
                       apply_temporal_disambiguation=False):
    """Build SAM3 tracker (Sam3TrackerPredictor, IS-A Sam3TrackerBase) WITH backbone.

    Architecture instantiates with random init (no checkpoint needed). Pretrained
    weights (facebook/sam3 sam3.pt, gated) are loaded separately and only required
    for actual training/perf — `strict=False` since sam3.pt is the FULL model
    (detector+tracker+backbone) and we keep only the tracker+backbone subset.
    Requires `semseg/models/sam3` on PYTHONPATH (for `import sam3`).
    """
    from sam3.model_builder import build_tracker
    tracker = build_tracker(
        apply_temporal_disambiguation=apply_temporal_disambiguation,
        with_backbone=True,
    )
    if checkpoint_path is None and load_from_HF:
        from sam3.model_builder import download_ckpt_from_hf
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path:
        sd = torch.load(checkpoint_path, map_location="cpu")
        sd = sd.get("model", sd.get("model_state_dict", sd))
        # The FULL SAM3 checkpoint namespaces weights under 'detector.' (~1156) and
        # 'tracker.' (~309); our standalone tracker's keys carry NEITHER prefix. A plain
        # strict=False load therefore matches ZERO keys -> the ViT backbone stays RANDOM
        # and frozen -> features are garbage -> val mIoU collapses (~2%). Remap by prefix,
        # priority tracker.* (authoritative for tracker-specific modules), then
        # detector.* (the backbone is stored only under detector.backbone.*).
        model_sd = tracker.state_dict()
        remapped, src = {}, {"tracker.": 0, "detector.": 0}
        for m, mt in model_sd.items():
            for pref in ("tracker.", "detector."):
                c = sd.get(pref + m)
                if c is not None and c.shape == mt.shape:
                    remapped[m] = c
                    src[pref] += 1
                    break
        missing, unexpected = tracker.load_state_dict(remapped, strict=False)
        _bb = lambda k: any(t in k for t in ("trunk", "visual", "backbone", "vision"))
        n_bb = sum(1 for k in model_sd if _bb(k))
        n_bb_loaded = sum(1 for k in remapped if _bb(k))
        print(f"[sam3 ckpt] {checkpoint_path}: loaded={len(remapped)}/{len(model_sd)} "
              f"(tracker.={src['tracker.']} detector.={src['detector.']}) "
              f"missing={len(missing)} unexpected={len(unexpected)} | backbone={n_bb_loaded}/{n_bb}")
        # Guard: never silently train on a random frozen backbone again.
        if n_bb == 0 or n_bb_loaded < 0.9 * n_bb:
            raise RuntimeError(
                f"SAM3 backbone load FAILED: only {n_bb_loaded}/{n_bb} backbone keys filled "
                f"from {checkpoint_path}. Check key prefixes / checkpoint file. "
                f"(Set MODEL.CHECKPOINT_PATH='' to intentionally use random init.)")
    return tracker


class LoRA_Sam3_RBMA(nn.Module):
    """RBMA on SAM3 (plain LoRA encoder + reliability-biased memory attention).

    Phase 1 (this commit): build tracker + plain LoRA on backbone ViT qkv (q,v).
    Phase 2+: modality-as-frame forward, RBMA memory_mask bias, semantic decoder.
    """

    SEM_EPS = 1e-6

    def __init__(self, r: int = 4, num_modalities: int = 3, num_classes: int = 4,
                 lora_layer=None,
                 checkpoint_path=None, load_from_HF: bool = False,
                 apply_temporal_disambiguation: bool = False,
                 lambda_bias_init: float = 1.0,
                 fpn_channels: int = 256, fpn_hi_ch: int = 32, fpn_mid_ch: int = 64,
                 decoder_high_res: bool = False):
        super().__init__()
        assert r > 0
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.r = r

        # SAM3 tracker (predictor IS-A base): backbone + forward_image/_prepare_memory_*/_forward_sam_heads
        self.sam = build_sam3_tracker(
            checkpoint_path=checkpoint_path, load_from_HF=load_from_HF,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
        )

        # plain LoRA (q,v) on backbone ViT blocks; freezes the rest.
        # LoRA Linears are registered INSIDE self.sam via _LoRA_qkv → keep only plain
        # refs here (NOT nn.ModuleList) to avoid double-registration / duplicate state_dict keys.
        w_As, w_Bs, n = inject_plain_lora(self.sam, r)
        assert n > 0, "no ViT qkv found to inject LoRA — check backbone path/name_filter"
        self._lora_A = w_As   # plain lists (already submodules of self.sam)
        self._lora_B = w_Bs
        self.n_lora_blocks = n

        # Reliability head (small): standalone per-modality class logits → predictive
        # entropy = RBMA reliability signal (detached for the bias), and trained by the
        # per-modality auxiliary CE so the signal is meaningful.
        self.reliab_head = SemanticHead(in_channels=fpn_channels, num_classes=num_classes)

        # Semantic decoder = SAM3 mask decoder REPURPOSED for num_classes (like SAM2
        # MemorySAM's high_res_multimasks). Its TwoWayTransformer lets per-class tokens
        # ATTEND to the memory-conditioned image features, then transposed-conv upscaling —
        # far stronger than a conv head (the conv head collapsed to the 3 biggest classes).
        # We run it ourselves (see _semantic_decode) on the captured memory-conditioned
        # feature; track_step's own _forward_sam_heads is left untouched for memory/RBMA
        # (its object-score gating / best-mask selection would corrupt semantic output).
        from sam3.sam.mask_decoder import MaskDecoder
        from sam3.sam.transformer import TwoWayTransformer
        edim = self.sam.sam_prompt_embed_dim
        # decoder_high_res: feed fpn0(@288,32ch)/fpn1(@144,64ch) into the decoder's
        # transposed-conv upscaling as skip connections → finer detail for tiny/boundary
        # classes (Pedestrian/Pole/sign/RoadLine). Helps small classes but NOT a path to
        # SOTA mIoU — the binding limit is SAM3's single-scale 72x72 frozen ViT.
        self.decoder_high_res = decoder_high_res
        self.sem_decoder = MaskDecoder(
            num_multimask_outputs=num_classes,   # multimask path returns masks[:,1:] = num_classes
            transformer=TwoWayTransformer(depth=2, embedding_dim=edim, mlp_dim=2048, num_heads=8),
            transformer_dim=edim,
            iou_head_depth=3, iou_head_hidden_dim=256,
            use_high_res_features=decoder_high_res,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=False,                # semantic: NO object-presence gating
            dynamic_multimask_via_stability=False,
        )
        # warm-start the transformer + upscaling from the pretrained SAM3 mask decoder.
        # Only SHAPE-MATCHING keys (transformer, output_upscaling, iou_token, hypernet
        # MLPs 0..3) — the num_classes mask_tokens/iou_head differ in shape and STAY FRESH.
        # (strict=False still ERRORS on shape mismatch, so we must pre-filter.)
        try:
            own = self.sem_decoder.state_dict()
            filt = {k: v for k, v in self.sam.sam_mask_decoder.state_dict().items()
                    if k in own and own[k].shape == v.shape}
            self.sem_decoder.load_state_dict(filt, strict=False)
            print(f"[sem_decoder] warm-start: loaded {len(filt)}/{len(own)} keys (rest fresh: num_classes tokens)")
        except Exception as e:
            print(f"[sem_decoder] warm-start skipped: {e}")

        # RBMA learnable bias magnitude (used from Phase 3)
        self.lambda_bias = nn.Parameter(torch.tensor(float(lambda_bias_init)))

        # Output fusion temperature: per-modality semantic logits are combined by a softmax
        # over modalities of (lambda_fuse * reliability) instead of a naive mean — a naive
        # mean dilutes RGB's strong 25-class signal with weak depth/event/lidar logits
        # (collapse to dominant classes). Reuses the RBMA reliability as the fusion weight.
        self.lambda_fuse = nn.Parameter(torch.tensor(1.0))

        # RBMA memory-attention bias injection (Phase 3).
        # Hook injects a float attn_mask into each memory-fusion cross-attention
        # (MultiheadAttentionWrapper, torch semantics → added pre-softmax).
        # `_mem_bias_fn(L, S, device, dtype) -> (L, S) tensor | None` is set per-forward.
        self._mem_bias_fn = None
        self._mem_hooks = []
        self._mem_hook_fires = 0
        self._rbma_state = None
        self._captured_mem_feat = None
        self._register_memory_bias_hooks()

        # capture memory-conditioned feature (pix_feat_with_mem) so the semantic head
        # consumes the RBMA-affected feature (else the bias wouldn't reach the output).
        _orig_pmcf = self.sam._prepare_memory_conditioned_features
        def _pmcf_capture(*a, **k):
            out = _orig_pmcf(*a, **k)
            self._captured_mem_feat = out
            return out
        self.sam._prepare_memory_conditioned_features = _pmcf_capture

    def _register_memory_bias_hooks(self):
        """Set/clear RBMA bias on each memory-fusion cross-attn (RoPEAttention).

        The tracker's memory cross-attention is RoPEAttention(q, k, v, num_k_exclude_rope)
        whose SDPA was patched (sam3/sam/transformer.py) to add `self._rbma_attn_bias`
        to pre-softmax logits. We set that attribute per-call via a forward_pre_hook
        (from `self._mem_bias_fn(q, k) -> bias | None`) and clear it post-call.
        q,k are batch-first (B, N, E) at RoPEAttention input.
        """
        def _pre_hook(module, args, kwargs):
            self._mem_hook_fires += 1
            fn = self._mem_bias_fn
            if fn is None:
                module._rbma_attn_bias = None
                return None
            q = kwargs.get("q", args[0] if len(args) > 0 else None)
            k = kwargs.get("k", args[1] if len(args) > 1 else None)
            if q is None or k is None:
                module._rbma_attn_bias = None
                return None
            module._rbma_attn_bias = fn(q, k)   # bias broadcastable to (B,nh,Lq,Sk) | None
            return None

        def _post_hook(module, args, kwargs, output):
            module._rbma_attn_bias = None        # clear so it never leaks to other calls

        n = 0
        for name, mod in self.sam.named_modules():
            if name.endswith("cross_attn_image"):
                self._mem_hooks.append(mod.register_forward_pre_hook(_pre_hook, with_kwargs=True))
                self._mem_hooks.append(mod.register_forward_hook(_post_hook, with_kwargs=True))
                n += 1
        self._n_mem_hooks = n

        # capture num_obj_ptr_tokens (memory tail) from the memory-fusion encoder call,
        # so the bias fn can split the spatial-memory key columns per modality-frame.
        def _enc_pre(module, args, kwargs):
            if self._rbma_state is not None:
                self._rbma_state["num_obj_ptr"] = int(kwargs.get("num_obj_ptr_tokens", 0) or 0)
            return None
        try:
            self._mem_hooks.append(
                self.sam.transformer.encoder.register_forward_pre_hook(_enc_pre, with_kwargs=True))
        except AttributeError:
            pass

    def _rbma_bias_fn(self, q, k):
        """Build (1,1,1,Sk) additive reliability bias for the current cross-attn call.

        Memory key layout (our sequential modality-as-frame setup, small m):
        [frame0 spatial block] ... [frame_{i-1} spatial block] [obj_ptr tail].
        Each spatial block = tokens_per_frame; assign that modality-frame's reliability
        (resized to the block grid), λ-scaled and centered across frames. obj_ptr tail = 0.
        """
        st = self._rbma_state
        if st is None or st.get("frame", 0) == 0 or not st.get("reliab"):
            return None
        Sk = k.shape[1]
        nop = int(st.get("num_obj_ptr", 0) or 0)
        rels = st["reliab"]                      # list of (B,1,h,w), frame order 0..i-1
        nf = len(rels)
        n_spatial = Sk - nop
        if nf == 0 or n_spatial <= 0 or n_spatial % nf != 0:
            return None                           # layout assumption broken → skip (safe)
        tpf = n_spatial // nf
        hm = int(round(math.sqrt(tpf)))
        if hm * hm != tpf:
            return None
        B = rels[0].shape[0]
        # resize each frame reliability to (hm,hm), flatten → (B, tpf)
        blocks = [F.interpolate(r, size=(hm, hm), mode="bilinear", align_corners=False)
                  .flatten(1) for r in rels]      # each (B, tpf)
        rel_mat = torch.stack(blocks, dim=1)      # (B, nf, tpf)
        rel_mat = rel_mat - rel_mat.mean(dim=1, keepdim=True)   # center across frames
        rel_flat = (self.lambda_bias * rel_mat).reshape(B, nf * tpf)   # (B, n_spatial)
        if nop > 0:
            rel_flat = torch.cat([rel_flat, rel_flat.new_zeros(B, nop)], dim=1)
        return rel_flat.view(B, 1, 1, Sk).to(q.dtype)   # broadcast over heads & queries

    def train(self, mode: bool = True):
        """Keep the frozen SAM3 tracker in EVAL even when training. We train LoRA +
        reliab_head + sem_decoder + lambdas; the tracker/backbone is frozen and its
        training-only branches reference attributes not set by build_tracker (e.g.
        teacher_force_obj_scores_for_mem). super().train() already set our trainable
        submodules (reliab_head/sem_decoder) to `mode`; we only force sam back to eval.
        (No BatchNorm anywhere: reliab_head=GroupNorm, sem_decoder=LayerNorm → train==eval.)"""
        super().train(mode)
        self.sam.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ── Phase 4 — semantic head + per-modality predictive-uncertainty reliability ──
    @staticmethod
    def _reliability_from_logits(logits, eps=SEM_EPS):
        """class logits (B,C,h,w) -> reliability (B,1,h,w) = 1 - normalized pred. entropy.
        Training-free signal (no separate evidential/quality head): the RBMA 'A' axis."""
        C = logits.shape[1]
        p = F.softmax(logits, dim=1)
        ent = -(p * (p + eps).log()).sum(dim=1, keepdim=True) / math.log(C)
        return 1.0 - ent

    def _semantic_decode(self, pix_feat, high_res_features=None):
        """Repurposed SAM3 mask decoder on a (B,256,72,72) memory-conditioned feature
        → per-class logits (B, num_classes, 288, 288). Prompt-free: mirrors
        _forward_sam_heads' empty-point (label -1) + no-mask setup, but bypasses its
        object-score gating / best-mask selection (which are object-tracking specific).
        high_res_features (when decoder_high_res): [fpn0(B,32,288,288), fpn1(B,64,144,144)]
        skip-connections for the upscaling path."""
        sam = self.sam
        B = pix_feat.size(0); device = pix_feat.device
        pt_coords = torch.zeros(B, 1, 2, device=device)
        pt_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
        sparse, dense = sam.sam_prompt_encoder(points=(pt_coords, pt_labels), boxes=None, masks=None)
        image_pe = sam.sam_prompt_encoder.get_dense_pe()
        masks, _iou, _tok, _obj = self.sem_decoder(
            image_embeddings=pix_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,        # → masks[:, 1:] = num_classes channels
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return masks                       # (B, num_classes, 288, 288)

    def _backbone_lowres_feat(self, backbone_out):
        """Lowest-res FPN level (B, C, h, w) for the semantic head / reliability."""
        return backbone_out["backbone_fpn"][-1]

    # ── Phase 2+3+4 — modality-as-frame forward with RBMA bias + semantic head ──
    def forward(self, batched_input, multimask_output=True, gt_mask=None):
        """batched_input: list of m tensors (B, 3, 1008, 1008), one per modality.

        Per modality (frame): forward_image → backbone feats. Compute per-modality
        predictive-uncertainty reliability (semantic head on standalone backbone feat).
        For frames>0, set RBMA bias from PRIOR frames' reliability so the memory
        cross-attention up/down-weights each modality's memory tokens. track_step fuses
        via SAM3 memory. Semantic output = per-modality sem-head logits, averaged.
        """
        m = len(batched_input)
        sam = self.sam
        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        per_modal_sem = []     # per-modality class logits from sem_decoder (B, C, 288, 288)
        reliab_logits = []     # per-modality standalone logits (B, C, 72, 72) for aux CE
        reliab = []            # per-modality reliability (B,1,72,72) for RBMA bias + fusion
        self._mem_bias_fn = self._rbma_bias_fn

        for i in range(m):
            backbone_out = sam.forward_image(batched_input[i])
            _, vision_feats, vision_pos_embeds, feat_sizes = sam._prepare_backbone_features(backbone_out)
            fpn = backbone_out["backbone_fpn"]                       # [fpn0@288/32, fpn1@144/64, low@72/256]
            f_low = fpn[-1]                                          # (B, 256, 72, 72)
            hr = [fpn[0], fpn[1]] if self.decoder_high_res else None  # high-res skips (per modality)

            # reliability: standalone (pre-fusion) prediction → entropy. detached for the
            # RBMA bias (no circularity); the raw logits are kept for the aux CE.
            rl = self.reliab_head(f_low)                            # (B, C, 72, 72)
            reliab_logits.append(rl)
            reliab.append(self._reliability_from_logits(rl).detach())

            # RBMA: bias memory cross-attn by PRIOR frames' reliability (frames 0..i-1)
            self._rbma_state = {"frame": i, "reliab": reliab[:i], "num_obj_ptr": 0}

            step = sam.track_step(
                frame_idx=i, is_init_cond_frame=(i == 0),
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes, image=batched_input[i],
                point_inputs=None, mask_inputs=None,
                output_dict=output_dict, num_frames=m,
                run_mem_encoder=True, prev_sam_mask_logits=None,
            )
            bucket = "cond_frame_outputs" if i == 0 else "non_cond_frame_outputs"
            output_dict[bucket][i] = step

            # semantic output: repurposed SAM decoder on the MEMORY-CONDITIONED feature
            # (cross-modal fusion + RBMA bias flow here). Per-class tokens attend to it.
            mem_feat = self._captured_mem_feat                      # (B, 256, 72, 72)
            per_modal_sem.append(self._semantic_decode(mem_feat, hr))  # (B, C, 288, 288)

        self._rbma_state = None
        self._mem_bias_fn = None
        self._last_reliab = reliab
        self._last_reliab_logits = reliab_logits   # standalone logits → aux CE (trains reliab_head)

        # reliability-gated fusion of the per-modality decoder outputs (NOT a naive mean,
        # which dilutes RGB). w_i = softmax_i(lambda_fuse * reliability_i); degrades to a
        # mean when reliabilities are uniform (safe at init).
        sem_stack = torch.stack(per_modal_sem, dim=1)              # (B, m, C, 288, 288)
        Hs, Ws = sem_stack.shape[-2:]
        rel = torch.stack(reliab, dim=1).flatten(0, 1)            # (B*m, 1, 72, 72)
        rel = F.interpolate(rel, size=(Hs, Ws), mode="bilinear", align_corners=False)
        rel = rel.view(sem_stack.size(0), m, 1, Hs, Ws)           # (B, m, 1, 288, 288)
        w = torch.softmax(self.lambda_fuse * rel, dim=1)
        sem = (w * sem_stack).sum(dim=1)                          # (B, C, 288, 288)
        # head-res output (eval resizes to label size; loss downsamples GT to this res).
        return sem

    # ── Phase 4.3 — losses (main semantic CE + per-modality auxiliary CE) ──
    def compute_losses(self, sem_logits, gt, loss_fn=None, aux_weight: float = 0.5,
                       ignore_index: int = 255):
        """sem_logits: (B,C,H,W) fused output; gt: (B,H,W) long. Returns (total, dict).
        Main CE on fused output + per-modality aux CE (trains sem_head on each modality)."""
        if loss_fn is None:
            loss_fn = lambda x, y: F.cross_entropy(x, y, ignore_index=ignore_index)
        gt = gt.long()

        # CE at each logit's own resolution (downsample GT nearest → preserves ignore 255).
        # Main = fused sem_decoder output (288); aux = per-modality standalone reliab_head
        # logits (72) — trains the reliability head so the RBMA signal is meaningful.
        def ce_at(logits):
            hh, ww = logits.shape[-2:]
            g = gt
            if g.shape[-2:] != (hh, ww):
                g = F.interpolate(gt.unsqueeze(1).float(), size=(hh, ww), mode="nearest").squeeze(1).long()
            return loss_fn(logits, g)

        main = ce_at(sem_logits)
        aux = sem_logits.new_zeros(())
        rl = getattr(self, "_last_reliab_logits", None)
        if rl:
            for s in rl:
                aux = aux + ce_at(s)
            aux = aux / len(rl)
        total = main + aux_weight * aux
        return total, {"main": main.detach(), "aux": aux.detach()}
