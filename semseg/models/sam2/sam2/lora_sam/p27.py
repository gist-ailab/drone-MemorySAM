"""LoRA_Sam_P27 (verbatim 이동)."""
import copy
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as tv_models
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
from icecream import ic

from ..modeling.sam2_base import SAM2Base
from ..modules import *  # noqa: F401,F403
from ..modules.moe import _LoRA_qkv, _MoE_LoRA_qkv, _SoftMoE_LoRA_qkv, _MoE_DeBA_BB_qkv  # noqa: F401
from .base import LoRA_Sam  # noqa: F401
from .viz import (save_sam2_full_report, _denormalize, _compute_gpu_pca_single,
                  _save_image, _save_heatmap, _save_pca)  # noqa: F401
from .heads import (ConfidenceAuxHead, ModalAuxDecoder, MultiScaleModalAuxDecoder,
                    ResNetAuxBackbone, ResNetAuxDecoder,
                    compute_energy_confidence, compute_spatial_energy_confidence,
                    compute_spatial_entropy_confidence)  # noqa: F401
from .legacy import LoRA_Sam_P26  # noqa: F401


class LoRA_Sam_P27(LoRA_Sam_P26):
    """
    LoRA_Sam_P27: Additive Attention Bias on Cross-Modal Memory Attention.

    핵심 차이 (P26 → P27):
      - [제거] UAMM feature multiplication (`level_feat * q_flat`) in Phase 3
      - [추가] Cross-attention additive bias: attn = softmax(QK^T/√d + λ·B) V
              B는 각 memory token의 source-modality quality_logit을 spatial 대응시킨 것
              λ는 학습 가능 스칼라
      - [유지] SQG + KL teacher loss, SoftMoE LoRA, per-modal decoder, multi-scale FPN
      - [옵션] AMF는 `amf_mode='sqg_quality'`(기본) 또는 `'uniform'` 선택

    Memory attention 내부에서 열화된 modality의 K/V에 직접 페널티 → content-sensitive
    attention routing. Diagnosis (MISC/diagnose_memory_attention.py)에서 확인된
    "attention insensitivity" 문제를 정면 대응.
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.3,
                 tau_uamm=1.0, tau_teacher=0.5,
                 memory_mod=False, amf_mode='sqg_quality',
                 multi_scale_sqg=True, per_modality_decoder=True,
                 cond_dim=8,
                 lambda_bias_init=1.0):
        super().__init__(
            sam_model=sam_model, r=r, lora_layer=lora_layer,
            num_experts=num_experts, num_modalities=num_modalities,
            quality_hidden_dim=quality_hidden_dim, quality_min=quality_min,
            tau_uamm=tau_uamm, tau_teacher=tau_teacher,
            memory_mod=memory_mod, amf_mode=amf_mode,
            multi_scale_sqg=multi_scale_sqg, per_modality_decoder=per_modality_decoder,
            cond_dim=cond_dim,
        )
        # [P27] Learnable scalar for attention bias magnitude
        self.lambda_bias = nn.Parameter(torch.tensor(float(lambda_bias_init)))
        # Runtime state for pre-hook — set by forward, read by the hook
        self._p27_state = {
            'enabled': False,
            'quality_logits': None,
            'current_frame': 0,
        }
        self._p27_hook_handle = None
        self._register_memory_attention_hook()

    # ─────────────────────────────────────────────────────────────────
    # Memory attention bias injection
    # ─────────────────────────────────────────────────────────────────
    def _register_memory_attention_hook(self):
        """Register a forward pre-hook on sam.memory_attention that computes
        per-K-token additive bias and sets it on each cross-attn module.
        """
        if self._p27_hook_handle is not None:
            return

        def _pre_hook(module, args, kwargs):
            state = self._p27_state
            if not state.get('enabled', False):
                return
            memory = kwargs.get('memory', None)
            if memory is None and len(args) >= 2:
                memory = args[1]
            if memory is None:
                return
            quality_logits = state['quality_logits']
            current_frame = state['current_frame']
            if quality_logits is None or current_frame == 0:
                return

            B = quality_logits[0].shape[0]
            if memory.dim() != 3:
                return
            # MemoryAttention receives memory seq-first (pre-transpose) when batch_first=True
            if memory.shape[0] == B and memory.shape[1] != B:
                N_k = memory.shape[1]
            elif memory.shape[1] == B and memory.shape[0] != B:
                N_k = memory.shape[0]
            else:
                # Ambiguous shape (e.g. B==N_k) — default to SAM2 convention (seq, B, D)
                N_k = memory.shape[0]
            num_obj_ptr = kwargs.get('num_obj_ptr_tokens', 0)
            n_spatial = N_k - num_obj_ptr
            f = current_frame
            if f <= 0 or n_spatial <= 0 or n_spatial % f != 0:
                return
            tokens_per_frame = n_spatial // f
            h_mem = int(math.sqrt(tokens_per_frame))
            if h_mem <= 0:
                return
            w_mem = tokens_per_frame // h_mem
            if h_mem * w_mem != tokens_per_frame:
                # Fallback: flat interpolation using 1D
                h_mem, w_mem = 1, tokens_per_frame

            device = memory.device
            dtype = memory.dtype
            bias_parts = []
            for j in range(f):
                q_logit_j = quality_logits[j]  # (B, 1, H_fpn, W_fpn)
                q_bias_j = F.interpolate(
                    q_logit_j, size=(h_mem, w_mem),
                    mode='bilinear', align_corners=False,
                )
                q_bias_flat = q_bias_j.flatten(2).squeeze(1)  # (B, tokens_per_frame)
                bias_parts.append(q_bias_flat.to(dtype))
            if num_obj_ptr > 0:
                bias_parts.append(torch.zeros(B, num_obj_ptr, device=device, dtype=dtype))
            bias_all = torch.cat(bias_parts, dim=-1)  # (B, N_k)
            bias_all = bias_all * self.lambda_bias
            bias_all = bias_all.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N_k)

            for layer in module.layers:
                if hasattr(layer.cross_attn_image, '_p27_attn_bias'):
                    layer.cross_attn_image._p27_attn_bias = bias_all

        self._p27_hook_handle = self.sam.memory_attention.register_forward_pre_hook(
            _pre_hook, with_kwargs=True,
        )

    def _clear_memory_attention_bias(self):
        for layer in self.sam.memory_attention.layers:
            if hasattr(layer.cross_attn_image, '_p27_attn_bias'):
                layer.cross_attn_image._p27_attn_bias = None

    def _compute_bias_source(self, quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m):
        """Per-modality maps (list of m x (B,1,H_fpn,W_fpn)) used as the additive
        memory-attention logit-bias source (consumed by the pre-hook).

        P27 default = SpatialQualityGating quality logits. Subclasses override:
        e.g. P28/RBMA replaces this with training-free per-modality decoder
        predictive uncertainty. Identity default → preserves P27 behavior."""
        return quality_logits

    # ─────────────────────────────────────────────────────────────────
    # Detection bridge (P29-Det / P30-Det): run the full encoder + cross-modal
    # memory-attention pipeline and hand per-modality features to a detection head.
    # ─────────────────────────────────────────────────────────────────
    def extract_det_features(self, batched_input):
        """Run the full encoder + cross-modal memory-attention pipeline and return
        per-modality features for an object-detection head, keeping the graph intact
        (gradients flow to LoRA / memory_attention / RBMA λ).

        The cross-modal memory is built from the mask-decoder outputs encoded by the
        memory encoder, so the regular forward() (track_step loop) must run for the
        memory attention to be meaningful — we capture intermediate tensors rather than
        reimplement it.

        Returns dict of in-graph tensor lists (length m, modality order = input):
          'fpn0'  : (B, 32,  H/4,  W/4)   encoder high-res detail        · per modality
          'fpn1'  : (B, 64,  H/8,  W/8)   encoder mid-res detail         · per modality
          'mem'   : (B, 256, H/16, W/16)  memory-conditioned coarse      · per modality
                    (frame 0 = +no_mem_embed; frames>=1 = memory attention + RBMA bias)
          'output': (B, Cseg, H/4, W/4)   per-modality seg logits        · per modality
                    (used by P30-Det as a training-free reliability source 1-H/logC)
        """
        mem_feats = []
        orig_prep = self.sam._prepare_memory_conditioned_features

        def _capture_prep(*args, **kwargs):
            out = orig_prep(*args, **kwargs)   # (B, C, H, W), in-graph
            mem_feats.append(out)
            return out

        self.sam._prepare_memory_conditioned_features = _capture_prep
        self._capture_det_features = True
        self._det_fpn0 = None
        self._det_fpn1 = None
        self._det_output = None
        try:
            # gt_mask=None → aux/KL path skipped; the fused seg output is discarded.
            self.forward(batched_input, multimask_output=True)
        finally:
            self.sam._prepare_memory_conditioned_features = orig_prep
            self._capture_det_features = False

        feats = {
            'fpn0': self._det_fpn0,
            'fpn1': self._det_fpn1,
            'mem': mem_feats,
            'output': self._det_output,
        }
        self._det_fpn0 = None
        self._det_fpn1 = None
        self._det_output = None
        if (feats['fpn0'] is None or feats['output'] is None
                or len(mem_feats) != len(batched_input)):
            raise RuntimeError(
                f"extract_det_features capture failed: fpn0={feats['fpn0'] is not None}, "
                f"output={feats['output'] is not None}, "
                f"mem_feats={len(mem_feats)} (expected {len(batched_input)})."
            )
        return feats

    # ─────────────────────────────────────────────────────────────────
    # Forward — P26와 동일하되 Phase 3에서 UAMM multiplication 제거
    # ─────────────────────────────────────────────────────────────────
    def forward(self, batched_input, multimask_output, gt_mask=None):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []
        raw_backbone_fpns = []

        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        trunk = self.sam.image_encoder.trunk
        _orig_gc = getattr(trunk, 'gradient_checkpointing', False)

        try:
            # ── Phase 1: Image Encoding (same as P26) ──
            device = batched_input[0].device
            use_hr = getattr(self.sam, "use_high_res_features_in_sam", False)
            scalp = getattr(self.sam.image_encoder, "scalp", 0)
            n_fpn = len(self.sam.image_encoder.neck.convs) - scalp

            for i in range(m):
                idx_tensor = torch.tensor(i, device=device)

                if _orig_gc and self.training:
                    outs = torch.utils.checkpoint.checkpoint(
                        self._encode_single_modality,
                        batched_input[i], idx_tensor,
                        use_reentrant=False,
                    )
                else:
                    outs = self._encode_single_modality(batched_input[i], idx_tensor)

                fpn_list = list(outs[:n_fpn])
                pos_list = list(outs[n_fpn:n_fpn * 2])
                img_emb_raw = {'backbone_fpn': fpn_list, 'vision_pos_enc': pos_list}

                if self.multi_scale_sqg and use_hr:
                    raw_fpn = list(outs[n_fpn * 2:])
                    raw_backbone_fpns.append(raw_fpn)

                image_embedding.append(img_emb_raw)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb_raw)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            for layer in self.moe_layers_q + self.moe_layers_v:
                layer.set_condition(None)

            # ── Phase 2: Per-Modality SQG (same as P26) ──
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]

            # [Det bridge] Expose encoder FPN detail levels (in-graph) for an object
            # detection head. Behaviour-neutral: only activated by extract_det_features().
            if getattr(self, '_capture_det_features', False):
                self._det_fpn0 = all_backbone_feats
                self._det_fpn1 = [image_embedding[i]['backbone_fpn'][1] for i in range(m)]

            quality_logits = []
            quality_maps = []
            for i in range(m):
                if self.multi_scale_sqg and len(raw_backbone_fpns) > 0:
                    sqg_input = self._fuse_fpn_multiscale(raw_backbone_fpns[i])
                else:
                    sqg_input = all_backbone_feats[i]
                q_logit = self.quality_gatings[i](sqg_input)
                quality_logits.append(q_logit)
                quality_maps.append(self.quality_gatings[i].logits_to_quality(q_logit))

            # ── Phase 2.5: Aux CE + KL teacher (same as P26) ──
            gate_loss_data = None
            if self.training and gt_mask is not None:
                fpn_h, fpn_w = quality_logits[0].shape[-2:]
                gt_safe = gt_mask.long().clone()
                ignore_mask_full = (gt_safe == 255)
                gt_safe[ignore_mask_full] = 0
                ignore_mask_fpn = F.interpolate(
                    ignore_mask_full.unsqueeze(1).float(), size=(fpn_h, fpn_w),
                    mode='nearest',
                ).bool()

                ce_maps = []
                aux_losses = []
                for i in range(m):
                    aux_logits = self._auxiliary_decode_single(
                        self.per_modal_decoders[i],
                        vision_feats[i], vision_pos_embeds[i], feat_sizes[i],
                    )
                    if aux_logits.shape[-2:] != gt_mask.shape[-2:]:
                        aux_logits_resized = F.interpolate(
                            aux_logits, size=gt_mask.shape[-2:],
                            mode='bilinear', align_corners=False,
                        )
                    else:
                        aux_logits_resized = aux_logits

                    aux_ce = F.cross_entropy(aux_logits_resized, gt_safe, ignore_index=255)
                    aux_losses.append(aux_ce)

                    with torch.no_grad():
                        ce_map = F.cross_entropy(
                            aux_logits_resized.detach(), gt_safe,
                            reduction='none',
                        )
                        ce_map[ignore_mask_full] = 0.0
                        ce_map_fpn = F.interpolate(
                            ce_map.unsqueeze(1), size=(fpn_h, fpn_w),
                            mode='bilinear', align_corners=False,
                        )
                    ce_maps.append(ce_map_fpn)

                ce_stack = torch.stack(ce_maps, dim=0)
                quality_target_dist = F.softmax(-ce_stack / self.tau_teacher, dim=0)

                gate_loss_data = {
                    'predicted_logits': quality_logits,
                    'quality_target_dist': quality_target_dist,
                    'ignore_mask': ignore_mask_fpn,
                    'loss_type': 'kl',
                    'aux_ce_losses': aux_losses,
                }

            # ── Phase 3 (P27): No UAMM multiplication; inject memory attention bias ──
            q_logit_stack = torch.stack(quality_logits, dim=0)
            q_uamm_norm = F.softmax(q_logit_stack / self.tau_uamm, dim=0)

            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            # Prepare bias state for pre-hook (P27=SQG logits; P28/RBMA=decoder uncertainty)
            self._p27_state['quality_logits'] = self._compute_bias_source(
                quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m)
            self._p27_state['enabled'] = True

            for frame_idx in range(m):
                is_init = (frame_idx == 0)
                self._p27_state['current_frame'] = frame_idx

                multi_mask_output_step = self.sam.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init,
                    current_vision_feats=vision_feats[frame_idx],
                    current_vision_pos_embeds=vision_pos_embeds[frame_idx],
                    feat_sizes=feat_sizes[frame_idx],
                    point_inputs=None,
                    mask_inputs=None,
                    output_dict=output_dict,
                    num_frames=m,
                    track_in_reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=None,
                )
                self._clear_memory_attention_bias()

                if self.memory_mod and multi_mask_output_step.get("maskmem_features") is not None:
                    maskmem = multi_mask_output_step["maskmem_features"]
                    q_map = quality_maps[frame_idx]
                    if q_map.shape[-2:] != maskmem.shape[-2:]:
                        q_map_resized = F.interpolate(
                            q_map, size=maskmem.shape[-2:],
                            mode='bilinear', align_corners=False,
                        )
                    else:
                        q_map_resized = q_map
                    multi_mask_output_step["maskmem_features"] = maskmem * q_map_resized

                output_dict["cond_frame_outputs"][frame_idx] = multi_mask_output_step
                output.append(multi_mask_output_step["high_res_multimasks"])

            # Disable bias injection after all track_step calls
            self._p27_state['enabled'] = False
            self._p27_state['quality_logits'] = None

            # [Det bridge] Capture per-modality memory-conditioned seg logits, used by
            # the detection head as a training-free per-modality reliability source
            # (1 - H(softmax)/logC). Behaviour-neutral unless extract_det_features().
            if getattr(self, '_capture_det_features', False):
                self._det_output = output

            # ── Phase 4: Fusion (amf_mode 선택) ──
            out_h, out_w = output[0].shape[-2:]
            num_classes = output[0].shape[1]

            if self.amf_mode == 'uniform':
                # [P27] Simplest fusion — equal weight per modality
                amf_norm = torch.stack([
                    torch.full_like(q_uamm_norm[0], 1.0 / m)
                    for _ in range(m)
                ], dim=0)
                amf_norm_list = []
                for i in range(m):
                    amf_i = F.interpolate(
                        amf_norm[i], size=(out_h, out_w),
                        mode='bilinear', align_corners=False,
                    )
                    amf_norm_list.append(amf_i)
                amf_norm = torch.stack(amf_norm_list, dim=0)
            elif self.amf_mode == 'sqg_quality':
                amf_norm_list = []
                for i in range(m):
                    amf_i = F.interpolate(
                        q_uamm_norm[i], size=(out_h, out_w),
                        mode='bilinear', align_corners=False,
                    )
                    amf_norm_list.append(amf_i)
                amf_norm = torch.stack(amf_norm_list, dim=0)
            elif self.amf_mode == 'output_entropy':
                amf_weights = []
                for i in range(m):
                    prob = F.softmax(output[i], dim=1)
                    entropy = -(prob * (prob + 1e-8).log()).sum(dim=1, keepdim=True)
                    confidence = 1.0 - entropy / math.log(num_classes)
                    amf_weights.append(confidence)
                amf_stack = torch.stack(amf_weights, dim=0)
                amf_norm = amf_stack / amf_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)
            else:
                q_amf_list = []
                for i in range(m):
                    q_amf_i = F.interpolate(
                        quality_maps[i], size=(out_h, out_w),
                        mode='bilinear', align_corners=False,
                    )
                    q_amf_list.append(q_amf_i)
                amf_stack = torch.stack(q_amf_list, dim=0)
                amf_norm = amf_stack / amf_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)

            # [P31.2 fix] Route through the overridable _fuse_outputs hook (P26 hook).
            # P27's original inline fusion bypassed the hook, so LoRA_Sam_P30/P31's
            # ReliabilityAnchoredRouter override was NEVER executed for P27-lineage
            # models (P30's 200ep run trained with the router silently inactive —
            # find_unused_parameters=True masked the dead params). The default hook
            # body is byte-identical to the previous inline sums (P27/P28/P29 unchanged):
            #   m_output = Σ amf_norm[i]·output[i];  m_feat = Σ q_uamm_norm[i]·feat[i]
            m_output, m_feat = self._fuse_outputs(
                output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes)

            # Logging
            uamm_scalar = torch.stack(
                [q_uamm_norm[i].mean(dim=[1, 2, 3]) for i in range(m)], dim=1
            )
            amf_log = torch.stack(
                [amf_norm[i].mean(dim=[1, 2, 3]) for i in range(m)], dim=1
            )
            amf_log = amf_log / amf_log.sum(dim=1, keepdim=True).clamp(min=1e-6)

            self._last_uamm_scores = uamm_scalar.detach().float().cpu().numpy()
            self._last_amf_weights = amf_log.detach().float().cpu().numpy()
            self._last_quality_maps = [q.detach().float().cpu().numpy() for q in quality_maps]
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None
            self._last_per_modal_outputs = [o.detach().cpu() for o in output]
            self._last_per_modal_feats = [f.detach().cpu() for f in all_backbone_feats]
            self._last_uamm_spatial = [q_uamm_norm[i].detach().float().cpu().numpy() for i in range(m)]
            self._last_amf_spatial = [amf_norm[i].detach().float().cpu().numpy() for i in range(m)]
            ent_maps = []
            for i in range(m):
                prob_i = F.softmax(output[i], dim=1)
                ent_i = -(prob_i * (prob_i + 1e-8).log()).sum(dim=1, keepdim=True)
                ent_maps.append(ent_i.detach().float().cpu().numpy())
            self._last_entropy_maps = ent_maps
        finally:
            self._p27_state['enabled'] = False
            self._p27_state['quality_logits'] = None
            self._clear_memory_attention_bias()
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None
                layer.set_condition(None)

        if self.training and gate_loss_data is not None:
            return m_output, m_feat, gate_loss_data
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        super().save_lora_parameters(filename)
        # Append λ to the saved checkpoint
        state = torch.load(filename)
        state['p27_lambda_bias'] = self.lambda_bias.detach().cpu()
        torch.save(state, filename)

    def load_lora_parameters(self, filename: str) -> None:
        super().load_lora_parameters(filename)
        state = torch.load(filename)
        if 'p27_lambda_bias' in state:
            with torch.no_grad():
                self.lambda_bias.copy_(state['p27_lambda_bias'].to(self.lambda_bias.device))
