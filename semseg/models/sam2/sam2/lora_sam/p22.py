"""LoRA_Sam_P22 (verbatim 이동)."""
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


class LoRA_Sam_P22(nn.Module):
    """
    LoRA_Sam_P22: P9 + Multi-Scale DeBA-FP (all FPN levels, Phase 1)

    P21과 동일한 DeBA-FP 원리이나 적용 범위가 다름:
      - P21: fpn[0]만, Phase 2에서 적용 → CrossModalFusionHead와 m_feat에만 영향
      - P22: fpn[0,1,2] 전부, Phase 1에서 적용 → vision_feats, tracking, decoder,
             CrossModalFusionHead 등 전체 파이프라인에 refined features 전파

    Ref: "Rethinking Deformable Convolution as an Adapter with Cross-layer
          Weight Sharing for Robust Semantic Segmentation in the Wild" (CVPR 2026)

    Cross-layer weight sharing (DeBA paper 원칙):
      - Shared: DCM (offset+deform conv), LayerNorm — across ALL FPN levels & modalities
      - Per-level: W_d, W_u (fpn[0]=32ch, fpn[1]=64ch, fpn[2]=256ch)
      - Per-modality: α scaling (shared across levels)

    Components:
      1. Structure: Soft-MoE LoRA (P9 동일)
      2. DeBA-FP MultiScale: all FPN levels (신규)
      3. UAMM (Memory): CrossModal max-normalized softmax (P9 동일)
      4. AMF (Fusion): CrossModal raw softmax (P9 동일)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 deba_bottleneck_dim=64, deba_kernel_size=3):
        nn.Module.__init__(self)

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        # Freeze original parameters
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Inject SoftMoE-LoRA (P9 동일)
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # Determine FPN channel dimensions
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        transformer_dim = self.sam.sam_mask_decoder.transformer_dim  # 256
        if use_high_res:
            # After forward_image: conv_s0→32ch, conv_s1→64ch, fpn[2]→256ch
            fpn_channels = [
                transformer_dim // 8,   # fpn[0] = 32
                transformer_dim // 4,   # fpn[1] = 64
                transformer_dim,        # fpn[2] = 256
            ]
            fusion_dim = fpn_channels[0]  # 32, for CrossModalFusionHead
        else:
            fpn_channels = [transformer_dim]
            fusion_dim = transformer_dim

        # CrossModalFusionHead (P9 동일)
        self.cross_modal_head = CrossModalFusionHead(
            in_channels=fusion_dim,
            num_modalities=num_modalities,
        )

        # [P22 신규] DeBA-FP MultiScale: all FPN levels
        self.deba_fp_ms = DeBAFP_MultiScale(
            fpn_channels=fpn_channels,
            bottleneck_dim=deba_bottleneck_dim,
            kernel_size=deba_kernel_size,
            num_modalities=num_modalities,
        )

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # Collect MoE gate weights for visualization
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb
        try:
            # ============================================
            # Phase 1: Image Encoding + DeBA-FP (all FPN levels)
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])

                # [P22] DeBA-FP: refine ALL FPN levels before _prepare_backbone_features
                # → refined features flow through vision_feats, tracking, decoder
                num_fpn_levels = len(img_emb['backbone_fpn'])
                for level in range(num_fpn_levels):
                    img_emb['backbone_fpn'][level] = self.deba_fp_ms(
                        img_emb['backbone_fpn'][level],
                        modality_idx=i,
                        level_idx=level,
                    )

                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Cross-Modal 가중치 산출
            # (fpn[0] is already refined from Phase 1)
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
            cross_weights, cross_logits = self.cross_modal_head(all_backbone_feats)

            # UAMM용: max-normalize → best modality = 1.0
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)

            # ============================================
            # Phase 3: UAMM Modulation + Tracking
            # ============================================
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                current_score = uamm_scores[:, frame_idx].unsqueeze(1)
                score_expanded = current_score.transpose(0, 1).unsqueeze(-1)
                modulated_vision_feats = [feat * score_expanded for feat in vision_feats[frame_idx]]

                multi_mask_output_step = self.sam.track_step(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init,
                    current_vision_feats=modulated_vision_feats,
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
                output_dict["cond_frame_outputs"][frame_idx] = multi_mask_output_step
                output.append(multi_mask_output_step["high_res_multimasks"])

            # ============================================
            # Phase 4: AMF — raw softmax weights로 Output Fusion
            # ============================================
            amf_weights = cross_weights

            w0 = amf_weights[:, 0].view(-1, 1, 1, 1)
            m_output = output[0] * w0
            m_feat = all_backbone_feats[0] * w0

            for i in range(1, m):
                wi = amf_weights[:, i].view(-1, 1, 1, 1)
                m_output = m_output + output[i] * wi
                m_feat = m_feat + all_backbone_feats[i] * wi

            # Store for visualization
            self._last_uamm_scores = uamm_scores.detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().float().cpu().numpy()
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None
            # Feature analysis: per-modal outputs, backbone feats
            self._last_per_modal_outputs = [o.detach().cpu() for o in output]
            self._last_per_modal_feats = [f.detach().cpu() for f in all_backbone_feats]
        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        # SoftMoE Parameters
        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        cross_modal_tensors = {
            f"cross_modal_head.{k}": v
            for k, v in self.cross_modal_head.state_dict().items()
        }

        # [P22] DeBA-FP MultiScale parameters
        deba_fp_ms_tensors = {
            f"deba_fp_ms.{k}": v
            for k, v in self.deba_fp_ms.state_dict().items()
        }

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        model_ref = self.sam.module if isinstance(self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else self.sam
        state_dict = model_ref.state_dict()

        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {
            **moe_params,
            **prompt_encoder_tensors,
            **mask_decoder_tensors,
            **cross_modal_tensors,
            **deba_fp_ms_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        # Load SoftMoE Layers
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict: mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict: mv.load_state_dict(state_dict[v_key])

        # Load CrossModalFusionHead
        cross_modal_dict = {}
        for k, v in state_dict.items():
            if k.startswith("cross_modal_head."):
                cross_modal_dict[k.replace("cross_modal_head.", "")] = v
        if cross_modal_dict:
            self.cross_modal_head.load_state_dict(cross_modal_dict)

        # [P22] Load DeBA-FP MultiScale
        deba_fp_ms_dict = {}
        for k, v in state_dict.items():
            if k.startswith("deba_fp_ms."):
                deba_fp_ms_dict[k.replace("deba_fp_ms.", "")] = v
        if deba_fp_ms_dict:
            self.deba_fp_ms.load_state_dict(deba_fp_ms_dict)

        # Load SAM components
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)

        self.sam.load_state_dict(sam_dict)
