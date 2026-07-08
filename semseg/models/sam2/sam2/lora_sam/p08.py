"""LoRA_Sam_P8 (verbatim 이동)."""
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
from .legacy import LoRA_Sam_P6  # noqa: F401


class LoRA_Sam_P8(LoRA_Sam_P6):
    """
    LoRA_Sam_P8: The Hybrid Approach with Soft-MoE
    
    Components:
    1. Structure: Soft-MoE LoRA (Dynamic Ensemble of all experts)
    2. UAMM (Memory): Uses SIGMOID Scores (Absolute Quality) -> Feature Suppression
    3. AMF (Fusion): Uses Normalized SCORES (Relative Importance)
    """
    
    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None, num_experts=4, top_k=None):
        # Top-k는 Soft-MoE에서 사용되지 않으므로 무시하거나 None 처리
        
        # P6 등 상위 클래스의 init을 호출하면 Hard MoE가 생성되므로, 
        # 여기서는 기본 nn.Module 초기화 후 직접 레이어를 주입합니다.
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

        # Inject SoftMoE-LoRA
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            
            # [변경점] SoftMoE Layer 사용 (Top-k 인자 없음)
            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            
            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)
            
            # Replace QKV with SoftMoE version
            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)
            
        self.sam = sam_model
        
        # Determine Fusion Dim
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # Use ConfidenceHeadV2 (Deep Head + Zero Init) from P6
        self.confidence_head = ConfidenceHeadV2(in_channels=fusion_dim)

    def modulate_features_sigmoid(self, vision_feats_list, logits):
        """
        [UAMM] Apply Sigmoid for Memory Modulation (Same as P8 Revised)
        Input: logits (B, 1) -> Sigmoid -> Scores (B, 1) -> Expand -> Multiply
        Returns: Modulated Features AND the Score (0~1) for later use in AMF
        """
        scores = torch.sigmoid(logits) # 0.0 ~ 1.0 (Absolute score)
        
        # (B, 1) -> (1, B, 1) to match (HW, B, C)
        scores_expanded = scores.transpose(0, 1).unsqueeze(-1)
        
        modulated_list = [feat * scores_expanded for feat in vision_feats_list]
        return modulated_list, scores

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats, vision_pos_embeds, feat_sizes, output = [], [], [], [], [], []

        # Collect MoE gate weights for visualization (Soft-MoE: softmax over experts)
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb
        try:
            # 1. Image Encoding
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)
        
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            modality_scores = []  # Store Scores (0~1) for AMF

            # 2. Tracking Loop (Hybrid Logic)
            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                # (A) Compute Logits
                score_source = image_embedding[frame_idx]['backbone_fpn'][0]
                current_logits = self.confidence_head(score_source)  # (B, 1)

                # (B) UAMM: Use SIGMOID for Memory Modulation
                modulated_vision_feats, current_score = self.modulate_features_sigmoid(
                    vision_feats[frame_idx],
                    current_logits
                )
                modality_scores.append(current_score)

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

            # 3. AMF: Use Normalized SCORES for Output Fusion
            all_scores = torch.cat(modality_scores, dim=1)  # (B, m) - Values are 0~1
            sum_scores = all_scores.sum(dim=1, keepdim=True) + 1e-8
            weights = all_scores / sum_scores  # (B, m)

            w0 = weights[:, 0].view(-1, 1, 1, 1)
            m_output = output[0] * w0
            m_feat = image_embedding[0]['backbone_fpn'][0] * w0

            for i in range(1, m):
                wi = weights[:, i].view(-1, 1, 1, 1)
                m_output = m_output + output[i] * wi
                m_feat = m_feat + image_embedding[i]['backbone_fpn'][0] * wi

            # Store for visualization
            self._last_uamm_scores = all_scores.detach().float().cpu().numpy()
            self._last_amf_weights = weights.detach().float().cpu().numpy()
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None
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
            
        confidence_tensors = {f"confidence_head.{k}": v for k, v in self.confidence_head.state_dict().items()}
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
            **confidence_tensors
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

        confidence_dict = {}
        for k, v in state_dict.items():
            if k.startswith("confidence_head."):
                confidence_dict[k.replace("confidence_head.", "")] = v
        if confidence_dict:
            self.confidence_head.load_state_dict(confidence_dict)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)

        self.sam.load_state_dict(sam_dict)
