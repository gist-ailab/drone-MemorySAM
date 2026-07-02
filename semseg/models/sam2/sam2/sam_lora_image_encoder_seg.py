import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file
import torchvision.models as tv_models

from icecream import ic
from .modeling.sam2_base import SAM2Base
import torch.nn.init as init
import random
from .sam_lora_image_encoder_seg_bkup import LoRA_Sam
from .sam_lola_utils import (
    MLP_my,
    _LoRA_qkv,
    random_element_swap,
    ConfidenceHeadV2,
    ConfidenceHead,
    CrossModalFusionHead,
    CrossModalFusionHeadV2,
    SpatialCrossModalFusionHead,
    ModalAuxHead,
    MoE_LoRA_Layer,
    _MoE_LoRA_qkv,
    SoftMoE_LoRA_Layer,
    SelfDerivedCondition,
    ReliabilityAnchoredRouter,
    ClassTokenDecoder,
    ClassTokenDecoderMS,
    _SoftMoE_LoRA_qkv,
    SharedGateMLP,
    SoftMoE_LoRA_Layer_V2,
    DeBAFP,
    DeBAFP_MultiScale,
    MoE_DeBA_BB,
    _MoE_DeBA_BB_qkv,
    SpatialQualityGating,
)


'''CUSTOM VISUALIZATIOM'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
def save_sam2_full_report(
    batched_input,    # 원본 입력 리스트 [tensor(B,C,H,W), ...]
    image_embedding, 
    vision_feats, 
    feat_sizes, 
    output_list, 
    m_output, 
    m_feat, 
    save_dir="viz_results"
):
    """
    배치 사이즈(B)에 대응하며, 입력 원본 이미지까지 포함한 통합 분석 리포트를 저장합니다.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.switch_backend('Agg')
    
    modal_names = ["00", "01", "02", "03"]
    batch_size = batched_input[0].shape[0] # B 추출
    num_modalities = len(batched_input)

    with torch.no_grad():
        for b in range(batch_size):
            # 배치별 폴더 생성 (예: viz_results/batch_00)
            b_dir = os.path.join(save_dir, f"batch_{b:02d}")
            os.makedirs(b_dir, exist_ok=True)

            for i in range(num_modalities):
                prefix = f"mod{i:02d}_{modal_names[i]}"
                
                # --- 1. 원본 입력 데이터 역정규화 및 저장 ---
                # batched_input[i][b] -> (C, H, W)
                input_tensor = batched_input[i][b].cpu()
                input_img = _denormalize(input_tensor, modal_type=modal_names[i])
                _save_image(input_img, f"{prefix}_00_input.png", f"{modal_names[i]} Input", b_dir)

                # --- 2. 개별 마스크 (Heatmap) ---
                mask_prob = torch.sigmoid(output_list[i][b, 0]).cpu().numpy()
                _save_heatmap(mask_prob, f"{prefix}_01_mask.png", f"{modal_names[i]} Mask", b_dir)

                # --- 3. 개별 피쳐 PCA ---
                fpn_feat = image_embedding[i]['backbone_fpn'][0][b] # (C, H, W)
                pca_backbone = _compute_gpu_pca_single(fpn_feat)
                _save_pca(pca_backbone, f"{prefix}_02_feat_pca.png", f"{modal_names[i]} Feature PCA", b_dir)

            # --- 4. 최종 융합 결과 (Fused) ---
            f_prefix = "fused_final"
            f_mask_prob = torch.sigmoid(m_output[b, 0]).cpu().numpy()
            _save_heatmap(f_mask_prob, f"{f_prefix}_01_mask.png", "Final Fused Mask", b_dir)

            f_feat = m_feat[b]
            pca_fused = _compute_gpu_pca_single(f_feat)
            _save_pca(pca_fused, f"{f_prefix}_02_feat_pca.png", "Final Fused Feature PCA", b_dir)

    print(f"✅ 배치 {batch_size}개에 대한 리포트 저장 완료: {save_dir}")

# --- Helper Functions ---

def _denormalize(tensor, modal_type="RGB"):
    """정규화된 텐서를 시각화 가능한 [0, 1] 범위의 numpy로 변환"""
    # RGB의 경우 표준 ImageNet 수치 사용 (사용자 설정에 따라 수정 가능)
    if modal_type == "RGB" and tensor.shape[0] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean
    else:
        # Depth, Event, LiDAR 등은 보통 Min-Max 정규화만 해제
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    img = tensor.permute(1, 2, 0).numpy() # (H, W, C)
    return np.clip(img, 0, 1)

def _compute_gpu_pca_single(feat_tensor):
    """단일 이미지 피쳐(C, H, W)에 대한 PCA"""
    C, H, W = feat_tensor.shape
    x = feat_tensor.permute(1, 2, 0).reshape(-1, C)
    _, _, V = torch.pca_lowrank(x, q=3)
    pca = torch.matmul(x, V[:, :3])
    pca = (pca - pca.min()) / (pca.max() - pca.min() + 1e-8)
    return pca.reshape(H, W, 3).cpu().numpy()

def _save_image(data, filename, title, save_dir):
    plt.figure(figsize=(6, 6))
    if data.shape[-1] == 1: # Single channel (Depth/Event)
        plt.imshow(data.squeeze(), cmap='gray')
    else:
        plt.imshow(data)
    plt.title(title); plt.axis('off')
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight'); plt.close()

def _save_heatmap(data, filename, title, save_dir, cmap='jet'):
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap=cmap)
    plt.title(title); plt.axis('off'); plt.colorbar()
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight'); plt.close()

def _save_pca(data, filename, title, save_dir):
    plt.figure(figsize=(6, 6)); plt.imshow(data)
    plt.title(title); plt.axis('off')
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight'); plt.close()

'''---'''


class LoRA_Sam_P1(LoRA_Sam):
    """
    LoRA_Sam with Adaptive Output Fusion (P1 Contribution).
    Inherits from LoRA_Sam to reuse LoRA injection and initialization logic.
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None):
        # 1. 부모 클래스(LoRA_Sam)의 __init__ 호출
        # 여기서 LoRA 레이어 주입(Surgery)과 초기화가 자동으로 수행됩니다.
        super().__init__(sam_model, r, lora_layer)
        
        # 2. [P1 Contribution] Adaptive Fusion을 위한 Confidence Head 추가
        # [Bug Fix] Correctly set input channels based on use_high_res_features_in_sam
        # SAM2 usually outputs features at stride 4, but the high-res feature fed to mask decoder
        # might be different depending on configuration.
        # Check if attribute exists, default to False if not present (safe check)
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        
        if use_high_res:
            # When True, the feature is from conv_s0 or similar low-level feature
            # The channel count is typically transformer_dim // 8 (e.g., 256 // 8 = 32)
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            # Standard feature dimension
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        self.confidence_head = ConfidenceHead(in_channels=fusion_dim)

    def forward(self, batched_input, multimask_output):
        """
        Override forward to implement Adaptive Output Fusion.
        """
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats, vision_pos_embeds, feat_sizes, output = [], [], [], [], [], []
        
        # 1. Image Encoding Loop (Base logic reuse)
        for i in range(m):
            img_emb = self.sam.forward_image(batched_input[i])
            image_embedding.append(img_emb)
            bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
            backbone_out.append(bb_out)
            vision_feats.append(v_feats)
            vision_pos_embeds.append(v_pos)
            feat_sizes.append(f_sizes)
        
        output_dict={
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        
        # 2. Modality Tracking Loop (MAM)
        for frame_idx in range(m):
            is_init = (frame_idx == 0)
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
            output_dict["cond_frame_outputs"][frame_idx] = multi_mask_output_step
            output.append(multi_mask_output_step["high_res_multimasks"])
        
        # 3. [P1 Contribution] Adaptive Output Fusion Logic
        # Calculate confidence scores for each modality
        confidences = []
        for i in range(m):
            # Use the high-res FPN feature for confidence estimation
            # image_embedding[i]['backbone_fpn'][0] shape: (B, C, H, W)
            feat = image_embedding[i]['backbone_fpn'][0]
            conf = self.confidence_head(feat) # (B, 1)
            confidences.append(conf)
            
        # Stack scores: (B, m)
        confidences = torch.cat(confidences, dim=1)
        
        # Normalize weights across modalities using Softmax
        # This makes the weights sum to 1, acting as a soft-selection mechanism
        weights = F.softmax(confidences, dim=1) # (B, m)
        
        # Perform Weighted Sum
        # Weights need to be reshaped to (B, 1, 1, 1) for broadcasting
        w0 = weights[:, 0].view(-1, 1, 1, 1)
        m_output = output[0] * w0
        m_feat = image_embedding[0]['backbone_fpn'][0] * w0
        
        for i in range(1, m):
            wi = weights[:, i].view(-1, 1, 1, 1)
            m_output = m_output + output[i] * wi
            m_feat = m_feat + image_embedding[i]['backbone_fpn'][0] * wi

        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        """
        Override to save confidence_head parameters along with LoRA weights.
        """
        assert filename.endswith(".pt") or filename.endswith('.pth')

        # Construct dictionary similar to parent, but including confidence_head
        num_layer = len(self.w_As)
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        # [New] Save Confidence Head
        confidence_tensors = {f"confidence_head.{k}": v for k, v in self.confidence_head.state_dict().items()}

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
            
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {
            **a_tensors, 
            **b_tensors, 
            **prompt_encoder_tensors, 
            **mask_decoder_tensors, 
            **confidence_tensors
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        """
        Override to load confidence_head parameters.
        """
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        # 1. Load LoRA weights (same logic as parent)
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            if saved_key in state_dict:
                w_A_linear.weight = Parameter(state_dict[saved_key])

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            if saved_key in state_dict:
                w_B_linear.weight = Parameter(state_dict[saved_key])
        
        # 2. [New] Load Confidence Head
        confidence_dict = {}
        for k, v in state_dict.items():
            if k.startswith("confidence_head."):
                confidence_dict[k.replace("confidence_head.", "")] = v
        if confidence_dict:
            self.confidence_head.load_state_dict(confidence_dict)

        # 3. Load SAM parts
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)
            
        self.sam.load_state_dict(sam_dict)

class LoRA_Sam_P2(LoRA_Sam_P1):
    """
    LoRA_Sam_P2: Adds Quality-Aware Memory Update (QAMU) on top of Adaptive Output Fusion (P1).
    
    Inheritance Chain: LoRA_Sam -> LoRA_Sam_P1 -> LoRA_Sam_P2
    - LoRA_Sam: Basic LoRA injection
    - LoRA_Sam_P1: Adds Adaptive Output Fusion (Weighted Sum)
    - LoRA_Sam_P2: Adds QAMU (Selective Memory Encoding based on Confidence)
    """

    def __init__(self, sam_model, r: int, lora_layer=None, qamu_threshold=0.4):
        # Initialize P1 (which initializes LoRA_Sam)
        # This sets up LoRA layers and the ConfidenceHead
        super().__init__(sam_model, r, lora_layer)
        
        # Hyperparameter for QAMU
        # If confidence < threshold, we skip memory encoding for that modality
        self.qamu_threshold = qamu_threshold

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats, vision_pos_embeds, feat_sizes, output = [], [], [], [], [], []
        
        # 1. Image Encoding Loop
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
        
        # --- Pre-calculate Confidences for QAMU & Adaptive Fusion ---
        # We need confidence scores *during* the tracking loop to decide on memory update
        confidences_list = []
        for i in range(m):
            feat = image_embedding[i]['backbone_fpn'][0] # (B, C, H, W)
            # Use the ConfidenceHead inherited from P1
            conf = self.confidence_head(feat) # (B, 1)
            confidences_list.append(conf)
            
        # 2. Modality Tracking Loop with QAMU
        for frame_idx in range(m):
            is_init = (frame_idx == 0)
            
            # [QAMU Logic]
            # Determine if we should update memory based on confidence
            current_conf = torch.sigmoid(confidences_list[frame_idx]) # Normalize to 0~1
            
            # We use the average confidence of the batch to make the decision
            # (Alternatively, could be per-sample decision but SAM2 memory API is batch-centric)
            avg_conf = current_conf.mean().item()
            
            # Rule: Always encode the first frame (to initialize memory), 
            # otherwise encode only if quality is sufficient.
            if is_init:
                run_mem_encoder = True
            else:
                run_mem_encoder = avg_conf > self.qamu_threshold
            

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
                # [QAMU Injection] Apply the decision here
                run_mem_encoder=run_mem_encoder,
                prev_sam_mask_logits=None,
            )
            output_dict["cond_frame_outputs"][frame_idx] = multi_mask_output_step
            output.append(multi_mask_output_step["high_res_multimasks"])
        
        # 3. Adaptive Output Fusion (Logic from P1)
        # We reuse the pre-calculated confidences
        confidences = torch.cat(confidences_list, dim=1) # (B, m)
        weights = F.softmax(confidences, dim=1) # (B, m)
        
        w0 = weights[:, 0].view(-1, 1, 1, 1)
        m_output = output[0] * w0
        m_feat = image_embedding[0]['backbone_fpn'][0] * w0
        
        for i in range(1, m):
            wi = weights[:, i].view(-1, 1, 1, 1)
            m_output = m_output + output[i] * wi
            m_feat = m_feat + image_embedding[i]['backbone_fpn'][0] * wi

        return m_output, m_feat


class LoRA_Sam_P3(LoRA_Sam_P2):
    """
    LoRA_Sam_P3 with Fixed Dimension Handling and MoE Integration.
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None, qamu_threshold=0.4, num_experts=4, top_k=2):
        nn.Module.__init__(self)

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
                
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            
            moe_q = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            moe_v = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            
            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)
            
            blk.attn.qkv = _MoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)
            
        self.sam = sam_model
        
        # [Bug Fix] Correctly set input channels based on SAM2 configuration
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        
        if use_high_res:
            # When True, the feature is from conv_s0 (transformer_dim // 8)
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        self.confidence_head = ConfidenceHead(in_channels=fusion_dim)
        self.qamu_threshold = qamu_threshold

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
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

class LoRA_Sam_P4(nn.Module):
    """
    LoRA_Sam_P4: MoE-LoRA + Adaptive Mask Fusion (AMF).
    (QAMU Removed)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None, num_experts=4, top_k=2):
        super().__init__()

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

        # Apply MoE-LoRA Surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
                
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            
            moe_q = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            moe_v = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            
            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)
            
            blk.attn.qkv = _MoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)
            
        self.sam = sam_model
        
        # [Bug Fix] Correctly set AMF input channels
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        
        if use_high_res:
            # When True, feature channel is reduced (e.g., 256 -> 32)
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        self.confidence_head = ConfidenceHead(in_channels=fusion_dim)

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats, vision_pos_embeds, feat_sizes, output = [], [], [], [], [], []
        
        # 1. Image Encoding Loop
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
        
        # 2. Modality Tracking Loop (MAM)
        # Note: QAMU is removed, so we run memory encoder for all frames (Standard behavior)
        for frame_idx in range(m):
            is_init = (frame_idx == 0)
            
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
                run_mem_encoder=True, # Always memorize
                prev_sam_mask_logits=None,
            )
            output_dict["cond_frame_outputs"][frame_idx] = multi_mask_output_step
            output.append(multi_mask_output_step["high_res_multimasks"])
        
        # 3. Adaptive Mask Fusion (AMF)
        confidences = []
        for i in range(m):
            feat = image_embedding[i]['backbone_fpn'][0]
            conf = self.confidence_head(feat) # (B, 1)
            confidences.append(conf)
            
        # Stack scores & Softmax
        confidences = torch.cat(confidences, dim=1) # (B, m)
        weights = F.softmax(confidences, dim=1)     # (B, m)
        
        # Weighted Sum
        w0 = weights[:, 0].view(-1, 1, 1, 1)
        m_output = output[0] * w0
        m_feat = image_embedding[0]['backbone_fpn'][0] * w0
        
        for i in range(1, m):
            wi = weights[:, i].view(-1, 1, 1, 1)
            m_output = m_output + output[i] * wi
            m_feat = m_feat + image_embedding[i]['backbone_fpn'][0] * wi

        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
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


class LoRA_Sam_P5(nn.Module):
    """
    LoRA_Sam_P5:
    1. MoE-LoRA (Structure)
    2. Differentiable Soft-Gating Memory 
    3. Adaptive Mask Fusion (AMF)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None, num_experts=4, top_k=2):
        super().__init__()
        
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            
            moe_q = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            moe_v = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            
            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)
            
            blk.attn.qkv = _MoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)
            
        self.sam = sam_model
        
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        self.confidence_head = ConfidenceHead(in_channels=fusion_dim)


    def modulate_features_with_soft_gating(self, vision_feats_list, score_source_feat):
        """
        [Original] Differentiable Memory Modulation using sigmoid.
        """
        # 1. Calculate Confidence
        logits = self.confidence_head(score_source_feat) # (B, 1)
        scores = torch.sigmoid(logits) # (B, 1)
        
        # 2. Correct Reshape for Broadcasting
        # vision_feats_list elements are (HW, B, C) - 3 Dimensions
        # scores: (B, 1) -> (1, B, 1) to match (HW, B, C)
        scores_expanded = scores.transpose(0, 1).unsqueeze(-1) # (1, B, 1)
        
        # 3. Apply Modulation
        modulated_list = [feat * scores_expanded for feat in vision_feats_list]
        
        return modulated_list, logits

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats, vision_pos_embeds, feat_sizes, output = [], [], [], [], [], []
        
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
        
        modality_logits = []

        for frame_idx in range(m):
            is_init = (frame_idx == 0)
            
            score_source = image_embedding[frame_idx]['backbone_fpn'][0]
            
            # Use corrected modulation function
            modulated_vision_feats, current_logits = self.modulate_features_with_soft_gating(
                vision_feats[frame_idx], 
                score_source
            )
            modality_logits.append(current_logits)

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
        
        all_logits = torch.cat(modality_logits, dim=1) # (B, m)
        weights = F.softmax(all_logits, dim=1)         # (B, m)
        
        w0 = weights[:, 0].view(-1, 1, 1, 1)
        m_output = output[0] * w0
        m_feat = image_embedding[0]['backbone_fpn'][0] * w0
        
        for i in range(1, m):
            wi = weights[:, i].view(-1, 1, 1, 1)
            m_output = m_output + output[i] * wi
            m_feat = m_feat + image_embedding[i]['backbone_fpn'][0] * wi

        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
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


class LoRA_Sam_P6(LoRA_Sam_P5):
    """
    LoRA_Sam_P6:
    Inherits from LoRA_Sam_P5 (MoE-LoRA + UDMM + AMF)
    but uses a deeper ConfidenceHeadV2 for better uncertainty estimation.
    """

    def __init__(self, sam_model, r: int, lora_layer=None, num_experts=4, top_k=2):
        # Initialize P5 first (this sets up MoE-LoRA layers and basic structure)
        super().__init__(sam_model, r, lora_layer, num_experts, top_k)
        
        # Override the confidence_head with the V2 (Deep) version
        # We need to determine the correct input dimension again
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # Replace the head
        self.confidence_head = ConfidenceHeadV2(in_channels=fusion_dim)


class LoRA_Sam_P7(nn.Module):
    """
    LoRA_Sam_P7: Improved Modality-Aware Adaptive Fusion
    
    Key Improvement over P5/P6:
    - P5/P6: Uses sigmoid for modulation → saturates when logit > 3 (all ~0.97)
    - P7: Uses SOFTMAX for modulation → always relative comparison (sum to 1)
    
    This ensures meaningful differentiation between modalities regardless of logit magnitude.
    
    Architecture:
    1. MoE-LoRA (Mixture of Experts LoRA for multi-modal adaptation)
    2. Softmax-based Relative Modulation (NOT sigmoid!)
    3. Adaptive Mask Fusion (AMF) with softmax weights
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None, num_experts=4, top_k=2):
        super().__init__()
        
        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            
            moe_q = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            moe_v = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            
            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)
            
            blk.attn.qkv = _MoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)
            
        self.sam = sam_model
        
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # Use ConfidenceHeadV2 for better uncertainty estimation
        self.confidence_head = ConfidenceHeadV2(in_channels=fusion_dim)
        
        # Store for analysis
        self.last_modality_logits = None
        self.last_modality_weights = None

    def modulate_features_with_relative_weight(self, vision_feats_list, modulation_weight):
        """
        [P7 Key Innovation] Differentiable Memory Modulation using RELATIVE weights.
        
        Unlike P5/P6 which uses sigmoid(logits) leading to saturation,
        P7 uses softmax-normalized weights ensuring meaningful differentiation.
        
        Args:
            vision_feats_list: List of vision features (HW, B, C)
            modulation_weight: Softmax-normalized weight for this modality (B, 1)
                              Range: [0, 1], sum across modalities = 1
        
        Returns:
            modulated_list: Modulated vision features
        """
        # Reshape for broadcasting: (B, 1) -> (1, B, 1) to match (HW, B, C)
        weight_expanded = modulation_weight.transpose(0, 1).unsqueeze(-1)  # (1, B, 1)
        
        # Apply modulation with relative weight
        modulated_list = [feat * weight_expanded for feat in vision_feats_list]
        
        return modulated_list

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats, vision_pos_embeds, feat_sizes, output = [], [], [], [], [], []
        
        # ============================================
        # Phase 1: Extract features from ALL modalities first
        # ============================================
        for i in range(m):
            img_emb = self.sam.forward_image(batched_input[i])
            image_embedding.append(img_emb)
            bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
            backbone_out.append(bb_out)
            vision_feats.append(v_feats)
            vision_pos_embeds.append(v_pos)
            feat_sizes.append(f_sizes)
        
        # ============================================
        # Phase 2: Compute confidence logits for ALL modalities
        # Then apply SOFTMAX for relative comparison
        # ============================================
        modality_logits = []
        for i in range(m):
            score_source = image_embedding[i]['backbone_fpn'][0]  # (B, C, H, W)
            logits = self.confidence_head(score_source)  # (B, 1)
            modality_logits.append(logits)
        
        # Stack logits and compute SOFTMAX weights (not sigmoid!)
        # This ensures relative comparison: weights sum to 1 per sample
        all_logits = torch.cat(modality_logits, dim=1)  # (B, m)
        modality_weights = F.softmax(all_logits, dim=1)  # (B, m)
        
        # Store for analysis
        self.last_modality_logits = all_logits.detach()
        self.last_modality_weights = modality_weights.detach()
        
        # ============================================
        # Phase 3: Apply modulation with RELATIVE weights, then track
        # ============================================
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }

        for frame_idx in range(m):
            is_init = (frame_idx == 0)
            
            # Get the softmax weight for this modality: (B,) -> (B, 1)
            current_weight = modality_weights[:, frame_idx].unsqueeze(1)  # (B, 1)
            
            # Modulate features using RELATIVE weight (softmax-based)
            modulated_vision_feats = self.modulate_features_with_relative_weight(
                vision_feats[frame_idx], 
                current_weight
            )

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
        # Phase 4: Adaptive Mask Fusion using the same softmax weights
        # ============================================
        w0 = modality_weights[:, 0].view(-1, 1, 1, 1)
        m_output = output[0] * w0
        m_feat = image_embedding[0]['backbone_fpn'][0] * w0
        
        for i in range(1, m):
            wi = modality_weights[:, i].view(-1, 1, 1, 1)
            m_output = m_output + output[i] * wi
            m_feat = m_feat + image_embedding[i]['backbone_fpn'][0] * wi

        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
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


class LoRA_Sam_P9(nn.Module):
    """
    LoRA_Sam_P9: Cross-Modal Dual-Path Architecture

    P8의 문제점:
      - UAMM: sigmoid 포화 (logit > 3 → score ≈ 1.0) → 사실상 비활성화 (no-op)
      - AMF: 동일 score 재사용 → 항상 1/3 균등 분배
      - 로짓을 강제로 낮추면 feat * score에서 feature 약화 → 성능 하락

    P9 핵심 개선:
      1. ConfidenceHeadV2(독립 평가) → CrossModalFusionHead(상대 비교) 교체
         - 모든 모달리티 feature를 동시에 보고 상대 중요도를 softmax로 산출
      2. UAMM: max-normalized softmax → best modality = 1.0, 나머지 상대적 억제
         - best modality feature 완전 보존 (sigmoid 포화 문제 해결)
         - worst modality는 상대적으로 억제 (차별화 보장)
      3. AMF: raw softmax weights → 차별화된 출력 융합 가중치
      4. 2-pass forward: 전체 encode → cross-modal 비교 → modulate+track

    Components:
      1. Structure: Soft-MoE LoRA (P8과 동일)
      2. UAMM (Memory): CrossModal max-normalized softmax → Feature Modulation
      3. AMF (Fusion): CrossModal raw softmax → Output Fusion
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3):
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

        # Inject SoftMoE-LoRA (P8과 동일)
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

        # Determine Fusion Dim
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # [P9 핵심] CrossModalFusionHead: ConfidenceHeadV2 대체
        self.cross_modal_head = CrossModalFusionHead(
            in_channels=fusion_dim,
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
            # Phase 1: 모든 모달리티 Image Encoding
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Cross-Modal 가중치 산출
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
            cross_weights, cross_logits = self.cross_modal_head(all_backbone_feats)  # (B, m)

            # UAMM용: max-normalize → best modality = 1.0, 나머지 상대적 억제
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m)

            # ============================================
            # Phase 3: UAMM Modulation + Tracking
            # ============================================
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            feats_before_uamm = []
            feats_after_uamm = []

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                # UAMM: max-normalized score로 feature modulation
                current_score = uamm_scores[:, frame_idx].unsqueeze(1)  # (B, 1)
                score_expanded = current_score.transpose(0, 1).unsqueeze(-1)  # (1, B, 1)

                if not self.training:
                    feats_before_uamm.append(vision_feats[frame_idx][0].detach().cpu())

                modulated_vision_feats = [feat * score_expanded for feat in vision_feats[frame_idx]]

                if not self.training:
                    feats_after_uamm.append(modulated_vision_feats[0].detach().cpu())

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
            amf_weights = cross_weights  # (B, m), 이미 softmax 정규화

            w0 = amf_weights[:, 0].view(-1, 1, 1, 1)
            m_output = output[0] * w0
            m_feat = all_backbone_feats[0] * w0

            for i in range(1, m):
                wi = amf_weights[:, i].view(-1, 1, 1, 1)
                m_output = m_output + output[i] * wi
                m_feat = m_feat + all_backbone_feats[i] * wi

            # Store for visualization (val_multiaqua.py 호환)
            self._last_uamm_scores = uamm_scores.detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().float().cpu().numpy()
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None
            # Feature analysis: per-modal outputs, backbone feats
            self._last_per_modal_outputs = [o.detach().cpu() for o in output]
            self._last_per_modal_feats = [f.detach().cpu() for f in all_backbone_feats]
            # UAMM 전/후 feature 비교 (ISSUE-018)
            self._last_feats_before_uamm = feats_before_uamm if not self.training else None
            self._last_feats_after_uamm = feats_after_uamm if not self.training else None
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

class LoRA_Sam_P12(nn.Module):
    """
    LoRA_Sam_P12: Input-Conditioned Soft MoE LoRA

    P9의 문제점:
      - MoE routing이 정적: 모든 이미지에서 동일한 routing (Block9에서 E1 사망)
      - UAMM/AMF 값 정적: img=0.745, lidar=0.961, thermal=1.0 (val/test 전체 동일)
      - 원인: frozen encoder의 deep feature가 quality 정보를 소실

    P12 핵심 개선:
      1. Raw RGB input statistics(mean, std)를 MoE gating에 condition으로 주입
         - Night RGB(mean≈-2.0) vs Day RGB(mean≈0.0) → 다른 expert routing
      2. 동일 condition을 CrossModalFusionHead에도 주입
         - RGB quality에 따라 UAMM/AMF weight가 adaptive하게 변경
      3. Zero-init: 시작점 = P9과 동일, 점진적 학습
      4. Loss 추가 없음: main OHEM + proto loss로만 학습 (P10/P11 실패 원인 회피)
      5. RGB만 condition (thermal/lidar는 day/night quality 변화 없음)

    Components:
      1. Structure: Input-Conditioned Soft-MoE LoRA (cond_dim=6)
      2. UAMM (Memory): Input-Conditioned CrossModal max-normalized softmax
      3. AMF (Fusion): Input-Conditioned CrossModal raw softmax
    """

    RGB_COND_DIM = 6  # 3ch mean + 3ch std

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3):
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

        # Inject Input-Conditioned SoftMoE-LoRA
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts,
                                        cond_dim=self.RGB_COND_DIM)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts,
                                        cond_dim=self.RGB_COND_DIM)

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # Determine Fusion Dim
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # [P12] Input-Conditioned CrossModalFusionHead
        self.cross_modal_head = CrossModalFusionHead(
            in_channels=fusion_dim,
            num_modalities=num_modalities,
            cond_dim=self.RGB_COND_DIM,
        )

    @staticmethod
    def compute_rgb_stats(rgb_tensor):
        """Compute per-channel mean and std from normalized RGB input.

        Args:
            rgb_tensor: (B, 3, H, W) — ImageNet-normalized RGB tensor
        Returns:
            stats: (B, 6) — [ch0_mean, ch1_mean, ch2_mean, ch0_std, ch1_std, ch2_std]
        """
        mean = rgb_tensor.mean(dim=[2, 3])  # (B, 3)
        std = rgb_tensor.std(dim=[2, 3])    # (B, 3)
        return torch.cat([mean, std], dim=1)  # (B, 6)

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
            # Phase 0: RGB stats 계산 (encoding 전, raw input에서)
            # ============================================
            rgb_stats = self.compute_rgb_stats(batched_input[0])  # (B, 6)

            # ============================================
            # Phase 1: 모든 모달리티 Image Encoding
            # ============================================
            for i in range(m):
                # [P12] RGB일 때만 condition 설정, 나머지는 None
                cond = rgb_stats if i == 0 else None
                for layer in self.moe_layers_q:
                    layer.set_condition(cond)
                for layer in self.moe_layers_v:
                    layer.set_condition(cond)

                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # Clear conditions after encoding
            for layer in self.moe_layers_q:
                layer.set_condition(None)
            for layer in self.moe_layers_v:
                layer.set_condition(None)

            # ============================================
            # Phase 2: Cross-Modal 가중치 산출 (with condition)
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
            cross_weights, cross_logits = self.cross_modal_head(
                all_backbone_feats, condition=rgb_stats
            )  # (B, m)

            # UAMM용: max-normalize → best modality = 1.0, 나머지 상대적 억제
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m)

            # ============================================
            # Phase 3: UAMM Modulation + Tracking
            # ============================================
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                # UAMM: max-normalized score로 feature modulation
                current_score = uamm_scores[:, frame_idx].unsqueeze(1)  # (B, 1)
                score_expanded = current_score.transpose(0, 1).unsqueeze(-1)  # (1, B, 1)
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
            amf_weights = cross_weights  # (B, m), 이미 softmax 정규화

            w0 = amf_weights[:, 0].view(-1, 1, 1, 1)
            m_output = output[0] * w0
            m_feat = all_backbone_feats[0] * w0

            for i in range(1, m):
                wi = amf_weights[:, i].view(-1, 1, 1, 1)
                m_output = m_output + output[i] * wi
                m_feat = m_feat + all_backbone_feats[i] * wi

            # Store for visualization (val_multiaqua.py 호환)
            self._last_uamm_scores = uamm_scores.detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().float().cpu().numpy()
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None
        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None
            # Ensure conditions are cleared
            for layer in self.moe_layers_q:
                layer.set_condition(None)
            for layer in self.moe_layers_v:
                layer.set_condition(None)

        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        # SoftMoE Parameters (includes cond_proj)
        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        cross_modal_tensors = {
            f"cross_modal_head.{k}": v
            for k, v in self.cross_modal_head.state_dict().items()
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
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        # Load SoftMoE Layers (includes cond_proj)
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict: mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict: mv.load_state_dict(state_dict[v_key])

        # Load CrossModalFusionHead (includes cond_compare)
        cross_modal_dict = {}
        for k, v in state_dict.items():
            if k.startswith("cross_modal_head."):
                cross_modal_dict[k.replace("cross_modal_head.", "")] = v
        if cross_modal_dict:
            self.cross_modal_head.load_state_dict(cross_modal_dict)

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


class LoRA_Sam_P10(nn.Module):
    """
    LoRA_Sam_P10: Quality-Aware Adaptive Gating

    P9 문제:
      - CrossModalFusionHead가 semantic feature만 보고 품질을 판단 → gating 상수화
      - UAMM/AMF 값이 200장 전체에서 고정 (thermal=1.0, lidar=0.96, img=0.74)
      - Training signal 없음: seg_loss 최소화에 constant gating으로도 충분

    P10 개선:
      1. CrossModalFusionHeadV2: GAP+GMP+Std multi-pool + per-modality compress
         - Std가 텍스처/노이즈 정도를 품질 proxy로 제공
         - Per-modal compress로 모달리티별 특성 독립 학습
      2. Per-modality Auxiliary Loss (gating oracle supervision):
         - 각 모달리티의 backbone feature → ModalAuxHead → 독립 segmentation 예측
         - 예측 품질(aux loss)이 낮을수록 해당 모달리티에 높은 oracle weight
         - KL(AMF || oracle): gating이 oracle을 따르도록 학습

    Components:
      1. Structure: Soft-MoE LoRA (P8/P9와 동일)
      2. Gating: CrossModalFusionHeadV2 (multi-pool + per-modal compress)
      3. UAMM: max-normalized softmax → Feature Modulation
      4. AMF: raw softmax → Output Fusion
      5. Aux Heads: ModalAuxHead × num_modalities (gating oracle 생성)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3, num_classes=4):
        nn.Module.__init__(self)

        assert r > 0
        self.num_modalities = num_modalities

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

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

        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # [P10] CrossModalFusionHeadV2: multi-pool + per-modality compress
        self.cross_modal_head = CrossModalFusionHeadV2(
            in_channels=fusion_dim,
            num_modalities=num_modalities,
        )

        # [P10] Per-modality auxiliary segmentation heads (gating oracle 생성)
        self.aux_heads = nn.ModuleList([
            ModalAuxHead(in_channels=fusion_dim, num_classes=num_classes)
            for _ in range(num_modalities)
        ])

        # Visualization buffers (항상 detach)
        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        try:
            # ============================================================
            # Phase 1: 모든 모달리티 Image Encoding
            # ============================================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================================
            # Phase 2: Cross-Modal 가중치 산출 (CrossModalFusionHeadV2)
            # ============================================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
            cross_weights, cross_logits = self.cross_modal_head(all_backbone_feats)  # (B, m)

            # UAMM: max-normalize → best modality = 1.0, 나머지 상대적 억제
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)   # (B, m)

            # [P10] Per-modality aux predictions
            # backbone feature에서 독립 예측 (track_step 공유 memory와 무관)
            aux_outputs = [self.aux_heads[i](all_backbone_feats[i]) for i in range(m)]

            # ============================================================
            # Phase 3: UAMM Modulation + Tracking
            # ============================================================
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                current_score = uamm_scores[:, frame_idx].unsqueeze(1)        # (B, 1)
                score_expanded = current_score.transpose(0, 1).unsqueeze(-1)   # (1, B, 1)
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

            # ============================================================
            # Phase 4: AMF — raw softmax weights로 Output Fusion
            # ============================================================
            amf_weights = cross_weights  # (B, m)

            w0 = amf_weights[:, 0].view(-1, 1, 1, 1)
            m_output = output[0] * w0
            m_feat = all_backbone_feats[0] * w0

            for i in range(1, m):
                wi = amf_weights[:, i].view(-1, 1, 1, 1)
                m_output = m_output + output[i] * wi
                m_feat = m_feat + all_backbone_feats[i] * wi

            # Visualization용 버퍼 (detach)
            self._last_uamm_scores = uamm_scores.detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().float().cpu().numpy()
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None

        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        # [P10] aux_outputs, amf_weights를 리턴값에 포함
        # → DDP가 aux_heads 파라미터를 "used"로 인식하여 gradient hook 충돌 방지
        if self.training:
            return m_output, m_feat, aux_outputs, amf_weights
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        cross_modal_tensors = {
            f"cross_modal_head.{k}": v
            for k, v in self.cross_modal_head.state_dict().items()
        }

        aux_heads_tensors = {
            f"aux_heads.{k}": v
            for k, v in self.aux_heads.state_dict().items()
        }

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        model_ref = self.sam.module if isinstance(
            self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ) else self.sam
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
            **aux_heads_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict:
                mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict:
                mv.load_state_dict(state_dict[v_key])

        # CrossModalFusionHeadV2
        cross_modal_dict = {
            k.replace("cross_modal_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("cross_modal_head.")
        }
        if cross_modal_dict:
            self.cross_modal_head.load_state_dict(cross_modal_dict)

        # ModalAuxHeads
        aux_heads_dict = {
            k.replace("aux_heads.", ""): v
            for k, v in state_dict.items()
            if k.startswith("aux_heads.")
        }
        if aux_heads_dict:
            self.aux_heads.load_state_dict(aux_heads_dict)

        # SAM components
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)

        self.sam.load_state_dict(sam_dict)


class LoRA_Sam_P11(nn.Module):
    """
    LoRA_Sam_P11: MI-based MoE Expert Specialization

    P9/P10 문제:
      - Soft-MoE gate가 uniform으로 수렴 (E0≈0.33, E1≈0.32, E2≈0.35)
        → 사실상 단일 LoRA와 동일 → expert specialization 미발생
      - 원인: seg_loss 단독으로는 expert 분화 gradient 부족
      - AMF/UAMM도 상수로 수렴 (연쇄 문제)

    P11 해결:
      Mutual Information Maximization으로 MoE routing 학습.
      MI(Input, Expert) = H(Expert_marginal) - H(Expert|Input)
      = "전체적으로 expert를 골고루 쓰되, 각 입력에 대해선 특정 expert를 확실히 써라"

      - Modality label 불사용, for문 순서 불사용
      - 순수 정보이론 기반, gradient가 gate network로 직접 흐름
      - MoE expert 분화 → feature가 modality-adaptive → feature-level oracle 강화

    Loss 구조:
      total = seg_loss + proto_loss + λ_gate * gating_loss + λ_mi * mi_loss
      gating_loss = oracle_kl + 0.3 * aux_seg  (P10 동일)
      mi_loss = H(gate|input) - H(gate_marginal)  (minimize → MI 최대화)

    Components (P10 base + MI loss):
      1. Structure: Soft-MoE LoRA (동일) + gradient-enabled gate 수집
      2. Gating: CrossModalFusionHeadV2 (P10 동일)
      3. UAMM: softmax with temperature (max-norm 대체)
      4. AMF: raw softmax weights (P10 동일)
      5. Aux Heads: ModalAuxHead × num_modalities (P10 동일)
      6. NEW: per-modality gate distribution → MI routing loss
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3, num_classes=4):
        nn.Module.__init__(self)

        assert r > 0
        self.num_modalities = num_modalities

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

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

        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # [P10 계승] CrossModalFusionHeadV2: multi-pool + per-modality compress
        self.cross_modal_head = CrossModalFusionHeadV2(
            in_channels=fusion_dim,
            num_modalities=num_modalities,
        )

        # [P10 계승] Per-modality auxiliary segmentation heads (gating oracle)
        self.aux_heads = nn.ModuleList([
            ModalAuxHead(in_channels=fusion_dim, num_classes=num_classes)
            for _ in range(num_modalities)
        ])

        # [P11] UAMM: softmax with temperature (max-norm 대체)
        self.uamm_temperature = 2.0

        # Visualization buffers (항상 detach)
        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # Visualization용 (detach)
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)

        all_moe_layers = list(self.moe_layers_q) + list(self.moe_layers_v)

        try:
            # ============================================================
            # Phase 1: 모든 모달리티 Image Encoding + Gate Distribution 수집
            # ============================================================
            per_modal_gate_dists = []  # List of (E,) per modality

            for i in range(m):
                # 각 modality encoding 시 gradient gate 수집
                grad_collector = []
                for layer in all_moe_layers:
                    layer._gate_callback = _moe_gate_cb
                    layer._grad_gate_collector = grad_collector

                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

                # 이 modality의 평균 gate distribution (전 layer 평균)
                if grad_collector:
                    modal_gate = torch.stack(grad_collector, dim=0).mean(dim=0)  # (E,)
                    per_modal_gate_dists.append(modal_gate)

                # Collector 정리 (다음 modality를 위해)
                for layer in all_moe_layers:
                    layer._grad_gate_collector = None

            # ============================================================
            # Phase 2: Cross-Modal 가중치 산출 (CrossModalFusionHeadV2)
            # ============================================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
            cross_weights, cross_logits = self.cross_modal_head(all_backbone_feats)  # (B, m)

            # [P11] UAMM: softmax with temperature (max-norm 대체)
            uamm_scores = F.softmax(cross_logits / self.uamm_temperature, dim=1)  # (B, m)
            uamm_scores = uamm_scores * m  # 범위 [0, m], 평균=1

            # Per-modality aux predictions
            aux_outputs = [self.aux_heads[i](all_backbone_feats[i]) for i in range(m)]

            # ============================================================
            # Phase 3: UAMM Modulation + Tracking
            # ============================================================
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                current_score = uamm_scores[:, frame_idx].unsqueeze(1)        # (B, 1)
                score_expanded = current_score.transpose(0, 1).unsqueeze(-1)   # (1, B, 1)
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

            # ============================================================
            # Phase 4: AMF — Output Fusion
            # ============================================================
            amf_weights = cross_weights  # (B, m)

            w0 = amf_weights[:, 0].view(-1, 1, 1, 1)
            m_output = output[0] * w0
            m_feat = all_backbone_feats[0] * w0

            for i in range(1, m):
                wi = amf_weights[:, i].view(-1, 1, 1, 1)
                m_output = m_output + output[i] * wi
                m_feat = m_feat + all_backbone_feats[i] * wi

            # Visualization용 버퍼 (detach)
            self._last_uamm_scores = uamm_scores.detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().float().cpu().numpy()
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None

        finally:
            for layer in all_moe_layers:
                layer._gate_callback = None
                layer._grad_gate_collector = None

        if self.training:
            return m_output, m_feat, aux_outputs, amf_weights, per_modal_gate_dists
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        cross_modal_tensors = {
            f"cross_modal_head.{k}": v
            for k, v in self.cross_modal_head.state_dict().items()
        }

        aux_heads_tensors = {
            f"aux_heads.{k}": v
            for k, v in self.aux_heads.state_dict().items()
        }

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        model_ref = self.sam.module if isinstance(
            self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ) else self.sam
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
            **aux_heads_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict:
                mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict:
                mv.load_state_dict(state_dict[v_key])

        # CrossModalFusionHeadV2
        cross_modal_dict = {
            k.replace("cross_modal_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("cross_modal_head.")
        }
        if cross_modal_dict:
            self.cross_modal_head.load_state_dict(cross_modal_dict)

        # ModalAuxHeads
        aux_heads_dict = {
            k.replace("aux_heads.", ""): v
            for k, v in state_dict.items()
            if k.startswith("aux_heads.")
        }
        if aux_heads_dict:
            self.aux_heads.load_state_dict(aux_heads_dict)

        # SAM components
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)

        self.sam.load_state_dict(sam_dict)


# ================================================================
# P13: Energy-Confidence Fusion + Expert Collapse Fix
# ================================================================

class ConfidenceAuxHead(nn.Module):
    """공유 auxiliary segmentation head.
    모든 모달리티가 동일한 head를 사용하여 파라미터 최소화.
    Energy score 계산을 위한 raw logit을 출력한다 (softmax 적용 전).
    """

    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        mid_channels = max(in_channels // 4, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, feat):
        """
        Args:
            feat: backbone feature (B, C, H, W)
        Returns:
            logits: (B, num_classes, H, W) — raw logits
        """
        return self.head(feat)


class ModalAuxDecoder(nn.Module):
    """P14 전용: 모달리티별 독립 auxiliary segmentation decoder.

    P13의 공유 ConfidenceAuxHead와 달리 각 모달리티가 고유 파라미터를 가짐.
    첫 conv를 3×3으로 변경 → 모달리티별 공간 패턴에 특화:
      - RGB: 텍스처 경계 / 색상 gradient
      - LiDAR: 점군 집적 패턴 / 깊이 불연속
      - Thermal: 온도 gradient / 열 분포 형태
    Energy score 계산을 위한 raw logit 출력 (softmax 미적용).
    """

    def __init__(self, in_channels, num_classes=4):
        super().__init__()
        mid_channels = max(in_channels // 4, 32)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),  # 3×3 spatial context
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(self, feat):
        """
        Args:
            feat: backbone feature (B, C, H, W)
        Returns:
            logits: (B, num_classes, H, W) — raw logits
        """
        return self.decoder(feat)


class MultiScaleModalAuxDecoder(nn.Module):
    """P17: Multi-scale FPN feature를 활용한 aux segmentation decoder.

    기존 ModalAuxDecoder는 backbone_fpn[0](32ch)만 사용.
    이 decoder는 3개 FPN 레벨 모두 활용:
      - fpn[0]: 256×256, 32ch  (high-res spatial detail)
      - fpn[1]: 128×128, 64ch  (mid-level features)
      - fpn[2]:  64×64, 256ch  (semantic context)

    전략: 각 레벨을 proj_dim으로 project → fpn[0] 해상도로 upsample → concat → decode
    """

    def __init__(self, fpn_channels=(32, 64, 256), proj_dim=32, num_classes=4):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, proj_dim, 1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for ch in fpn_channels
        ])

        fused_dim = proj_dim * len(fpn_channels)  # 32 * 3 = 96
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_dim, fused_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_dim // 2, num_classes, 1),
        )

    def forward(self, fpn_feats):
        """
        Args:
            fpn_feats: list of 3 tensors [fpn0, fpn1, fpn2]
                fpn0: (B, 32, H0, W0)   — highest resolution
                fpn1: (B, 64, H1, W1)   — mid resolution
                fpn2: (B, 256, H2, W2)  — lowest resolution
        Returns:
            logits: (B, num_classes, H0, W0) — raw logits at fpn[0] resolution
        """
        target_size = fpn_feats[0].shape[2:]

        projected = []
        for i, feat in enumerate(fpn_feats):
            p = self.proj_layers[i](feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear',
                                  align_corners=False)
            projected.append(p)

        fused = torch.cat(projected, dim=1)
        return self.decoder(fused)


class ResNetAuxBackbone(nn.Module):
    """P18: Trainable ResNet-18 aux backbone with per-modality input stems.

    Frozen SAM2 FPN feature의 정보량 한계(ISSUE-008)를 극복하기 위해
    ImageNet pretrained ResNet-18을 trainable aux backbone으로 사용.

    구조: 3개 독립 stem(모달리티별 low-level 특화) + 1개 공유 body(layer1~3)
    Output: layer2(128ch, H/8) + layer3(256ch, H/16)
    Params: ~11.2M (shared body) + ~28K (3 stems)
    """

    def __init__(self, num_modalities=3, pretrained=True):
        super().__init__()

        resnet = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Per-modality input stems (conv1+bn1 replacement)
        # 모든 모달리티는 dataset에서 3ch로 repeat → 입력 3ch 통일
        self.stems = nn.ModuleList()
        for _ in range(num_modalities):
            stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            if pretrained:
                stem[0].weight.data.copy_(resnet.conv1.weight.data)
                stem[1].weight.data.copy_(resnet.bn1.weight.data)
                stem[1].bias.data.copy_(resnet.bn1.bias.data)
                stem[1].running_mean.copy_(resnet.bn1.running_mean)
                stem[1].running_var.copy_(resnet.bn1.running_var)
            self.stems.append(stem)

        # Shared body: maxpool + layer1~3
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # 64ch, H/4
        self.layer2 = resnet.layer2   # 128ch, H/8
        self.layer3 = resnet.layer3   # 256ch, H/16

    def forward(self, x, modal_idx):
        """Extract multi-scale features for a single modality.

        Args:
            x: (B, 3, H, W) input tensor
            modal_idx: modality index (0=RGB, 1=LiDAR, 2=Thermal)
        Returns:
            (layer2_feat, layer3_feat): tuple
                layer2_feat: (B, 128, H/8, W/8)
                layer3_feat: (B, 256, H/16, W/16)
        """
        x = self.stems[modal_idx](x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layer2_feat = self.layer2(x)
        layer3_feat = self.layer3(layer2_feat)
        return layer2_feat, layer3_feat


class ResNetAuxDecoder(nn.Module):
    """P18: ResNet layer2(128ch) + layer3(256ch) → aux segmentation logits.

    MultiScaleModalAuxDecoder와 유사하나 ResNet 2-level feature 사용.
    Output resolution: layer2와 동일 (H/8 × W/8 = 128×128 for 1024 input)
    Params: ~53K per modality
    """

    def __init__(self, resnet_channels=(128, 256), proj_dim=32, num_classes=4):
        super().__init__()
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, proj_dim, 1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for ch in resnet_channels
        ])

        fused_dim = proj_dim * len(resnet_channels)  # 32 * 2 = 64
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_dim, fused_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_dim, num_classes, 1),
        )

    def forward(self, resnet_feats):
        """
        Args:
            resnet_feats: (layer2_feat, layer3_feat)
                layer2_feat: (B, 128, H2, W2)
                layer3_feat: (B, 256, H3, W3) where H3=H2/2
        Returns:
            logits: (B, num_classes, H2, W2) at layer2 resolution
        """
        layer2_feat, layer3_feat = resnet_feats
        target_size = layer2_feat.shape[2:]

        projected = []
        for i, feat in enumerate([layer2_feat, layer3_feat]):
            p = self.proj_layers[i](feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear',
                                  align_corners=False)
            projected.append(p)

        fused = torch.cat(projected, dim=1)
        return self.decoder(fused)


def compute_energy_confidence(aux_logits_list, temperature=1.0):
    """Energy score 기반 modality confidence 계산.

    학습 가능 파라미터 없음 — computed signal이므로 상수로 수렴 불가.
    학습/추론 동일 메커니즘 — oracle-at-train / guess-at-test 불일치 없음.

    Args:
        aux_logits_list: List[Tensor], 길이 = num_modalities
            각 원소: (B, num_classes, H, W) raw logits
        temperature: energy score temperature (default 1.0)
    Returns:
        weights: (B, num_modalities) — softmax normalized fusion weights
    """
    confidences = []
    for z in aux_logits_list:
        # Energy score: E(x) = -T * log(sum(exp(z_k / T)))
        energy = -temperature * torch.logsumexp(z / temperature, dim=1)  # (B, H, W)
        # 높은 confidence = 낮은 energy (더 음수) → -energy가 클수록 confident
        conf = -energy.mean(dim=[1, 2])  # (B,)
        confidences.append(conf)

    confidences = torch.stack(confidences, dim=1)  # (B, num_modalities)
    weights = F.softmax(confidences / temperature, dim=1)  # (B, num_modalities)
    return weights


def compute_spatial_energy_confidence(aux_logits_list, temperature=1.0):
    """P15 전용: spatial-wise modality confidence map 계산.

    compute_energy_confidence()에서 spatial mean(.mean(dim=[1,2]))을 제거해
    위치별 서로 다른 모달리티 가중치 map을 반환한다.

    Args:
        aux_logits_list: List[Tensor], 길이 = num_modalities
            각 원소: (B, num_classes, H_feat, W_feat) raw logits
        temperature: energy score temperature (default 1.0)
    Returns:
        weights: (B, num_modalities, H_feat, W_feat) — 위치별 softmax 가중치
    """
    conf_maps = []
    for z in aux_logits_list:
        energy = -temperature * torch.logsumexp(z / temperature, dim=1)
        conf_map = -energy
        conf_maps.append(conf_map)

    stacked = torch.stack(conf_maps, dim=1)
    weights = F.softmax(stacked / temperature, dim=1)
    return weights


def compute_spatial_entropy_confidence(aux_logits_list, temperature=1.0, num_classes=4):
    """P15 전용: Calibrated Entropy 기반 spatial confidence map.

    Energy Score 대신 예측 분포의 entropy를 사용.
    - Energy: logit magnitude 기반 → "자신있게 틀리면" 높은 점수 (dangerous)
    - Entropy: 확률 분포 균등도 → 4클래스에 골고루 분산 = 낮은 confidence (safe)

    예: LiDAR가 Sky에서 Water로 확신있게 오예측
       → Energy: 높음 (logit 크니까) → 나쁨
       → Entropy: 낮음 (한 클래스에 집중) → confidence 높음... BUT aux head가
         부정확하면 분산된 예측 → 높은 entropy → 낮은 confidence → 안전

    Args:
        aux_logits_list: List[Tensor], 길이 = num_modalities
            각 원소: (B, num_classes, H_feat, W_feat) raw logits (반드시 .detach()된 것)
        temperature: softmax temperature (default 1.0)
        num_classes: 클래스 수 (entropy 정규화용, default 4)
    Returns:
        weights: (B, num_modalities, H_feat, W_feat) — 위치별 softmax 가중치
    """
    conf_maps = []
    max_entropy = math.log(num_classes)

    for z in aux_logits_list:
        # Temperature-scaled softmax → calibrated probability
        probs = F.softmax(z / temperature, dim=1)           # (B, C, H, W)
        log_probs = F.log_softmax(z / temperature, dim=1)   # (B, C, H, W)
        entropy = -(probs * log_probs).sum(dim=1)           # (B, H, W)
        # Normalize: 0 (완전 확신) ~ 1 (완전 균등), confidence = 1 - normalized_entropy
        confidence = 1.0 - entropy / max_entropy            # (B, H, W)
        conf_maps.append(confidence)

    stacked = torch.stack(conf_maps, dim=1)                 # (B, m, H, W)
    weights = F.softmax(stacked / temperature, dim=1)       # (B, m, H, W)
    return weights


class LoRA_Sam_P13(nn.Module):
    """
    LoRA_Sam_P13: Energy-Confidence Fusion + Expert Collapse Fix

    P9 기반, 두 가지 핵심 변경:
      1. CrossModalFusionHead → ConfidenceAuxHead + Energy Score 기반 fusion weight
         - 학습 가능 파라미터로 weight를 직접 예측하지 않음 (상수 수렴 방지)
         - aux head의 raw logit에서 energy score를 계산 → computed signal
         - 학습/추론 동일 메커니즘 (P10의 oracle 불일치 문제 해결)
      2. SoftMoE_LoRA_Layer experts_b: zero-init → kaiming*0.01
         - 대칭 시작 → rich-get-richer 방지
         - expert collapse (Block6-20에서 E1 사망) 해결

    P9에서 유지:
      - UAMM: max-norm (best modality = 1.0, 나머지 억제)
      - AMF: softmax weights로 output fusion
      - SoftMoE LoRA 구조 자체
      - Memory attention, mask decoder, prompt encoder

    Loss 구조:
      total = seg_loss + proto_loss + λ_aux * aux_loss
      aux_loss = mean(CE(aux_head(feat_i), gt) for i in modalities)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3, num_classes=4):
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

        # Inject SoftMoE-LoRA (P9과 동일, experts_b init은 아래에서 재초기화)
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)

            # [P13] Expert collapse fix: experts_b를 비영 초기화로 대칭 깨기
            for expert_b in moe_q.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01
            for expert_b in moe_v.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # Determine fusion dim (P9과 동일 로직)
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # [P13] ConfidenceAuxHead: 공유 1개 (CrossModalFusionHead 대체)
        self.aux_head = ConfidenceAuxHead(
            in_channels=fusion_dim,
            num_classes=num_classes,
        )

        # [P13] Energy temperature
        self.energy_temperature = 1.0

        # Visualization buffers
        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None
        self._last_aux_logits = None  # [P13] List[(B, C, H_feat, W_feat)] per modality

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # MoE gate visualization collector
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        try:
            # ============================================
            # Phase 1: 모든 모달리티 Image Encoding (P9 동일)
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Aux Prediction + Energy Confidence (P13 NEW)
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]

            # 공유 aux head로 각 modality의 독립 segmentation logit 계산
            aux_logits_list = [self.aux_head(feat) for feat in all_backbone_feats]

            # Energy score → fusion weights (학습 가능 파라미터 없음)
            cross_weights = compute_energy_confidence(
                aux_logits_list,
                temperature=self.energy_temperature,
            )  # (B, m)

            # ============================================
            # Phase 3: UAMM max-norm + Tracking (P9 동일)
            # ============================================
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m)

            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                current_score = uamm_scores[:, frame_idx].unsqueeze(1)        # (B, 1)
                score_expanded = current_score.transpose(0, 1).unsqueeze(-1)   # (1, B, 1)
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
            # Phase 4: AMF — Output Fusion (P9 동일)
            # ============================================
            amf_weights = cross_weights  # (B, m)

            w0 = amf_weights[:, 0].view(-1, 1, 1, 1)
            m_output = output[0] * w0
            m_feat = all_backbone_feats[0] * w0

            for i in range(1, m):
                wi = amf_weights[:, i].view(-1, 1, 1, 1)
                m_output = m_output + output[i] * wi
                m_feat = m_feat + all_backbone_feats[i] * wi

            # Visualization buffers
            self._last_uamm_scores = uamm_scores.detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().float().cpu().numpy()
            # [P13] aux logits per modality: List[(B, C, H_feat, W_feat)]
            self._last_aux_logits = [z.detach().cpu() for z in aux_logits_list]
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None

        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        if self.training:
            return m_output, m_feat, aux_logits_list
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        aux_head_tensors = {
            f"aux_head.{k}": v
            for k, v in self.aux_head.state_dict().items()
        }

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        model_ref = self.sam.module if isinstance(
            self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ) else self.sam
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
            **aux_head_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        # Load SoftMoE Layers
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict:
                mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict:
                mv.load_state_dict(state_dict[v_key])

        # Load ConfidenceAuxHead
        aux_head_dict = {
            k.replace("aux_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("aux_head.")
        }
        if aux_head_dict:
            self.aux_head.load_state_dict(aux_head_dict)

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


class LoRA_Sam_P14(nn.Module):
    """
    LoRA_Sam_P14: Per-Modality Separate Aux Decoders

    P13 기반, 핵심 변경 1가지:
      - ConfidenceAuxHead (공유 1개) → ModalAuxDecoder × num_modalities (모달리티별 독립)
        · 첫 conv를 3×3으로 변경 → RGB 텍스처 경계/LiDAR 점군/Thermal gradient 패턴 특화
        · 각 모달리티가 고유 파라미터를 가져 inter-modality gradient interference 제거

    P13에서 유지:
      - Energy Score 기반 fusion weight (학습 가능 파라미터 없음)
      - UAMM: max-norm
      - AMF: softmax weights로 output fusion
      - SoftMoE LoRA + expert collapse fix (kaiming*0.01)
      - 동일 리턴 형식: (output, m_feat, aux_logits_list) in train / (output, m_feat) in eval

    Loss 구조:
      total = seg_loss + proto_loss + λ_aux * aux_loss
      aux_loss = mean(CE(aux_heads[i](feat_i), gt) for i in modalities)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3, num_classes=4):
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

        # Inject SoftMoE-LoRA (P13과 동일, experts_b init 포함)
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)

            # [P13/P14] Expert collapse fix: experts_b를 비영 초기화로 대칭 깨기
            for expert_b in moe_q.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01
            for expert_b in moe_v.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # Determine fusion dim (P13과 동일 로직)
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # [P14] 모달리티별 독립 ModalAuxDecoder (P13의 공유 ConfidenceAuxHead 대체)
        self.aux_heads = nn.ModuleList([
            ModalAuxDecoder(in_channels=fusion_dim, num_classes=num_classes)
            for _ in range(num_experts)
        ])

        # [P13/P14] Energy temperature
        self.energy_temperature = 1.0

        # Visualization buffers
        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None
        self._last_aux_logits = None  # List[(B, C, H_feat, W_feat)] per modality

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # MoE gate visualization collector
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        try:
            # ============================================
            # Phase 1: 모든 모달리티 Image Encoding (P13 동일)
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Aux Prediction + Energy Confidence (P14: 모달리티별 독립 decoder)
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]

            # [P14] 각 modality에 독립 decoder 적용 (P13: 공유 aux_head)
            aux_logits_list = [self.aux_heads[i](feat) for i, feat in enumerate(all_backbone_feats)]

            # Energy score → fusion weights (학습 가능 파라미터 없음, P13과 동일)
            cross_weights = compute_energy_confidence(
                aux_logits_list,
                temperature=self.energy_temperature,
            )  # (B, m)

            # ============================================
            # Phase 3: UAMM max-norm + Tracking (P13 동일)
            # ============================================
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m)

            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                current_score = uamm_scores[:, frame_idx].unsqueeze(1)        # (B, 1)
                score_expanded = current_score.transpose(0, 1).unsqueeze(-1)   # (1, B, 1)
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
            # Phase 4: AMF — Output Fusion (P13 동일)
            # ============================================
            amf_weights = cross_weights  # (B, m)

            w0 = amf_weights[:, 0].view(-1, 1, 1, 1)
            m_output = output[0] * w0
            m_feat = all_backbone_feats[0] * w0

            for i in range(1, m):
                wi = amf_weights[:, i].view(-1, 1, 1, 1)
                m_output = m_output + output[i] * wi
                m_feat = m_feat + all_backbone_feats[i] * wi

            # Visualization buffers
            self._last_uamm_scores = uamm_scores.detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().float().cpu().numpy()
            self._last_aux_logits = [z.detach().cpu() for z in aux_logits_list]
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None

        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        if self.training:
            return m_output, m_feat, aux_logits_list
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        # [P14] ModuleList → aux_heads.{i}.* 키로 자동 직렬화
        aux_heads_tensors = {
            f"aux_heads.{k}": v
            for k, v in self.aux_heads.state_dict().items()
        }

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        model_ref = self.sam.module if isinstance(
            self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ) else self.sam
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
            **aux_heads_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        # Load SoftMoE Layers
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict:
                mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict:
                mv.load_state_dict(state_dict[v_key])

        # [P14] Load per-modality ModalAuxDecoder (aux_heads.{i}.* 키)
        aux_heads_dict = {
            k.replace("aux_heads.", ""): v
            for k, v in state_dict.items()
            if k.startswith("aux_heads.")
        }
        if aux_heads_dict:
            self.aux_heads.load_state_dict(aux_heads_dict)

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


class LoRA_Sam_P15(nn.Module):
    """
    LoRA_Sam_P15: Spatial-wise Energy Weighting

    P14 기반, 핵심 변경 1가지:
      - image-level 스칼라 Energy Score (B, m) → spatial map (B, m, H_feat, W_feat)
        · compute_energy_confidence()의 .mean(dim=[1,2]) 제거
        · UAMM: 위치별 max-norm → feat_sizes에 맞게 F.interpolate 후 vision_feats 변조
        · AMF: 위치별 가중치 합산 → output 해상도에 맞게 F.interpolate 후 fusion

    기대 효과:
      - Sky 영역: LiDAR 가중치 자동 억제 (상공 포인트 없음 → energy 높음)
      - Water 영역: RGB 가중치 억제 (야간 수면 암전 → energy 높음) → LiDAR/Thermal 활용
      - Dynamic 영역: 위치별 최적 모달리티 선택 → Dynamic IoU 개선

    P14에서 유지:
      - ModalAuxDecoder × num_modalities (per-modality 독립 decoder)
      - SoftMoE LoRA + expert collapse fix (kaiming*0.01)
      - 동일 리턴 형식: (output, m_feat, aux_logits_list) train / (output, m_feat) eval

    Loss 구조:
      total = seg_loss + proto_loss + λ_aux * aux_loss
      aux_loss = mean(CE(aux_heads[i](feat_i), gt) for i in modalities)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3, num_classes=4):
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

        # Inject SoftMoE-LoRA (P13/P14와 동일, experts_b init 포함)
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)

            # [P13~P15] Expert collapse fix
            for expert_b in moe_q.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01
            for expert_b in moe_v.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # Determine fusion dim (P13/P14와 동일 로직)
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # [P14/P15] 모달리티별 독립 ModalAuxDecoder
        self.aux_heads = nn.ModuleList([
            ModalAuxDecoder(in_channels=fusion_dim, num_classes=num_classes)
            for _ in range(num_experts)
        ])

        # [P13~P15] Energy temperature
        self.energy_temperature = 1.0

        # Visualization buffers (spatial mean으로 압축해 backward compat 유지)
        self._last_uamm_scores = None   # (B, m) — spatial mean
        self._last_amf_weights = None   # (B, m) — spatial mean
        self._last_moe_gates = None
        self._last_aux_logits = None    # List[(B, C, H_feat, W_feat)] per modality

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # MoE gate visualization collector
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        try:
            # ============================================
            # Phase 1: 모든 모달리티 Image Encoding (P13/P14 동일)
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Aux Prediction + Spatial Energy Confidence (P15 NEW)
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]

            # 각 modality에 독립 decoder 적용 (P14와 동일)
            aux_logits_list = [self.aux_heads[i](feat) for i, feat in enumerate(all_backbone_feats)]

            # [P15] Spatial energy confidence: (B, m, H_feat, W_feat)
            # compute_energy_confidence의 .mean(dim=[1,2]) 제거 버전
            cross_weights = compute_spatial_energy_confidence(
                aux_logits_list,
                temperature=self.energy_temperature,
            )  # (B, m, H_feat, W_feat)

            # ============================================
            # Phase 3: Spatial UAMM + Tracking (P15 변경)
            # ============================================
            # max-norm: 각 위치에서 가장 confident한 modality = 1.0
            max_w = cross_weights.max(dim=1, keepdim=True)[0]  # (B, 1, H_feat, W_feat)
            uamm_scores = cross_weights / (max_w + 1e-8)        # (B, m, H_feat, W_feat)

            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                # [P15] 위치별 score map → 각 level의 vision_feat 해상도에 맞게 resize
                spatial_score = uamm_scores[:, frame_idx]  # (B, H_feat, W_feat)

                modulated_vision_feats = []
                for level, feat in enumerate(vision_feats[frame_idx]):
                    # feat: (num_tokens, B, C)  — SAM2 Hiera flattened format
                    h, w = feat_sizes[frame_idx][level]
                    score_resized = F.interpolate(
                        spatial_score.unsqueeze(1),   # (B, 1, H_feat, W_feat)
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    )  # (B, 1, h, w)
                    score_flat = score_resized.flatten(2).permute(2, 0, 1)  # (h*w, B, 1)
                    modulated_vision_feats.append(feat * score_flat)

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
            # Phase 4: Spatial AMF — Output Fusion (P15 변경)
            # ============================================
            amf_weights = cross_weights  # (B, m, H_feat, W_feat)

            # output[i]: (B, num_masks, H_out, W_out) — 고해상도 mask logits
            # amf_weights: (B, m, H_feat, W_feat) → H_out×W_out으로 resize
            def _resize_weight(w_map, target_hw):
                return F.interpolate(
                    w_map,          # (B, 1, H_feat, W_feat)
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False,
                )  # (B, 1, H_out, W_out)

            w0_out = _resize_weight(amf_weights[:, 0:1], output[0].shape[2:])
            m_output = output[0] * w0_out
            # m_feat: backbone feature fusion (H_feat 해상도, resize 불필요)
            m_feat = all_backbone_feats[0] * amf_weights[:, 0:1]

            for i in range(1, m):
                wi_out = _resize_weight(amf_weights[:, i:i+1], output[i].shape[2:])
                m_output = m_output + output[i] * wi_out
                m_feat = m_feat + all_backbone_feats[i] * amf_weights[:, i:i+1]

            # Visualization buffers (spatial mean으로 압축)
            self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().float().cpu().numpy()  # (B, m)
            self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().float().cpu().numpy()  # (B, m)
            self._last_aux_logits = [z.detach().cpu() for z in aux_logits_list]
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None

        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        if self.training:
            return m_output, m_feat, aux_logits_list
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        aux_heads_tensors = {
            f"aux_heads.{k}": v
            for k, v in self.aux_heads.state_dict().items()
        }

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        model_ref = self.sam.module if isinstance(
            self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ) else self.sam
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
            **aux_heads_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict:
                mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict:
                mv.load_state_dict(state_dict[v_key])

        aux_heads_dict = {
            k.replace("aux_heads.", ""): v
            for k, v in state_dict.items()
            if k.startswith("aux_heads.")
        }
        if aux_heads_dict:
            self.aux_heads.load_state_dict(aux_heads_dict)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)

        self.sam.load_state_dict(sam_dict)


class LoRA_Sam_P16(nn.Module):
    """
    LoRA_Sam_P16: Calibrated Spatial Entropy Fusion

    P15 기반, P12~P14 실패 분석에서 도출된 4가지 수정사항 통합:

      Fix 1: Gradient 격리 (.detach())
        - aux logits → fusion weight 계산 시 .detach()로 gradient 차단
        - aux head는 자기 CE loss만으로 학습 → 정직한 confidence 출력
        - main loss gradient가 energy→aux→LoRA로 역전파되는 경로 차단

      Fix 2: Energy Score → Calibrated Entropy
        - Energy: logit magnitude 기반 → "자신있게 틀리면" 높은 점수 (dangerous)
        - Entropy: 확률 분포 균등도 → confidence = 1 - normalized_entropy
        - LiDAR Sky에서 Water로 확신있게 오예측 → entropy 낮음 but aux head 부정확
          → 실제로는 분산된 예측 → 높은 entropy → 낮은 confidence → 안전

      Fix 3: Spatial-wise (B, m, H, W) — P15에서 유지
        - 위치별 다른 모달리티 가중치 (Sky: LiDAR 억제, Water: RGB 억제)

      Fix 4: Aux Warmup Schedule
        - 초기 N epoch: uniform weights (1/m) → aux head 학습 시간 확보
        - N~N+5 epoch: linear ramp → 점진적 entropy 반영
        - N+5 이후: full entropy weights

    P15에서 유지:
      - ModalAuxDecoder × num_modalities
      - SoftMoE LoRA + expert collapse fix (kaiming*0.01)
      - Spatial UAMM/AMF interpolation
      - 동일 리턴: (output, m_feat, aux_logits_list) train / (output, m_feat) eval
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3, num_classes=4,
                 aux_warmup_epochs=10):
        nn.Module.__init__(self)

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.num_classes = num_classes
        self.aux_warmup_epochs = aux_warmup_epochs
        self._current_epoch = 0  # 학습 스크립트에서 매 epoch 설정

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        # Freeze original parameters
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Inject SoftMoE-LoRA (P13~P15와 동일, experts_b init 포함)
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)

            # Expert collapse fix
            for expert_b in moe_q.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01
            for expert_b in moe_v.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # Determine fusion dim
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # 모달리티별 독립 ModalAuxDecoder (P14/P15 유지)
        self.aux_heads = nn.ModuleList([
            ModalAuxDecoder(in_channels=fusion_dim, num_classes=num_classes)
            for _ in range(num_experts)
        ])

        # Entropy temperature (val에서 grid search 가능)
        self.energy_temperature = 1.0

        # Visualization buffers
        self._last_uamm_scores = None   # (B, m) — spatial mean
        self._last_amf_weights = None   # (B, m) — spatial mean
        self._last_moe_gates = None
        self._last_aux_logits = None

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # MoE gate visualization collector
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        try:
            # ============================================
            # Phase 1: 모든 모달리티 Image Encoding
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Aux Prediction + Calibrated Entropy Confidence
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]

            # 각 modality에 독립 decoder 적용
            aux_logits_list = [self.aux_heads[i](feat) for i, feat in enumerate(all_backbone_feats)]

            # [Fix 4] Aux Warmup: 초기 N epoch uniform → linear ramp → full entropy
            warmup_ramp = 5  # ramp 구간 (epoch)
            if self._current_epoch < self.aux_warmup_epochs:
                ramp = 0.0
            elif self._current_epoch < self.aux_warmup_epochs + warmup_ramp:
                ramp = (self._current_epoch - self.aux_warmup_epochs) / warmup_ramp
            else:
                ramp = 1.0

            if ramp < 1e-6:
                # Pure uniform: aux head 아직 학습 중
                B = all_backbone_feats[0].shape[0]
                H_feat, W_feat = all_backbone_feats[0].shape[2:]
                device = all_backbone_feats[0].device
                cross_weights = torch.ones(B, m, H_feat, W_feat, device=device) / m
            else:
                # [Fix 1] .detach(): aux logits gradient 격리
                # [Fix 2] Calibrated Entropy (Energy Score 대체)
                entropy_weights = compute_spatial_entropy_confidence(
                    [z.detach() for z in aux_logits_list],
                    temperature=self.energy_temperature,
                    num_classes=self.num_classes,
                )  # (B, m, H_feat, W_feat)

                if ramp < 1.0:
                    # Linear ramp: uniform → entropy 전환
                    B = all_backbone_feats[0].shape[0]
                    H_feat, W_feat = all_backbone_feats[0].shape[2:]
                    device = all_backbone_feats[0].device
                    uniform = torch.ones(B, m, H_feat, W_feat, device=device) / m
                    cross_weights = (1.0 - ramp) * uniform + ramp * entropy_weights
                else:
                    cross_weights = entropy_weights

            # ============================================
            # Phase 3: Spatial UAMM + Tracking
            # ============================================
            max_w = cross_weights.max(dim=1, keepdim=True)[0]  # (B, 1, H_feat, W_feat)
            uamm_scores = cross_weights / (max_w + 1e-8)        # (B, m, H_feat, W_feat)

            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                spatial_score = uamm_scores[:, frame_idx]  # (B, H_feat, W_feat)

                modulated_vision_feats = []
                for level, feat in enumerate(vision_feats[frame_idx]):
                    h, w = feat_sizes[frame_idx][level]
                    score_resized = F.interpolate(
                        spatial_score.unsqueeze(1),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    )  # (B, 1, h, w)
                    score_flat = score_resized.flatten(2).permute(2, 0, 1)  # (h*w, B, 1)
                    modulated_vision_feats.append(feat * score_flat)

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
            # Phase 4: Spatial AMF — Output Fusion
            # ============================================
            amf_weights = cross_weights  # (B, m, H_feat, W_feat)

            def _resize_weight(w_map, target_hw):
                return F.interpolate(
                    w_map,
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False,
                )

            w0_out = _resize_weight(amf_weights[:, 0:1], output[0].shape[2:])
            m_output = output[0] * w0_out
            m_feat = all_backbone_feats[0] * amf_weights[:, 0:1]

            for i in range(1, m):
                wi_out = _resize_weight(amf_weights[:, i:i+1], output[i].shape[2:])
                m_output = m_output + output[i] * wi_out
                m_feat = m_feat + all_backbone_feats[i] * amf_weights[:, i:i+1]

            # Visualization buffers
            self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().float().cpu().numpy()
            self._last_aux_logits = [z.detach().cpu() for z in aux_logits_list]
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None

        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        if self.training:
            return m_output, m_feat, aux_logits_list
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        aux_heads_tensors = {
            f"aux_heads.{k}": v
            for k, v in self.aux_heads.state_dict().items()
        }

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        model_ref = self.sam.module if isinstance(
            self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ) else self.sam
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
            **aux_heads_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict:
                mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict:
                mv.load_state_dict(state_dict[v_key])

        aux_heads_dict = {
            k.replace("aux_heads.", ""): v
            for k, v in state_dict.items()
            if k.startswith("aux_heads.")
        }
        if aux_heads_dict:
            self.aux_heads.load_state_dict(aux_heads_dict)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)

        self.sam.load_state_dict(sam_dict)


class LoRA_Sam_P17(nn.Module):
    """
    LoRA_Sam_P17: Multi-Scale FPN Aux Decoder + Calibrated Spatial Entropy Fusion

    P16 기반, ISSUE-008(frozen backbone bottleneck)에 대한 직접적 해결:

      기존 (P13~P16): backbone_fpn[0](32ch, 256×256)만 aux decoder에 사용
        → 32채널 단일 스케일: 정보량 부족 → aux mask 부정확 → entropy 부정확 → fusion 실패

      P17 변경: 3개 FPN 레벨 전부 활용 (MultiScaleModalAuxDecoder)
        - fpn[0]: 32ch, 256×256  (high-res spatial detail)
        - fpn[1]: 64ch, 128×128  (mid-level features)
        - fpn[2]: 256ch, 64×64   (semantic context)
        → 352채널 멀티스케일: 11배 정보량 증가, 추가 backbone 연산 0

      P16의 4가지 Fix 유지:
        1. .detach() gradient 격리
        2. Calibrated Entropy (Energy Score 대체)
        3. Spatial-wise (B, m, H, W) 가중치
        4. Aux Warmup Schedule (10ep uniform + 5ep ramp)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3, num_classes=4,
                 aux_warmup_epochs=10):
        nn.Module.__init__(self)

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.aux_warmup_epochs = aux_warmup_epochs
        self._current_epoch = 0

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        # Freeze original parameters
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Inject SoftMoE-LoRA (P16과 동일, experts_b init 포함)
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)

            # Expert collapse fix
            for expert_b in moe_q.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01
            for expert_b in moe_v.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # [P17] Multi-scale FPN channels
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            self.fpn_channels = (
                self.sam.sam_mask_decoder.transformer_dim // 8,  # fpn[0]: 32
                self.sam.sam_mask_decoder.transformer_dim // 4,  # fpn[1]: 64
                self.sam.sam_mask_decoder.transformer_dim,       # fpn[2]: 256
            )
        else:
            td = self.sam.sam_mask_decoder.transformer_dim
            self.fpn_channels = (td, td, td)

        # [P17] 모달리티별 독립 MultiScaleModalAuxDecoder
        self.aux_heads = nn.ModuleList([
            MultiScaleModalAuxDecoder(
                fpn_channels=self.fpn_channels,
                proj_dim=32,
                num_classes=num_classes,
            )
            for _ in range(num_modalities)
        ])

        # Entropy temperature
        self.energy_temperature = 1.0

        # Visualization buffers
        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None
        self._last_aux_logits = None

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # MoE gate visualization collector
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        try:
            # ============================================
            # Phase 1: 모든 모달리티 Image Encoding
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Multi-Scale Aux Prediction + Calibrated Entropy
            # ============================================
            # [P17] 3개 FPN 레벨 전부 추출
            all_fpn_feats = [
                [image_embedding[i]['backbone_fpn'][j] for j in range(3)]
                for i in range(m)
            ]  # all_fpn_feats[modality_idx][fpn_level]

            # 각 modality에 독립 MultiScaleModalAuxDecoder 적용
            aux_logits_list = [
                self.aux_heads[i](all_fpn_feats[i]) for i in range(m)
            ]  # List[(B, num_classes, H0, W0)]

            # m_feat fusion용 backbone feature (fpn[0] 유지)
            all_backbone_feats = [all_fpn_feats[i][0] for i in range(m)]

            # [Fix 4] Aux Warmup: 초기 N epoch uniform → linear ramp → full entropy
            warmup_ramp = 5
            if self._current_epoch < self.aux_warmup_epochs:
                ramp = 0.0
            elif self._current_epoch < self.aux_warmup_epochs + warmup_ramp:
                ramp = (self._current_epoch - self.aux_warmup_epochs) / warmup_ramp
            else:
                ramp = 1.0

            if ramp < 1e-6:
                B = all_backbone_feats[0].shape[0]
                H_feat, W_feat = all_backbone_feats[0].shape[2:]
                device = all_backbone_feats[0].device
                cross_weights = torch.ones(B, m, H_feat, W_feat, device=device) / m
            else:
                # [Fix 1] .detach() + [Fix 2] Calibrated Entropy
                entropy_weights = compute_spatial_entropy_confidence(
                    [z.detach() for z in aux_logits_list],
                    temperature=self.energy_temperature,
                    num_classes=self.num_classes,
                )

                if ramp < 1.0:
                    B = all_backbone_feats[0].shape[0]
                    H_feat, W_feat = all_backbone_feats[0].shape[2:]
                    device = all_backbone_feats[0].device
                    uniform = torch.ones(B, m, H_feat, W_feat, device=device) / m
                    cross_weights = (1.0 - ramp) * uniform + ramp * entropy_weights
                else:
                    cross_weights = entropy_weights

            # ============================================
            # Phase 3: Spatial UAMM + Tracking
            # ============================================
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)

            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                spatial_score = uamm_scores[:, frame_idx]

                modulated_vision_feats = []
                for level, feat in enumerate(vision_feats[frame_idx]):
                    h, w = feat_sizes[frame_idx][level]
                    score_resized = F.interpolate(
                        spatial_score.unsqueeze(1),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    )
                    score_flat = score_resized.flatten(2).permute(2, 0, 1)
                    modulated_vision_feats.append(feat * score_flat)

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
            # Phase 4: Spatial AMF — Output Fusion
            # ============================================
            amf_weights = cross_weights

            def _resize_weight(w_map, target_hw):
                return F.interpolate(
                    w_map,
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False,
                )

            w0_out = _resize_weight(amf_weights[:, 0:1], output[0].shape[2:])
            m_output = output[0] * w0_out
            m_feat = all_backbone_feats[0] * amf_weights[:, 0:1]

            for i in range(1, m):
                wi_out = _resize_weight(amf_weights[:, i:i+1], output[i].shape[2:])
                m_output = m_output + output[i] * wi_out
                m_feat = m_feat + all_backbone_feats[i] * amf_weights[:, i:i+1]

            # Visualization buffers
            self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().float().cpu().numpy()
            self._last_aux_logits = [z.detach().cpu() for z in aux_logits_list]
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None

        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        if self.training:
            return m_output, m_feat, aux_logits_list
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        aux_heads_tensors = {
            f"aux_heads.{k}": v
            for k, v in self.aux_heads.state_dict().items()
        }

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        model_ref = self.sam.module if isinstance(
            self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ) else self.sam
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
            **aux_heads_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict:
                mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict:
                mv.load_state_dict(state_dict[v_key])

        aux_heads_dict = {
            k.replace("aux_heads.", ""): v
            for k, v in state_dict.items()
            if k.startswith("aux_heads.")
        }
        if aux_heads_dict:
            self.aux_heads.load_state_dict(aux_heads_dict)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)

        self.sam.load_state_dict(sam_dict)


class LoRA_Sam_P18(nn.Module):
    """
    LoRA_Sam_P18: Trainable ResNet-18 Aux Backbone + Configurable Fusion

    ISSUE-008(frozen backbone bottleneck) 근본 해결:
      P13~P17: frozen SAM2 FPN feature → aux decoder → aux mask 품질 한계
      P18: trainable ResNet-18 → aux decoder → 도메인 특화 feature 학습 가능

    use_entropy_fusion으로 두 가지 서브 옵션:
      P18-A (False): P9-style CrossModalFusionHead → 고정상수 UAMM/AMF
        ResNet aux는 CE loss로만 학습, fusion weight에 영향 없음. 안전한 baseline.
      P18-B (True): P17-style spatial entropy → adaptive UAMM/AMF
        정확한 ResNet aux mask → 정확한 entropy → dynamic fusion 비로소 작동.

    아키텍처:
      Input → SAM2 Hiera B+ (frozen) → backbone_fpn → memory attention → prediction
                                                                    ↑ (UAMM/AMF)
      Input → ResNet-18 (trainable) → layer2+layer3 → ResNetAuxDecoder → aux_logits
                                                                           ↓
                                                            aux CE loss (trains ResNet)
                                                            entropy confidence (P18-B only)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3, num_classes=4,
                 aux_warmup_epochs=10, use_entropy_fusion=False):
        nn.Module.__init__(self)

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.use_entropy_fusion = use_entropy_fusion
        self.aux_warmup_epochs = aux_warmup_epochs
        self._current_epoch = 0

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        # Freeze original parameters
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Inject SoftMoE-LoRA (P17 동일, experts_b init 포함)
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)

            # Expert collapse fix
            for expert_b in moe_q.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01
            for expert_b in moe_v.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # [P18] Trainable ResNet-18 Aux Backbone
        self.aux_backbone = ResNetAuxBackbone(
            num_modalities=num_modalities, pretrained=True
        )

        # [P18] Per-modality aux decoders on ResNet features
        self.aux_heads = nn.ModuleList([
            ResNetAuxDecoder(
                resnet_channels=(128, 256),
                proj_dim=32,
                num_classes=num_classes,
            )
            for _ in range(num_modalities)
        ])

        # [P18-A] P9-style CrossModalFusionHead (scalar fusion)
        if not use_entropy_fusion:
            use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
            if use_high_res:
                fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
            else:
                fusion_dim = self.sam.sam_mask_decoder.transformer_dim
            self.cross_modal_head = CrossModalFusionHead(
                in_channels=fusion_dim,
                num_modalities=num_modalities,
            )

        # [P18-B] Entropy temperature
        self.energy_temperature = 1.0

        # Visualization buffers
        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None
        self._last_aux_logits = None

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # MoE gate visualization collector
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        try:
            # ============================================
            # Phase 1: 모든 모달리티 SAM2 Image Encoding
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # m_feat fusion용 SAM2 fpn[0] (P9/P17 동일)
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]

            # ============================================
            # Phase 2: ResNet-18 Aux Prediction
            # ============================================
            aux_logits_list = []
            for i in range(m):
                resnet_feats = self.aux_backbone(batched_input[i], modal_idx=i)
                aux_logits = self.aux_heads[i](resnet_feats)
                aux_logits_list.append(aux_logits)
            # aux_logits_list: List[(B, num_classes, H/8, W/8)] = (B, 4, 128, 128)

            # ============================================
            # Phase 2b: Fusion Weights
            # ============================================
            if self.use_entropy_fusion:
                # [P18-B] Spatial entropy from ResNet aux (P17 로직)
                warmup_ramp = 5
                if self._current_epoch < self.aux_warmup_epochs:
                    ramp = 0.0
                elif self._current_epoch < self.aux_warmup_epochs + warmup_ramp:
                    ramp = (self._current_epoch - self.aux_warmup_epochs) / warmup_ramp
                else:
                    ramp = 1.0

                B = all_backbone_feats[0].shape[0]
                H_feat, W_feat = all_backbone_feats[0].shape[2:]
                device = all_backbone_feats[0].device

                if ramp < 1e-6:
                    cross_weights = torch.ones(B, m, H_feat, W_feat, device=device) / m
                else:
                    entropy_weights = compute_spatial_entropy_confidence(
                        [z.detach() for z in aux_logits_list],
                        temperature=self.energy_temperature,
                        num_classes=self.num_classes,
                    )
                    # ResNet aux output (128x128) -> SAM2 fpn[0] resolution (256x256)
                    if entropy_weights.shape[2:] != (H_feat, W_feat):
                        entropy_weights = F.interpolate(
                            entropy_weights, size=(H_feat, W_feat),
                            mode='bilinear', align_corners=False
                        )

                    if ramp < 1.0:
                        uniform = torch.ones(B, m, H_feat, W_feat, device=device) / m
                        cross_weights = (1.0 - ramp) * uniform + ramp * entropy_weights
                    else:
                        cross_weights = entropy_weights

                # Spatial UAMM
                max_w = cross_weights.max(dim=1, keepdim=True)[0]
                uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m, H, W)
            else:
                # [P18-A] P9-style CrossModalFusionHead -> scalar (B, m)
                cross_weights, cross_logits = self.cross_modal_head(all_backbone_feats)
                max_w = cross_weights.max(dim=1, keepdim=True)[0]
                uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m)

            # ============================================
            # Phase 3: UAMM Modulation + Tracking
            # ============================================
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                if self.use_entropy_fusion:
                    # Spatial UAMM (P18-B, P17 패턴)
                    spatial_score = uamm_scores[:, frame_idx]  # (B, H, W)
                    modulated_vision_feats = []
                    for level, feat in enumerate(vision_feats[frame_idx]):
                        h, w = feat_sizes[frame_idx][level]
                        score_resized = F.interpolate(
                            spatial_score.unsqueeze(1),
                            size=(h, w), mode='bilinear', align_corners=False,
                        )
                        score_flat = score_resized.flatten(2).permute(2, 0, 1)
                        modulated_vision_feats.append(feat * score_flat)
                else:
                    # Scalar UAMM (P18-A, P9 패턴)
                    current_score = uamm_scores[:, frame_idx].unsqueeze(1)  # (B, 1)
                    score_expanded = current_score.transpose(0, 1).unsqueeze(-1)  # (1, B, 1)
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
            # Phase 4: AMF -- Output Fusion
            # ============================================
            amf_weights = cross_weights

            if self.use_entropy_fusion:
                # Spatial AMF (P18-B, P17 패턴)
                def _resize_weight(w_map, target_hw):
                    return F.interpolate(w_map, size=target_hw, mode='bilinear', align_corners=False)

                w0_out = _resize_weight(amf_weights[:, 0:1], output[0].shape[2:])
                m_output = output[0] * w0_out
                m_feat = all_backbone_feats[0] * amf_weights[:, 0:1]

                for i in range(1, m):
                    wi_out = _resize_weight(amf_weights[:, i:i+1], output[i].shape[2:])
                    m_output = m_output + output[i] * wi_out
                    m_feat = m_feat + all_backbone_feats[i] * amf_weights[:, i:i+1]
            else:
                # Scalar AMF (P18-A, P9 패턴)
                w0 = amf_weights[:, 0].view(-1, 1, 1, 1)
                m_output = output[0] * w0
                m_feat = all_backbone_feats[0] * w0

                for i in range(1, m):
                    wi = amf_weights[:, i].view(-1, 1, 1, 1)
                    m_output = m_output + output[i] * wi
                    m_feat = m_feat + all_backbone_feats[i] * wi

            # Visualization buffers
            if self.use_entropy_fusion:
                self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().float().cpu().numpy()
                self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().float().cpu().numpy()
            else:
                self._last_uamm_scores = uamm_scores.detach().float().cpu().numpy()
                self._last_amf_weights = amf_weights.detach().float().cpu().numpy()
            self._last_aux_logits = [z.detach().cpu() for z in aux_logits_list]
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None

        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        if self.training:
            return m_output, m_feat, aux_logits_list
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        aux_backbone_tensors = {
            f"aux_backbone.{k}": v
            for k, v in self.aux_backbone.state_dict().items()
        }
        aux_heads_tensors = {
            f"aux_heads.{k}": v
            for k, v in self.aux_heads.state_dict().items()
        }
        cross_modal_tensors = {}
        if not self.use_entropy_fusion:
            cross_modal_tensors = {
                f"cross_modal_head.{k}": v
                for k, v in self.cross_modal_head.state_dict().items()
            }

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        model_ref = self.sam.module if isinstance(
            self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ) else self.sam
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
            **aux_backbone_tensors,
            **aux_heads_tensors,
            **cross_modal_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict:
                mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict:
                mv.load_state_dict(state_dict[v_key])

        aux_backbone_dict = {
            k.replace("aux_backbone.", ""): v
            for k, v in state_dict.items()
            if k.startswith("aux_backbone.")
        }
        if aux_backbone_dict:
            self.aux_backbone.load_state_dict(aux_backbone_dict)

        aux_heads_dict = {
            k.replace("aux_heads.", ""): v
            for k, v in state_dict.items()
            if k.startswith("aux_heads.")
        }
        if aux_heads_dict:
            self.aux_heads.load_state_dict(aux_heads_dict)

        if not self.use_entropy_fusion:
            cross_dict = {
                k.replace("cross_modal_head.", ""): v
                for k, v in state_dict.items()
                if k.startswith("cross_modal_head.")
            }
            if cross_dict:
                self.cross_modal_head.load_state_dict(cross_dict)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()
        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)
        self.sam.load_state_dict(sam_dict)


class LoRA_Sam_P19(nn.Module):
    """
    LoRA_Sam_P19: Learned Spatial Cross-Modal Fusion

    P9 base + SpatialCrossModalFusionHead:
      - P9: GAP → scalar (B,m) CrossModalFusionHead (fpn[0] only)
      - P19: multi-scale FPN → spatial (B,m,H,W) SpatialCrossModalFusionHead

    핵심 차이:
      1. 3개 FPN 레벨 전부 사용 (P9: fpn[0] only)
      2. 공간 가중치 (B,m,H,W) → per-location 모달리티 선택 (P9: 전체 이미지 동일)
      3. Spatial UAMM/AMF (P17 패턴) — vision_feats level별 resize 적용
      4. Aux decoder 없음 → 깔끔한 구조, main loss만으로 학습

    Components:
      1. Structure: Soft-MoE LoRA (P9과 동일)
      2. UAMM (Memory): Spatial max-normalized softmax → per-location Feature Modulation
      3. AMF (Fusion): Spatial raw softmax → per-location Output Fusion
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3):
        nn.Module.__init__(self)

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.num_modalities = num_modalities
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

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts)

            # Expert collapse fix (from P17)
            for expert_b in moe_q.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01
            for expert_b in moe_v.experts_b:
                nn.init.kaiming_uniform_(expert_b.weight, a=math.sqrt(5))
                expert_b.weight.data *= 0.01

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # FPN channels
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            self.fpn_channels = (
                self.sam.sam_mask_decoder.transformer_dim // 8,  # 32
                self.sam.sam_mask_decoder.transformer_dim // 4,  # 64
                self.sam.sam_mask_decoder.transformer_dim,       # 256
            )
        else:
            td = self.sam.sam_mask_decoder.transformer_dim
            self.fpn_channels = (td, td, td)

        # [P19] SpatialCrossModalFusionHead
        self.spatial_fusion_head = SpatialCrossModalFusionHead(
            fpn_channels=self.fpn_channels,
            num_modalities=num_modalities,
            proj_dim=32,
            hidden_dim=64,
        )

        # Visualization buffers
        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # MoE gate visualization collector
        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        try:
            # ============================================
            # Phase 1: 모든 모달리티 Image Encoding
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Multi-Scale FPN → Spatial Fusion Weights
            # ============================================
            # [P19] 3개 FPN 레벨 전부 추출
            all_fpn_feats = [
                [image_embedding[i]['backbone_fpn'][j] for j in range(3)]
                for i in range(m)
            ]  # all_fpn_feats[modality_idx][fpn_level]

            # m_feat fusion용 backbone feature (fpn[0])
            all_backbone_feats = [all_fpn_feats[i][0] for i in range(m)]

            # SpatialCrossModalFusionHead → (B, m, H, W)
            cross_weights, cross_logits = self.spatial_fusion_head(all_fpn_feats)

            # UAMM용: spatial max-normalize → best modality location = 1.0
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m, H, W)

            # ============================================
            # Phase 3: Spatial UAMM + Tracking (P17 패턴)
            # ============================================
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                spatial_score = uamm_scores[:, frame_idx]  # (B, H, W)

                modulated_vision_feats = []
                for level, feat in enumerate(vision_feats[frame_idx]):
                    h, w = feat_sizes[frame_idx][level]
                    score_resized = F.interpolate(
                        spatial_score.unsqueeze(1),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False,
                    )
                    score_flat = score_resized.flatten(2).permute(2, 0, 1)  # (hw, B, 1)
                    modulated_vision_feats.append(feat * score_flat)

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
            # Phase 4: Spatial AMF — Output Fusion (P17 패턴)
            # ============================================
            amf_weights = cross_weights  # (B, m, H, W)

            def _resize_weight(w_map, target_hw):
                return F.interpolate(
                    w_map,
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False,
                )

            w0_out = _resize_weight(amf_weights[:, 0:1], output[0].shape[2:])
            m_output = output[0] * w0_out
            m_feat = all_backbone_feats[0] * amf_weights[:, 0:1]

            for i in range(1, m):
                wi_out = _resize_weight(amf_weights[:, i:i+1], output[i].shape[2:])
                m_output = m_output + output[i] * wi_out
                m_feat = m_feat + all_backbone_feats[i] * amf_weights[:, i:i+1]

            # Visualization buffers (spatial mean for backward compat with P9 visualizers)
            self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().float().cpu().numpy()
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

        # SpatialCrossModalFusionHead
        spatial_head_tensors = {
            f"spatial_fusion_head.{k}": v
            for k, v in self.spatial_fusion_head.state_dict().items()
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
            **spatial_head_tensors,
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

        # Load SpatialCrossModalFusionHead
        spatial_dict = {
            k.replace("spatial_fusion_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("spatial_fusion_head.")
        }
        if spatial_dict:
            self.spatial_fusion_head.load_state_dict(spatial_dict)

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


class LoRA_Sam_P20(nn.Module):
    """
    LoRA_Sam_P20: Per-Layer Independent MLP Gate + Higher Rank MoE

    P9 대비 개선:
      - Gate: Linear(C→E) → 2-layer MLP(C→C//4→E), per-layer 독립
      - Rank 상향: 4 → 8 (J-A 실험), 16 (J-B 실험)
      - SoftMoE_LoRA_Layer_V2: 내부 독립 MLP gate 보유

    (이전 버전은 gate를 dim별로 공유했으나, Stage 2-3에서 38개 레이어가
     1개 gate를 공유하면 uniform compromise에 빠지는 문제가 확인되어
     per-layer 독립 MLP gate로 변경.)

    나머지 (UAMM, AMF, CrossModalFusionHead)는 P9과 동일.
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3, gate_hidden_ratio=4):
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

        # Inject SoftMoE-LoRA V2 with per-layer independent MLP gates
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer_V2(dim, r, num_experts=num_experts,
                                           gate_hidden_ratio=gate_hidden_ratio)
            moe_v = SoftMoE_LoRA_Layer_V2(dim, r, num_experts=num_experts,
                                           gate_hidden_ratio=gate_hidden_ratio)

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # Determine Fusion Dim (P9과 동일)
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # CrossModalFusionHead (P9과 동일)
        self.cross_modal_head = CrossModalFusionHead(
            in_channels=fusion_dim,
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
            # Phase 1: 모든 모달리티 Image Encoding
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Cross-Modal 가중치 산출 (P9과 동일)
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
            cross_weights, cross_logits = self.cross_modal_head(all_backbone_feats)

            # UAMM용: max-normalize
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)

            # ============================================
            # Phase 3: UAMM Modulation + Tracking (P9과 동일)
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
            # Phase 4: AMF — raw softmax weights로 Output Fusion (P9과 동일)
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
        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        # SoftMoE Expert + Gate Parameters (gate는 V2 내부에 포함)
        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        cross_modal_tensors = {
            f"cross_modal_head.{k}": v
            for k, v in self.cross_modal_head.state_dict().items()
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
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        # Load SoftMoE Expert + Gate Layers (gate는 V2 내부에 포함)
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


class LoRA_Sam_P21(nn.Module):
    """
    LoRA_Sam_P21: P9 + DeBA-FP (Deformable Bottleneck Adapter for Feature Pyramid)

    P9 base architecture에 DeBA-FP를 추가하여 FPN 피처를 spatial-aware하게 refine.
    Day→Night domain gap에서 구조적 정보(경계, 형태)가 domain-invariant하다는
    가정 하에, deformable convolution으로 structural information을 포착.

    Ref: "Rethinking Deformable Convolution as an Adapter with Cross-layer
          Weight Sharing for Robust Semantic Segmentation in the Wild" (CVPR 2026)

    P9 대비 변경:
      1. DeBA-FP 모듈: fpn[0] → DeBA-FP → refined fpn[0] → CrossModalFusionHead
      2. Cross-modal weight sharing: DCM, norm, W_d, W_u 공유, α만 per-modality
      3. DeBA-BB는 미적용 (SAM2 Hiera와 DINOv2 구조 차이로 향후 과제)

    Components:
      1. Structure: Soft-MoE LoRA (P9 동일)
      2. DeBA-FP: Deformable bottleneck adapter on FPN features (신규)
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

        # Determine Fusion Dim
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # CrossModalFusionHead (P9 동일)
        self.cross_modal_head = CrossModalFusionHead(
            in_channels=fusion_dim,
            num_modalities=num_modalities,
        )

        # [P21 신규] DeBA-FP: Deformable Bottleneck Adapter
        self.deba_fp = DeBAFP(
            in_channels=fusion_dim,
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
            # Phase 1: 모든 모달리티 Image Encoding
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: DeBA-FP → Cross-Modal 가중치 산출
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]

            # [P21] DeBA-FP: 각 모달리티 FPN feature를 deformable conv로 refine
            all_backbone_feats = [
                self.deba_fp(feat, modality_idx=i)
                for i, feat in enumerate(all_backbone_feats)
            ]

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

        # [P21] DeBA-FP parameters
        deba_fp_tensors = {
            f"deba_fp.{k}": v
            for k, v in self.deba_fp.state_dict().items()
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
            **deba_fp_tensors,
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

        # [P21] Load DeBA-FP
        deba_fp_dict = {}
        for k, v in state_dict.items():
            if k.startswith("deba_fp."):
                deba_fp_dict[k.replace("deba_fp.", "")] = v
        if deba_fp_dict:
            self.deba_fp.load_state_dict(deba_fp_dict)

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


class LoRA_Sam_P23(nn.Module):
    """
    LoRA_Sam_P23: MoE DeBA-BB (Deformable Bottleneck Adapter for Backbone)

    P9의 SoftMoE-LoRA를 MoE-DeBA-BB로 교체:
      - SoftMoE LoRA: Linear_a(C→r) → Linear_b(r→C) per expert
      - DeBA-BB: W_d(C→d_b) → DCM_i(3×3, multi-scale) → LN → GELU → W_u(d_b→C) per expert

    핵심 특징:
      1. GAP gating (per-image routing) — ConvLoRA 스타일
      2. Multi-scale expert differentiation (×1, ×2 해상도)
      3. Cross-layer weight sharing (DeBA paper):
         - Shared across layers + experts: LayerNorm
         - Shared across layers, per-expert: DCM
         - Per-stage: W_d, W_u, gate (Hiera stage별 dim 차이)
         - Per-modality: α scaling
      4. 나머지 (UAMM, AMF, CrossModalFusionHead)는 P9 동일

    Refs:
      - DeBA (CVPR 2026): cross-layer shared deformable bottleneck adapter
      - ConvLoRA (ICLR 2024): multi-scale MoE conv adapter with GAP gating

    Components:
      1. Structure: MoE DeBA-BB (replaces SoftMoE-LoRA)
      2. UAMM (Memory): CrossModal max-normalized softmax (P9 동일)
      3. AMF (Fusion): CrossModal raw softmax (P9 동일)
    """

    # Block → stage mapping for Hiera-B+
    # stages=(2,3,16,3), stage_ends=[1,4,20,23]
    # Block  0-2:  qkv.in_features=112 (stage 0)
    # Block  3-5:  qkv.in_features=224 (stage 1)
    # Block  6-21: qkv.in_features=448 (stage 2)
    # Block  22-23: qkv.in_features=896 (stage 3)
    _BLOCK_TO_STAGE = {}
    for _b in range(0, 3):   _BLOCK_TO_STAGE[_b] = 0   # dim 112
    for _b in range(3, 6):   _BLOCK_TO_STAGE[_b] = 1   # dim 224
    for _b in range(6, 22):  _BLOCK_TO_STAGE[_b] = 2   # dim 448
    for _b in range(22, 24): _BLOCK_TO_STAGE[_b] = 3   # dim 896

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=2, num_modalities=3,
                 deba_bottleneck_dim=64, deba_kernel_size=3,
                 deba_scales=None, deba_gate_noise_std=0.1):
        """
        Args:
            sam_model: pretrained SAM2 model
            r: unused (kept for dispatch compatibility, use deba_bottleneck_dim instead)
            lora_layer: which backbone blocks to apply adapter (default: all)
            num_experts: MoE expert count (default 2: ×1 and ×2 scale)
            num_modalities: number of input modalities
            deba_bottleneck_dim: DCM bottleneck dimension
            deba_kernel_size: DCM kernel size
            deba_scales: multi-scale factors per expert (default [1, 2])
            deba_gate_noise_std: gate noise during training
        """
        nn.Module.__init__(self)

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        # Freeze original parameters
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Hiera-B+ stage dims
        stage_dims = [112, 224, 448, 896]

        # Create shared MoE DeBA-BB module (cross-layer weight sharing)
        self.deba_bb = MoE_DeBA_BB(
            stage_dims=stage_dims,
            bottleneck_dim=deba_bottleneck_dim,
            kernel_size=deba_kernel_size,
            num_experts=num_experts,
            num_modalities=num_modalities,
            scales=deba_scales,
            gate_noise_std=deba_gate_noise_std,
        )

        # Inject MoE-DeBA-BB into backbone blocks
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            stage_idx = self._BLOCK_TO_STAGE[t_layer_i]
            blk.attn.qkv = _MoE_DeBA_BB_qkv(
                blk.attn.qkv, self.deba_bb, stage_idx,
            )

        self.sam = sam_model

        # Determine Fusion Dim
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # CrossModalFusionHead (P9 동일)
        self.cross_modal_head = CrossModalFusionHead(
            in_channels=fusion_dim,
            num_modalities=num_modalities,
        )

    def forward(self, batched_input, multimask_output):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        # Collect DeBA-BB gate weights for visualization
        deba_gate_collector = []
        def _deba_gate_cb(gw):
            deba_gate_collector.append(gw)
        self.deba_bb._gate_callback = _deba_gate_cb
        try:
            # ============================================
            # Phase 1: 모든 모달리티 Image Encoding
            # ============================================
            for i in range(m):
                # Set modality for per-modality α scaling
                self.deba_bb.set_modality(i)

                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Cross-Modal 가중치 산출
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
            cross_weights, cross_logits = self.cross_modal_head(all_backbone_feats)  # (B, m)

            # UAMM용: max-normalize → best modality = 1.0
            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)  # (B, m)

            # ============================================
            # Phase 3: UAMM Modulation + Tracking
            # ============================================
            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                current_score = uamm_scores[:, frame_idx].unsqueeze(1)  # (B, 1)
                score_expanded = current_score.transpose(0, 1).unsqueeze(-1)  # (1, B, 1)
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
            amf_weights = cross_weights  # (B, m)

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
            if deba_gate_collector:
                self._last_moe_gates = np.stack(deba_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None
        finally:
            self.deba_bb._gate_callback = None

        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        # MoE DeBA-BB parameters (single shared module)
        deba_bb_tensors = {
            f"deba_bb.{k}": v
            for k, v in self.deba_bb.state_dict().items()
        }

        cross_modal_tensors = {
            f"cross_modal_head.{k}": v
            for k, v in self.cross_modal_head.state_dict().items()
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
            **deba_bb_tensors,
            **prompt_encoder_tensors,
            **mask_decoder_tensors,
            **cross_modal_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        # Load MoE DeBA-BB
        deba_bb_dict = {}
        for k, v in state_dict.items():
            if k.startswith("deba_bb."):
                deba_bb_dict[k.replace("deba_bb.", "")] = v
        if deba_bb_dict:
            self.deba_bb.load_state_dict(deba_bb_dict)

        # Load CrossModalFusionHead
        cross_modal_dict = {}
        for k, v in state_dict.items():
            if k.startswith("cross_modal_head."):
                cross_modal_dict[k.replace("cross_modal_head.", "")] = v
        if cross_modal_dict:
            self.cross_modal_head.load_state_dict(cross_modal_dict)

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


class LoRA_Sam_P24(nn.Module):
    """
    LoRA_Sam_P24: P9 + Quality-aware Memory Gating via Per-Modality Decoder Distillation

    P9 (SoftMoE-LoRA + CrossModalFusionHead) 기반에
    SpatialQualityGating을 추가하여 memory bank modulation.

    핵심 아이디어:
      - 각 모달리티의 encoded feature 품질을 spatial quality map으로 예측
      - 예측된 quality map으로 memory bank 저장 시 maskmem_features를 modulate
      - 열화된 영역의 memory 기여 ↓, 잘 예측하는 영역의 memory 기여 ↑

    학습 시 (Teacher-Student):
      - Teacher: per-modality feature를 SAM2 decoder에 single-frame으로 입력 (no memory)
        → per-pixel sigmoid confidence → quality_target → downsample
      - Student: SpatialQualityGating(feat) → predicted quality_map
      - Loss: MSE(predicted, target.detach())

    추론 시:
      - SpatialQualityGating만 실행 (teacher decoding 불필요)
      - Memory modulation은 학습 시와 동일

    Components:
      1. Structure: Soft-MoE LoRA (P9 동일)
      2. Quality: SpatialQualityGating → memory modulation (P24 신규)
      3. UAMM (Memory): CrossModal max-normalized softmax (P9 동일)
      4. AMF (Fusion): CrossModal raw softmax (P9 동일)

    Return:
      Training:  (m_output, m_feat, gate_loss_data)
        gate_loss_data = {'predicted': [quality_maps], 'target': [quality_targets]}
      Inference: (m_output, m_feat)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.1):
        nn.Module.__init__(self)

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # ── SoftMoE-LoRA injection (P9 동일) ──
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

        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # ── CrossModalFusionHead (P9 동일) ──
        self.cross_modal_head = CrossModalFusionHead(
            in_channels=fusion_dim,
            num_modalities=num_modalities,
        )

        # ── SpatialQualityGating (P24 신규) ──
        self.quality_gating = SpatialQualityGating(
            in_channels=fusion_dim,
            hidden_dim=quality_hidden_dim,
            min_quality=quality_min,
        )

        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None
        self._last_quality_maps = None

    def _teacher_decode_single(self, vision_feats, vision_pos_embeds, feat_sizes):
        """
        Teacher: single-frame decoding (no memory attention).
        Training only — generates per-modality quality targets.
        """
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        B = vision_feats[-1].size(1)
        C = self.sam.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        sam_outputs = self.sam._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=True,
        )
        _, high_res_multimasks, _, _, _, _, _ = sam_outputs
        return high_res_multimasks  # (B, num_classes, H_img, W_img)

    def forward(self, batched_input, multimask_output, gt_mask=None):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb
        try:
            # ============================================
            # Phase 1: Image Encoding (P9 동일)
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Cross-Modal Weights + Quality Map
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]
            cross_weights, cross_logits = self.cross_modal_head(all_backbone_feats)

            max_w = cross_weights.max(dim=1, keepdim=True)[0]
            uamm_scores = cross_weights / (max_w + 1e-8)

            # Per-modality spatial quality prediction (raw logits)
            quality_logits = []
            quality_maps = []  # sigmoid-ed for memory modulation
            for i in range(m):
                q_logit = self.quality_gating(all_backbone_feats[i])  # raw logits
                quality_logits.append(q_logit)
                quality_maps.append(self.quality_gating.logits_to_quality(q_logit))

            # ============================================
            # Phase 2.5 (Training): Teacher → CE-based quality targets
            # ============================================
            gate_loss_data = None
            if self.training and gt_mask is not None:
                quality_targets = []
                # Build ignore mask once → resize to FPN size for loss masking
                fpn_h, fpn_w = all_backbone_feats[0].shape[-2:]
                gt_safe = gt_mask.long().clone()
                ignore_mask_full = (gt_safe == 255)       # (B, H_img, W_img)
                gt_safe[ignore_mask_full] = 0
                # Resize ignore mask to FPN size (nearest to keep binary)
                ignore_mask_fpn = F.interpolate(
                    ignore_mask_full.unsqueeze(1).float(), size=(fpn_h, fpn_w),
                    mode='nearest',
                ).bool()                                   # (B, 1, fpn_h, fpn_w)

                for i in range(m):
                    with torch.no_grad():
                        teacher_logits = self._teacher_decode_single(
                            vision_feats[i], vision_pos_embeds[i], feat_sizes[i],
                        )
                        # teacher_logits: (B, num_classes, H_img, W_img)
                        if teacher_logits.shape[-2:] != gt_mask.shape[-2:]:
                            teacher_logits_resized = F.interpolate(
                                teacher_logits, size=gt_mask.shape[-2:],
                                mode='bilinear', align_corners=False,
                            )
                        else:
                            teacher_logits_resized = teacher_logits
                        # Per-pixel multi-class CE (ignore-safe)
                        ce_map = F.cross_entropy(
                            teacher_logits_resized, gt_safe,
                            reduction='none',
                        )  # (B, H, W)
                        ce_map[ignore_mask_full] = 0.0
                        # exp(-CE): low CE → high quality, ignore → 1.0
                        quality_target = torch.exp(-ce_map).unsqueeze(1)  # (B, 1, H, W)
                        quality_target = F.interpolate(
                            quality_target, size=(fpn_h, fpn_w),
                            mode='bilinear', align_corners=False,
                        )
                    quality_targets.append(quality_target)

                gate_loss_data = {
                    'predicted': quality_logits,    # raw logits
                    'target': quality_targets,       # exp(-CE) ∈ (0, 1]
                    'ignore_mask': ignore_mask_fpn,  # (B, 1, fpn_h, fpn_w) bool
                }

            # ============================================
            # Phase 3: UAMM + Tracking + Memory Modulation
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

                # ── Memory Modulation: quality-gated memory ──
                if multi_mask_output_step.get("maskmem_features") is not None:
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

            # ============================================
            # Phase 4: AMF Output Fusion
            # ============================================
            amf_weights = cross_weights

            w0 = amf_weights[:, 0].view(-1, 1, 1, 1)
            m_output = output[0] * w0
            m_feat = all_backbone_feats[0] * w0

            for i in range(1, m):
                wi = amf_weights[:, i].view(-1, 1, 1, 1)
                m_output = m_output + output[i] * wi
                m_feat = m_feat + all_backbone_feats[i] * wi

            self._last_uamm_scores = uamm_scores.detach().float().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().float().cpu().numpy()
            self._last_quality_maps = [q.detach().float().cpu().numpy() for q in quality_maps]
            if moe_gate_collector:
                self._last_moe_gates = np.stack(moe_gate_collector, axis=0).mean(axis=0)
            else:
                self._last_moe_gates = None
        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None

        if self.training and gate_loss_data is not None:
            return m_output, m_feat, gate_loss_data
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        cross_modal_tensors = {
            f"cross_modal_head.{k}": v for k, v in self.cross_modal_head.state_dict().items()
        }
        quality_gating_tensors = {
            f"quality_gating.{k}": v for k, v in self.quality_gating.state_dict().items()
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
            **cross_modal_tensors,
            **quality_gating_tensors,
            **prompt_encoder_tensors,
            **mask_decoder_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict: mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict: mv.load_state_dict(state_dict[v_key])

        cross_modal_dict = {k.replace("cross_modal_head.", ""): v for k, v in state_dict.items() if k.startswith("cross_modal_head.")}
        if cross_modal_dict:
            self.cross_modal_head.load_state_dict(cross_modal_dict)

        quality_dict = {k.replace("quality_gating.", ""): v for k, v in state_dict.items() if k.startswith("quality_gating.")}
        if quality_dict:
            self.quality_gating.load_state_dict(quality_dict)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()
        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)
        self.sam.load_state_dict(sam_dict)


# ============================================================================
# P25: Unified Spatial Quality Fusion
# ============================================================================

class LoRA_Sam_P25(nn.Module):
    """
    LoRA_Sam_P25: Unified Spatial Quality Fusion

    P24 기반에서 CrossModalFusionHead를 제거하고,
    SpatialQualityGating의 quality map으로 UAMM + AMF + Memory를 통합 제어.

    P24 → P25 변경:
      - CrossModalFusionHead 제거 (상수 수렴하므로 불필요)
      - UAMM: scalar max-norm (B, m) → spatial max-norm (B, 1, H, W) per modality
      - AMF: scalar softmax (B, m) → spatial softmax (B, 1, H, W) per modality
      - Memory modulation: P24 동일

    Teacher signal: P24 동일 (4-class CE → exp(-CE) → quality target)

    Return:
      Training:  (m_output, m_feat, gate_loss_data)
      Inference: (m_output, m_feat)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.1):
        nn.Module.__init__(self)

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # ── SoftMoE-LoRA injection (P9 동일) ──
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

        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        # ── No CrossModalFusionHead (P25: 제거됨) ──
        # ── SpatialQualityGating (P24 동일) ──
        self.quality_gating = SpatialQualityGating(
            in_channels=fusion_dim,
            hidden_dim=quality_hidden_dim,
            min_quality=quality_min,
        )

        self._last_uamm_scores = None   # (B, m) mean quality per modality (for logging)
        self._last_amf_weights = None    # (B, m) mean AMF weight per modality (for logging)
        self._last_moe_gates = None
        self._last_quality_maps = None   # list of numpy (B, 1, H, W)

    def _teacher_decode_single(self, vision_feats, vision_pos_embeds, feat_sizes):
        """Teacher: single-frame decoding (no memory attention)."""
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        B = vision_feats[-1].size(1)
        C = self.sam.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        sam_outputs = self.sam._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=True,
        )
        _, high_res_multimasks, _, _, _, _, _ = sam_outputs
        return high_res_multimasks  # (B, num_classes, H_img, W_img)

    def forward(self, batched_input, multimask_output, gt_mask=None):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []

        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb
        try:
            # ============================================
            # Phase 1: Image Encoding (P9 동일)
            # ============================================
            for i in range(m):
                img_emb = self.sam.forward_image(batched_input[i])
                image_embedding.append(img_emb)
                bb_out, v_feats, v_pos, f_sizes = self.sam._prepare_backbone_features(img_emb)
                backbone_out.append(bb_out)
                vision_feats.append(v_feats)
                vision_pos_embeds.append(v_pos)
                feat_sizes.append(f_sizes)

            # ============================================
            # Phase 2: Spatial Quality Map (CrossModalFusionHead 없음)
            # ============================================
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]

            quality_logits = []
            quality_maps = []
            for i in range(m):
                q_logit = self.quality_gating(all_backbone_feats[i])
                quality_logits.append(q_logit)
                quality_maps.append(self.quality_gating.logits_to_quality(q_logit))

            # ============================================
            # Phase 2.5 (Training): Teacher → CE-based quality targets
            # ============================================
            gate_loss_data = None
            if self.training and gt_mask is not None:
                quality_targets = []
                fpn_h, fpn_w = all_backbone_feats[0].shape[-2:]
                gt_safe = gt_mask.long().clone()
                ignore_mask_full = (gt_safe == 255)
                gt_safe[ignore_mask_full] = 0
                ignore_mask_fpn = F.interpolate(
                    ignore_mask_full.unsqueeze(1).float(), size=(fpn_h, fpn_w),
                    mode='nearest',
                ).bool()

                for i in range(m):
                    with torch.no_grad():
                        teacher_logits = self._teacher_decode_single(
                            vision_feats[i], vision_pos_embeds[i], feat_sizes[i],
                        )
                        if teacher_logits.shape[-2:] != gt_mask.shape[-2:]:
                            teacher_logits_resized = F.interpolate(
                                teacher_logits, size=gt_mask.shape[-2:],
                                mode='bilinear', align_corners=False,
                            )
                        else:
                            teacher_logits_resized = teacher_logits
                        ce_map = F.cross_entropy(
                            teacher_logits_resized, gt_safe,
                            reduction='none',
                        )
                        ce_map[ignore_mask_full] = 0.0
                        quality_target = torch.exp(-ce_map).unsqueeze(1)
                        quality_target = F.interpolate(
                            quality_target, size=(fpn_h, fpn_w),
                            mode='bilinear', align_corners=False,
                        )
                    quality_targets.append(quality_target)

                gate_loss_data = {
                    'predicted': quality_logits,
                    'target': quality_targets,
                    'ignore_mask': ignore_mask_fpn,
                }

            # ============================================
            # Phase 3: Spatial UAMM + Tracking + Memory Modulation
            # ============================================
            # Compute spatial max-norm across modalities
            q_stack = torch.stack(quality_maps, dim=1)          # (B, m, 1, H_fpn, W_fpn)
            q_max = q_stack.max(dim=1, keepdim=True).values     # (B, 1, 1, H_fpn, W_fpn)
            q_uamm_norm = q_stack / q_max.clamp(min=1e-6)      # (B, m, 1, H_fpn, W_fpn)

            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                # Spatial UAMM: interpolate quality to each vision_feat level
                q_uamm_i = q_uamm_norm[:, frame_idx]  # (B, 1, H_fpn, W_fpn)
                modulated_vision_feats = []
                for level_feat in vision_feats[frame_idx]:
                    # level_feat: (HW, B, C) — SAM2 format
                    _, B_feat, C_feat = level_feat.shape
                    # Infer spatial size from token count
                    hw = level_feat.shape[0]
                    # Estimate H, W from feat_sizes
                    h_l, w_l = None, None
                    for fs in feat_sizes[frame_idx]:
                        if fs[0] * fs[1] == hw:
                            h_l, w_l = fs
                            break
                    if h_l is None:
                        # fallback: square assumption
                        h_l = int(hw ** 0.5)
                        w_l = hw // h_l

                    q_resized = F.interpolate(
                        q_uamm_i, size=(h_l, w_l),
                        mode='bilinear', align_corners=False,
                    )  # (B, 1, h_l, w_l)
                    q_flat = q_resized.flatten(2).permute(2, 0, 1)  # (HW, B, 1)
                    modulated_vision_feats.append(level_feat * q_flat)

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

                # Memory Modulation (P24 동일)
                if multi_mask_output_step.get("maskmem_features") is not None:
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

            # ============================================
            # Phase 4: Spatial AMF Output Fusion
            # ============================================
            out_h, out_w = output[0].shape[-2:]
            q_amf_list = []
            for i in range(m):
                q_amf_i = F.interpolate(
                    quality_maps[i], size=(out_h, out_w),
                    mode='bilinear', align_corners=False,
                )  # (B, 1, H_out, W_out)
                q_amf_list.append(q_amf_i)

            q_amf_stack = torch.stack(q_amf_list, dim=0)       # (m, B, 1, H_out, W_out)
            q_amf_norm = q_amf_stack / q_amf_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)

            m_output = sum(q_amf_norm[i] * output[i] for i in range(m))

            # Feature fusion (for proto loss) — use same spatial weights at fpn resolution
            fpn_h_f, fpn_w_f = all_backbone_feats[0].shape[-2:]
            q_feat_list = []
            for i in range(m):
                q_fi = F.interpolate(
                    quality_maps[i], size=(fpn_h_f, fpn_w_f),
                    mode='bilinear', align_corners=False,
                )
                q_feat_list.append(q_fi)
            q_feat_stack = torch.stack(q_feat_list, dim=0)
            q_feat_norm = q_feat_stack / q_feat_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)
            m_feat = sum(q_feat_norm[i] * all_backbone_feats[i] for i in range(m))

            # ── Logging (scalar summaries for compatibility) ──
            # UAMM: per-modality mean quality (spatial mean → scalar)
            uamm_scalar = torch.stack(
                [q.mean(dim=[2, 3]).squeeze(1) for q in quality_maps], dim=1
            )  # (B, m)
            uamm_max = uamm_scalar.max(dim=1, keepdim=True)[0]
            uamm_log = uamm_scalar / uamm_max.clamp(min=1e-6)

            # AMF: per-modality mean weight (spatial mean → scalar)
            amf_log = torch.stack(
                [q_amf_norm[i].mean(dim=[1, 2, 3]) for i in range(m)], dim=1  # q_amf_norm[i]: (B, 1, H, W)
            )  # (B, m)
            # Renormalize to sum=1
            amf_log = amf_log / amf_log.sum(dim=1, keepdim=True).clamp(min=1e-6)

            self._last_uamm_scores = uamm_log.detach().float().cpu().numpy()
            self._last_amf_weights = amf_log.detach().float().cpu().numpy()
            self._last_quality_maps = [q.detach().float().cpu().numpy() for q in quality_maps]
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

        if self.training and gate_loss_data is not None:
            return m_output, m_feat, gate_loss_data
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        quality_gating_tensors = {
            f"quality_gating.{k}": v for k, v in self.quality_gating.state_dict().items()
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
            **quality_gating_tensors,
            **prompt_encoder_tensors,
            **mask_decoder_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict: mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict: mv.load_state_dict(state_dict[v_key])

        quality_dict = {k.replace("quality_gating.", ""): v for k, v in state_dict.items() if k.startswith("quality_gating.")}
        if quality_dict:
            self.quality_gating.load_state_dict(quality_dict)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()
        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)
        self.sam.load_state_dict(sam_dict)



# ═══════════════════════════════════════════════════════════════════════
# P26 v5: Per-Modality SQG + Multi-Scale FPN + Per-Modality Decoder
#          + Modal-Cond MoE + UAMM Softmax + AMF Entropy + No MemMod
# ═══════════════════════════════════════════════════════════════════════

class LoRA_Sam_P26(nn.Module):
    """
    LoRA_Sam_P26: Full v5 — 8 changes from P25.

    ① Per-Modality SQG (ModuleList, multi-task 충돌 해소)
    ② UAMM softmax (max-norm → softmax, 불연속 제거)
    ③ Relative quality teacher + KL loss
    ④ AMF output entropy (SQG와 분리, triple-duty 해소)
    ⑤ Memory modulation 제거 (이중 페널티 방지)
    ⑥ Multi-Scale FPN input for SQG (fpn[0,1,2] concat)
    ⑦ Per-Modality Decoder (decoder ×m deepcopy)
    ⑧ Modality-Conditioned MoE LoRA Gate (modal_embed → gate bias)

    Return:
      Training:  (m_output, m_feat, gate_loss_data)
      Inference: (m_output, m_feat)
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.3,
                 tau_uamm=1.0, tau_teacher=0.5,
                 memory_mod=False, amf_mode='sqg_quality',
                 multi_scale_sqg=True, per_modality_decoder=True,
                 cond_dim=8):
        nn.Module.__init__(self)

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))

        self.num_modalities = num_modalities
        self.tau_uamm = tau_uamm
        self.tau_teacher = tau_teacher
        self.memory_mod = memory_mod
        self.amf_mode = amf_mode
        self.multi_scale_sqg = multi_scale_sqg
        self.per_modality_decoder = per_modality_decoder
        self.cond_dim = cond_dim

        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # ── SoftMoE-LoRA injection with cond_dim (변경 ⑧) ──
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            moe_q = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts, cond_dim=cond_dim)
            moe_v = SoftMoE_LoRA_Layer(dim, r, num_experts=num_experts, cond_dim=cond_dim)

            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)

            blk.attn.qkv = _SoftMoE_LoRA_qkv(w_qkv_linear, moe_q, moe_v)

        self.sam = sam_model

        # ── 변경 ⑧: Modality Embedding for MoE gate conditioning ──
        self.modal_embed = nn.Embedding(num_modalities, cond_dim)

        # ── 변경 ⑦ (v6): Per-Modality Decoder (학습 전용 auxiliary) ──
        # per_modal_decoders: 학습 시 직접 CE loss → SQG target 생성 (추론 시 미사용)
        # self.sam.sam_mask_decoder: shared inference decoder (학습+추론 모두 사용)
        if per_modality_decoder:
            self.per_modal_decoders = nn.ModuleList([
                copy.deepcopy(sam_model.sam_mask_decoder)
                for _ in range(num_modalities)
            ])

        # ── 변경 ⑥: Multi-Scale FPN projection ──
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8  # 32
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim       # 256

        if multi_scale_sqg and use_high_res:
            # Raw FPN from backbone neck: ALL levels are d_model=256ch (before conv_s0/s1)
            # Project each to fusion_dim(32ch), then concat → 96ch
            d_model = self.sam.sam_mask_decoder.transformer_dim  # 256
            self.fpn_proj0 = nn.Conv2d(d_model, fusion_dim, 1)
            self.fpn_proj1 = nn.Conv2d(d_model, fusion_dim, 1)
            self.fpn_proj2 = nn.Conv2d(d_model, fusion_dim, 1)
            sqg_in_channels = fusion_dim * 3  # 96
        else:
            sqg_in_channels = fusion_dim

        # ── 변경 ①: Per-modality SpatialQualityGating ──
        self.quality_gatings = nn.ModuleList([
            SpatialQualityGating(
                in_channels=sqg_in_channels,
                hidden_dim=quality_hidden_dim,
                min_quality=quality_min,
            )
            for _ in range(num_modalities)
        ])

        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None
        self._last_quality_maps = None
        self._last_per_modal_outputs = None
        self._last_per_modal_feats = None
        # P26 detailed viz: spatial maps
        self._last_uamm_spatial = None    # list of m numpy (B,1,H_fpn,W_fpn)
        self._last_amf_spatial = None     # list of m numpy (B,1,H,W)
        self._last_entropy_maps = None    # list of m numpy (B,1,H,W)

    def _fuse_fpn_multiscale(self, backbone_fpn):
        """Fuse fpn[0,1,2] to fpn[0] resolution for SQG input.
        Raw FPN levels are all 256ch (d_model). Project each to 32ch then concat."""
        f0 = self.fpn_proj0(backbone_fpn[0])  # (B, 256→32, H0, W0)
        f1 = F.interpolate(
            self.fpn_proj1(backbone_fpn[1]),  # (B, 256→32, H1, W1)
            size=f0.shape[-2:], mode='bilinear', align_corners=False,
        )
        f2 = F.interpolate(
            self.fpn_proj2(backbone_fpn[2]),  # (B, 256→32, H2, W2)
            size=f0.shape[-2:], mode='bilinear', align_corners=False,
        )
        return torch.cat([f0, f1, f2], dim=1)  # (B, 96, H0, W0)

    def _swap_decoder(self, modal_idx):
        """Temporarily swap SAM2's mask decoder with per-modality decoder."""
        if self.per_modality_decoder:
            self.sam.sam_mask_decoder = self.per_modal_decoders[modal_idx]

    def _encode_single_modality(self, img, modal_idx_tensor):
        """Checkpoint-safe: encode one modality with correct condition.
        Returns flat tuple of tensors for torch.utils.checkpoint compatibility.
        Layout: (*backbone_fpn, *vision_pos_enc[, *raw_fpn if multi_scale_sqg])

        v6: Always uses shared decoder (self.sam.sam_mask_decoder) conv_s0/s1.
        Per-modal decoders are only used in Phase 2.5 auxiliary path.

        Nested checkpointing is used when gradient_checkpointing is enabled:
        - Outer: per-modality checkpoint wraps this entire function
        - Inner: per-block checkpoint inside HieraDet trunk
        set_condition() is called here, so both outer and inner recomputation
        see the correct _condition state.
        """
        B = img.shape[0]

        # Set MoE condition for this modality
        modal_cond = self.modal_embed(modal_idx_tensor).unsqueeze(0).expand(B, -1)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer.set_condition(modal_cond)

        # v6: No _swap_decoder — always use shared decoder's conv_s0/s1

        # Run backbone
        emb = self.sam.image_encoder(img)
        use_hr = getattr(self.sam, "use_high_res_features_in_sam", False)

        # Clone raw FPN before conv_s0/s1 (for multi-scale SQG)
        raw_fpn = []
        if self.multi_scale_sqg and use_hr:
            raw_fpn = [f.clone() for f in emb['backbone_fpn']]

        # Apply conv_s0/s1 from shared decoder (모든 모달리티 공통)
        if use_hr:
            emb["backbone_fpn"][0] = self.sam.sam_mask_decoder.conv_s0(emb["backbone_fpn"][0])
            emb["backbone_fpn"][1] = self.sam.sam_mask_decoder.conv_s1(emb["backbone_fpn"][1])

        return tuple(emb['backbone_fpn']) + tuple(emb['vision_pos_enc']) + tuple(raw_fpn)

    def _auxiliary_decode_single(self, decoder, vision_feats, vision_pos_embeds, feat_sizes):
        """v6: Auxiliary decode using a specific per-modal decoder (with grad).

        Unlike _teacher_decode_single, this takes an explicit decoder module
        and does NOT wrap in no_grad — used for per-modal auxiliary CE loss.
        The decoder is temporarily swapped into SAM2 for _forward_sam_heads,
        then restored to the shared decoder.
        """
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        B = vision_feats[-1].size(1)
        C = self.sam.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        # Temporarily swap decoder for _forward_sam_heads
        orig_decoder = self.sam.sam_mask_decoder
        self.sam.sam_mask_decoder = decoder
        try:
            sam_outputs = self.sam._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=True,
            )
        finally:
            self.sam.sam_mask_decoder = orig_decoder

        _, high_res_multimasks, _, _, _, _, _ = sam_outputs
        return high_res_multimasks

    def _fuse_outputs(self, output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes):
        """[P26 default] AMF-weighted output fusion + UAMM-weighted feature fusion.
        Overridable hook: P30 replaces this with a learned reliability-anchored modality
        router. Default body is byte-identical to the original inline fusion so P26/P27/
        P28/P29 behavior is unchanged."""
        m_output = sum(amf_norm[i] * output[i] for i in range(m))
        m_feat = sum(q_uamm_norm[i] * all_backbone_feats[i] for i in range(m))
        return m_output, m_feat

    def _teacher_decode_single(self, vision_feats, vision_pos_embeds, feat_sizes):
        """Teacher: single-frame decoding using current sam_mask_decoder (no memory attention)."""
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        B = vision_feats[-1].size(1)
        C = self.sam.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        sam_outputs = self.sam._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=True,
        )
        _, high_res_multimasks, _, _, _, _, _ = sam_outputs
        return high_res_multimasks

    def forward(self, batched_input, multimask_output, gt_mask=None):
        m = len(batched_input)
        image_embedding, backbone_out, vision_feats = [], [], []
        vision_pos_embeds, feat_sizes, output = [], [], []
        # Store raw fpn (before conv_s0/s1) for multi-scale SQG
        raw_backbone_fpns = []

        moe_gate_collector = []
        def _moe_gate_cb(gw):
            moe_gate_collector.append(gw)
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer._gate_callback = _moe_gate_cb

        # ⑧ Modal-Conditioned MoE + gradient checkpointing:
        # Uses nested checkpointing — outer per-modality + inner per-block.
        # set_condition() is called inside _encode_single_modality, so both
        # outer and inner recomputation see the correct _condition state.
        trunk = self.sam.image_encoder.trunk
        _orig_gc = getattr(trunk, 'gradient_checkpointing', False)

        try:
            # ============================================
            # Phase 1: Image Encoding + 변경 ⑧ Modal-Cond MoE + ⑦ Per-Modal Decoder
            # Per-modality checkpointing: each modality encoding is a checkpoint unit.
            # Condition & decoder are set inside _encode_single_modality,
            # so recomputation during backward produces identical results.
            # ============================================
            device = batched_input[0].device
            use_hr = getattr(self.sam, "use_high_res_features_in_sam", False)
            scalp = getattr(self.sam.image_encoder, "scalp", 0)
            n_fpn = len(self.sam.image_encoder.neck.convs) - scalp  # 4 convs - 1 scalp = 3

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

                # Unpack flat tuple → dict
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

            # Clear MoE conditions
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer.set_condition(None)

            # ============================================
            # Phase 2: Per-Modality Spatial Quality Map (변경 ① + ⑥)
            # ============================================
            # all_backbone_feats: fpn[0] after conv_s0 (32ch) for UAMM modulation
            all_backbone_feats = [image_embedding[i]['backbone_fpn'][0] for i in range(m)]

            quality_logits = []
            quality_maps = []
            for i in range(m):
                # ⑥: Multi-Scale FPN input for SQG
                if self.multi_scale_sqg and len(raw_backbone_fpns) > 0:
                    sqg_input = self._fuse_fpn_multiscale(raw_backbone_fpns[i])
                else:
                    sqg_input = all_backbone_feats[i]

                q_logit = self.quality_gatings[i](sqg_input)
                quality_logits.append(q_logit)
                quality_maps.append(self.quality_gatings[i].logits_to_quality(q_logit))

            # ============================================
            # Phase 2.5 (Training): v6 — Per-modal auxiliary CE + Relative Quality Teacher
            # per_modal_decoders로 직접 CE loss (grad 있음) + SQG target (detach)
            # ============================================
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
                    # v6: per-modal decoder로 직접 decode (grad 있음)
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

                    # (1) Auxiliary CE loss (grad flows to per_modal_decoder + encoder)
                    aux_ce = F.cross_entropy(aux_logits_resized, gt_safe, ignore_index=255)
                    aux_losses.append(aux_ce)

                    # (2) CE map for SQG target (detach — SQG target은 gradient 차단)
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

                ce_stack = torch.stack(ce_maps, dim=0)  # (m, B, 1, fpn_h, fpn_w)
                quality_target_dist = F.softmax(-ce_stack / self.tau_teacher, dim=0)

                gate_loss_data = {
                    'predicted_logits': quality_logits,
                    'quality_target_dist': quality_target_dist,
                    'ignore_mask': ignore_mask_fpn,
                    'loss_type': 'kl',
                    'aux_ce_losses': aux_losses,  # v6: per-modal auxiliary CE losses
                }

            # ============================================
            # Phase 3: v6 — UAMM softmax + Shared Decoder track_step + ⑤ No MemMod
            # ============================================
            q_logit_stack = torch.stack(quality_logits, dim=0)  # (m, B, 1, H_fpn, W_fpn)
            q_uamm_norm = F.softmax(q_logit_stack / self.tau_uamm, dim=0)

            output_dict = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }

            for frame_idx in range(m):
                is_init = (frame_idx == 0)

                # v6: No _swap_decoder — shared decoder (sam.sam_mask_decoder) 사용
                # per_modal_decoders는 Phase 2.5 auxiliary에서만 사용

                q_uamm_i = q_uamm_norm[frame_idx]  # (B, 1, H_fpn, W_fpn)
                modulated_vision_feats = []
                for level_feat in vision_feats[frame_idx]:
                    _, B_feat, C_feat = level_feat.shape
                    hw = level_feat.shape[0]
                    h_l, w_l = None, None
                    for fs in feat_sizes[frame_idx]:
                        if fs[0] * fs[1] == hw:
                            h_l, w_l = fs
                            break
                    if h_l is None:
                        h_l = int(hw ** 0.5)
                        w_l = hw // h_l

                    q_resized = F.interpolate(
                        q_uamm_i, size=(h_l, w_l),
                        mode='bilinear', align_corners=False,
                    )
                    q_flat = q_resized.flatten(2).permute(2, 0, 1)
                    modulated_vision_feats.append(level_feat * q_flat)

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

                # ⑤: Memory Modulation — config로 제어 (기본: 비활성화)
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

            # ============================================
            # Phase 4: v6 — AMF (SQG quality softmax 또는 output entropy fallback)
            # ============================================
            out_h, out_w = output[0].shape[-2:]
            num_classes = output[0].shape[1]

            if self.amf_mode == 'sqg_quality':
                # v6: UAMM과 동일한 sqg_weight 재사용 — tau 없음
                # q_uamm_norm은 Phase 3에서 이미 계산됨
                amf_norm_list = []
                for i in range(m):
                    amf_i = F.interpolate(
                        q_uamm_norm[i], size=(out_h, out_w),
                        mode='bilinear', align_corners=False,
                    )
                    amf_norm_list.append(amf_i)
                amf_norm = torch.stack(amf_norm_list, dim=0)  # (m, B, 1, H_out, W_out)
            elif self.amf_mode == 'output_entropy':
                # Legacy: output entropy 기반 AMF
                amf_weights = []
                for i in range(m):
                    prob = F.softmax(output[i], dim=1)
                    entropy = -(prob * (prob + 1e-8).log()).sum(dim=1, keepdim=True)
                    confidence = 1.0 - entropy / math.log(num_classes)
                    amf_weights.append(confidence)
                amf_stack = torch.stack(amf_weights, dim=0)
                amf_norm = amf_stack / amf_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)
            else:
                # Fallback: quality_map sum-norm
                q_amf_list = []
                for i in range(m):
                    q_amf_i = F.interpolate(
                        quality_maps[i], size=(out_h, out_w),
                        mode='bilinear', align_corners=False,
                    )
                    q_amf_list.append(q_amf_i)
                amf_stack = torch.stack(q_amf_list, dim=0)
                amf_norm = amf_stack / amf_stack.sum(dim=0, keepdim=True).clamp(min=1e-6)

            # [P30 hook] default = AMF/UAMM weighted sums (P26–P29 unchanged);
            # P30 overrides _fuse_outputs with a learned reliability-anchored router.
            m_output, m_feat = self._fuse_outputs(
                output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes)

            # ── Logging ──
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
            # P26 detailed viz: spatial maps
            self._last_uamm_spatial = [q_uamm_norm[i].detach().float().cpu().numpy() for i in range(m)]
            self._last_amf_spatial = [amf_norm[i].detach().float().cpu().numpy() for i in range(m)]
            # Entropy maps (for viz — computed regardless of amf_mode)
            ent_maps = []
            for i in range(m):
                prob_i = F.softmax(output[i], dim=1)
                ent_i = -(prob_i * (prob_i + 1e-8).log()).sum(dim=1, keepdim=True)
                ent_maps.append(ent_i.detach().float().cpu().numpy())
            self._last_entropy_maps = ent_maps
        finally:
            for layer in self.moe_layers_q + self.moe_layers_v:
                layer._gate_callback = None
                layer.set_condition(None)

        if self.training and gate_loss_data is not None:
            return m_output, m_feat, gate_loss_data
        return m_output, m_feat

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()

        # Per-modality quality gatings
        quality_gating_tensors = {}
        for idx, qg in enumerate(self.quality_gatings):
            for k, v in qg.state_dict().items():
                quality_gating_tensors[f"quality_gating_{idx}.{k}"] = v

        # Multi-scale FPN projections
        extra_tensors = {}
        if self.multi_scale_sqg and hasattr(self, 'fpn_proj0'):
            for k, v in self.fpn_proj0.state_dict().items():
                extra_tensors[f"fpn_proj0.{k}"] = v
            for k, v in self.fpn_proj1.state_dict().items():
                extra_tensors[f"fpn_proj1.{k}"] = v
            for k, v in self.fpn_proj2.state_dict().items():
                extra_tensors[f"fpn_proj2.{k}"] = v

        # Modal embedding
        for k, v in self.modal_embed.state_dict().items():
            extra_tensors[f"modal_embed.{k}"] = v

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        model_ref = self.sam.module if isinstance(self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else self.sam
        state_dict = model_ref.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        # Per-modality decoders
        per_decoder_tensors = {}
        if self.per_modality_decoder:
            for idx, dec in enumerate(self.per_modal_decoders):
                for k, v in dec.state_dict().items():
                    per_decoder_tensors[f"per_modal_decoder_{idx}.{k}"] = v

        merged_dict = {
            **moe_params,
            **quality_gating_tensors,
            **extra_tensors,
            **prompt_encoder_tensors,
            **mask_decoder_tensors,
            **per_decoder_tensors,
        }
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            if q_key in state_dict: mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict: mv.load_state_dict(state_dict[v_key])

        # Per-modality quality gatings
        for idx, qg in enumerate(self.quality_gatings):
            prefix = f"quality_gating_{idx}."
            qg_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
            if qg_dict:
                qg.load_state_dict(qg_dict)

        # Multi-scale FPN projections
        if hasattr(self, 'fpn_proj0'):
            p0_dict = {k.replace("fpn_proj0.", ""): v for k, v in state_dict.items() if k.startswith("fpn_proj0.")}
            if p0_dict: self.fpn_proj0.load_state_dict(p0_dict)
            p1_dict = {k.replace("fpn_proj1.", ""): v for k, v in state_dict.items() if k.startswith("fpn_proj1.")}
            if p1_dict: self.fpn_proj1.load_state_dict(p1_dict)
            p2_dict = {k.replace("fpn_proj2.", ""): v for k, v in state_dict.items() if k.startswith("fpn_proj2.")}
            if p2_dict: self.fpn_proj2.load_state_dict(p2_dict)

        # Modal embedding
        me_dict = {k.replace("modal_embed.", ""): v for k, v in state_dict.items() if k.startswith("modal_embed.")}
        if me_dict:
            self.modal_embed.load_state_dict(me_dict)

        # Per-modality decoders
        if self.per_modality_decoder:
            for idx, dec in enumerate(self.per_modal_decoders):
                prefix = f"per_modal_decoder_{idx}."
                dec_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
                if dec_dict:
                    dec.load_state_dict(dec_dict)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()
        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)
        self.sam.load_state_dict(sam_dict)


class LoRA_Sam_P26_AblB(LoRA_Sam_P26):
    """
    LoRA_Sam_P26_AblB: Ablation B — Single LoRA (parameter-matched).

    P26과 동일한 forward / SQG / UAMM / AMF / Per-Modal Decoder / Multi-Scale FPN을
    상속받고, **SoftMoE LoRA만 Single LoRA로 교체**한다.
    파라미터 수를 매칭하여 MoE routing의 기여도를 분리 검증.

    제거 (vs P26): SoftMoE LoRA, Modality-Conditioned Gate (modal_embed, cond_dim),
                   MoE gate collection
    유지 (상속):   forward, _fuse_fpn_multiscale, _auxiliary_decode_single,
                   ① SQG / ② UAMM / ③ KL teacher / ④ AMF / ⑤ MemMod / ⑥ MS-FPN / ⑦ per-modal decoder

    P26 forward의 `for layer in self.moe_layers_q + self.moe_layers_v` 구문은 빈
    ModuleList일 때 자동으로 no-op이 되므로, 빈 더미만 두면 그대로 상속 가능.
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.3,
                 tau_uamm=1.0, tau_teacher=0.5,
                 memory_mod=False, amf_mode='sqg_quality',
                 multi_scale_sqg=True, per_modality_decoder=True):
        # P26.__init__는 호출하지 않는다 — SoftMoE 생성 경로를 우회해야 함.
        nn.Module.__init__(self)

        assert r > 0
        self.lora_layer = lora_layer if lora_layer else list(
            range(len(sam_model.image_encoder.trunk.blocks))
        )

        self.num_modalities = num_modalities
        self.tau_uamm = tau_uamm
        self.tau_teacher = tau_teacher
        self.memory_mod = memory_mod
        self.amf_mode = amf_mode
        self.multi_scale_sqg = multi_scale_sqg
        self.per_modality_decoder = per_modality_decoder

        # ── Single LoRA injection (SoftMoE LoRA 자리) ──
        self.w_As = []
        self.w_Bs = []
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features

            w_a_q = nn.Linear(dim, r, bias=False)
            w_b_q = nn.Linear(r, dim, bias=False)
            w_a_v = nn.Linear(dim, r, bias=False)
            w_b_v = nn.Linear(r, dim, bias=False)
            self.w_As.extend([w_a_q, w_a_v])
            self.w_Bs.extend([w_b_q, w_b_v])

            blk.attn.qkv = _LoRA_qkv(w_qkv_linear, w_a_q, w_b_q, w_a_v, w_b_v)

        # LoRA 파라미터는 _LoRA_qkv → self.sam.image_encoder 경로로 등록되므로
        # 별도 ParameterList는 두지 않음 (이중 카운팅 방지).
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

        # P26 forward는 self.moe_layers_q/v를 순회하므로 빈 더미를 둔다 (no-op).
        self.moe_layers_q = nn.ModuleList()
        self.moe_layers_v = nn.ModuleList()

        self.sam = sam_model

        # ── ⑦ Per-Modality Decoder (학습 전용 auxiliary) ──
        if per_modality_decoder:
            self.per_modal_decoders = nn.ModuleList([
                copy.deepcopy(sam_model.sam_mask_decoder)
                for _ in range(num_modalities)
            ])

        # ── ⑥ Multi-Scale FPN projection ──
        use_high_res = getattr(self.sam, "use_high_res_features_in_sam", False)
        if use_high_res:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim // 8
        else:
            fusion_dim = self.sam.sam_mask_decoder.transformer_dim

        if multi_scale_sqg and use_high_res:
            d_model = self.sam.sam_mask_decoder.transformer_dim
            self.fpn_proj0 = nn.Conv2d(d_model, fusion_dim, 1)
            self.fpn_proj1 = nn.Conv2d(d_model, fusion_dim, 1)
            self.fpn_proj2 = nn.Conv2d(d_model, fusion_dim, 1)
            sqg_in_channels = fusion_dim * 3
        else:
            sqg_in_channels = fusion_dim

        # ── ① Per-modality SpatialQualityGating ──
        self.quality_gatings = nn.ModuleList([
            SpatialQualityGating(
                in_channels=sqg_in_channels,
                hidden_dim=quality_hidden_dim,
                min_quality=quality_min,
            )
            for _ in range(num_modalities)
        ])

        # Visualization buffers (P26 forward가 채움)
        self._last_uamm_scores = None
        self._last_amf_weights = None
        self._last_moe_gates = None
        self._last_quality_maps = None
        self._last_per_modal_outputs = None
        self._last_per_modal_feats = None
        self._last_uamm_spatial = None
        self._last_amf_spatial = None
        self._last_entropy_maps = None

    def _encode_single_modality(self, img, modal_idx_tensor=None):
        """P26.forward에서 호출되는 시그니처와 호환 — modal_idx_tensor는 무시한다.
        Single LoRA는 modality condition이 없음."""
        emb = self.sam.image_encoder(img)
        use_hr = getattr(self.sam, "use_high_res_features_in_sam", False)

        raw_fpn = []
        if self.multi_scale_sqg and use_hr:
            raw_fpn = [f.clone() for f in emb['backbone_fpn']]

        if use_hr:
            emb["backbone_fpn"][0] = self.sam.sam_mask_decoder.conv_s0(emb["backbone_fpn"][0])
            emb["backbone_fpn"][1] = self.sam.sam_mask_decoder.conv_s1(emb["backbone_fpn"][1])

        return tuple(emb['backbone_fpn']) + tuple(emb['vision_pos_enc']) + tuple(raw_fpn)

    # forward, _fuse_fpn_multiscale, _auxiliary_decode_single 는 P26에서 그대로 상속.

    def save_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')

        a_tensors = {f"w_a_{i:03d}": w.weight for i, w in enumerate(self.w_As)}
        b_tensors = {f"w_b_{i:03d}": w.weight for i, w in enumerate(self.w_Bs)}

        quality_gating_tensors = {
            f"quality_gating_{idx}.{k}": v
            for idx, qg in enumerate(self.quality_gatings)
            for k, v in qg.state_dict().items()
        }

        extra_tensors = {}
        if self.multi_scale_sqg and hasattr(self, 'fpn_proj0'):
            for proj_name in ('fpn_proj0', 'fpn_proj1', 'fpn_proj2'):
                for k, v in getattr(self, proj_name).state_dict().items():
                    extra_tensors[f"{proj_name}.{k}"] = v

        model_ref = self.sam.module if isinstance(
            self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
        ) else self.sam
        state_dict = model_ref.state_dict()
        prompt_encoder_tensors = {k: v for k, v in state_dict.items() if 'prompt_encoder' in k}
        mask_decoder_tensors = {k: v for k, v in state_dict.items() if 'mask_decoder' in k}

        per_decoder_tensors = {}
        if self.per_modality_decoder:
            for idx, dec in enumerate(self.per_modal_decoders):
                for k, v in dec.state_dict().items():
                    per_decoder_tensors[f"per_modal_decoder_{idx}.{k}"] = v

        torch.save({
            **a_tensors, **b_tensors,
            **quality_gating_tensors, **extra_tensors,
            **prompt_encoder_tensors, **mask_decoder_tensors,
            **per_decoder_tensors,
        }, filename)

    def load_lora_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        for i, w_A in enumerate(self.w_As):
            key = f"w_a_{i:03d}"
            if key in state_dict:
                w_A.weight = Parameter(state_dict[key])
        for i, w_B in enumerate(self.w_Bs):
            key = f"w_b_{i:03d}"
            if key in state_dict:
                w_B.weight = Parameter(state_dict[key])

        for idx, qg in enumerate(self.quality_gatings):
            prefix = f"quality_gating_{idx}."
            sub = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
            if sub:
                qg.load_state_dict(sub)

        if hasattr(self, 'fpn_proj0'):
            for proj_name in ('fpn_proj0', 'fpn_proj1', 'fpn_proj2'):
                prefix = f"{proj_name}."
                sub = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
                if sub:
                    getattr(self, proj_name).load_state_dict(sub)

        if self.per_modality_decoder:
            for idx, dec in enumerate(self.per_modal_decoders):
                prefix = f"per_modal_decoder_{idx}."
                sub = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
                if sub:
                    dec.load_state_dict(sub)

        sam_dict = self.sam.state_dict()
        for module_name in ('prompt_encoder', 'mask_decoder'):
            module_keys = [k for k in sam_dict.keys() if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                sam_dict.update({k: v for k, v in zip(module_keys, module_values)})
        self.sam.load_state_dict(sam_dict)


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

            m_output = sum(amf_norm[i] * output[i] for i in range(m))

            # Feature fusion — keep using q_uamm_norm as spatial weight (no UAMM *feat multiplication
            # inside track_step; this is the final fusion, not the modulation)
            m_feat = sum(q_uamm_norm[i] * all_backbone_feats[i] for i in range(m))

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


class LoRA_Sam_P28(LoRA_Sam_P27):
    """
    LoRA_Sam_P28: RBMA — Reliability-Biased Memory Attention.

    P27의 additive memory-attention logit-bias 기구(softmax(QK^T/√d + λ·B)V)를 그대로
    재사용하되, bias 신호 B를 SpatialQualityGating(학습형 예측기, B-2 진단에서
    underfit·평탄·정적붕괴 확인)에서 **per-modality decoder의 training-free
    예측 불확실성**으로 교체한다.

      reliability_i(x) = 1 - H(softmax(D_i(f_i)))(x) / log(C)
        - D_i = per_modal_decoders[i] (모달리티 단독 디코드, memory 융합 이전 → 순환 없음)
        - H   = per-pixel predictive entropy (GT 불필요, 학습형 quality head 불필요)
      bias_i = λ · (reliability_i - mean_j reliability_j)   # 모달리티 간 상대 신뢰도

    설계 근거 (deep-research 2026-06-15):
      - 신규성 축 = B(attention LOGIT additive bias): 선행연구는 모두 feature-multiply /
        attention-output-scale / loss-level (ReliFusion/READ/UTFNet/HyperDUM 등). logit-bias
        전례 0. 신호(A)는 UTFNet/HyperDUM의 *학습 evidential/HD head* 대비 "training-free"가 차별점.
      - B-2 병목 직격: SQG(frozen feature 예측기)를 bias 경로에서 제거, decoder 자체
        예측분포에서 신뢰도 산출 (Target Q=exp(-CE)의 GT-free 버전).

    SQG/per-modal decoder/aux-CE/KL teacher loss는 학습용으로 유지(점진 제거는 ablation).
    추론 시 bias는 오직 decoder 불확실성에서 나옴. λ(self.lambda_bias)만 학습.
    순수 RBMA 평가를 위해 config에서 AMF_MODE: uniform 권장(출력 융합은 등가중).
    """
    RBMA_EPS = 1e-6

    def _compute_bias_source(self, quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m):
        fpn_h, fpn_w = quality_logits[0].shape[-2:]
        rel_maps = []
        for i in range(m):
            with torch.no_grad():
                aux_logits = self._auxiliary_decode_single(
                    self.per_modal_decoders[i],
                    vision_feats[i], vision_pos_embeds[i], feat_sizes[i],
                )  # (B, C, H, W) class logits
                C = aux_logits.shape[1]
                p = F.softmax(aux_logits, dim=1)
                ent = -(p * torch.log(p + self.RBMA_EPS)).sum(dim=1, keepdim=True)  # (B,1,H,W)
                ent = ent / math.log(C)                       # normalize → [0,1]
                rel = 1.0 - ent                               # reliability (high=confident)
                rel = F.interpolate(rel, size=(fpn_h, fpn_w),
                                    mode='bilinear', align_corners=False)
            rel_maps.append(rel)
        # center across modalities per pixel → zero-mean relative up/down-weight
        rel_stack = torch.stack(rel_maps, dim=0)              # (m, B, 1, H, W)
        rel_stack = rel_stack - rel_stack.mean(dim=0, keepdim=True)
        return [rel_stack[i] for i in range(m)]


class LoRA_Sam_P29(LoRA_Sam_P28):
    """
    LoRA_Sam_P29: Self-Derived Condition (SDC) routing for the Soft-MoE LoRA gate.

    계보: P29(P28) — RBMA(P27/P28 memory-attn logit bias)·SoftMoE LoRA·per-modal
    decoder·SQG는 그대로 상속. **변경점은 MoE 게이트의 조건화**뿐:

      - P28 게이트 조건 = modal_embed(modality id)만 → day/night/snow가 라우터에 부재.
      - P29: **이미지에서 무감독 자기파생한 조건 latent z_c**(SelfDerivedCondition)를
        modal_embed과 concat → 게이트에 **FiLM(scale+shift)** 로 주입(cond_mode='film',
        zero-init=identity). z_c는 RGB 입력의 채널 통계 → MLP → prototype bank(K) cosine
        soft-assign(라벨/텍스트 0). label-free clustering loss는 self._sdc_loss로 노출
        (trainer가 total loss에 가산).

    구현 방식(최소 침습): super().__init__로 P28(=add 게이트) 빌드 후, 기존 moe 게이트의
    조건 분기만 'film' + (cond_dim+sdc_latent)로 **사후 재구성**. P28 클래스 자체는 불변.

    deviation 노트: 설계의 "fpn[0] feature" 대신 **RGB 입력 이미지**(in_channels=3)에서
    z_c를 계산해 encoder cyclic-dependency를 회피(게이트가 인코딩 전에 조건 필요).
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.3,
                 tau_uamm=1.0, tau_teacher=0.5,
                 memory_mod=False, amf_mode='uniform',
                 multi_scale_sqg=True, per_modality_decoder=True,
                 cond_dim=8, lambda_bias_init=1.0,
                 sdc_enable=True, sdc_K=6, sdc_latent=32, sdc_in_channels=3):
        super().__init__(
            sam_model=sam_model, r=r, lora_layer=lora_layer,
            num_experts=num_experts, num_modalities=num_modalities,
            quality_hidden_dim=quality_hidden_dim, quality_min=quality_min,
            tau_uamm=tau_uamm, tau_teacher=tau_teacher,
            memory_mod=memory_mod, amf_mode=amf_mode,
            multi_scale_sqg=multi_scale_sqg, per_modality_decoder=per_modality_decoder,
            cond_dim=cond_dim, lambda_bias_init=lambda_bias_init,
        )
        self.sdc_enable = sdc_enable
        self.sdc_latent = sdc_latent
        self._z_c = None
        self._sdc_loss = None
        if sdc_enable:
            self.sdc = SelfDerivedCondition(
                in_channels=sdc_in_channels, latent_dim=sdc_latent, K=sdc_K)
            new_cd = cond_dim + sdc_latent
            # Post-hoc reconfigure existing Soft-MoE gates: add→film, widen cond.
            for layer in list(self.moe_layers_q) + list(self.moe_layers_v):
                layer.cond_mode = 'film'
                layer.cond_dim = new_cd
                if hasattr(layer, 'cond_proj'):
                    del layer.cond_proj
                film = nn.Linear(new_cd, 2 * layer.num_experts)
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
                layer.cond_film = film

    def _encode_single_modality(self, img, modal_idx_tensor):
        """P26 body + SDC: gate condition = [modal_embed ⊕ z_c]."""
        B = img.shape[0]
        modal_cond = self.modal_embed(modal_idx_tensor).unsqueeze(0).expand(B, -1)
        if self.sdc_enable and self._z_c is not None:
            cond = torch.cat([modal_cond, self._z_c], dim=-1)
        else:
            cond = modal_cond
        for layer in self.moe_layers_q + self.moe_layers_v:
            layer.set_condition(cond)

        emb = self.sam.image_encoder(img)
        use_hr = getattr(self.sam, "use_high_res_features_in_sam", False)
        raw_fpn = []
        if self.multi_scale_sqg and use_hr:
            raw_fpn = [f.clone() for f in emb['backbone_fpn']]
        if use_hr:
            emb["backbone_fpn"][0] = self.sam.sam_mask_decoder.conv_s0(emb["backbone_fpn"][0])
            emb["backbone_fpn"][1] = self.sam.sam_mask_decoder.conv_s1(emb["backbone_fpn"][1])
        return tuple(emb['backbone_fpn']) + tuple(emb['vision_pos_enc']) + tuple(raw_fpn)

    def forward(self, batched_input, multimask_output, gt_mask=None):
        # Compute the self-derived condition once per image from RGB (modality 0),
        # before the per-modality encoding loop (gates read self._z_c).
        if self.sdc_enable:
            rgb = batched_input[0]                      # (B, 3, H, W)
            z_c, sdc_loss = self.sdc(rgb)
            self._z_c = z_c
            self._sdc_loss = sdc_loss
        else:
            self._z_c = None
            self._sdc_loss = None
        return super().forward(batched_input, multimask_output, gt_mask)


class LoRA_Sam_P30(LoRA_Sam_P29):
    """
    LoRA_Sam_P30: rare-class + dead-modality fix, grounded in the P28 failure analysis
    (condclass: Water/Bridge=0.00, Wall/Other/Dynamic/Ground/TrafficLight ≪; ablation:
    event Δ-0.000 / LiDAR Δ+0.001 = UNUSED while depth Δ-0.224 / RGB Δ-0.097 dominate).
    Inherits P29 (SDC) + RBMA. Two config-gated mechanisms (both OFF by default → P28/P29
    byte-identical):

    ① Class-token decoder (rare-class fix): C learnable class queries cross-attend the fused
       cross-modal memory feature m_feat (all modalities + RBMA bias) → per-class masks.
       Ports the SAM3-RBMA class-collapse break (val 8.49→16.27). Applied post-hoc on the
       grad-attached m_feat returned by super().forward (trains end-to-end). APPROXIMATION:
       light MaskFormer/SAM-style decoder, not surgery on sam_mask_decoder weights.
    ② Reliability-anchored learned router (dead-modality fix): replaces the fixed UAMM scalar
       fusion with a learned (optionally per-class) modality router, anchored by the RBMA
       reliability so it can't collapse to a constant (P10–P27 'gate 상수수렴', ISSUE-002/015).
       Implemented by overriding _fuse_outputs (the P26 hook), using grad-attached per-modal
       feats/outputs. Reliability anchor = 1 − H(softmax(output_i))/logC (training-free).
       Exposes self._router_reg (modality-mixing entropy) for an optional diversity term.
    """
    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.3,
                 tau_uamm=1.0, tau_teacher=0.5,
                 memory_mod=False, amf_mode='uniform',
                 multi_scale_sqg=True, per_modality_decoder=True,
                 cond_dim=8, lambda_bias_init=1.0,
                 sdc_enable=True, sdc_K=6, sdc_latent=32, sdc_in_channels=3,
                 class_token_decoder=False, ctd_dim=128,
                 learned_router=False, router_per_class=False, router_anchor_lambda=1.0,
                 num_classes=25):
        super().__init__(
            sam_model=sam_model, r=r, lora_layer=lora_layer,
            num_experts=num_experts, num_modalities=num_modalities,
            quality_hidden_dim=quality_hidden_dim, quality_min=quality_min,
            tau_uamm=tau_uamm, tau_teacher=tau_teacher,
            memory_mod=memory_mod, amf_mode=amf_mode,
            multi_scale_sqg=multi_scale_sqg, per_modality_decoder=per_modality_decoder,
            cond_dim=cond_dim, lambda_bias_init=lambda_bias_init,
            sdc_enable=sdc_enable, sdc_K=sdc_K, sdc_latent=sdc_latent, sdc_in_channels=sdc_in_channels,
        )
        self.class_token_decoder_enable = class_token_decoder
        self.learned_router_enable = learned_router
        self.router_per_class = router_per_class
        self.num_classes_p30 = num_classes
        self._router_reg = None
        feat_ch = 32  # all_backbone_feats[i] = fpn[0] after conv_s0 (32ch)
        if class_token_decoder:
            self.class_decoder = ClassTokenDecoder(feat_ch, num_classes, dim=ctd_dim)
        if learned_router:
            self.router = ReliabilityAnchoredRouter(
                feat_ch, num_modalities, num_classes=num_classes,
                per_class=router_per_class, anchor_lambda=router_anchor_lambda)

    def _fuse_outputs(self, output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes):
        if not self.learned_router_enable:
            return super()._fuse_outputs(output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes)
        # Reliability anchor: training-free 1 - normalized entropy of each modality's logits
        rels = []
        for i in range(m):
            p = F.softmax(output[i], dim=1)
            ent = -(p * (p + 1e-8).log()).sum(dim=1, keepdim=True) / math.log(num_classes)
            rels.append(1.0 - ent)                                  # (B,1,H_out,W_out)
        rel = torch.stack(rels, dim=0)                              # (m,B,1,H_out,W_out)
        Bn = rel.shape[1]; hf, wf = all_backbone_feats[0].shape[-2:]
        rel_f = F.interpolate(rel.reshape(m * Bn, 1, *rel.shape[-2:]), size=(hf, wf),
                              mode='bilinear', align_corners=False).reshape(m, Bn, 1, hf, wf)
        w, reg = self.router(all_backbone_feats, rel_f)            # w: (m,B,K,hf,wf)
        self._router_reg = reg
        w_feat = w.mean(dim=2, keepdim=True) if self.router_per_class else w  # (m,B,1,hf,wf)
        m_feat = sum(w_feat[i] * all_backbone_feats[i] for i in range(m))
        Hh, Ww = output[0].shape[-2:]
        w_out = F.interpolate(w.reshape(m * Bn, w.shape[2], hf, wf), size=(Hh, Ww),
                              mode='bilinear', align_corners=False).reshape(m, Bn, w.shape[2], Hh, Ww)
        m_output = sum(w_out[i] * output[i] for i in range(m))     # per-class or broadcast scalar
        return m_output, m_feat

    def forward(self, batched_input, multimask_output, gt_mask=None):
        out = super().forward(batched_input, multimask_output, gt_mask)
        if not self.class_token_decoder_enable:
            return out
        m_output, m_feat = out[0], out[1]
        cls_logits = self.class_decoder(m_feat)                    # (B, num_classes, hf, wf)
        cls_logits = F.interpolate(cls_logits, size=m_output.shape[-2:],
                                   mode='bilinear', align_corners=False)
        return (cls_logits, m_feat) + tuple(out[2:])


class LoRA_Sam_P30_Det(LoRA_Sam_P30):
    """
    LoRA_Sam_P30_Det — P30 segmentation backbone repurposed for object detection.

    계보: `LoRA_Sam_P30_Det(LoRA_Sam_P30)` — RBMA(P27/P28 memory-attn logit bias) +
    SDC(P29 self-derived FiLM gate) + P30의 두 기구(class-token decoder / reliability-
    anchored router)를 **그대로 상속**. P30 seg 의 두 노벨티는 detection 헤드 쪽으로
    번역되며(아래), 백본의 seg-level class_token_decoder / learned_router 는 기본 OFF
    (detection 은 backbone 의 fused seg 출력을 쓰지 않고 detection feature 경로만 구동).

    detection feature 경로는 P27 에서 상속한 `extract_det_features()`:
      fpn0 (32,s4) · fpn1 (64,s8) · mem (256,s16, memory+RBMA) · output (per-modal seg
      logits → training-free reliability 1-H/logC). downstream
      `objdet.models.det_model.MemorySAMDetectorP30` 가 이를 소비:
        ① ReliabilityAnchoredRouter 로 per-modality FPN 융합 (mean 대신 — P30 ② 이식)
        ② Object-Query Decoder 가 융합된 mem(memory-conditioned)에 cross-attend (P30 ① 이식)
        + FCOS dense head (aux, 조기 수렴 안정화)

    백본 fine-tune(indoor domain shift): SAM2 base 는 freeze, LoRA/SQG/per-modal decoder/
    RBMA λ/SDC 는 학습, detection model 이 `sam.memory_attention` 도 unfreeze 가능
    (MemorySAMDetectorP30 `train_memory`).
    """
    pass


class LoRA_Sam_P29_Det(LoRA_Sam_P28):
    """
    LoRA_Sam_P29_Det — P28/RBMA backbone repurposed for object detection (mean-fusion
    baseline). Identical to P28 (SAM2 Hiera-B+ + SoftMoE-LoRA + cross-modal memory
    attention with RBMA logit bias); the only addition is the detection-feature path
    `extract_det_features()` inherited from P27. Used as SEG_MODEL for the P29-Det
    detector (objdet.models.det_model.MemorySAMDetector). Mirrors the same class on the
    worktree-p29-det branch so P29-Det configs resolve on this branch too.
    """
    pass


class LoRA_Sam_P31(LoRA_Sam_P30):
    """
    LoRA_Sam_P31: Calibrated Dual-Reliability RBMA + Multi-scale Class-Token Decoding
    (doc 20 P31-Seg core; grounded in the doc 16 §7 module diagnostics).

    확정 진단 (P28/P29 실측):
      - reliability AUROC [img .77, depth .62, event .30, lidar .22] → event/LiDAR는
        anti-calibrated (틀린 곳에서 과확신) → RBMA bias 신호가 geometry 모달에서 무의미.
      - m_feat 단일 저해상(32ch, stride4) 질의 → thin-class(Bridge/Water/Wall) 경계 muffle.
      - AMF uniform → 죽은 event/lidar에 ~45% 질량 낭비.

    네 가지 config-gated 기구 (전부 OFF → P30과 byte-identical):

    ① [Seg-A] RBMA reliability 재보정 (rbma_calibrate):
       - per-modal learnable temperature T_i (rbma_log_temp): reliability = 1 − H(softmax
         (D_i(f_i)/T_i))/logC. _compute_bias_source에 적용 (여전히 training-free 신호;
         T는 스칼라 보정자일 뿐 학습형 quality head가 아님).
       - correctness-contrastive calibration loss (training): 틀린 픽셀 entropy↑ + 맞은
         픽셀 entropy↓ → reliability의 정답 AUROC를 직접 최적화. gate_loss_data
         ['rbma_cal_loss']로 노출 (trainer가 RBMA_CALIB.LAMBDA로 가산). GT는 학습 시에만
         사용 — 추론 시 bias는 여전히 GT-free.
    ② [Seg-B] Consistency 2차 bias (consistency_bias, A 성공 후 조건부 ON):
       Attention = softmax(QKᵀ/√d + λ_ent·B_ent + λ_ent·λ_cons·B_cons).
       B_cons_i = mean_{j≠i} Bhattacharyya(p_i, p_j) — cross-modal 일치도의 training-free
       2번째 additive 항 (RSGMamba의 learned gate와 차별: training-free + pre-softmax
       additive). λ_cons는 학습 스칼라 (hook의 λ_ent=lambda_bias가 전체에 곱해지므로
       유효 계수 = λ_ent·λ_cons — 2-dof로 동일 공간 커버).
    ③ [Seg-A2] Reliability-proportional AMF (amf_reliability; learned_router OFF일 때만):
       출력 융합 가중치 = softmax_modality(reliability/τ). "보정 후에만 전환" 게이트
       (doc 16 §7-2: 미보정 상태에서 비례가중은 악화 위험 → 기본 OFF).
    ④ [Seg-C] Multi-scale HR class-token decoder (ctd_multi_scale):
       self.class_decoder를 ClassTokenDecoderMS로 교체 — simple-FPN {4,8,16,32} 피라미드
       + coarse→fine cross-attend + 학습형 ConvTranspose(×up) 고해상 pixel-embed +
       training-only aux CE head @H/4 (gate_loss_data['ctd_aux_ce']). P30.forward의
       class_decoder(m_feat)+interpolate가 고해상 mask를 투명 처리 (forward 미수정).
       [P31.1] ctd_aux_only=True → CTD 강등: P30의 "cls_logits가 최종 출력 대체"를
       우회하고 최종 출력은 SAM decoder 융합(m_output) 유지, CTD는 학습 시 aux CE
       (gate_loss_data['ctd_seg_ce'])로만 기여(추론 경로 제거). 근거 = P30-seg 실측
       붕괴(Day-Val 49.76/Test 44.10, P29 대비 −13.4/−10.2) + det E0.1(query head 유죄).
       [P31.1 로깅] _calibration_loss가 per-modal reliability AUROC/μ/σ를 stash
       (_last_rel_auroc/_last_rel_stats) → trainer가 epoch마다 tb/wandb 기록
       (doc 20 "AUROC>0.5 선행 게이트"의 측정 구현; σ→0 = 엔트로피 상수화 퇴화 감지).

    학습 레버 (orthogonal, doc 16 ISSUE-008 — frozen-backbone ceiling의 유일한 지렛대):
    ⑤ unfreeze_last_n_blocks: Hiera trunk 마지막 N개 block unfreeze (Bridge/Other
       structural-dead class 직격). optimizer는 UNFREEZE_LR_SCALE로 감쇠 LR 적용.
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None,
                 num_experts=4, num_modalities=3,
                 quality_hidden_dim=64, quality_min=0.3,
                 tau_uamm=1.0, tau_teacher=0.5,
                 memory_mod=False, amf_mode='uniform',
                 multi_scale_sqg=True, per_modality_decoder=True,
                 cond_dim=8, lambda_bias_init=1.0,
                 sdc_enable=True, sdc_K=6, sdc_latent=32, sdc_in_channels=3,
                 class_token_decoder=False, ctd_dim=128,
                 learned_router=False, router_per_class=False, router_anchor_lambda=1.0,
                 num_classes=25,
                 rbma_calibrate=False, consistency_bias=False, lambda_cons_init=0.5,
                 amf_reliability=False, amf_rel_tau=0.25,
                 ctd_multi_scale=False, ctd_up=2, ctd_aux_ce=True, ctd_aux_only=False,
                 unfreeze_last_n_blocks=0, router_reg_mode='diversity'):
        super().__init__(
            sam_model=sam_model, r=r, lora_layer=lora_layer,
            num_experts=num_experts, num_modalities=num_modalities,
            quality_hidden_dim=quality_hidden_dim, quality_min=quality_min,
            tau_uamm=tau_uamm, tau_teacher=tau_teacher,
            memory_mod=memory_mod, amf_mode=amf_mode,
            multi_scale_sqg=multi_scale_sqg, per_modality_decoder=per_modality_decoder,
            cond_dim=cond_dim, lambda_bias_init=lambda_bias_init,
            sdc_enable=sdc_enable, sdc_K=sdc_K, sdc_latent=sdc_latent,
            sdc_in_channels=sdc_in_channels,
            class_token_decoder=class_token_decoder, ctd_dim=ctd_dim,
            learned_router=learned_router, router_per_class=router_per_class,
            router_anchor_lambda=router_anchor_lambda, num_classes=num_classes,
        )
        self.rbma_calibrate = rbma_calibrate
        self.consistency_bias = consistency_bias
        self.amf_reliability = amf_reliability
        self.amf_rel_tau = amf_rel_tau
        self.ctd_multi_scale = ctd_multi_scale
        self.ctd_aux_only = ctd_aux_only
        self._rbma_rel = None            # uncentered reliability stash (m,B,1,hf,wf)
        self._p31_aux_stash = None       # grad-attached per-modal aux logits (training)
        self._last_rel_auroc = None      # [P31 logging] per-modal reliability AUROC (list of m)
        self._last_rel_stats = None      # [P31 logging] (means, stds) of reliability per modality
        if rbma_calibrate:
            self.rbma_log_temp = nn.Parameter(torch.zeros(num_modalities))
        if consistency_bias:
            self.lambda_cons = nn.Parameter(torch.tensor(float(lambda_cons_init)))
        if class_token_decoder and ctd_multi_scale:
            # ④ drop-in swap (P30.forward 호출부는 동일 (feat) 시그니처를 그대로 사용)
            self.class_decoder = ClassTokenDecoderMS(
                32, num_classes, dim=ctd_dim, up=ctd_up, aux_ce=ctd_aux_ce)
        if learned_router and router_reg_mode != 'diversity':
            # ⑥ router 비적응(uniform) 해소: doc 16 §7 — 측정된 융합 가중치가 거의
            # uniform([.27,.28,.23,.23])인데 기존 'diversity' reg(entropy 보상)는 uniform
            # 방향으로 미는 모순 → 'decisive' = per-pixel commit + batch-marginal 다양성.
            self.router.reg_mode = router_reg_mode
        if unfreeze_last_n_blocks > 0:
            # ⑤ P26.__init__이 image_encoder 전체를 freeze한 것을 마지막 N block만 해제.
            # LoRA qkv wrapper 내부의 base linear도 함께 풀림 (의도된 backbone unfreeze).
            blocks = self.sam.image_encoder.trunk.blocks
            for blk in blocks[max(0, len(blocks) - unfreeze_last_n_blocks):]:
                for p_ in blk.parameters():
                    p_.requires_grad = True

    # ── ① aux-decode stash: Phase 2.5의 grad-attached per-modal logits를 붙잡아
    #    calibration loss에 재사용 (재디코딩 없음). _compute_bias_source의 no_grad
    #    호출은 torch.is_grad_enabled()==False라 stash되지 않음.
    def _auxiliary_decode_single(self, decoder, vision_feats, vision_pos_embeds, feat_sizes):
        out = super()._auxiliary_decode_single(decoder, vision_feats, vision_pos_embeds, feat_sizes)
        if self._p31_aux_stash is not None and torch.is_grad_enabled():
            self._p31_aux_stash.append(out)
        return out

    # ── ①/② bias 소스: temperature-보정 reliability (+ 선택적 consistency 2차 항)
    def _compute_bias_source(self, quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m):
        if not (self.rbma_calibrate or self.consistency_bias or self.amf_reliability):
            return super()._compute_bias_source(
                quality_logits, vision_feats, vision_pos_embeds, feat_sizes, m)
        fpn_h, fpn_w = quality_logits[0].shape[-2:]
        rel_maps, prob_maps = [], []
        T = (self.rbma_log_temp.exp().clamp(min=0.05, max=20.0).detach()
             if self.rbma_calibrate else None)
        for i in range(m):
            with torch.no_grad():
                aux_logits = self._auxiliary_decode_single(
                    self.per_modal_decoders[i],
                    vision_feats[i], vision_pos_embeds[i], feat_sizes[i],
                )  # (B, C, H, W)
                C = aux_logits.shape[1]
                lg = aux_logits.float()
                if T is not None:
                    lg = lg / T[i]
                p = F.softmax(lg, dim=1)
                ent = -(p * torch.log(p + self.RBMA_EPS)).sum(dim=1, keepdim=True)
                rel = 1.0 - ent / math.log(C)                 # (B,1,H,W) native res (P28 순서 유지)
                rel = F.interpolate(rel, size=(fpn_h, fpn_w),
                                    mode='bilinear', align_corners=False)
                rel_maps.append(rel)
                if self.consistency_bias:
                    p_f = F.interpolate(p, size=(fpn_h, fpn_w),
                                        mode='bilinear', align_corners=False)
                    prob_maps.append(p_f / p_f.sum(dim=1, keepdim=True).clamp(min=1e-6))
        rel_stack = torch.stack(rel_maps, dim=0)              # (m,B,1,hf,wf)
        self._rbma_rel = rel_stack                            # uncentered stash (③ AMF용)
        bias = rel_stack - rel_stack.mean(dim=0, keepdim=True)
        if self.consistency_bias:
            with torch.no_grad():
                cons_maps = []
                for i in range(m):
                    agree = [(prob_maps[i] * prob_maps[j]).clamp_min(0).sqrt()
                             .sum(dim=1, keepdim=True)
                             for j in range(m) if j != i]     # Bhattacharyya coeff ∈ [0,1]
                    cons_maps.append(torch.stack(agree, dim=0).mean(dim=0))
                cons_stack = torch.stack(cons_maps, dim=0)
                cons_stack = cons_stack - cons_stack.mean(dim=0, keepdim=True)
            # λ_cons 곱은 no_grad 밖 → attention을 통해 λ_cons에 grad 전달
            bias = bias + self.lambda_cons * cons_stack
        return [bias[i] for i in range(m)]

    # ── ③ reliability-proportional AMF (learned_router가 우선; doc 20 A "보정 후 전환")
    def _fuse_outputs(self, output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes):
        if (self.learned_router_enable or not self.amf_reliability
                or self._rbma_rel is None):
            return super()._fuse_outputs(
                output, all_backbone_feats, q_uamm_norm, amf_norm, m, num_classes)
        rel = self._rbma_rel                                  # (m,B,1,hf,wf) no-grad
        w = F.softmax(rel / self.amf_rel_tau, dim=0)
        Bn = rel.shape[1]; hf, wf = rel.shape[-2:]
        Hh, Ww = output[0].shape[-2:]
        w_out = F.interpolate(w.reshape(m * Bn, 1, hf, wf), size=(Hh, Ww),
                              mode='bilinear', align_corners=False).reshape(m, Bn, 1, Hh, Ww)
        w_out = w_out / w_out.sum(dim=0, keepdim=True).clamp(min=1e-6)
        m_output = sum(w_out[i] * output[i] for i in range(m))
        m_feat = sum(q_uamm_norm[i] * all_backbone_feats[i] for i in range(m))
        return m_output, m_feat

    # ── ① correctness-contrastive calibration loss (reliability AUROC 직접 최적화).
    #    1/4 해상도에서 계산 (reliability 소비 지점 = fpn 해상도; 메모리 16× 절약).
    def _calibration_loss(self, aux_logits_list, gt_mask):
        gt = gt_mask.long()
        Ht, Wt = max(1, gt.shape[-2] // 4), max(1, gt.shape[-1] // 4)
        gt_ds = F.interpolate(gt.unsqueeze(1).float(), size=(Ht, Wt),
                              mode='nearest').squeeze(1).long()
        valid = gt_ds != 255
        T = self.rbma_log_temp.exp().clamp(min=0.05, max=20.0)
        total = None
        aurocs, rel_mu, rel_sd = [], [], []
        for i, logits in enumerate(aux_logits_list):
            C = logits.shape[1]
            lg = F.interpolate(logits.float(), size=(Ht, Wt),
                               mode='bilinear', align_corners=False) / T[i]
            p = F.softmax(lg, dim=1)
            ent = -(p * (p + 1e-8).log()).sum(dim=1) / math.log(C)   # (B,Ht,Wt) ∈ [0,1]
            pred = lg.argmax(dim=1)
            wrong = (pred != gt_ds) & valid
            correct = (pred == gt_ds) & valid
            l_wrong = ((1.0 - ent) * wrong.float()).sum() / wrong.float().sum().clamp(min=1.0)
            l_correct = (ent * correct.float()).sum() / correct.float().sum().clamp(min=1.0)
            li = l_wrong + l_correct
            total = li if total is None else total + li
            # [P31 logging] doc 20 "AUROC>0.5 선행 게이트"의 측정: reliability(1−H)가
            # 정답을 rank하는지(Mann-Whitney AUROC) + 퇴화 해(엔트로피 상수화) 감지용 μ/σ.
            with torch.no_grad():
                score = (1.0 - ent).detach()[valid].float()
                lab = correct[valid]
                n1 = lab.sum().float(); n0 = lab.numel() - lab.sum().float()
                if n1 > 0 and n0 > 0:
                    order = score.argsort()
                    ranks = torch.zeros_like(score)
                    ranks[order] = torch.arange(1, score.numel() + 1,
                                                device=score.device, dtype=score.dtype)
                    auroc = ((ranks[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()
                else:
                    auroc = float('nan')
                aurocs.append(auroc)
                rel_mu.append(score.mean().item())
                rel_sd.append(score.std().item())
        self._last_rel_auroc = aurocs
        self._last_rel_stats = (rel_mu, rel_sd)
        return total / len(aux_logits_list)

    def forward(self, batched_input, multimask_output, gt_mask=None):
        self._rbma_rel = None
        self._p31_aux_stash = ([] if (self.training and gt_mask is not None
                                      and self.rbma_calibrate) else None)
        # [P31.1] CTD 강등(ctd_aux_only): P30의 "cls_logits가 최종 출력을 대체" 경로를 우회.
        # 근거: P30-seg 실측 붕괴(Day-Val 49.8/Test 44.1 = P29 대비 −13.4/−10.2) + det
        # E0.1(같은 ckpt에서 query head 0.256 vs FCOS aux 0.431 = head 단독 유죄) →
        # 경량 query decoder가 SAM decoder 출력을 대체하는 구조가 주범 후보.
        # aux-only에서 최종 출력 = SAM decoder 융합(m_output), CTD는 학습 시
        # auxiliary CE(gate_loss_data['ctd_seg_ce'])로만 rare-class gradient를 공급
        # (추론 경로에서 완전 제거 — GOOSE-M2F류 training-only head와 동일 지위).
        bypass = self.ctd_aux_only and self.class_token_decoder_enable
        if bypass:
            self.class_token_decoder_enable = False
        try:
            out = super().forward(batched_input, multimask_output, gt_mask)
        finally:
            if bypass:
                self.class_token_decoder_enable = True
        # 학습 시 반환 = (cls_logits|m_output, m_feat, gate_loss_data)
        if self.training and len(out) == 3 and isinstance(out[2], dict):
            gate_loss_data = out[2]
            if self.rbma_calibrate and self._p31_aux_stash:
                gate_loss_data['rbma_cal_loss'] = self._calibration_loss(
                    self._p31_aux_stash, gt_mask)
            if bypass and gt_mask is not None:
                cls_logits = self.class_decoder(out[1])          # grad-attached m_feat
                cls_logits = F.interpolate(cls_logits.float(), size=gt_mask.shape[-2:],
                                           mode='bilinear', align_corners=False)
                gate_loss_data['ctd_seg_ce'] = F.cross_entropy(
                    cls_logits, gt_mask.long(), ignore_index=255)
            aux_logits = getattr(self.class_decoder, 'last_aux_logits', None) \
                if self.class_token_decoder_enable else None
            if self.ctd_multi_scale and aux_logits is not None and gt_mask is not None:
                al = F.interpolate(aux_logits.float(), size=gt_mask.shape[-2:],
                                   mode='bilinear', align_corners=False)
                gate_loss_data['ctd_aux_ce'] = F.cross_entropy(
                    al, gt_mask.long(), ignore_index=255)
        self._p31_aux_stash = None
        return out


class LoRA_Sam_P31_Det(LoRA_Sam_P31):
    """
    LoRA_Sam_P31_Det — P31.1 backbone repurposed for object detection.

    계보: `LoRA_Sam_P31_Det(LoRA_Sam_P31)` — RBMA + P31.1의 calibrated reliability
    (per-modal learnable temperature `rbma_log_temp`) + decisive router + backbone
    부분 unfreeze를 그대로 상속. detection feature 경로는 P27에서 상속한
    `extract_det_features()` (fpn0/fpn1/mem + per-modal seg output). downstream
    `MemorySAMDetectorP30`(primary_head='fcos', use_calibrated_reliability=True,
    router_reg_mode='decisive')가 이를 소비한다.

    **calibrated reliability**: detector는 `self.rbma_log_temp`(있으면)를 읽어
    reliability = 1 − H(softmax(seg_output_i / T_i))/logC 로 계산 (P30-Det의 raw
    1−H/logC 대비 event/lidar anti-calibration 수리). T_i는 P31.1 seg 학습에서
    보정되므로 **P31.1 seg 체크포인트 warm-start**(SEG_CHECKPOINT)가 권장된다.

    P31.1 교훈 반영: object-query decoder를 최종 출력으로 쓰지 않음(FCOS primary).
    calibration/CTD aux loss는 seg-GT 의존이라 detection-only 학습에선 비활성
    (backbone은 seg warm-start로 보정 흡수).
    """
    pass
