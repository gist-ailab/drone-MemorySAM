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
            self._last_uamm_scores = all_scores.detach().cpu().numpy()
            self._last_amf_weights = weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().cpu().numpy()  # (B, m)
            self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().cpu().numpy()  # (B, m)
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
            self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().cpu().numpy()
            self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().cpu().numpy()
            self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().cpu().numpy()
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
                self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().cpu().numpy()
                self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().cpu().numpy()
            else:
                self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
                self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.mean(dim=[2, 3]).detach().cpu().numpy()
            self._last_amf_weights = amf_weights.mean(dim=[2, 3]).detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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
            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
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

            self._last_uamm_scores = uamm_scores.detach().cpu().numpy()
            self._last_amf_weights = amf_weights.detach().cpu().numpy()
            self._last_quality_maps = [q.detach().cpu().numpy() for q in quality_maps]
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
                [q_amf_norm[i].mean(dim=[2, 3, 4]) for i in range(m)], dim=1  # q_amf_norm: (m, B, 1, H, W)
            )  # (B, m)
            # Renormalize to sum=1
            amf_log = amf_log / amf_log.sum(dim=1, keepdim=True).clamp(min=1e-6)

            self._last_uamm_scores = uamm_log.detach().cpu().numpy()
            self._last_amf_weights = amf_log.detach().cpu().numpy()
            self._last_quality_maps = [q.detach().cpu().numpy() for q in quality_maps]
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
