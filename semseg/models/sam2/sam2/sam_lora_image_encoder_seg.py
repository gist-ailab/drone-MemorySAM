import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from safetensors import safe_open
from safetensors.torch import save_file

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
    ModalAuxHead,
    MoE_LoRA_Layer,
    _MoE_LoRA_qkv,
    SoftMoE_LoRA_Layer,
    _SoftMoE_LoRA_qkv,
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
        # Energy score: E(x) = -T * log(sum(exp(z_k / T)))
        energy = -temperature * torch.logsumexp(z / temperature, dim=1)  # (B, H_feat, W_feat)
        # 낮은 energy = 높은 confidence → conf_map이 클수록 confident
        conf_map = -energy  # (B, H_feat, W_feat)
        conf_maps.append(conf_map)

    stacked = torch.stack(conf_maps, dim=1)  # (B, num_modalities, H_feat, W_feat)
    weights = F.softmax(stacked / temperature, dim=1)  # (B, num_modalities, H_feat, W_feat)
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
