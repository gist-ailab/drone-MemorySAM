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
from sam_lora_image_encoder_seg_bkup import LoRA_Sam 


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


class MLP_my(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    
class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
        
def random_element_swap(tensor_list):
    if len(tensor_list) != 2:
        raise ValueError("列表必须包含两个张量")

    tensor1, tensor2 = tensor_list

    if tensor1.size() != tensor2.size():
        raise ValueError("两个张量的大小必须相同")

        # 生成与张量大小相同的随机布尔掩码
    swap_mask = torch.rand(tensor1.size()) > 0.5
        # 使用掩码进行元素交换
    temp = tensor1.clone()
    tensor1[swap_mask] = tensor2[swap_mask]
    tensor2[swap_mask] = temp[swap_mask]

    return [tensor1, tensor2]


class MoE_LoRA_Layer(nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) LoRA layer.
    Instead of one pair of A/B matrices, it holds 'num_experts' pairs.
    A gating network selects top-k experts for each token.
    """
    def __init__(self, in_features, rank, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.rank = rank
        self.in_features = in_features
        
        # Gating Network: Linear projection to expert scores
        self.gate = nn.Linear(in_features, num_experts)
        
        # Experts: ModuleList of LoRA Adapters
        # Expert A: in -> rank
        self.experts_a = nn.ModuleList([
            nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)
        ])
        # Expert B: rank -> in
        self.experts_b = nn.ModuleList([
            nn.Linear(rank, in_features, bias=False) for _ in range(num_experts)
        ])
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize Gate: Near zero to start with uniform routing
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)
        
        # Initialize Experts
        for i in range(self.num_experts):
            # A: Kaiming Uniform
            nn.init.kaiming_uniform_(self.experts_a[i].weight, a=math.sqrt(5))
            # B: Zero init to ensure identity function at start
            nn.init.zeros_(self.experts_b[i].weight)

    def forward(self, x):
        # x shape: (B, N, C)
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features) # (B*N, C)
        
        # 1. Calculate Routing Logits
        gate_logits = self.gate(x_flat) # (B*N, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # 2. Select Top-K Experts
        # weights: (B*N, k), indices: (B*N, k)
        weights, indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Normalize weights so they sum to 1
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 3. Compute Expert Outputs
        # Optimization: We compute output as weighted sum of selected experts.
        # Since rank is small, we can iterate experts or use efficient gathering.
        # Here we iterate for clarity and stability.
        
        final_output = torch.zeros_like(x_flat)
        
        # Create a mask for efficient computation/addition
        # mask: (B*N, num_experts)
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(1, indices, 1.0)
        
        # Apply mask to probabilities to zero out non-top-k
        masked_probs = gate_probs * mask
        # Re-normalize over the top-k selection
        masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

        for i in range(self.num_experts):
            # Check if this expert contributes to any token (Optimization)
            expert_weight = masked_probs[:, i].unsqueeze(-1) # (B*N, 1)
            if expert_weight.sum() == 0:
                continue
            
            # Compute expert output: B(A(x))
            # Note: We compute for all tokens. For very large models, we would only compute for selected tokens.
            # But for LoRA (low rank), computing all is often faster than gather/scatter overhead in PyTorch.
            expert_out = self.experts_b[i](self.experts_a[i](x_flat))
            
            final_output += expert_weight * expert_out
            
        return final_output.view(original_shape)

class _MoE_LoRA_qkv(nn.Module):
    """
    QKV layer wrapper that replaces standard LoRA with MoE-LoRA.
    """
    def __init__(
            self,
            qkv: nn.Module,
            moe_layer_q: MoE_LoRA_Layer,
            moe_layer_v: MoE_LoRA_Layer,
    ):
        super().__init__()
        self.qkv = qkv
        self.moe_layer_q = moe_layer_q
        self.moe_layer_v = moe_layer_v
        self.dim = qkv.in_features
        
    def forward(self, x):
        # Original QKV
        qkv = self.qkv(x)  # B, N, 3*dim
        
        # MoE-LoRA Update for Query
        new_q = self.moe_layer_q(x)
        
        # MoE-LoRA Update for Value
        new_v = self.moe_layer_v(x)
        
        # Add residual connection
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        
        return qkv


class ConfidenceHead(nn.Module):
    """
    Lightweight module to estimate the confidence of a modality based on its features.
    Architecture: GlobalAvgPool -> MLP -> Linear (Logits for Softmax)
    """
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

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
        # Transformer dimension (일반적으로 256)을 가져옵니다.
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
    LoRA_Sam_P3: The Ultimate Architecture.
    Combines:
    1. Adaptive Output Fusion (Inherited from P1)
    2. Quality-Aware Memory Update (QAMU) (Inherited from P2)
    3. Mixture-of-Experts (MoE) LoRA (New in P3)
    
    This class overrides __init__ to inject MoE layers instead of standard LoRA.
    """

    def __init__(self, sam_model: SAM2Base, r: int, lora_layer=None, qamu_threshold=0.4, num_experts=4, top_k=2):
        # We cannot call super().__init__ easily because we want to inject different layers.
        # So we reconstruct the initialization logic here.
        nn.Module.__init__(self) # Base Module init

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

        # MoE-LoRA Surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks):
            if t_layer_i not in self.lora_layer:
                continue
                
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            
            # Create MoE Layers for Q and V
            moe_q = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            moe_v = MoE_LoRA_Layer(dim, r, num_experts=num_experts, top_k=top_k)
            
            self.moe_layers_q.append(moe_q)
            self.moe_layers_v.append(moe_v)
            
            # Replace QKV
            blk.attn.qkv = _MoE_LoRA_qkv(
                w_qkv_linear,
                moe_q,
                moe_v,
            )
            
        self.sam = sam_model
        
        # P1/P2 Features: Adaptive Fusion & QAMU
        fusion_dim = self.sam.sam_mask_decoder.transformer_dim
        self.confidence_head = ConfidenceHead(in_channels=fusion_dim)
        self.qamu_threshold = qamu_threshold

    def save_lora_parameters(self, filename: str) -> None:
        """
        Custom save function for MoE structure.
        """
        assert filename.endswith(".pt") or filename.endswith('.pth')

        # Collect MoE parameters
        moe_params = {}
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            moe_params[f"moe_q_{i:03d}"] = mq.state_dict()
            moe_params[f"moe_v_{i:03d}"] = mv.state_dict()
            
        # Collect Confidence Head
        confidence_tensors = {f"confidence_head.{k}": v for k, v in self.confidence_head.state_dict().items()}

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        
        # Handle DDP wrapping
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
        """
        Custom load function for MoE structure.
        """
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)

        # Load MoE Layers
        for i, (mq, mv) in enumerate(zip(self.moe_layers_q, self.moe_layers_v)):
            q_key = f"moe_q_{i:03d}"
            v_key = f"moe_v_{i:03d}"
            
            if q_key in state_dict:
                mq.load_state_dict(state_dict[q_key])
            if v_key in state_dict:
                mv.load_state_dict(state_dict[v_key])
        
        # Load Confidence Head
        confidence_dict = {}
        for k, v in state_dict.items():
            if k.startswith("confidence_head."):
                confidence_dict[k.replace("confidence_head.", "")] = v
        if confidence_dict:
            self.confidence_head.load_state_dict(confidence_dict)

        # Load SAM parts
        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        for module_name in ['prompt_encoder', 'mask_decoder']:
            module_keys = [k for k in sam_keys if module_name in k]
            module_values = [state_dict[k] for k in module_keys if k in state_dict]
            if len(module_keys) == len(module_values):
                module_new_state_dict = {k: v for k, v in zip(module_keys, module_values)}
                sam_dict.update(module_new_state_dict)
            
        self.sam.load_state_dict(sam_dict)