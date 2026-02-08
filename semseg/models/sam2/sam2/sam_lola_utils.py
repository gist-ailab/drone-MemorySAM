"""
LoRA/LoLA 공통 유틸: 실험 모델(LoRA_Sam_P1~P7)이 아닌 보조 클래스·함수.
- MLP_my, _LoRA_qkv, ConfidenceHead, ConfidenceHeadV2
- MoE_LoRA_Layer, _MoE_LoRA_qkv
- random_element_swap
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    swap_mask = torch.rand(tensor1.size()) > 0.5
    temp = tensor1.clone()
    tensor1[swap_mask] = tensor2[swap_mask]
    tensor2[swap_mask] = temp[swap_mask]

    return [tensor1, tensor2]


class ConfidenceHeadV2(nn.Module):
    """
    A deeper Confidence Head using CNN layers to capture spatial features
    before global pooling. This helps in detecting local noise (like rain drops)
    better than a simple MLP on avg-pooled features.
    """
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, x):
        return self.net(x)


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

        self.gate = nn.Linear(in_features, num_experts)
        self.experts_a = nn.ModuleList([
            nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)
        ])
        self.experts_b = nn.ModuleList([
            nn.Linear(rank, in_features, bias=False) for _ in range(num_experts)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.experts_a[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_b[i].weight)

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)

        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        weights, indices = torch.topk(gate_probs, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        final_output = torch.zeros_like(x_flat)
        mask = torch.zeros_like(gate_probs)
        mask.scatter_(1, indices, 1.0)
        masked_probs = gate_probs * mask
        masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

        for i in range(self.num_experts):
            expert_weight = masked_probs[:, i].unsqueeze(-1)
            if expert_weight.sum() == 0:
                continue
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
        qkv = self.qkv(x)
        new_q = self.moe_layer_q(x)
        new_v = self.moe_layer_v(x)
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
class SoftMoE_LoRA_Layer(nn.Module):
    """
    Soft-MoE (Soft Mixture-of-Experts) LoRA Layer.
    
    기존의 Top-k 방식(Hard Routing) 대신, 입력에 따라 모든 전문가(Expert)의 출력을
    Softmax 가중치로 섞어서 사용합니다.
    
    장점:
    1. 전문가 수가 적을 때(예: 4개) Top-k보다 정보 손실이 적음.
    2. 모든 전문가에게 Gradient가 흘러 Dead Expert 문제가 완화됨.
    3. 미분 가능성이 완벽하게 보장됨.
    """
    def __init__(self, in_features, rank, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank
        self.in_features = in_features
        
        # Gating Network: 입력 토큰별로 전문가 가중치를 계산
        self.gate = nn.Linear(in_features, num_experts)
        
        # Experts: LoRA 어댑터들
        self.experts_a = nn.ModuleList([
            nn.Linear(in_features, rank, bias=False) for _ in range(num_experts)
        ])
        self.experts_b = nn.ModuleList([
            nn.Linear(rank, in_features, bias=False) for _ in range(num_experts)
        ])
        
        self.reset_parameters()

    def reset_parameters(self):
        # Gate 초기화: 초기에는 모든 전문가를 균등하게 사용하도록 0 근처로 설정
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)
        
        # Expert 초기화
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.experts_a[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_b[i].weight)

    def forward(self, x):
        # x shape: (B, N_tokens, C)
        
        # 1. Routing Weights 계산 (Softmax over all experts)
        gate_logits = self.gate(x) # (B, N, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1) # (B, N, num_experts)
        
        # 2. 모든 Expert 연산 및 가중 합 (Weighted Sum)
        final_output = 0
        for i in range(self.num_experts):
            # Expert Output: B(A(x)) -> (B, N, C)
            expert_out = self.experts_b[i](self.experts_a[i](x))
            
            # Weighting: (B, N, 1) * (B, N, C)
            weight = gate_weights[:, :, i].unsqueeze(-1)
            final_output += weight * expert_out
            
        return final_output

class _SoftMoE_LoRA_qkv(nn.Module):
    """
    QKV layer wrapper that replaces standard LoRA with Soft-MoE LoRA.
    """
    def __init__(
            self,
            qkv: nn.Module,
            moe_layer_q: SoftMoE_LoRA_Layer,
            moe_layer_v: SoftMoE_LoRA_Layer,
    ):
        super().__init__()
        self.qkv = qkv
        self.moe_layer_q = moe_layer_q
        self.moe_layer_v = moe_layer_v
        self.dim = qkv.in_features
        
    def forward(self, x):
        # Original QKV
        qkv = self.qkv(x)
        
        # Soft-MoE LoRA Update for Query
        new_q = self.moe_layer_q(x)
        
        # Soft-MoE LoRA Update for Value
        new_v = self.moe_layer_v(x)
        
        # Add residual connection
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        
        return qkv
