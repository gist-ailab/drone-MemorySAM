"""공통 소형 모듈: MLP, decoder, 텐서 유틸 (구 sam_lola_utils.py에서 verbatim 이동)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MLP_my', 'random_element_swap', 'ClassTokenDecoder', 'ClassTokenDecoderMS']

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


class ClassTokenDecoder(nn.Module):
    """[P30] Class-token decoder (faithful approximation of repurposing the SAM2 mask decoder
    to class tokens — ports the SAM3-RBMA class-collapse break). C learnable class queries
    cross-attend the fused cross-modal memory feature (m_feat, already carries all modalities
    + RBMA bias) → per-class masks via a light transformer-decoder block + dynamic-kernel
    dot-product. Gives thin/rare classes an active query mechanism instead of losing a
    per-pixel argmax to dominant classes.

    APPROXIMATION: a small MaskFormer/SAM-style decoder block, NOT surgery on the actual
    `sam_mask_decoder` weights (kept faithful to the mechanism; noted in docs)."""

    def __init__(self, feat_ch, num_classes, dim=128, heads=4, ffn=256):
        super().__init__()
        self.proj = nn.Conv2d(feat_ch, dim, 1)
        self.class_tokens = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, dim))
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim); self.n3 = nn.LayerNorm(dim)
        self.kernel = nn.Linear(dim, dim)
        self.pix = nn.Conv2d(dim, dim, 1)

    def forward(self, feat):
        # feat: (B, feat_ch, h, w) fused feature → (B, num_classes, h, w) class logits
        B, _, h, w = feat.shape
        f = self.proj(feat)                                  # (B, dim, h, w)
        mem = f.flatten(2).permute(0, 2, 1)                  # (B, hw, dim)
        q = self.class_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, C, dim)
        q = self.n1(q + self.self_attn(q, q, q)[0])
        q = self.n2(q + self.cross_attn(q, mem, mem)[0])
        q = self.n3(q + self.ffn(q))
        k = self.kernel(q)                                   # (B, C, dim)
        p = self.pix(f)                                      # (B, dim, h, w)
        masks = torch.einsum('bcd,bdhw->bchw', k, p)         # (B, C, h, w)
        return masks


class ClassTokenDecoderMS(ClassTokenDecoder):
    """[P31] Multi-scale high-res class-token decoder (doc 20 Seg-C).

    Extends the P30 ClassTokenDecoder along two axes identified in the P28/P29 failure
    analysis (doc 16: m_feat@32ch single low-res query → thin-class boundary muffle):

    ① simple-FPN pyramid (ViTDet recipe): the fused RBMA memory feature (stride-4) is
       expanded into a {4,8,16,32} pyramid with stride-2 convs; class tokens cross-attend
       every level coarse→fine (one attn+FFN block per level, learned scale embedding).
    ② learned-upsample pixel branch (ClassTokenDecoderHR recipe): the dynamic-kernel
       dot-product runs on a ConvTranspose2d(×up) high-res pixel embed instead of relying
       on the caller's bilinear upsample — recovers thin-class boundaries.
    ③ training-only auxiliary per-pixel CE head @H/4 (GOOSE-M2F recipe): a 1×1 conv on the
       stride-4 pixel embed, exposed as `self.last_aux_logits` (None at inference; the
       head adds no inference cost). Fixes thin-class gradient starvation.

    Drop-in replacement: same `forward(feat) -> (B, num_classes, h*up, w*up)` contract as
    the base class (the P30 caller interpolates to output size transparently).
    NOTE: the base class' single-scale `cross_attn`/`n2` remain unused here (DDP runs with
    find_unused_parameters=True); kept so base-class checkpoints partially load."""

    def __init__(self, feat_ch, num_classes, dim=128, heads=4, ffn=256,
                 up=2, num_scales=4, aux_ce=True):
        super().__init__(feat_ch, num_classes, dim=dim, heads=heads, ffn=ffn)
        self.up = up
        self.num_scales = num_scales
        # ① simple-FPN downsample chain: stride-4 → {8, 16, 32}
        self.downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1),
                          nn.GroupNorm(8, dim), nn.GELU())
            for _ in range(num_scales - 1)])
        self.scale_embed = nn.Parameter(torch.zeros(num_scales, dim))
        self.ms_cross = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(num_scales)])
        self.ms_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_scales)])
        self.ms_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))
            for _ in range(num_scales)])
        self.ms_ffn_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_scales)])
        # ② learned high-res pixel branch
        self.up_conv = nn.ConvTranspose2d(dim, dim, kernel_size=up, stride=up) if up > 1 else None
        # ③ training-only aux CE head @ stride-4
        self.aux_head = nn.Conv2d(dim, num_classes, 1) if aux_ce else None
        self.last_aux_logits = None

    def forward(self, feat):
        # feat: (B, feat_ch, h, w) fused stride-4 feature → (B, C, h*up, w*up) class logits
        B = feat.shape[0]
        f = self.proj(feat)                                   # (B, dim, h, w) stride-4 embed
        pyr = [f]
        for down in self.downs:
            pyr.append(down(pyr[-1]))                         # stride 8, 16, 32
        q = self.class_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, C, dim)
        q = self.n1(q + self.self_attn(q, q, q)[0])
        for s in range(self.num_scales):                      # coarse → fine
            lvl = self.num_scales - 1 - s
            mem = pyr[lvl].flatten(2).permute(0, 2, 1) + self.scale_embed[lvl]
            q = self.ms_norm[s](q + self.ms_cross[s](q, mem, mem)[0])
            q = self.ms_ffn_norm[s](q + self.ms_ffn[s](q))
        q = self.n3(q + self.ffn(q))
        k = self.kernel(q)                                    # (B, C, dim)
        p = self.pix(f)                                       # (B, dim, h, w)
        self.last_aux_logits = (self.aux_head(p)
                                if (self.aux_head is not None and self.training) else None)
        if self.up_conv is not None:
            p = self.up_conv(p)                               # (B, dim, h*up, w*up)
        masks = torch.einsum('bcd,bdhw->bchw', k, p)          # (B, C, h*up, w*up)
        return masks
