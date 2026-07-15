"""P37b — ClassToken-lite-Learned auxiliary head (Mask2Former-lite).

25 LEARNED class tokens (fixed assignment token k <-> class k — NO Hungarian
matching, per the in-house P30 collapse lesson; NO CLIP/text init — user
directive) run 3 layers of masked cross-attention over the GATED fused
stride-16 map, then dot-product mask embeddings against the FPN stride-4
feature (the `feat` FPNSegHead already returns) to produce per-class
`token_logits` (B, K, H/4, W/4).

Merge (in model.py, collapse-safe):
    final = conv_head_logits + beta * token_logits (+ router residual)
with `beta` a ZERO-INIT scalar -> at init the model is byte-identical to the
classtoken-off path (same recipe as router_alpha / the P10-P27 zero-init fix).

Aux supervision: training-only CE on token_logits at 1/4 label res, returned
by the model as aux['ctd_ce'] and weighted in the trainer by
MODEL.CLASS_TOKEN.AUX_CE_W (default 0.4) — this is what gives the tokens /
attention weights gradients while beta is still ~0.

Masking (Mask2Former-style, "lite"): layer 1 is plain cross-attention (no
usable mask exists yet from randomly-initialized tokens); layers 2-3 mask each
token's attention to the stride-16 map locations where its CURRENT mask
prediction (sigmoid of a dim_t-space dot product against the map tokens) is
< 0.5. Fully-masked rows are un-masked (standard Mask2Former NaN guard).

Params @ dim=1024 (ViT-L), dim_t=256, 3 layers, mlp_ratio 2.0: ~2.9M.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TokenDecoderLayer(nn.Module):
    """One pre-norm block: masked cross-attn (tokens -> map) + self-attn + FFN."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.num_heads = num_heads
        # cross-attention (queries = class tokens, keys/values = fused map)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cq = nn.Linear(dim, dim)
        self.ck = nn.Linear(dim, dim)
        self.cv = nn.Linear(dim, dim)
        self.cproj = nn.Linear(dim, dim)
        # self-attention among the K tokens
        self.norm_s = nn.LayerNorm(dim)
        self.sqkv = nn.Linear(dim, 3 * dim)
        self.sproj = nn.Linear(dim, dim)
        # FFN
        self.norm_f = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, src: torch.Tensor,
                attn_bias: torch.Tensor | None) -> torch.Tensor:
        """x: (B, K, C) tokens; src: (B, N, C) map tokens;
        attn_bias: (B, 1, K, N) additive pre-softmax mask (0 / -inf) or None."""
        B, K, C = x.shape
        h = self.num_heads
        # masked cross-attention
        q = self.cq(self.norm_q(x)).reshape(B, K, h, C // h).transpose(1, 2)
        kvn = self.norm_kv(src)
        k = self.ck(kvn).reshape(B, -1, h, C // h).transpose(1, 2)
        v = self.cv(kvn).reshape(B, -1, h, C // h).transpose(1, 2)
        mask = attn_bias.to(q.dtype) if attn_bias is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.cproj(out.transpose(1, 2).reshape(B, K, C))
        # self-attention
        qkv = self.sqkv(self.norm_s(x)).reshape(B, K, 3, h, C // h).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        x = x + self.sproj(out.transpose(1, 2).reshape(B, K, C))
        # FFN
        x = x + self.mlp(self.norm_f(x))
        return x


class ClassTokenLiteHead(nn.Module):
    """[P37b] Learned-class-token auxiliary decoder.

    forward(fused, feat_s4) -> token_logits (B, K, H/4, W/4)
      fused   : (B, dim, h, w)      gated fused stride-16 map (fusion output)
      feat_s4 : (B, fpn_dim, H/4, W/4)  FPNSegHead pre-classifier feature

    The residual scale `self.beta` (zero-init scalar) lives here; the model
    applies it: logits = head_logits + beta * token_logits.
    """

    def __init__(self, dim: int, fpn_dim: int, num_classes: int,
                 num_layers: int = 3, dim_t: int = 256, num_heads: int = 8,
                 mlp_ratio: float = 2.0, beta_init: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        # 25 learned class tokens — fixed assignment token k <-> class k.
        self.tokens = nn.Parameter(torch.empty(num_classes, dim_t))
        nn.init.normal_(self.tokens, std=0.02)
        self.in_proj = nn.Linear(dim, dim_t)          # fused map -> dim_t
        self.layers = nn.ModuleList(
            _TokenDecoderLayer(dim_t, num_heads, mlp_ratio) for _ in range(num_layers))
        # intermediate mask predictor (dim_t space, against the stride-16 map)
        self.mask_proj = nn.Linear(dim_t, dim_t)
        # final mask-embedding MLP (Mask2Former convention: 3-layer)
        self.norm_out = nn.LayerNorm(dim_t)
        self.mask_mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.GELU(),
            nn.Linear(dim_t, dim_t), nn.GELU(),
            nn.Linear(dim_t, fpn_dim))
        # zero-init residual scale -> byte-identical to classtoken-off at start.
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def _attn_bias(self, q: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Mask2Former-style attention mask from the CURRENT token state.
        q: (B, K, dim_t), src: (B, N, dim_t) -> (B, 1, K, N) additive bias."""
        mask_logits = torch.einsum('bkc,bnc->bkn', self.mask_proj(q), src)
        masked = mask_logits.sigmoid() < 0.5                  # True = don't attend
        # NaN guard: a token whose mask covers nothing attends everywhere.
        masked = masked & ~masked.all(dim=-1, keepdim=True)
        bias = torch.zeros_like(mask_logits)
        bias = bias.masked_fill(masked, float('-inf'))
        return bias.unsqueeze(1)                              # (B, 1, K, N)

    def forward(self, fused: torch.Tensor, feat_s4: torch.Tensor) -> torch.Tensor:
        B = fused.shape[0]
        src = self.in_proj(fused.flatten(2).transpose(1, 2))  # (B, N, dim_t)
        q = self.tokens.unsqueeze(0).expand(B, -1, -1)        # (B, K, dim_t)
        for li, layer in enumerate(self.layers):
            # layer 1: plain cross-attn (random tokens carry no usable mask yet);
            # layers 2..L: masked by the current per-token mask prediction.
            bias = self._attn_bias(q, src) if li > 0 else None
            q = layer(q, src, bias)
        mask_embed = self.mask_mlp(self.norm_out(q))          # (B, K, fpn_dim)
        token_logits = torch.einsum('bkc,bchw->bkhw', mask_embed, feat_s4)
        return token_logits
