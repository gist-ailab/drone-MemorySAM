"""RF-DETR NMS-free detection head, transplanted onto ReliaDINO fusion features.

The decoder under ``_vendor/`` is a vendored subset of roboflow/rf-detr (Apache-2.0,
see ``_vendor/LICENSE``). Only the transformer decoder + matcher/criterion/postprocess
are vendored; RF-DETR's own DINOv2 backbone is dropped because our ReliaDINO fusion
pyramid takes its place. That keeps the vendored tree free of transformers/timm.

Why this exists: our from-scratch FCOS head is the measured bottleneck against
COCO-pretrained YOLO on normal-light frames. This head is initialised from RF-DETR's
COCO weights and predicts sets directly, so no NMS is applied anywhere.

Geometry note: RF-DETR is natively single-scale at stride 16 (``num_feature_levels=1``),
and its two-stage proposal branch runs dense heads over *every* token, so token count
matters. Our stride-4 level alone is 36,864 tokens at 768px (25x RF-DETR's native
1,936) — feed ``P4_INDEX`` only unless you have measured a reason not to. The decoder's
initial box prior is ``0.05 * 2**level_index`` (transformer.py), which matches COCO
exactly only when the stride-16 feature sits at level 0.
"""

import math
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_vendor')
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from rfdetr.models.transformer import Transformer  # noqa: E402
from rfdetr.models.position_encoding import PositionEmbeddingSine  # noqa: E402
from rfdetr.models.math import MLP  # noqa: E402
from rfdetr.models.matcher import HungarianMatcher  # noqa: E402
from rfdetr.models.criterion import SetCriterion  # noqa: E402
from rfdetr.models.postprocess import PostProcess  # noqa: E402
from rfdetr.utilities.tensors import NestedTensor  # noqa: E402


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of an NCHW tensor (ConvNeXt-style).

    RF-DETR's projector ends in exactly this op, so the COCO decoder was trained on
    features with this normalisation applied. Our SimpleFPN emits un-normalised
    features, so we re-apply it at the interface and seed it from the checkpoint —
    without it the pretrained two-stage class head sees off-distribution activations
    and fires wildly (measured: enc logits up to +6.3 vs the decoder's max -0.8).
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.normalized_shape = (num_channels,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def _resize_linear(linear: nn.Linear, num_classes: int) -> nn.Linear:
    """Tile/truncate a Linear's rows to `num_classes` outputs (upstream lwdetr._resize_linear)."""
    base = linear.weight.shape[0]
    num_repeats = int(math.ceil(num_classes / base))
    new_weight = linear.weight.detach().repeat(num_repeats, 1)[:num_classes]
    new_bias = linear.bias.detach().repeat(num_repeats)[:num_classes] if linear.bias is not None else None
    new_linear = nn.Linear(linear.in_features, num_classes, bias=new_bias is not None)
    with torch.no_grad():
        new_linear.weight.copy_(new_weight)
        if new_bias is not None and new_linear.bias is not None:
            new_linear.bias.copy_(new_bias)
    return new_linear


def _expand_msdeform(tsd: Dict[str, torch.Tensor], n_levels: int, p4_index: int,
                     n_heads: int = 16, n_points: int = 2) -> Dict[str, torch.Tensor]:
    """Re-shape the checkpoint's single-level MSDeformAttn params to `n_levels`.

    The COCO weights only ever attend to one level. We replicate the sampling offsets
    across levels and drive the *other* levels' attention logits to -inf, so at init the
    cross-attention reproduces the single-scale COCO behaviour exactly; training is then
    free to open the extra levels up.
    """
    out = dict(tsd)
    for k, v in tsd.items():
        if k.endswith('cross_attn.sampling_offsets.weight'):
            w = v.view(n_heads, 1, n_points, 2, -1).repeat(1, n_levels, 1, 1, 1)
            out[k] = w.reshape(n_heads * n_levels * n_points * 2, -1).clone()
        elif k.endswith('cross_attn.sampling_offsets.bias'):
            b = v.view(n_heads, 1, n_points, 2).repeat(1, n_levels, 1, 1)
            out[k] = b.reshape(-1).clone()
        elif k.endswith('cross_attn.attention_weights.weight'):
            w = v.view(n_heads, 1, n_points, -1).repeat(1, n_levels, 1, 1).clone()
            for lvl in range(n_levels):
                if lvl != p4_index:
                    w[:, lvl] = 0.0
            out[k] = w.reshape(n_heads * n_levels * n_points, -1).clone()
        elif k.endswith('cross_attn.attention_weights.bias'):
            b = v.view(n_heads, 1, n_points).repeat(1, n_levels, 1).clone()
            for lvl in range(n_levels):
                if lvl != p4_index:
                    b[:, lvl] = -1e4
            out[k] = b.reshape(-1).clone()
    return out


class RFDETRHead(nn.Module):
    """RF-DETR decoder + set-prediction heads over an externally supplied feature pyramid.

    Mirrors ``LWDETR.forward`` minus backbone/keypoints/segmentation. The
    ``self.training`` gates upstream uses (all groups while training, one group at
    inference) are reproduced here — dropping them silently corrupts eval.
    """

    def __init__(
        self,
        n_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 300,
        group_detr: int = 13,
        dec_layers: int = 4,
        num_feature_levels: int = 1,
        dec_n_points: int = 2,
        sa_nhead: int = 8,
        ca_nhead: int = 16,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        two_stage: bool = True,
        aux_loss: bool = True,
        bbox_reparam: bool = True,
        lite_refpoint_refine: bool = True,
    ):
        super().__init__()
        # Upstream convention: `num_classes` is max_obj_id + 1, not the class count.
        # Our labels are dense 0..n_classes-1, so n_classes + 1 leaves one dead slot,
        # matching the shape the COCO weights were trained in.
        self.num_slots = n_classes + 1
        self.num_queries = num_queries
        self.group_detr = group_detr
        self.two_stage = two_stage
        self.aux_loss = aux_loss
        self.bbox_reparam = bbox_reparam
        self.num_feature_levels = num_feature_levels

        self.transformer = Transformer(
            d_model=hidden_dim, sa_nhead=sa_nhead, ca_nhead=ca_nhead,
            num_queries=num_queries, dropout=dropout, dim_feedforward=dim_feedforward,
            num_decoder_layers=dec_layers, return_intermediate_dec=True,
            group_detr=group_detr, two_stage=two_stage,
            num_feature_levels=num_feature_levels, dec_n_points=dec_n_points,
            lite_refpoint_refine=lite_refpoint_refine, decoder_norm_type='LN',
            bbox_reparam=bbox_reparam,
        )

        self.class_embed = nn.Linear(hidden_dim, self.num_slots)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.refpoint_embed = nn.Embedding(num_queries * group_detr, 4)
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        nn.init.constant_(self.refpoint_embed.weight.data, 0)

        # Attached by LWDETR.__init__ upstream, not by Transformer.__init__.
        if two_stage:
            self.transformer.enc_out_class_embed = nn.ModuleList(
                [nn.Linear(hidden_dim, self.num_slots) for _ in range(group_detr)])
            self.transformer.enc_out_bbox_embed = nn.ModuleList(
                [MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(group_detr)])

        self.transformer.decoder.bbox_embed = None if lite_refpoint_refine else self.bbox_embed

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_slots) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # Interface norm: stands in for the last op of RF-DETR's projector, so the
        # decoder receives the feature distribution its COCO weights were trained on.
        self.input_norm = nn.ModuleList(
            [ChannelLayerNorm(hidden_dim) for _ in range(num_feature_levels)])

        self.pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    # ------------------------------------------------------------------ weights
    def load_coco_pretrained(self, ckpt_path: str, p4_index: int = 0, verbose: bool = True) -> None:
        """Load RF-DETR COCO weights, resizing only the class heads to our label space."""
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = sd.get('model', sd)

        tsd = {k[len('transformer.'):]: v for k, v in sd.items() if k.startswith('transformer.')}
        if self.num_feature_levels > 1:
            tsd = _expand_msdeform(tsd, self.num_feature_levels, p4_index)

        # Class heads carry 91 COCO slots; every other tensor transfers as-is.
        named = dict(self.transformer.named_parameters())
        skipped = [k for k, v in tsd.items()
                   if k in named and tuple(named[k].shape) != tuple(v.shape)]
        tsd_load = {k: v for k, v in tsd.items() if k not in skipped}
        r = self.transformer.load_state_dict(tsd_load, strict=False)

        with torch.no_grad():
            src_cls = nn.Linear(self.class_embed.in_features, sd['class_embed.weight'].shape[0])
            src_cls.weight.copy_(sd['class_embed.weight'])
            src_cls.bias.copy_(sd['class_embed.bias'])
            resized = _resize_linear(src_cls, self.num_slots)
            self.class_embed.weight.copy_(resized.weight)
            # Tiling carries COCO's per-class bias over; refocus on our prior instead.
            self.class_embed.bias.fill_(-math.log((1 - 0.01) / 0.01))

            self.bbox_embed.load_state_dict(
                {k[len('bbox_embed.'):]: v for k, v in sd.items() if k.startswith('bbox_embed.')})

            # Seed the interface norm from the projector's final LayerNorm, which is
            # what fed the decoder during COCO training.
            proj_ln_w = sd.get('backbone.0.projector.stages.0.1.weight')
            proj_ln_b = sd.get('backbone.0.projector.stages.0.1.bias')
            if proj_ln_w is not None and proj_ln_b is not None:
                for ln in self.input_norm:
                    ln.weight.copy_(proj_ln_w)
                    ln.bias.copy_(proj_ln_b)

            n_q = self.refpoint_embed.weight.shape[0]
            self.refpoint_embed.weight.copy_(sd['refpoint_embed.weight'][:n_q])
            self.query_feat.weight.copy_(sd['query_feat.weight'][:n_q])

            if self.two_stage:
                for g in range(self.group_detr):
                    src = nn.Linear(self.class_embed.in_features,
                                    sd[f'transformer.enc_out_class_embed.{g}.weight'].shape[0])
                    src.weight.copy_(sd[f'transformer.enc_out_class_embed.{g}.weight'])
                    src.bias.copy_(sd[f'transformer.enc_out_class_embed.{g}.bias'])
                    rz = _resize_linear(src, self.num_slots)
                    self.transformer.enc_out_class_embed[g].weight.copy_(rz.weight)
                    self.transformer.enc_out_class_embed[g].bias.fill_(-math.log((1 - 0.01) / 0.01))

        if verbose:
            print(f"[RFDETRHead] COCO init from {os.path.basename(ckpt_path)}: "
                  f"transformer loaded={len(tsd_load)}/{len(tsd)} "
                  f"missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)} "
                  f"| class heads resized 91 -> {self.num_slots}")

    # ------------------------------------------------------------------ forward
    def forward(self, srcs: List[torch.Tensor]) -> Dict[str, object]:
        srcs = [ln(s) for ln, s in zip(self.input_norm, srcs)]
        masks = [torch.zeros(s.shape[0], s.shape[-2], s.shape[-1],
                             dtype=torch.bool, device=s.device) for s in srcs]
        poss = [self.pos_enc(NestedTensor(s, m), align_dim_orders=False)
                for s, m in zip(srcs, masks)]

        # Upstream gate: all query groups while training, a single group at inference.
        if self.training:
            refpoint_w = self.refpoint_embed.weight
            query_w = self.query_feat.weight
        else:
            refpoint_w = self.refpoint_embed.weight[:self.num_queries]
            query_w = self.query_feat.weight[:self.num_queries]

        hs, ref_unsig, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_w, query_w)[:4]

        if self.bbox_reparam:
            delta = self.bbox_embed(hs)
            coord = torch.cat([delta[..., :2] * ref_unsig[..., 2:] + ref_unsig[..., :2],
                               delta[..., 2:].exp() * ref_unsig[..., 2:]], dim=-1)
        else:
            coord = (self.bbox_embed(hs) + ref_unsig).sigmoid()
        cls = self.class_embed(hs)

        out: Dict[str, object] = {'pred_logits': cls[-1], 'pred_boxes': coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                  for a, b in zip(cls[:-1], coord[:-1])]

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = torch.cat(
                [self.transformer.enc_out_class_embed[g](hs_enc_list[g]) for g in range(group_detr)],
                dim=1)
            out['enc_outputs'] = {'pred_logits': cls_enc, 'pred_boxes': ref_enc}
        return out


def build_rfdetr_criterion(n_classes: int, group_detr: int = 13, dec_layers: int = 4,
                           two_stage: bool = True, aux_loss: bool = True) -> SetCriterion:
    """SetCriterion with RF-DETR's shipped coefficients (IoU-aware BCE + Hungarian o2o)."""
    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal_alpha=0.25)
    weight_dict = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
    if aux_loss:
        aux = {}
        for i in range(dec_layers - 1):
            aux.update({f'{k}_{i}': v for k, v in weight_dict.items()})
        if two_stage:
            aux.update({f'{k}_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux)
    return SetCriterion(
        n_classes + 1, matcher=matcher, weight_dict=weight_dict, focal_alpha=0.25,
        losses=['labels', 'boxes', 'cardinality'], group_detr=group_detr, ia_bce_loss=True,
    )


def build_rfdetr_postprocessor(num_select: int = 300) -> PostProcess:
    """Top-k over (query x class) — RF-DETR ships no NMS anywhere."""
    return PostProcess(num_select=num_select)
