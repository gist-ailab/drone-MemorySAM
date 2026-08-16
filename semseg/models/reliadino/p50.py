"""[P50-MAP] Modal Alignment Pretraining — 공용 부품.

정본 설계 = `.claude_logs/decisions/2026-08-17-p50-map-modal-alignment-pretraining-proposal.md`

여기 있는 것은 **사전학습 전용**이다. 추론 그래프는 한 글자도 바뀌지 않는다:
  - `ReconHead` 는 사전학습에서만 만들어지고, 산출 state_dict 에서 **제외**된다.
  - `sample_modal_token_masks` / `token_mask_to_pixel_mask` 는 MultiMAE 식
    cross-modal masked reconstruction 의 마스크 샘플러다 (학습 루프에서만 호출).
  - `load_pretrained_adapters` 는 파인튠(`train_reliadino.py`)이 부르는 유일한
    진입점이며, `MODEL.PRETRAINED_ADAPTERS` 키가 **없으면 아무 일도 하지 않는다**
    (기존 config 전부 바이트 무영향).

산출 state_dict 에 담기는 것 = LoRA(모달별) + 융합 + 트렁크 + FPN.
담기지 않는 것 = frozen 백본(원본 DINOv3 그대로), `head`(클래스 수 의존),
M2F/P43/router 등 과제별 헤드, 그리고 recon 헤드(사전학습 후 폐기).
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 1. 어댑터 state_dict 그룹 정의
# ─────────────────────────────────────────────────────────────────────────────
#  키 이름의 출처(하드코딩이 아니라 실제 모듈 구조):
#   lora   : encoder.backbone.blocks.<k>.attn.qkv.{a_q,b_q,a_v,b_v}
#            (encoder.MultiModalLoRAQKV — 모달별 Q/V 저랭크 어댑터)
#   fusion : fusion.*            (ReliabilityGatedFusion 전체)
#   trunk  : trunk_exp / trunk_xattn / trunk_gamma  (P39-V1 modal subspace
#            restoration; FUSION.TRUNK 가 xattn 이면 trunk_xattn 쪽)
#   fpn    : fpn.*               (SimpleFPN — 클래스 수 무관)
ADAPTER_GROUPS: Dict[str, Callable[[str], bool]] = {
    'lora':   lambda k: k.endswith(('.a_q', '.b_q', '.a_v', '.b_v')),
    'fusion': lambda k: k.startswith('fusion.'),
    'trunk':  lambda k: k.startswith(('trunk_exp', 'trunk_xattn', 'trunk_gamma')),
    'fpn':    lambda k: k.startswith('fpn.'),
}
DEFAULT_ADAPTER_GROUPS: Tuple[str, ...] = ('lora', 'fusion', 'trunk', 'fpn')


def filter_adapter_state_dict(state_dict: Dict[str, torch.Tensor],
                              groups: Sequence[str] = DEFAULT_ADAPTER_GROUPS,
                              ) -> 'OrderedDict[str, torch.Tensor]':
    """전체 state_dict 에서 P50 산출 대상 키만 남긴다 (DDP `module.` 접두 제거)."""
    unknown = [g for g in groups if g not in ADAPTER_GROUPS]
    if unknown:
        raise ValueError(f"unknown adapter group(s) {unknown}; "
                         f"available={sorted(ADAPTER_GROUPS)}")
    preds = [ADAPTER_GROUPS[g] for g in groups]
    out: 'OrderedDict[str, torch.Tensor]' = OrderedDict()
    for k, v in state_dict.items():
        kk = k[len('module.'):] if k.startswith('module.') else k
        if any(p(kk) for p in preds):
            out[kk] = v.detach().cpu() if torch.is_tensor(v) else v
    return out


def load_pretrained_adapters(model: nn.Module,
                             model_cfg: Optional[dict],
                             verbose: bool = True) -> Optional[dict]:
    """[P50-MAP] `MODEL.PRETRAINED_ADAPTERS` 를 파인튠 모델에 얹는다.

    키가 없거나 빈 문자열이면 **None 을 돌려주고 모델을 건드리지 않는다** — 기존
    config 는 이 함수가 없는 것과 완전히 동일하게 동작한다.

    반대로 키가 있는데 파일이 없으면 **예외로 죽인다**. 조용히 건너뛰면
    "사전학습 팔"이라고 이름 붙은 런이 실제로는 무사전학습으로 돌아가고, A/B의
    유일 변수가 사라진다 (게이트 §3 falsifiable 조항이 무너짐).

    Returns: {'path', 'loaded', 'missing', 'unexpected'} 또는 None.
    """
    cfg = model_cfg or {}
    path = str(cfg.get('PRETRAINED_ADAPTERS', '') or '')
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[P50-MAP] MODEL.PRETRAINED_ADAPTERS='{path}' 파일이 없다. "
            f"사전학습 ckpt 경로를 고치거나 키를 지우고 돌려라 "
            f"(조용한 무사전학습 폴백은 금지 — A/B 유일변수 보호).")
    strict_unexpected = bool(cfg.get('PRETRAINED_ADAPTERS_STRICT', True))
    ck = torch.load(path, map_location='cpu')
    sd = ck
    if isinstance(ck, dict):
        for key in ('model_state_dict', 'state_dict', 'adapters'):
            if key in ck and isinstance(ck[key], dict):
                sd = ck[key]
                break
    sd = {(k[len('module.'):] if k.startswith('module.') else k): v
          for k, v in sd.items()}

    info = model.load_state_dict(sd, strict=False)
    unexpected = list(info.unexpected_keys)
    if unexpected and strict_unexpected:
        raise RuntimeError(
            f"[P50-MAP] 사전학습 ckpt 에 현재 모델이 모르는 키가 {len(unexpected)}개 "
            f"있다 (아키텍처 불일치 — 사전학습 cfg 와 파인튠 cfg 의 MODEL 절이 "
            f"다르다는 뜻). 예: {unexpected[:8]}. "
            f"의도한 것이면 MODEL.PRETRAINED_ADAPTERS_STRICT: false 로 풀어라.")
    out = {'path': path, 'loaded': len(sd),
           'missing': len(info.missing_keys), 'unexpected': len(unexpected),
           'meta': ck.get('p50_meta') if isinstance(ck, dict) else None}
    if verbose:
        print(f"[P50-MAP] loaded {len(sd)} adapter tensors from {path} "
              f"(model-side missing={len(info.missing_keys)} "
              f"unexpected={len(unexpected)})")
        if out['meta']:
            print(f"[P50-MAP]   pretrain meta: {out['meta']}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. MultiMAE 식 cross-modal 토큰 마스킹
# ─────────────────────────────────────────────────────────────────────────────
def sample_modal_token_masks(batch: int,
                             num_modalities: int,
                             num_tokens: int,
                             mask_ratio: float = 0.75,
                             alpha: float = 1.0,
                             device: Optional[torch.device] = None,
                             generator: Optional[torch.Generator] = None,
                             ) -> torch.Tensor:
    """MultiMAE(2204.01678) §3.1 의 마스킹: **모달 전체를 합친** 토큰 풀에서
    가시 토큰 예산을 Dirichlet(α) 비율로 모달에 나눠 준다.

    모달마다 독립적으로 75%를 가리면 "다른 모달을 봐야만 복원 가능"이라는 압력이
    약해진다 — 예산을 모달 간에 경쟁시켜야 어떤 스텝에서는 depth 가 거의 다
    가려지고 RGB 로부터만 복원해야 하는 국면이 생긴다. 그 국면이 정렬을 만든다.

    Returns: bool (M, B, N) — True = **가시(visible)**, False = 마스킹.
    """
    m, b, n = int(num_modalities), int(batch), int(num_tokens)
    total_visible = int(round((1.0 - float(mask_ratio)) * n * m))
    total_visible = max(m, min(total_visible, n * m))     # 모달당 최소 1개는 보장

    dev = device or torch.device('cpu')
    conc = torch.full((b, m), float(alpha), device=dev)
    props = torch._standard_gamma(conc)                    # Dirichlet via Gamma
    props = props / props.sum(dim=1, keepdim=True).clamp(min=1e-8)

    counts = torch.floor(props * total_visible).long().clamp(min=1, max=n)
    # 잔여 예산을 비율이 큰 모달부터 채워 정확히 total_visible 을 맞춘다.
    for _ in range(m):
        deficit = total_visible - counts.sum(dim=1)        # (B,)
        if not bool((deficit > 0).any()):
            break
        room = (n - counts)                                # (B, M)
        order = torch.argsort(props * (room > 0), dim=1, descending=True)
        top = order[:, 0]
        add = torch.minimum(deficit.clamp(min=0), room.gather(1, top[:, None]).squeeze(1))
        counts.scatter_add_(1, top[:, None], add[:, None])
        if not bool((add > 0).any()):
            break
    counts = counts.clamp(min=1, max=n)

    noise = torch.rand(b, m, n, device=dev, generator=generator)
    rank = noise.argsort(dim=2).argsort(dim=2)             # 0..n-1 무작위 순위
    visible = rank < counts[:, :, None]                    # (B, M, N)
    return visible.permute(1, 0, 2).contiguous()           # (M, B, N)


def token_mask_to_pixel_mask(visible: torch.Tensor, h: int, w: int,
                             out_hw: Tuple[int, int]) -> torch.Tensor:
    """(M,B,N) 토큰 가시 마스크 → (M,B,1,H,W) 픽셀 가시 마스크 (nearest 확대)."""
    m, b, n = visible.shape
    assert n == h * w, f"token count {n} != grid {h}x{w}"
    v = visible.reshape(m * b, 1, h, w).float()
    v = F.interpolate(v, size=out_hw, mode='nearest')
    return v.reshape(m, b, 1, *out_hw)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 사전학습 전용 복원 헤드 (사전학습 후 폐기 — 산출 state_dict 에 없다)
# ─────────────────────────────────────────────────────────────────────────────
class ReconHead(nn.Module):
    """stride-4 피라미드 레벨 → 원해상도 모달 재구성 (PixelShuffle 업샘플).

    일부러 얕다 — 복원 품질이 아니라 **트렁크가 모달 간 정보를 실어 나르는지**가
    목적이고, 헤드가 두꺼우면 헤드가 그 일을 대신 해 버린다.
    """

    def __init__(self, in_dim: int, out_ch: int = 3, upscale: int = 4,
                 hidden: Optional[int] = None):
        super().__init__()
        hidden = int(hidden or in_dim)
        groups = 32 if hidden % 32 == 0 else 1
        self.upscale = int(upscale)
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch * upscale * upscale, 1),
        )
        self.shuffle = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.body(x))


def masked_recon_loss(pred: torch.Tensor, target: torch.Tensor,
                      visible: torch.Tensor) -> torch.Tensor:
    """가려진 픽셀에서만 L2. `visible` 은 (B,1,H,W) float (1=가시).

    가시 픽셀에 손실을 걸면 항등사상(입력 복사)이 최적해가 되어 모달 간 참조를
    배우지 않는다 — MAE/MultiMAE 와 동일한 이유로 마스킹 영역만 센다.
    """
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear',
                             align_corners=False)
    masked = (1.0 - visible)
    denom = masked.sum() * pred.shape[1]
    if float(denom) <= 0:
        return pred.sum() * 0.0
    return (((pred - target) ** 2) * masked).sum() / denom


__all__ = [
    'ADAPTER_GROUPS', 'DEFAULT_ADAPTER_GROUPS', 'filter_adapter_state_dict',
    'load_pretrained_adapters', 'sample_modal_token_masks',
    'token_mask_to_pixel_mask', 'ReconHead', 'masked_recon_loss',
]
