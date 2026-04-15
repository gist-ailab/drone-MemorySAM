"""mmcv.cnn의 최소 호환 shim — pure PyTorch 구현.

이 프로젝트에서 실제로 쓰이는 mmcv API만 재현한다:
  - build_norm_layer
  - build_activation_layer
  - ConvModule

state_dict 호환을 위해 mmcv 1.6.1과 동일한 attribute 이름 규칙을 따른다:
  - ConvModule.conv
  - ConvModule.bn / ConvModule.gn / ConvModule.ln / ...  (norm type에 따라 동적)
  - ConvModule.activate

이렇게 하면 기존 mmcv 기반으로 학습한 체크포인트를 그대로 로드할 수 있다.
"""

import torch.nn as nn


# mmcv가 사용하는 norm prefix 규칙 (1.6.1 소스 기준)
_NORM_ABBR = {
    'BN': 'bn', 'BN1d': 'bn', 'BN2d': 'bn', 'BN3d': 'bn',
    'SyncBN': 'bn',
    'GN': 'gn',
    'LN': 'ln',
    'IN': 'in', 'IN1d': 'in', 'IN2d': 'in', 'IN3d': 'in',
}


def build_norm_layer(cfg, num_features, postfix=''):
    """mmcv.cnn.build_norm_layer 호환.

    Args:
        cfg (dict): {'type': 'BN'|'GN'|..., ...extra kwargs}
        num_features (int): 채널 수
        postfix (str|int): attribute 이름 접미사 (mmcv는 빈 문자열 or 숫자)

    Returns:
        (name, norm_module): name은 'bn'/'gn' 등 + postfix, norm_module은 nn.Module
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    norm_type = cfg.pop('type')
    requires_grad = cfg.pop('requires_grad', True)
    eps = cfg.pop('eps', 1e-5)

    if norm_type in ('BN', 'BN2d'):
        layer = nn.BatchNorm2d(num_features, eps=eps, **cfg)
    elif norm_type == 'BN1d':
        layer = nn.BatchNorm1d(num_features, eps=eps, **cfg)
    elif norm_type == 'BN3d':
        layer = nn.BatchNorm3d(num_features, eps=eps, **cfg)
    elif norm_type == 'SyncBN':
        layer = nn.SyncBatchNorm(num_features, eps=eps, **cfg)
    elif norm_type == 'GN':
        num_groups = cfg.pop('num_groups')
        layer = nn.GroupNorm(num_groups, num_features, eps=eps, **cfg)
    elif norm_type == 'LN':
        layer = nn.LayerNorm(num_features, eps=eps, **cfg)
    elif norm_type in ('IN', 'IN2d'):
        layer = nn.InstanceNorm2d(num_features, eps=eps, **cfg)
    elif norm_type == 'IN1d':
        layer = nn.InstanceNorm1d(num_features, eps=eps, **cfg)
    elif norm_type == 'IN3d':
        layer = nn.InstanceNorm3d(num_features, eps=eps, **cfg)
    else:
        raise KeyError(f'Unsupported norm type: {norm_type}')

    for p in layer.parameters():
        p.requires_grad = requires_grad

    name = _NORM_ABBR[norm_type] + str(postfix)
    return name, layer


def build_activation_layer(cfg):
    """mmcv.cnn.build_activation_layer 호환 (최소 범위)."""
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg = cfg.copy()
    act_type = cfg.pop('type')
    if act_type == 'ReLU':
        return nn.ReLU(**cfg)
    if act_type == 'LeakyReLU':
        return nn.LeakyReLU(**cfg)
    if act_type == 'PReLU':
        return nn.PReLU(**cfg)
    if act_type == 'RReLU':
        return nn.RReLU(**cfg)
    if act_type == 'ReLU6':
        return nn.ReLU6(**cfg)
    if act_type == 'ELU':
        return nn.ELU(**cfg)
    if act_type == 'GELU':
        return nn.GELU()
    if act_type == 'Tanh':
        return nn.Tanh()
    if act_type == 'Sigmoid':
        return nn.Sigmoid()
    raise KeyError(f'Unsupported act type: {act_type}')


class ConvModule(nn.Module):
    """mmcv.cnn.ConvModule의 완전 호환 shim.

    호환 범위 — 이 프로젝트에서 실제 사용되는 동작을 모두 지원:
      - conv_cfg=None (표준 nn.Conv2d) 만 지원 (DCN/plugins 등 미지원)
      - norm_cfg: None | BN | GN | SyncBN | LN | IN
      - act_cfg : None | ReLU | GELU | LeakyReLU | …
      - bias='auto': norm 있으면 False, 없으면 True (mmcv 동작)
      - order=('conv', 'norm', 'act') 고정

    Attribute 이름 (state_dict 키 호환):
      - self.conv
      - self.<norm_abbr>  (bn/gn/ln/in — norm type에 따라 동적)
      - self.activate
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 padding_mode='zeros'):
        super().__init__()
        assert conv_cfg is None, (
            'ConvModule shim only supports conv_cfg=None (standard Conv2d). '
            'DCN/plugins are not used in this project.'
        )
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        if bias == 'auto':
            bias = not self.with_norm

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
        )
        # mmcv가 노출하는 conv 메타 속성 재현 (호출처 호환)
        for attr in ('in_channels', 'out_channels', 'kernel_size', 'stride',
                     'padding', 'dilation', 'groups'):
            setattr(self, attr, getattr(self.conv, attr))

        if self.with_norm:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        if self.with_activation:
            act_cfg_ = dict(act_cfg)
            # mmcv는 ReLU류에만 inplace를 주입한다 (Tanh/Sigmoid/GELU 등에는 주입 X)
            if act_cfg_['type'] in ('ReLU', 'LeakyReLU', 'ReLU6', 'ELU', 'RReLU'):
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

    @property
    def norm(self):
        return getattr(self, self.norm_name) if self.norm_name else None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = getattr(self, self.norm_name)(x)
        if self.with_activation:
            x = self.activate(x)
        return x
