from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01,
                  backbone_lr_scale: float = 1.0, backbone_prefix: str = None):
    # [P31] parameters under backbone_prefix (e.g. unfrozen Hiera trunk blocks) train at
    # lr * backbone_lr_scale; defaults keep the original two-group behavior byte-identical.
    use_bb = backbone_prefix is not None and backbone_lr_scale != 1.0
    wd_params, nwd_params = [], []
    bb_wd_params, bb_nwd_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = use_bb and name.startswith(backbone_prefix)
        if p.dim() == 1:
            (bb_nwd_params if is_bb else nwd_params).append(p)
        else:
            (bb_wd_params if is_bb else wd_params).append(p)

    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]
    if bb_wd_params:
        params.append({"params": bb_wd_params, "lr": lr * backbone_lr_scale})
    if bb_nwd_params:
        params.append({"params": bb_nwd_params, "lr": lr * backbone_lr_scale, "weight_decay": 0})

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
