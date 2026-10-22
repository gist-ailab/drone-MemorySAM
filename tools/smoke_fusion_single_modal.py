"""[2026-10-22] ReliaDINO 단일 센서 항등 융합 CPU 스모크.

실행: /home/jemo/anaconda3/envs/MMSS_SAM/bin/python -I tools/smoke_fusion_single_modal.py
"""
from __future__ import annotations

import copy
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from semseg.models.reliadino.model import build_reliadino


def main() -> None:
    torch.set_num_threads(2)
    config_path = REPO / 'configs/bengio-deliver_rgbdel_P46_c3only_seed20260821_screen40_E1.yaml'
    with config_path.open(encoding='utf-8') as file:
        base_cfg = yaml.safe_load(file)
    base_cfg['MODEL']['PRETRAINED_BACKBONE'] = False
    size = 256
    x = torch.randn(1, 3, size, size)

    single_cfg = copy.deepcopy(base_cfg)
    single_cfg['DATASET']['MODALS'] = ['img']
    model = build_reliadino(single_cfg, 25).cpu()
    model.eval()
    with torch.no_grad():
        logits, _ = model([x], True)
    assert logits.shape == (1, 25, size, size), logits.shape
    rel = model.fusion._last_rel_spatial
    assert rel is not None and rel.shape[0:3] == (1, 1, 1)
    assert torch.all(rel == 1), 'single-modal reliability must be one'
    del logits

    model.train()
    gt = torch.randint(0, 25, (1, size, size), dtype=torch.long)
    # train_reliadino.py 의 호출 형태와 동일.
    logits, _, aux = model([x], True, gt_mask=gt)
    assert logits.shape == (1, 25, size, size), logits.shape
    assert 'aux_ce' in aux and aux['aux_ce'].ndim == 0 and aux['aux_ce'].item() == 0
    assert not ({'gate_entropy', 'routed_logits', 'router_reg'} & aux.keys())
    loss = F.cross_entropy(logits, gt) + aux['aux_ce']
    loss.backward()
    # LoRA 래퍼는 backbone.blocks.*.attn.qkv 아래에 등록돼 이름에 'lora'가 없다.
    lora_grads = [(name, param.grad) for layer in model.encoder.lora_layers
                  for name, param in layer.named_parameters()
                  if param.requires_grad]
    assert lora_grads, 'no trainable LoRA parameters found'
    assert any(grad is not None and torch.isfinite(grad).all()
               and grad.abs().sum().item() > 0 for _, grad in lora_grads), \
        'no nonzero LoRA gradient'
    del loss, logits, aux, model
    gc.collect()

    multi_cfg = copy.deepcopy(base_cfg)
    multi_cfg['DATASET']['MODALS'] = ['img', 'depth', 'event', 'lidar']
    model = build_reliadino(multi_cfg, 25).cpu().eval()
    with torch.no_grad():
        logits, _ = model([x, x.clone(), x.clone(), x.clone()], True)
    assert logits.shape == (1, 25, size, size), logits.shape
    print('SMOKE OK')


if __name__ == '__main__':
    main()
