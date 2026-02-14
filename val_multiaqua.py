"""
MULTIAQUA 데이터셋용 Validation 및 Test Inference 스크립트.
- val: validation set 평가 (mIoU, mAcc, Dynamic IoU 별도) - annotation 필요
- test: test set 인퍼런스만 (annotation 없음) - seg/, seg_viz/ 저장

저장 구조 (test 모드):
  save_dir/seg/      : 클래스값 0,1,2,3 (uint8) raw segmentation
  save_dir/seg_viz/  : colormap 시각화

사용:
  python val_multiaqua.py --cfg configs/lecun_multiaqua_rgbtl_P8.yaml --mode val --model_path outputs/.../epoch15_93.95_checkpoint.pth
  python val_multiaqua.py --cfg configs/lecun_multiaqua_rgbtl_P8.yaml --mode test --model_path outputs/.../epoch15_93.95_checkpoint.pth --save_dir outputs/.../test_pred
"""
import torch
import argparse
import yaml
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import inspect

from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import *


def load_model(cfg, model_path, device):
    """Config 기반 LoRA 모델 로드."""
    model_cfg = cfg['MODEL']
    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']

    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_config_file = "sam2_hiera_b+.yaml"
    num_modalities = len(dataset_cfg['MODALS'])

    sam2 = build_sam2(
        sam2_config_file,
        checkpoint,
        hydra_overrides_extra=[
            "++model.pred_obj_scores=false",
            "++model.fixed_no_obj_ptr=false",
            "++model.pred_obj_scores_mlp=false"
        ]
    )

    lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P8')
    lora_r = model_cfg.get('LORA_R', 4)
    lora_num_experts = model_cfg.get('LORA_NUM_EXPERTS')
    if lora_num_experts is None:
        lora_num_experts = num_modalities
    lora_top_k = model_cfg.get('LORA_TOP_K')
    lora_layer = model_cfg.get('LORA_LAYER')

    lora_model_class = eval(lora_model_name)
    model_kwargs = {
        'sam_model': sam2,
        'r': lora_r,
        'lora_layer': lora_layer,
    }
    sig = inspect.signature(lora_model_class.__init__)
    if 'num_experts' in sig.parameters:
        model_kwargs['num_experts'] = lora_num_experts
    if 'top_k' in sig.parameters:
        model_kwargs['top_k'] = lora_top_k

    model = lora_model_class(**model_kwargs)

    ckpt = torch.load(str(model_path), map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    msg = model.load_state_dict(state, strict=False)
    print(f"Model load: {msg}")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Validation 평가. Dynamic(class 1) IoU 별도 계산."""
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader, desc="Val"):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        output, _ = model(images, multimask_output=True)
        preds = output.softmax(dim=1)
        metrics.update(preds, labels)

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    # Dynamic = class index 1
    dynamic_iou = float(ious[1])
    return acc, macc, f1, mf1, ious, miou, dynamic_iou


@torch.no_grad()
def run_test_inference(model, dataloader, device, save_dir):
    """
    Test set 인퍼런스 후 저장.
    - seg/: 클래스값 0,1,2,3 (uint8) raw segmentation
    - seg_viz/: colormap 시각화
    """
    model.eval()
    n_classes = dataloader.dataset.n_classes
    palette = dataloader.dataset.PALETTE
    stems = dataloader.dataset.stems

    save_dir = Path(save_dir)
    seg_dir = save_dir / "seg"
    seg_viz_dir = save_dir / "seg_viz"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_viz_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    idx = 0
    for images, _ in tqdm(dataloader, desc="Test inference"):
        images = [x.to(device) for x in images]
        output, _ = model(images, multimask_output=True)
        preds = output.softmax(dim=1)

        # MULTIAQUA: 4 classes (0=Static, 1=Dynamic, 2=Water, 3=Sky). 모델 25채널 → 앞 4채널만 사용.
        pred_labels = preds[:, :n_classes].argmax(dim=1)  # (B, H, W)

        for b in range(pred_labels.shape[0]):
            stem = stems[idx] if idx < len(stems) else f"pred_{idx:06d}"
            pred_np = pred_labels[b].cpu().numpy().astype(np.uint8)

            # seg: raw class values (0,1,2,3)
            seg_path = seg_dir / f"{stem}.png"
            Image.fromarray(pred_np).save(str(seg_path))

            # seg_viz: colormap 시각화
            colored = MULTIAQUA.decode_segmap(pred_np, palette)
            viz_path = seg_viz_dir / f"{stem}.png"
            Image.fromarray(colored).save(str(viz_path))
            idx += 1

    print(f"Saved {idx} predictions: seg/ and seg_viz/ under {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/lecun_multiaqua_rgbtl_P8.yaml')
    parser.add_argument('--mode', type=str, choices=['val', 'test'], default='val')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device(cfg['DEVICE'])
    setup_cudnn()

    dataset_cfg = cfg['DATASET']
    eval_cfg = cfg['EVAL']
    test_cfg = cfg.get('TEST', {})

    image_size = eval_cfg['IMAGE_SIZE'] if args.mode == 'val' else test_cfg.get('IMAGE_SIZE', eval_cfg['IMAGE_SIZE'])
    transform = get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    split = 'val' if args.mode == 'val' else 'test'
    require_annotation = args.mode == 'val'  # val=평가용(annotation필요), test=인퍼런스만(RGB만)
    dataset = MULTIAQUA(
        dataset_cfg['ROOT'],
        split=split,
        transform=transform,
        modals=dataset_cfg['MODALS'],
        require_annotation=require_annotation,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=eval_cfg['BATCH_SIZE'],
        num_workers=4,
        pin_memory=False,
    )

    model = load_model(cfg, model_path, device)

    if args.mode == 'val':
        acc, macc, f1, mf1, ious, miou, dynamic_iou = evaluate(model, dataloader, device)
        table = {
            'Class': list(dataset.CLASSES) + ['Mean'],
            'IoU': [f"{iou:.2f}" for iou in ious] + [f"{miou:.2f}"],
            'Acc': [f"{a:.2f}" for a in acc] + [f"{macc:.2f}"],
        }
        print("\n" + "=" * 60)
        print(f"MULTIAQUA Validation ({len(dataset)} images)")
        print("=" * 60)
        print(tabulate(table, headers='keys', tablefmt='grid'))
        print(f"\nmIoU: {miou:.2f}  mAcc: {macc:.2f}")
        print(f"Dynamic IoU (class 1): {dynamic_iou:.2f}")

        out_txt = model_path.parent / f"eval_{split}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(out_txt, 'w') as f:
            f.write(f"Model: {model_path}\n")
            f.write(f"Split: {split}  N={len(dataset)}\n")
            f.write(tabulate(table, headers='keys') + "\n")
            f.write(f"\nDynamic IoU (class 1): {dynamic_iou:.2f}\n")
        print(f"Results saved to {out_txt}")

    else:
        save_dir = args.save_dir or (model_path.parent / "test_pred")
        run_test_inference(model, dataloader, device, save_dir)


if __name__ == '__main__':
    main()
