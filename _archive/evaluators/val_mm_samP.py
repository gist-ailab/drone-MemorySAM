import torch
import argparse
import yaml
import math
import os
import time
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from semseg.models.sam2.sam2.build_sam import build_sam2 as build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg_bkup import LoRA_Sam
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import LoRA_Sam_P3, LoRA_Sam_P2, LoRA_Sam_P1, LoRA_Sam_P5, LoRA_Sam_P4, LoRA_Sam_P6
import inspect

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2]*1)), int(ceil(image_size[3]*1)))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction, _ = model(padded_img, multimask_output=True)
            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions, _ = model(fliped_img, multimask_output=True)
                padded_prediction += fliped_predictions.flip(-1)
            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)

    return total_predictions.unsqueeze(0)
    
@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    sliding = False
    for images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        if sliding:
            preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
        else:
            output, _ = model(images, multimask_output=True)
            preds = output.softmax(dim=1)
        metrics.update(preds, labels)
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou

@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in images]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
            logits, _ = model(scaled_images, multimask_output=True)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                logits, _ = model(scaled_images, multimask_output=True)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    dataset_cfg = cfg['DATASET']
    model_cfg = cfg['MODEL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    cases = [None] # all
    
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): 
        raise FileNotFoundError(f"Model path not found: {model_path}")
    print(f"Evaluating {model_path}...")

    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))

    for case in cases:
        dataset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', transform, dataset_cfg['MODALS'], case)
        # --- test set
        # dataset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'test', transform, dataset_cfg['MODALS'], case)

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
        
        # Get LoRA model configuration from config
        lora_model_name = model_cfg.get('LORA_MODEL', 'LoRA_Sam_P6')
        lora_r = model_cfg.get('LORA_R', 4)
        lora_num_experts = model_cfg.get('LORA_NUM_EXPERTS')
        if lora_num_experts is None:
            lora_num_experts = num_modalities
        lora_top_k = model_cfg.get('LORA_TOP_K', 2)
        lora_layer = model_cfg.get('LORA_LAYER', None)
        
        # Dynamically load LoRA model class
        lora_model_class = eval(lora_model_name)
        
        # Build model with config parameters
        model_kwargs = {
            'sam_model': sam2,
            'r': lora_r,
            'lora_layer': lora_layer,
        }
        
        # Add optional parameters if they exist in the model signature
        sig = inspect.signature(lora_model_class.__init__)
        if 'num_experts' in sig.parameters:
            model_kwargs['num_experts'] = lora_num_experts
        if 'top_k' in sig.parameters:
            model_kwargs['top_k'] = lora_top_k
        
        model = lora_model_class(**model_kwargs).cpu()
        print(f"Using LoRA model: {lora_model_name}")
        print(f"LoRA parameters: r={lora_r}, num_experts={lora_num_experts}, top_k={lora_top_k}, lora_layer={lora_layer}")
        
        # Load model weights
        msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'), strict=False)
        print(f"Model loading message: {msg}")
        model = model.to(device)
        model.eval()
        
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=4, pin_memory=False, sampler=sampler_val)
        
        if eval_cfg['MSF']['ENABLE']:
            acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
        else:
            acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device)

        table = {
            'Class': list(dataset.CLASSES) + ['Mean'],
            'IoU': [f"{iou:.4f}" for iou in ious] + [f"{miou:.4f}"],
            'F1': [f"{f:.4f}" for f in f1] + [f"{mf1:.4f}"],
            'Acc': [f"{a:.4f}" for a in acc] + [f"{macc:.4f}"]
        }
        print("\n" + "="*80)
        print("Evaluation Results")
        print("="*80)
        print(tabulate(table, headers='keys', tablefmt='grid'))
        print(f"\nmIoU: {miou:.4f}")
        print(f"Results saved in {eval_path}")

        with open(eval_path, 'a+') as f:
            f.write(f"{eval_cfg['MODEL_PATH']}\n")
            f.write(f"============== Eval on {case} {len(dataset)} images =================\n")
            f.write(f"LoRA Model: {lora_model_name}\n")
            f.write(f"LoRA Parameters: r={lora_r}, num_experts={lora_num_experts}, top_k={lora_top_k}\n")
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)
