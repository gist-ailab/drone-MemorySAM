import os
import sys
import torch
import argparse
import yaml
import numpy as np
from pathlib import Path
from torchvision import io
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))
from semseg.augmentations_mm import get_val_augmentation
from semseg.datasets.deliver import DELIVER
from semseg.models.sam2.sam2.build_sam import build_sam2
from semseg.models.sam2.sam2.sam_lora_image_encoder_seg import LoRA_Sam


def find_modality_paths(rgb_path, modals):
    """
    RGB 이미지 경로에서 다른 모달리티 경로를 생성
    
    Args:
        rgb_path: RGB 이미지 파일 경로
        modals: 사용할 모달리티 리스트
    
    Returns:
        dict: 모달리티별 파일 경로
    """
    paths = {}
    
    # RGB는 원본 경로
    if 'img' in modals:
        paths['img'] = rgb_path
    
    # 다른 모달리티 경로 생성
    if 'depth' in modals:
        depth_path = rgb_path.replace('/img', '/hha').replace('_rgb', '_depth')
        paths['depth'] = depth_path
    
    if 'lidar' in modals:
        lidar_path = rgb_path.replace('/img', '/lidar').replace('_rgb', '_lidar')
        paths['lidar'] = lidar_path
    
    if 'event' in modals:
        event_path = rgb_path.replace('/img', '/event').replace('_rgb', '_event')
        paths['event'] = event_path
    
    # Semantic GT 경로 (있으면 사용)
    semantic_path = rgb_path.replace('/img', '/semantic').replace('_rgb', '_semantic')
    paths['semantic'] = semantic_path
    
    return paths


def load_image(file_path):
    """이미지 로드 (CHW format)"""
    if not os.path.exists(file_path):
        return None
    
    img = io.read_image(file_path)
    C, H, W = img.shape
    
    # 채널 처리
    if C == 4:
        img = img[:3, ...]
    if C == 1:
        img = img.repeat(3, 1, 1)
    
    return img


def tensor_to_numpy(img_tensor):
    """PyTorch tensor를 numpy array로 변환 (시각화용)"""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = img_tensor
    
    # CHW -> HWC 변환
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    # 정규화된 값인 경우 [0, 1] 범위로 가정하고 [0, 255]로 변환
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


def visualize_result(sample, pred, gt_label=None, palette=None, save_path=None):
    """
    결과 시각화
    
    Args:
        sample: dict with keys 'img', 'depth', 'lidar', 'event'
        pred: 예측 결과 (H, W) numpy array
        gt_label: GT 레이블 (H, W) numpy array (optional)
        palette: 색상 팔레트
        save_path: 저장 경로
    """
    modals = ['img', 'depth', 'lidar', 'event']
    n_modals = sum(1 for m in modals if m in sample)
    
    # GT가 있으면 추가 컬럼
    n_cols = n_modals + 1 + (1 if gt_label is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    idx = 0
    
    # RGB 이미지
    if 'img' in sample:
        rgb = tensor_to_numpy(sample['img'])
        axes[idx].imshow(rgb)
        axes[idx].set_title('RGB', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    # Depth 이미지
    if 'depth' in sample:
        depth = tensor_to_numpy(sample['depth'])
        axes[idx].imshow(depth)
        axes[idx].set_title('Depth', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    # LiDAR 이미지
    if 'lidar' in sample:
        lidar = tensor_to_numpy(sample['lidar'])
        axes[idx].imshow(lidar)
        axes[idx].set_title('LiDAR', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    # Event 이미지
    if 'event' in sample:
        event = tensor_to_numpy(sample['event'])
        axes[idx].imshow(event)
        axes[idx].set_title('Event', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
        idx += 1
    
    # 예측 결과
    if palette is not None:
        h, w = pred.shape
        colored_pred = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in range(len(palette)):
            mask = pred == class_id
            colored_pred[mask] = palette[class_id].cpu().numpy() if isinstance(palette, torch.Tensor) else palette[class_id]
        axes[idx].imshow(colored_pred)
    else:
        axes[idx].imshow(pred, cmap='tab20')
    axes[idx].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[idx].axis('off')
    idx += 1
    
    # GT 레이블 (있다면)
    if gt_label is not None:
        if palette is not None:
            h, w = gt_label.shape
            colored_gt = np.zeros((h, w, 3), dtype=np.uint8)
            for class_id in range(len(palette)):
                mask = gt_label == class_id
                colored_gt[mask] = palette[class_id].cpu().numpy() if isinstance(palette, torch.Tensor) else palette[class_id]
            axes[idx].imshow(colored_gt)
        else:
            axes[idx].imshow(gt_label, cmap='tab20')
        axes[idx].set_title('GT', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


@torch.no_grad()
def inference_single_image(cfg, image_path, output_path):
    """
    단일 이미지에 대한 추론 수행
    
    Args:
        cfg: 설정 딕셔너리
        image_path: 입력 이미지 경로 (RGB)
        output_path: 출력 이미지 저장 경로
    """
    device = torch.device(cfg['DEVICE'])
    eval_cfg = cfg['EVAL']
    dataset_cfg = cfg['DATASET']
    modals = dataset_cfg['MODALS']
    
    # 팔레트 로드
    palette = DELIVER.PALETTE
    
    # 모델 로드
    print("Loading model...")
    checkpoint = "semseg/models/sam2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2 = build_sam2("sam2_hiera_b+.yaml", checkpoint)
    model = LoRA_Sam(sam2, 4).cpu()
    
    # 웨이트 로드
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    print(f"Loading weights from {model_path}...")
    msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'), strict=False)
    print(msg)
    
    model = model.to(device)
    model.eval()
    
    # 전처리 변환
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    
    # 모달리티 경로 찾기
    modality_paths = find_modality_paths(image_path, modals)
    
    # 이미지 로드 (원본 해상도)
    print("Loading images...")
    sample_orig = {}  # 원본 이미지 (시각화용)
    sample = {}  # 전처리용
    
    if 'img' in modals:
        img = load_image(modality_paths['img'])
        if img is None:
            raise FileNotFoundError(f"Image not found: {modality_paths['img']}")
        sample_orig['img'] = img
        sample['img'] = img
    
    H_orig, W_orig = sample['img'].shape[1:]
    
    if 'depth' in modals:
        depth = load_image(modality_paths['depth'])
        if depth is not None:
            sample_orig['depth'] = depth
            sample['depth'] = depth
        else:
            print(f"Warning: Depth image not found: {modality_paths['depth']}")
    
    if 'lidar' in modals:
        lidar = load_image(modality_paths['lidar'])
        if lidar is not None:
            sample_orig['lidar'] = lidar
            sample['lidar'] = lidar
        else:
            print(f"Warning: LiDAR image not found: {modality_paths['lidar']}")
    
    if 'event' in modals:
        event = load_image(modality_paths['event'])
        if event is not None:
            # Event 이미지 리사이즈 (원본 해상도로)
            event_resized = TF.resize(event, (H_orig, W_orig), TF.InterpolationMode.NEAREST)
            sample_orig['event'] = event_resized
            sample['event'] = event
        else:
            print(f"Warning: Event image not found: {modality_paths['event']}")
    
    # GT 레이블 로드 (있으면) - 원본 해상도
    gt_label = None
    if os.path.exists(modality_paths['semantic']):
        try:
            label = io.read_image(modality_paths['semantic'])[0, ...].unsqueeze(0)
            label[label == 255] = 0
            label = label - 1
            gt_label = label.squeeze().cpu().numpy().astype(np.int64)
            print("GT label found and loaded.")
        except Exception as e:
            print(f"Warning: Failed to load GT label: {e}")
    
    # 전처리 적용 (리사이즈 및 정규화)
    if transform:
        sample = transform(sample)
    
    # 모델 입력 형식으로 변환 (리스트)
    model_input = [sample[k] for k in modals]
    model_input = [x.unsqueeze(0).to(device) for x in model_input]
    
    # 추론
    print("Running inference...")
    output, _ = model(model_input, multimask_output=True)
    pred = output.softmax(dim=1).argmax(dim=1).squeeze().cpu().numpy()
    
    # 예측 결과를 원본 해상도로 리사이즈 (필요시)
    H_pred, W_pred = pred.shape
    if H_pred != H_orig or W_pred != W_orig:
        pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
        pred_resized = TF.resize(pred_tensor, (H_orig, W_orig), TF.InterpolationMode.NEAREST)
        pred = pred_resized.squeeze().cpu().numpy().astype(np.int64)
    
    # GT도 예측 해상도에 맞춰 리사이즈 (필요시)
    if gt_label is not None:
        H_gt, W_gt = gt_label.shape
        if H_gt != H_orig or W_gt != W_orig:
            gt_tensor = torch.from_numpy(gt_label).unsqueeze(0).unsqueeze(0).float()
            gt_resized = TF.resize(gt_tensor, (H_orig, W_orig), TF.InterpolationMode.NEAREST)
            gt_label = gt_resized.squeeze().cpu().numpy().astype(np.int64)
    
    # 결과 시각화 및 저장
    print(f"Saving result to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visualize_result(sample_orig, pred, gt_label, palette, save_path=output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Single Image Inference')
    parser.add_argument('--cfg', type=str, required=True, help='Configuration file path')
    parser.add_argument('--image', type=str, required=True, help='Input image path (RGB)')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    
    args = parser.parse_args()
    
    # Config 로드
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    # 추론 수행
    inference_single_image(cfg, args.image, args.output)


if __name__ == '__main__':
    main()
