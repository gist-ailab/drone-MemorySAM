import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from pathlib import Path
from tqdm import tqdm
import cv2
from collections import defaultdict

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))
from semseg.datasets.deliver import DELIVER
from semseg.augmentations_mm import get_val_augmentation


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


def load_lidar_color(rgb_path):
    """
    RGB 파일 경로에서 LiDAR 컬러 이미지 경로를 생성하고 로드
    
    Args:
        rgb_path: RGB 이미지 파일 경로
    
    Returns:
        LiDAR 컬러 이미지 tensor 또는 None
    """
    try:
        lidar_color_path = rgb_path.replace('/img', '/lidar').replace('_rgb', '_lidar_color')
        if os.path.exists(lidar_color_path):
            from torchvision import io
            img = io.read_image(lidar_color_path)
            C, H, W = img.shape
            if C == 4:
                img = img[:3, ...]
            if C == 1:
                img = img.repeat(3, 1, 1)
            return img
    except Exception as e:
        pass
    return None


def visualize_modalities(sample, label, palette, save_path=None, rgb_file_path=None):
    """
    모든 모달리티를 한번에 시각화
    
    Args:
        sample: dict with keys 'img', 'depth', 'lidar', 'event'
        label: semantic label tensor
        palette: color palette for semantic segmentation
        save_path: 저장 경로 (None이면 표시만)
        rgb_file_path: RGB 파일 경로 (LiDAR 컬러 로드를 위해)
    """
    modals = ['img', 'depth', 'lidar', 'event']
    n_modals = sum(1 for m in modals if m in sample)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # RGB 이미지
    if 'img' in sample:
        rgb = tensor_to_numpy(sample['img'])
        axes[0].imshow(rgb)
        axes[0].set_title('RGB', fontsize=14, fontweight='bold')
        axes[0].axis('off')
    
    # Depth 이미지
    if 'depth' in sample:
        depth = tensor_to_numpy(sample['depth'])
        axes[1].imshow(depth)
        axes[1].set_title('Depth (HHA)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
    
    # LiDAR 이미지
    if 'lidar' in sample:
        lidar = tensor_to_numpy(sample['lidar'])
        axes[2].imshow(lidar)
        axes[2].set_title('LiDAR', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    # Event 이미지
    if 'event' in sample:
        event = tensor_to_numpy(sample['event'])
        axes[3].imshow(event)
        axes[3].set_title('Event', fontsize=14, fontweight='bold')
        axes[3].axis('off')
    
    # Semantic Label
    if label is not None:
        label_np = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        if label_np.ndim == 2:
            # 색상 팔레트 적용
            h, w = label_np.shape
            colored_label = np.zeros((h, w, 3), dtype=np.uint8)
            for class_id in range(len(palette)):
                mask = label_np == class_id
                colored_label[mask] = palette[class_id].cpu().numpy() if isinstance(palette, torch.Tensor) else palette[class_id]
            axes[4].imshow(colored_label)
            axes[4].set_title('Semantic Label', fontsize=14, fontweight='bold')
            axes[4].axis('off')
    
    # LiDAR Color 이미지 (마지막 그리드)
    if rgb_file_path is not None:
        lidar_color = load_lidar_color(rgb_file_path)
        if lidar_color is not None:
            lidar_color_np = tensor_to_numpy(lidar_color)
            axes[5].imshow(lidar_color_np)
            axes[5].set_title('LiDAR Color', fontsize=14, fontweight='bold')
            axes[5].axis('off')
        else:
            axes[5].axis('off')
    else:
        axes[5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def analyze_brightness(img_tensor, modal_name='RGB'):
    """
    이미지의 밝기를 정량적으로 분석
    
    Returns:
        dict: 평균 밝기, 표준편차, 최소값, 최대값, 히스토그램 등
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = img_tensor
    
    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    # RGB to Grayscale for brightness analysis
    if img.ndim == 3 and img.shape[2] == 3:
        # Convert to grayscale using standard weights
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img if img.ndim == 2 else img[:, :, 0]
    
    # Normalize to [0, 1] if needed
    if gray.max() > 1.0:
        gray = gray.astype(np.float32) / 255.0
    
    stats = {
        'mean': float(np.mean(gray)),
        'std': float(np.std(gray)),
        'min': float(np.min(gray)),
        'max': float(np.max(gray)),
        'median': float(np.median(gray)),
        'percentile_10': float(np.percentile(gray, 10)),
        'percentile_90': float(np.percentile(gray, 90)),
        'histogram': np.histogram(gray, bins=256, range=(0, 1))[0]
    }
    
    return stats


class InteractiveViewer:
    """인터랙티브 데이터셋 뷰어"""
    
    def __init__(self, dataset, palette):
        self.dataset = dataset
        self.palette = palette
        self.current_idx = 0
        self.fig = None
        self.axes = None
        
    def update_image(self, idx):
        """이미지 업데이트"""
        if idx < 0:
            idx = len(self.dataset) - 1
        elif idx >= len(self.dataset):
            idx = 0
        
        self.current_idx = idx
        
        # 데이터 로드
        sample_list, label = self.dataset[idx]
        sample = {}
        modals = ['img', 'depth', 'lidar', 'event']
        for i, modal in enumerate(modals):
            if i < len(sample_list):
                sample[modal] = sample_list[i]
        
        # RGB 파일 경로 가져오기
        rgb_file_path = str(self.dataset.files[idx]) if hasattr(self.dataset, 'files') and idx < len(self.dataset.files) else None
        
        # 시각화
        if self.fig is None:
            self.fig = visualize_modalities(sample, label, self.palette, rgb_file_path=rgb_file_path)
            self.fig.suptitle(f'Image {idx+1}/{len(self.dataset)}', fontsize=16, fontweight='bold')
        else:
            self.fig.clear()
            self.fig = visualize_modalities(sample, label, self.palette, rgb_file_path=rgb_file_path)
            self.fig.suptitle(f'Image {idx+1}/{len(self.dataset)}', fontsize=16, fontweight='bold')
        
        plt.draw()
    
    def on_key(self, event):
        """키보드 이벤트 핸들러"""
        if event.key == 'right' or event.key == 'n':
            self.update_image(self.current_idx + 1)
        elif event.key == 'left' or event.key == 'p':
            self.update_image(self.current_idx - 1)
        elif event.key == 'q':
            plt.close('all')
    
    def show(self):
        """인터랙티브 뷰어 시작"""
        self.update_image(0)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()


def create_video(dataset, palette, output_path, split='train', case=None, max_frames=100):
    """
    데이터셋을 비디오로 저장
    
    Args:
        dataset: DELIVER 데이터셋
        palette: 색상 팔레트
        output_path: 출력 비디오 경로
        split: 'train' or 'val'
        case: 조건 (None이면 전체)
        max_frames: 최대 프레임 수 (-1이면 모든 프레임 사용)
    """
    print(f"Creating video for {split} split, case: {case}")
    
    # 비디오 작성자 설정
    fps = 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 첫 번째 이미지로 해상도 결정
    sample_list, label = dataset[0]
    sample = {}
    modals = ['img', 'depth', 'lidar', 'event']
    for i, modal in enumerate(modals):
        if i < len(sample_list):
            sample[modal] = sample_list[i]
    
    # RGB 파일 경로 가져오기
    rgb_file_path = str(dataset.files[0]) if hasattr(dataset, 'files') and len(dataset.files) > 0 else None
    
    # 임시 이미지 생성하여 해상도 확인
    fig = visualize_modalities(sample, label, palette, rgb_file_path=rgb_file_path)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    h, w = frame.shape[:2]
    plt.close(fig)
    
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # max_frames가 -1이면 모든 프레임 사용
    if max_frames == -1:
        n_frames = len(dataset)
        print(f"Using all {n_frames} frames")
    else:
        n_frames = min(len(dataset), max_frames)
        print(f"Using {n_frames} frames (max: {max_frames})")
    for idx in tqdm(range(n_frames), desc="Creating video"):
        sample_list, label = dataset[idx]
        sample = {}
        for i, modal in enumerate(modals):
            if i < len(sample_list):
                sample[modal] = sample_list[i]
        
        # RGB 파일 경로 가져오기
        rgb_file_path = str(dataset.files[idx]) if hasattr(dataset, 'files') and idx < len(dataset.files) else None
        
        fig = visualize_modalities(sample, label, palette, rgb_file_path=rgb_file_path)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        plt.close(fig)
    
    video_writer.release()
    print(f"Video saved to {output_path}")


def analyze_condition_brightness(root_path, cases, split='val'):
    """
    조건별 밝기 분석
    
    Args:
        root_path: 데이터셋 루트 경로
        cases: 분석할 조건 리스트
        split: 'train' or 'val'
    
    Returns:
        dict: 조건별 통계
    """
    results = {}
    
    for case in tqdm(cases, desc="Analyzing conditions"):
        try:
            # 각 조건별 데이터셋 생성
            case_dataset = DELIVER(
                root=root_path,
                split=split,
                transform=None,
                modals=['img', 'depth', 'lidar', 'event'],
                case=case
            )
            
            case_stats = {
                'rgb_brightness': [],
                'depth_brightness': [],
                'lidar_brightness': [],
                'event_brightness': []
            }
            
            for idx in range(len(case_dataset)):
                sample_list, label = case_dataset[idx]
                
                # RGB 밝기 분석
                if len(sample_list) > 0:
                    rgb_stats = analyze_brightness(sample_list[0], 'RGB')
                    case_stats['rgb_brightness'].append(rgb_stats['mean'])
                
                # Depth 밝기 분석
                if len(sample_list) > 1:
                    depth_stats = analyze_brightness(sample_list[1], 'Depth')
                    case_stats['depth_brightness'].append(depth_stats['mean'])
                
                # LiDAR 밝기 분석
                if len(sample_list) > 2:
                    lidar_stats = analyze_brightness(sample_list[2], 'LiDAR')
                    case_stats['lidar_brightness'].append(lidar_stats['mean'])
                
                # Event 밝기 분석
                if len(sample_list) > 3:
                    event_stats = analyze_brightness(sample_list[3], 'Event')
                    case_stats['event_brightness'].append(event_stats['mean'])
            
            # 통계 요약
            results[case] = {
                'count': len(case_dataset),
                'rgb_mean': np.mean(case_stats['rgb_brightness']) if case_stats['rgb_brightness'] else None,
                'rgb_std': np.std(case_stats['rgb_brightness']) if case_stats['rgb_brightness'] else None,
                'depth_mean': np.mean(case_stats['depth_brightness']) if case_stats['depth_brightness'] else None,
                'depth_std': np.std(case_stats['depth_brightness']) if case_stats['depth_brightness'] else None,
                'lidar_mean': np.mean(case_stats['lidar_brightness']) if case_stats['lidar_brightness'] else None,
                'lidar_std': np.std(case_stats['lidar_brightness']) if case_stats['lidar_brightness'] else None,
                'event_mean': np.mean(case_stats['event_brightness']) if case_stats['event_brightness'] else None,
                'event_std': np.std(case_stats['event_brightness']) if case_stats['event_brightness'] else None,
            }
            
        except Exception as e:
            print(f"Error analyzing case {case}: {e}")
            results[case] = {'error': str(e)}
    
    return results


def plot_brightness_analysis(results, save_path=None):
    """밝기 분석 결과 시각화"""
    cases = list(results.keys())
    
    # RGB 밝기 비교
    rgb_means = [results[c].get('rgb_mean', 0) for c in cases if 'error' not in results[c]]
    rgb_stds = [results[c].get('rgb_std', 0) for c in cases if 'error' not in results[c]]
    valid_cases = [c for c in cases if 'error' not in results[c]]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RGB 밝기
    axes[0, 0].bar(valid_cases, rgb_means, yerr=rgb_stds, capsize=5, alpha=0.7, color='blue')
    axes[0, 0].set_title('RGB Average Brightness by Condition', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Brightness (0-1)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Depth 밝기
    depth_means = [results[c].get('depth_mean', 0) for c in valid_cases]
    depth_stds = [results[c].get('depth_std', 0) for c in valid_cases]
    axes[0, 1].bar(valid_cases, depth_means, yerr=depth_stds, capsize=5, alpha=0.7, color='green')
    axes[0, 1].set_title('Depth Average Brightness by Condition', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Brightness (0-1)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # LiDAR 밝기
    lidar_means = [results[c].get('lidar_mean', 0) for c in valid_cases]
    lidar_stds = [results[c].get('lidar_std', 0) for c in valid_cases]
    axes[1, 0].bar(valid_cases, lidar_means, yerr=lidar_stds, capsize=5, alpha=0.7, color='red')
    axes[1, 0].set_title('LiDAR Average Brightness by Condition', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Brightness (0-1)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Event 밝기
    event_means = [results[c].get('event_mean', 0) for c in valid_cases]
    event_stds = [results[c].get('event_std', 0) for c in valid_cases]
    axes[1, 1].bar(valid_cases, event_means, yerr=event_stds, capsize=5, alpha=0.7, color='orange')
    axes[1, 1].set_title('Event Average Brightness by Condition', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Brightness (0-1)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DELIVER Dataset Analysis and Visualization')
    parser.add_argument('--root', type=str, default='/ailab_mat2/dataset/DELIVER', help='Dataset root path')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--case', type=str, default=None, help='Specific case to analyze')
    parser.add_argument('--mode', type=str, default='interactive', 
                       choices=['interactive', 'video', 'analyze', 'visualize'],
                       help='Analysis mode')
    parser.add_argument('--output', type=str, default='./MISC/outputs', help='Output directory')
    parser.add_argument('--max_frames', type=int, default=100, help='Max frames for video (-1 to use all frames)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 데이터셋 로드 (transform 없이 원본 사용)
    dataset = DELIVER(
        root=args.root,
        split=args.split,
        transform=None,
        modals=['img', 'depth', 'lidar', 'event'],
        case=args.case
    )
    
    print(f"Loaded dataset: {len(dataset)} images")
    print(f"Split: {args.split}, Case: {args.case}")
    
    # 팔레트 로드
    palette = dataset.PALETTE
    
    if args.mode == 'interactive':
        # 인터랙티브 뷰어
        print("\n=== Interactive Viewer ===")
        print("Controls:")
        print("  Right arrow / 'n': Next image")
        print("  Left arrow / 'p': Previous image")
        print("  'q': Quit")
        viewer = InteractiveViewer(dataset, palette)
        viewer.show()
    
    elif args.mode == 'video':
        # 비디오 생성
        print("\n=== Creating Video ===")
        video_path = os.path.join(args.output, f'deliver_{args.split}_{args.case or "all"}.mp4')
        create_video(dataset, palette, video_path, args.split, args.case, args.max_frames)
    
    elif args.mode == 'analyze':
        # 밝기 분석
        print("\n=== Analyzing Brightness ===")
        cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
        
        # 밝기 분석 수행
        results = analyze_condition_brightness(args.root, cases, args.split)
        
        # 결과 출력
        print("\n=== Brightness Analysis Results ===")
        for case, stats in results.items():
            if 'error' not in stats:
                print(f"\n{case.upper()}:")
                print(f"  Count: {stats['count']}")
                print(f"  RGB Brightness: {stats['rgb_mean']:.4f} ± {stats['rgb_std']:.4f}")
                print(f"  Depth Brightness: {stats['depth_mean']:.4f} ± {stats['depth_std']:.4f}")
                print(f"  LiDAR Brightness: {stats['lidar_mean']:.4f} ± {stats['lidar_std']:.4f}")
                print(f"  Event Brightness: {stats['event_mean']:.4f} ± {stats['event_std']:.4f}")
        
        # 시각화
        plot_path = os.path.join(args.output, f'brightness_analysis_{args.split}.png')
        plot_brightness_analysis(results, plot_path)
        print(f"\nPlot saved to {plot_path}")
    
    elif args.mode == 'visualize':
        # 단일 이미지 시각화
        print("\n=== Visualizing Sample Images ===")
        n_samples = min(10, len(dataset))
        for i in range(n_samples):
            sample_list, label = dataset[i]
            sample = {}
            modals = ['img', 'depth', 'lidar', 'event']
            for j, modal in enumerate(modals):
                if j < len(sample_list):
                    sample[modal] = sample_list[j]
            
            # RGB 파일 경로 가져오기
            rgb_file_path = str(dataset.files[i]) if hasattr(dataset, 'files') and i < len(dataset.files) else None
            
            save_path = os.path.join(args.output, f'sample_{i:04d}.png')
            visualize_modalities(sample, label, palette, save_path, rgb_file_path=rgb_file_path)
            print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
