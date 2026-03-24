import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation

class DELIVER(Dataset):
    """
    num_classes: 25
    """
    CLASSES = ["Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road", "SideWalk", "Vegetation", 
                "Cars", "Wall", "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GroundRail", 
                "TrafficLight", "Static", "Dynamic", "Water", "Terrain", "TwoWheeler", "Bus", "Truck"]

    # 원본 GT 인덱스와 매칭되는 Palette (1-based 인덱싱)
    # 원본 GT: 클래스 1, 2, 3, ..., 25 (ignore: 255)
    # 학습용 변환: label -= 1 (0-based: 0, 1, 2, ..., 24)
    # Palette[0] = 클래스 1 (Building), Palette[1] = 클래스 2 (Fence), ...
    PALETTE = torch.tensor([[70, 70, 70],        # 0: Building (원본 GT 클래스 1)
            [100, 40, 40],                       # 1: Fence (원본 GT 클래스 2)
            [55, 90, 80],                        # 2: Other (원본 GT 클래스 3)
            [220, 20, 60],                       # 3: Pedestrian (원본 GT 클래스 4)
            [153, 153, 153],                     # 4: Pole (원본 GT 클래스 5)
            [157, 234, 50],                      # 5: RoadLine (원본 GT 클래스 6)
            [128, 64, 128],                      # 6: Road (원본 GT 클래스 7)
            [244, 35, 232],                      # 7: SideWalk (원본 GT 클래스 8)
            [107, 142, 35],                      # 8: Vegetation (원본 GT 클래스 9)
            [0, 0, 142],                         # 9: Cars (원본 GT 클래스 10)
            [102, 102, 156],                     # 10: Wall (원본 GT 클래스 11)
            [220, 220, 0],                       # 11: TrafficSign (원본 GT 클래스 12)
            [70, 130, 180],                      # 12: Sky (원본 GT 클래스 13)
            [81, 0, 81],                         # 13: Ground (원본 GT 클래스 14)
            [150, 100, 100],                     # 14: Bridge (원본 GT 클래스 15)
            [230, 150, 140],                     # 15: RailTrack (원본 GT 클래스 16)
            [180, 165, 180],                     # 16: GroundRail (원본 GT 클래스 17)
            [250, 170, 30],                      # 17: TrafficLight (원본 GT 클래스 18)
            [110, 190, 160],                     # 18: Static (원본 GT 클래스 19)
            [170, 120, 50],                      # 19: Dynamic (원본 GT 클래스 20)
            [45, 60, 150],                       # 20: Water (원본 GT 클래스 21)
            [145, 170, 100],                     # 21: Terrain (원본 GT 클래스 22)
            [  0,  0, 230],                      # 22: TwoWheeler (원본 GT 클래스 23)
            [  0, 60, 100],                      # 23: Bus (원본 GT 클래스 24)
            [  0,  0, 70],                       # 24: Truck (원본 GT 클래스 25)
            ])
    
    @staticmethod
    def decode_segmap(label, palette=None):
        """
        학습용 label (0-based: 0~24)을 원본 GT 색상으로 변환
        Args:
            label: 학습용 label tensor/array (0-based, 0~24)
            palette: 사용할 palette (None이면 기본 PALETTE 사용)
        Returns:
            colored_label: RGB 이미지 (H, W, 3)
        """
        if palette is None:
            palette = DELIVER.PALETTE
        
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        
        h, w = label.shape
        colored_label = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 학습용 label (0-based)을 palette 인덱스로 직접 사용
        for cls_id in range(len(palette)):
            mask = label == cls_id
            colored_label[mask] = palette[cls_id].cpu().numpy() if isinstance(palette, torch.Tensor) else palette[cls_id]
        
        # ignore label (255) 처리 - 검은색으로 표시
        ignore_mask = label == 255
        colored_label[ignore_mask] = [0, 0, 0]
        
        return colored_label
    
    @staticmethod
    def decode_segmap_from_original(label_original, palette=None):
        """
        원본 GT label (1-based: 1~25)을 원본 GT 색상으로 변환
        Args:
            label_original: 원본 GT label tensor/array (1-based, 1~25, ignore: 255)
            palette: 사용할 palette (None이면 기본 PALETTE 사용)
        Returns:
            colored_label: RGB 이미지 (H, W, 3)
        """
        if palette is None:
            palette = DELIVER.PALETTE
        
        if isinstance(label_original, torch.Tensor):
            label_original = label_original.cpu().numpy()
        
        h, w = label_original.shape
        colored_label = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 원본 GT는 1-based이므로 palette 인덱스는 (label - 1)
        for cls_id in range(1, len(palette) + 1):  # 1~25
            mask = label_original == cls_id
            colored_label[mask] = palette[cls_id - 1].cpu().numpy() if isinstance(palette, torch.Tensor) else palette[cls_id - 1]
        
        # ignore label (255) 처리 - 검은색으로 표시
        ignore_mask = label_original == 255
        colored_label[ignore_mask] = [0, 0, 0]
        
        return colored_label
    
    def __init__(self, root: str = 'data/DELIVER', split: str = 'train', transform = None,
                 modals = ['img'], case = None, return_meta: bool = False) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.return_meta = return_meta
        self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))
        # --- debug
        # self.files = sorted(glob.glob(os.path.join(*[root, 'img', '*', split, '*', '*.png'])))[:100]
        # --- split as case
        if case is not None:
            assert case in ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'], "Case name not available."
            _temp_files = [f for f in self.files if case in f]
            self.files = _temp_files
        if not self.files:
            raise Exception(f"No images found in {root}/img/*/{split}/*/*.png")
        print(f"Found {len(self.files)} {split} {case} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        rgb = str(self.files[index])
        x1 = rgb.replace('/img', '/hha').replace('_rgb', '_depth')
        x2 = rgb.replace('/img', '/lidar').replace('_rgb', '_lidar')
        x3 = rgb.replace('/img', '/event').replace('_rgb', '_event')
        lbl_path = rgb.replace('/img', '/semantic').replace('_rgb', '_semantic')

        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        H, W = sample['img'].shape[1:]
        if 'depth' in self.modals:
            sample['depth'] = self._open_img(x1)
        if 'lidar' in self.modals:
            sample['lidar'] = self._open_img(x2)
        if 'event' in self.modals:
            eimg = self._open_img(x3)
            sample['event'] = TF.resize(eimg, (H, W), TF.InterpolationMode.NEAREST)
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        label[label==255] = 0
        label -= 1  # 1-25 → 0-24, 0 → 255 (uint8 underflow = ignore)
        sample['mask'] = label

        # Save original-size label before transform (for metric computation at orig size)
        if self.return_meta:
            orig_label = label.clone().squeeze().long()  # (H, W), values 0-24 + 255

        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]

        if self.return_meta:
            meta = {
                'stem': Path(rgb).stem,
                'orig_h': int(H),
                'orig_w': int(W),
                'orig_label': orig_label,
                'paths': {'img': rgb, 'depth': x1, 'lidar': x2, 'event': x3},
            }
            return sample, label, meta
        return sample, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)


if __name__ == '__main__':
    cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    traintransform = get_train_augmentation((1024, 1024), seg_fill=255)
    for case in cases:

        trainset = DELIVER(transform=traintransform, split='val', case=case)
        trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

        for i, (sample, lbl) in enumerate(trainloader):
            print(torch.unique(lbl))