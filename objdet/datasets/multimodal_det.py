"""
Multimodal Object Detection Dataset (COCO format).

COCO JSON annotation + per-modality image root 구조를 지원.
모달리티별 이미지는 동일한 file_name을 공유하며, config의 ROOT prefix로 경로를 결정.

Config 예시:
  DATASET:
    ANNOTATION_TRAIN: /path/to/train.json
    ANNOTATION_VAL: /path/to/val.json
    MODALITIES:
      img:
        ROOT: /path/to/data/zed
      lidar:
        ROOT: /path/to/data/lidar_processed
      thermal:
        ROOT: /path/to/data/thermal_processed
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torchvision.transforms.functional as TF


class MultiModalDetDataset(Dataset):
    """
    COCO-format multimodal object detection dataset.

    Each image has the same file_name across modalities, loaded from
    different ROOT directories specified in the config.

    Args:
        annotation_path: Path to COCO JSON annotation file.
        modality_roots: Dict[str, str] mapping modality name → image root dir.
            e.g. {'img': '/data/zed', 'lidar': '/data/lidar', 'thermal': '/data/thermal'}
        img_size: Resize target (H, W). All modalities resized to this.
        transform: Optional augmentation callable. Should handle bbox-aware transforms.
            Expected signature: transform(images: Dict[str, np.ndarray],
                                          bboxes: np.ndarray, labels: np.ndarray)
                                → (images, bboxes, labels)
        modals: List of modality keys to load. If None, uses all keys from modality_roots.
        min_area: Minimum bbox area to keep (filters tiny annotations).
    """

    def __init__(
        self,
        annotation_path: str,
        modality_roots: Dict[str, str],
        img_size: Tuple[int, int] = (1024, 1024),
        transform=None,
        modals: Optional[List[str]] = None,
        min_area: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size  # (H, W)
        self.transform = transform
        self.modals = modals or list(modality_roots.keys())
        self.modality_roots = {k: modality_roots[k] for k in self.modals}
        self.min_area = min_area

        # Load COCO JSON
        with open(annotation_path, 'r') as f:
            coco = json.load(f)

        # Build category mapping: original_id → contiguous 0-based index
        self.categories = coco['categories']
        self.cat_id_to_idx = {}
        self.idx_to_cat = {}
        for idx, cat in enumerate(sorted(self.categories, key=lambda c: c['id'])):
            self.cat_id_to_idx[cat['id']] = idx
            self.idx_to_cat[idx] = cat

        self.n_classes = len(self.categories)
        self.class_names = [self.idx_to_cat[i]['name'] for i in range(self.n_classes)]

        # Build image lookup
        self.images = {img['id']: img for img in coco['images']}

        # Group annotations by image_id
        self.img_ids = []
        self.img_anns = {}
        ann_by_img = {}
        for ann in coco.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in ann_by_img:
                ann_by_img[img_id] = []
            ann_by_img[img_id].append(ann)

        # Keep images that have annotations (or all images if no annotations)
        for img_id, img_info in self.images.items():
            self.img_ids.append(img_id)
            self.img_anns[img_id] = ann_by_img.get(img_id, [])

        # Sort for reproducibility
        self.img_ids.sort()

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_image(self, file_name: str, modal: str) -> np.ndarray:
        """Load image from modality root. Returns (H, W, 3) uint8."""
        root = self.modality_roots[modal]
        path = os.path.join(root, file_name)
        img = Image.open(path).convert('RGB')
        return np.array(img)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        file_name = img_info['file_name']
        orig_h, orig_w = img_info['height'], img_info['width']

        # Load all modalities
        images = {}
        for modal in self.modals:
            images[modal] = self._load_image(file_name, modal)

        # Parse annotations → bboxes (x1, y1, x2, y2), labels
        anns = self.img_anns[img_id]
        bboxes = []
        labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']  # COCO format: (x, y, w, h)
            area = w * h
            if area < self.min_area:
                continue
            bboxes.append([x, y, x + w, y + h])  # → (x1, y1, x2, y2)
            labels.append(self.cat_id_to_idx[ann['category_id']])

        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        # Apply augmentation (bbox-aware)
        if self.transform is not None:
            images, bboxes, labels = self.transform(images, bboxes, labels)

        # Resize all modalities and scale bboxes
        target_h, target_w = self.img_size
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        sample = {}
        for modal in self.modals:
            img = images[modal]
            img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            img = img.resize((target_w, target_h), Image.BILINEAR)
            img = TF.to_tensor(img)  # (3, H, W), float32 [0, 1]
            sample[modal] = img

        # Scale bboxes to resized image
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] *= scale_x
            bboxes[:, [1, 3]] *= scale_y
            # Clip to image bounds
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, target_w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, target_h)

        sample['bboxes'] = torch.from_numpy(bboxes)       # (N, 4) x1y1x2y2
        sample['labels'] = torch.from_numpy(labels)        # (N,)
        sample['image_id'] = img_id
        sample['orig_size'] = torch.tensor([orig_h, orig_w])
        sample['file_name'] = file_name

        return sample

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """
        Custom collate for variable-length bboxes.
        Modality images are stacked; bboxes/labels remain as lists.
        """
        modals = [k for k in batch[0].keys()
                  if isinstance(batch[0][k], torch.Tensor) and batch[0][k].dim() == 3]

        collated = {}
        for modal in modals:
            collated[modal] = torch.stack([b[modal] for b in batch])

        collated['bboxes'] = [b['bboxes'] for b in batch]
        collated['labels'] = [b['labels'] for b in batch]
        collated['image_id'] = [b['image_id'] for b in batch]
        collated['orig_size'] = torch.stack([b['orig_size'] for b in batch])
        collated['file_name'] = [b['file_name'] for b in batch]

        return collated
