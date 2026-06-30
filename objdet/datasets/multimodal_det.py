"""
Multimodal Object Detection Dataset (COCO format).

Two path-resolution modes are supported:

(A) modalities-map mode  [preferred for the poongsan indoor dataset]
    Each COCO image entry carries a per-image ``modalities`` dict mapping a
    modality key (``rgb`` / ``thermal_aligned`` / ``depth_map_lidar`` / ...) to a
    path **relative to a single DATASET ROOT**. ``modality_keys`` maps the model's
    modality names (``img`` / ``thermal`` / ``lidar``) onto those json keys.
    Frames missing any requested modality (key absent or file not on disk) are
    dropped → only the 3-modality intersection is used (lidar coverage is partial).

(B) parallel-root mode  [legacy]
    Every modality shares the same ``file_name`` but lives under a different ROOT
    directory given in ``modality_roots``.

Config (mode A) example:
  DATASET:
    ROOT: /drone_nas/drone/dataset
    ANNOTATION_TRAIN: .../det_train.json
    ANNOTATION_VAL:   .../det_val.json
    MODALS: ['img', 'lidar', 'thermal']
    MODALITY_KEYS: { img: rgb, thermal: thermal_aligned, lidar: depth_map_lidar }
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import torchvision.transforms.functional as TF


def letterbox_params(orig_w, orig_h, target_w, target_h):
    """Aspect-preserving resize+center-pad params. Returns (r, pad_x, pad_y) such that
    a box at original (x,y) maps to model-input (x*r+pad_x, y*r+pad_y)."""
    r = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = round(orig_w * r), round(orig_h * r)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    return r, pad_x, pad_y


def rescale_boxes_to_orig(boxes, orig_h, orig_w, in_h, in_w, resize_mode='stretch'):
    """Invert the dataset resize: map predicted boxes (model-input px, xyxy) back to
    original image px. `boxes` is a tensor (N,4); returns a new tensor. Used by eval/COCO
    so it matches the dataset's train/val resize_mode exactly."""
    boxes = boxes.clone()
    if resize_mode == 'letterbox':
        r, pad_x, pad_y = letterbox_params(orig_w, orig_h, in_w, in_h)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / r
    else:  # stretch
        boxes[:, [0, 2]] *= orig_w / in_w
        boxes[:, [1, 3]] *= orig_h / in_h
    return boxes


class MultiModalDetDataset(Dataset):
    def __init__(
        self,
        annotation_path: str,
        modality_roots: Optional[Dict[str, str]] = None,
        img_size: Tuple[int, int] = (1024, 1024),
        transform=None,
        modals: Optional[List[str]] = None,
        min_area: float = 0.0,
        root: Optional[str] = None,
        modality_keys: Optional[Dict[str, str]] = None,
        require_all_modalities: bool = True,
        resize_mode: str = 'stretch',
        drop_empty: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.img_size = img_size  # (H, W)
        self.transform = transform
        self.min_area = min_area
        self.root = root
        self.modality_keys = modality_keys
        self.require_all_modalities = require_all_modalities
        # resize_mode: 'stretch' (force to img_size, aspect distorted) or 'letterbox'
        # (aspect-preserving resize + center pad). 'letterbox' removes the 640x480→1024
        # distortion; eval must invert with the same params (see letterbox_params()).
        self.resize_mode = resize_mode
        # drop_empty: skip images whose annotation set is empty (no non-crowd box).
        # Prevents all-background frames from collapsing FCOS confidence (the v1 failure).
        self.drop_empty = drop_empty

        # ── Resolve mode ──────────────────────────────────────────────
        if modality_keys is not None:
            self.mode = 'modalities_map'
            self.modals = modals or list(modality_keys.keys())
            if root is None:
                raise ValueError("modalities_map mode requires DATASET.ROOT")
        else:
            if modality_roots is None:
                raise ValueError("parallel-root mode requires modality_roots")
            self.mode = 'parallel_root'
            self.modals = modals or list(modality_roots.keys())
            self.modality_roots = {k: modality_roots[k] for k in self.modals}

        # ── Load COCO ─────────────────────────────────────────────────
        with open(annotation_path, 'r') as f:
            coco = json.load(f)

        self.categories = coco['categories']
        self.cat_id_to_idx, self.idx_to_cat = {}, {}
        for idx, cat in enumerate(sorted(self.categories, key=lambda c: c['id'])):
            self.cat_id_to_idx[cat['id']] = idx
            self.idx_to_cat[idx] = cat
        self.n_classes = len(self.categories)
        self.class_names = [self.idx_to_cat[i]['name'] for i in range(self.n_classes)]

        self.images = {img['id']: img for img in coco['images']}

        ann_by_img: Dict[int, list] = {}
        for ann in coco.get('annotations', []):
            ann_by_img.setdefault(ann['image_id'], []).append(ann)

        # ── Build sample list (+ intersection filter in map mode) ─────
        self.img_ids: List[int] = []
        self.img_anns: Dict[int, list] = {}
        self.modal_paths: Dict[int, Dict[str, str]] = {}
        n_drop = 0
        for img_id, info in self.images.items():
            paths = self._resolve_paths(info)
            if paths is None:        # missing a required modality
                n_drop += 1
                continue
            anns = ann_by_img.get(img_id, [])
            if self.drop_empty and not any(not a.get('iscrowd', 0) for a in anns):
                n_drop += 1
                continue
            self.img_ids.append(img_id)
            self.img_anns[img_id] = anns
            self.modal_paths[img_id] = paths
        self.img_ids.sort()

        if verbose:
            print(f"[MultiModalDetDataset] mode={self.mode} modals={self.modals} "
                  f"kept={len(self.img_ids)} dropped={n_drop} "
                  f"(require_all_modalities={require_all_modalities})")

    # ──────────────────────────────────────────────────────────────────
    def _resolve_paths(self, info: dict) -> Optional[Dict[str, str]]:
        """Return {modal: abs_path} or None if a required modality is missing."""
        paths = {}
        if self.mode == 'modalities_map':
            mod_map = info.get('modalities', {})
            for modal in self.modals:
                key = self.modality_keys[modal]
                rel = mod_map.get(key)
                if not rel:
                    return None
                ap = os.path.join(self.root, rel)
                if self.require_all_modalities and not os.path.exists(ap):
                    return None
                paths[modal] = ap
        else:
            for modal in self.modals:
                ap = os.path.join(self.modality_roots[modal], info['file_name'])
                if self.require_all_modalities and not os.path.exists(ap):
                    return None
                paths[modal] = ap
        return paths

    def __len__(self) -> int:
        return len(self.img_ids)

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        return np.array(Image.open(path).convert('RGB'))

    def __getitem__(self, idx: int) -> dict:
        img_id = self.img_ids[idx]
        info = self.images[img_id]
        orig_h, orig_w = info['height'], info['width']
        paths = self.modal_paths[img_id]

        images = {modal: self._load_image(paths[modal]) for modal in self.modals}

        # annotations → (x1,y1,x2,y2), labels
        bboxes, labels = [], []
        for ann in self.img_anns[img_id]:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w * h < self.min_area:
                continue
            bboxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[ann['category_id']])

        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        if self.transform is not None:
            images, bboxes, labels = self.transform(images, bboxes, labels)

        # Resize to model input. Scale by the ACTUAL (post-aug) image size, not the COCO
        # width/height — random_crop can change it (else bboxes misalign once aug is on).
        target_h, target_w = self.img_size
        cur_h, cur_w = images[self.modals[0]].shape[:2]

        sample = {}
        if self.resize_mode == 'letterbox':
            r, pad_x, pad_y = letterbox_params(cur_w, cur_h, target_w, target_h)
            new_w, new_h = max(1, round(cur_w * r)), max(1, round(cur_h * r))
            for modal in self.modals:
                img = images[modal]
                img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                img = img.resize((new_w, new_h), Image.BILINEAR)
                canvas = Image.new('RGB', (target_w, target_h), (114, 114, 114))
                canvas.paste(img, (pad_x, pad_y))
                sample[modal] = TF.to_tensor(canvas)
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * r + pad_x
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * r + pad_y
        else:  # 'stretch' — original behaviour (aspect distorted)
            scale_x = target_w / cur_w
            scale_y = target_h / cur_h
            for modal in self.modals:
                img = images[modal]
                img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                img = img.resize((target_w, target_h), Image.BILINEAR)
                sample[modal] = TF.to_tensor(img)
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] *= scale_x
                bboxes[:, [1, 3]] *= scale_y

        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, target_w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, target_h)

        sample['bboxes'] = torch.from_numpy(bboxes)
        sample['labels'] = torch.from_numpy(labels)
        sample['image_id'] = img_id
        sample['orig_size'] = torch.tensor([orig_h, orig_w])
        sample['file_name'] = info['file_name']
        return sample

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        modals = [k for k in batch[0].keys()
                  if isinstance(batch[0][k], torch.Tensor) and batch[0][k].dim() == 3]
        collated = {m: torch.stack([b[m] for b in batch]) for m in modals}
        collated['bboxes'] = [b['bboxes'] for b in batch]
        collated['labels'] = [b['labels'] for b in batch]
        collated['image_id'] = [b['image_id'] for b in batch]
        collated['orig_size'] = torch.stack([b['orig_size'] for b in batch])
        collated['file_name'] = [b['file_name'] for b in batch]
        return collated
