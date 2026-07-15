import os
import glob
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io

cv2.setNumThreads(0)  # DataLoader workers: avoid cv2 thread oversubscription


class MUSES(Dataset):
    """MUSES (MUlti-SEnsor Semantic perception, ETH Zurich) — 19 Cityscapes trainId classes.

    Interface is a drop-in match for semseg/datasets/deliver.py (DELIVER):
      __init__(root, split, transform, modals)  ->  __getitem__ returns ([modal tensors], label)

    Layout (as shipped/generated on B200):
      {root}/frame_camera/{split}/{weather}/{tod}/{stem}_frame_camera.png        RGB uint8
      {root}/gt_semantic/{split}/{weather}/{tod}/{stem}_gt_labelTrainIds.png     GT uint8 (0-18, 255=ignore)
      {root}/projected_to_rgb/lidar/{split}/{weather}/{tod}/{stem}_lidar.png     uint16 PNG
      {root}/projected_to_rgb/event_camera/{split}/{weather}/{tod}/{stem}_event_camera.png   uint8 PNG
      {root}/projected_to_rgb/radar/{split}/{weather}/{tod}/{stem}_radar.png     uint16 PNG
    weather in {clear, fog, rain, snow}, tod in {day, night}.
    Splits: train 1500 / val 250 / test 750 (test GT is WITHHELD by the benchmark -> unusable here).

    NOTE (uint16 lidar/radar): the MUSES SDK stores the float projection as
    uint16 via `encoded = (value + 100) * 150`  (scripts/project_sensors_to_rgb.py
    -> processing/utils.py:rescale_and_shift_image). It MUST be read with
    cv2.IMREAD_UNCHANGED and decoded as `value = encoded / 150 - 100`.
    torchvision.io.read_image() CANNOT be used: it yields a UInt16 tensor whose
    ops are unimplemented ('"min_all" not implemented for UInt16'), and an 8-bit
    cast would destroy the range/height precision.
    Channel order (processing/utils.py:create_image_from_point_cloud, preserved by
    the cv2.imwrite -> cv2.imread round-trip) is [range, intensity, height];
    height is signed and a decoded value of exactly 0 in all channels means
    "no lidar return" (only ~6.7% of pixels carry a point).

    Aspect ratio: MUSES frames are 1080x1920 (16:9). Every modality + the label is
    letterboxed to a square (pad top/bottom, label padded with ignore=255) inside
    __getitem__, so the shared augmentations_mm Resize/RandomResizedCrop then land
    on an exact NxN that is divisible by the ViT patch size. Without this, the val
    Resize((1024,1024)) would emit 1024x1820, and 1820/16 is not an integer.
    """

    # Cityscapes 19 trainId classes (MUSES gt_semantic *_gt_labelTrainIds.png)
    CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole",
               "traffic light", "traffic sign", "vegetation", "terrain", "sky",
               "person", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle"]

    PALETTE = torch.tensor([
        [128, 64, 128],    # 0  road
        [244, 35, 232],    # 1  sidewalk
        [70, 70, 70],      # 2  building
        [102, 102, 156],   # 3  wall
        [190, 153, 153],   # 4  fence
        [153, 153, 153],   # 5  pole
        [250, 170, 30],    # 6  traffic light
        [220, 220, 0],     # 7  traffic sign
        [107, 142, 35],    # 8  vegetation
        [152, 251, 152],   # 9  terrain
        [70, 130, 180],    # 10 sky
        [220, 20, 60],     # 11 person
        [255, 0, 0],       # 12 rider
        [0, 0, 142],       # 13 car
        [0, 0, 70],        # 14 truck
        [0, 60, 100],      # 15 bus
        [0, 80, 100],      # 16 train
        [0, 0, 230],       # 17 motorcycle
        [119, 11, 32],     # 18 bicycle
    ])

    # ---- uint16 PNG codec (MUSES SDK constants; do not change) ----
    PNG_SCALE = 150.0
    PNG_SHIFT = 100.0

    # ---- physical -> [0,1] normalisation (from 200-train-image statistics) ----
    # range     p50 14.6 m, p99 89.1 m, max 199.9 m -> clip at 100 m
    # intensity p50 24,     p99 146,    max 255     -> native 0-255
    # height    p1 -1.68 m, p99 9.27 m, min -11.2, max 40.6 -> clip [-10, 30] m
    LIDAR_RANGE_MAX = 100.0
    LIDAR_INTENSITY_MAX = 255.0
    LIDAR_HEIGHT_MIN = -10.0
    LIDAR_HEIGHT_MAX = 30.0
    # event counts are heavy-tailed (median nonzero 1, p99 5, max 255): a plain
    # /255 would squash typical events to ~0.004, so compress with log1p.
    EVENT_COUNT_MAX = 255.0

    # modal name -> (projected_to_rgb subdir, filename suffix)
    _PROJ = {
        'lidar': ('lidar', '_lidar'),
        'event': ('event_camera', '_event_camera'),
        'radar': ('radar', '_radar'),
    }
    WEATHER = ('clear', 'fog', 'rain', 'snow')
    TIME_OF_DAY = ('day', 'night')

    @staticmethod
    def decode_segmap(label, palette=None):
        """trainId label (0-18, 255=ignore) -> RGB uint8 (H, W, 3)."""
        if palette is None:
            palette = MUSES.PALETTE
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        h, w = label.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(len(palette)):
            p = palette[cls_id]
            out[label == cls_id] = p.cpu().numpy() if isinstance(p, torch.Tensor) else p
        out[label == 255] = [0, 0, 0]
        return out

    def __init__(self, root: str = 'data/MUSES', split: str = 'train', transform=None,
                 modals=['img', 'lidar', 'event'], case=None, return_meta: bool = False) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        if split == 'test':
            # gt_semantic/test does not exist: MUSES withholds test GT for the
            # public benchmark. train_reliadino.py catches this and sets
            # testset=None, so training evaluates on val only.
            raise FileNotFoundError(
                "MUSES test GT is withheld by the benchmark (gt_semantic/test has 0 files); "
                "train/val only.")
        self.root = root
        self.split = split
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.return_meta = return_meta

        unknown = [m for m in modals if m != 'img' and m not in self._PROJ]
        if unknown:
            raise ValueError(f"MUSES: unsupported modal(s) {unknown}. "
                             f"Available: 'img', {list(self._PROJ)} (MUSES has NO depth).")

        self.files = sorted(glob.glob(os.path.join(root, 'frame_camera', split, '*', '*', '*.png')))
        if case is not None:
            assert case in self.WEATHER + self.TIME_OF_DAY, \
                f"Case must be one of {self.WEATHER + self.TIME_OF_DAY}."
            self.files = [f for f in self.files if f'{os.sep}{case}{os.sep}' in f]
        if not self.files:
            raise FileNotFoundError(
                f"No images found in {root}/frame_camera/{split}/*/*/*.png")
        print(f"Found {len(self.files)} {split} {case if case else ''} images.")

    def __len__(self) -> int:
        return len(self.files)

    # ---- path helpers ----
    def _sibling(self, rgb: str, kind: str) -> str:
        """rgb -> path of the matching modality/label file (same weather/tod/stem)."""
        rel = os.path.relpath(rgb, os.path.join(self.root, 'frame_camera'))
        cond_dir, fname = os.path.split(rel)              # 'train/clear/day', '<stem>_frame_camera.png'
        stem = fname[:-len('_frame_camera.png')]
        if kind == 'mask':
            return os.path.join(self.root, 'gt_semantic', cond_dir, stem + '_gt_labelTrainIds.png')
        sub, suffix = self._PROJ[kind]
        return os.path.join(self.root, 'projected_to_rgb', sub, cond_dir, stem + suffix + '.png')

    # ---- readers: every modality is returned as float/uint8 on a 0-255 scale, so
    #      that augmentations_mm.Normalize's `/= 255` lands it in [0, 1] (the same
    #      contract DELIVER's uint8 PNGs satisfy).
    def _read_uint16_proj(self, path: str) -> np.ndarray:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"MUSES: cannot read {path}")
        if raw.dtype != np.uint16:
            raise ValueError(f"MUSES: expected uint16 PNG, got {raw.dtype} for {path}")
        return raw.astype(np.float32) / self.PNG_SCALE - self.PNG_SHIFT

    def _open_lidar(self, path: str) -> Tensor:
        v = self._read_uint16_proj(path)                       # (H,W,3) [range, intensity, height]
        valid = (v[:, :, 0] > 0).astype(np.float32)            # exact: encoded 15000 -> 0.0
        rng = np.clip(v[:, :, 0] / self.LIDAR_RANGE_MAX, 0.0, 1.0)
        inten = np.clip(v[:, :, 1] / self.LIDAR_INTENSITY_MAX, 0.0, 1.0)
        hgt = np.clip((v[:, :, 2] - self.LIDAR_HEIGHT_MIN)
                      / (self.LIDAR_HEIGHT_MAX - self.LIDAR_HEIGHT_MIN), 0.0, 1.0)
        out = np.stack([rng, inten, hgt], axis=0) * valid[None]   # empty pixels -> 0 in all channels
        return torch.from_numpy(np.ascontiguousarray(out * 255.0))

    def _open_radar(self, path: str) -> Tensor:
        # radar uses the same (value+100)*150 codec; channels [range, intensity, height]
        return self._open_lidar(path)

    def _open_event(self, path: str) -> Tensor:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)           # (H,W,3) uint8 [pos, neg, 0]
        if raw is None:
            raise FileNotFoundError(f"MUSES: cannot read {path}")
        e = np.log1p(raw.astype(np.float32)) / np.log1p(self.EVENT_COUNT_MAX)
        e = e.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(e * 255.0))

    @staticmethod
    def _pad_to_square(t: Tensor, fill: float) -> Tensor:
        """(C,H,W) -> (C,S,S), S=max(H,W); pads the short side symmetrically."""
        _, h, w = t.shape
        if h == w:
            return t
        if w > h:
            top = (w - h) // 2
            padding = [0, top, 0, w - h - top]      # l, t, r, b
        else:
            left = (h - w) // 2
            padding = [left, 0, h - w - left, 0]
        return TF.pad(t, padding, fill=fill)

    def __getitem__(self, index: int) -> Tuple[list, Tensor]:
        rgb = str(self.files[index])
        lbl_path = self._sibling(rgb, 'mask')

        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]            # uint8 RGB
        H, W = sample['img'].shape[1:]
        for m in self.modals:
            if m == 'img':
                continue
            p = self._sibling(rgb, m)
            x = self._open_event(p) if m == 'event' else self._open_lidar(p)
            if x.shape[1:] != (H, W):
                x = TF.resize(x, [H, W], TF.InterpolationMode.NEAREST)
            sample[m] = x

        label = io.read_image(lbl_path)[0, ...].unsqueeze(0)   # (1,H,W) uint8, already 0-18 + 255
        sample['mask'] = label

        # letterbox to square BEFORE the shared transforms (see class docstring)
        for k in sample:
            sample[k] = self._pad_to_square(sample[k], fill=self.ignore_label if k == 'mask' else 0)

        if self.return_meta:
            orig_label = label.clone().squeeze().long()

        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        out = [sample[k] for k in self.modals]

        if self.return_meta:
            meta = {
                'stem': Path(rgb).stem,
                'orig_h': int(H),
                'orig_w': int(W),
                'orig_label': orig_label,
                'paths': {m: (rgb if m == 'img' else self._sibling(rgb, m)) for m in self.modals},
            }
            return out, label, meta
        return out, label

    def encode(self, label: np.ndarray) -> Tensor:
        return torch.from_numpy(label)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from semseg.augmentations_mm import get_train_augmentation

    root = '/NHNHOME/ailab/Workspaces/jemo_maeng/dset/MUSES'
    t = get_train_augmentation((1024, 1024), seg_fill=255)
    ds = MUSES(root, 'train', t, ['img', 'lidar', 'event'])
    for i, (sample, lbl) in enumerate(DataLoader(ds, batch_size=2, num_workers=2)):
        print([s.shape for s in sample], lbl.shape, torch.unique(lbl))
        if i > 2:
            break
