"""
MULTIAQUA dataset (DELIVER-style interface).

Dataset root: .../MULTIAQUA_night
- train.txt, val.txt, test.txt: one stem per line (e.g. bl1_0_051235)
- MULTIAQUA_night/annotations/: segmentation masks {stem}.png
- MULTIAQUA_night/data/zed/: RGB images {stem}.png
- MULTIAQUA_night/data/lidar_processed/: {stem}_lidar.png, {stem}_lidar_color.png
- MULTIAQUA_night/data/thermal_processed/: {stem}_thermal.png
"""

import os
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import io
from pathlib import Path
from typing import Tuple, List

from semseg.augmentations_mm import get_train_augmentation
from torch.utils.data import DataLoader


class MULTIAQUA(Dataset):
    """
    MULTIAQUA multimodal semantic segmentation.
    modals: ['img'], ['img','lidar'], ['img','thermal'], ['img','lidar','thermal'], etc.
    """

    # Same 25 classes as DELIVER for compatibility; adjust if MULTIAQUA uses different taxonomy
    CLASSES = [
        "Building", "Fence", "Other", "Pedestrian", "Pole", "RoadLine", "Road", "SideWalk", "Vegetation",
        "Cars", "Wall", "TrafficSign", "Sky", "Ground", "Bridge", "RailTrack", "GroundRail",
        "TrafficLight", "Static", "Dynamic", "Water", "Terrain", "TwoWheeler", "Bus", "Truck",
    ]

    PALETTE = torch.tensor([
        [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60], [153, 153, 153],
        [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35], [0, 0, 142],
        [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81], [150, 100, 100],
        [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160], [170, 120, 50],
        [45, 60, 150], [145, 170, 100], [0, 0, 230], [0, 60, 100], [0, 0, 70],
    ], dtype=torch.uint8)

    def __init__(
        self,
        root: str = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night",
        split: str = "train",
        transform=None,
        modals: List[str] = None,
    ) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.modals = modals if modals is not None else ["img"]
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        # Paths under root
        self.data_root = self.root / "MULTIAQUA_night"
        self.rgb_dir = self.data_root / "data" / "zed"
        self.lidar_dir = self.data_root / "data" / "lidar_processed"
        self.thermal_dir = self.data_root / "data" / "thermal_processed"
        self.ann_dir = self.data_root / "annotations"

        # Load stem list from {split}.txt
        split_file = self.root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            stems = [line.strip() for line in f if line.strip()]
        # Keep only stems that have RGB and annotation (required)
        self.stems = []
        for s in stems:
            rgb_path = self.rgb_dir / f"{s}.png"
            ann_path = self.ann_dir / f"{s}.png"
            if rgb_path.exists() and ann_path.exists():
                self.stems.append(s)
        if not self.stems:
            raise Exception(f"No samples found for {split} in {self.root}")
        print(f"Found {len(self.stems)} {split} images.")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> Tuple:
        stem = self.stems[index]
        rgb_path = self.rgb_dir / f"{stem}.png"
        lbl_path = self.ann_dir / f"{stem}.png"

        sample = {}
        sample["img"] = io.read_image(str(rgb_path))[:3, ...]
        H, W = sample["img"].shape[1:]

        if "lidar" in self.modals:
            lidar_path = self.lidar_dir / f"{stem}_lidar.png"
            sample["lidar"] = self._open_img(lidar_path, H, W)
        if "thermal" in self.modals:
            thermal_path = self.thermal_dir / f"{stem}_thermal.png"
            sample["thermal"] = self._open_img(thermal_path, H, W)

        label = io.read_image(str(lbl_path))[0, ...].unsqueeze(0)
        label[label == 255] = 0
        label = label - 1
        sample["mask"] = label

        if self.transform:
            sample = self.transform(sample)
        label = sample["mask"]
        del sample["mask"]
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        return sample, label

    def _open_img(self, path: Path, H: int, W: int) -> Tensor:
        if not path.exists():
            return torch.zeros(3, H, W, dtype=torch.uint8)
        img = io.read_image(str(path))
        C, h, w = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        if (h, w) != (H, W):
            img = TF.resize(img, (H, W), TF.InterpolationMode.NEAREST)
        return img

    def encode(self, label: np.ndarray) -> Tensor:
        return torch.from_numpy(label)

    @staticmethod
    def decode_segmap(label, palette=None):
        """0-based label (0~24) to RGB visualization."""
        if palette is None:
            palette = MULTIAQUA.PALETTE
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        h, w = label.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(len(palette)):
            mask = label == cls_id
            colored[mask] = palette[cls_id].numpy() if isinstance(palette, torch.Tensor) else np.array(palette[cls_id])
        colored[label == 255] = [0, 0, 0]
        return colored


if __name__ == "__main__":
    traintransform = get_train_augmentation((1024, 1024), seg_fill=255)
    for split in ["train", "val"]:
        ds = MULTIAQUA(
            root="/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night",
            split=split,
            transform=traintransform,
            modals=["img", "lidar", "thermal"],
        )
        loader = DataLoader(ds, batch_size=2, num_workers=0, drop_last=False)
        for i, (sample, lbl) in enumerate(loader):
            print(split, "sample keys:", [x.shape for x in sample], "label unique:", torch.unique(lbl))
            if i >= 1:
                break
