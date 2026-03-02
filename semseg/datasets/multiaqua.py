"""
MULTIAQUA dataset (DELIVER-style interface).

Ref: https://arxiv.org/pdf/2512.17450
- Sky, Water, Static obstacle, Dynamic obstacle, Recording boat (ignored during training)

Dataset root: .../MULTIAQUA_night
- train.txt, val.txt, test.txt: one stem per line
- MULTIAQUA_night/annotations/: segmentation masks {stem}.png
- MULTIAQUA_night/data/zed/: RGB images {stem}.png
- MULTIAQUA_night/data/lidar_processed/: {stem}_lidar.png
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
from typing import Tuple, List, Optional

from semseg.augmentations_mm import get_train_augmentation
from torch.utils.data import DataLoader


def count_num_classes_from_annotations(
    ann_dir: Path,
    ignore_labels: Tuple[int, ...] = (255,),
    max_files: int = 500,
) -> int:
    """
    Scan annotation images to infer the number of classes.
    Returns max(unique_values) + 1, excluding ignore_labels.
    """
    from PIL import Image
    all_vals = set()
    for f in list(ann_dir.glob("*.png"))[:max_files]:
        arr = np.array(Image.open(str(f)))
        if arr.ndim >= 3:
            arr = arr[:, :, 0]
        for v in np.unique(arr):
            if int(v) not in ignore_labels:
                all_vals.add(int(v))
    if not all_vals:
        return 0
    return max(all_vals) + 1


class MULTIAQUA(Dataset):
    """
    MULTIAQUA multimodal semantic segmentation.
    Paper: https://arxiv.org/pdf/2512.17450
    Classes: Sky, Water, Static obstacle, Dynamic obstacle. Recording boat is ignored during training.
    modals: ['img'], ['img','lidar'], ['img','thermal'], ['img','lidar','thermal'], etc.
    """

    # Annotation: 0=Recording Boat (ignore), 1=Static, 2=Dynamic, 3=Water, 4=Sky
    # n_classes=4: Static, Dynamic, Water, Sky (Recording Boat 제외)
    _BASE_CLASSES = ["Static", "Dynamic", "Water", "Sky"]
    _BASE_PALETTE = torch.tensor([
        [107, 142, 35],   # 0: Static
        [220, 20, 60],    # 1: Dynamic
        [45, 60, 150],   # 2: Water
        [70, 130, 180],   # 3: Sky
    ], dtype=torch.uint8)
    CLASSES = _BASE_CLASSES
    PALETTE = _BASE_PALETTE

    def __init__(
        self,
        root: str = "/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night",
        split: str = "train",
        transform=None,
        modals: List[str] = None,
        n_classes: Optional[int] = None,
        require_annotation: bool = True,
        return_meta: bool = False,
        night_translation: bool = False,
    ) -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.modals = modals if modals is not None else ["img"]
        self.ignore_label = 255
        self.require_annotation = require_annotation
        self.return_meta = return_meta

        # Paths under root
        self.data_root = self.root / "MULTIAQUA_night"
        self.rgb_dir = self.data_root / "data" / "zed"
        self.lidar_dir = self.data_root / "data" / "lidar_processed"
        self.thermal_dir = self.data_root / "data" / "thermal_processed"
        self.ann_dir = self.data_root / "annotations"

        # n_classes=4 (Static, Dynamic, Water, Sky). Recording Boat(0)는 ignore.
        self.n_classes = n_classes if n_classes is not None else 4
        self.CLASSES = self._BASE_CLASSES
        self.PALETTE = self._BASE_PALETTE

        # Night translation: zed_night*, zed_night2* 등 img2img 변환 폴더도 사용
        self.night_translation = night_translation
        if self.night_translation:
            data_dir = self.data_root / "data"
            zed_dirs = sorted([
                d for d in data_dir.iterdir()
                if d.is_dir() and d.name.startswith("zed")
            ])
            if not zed_dirs:
                zed_dirs = [self.rgb_dir]
        else:
            zed_dirs = [self.rgb_dir]

        # Load stem list from {split}.txt
        split_file = self.root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            stems = [line.strip() for line in f if line.strip()]
        # require_annotation=True: RGB+annotation 둘 다 필요 (val, train)
        # require_annotation=False: RGB만 필요 (test inference, annotation 없음)
        # samples: list of (stem, rgb_dir) — 하나의 stem이 여러 zed 변환에 대해 복수 등장 가능
        self.samples = []
        for s in stems:
            if require_annotation:
                ann_path = self.ann_dir / f"{s}.png"
                if not ann_path.exists():
                    continue
            for zed_dir in zed_dirs:
                rgb_path = zed_dir / f"{s}.png"
                if rgb_path.exists():
                    self.samples.append((s, zed_dir))
        if not self.samples:
            raise Exception(f"No samples found for {split} in {self.root} (require_annotation={require_annotation})")
        # Backward compat: self.stems (unique stems only)
        self.stems = list(dict.fromkeys(s for s, _ in self.samples))
        n_variants = len(zed_dirs)
        print(f"Found {len(self.samples)} {split} samples "
              f"({len(self.stems)} stems x {n_variants} RGB variant{'s' if n_variants > 1 else ''}: "
              f"{[d.name for d in zed_dirs]})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        stem, rgb_dir = self.samples[index]
        rgb_path = rgb_dir / f"{stem}.png"

        sample = {}
        sample["img"] = io.read_image(str(rgb_path))[:3, ...]
        H, W = sample["img"].shape[1:]

        if "lidar" in self.modals:
            lidar_path = self.lidar_dir / f"{stem}_lidar.png"
            sample["lidar"] = self._open_img(lidar_path, H, W)
        if "thermal" in self.modals:
            thermal_path = self.thermal_dir / f"{stem}_thermal.png"
            sample["thermal"] = self._open_img(thermal_path, H, W)

        if self.require_annotation:
            lbl_path = self.ann_dir / f"{stem}.png"
            label = io.read_image(str(lbl_path))[0, ...]  # (H, W)
            # MULTIAQUA: 0=Recording Boat(ignore), 1=Static, 2=Dynamic, 3=Water, 4=Sky
            # Output: 0=Static, 1=Dynamic, 2=Water, 3=Sky, 255=ignore
            label = label.numpy().astype(np.int64)
            # 유효 클래스(1~4)만 리매핑, 나머지(0, 5+, 255 등)는 모두 ignore(255)
            orig_label = np.where(
                (label >= 1) & (label <= 4),
                label - 1,   # 1→0, 2→1, 3→2, 4→3
                255           # 0(Recording Boat), 255, 기타 → ignore
            )
            sample["mask"] = torch.from_numpy(orig_label.copy()).unsqueeze(0)  # (1, H, W)
            mh, mw = int(sample["mask"].shape[1]), int(sample["mask"].shape[2])
            assert (mh, mw) == (int(H), int(W)), f"stem={stem} img={H}x{W} mask={mh}x{mw}"
        else:
            # inference only: dummy mask (not used)
            sample["mask"] = torch.zeros(1, H, W, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)
        label = sample["mask"]
        del sample["mask"]
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]

        if self.return_meta:
            meta = {"stem": stem, "orig_h": int(H), "orig_w": int(W)}
            if self.require_annotation:
                meta["orig_label"] = torch.from_numpy(orig_label).long()
            return sample, label, meta
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
        """0-based label to RGB visualization."""
        if palette is None:
            palette = MULTIAQUA._BASE_PALETTE
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
