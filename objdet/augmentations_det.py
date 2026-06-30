"""
Bbox-aware augmentations for multimodal detection.

모든 transform은 동일한 변환을 모든 모달리티에 적용하고,
bounding box도 함께 변환.
"""

import numpy as np
import random
from PIL import Image
from typing import Dict, Tuple


def hflip(
    images: Dict[str, np.ndarray],
    bboxes: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Horizontal flip."""
    flipped = {}
    for k, img in images.items():
        flipped[k] = img[:, ::-1].copy()

    if bboxes.shape[0] > 0:
        w = images[list(images.keys())[0]].shape[1]
        new_bboxes = bboxes.copy()
        new_bboxes[:, 0] = w - bboxes[:, 2]  # x1 = w - x2
        new_bboxes[:, 2] = w - bboxes[:, 0]  # x2 = w - x1
        bboxes = new_bboxes

    return flipped, bboxes, labels


def random_brightness(
    images: Dict[str, np.ndarray],
    bboxes: np.ndarray,
    labels: np.ndarray,
    range: Tuple[float, float] = (0.5, 1.5),
    modals: list = ['img'],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Random brightness adjustment (RGB only by default)."""
    factor = random.uniform(*range)
    result = {}
    for k, img in images.items():
        if k in modals:
            result[k] = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        else:
            result[k] = img
    return result, bboxes, labels


def random_contrast(
    images: Dict[str, np.ndarray],
    bboxes: np.ndarray,
    labels: np.ndarray,
    range: Tuple[float, float] = (0.7, 1.3),
    modals: list = ['img'],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Random contrast around per-image mean (RGB only by default)."""
    factor = random.uniform(*range)
    result = {}
    for k, img in images.items():
        if k in modals:
            mean = img.astype(np.float32).mean()
            result[k] = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        else:
            result[k] = img
    return result, bboxes, labels


def random_crop(
    images: Dict[str, np.ndarray],
    bboxes: np.ndarray,
    labels: np.ndarray,
    min_scale: float = 0.8,
    max_scale: float = 1.0,
    min_iou_with_crop: float = 0.5,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Random crop that keeps bboxes with sufficient IoU with crop region.
    """
    first_img = images[list(images.keys())[0]]
    h, w = first_img.shape[:2]

    scale = random.uniform(min_scale, max_scale)
    crop_h = int(h * scale)
    crop_w = int(w * scale)

    y1 = random.randint(0, h - crop_h)
    x1 = random.randint(0, w - crop_w)
    y2 = y1 + crop_h
    x2 = x1 + crop_w

    cropped = {}
    for k, img in images.items():
        cropped[k] = img[y1:y2, x1:x2].copy()

    if bboxes.shape[0] > 0:
        new_bboxes = bboxes.copy()
        new_bboxes[:, [0, 2]] -= x1
        new_bboxes[:, [1, 3]] -= y1
        new_bboxes[:, [0, 2]] = np.clip(new_bboxes[:, [0, 2]], 0, crop_w)
        new_bboxes[:, [1, 3]] = np.clip(new_bboxes[:, [1, 3]], 0, crop_h)

        # Filter out degenerate boxes
        ws = new_bboxes[:, 2] - new_bboxes[:, 0]
        hs = new_bboxes[:, 3] - new_bboxes[:, 1]
        valid = (ws > 1) & (hs > 1)

        # IoU with crop: check overlap ratio
        orig_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        new_areas = ws * hs
        iou_ratios = new_areas / np.maximum(orig_areas, 1e-6)
        valid = valid & (iou_ratios > min_iou_with_crop)

        bboxes = new_bboxes[valid]
        labels = labels[valid]
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)

    return cropped, bboxes, labels


class DetAugmentation:
    """
    Compose bbox-aware augmentations for detection training.

    Args:
        hflip_prob: Probability of horizontal flip.
        brightness_range: (min, max) brightness factor.
        crop_prob: Probability of random crop.
        crop_scale: (min, max) crop scale.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.6, 1.4),
        contrast_range: Tuple[float, float] = (0.7, 1.3),
        crop_prob: float = 0.5,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
    ):
        self.hflip_prob = hflip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.crop_prob = crop_prob
        self.crop_scale = crop_scale

    def __call__(
        self,
        images: Dict[str, np.ndarray],
        bboxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:

        if random.random() < self.hflip_prob:
            images, bboxes, labels = hflip(images, bboxes, labels)

        if random.random() < 0.5:
            images, bboxes, labels = random_brightness(
                images, bboxes, labels, range=self.brightness_range
            )

        if random.random() < 0.5:
            images, bboxes, labels = random_contrast(
                images, bboxes, labels, range=self.contrast_range
            )

        if random.random() < self.crop_prob:
            images, bboxes, labels = random_crop(
                images, bboxes, labels,
                min_scale=self.crop_scale[0],
                max_scale=self.crop_scale[1],
            )

        return images, bboxes, labels
