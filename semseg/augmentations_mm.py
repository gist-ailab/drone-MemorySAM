import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, sample: list) -> list:
        img, mask = sample['img'], sample['mask']
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            sample = transform(sample)

        return sample


class Normalize:
    def __init__(
        self,
        mean: list = (0.485, 0.456, 0.406),
        std: list = (0.229, 0.224, 0.225),
        thermal_mean: Optional[float] = None,
        thermal_std: Optional[float] = None,
    ):
        self.mean = mean
        self.std = std
        self.thermal_mean = thermal_mean
        self.thermal_std = thermal_std

    def __call__(self, sample: list) -> list:
        for k, v in sample.items():
            if k == 'mask':
                continue
            elif k == 'img':
                sample[k] = sample[k].float()
                sample[k] /= 255
                sample[k] = TF.normalize(sample[k], self.mean, self.std)
            elif k == 'thermal' and self.thermal_mean is not None and self.thermal_std is not None:
                # z-score on raw 0-255 scale (mean/std from cal_meanstd_thermal.py)
                sample[k] = sample[k].float()
                m = (self.thermal_mean,) * 3
                s = (self.thermal_std,) * 3
                sample[k] = TF.normalize(sample[k], m, s)
            else:
                sample[k] = sample[k].float()
                sample[k] /= 255
        return sample


class RandomColorJitter:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            self.brightness = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_brightness(sample['img'], self.brightness)
            self.contrast = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_contrast(sample['img'], self.contrast)
            self.saturation = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_saturation(sample['img'], self.saturation)
        return sample


class RandomRGBNightSimulation:
    """
    RGB만 야간/저조도로 시뮬레이션. thermal, lidar는 유지 → 모델이 보조 모달리티에 집중하도록 유도.
    MULTIAQUA 야간(lj4) 도메인 적응용. Ref: https://arxiv.org/pdf/2512.17450

    brightness_sampling: "uniform" | "log_uniform" | "dark_biased"
    - uniform: random.uniform(min, max)
    - log_uniform: 10^U(log10(min), log10(max)) → 어두운 값 더 자주
    - dark_biased: 70% 확률로 [dark_min, dark_max], 30%로 [moderate_min, moderate_max]
    """
    def __init__(self, p: float = 0.3, brightness_range: Tuple[float, float] = (0.03, 0.25),
                 contrast_range: Tuple[float, float] = (0.3, 0.7), gamma_range: Tuple[float, float] = (0.4, 0.8),
                 noise_std: float = 0.02, brightness_sampling: str = "dark_biased",
                 dark_biased_ratio: float = 0.7, dark_range: Optional[Tuple[float, float]] = None,
                 moderate_range: Optional[Tuple[float, float]] = None) -> None:
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.brightness_sampling = brightness_sampling
        self.dark_biased_ratio = dark_biased_ratio
        self.dark_range = dark_range or (0.03, 0.15)
        self.moderate_range = moderate_range or (0.15, 0.5)

    def _sample_brightness(self) -> float:
        if self.brightness_sampling == "uniform":
            return random.uniform(*self.brightness_range)
        elif self.brightness_sampling == "log_uniform":
            import math
            lo, hi = self.brightness_range
            log_lo, log_hi = math.log10(max(lo, 1e-6)), math.log10(max(hi, 1e-6))
            return 10 ** random.uniform(log_lo, log_hi)
        elif self.brightness_sampling == "dark_biased":
            if random.random() < self.dark_biased_ratio:
                return random.uniform(*self.dark_range)
            else:
                return random.uniform(*self.moderate_range)
        else:
            return random.uniform(*self.brightness_range)

    def __call__(self, sample: dict) -> dict:
        if 'img' not in sample or random.random() >= self.p:
            return sample
        img = sample['img'].float() / 255.0
        # 1) Brightness reduction (다양한 강도, 어두운 쪽 편향)
        brightness = self._sample_brightness()
        img = img * brightness
        # 2) Contrast reduction
        contrast = random.uniform(*self.contrast_range)
        img = (img - img.mean()) * contrast + img.mean()
        # 3) Gamma (gamma < 1 → shadows darker)
        gamma = random.uniform(*self.gamma_range)
        img = torch.clamp(img, 1e-6, 1.0) ** gamma
        img = torch.clamp(img, 0.0, 1.0)
        # 4) Optional sensor noise (저조도에서 증가)
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = torch.clamp(img + noise, 0.0, 1.0)
        sample['img'] = (img * 255).clamp(0, 255).to(sample['img'].dtype)
        return sample


class RandomRGBComplementaryMasking:
    """
    CRM (Complementary Random Masking): RGB의 일부 영역을 0으로 마스킹.
    thermal/lidar는 그대로 유지 → 보조 모달리티 활용 강제. Ref: CRM paper, MULTIAQUA
    """
    def __init__(self, p: float = 0.3, mask_ratio_range: Tuple[float, float] = (0.2, 0.5),
                 num_patches: int = 4) -> None:
        self.p = p
        self.mask_ratio_range = mask_ratio_range
        self.num_patches = num_patches

    def __call__(self, sample: dict) -> dict:
        if 'img' not in sample or random.random() >= self.p:
            return sample
        img = sample['img']
        C, H, W = img.shape
        total_masked = 0
        target_ratio = random.uniform(*self.mask_ratio_range)
        for _ in range(self.num_patches):
            h_size = random.randint(int(H * 0.15), int(H * 0.4))
            w_size = random.randint(int(W * 0.15), int(W * 0.4))
            y = random.randint(0, max(0, H - h_size))
            x = random.randint(0, max(0, W - w_size))
            img[:, y : y + h_size, x : x + w_size] = 0
            total_masked += h_size * w_size
            if total_masked / (H * W) >= target_ratio:
                break
        sample['img'] = img
        return sample


class RandomRGBZeroOut:
    """
    RGB 전체를 0으로 대체 (확률 p). Double forward pass와 유사한 효과.
    야간 극한 상황(완전 암실) 시뮬레이션.
    """
    def __init__(self, p: float = 0.15) -> None:
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if 'img' not in sample or random.random() >= self.p:
            return sample
        sample['img'] = torch.zeros_like(sample['img'])
        return sample


class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.adjust_gamma(img, self.gamma, self.gain), mask


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.adjust_sharpness(sample['img'], self.sharpness)
        return sample


class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.autocontrast(sample['img'])
        return sample


class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.gaussian_blur(sample['img'], self.kernel_size)
            # img = TF.gaussian_blur(img, self.kernel_size)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            for k, v in sample.items():
                sample[k] = TF.hflip(v)
            return sample
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, 3)
        return img, mask


class Equalize:
    def __call__(self, image, label):
        return TF.equalize(image), label


class Posterize:
    def __init__(self, bits=2):
        self.bits = bits # 0-8
        
    def __call__(self, image, label):
        return TF.posterize(image, self.bits), label


class Affine:
    def __init__(self, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0], seg_fill=0):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.seg_fill = seg_fill
        
    def __call__(self, img, label):
        return TF.affine(img, self.angle, self.translate, self.scale, self.shear, TF.InterpolationMode.BILINEAR, 0), TF.affine(label, self.angle, self.translate, self.scale, self.shear, TF.InterpolationMode.NEAREST, self.seg_fill) 


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 0, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag. 
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
                else:
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
            # img = TF.rotate(img, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
            # mask = TF.rotate(mask, random_angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
        return sample
    

class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(img, self.size), TF.center_crop(mask, self.size)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h+1)
            x1 = random.randint(0, margin_w+1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]
        return img, mask


class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int], seg_fill: int = 0) -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            size: expected output image size (h, w)
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        padding = (0, 0, self.size[1]-img.shape[2], self.size[0]-img.shape[1])
        return TF.pad(img, padding), TF.pad(mask, padding, self.seg_fill)


class ResizeWidthPadToSquare:
    """
    MULTIAQUA 전용: 가로가 긴 이미지를 target_size x target_size로 변환.
    가로를 target_size로 리사이즈, 세로는 위아래 패딩으로 맞춤.
    Normalize 전에 적용하여 thermal 패딩 값 불일치 방지.
    """
    def __init__(self, target_size: int, seg_fill: int = 0) -> None:
        self.target_size = target_size if isinstance(target_size, int) else target_size[0]
        self.seg_fill = seg_fill

    def __call__(self, sample: dict) -> dict:
        H, W = sample['img'].shape[1:]
        t = self.target_size
        if W >= H:
            scale = t / W
            nH, nW = round(H * scale), t
            pad_top = (t - nH) // 2
            pad_bottom = t - nH - pad_top
            padding = [0, pad_top, 0, pad_bottom]  # (left, top, right, bottom)
        else:
            scale = t / H
            nH, nW = t, round(W * scale)
            pad_left = (t - nW) // 2
            pad_right = t - nW - pad_left
            padding = [pad_left, 0, pad_right, 0]  # (left, top, right, bottom)

        for k, v in sample.items():
            if k == 'mask':
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
                sample[k] = TF.pad(sample[k], padding, fill=self.seg_fill)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
                sample[k] = TF.pad(sample[k], padding, fill=0)
        return sample


class ResizePad:
    def __init__(self, size: Union[int, Tuple[int], List[int]], seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # scale the image 
        scale_factor = min(tH/H, tW/W) if W > H else max(tH/H, tW/W)
        # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # pad the image
        padding = [0, 0, tW - nW, tH - nH]
        img = TF.pad(img, padding, fill=0)
        mask = TF.pad(mask, padding, fill=self.seg_fill)
        return img, mask 


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, sample:list) -> list:
        H, W = sample['img'].shape[1:]

        # scale the image 
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
        # img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        # mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.BILINEAR)
        # img = TF.resize(img, (alignH, alignW), TF.InterpolationMode.BILINEAR)
        # mask = TF.resize(mask, (alignH, alignW), TF.InterpolationMode.NEAREST)
        return sample


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        # img, mask = sample['img'], sample['mask']
        H, W = sample['img'].shape[1:]
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)
        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)

        # random crop
        margin_h = max(sample['img'].shape[1] - tH, 0)
        margin_w = max(sample['img'].shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]

        # pad the image
        if sample['img'].shape[1:] != self.size:
            padding = [0, 0, tW - sample['img'].shape[2], tH - sample['img'].shape[1]]
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.pad(v, padding, fill=self.seg_fill)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)

        return sample



def _get_thermal_stats(dataset_cfg: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    """MULTIAQUA + thermal일 때만 thermal mean/std 반환. 그 외 None."""
    if not dataset_cfg:
        return None, None
    if dataset_cfg.get('NAME') != 'MULTIAQUA':
        return None, None
    modals = dataset_cfg.get('MODALS') or []
    if 'thermal' not in modals:
        return None, None
    # config에 있으면 사용, 없으면 MULTIAQUA thermal 기본값 (cal_meanstd_thermal 결과)
    m = dataset_cfg.get('THERMAL_MEAN', 84.1594)
    s = dataset_cfg.get('THERMAL_STD', 11.9157)
    return float(m), float(s)


def _use_multiaqua_resize_pad(dataset_cfg: Optional[dict]) -> bool:
    """MULTIAQUA일 때만 2208x1242 → 1024x1024 (가로 리사이즈 + 위아래 패딩) 적용."""
    return bool(dataset_cfg and dataset_cfg.get('NAME') == 'MULTIAQUA')


def _get_night_aug_config(dataset_cfg: Optional[dict]) -> dict:
    """MULTIAQUA 야간 도메인 적응용 augmentation 설정."""
    if not dataset_cfg or dataset_cfg.get('NAME') != 'MULTIAQUA':
        return {}
    return dataset_cfg.get('NIGHT_AUG', {})


def get_train_augmentation(
    size: Union[int, Tuple[int], List[int]],
    seg_fill: int = 0,
    dataset_cfg: Optional[dict] = None,
):
    tm, ts = _get_thermal_stats(dataset_cfg)
    t_size = size if isinstance(size, int) else (size[0] if isinstance(size, (list, tuple)) else size)
    transforms = []
    if _use_multiaqua_resize_pad(dataset_cfg):
        transforms.append(ResizeWidthPadToSquare(t_size, seg_fill=seg_fill))
    transforms.append(RandomColorJitter(p=0.2))

    # MULTIAQUA 전용: RGB 야간 시뮬레이션 augmentation (thermal/lidar는 유지)
    night_cfg = _get_night_aug_config(dataset_cfg)
    if night_cfg.get('ENABLE', False):
        transforms.append(RandomRGBNightSimulation(
            p=night_cfg.get('NIGHT_SIM_P', 0.5),
            brightness_range=tuple(night_cfg.get('BRIGHTNESS_RANGE', [0.03, 0.5])),
            contrast_range=tuple(night_cfg.get('CONTRAST_RANGE', [0.3, 0.7])),
            gamma_range=tuple(night_cfg.get('GAMMA_RANGE', [0.4, 0.8])),
            noise_std=night_cfg.get('NOISE_STD', 0.02),
            brightness_sampling=night_cfg.get('BRIGHTNESS_SAMPLING', 'dark_biased'),
            dark_biased_ratio=night_cfg.get('DARK_BIASED_RATIO', 0.7),
            dark_range=tuple(night_cfg.get('DARK_RANGE', [0.03, 0.15])) if night_cfg.get('DARK_RANGE') else None,
            moderate_range=tuple(night_cfg.get('MODERATE_RANGE', [0.15, 0.5])) if night_cfg.get('MODERATE_RANGE') else None,
        ))
        if night_cfg.get('CRM_P', 0) > 0:
            transforms.append(RandomRGBComplementaryMasking(
                p=night_cfg.get('CRM_P', 0.3),
                mask_ratio_range=tuple(night_cfg.get('CRM_MASK_RATIO', [0.2, 0.5])),
            ))
        if night_cfg.get('ZERO_P', 0) > 0:
            transforms.append(RandomRGBZeroOut(p=night_cfg.get('ZERO_P', 0.15)))

    transforms.extend([
        RandomHorizontalFlip(p=0.5),
        RandomGaussianBlur((3, 3), p=0.2),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), thermal_mean=tm, thermal_std=ts)
    ])
    return Compose(transforms)


def get_val_augmentation(
    size: Union[int, Tuple[int], List[int]],
    dataset_cfg: Optional[dict] = None,
):
    tm, ts = _get_thermal_stats(dataset_cfg)
    t_size = size if isinstance(size, int) else (size[0] if isinstance(size, (list, tuple)) else size)
    transforms = []
    if _use_multiaqua_resize_pad(dataset_cfg):
        transforms.append(ResizeWidthPadToSquare(t_size, seg_fill=dataset_cfg.get('IGNORE_LABEL', 255) if dataset_cfg else 255))
    transforms.extend([
        Resize(size),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), thermal_mean=tm, thermal_std=ts)
    ])
    return Compose(transforms)


if __name__ == '__main__':
    h = 230
    w = 420
    sample = {}
    sample['img'] = torch.randn(3, h, w)
    sample['depth'] = torch.randn(3, h, w)
    sample['lidar'] = torch.randn(3, h, w)
    sample['event'] = torch.randn(3, h, w)
    sample['mask'] = torch.randn(1, h, w)
    aug = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop((512, 512)),
        Resize((224, 224)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    sample = aug(sample)
    for k, v in sample.items():
        print(k, v.shape)