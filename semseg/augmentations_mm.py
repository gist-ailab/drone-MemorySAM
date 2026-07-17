import torchvision.transforms.functional as TF
import torchvision.io as io
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
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


class ColorAugSSD:
    """DGFusion(Detectron2 ColorAugSSDTransform) 정합 photometric aug — RGB 전용.

    각 서브증강 독립 p=0.5 (SSD 표준): brightness 가산 ±32/255 · contrast ×U(0.5,1.5)
    · saturation ×U(0.5,1.5) · hue ±36°(=0.1 cycle; OpenCV hue_delta=18 등가).
    공정성: DGFusion DELIVER 학습 파이프라인과 동일 강도 — PhysAug의 합법 대체.
    (deliver_semantic_dataset_mapper: 비-camera 모달에는 identity — 여기서도 img만.)
    """
    def __init__(self, brightness_delta: float = 32.0 / 255.0,
                 contrast: Tuple[float, float] = (0.5, 1.5),
                 saturation: Tuple[float, float] = (0.5, 1.5),
                 hue_delta: float = 0.1) -> None:
        self.bd = brightness_delta
        self.contrast = contrast
        self.saturation = saturation
        self.hd = hue_delta

    def __call__(self, sample: list) -> list:
        img = sample['img']
        if random.random() < 0.5:
            img = (img + random.uniform(-self.bd, self.bd)).clamp(0.0, 1.0) \
                if torch.is_tensor(img) else TF.adjust_brightness(img, 1.0 + random.uniform(-self.bd, self.bd))
        # SSD는 contrast를 (sat/hue) 앞 또는 뒤 랜덤 순서로 적용
        contrast_first = random.random() < 0.5
        if contrast_first and random.random() < 0.5:
            img = TF.adjust_contrast(img, random.uniform(*self.contrast))
        if random.random() < 0.5:
            img = TF.adjust_saturation(img, random.uniform(*self.saturation))
        if random.random() < 0.5:
            img = TF.adjust_hue(img, random.uniform(-self.hd, self.hd))
        if (not contrast_first) and random.random() < 0.5:
            img = TF.adjust_contrast(img, random.uniform(*self.contrast))
        sample['img'] = img
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
                 moderate_range: Optional[Tuple[float, float]] = None,
                 shot_noise_gain_range: Optional[Tuple[float, float]] = None) -> None:
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.brightness_sampling = brightness_sampling
        self.dark_biased_ratio = dark_biased_ratio
        self.dark_range = dark_range or (0.03, 0.15)
        self.moderate_range = moderate_range or (0.15, 0.5)
        self.shot_noise_gain_range = shot_noise_gain_range  # (min_gain, max_gain), None=disabled

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
        # 3) Shot noise (Poisson) — signal-dependent, 밝은 영역에 더 강한 노이즈
        #    물리적 순서: 광자 도달 → shot noise → gain → gamma → read noise
        if self.shot_noise_gain_range is not None:
            gain = random.uniform(*self.shot_noise_gain_range)
            if gain > 0:
                img_photon = torch.clamp(img, min=0) * gain
                img = torch.poisson(img_photon) / gain
                img = torch.clamp(img, 0.0, 1.0)
        # 4) Gamma (gamma < 1 → shadows darker)
        gamma = random.uniform(*self.gamma_range)
        img = torch.clamp(img, 1e-6, 1.0) ** gamma
        img = torch.clamp(img, 0.0, 1.0)
        # 5) Read noise (Gaussian) — signal-independent sensor 전자회로 노이즈
        if self.noise_std > 0:
            noise = torch.randn_like(img) * self.noise_std
            img = torch.clamp(img + noise, 0.0, 1.0)
        sample['img'] = (img * 255).clamp(0, 255).to(sample['img'].dtype)
        return sample


class RandomGammaWide:
    """NightSim과 독립적인 wide gamma augmentation. Contrast invariance 학습용.

    NightSim 적용 여부와 무관하게 모든 RGB 이미지에 적용 가능.
    NightSim 이전에 배치: 주간 이미지에 gamma 적용 → NightSim → 다양한 야간.
    gamma < 1: 어둡게 (contrast 증가), gamma > 1: 밝게 (contrast 감소).

    hardaug6과 차이: hardaug6은 NightSim 내부 gamma를 변경 → 비현실적 야간 생성.
    본 클래스는 NightSim 파라미터를 건드리지 않고 독립적으로 domain randomization.
    """
    def __init__(self, gamma_range: Tuple[float, float] = (0.3, 2.5), p: float = 0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if 'img' not in sample or random.random() >= self.p:
            return sample
        gamma = random.uniform(*self.gamma_range)
        img = sample['img'].float() / 255.0
        img = torch.clamp(img, 1e-6, 1.0) ** gamma
        sample['img'] = (img * 255).clamp(0, 255).to(sample['img'].dtype)
        return sample


class RandomFDA:
    """FDA (Fourier Domain Adaptation): 야간 이미지의 저주파 amplitude를 주간 RGB에 적용.

    Ref: Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation" (CVPR 2020)

    low-frequency amplitude = 조명, global color tone, 전체 밝기
    high-frequency = edge, texture, semantic structure (보존됨)

    NightSim이 brightness/contrast/gamma로 야간을 '흉내'내는 반면,
    FDA는 실제 야간 이미지의 주파수 특성을 직접 전이 → 더 사실적인 야간 스타일.
    RGB에만 적용 (thermal/lidar 유지).

    Args:
        p: 적용 확률
        beta: 저주파 대역 비율 고정값 (beta_range 미지정 시 사용)
        beta_range: (min, max) 범위에서 매 호출마다 랜덤 샘플링. 지정 시 beta 무시.
        target_dir: reference 이미지 디렉토리
        blend_ratio: 원본과 FDA 결과의 블렌딩 비율 (0=원본, 1=full FDA)
        target_prefix: 파일명 prefix 필터 (예: 'lj4' → 야간만)
    """
    def __init__(self, p: float = 0.3, beta: float = 0.03, target_dir: str = None,
                 blend_ratio: float = 1.0, target_prefix: str = None,
                 beta_range: tuple = None):
        self.p = p
        self.beta = beta
        self.beta_range = beta_range
        self.blend_ratio = blend_ratio
        all_paths = sorted(Path(target_dir).glob('*.png'))
        if target_prefix:
            self.target_paths = [p for p in all_paths if p.stem.startswith(target_prefix)]
        else:
            self.target_paths = all_paths
        assert len(self.target_paths) > 0, (
            f"No target images in {target_dir} (prefix={target_prefix})")
        beta_str = f"{beta_range[0]:.3f}~{beta_range[1]:.3f}" if beta_range else f"{beta}"
        print(f"[RandomFDA] Loaded {len(self.target_paths)} target style images "
              f"(beta={beta_str}, p={p}, blend={blend_ratio}, prefix={target_prefix})")

    def _fda_swap(self, src_img: Tensor, tgt_img: Tensor, beta: float) -> Tensor:
        """src의 저주파 amplitude를 tgt의 것으로 교체.

        Args:
            src_img: (3, H, W) float [0, 1]
            tgt_img: (3, H, W) float [0, 1]
            beta: 저주파 대역 비율
        Returns:
            (3, H, W) float [0, 1]
        """
        _, h, w = src_img.shape
        src_fft = torch.fft.fft2(src_img, dim=(-2, -1))
        tgt_fft = torch.fft.fft2(tgt_img, dim=(-2, -1))

        src_amp = torch.abs(src_fft)
        src_phase = torch.angle(src_fft)
        tgt_amp = torch.abs(tgt_fft)

        h_cut = max(1, int(h * beta))
        w_cut = max(1, int(w * beta))

        # fft2 출력: corners가 저주파 → 4개 코너의 amplitude 교체
        new_amp = src_amp.clone()
        new_amp[:, :h_cut, :w_cut] = tgt_amp[:, :h_cut, :w_cut]
        new_amp[:, -h_cut:, :w_cut] = tgt_amp[:, -h_cut:, :w_cut]
        new_amp[:, :h_cut, -w_cut:] = tgt_amp[:, :h_cut, -w_cut:]
        new_amp[:, -h_cut:, -w_cut:] = tgt_amp[:, -h_cut:, -w_cut:]

        fda_fft = new_amp * torch.exp(1j * src_phase)
        fda_img = torch.fft.ifft2(fda_fft, dim=(-2, -1)).real
        return torch.clamp(fda_img, 0.0, 1.0)

    def _sample_beta(self) -> float:
        """beta_range 설정 시 uniform 랜덤 샘플링, 아니면 고정값 반환."""
        if self.beta_range:
            return random.uniform(self.beta_range[0], self.beta_range[1])
        return self.beta

    def __call__(self, sample: dict) -> dict:
        if 'img' not in sample or random.random() >= self.p:
            return sample

        beta = self._sample_beta()

        src = sample['img'].float() / 255.0
        tgt_path = random.choice(self.target_paths)
        tgt = io.read_image(str(tgt_path))[:3, ...].float() / 255.0

        if tgt.shape[1:] != src.shape[1:]:
            tgt = TF.resize(tgt, list(src.shape[1:]), TF.InterpolationMode.BILINEAR)

        fda_img = self._fda_swap(src, tgt, beta)

        if self.blend_ratio < 1.0:
            fda_img = (1 - self.blend_ratio) * src + self.blend_ratio * fda_img
            fda_img = torch.clamp(fda_img, 0.0, 1.0)

        sample['img'] = (fda_img * 255).clamp(0, 255).to(sample['img'].dtype)
        return sample


class RandomPhysAug:
    """PhysAug: 물리 기반 공간적 교란 augmentation.

    Ref: PhysAug (AAAI 2025) — Physical-guided and Frequency-based Data Augmentation
         for Single-Domain Generalized Object Detection.

    두 모듈로 구성:
    1. Filter: identity kernel + Gaussian noise로 random convolution → 비균일 조명 변화
    2. Fourier: planar sinusoidal wave → 대기 입자 산란/회절 패턴

    NightSim(global scalar: brightness/contrast/gamma)과 orthogonal하게,
    공간적으로 비균일한 물리적 교란을 추가하여 야간 domain 강건성 향상.
    RGB에만 적용 (thermal/lidar 유지).

    Args:
        p: 전체 적용 확률
        filter_enable: Filter 모듈 활성화
        filter_sigma_range: (min, max) random conv noise 강도. 0=무변환.
        filter_kernel_size: conv 커널 크기 (홀수)
        fourier_enable: Fourier 모듈 활성화
        fourier_groups: (min, max) 주파수 범위. freq = group / img_size.
        fourier_phases: (start, end) 위상 범위 (pi의 배수)
        fourier_granularity: 위상 이산화 해상도
        fourier_mean_str: wave strength의 exponential 분포 평균의 역수. 높을수록 약함.
        fourier_decay: Gaussian spatial decay 강도. 0=전역 균일, >0=국소적.
        fourier_f_cut: 동시 사용 주파수 수
        fourier_p_cut: 주파수당 위상 수
    """
    def __init__(
        self,
        p: float = 0.4,
        filter_enable: bool = True,
        filter_sigma_range: Tuple[float, float] = (0.0, 1.5),
        filter_kernel_size: int = 3,
        fourier_enable: bool = True,
        fourier_groups: Tuple[int, int] = (1, 513),
        fourier_phases: Tuple[float, float] = (0.0, 1.0),
        fourier_granularity: int = 256,
        fourier_mean_str: float = 8.0,
        fourier_decay: float = 0.3,
        fourier_f_cut: int = 1,
        fourier_p_cut: int = 1,
    ):
        self.p = p
        # Filter params
        self.filter_enable = filter_enable
        self.filter_sigma_range = filter_sigma_range
        self.filter_kernel_size = filter_kernel_size
        self.filter_kernel_candidates = [filter_kernel_size, filter_kernel_size + 2]
        # Fourier params
        self.fourier_enable = fourier_enable
        self.fourier_f_cut = fourier_f_cut
        self.fourier_p_cut = fourier_p_cut
        self.fourier_mean_str = fourier_mean_str
        self.fourier_decay = fourier_decay
        # 주파수 후보 풀
        num_groups = fourier_groups[1] - fourier_groups[0]
        self.fourier_freqs = [g / 1024.0 for g in range(fourier_groups[0], fourier_groups[1])]
        self.fourier_num_groups = len(self.fourier_freqs)
        # 위상 후보 풀
        self.fourier_phases_arr = -np.pi * np.linspace(
            fourier_phases[0], fourier_phases[1], num=fourier_granularity
        )
        self.fourier_num_phases = fourier_granularity
        self.fourier_eps_scale = 1024.0 / 32.0  # = 32
        # meshgrid cache
        self._cached_mesh = None
        self._cached_mesh_size = None

        modules = []
        if filter_enable:
            modules.append(f"filter(sigma={filter_sigma_range}, k={filter_kernel_size})")
        if fourier_enable:
            modules.append(f"fourier(groups={fourier_groups}, mean_str={fourier_mean_str}, decay={fourier_decay})")
        print(f"[RandomPhysAug] p={p}, modules: {', '.join(modules)}")

    def _get_meshgrid(self, H: int, W: int):
        """좌표 meshgrid 생성/캐시."""
        if self._cached_mesh_size == (H, W):
            return self._cached_mesh
        _x = np.linspace(-H / 2, H / 2, H)
        _y = np.linspace(-W / 2, W / 2, W)
        mesh_x, mesh_y = np.meshgrid(_x, _y, indexing='ij')
        self._cached_mesh = (
            torch.tensor(mesh_x, dtype=torch.float32),
            torch.tensor(mesh_y, dtype=torch.float32),
        )
        self._cached_mesh_size = (H, W)
        return self._cached_mesh

    def _apply_filter(self, img: Tensor, sigma: float) -> Tensor:
        """Random convolution: identity kernel + Gaussian noise.

        Args:
            img: (C, H, W) float [0, 1]
            sigma: noise 강도
        Returns:
            (C, H, W) float [0, 1], per-channel min-max normalized.
        """
        C, H, W = img.shape
        # 커널 크기 랜덤 선택
        ks = random.choice(self.filter_kernel_candidates)
        pad = ks // 2

        # identity + Gaussian noise 커널
        delta = torch.zeros(ks, ks)
        delta[ks // 2, ks // 2] = 1.0
        conv_weight = sigma * torch.randn(ks, ks) + delta
        # (1, 1, ks, ks) for depthwise-like conv (채널별 동일 커널)
        conv_weight = conv_weight.unsqueeze(0).unsqueeze(0)

        # 채널별 conv
        filtered = torch.zeros_like(img)
        for c in range(C):
            inp = img[c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            out = F.conv2d(inp, conv_weight, padding=pad)
            filtered[c] = out.squeeze()

        filtered = filtered.abs()

        # per-channel min-max normalization → [0, 1]
        for c in range(C):
            ch = filtered[c]
            mn, mx = ch.min(), ch.max()
            if mx - mn > 1e-6:
                filtered[c] = (ch - mn) / (mx - mn)
            else:
                filtered[c] = ch.clamp(0.0, 1.0)

        return filtered

    def _gen_planar_waves(self, freqs: Tensor, phases: Tensor,
                          H: int, W: int) -> Tensor:
        """평면파 정현파 생성.

        wave = sin(2π·f·(x·cos(φ) + y·sin(φ)) - π/4)

        Args:
            freqs: (1, C, f_cut, 1) 주파수
            phases: (1, C, f_cut, p_cut) 위상
            H, W: 이미지 크기
        Returns:
            (1, C, f_cut, p_cut, H, W) 정규화된 파동
        """
        mesh_x, mesh_y = self._get_meshgrid(H, W)

        # reshape for broadcasting: (1, C, f_cut, p_cut, 1, 1)
        f = freqs.unsqueeze(-1).unsqueeze(-1)
        p = phases.unsqueeze(-1).unsqueeze(-1)

        waves = torch.sin(
            2 * math.pi * f * (mesh_x * torch.cos(p) + mesh_y * torch.sin(p))
            - math.pi / 4
        )
        # L2 normalize per wave
        norm = torch.norm(waves, dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        waves = waves / norm * self.fourier_eps_scale

        return waves

    def _apply_gaussian_decay(self, aug: Tensor, H: int, W: int) -> Tensor:
        """Gaussian spatial decay: 랜덤 중심에서 가장자리로 감쇠.

        Args:
            aug: (H, W, C) perturbation
        Returns:
            (H, W, C) decayed perturbation
        """
        center_x = random.randint(0, max(0, H - 13))
        center_y = random.randint(0, max(0, W - 13))
        sigma_x, sigma_y = H / 6.0, W / 6.0

        x = torch.arange(H, dtype=torch.float32) - center_x
        y = torch.arange(W, dtype=torch.float32) - center_y
        X, Y = torch.meshgrid(x, y, indexing='ij')

        gaussian = torch.exp(-((X ** 2 / (2 * sigma_x ** 2)) + (Y ** 2 / (2 * sigma_y ** 2))))
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min() + 1e-6)
        decay_map = (1 - self.fourier_decay) + self.fourier_decay * gaussian
        # (H, W) → (H, W, 1) for broadcast
        return aug * decay_map.unsqueeze(-1)

    def _apply_fourier(self, img: Tensor) -> Tensor:
        """Planar wave perturbation + atmospheric light.

        Args:
            img: (C, H, W) float [0, 1]
        Returns:
            (C, H, W) float [0, 1]
        """
        C, H, W = img.shape

        # 주파수/위상 샘플링 (채널별 독립)
        freqs_np = np.array(self.fourier_freqs, dtype=np.float32)
        phases_np = np.array(self.fourier_phases_arr, dtype=np.float32)

        f_idx = np.random.randint(0, self.fourier_num_groups,
                                  (1, C, self.fourier_f_cut, 1))
        p_idx = np.random.randint(0, self.fourier_num_phases,
                                  (1, C, self.fourier_f_cut, self.fourier_p_cut))

        freqs = torch.tensor(freqs_np[f_idx], dtype=torch.float32)
        phases = torch.tensor(phases_np[p_idx], dtype=torch.float32)

        # wave strength: exponential 분포
        strengths = np.random.exponential(
            1.0 / self.fourier_mean_str,
            (1, C, self.fourier_f_cut, self.fourier_p_cut)
        )
        strengths_t = torch.tensor(strengths, dtype=torch.float32)

        # planar wave 생성
        waves = self._gen_planar_waves(freqs, phases, H, W)

        # einsum: strengths(1,C,f,p) * waves(1,C,f,p,H,W) → (1,C,H,W)
        aug = torch.einsum('bcfp,bcfphw->bchw', strengths_t, waves)
        aug = aug / (self.fourier_f_cut * self.fourier_p_cut)

        # interpolate to actual size (waves가 이미 H,W이면 불필요하지만 안전장치)
        if aug.shape[-2:] != (H, W):
            aug = F.interpolate(aug, size=(H, W), mode='bilinear', align_corners=False)

        # (1, C, H, W) → (H, W, C) for gaussian decay
        aug_hwc = aug[0].permute(1, 2, 0)

        if self.fourier_decay > 0:
            aug_hwc = self._apply_gaussian_decay(aug_hwc, H, W)

        # additive perturbation
        img_hwc = img.permute(1, 2, 0)
        result = torch.clamp(img_hwc + aug_hwc, 0.0, 1.0)

        # atmospheric light: L = L_inf * (1 - exp(-d))
        log_sample = random.uniform(-3, -1)
        L_inf = 10 ** log_sample
        dx = random.uniform(0, 10)
        L = L_inf * (1 - math.exp(-dx))
        result = torch.clamp(result + L, 0.0, 1.0)

        return result.permute(2, 0, 1)

    def __call__(self, sample: dict) -> dict:
        if 'img' not in sample or random.random() >= self.p:
            return sample

        img = sample['img'].float() / 255.0  # (C, H, W) [0, 1]

        if self.filter_enable:
            sigma = random.uniform(*self.filter_sigma_range)
            if sigma > 0.01:
                img = self._apply_filter(img, sigma)

        if self.fourier_enable:
            img = self._apply_fourier(img)

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
    _dgf_aug = bool((dataset_cfg or {}).get('DGFUSION_AUG', False))
    if _dgf_aug:
        # DGFusion 정합 레시피: SSD photometric (PhysAug 대체, 공정선 내)
        transforms.append(ColorAugSSD())
    else:
        transforms.append(RandomColorJitter(p=0.2))

    # MULTIAQUA 전용: RGB 야간 시뮬레이션 augmentation (thermal/lidar는 유지)
    night_cfg = _get_night_aug_config(dataset_cfg)

    # RandomGammaWide: NightSim 이전에 적용 → 주간 RGB에 다양한 gamma → NightSim → 다양한 야간
    rgw_cfg = night_cfg.get('RANDOM_GAMMA_WIDE', {})
    if rgw_cfg.get('ENABLE', False):
        transforms.append(RandomGammaWide(
            gamma_range=tuple(rgw_cfg.get('GAMMA_RANGE', [0.3, 2.5])),
            p=rgw_cfg.get('P', 0.5),
        ))

    # FDA: Fourier Domain Adaptation — 야간 style을 주파수 도메인에서 전이
    fda_cfg = night_cfg.get('FDA', {})
    if fda_cfg.get('ENABLE', False):
        target_dir = fda_cfg.get('TARGET_DIR')
        if target_dir:
            transforms.append(RandomFDA(
                p=fda_cfg.get('P', 0.3),
                beta=fda_cfg.get('BETA', 0.03),
                target_dir=target_dir,
                blend_ratio=fda_cfg.get('BLEND_RATIO', 1.0),
                target_prefix=fda_cfg.get('TARGET_PREFIX', None),
                beta_range=fda_cfg.get('BETA_RANGE', None),
            ))

    # PhysAug: 물리 기반 공간적 교란 (random conv + planar wave)
    # dataset_cfg.PHYSAUG (top-level) 우선, 없으면 NIGHT_AUG.PHYSAUG fallback (하위 호환)
    physaug_cfg = (dataset_cfg or {}).get('PHYSAUG') or night_cfg.get('PHYSAUG', {})
    if physaug_cfg.get('ENABLE', False):
        filter_cfg = physaug_cfg.get('FILTER', {})
        fourier_cfg = physaug_cfg.get('FOURIER', {})
        transforms.append(RandomPhysAug(
            p=physaug_cfg.get('P', 0.4),
            filter_enable=filter_cfg.get('ENABLE', True),
            filter_sigma_range=tuple(filter_cfg.get('SIGMA_RANGE', [0.0, 1.5])),
            filter_kernel_size=filter_cfg.get('KERNEL_SIZE', 3),
            fourier_enable=fourier_cfg.get('ENABLE', True),
            fourier_groups=tuple(fourier_cfg.get('GROUPS', [1, 513])),
            fourier_phases=tuple(fourier_cfg.get('PHASES', [0.0, 1.0])),
            fourier_granularity=fourier_cfg.get('GRANULARITY', 256),
            fourier_mean_str=fourier_cfg.get('MEAN_STR', 8.0),
            fourier_decay=fourier_cfg.get('DECAY', 0.3),
            fourier_f_cut=fourier_cfg.get('F_CUT', 1),
            fourier_p_cut=fourier_cfg.get('P_CUT', 1),
        ))

    if night_cfg.get('ENABLE', False):
        _sng = night_cfg.get('SHOT_NOISE_GAIN_RANGE')
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
            shot_noise_gain_range=tuple(_sng) if _sng else None,
        ))
        if night_cfg.get('CRM_P', 0) > 0:
            transforms.append(RandomRGBComplementaryMasking(
                p=night_cfg.get('CRM_P', 0.3),
                mask_ratio_range=tuple(night_cfg.get('CRM_MASK_RATIO', [0.2, 0.5])),
            ))
        if night_cfg.get('ZERO_P', 0) > 0:
            transforms.append(RandomRGBZeroOut(p=night_cfg.get('ZERO_P', 0.15)))

    transforms.append(RandomHorizontalFlip(p=0.5))
    if not _dgf_aug:
        # DGFusion 파이프라인에는 blur 없음 — 정합 모드에서는 제외
        transforms.append(RandomGaussianBlur((3, 3), p=0.2))
    transforms.extend([
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),   # = DGFusion multi-scale 0.5-2.0 + crop
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


def get_nightval_augmentation(
    size: Union[int, Tuple[int], List[int]],
    dataset_cfg: Optional[dict] = None,
):
    """MULTIAQUA 야간 조건 Validation augmentation (ISSUE-001 대응).

    기존 get_val_augmentation()은 수정하지 않고, 야간 시뮬레이션을 추가한 별도 함수.
    NightSim p=1.0 (항상 적용) → dice-roll 없이 모든 이미지에 야간 조건 부여.
    CRM / Zero-out은 config 확률 그대로 유지 (더 realistic한 mixed-condition 평가).
    기하학적 증강(Flip, Crop)은 적용하지 않음 (val 특성 유지).

    best checkpoint 기준:
      - 기존 val   → day-val  best (주간 성능 기준)
      - 이 함수    → night-val best (야간 시뮬 성능 기준) → test와 더 가까운 체크포인트
    """
    tm, ts = _get_thermal_stats(dataset_cfg)
    t_size = size if isinstance(size, int) else (size[0] if isinstance(size, (list, tuple)) else size)
    transforms = []

    if _use_multiaqua_resize_pad(dataset_cfg):
        transforms.append(ResizeWidthPadToSquare(
            t_size,
            seg_fill=dataset_cfg.get('IGNORE_LABEL', 255) if dataset_cfg else 255,
        ))

    night_cfg = _get_night_aug_config(dataset_cfg)
    if night_cfg.get('ENABLE', False):
        transforms.append(RandomRGBNightSimulation(
            p=1.0,  # val 시 항상 적용 — dice-roll 제거로 모든 샘플에 야간 조건 부여
            brightness_range=tuple(night_cfg.get('BRIGHTNESS_RANGE', [0.03, 0.45])),
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
            transforms.append(RandomRGBZeroOut(p=night_cfg.get('ZERO_P', 0.09)))

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