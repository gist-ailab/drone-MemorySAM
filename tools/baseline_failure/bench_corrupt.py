"""DRN-261027-01 R1 Fig.3/Fig.6 센서 고장 덤프용 공통 손상 정의.
S4 노이즈는 벤치(arXiv 2503.18445,
third_party/mmss_robust_bench/upstream/val_mm_NM.py:64-84) 함수와 같은 식이다.
정규화 입력과 detectron2 원본 입력에 동일한 설정을 적용한다.
"""

import numpy as np
import torch


# DELIVER 모달 이름을 사용한다. noise 는 정규화 공간의 S&P 다음 Gaussian 이다.
SETTINGS = {
    "none": {"zero": (), "noise": ()},
    "depth0": {"zero": ("depth",), "noise": ()},
    "rgb0": {"zero": ("img",), "noise": ()},
    "lidarevent0": {"zero": ("lidar", "event"), "noise": ()},
    "rgbnmhigh": {"zero": (), "noise": ("img",)},
}
NM_HIGH = (0.2, 0.5)  # upstream level 2: density, std


def bench_sp(x, density, gen):
    """원본 텐서의 극값으로 salt/pepper 를 채우는 벤치 수식."""
    noisy = x.clone()
    n = int(np.ceil(density * x.numel() * 0.5))
    coords = [torch.randint(0, s, (n,), generator=gen).to(x.device)
              for s in x.shape]
    noisy[coords] = x.max()
    coords = [torch.randint(0, s, (n,), generator=gen).to(x.device)
              for s in x.shape]
    noisy[coords] = x.min()
    return noisy


def bench_gauss(x, std, gen):
    """CPU 생성기를 사용하고 입력 텐서의 device/dtype 으로 옮긴다."""
    return x + torch.randn(x.shape, generator=gen).to(x.device, x.dtype) * std


def apply_normalized(images, modals, setting, gen):
    """[B,C,H,W] 정규화 텐서 목록에 DELIVER 모달별 손상을 적용한다."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown corruption setting: {setting!r}")
    spec = SETTINGS[setting]
    required = spec["zero"] + spec["noise"]
    missing = [modal for modal in required if modal not in modals]
    if missing:
        raise ValueError(f"required modals missing for {setting}: {missing}")
    result = list(images)
    for i, modal in enumerate(modals):
        if modal in spec["zero"]:
            result[i] = torch.zeros_like(images[i])
        elif modal in spec["noise"]:
            assert images[i].shape[0] == 1, "rgbnmhigh requires batch size 1"
            result[i] = bench_gauss(
                bench_sp(images[i], NM_HIGH[0], gen), NM_HIGH[1], gen)
    return result


def _cfg_get(cfg, *keys):
    value = cfg
    try:
        for key in keys:
            value = value[key] if isinstance(value, dict) else getattr(value, key)
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"missing cfg.{'.'.join(keys)}") from exc
    return value


def register_d2_hook(model, cfg, setting, seed=0):
    """DGFusion/CAFuser 의 raw batched_inputs 에 공통 손상을 주입한다."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown corruption setting: {setting!r}")
    order = _cfg_get(cfg, "DATASETS", "MODALITIES", "ORDER")
    main = _cfg_get(cfg, "DATASETS", "MODALITIES", "MAIN_MODALITY")
    main = str(main).strip().upper()
    order = [str(modal).strip().upper() for modal in order]
    modal_order = [main] + [modal for modal in order if modal != main]
    spec = SETTINGS[setting]
    target_to_d2 = {"img": "CAMERA", "depth": "DEPTH",
                    "lidar": "LIDAR", "event": "EVENT"}
    needed = spec["zero"] + spec["noise"]
    missing = [modal for modal in needed
               if target_to_d2[modal] not in modal_order]
    if missing:
        raise RuntimeError(f"required modals missing from cfg order: {missing}")

    pm = getattr(model, "pixel_mean", None)
    ps = getattr(model, "pixel_std", None)
    if not (torch.is_tensor(pm) and torch.is_tensor(ps)):
        raise RuntimeError("model.pixel_mean/pixel_std are missing")
    pm = pm.detach().reshape(-1)
    ps = ps.detach().reshape(-1)
    expected = 3 * len(modal_order)
    if pm.numel() != expected or ps.numel() != expected:
        raise RuntimeError(
            f"pixel_mean/std length must be {expected} (3 * {len(modal_order)} modals); "
            f"got {pm.numel()}/{ps.numel()}")
    slices = {modal: slice(3 * modal_order.index(target_to_d2[modal]),
                           3 * modal_order.index(target_to_d2[modal]) + 3)
              for modal in needed}
    gen = torch.Generator().manual_seed(seed)

    def hook(_model, inputs):
        if (not inputs or not isinstance(inputs[0], (list, tuple))
                or not inputs[0]):
            raise RuntimeError("expected batched_inputs as a list of dicts")
        for d in inputs[0]:
            if not isinstance(d, dict):
                raise RuntimeError("expected each batched input to be a dict")
            for modal in needed:
                key = "image" if modal == "img" else target_to_d2[modal]
                if key not in d or not torch.is_tensor(d[key]):
                    raise RuntimeError(f"required input key {key!r} is absent")
                v = d[key]
                if v.ndim != 3 or v.shape[0] != 3:
                    raise RuntimeError(f"{key!r} must have shape [3,H,W]")
                # 출력은 항상 float32: 입력이 uint8 이면 v.dtype 으로 되돌릴 때 채움 평균이
                # 잘리고(123.675→123) 노이즈의 0 미만·255 초과 값이 깨진다. 모델 내부
                # 정규화 (x - mean) / std 는 float 입력을 그대로 받는다.
                mean = pm[slices[modal]].to(device=v.device, dtype=torch.float32).view(3, 1, 1)
                if modal in spec["zero"]:
                    changed = mean.expand(v.shape).clone()
                else:
                    std = ps[slices[modal]].to(device=v.device, dtype=torch.float32).view(3, 1, 1)
                    normalized = ((v.to(torch.float32) - mean) / std).unsqueeze(0)
                    changed = bench_gauss(
                        bench_sp(normalized, NM_HIGH[0], gen), NM_HIGH[1], gen
                    ).squeeze(0).mul(std).add(mean)
                d[key] = changed
                if modal == "img" and "CAMERA" in d:
                    d["CAMERA"] = changed

    handle = model.register_forward_pre_hook(hook)
    slice_desc = {modal: (part.start, part.stop) for modal, part in slices.items()}
    print(f"[bench_corrupt] registered setting={setting} modal_order={modal_order} "
          f"slices={slice_desc}", flush=True)
    return handle
