"""CPU checks against the DELIVER robustness definitions."""
import torch

from tools.missing_modality_eval import (enumerate_present_subsets, missing_of,
                                         rmm_degrade, zero_fill)
from tools.muses_robust_eval import LEVELS, rgb_noise
from third_party.dgfusion_train_restore.muses_robust import muses_robust_hook


def test_fifteen_subsets():
    subsets = enumerate_present_subsets(4)
    assert len(subsets) == 15
    assert len(set(subsets)) == 15
    assert subsets[0] == (0, 1, 2, 3)
    assert all(subset for subset in subsets)


def test_rmm_seed_and_elementwise_bench_mask():
    base = [torch.ones(1, 3, 5, 7) for _ in range(4)]
    missing = missing_of((0, 2), 4)
    seed = 41
    result = rmm_degrade(base, missing, 0.5, torch.Generator().manual_seed(seed))
    expected_gen = torch.Generator().manual_seed(seed)
    assert torch.equal(result[1], (torch.rand(base[1].shape, generator=expected_gen) >= 0.5).float())
    assert torch.equal(result[3], (torch.rand(base[3].shape, generator=expected_gen) >= 0.5).float())
    repeat = rmm_degrade(base, missing, 0.5, torch.Generator().manual_seed(seed))
    assert all(torch.equal(a, b) for a, b in zip(result, repeat))
    assert torch.equal(result[0], base[0]) and torch.equal(result[2], base[2])


def test_rgb_nm_levels_and_other_sensors_untouched():
    assert LEVELS == {'low': (0.05, 0.1), 'mid': (0.1, 0.2), 'high': (0.2, 0.5)}
    base = [torch.linspace(-1, 1, 120).reshape(1, 3, 5, 8) for _ in range(4)]
    for level in LEVELS:
        a = rgb_noise(base, level, torch.Generator().manual_seed(7))
        b = rgb_noise(base, level, torch.Generator().manual_seed(7))
        assert torch.equal(a[0], b[0]) and not torch.equal(a[0], base[0])
        assert all(torch.equal(a[i], base[i]) for i in range(1, 4))


def test_zero_fill_after_normalization():
    raw = torch.tensor([[[[10.0]]], [[[20.0]]]])
    normalized = (raw - 15.0) / 5.0
    result = zero_fill([normalized], (0,))
    assert torch.equal(result[0], torch.zeros_like(normalized))
    assert torch.equal(normalized, torch.tensor([[[[-1.0]]], [[[1.0]]]]))


def test_d2_raw_mean_fill_becomes_normalized_zero(monkeypatch):
    class Model:
        modalities = list(muses_robust_hook.MODALS)
        pixel_mean = torch.arange(1, 13, dtype=torch.float32).reshape(12, 1, 1)
        pixel_std = torch.full((12, 1, 1), 2.0)

        def register_forward_pre_hook(self, fn):
            self.hook = fn

    monkeypatch.setenv('MUSES_ROBUST_PROTOCOL', 'emm')
    monkeypatch.setenv('MUSES_ROBUST_SUBSET', 'lidar,event,radar')
    model = Model()
    muses_robust_hook.install(model)
    raw = torch.full((3, 2, 2), 100, dtype=torch.uint8)
    sample = {name: raw.clone() for name in model.modalities}
    sample['image'] = raw.clone()
    model.hook(model, ([sample],))
    mu = model.pixel_mean[:3]
    assert torch.equal((sample['CAMERA'] - mu) / model.pixel_std[:3],
                       torch.zeros_like(sample['CAMERA']))
    assert torch.equal(sample['CAMERA'], sample['image'])
