#!/usr/bin/env python3
"""CPU checks for H1 SPM and tap sensor selection, without pretrained weights."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semseg.models.reliadino.hires_adapter import HiResAdapter  # noqa: E402


NAMES = ['img', 'depth', 'event']


def make_adapter(modals):
    return HiResAdapter(fpn_dim=8, num_modalities=3, tap_in_dim=8,
                        dim=8, stem_dim=8, n_heads=2, n_points=2,
                        extra_extractors=0, modal_names=NAMES, modals=modals).eval()


def project_sum(adapter, taps, indices):
    acc = None
    for i in indices:
        pj = adapter.project_sensor_taps(i, taps[i])
        acc = pj if acc is None else [a + b for a, b in zip(acc, pj)]
    return acc


def manual_spm(spm, xs):
    acc = [None] * 4
    for i in range(len(xs)):
        stem = spm.stems[i] if spm.mode == 'per_modal' else spm.stem
        raws = stem(xs[i])
        for k in range(4):
            pj = spm.fcs[i][k](raws[k])
            acc[k] = pj if acc[k] is None else acc[k] + pj
    return acc


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    xs = [torch.randn(1, 3, 64, 64) for _ in NAMES]
    taps = [[torch.randn(1, 8, 2, 2) for _ in range(4)] for _ in NAMES]
    all_adapter = make_adapter(None)
    with torch.no_grad():
        expected_spm = manual_spm(all_adapter.spm, xs)
        actual_spm = all_adapter.spm(xs)
        assert all(torch.equal(a, b) for a, b in zip(expected_spm, actual_spm))
        all_taps = project_sum(all_adapter, taps, range(len(NAMES)))
        output = all_adapter(xs, all_taps)
        original_forward = all_adapter.spm.forward
        all_adapter.spm.forward = lambda inputs: manual_spm(all_adapter.spm, inputs)
        expected_output = all_adapter(xs, all_taps)
        all_adapter.spm.forward = original_forward
        assert all(torch.equal(a, b) for a, b in zip(output, expected_output))

    rgb_adapter = make_adapter(['img'])
    assert rgb_adapter.modal_indices == (0,)
    assert tuple(all_adapter.state_dict()) == tuple(rgb_adapter.state_dict())
    xs_changed = [xs[0], torch.randn_like(xs[1]), torch.randn_like(xs[2])]
    taps_changed = [taps[0], [torch.randn_like(t) for t in taps[1]],
                    [torch.randn_like(t) for t in taps[2]]]
    with torch.no_grad():
        a = rgb_adapter(xs, project_sum(rgb_adapter, taps, rgb_adapter.modal_indices))
        b = rgb_adapter(xs_changed, project_sum(rgb_adapter, taps_changed,
                                               rgb_adapter.modal_indices))
        assert all(torch.equal(u, v) for u, v in zip(a, b))
        xs_rgb_changed = [torch.randn_like(xs[0]), xs[1], xs[2]]
        c = rgb_adapter(xs_rgb_changed,
                        project_sum(rgb_adapter, taps, rgb_adapter.modal_indices))
        assert any(not torch.equal(u, v) for u, v in zip(a, c))

    rgb_adapter.zero_grad(set_to_none=True)
    rgb_adapter(xs, project_sum(rgb_adapter, taps, rgb_adapter.modal_indices))[0].sum().backward()
    for i in (1, 2):
        for module in (rgb_adapter.spm.fcs[i], rgb_adapter.tap_proj[i]):
            for p in module.parameters():
                assert p.grad is None or torch.count_nonzero(p.grad) == 0
    assert any(p.grad is not None for p in rgb_adapter.spm.fcs[0].parameters())
    assert any(p.grad is not None for p in rgb_adapter.tap_proj[0].parameters())
    for bad in ([], ['unknown']):
        try:
            make_adapter(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid MODALS accepted: {bad}')
    print('SMOKE OK')


if __name__ == '__main__':
    main()
