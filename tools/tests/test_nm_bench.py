"""missing_modality_eval NM bench 정의와 기존 sp 경로의 CPU 회귀 검사."""

import argparse
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools import missing_modality_eval as nm


def generator(seed=37):
    return torch.Generator().manual_seed(seed)


def reference_sp(x, density, gen):
    """공개 val_mm_NM.py 의 randint 좌표 순서를 독립적으로 재현."""
    out = x.clone()
    count = math.ceil(density * x.numel() * 0.5)
    salt_coords = tuple(torch.randint(0, size, (count,), generator=gen)
                        for size in x.shape)
    out[salt_coords] = x.max()
    pepper_coords = tuple(torch.randint(0, size, (count,), generator=gen)
                          for size in x.shape)
    out[pepper_coords] = x.min()
    return out


class NMBenchTest(unittest.TestCase):
    def test_default_sp_matches_existing_function_bytes(self):
        x = torch.randn(1, 3, 64, 64, generator=generator(11))
        cases = nm.build_cases("nm", 1, ["img"], [], [0.1], generator)
        self.assertEqual([c.id for c in cases], ["clean", "nm0.1"])
        actual = cases[1].apply_fn([x])[0]
        expected = nm.sp_noise(x, 0.1, generator())[0]
        self.assertEqual(actual.numpy().tobytes(), expected.numpy().tobytes())
        print("default sp: byte-identical to sp_noise")

    def test_bench_sp_counts_channels_and_exact_reference(self):
        x = torch.randn(1, 3, 64, 64, generator=generator(5))
        original = x.clone()
        d = 0.1
        actual = nm.bench_sp_noise(x, d, generator())
        expected = reference_sp(x, d, generator())
        self.assertTrue(torch.equal(actual, expected))
        count = math.ceil(d * x.numel() * 0.5)
        salt = actual == x.max()
        pepper = actual == x.min()
        for name, mask in (("salt", salt), ("pepper", pepper)):
            self.assertGreater(mask.sum().item(), 0.85 * count, name)
            self.assertLessEqual(mask.sum().item(), count, name)
            self.assertFalse(torch.equal(mask[0, 0], mask[0, 1]), name)
        self.assertTrue(torch.equal(x, original))
        print(f"bench S&P: exact reference; salt={salt.sum().item()}, "
              f"pepper={pepper.sum().item()}, requested each={count}")

    def test_gaussian_skips_event_by_name(self):
        names = ["img", "event", "lidar"]  # MUSES 순서와 같은 3센서 사례
        base = [torch.randn(1, 3, 64, 64, generator=generator(i))
                for i in range(3)]
        d, sigma = 0.1, 0.2
        actual = nm.bench_noise(base, names, d, sigma, generator())
        ref_gen = generator()
        sp_only = [reference_sp(x, d, ref_gen) for x in base]
        self.assertTrue(torch.equal(actual[1], sp_only[1]))
        changed = actual[1] != base[1]
        self.assertTrue(torch.all((actual[1][changed] == base[1].max()) |
                                  (actual[1][changed] == base[1].min())))
        for i in (0, 2):
            expected = sp_only[i] + torch.randn(sp_only[i].size(), generator=ref_gen) * sigma
            self.assertTrue(torch.equal(actual[i], expected))
            observed_std = (actual[i] - sp_only[i]).std().item()
            self.assertAlmostEqual(observed_std, sigma, delta=0.01)
        print("Gaussian: Event unchanged after S&P; img/lidar std near sigma")

    def test_bench_cases_seed_and_legacy_flags(self):
        names = ["img", "depth", "event", "lidar"]
        levels = nm.parse_nm_bench_levels("0.05:0.1,0.1:0.2,0.2:0.5")

        def cases():
            return nm.build_cases("nm", 4, names, [], [0.9], generator,
                                  nm_gaussian=True, nm_gaussian_std=9.0,
                                  nm_mode="bench", nm_bench_levels=levels)
        first, second = cases(), cases()
        self.assertEqual(len(first), 4)
        self.assertEqual([c.label.split(" ", 1)[0] for c in first[1:]],
                         ["Low", "Mid", "High"])
        for c in first[1:]:
            self.assertIn(f"d={c.density}", c.id)
            self.assertIn(f"sigma={c.std}", c.id)
            self.assertIn(f"σ={c.std}", c.label)
        base = [torch.randn(1, 2, 8, 8, generator=generator(i)) for i in range(4)]
        for a, b in zip(first[1:], second[1:]):
            for xa, xb in zip(a.apply_fn(base), b.apply_fn(base)):
                self.assertTrue(torch.equal(xa, xb))
        with self.assertRaises(argparse.ArgumentTypeError):
            nm.parse_nm_bench_levels("0.1:0.2,0.2:0.5")
        print("bench cases: Low/Mid/High IDs, labels, legacy flags ignored, seed stable")

    def test_bench_summary_and_csv(self):
        names = ["img", "event", "lidar"]
        levels = nm.parse_nm_bench_levels("0.03:0.12,0.12:0.3,0.3:0.6")
        cases = nm.build_cases("nm", 3, names, [], [0.9], generator,
                               nm_mode="bench", nm_bench_levels=levels)
        hists = {c.id: np.eye(2, dtype=np.int64) for c in cases}
        with tempfile.TemporaryDirectory() as tmp:
            base, summary = nm.write_outputs(
                tmp, "val", cases, hists, 3, names, ["a", "b"], [], [0.9],
                "nm", False, True, 9.0, "model.pth", "config.yaml",
                nm_mode="bench")
            self.assertEqual(list(summary["NM"]), ["Low", "Mid", "High"])
            self.assertEqual(summary["NM"]["Low"]["density"], 0.03)
            self.assertEqual(summary["NM"]["High"]["sigma"], 0.6)
            definition = summary["protocol_defs"]["NM"]
            self.assertIn("Event 제외", definition)
            self.assertIn("원소별 위치", definition)
            self.assertIn("σ", definition)
            self.assertIn("Low (d=0.03, σ=0.12)", definition)
            self.assertIn("label", (Path(base) / "nm.csv").read_text())
            self.assertIn("Gaussian σ", (Path(base) / "summary.md").read_text())
        print("bench outputs: level mIoU, definition, CSV labels, Markdown table")


if __name__ == "__main__":
    unittest.main(verbosity=2)
