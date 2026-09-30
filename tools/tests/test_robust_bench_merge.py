#!/usr/bin/env python3
"""detectron2 없이 robust_bench_eval 샤드 병합을 검증한다."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402
from tools.baseline_failure.robust_bench_eval import build_all_cases, main  # noqa: E402


def run_merge(paths, out):
    previous = sys.argv
    try:
        sys.argv = ["robust_bench_eval.py", "--merge", *map(str, paths),
                    "--out", str(out)]
        main()
    finally:
        sys.argv = previous


def main_test():
    modals = ["CAMERA", "LIDAR", "EVENT", "DEPTH"]
    cases, gaussian = build_all_cases("all", modals, [0.25, 0.5, 0.75],
                                      [0.05, 0.1, 0.2], 0.2, 0)
    all_cases = cases + [gaussian]
    ids = [c.id for c in all_cases]
    assert len(ids) == 61 and len(set(ids)) == 61
    assert ids[0] == "clean"

    rng = np.random.default_rng(20260930)
    hists = rng.integers(0, 20, size=(61, 25, 25), dtype=np.int64)
    hists[:, np.arange(25), np.arange(25)] += 200

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = []
        for i, indices in enumerate(np.array_split(np.arange(61), 3)):
            path = root / f"shard_{i}.npz"
            np.savez(path, ids=np.array([ids[j] for j in indices]),
                     hists=hists[indices], modals=np.array(modals), n_seen=2)
            paths.append(path)

        run_merge(paths, root / "output")
        summary_path = root / "output" / "test" / "missing_modality" / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_miou = round(common.global_miou_from_cm(hists[0])[1], 4)
        assert summary["clean_mIoU"] == expected_miou
        assert "NM_gaussian" in summary
        assert summary["NM_gaussian"]["std"] == 0.2
        assert all(str(path) in summary["model_path"] for path in paths)
        assert summary["cfg"] == summary["model_path"]

        duplicate_path = root / "duplicate.npz"
        np.savez(duplicate_path, ids=np.array([ids[0]]), hists=hists[:1],
                 modals=np.array(modals), n_seen=2)
        run_merge([*paths, duplicate_path], root / "duplicate_ok")
        bad_duplicate_path = root / "duplicate_bad.npz"
        np.savez(bad_duplicate_path, ids=np.array([ids[0]]), hists=hists[:1] + 1,
                 modals=np.array(modals), n_seen=2)
        try:
            run_merge([*paths, bad_duplicate_path], root / "duplicate_bad")
        except RuntimeError as exc:
            assert ids[0] in str(exc)
        else:
            raise AssertionError("내용이 다른 중복 케이스 id 를 허용했다")

        missing_path = root / "shard_missing.npz"
        with np.load(paths[-1], allow_pickle=False) as last:
            np.savez(missing_path, ids=last["ids"][:-1],
                     hists=last["hists"][:-1], modals=last["modals"],
                     n_seen=last["n_seen"])
        try:
            run_merge([*paths[:-1], missing_path], root / "incomplete")
        except RuntimeError as exc:
            assert ids[-1] in str(exc)
        else:
            raise AssertionError("누락된 케이스 id 를 허용했다")

    print("OK")


if __name__ == "__main__":
    main_test()
