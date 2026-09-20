"""
degradation.py 스모크 테스트 (2026-09-20).

네 항목을 합성 배열(실제 데이터셋을 읽지 않는다)로 검사하고 각 PASS/FAIL 을 찍는다.

  1. 꺼짐 불변  — DEGRADE.ENABLED=False 면 열화를 거친 배열이 원본과 바이트 동일.
  2. 비율 정확  — 패치 드롭 r 을 고정하면 실제로 평균으로 바뀐 화소 비율이 r 에 가깝다(±0.05).
  3. 재현      — 같은 seed 두 번은 바이트 동일, 다른 seed 는 다르다.
  4. 커리큘럼 단조 — 공유 step 0 -> 50% -> 100% 에서 current_severity 가 0.3 -> 0.6 -> 1.0.

의존은 numpy·cv2 만. cv2 가 없으면 그 사실을 보고하고 종료한다(테스트 실패로 세지 않는다).
"""

import multiprocessing as mp
import sys
import types

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError as exc:  # pragma: no cover
    print("SKIP: cv2 를 import 할 수 없다 — 이 환경에는 cv2 가 없다: %r" % (exc,))
    sys.exit(2)

import degradation as dg


def _make_cfg(enabled=True, per_modal_prob=0.5,
              curriculum=(0.3, 0.6, 1.0), fractions=(0.33, 0.66, 1.0)):
    return types.SimpleNamespace(
        ENABLED=enabled,
        PER_MODAL_PROB=per_modal_prob,
        CURRICULUM=list(curriculum),
        CURRICULUM_FRACTIONS=list(fractions),
        SEED=0,
    )


def _report(name, ok, detail=""):
    print("%-14s %s%s" % (name, "PASS" if ok else "FAIL",
                          ("  (%s)" % detail) if detail else ""))
    return ok


def test_disabled_invariant():
    cfg = _make_cfg(enabled=False)
    rng = np.random.default_rng(0)
    imgs = {
        "CAMERA": (np.arange(64 * 64 * 3, dtype=np.uint8) % 251).reshape(64, 64, 3),
        "DEPTH": (np.arange(64 * 64 * 3, dtype=np.uint8) % 199).reshape(64, 64, 3),
    }
    means = {"CAMERA": [40, 50, 60], "DEPTH": [70, 70, 70]}
    out = dg.degrade_sample(imgs, means, cfg, rng, max_iter=100)
    ok = all(np.array_equal(out[k], imgs[k]) and out[k].tobytes() == imgs[k].tobytes()
             for k in imgs)
    return _report("disabled", ok)


def test_ratio_accuracy():
    rng = np.random.default_rng(123)
    fill = np.array([50, 50, 50], dtype=np.uint8)
    img = np.full((64, 64, 3), 200, dtype=np.uint8)  # 원본은 평균값과 다르게
    r = 0.3
    out = dg.patch_drop(img, fill, sev=1.0, rng=rng, ratio=r, grid=8)
    changed = np.all(out == fill, axis=2)
    frac = float(changed.mean())
    ok = abs(frac - r) <= 0.05
    return _report("ratio", ok, "r=%.2f measured=%.4f" % (r, frac))


def test_reproducibility():
    cfg = _make_cfg(enabled=True, per_modal_prob=1.0)
    dg.set_shared_step(mp.Value("i", 0))  # 진행도 0 -> severity 상한 0.3 (결정적)
    imgs = {
        "CAMERA": (np.arange(64 * 64 * 3, dtype=np.uint8) % 251).reshape(64, 64, 3),
        "DEPTH": (np.arange(64 * 64 * 3, dtype=np.uint8) % 199).reshape(64, 64, 3),
        "LIDAR": (np.arange(64 * 64 * 3, dtype=np.uint8) % 197).reshape(64, 64, 3),
        "EVENT": (np.arange(64 * 64 * 3, dtype=np.uint8) % 193).reshape(64, 64, 3),
    }
    means = {"CAMERA": [40, 50, 60], "DEPTH": [70, 70, 70],
             "LIDAR": [80, 80, 80], "EVENT": [90, 90, 90]}

    def run(seed):
        rng = np.random.default_rng(seed)
        out = dg.degrade_sample(imgs, means, cfg, rng, max_iter=100)
        return b"".join(out[k].tobytes() for k in imgs)

    a, b, c = run(7), run(7), run(8)
    same_seed_equal = a == b
    diff_seed_differ = a != c
    ok = same_seed_equal and diff_seed_differ
    return _report("reproducible", ok,
                   "same=%s diff=%s" % (same_seed_equal, diff_seed_differ))


def test_curriculum_monotonic():
    cfg = _make_cfg()
    v = mp.Value("i", 0)
    dg.set_shared_step(v)
    got = []
    for step in (0, 50, 100):  # max_iter=100 -> frac 0, 0.5, 1.0
        v.value = step
        got.append(dg.current_severity(cfg, max_iter=100))
    expected = [0.3, 0.6, 1.0]
    ok = all(abs(g - e) < 1e-9 for g, e in zip(got, expected))
    return _report("curriculum", ok, "got=%s expected=%s" % (got, expected))


def main():
    results = [
        test_disabled_invariant(),
        test_ratio_accuracy(),
        test_reproducibility(),
        test_curriculum_monotonic(),
    ]
    print("---")
    print("%d/%d PASS" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
