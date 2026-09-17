#!/usr/bin/env python3
"""
D4 — DGFusion 내부 기제 프로브(기준선 저장소 전용). forward hook 으로 다음을
이미지별로 잰다:
  (a) depth 보조 헤드 출력 vs GT depth 의 AbsRel·d1
  (b) 로컬 depth 토큰·전역 조건 토큰 텐서의 평균/분산
  (c) 각 레벨 cross-attention 의 softmax 가중을 K/V 모달 구간별로 합산(RGB 쿼리 기준
      모달별 가중)
BF_ZERO_DEPTH_TOKEN=1 이면 depth 토큰을 0 으로 치환한 추론과 대조한다.

⚠️ 모듈 이름은 저장소마다 다르다. 정규식으로 모듈을 찾고 **못 찾으면 명확한 에러로
멈춘다**(추측으로 아무 층이나 잡지 않는다). 첫 실행에서 `--list-modules` 로 이름을
확인한 뒤 정규식을 맞춰라. 우리 쪽 대응 측정은 tools/module_diagnostics.py 참조.

detectron2 는 지연 import — tools 패키지 스모크(detectron2 부재)에서도 이 모듈은
import·인자 파싱이 되어야 한다.
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def find_modules(model, pattern):
    """정규식에 이름이 매칭되는 (name, module) 목록. 없으면 RuntimeError."""
    rx = re.compile(pattern)
    hits = [(n, m) for n, m in model.named_modules() if n and rx.search(n)]
    if not hits:
        raise RuntimeError(
            f"정규식 '{pattern}' 에 맞는 모듈이 없다. --list-modules 로 실제 이름을 "
            f"확인하고 정규식을 맞춰라(추측 금지).")
    return hits


class ActivationProbe:
    """이름 정규식으로 고른 모듈에 forward hook 을 걸어 출력 텐서를 모은다."""

    def __init__(self, model):
        self.model = model
        self._handles = []
        self.captured = {}

    def watch(self, pattern, tag):
        hits = find_modules(self.model, pattern)
        for name, mod in hits:
            def hook(_m, _i, out, _name=name, _tag=tag):
                self.captured.setdefault(_tag, {})[_name] = out
            self._handles.append(mod.register_forward_hook(hook))
        return [n for n, _ in hits]

    def clear(self):
        self.captured = {}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def _to_np(t):
    import torch
    if isinstance(t, (tuple, list)):
        t = t[0]
    if torch.is_tensor(t):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def tensor_stats(arr):
    a = np.asarray(arr, dtype=np.float64).ravel()
    return {"mean": float(a.mean()), "var": float(a.var()),
            "min": float(a.min()), "max": float(a.max())}


def depth_absrel_d1(pred_depth, gt_depth, mask=None):
    """AbsRel = mean(|p-g|/g), d1 = mean(max(p/g,g/p)<1.25). 유효 픽셀만."""
    p = np.asarray(pred_depth, dtype=np.float64).ravel()
    g = np.asarray(gt_depth, dtype=np.float64).ravel()
    valid = np.isfinite(p) & np.isfinite(g) & (g > 1e-3)
    if mask is not None:
        valid &= np.asarray(mask).ravel().astype(bool)
    if not valid.any():
        return float("nan"), float("nan")
    p, g = p[valid], g[valid]
    absrel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / p)
    d1 = float(np.mean(ratio < 1.25))
    return absrel, d1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", help="DGFusion detectron2 config")
    ap.add_argument("--weights", help="체크포인트 .pth")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", help="이미지별 CSV 출력 경로")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list-modules", action="store_true",
                    help="모델 모듈 이름을 출력하고 종료(정규식 잡을 때 사용)")
    ap.add_argument("--depth-head-regex", default=r"depth.*head|head.*depth")
    ap.add_argument("--depth-token-regex", default=r"depth.*token|depth.*embed")
    ap.add_argument("--cond-token-regex", default=r"cond.*token|condition.*embed")
    ap.add_argument("--xattn-regex", default=r"cross.*attn|fusion.*attn")
    args = ap.parse_args()

    if not args.config_file or not args.weights:
        ap.error("--config-file 과 --weights 는 실행에 필요(스모크는 인자 파싱만 검사)")

    # 지연 import — 기준선 repo 에서만 사용 가능.
    import torch  # noqa: F401
    from detectron2.checkpoint import DetectionCheckpointer
    from train_net import Trainer, setup  # 기준선 repo 루트에서 실행

    class _A:
        config_file = args.config_file
        opts = ["MODEL.WEIGHTS", args.weights, "MODEL.IS_TRAIN", "False"]
        eval_only = True
        inference_only = False
        resume = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "auto"

    cfg = setup(_A)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(args.weights, resume=False)
    model.eval()

    if args.list_modules:
        for n, _m in model.named_modules():
            if n:
                print(n)
        return

    probe = ActivationProbe(model)
    depth_names = probe.watch(args.depth_head_regex, "depth_head")
    dtok_names = probe.watch(args.depth_token_regex, "depth_token")
    ctok_names = probe.watch(args.cond_token_regex, "cond_token")
    xattn_names = probe.watch(args.xattn_regex, "xattn")
    print(f"[probe_dgfusion] depth_head={depth_names} depth_token={dtok_names}\n"
          f"  cond_token={ctok_names} xattn={xattn_names}")
    if os.environ.get("BF_ZERO_DEPTH_TOKEN") == "1":
        print("[probe_dgfusion] BF_ZERO_DEPTH_TOKEN=1 — depth 토큰 0 치환 모드")

    from detectron2.data import build_detection_test_loader
    ds_name = cfg.DATASETS.TEST_SEMANTIC[0] if hasattr(cfg.DATASETS, "TEST_SEMANTIC") else cfg.DATASETS.TEST[0]
    loader = build_detection_test_loader(cfg, ds_name)

    rows = []
    for i, batch in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        probe.clear()
        with torch.no_grad():
            _ = model(batch)
        rec = {"idx": i, "file_name": batch[0].get("file_name", "")}
        for tag in ("depth_token", "cond_token"):
            caps = probe.captured.get(tag, {})
            if caps:
                st = tensor_stats(_to_np(next(iter(caps.values()))))
                rec[f"{tag}_mean"], rec[f"{tag}_var"] = st["mean"], st["var"]
        rows.append(rec)

    probe.remove()
    if args.out and rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"[probe_dgfusion] {len(rows)} 행 -> {args.out}")


if __name__ == "__main__":
    main()
