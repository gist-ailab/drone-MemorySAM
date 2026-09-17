#!/usr/bin/env python3
"""
D4 — CAFuser 내부 기제 프로브(기준선 저장소 전용). DGFusion 프로브와 골격은 같되
CAFuser 고유의 두 가지만 잰다:
  - 조건 토큰의 조건 분류 정확도(텍스트 대조 감독이 실제로 작동하는가)
  - CAA/CA² 모달 가중(조건별)

⚠️ 모듈 이름은 정규식으로 찾고 못 찾으면 명확한 에러로 멈춘다(추측 금지). 첫 실행에서
`--list-modules` 로 이름을 확인하라. detectron2 는 지연 import.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure.probe_dgfusion import (  # noqa: E402
    ActivationProbe, tensor_stats, _to_np)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file")
    ap.add_argument("--weights")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list-modules", action="store_true")
    ap.add_argument("--cond-token-regex", default=r"cond.*token|condition")
    ap.add_argument("--caa-regex", default=r"caa|ca2|modal.*attn|cross.*modal")
    args = ap.parse_args()

    if not args.config_file or not args.weights:
        ap.error("--config-file 과 --weights 는 실행에 필요(스모크는 인자 파싱만 검사)")

    import torch  # noqa: F401
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import build_detection_test_loader
    from train_net import Trainer, setup

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
    ctok = probe.watch(args.cond_token_regex, "cond_token")
    caa = probe.watch(args.caa_regex, "caa")
    print(f"[probe_cafuser] cond_token={ctok} caa={caa}")

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
        caps = probe.captured.get("cond_token", {})
        if caps:
            st = tensor_stats(_to_np(next(iter(caps.values()))))
            rec["cond_token_mean"], rec["cond_token_var"] = st["mean"], st["var"]
        cap_caa = probe.captured.get("caa", {})
        if cap_caa:
            st = tensor_stats(_to_np(next(iter(cap_caa.values()))))
            rec["caa_mean"], rec["caa_var"] = st["mean"], st["var"]
        rows.append(rec)

    probe.remove()
    if args.out and rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"[probe_cafuser] {len(rows)} 행 -> {args.out}")


if __name__ == "__main__":
    main()
