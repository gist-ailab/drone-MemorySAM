"""스모크 ③: 열화 패치 off 일 때 학습 손실이 원본 코드와 바이트 동일한가 (GPU 1장).

같은 체크포인트(재개용 iter90k)·같은 샘플·같은 시드로 학습 모드 forward 한 번의 손실 딕셔너리를
float32 비트까지 비교한다. 코드 트리 두 개(패치 전 overlay / 패치 후)에서 같은 명령으로 돌려 json 을 대조한다.

  python smoke_cafb_loss.py <out.json> [--ckpt <model.pth>] [--config <yaml>] [--opts k v ...]
  python smoke_cafb_loss.py cmp a.json b.json
"""
import argparse
import copy
import json
import os
import random
import struct
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import torch  # noqa: E402


def bits(t):
    return struct.pack(">f", float(t)).hex()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    if len(sys.argv) > 3 and sys.argv[1] == "cmp":
        a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
        bad = [k for k in a if a[k] != b.get(k)]
        ok = not bad and a.keys() == b.keys()
        print(f"[cmp] 손실 항목 {len(a)}개 · 불일치 {len(bad)}개 →", "바이트 동일" if ok else f"불일치 {bad[:6]}")
        sys.exit(0 if ok else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ckpt", default="output/cafuser_swin_tiny_bs8_200k_deliver_clde_degrade/model_0089999.pth")
    ap.add_argument("--config", default="configs/deliver/swin/cafuser_swin_tiny_bs8_200k_deliver_clde.yaml")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog
    from train_net import Trainer, setup
    from cafuser.data.dataset_mappers.deliver_semantic_dataset_mapper import DELIVERSemanticDatasetMapper as CM

    class A:
        eval_only = False
        inference_only = False
        resume = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "auto"
    A.config_file = args.config
    A.opts = ["MODEL.WEIGHTS", args.ckpt, "OUTPUT_DIR", "/tmp/smoke_cafb_loss_out"] + list(args.opts)
    cfg = setup(A)
    seed_all(0)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(args.ckpt)
    model.train()
    ds = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    mapper = CM(cfg, True)
    batch = []
    for i in range(args.n):
        seed_all(500 + i)
        batch.append(mapper(copy.deepcopy(ds[i * 997 % len(ds)])))
    seed_all(7)
    with torch.no_grad():
        losses = model(batch)
    res = {k: bits(v) for k, v in sorted(losses.items())}
    total = sum(float(v) for v in losses.values())
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[loss] 항목 {len(res)}개 · 합계 {total:.6f} · 예) loss_ce {float(losses.get('loss_ce', -1)):.6f} loss_mask {float(losses.get('loss_mask', -1)):.6f} → {args.out}")
    print(f"[mem] max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB (bs{args.n}, 학습 모드 forward, no_grad)")


if __name__ == "__main__":
    main()
