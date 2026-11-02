"""Inference bank for the qualitative figures: whole-split predictions -> per-image scores -> candidate scenes.

Subcommands (run from _paper/ieee-access-2026; registry = figures/qualbank/registry.yaml):
  status                 table of datasets x models: bank count, scores present, status/note
  ingest  DS MODEL       copy MODEL's whole-split predictions (and DS gt, once) into the bank at GT resolution,
                         write <model>/MANIFEST.json; a model with resize_from is resized with nearest neighbour
                         (tools/baseline_failure/common.resize_nearest, the same as the scoring)
  score   DS [MODEL..]   per-image confusion matrix hist[gt, pred] of every ready model -> scores/<model>.npz;
                         prints the dataset mIoU (Metrics.compute_iou definition) next to expected_miou
  candidates DS          per-image table + ranked candidate scenes -> figures/source_data/qualbank/<DS>_*.csv
  all DS                 ingest every ready model that is not in the bank yet, score, candidates

Definitions (all from tools/baseline_failure/common.py, nothing re-implemented):
  image mIoU   = mean IoU over the classes present in that image's ground truth (common.per_image_iou)
  thin mIoU    = the same, restricted to the dataset's thin_classes present in the image
  delta        = image mIoU(ours) - max over `compare` models; delta_<m> = ours - m
  class delta  = IoU_c(ours) - max_m IoU_c(m), only where class c has >= MIN_CLASS_PX ground-truth pixels
Candidates (written in full, never just the winners):
  top      rank by delta within each group (condition), greedy: no two picks of a group from one sequence
  reverse  the same, ascending (scenes where a compared model is best) -> limitation figures
  typical  the TYPICAL_K scenes closest to the group's median delta -> "not cherry-picked" rows
  class    for each thin class, scenes with the largest class delta (one per sequence)
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

PAPER = Path(__file__).resolve().parents[2]
REPO = PAPER.parents[1]
sys.path.insert(0, str(REPO))
from tools.baseline_failure import common  # noqa: E402

REG = PAPER / "figures" / "qualbank" / "registry.yaml"
OUT = PAPER / "figures" / "source_data" / "qualbank"
TOP_K, REVERSE_K, TYPICAL_K, CLASS_K = 10, 5, 3, 6
MIN_CLASS_PX = 400


def load_reg():
    return yaml.safe_load(open(REG, encoding="utf-8"))


def classes_of(ds):
    src = (REPO / ds["classes_from"]).read_text(encoding="utf-8")
    block = src[src.index("CLASSES"):]
    block = block[block.index("[") + 1:block.index("]")]
    return re.findall(r"""["']([^"']+)["']""", block)


def palette_of(ds):
    src = (REPO / ds["classes_from"]).read_text(encoding="utf-8")
    block = src[src.index("PALETTE"):]
    block = block[:block.index("])") + 2]
    rgb = [tuple(int(v) for v in m) for m in re.findall(r"\[\s*(\d+),\s*(\d+),\s*(\d+)\s*\]", block)]
    lut = np.zeros((256, 3), np.uint8)
    lut[:len(rgb)] = np.array(rgb, np.uint8)
    return lut, len(rgb)


def bank_dir(reg, name):
    return Path(reg["bank_root"]) / f"{name}_{reg['datasets'][name]['split']}"


def list_ids(root):
    return sorted(str(p.relative_to(root).with_suffix("")) for p in Path(root).rglob("*.png"))


def group_of(ds, image_id):
    for r in ds["group_rules"]:
        if "contains" in r and r["contains"] in image_id:
            return r["group"]
        if "regex" in r:
            m = re.search(r["regex"], image_id)
            if m:
                return m.expand(r["group"])
    return "all"


def sequence_of(ds, image_id):
    rx = ds.get("sequence_regex")
    m = re.search(rx, image_id) if rx else None
    return m.group(1) if m else image_id


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ ingest
def _copy_one(args):
    src, dst, gt, resize = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not resize:
        shutil.copyfile(src, dst)
        return
    g = common.load_label_png(gt)
    p = common.load_label_png(src)
    assert p.shape == (resize, resize), (src, p.shape)
    common.save_label_png(dst, common.resize_nearest(p, g.shape[0], g.shape[1]))


def ingest(reg, name, model, workers=8):
    ds = reg["datasets"][name]
    bd = bank_dir(reg, name)
    gt_bank = bd / "gt"
    if not gt_bank.exists():
        assert ds.get("gt_source"), f"{name}: gt_source missing in registry"
        ids = list_ids(ds["gt_source"])
        assert len(ids) == ds["n_images"], (len(ids), ds["n_images"])
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_copy_one, [(Path(ds["gt_source"]) / f"{i}.png", gt_bank / f"{i}.png", None, None)
                                    for i in ids], chunksize=32))
        json.dump({"source": ds["gt_source"], "count": len(ids)}, open(gt_bank / "MANIFEST.json", "w"), indent=1)
    m = ds["models"][model]
    assert m["status"] == "ready", f"{model} is {m['status']}: {m.get('note')}"
    src = Path(m["source"])
    ids = list_ids(src)
    gt_ids = list_ids(gt_bank)
    assert ids == gt_ids, f"{model}: {len(ids)} predictions vs {len(gt_ids)} gt, or different ids"
    dst = bd / model
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_copy_one, [(src / f"{i}.png", dst / f"{i}.png", gt_bank / f"{i}.png", m.get("resize_from"))
                                for i in ids], chunksize=32))
    json.dump({"source": str(src), "ckpt": m.get("ckpt"), "resize_from": m.get("resize_from"),
               "count": len(ids), "script": "figures/qualbank/qualbank.py ingest"},
              open(dst / "MANIFEST.json", "w"), indent=1)
    print(f"[ingest] {name}/{model}: {len(ids)} images -> {dst}")


# ------------------------------------------------------------------ score
def _cm_one(args):
    pred, gt, n = args
    return common.confusion_matrix(common.load_label_png(pred), common.load_label_png(gt), n=n)


def score(reg, name, models=None, workers=8):
    ds = reg["datasets"][name]
    _, n = palette_of(ds)
    bd = bank_dir(reg, name)
    ids = list_ids(bd / "gt")
    (bd / "scores").mkdir(exist_ok=True)
    for model in models or [k for k, v in ds["models"].items() if (bd / k).exists()]:
        with ProcessPoolExecutor(workers) as ex:
            cms = list(ex.map(_cm_one, [(bd / model / f"{i}.png", bd / "gt" / f"{i}.png", n) for i in ids],
                              chunksize=16))
        cm = np.stack(cms).astype(np.uint32)
        np.savez_compressed(bd / "scores" / f"{model}.npz", ids=np.array(ids), cm=cm)
        _, miou = common.global_miou_from_cm(cm.sum(0))
        exp = ds["models"][model].get("expected_miou")
        flag = "" if exp is None else ("  OK" if abs(miou - exp) < 0.05 else f"  MISMATCH vs expected {exp}")
        print(f"[score] {name}/{model}: dataset mIoU {miou:.2f}{flag}")


# ------------------------------------------------------------------ candidates
def candidates(reg, name):
    ds = reg["datasets"][name]
    bd = bank_dir(reg, name)
    cls = classes_of(ds)
    thin = [cls.index(c) for c in ds.get("thin_classes", []) if c in cls]
    models = [m for m in ds["models"] if (bd / "scores" / f"{m}.npz").exists()]
    ours, comp = ds["ours"], [m for m in ds["compare"] if m in models]
    assert ours in models and comp, f"need scores for {ours} and at least one of {ds['compare']}"
    S = {m: np.load(bd / "scores" / f"{m}.npz") for m in models}
    ids = list(S[ours]["ids"])
    for m in models:
        assert list(S[m]["ids"]) == ids, m
    iou = {m: np.stack([common.per_image_iou(c) for c in S[m]["cm"]]) for m in models}   # (N, K), NaN = absent
    gt_px = S[ours]["cm"].sum(2)                                                          # (N, K)
    nanmean = lambda a: np.array([common.nanmean(r) for r in a])
    miou = {m: nanmean(iou[m]) for m in models}
    tmiou = {m: nanmean(iou[m][:, thin]) if thin else np.full(len(ids), np.nan) for m in models}
    best = np.max(np.stack([miou[m] for m in comp]), 0)
    delta = miou[ours] - best
    grp = [group_of(ds, i) for i in ids]
    seq = [sequence_of(ds, i) for i in ids]
    OUT.mkdir(parents=True, exist_ok=True)

    import csv
    with open(OUT / f"{name}_per_image.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group", "sequence"] + [f"mIoU_{m}" for m in models]
                   + [f"thin_{m}" for m in models] + ["best_compare", "delta"] + [f"delta_{m}" for m in comp])
        for k, i in enumerate(ids):
            bc = comp[int(np.argmax([miou[m][k] for m in comp]))]
            w.writerow([i, grp[k], seq[k]] + [f"{miou[m][k]:.4f}" for m in models]
                       + [f"{tmiou[m][k]:.4f}" for m in models] + [bc, f"{delta[k]:+.4f}"]
                       + [f"{miou[ours][k] - miou[m][k]:+.4f}" for m in comp])

    def pick(order, k):
        out, used = [], set()
        for j in order:
            if seq[j] in used or np.isnan(delta[j]):
                continue
            out.append(j); used.add(seq[j])
            if len(out) == k:
                break
        return out

    rows = []
    for g in sorted(set(grp)):
        idx = np.array([k for k in range(len(ids)) if grp[k] == g])
        d = delta[idx]
        for kind, order, kk in [("top", idx[np.argsort(-d)], TOP_K), ("reverse", idx[np.argsort(d)], REVERSE_K),
                                ("typical", idx[np.argsort(np.abs(d - np.nanmedian(d)))], TYPICAL_K)]:
            for r, j in enumerate(pick(order, kk), 1):
                rows.append([kind, g, r, ids[j], seq[j], "", f"{delta[j]:+.4f}"]
                            + [f"{miou[m][j]:.4f}" for m in models])
    for c in thin:
        valid = (gt_px[:, c] >= MIN_CLASS_PX) & ~np.isnan(iou[ours][:, c])
        cd = iou[ours][:, c] - np.max(np.stack([np.nan_to_num(iou[m][:, c], nan=0.0) for m in comp]), 0)
        cd = np.where(valid, cd, -np.inf)
        for r, j in enumerate(pick(np.argsort(-cd), CLASS_K), 1):
            rows.append(["class", cls[c], r, ids[j], seq[j], f"{cd[j]:+.4f}", f"{delta[j]:+.4f}"]
                        + [f"{miou[m][j]:.4f}" for m in models])
    with open(OUT / f"{name}_candidates.csv", "w", newline="") as f:
        f.write(f"# rule: delta = image mIoU({ours}) - max({', '.join(comp)}); image mIoU over GT-present classes "
                f"(tools/baseline_failure/common.per_image_iou). top/reverse/typical per group, one scene per "
                f"sequence; class = thin-class IoU gain (>= {MIN_CLASS_PX} GT px). bank: {bd}\n")
        w = csv.writer(f)
        w.writerow(["kind", "group", "rank", "image_id", "sequence", "class_delta", "delta"]
                   + [f"mIoU_{m}" for m in models])
        w.writerows(rows)
    json.dump({"script": "figures/qualbank/qualbank.py candidates", "registry_md5": md5(REG),
               "scores": {m: md5(bd / "scores" / f"{m}.npz") for m in models}},
              open(OUT / f"{name}_candidates.json", "w"), indent=1)
    print(f"[candidates] {name}: {len(ids)} images, {len(rows)} candidate rows -> {OUT}")


def status(reg):
    for name, ds in reg["datasets"].items():
        bd = bank_dir(reg, name)
        g = (bd / "gt" / "MANIFEST.json")
        print(f"== {name} ({ds['split']}, {ds['n_images']} images)  gt in bank: {'yes' if g.exists() else 'no'}")
        for m, v in ds["models"].items():
            man = bd / m / "MANIFEST.json"
            cnt = json.load(open(man))["count"] if man.exists() else 0
            sc = "scored" if (bd / "scores" / f"{m}.npz").exists() else "-"
            print(f"   {m:14s} {v['status']:8s} bank {cnt:5d}  {sc:6s} {v.get('note', '')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "ingest", "score", "candidates", "all"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    reg = load_reg()
    if a.cmd == "status":
        status(reg)
    elif a.cmd == "ingest":
        ingest(reg, a.args[0], a.args[1], a.workers)
    elif a.cmd == "score":
        score(reg, a.args[0], a.args[1:] or None, a.workers)
    elif a.cmd == "candidates":
        candidates(reg, a.args[0])
    elif a.cmd == "all":
        name = a.args[0]
        for m, v in reg["datasets"][name]["models"].items():
            if v["status"] == "ready" and not (bank_dir(reg, name) / m / "MANIFEST.json").exists():
                ingest(reg, name, m, a.workers)
        score(reg, name, None, a.workers)
        candidates(reg, name)
