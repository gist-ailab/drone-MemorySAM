"""Diagnose whether SAM3 pretrained weights actually load into the tracker.

Why: LoRA_Sam3_RBMA freezes the SAM3 backbone and trains only rank-4 LoRA + a tiny
semantic head. If sam3.pt fails to load (key-name mismatch under strict=False), the
backbone stays RANDOM and frozen -> features are garbage -> val mIoU collapses (~2%).

This script answers "did the weights load?" AND "how do we remap the keys?" with no
multi-line paste and no PYTHONPATH needed (it self-adds semseg/models/sam3).

Run:
  python diag_sam3_ckpt.py
"""
import os, sys
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "semseg/models/sam3"))

import torch

CKPT = "semseg/models/sam3/checkpoints/sam3.pt"


def prefix_hist(keys, depth=1):
    c = Counter(".".join(k.split(".")[:depth]) for k in keys)
    return c.most_common()


print("=== file ===")
print(f"path: {CKPT}  exists: {os.path.isfile(CKPT)}")
if os.path.isfile(CKPT):
    print(f"size: {os.path.getsize(CKPT)/1e9:.3f} GB")

print("\n=== ckpt state_dict ===")
sd = torch.load(CKPT, map_location="cpu")
sd = sd.get("model", sd.get("model_state_dict", sd))
ckpt_keys = list(sd.keys())
print(f"ckpt keys: {len(ckpt_keys)}")
print("top-level prefixes:", prefix_hist(ckpt_keys, 1))

print("\n=== build tracker (random init) ===")
from sam3.model_builder import build_tracker
t = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
model_sd = t.state_dict()
model_keys = list(model_sd.keys())
print(f"tracker keys: {len(model_keys)}")
print("top-level prefixes:", prefix_hist(model_keys, 1))
print("sample tracker keys:")
for k in model_keys[:8]:
    print("    ", k)

# ── strategy A: simple prefix-strip transforms on ckpt keys ──
print("\n=== prefix-strip match counts (transform ckpt keys -> intersect tracker keys) ===")
model_key_set = set(model_keys)
strategies = {
    "identity": lambda k: k,
    "strip 'detector.'": lambda k: k[len("detector."):] if k.startswith("detector.") else k,
    "strip 'tracker.'": lambda k: k[len("tracker."):] if k.startswith("tracker.") else k,
    "strip 'model.'": lambda k: k[len("model."):] if k.startswith("model.") else k,
    "strip 'detector.'->add nothing, also try 'tracker.'": None,  # placeholder
}
best = None
for name, fn in strategies.items():
    if fn is None:
        continue
    hit = sum(1 for k in ckpt_keys if fn(k) in model_key_set)
    print(f"  {name:24s}: {hit}/{len(model_keys)} tracker keys fillable")
    if best is None or hit > best[1]:
        best = (name, hit, fn)

# ── strategy B: shape-checked suffix match (robust, gives real remap) ──
print("\n=== suffix match (model key m <- unique ckpt key ending in m, shape-checked) ===")
remap = {}          # model_key -> ckpt_key
unmatched = []
ambiguous = 0
for m in model_keys:
    cand = [c for c in ckpt_keys if c == m or c.endswith("." + m)]
    cand = [c for c in cand if sd[c].shape == model_sd[m].shape]
    if len(cand) == 1:
        remap[m] = cand[0]
    elif len(cand) == 0:
        unmatched.append(m)
    else:
        ambiguous += 1
print(f"  matched (shape-ok, unique): {len(remap)}/{len(model_keys)}")
print(f"  unmatched: {len(unmatched)}   ambiguous: {ambiguous}")

def is_backbone(k):
    return any(tok in k for tok in ("trunk", "visual", "backbone", "vision"))

bb_total = [k for k in model_keys if is_backbone(k)]
bb_matched = [k for k in bb_total if k in remap]
print("\n=== BACKBONE (must be pretrained) ===")
print(f"backbone tracker keys: {len(bb_total)}   matched: {len(bb_matched)}")
if bb_total:
    rand_pct = 100.0 * (len(bb_total) - len(bb_matched)) / len(bb_total)
    print(f"backbone RANDOM fraction (current strict=False, identity load): "
          f"{100.0*sum(1 for k in bb_total if k not in model_key_set or k not in sd):.1f}% (identity)")
    print(f"backbone RANDOM fraction (after suffix-remap fix): {rand_pct:.1f}%")

print("\n=== VERDICT ===")
identity_hit = sum(1 for k in ckpt_keys if k in model_key_set)
print(f"current code (identity, strict=False) fills: {identity_hit}/{len(model_keys)} keys")
if identity_hit < 0.5 * len(model_keys) and len(remap) > 0.8 * len(model_keys):
    print(">>> CONFIRMED: current load leaves backbone RANDOM (key prefix mismatch).")
    print(f">>> FIX: best simple transform = '{best[0]}' ({best[1]} keys), or use suffix-remap.")
elif identity_hit > 0.8 * len(model_keys):
    print(">>> Weights load fine under current code. Look elsewhere (sem_head/eval).")
else:
    print(">>> Partial/odd match. Inspect samples below.")

print("\nsample tracker keys NOT matched by suffix-remap:")
for k in unmatched[:15]:
    print("    ", k)
print("\nsample remap (tracker_key <- ckpt_key):")
for m in list(remap)[:8]:
    print(f"    {m}  <-  {remap[m]}")
