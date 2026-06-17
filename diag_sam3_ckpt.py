"""Diagnose whether SAM3 pretrained weights actually load into the tracker.

Why: LoRA_Sam3_RBMA freezes the SAM3 backbone and trains only rank-4 LoRA + a tiny
semantic head. If sam3.pt fails to load (key-name mismatch under strict=False), the
backbone stays RANDOM and frozen → features are garbage → val mIoU collapses (~2%).
This script answers "did the weights load?" definitively, with no multi-line paste.

Run (single line):
  PYTHONPATH=semseg/models/sam3 HF_HUB_OFFLINE=1 python diag_sam3_ckpt.py
"""
import os, sys
import torch

CKPT = "semseg/models/sam3/checkpoints/sam3.pt"

print(f"=== file ===")
print(f"path: {CKPT}")
print(f"exists: {os.path.isfile(CKPT)}")
if os.path.isfile(CKPT):
    print(f"size: {os.path.getsize(CKPT)/1e9:.3f} GB")

print(f"\n=== load ckpt state_dict ===")
sd = torch.load(CKPT, map_location="cpu")
sd = sd.get("model", sd.get("model_state_dict", sd))
print(f"ckpt keys: {len(sd)}")
print(f"sample ckpt keys:")
for k in list(sd.keys())[:10]:
    print(f"    {k}")

print(f"\n=== build tracker (random init) ===")
from sam3.model_builder import build_tracker
t = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
model_sd = t.state_dict()
print(f"tracker total keys: {len(model_sd)}")
print(f"sample tracker keys:")
for k in list(model_sd.keys())[:10]:
    print(f"    {k}")

print(f"\n=== load_state_dict(strict=False) ===")
miss, unexp = t.load_state_dict(sd, strict=False)
matched = len(model_sd) - len(miss)
print(f"matched (loaded):  {matched} / {len(model_sd)}")
print(f"missing (left RANDOM): {len(miss)}")
print(f"unexpected (in ckpt, unused): {len(unexp)}")

# the decisive number: are the ViT backbone weights actually filled?
def is_backbone(k):
    return any(tok in k for tok in ("trunk", "visual", "backbone", "vision"))

bb_total = [k for k in model_sd if is_backbone(k)]
bb_miss = [k for k in miss if is_backbone(k)]
print(f"\n=== BACKBONE (the part that must be pretrained) ===")
print(f"backbone keys total: {len(bb_total)}")
print(f"backbone keys MISSING (random!): {len(bb_miss)}")
if bb_total:
    pct = 100.0 * len(bb_miss) / len(bb_total)
    print(f"backbone RANDOM fraction: {pct:.1f}%")
    if pct > 50:
        print(">>> VERDICT: backbone is mostly RANDOM → this is why val mIoU ~2%.")
        print(">>> Likely key-name prefix mismatch. Compare the sample keys above.")
    elif len(miss) == 0:
        print(">>> VERDICT: full load, weights OK. Look elsewhere (decoder/head/eval).")
    else:
        print(">>> VERDICT: backbone loaded OK; some non-backbone keys missing (may be fine).")

print(f"\nsample MISSING keys:")
for k in miss[:15]:
    print(f"    {k}")
print(f"\nsample UNEXPECTED keys:")
for k in unexp[:15]:
    print(f"    {k}")
