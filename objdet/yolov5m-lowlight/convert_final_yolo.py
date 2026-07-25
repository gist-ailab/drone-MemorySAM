#!/usr/bin/env python3
"""final low-light split → YOLO(RGB-only). 모달리티 필터 없음(RGB 전 프레임 사용).
클립별 저조도/정상 평가를 위해 test 이미지 목록 txt 도 산출.

사용: python convert_final_yolo.py <ANN_DIR> <DATA_BASE> <OUT_DIR>
  ANN_DIR: instances_train.json / instances_test.json (+ split_info.json)
"""
import json, os, shutil, sys, collections

ANN, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
info = json.load(open(os.path.join(ANN, "split_info.json")))
LOWLIGHT = set(info["low_light_clips"])
names = None

def convert(split):
    global names
    coco = json.load(open(os.path.join(ANN, f"instances_{split}.json")))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    if names is None:
        names = [c["name"] for c in cats]
    ann_by = collections.defaultdict(list)
    for a in coco["annotations"]:
        ann_by[a["image_id"]].append(a)
    idir, ldir = f"{OUT}/images/{split}", f"{OUT}/labels/{split}"
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    lists = collections.defaultdict(list)   # 'all','lowlight','normal' → abs img paths
    n = 0
    for im in coco["images"]:
        rel = im["modalities"]["rgb"]
        cap = rel.split("/")[0]
        flat = rel.replace("/rgb/", "_").replace("/", "_")
        dst = os.path.join(idir, flat)
        src = os.path.join(BASE, rel)
        if not os.path.exists(src):
            continue
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        W, H = im["width"], im["height"]
        lines = []
        for a in ann_by.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx, cy = (x + w/2)/W, (y + h/2)/H
            lines.append(f"{cat_to_idx[a['category_id']]} {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} {min(w/W,1):.6f} {min(h/H,1):.6f}")
        open(os.path.join(ldir, os.path.splitext(flat)[0] + ".txt"), "w").write("\n".join(lines))
        ap = os.path.abspath(dst)
        lists["all"].append(ap)
        lists["lowlight" if cap in LOWLIGHT else "normal"].append(ap)
        n += 1
    if split == "test":
        for k, v in lists.items():
            open(f"{OUT}/test_{k}.txt", "w").write("\n".join(v) + "\n")
            print(f"  test_{k}: {len(v)} imgs")
    print(f"{split}: {n} imgs")

convert("train"); convert("test")
with open(f"{OUT}/final_rgb.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: test_all.txt\ntest: test_all.txt\nnames:\n")
    for i, nm in enumerate(names):
        f.write(f"  {i}: {nm}\n")
print("wrote final_rgb.yaml + test_{all,lowlight,normal}.txt ->", OUT)
