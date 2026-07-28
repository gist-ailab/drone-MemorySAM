"""poongsan_v2 -> YOLO(RGB-only), restricted to the EXACT frames ViT-S+ (D1) used
(REQUIRE_ALL_MODALITIES: raw depth_map_lidar + thermal_aligned present). Apples-to
-apples vs D1: same frames, RGB channel only. Target: TRAIN 7043 / TEST 2066."""
import json, os, shutil, sys, collections
BASE = "/SSDd/jemo_maeng/dset/poongsan_v2"
ANN  = f"{BASE}/_final_ann"
OUT  = sys.argv[1]
SHARED_PATH = sys.argv[2]
SPLITS = {"train": "instances_train_egofill.json", "test": "instances_test_common.json"}

def keep(im):
    rel = im["modalities"]["rgb"]; clip = rel.split("/")[0]; ts = os.path.basename(rel)
    return (os.path.exists(f"{BASE}/{clip}/depth_map_lidar/{ts}") and
            os.path.exists(f"{BASE}/{clip}/thermal_aligned/{ts}"))

names = None
def convert(split, ann_file):
    global names
    coco = json.load(open(f"{ANN}/{ann_file}"))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    if names is None: names = [c["name"] for c in cats]
    ann_by = collections.defaultdict(list)
    for a in coco["annotations"]: ann_by[a["image_id"]].append(a)
    idir, ldir = f"{OUT}/images/{split}", f"{OUT}/labels/{split}"
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    subset = {"lowlight": [], "normal": []}
    n = 0
    for im in coco["images"]:
        if not keep(im): continue
        rel = im["file_name"]
        flat = rel.replace("/rgb/", "_").replace("/", "_")
        src, dst = f"{BASE}/{rel}", f"{idir}/{flat}"
        if not os.path.exists(src): continue
        if not os.path.exists(dst): shutil.copy2(src, dst)
        W, H = im["width"], im["height"]; lines = []
        for a in ann_by.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0: continue
            cx, cy = (x + w/2)/W, (y + h/2)/H
            lines.append(f"{cat_to_idx[a['category_id']]} {min(max(cx,0),1):.6f} "
                         f"{min(max(cy,0),1):.6f} {min(w/W,1):.6f} {min(h/H,1):.6f}")
        open(f"{ldir}/{os.path.splitext(flat)[0]}.txt", "w").write("\n".join(lines))
        if split == "test":
            subset["lowlight" if im.get("low_light") else "normal"].append(flat)
        n += 1
    print(f"{split}: {n} imgs")
    if split == "test":
        for k, v in subset.items():
            open(f"{OUT}/test_{k}_basenames.txt", "w").write("\n".join(sorted(v)) + "\n")
            print(f"  test_{k}: {len(v)}")

for sp, f in SPLITS.items(): convert(sp, f)
with open(f"{OUT}/poongsan_v2_rgb.yaml", "w") as f:
    f.write("# RGB-only, ViT-S+ same-frames subset (all-modal: raw lidar+thermal). test=2066 (night 1119/normal 947)\n")
    f.write(f"path: {SHARED_PATH}\ntrain: images/train\nval: images/test\ntest: images/test\nnames:\n")
    for i, nm in enumerate(names): f.write(f"  {i}: {nm}\n")
print("classes:", names, "-> wrote", OUT)
