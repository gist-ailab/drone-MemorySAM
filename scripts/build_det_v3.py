#!/usr/bin/env python3
"""v3 det split builder — NEW annotations (instances.json = v20260702_2303) + CLIP holdout.

지난번 v2와 동일한 **클립(캡처) 단위 holdout** (test = V2_TEST 캡처) 로 나누되, 갱신된
어노테이션(2026-07-02)을 사용하고 결과를 **v3** 로 명명한다. 새 어노의 modalities dict 는
lidar(depth_map_lidar) 를 빠뜨렸으나 디스크에는 존재(커버리지 ~53%, v2와 동일)하므로
rgb basename 으로 depth_map_lidar 경로를 복원해 v2(rgb+lidar+thermal) 드롭인으로 만든다.
(build_det_splits.py 의 v3=시간분할과 달리 여기 v3=클립 holdout — 사용자 지시 "클립 기준")

사용: python scripts/build_det_v3.py [BASE] [OUT]
"""
import json, glob, os, sys, collections

BASE = sys.argv[1] if len(sys.argv) > 1 else "/ailab_mat2/Projects/Drone/DATA/260618_poongsan"
OUT  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "_det_splits")
V2_TEST = {"capture_20260618_115206", "capture_20260618_114808"}  # v2와 동일 test 캡처

os.makedirs(OUT, exist_ok=True)
caps = sorted(glob.glob(os.path.join(BASE, "capture_*")))
CATS = None
gid = [0]

def new_store():
    return {"images": [], "annotations": [], "_aid": [0]}

def add_image(store, img, anns_by_old, capname):
    gid[0] += 1; nid = gid[0]
    im = dict(img); im["id"] = nid
    im["file_name"] = f"{capname}/{img['file_name']}"          # capture_XXX/rgb/<ts>.png
    bn = os.path.basename(img["file_name"])                    # <ts>.png
    mod = dict(img.get("modalities") or {})
    mod.setdefault("depth_map_lidar", f"depth_map_lidar/{bn}")  # lidar 경로 복원 (v2 호환)
    im["modalities"] = {k: f"{capname}/{v}" for k, v in mod.items()}
    store["images"].append(im)
    for a in anns_by_old.get(img["id"], []):
        store["_aid"][0] += 1
        na = dict(a); na["id"] = store["_aid"][0]; na["image_id"] = nid
        store["annotations"].append(na)

v3_tr, v3_te = new_store(), new_store()
for c in caps:
    cn = os.path.basename(c)
    d = json.load(open(os.path.join(c, "annotations", "instances.json")))
    if CATS is None:
        CATS = d["categories"]
    anns_by_old = collections.defaultdict(list)
    for a in d["annotations"]:
        anns_by_old[a["image_id"]].append(a)
    store = v3_te if cn in V2_TEST else v3_tr
    for im in d["images"]:
        add_image(store, im, anns_by_old, cn)

def finalize(store, name):
    json.dump({"images": store["images"], "annotations": store["annotations"],
               "categories": CATS}, open(os.path.join(OUT, name), "w"))
    cc = collections.Counter(a["category_id"] for a in store["annotations"])
    zero = len(store["images"]) - len({a["image_id"] for a in store["annotations"]})
    missing = [i for i in range(1, 11) if cc.get(i, 0) == 0]
    print(f"{name}: imgs={len(store['images'])} anns={len(store['annotations'])} "
          f"zerobox={zero} classes={10 - len(missing)}/10 missing={missing}")

print(f"BASE={BASE}\nTEST caps={sorted(V2_TEST)}")
finalize(v3_tr, "det_train_v3.json")
finalize(v3_te, "det_test_v3.json")
print("OUT:", OUT)
