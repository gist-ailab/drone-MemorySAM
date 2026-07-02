#!/usr/bin/env python3
"""P29-Det poongsan 깨끗한 라벨셋(8 caps, empty=0%) → train/test split 빌더.

각 capture 의 annotations/instances.json(COCO, per-image `modalities` dict 포함)을
병합하면서 image/annotation id 전역 재부여 + 모든 경로에 `capture_XXX/` prefix
(단일 ROOT 에서 resolve 되도록). 두 가지 split 산출:

  v2 = 캡처 단위 holdout (test = V2_TEST 캡처들).        temporal leakage 0.
  v3 = 캡처 내 타임스탬프 80/20 + 경계 GAP 프레임 drop.   near-duplicate leakage 완화.

사용: python scripts/build_det_splits.py [BASE] [OUT]
산출 후 OUT/*.json 을 학습 서버 ROOT/_det_splits/ 로 rsync.
"""
import json, glob, os, sys, collections

BASE = sys.argv[1] if len(sys.argv) > 1 else "/ailab_mat2/Projects/Drone/DATA/260618_poongsan"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "./_det_splits_v2v3"

V2_TEST   = {"capture_20260618_115206", "capture_20260618_114808"}  # 둘 다 10클래스 포함
GAP       = 15     # v3 경계 drop 프레임 수
TEST_FRAC = 0.20   # v3 캡처별 뒤쪽 test 비율

os.makedirs(OUT, exist_ok=True)
caps = sorted(glob.glob(os.path.join(BASE, "capture_*")))
CATS = None
gid = [0]

def new_store():
    return {"images": [], "annotations": [], "_aid": [0]}

def add_image(store, img, anns_by_old, capname):
    gid[0] += 1; nid = gid[0]
    im = dict(img); im["id"] = nid
    im["file_name"] = f"{capname}/{img['file_name']}"
    if "modalities" in img:
        im["modalities"] = {k: f"{capname}/{v}" for k, v in img["modalities"].items()}
    store["images"].append(im)
    for a in anns_by_old.get(img["id"], []):
        store["_aid"][0] += 1
        na = dict(a); na["id"] = store["_aid"][0]; na["image_id"] = nid
        store["annotations"].append(na)

v2_tr, v2_te, v3_tr, v3_te = new_store(), new_store(), new_store(), new_store()

for c in caps:
    capname = os.path.basename(c)
    d = json.load(open(os.path.join(c, "annotations", "instances.json")))
    if CATS is None: CATS = d["categories"]
    anns_by_old = collections.defaultdict(list)
    for a in d["annotations"]: anns_by_old[a["image_id"]].append(a)
    imgs = d["images"]

    # v2: whole-capture holdout
    v2_store = v2_te if capname in V2_TEST else v2_tr
    for im in imgs: add_image(v2_store, im, anns_by_old, capname)

    # v3: temporal 80/20 within capture, with boundary gap
    def ts(im):
        b = os.path.splitext(os.path.basename(im["file_name"]))[0]
        try: return int(b)
        except Exception: return b
    s = sorted(imgs, key=ts); n = len(s); cut = int(n * (1 - TEST_FRAC))
    for im in s[:cut]:        add_image(v3_tr, im, anns_by_old, capname)
    for im in s[cut + GAP:]:  add_image(v3_te, im, anns_by_old, capname)

def finalize(store, name):
    json.dump({"images": store["images"], "annotations": store["annotations"],
               "categories": CATS}, open(os.path.join(OUT, name), "w"))
    cc = collections.Counter(a["category_id"] for a in store["annotations"])
    missing = [i for i in range(1, 11) if cc.get(i, 0) == 0]
    print(f"{name}: imgs={len(store['images'])} anns={len(store['annotations'])} "
          f"classes={10 - len(missing)}/10 missing={missing}")

finalize(v2_tr, "det_train_v2.json"); finalize(v2_te, "det_test_v2.json")
finalize(v3_tr, "det_train_v3.json"); finalize(v3_te, "det_test_v3.json")
print("OUT:", OUT)
