#!/usr/bin/env python3
"""poongsan_v2 COCO det split → YOLO(ultralytics) 포맷 변환. E1.1 YOLO 기준점용.

우리 파이프라인(objdet/datasets/multimodal_det.py)과 정확히 동일한 프레임 집합을
쓰기 위해 REQUIRE_ALL_MODALITIES 필터를 재현한다: rgb/thermal_aligned/
depth_map_lidar 3개 modality 키가 모두 있고 파일이 디스크에 존재하는 프레임만
유지 (기대: train 5,862 / test 1,772). 이미지는 RGB만 복사 (RGB-only 비교군).

클래스 리맵도 동일: categories를 id 기준 정렬 후 0-base 인덱스.

사용: python convert_poongsan_to_yolo.py <SPLIT_DIR> <DATA_BASE> <OUT_DIR> [VER]
  SPLIT_DIR: det_train_<VER>.json / det_test_<VER>.json 위치
  DATA_BASE: capture_*/ 들이 있는 원본 루트 (/ailab_mat2/...)
  OUT_DIR  : YOLO dataset 루트 (images/, labels/, poongsan_<VER>_rgb.yaml 생성)
  VER      : split 버전 (기본 v2)
"""
import json, os, shutil, sys, collections

SPLIT_DIR, DATA_BASE, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
VER = sys.argv[4] if len(sys.argv) > 4 else "v2"
MODALITY_KEYS = ["rgb", "thermal_aligned", "depth_map_lidar"]  # 전부 있어야 유지
SPLITS = {"train": f"det_train_{VER}.json", "test": f"det_test_{VER}.json"}

names = None
stats = {}
for split, fname in SPLITS.items():
    coco = json.load(open(os.path.join(SPLIT_DIR, fname)))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    if names is None:
        names = [c["name"] for c in cats]

    ann_by_img = collections.defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    img_dir = os.path.join(OUT_DIR, "images", split)
    lbl_dir = os.path.join(OUT_DIR, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    kept, dropped, n_box = 0, 0, 0
    for im in coco["images"]:
        mods = im.get("modalities", {})
        paths = {k: os.path.join(DATA_BASE, mods[k]) for k in MODALITY_KEYS if k in mods}
        if len(paths) < len(MODALITY_KEYS) or not all(os.path.exists(p) for p in paths.values()):
            dropped += 1
            continue
        kept += 1
        W, H = im["width"], im["height"]
        # capture_X/rgb/123.png → capture_X_123.png (캡처 간 파일명 충돌 방지)
        rel = mods["rgb"]
        flat = rel.replace("/rgb/", "_").replace("/", "_")
        dst = os.path.join(img_dir, flat)
        if not os.path.exists(dst):
            shutil.copy2(paths["rgb"], dst)
        lines = []
        for a in ann_by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            bw, bh = w / W, h / H
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            bw, bh = min(bw, 1.0), min(bh, 1.0)
            lines.append(f"{cat_to_idx[a['category_id']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_box += 1
        with open(os.path.join(lbl_dir, os.path.splitext(flat)[0] + ".txt"), "w") as f:
            f.write("\n".join(lines))
    stats[split] = (kept, dropped, n_box)
    print(f"{split}: kept={kept} dropped={dropped} boxes={n_box}")

yaml_path = os.path.join(OUT_DIR, f"poongsan_{VER}_rgb.yaml")
with open(yaml_path, "w") as f:
    f.write(f"# poongsan RGB-only, {VER} split (YOLO baseline)\n")
    f.write(f"path: {os.path.abspath(OUT_DIR)}\n")
    f.write("train: images/train\nval: images/test\ntest: images/test\n")
    f.write("names:\n")
    for i, n in enumerate(names):
        f.write(f"  {i}: {n}\n")
print("wrote", yaml_path)
