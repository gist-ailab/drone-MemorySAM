#!/usr/bin/env python3
"""P29-Det 멀티모달 추론 시각화: 프레임마다 2x3 패널.

  [ RGB          | LiDAR(depth)  | Thermal        ]   (입력 3 모달리티)
  [ Ground Truth | Prediction    | GT vs Pred     ]   (검출 결과)

GT = COCO GT json, Pred = val_det.py 가 뽑은 predictions.json (COCO), 이미지는
데이터셋 BASE 에서 로드. 클래스별 색상(10색), 색배경 라벨 배지, 패널 배지.

사용: python viz_multimodal_det.py <GT_JSON> <PRED_JSON> <DATA_BASE> <OUT_DIR> [CONF]
"""
import json, os, sys, collections
import cv2
import numpy as np

GT_JSON, PRED_JSON, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
CONF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.30
os.makedirs(OUT, exist_ok=True)

gt = json.load(open(GT_JSON))
names = {c["id"]: c["name"] for c in gt["categories"]}
images = {im["id"]: im for im in gt["images"]}
gt_by_img = collections.defaultdict(list)
for a in gt["annotations"]:
    gt_by_img[a["image_id"]].append(a)
pred_by_img = collections.defaultdict(list)
for p in json.load(open(PRED_JSON)):
    pred_by_img[p["image_id"]].append(p)

# 클래스별 팔레트 (BGR) — cat_id 1..10 → index 0..9
PALETTE = [
    (60, 200, 60), (60, 60, 235), (0, 160, 255), (235, 180, 50), (200, 100, 30),
    (0, 230, 230), (230, 80, 220), (140, 230, 140), (180, 60, 255), (255, 220, 160),
]
def color(cat_id):
    return PALETTE[(cat_id - 1) % len(PALETTE)]
def _txtcol(bgr):
    b, g, r = bgr
    return (20, 20, 20) if (0.114*b + 0.587*g + 0.299*r) > 150 else (255, 255, 255)

def draw_box(img, x1, y1, x2, y2, text, col, thick=2, label=True):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
    if not label:
        return
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    ty = y1 - th - 8 if y1 - th - 8 >= 0 else y1 + 2
    cv2.rectangle(img, (x1, ty), (x1 + tw + 8, ty + th + 7), col, -1)
    cv2.putText(img, text, (x1 + 4, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                _txtcol(col), 1, cv2.LINE_AA)

def badge(img, text, col=(25, 25, 25)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (tw + 24, th + 16), col, -1)
    cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
    cv2.putText(img, text, (12, th + 8), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

def load(rel, W, H):
    p = os.path.join(BASE, rel)
    im = cv2.imread(p)
    if im is None:
        return np.zeros((H, W, 3), np.uint8)
    if im.shape[:2] != (H, W):
        im = cv2.resize(im, (W, H))
    return im

miss = 0
imgs = sorted(images.values(), key=lambda im: im["file_name"])
for n, im in enumerate(imgs):
    mods = im["modalities"]
    rgb_p = os.path.join(BASE, mods["rgb"])
    rgb = cv2.imread(rgb_p)
    if rgb is None:
        miss += 1
        continue
    H, W = rgb.shape[:2]

    # --- 입력 모달리티 ---
    rgb_panel = rgb.copy()
    lidar_raw = load(mods["depth_map_lidar"], W, H)     # JET 점(검은 배경)
    dim = (rgb * 0.35).astype(np.uint8)                 # lidar 점을 dim rgb 위에 얹어 맥락 부여
    m = np.any(lidar_raw > 10, axis=2)
    lidar_panel = dim.copy(); lidar_panel[m] = lidar_raw[m]
    thermal_panel = load(mods["thermal_aligned"], W, H)

    # --- 검출 결과 ---
    gt_panel = rgb.copy()
    for a in gt_by_img.get(im["id"], []):
        x, y, w, h = a["bbox"]
        draw_box(gt_panel, x, y, x + w, y + h, names[a["category_id"]], color(a["category_id"]))

    pred_panel = rgb.copy()
    for p in pred_by_img.get(im["id"], []):
        if p["score"] < CONF:
            continue
        x, y, w, h = p["bbox"]
        draw_box(pred_panel, x, y, x + w, y + h,
                 f'{names[p["category_id"]]} {p["score"]:.2f}', color(p["category_id"]))

    # overlay: GT(흰 얇은 선) vs Pred(클래스색 굵은 선)
    ov = rgb.copy()
    for a in gt_by_img.get(im["id"], []):
        x, y, w, h = a["bbox"]
        draw_box(ov, x, y, x + w, y + h, "", (245, 245, 245), thick=1, label=False)
    for p in pred_by_img.get(im["id"], []):
        if p["score"] < CONF:
            continue
        x, y, w, h = p["bbox"]
        draw_box(ov, x, y, x + w, y + h, "", color(p["category_id"]), thick=2, label=False)

    badge(rgb_panel, "RGB")
    badge(lidar_panel, "LiDAR (depth)")
    badge(thermal_panel, "Thermal")
    badge(gt_panel, "Ground Truth")
    badge(pred_panel, "Prediction  P29-Det (RGB+LiDAR+Thermal)")
    badge(ov, "GT (white)  vs  Pred (color)")

    vdiv = np.full((H, 4, 3), 255, np.uint8)
    top = cv2.hconcat([rgb_panel, vdiv, lidar_panel, vdiv, thermal_panel])
    bot = cv2.hconcat([gt_panel, vdiv, pred_panel, vdiv, ov])
    hdiv = np.full((4, top.shape[1], 3), 255, np.uint8)
    grid = cv2.vconcat([top, hdiv, bot])
    cv2.imwrite(os.path.join(OUT, os.path.splitext(os.path.basename(mods["rgb"]))[0] + ".jpg"),
                grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if (n + 1) % 200 == 0:
        print(f"{n + 1}/{len(imgs)}", flush=True)

print(f"DONE ({len(imgs)-miss} written, {miss} missing rgb)", flush=True)
