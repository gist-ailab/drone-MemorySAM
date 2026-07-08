#!/usr/bin/env python3
"""YOLO 기준점: 전체 test셋 GT|Pred 나란히 시각화 (클래스별 색상).

각 test 이미지에 대해 [Ground Truth | Prediction] 을 가로로 붙인 jpg 저장.
클래스마다 고유 색 (GT/Pred 동일 클래스 = 동일 색), 라벨은 색 배경 배지,
패널 상단에 반투명 배지 표기.

사용: python viz_test_set.py <WEIGHTS> <DATASET_ROOT> <OUT_DIR> [DEVICE] [CONF]
"""
import os, sys, glob
import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS, ROOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "0"
CONF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25

model = YOLO(WEIGHTS)
names = model.names
os.makedirs(OUT, exist_ok=True)

# 클래스별 팔레트 (BGR, tab10 기반 고채도) — 인덱스 = 클래스 id
PALETTE = [
    (60, 200, 60),    # 0 Allies          green
    (60, 60, 235),    # 1 Enemies         red
    (0, 160, 255),    # 2 Casualties      orange
    (235, 180, 50),   # 3 Windows         sky blue
    (200, 100, 30),   # 4 Doors           blue
    (0, 230, 230),    # 5 Obstacles       yellow
    (230, 80, 220),   # 6 Lighting        magenta
    (140, 230, 140),  # 7 Emergency Exits mint
    (180, 60, 255),   # 8 Fire Ext.       pink-violet
    (255, 220, 160),  # 9 Landing Markers pale cyan
]

def _text_color(bgr):
    b, g, r = bgr
    return (20, 20, 20) if (0.114 * b + 0.587 * g + 0.299 * r) > 150 else (255, 255, 255)

def draw_box(img, x1, y1, x2, y2, text, color):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    ty0 = y1 - th - 8
    if ty0 < 0:                      # 상단 벗어나면 박스 안쪽 위에
        ty0 = y1 + 2
    cv2.rectangle(img, (x1, ty0), (x1 + tw + 8, ty0 + th + 7), color, -1)
    cv2.putText(img, text, (x1 + 4, ty0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, _text_color(color), 1, cv2.LINE_AA)

def panel_badge(img, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)
    pad = 8
    over = img.copy()
    cv2.rectangle(over, (0, 0), (tw + pad * 2 + 8, th + pad * 2), (25, 25, 25), -1)
    cv2.addWeighted(over, 0.65, img, 0.35, 0, img)
    cv2.putText(img, text, (pad + 4, pad + th - 1), cv2.FONT_HERSHEY_DUPLEX,
                0.62, (255, 255, 255), 1, cv2.LINE_AA)

imgs = sorted(glob.glob(os.path.join(ROOT, "images/test/*")))
print(f"{len(imgs)} test images, conf={CONF}", flush=True)

for i, r in enumerate(model.predict(source=os.path.join(ROOT, "images/test"),
                                    stream=True, batch=1, conf=CONF,
                                    device=DEVICE, verbose=False)):
    p = r.path
    base = os.path.basename(p)
    img = cv2.imread(p)
    H, W = img.shape[:2]

    gt = img.copy()
    lp = os.path.join(ROOT, "labels/test", os.path.splitext(base)[0] + ".txt")
    if os.path.exists(lp):
        for line in open(lp):
            v = line.split()
            if len(v) != 5:
                continue
            c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:])
            draw_box(gt, (cx - bw / 2) * W, (cy - bh / 2) * H,
                     (cx + bw / 2) * W, (cy + bh / 2) * H,
                     names[c], PALETTE[c % len(PALETTE)])

    pr = img.copy()
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        c = int(b.cls)
        draw_box(pr, x1, y1, x2, y2,
                 f"{names[c]} {float(b.conf):.2f}", PALETTE[c % len(PALETTE)])

    panel_badge(gt, "Ground Truth")
    panel_badge(pr, f"Prediction (conf>={CONF:g})")
    div = np.full((H, 4, 3), 255, dtype=img.dtype)
    out = cv2.hconcat([gt, div, pr])
    cv2.imwrite(os.path.join(OUT, os.path.splitext(base)[0] + ".jpg"), out,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    if (i + 1) % 200 == 0:
        print(f"{i + 1}/{len(imgs)}", flush=True)

print("DONE", flush=True)
