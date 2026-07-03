#!/usr/bin/env python3
"""E1.1 YOLO11-m RGB 기준점: 전체 test셋 GT|Pred 나란히 시각화.

각 test 이미지에 대해 [GT(초록) | Pred(빨강, conf 표기)] 를 가로로 붙인
한 장의 jpg 를 저장. GT 는 YOLO 라벨 txt, Pred 는 best.pt 추론 결과.

사용: python viz_test_set.py <WEIGHTS> <DATASET_ROOT> <OUT_DIR> [DEVICE] [CONF]
"""
import os, sys, glob
import cv2
from ultralytics import YOLO

WEIGHTS, ROOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "0"
CONF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25

model = YOLO(WEIGHTS)
names = model.names
os.makedirs(OUT, exist_ok=True)

GT_COLOR = (0, 200, 0)
PR_COLOR = (0, 0, 255)

def draw_box(img, x1, y1, x2, y2, text, color):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    ty = int(y1) - 4 if y1 > th + 6 else int(y2) + th + 4
    cv2.rectangle(img, (int(x1), ty - th - 3), (int(x1) + tw + 2, ty + 2), color, -1)
    cv2.putText(img, text, (int(x1) + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

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
                     (cx + bw / 2) * W, (cy + bh / 2) * H, names[c], GT_COLOR)

    pr = img.copy()
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        draw_box(pr, x1, y1, x2, y2,
                 f"{names[int(b.cls)]} {float(b.conf):.2f}", PR_COLOR)

    cv2.putText(gt, "GT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(pr, "PRED", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, PR_COLOR, 2, cv2.LINE_AA)
    div = 255 * (img[:, :4] * 0 + 1)
    out = cv2.hconcat([gt, div.astype(img.dtype), pr])
    cv2.imwrite(os.path.join(OUT, os.path.splitext(base)[0] + ".jpg"), out,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    if (i + 1) % 200 == 0:
        print(f"{i + 1}/{len(imgs)}", flush=True)

print("DONE", flush=True)
