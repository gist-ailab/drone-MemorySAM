"""GISTOLO KD YOLOv5m -> ONNX (i.MX 포팅용).

yolov5 의 export.py 는 torch 2.x 의 dynamo exporter 를 타서 opset 12 요청이
무시되고 18 로 나온다(Resize 버전 변환 어댑터 부재). i.MX eIQ 는 보통 opset
12~13 을 요구하므로, 레거시 TorchScript exporter(dynamo=False)로 직접 내보내
opset 을 확실히 고정한다.
"""
import argparse, os, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True)
ap.add_argument('--yolo-dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--img', type=int, default=640)
ap.add_argument('--opset', type=int, default=12)
a = ap.parse_args()

sys.path.insert(0, a.yolo_dir)
from models.experimental import attempt_load
from models.yolo import Detect

model = attempt_load(a.weights, device=torch.device('cpu'), inplace=True, fuse=True)
model.eval()
for m in model.modules():
    if isinstance(m, Detect):
        m.inplace = True
        m.dynamic = False
        m.export = True          # NMS 없는 raw 출력 (배포측에서 후처리)

im = torch.zeros(1, 3, a.img, a.img)
for _ in range(2):               # shape 확정용 워밍업
    y = model(im)

torch.onnx.export(
    model, im, a.out,
    input_names=['images'], output_names=['output0'],
    opset_version=a.opset,
    do_constant_folding=True,
    dynamo=False,                # 레거시 경로 — opset 을 그대로 지킨다
)
print(f'[export] saved {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)')
