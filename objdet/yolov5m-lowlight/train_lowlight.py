"""Low-light training wrapper for classic YOLOv5 — architecture-frozen.

This does NOT edit YOLOv5 source. It imports the stock ``ultralytics/yolov5``
``train.py`` and, gated by env flags, monkeypatches exactly two train-time
things before training starts:

    YOLO_NIGHTAUG=1   -> append dark-tail low-light augmentation (night_aug.py)
    YOLO_EIOU=1       -> swap the box-regression criterion to EIoU (eiou.py)
    YOLO_NIGHT_STR=calibrated|mild|strong   (augmentation strength)

Both live only in the augmentation pipeline and the loss function. The network
graph, the exported ONNX/TFLite, and therefore i.MX-NPU portability are
untouched. Remove the env flags and you get bit-identical stock YOLOv5.

Usage (all stock YOLOv5 train.py args pass straight through):

    YOLOV5_DIR=/path/to/yolov5 YOLO_NIGHTAUG=1 YOLO_EIOU=1 \
    python train_lowlight.py --data poongsan.yaml --weights yolov5m.pt \
        --hyp hyp.lowlight.yaml --img 640 --batch 16 --epochs 100 \
        --label-smoothing 0.1 --device 0 --name b3
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so night_aug / eiou import


def _find_yolov5():
    for cand in (os.environ.get("YOLOV5_DIR"),
                 os.path.join(HERE, "yolov5"),
                 os.path.join(HERE, "ultralytics_yolov5")):
        if cand and os.path.isfile(os.path.join(cand, "train.py")):
            return os.path.abspath(cand)
    sys.exit("[train_lowlight] YOLOv5 repo not found. Clone ultralytics/yolov5 "
             "and set YOLOV5_DIR=<path> (must contain train.py).")


def _patch_night(strength):
    """Append dark-tail augmentation into YOLOv5's Albumentations pipeline."""
    import albumentations as A
    import utils.augmentations as aug
    import utils.dataloaders as dl
    from night_aug import build_night_transforms

    Orig = aug.Albumentations

    class NightAlbumentations(Orig):
        def __init__(self, size=640):
            super().__init__(size)
            if self.transform is not None:
                base = list(self.transform.transforms)
                extra = build_night_transforms(strength)
                self.transform = A.Compose(
                    base + extra,
                    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                print(f"[train_lowlight] night-aug ON (strength={strength}): "
                      f"+{len(extra)} dark-tail transforms appended "
                      f"({len(base)} stock -> {len(base)+len(extra)} total)")

    aug.Albumentations = NightAlbumentations
    dl.Albumentations = NightAlbumentations  # the name dataloaders actually instantiates


def _patch_eiou():
    """Replace the box-regression IoU used by ComputeLoss with EIoU."""
    import utils.loss as loss
    from eiou import bbox_iou_eiou
    loss.bbox_iou = bbox_iou_eiou
    print("[train_lowlight] EIoU box loss ON (CIoU -> EIoU, arch unchanged)")


def main():
    y5 = _find_yolov5()
    sys.path.insert(0, y5)
    os.chdir(y5)  # yolov5 resolves data/ hyp/ relative to its own root

    import train  # noqa: E402  (stock yolov5 train.py)

    if os.environ.get("YOLO_NIGHTAUG") == "1":
        _patch_night(os.environ.get("YOLO_NIGHT_STR", "calibrated"))
    if os.environ.get("YOLO_EIOU") == "1":
        _patch_eiou()

    opt = train.parse_opt()          # parses the pass-through argv
    train.main(opt)


if __name__ == "__main__":
    main()
