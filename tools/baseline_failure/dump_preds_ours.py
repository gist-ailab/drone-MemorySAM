#!/usr/bin/env python3
"""
D1 — 우리 모델(ReliaDINO/LoRA_Sam) 예측 전수 덤프.

val.py 의 추론 경로(load_model → forward → softmax → _argmax_pred →
_unpad_resize_to_orig)를 **그대로 import 해서** 재사용한다. val.py 는 수정하지
않으므로 기존 평가 경로는 바이트 동일하게 유지된다(설계서 D1 의 "저장 옵션은 기본
off" 요구를 val.py 미변경으로 충족).

각 이미지에 대해 원본 해상도로 저장한다. image_id = **데이터셋 루트 기준 상대 경로**
(`img/<condition>/<split>/<scene>/<stem>_rgb_front`, 확장자 제외)이며, 그대로 중첩
디렉터리로 풀려 저장되므로 DELIVER 의 반복 basename 이 서로 덮어쓰지 않는다:
- `<out>/<split>/pred/<image_id>.png` : trainID 0~24 (ignore 없음, uint8)
- `<out>/<split>/conf/<image_id>.png` : softmax 최댓값 × 255 (uint8)
- `<out>/<split>/gt/<image_id>.png`   : native GT trainID 0~24 + 255 (교차 채점용)
- `<out>/<split>/pred1024/<image_id>.png` : 원본(1042) 복원 **전** 1024² argmax 라벨.
  `--save_1024` 사용 시만 저장한다(정본 채점 격자 = 1024 방침, A9-1). 라벨이
  1024² 가 아니면 임의 보정 없이 에러로 멈춘다.
그리고 같은 실행에서 mIoU 를 누적해 `<out>/<split>/summary.json` 을 남긴다. 이
mIoU 가 등록 수치(56.99 / E1 시드1)와 ±0.05 안에서 일치해야 덤프가 유효하다.
저장이 끝나면 장수를 세어 기대값(val 2005 / test 1897)과 다르면 assert 로 멈춘다
(--limit 디버그 실행은 예외).

예:
  python tools/baseline_failure/dump_preds_ours.py \
    --cfg configs/eval/<ours>.yaml \
    --model_path <ckpt.pth> --split test \
    --out /drone_nas/.../analysis_logs/baseline_failure_20260917/ours
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# 리포 루트를 path 에 추가해 val.py / semseg 를 import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402


def _unpad_resize_conf(conf, orig_h, orig_w, model_size):
    """confidence(float) 맵을 _unpad_resize_to_orig 와 같은 crop 규약으로 원본
    해상도에 맞춘다. 라벨용 _unpad_resize_to_orig 는 마지막에 long 캐스팅을 하므로
    float confidence 에는 쓸 수 없어, 같은 crop 수식을 bilinear 로 재현한다."""
    H, W, t = orig_h, orig_w, model_size
    if W >= H:
        scale = t / W
        nH = round(H * scale)
        pad_top = (t - nH) // 2
        content = conf[pad_top:pad_top + nH, :t]
    else:
        scale = t / H
        nW = round(W * scale)
        pad_left = (t - nW) // 2
        content = conf[:t, pad_left:pad_left + nW]
    content = content.unsqueeze(0).unsqueeze(0).float()
    out = F.interpolate(content, size=(H, W), mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True, help="평가 config (val.py 와 동일)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out", required=True, help="덤프 루트(<out>/<split>/... 생성)")
    ap.add_argument("--batch", type=int, default=1,
                    help="legal 프로토콜은 BS1 — 기본 1 유지 권장")
    ap.add_argument("--expected-miou", type=float, default=None,
                    help="등록 수치. 주면 ±--tol 밖일 때 exit≠0")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0,
                    help="디버그용 앞 N 장만(0=전체)")
    ap.add_argument("--save_1024", action="store_true",
                    help="원본(1042) 복원 전의 1024x1024 argmax 라벨을 "
                         "<out>/<split>/pred1024/ 에 추가 저장한다(정본 채점 격자 = "
                         "1024 방침). 기존 pred/conf/gt 저장은 그대로. 라벨이 1024²"
                         "가 아니면 임의 보정 없이 에러로 멈춘다")
    args = ap.parse_args()

    # val.py 는 무거운 top-level import 를 가지므로 여기서 지연 import.
    import val as valmod

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    valmod.setup_cudnn()

    dataset_cfg = cfg["DATASET"]
    eval_cfg = cfg["EVAL"]
    test_cfg = cfg.get("TEST", {})
    mode = args.split
    image_size = (eval_cfg["IMAGE_SIZE"] if mode == "val"
                  else test_cfg.get("IMAGE_SIZE", eval_cfg["IMAGE_SIZE"]))
    model_size = image_size if isinstance(image_size, int) else int(image_size[0])
    transform = valmod.get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    dataset, has_gt = valmod.create_dataset(dataset_cfg, mode, transform, mode,
                                            macvi=False, eval_day=False)
    if not has_gt:
        print("[dump_ours] ⚠️ has_gt=False — GT 없는 split 은 mIoU 검증 불가", flush=True)

    loader = DataLoader(dataset, batch_size=args.batch, num_workers=4,
                        pin_memory=False, collate_fn=valmod._collate_fn)

    model = valmod.load_model(cfg, Path(args.model_path), device)
    model.eval()

    n_classes = dataset.n_classes
    ignore = dataset.ignore_label
    out_root = Path(args.out) / mode
    pred_dir = out_root / "pred"
    conf_dir = out_root / "conf"
    gt_dir = out_root / "gt"
    pred1024_dir = out_root / "pred1024"
    for d in (pred_dir, conf_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)
    if args.save_1024:
        pred1024_dir.mkdir(parents=True, exist_ok=True)

    hist = np.zeros((n_classes, n_classes), dtype=np.int64)
    n_saved = 0
    seen_ids = set()

    with torch.no_grad():
        for images, labels, metas in tqdm(loader, desc=f"dump ours [{mode}]"):
            images = [x.to(device) for x in images]
            output, _ = model(images, multimask_output=True)
            probs = output.softmax(dim=1)                       # [B, C, h, w]
            pred_labels = valmod._argmax_pred(probs, n_classes, None)
            conf_map = probs[:, :n_classes].max(dim=1).values   # [B, h, w]

            for b in range(pred_labels.shape[0]):
                meta = metas[b]
                orig_h, orig_w = meta["orig_h"], meta["orig_w"]
                rgb_path = meta.get("paths", {}).get("img")
                if rgb_path is None:
                    raise RuntimeError(
                        "meta['paths']['img'] 가 없다 — 상대 경로 image_id 를 만들 수 "
                        "없다(dataset 이 return_meta=True 로 img 경로를 넘겨야 함).")
                image_id = common.image_id_from_rel(rgb_path)
                if image_id in seen_ids:
                    raise RuntimeError(
                        f"image_id 충돌: {image_id} — 경로 규약 확인 필요")
                seen_ids.add(image_id)

                if args.save_1024:
                    # A9-1: _unpad_resize_to_orig(1042 복원) 직전의 1024² argmax 라벨을
                    # 그대로 저장(정본 채점 격자 = 1024). floor 정렬 복원을 거치지 않는다.
                    lab1024 = pred_labels[b]
                    if tuple(lab1024.shape) != (1024, 1024):
                        raise RuntimeError(
                            f"--save_1024 인데 복원 전 라벨 해상도가 "
                            f"{lab1024.shape[1]}x{lab1024.shape[0]} (기대 1024x1024, "
                            f"image_id={image_id}) — 정본 격자와 어긋난다. 임의 보정 없이 "
                            f"멈춘다(EVAL.IMAGE_SIZE 확인 필요).")
                    common.save_label_png(
                        pred1024_dir / f"{image_id}.png",
                        lab1024.cpu().numpy().astype(np.uint8))

                pred_resized = valmod._unpad_resize_to_orig(
                    pred_labels[b], orig_h, orig_w, model_size=model_size)
                pred_np = pred_resized.cpu().numpy().astype(np.uint8)
                common.save_label_png(pred_dir / f"{image_id}.png", pred_np)

                conf_resized = _unpad_resize_conf(
                    conf_map[b], orig_h, orig_w, model_size)
                conf_np = (conf_resized.clamp(0, 1).cpu().numpy() * 255.0
                           ).round().astype(np.uint8)
                common.save_label_png(conf_dir / f"{image_id}.png", conf_np)

                orig_label = meta.get("orig_label")
                if orig_label is not None:
                    gt_np = orig_label.cpu().numpy().astype(np.uint8)
                    common.save_label_png(gt_dir / f"{image_id}.png", gt_np)
                    hist += common.confusion_matrix(pred_np, gt_np, n_classes, ignore)

                n_saved += 1
            if args.limit and n_saved >= args.limit:
                break

    # 장수 검증 — 상대 경로 보존이 성공했는지 확인(--limit 디버그는 예외).
    if not args.limit:
        common.assert_expected_count(n_saved, mode,
                                     extra=f"(dump_preds_ours, out={out_root})")

    ious, miou = common.global_miou_from_cm(hist)
    summary = {
        "split": mode,
        "model_path": str(args.model_path),
        "cfg": str(args.cfg),
        "num_images": n_saved,
        "expected_count": common.EXPECTED_COUNTS.get(mode),
        "saved_1024": bool(args.save_1024),
        "pred1024_resolution": 1024 if args.save_1024 else None,
        "mIoU": miou,
        "per_class_iou": {common.CLASSES[i]: ious[i] for i in range(n_classes)},
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[dump_ours] {mode}: {n_saved} imgs  mIoU={miou:.2f}")
    print(f"[dump_ours] -> {out_root}")

    if args.expected_miou is not None:
        delta = abs(miou - args.expected_miou)
        status = "OK" if delta <= args.tol else "MISMATCH"
        print(f"[dump_ours] 등록수치 {args.expected_miou:.2f} 대비 Δ={delta:.3f} → {status}")
        if delta > args.tol:
            print("[dump_ours] ⚠️ 재현 실패 — 덤프 무효, 전처리/ckpt 확인 후 재실행")
            sys.exit(2)


if __name__ == "__main__":
    main()
