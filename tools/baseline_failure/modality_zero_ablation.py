#!/usr/bin/env python3
"""
D4 — 우리 모델 모달 제거(zero-out) ablation. 각 모달 입력을 0 으로 두고 mIoU 를
재는 추론을 모달 수만큼 반복해, 어느 모달에 실제로 의존하는지 본다.

val.py 의 load_model/create_dataset 을 재사용한다. 기준선(detectron2)용 zero-out 은
tools/baseline_failure/d2_zero_modality.patch (BF_ZERO_MODAL) 를 쓴다 — README 참조.
tools/module_ablation.py 는 모듈 토글(rbma_off 등)만 다루고 모달 zero-out 기능은 없어
여기서 별도 구현한다.

예:
  python tools/baseline_failure/modality_zero_ablation.py \
    --cfg configs/eval/<ours>.yaml --model_path <ckpt> \
    --split test --out .../mining/modal_zero_ours.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402


def _run_once(valmod, model, loader, device, n_classes, ignore, model_size, zero_idx,
              per_image=None, ratio=1.0, generator=None):
    """전역 혼동행렬 기준 mIoU 를 돌려준다.

    per_image 에 리스트를 주면 이미지별 mIoU 도 같이 담는다(조건별 Δ 산출용). 이미지별
    값은 그 이미지 한 장의 혼동행렬로 계산한 것이라 전역 mIoU 와 집계가 다르다 —
    두 수치를 섞어 인용하면 안 된다.

    ratio 는 모달 열화 비율(RMM r)이다. 1.0 이면 완전 zero-out, 1.0 미만이면 그 비율만큼
    픽셀×채널 독립 드롭으로 열화한다. generator 는 드롭 마스크 재현용 난수 생성기.
    """
    hist = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for images, labels, metas in tqdm(loader, desc=f"zero={zero_idx}", leave=False):
            images = [x.to(device) for x in images]
            if zero_idx is not None:
                if ratio >= 1.0:
                    images[zero_idx] = torch.zeros_like(images[zero_idx])
                else:
                    # 부분 열화: 정규화된 텐서에서 원소별로 rand<ratio 인 자리만 0 으로
                    # 만든다(missing_modality_eval.rmm_mask 와 같은 규약, 픽셀×채널 독립).
                    # CPU 생성기로 마스크를 뽑은 뒤 device/dtype 로 옮긴다.
                    keep = (torch.rand(images[zero_idx].shape, generator=generator)
                            >= ratio).to(dtype=images[zero_idx].dtype,
                                         device=images[zero_idx].device)
                    images[zero_idx] = images[zero_idx] * keep
            output, _ = model(images, multimask_output=True)
            probs = output.softmax(dim=1)
            pred_labels = valmod._argmax_pred(probs, n_classes, None)
            for b in range(pred_labels.shape[0]):
                meta = metas[b]
                orig_label = meta.get("orig_label")
                if orig_label is None:
                    continue
                pr = valmod._unpad_resize_to_orig(
                    pred_labels[b], meta["orig_h"], meta["orig_w"], model_size=model_size)
                cm = common.confusion_matrix(
                    pr.cpu().numpy(), orig_label.cpu().numpy(), n_classes, ignore)
                hist += cm
                if per_image is not None:
                    _, miou_i = common.global_miou_from_cm(cm)
                    per_image.append({
                        "stem": meta.get("stem", ""),
                        "img_path": (meta.get("paths") or {}).get("img", ""),
                        "miou": round(float(miou_i), 4),
                    })
    _, miou = common.global_miou_from_cm(hist)
    return miou


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per_image_csv", default=None,
                    help="이미지별 mIoU 를 이 CSV 에 남긴다(조건별 Δ 산출용). 이미지별 값은 "
                         "그 한 장의 혼동행렬로 계산하므로 전역 mIoU 와 집계가 다르다.")
    ap.add_argument("--only", nargs="*", default=[],
                    help="돌릴 조건만 고른다(예: --only base zero_depth). 생략하면 전부. "
                         "조건 이름은 base 와 zero_<모달> 이다.")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="모달 열화 비율(RMM r). 1.0(기본)이면 지금처럼 완전 제거(zero-out), "
                         "1.0 미만이면 그 비율만큼 픽셀·채널 독립으로 0 으로 드롭해 열화한다.")
    ap.add_argument("--ratio_seed", type=int, default=0,
                    help="부분 열화(ratio<1.0) 드롭 마스크의 시드. 모든 조건이 같은 난수 "
                         "스트림을 공유해 모달 간 비교가 공정해진다.")
    args = ap.parse_args()

    if not (0 < args.ratio <= 1.0):
        raise SystemExit(f"--ratio 는 0 초과 1 이하의 값이어야 합니다(입력: {args.ratio}). "
                         "1.0 = 완전 제거, 1.0 미만 = 그 비율만큼 부분 열화입니다.")

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
    dataset, _ = valmod.create_dataset(dataset_cfg, mode, transform, mode)
    if args.limit:
        dataset.files = dataset.files[:args.limit]
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=4,
                        pin_memory=False, collate_fn=valmod._collate_fn)
    model = valmod.load_model(cfg, Path(args.model_path), device)
    model.eval()

    modals = list(dataset_cfg["MODALS"])
    n_classes = dataset.n_classes
    ignore = dataset.ignore_label

    # 조건 목록 = 기준 + 모달별 제거. --only 로 일부만 돌리면 조건 하나를 GPU 한 장에
    # 맡겨 동시에 처리할 수 있다(전량을 한 프로세스로 돌리면 모달 수만큼 직렬로 걸린다).
    conditions = [("base", None)] + [(f"zero_{m}", i) for i, m in enumerate(modals)]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {name for name, _ in conditions}
        if unknown:
            raise SystemExit(f"--only 에 없는 조건: {sorted(unknown)} "
                             f"(가능한 값: {[n for n, _ in conditions]})")
        conditions = [(name, idx) for name, idx in conditions if name in wanted]

    # 부분 열화 마스크용 생성기는 한 번만 만들어 모든 조건 호출에 이어 쓴다. 기준선 쪽
    # 개입(d2_zero_modality.patch 의 BF_ZERO_SEED)도 같은 방식이라 규약이 맞는다.
    # 조건을 한 번에 여러 개 돌리면 뒤 조건은 앞 조건이 쓰고 남은 난수를 이어받는다 —
    # 모달 사이의 마스크를 똑같이 맞추고 싶으면 --only 로 조건 하나씩 따로 돌려라
    # (프로세스마다 같은 시드에서 시작하므로 같은 마스크가 나온다).
    ratio_generator = torch.Generator().manual_seed(args.ratio_seed)

    results = {}
    per_image_rows = []
    for name, zero_idx in conditions:
        rows = [] if args.per_image_csv else None
        results[name] = _run_once(valmod, model, loader, device, n_classes, ignore,
                                  model_size, zero_idx, per_image=rows,
                                  ratio=args.ratio, generator=ratio_generator)
        for r in (rows or []):
            r["condition"] = name
            per_image_rows.append(r)
    if args.per_image_csv and per_image_rows:
        import csv
        out_csv = Path(args.per_image_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["condition", "stem", "img_path", "miou"])
            w.writeheader()
            w.writerows(per_image_rows)
        print(f"[modality_zero] 이미지별 {len(per_image_rows)} 행 -> {out_csv}")

    report = {"cfg": args.cfg, "model_path": args.model_path, "split": mode,
              "modals": modals, "conditions": [n for n, _ in conditions],
              "ratio": args.ratio, "ratio_seed": args.ratio_seed, "mIoU": results}
    if "base" in results:
        report["drop_vs_base"] = {k: round(results["base"] - v, 3)
                                  for k, v in results.items() if k != "base"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["mIoU"], indent=2, ensure_ascii=False))
    print(f"[modality_zero] -> {args.out}")


if __name__ == "__main__":
    main()
