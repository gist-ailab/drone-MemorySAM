#!/usr/bin/env python3
"""DELIVER 4모달 ReliaDINO의 사전 등록 축 ④ 예측 일치도와 ⑤ ECE를 측정한다.
④는 clean 대비 EMM 14개 설정 및 NM Mid의 유효 픽셀 예측 일치율이다.
⑤는 clean과 NM Mid의 최대 softmax 확률로 계산한 15-bin ECE다.
EMM zero-fill과 NM Mid 열화는 missing_modality_eval 함수를 그대로 재사용한다.
라벨 복원은 legal v2, confidence 복원은 기존 덤프 함수의 규약을 따른다.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SAM2_ROOT = _REPO_ROOT / "semseg" / "models" / "sam2"
if str(_SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM2_ROOT))

import val as valmod  # noqa: E402
from tools import legal_rescore_v2  # noqa: E402
from tools import missing_modality_eval as mme  # noqa: E402
from tools.baseline_failure import common  # noqa: E402
from tools.baseline_failure.dump_preds_ours import _unpad_resize_conf  # noqa: E402

N_BINS = 15
NM_ID = "nm_mid"


def accumulate_ece(counts, sum_conf, sum_correct, confidence, correct, valid):
    """유효 픽셀을 15개 등폭 bin에 누적한다. confidence/correct는 원본 해상도."""
    conf = confidence[valid].reshape(-1).to(dtype=torch.float64, device="cpu")
    corr = correct[valid].reshape(-1).to(dtype=torch.float64, device="cpu")
    if conf.numel() == 0:
        return
    if not bool(torch.isfinite(conf).all()) or bool(((conf < 0) | (conf > 1)).any()):
        raise ValueError("confidence must be finite and in [0, 1]")
    bin_idx = (conf * N_BINS).long().clamp(max=N_BINS - 1)
    counts += torch.bincount(bin_idx, minlength=N_BINS)
    sum_conf += torch.bincount(bin_idx, weights=conf, minlength=N_BINS)
    sum_correct += torch.bincount(bin_idx, weights=corr, minlength=N_BINS)


def ece_result(counts, sum_conf, sum_correct):
    """반환 ECE는 백분율, 각 bin의 conf/acc는 0~1 비율."""
    total = int(counts.sum().item())
    bins = []
    error = 0.0
    for count_t, conf_t, correct_t in zip(counts, sum_conf, sum_correct):
        count = int(count_t.item())
        conf = float(conf_t.item() / count) if count else None
        acc = float(correct_t.item() / count) if count else None
        bins.append({"count": count, "conf": conf, "acc": acc})
        if count:
            error += count * abs(acc - conf)
    return (100.0 * error / total if total else None), bins


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="앞 N장만 평가 (0=전체)")
    parser.add_argument("--subset_every", type=int, default=1,
                        help="데이터셋 순서에서 k장마다 1장 (1=전체)")
    args = parser.parse_args()
    if args.batch < 1 or args.limit < 0 or args.subset_every < 1:
        parser.error("--batch >= 1, --limit >= 0, --subset_every >= 1 이어야 합니다")

    valmod._unpad_resize_to_orig = legal_rescore_v2._unpad_resize_to_orig_v2
    with open(args.cfg, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    valmod.setup_cudnn()
    dataset_cfg = cfg["DATASET"]
    eval_cfg = cfg["EVAL"]
    image_size = (eval_cfg["IMAGE_SIZE"] if args.split == "val" else
                  cfg.get("TEST", {}).get("IMAGE_SIZE", eval_cfg["IMAGE_SIZE"]))
    transform = valmod.get_val_augmentation(image_size, dataset_cfg=dataset_cfg)
    dataset, has_gt = valmod.create_dataset(dataset_cfg, args.split, transform,
                                            args.split, macvi=False, eval_day=False)
    if not has_gt:
        raise ValueError(f"{args.split} split has no GT")
    modal_names = list(dataset_cfg["MODALS"])
    if len(modal_names) != 4:
        raise ValueError(f"Expected four DELIVER modalities, got {modal_names}")
    n_classes = dataset.n_classes
    ignore = dataset.ignore_label
    if ignore != 255:
        raise ValueError(f"Expected DELIVER ignore label 255, got {ignore}")
    if args.subset_every > 1:
        dataset = Subset(dataset, range(0, len(dataset), args.subset_every))
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=4,
                        pin_memory=False, collate_fn=valmod._collate_fn)
    model = valmod.load_model(cfg, Path(args.model_path), device)
    model.eval()

    present_subsets = [present for present in mme.enumerate_present_subsets(4)
                       if len(present) < 4]
    emm_cases = [(f"emm|present={mme._present_names(present, modal_names)}",
                  mme.missing_of(present, 4)) for present in present_subsets]
    if len(emm_cases) != 14:
        raise RuntimeError("Expected 14 non-clean EMM subsets")
    settings = ["clean"] + [name for name, _ in emm_cases] + [NM_ID]
    agree = {name: 0 for name in settings[1:]}
    total_valid = {name: 0 for name in settings}
    correct_count = {name: 0 for name in settings}
    hist = np.zeros((n_classes, n_classes), dtype=np.int64)
    ece_stats = {name: (torch.zeros(N_BINS, dtype=torch.int64),
                        torch.zeros(N_BINS, dtype=torch.float64),
                        torch.zeros(N_BINS, dtype=torch.float64))
                 for name in ("clean", NM_ID)}
    generator = torch.Generator().manual_seed(args.seed)
    n_images = 0

    with torch.no_grad():
        for images, _labels, metas in tqdm(loader, desc="prediction consistency / ECE"):
            if args.limit:
                remaining = args.limit - n_images
                if remaining <= 0:
                    break
                if remaining < len(metas):
                    images = [x[:remaining] for x in images]
                    metas = metas[:remaining]
            base = [x.to(device) for x in images]
            clean_preds = None
            for name, missing in [("clean", None), *emm_cases, (NM_ID, None)]:
                if name == "clean":
                    imgs = base
                elif name == NM_ID:
                    imgs = mme.bench_noise(base, modal_names, 0.1, 0.2, generator)
                else:
                    imgs = mme.zero_fill(base, missing)
                output = model(imgs, multimask_output=True)
                logits = output[0] if isinstance(output, (tuple, list)) else output
                probs = logits.softmax(dim=1)
                labels = valmod._argmax_pred(probs, n_classes, None)
                conf_maps = (probs[:, :n_classes].max(dim=1).values
                             if name in ece_stats else None)
                native_preds = []
                for b, meta in enumerate(metas):
                    orig_label = meta.get("orig_label")
                    if orig_label is None:
                        raise ValueError("meta['orig_label'] is missing")
                    label = labels[b]
                    pred = valmod._unpad_resize_to_orig(
                        label, meta["orig_h"], meta["orig_w"],
                        model_size=int(label.shape[0])).cpu()
                    gt = orig_label.cpu()
                    if pred.shape != gt.shape:
                        raise ValueError(f"prediction/GT shape mismatch: {pred.shape}, {gt.shape}")
                    valid = gt != 255
                    n_valid = int(valid.sum().item())
                    total_valid[name] += n_valid
                    is_correct = pred == gt
                    correct_count[name] += int((valid & is_correct).sum().item())
                    native_preds.append(pred)
                    if name == "clean":
                        hist += common.confusion_matrix(pred.numpy(), gt.numpy(),
                                                        n_classes, ignore)
                    else:
                        agree[name] += int((valid & (pred == clean_preds[b])).sum().item())
                    if conf_maps is not None:
                        conf = _unpad_resize_conf(conf_maps[b], meta["orig_h"],
                                                  meta["orig_w"], int(label.shape[0])).cpu()
                        accumulate_ece(*ece_stats[name], conf, is_correct, valid)
                if name == "clean":
                    clean_preds = native_preds
            n_images += len(metas)

    if n_images == 0 or total_valid["clean"] == 0:
        raise ValueError("No images with valid GT pixels were evaluated")
    agreement = {name: 100.0 * agree[name] / total_valid[name]
                 if total_valid[name] else None for name in settings[1:]}
    pixel_acc = {name: 100.0 * correct_count[name] / total_valid[name]
                 if total_valid[name] else None for name in settings}
    ece_pairs = {name: ece_result(*ece_stats[name]) for name in ("clean", NM_ID)}
    _, clean_miou = common.global_miou_from_cm(hist)
    summary = {
        "model_path": args.model_path,
        "split": args.split,
        "n_images": n_images,
        "settings": settings,
        "agreement": agreement,
        "agreement_emm14_mean": sum(agreement[name] for name, _ in emm_cases) / 14,
        "pixel_acc": pixel_acc,
        "ece": {name: pair[0] for name, pair in ece_pairs.items()},
        "ece_bins": {name: pair[1] for name, pair in ece_pairs.items()},
        "clean_mIoU": clean_miou,
        "batch": args.batch,
        "seed": args.seed,
        "limit": args.limit,
        "subset_every": args.subset_every,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    print(f"[pred-consistency] {args.split}: {n_images} images, clean mIoU={clean_miou:.4f}%")
    print(f"[pred-consistency] EMM14 agreement={summary['agreement_emm14_mean']:.3f}%, "
          f"NM Mid agreement={agreement[NM_ID]:.3f}%")
    print(f"[pred-consistency] ECE clean={summary['ece']['clean']:.3f}%, "
          f"NM Mid={summary['ece'][NM_ID]:.3f}% -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
