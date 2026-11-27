#!/usr/bin/env python3
"""융합 직후 특징(``model._last_fused_postfusion``)이 센서 제거에 얼마나 안정적인지 측정·시각화한다.

설정 5개(clean + 센서 하나를 제거한 no_<센서> 4개)에서 같은 이미지의 융합 특징을 비교한다.
- stats: 위치별 cosine(F_clean, F_removed) 의 이미지 평균을 모은다.
- panels: 대상 이미지마다 설정 5개의 특징·예측·RGB·GT 를 저장하고, 대상 전체의 clean 특징으로
  구한 PCA 기저 하나로 모든 설정 특징을 같은 색공간에 투영해 PNG 로 남긴다.

학습·모델·기존 도구를 수정하지 않는다. 추론·데이터 로딩·이미지 id·원본 해상도 복원은
``tools/viz_hires_pyramid.py``·``tools/missing_modality_eval.py`` 와 같은 방식(val.py 재사용)이다.
센서 제거는 ``tools/missing_modality_eval.py`` 의 zero_fill/missing_of 를 그대로 쓴다.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ===========================================================================
# 순수 함수 (GPU·데이터 없이 --selftest 로 직접 검증)
# ===========================================================================
def to_grid(feat):
    """융합 특징(배치 1 제거 전) → (C, h, w) 텐서.

    - (B, C, h, w) 공간 형태면 그대로 (B=1 가정, 0번째 사용).
    - (B, N, C) 토큰 형태면 N 이 정사각수일 때만 (C, h, w) 격자로 바꾼다(아니면 오류).
    """
    if feat.dim() == 4:
        return feat[0]
    if feat.dim() == 3:
        _, n, c = feat.shape
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(f"토큰 수 {n} 이 정사각이 아니라 격자로 바꿀 수 없습니다")
        return feat[0].reshape(side, side, c).permute(2, 0, 1).contiguous()
    raise ValueError(f"지원하지 않는 특징 차원: {tuple(feat.shape)}")


def cosine_map(a, b, eps=1e-8):
    """채널 축 cosine 유사도 맵. a, b = (C, h, w) → (h, w)."""
    if a.shape != b.shape:
        raise ValueError(f"cosine_map 입력 크기 불일치: {tuple(a.shape)} vs {tuple(b.shape)}")
    a = a.float()
    b = b.float()
    dot = (a * b).sum(dim=0)
    na = a.norm(dim=0)
    nb = b.norm(dim=0)
    return dot / (na * nb + eps)


def fit_pca_basis(feats, lo_pct=1.0, hi_pct=99.0):
    """위치 단위로 모은 clean 특징(K, C) → PCA 기저 dict.

    반환 {"mean": (C,), "components": (C, 3), "lo": (3,), "hi": (3,)}.
    lo/hi 는 clean 투영값의 성분별 1·99 백분위(색 범위 고정용)이다.
    components 는 직교(정규직교)다.
    """
    feats = feats.float()
    mean = feats.mean(dim=0)
    centered = feats - mean
    # q 는 성분 수 3 보다 넉넉히 둬 수치 안정성을 높인다.
    q = min(centered.shape[1], max(6, 3))
    _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    components = v[:, :3].contiguous()
    projected = centered @ components              # (K, 3)
    lo = torch.quantile(projected, lo_pct / 100.0, dim=0)
    hi = torch.quantile(projected, hi_pct / 100.0, dim=0)
    return {"mean": mean, "components": components, "lo": lo, "hi": hi}


def project_rgb(feat, basis):
    """(C, h, w) 특징을 기저로 투영해 [0,1] RGB (h, w, 3) float 로 만든다.

    같은 백분위(basis['lo']/['hi'])로 잘라 정규화하므로 출력은 항상 [0,1] 범위다.
    """
    c, h, w = feat.shape
    flat = feat.float().permute(1, 2, 0).reshape(-1, c)     # (h*w, C)
    projected = (flat - basis["mean"]) @ basis["components"]  # (h*w, 3)
    lo = basis["lo"]
    hi = basis["hi"]
    span = (hi - lo).clamp_min(1e-12)
    rgb = ((projected - lo) / span).clamp(0.0, 1.0)
    return rgb.reshape(h, w, 3)


def _selftest():
    torch.manual_seed(0)
    x = torch.randn(8, 5, 5)
    cos_same = cosine_map(x, x)
    assert torch.allclose(cos_same, torch.ones_like(cos_same), atol=1e-4), "cos(x,x)!=1"
    cos_neg = cosine_map(x, -x)
    assert torch.allclose(cos_neg, -torch.ones_like(cos_neg), atol=1e-4), "cos(x,-x)!=-1"

    feats = torch.randn(400, 32)
    basis = fit_pca_basis(feats)
    gram = basis["components"].T @ basis["components"]
    assert torch.allclose(gram, torch.eye(3), atol=1e-4), "PCA 기저가 직교가 아님"

    clean = torch.randn(32, 6, 7)
    rgb = project_rgb(clean, basis)
    assert rgb.shape == (6, 7, 3)
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0, "투영 RGB 가 [0,1] 밖"

    # 토큰 형태 격자 변환
    tok = torch.randn(1, 16, 32)
    grid = to_grid(tok)
    assert grid.shape == (32, 4, 4)
    print("SELFTEST_OK")


# ===========================================================================
# 추론·저장 (val.py 재사용 — GPU·데이터 필요)
# ===========================================================================
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _ckpt_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_ids(path):
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
           if line.strip()]
    if not ids:
        raise ValueError("--ids-file 에 image_id 가 없습니다")
    if len(ids) != len(set(ids)):
        raise ValueError("--ids-file 에 중복 image_id 가 있습니다")
    return ids


def _save_label_png(path, arr):
    from PIL import Image
    Image.fromarray(np.asarray(arr).astype(np.uint8)).save(str(path))


def _denorm_rgb(img_tensor):
    """(C, H, W) 정규화된 img 텐서 → 역정규화 uint8 (H, W, 3). img 는 ImageNet 정규화."""
    x = img_tensor.detach().float().cpu()
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    x = (x[:3] * std + mean).clamp(0.0, 1.0)
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="합성 텐서로 순수 함수만 검증하고 종료(GPU·데이터 불필요)")
    ap.add_argument("--cfg", help="ReliaDINO DELIVER 평가 YAML")
    ap.add_argument("--model_path", help="체크포인트 경로")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--out", help="출력 디렉터리")
    ap.add_argument("--mode", default="both", choices=("stats", "panels", "both"))
    ap.add_argument("--ids-file", help="패널 대상 image_id 목록(한 줄에 하나)")
    ap.add_argument("--random-n", type=int, default=0, help="무작위로 추가할 패널 대상 수")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="stats 대상 앞 N 장(0=전체)")
    ap.add_argument("--subset_every", type=int, default=1,
                    help="stats 대상 중 k 장마다 1 장(결정론적 부분집합)")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    for name in ("cfg", "model_path", "out"):
        if not getattr(args, name):
            ap.error(f"--{name} 가 필요합니다(--selftest 제외)")
    if args.limit < 0 or args.subset_every < 1 or args.random_n < 0:
        ap.error("--limit>=0, --subset_every>=1, --random-n>=0 이어야 합니다")
    want_stats = args.mode in ("stats", "both")
    want_panels = args.mode in ("panels", "both")

    import yaml
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import val as valmod
    from tools.baseline_failure import common
    from tools.missing_modality_eval import zero_fill, missing_of

    with open(args.cfg, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    dataset_cfg = cfg["DATASET"]
    eval_cfg = cfg["EVAL"]
    modal_names = list(dataset_cfg["MODALS"])
    M = len(modal_names)
    image_size = (eval_cfg["IMAGE_SIZE"] if args.split == "val" else
                  cfg.get("TEST", {}).get("IMAGE_SIZE", eval_cfg["IMAGE_SIZE"]))
    transform = valmod.get_val_augmentation(image_size, dataset_cfg=dataset_cfg)
    dataset, has_gt = valmod.create_dataset(dataset_cfg, args.split, transform,
                                            args.split, macvi=False, eval_day=False)
    if not has_gt:
        raise ValueError("native GT 가 필요합니다")

    device = torch.device(cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    valmod.setup_cudnn()
    model = valmod.load_model(cfg, Path(args.model_path), device).eval()
    n_classes = dataset.n_classes

    # 설정: clean + no_<센서> 4 개(MODALS 순서).
    settings = [("clean", None)]
    for i, name in enumerate(modal_names):
        settings.append((f"no_{name}", (i,)))

    # 패널 대상: ids-file + 무작위 N 개.
    all_ids = sorted(common.image_id_from_rel(f) for f in dataset.files)
    targets = []
    if want_panels:
        if args.ids_file:
            targets.extend(_read_ids(args.ids_file))
        random_ids = []
        if args.random_n > 0:
            pool = [i for i in all_ids if i not in set(targets)]
            if args.random_n > len(pool):
                raise ValueError(f"--random-n={args.random_n} 이 가용 id 수 {len(pool)} 보다 큽니다")
            rng = np.random.RandomState(args.random_seed)
            random_ids = list(rng.choice(pool, size=args.random_n, replace=False))
            targets.extend(random_ids)
        targets = list(dict.fromkeys(targets))
    target_set = set(targets)

    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=False,
                        collate_fn=valmod._collate_fn)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if want_panels and args.random_n > 0:
        (out_root / "random_ids.txt").write_text(
            "\n".join(str(i) for i in random_ids) + ("\n" if random_ids else ""),
            encoding="utf-8")

    stats_rows = []
    panel_feats = {}        # image_id -> {setting: (C,h,w) float16 np}
    seen = set()
    found = set()
    stats_count = 0
    global_idx = -1

    def forward_setting(base_imgs, missing_idx):
        imgs = base_imgs if missing_idx is None else zero_fill(base_imgs, missing_idx)
        output = model(imgs, multimask_output=True)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        fused = model._last_fused_postfusion
        if fused is None:
            raise RuntimeError("model._last_fused_postfusion 가 None 입니다(eval forward 확인)")
        grid = to_grid(fused).detach().float().cpu()
        return logits.detach(), grid

    with torch.no_grad():
        for images, _labels, metas in tqdm(loader, desc=f"fused-stability[{args.split}]"):
            global_idx += 1
            meta = metas[0]
            image_id = common.image_id_from_rel(meta["paths"]["img"])
            if image_id in seen:
                raise RuntimeError(f"중복 image_id: {image_id}")
            seen.add(image_id)

            do_stats = (want_stats and (global_idx % args.subset_every == 0)
                        and (not args.limit or stats_count < args.limit))
            do_panel = want_panels and image_id in target_set
            if not do_stats and not do_panel:
                continue

            base_imgs = [img.to(device) for img in images]
            grids = {}
            preds = {}
            for sname, miss in settings:
                logits, grid = forward_setting(base_imgs, miss)
                grids[sname] = grid
                if do_panel:
                    pred = valmod._argmax_pred(logits.softmax(dim=1), n_classes, None)[0]
                    pred_orig = valmod._unpad_resize_to_orig(
                        pred, meta["orig_h"], meta["orig_w"],
                        model_size=int(pred.shape[0]))
                    preds[sname] = pred_orig.cpu().numpy().astype(np.uint8)

            if do_stats:
                clean = grids["clean"]
                row = {"image_id": image_id}
                for sname, miss in settings:
                    if miss is None:
                        continue
                    cmap = cosine_map(clean, grids[sname])
                    row[sname] = float(cmap.mean())
                stats_rows.append(row)
                stats_count += 1

            if do_panel:
                panel_dir = out_root / "panels" / image_id
                panel_dir.mkdir(parents=True, exist_ok=True)
                img_idx = modal_names.index("img") if "img" in modal_names else 0
                rgb = _denorm_rgb(images[img_idx][0])
                from PIL import Image
                Image.fromarray(rgb).save(panel_dir / "rgb.png")
                gt = meta.get("orig_label")
                if gt is None:
                    raise RuntimeError(f"native GT 없음: {image_id}")
                _save_label_png(panel_dir / "gt.png",
                                gt.cpu().numpy().astype(np.uint8))
                feats_fp16 = {}
                for sname, _miss in settings:
                    g16 = grids[sname].numpy().astype(np.float16)
                    np.save(panel_dir / f"feat_{sname}.npy", g16)
                    feats_fp16[sname] = g16
                    _save_label_png(panel_dir / f"pred_{sname}.png", preds[sname])
                panel_feats[image_id] = feats_fp16
                found.add(image_id)

    missing_targets = target_set - found
    if missing_targets:
        raise ValueError(f"--ids-file/무작위 대상 중 찾지 못한 id: {sorted(missing_targets)}")

    # stats 저장
    if want_stats:
        if not stats_rows:
            raise RuntimeError("stats 대상 이미지가 없습니다")
        removed = [s for s, m in settings if m is not None]
        fields = ["image_id"] + removed
        with (out_root / "stats_per_image.csv").open("w", newline="",
                                                     encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(stats_rows)
        per_setting = {}
        for sname in removed:
            vals = np.array([r[sname] for r in stats_rows], dtype=np.float64)
            per_setting[sname] = {
                "mean_cosine": float(vals.mean()),
                "std_cosine": float(vals.std()),
                "n_images": int(vals.size),
            }
        summary = {
            "n_images": len(stats_rows),
            "removed_modals": removed,
            "modals": modal_names,
            "split": args.split,
            "subset_every": int(args.subset_every),
            "limit": int(args.limit),
            "per_setting": per_setting,
        }
        (out_root / "stats_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # PCA 기저: 대상 전체 clean 특징을 위치 단위로 모아 한 번만 구한다.
    if want_panels and panel_feats:
        clean_stack = []
        for feats in panel_feats.values():
            g = torch.from_numpy(feats["clean"].astype(np.float32))  # (C,h,w)
            c = g.shape[0]
            clean_stack.append(g.permute(1, 2, 0).reshape(-1, c))
        pooled = torch.cat(clean_stack, dim=0)
        basis = fit_pca_basis(pooled)
        np.savez(out_root / "pca_basis.npz",
                 mean=basis["mean"].numpy(), components=basis["components"].numpy(),
                 lo=basis["lo"].numpy(), hi=basis["hi"].numpy())
        from PIL import Image
        for image_id, feats in panel_feats.items():
            panel_dir = out_root / "panels" / image_id
            for sname, _miss in settings:
                g = torch.from_numpy(feats[sname].astype(np.float32))
                rgb01 = project_rgb(g, basis)                   # (h,w,3) [0,1]
                arr = (rgb01.numpy() * 255).round().astype(np.uint8)
                big = np.array(Image.fromarray(arr).resize((512, 512), Image.NEAREST))
                Image.fromarray(big).save(panel_dir / f"pca_{sname}.png")

    run_meta = {
        "n_stats_images": len(stats_rows) if want_stats else 0,
        "n_panel_images": len(found),
        "mode": args.mode,
        "split": args.split,
        "ckpt": str(args.model_path),
        "ckpt_md5": _ckpt_md5(args.model_path),
        "cfg": str(args.cfg),
        "settings": [s for s, _ in settings],
    }
    (out_root / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"processed stats={run_meta['n_stats_images']}, "
          f"panels={run_meta['n_panel_images']} -> {out_root}")


if __name__ == "__main__":
    main()
