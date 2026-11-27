#!/usr/bin/env python3
"""DELIVER test 에서 센서 하나를 0 으로 채웠을 때 조건별 mIoU 가 얼마나 변하는지 한 번의 순회로 잰다.

설정 5 개(clean + no_<센서> 4 개)를 이미지마다 모두 forward 해, (설정, 그룹)마다 혼동행렬을
누적한다. 그룹은 이미지 id 에서 읽은 날씨 조건(sun/cloud/rain/fog/night)·저노출(lowexp)·
전체(all)다. 끝에 (설정, 그룹)별 mIoU 와 clean 대비 하락폭(drop)을 낸다.

추론·데이터 로딩·원본 해상도 복원은 ``tools/viz_fused_stability.py`` 와 같은 방식(val.py 재사용,
배치 1, ``model(imgs, multimask_output=True)``)을 그대로 따른다. 센서 제거는 ``tools/
missing_modality_eval.py`` 의 zero_fill/missing_of 를 쓴다. 원본 해상도 복원은 legal v2 규칙
(nearest-exact)으로, 시작할 때 ``tools/mm_eval_v2.py`` 와 똑같이 val._unpad_resize_to_orig 를
``legal_rescore_v2._unpad_resize_to_orig_v2`` 로 바꾼다. 학습·모델·기존 도구는 수정하지 않는다.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 조건 그룹 정의(단일 출처: 이 상수).
CONDITIONS = ("sun", "cloud", "rain", "fog", "night")
LOWEXP_MARKER = "underexposure"
GROUP_ORDER = list(CONDITIONS) + ["lowexp", "all"]


# ===========================================================================
# 순수 함수 (GPU·데이터 없이 --selftest 로 직접 검증)
# ===========================================================================
def groups_for_id(image_id):
    """이미지 id → 이 이미지가 속하는 그룹 목록.

    - 날씨 조건: id 의 경로 구성요소 중 sun/cloud/rain/fog/night 하나(``img/<조건>/...``).
    - lowexp: id 에 ``underexposure`` 가 들어 있으면 포함(조건 그룹과 겹쳐도 된다).
    - all: 항상 포함.
    반환 순서는 GROUP_ORDER 를 따른다.
    """
    parts = str(image_id).replace("\\", "/").split("/")
    found = set()
    for p in parts:
        if p in CONDITIONS:
            found.add(p)
            break
    if LOWEXP_MARKER in str(image_id):
        found.add("lowexp")
    found.add("all")
    return [g for g in GROUP_ORDER if g in found]


def miou_from_cm(cm):
    """혼동행렬 → mIoU(클래스 IoU 의 nanmean ×100).

    IoU = tp / (열합 + 행합 − tp). 분모가 0 인 클래스(한 번도 등장·예측 안 됨)는 NaN 으로
    두어 nanmean 에서 제외한다. 전 클래스가 NaN 이면 NaN 을 돌려준다.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / denom
    iou[denom == 0] = np.nan
    if np.all(np.isnan(iou)):
        return float("nan")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(iou)) * 100.0


def _confusion(pred, gt, n, ignore):
    """혼동행렬 hist[gt, pred]. common.confusion_matrix 우선, 없으면 bincount."""
    try:
        from tools.baseline_failure import common
        return common.confusion_matrix(pred, gt, n, ignore)
    except Exception:  # noqa: BLE001 — common import 불가 환경(순수 selftest)에서도 동작
        pred = np.asarray(pred).astype(np.int64).ravel()
        gt = np.asarray(gt).astype(np.int64).ravel()
        keep = (gt != ignore) & (gt >= 0) & (gt < n) & (pred >= 0) & (pred < n)
        idx = gt[keep] * n + pred[keep]
        return np.bincount(idx, minlength=n * n).reshape(n, n)


def _selftest():
    # 그룹 분류
    assert groups_for_id("img/night/val/scene1/000050_rgb_front") == ["night", "all"]
    assert groups_for_id("img/sun/test/scene2_underexposure/000001_rgb") == \
        ["sun", "lowexp", "all"], groups_for_id("img/sun/test/scene2_underexposure/000001_rgb")
    assert groups_for_id("img/rain/val/scene/000002") == ["rain", "all"]
    assert groups_for_id("img/foobar/val/scene/000003") == ["all"]

    # mIoU: 완전 일치 = 100, drop 0
    n = 5
    rng = np.random.RandomState(0)
    gt = rng.randint(0, n, size=(32, 32))
    cm_perfect = _confusion(gt.copy(), gt.copy(), n, 255)
    m = miou_from_cm(cm_perfect)
    assert abs(m - 100.0) < 1e-6, f"완전 일치 mIoU != 100: {m}"
    # clean 과 같은 설정 → drop = clean - self = 0
    drop = m - m
    assert drop == 0.0, f"self drop != 0: {drop}"

    # ignore 픽셀은 집계에서 빠진다
    gt2 = gt.copy()
    gt2[:4, :4] = 255
    cm2 = _confusion(gt.copy(), gt2, n, 255)
    assert abs(miou_from_cm(cm2) - 100.0) < 1e-6

    print("SELFTEST_OK")


# ===========================================================================
# 추론·집계 (val.py 재사용 — GPU·데이터 필요)
# ===========================================================================
def _ckpt_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="합성 예측으로 그룹 분류·mIoU 계산만 검증하고 종료(GPU·데이터 불필요)")
    ap.add_argument("--cfg", help="ReliaDINO DELIVER 평가 YAML")
    ap.add_argument("--model_path", help="체크포인트 경로")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--out", help="출력 디렉터리")
    ap.add_argument("--limit", type=int, default=0, help="앞 N 장만(0=전체)")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    for name in ("cfg", "model_path", "out"):
        if not getattr(args, name):
            ap.error(f"--{name} 가 필요합니다(--selftest 제외)")
    if args.limit < 0:
        ap.error("--limit>=0 이어야 합니다")

    import torch
    import yaml
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import val as valmod
    from tools import legal_rescore_v2
    from tools.baseline_failure import common
    from tools.missing_modality_eval import zero_fill, missing_of  # noqa: F401

    # legal v2 복원(nearest-exact)으로 몽키패치 — mm_eval_v2.py 와 동일.
    valmod._unpad_resize_to_orig = legal_rescore_v2._unpad_resize_to_orig_v2
    print(f"[sensor-dep] _unpad_resize_to_orig -> {legal_rescore_v2.RESAMPLE_MODE}", flush=True)

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
        raise ValueError("native GT 가 필요합니다(결측 split 은 채점 불가)")

    device = torch.device(cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    valmod.setup_cudnn()
    model = valmod.load_model(cfg, Path(args.model_path), device).eval()
    n_classes = dataset.n_classes

    # 설정: clean + no_<센서> 4 개(MODALS 순서).
    settings = [("clean", None)]
    for i, name in enumerate(modal_names):
        settings.append((f"no_{name}", (i,)))

    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=False,
                        collate_fn=valmod._collate_fn)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # (설정, 그룹) -> 혼동행렬. 그룹별 이미지 수도 센다.
    confusions = {f"{s}__{g}": np.zeros((n_classes, n_classes), dtype=np.int64)
                  for s, _ in settings for g in GROUP_ORDER}
    group_n = {g: 0 for g in GROUP_ORDER}
    total_images = 0

    def forward_pred(base_imgs, missing_idx):
        imgs = base_imgs if missing_idx is None else zero_fill(base_imgs, missing_idx)
        output = model(imgs, multimask_output=True)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        pred = valmod._argmax_pred(logits.softmax(dim=1), n_classes, None)[0]
        pred_orig = valmod._unpad_resize_to_orig(
            pred, meta["orig_h"], meta["orig_w"], model_size=int(pred.shape[0]))
        return pred_orig.cpu().numpy().astype(np.int64)

    seen = set()
    with torch.no_grad():
        for images, _labels, metas in tqdm(loader, desc=f"sensor-dep[{args.split}]"):
            if args.limit and total_images >= args.limit:
                break
            meta = metas[0]
            image_id = common.image_id_from_rel(meta["paths"]["img"])
            if image_id in seen:
                raise RuntimeError(f"중복 image_id: {image_id}")
            seen.add(image_id)
            gt = meta.get("orig_label")
            if gt is None:
                raise RuntimeError(f"native GT 없음: {image_id}")
            gt_np = gt.cpu().numpy().astype(np.int64)

            groups = groups_for_id(image_id)
            for g in groups:
                group_n[g] += 1

            base_imgs = [img.to(device) for img in images]
            for sname, miss in settings:
                pred_np = forward_pred(base_imgs, miss)
                cm = common.confusion_matrix(pred_np, gt_np, n_classes, 255)
                for g in groups:
                    confusions[f"{sname}__{g}"] += cm
            total_images += 1

    # mIoU·drop 계산
    used_groups = [g for g in GROUP_ORDER if group_n[g] > 0]
    miou = {}            # (setting, group) -> mIoU
    for sname, _ in settings:
        for g in used_groups:
            miou[(sname, g)] = miou_from_cm(confusions[f"{sname}__{g}"])

    rows = []
    for sname, _ in settings:
        for g in used_groups:
            clean_m = miou[("clean", g)]
            m = miou[(sname, g)]
            drop = (clean_m - m) if (np.isfinite(clean_m) and np.isfinite(m)) else float("nan")
            rows.append({
                "setting": sname,
                "group": g,
                "n_images": group_n[g],
                "mIoU": round(m, 4) if np.isfinite(m) else "",
                "drop_vs_clean": round(drop, 4) if np.isfinite(drop) else "",
            })

    # 저장: confusions.npz / dependence.csv / summary.json
    np.savez(out_root / "confusions.npz",
             **{k: v for k, v in confusions.items()
                if k.split("__", 1)[1] in used_groups})

    with (out_root / "dependence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["setting", "group", "n_images", "mIoU", "drop_vs_clean"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "cfg": str(args.cfg),
        "model_path": str(args.model_path),
        "ckpt_md5": _ckpt_md5(args.model_path),
        "split": args.split,
        "n_images": total_images,
        "modals": modal_names,
        "settings": [s for s, _ in settings],
        "groups": used_groups,
        "group_n_images": {g: group_n[g] for g in used_groups},
        "rows": rows,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[sensor-dep] processed {total_images} images "
          f"(groups={used_groups}) -> {out_root}")


if __name__ == "__main__":
    main()
