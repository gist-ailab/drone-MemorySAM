#!/usr/bin/env python3
"""GT 객체의 fused 토큰과 고해상 s4 특징을 비교하는 학습 없는 선형 프로브.

DELIVER 25-class ReliaDINO A2를 대상으로 한다. 객체는 변환된 GT의 8연결
성분이며, 두 특징에서 모두 유효한 객체만 저장하고 같은 fold에 사용한다.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.baseline_failure import common


FEATURES = ("fused", "hires_s4")
DEFAULT_PAIRS = ("TrafficLight:Other", "TrafficLight:TrafficSign")
OBJECT_FIELDS = ("image_id", "condition", "case", "class", "area")
PROBE_FIELDS = (
    "pair", "feature", "n_pos", "n_neg", "auroc_pooled",
    "auroc_fold_mean", "auroc_fold_std", "reason",
)


def parse_pairs(specs):
    pairs = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2 or parts[0] == parts[1] or any(
            name not in common.CLASSES for name in parts
        ):
            raise ValueError(f"잘못된 --pairs 값: {spec} (DELIVER 클래스명 A:B 필요)")
        pair = tuple(parts)
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("--pairs 에 적어도 한 쌍이 필요합니다")
    return pairs


def pool_object(feature, mask):
    """(C,h,w) 특징을 원본 GT 객체의 면적 비율로 가중 평균한다."""
    if feature.ndim != 3 or mask.ndim != 2:
        raise ValueError("feature=(C,h,w), mask=(H,W) 형태가 필요합니다")
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=feature.device)
    weights = F.adaptive_avg_pool2d(mask_t[None, None], feature.shape[-2:])[0, 0]
    mass = weights.sum()
    if mass.item() < 0.25:
        return None
    vector = (feature.float() * weights[None]).sum(dim=(-2, -1)) / mass
    return vector.detach().cpu().numpy().astype(np.float32)


def extract_objects(label, feature_maps, image_id, class_names, min_area):
    """한 이미지의 관심 클래스 8연결 객체를 두 특징의 공통 표본으로 추출."""
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    label = np.asarray(label)
    if label.ndim == 3 and label.shape[0] == 1:
        label = label[0]
    if label.ndim != 2:
        raise ValueError(f"GT label shape 이 (H,W) 가 아닙니다: {label.shape}")
    for name in FEATURES:
        if name not in feature_maps or feature_maps[name].ndim != 3:
            raise ValueError(f"{name} 특징은 (C,h,w) 텐서여야 합니다")

    condition, case = common.parse_condition_case(image_id)
    rows = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_name in class_names:
        class_id = common.CLASSES.index(class_name)
        components, count = ndimage.label(label == class_id, structure=structure)
        if count == 0:
            continue
        areas = np.bincount(components.ravel(), minlength=count + 1)
        for component_id in range(1, count + 1):
            area = int(areas[component_id])
            if area < min_area:
                continue
            mask = components == component_id
            vectors = {name: pool_object(feature_maps[name], mask) for name in FEATURES}
            if any(vector is None for vector in vectors.values()):
                continue
            rows.append({
                "image_id": image_id, "condition": condition, "case": case,
                "class": class_name, "area": area, **vectors,
            })
    return rows


def sklearn_components():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("선형 프로브에는 scikit-learn(sklearn)이 필요합니다") from exc
    return LogisticRegression, roc_auc_score, GroupKFold, make_pipeline, StandardScaler


def probe_pair(objects, pair, feature, seed, sk):
    LogisticRegression, roc_auc_score, GroupKFold, make_pipeline, StandardScaler = sk
    positive, negative = pair
    selected = [row for row in objects if row["class"] in pair]
    y = np.asarray([int(row["class"] == positive) for row in selected], dtype=np.int8)
    groups = np.asarray([row["image_id"] for row in selected])
    result = {
        "pair": f"{positive}:{negative}", "feature": feature,
        "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
        "auroc_pooled": None, "auroc_fold_mean": None,
        "auroc_fold_std": None, "reason": "",
    }
    if min(result["n_pos"], result["n_neg"]) < 10:
        result["reason"] = "클래스당 객체 10개 미만"
        return result
    if len(np.unique(groups)) < 5:
        result["reason"] = "서로 다른 이미지 5장 미만"
        return result

    x = np.stack([row[feature] for row in selected]).astype(np.float32)
    scores = np.full(len(y), np.nan, dtype=np.float64)
    fold_aucs = []
    missing_eval_class = 0
    for train_idx, test_idx in GroupKFold(n_splits=5).split(x, y, groups):
        if len(np.unique(y[train_idx])) != 2:
            result["reason"] = "한 fold의 학습 집합에 클래스 하나만 있음"
            return result
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2", C=1.0, max_iter=2000,
                class_weight="balanced", random_state=seed,
            ),
        )
        classifier.fit(x[train_idx], y[train_idx])
        scores[test_idx] = classifier.decision_function(x[test_idx])
        if len(np.unique(y[test_idx])) == 2:
            fold_aucs.append(float(roc_auc_score(y[test_idx], scores[test_idx])))
        else:
            missing_eval_class += 1
    if not np.isfinite(scores).all():
        raise RuntimeError("GroupKFold 평가 점수가 모든 객체에 생성되지 않았습니다")
    result["auroc_pooled"] = float(roc_auc_score(y, scores))
    if fold_aucs:
        result["auroc_fold_mean"] = float(np.mean(fold_aucs))
        result["auroc_fold_std"] = float(np.std(fold_aucs))
    if missing_eval_class:
        result["reason"] = f"평가 클래스 하나인 fold {missing_eval_class}개는 fold 통계에서 제외"
    return result


def md5_file(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_model_objects(args, pairs):
    import yaml
    sam2_root = Path(__file__).resolve().parents[1] / "semseg" / "models" / "sam2"
    sys.path.insert(0, str(sam2_root))
    import val as valmod
    from torch.utils.data import DataLoader

    cfg_path = Path(args.cfg)
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    if str(cfg["DATASET"].get("NAME", "")).upper() != "DELIVER":
        raise ValueError("이 프로브는 DELIVER 25클래스 라벨만 지원합니다")
    device = torch.device(args.device)
    model = valmod.load_model(cfg, model_path, device)
    model.eval()
    if getattr(model, "hires", None) is None:
        raise ValueError("model.hires 가 없습니다. ReliaDINO A2 체크포인트가 필요합니다")

    transform = valmod.get_val_augmentation(
        [args.size, args.size], dataset_cfg=cfg["DATASET"]
    )
    dataset, has_gt = valmod.create_dataset(
        cfg["DATASET"], args.split, transform, args.split,
        macvi=False, eval_day=False,
    )
    if not has_gt:
        raise ValueError(f"{args.split} split 에 GT가 없습니다")
    loader = DataLoader(dataset, batch_size=1, collate_fn=valmod._collate_fn)
    classes = set(name for pair in pairs for name in pair)
    ordered_classes = [name for name in common.CLASSES if name in classes]
    captured = []

    def capture_fusion(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError("fusion hook 출력이 (B,C,h,w) 텐서가 아닙니다")
        captured.append(output.detach())

    hook = model.fusion.register_forward_hook(capture_fusion)
    objects = []
    image_count = 0
    try:
        with torch.no_grad():
            for images, labels, metas in loader:
                if args.limit and image_count >= args.limit:
                    break
                captured.clear()
                images = [image.to(device) for image in images]
                model(images, multimask_output=True)
                if len(captured) != 1:
                    raise RuntimeError(f"fusion forward hook 호출 횟수 {len(captured)} (기대 1)")
                hires_feats = getattr(model, "_hires_feats", None)
                if hires_feats is None or len(hires_feats) == 0 or hires_feats[0] is None:
                    raise RuntimeError("model._hires_feats[0] (s4) 이 없습니다")
                fused = captured[0]
                hires_s4 = hires_feats[0]
                if fused.shape[0] != 1 or hires_s4.ndim != 4 or hires_s4.shape[0] != 1:
                    raise RuntimeError("batch_size=1 특징 형식이 아닙니다")
                rgb_path = metas[0]["paths"]["img"]
                image_id = common.image_id_from_rel(
                    rgb_path, dataset_root=cfg["DATASET"]["ROOT"]
                )
                objects.extend(extract_objects(
                    labels[0], {"fused": fused[0], "hires_s4": hires_s4[0]},
                    image_id, ordered_classes, args.min_area,
                ))
                image_count += 1
                if image_count % 100 == 0:
                    print(f"images={image_count} objects={len(objects)}", flush=True)
    finally:
        hook.remove()
    return objects, image_count, model_path.resolve(), md5_file(model_path)


def dry_run_objects(seed):
    """동일 이미지의 네 GT 객체에 분리 가능/무작위 특징을 부여한다."""
    rng = np.random.default_rng(seed)
    names = ("TrafficLight", "Other", "TrafficSign", "Pole")
    corners = ((0, 0), (0, 24), (24, 0), (24, 24))
    objects = []
    for image_idx in range(100):
        label = np.full((32, 32), 255, dtype=np.uint8)
        maps = {"fused": torch.zeros((4, 4, 4)), "hires_s4": torch.zeros((4, 8, 8))}
        expected = {}
        for name, (top, left) in zip(names, corners):
            label[top:top + 8, left:left + 8] = common.CLASSES.index(name)
            for feature in FEATURES:
                vector = rng.normal(0, 1, size=4).astype(np.float32)
                if name == "TrafficLight":
                    vector[0] += 4
                elif name == "Other":
                    vector[0] -= 4
                grid = 8 if feature == "hires_s4" else 4
                scale = grid // 4
                y, x = (top // 8) * scale, (left // 8) * scale
                maps[feature][:, y:y + scale, x:x + scale] = torch.from_numpy(vector)[:, None, None]
                expected[name, feature] = vector
        image_id = f"img/sun/val/scene_{image_idx:04d}/frame_rgb"
        found = extract_objects(label, maps, image_id, names, min_area=64)
        assert len(found) == 4 and all(row["area"] == 64 for row in found)
        for row in found:
            for feature in FEATURES:
                assert np.allclose(row[feature], expected[row["class"], feature])
        objects.extend(found)
    assert len(objects) == 400
    return objects, 100


def write_outputs(out, objects, results, args, pairs, image_count, ckpt_path, ckpt_md5):
    out.mkdir(parents=True, exist_ok=True)
    meta = {field: np.asarray([row[field] for row in objects]) for field in OBJECT_FIELDS}
    vectors = {}
    for name in FEATURES:
        vectors[name] = (np.stack([row[name] for row in objects]).astype(np.float16)
                         if objects else np.empty((0, 0), dtype=np.float16))
    np.savez_compressed(out / "objects.npz", **meta, **vectors)
    with (out / "objects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OBJECT_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in OBJECT_FIELDS} for row in objects)
    with (out / "probe_auroc.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROBE_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    def fmt(value):
        return "null" if value is None else f"{value:.4f}"

    lines = [
        "# GT object pooling linear probe", "",
        f"Images: {image_count}; objects: {len(objects)}", "",
        "| Pair | Feature | Pos | Neg | Pooled AUROC | Fold mean | Fold std | Reason |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['pair']} | {row['feature']} | {row['n_pos']} | {row['n_neg']} | "
            f"{fmt(row['auroc_pooled'])} | {fmt(row['auroc_fold_mean'])} | "
            f"{fmt(row['auroc_fold_std'])} | {row['reason']} |"
        )
    (out / "probe_summary.md").write_text("\n".join(lines) + "\n")
    counts = {
        f"{a}:{b}": {a: sum(row["class"] == a for row in objects),
                      b: sum(row["class"] == b for row in objects)}
        for a, b in pairs
    }
    run = {
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "checkpoint_md5": ckpt_md5, "n_images": image_count,
        "n_objects": len(objects), "pair_object_counts": counts,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg")
    parser.add_argument("--model_path")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.size <= 0 or args.min_area <= 0 or args.limit < 0:
        parser.error("--size/--min_area 는 양수, --limit 는 0 이상이어야 합니다")
    pairs = parse_pairs(args.pairs)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dry_run:
        objects, image_count = dry_run_objects(args.seed)
        dry_pairs = (("TrafficLight", "Other"), ("TrafficSign", "Pole"))
        try:
            sk = sklearn_components()
        except ImportError:
            print("sklearn 없음: dry_run 프로브 단계 생략; 객체 추출·풀링 검증 통과")
            results = [
                {"pair": f"{a}:{b}", "feature": feature,
                 "n_pos": 100, "n_neg": 100, "auroc_pooled": None,
                 "auroc_fold_mean": None, "auroc_fold_std": None,
                 "reason": "sklearn 없음"}
                for a, b in dry_pairs for feature in FEATURES
            ]
        else:
            results = [probe_pair(objects, pair, feature, args.seed, sk)
                       for pair in dry_pairs for feature in FEATURES]
            for row in results:
                auc = row["auroc_pooled"]
                if row["pair"] == "TrafficLight:Other":
                    assert auc is not None and auc >= 0.95, row
                else:
                    assert auc is not None and 0.3 <= auc <= 0.7, row
            print("dry_run: 분리 쌍 AUROC >= 0.95, 무작위 쌍 AUROC 0.3~0.7 통과")
        write_outputs(args.out, objects, results, args, dry_pairs, image_count, None, None)
        return

    if not args.cfg or not args.model_path:
        parser.error("실제 실행에는 --cfg 와 --model_path 가 필요합니다")
    sk = sklearn_components()  # 실제 실행은 sklearn 부재 시 추론 전에 중단
    objects, image_count, ckpt_path, ckpt_md5 = collect_model_objects(args, pairs)
    results = [probe_pair(objects, pair, feature, args.seed, sk)
               for pair in pairs for feature in FEATURES]
    write_outputs(args.out, objects, results, args, pairs, image_count, ckpt_path, ckpt_md5)
    print(f"images={image_count} objects={len(objects)} output={args.out}")


if __name__ == "__main__":
    main()
