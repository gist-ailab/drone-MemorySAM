#!/usr/bin/env python3
"""
D1(기준선) — detectron2 `SemSegEvaluator` 예측 JSON → trainID PNG 복원 변환기.

기준선(DGFusion·CAFuser)은 재추론 대신, `--eval-only` 가 이미 남긴
`inference/sem_seg_predictions.json`(이미지별 `{file_name, category_id, segmentation
(COCO RLE)}` 목록, 약 100 MB)이 val·test 모두 존재한다. 이 JSON 을 우리 덤프와
**같은 저장 규약**(상대 경로 보존 중첩 PNG, trainID 0~24)으로 복원한다.

복원 규칙:
- 이미지(file_name)별로 category_id 별 RLE 를 `pycocotools.mask.decode` 로 풀어
  한 장의 라벨 맵으로 합친다. 겹치는 픽셀이 있으면 경고 카운트만 하고 나중에 그린
  category 로 남긴다(argmax 예측이라 정상적으로는 겹치지 않는다).
- 어느 category 에도 속하지 않은 픽셀은 255(ignore) 로 둔다.
- image_id = file_name 의 **데이터셋 루트 기준 상대 경로**(확장자 제외). 기준선 저장소
  의 file_name 은 절대 경로일 수 있으므로 `--dataset_root` 로 잘라낸다. RGB 경로의
  `img` 접두어를 다른 규약으로 바꿔야 하면 `--path-sub OLD NEW` 로 치환한다.
- 저장이 끝나면 장수를 세어 기대값(val 2005 / test 1897, 또는 --expected_count)과
  다르면 assert 로 멈춘다.

라벨 규약 보정(--label_offset):
- 기준선 category_id 가 우리 trainID(0~24)와 같은지는 `check_label_convention.py` 로
  먼저 확정하라. 기본 offset=0 은 **보정 없이** 그대로 쓰며, 보정이 필요하면 그
  도구가 알려준 이동량을 `--label_offset` 로 넘긴다(추측 보정 금지). offset 적용 뒤
  category_id 가 0~24 밖이면 명확한 에러로 멈춘다(예: max=25 → 1-based 의심).

재현 검산(D1 유효 판정, 선택):
- `--gt_dir <우리 GT 덤프>` 와 `--expected_miou <로그값>` 를 주면 변환된 PNG 로 전역
  mIoU 를 재계산해 로그 수치(DGFusion test 80k 55.6807 / final 55.5605,
  CAFuser test final 55.3761, val 66.0405)와 ±--tol(기본 0.05) 안에 들면 summary 에
  `reproduced: true` 를 기록한다. GT 해상도가 달라 pred 를 리샘플했으면 그 사실도
  summary 에 남긴다(기준선 evaluator 의 GT 읽기 방식 — native 1042 vs 리사이즈 — 을
  함께 확인·보고할 것).

1024 단언(--keep_1024, A8):
- 저장은 옵션·GT 유무와 무관하게 **항상 RLE 해상도 그대로** 한다(기본 동작. 이미
  만들어진 1024 덤프와 새 덤프의 저장 해상도가 섞이는 사고를 막는다).
- `--keep_1024` 는 저장 해상도를 바꾸는 옵션이 아니라 **RLE 의 size 가 정확히
  1024x1024 인지 단언(검증)**하는 옵션이다 — 어긋나면 임의 보정 없이 명확한 에러로
  멈춘다.
- summary.json 에 실제 저장 해상도 `pred_resolution` 과 `keep_1024`(옵션 사용 여부)를
  옵션과 무관하게 항상 기록한다.

예:
  python tools/baseline_failure/rle_json_to_png.py \
    --json output/.../inference/sem_seg_predictions.json \
    --dataset_root $PWD/datasets/DELIVER --split test \
    --out <ROOT>/dgffinal --expected_count 1897 \
    --gt_dir <ROOT>/ours/test/gt --expected_miou 55.5605
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402

N = common.N_CLASSES


def _decode_rle(seg):
    """COCO RLE(dict) → (H, W) uint8 0/1 마스크. JSON 직렬화로 str 이 된 counts 를
    pycocotools 가 요구하는 bytes 로 되돌린다."""
    from pycocotools import mask as mask_util
    rle = dict(seg)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("utf-8")
    return mask_util.decode(rle)


def group_by_image(records):
    """예측 레코드 목록을 file_name 별로 묶는다."""
    groups = defaultdict(list)
    for rec in records:
        groups[rec["file_name"]].append(rec)
    return groups


def build_label_map(recs, label_offset):
    """한 이미지의 category 별 RLE 를 합쳐 라벨 맵(255 배경)으로. 겹침 픽셀 수도 반환.

    반환 = (label(H,W uint8), overlap_pixels, seen_category_ids set).
    """
    # 크기: 첫 RLE 의 size([h, w]).
    h, w = recs[0]["segmentation"]["size"]
    label = np.full((h, w), 255, dtype=np.uint8)
    assigned = np.zeros((h, w), dtype=bool)
    overlap = 0
    seen = set()
    for rec in recs:
        cid = int(rec["category_id"]) + label_offset
        seen.add(int(rec["category_id"]))
        if not (0 <= cid < N):
            raise ValueError(
                f"category_id={rec['category_id']} (offset {label_offset} 적용 후 {cid}) "
                f"가 trainID 범위 0~{N - 1} 밖이다. check_label_convention.py 로 이동량을 "
                f"확정한 뒤 --label_offset 을 지정하라(추측 보정 금지).")
        m = _decode_rle(rec["segmentation"]).astype(bool)
        if m.shape != (h, w):
            raise ValueError(f"RLE 크기 불일치: {m.shape} != {(h, w)} (file={rec['file_name']})")
        overlap += int(np.count_nonzero(assigned & m))
        label[m] = cid
        assigned |= m
    return label, overlap, seen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="sem_seg_predictions.json")
    ap.add_argument("--dataset_root", default=None,
                    help="file_name 절대 경로에서 잘라낼 데이터셋 루트")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--out", required=True, help="덤프 루트(<out>/<split>/pred/... 생성)")
    ap.add_argument("--expected_count", type=int, default=None,
                    help="기대 장수(미지정 시 split 기본값: val 2005 / test 1897)")
    ap.add_argument("--label_offset", type=int, default=0,
                    help="category_id 에 더할 이동량(기본 0=보정 없음)")
    ap.add_argument("--path-sub", nargs=2, metavar=("OLD", "NEW"), default=None,
                    help="상대 경로 문자열 치환 규칙(예: img 접두어 교체)")
    ap.add_argument("--gt_dir", default=None,
                    help="재현 검산용 GT 덤프 디렉터리(우리 GT, trainID)")
    ap.add_argument("--expected_miou", type=float, default=None,
                    help="로그 mIoU(주면 재현 검산 후 summary 에 reproduced 기록)")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--keep_1024", action="store_true",
                    help="저장 해상도가 정확히 1024x1024 임을 단언(검증)한다 — RLE"
                         " size 가 1024² 가 아니면 임의 보정 없이 에러로 멈춘다."
                         " 저장 해상도 자체는 옵션과 무관하게 항상 RLE 해상도 그대로다")
    args = ap.parse_args()

    out_root = Path(args.out) / args.split
    pred_dir = out_root / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rle2png] JSON 로드: {args.json}")
    records = json.loads(Path(args.json).read_text(encoding="utf-8"))
    if isinstance(records, dict):                 # {"predictions": [...]} 형태 방어
        records = records.get("predictions", records.get("annotations", []))
    groups = group_by_image(records)
    print(f"[rle2png] 이미지 {len(groups)} 장 · 레코드 {len(records)} 건")

    path_sub = tuple(args.path_sub) if args.path_sub else None
    n_saved = 0
    total_overlap = 0
    total_ignore = 0
    total_px = 0
    seen_ids = set()
    saved_shapes = set()
    cat_min, cat_max = None, None
    for fname, recs in groups.items():
        image_id = common.image_id_from_rel(
            fname, dataset_root=args.dataset_root, path_sub=path_sub)
        if image_id in seen_ids:
            raise RuntimeError(f"image_id 충돌: {image_id} — 경로 규약 확인 필요")
        seen_ids.add(image_id)
        label, overlap, seen = build_label_map(recs, args.label_offset)
        if args.keep_1024:
            # 1024 단언(A8): 저장은 항상 RLE 해상도 그대로 — 여기선 size 검증만 한다.
            if label.shape != (1024, 1024):
                raise ValueError(
                    f"--keep_1024 인데 디코드한 RLE 해상도가 {label.shape[1]}x{label.shape[0]}"
                    f"(기대 1024x1024, file={fname}) — 다른 해상도로 추론한 예측이다. "
                    f"해상도를 확인해 다시 덤프하라(임의 보정 금지).")
        common.save_label_png(pred_dir / f"{image_id}.png", label)
        n_saved += 1
        total_overlap += overlap
        total_ignore += int(np.count_nonzero(label == 255))
        total_px += label.size
        saved_shapes.add((label.shape[0], label.shape[1]))
        if seen:
            lo, hi = min(seen), max(seen)
            cat_min = lo if cat_min is None else min(cat_min, lo)
            cat_max = hi if cat_max is None else max(cat_max, hi)

    # 장수 검증.
    expected = args.expected_count or common.EXPECTED_COUNTS.get(args.split)
    if expected is not None:
        assert n_saved == expected, (
            f"장수 불일치: split='{args.split}' 에 {n_saved}장 저장(기대 {expected}). "
            f"JSON 이 스플릿 전체를 담고 있는지, file_name 상대 경로가 유일한지 확인.")

    summary = {
        "split": args.split,
        "json": str(args.json),
        "dataset_root": args.dataset_root,
        "label_offset": args.label_offset,
        "num_images": n_saved,
        "expected_count": expected,
        "overlap_pixels": total_overlap,
        "ignore_pixel_ratio": round(total_ignore / max(total_px, 1), 6),
        "category_id_min": cat_min,
        "category_id_max": cat_max,
    }
    if total_overlap:
        print(f"[rle2png] ⚠️ 겹침 픽셀 {total_overlap} — argmax 예측이면 0 이어야 한다.")

    # 저장 해상도 요약 — 모든 이미지가 같은 정사각 해상도면 int, 아니면 [h, w] 목록.
    shapes = sorted(saved_shapes)
    pred_resolution = (shapes[0][0] if len(shapes) == 1 and shapes[0][0] == shapes[0][1]
                       else [list(s) for s in shapes])
    summary["keep_1024"] = bool(args.keep_1024)
    summary["pred_resolution"] = pred_resolution

    # 재현 검산(D1 유효 판정) — pred 를 GT 해상도로 최근접 리샘플하는 기존 로직 그대로.
    if args.gt_dir and args.expected_miou is not None:
        summary.update(_reproduce_check(pred_dir, args.gt_dir, args.expected_miou, args.tol))

    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[rle2png] {n_saved} 장 저장 -> {pred_dir}")
    print(f"[rle2png] summary -> {out_root / 'summary.json'}")

    if summary.get("reproduced") is False:
        print("[rle2png] ⚠️ 재현 실패 — 덤프 무효. 라벨 규약(--label_offset)·GT 해상도 확인.")
        sys.exit(2)


def _reproduce_check(pred_dir, gt_dir, expected_miou, tol):
    """변환 PNG vs GT 덤프로 전역 mIoU 재계산 → 로그값과 ±tol 비교.

    저장된 PNG 는 RLE 해상도(예: 1024²)이고 우리 GT 덤프는 native(예: 1042²)일 수
    있다 — 해상도가 다르면 pred 를 GT 해상도로 최근접 리샘플해 같은 축에서 채점한다
    (기존 로직 그대로). 사용한 방식은 reproduce_method 로 summary 에 남긴다.
    """
    pred_index = common.index_label_pngs(pred_dir)
    gt_index = common.index_label_pngs(gt_dir)
    ids = sorted(set(pred_index) & set(gt_index))
    if not ids:
        print("[rle2png] ⚠️ 재현 검산 불가 — GT 와 공통 image_id 없음(경로 규약 확인).")
        return {"reproduced": None, "reproduce_note": "no common image_id"}
    hist = np.zeros((N, N), dtype=np.int64)
    resized = 0
    for image_id in ids:
        pred = common.load_label_png(pred_index[image_id])
        gt = common.load_label_png(gt_index[image_id])
        if pred.shape != gt.shape:      # GT 해상도가 다르면 pred 를 최근접 리샘플
            pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
            resized += 1
        hist += common.confusion_matrix(pred, gt, N, common.IGNORE_LABEL)
    _, miou = common.global_miou_from_cm(hist)
    delta = abs(miou - expected_miou)
    ok = delta <= tol
    note = (f"pred {resized}/{len(ids)} 장을 GT 해상도로 최근접 리샘플" if resized
            else "pred·GT 해상도 동일(리샘플 없음)")
    method = "pred→GT 최근접" if resized else "리샘플 없음"
    print(f"[rle2png] 재현 mIoU={miou:.4f} vs 로그 {expected_miou:.4f} "
          f"Δ={delta:.4f} → {'OK' if ok else 'MISMATCH'} ({note})")
    if resized:
        print("[rle2png] ⚠️ GT 해상도가 달라 리샘플했다 — 기준선 evaluator 의 GT "
              "읽기 방식(native 1042 vs 리사이즈)을 확인해 보고하라.")
    return {"reproduced": bool(ok), "reproduce_miou": miou,
            "reproduce_delta": round(delta, 4), "reproduce_note": note,
            "reproduce_method": method, "reproduce_n": len(ids)}


if __name__ == "__main__":
    main()
