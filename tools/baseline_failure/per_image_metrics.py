#!/usr/bin/env python3
"""
D2 — 이미지별 지표 CSV + 모델 간 join.

각 모델의 예측 덤프(`<dir>/<image_id>.png`)와 공통 GT 덤프를 대조해 이미지별
per-class IoU CSV 를 만들고, 전체 혼동행렬 합에서 mIoU 를 재계산해 각 모델의
summary.json 과 대조한다(불일치 시 경고 + exit≠0). 여러 모델을 image_id 로 join 해
모델별 mIoU·ΔmIoU·얇은 객체/큰 영역 평균 IoU 표(joined_<split>.csv)도 쓴다.

예:
  python tools/baseline_failure/per_image_metrics.py \
    --gt   .../ours/test/gt \
    --split test --out .../mining_in \
    --models ours=.../ours/test/pred \
             dgf80k=.../dgf80k/test/pred \
             dgffinal=.../dgffinal/test/pred \
             caf=.../caf/test/pred

채점 프로토콜(--gt_protocol): 기준선(DGFusion·CAFuser)은 1024²로 리사이즈해 채점하는
반면 우리 덤프는 native 1042² GT 로 채점한다. `resized1024` 를 주면 GT·예측을 모두
1024² 최근접으로 맞춰 같은 축에서 재채점한다(파일명에 프로토콜이 붙어 덮어쓰지 않는다).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402

N = common.N_CLASSES
CLASSES = common.CLASSES

GT_PROTOCOLS = ("native", "resized1024")

# resized1024 프로토콜의 한계 — 요약 JSON note 필드와 함수 docstring 에 그대로 남긴다.
RESIZED1024_NOTE = (
    "1042→1024 는 정수배 축소가 아니므로, 원본 해상도(예: 1042)로 복원 저장된 예측 PNG 를 "
    "다시 1024x1024 로 줄이면 기준선 추론 시점의 원래 1024 예측과 픽셀 단위로 완전히 같지는 "
    "않다(근사 재현). 임의 보정 없이 GT·예측을 같은 규칙(최근접)으로 줄여 채점한다.")
NATIVE_NOTE = "GT 덤프 원 해상도 그대로 채점 — 기존 프로토콜(덤프·summary.json 검산)과 동일."


def _resolve_pred_dir(path):
    """모델 인자가 pred 디렉터리 자체이거나 split 루트일 수 있으니 정규화한다."""
    p = Path(path)
    if (p / "pred").is_dir():
        return p / "pred"
    return p


def _find_summary(pred_dir):
    """pred 디렉터리 기준으로 summary.json 을 찾는다(<split>/summary.json)."""
    cand = pred_dir.parent / "summary.json"
    return cand if cand.exists() else None


def score_model(name, pred_dir, gt_dir, split, out_dir, tol=0.05, gt_protocol="native"):
    """한 모델의 이미지별 CSV 를 쓰고, 전역 mIoU 재계산·summary 대조 결과를 담아
    per-image 요약 dict(image_id→{condition,case,miou_img,thin,large,iou_vec})를 반환.

    image_id = 각 디렉터리 기준 **상대 경로**(중첩) — common.index_label_pngs 로
    재귀 인덱싱한다. 평탄 파일명 중복(1차 덤프 결함)을 만나면 그 함수가 에러로 멈춘다.

    gt_protocol='native'(기본) = GT 원 해상도 그대로 채점(기존 동작). 'resized1024' 는
    GT 와 예측을 모두 1024x1024 최근접으로 맞춘 뒤 채점한다. ⚠️ 1042→1024 는 정수배가
    아니므로, 이미 1042 로 복원 저장된 예측 PNG 를 다시 1024 로 줄이면 기준선 추론 시점의
    원래 1024 예측과 픽셀 단위로 완전히 같지는 않다 — 근사이며 임의 보정은 하지 않는다
    (요약 JSON 의 note 필드에도 같은 문구를 남긴다). 출력 CSV 는 프로토콜별 파일명
    (native 는 기존 이름 그대로, resized1024 는 접미사)이라 서로 덮어쓰지 않는다.
    """
    if gt_protocol not in GT_PROTOCOLS:
        raise ValueError(f"알 수 없는 gt_protocol: {gt_protocol!r} ({GT_PROTOCOLS} 중 하나)")
    pred_dir = _resolve_pred_dir(pred_dir)
    gt_index = common.index_label_pngs(gt_dir, require_nested=True)
    pred_index = common.index_label_pngs(pred_dir, require_nested=True)
    gt_ids = set(gt_index)
    pred_ids = set(pred_index)
    ids = sorted(gt_ids & pred_ids)
    missing = gt_ids - pred_ids
    if missing:
        print(f"[{name}] ⚠️ GT 에 있는데 예측 없는 이미지 {len(missing)} 장 "
              f"(예: {sorted(missing)[:3]})")
    if not ids:
        raise RuntimeError(f"[{name}] GT 와 예측의 공통 image_id 가 없다 "
                           f"(gt={len(gt_ids)} pred={len(pred_ids)}). 상대 경로 규약이 "
                           f"어긋났을 수 있다 — 두 덤프가 같은 image_id 를 쓰는지 확인.")
    # 공통 비율이 낮으면 규약 어긋남(평탄 vs 중첩)을 크게 경고.
    smaller = min(len(gt_ids), len(pred_ids))
    if smaller and len(ids) < 0.5 * smaller:
        print(f"[{name}] ⚠️ 공통 image_id 가 {len(ids)}/{smaller} 로 과소 — "
              f"평탄 덤프와 중첩 GT 처럼 규약이 어긋났을 가능성이 크다.")

    csv_path = Path(out_dir) / f"per_image_{name}_{split}{_protocol_suffix(gt_protocol)}.csv"
    header = (["image_id", "condition", "case"]
              + [f"IoU_{i}" for i in range(N)]
              + ["mIoU_img", "pixel_acc"]
              + [f"px_{i}" for i in range(N)])

    global_hist = np.zeros((N, N), dtype=np.int64)
    per_image = {}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for image_id in ids:
            gt = common.load_label_png(gt_index[image_id])
            pred = common.load_label_png(pred_index[image_id])
            if gt_protocol == "resized1024":
                # GT·예측을 모두 1024x1024 최근접으로 맞춘 뒤 채점(RESIZED1024_NOTE 참조).
                gt = common.resize_nearest(gt, 1024, 1024)
                pred = common.resize_nearest(pred, 1024, 1024)
            elif pred.shape != gt.shape:
                pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
            cm = common.confusion_matrix(pred, gt, N, common.IGNORE_LABEL)
            global_hist += cm
            iou = common.per_image_iou(cm)
            miou_img = common.nanmean(iou)
            pacc = common.pixel_acc_from_cm(cm)
            px = cm.sum(1).astype(np.int64)          # GT 클래스별 픽셀 수
            cond, case = common.parse_condition_case(image_id)
            row = ([image_id, cond, case]
                   + [("" if np.isnan(v) else round(float(v), 6)) for v in iou]
                   + [round(miou_img, 6), round(pacc, 6)]
                   + px.tolist())
            w.writerow(row)
            per_image[image_id] = {
                "condition": cond, "case": case,
                "miou_img": miou_img, "iou": iou,
                "thin": common.nanmean(iou[common.THIN_CLASS_IDS]),
                "large": common.nanmean(iou[common.LARGE_CLASS_IDS]),
            }

    _, miou = common.global_miou_from_cm(global_hist)
    print(f"[{name}] gt_protocol={gt_protocol}  {len(ids)} imgs  "
          f"전역 재계산 mIoU={miou:.2f}  -> {csv_path.name}")

    ok = True
    check_status = "no_summary"
    summ = _find_summary(pred_dir)
    if gt_protocol != "native":
        # summary.json 의 mIoU 는 native 프로토콜로 계산된 값 — 프로토콜이 다르면
        # 비교 자체가 성립하지 않으므로 대조를 건너뛴다(결과 무효가 아님).
        print(f"[{name}] gt_protocol={gt_protocol} — summary.json 대조 생략(프로토콜 상이)")
        check_status = "skipped_protocol"
        return per_image, miou, ok, check_status
    if summ is not None:
        ref = json.loads(summ.read_text(encoding="utf-8")).get("mIoU")
        if ref is not None:
            delta = abs(miou - float(ref))
            tag = "OK" if delta <= tol else "MISMATCH"
            print(f"[{name}] summary.json mIoU={ref:.2f} Δ={delta:.3f} → {tag}")
            ok = delta <= tol
            check_status = "ok" if ok else "mismatch"
    else:
        print(f"[{name}] summary.json 없음 — 전역 mIoU 대조 생략")

    return per_image, miou, ok, check_status


def _protocol_suffix(gt_protocol):
    """파일명 접미사 — native 는 기존 이름을 그대로 쓰고(하위 호환), 그 외는 붙인다."""
    return "" if gt_protocol == "native" else f"_{gt_protocol}"


def write_join(models_pi, split, out_dir, gt_protocol="native"):
    """공통 image_id 로 모델별 mIoU·ΔmIoU·얇은/큰 영역 평균 IoU join CSV 를 쓴다."""
    names = list(models_pi.keys())
    common_ids = None
    for pi in models_pi.values():
        s = set(pi.keys())
        common_ids = s if common_ids is None else (common_ids & s)
    common_ids = sorted(common_ids or [])
    ref = "ours" if "ours" in names else names[0]
    others = [n for n in names if n != ref]

    header = ["image_id", "condition", "case"]
    header += [f"mIoU_{n}" for n in names]
    header += [f"dmIoU_{ref}_minus_{n}" for n in others]
    header += [f"thin_{n}" for n in names] + [f"large_{n}" for n in names]

    join_path = Path(out_dir) / f"joined_{split}{_protocol_suffix(gt_protocol)}.csv"
    with open(join_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for image_id in common_ids:
            base = models_pi[ref][image_id]
            row = [image_id, base["condition"], base["case"]]
            row += [round(models_pi[n][image_id]["miou_img"], 6) for n in names]
            row += [round(models_pi[ref][image_id]["miou_img"]
                          - models_pi[n][image_id]["miou_img"], 6) for n in others]
            row += [round(models_pi[n][image_id]["thin"], 6)
                    if not np.isnan(models_pi[n][image_id]["thin"]) else ""
                    for n in names]
            row += [round(models_pi[n][image_id]["large"], 6)
                    if not np.isnan(models_pi[n][image_id]["large"]) else ""
                    for n in names]
            w.writerow(row)
    print(f"[join] {len(common_ids)} 공통 이미지  기준모델={ref}  -> {join_path.name}")
    return join_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="공통 GT trainID PNG 디렉터리")
    ap.add_argument("--models", required=True, nargs="+", metavar="NAME=DIR",
                    help="모델별 예측 디렉터리. 예: ours=.../pred dgf80k=.../pred")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--gt_protocol", default="native", choices=list(GT_PROTOCOLS),
                    help="채점 GT 프로토콜: native=GT 원 해상도(기존 동작), "
                         "resized1024=GT·예측을 모두 1024x1024 최근접으로 맞춰 채점")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for spec in args.models:
        if "=" not in spec:
            ap.error(f"--models 는 name=dir 형식이어야 함: {spec}")
        name, d = spec.split("=", 1)
        models.append((name, d))

    models_pi = {}
    models_stats = {}
    all_ok = True
    for name, d in models:
        pi, miou, ok, check = score_model(name, d, args.gt, args.split, out_dir,
                                          args.tol, gt_protocol=args.gt_protocol)
        models_pi[name] = pi
        models_stats[name] = {"global_miou": miou, "n_images": len(pi),
                              "summary_check": check}
        all_ok = all_ok and ok

    if len(models_pi) >= 2:
        write_join(models_pi, args.split, out_dir, gt_protocol=args.gt_protocol)
    else:
        print("[join] 모델이 1개뿐 — join 생략")

    # 요약 JSON — 사용한 프로토콜과 (resized1024 의) 한계를 남긴다.
    note = RESIZED1024_NOTE if args.gt_protocol == "resized1024" else NATIVE_NOTE
    summary = {"split": args.split, "gt_protocol": args.gt_protocol, "note": note,
               "models": models_stats}
    summary_path = out_dir / f"summary_per_image_{args.split}_{args.gt_protocol}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"[per_image_metrics] gt_protocol={args.gt_protocol} 요약 -> {summary_path.name}")

    if not all_ok:
        print("[per_image_metrics] ⚠️ summary.json 과 mIoU 불일치 — exit 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
