#!/usr/bin/env python3
"""
D4 — DGFusion 내부 기제 프로브(기준선 저장소 전용). forward hook 으로 다음을
이미지별로 잰다:
  (a) depth 보조 헤드 출력 vs GT depth 의 AbsRel·d1
  (b) 로컬 depth 토큰·전역 조건 토큰 텐서의 평균/분산
  (c) 각 레벨 cross-attention 의 softmax 가중을 K/V 모달 구간별로 합산(RGB 쿼리 기준
      모달별 가중)
  (d) --zero-modal 지정 시 해당 모달 입력을 0 으로 채운 추론에서 (a) 의 AbsRel·delta1
      와 그 이미지의 분할 mIoU 를 같은 CSV 행에 기록(모달 zero-out 기제 측정).
BF_ZERO_DEPTH_TOKEN=1 이면 depth 토큰을 0 으로 치환한 추론과 대조한다.

depth 정답은 DELIVER `depth/` 원본 depth. depth 헤드가 로그 스케일로 학습되었는지는
config `MODEL.DEPTH_HEAD.LOSS.LOG_SCALE` 를 읽어 처리한다(참이면 exp 로 되돌려 비교).
depth 가 0 인 픽셀(무효)은 AbsRel·delta1 계산에서 제외한다.

⚠️ 모듈 이름은 저장소마다 다르다. 정규식으로 모듈을 찾고 **못 찾으면 명확한 에러로
멈춘다**(추측으로 아무 층이나 잡지 않는다). 첫 실행에서 `--list-modules` 로 이름을
확인한 뒤 정규식을 맞춰라. 우리 쪽 대응 측정은 tools/module_diagnostics.py 참조.

detectron2 는 지연 import — tools 패키지 스모크(detectron2 부재)에서도 이 모듈은
import·인자 파싱이 되어야 한다.
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402


def find_modules(model, pattern):
    """정규식에 이름이 매칭되는 (name, module) 목록. 없으면 RuntimeError."""
    rx = re.compile(pattern)
    hits = [(n, m) for n, m in model.named_modules() if n and rx.search(n)]
    if not hits:
        raise RuntimeError(
            f"정규식 '{pattern}' 에 맞는 모듈이 없다. --list-modules 로 실제 이름을 "
            f"확인하고 정규식을 맞춰라(추측 금지).")
    return hits


class ActivationProbe:
    """이름 정규식으로 고른 모듈에 forward hook 을 걸어 출력 텐서를 모은다."""

    def __init__(self, model):
        self.model = model
        self._handles = []
        self.captured = {}

    def watch(self, pattern, tag):
        hits = find_modules(self.model, pattern)
        for name, mod in hits:
            def hook(_m, _i, out, _name=name, _tag=tag):
                self.captured.setdefault(_tag, {})[_name] = out
            self._handles.append(mod.register_forward_hook(hook))
        return [n for n, _ in hits]

    def clear(self):
        self.captured = {}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def _to_np(t):
    import torch
    if isinstance(t, (tuple, list)):
        t = t[0]
    if torch.is_tensor(t):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def tensor_stats(arr):
    a = np.asarray(arr, dtype=np.float64).ravel()
    return {"mean": float(a.mean()), "var": float(a.var()),
            "min": float(a.min()), "max": float(a.max())}


def depth_absrel_d1(pred_depth, gt_depth, mask=None):
    """AbsRel = mean(|p-g|/g), d1 = mean(max(p/g,g/p)<1.25). 유효 픽셀만.

    depth 가 0 인 픽셀(무효 depth)은 g>1e-3 조건으로 제외된다.
    """
    p = np.asarray(pred_depth, dtype=np.float64).ravel()
    g = np.asarray(gt_depth, dtype=np.float64).ravel()
    valid = np.isfinite(p) & np.isfinite(g) & (g > 1e-3)
    if mask is not None:
        valid &= np.asarray(mask).ravel().astype(bool)
    if not valid.any():
        return float("nan"), float("nan")
    p, g = p[valid], g[valid]
    absrel = float(np.mean(np.abs(p - g) / g))
    ratio = np.maximum(p / g, g / p)
    d1 = float(np.mean(ratio < 1.25))
    return absrel, d1


# ---------------------------------------------------------------------------
# A4 — 모달 zero-out + GT 기반 depth/분할 채점
# ---------------------------------------------------------------------------
MODAL_KEYS = ("CAMERA", "LIDAR", "EVENT", "DEPTH")


def zero_modal_in_batch(batch, modal):
    """batched_inputs(list[dict]) 안의 지정 모달 입력 텐서를 0 으로 채운다(in-place).

    모델 입력 dict 의 모달 키는 CAMERA·LIDAR·EVENT·DEPTH 이며 주 모달(RGB)은 image
    키에도 중복 저장된다. modal=CAMERA 면 image 키까지 함께 0 으로 채운다(두 키가
    같은 텐서를 공유하지 않을 수 있으므로 둘 다 처리).
    """
    import torch
    if modal not in MODAL_KEYS:
        raise ValueError(f"modal 은 {MODAL_KEYS} 중 하나여야 한다: {modal!r}")
    keys = {modal}
    if modal == "CAMERA":
        keys.add("image")
    for d in batch:
        if isinstance(d, dict):
            for k in keys:
                v = d.get(k)
                if torch.is_tensor(v):
                    d[k] = torch.zeros_like(v)
    return batch


def depth_pred_2d(out):
    """depth 헤드 출력 → (H, W) float 배열. 싱글톤 차원만 벗긴다(의미 추측 금지).

    dict 이거나 채널/배치 차원이 1 이 아니면 어느 축이 depth 인지 알 수 없으므로
    명확한 에러로 멈춘다.
    """
    import torch
    t = out
    if isinstance(t, dict):
        raise RuntimeError(
            f"depth 헤드 출력이 dict 다(키: {sorted(t)}) — 출력 구조에 맞게 도구를 "
            f"확장해야 한다(키를 추측해 고르지 않는다).")
    if isinstance(t, (tuple, list)):
        if len(t) != 1:
            raise RuntimeError(f"depth 헤드 출력이 다중 원소({len(t)})다 — 구조 확인 필요.")
        t = t[0]
    if not torch.is_tensor(t):
        raise RuntimeError(f"depth 헤드 출력이 tensor 가 아니다: {type(t)}")
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4 and a.shape[0] == 1 and a.shape[1] == 1:
        a = a[0, 0]
    elif a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise RuntimeError(f"depth 헤드 출력을 (H, W) 로 해석할 수 없다: shape={a.shape}")
    return a


def resize_nearest_float(arr, out_h, out_w):
    """float 맵(depth 등)용 최근접 리사이즈(common.resize_nearest 는 uint8 로 캐스팅한다)."""
    if arr.shape[0] == out_h and arr.shape[1] == out_w:
        return arr
    from PIL import Image
    return np.array(Image.fromarray(arr.astype(np.float32))
                    .resize((out_w, out_h), Image.NEAREST), dtype=np.float64)


def load_depth_gt(path):
    """DELIVER `depth/` 원본 depth PNG → (H, W) float64."""
    from PIL import Image
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"depth GT 를 찾을 수 없다: {p} — --deliver-root 확인")
    return np.array(Image.open(p)).astype(np.float64)


def deliver_gt_paths(file_name, deliver_root=None):
    """RGB file_name → (depth GT 경로, semantic GT 경로). deliver.py 경로 치환 규약.

    `<root>/img/<cond>/<split>/<scene>/<stem>_rgb_front.png` 의 img→depth·semantic,
    _rgb→_depth·_semantic 치환. deliver_root 를 주면 그 기준 상대 경로로 만들고,
    못 주면 file_name 의 '/img/' 마커에서 루트를 유도한다(둘 다 아니면 명확한 에러).
    """
    p = str(file_name).replace("\\", "/")
    if deliver_root:
        image_id = common.image_id_from_rel(p, dataset_root=str(deliver_root))
        root = Path(deliver_root)
    else:
        if "/img/" not in p:
            raise ValueError(
                f"file_name 에 '/img/' 마커가 없어 GT 경로를 유도할 수 없다: {p} — "
                f"--deliver-root 를 지정하라.")
        image_id = common.image_id_from_rel(p)
        root = Path(p[: p.index("/img/")])
    if not image_id.startswith("img/"):
        raise ValueError(f"image_id 가 img/ 접두어로 시작하지 않는다: {image_id}")
    rest = image_id[len("img/"):]
    depth_rel = ("depth/" + rest).replace("_rgb", "_depth")
    sem_rel = ("semantic/" + rest).replace("_rgb", "_semantic")
    return root / f"{depth_rel}.png", root / f"{sem_rel}.png"


def seg_miou_from_output(output, sem_gt_path):
    """detectron2 출력(list[dict] 의 'sem_seg') × DELIVER semantic GT → 이미지별 mIoU(%).

    sem_seg 는 [C,H,W](원본 크기, trainID 0~24)로 가정해 argmax 하고, GT 는
    common.load_gt_deliver(raw 1~25 → trainID)로 읽는다. GT 에 없는 클래스는 NaN
    제외 평균(common 규약).
    """
    import torch
    if not isinstance(output, (list, tuple)) or not output:
        raise RuntimeError(f"모델 출력이 list[dict] 가 아니다: {type(output)}")
    sem = output[0].get("sem_seg") if isinstance(output[0], dict) else None
    if sem is None:
        raise RuntimeError("모델 출력에 'sem_seg' 키가 없다 — 분할 mIoU 를 계산할 수 없다.")
    if torch.is_tensor(sem):
        sem = sem.float().cpu().numpy()
    sem = np.asarray(sem)
    if sem.ndim == 3:
        pred = sem.argmax(0).astype(np.uint8)
    elif sem.ndim == 2:
        pred = sem.astype(np.uint8)
    else:
        raise RuntimeError(f"sem_seg shape 해석 불가: {sem.shape}")
    gt = common.load_gt_deliver(sem_gt_path)
    if pred.shape != gt.shape:
        pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
    cm = common.confusion_matrix(pred, gt, common.N_CLASSES, common.IGNORE_LABEL)
    return common.nanmean(common.per_image_iou(cm)) * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", help="DGFusion detectron2 config")
    ap.add_argument("--weights", help="체크포인트 .pth")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", help="이미지별 CSV 출력 경로")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--list-modules", action="store_true",
                    help="모델 모듈 이름을 출력하고 종료(정규식 잡을 때 사용)")
    ap.add_argument("--depth-head-regex", default=r"depth.*head|head.*depth")
    ap.add_argument("--depth-token-regex", default=r"depth.*token|depth.*embed")
    ap.add_argument("--cond-token-regex", default=r"cond.*token|condition.*embed")
    ap.add_argument("--xattn-regex", default=r"cross.*attn|fusion.*attn")
    ap.add_argument("--zero-modal", default="none",
                    choices=["none"] + list(MODAL_KEYS),
                    help="지정 모달의 입력 텐서를 0 으로 채운 상태로 추론(A4). "
                         "CAMERA 면 주 모달 중복 키 image 까지 함께 0 으로 채운다.")
    ap.add_argument("--deliver-root", default=None,
                    help="DELIVER 데이터셋 루트(GT depth/semantic 위치). 못 주면 "
                         "file_name 의 '/img/' 마커에서 유도한다.")
    args = ap.parse_args()

    if not args.config_file or not args.weights:
        ap.error("--config-file 과 --weights 는 실행에 필요(스모크는 인자 파싱만 검사)")

    # 지연 import — 기준선 repo 에서만 사용 가능.
    import torch  # noqa: F401
    from detectron2.checkpoint import DetectionCheckpointer
    from train_net import Trainer, setup  # 기준선 repo 루트에서 실행

    class _A:
        config_file = args.config_file
        opts = ["MODEL.WEIGHTS", args.weights, "MODEL.IS_TRAIN", "False"]
        eval_only = True
        inference_only = False
        resume = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "auto"

    cfg = setup(_A)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(args.weights, resume=False)
    model.eval()

    if args.list_modules:
        for n, _m in model.named_modules():
            if n:
                print(n)
        return

    probe = ActivationProbe(model)
    depth_names = probe.watch(args.depth_head_regex, "depth_head")
    dtok_names = probe.watch(args.depth_token_regex, "depth_token")
    ctok_names = probe.watch(args.cond_token_regex, "cond_token")
    xattn_names = probe.watch(args.xattn_regex, "xattn")
    print(f"[probe_dgfusion] depth_head={depth_names} depth_token={dtok_names}\n"
          f"  cond_token={ctok_names} xattn={xattn_names}")
    if os.environ.get("BF_ZERO_DEPTH_TOKEN") == "1":
        print("[probe_dgfusion] BF_ZERO_DEPTH_TOKEN=1 — depth 토큰 0 치환 모드")
    if args.zero_modal != "none":
        print(f"[probe_dgfusion] --zero-modal {args.zero_modal} — 해당 모달 입력을 0 으로 "
              f"채워 추론(A4). CAMERA 면 image 키까지 함께 0 으로 채운다.")

    # depth 헤드의 로그 스케일 여부 — config 에서 읽는다(없으면 선형으로 간주하고 경고).
    try:
        log_scale = bool(cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE)
    except AttributeError:
        log_scale = False
        print("[probe_dgfusion] ⚠️ cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE 키가 없다 — "
              "False(선형 스케일)로 간주한다. 실제 로그 스케일 학습이면 config 를 확인하라.")
    print(f"[probe_dgfusion] depth 헤드 로그 스케일 = {log_scale}"
          + (" (exp 로 되돌려 비교)" if log_scale else ""))

    from detectron2.data import build_detection_test_loader
    ds_name = cfg.DATASETS.TEST_SEMANTIC[0] if hasattr(cfg.DATASETS, "TEST_SEMANTIC") else cfg.DATASETS.TEST[0]
    loader = build_detection_test_loader(cfg, ds_name)

    rows = []
    for i, batch in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        probe.clear()
        if args.zero_modal != "none":
            zero_modal_in_batch(batch, args.zero_modal)
        with torch.no_grad():
            outputs = model(batch)
        file_name = batch[0].get("file_name", "")
        rec = {"idx": i, "file_name": file_name, "zero_modal": args.zero_modal}
        for tag in ("depth_token", "cond_token"):
            caps = probe.captured.get(tag, {})
            if caps:
                st = tensor_stats(_to_np(next(iter(caps.values()))))
                rec[f"{tag}_mean"], rec[f"{tag}_var"] = st["mean"], st["var"]
        # (a) depth 헤드 AbsRel·delta1 + (2) 분할 mIoU — DELIVER GT 원본으로 채점.
        depth_gt_path, sem_gt_path = deliver_gt_paths(file_name, args.deliver_root)
        depth_caps = probe.captured.get("depth_head", {})
        if depth_caps:
            pred_d = depth_pred_2d(next(iter(depth_caps.values())))
            if log_scale:
                pred_d = np.exp(pred_d)
            gt_d = load_depth_gt(depth_gt_path)
            if pred_d.shape != gt_d.shape:
                pred_d = resize_nearest_float(pred_d, gt_d.shape[0], gt_d.shape[1])
            rec["depth_absrel"], rec["depth_d1"] = depth_absrel_d1(pred_d, gt_d)
        rec["seg_miou_img"] = seg_miou_from_output(outputs, sem_gt_path)
        rows.append(rec)

    probe.remove()
    if args.out and rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"[probe_dgfusion] {len(rows)} 행 -> {args.out}")


if __name__ == "__main__":
    main()
