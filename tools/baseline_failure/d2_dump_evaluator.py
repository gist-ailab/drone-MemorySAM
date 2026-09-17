#!/usr/bin/env python3
"""
D1 — DGFusion·CAFuser(detectron2/OneFormer) 예측 전수 덤프 evaluator.

detectron2 `SemSegEvaluator` 를 상속한 `DumpSemSegEvaluator` 를 제공한다. `process()`
에서 argmax·confidence 를 우리 덤프와 **같은 파일명·값 범위**로 저장하고, 부모
`process()` 도 호출해 공식 mIoU 가 그대로 나오게 한다. 저장 위치는 `<dump_dir>/{pred,conf,gt}/<image_id>.png`.

이 파일은 **기준선 저장소 루트에 복사해서** 쓴다(detectron2·OneFormer 가 그 env
에만 있으므로). 따라서 tools 패키지에 의존하지 않도록 image_id·저장 헬퍼를 자체
포함한다(로직은 common.py 와 동일 — smoke 가 동치성을 assert 한다).

⚠️ 규약 점검: 기준선 GT 가 우리와 같은 trainID(0~24 + 255)인지 `check_label_convention`
으로 실행 초기에 단언하라. detectron2 의 DELIVER 로더(create_deliver_gt_sem_seg_loading_fn)
는 raw 1~25 를 0~24 로 변환하므로 evaluator 가 보는 GT 는 이미 0~24 여야 한다.

detectron2 는 여기서 import 하지 않는다(지연 import) — tools 패키지 스모크가
detectron2 없이 이 모듈을 import 할 수 있어야 하기 때문이다.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np

KNOWN_CONDITIONS = ["cloud", "fog", "night", "rain", "sun"]
VALID_TRAINIDS = set(range(25)) | {255}
EXPECTED_COUNTS = {"val": 2005, "test": 1897}


# ---------------------------------------------------------------------------
# 자체 포함 헬퍼 (common.image_id_from_rel 와 동일 로직 — 기준선 repo 에 단독 복사 가능)
# ---------------------------------------------------------------------------
def _image_id_from_rel(rgb_path, dataset_root=None):
    """common.image_id_from_rel 와 동일 규약(데이터셋 루트 기준 상대 경로, 확장자 제외).

    우리 덤프와 기준선 덤프가 같은 RGB 경로에서 **같은 상대 경로 id** 를 내야 join 이
    성립한다. DELIVER 의 반복 basename 이 서로 덮어쓰지 않도록 폴더 구조를 보존한다.
    dataset_root(또는 환경변수 BF_DATASET_ROOT)를 주면 그 접두어를 잘라 상대 경로를
    만들고, 없으면 '/img/' 마커 이후('img/...' 포함)를 상대 경로로 삼는다.
    """
    p = str(rgb_path).replace("\\", "/")
    dataset_root = dataset_root or os.environ.get("BF_DATASET_ROOT")
    rel = None
    if dataset_root:
        dr = str(dataset_root).replace("\\", "/").rstrip("/")
        if p.startswith(dr + "/"):
            rel = p[len(dr) + 1:]
    if rel is None:
        if "/img/" in p:
            rel = "img/" + p.split("/img/", 1)[1]
        elif p.startswith("img/"):
            rel = p
        else:
            raise ValueError(
                f"데이터셋 루트를 특정할 수 없어 상대 경로 id 를 못 만든다: {p}. "
                f"BF_DATASET_ROOT 를 지정하거나 경로에 '/img/' 마커가 있어야 한다.")
    return re.sub(r"\.[^./]+$", "", rel)


# 하위 호환 별칭.
def _image_id_from_rgb_path(rgb_path):
    return _image_id_from_rel(rgb_path)


def _save_label_png(path, arr):
    from PIL import Image
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    Image.fromarray(np.asarray(arr).astype(np.uint8)).save(str(path))


def check_label_convention(gt_dir, loading_fn=None, sample=64):
    """gt_dir 의 라벨 PNG 규약을 판정한다.

    반환 dict: convention ∈ {trainid_0_24, labelid_1_25, unknown}, uniques,
    ok(=우리 규약과 같은가), remap(우리 규약으로 통일하는 함수 또는 None).
    loading_fn 을 주면 그 함수로 로드한 뒤(=evaluator 가 보는 형태) 판정한다.
    """
    from PIL import Image
    files = sorted(str(p) for p in Path(gt_dir).glob("**/*.png"))[:sample]
    if not files:
        raise FileNotFoundError(f"라벨 PNG 를 찾지 못함: {gt_dir}")
    uniques = set()
    for f in files:
        if loading_fn is not None:
            arr = np.asarray(loading_fn(f))
        else:
            arr = np.array(Image.open(f))
            if arr.ndim == 3:
                arr = arr[..., 0]
        uniques.update(np.unique(arr).tolist())

    non_ignore = sorted(v for v in uniques if v != 255)
    lo = min(non_ignore) if non_ignore else 0
    hi = max(non_ignore) if non_ignore else 0

    if uniques <= VALID_TRAINIDS:
        conv, ok, remap = "trainid_0_24", True, None
    elif lo >= 1 and hi <= 25:
        conv, ok = "labelid_1_25", False

        def remap(a):
            a = np.asarray(a).copy()
            m = a != 255
            a[m] = a[m] - 1
            a[a < 0] = 255
            return a
    else:
        conv, ok, remap = "unknown", False, None

    report = {"convention": conv, "uniques": sorted(uniques), "ok": ok,
              "remap": remap, "n_sampled": len(files)}
    print(f"[check_label_convention] {gt_dir}\n  convention={conv} ok={ok} "
          f"uniques={sorted(uniques)[:30]}")
    if not ok:
        print("  ⚠️ 우리 규약(0~24+255)과 다르다 — remap 으로 통일해 저장하라." if remap
              else "  ⚠️ 알 수 없는 규약 — 수동 확인 필요.")
    return report


# ---------------------------------------------------------------------------
# DumpSemSegEvaluator — 지연 정의(detectron2 필요)
# ---------------------------------------------------------------------------
def make_dump_evaluator_class():
    """detectron2 `SemSegEvaluator` 를 상속한 DumpSemSegEvaluator 를 만들어 준다.

    detectron2 import 를 함수 안에 두어 tools 패키지 스모크(detectron2 부재)에서도
    이 모듈 자체는 import 가능하게 한다.
    """
    from detectron2.evaluation import SemSegEvaluator  # noqa: WPS433
    import torch

    class DumpSemSegEvaluator(SemSegEvaluator):
        """공식 SemSegEvaluator + argmax/confidence PNG 덤프.

        BF_DUMP_DIR 이 있을 때만 build_evaluator 에서 생성되며, process() 는 부모를
        그대로 호출해 공식 mIoU 를 보존한 뒤 예측을 추가로 저장한다.
        """

        def __init__(self, *args, dump_dir=None, **kwargs):
            super().__init__(*args, **kwargs)
            dump_dir = dump_dir or os.environ.get("BF_DUMP_DIR")
            assert dump_dir, "dump_dir(또는 BF_DUMP_DIR) 필요"
            self._bf_root = Path(dump_dir)
            self._bf_pred = self._bf_root / "pred"
            self._bf_conf = self._bf_root / "conf"
            self._bf_gt = self._bf_root / "gt"
            for d in (self._bf_pred, self._bf_conf, self._bf_gt):
                d.mkdir(parents=True, exist_ok=True)
            self._bf_saved = 0

        def process(self, inputs, outputs):
            # 공식 경로를 먼저 그대로 실행(mIoU 누적은 여기서만).
            super().process(inputs, outputs)
            for inp, out in zip(inputs, outputs):
                fname = inp["file_name"]
                image_id = _image_id_from_rel(fname)
                sem = out["sem_seg"]                        # [C, H, W] 원본 크기
                if not torch.is_tensor(sem):
                    sem = torch.as_tensor(np.asarray(sem))
                sem = sem.float()
                # OneFormer 의 sem_seg 는 이미 softmax 확률일 수 있다. 확률이면(음수 없고
                # 채널합≈1) 그대로 confidence 로 쓰고, 로짓이면 softmax 한다. argmax 는 무관.
                col_sum = sem.sum(dim=0)
                is_prob = bool((sem.min() >= -1e-4) and
                               torch.allclose(col_sum, torch.ones_like(col_sum), atol=1e-2))
                probs = sem if is_prob else torch.softmax(sem, dim=0)
                pred = probs.argmax(dim=0).to("cpu").numpy().astype(np.uint8)
                conf = (probs.max(dim=0).values.clamp(0, 1).to("cpu").numpy()
                        * 255.0).round().astype(np.uint8)
                _save_label_png(self._bf_pred / f"{image_id}.png", pred)
                _save_label_png(self._bf_conf / f"{image_id}.png", conf)

                # GT 도 같은 규약으로 저장(교차 채점용). 부모의 로더·매핑을 재사용.
                gt_file = getattr(self, "input_file_to_gt_file", {}).get(fname)
                if gt_file is not None and getattr(self, "sem_seg_loading_fn", None):
                    gt = np.asarray(self.sem_seg_loading_fn(
                        gt_file, dtype=np.int64)).astype(np.uint8)
                    _save_label_png(self._bf_gt / f"{image_id}.png", gt)
                self._bf_saved += 1

        def evaluate(self):
            res = super().evaluate()
            print(f"[DumpSemSegEvaluator] 덤프 {self._bf_saved} 장 -> {self._bf_root}")
            # 장수 검증 — 상대 경로 보존 저장이 성공했는지(반복 basename 덮어쓰기 없음).
            # split 은 BF_SPLIT 로 지정(val|test). 미지정이면 root 이름에서 추정.
            split = os.environ.get("BF_SPLIT") or self._bf_root.name
            exp = EXPECTED_COUNTS.get(split)
            if exp is not None:
                assert self._bf_saved == exp, (
                    f"장수 불일치: split='{split}' 에 {self._bf_saved}장 저장(기대 {exp}). "
                    f"평탄화 덮어쓰기·데이터셋 등록 오류 의심 — 상대 경로 보존 확인.")
                print(f"[DumpSemSegEvaluator] 장수 검증 OK ({self._bf_saved}=={exp}).")
            else:
                print(f"[DumpSemSegEvaluator] ⚠️ split='{split}' 기대 장수 미정 — "
                      f"BF_SPLIT 을 val|test 로 지정하면 장수 검증을 한다.")
            return res

    return DumpSemSegEvaluator


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-gt", metavar="DIR",
                    help="라벨 PNG 규약 점검만 실행하고 종료")
    ap.add_argument("--sample", type=int, default=64)
    args = ap.parse_args()
    if args.check_gt:
        rep = check_label_convention(args.check_gt, sample=args.sample)
        raise SystemExit(0 if rep["ok"] else 3)
    ap.print_help()


if __name__ == "__main__":
    main()
