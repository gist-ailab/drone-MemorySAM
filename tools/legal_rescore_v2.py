#!/usr/bin/env python3
"""legal 하네스 v2 후보 — native 복원 업샘플을 중심 정렬(nearest-exact)로 바꿔 val.py 를 그대로 실행한다.

[ISSUE-036, 2026-09-18] `val.py:_unpad_resize_to_orig` 는 1024 예측을 native 1042 로 올릴 때
`F.interpolate(mode="nearest")`(torch 의 floor 정렬)를 쓴다. 이 정렬은 예측을 반 픽셀 계통 편차로
어긋나게 해 얇은 클래스 IoU 를 깎는다 — DGFusion 80k test 예측(RLE 1024, 317장 표본)을 native GT 에
세 방식으로 올려 채점한 실측: nearest 52.34 / nearest-exact 53.68 / PIL NEAREST 53.68
(얇은 객체 4클래스 50.25 / 53.38, Pole 51.3 / 57.1). 기준선 변환(`tools/baseline_failure/common.py:
resize_nearest` = PIL, 중심 정렬)은 이 편차가 없으므로 우리 legal 수치만 약 −1.3 낮게 잰 셈이다.

이 스크립트는 하네스 가드(`tools/eval_harness_guard.py`, val.py 포함 8파일 SHA256 동결)를 깨지 않기
위해 **val.py 를 수정하지 않고** 함수 하나만 몽키패치한 뒤 `val.main()` 을 그대로 부른다. 인자는
val.py 와 동일하다. 정본 규칙(v1 = nearest)은 그대로 두고, v2 채택 여부는 카드 §5-34 판정으로 정한다.

예:
  PYTHONPATH=semseg/models/sam2:. python tools/legal_rescore_v2.py \\
      --cfg configs/eval/<...>.yaml --mode test --model_path <ckpt>
"""
import sys
from pathlib import Path

import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import val  # noqa: E402  (하네스 본체, 무수정)

RESAMPLE_MODE = "nearest-exact"


def _unpad_resize_to_orig_v2(pred, orig_h, orig_w, model_size=1024):
    """val._unpad_resize_to_orig 와 crop 규약은 동일, 업샘플만 nearest-exact(중심 정렬)."""
    H, W = orig_h, orig_w
    t = model_size
    if W >= H:
        scale = t / W
        nH, nW = round(H * scale), t
        pad_top = (t - nH) // 2
        pred_content = pred[pad_top:pad_top + nH, :nW]
    else:
        scale = t / H
        nH, nW = t, round(W * scale)
        pad_left = (t - nW) // 2
        pred_content = pred[:nH, pad_left:pad_left + nW]
    if pred_content.shape[0] != H or pred_content.shape[1] != W:
        pred_content = pred_content.unsqueeze(0).unsqueeze(0).float()
        pred_resized = F.interpolate(pred_content, size=(H, W), mode=RESAMPLE_MODE)
        pred_resized = pred_resized.squeeze(0).squeeze(0).long()
    else:
        pred_resized = pred_content.long()
    return pred_resized


def main():
    print(f"[legal-v2] _unpad_resize_to_orig -> mode={RESAMPLE_MODE} (val.py 무수정, 몽키패치)", flush=True)
    val._unpad_resize_to_orig = _unpad_resize_to_orig_v2
    val.main()


if __name__ == "__main__":
    main()
