#!/usr/bin/env python3
"""
D4 — DGFusion 내부 기제 프로브(기준선 저장소 전용). forward hook 으로 다음을
이미지별로 잰다:
  (a) depth 보조 헤드 출력 vs GT depth 의 AbsRel·d1
  (b) 로컬 depth 토큰·전역 조건 토큰 텐서의 평균/분산
  (c) 각 레벨 cross-attention 의 softmax 가중을 K/V 모달 구간별로 합산(RGB 쿼리 기준
      모달별 가중)
  (d) --zero-modal 지정 시 해당 모달 입력을 채운 추론에서 (a) 의 AbsRel·delta1
      와 그 이미지의 분할 mIoU 를 같은 CSV 행에 기록(모달 zero-out 기제 측정).
      개입 규약(A7·A12) --zero-mode normalized(기본: 원본을 모델 버퍼 pixel_mean 의
      모달 슬라이스[3i:3i+3]로 채워 정규화 후 0 — 우리 modality_zero_ablation 과
      동일 규약) | raw(옛 동작: 원본 0 채움).
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
        # 정규식으로 "none" 을 주면 그 항목은 감시하지 않는다. 모델 계열에 따라 아예 없는
        # 구성요소(예: CAFuser 에는 depth 헤드가 없다)를 억지로 다른 모듈에 붙이지 않기
        # 위한 것이다. 없는 것을 감시하려다 멈추는 기본 동작은 그대로 둔다.
        if pattern == "none":
            return []
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
# A4 — 모달 zero-out + GT 기반 depth/분할 채점 · A7 — 개입 규약(normalized|raw)
# ---------------------------------------------------------------------------
MODAL_KEYS = ("CAMERA", "LIDAR", "EVENT", "DEPTH")


def _obj_attr_items(node):
    """단순 객체의 (속성, 값) 목록 — 인스턴스·클래스 속성(밑줄·콜러블 제외)."""
    d = {}
    for src in (getattr(type(node), "__dict__", {}) or {}, vars(node)):
        for k, v in src.items():
            if not k.startswith("_") and not callable(v):
                d[k] = v
    return list(d.items())


def _iter_cfg_items(node, prefix=""):
    """cfg(CfgNode·dict·속성 객체) 를 (dotted 경로, 값) 쌍으로 재귀 열거."""
    if isinstance(node, dict):
        items = list(node.items())
    elif hasattr(node, "keys") and hasattr(node, "__getitem__"):
        try:
            items = [(str(k), node[k]) for k in node.keys()]
        except Exception:
            items = []
    elif hasattr(node, "__dict__"):
        items = _obj_attr_items(node)
    else:
        yield prefix, node
        return
    for k, v in items:
        p = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, (bool, int, float, str, list, tuple)):
            yield p, v
        else:
            yield from _iter_cfg_items(v, p)


def _as_mean_tuple(val, path):
    """PIXEL_MEAN 후보 값 → (float, ...) 튜플. 숫자가 아니면 RuntimeError."""
    try:
        seq = val if isinstance(val, (list, tuple)) else [val]
        out = tuple(float(x) for x in seq)
    except (TypeError, ValueError):
        raise RuntimeError(f"PIXEL_MEAN 후보({path}) 값이 숫자가 아니다: {val!r}")
    if not out:
        raise RuntimeError(f"PIXEL_MEAN 후보({path}) 가 빈 값이다")
    return out


def _seg_has_pixel_mean(seg):
    """세그먼트에 PIXEL_MEAN 토큰쌍이 들어가는지 — PIXEL_MEAN·PIXEL_MEAN_LIDAR·
    LIDAR_PIXEL_MEAN 등 (PIXEL_MEANING 같은 우연 일치는 제외)."""
    toks = seg.split("_")
    return any(toks[i] == "PIXEL" and toks[i + 1] == "MEAN"
               for i in range(len(toks) - 1))


def _strip_pixel_mean_marker(seg):
    """세그먼트에서 PIXEL_MEAN 토큰쌍 하나를 떼어 낸 나머지(조인). 쌍이 없으면 None.

    LIDAR_PIXEL_MEAN→LIDAR, PIXEL_MEAN_LIDAR→LIDAR, PIXEL_MEAN→"", EVENT_CAMERA→None.
    """
    toks = seg.split("_")
    for i in range(len(toks) - 1):
        if toks[i] == "PIXEL" and toks[i + 1] == "MEAN":
            return "_".join(toks[:i] + toks[i + 2:])
    return None


def _last_seg_is_modal(last, modal):
    """A11 — 후보 판정: 경로의 마지막 세그먼트가 모달 이름과 **정확히** 일치하는가.

    세그먼트를 `_` 로 쪼갠 부분일치는 인정하지 않는다(DATASETS.PIXEL_MEAN.EVENT_CAMERA
    는 마지막 세그먼트가 EVENT_CAMERA 라 CAMERA 의 후보도 EVENT 의 후보도 아니다).
    예외 두 가지:
      (a) 무명평탄 키 — PIXEL_MEAN 마커를 떼면 정확히 모달 이름만 남는 세그먼트
          (LIDAR_PIXEL_MEAN 등). 마커 외 토큰이 모달 이름뿐이므로 다른 모달이 끼어들
          여지가 없다.
      (b) CAMERA(주 모달)만 마지막 세그먼트가 그냥 PIXEL_MEAN 인 경로(MODEL.PIXEL_MEAN).
    """
    if last == modal:
        return True
    if modal == "CAMERA" and last == "PIXEL_MEAN":
        return True
    return _strip_pixel_mean_marker(last) == modal


# A11 — 후보 우선순위: 데이터셋 특화(경로에 DELIVER) > 그 외 DATASETS.* > MODEL.* > 기타.
_MEAN_PRIO_NAMES = {3: "데이터셋 특화(DELIVER)", 2: "DATASETS.*", 1: "MODEL.*",
                    0: "기타"}


def _mean_cand_priority(up):
    """대문자 경로의 후보 우선순위(A11) — 클수록 구체적이다."""
    if "DELIVER" in up:
        return 3
    first = up.split(".")[0]
    if first == "DATASETS":
        return 2
    if first == "MODEL":
        return 1
    return 0


def find_modal_pixel_mean(cfg, modal):
    """cfg 에서 모달의 PIXEL_MEAN(채널별 평균) 을 찾아 (float, ...) 로 반환한다.

    키 이름을 추측하지 않는다: cfg 를 재귀 열거해 (1) 경로 어딘가에 PIXEL_MEAN 토큰쌍이
    있고 (2) 경로의 마지막 세그먼트가 모달 이름과 정확히 일치하는(_last_seg_is_modal,
    부분일치 부정) 항목만 후보로 쓴다. 후보가 여럿이면 우선순위(_mean_cand_priority)로
    하나를 고르고, 고른 후보와 값이 같은 다른 후보를 로그로 남긴다(A11). 같은
    우선순위 안에서 값이 서로 다르거나 후보가 아예 없으면 명확한 에러로 멈춘다
    (임의값 대입 금지). d2_zero_modality.patch 안의 복사본과 규칙이 같아야 한다
    (smoke_baseline_failure A11 이 같은 입력으로 동시 검증).
    """
    if modal not in MODAL_KEYS:
        raise ValueError(f"modal 은 {MODAL_KEYS} 중 하나여야 한다: {modal!r}")
    cands = []
    for path, val in _iter_cfg_items(cfg):
        up = path.upper()
        segs = up.split(".")
        if not any(_seg_has_pixel_mean(s) for s in segs):
            continue
        if not _last_seg_is_modal(segs[-1], modal):
            continue
        cands.append((path, _as_mean_tuple(val, path)))
    if not cands:
        raise RuntimeError(
            f"cfg 에서 {modal} 의 PIXEL_MEAN 을 찾지 못했다(임의값 대입 금지) — config "
            f"의 실제 키 이름을 확인한 뒤 다시 실행하라.")
    top = max(_mean_cand_priority(p.upper()) for p, _m in cands)
    tier = [(p, m) for p, m in cands if _mean_cand_priority(p.upper()) == top]
    means = {m for _p, m in tier}
    if len(means) != 1:
        raise RuntimeError(
            f"{modal} 의 PIXEL_MEAN 후보가 같은 우선순위({_MEAN_PRIO_NAMES[top]}) 안에서 "
            f"값이 달라 모호하다: {tier} — config 의 모달별 통계 정의를 확인하라(추측 금지).")
    chosen_path, mean = tier[0]
    same_value = [p for p, m in cands if m == mean and p != chosen_path]
    print(f"[probe_dgfusion] {modal} PIXEL_MEAN 후보 {len(cands)}개 — 채택 "
          f"{chosen_path}({_MEAN_PRIO_NAMES[top]}), 값이 같은 다른 후보 "
          f"{same_value or '없음'} -> {list(mean)}")
    return mean


def _fill_modal(v, mean):
    """텐서 v 를 채널별 평균 mean 으로 채운 새 텐서(정규화 후 0 이 되는 원본 값)."""
    import torch
    m = torch.as_tensor(mean, dtype=v.dtype, device=v.device).reshape(-1)
    if v.ndim == 3 and m.numel() == v.shape[0]:
        return torch.ones_like(v) * m.view(-1, 1, 1)
    if m.numel() == 1:
        return torch.full_like(v, float(m[0]))
    raise RuntimeError(
        f"입력 텐서 채널 수({tuple(v.shape)}) 와 PIXEL_MEAN 길이({m.numel()}) 가 맞지 "
        f"않아 평균 채움을 정의할 수 없다 — config 의 모달별 통계를 확인하라.")


def _cfg_get(node, dotted):
    """cfg(dict·CfgNode·속성 객체) 에서 점 경로의 값 읽기 — 없으면 KeyError·AttributeError."""
    cur = node
    for k in dotted.split("."):
        cur = cur[k] if isinstance(cur, dict) else getattr(cur, k)
    return cur


def modal_order_from_cfg(cfg):
    """A12 — 모델 pixel_mean 버퍼의 모달 순서를 cfg 에서 구한다.

    dgfusion.py:108-132 버퍼 구성 규칙 그대로: 주 모달(cfg.DATASETS.MODALITIES.
    MAIN_MODALITY)이 먼저 오고 나머지가 cfg.DATASETS.MODALITIES.ORDER 순으로 이어진다
    (모달 키 이름은 대문자로 정규화해 비교). 두 키가 없으면 명확한 에러(추측 금지).
    d2_zero_modality.patch 의 _bf_modal_order 와 규칙이 같아야 한다(smoke A7 이 같은
    입력으로 동시 검증).
    """
    try:
        order = _cfg_get(cfg, "DATASETS.MODALITIES.ORDER")
        main = _cfg_get(cfg, "DATASETS.MODALITIES.MAIN_MODALITY")
    except (AttributeError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"cfg.DATASETS.MODALITIES.ORDER·MAIN_MODALITY 를 읽을 수 없다({e!r}) — "
            f"모델 pixel_mean 버퍼의 모달 순서를 알 수 없다. config 를 확인하라(추측 금지).")
    order = [str(m).strip().upper() for m in order]
    main = str(main).strip().upper()
    return [main] + [m for m in order if m != main]


def model_modal_mean(model, cfg, modal):
    """A12 — 채움 값의 출처: 모델 버퍼 pixel_mean[3i:3i+3] 을 (인덱스 i, (float,...)) 로.

    모델은 모달별 3채널 평균·표준편차를 하나의 긴 버퍼로 이어 붙여 갖고(dgfusion.py:
    108-132), 정규화 (x - pixel_mean[3i:3i+3]) / pixel_std[3i:3i+3] 에 모달 순서 i 로
    잘라 쓴다(dgfusion.py:347-348) — 이 슬라이스로 원본을 채우면 정규화 후 0 이 된다.
    모달이 cfg 순서에 없거나 버퍼가 없거나 길이가 3*모달수 가 아니면 명확한 에러로
    멈춘다(추측 보정 금지).
    """
    import torch
    if modal not in MODAL_KEYS:
        raise ValueError(f"modal 은 {MODAL_KEYS} 중 하나여야 한다: {modal!r}")
    modals = modal_order_from_cfg(cfg)
    if modal not in modals:
        raise RuntimeError(
            f"모달 {modal} 이 cfg 모달 순서({modals}) 에 없다 — config 의 "
            f"DATASETS.MODALITIES.ORDER·MAIN_MODALITY 를 확인하라(추측 금지).")
    i = modals.index(modal)
    pm = getattr(model, "pixel_mean", None)
    if not torch.is_tensor(pm):
        raise RuntimeError(
            "model.pixel_mean 버퍼가 없다(또는 tensor 가 아니다) — 이 모델의 정규화는 "
            "dgfusion.py:347-348 구조(모달별 3채널 평균·표준편차를 하나의 긴 버퍼로 이어 "
            "붙인 pixel_mean·pixel_std)가 아니다. BF_ZERO_MODE=raw 로 돌리면 정규화 전 "
            "0 채움이 된다(추측 금지).")
    buf = pm.detach().float().cpu().reshape(-1)
    n = len(modals)
    if buf.numel() != 3 * n:
        raise RuntimeError(
            f"model.pixel_mean 길이({buf.numel()}) 가 3*모달수({3 * n}) 가 아니다 — 버퍼 "
            f"구조가 다르다. BF_ZERO_MODE=raw 로 돌리면 정규화 전 0 채움이 된다(추측 금지).")
    return i, tuple(float(v) for v in buf[3 * i:3 * i + 3])


def zero_modal_in_batch(batch, modal, mode="normalized", cfg=None, model=None):
    """batched_inputs(list[dict]) 안의 지정 모달 입력 텐서를 채운다(in-place).

    mode(A7·A12):
      - "normalized"(기본): 원본 텐서를 **모델 버퍼** model.pixel_mean[3i:3i+3](i = cfg
        모달 순서상 인덱스)로 채운다. 모델 정규화 (x-mean)/std(dgfusion.py:347-348)가
        그 버퍼를 쓰므로 모델이 보는 값이 0 — 우리 modality_zero_ablation.py(정규화 후
        0)와 같은 개입 규약. cfg·model 필수. cfg 의 PIXEL_MEAN(A11 탐색) 값은
        assert_normalized_fill_is_zero 의 대조용으로만 쓴다(채움 값의 출처는 모델 버퍼).
      - "raw": 원본 텐서를 0 으로 채운다(옛 동작 — 정규화 후에는 -PIXEL_MEAN/PIXEL_STD
        상수가 된다).

    모델 입력 dict 의 모달 키는 CAMERA·LIDAR·EVENT·DEPTH 이며 주 모달(RGB)은 image
    키에도 중복 저장된다. modal=CAMERA 면 image 키까지 함께 채운다(두 키가 같은
    텐서를 공유하지 않을 수 있으므로 둘 다 처리).
    정규화 구조·cfg 대조 검증은 여기서 하지 않는다 — 모델이 있는 호출부(main) 가
    assert_normalized_fill_is_zero 로 검증한다.
    """
    import torch
    if modal not in MODAL_KEYS:
        raise ValueError(f"modal 은 {MODAL_KEYS} 중 하나여야 한다: {modal!r}")
    if mode not in ("normalized", "raw"):
        raise ValueError(f"mode 는 normalized|raw 중 하나여야 한다: {mode!r}")
    mean = None
    if mode == "normalized":
        if cfg is None or model is None:
            raise ValueError(
                "mode='normalized' 는 모달 인덱스를 구할 cfg 와 채움 값의 출처가 될 "
                "모델 버퍼의 model 이 필요하다(raw 면 불필요).")
        _i, mean = model_modal_mean(model, cfg, modal)
    keys = {modal}
    if modal == "CAMERA":
        keys.add("image")
    for d in batch:
        if isinstance(d, dict):
            for k in keys:
                v = d.get(k)
                if torch.is_tensor(v):
                    d[k] = (_fill_modal(v, mean) if mode == "normalized"
                            else torch.zeros_like(v))
    return batch


def assert_normalized_fill_is_zero(model, modal, cfg):
    """A12 — '원본을 모달 평균으로 채우면 정규화 후 0' 임을 모델 구조로 검증한다.

    모델은 모달별 3채널 평균·표준편차를 하나의 긴 버퍼(pixel_mean·pixel_std)로 이어
    붙여 갖고(dgfusion.py:108-132), 모달 순서 i 로 잘라 (x - mean[3i:3i+3]) /
    std[3i:3i+3] 정규화를 쓴다(dgfusion.py:347-348). 검증 내용:
      - 모달 인덱스 i — cfg.DATASETS.MODALITIES.ORDER(주 모달=MAIN_MODALITY 먼저)에서,
      - model.pixel_mean 버퍼 존재·길이 3*모달수(model_modal_mean 이 확인),
      - model.pixel_mean[3i:3i+3] 이 A11 cfg 탐색값과 같은지(허용오차 1e-4).
    어느 것이든 성립하지 않으면 RuntimeError — normalized 를 적용하지 않고 멈춘다
    (추측 금지). normalize_image 메서드 유무는 더 이상 보지 않는다(A12 — 그 메서드가
    없어 2026-09-18 jarvis DGFusion 80k 에서 실패한 실측).
    """
    import torch
    if modal not in MODAL_KEYS:
        raise ValueError(f"modal 은 {MODAL_KEYS} 중 하나여야 한다: {modal!r}")
    i, model_mean = model_modal_mean(model, cfg, modal)
    cfg_mean = find_modal_pixel_mean(cfg, modal)
    if not torch.allclose(torch.as_tensor(model_mean, dtype=torch.float32),
                          torch.as_tensor(cfg_mean, dtype=torch.float32), atol=1e-4):
        raise RuntimeError(
            f"모델 버퍼 pixel_mean[{3 * i}:{3 * i + 3}]={list(model_mean)} 와 cfg 평균 "
            f"{list(cfg_mean)} 이 다르다(허용오차 1e-4) — 모달 순서나 통계 정의를 확인"
            f"하라(추측 금지). BF_ZERO_MODE=raw 로 돌리면 정규화 전 0 채움이 된다.")
    print(f"[probe_dgfusion] 모달 {modal} 인덱스 {i}, 모델 평균 {list(model_mean)} = "
          f"cfg 평균 {list(cfg_mean)} — 평균으로 채우면 정규화 후 0")


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
                    help="지정 모달의 입력 텐서를 채운 상태로 추론(A4). CAMERA 면 주 모달 "
                         "중복 키 image 까지 함께 채운다. 채우는 값은 --zero-mode 참조.")
    ap.add_argument("--zero-mode", default="normalized",
                    choices=["normalized", "raw"],
                    help="모달 zero-out 개입 방식(A7·A12). normalized(기본)=원본을 모델 "
                         "버퍼 pixel_mean[3i:3i+3](모달 인덱스 i 는 cfg.DATASETS."
                         "MODALITIES 기준)로 채워 정규화 후 0 — 우리 "
                         "modality_zero_ablation 과 같은 규약. cfg PIXEL_MEAN 은 검증 "
                         "대조용. raw=옛 동작(원본 0 채움, 정규화 후 -PIXEL_MEAN/"
                         "PIXEL_STD 상수). 버퍼 구조가 다른 모델이면 에러로 멈춘다.")
    ap.add_argument("--deliver-root", default=None,
                    help="DELIVER 데이터셋 루트(GT depth/semantic 위치). 못 주면 "
                         "file_name 의 '/img/' 마커에서 유도한다.")
    ap.add_argument("--opts", nargs="*", default=[],
                    help="detectron2 config 덮어쓰기(KEY VALUE 쌍). 예: "
                         "--opts DATASETS.TEST_SEMANTIC \"('deliver_semantic_test',)\"")
    args = ap.parse_args()

    if not args.config_file or not args.weights:
        ap.error("--config-file 과 --weights 는 실행에 필요(스모크는 인자 파싱만 검사)")

    # 지연 import — 기준선 repo 에서만 사용 가능.
    import torch  # noqa: F401
    from detectron2.checkpoint import DetectionCheckpointer
    from train_net import Trainer, setup  # 기준선 repo 루트에서 실행

    class _A:
        config_file = args.config_file
        opts = ["MODEL.WEIGHTS", args.weights, "MODEL.IS_TRAIN", "False"] + list(args.opts)
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
        print(f"[probe_dgfusion] --zero-modal {args.zero_modal} zero-mode={args.zero_mode}"
              + (" — 원본을 모델 버퍼 pixel_mean 모달 슬라이스로 채운다(정규화 후 0, A7·A12)"
                 if args.zero_mode == "normalized" else
                 " — 원본을 0 으로 채운다(정규화 후 -PIXEL_MEAN/PIXEL_STD 상수, 옛 동작)")
              + ". CAMERA 면 image 키까지 함께 채운다.")
        if args.zero_mode == "normalized":
            # A12 — 채움 값의 출처는 모델 버퍼 슬라이스. cfg 값(PIXEL_MEAN)은 아래
            # 검증에서 대조용으로만 쓴다.
            _zm_i, _zm_mean = model_modal_mean(model, cfg, args.zero_modal)
            assert_normalized_fill_is_zero(model, args.zero_modal, cfg)
            print(f"[probe_dgfusion] 모달 {args.zero_modal} 인덱스 {_zm_i} "
                  f"zero_mode={args.zero_mode} -> 채운 값 {list(_zm_mean)}"
                  f"(모델 버퍼 pixel_mean 슬라이스, 정규화 후 0 — CSV zero_mode 열에 기록).")

    # depth 헤드의 로그 스케일 여부 — config 에서 읽는다(없으면 선형으로 간주하고 경고).
    try:
        log_scale = bool(cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE)
    except AttributeError:
        log_scale = False
        print("[probe_dgfusion] ⚠️ cfg.MODEL.DEPTH_HEAD.LOSS.LOG_SCALE 키가 없다 — "
              "False(선형 스케일)로 간주한다. 실제 로그 스케일 학습이면 config 를 확인하라.")
    print(f"[probe_dgfusion] depth 헤드 로그 스케일 = {log_scale}"
          + (" (exp 로 되돌려 비교)" if log_scale else ""))

    ds_name = cfg.DATASETS.TEST_SEMANTIC[0] if hasattr(cfg.DATASETS, "TEST_SEMANTIC") else cfg.DATASETS.TEST[0]
    # 기준선 Trainer 의 test 로더를 그대로 쓴다. detectron2 기본 DatasetMapper 를 쓰면
    # 모달 입력(batched_inputs[0]['modalities'])이 없어 모델 forward 가 즉시 실패한다.
    loader = Trainer.build_test_loader(cfg, ds_name)

    rows = []
    for i, batch in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        probe.clear()
        if args.zero_modal != "none":
            zero_modal_in_batch(batch, args.zero_modal, mode=args.zero_mode,
                                cfg=cfg, model=model)
        with torch.no_grad():
            outputs = model(batch)
        file_name = batch[0].get("file_name", "")
        rec = {"idx": i, "file_name": file_name, "zero_modal": args.zero_modal,
               "zero_mode": args.zero_mode if args.zero_modal != "none" else "none"}
        for tag in ("depth_token", "cond_token"):
            caps = probe.captured.get(tag, {})
            if caps:
                st = tensor_stats(_to_np(next(iter(caps.values()))))
                rec[f"{tag}_mean"], rec[f"{tag}_var"] = st["mean"], st["var"]
        # (a) depth 헤드 AbsRel·delta1 + (2) 분할 mIoU — DELIVER GT 원본으로 채점.
        depth_gt_path, sem_gt_path = deliver_gt_paths(file_name, args.deliver_root)
        # 모델이 내보낸 pred_depth 가 있으면 그것을 쓴다. 후처리(패딩 제거 + 원본
        # 해상도 보간)를 이미 거쳤으므로 GT 와 좌표계가 맞는다. depth_head 훅의 원출력은
        # 패딩된 입력 좌표계라 GT 로 늘리면 내용이 어긋난다(dgfusion.py:505-513).
        depth_src = None
        if isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], dict) \
                and outputs[0].get("pred_depth") is not None:
            depth_src = outputs[0]["pred_depth"]
            rec["depth_src"] = "pred_depth"
        else:
            depth_caps = probe.captured.get("depth_head", {})
            if depth_caps:
                depth_src = next(iter(depth_caps.values()))
                rec["depth_src"] = "hook_padded"
        if depth_src is not None:
            pred_d = depth_pred_2d(depth_src)
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
