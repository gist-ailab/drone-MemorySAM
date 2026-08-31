#!/usr/bin/env python
"""[P50-MAP] ImageNet → pseudo-모달(depth / pseudo-LiDAR / event-proxy) 생성기.

정본 설계 = `.claude_logs/decisions/2026-08-17-p50-map-modal-alignment-pretraining-proposal.md`
(OmniSegmentor 2509.15096 의 ImageNeXt 레시피 재구축 — 공개본이 없어 자체 생성)

사용:
    # 실제 생성 (ImageNet train 서브셋 200k, GPU 4장 분산)
    python tools/p50_gen_pseudomodal.py \
        --imagenet /path/to/imagenet/train --out /path/to/ImageNeXt_p50 \
        --n 200000 --size 448 --gpu 0,1,2,3

    # 오프라인/스모크 (네트워크·가중치 불요)
    python tools/p50_gen_pseudomodal.py --imagenet <dir> --out <dir> \
        --n 8 --size 128 --depth-backend synthetic

    # [P50-EXT] 증분 — 기존 코퍼스(예: 200k 산출물)에 aolp/dolp/nir 만 추가.
    # depth 백엔드는 기동하지 않고 이미 생성된 depth(HHA) 캐시에서 유도한다.
    python tools/p50_gen_pseudomodal.py --imagenet <dir> --out <기존out> \
        --modals aolp,dolp,nir

출력 (DELIVER 로더가 읽는 것과 같은 3채널 uint8 PNG):
    <out>/rgb/<stem>.png      원본 RGB(정사각 리사이즈본)
    <out>/depth/<stem>.png    HHA 3채널 (DELIVER 는 depth 모달을 'hha' 디렉토리에서
                              읽는다 — semseg/datasets/deliver.py __getitem__ 참조)
    <out>/lidar/<stem>.png    희소 range 이미지 (빔/방위각 래스터화, 0 = no return)
    <out>/event/<stem>.png    event **프록시** (ch0=+극성, ch1=−극성, ch2=0)
    <out>/index.txt           생성 완료 stem 목록
    <out>/meta.json           도구·버전·파라미터·실제 사용된 depth 백엔드

[P50-EXT] MCubeS 팔용 추가 출력 (--modals aolp,dolp,nir) — 원자 파일 규격
(semseg/datasets/mcubes.py 로더가 읽는 형태. 3채널로 미리 합치지 않는다 — 로더가 stack):
    <out>/aolp_sin/<stem>.npy  H×W float32 ∈ [-1,1] (편광각 sin(2θ))
    <out>/aolp_cos/<stem>.npy  H×W float32 ∈ [-1,1] (편광각 cos(2θ))
    <out>/dolp/<stem>.npy      H×W float32 ∈ [0,1]
    <out>/nir/<stem>.png       H×W uint8 (8bit [0,1] 단채널)

🔴 event 는 **프록시**다. N-ImageNet 을 받을 수 없는 환경 전제라, 단일 이미지의
   로그 강도 공간 그래디언트에 ± 극성을 붙여 event 로더와 **채널 인터페이스만**
   맞춘 것이다. 실제 event 스트림의 시간 통계(노이즈·refractory·모션 블러)는
   담기지 않는다. 논문/문서에 반드시 'proxy' 로 적을 것.
🔴 aolp/dolp/nir 도 전부 **프록시**다. 편광 두 모달은 그 이미지의 생성 depth 에서
   유도한 법선의 기하(방위각·천정각) 뿐이며 실측 편광 스펙트럼이 아니고, nir 는
   RGB 의 결정론적 변환(luminance + excess-green)일 뿐 실측 밴드가 아니다.
   신경망 forward 는 한 번도 추가하지 않는다(증분 시 depth 백엔드 기동 0회).

재개: 선택 모달(+rgb) 출력이 모두 있는 stem 은 건너뛴다. stem 처리 시에는
      **없는 파일만** 쓴다 — 증분 생성이 기존 산출물을 절대 다시 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOOL_VERSION = '1.1.0'
MODAL_DIRS = ('rgb', 'depth', 'lidar', 'event')
# [P50-EXT] --modals 로 선택 가능한 모달 (rgb 는 코퍼스 기저라 항상 생성)
SELECTABLE_MODALS = ('depth', 'lidar', 'event', 'aolp', 'dolp', 'nir')
DEFAULT_MODALS = 'depth,lidar,event'      # 종전 동작(4모달 전량)과 동일
IMG_EXTS = ('.jpeg', '.jpg', '.png', '.bmp', '.webp')

# 가정 카메라 (ImageNet 은 intrinsics 가 없다 — 고정값을 쓰고 meta.json 에 남긴다)
DEFAULT_FOV_DEG = 60.0
MIN_DEPTH_M = 1.0        # 정규화 depth 0 이 대응하는 거리
MAX_DEPTH_M = 60.0       # 정규화 depth 1 이 대응하는 거리
HEIGHT_MAX_M = 5.0       # HHA ch1 포화 높이
LIDAR_VALUE_SCALE = 0.38  # DELIVER lidar 실측 동적범위 [0, 0.38] (augmentations_mm 주석)

# [P50-EXT] 편광/NIR proxy 파라미터 (유도식은 meta.json 에도 기록된다)
DOLP_FRESNEL_COEF = 0.8   # DoLP Fresnel 근사 계수 (굴절률 1.5, specular-지배 가정)
NIR_EXCESS_GREEN_W = 0.5  # NIR excess-green 보정 가중치 (식생 NIR 밝음 근사)


# ═══════════════════════════════════════════════════════════════════════════
# 1. depth 백엔드
# ═══════════════════════════════════════════════════════════════════════════
class DepthBackend:
    """모든 백엔드는 (B,3,H,W) float [0,1] RGB → (B,H,W) float **정규화 depth**
    (0 = 가까움, 1 = 멂) 를 돌려준다. 백엔드별 관례(disparity vs depth)는 각
    구현이 흡수하고, 밖에서는 항상 depth 로 본다."""

    name = 'base'

    def __call__(self, rgb: 'torch.Tensor') -> 'torch.Tensor':  # noqa: F821
        raise NotImplementedError


def _robust_norm(x, lo_q: float = 2.0, hi_q: float = 98.0):
    """샘플별 로버스트 min-max → [0,1]. (백엔드 출력이 relative scale 이라 필수)"""
    import torch
    flat = x.flatten(1)
    lo = torch.quantile(flat.float(), lo_q / 100.0, dim=1)[:, None, None]
    hi = torch.quantile(flat.float(), hi_q / 100.0, dim=1)[:, None, None]
    return ((x - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)


class OmnidataDepth(DepthBackend):
    """Omnidata DPT (2110.04994). `pip install omnidata-tools` + 공식 ckpt 필요.

    Omnidata 는 HF Hub 에 없어서 완전 자동 다운로드가 불가능하다 —
    `--omnidata-ckpt` 로 파일을 주면 쓰고, 없으면 조용히가 아니라 **경고와 함께**
    다음 백엔드로 넘어간다(실제 사용 백엔드는 meta.json 에 기록된다)."""

    name = 'omnidata_dpt'

    def __init__(self, ckpt: Optional[str], device):
        import torch
        try:
            from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel
        except Exception as e:      # 패키지 없음
            raise RuntimeError(f"omnidata-tools import 실패: {e}")
        if not ckpt or not os.path.isfile(ckpt):
            raise RuntimeError(
                "omnidata ckpt 미지정/미존재 (--omnidata-ckpt). Omnidata 가중치는 "
                "공식 배포처에서 직접 받아야 한다 (HF Hub 자동 다운로드 없음).")
        model = DPTDepthModel(backbone='vitb_rn50_384')
        sd = torch.load(ckpt, map_location='cpu')
        sd = sd.get('state_dict', sd)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd)
        self.model = model.to(device).eval()
        self.device = device
        self.size = 384

    def __call__(self, rgb):
        import torch
        import torch.nn.functional as F
        x = F.interpolate(rgb, size=(self.size, self.size), mode='bilinear',
                          align_corners=False)
        x = (x - 0.5) / 0.5
        with torch.no_grad():
            d = self.model(x)
        if d.dim() == 4:
            d = d.squeeze(1)
        d = F.interpolate(d[:, None], size=rgb.shape[-2:], mode='bilinear',
                          align_corners=False)[:, 0]
        return _robust_norm(d)          # omnidata 는 depth 관례 (클수록 멂)


class DepthAnythingDepth(DepthBackend):
    """Depth-Anything V2 (HF `transformers` 자동 다운로드). 출력 = disparity."""

    name = 'depth_anything_v2'

    def __init__(self, model_id: str, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.model_id = model_id
        self.name = f'depth_anything({model_id})'

    def __call__(self, rgb):
        import torch
        import torch.nn.functional as F
        arrs = [(r.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for r in rgb]
        inp = self.proc(images=arrs, return_tensors='pt').to(self.device)
        with torch.no_grad():
            disp = self.model(**inp).predicted_depth        # (B,h,w) disparity
        disp = F.interpolate(disp[:, None].float(), size=rgb.shape[-2:],
                             mode='bilinear', align_corners=False)[:, 0]
        depth = 1.0 / _robust_norm(disp).clamp(min=1e-3)    # disparity → depth
        return _robust_norm(depth)


class MiDaSDepth(DepthBackend):
    """MiDaS (torch.hub 자동 다운로드). 출력 = inverse depth(disparity)."""

    name = 'midas'

    def __init__(self, variant: str, device):
        import torch
        self.model = torch.hub.load('intel-isl/MiDaS', variant, trust_repo=True)
        self.model = self.model.to(device).eval()
        self.size = 384 if 'small' not in variant.lower() else 256
        self.device = device
        self.name = f'midas({variant})'

    def __call__(self, rgb):
        import torch
        import torch.nn.functional as F
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device)[:, None, None]
        x = F.interpolate(rgb, size=(self.size, self.size), mode='bicubic',
                          align_corners=False)
        x = (x - mean) / std
        with torch.no_grad():
            disp = self.model(x)
        disp = F.interpolate(disp[:, None].float(), size=rgb.shape[-2:],
                             mode='bilinear', align_corners=False)[:, 0]
        depth = 1.0 / _robust_norm(disp).clamp(min=1e-3)
        return _robust_norm(depth)


class SyntheticDepth(DepthBackend):
    """오프라인 결정론적 대체물 — **스모크 전용**. 밝기 기반 가짜 depth 라
    실제 생성 런에 쓰면 안 된다(meta.json 에 그대로 남아 사후에 드러난다)."""

    name = 'synthetic(SMOKE-ONLY)'

    def __init__(self, device):
        self.device = device

    def __call__(self, rgb):
        import torch
        gray = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2])
        h, w = gray.shape[-2:]
        yy = torch.linspace(0, 1, h, device=gray.device)[:, None]
        d = 0.6 * (1.0 - gray) + 0.4 * (1.0 - yy)     # 아래쪽=가까움 가정
        return _robust_norm(d)


def build_depth_backend(pref: str, device, omnidata_ckpt: Optional[str],
                        da_model: str, midas_variant: str,
                        verbose: bool = True) -> DepthBackend:
    """요청 백엔드 → 실패 시 문서화된 순서로 폴백. 실제 선택은 반환값 .name."""
    order = {
        'omnidata': ['omnidata', 'depth_anything', 'midas'],
        'depth_anything': ['depth_anything', 'midas', 'omnidata'],
        'midas': ['midas', 'depth_anything', 'omnidata'],
        'synthetic': ['synthetic'],
        'auto': ['omnidata', 'depth_anything', 'midas'],
    }[pref]
    errs = []
    for cand in order:
        try:
            if cand == 'omnidata':
                return OmnidataDepth(omnidata_ckpt, device)
            if cand == 'depth_anything':
                return DepthAnythingDepth(da_model, device)
            if cand == 'midas':
                return MiDaSDepth(midas_variant, device)
            if cand == 'synthetic':
                return SyntheticDepth(device)
        except Exception as e:
            errs.append(f"{cand}: {type(e).__name__}: {e}")
            if verbose:
                print(f"[P50-gen] depth backend '{cand}' 사용 불가 → 폴백. ({e})")
    raise RuntimeError("사용 가능한 depth 백엔드가 없다:\n  " + "\n  ".join(errs))


# ═══════════════════════════════════════════════════════════════════════════
# 2. depth → 모달 렌더링 (numpy, per-image)
# ═══════════════════════════════════════════════════════════════════════════
def _intrinsics(h: int, w: int, fov_deg: float) -> Tuple[float, float, float, float]:
    f = 0.5 * max(h, w) / np.tan(np.deg2rad(fov_deg) * 0.5)
    return f, f, w * 0.5, h * 0.5


def backproject(depth_norm: np.ndarray, fov_deg: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """정규화 depth [0,1] → 카메라 좌표 (X 우, Y 하, Z 전방) 미터."""
    h, w = depth_norm.shape
    fx, fy, cx, cy = _intrinsics(h, w, fov_deg)
    z = MIN_DEPTH_M + depth_norm.astype(np.float32) * (MAX_DEPTH_M - MIN_DEPTH_M)
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def render_hha(depth_norm: np.ndarray, fov_deg: float) -> np.ndarray:
    """HHA (Gupta et al. ECCV'14) 3채널 uint8 — DELIVER 의 depth 모달과 같은 형식.

      ch0 horizontal disparity  1/Z 를 [1/MAX, 1/MIN] 로 고정 정규화
      ch1 height above ground   중력(=이미지 +Y) 방향 기준 지면 대비 높이, 0~5m
      ch2 angle with gravity    표면 법선과 중력의 각도 0~180°

    ⚠️ ImageNet 에는 intrinsics 도 중력 방향도 없다. fx=fy=fov(기본 60°),
       중력 = 이미지 아래 방향, 지면 = Y 의 99 백분위로 **가정**한다. 이 가정은
       meta.json 에 그대로 기록된다 (합성 코퍼스의 한계이지 은폐 대상이 아니다).
    """
    x, y, z = backproject(depth_norm, fov_deg)

    disp = 1.0 / np.maximum(z, 1e-3)
    lo, hi = 1.0 / MAX_DEPTH_M, 1.0 / MIN_DEPTH_M
    ch0 = np.clip((disp - lo) / (hi - lo), 0.0, 1.0)

    y_ground = float(np.percentile(y, 99.0))
    ch1 = np.clip((y_ground - y) / HEIGHT_MAX_M, 0.0, 1.0)

    # 표면 법선 = 접선벡터 외적 (u, v 방향 유한차분)
    p = np.stack([x, y, z], axis=-1)
    tu = np.zeros_like(p); tu[:, 1:-1] = (p[:, 2:] - p[:, :-2]) * 0.5
    tv = np.zeros_like(p); tv[1:-1] = (p[2:] - p[:-2]) * 0.5
    n = np.cross(tu, tv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    cos_g = np.clip(np.abs(n[..., 1]), 0.0, 1.0)          # 중력 g = (0,1,0)
    ch2 = np.degrees(np.arccos(cos_g)) / 180.0

    hha = np.stack([ch0, ch1, ch2], axis=-1)
    return np.clip(hha * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render_pseudo_lidar(depth_norm: np.ndarray, fov_deg: float,
                        beams: int = 64, az_bins: int = 1024,
                        beam_width: float = 0.25,
                        value_scale: float = LIDAR_VALUE_SCALE) -> np.ndarray:
    """추정 depth → 회전식 LiDAR 를 카메라 뷰에 투영한 **희소** range 이미지.

    dense depth 를 그대로 lidar 라고 부르면 두 모달이 같은 것이 되어 정렬
    사전학습이 학습할 것이 없어진다. 실제 LiDAR 의 결손 구조를 흉내낸다:
      1) 픽셀별 고도각(elevation)을 계산하고 `beams` 개의 균등 빔에 배정 —
         빔 중심에서 `beam_width` 밖은 버린다 (수직 서브샘플 = 스캔라인 사이 공백)
      2) 방위각(azimuth)을 `az_bins` 로 양자화하고 (빔, 방위각) 셀당 **최근접
         거리 1점만** 남긴다 (수평 각분해능 + 가림 처리)

    저장: 3채널 동일값 uint8 (DELIVER 로더는 1ch 면 3ch 로 복제한다).
      0 = no return. 반환값은 1..round(255*value_scale) — DELIVER lidar 실측
      동적범위 [0, 0.38] 에 맞춘다(augmentations_mm §Normalize 주석).
    """
    h, w = depth_norm.shape
    x, y, z = backproject(depth_norm, fov_deg)
    rng = np.sqrt(x * x + y * y + z * z)
    horiz = np.sqrt(x * x + z * z)
    el = np.arctan2(-y, np.maximum(horiz, 1e-6))          # 위쪽 = +
    az = np.arctan2(x, np.maximum(z, 1e-6))

    el_min, el_max = float(el.min()), float(el.max())
    if el_max - el_min < 1e-6:
        el_max = el_min + 1e-6
    step = (el_max - el_min) / max(beams - 1, 1)
    beam_idx = np.rint((el - el_min) / step)
    keep = np.abs(el - (el_min + beam_idx * step)) <= (step * beam_width)
    beam_idx = beam_idx.astype(np.int64)

    az_min, az_max = float(az.min()), float(az.max())
    az_span = max(az_max - az_min, 1e-6)
    az_idx = np.clip(((az - az_min) / az_span * (az_bins - 1)).astype(np.int64),
                     0, az_bins - 1)

    out = np.zeros((h, w), dtype=np.float32)
    ys, xs = np.nonzero(keep)
    if ys.size:
        cell = beam_idx[ys, xs] * az_bins + az_idx[ys, xs]
        r = rng[ys, xs]
        # 셀당 최근접 1점: (cell, range) 사전식 정렬 후 셀 첫 원소만 취한다.
        order = np.lexsort((r, cell))
        cell_s = cell[order]
        first = np.ones(cell_s.shape[0], dtype=bool)
        first[1:] = cell_s[1:] != cell_s[:-1]
        sel = order[first]
        out[ys[sel], xs[sel]] = np.clip(
            (r[sel] - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M), 0.0, 1.0)

    vmax = max(int(round(255 * value_scale)), 2)
    px = np.where(out > 0, 1.0 + out * (vmax - 1.0), 0.0)
    px = np.clip(px + 0.5, 0, 255).astype(np.uint8)
    return np.repeat(px[:, :, None], 3, axis=2)


def render_event_proxy(rgb_u8: np.ndarray, contrast_thresh: float = 0.15,
                       motion: str = 'x', rng: Optional[random.Random] = None,
                       saturate: int = 4) -> np.ndarray:
    """단일 이미지 event **프록시**.

    event 카메라는 log 강도의 시간 변화 ΔlogI 에 발화한다. 단일 프레임에는 시간
    축이 없으므로, 균일 병진 운동을 가정해 ΔlogI ≈ −v·∇logI 로 근사하고
    (공간 그래디언트) 극성별로 발화 수를 센다.

      ch0 = +극성 발화 수 / saturate,  ch1 = −극성,  ch2 = 0
      (DELIVER event 로더가 읽는 3채널 uint8 PNG 와 인터페이스 동일)

    🔴 실제 event 스트림의 시간 통계는 담기지 않는다. 항상 'proxy' 로 표기할 것.
    """
    import cv2
    g = rgb_u8.astype(np.float32) / 255.0
    gray = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    log_i = np.log(gray + 0.01)
    if motion == 'random' and rng is not None:
        motion = rng.choice(['x', 'y'])
    if motion == 'y':
        d = cv2.Sobel(log_i, cv2.CV_32F, 0, 1, ksize=3)
    else:
        d = cv2.Sobel(log_i, cv2.CV_32F, 1, 0, ksize=3)
    n = d / max(contrast_thresh, 1e-6)                    # 발화 "개수" 근사
    pos = np.clip(n, 0, saturate) / saturate
    neg = np.clip(-n, 0, saturate) / saturate
    ev = np.stack([pos, neg, np.zeros_like(pos)], axis=-1)
    return np.clip(ev * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ── [P50-EXT] 편광/NIR proxy (MCubeS 팔) ────────────────────────────────────
# 🔴 전부 proxy 다. depth 유도(편광)·RGB 결정론 변환(NIR)이며 실측 모달이 아니다.
#    새 신경망/백엔드 forward 는 추가하지 않는다.
def surface_normal(depth_norm: np.ndarray) -> np.ndarray:
    """생성 depth D → 근사 표면 법선 n = normalize([-∂D/∂x, -∂D/∂y, 1]) (Sobel 3×3).

    D 를 높이장(height field)으로 보는 근사다. depth 척도(미터↔정규화)는 단위벡터화
    로 흡수되고, 하류(DoLP 곡선)는 단조이기만 하면 되므로 척도 정확도는 요구하지
    않는다 — proxy 유도식은 meta.json 에 기록된다."""
    import cv2
    gx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)


def render_aolp_proxy(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """편광각 **프록시** — (sin(2θ), cos(2θ)) 반환.

    관행: 편광각은 법선 방위각과 직교 → θ = atan2(n_y, n_x) + π/2. 편광각은 π 주기라
    2θ 인코딩으로 출력하며, sin/cos 를 분리 저장하는 것은 MCubeS 원본 관행
    (polL_aolp_sin / polL_aolp_cos) 유지다 — 로더가 [sin, cos, sin] 3채널로 stack 한다."""
    theta = np.arctan2(normal[..., 1], normal[..., 0]) + np.pi / 2.0
    return np.sin(2.0 * theta), np.cos(2.0 * theta)


def render_dolp_proxy(normal: np.ndarray) -> np.ndarray:
    """DoLP **프록시** — 천정각 z = arccos(n_z) 의 Fresnel 근사(굴절률 1.5,
    specular-지배 가정): dolp = sin²z·DOLP_FRESNEL_COEF / (2 − sin²z), [0,1] 클립.

    브루스터각 근방에서 최대가 되는 단조 곡선이면 계수 정확도는 중요치 않다(proxy)."""
    nz = np.clip(normal[..., 2], -1.0, 1.0)
    s2 = 1.0 - nz * nz                                     # sin²(z)
    return np.clip(s2 * DOLP_FRESNEL_COEF / (2.0 - s2), 0.0, 1.0)


def render_nir_proxy(rgb_u8: np.ndarray) -> np.ndarray:
    """NIR **프록시**(결정론적 변환, 실측 밴드 아님) — 8bit 단채널 uint8.

      nir = clip(0.299R + 0.587G + 0.114B + NIR_EXCESS_GREEN_W·max(0, G − 0.5(R+B)), 0, 1)
      = luminance + excess-green 보정(식생이 NIR 에서 밝게 보이는 것의 근사).
    """
    g = rgb_u8.astype(np.float32) / 255.0
    r, gg, b = g[..., 0], g[..., 1], g[..., 2]
    nir = (0.299 * r + 0.587 * gg + 0.114 * b
           + NIR_EXCESS_GREEN_W * np.maximum(0.0, gg - 0.5 * (r + b)))
    return np.clip(nir * 255.0 + 0.5, 0, 255).astype(np.uint8)


def depth_from_cached_hha(path: Path) -> Optional[np.ndarray]:
    """증분 모드용 — 기존 생성 depth(HHA PNG)에서 정규화 depth D 를 역복환한다.

    render_hha 의 ch0 = (1/z − 1/MAX)/(1/MIN − 1/MAX) 를 역으로 풀어
    D = (z − MIN)/(MAX − MIN) 을 복원한다(8bit 양자화 오차만 포함).
    파일이 없으면 None — 이 경우 aolp/dolp 는 근거 없는 값을 만들지 않고 스킵한다."""
    if not path.is_file():
        return None
    hha = np.asarray(Image.open(path))
    lo, hi = 1.0 / MAX_DEPTH_M, 1.0 / MIN_DEPTH_M
    ch0 = hha[..., 0].astype(np.float32) / 255.0
    disp = np.maximum(ch0 * (hi - lo) + lo, 1e-3)
    z = 1.0 / disp
    return np.clip((z - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M), 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 입력 목록 / IO
# ═══════════════════════════════════════════════════════════════════════════
def list_images(root: str, n: int, seed: int = 0) -> List[Path]:
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"--imagenet 경로가 디렉토리가 아니다: {root}")
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root_p):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                files.append(Path(dirpath) / fn)
    files.sort()
    if not files:
        raise FileNotFoundError(f"이미지가 없다: {root} (확장자 {IMG_EXTS})")
    if n > 0 and len(files) > n:
        random.Random(seed).shuffle(files)
        files = sorted(files[:n])
    return files


def stem_of(path: Path, root: Path) -> str:
    """ImageNet 은 파일명이 전역 고유(n01440764_10026)지만, 임의 코퍼스도 받도록
    클래스 디렉토리를 접두어로 붙여 충돌을 원천 차단한다."""
    rel = path.relative_to(root)
    parts = list(rel.parts[:-1]) + [rel.stem]
    return '__'.join(parts[-2:]) if len(parts) > 1 else parts[-1]


def load_square(path: Path, size: int) -> Optional[np.ndarray]:
    try:
        im = Image.open(path).convert('RGB')
    except Exception:
        return None
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(int(round(w * s)), size), max(int(round(h * s)), size)),
                   Image.BICUBIC)
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(im.crop((left, top, left + size, top + size)), dtype=np.uint8)


def parse_modals(spec: str) -> List[str]:
    """'--modals' 파싱·검증 (예: 'depth,lidar,event,aolp,dolp,nir'). 순서를 유지한다."""
    toks = [t.strip() for t in str(spec).split(',') if t.strip()]
    if not toks:
        raise ValueError(f"--modals 파싱 결과가 비었다: '{spec}'")
    out: List[str] = []
    for t in toks:
        if t not in SELECTABLE_MODALS:
            raise ValueError(f"알 수 없는 모달 '{t}' (선택 가능: {list(SELECTABLE_MODALS)})")
        if t in out:
            raise ValueError(f"--modals 에 '{t}' 가 중복된다")
        out.append(t)
    return out


def modal_dirs_of(modals: Sequence[str]) -> List[str]:
    """선택 모달 → 생성할 산출 디렉토리 목록 (aolp 은 sin/cos 2원자 파일)."""
    dirs: List[str] = []
    for m in modals:
        dirs.extend(['aolp_sin', 'aolp_cos'] if m == 'aolp' else [m])
    return dirs


def modal_outputs(out_root: Path, stem: str, modals: Sequence[str]) -> dict:
    """stem 의 산출 파일 목록 — rgb(코퍼스 기저, 항상) + 선택 모달.
    확장자는 MCubeS 로더 원자 파일 규격: aolp/dolp = .npy, 나머지 = .png."""
    out: dict = {'rgb': out_root / 'rgb' / f"{stem}.png"}
    for m in modals:
        if m == 'aolp':
            out['aolp_sin'] = out_root / 'aolp_sin' / f"{stem}.npy"
            out['aolp_cos'] = out_root / 'aolp_cos' / f"{stem}.npy"
        elif m == 'dolp':
            out['dolp'] = out_root / 'dolp' / f"{stem}.npy"
        else:
            out[m] = out_root / m / f"{stem}.png"
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. 워커
# ═══════════════════════════════════════════════════════════════════════════
def worker(rank: int, args, files: List[str], in_root: str, ret: Optional[dict] = None):
    import torch

    sel = parse_modals(args.modals)
    # depth 백엔드 forward 는 depth/lidar 재(신규)생성할 때만 필요하다.
    # aolp/dolp 는 그 이미지의 '생성된 depth' 를 재사용(당회 산출 또는 캐시 HHA)하고,
    # nir/event 는 RGB 만 본다 — 증분 런(--modals aolp,dolp,nir)은 백엔드를 만들지도
    # 않는다(기존 200k 산출물에 신규 모달만 얹는 경로가 주 사용례).
    need_backend = ('depth' in sel) or ('lidar' in sel)

    gpus = [int(g) for g in str(args.gpu).split(',') if g.strip() != '']
    world = max(len(gpus), 1)
    dev_id = gpus[rank] if gpus else None
    if dev_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(dev_id)
        device = torch.device(f'cuda:{dev_id}')
    else:
        device = torch.device('cpu')

    backend = None
    if need_backend:
        backend = build_depth_backend(args.depth_backend, device, args.omnidata_ckpt,
                                      args.da_model, args.midas_variant,
                                      verbose=(rank == 0))
    if rank == 0:
        bname = backend.name if backend is not None else 'none (depth/lidar 미선택 — 캐시/RGB 유도만)'
        print(f"[P50-gen] depth backend = {bname} | device={device} | "
              f"workers={world} | modals={sel}")

    out_root = Path(args.out)
    in_root_p = Path(in_root)
    shard = [Path(f) for f in files[rank::world]]
    rng = random.Random(args.seed + rank)

    done, skipped, failed = 0, 0, 0
    t0 = time.time()
    batch: List[Tuple[str, np.ndarray]] = []

    def save_stem(stem: str, img: Optional[np.ndarray], dn: Optional[np.ndarray]):
        """stem 의 **없는 산출만** 쓴다 — 증분 생성이 기존 파일을 다시 쓰지 않는다."""
        paths = modal_outputs(out_root, stem, sel)
        if img is not None:
            if not paths['rgb'].is_file():
                Image.fromarray(img).save(paths['rgb'])
            if 'event' in sel and not paths['event'].is_file():
                Image.fromarray(render_event_proxy(
                    img, contrast_thresh=args.event_thresh,
                    motion=args.event_motion, rng=rng)).save(paths['event'])
            if 'nir' in sel and not paths['nir'].is_file():
                Image.fromarray(render_nir_proxy(img)).save(paths['nir'])
        if dn is not None:
            if 'depth' in sel and not paths['depth'].is_file():
                Image.fromarray(render_hha(dn, args.fov)).save(paths['depth'])
            if 'lidar' in sel and not paths['lidar'].is_file():
                Image.fromarray(render_pseudo_lidar(
                    dn, args.fov, beams=args.lidar_beams, az_bins=args.lidar_az_bins,
                    beam_width=args.lidar_beam_width,
                    value_scale=args.lidar_value_scale)).save(paths['lidar'])
            if 'aolp' in sel or 'dolp' in sel:
                n = surface_normal(dn)
                if 'aolp' in sel and not (paths['aolp_sin'].is_file()
                                          and paths['aolp_cos'].is_file()):
                    a_sin, a_cos = render_aolp_proxy(n)
                    np.save(paths['aolp_sin'], a_sin.astype(np.float32))
                    np.save(paths['aolp_cos'], a_cos.astype(np.float32))
                if 'dolp' in sel and not paths['dolp'].is_file():
                    np.save(paths['dolp'], render_dolp_proxy(n).astype(np.float32))

    def flush():
        nonlocal done, failed, batch
        if not batch:
            return
        arr = np.stack([b[1] for b in batch]).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
        try:
            depth = backend(t).float().cpu().numpy()
        except Exception as e:
            failed += len(batch)
            print(f"[P50-gen][rank{rank}] depth 추정 실패 {len(batch)}장: {e}")
            batch = []
            return
        for (stem, img), dn in zip(batch, depth):
            try:
                save_stem(stem, img, dn)
                done += 1
            except Exception as e:
                failed += 1
                print(f"[P50-gen][rank{rank}] 저장 실패 {stem}: {e}")
        batch = []

    for i, f in enumerate(shard):
        stem = stem_of(f, in_root_p)
        paths = modal_outputs(out_root, stem, sel)
        if all(p.is_file() for p in paths.values()):
            skipped += 1
            continue
        # 원본 RGB 픽셀이 필요한가: rgb 산출이 없거나, event/nir 를 만들거나, 백엔드 입력
        need_img = (not paths['rgb'].is_file()) or ('event' in sel) \
            or ('nir' in sel) or need_backend
        need_dn = any(m in sel for m in ('depth', 'lidar', 'aolp', 'dolp'))
        img = load_square(f, args.size) if need_img else None
        if need_img and img is None:
            failed += 1
            continue
        if need_backend:
            batch.append((stem, img))
            if len(batch) >= args.batch:
                flush()
        else:
            dn = None
            if need_dn:
                dn = depth_from_cached_hha(out_root / 'depth' / f"{stem}.png")
                if dn is None:
                    print(f"[P50-gen][rank{rank}] {stem}: depth 캐시 없음 → aolp/dolp "
                          f"스킵 (새로 만들려면 --modals 에 depth 를 포함할 것)",
                          flush=True)
            try:
                save_stem(stem, img, dn)
                done += 1
            except Exception as e:
                failed += 1
                print(f"[P50-gen][rank{rank}] 저장 실패 {stem}: {e}")
        if rank == 0 and (i + 1) % max(args.log_interval, 1) == 0:
            el = time.time() - t0
            rate = (done + skipped) / max(el, 1e-6)
            print(f"[P50-gen][rank0] {i+1}/{len(shard)} done={done} skip={skipped} "
                  f"fail={failed} {rate:.1f} img/s "
                  f"ETA={(len(shard)-i-1)/max(rate,1e-6)/60:.1f}min", flush=True)
    flush()

    stat = {'rank': rank, 'done': done, 'skipped': skipped, 'failed': failed,
            'backend': backend.name if backend is not None else 'none'}
    print(f"[P50-gen][rank{rank}] finished {stat}")
    if ret is not None:
        ret[rank] = stat
    return stat


# ═══════════════════════════════════════════════════════════════════════════
# 5. main
# ═══════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='[P50-MAP] pseudo-모달 생성기')
    p.add_argument('--imagenet', required=True, help='ImageNet train 루트(재귀 탐색)')
    p.add_argument('--out', required=True, help='출력 루트')
    p.add_argument('--n', type=int, default=200000, help='서브셋 크기 (0=전량)')
    p.add_argument('--size', type=int, default=448, help='정사각 출력 변 길이')
    p.add_argument('--batch', type=int, default=8, help='depth 추정 배치')
    p.add_argument('--gpu', type=str, default='0', help="예: '0,1,2,3' (프로세스 분산)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-interval', type=int, default=200)
    p.add_argument('--depth-backend', default='auto',
                   choices=['auto', 'omnidata', 'depth_anything', 'midas', 'synthetic'])
    p.add_argument('--omnidata-ckpt', default=os.environ.get('OMNIDATA_CKPT', ''))
    p.add_argument('--da-model', default='depth-anything/Depth-Anything-V2-Small-hf')
    p.add_argument('--midas-variant', default='DPT_Large')
    p.add_argument('--fov', type=float, default=DEFAULT_FOV_DEG, help='가정 수평 FOV(도)')
    p.add_argument('--modals', type=str, default=DEFAULT_MODALS,
                   help="쉼표 목록. 기본 'depth,lidar,event' = 종전 4모달(rgb 포함) 동작. "
                        f"추가 가능: {list(SELECTABLE_MODALS)} — aolp/dolp 는 생성 depth "
                        "재사용(캐시 HHA 역변환, 백엔드 기동 0회), nir 는 RGB 결정론 변환")
    p.add_argument('--lidar-beams', type=int, default=64)
    p.add_argument('--lidar-az-bins', type=int, default=1024)
    p.add_argument('--lidar-beam-width', type=float, default=0.25,
                   help='빔 중심 허용폭 (빔 간격 대비 비율) — 작을수록 희소')
    p.add_argument('--lidar-value-scale', type=float, default=LIDAR_VALUE_SCALE)
    p.add_argument('--event-thresh', type=float, default=0.15, help='event 대비 임계 C')
    p.add_argument('--event-motion', default='x', choices=['x', 'y', 'random'])
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sel = parse_modals(args.modals)
    use_backend = ('depth' in sel) or ('lidar' in sel)
    out_root = Path(args.out)
    for d in ['rgb'] + modal_dirs_of(sel):
        (out_root / d).mkdir(parents=True, exist_ok=True)

    files = list_images(args.imagenet, args.n, args.seed)
    print(f"[P50-gen] 입력 {len(files)}장 (root={args.imagenet}) → {out_root} "
          f"(modals={sel})")

    gpus = [g for g in str(args.gpu).split(',') if g.strip() != '']
    fstr = [str(f) for f in files]
    stats: List[dict] = []
    if len(gpus) <= 1:
        stats.append(worker(0, args, fstr, args.imagenet))
    else:
        import torch.multiprocessing as mp
        mgr = mp.Manager()
        ret = mgr.dict()
        mp.spawn(worker, args=(args, fstr, args.imagenet, ret),
                 nprocs=len(gpus), join=True)
        stats = [ret[k] for k in sorted(ret.keys())]

    in_root_p = Path(args.imagenet)
    stems, complete = [], 0
    for f in files:
        s = stem_of(f, in_root_p)
        if all(p.is_file() for p in modal_outputs(out_root, s, sel).values()):
            stems.append(s)
            complete += 1
    (out_root / 'index.txt').write_text('\n'.join(stems) + '\n')

    import torch
    # 증분 런은 기존 meta 를 **병합**한다 — 원본 생성 provenance(depth 백엔드 등)를
    # 덮어쓰지 않는다. 이번 런이 백엔드를 안 돌렸다면 depth_backend 는 캐시의
    # 원 출처 기록을 유지한다.
    meta_path = out_root / 'meta.json'
    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    now_backend = stats[0]['backend'] if stats else 'unknown'
    mod_meta: dict = {
        'rgb': {'format': 'uint8 PNG RGB', 'note': 'short-side resize + center crop'},
    }
    if 'depth' in sel:
        mod_meta['depth'] = {
            'format': 'uint8 PNG 3ch = HHA',
            'channels': ['horizontal disparity', 'height above ground',
                         'angle with gravity'],
            'note': 'DELIVER 의 depth 모달은 hha 디렉토리를 읽는다',
            'assumptions': {'fov_deg': args.fov,
                            'min_depth_m': MIN_DEPTH_M,
                            'max_depth_m': MAX_DEPTH_M,
                            'height_max_m': HEIGHT_MAX_M,
                            'gravity': 'image +y (assumed level camera)',
                            'ground': 'Y 99th percentile'}}
    if 'lidar' in sel:
        mod_meta['lidar'] = {
            'format': 'uint8 PNG 3ch (동일값 복제), 0 = no return',
            'method': 'elevation-beam + azimuth-bin rasterization of '
                      'estimated depth (nearest return per cell)',
            'beams': args.lidar_beams, 'az_bins': args.lidar_az_bins,
            'beam_width': args.lidar_beam_width,
            'value_scale': args.lidar_value_scale,
            'note': 'DELIVER lidar 실측 동적범위 [0,0.38] 에 정합'}
    if 'event' in sel:
        mod_meta['event'] = {
            'format': 'uint8 PNG 3ch — ch0=+polarity, ch1=-polarity, ch2=0',
            'method': 'PROXY: spatial gradient of log intensity under an '
                      'assumed uniform translation (no temporal statistics)',
            'contrast_thresh': args.event_thresh,
            'motion': args.event_motion,
            '🔴 proxy': 'N-ImageNet 대체물. 논문/문서에 proxy 로 명기할 것'}
    # [P50-EXT] aolp/dolp 의 D 출처 — 당회 백엔드 산출인지 캐시 역변환인지
    dsrc = ('depth 백엔드 당회 산출 D' if use_backend
            else '캐시 depth/<stem>.png (HHA ch0 역변환, 8bit 양자화 오차 포함)')
    if 'aolp' in sel:
        mod_meta['aolp'] = {
            'format': 'float32 .npy H×W — aolp_sin / aolp_cos 원자 2파일 '
                      '(로더가 [sin,cos,sin] 3ch 로 stack — mcubes.py 관행)',
            'method': 'PROXY: 법선 n = normalize([-∂D/∂x, -∂D/∂y, 1]) (Sobel 3×3) 의 '
                      '방위각과 직교 관행 θ = atan2(n_y, n_x) + π/2 → sin(2θ), cos(2θ) '
                      '(편광각은 π 주기 → 2θ 인코딩, MCubeS 원본 sin/cos 분리 저장 관행)',
            'depth_source': dsrc,
            '🔴 proxy': 'depth 유도 기하 proxy — 실측 편광 아님. 논문/문서에 proxy 로 명기할 것'}
    if 'dolp' in sel:
        mod_meta['dolp'] = {
            'format': 'float32 .npy H×W [0,1]',
            'method': f'PROXY: 천정각 z = arccos(n_z) → Fresnel 근사(굴절률 1.5, '
                      f'specular-지배 가정) dolp = sin²z·{DOLP_FRESNEL_COEF} / '
                      f'(2 − sin²z), [0,1] 클립',
            'depth_source': dsrc,
            '🔴 proxy': '단조 Fresnel 근사 곡선 proxy — 실측 DoLP 아님. '
                        '논문/문서에 proxy 로 명기할 것'}
    if 'nir' in sel:
        mod_meta['nir'] = {
            'format': 'uint8 PNG H×W 단채널 [0,1]',
            'method': f'PROXY(결정론적 변환): nir = clip(0.299R + 0.587G + 0.114B + '
                      f'{NIR_EXCESS_GREEN_W}·max(0, G − 0.5(R+B)), 0, 1) — luminance + '
                      f'excess-green 보정(식생 NIR 밝음 근사). 로더가 3채널로 복제',
            '🔴 proxy': 'RGB 결정론 변환 — 실측 NIR 밴드 아님. 논문/문서에 proxy 로 명기할 것'}
    meta.update({
        'tool': 'tools/p50_gen_pseudomodal.py',
        'tool_version': TOOL_VERSION,
        'created': meta.get('created', time.strftime('%Y-%m-%dT%H:%M:%S')),
        'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'design_doc': '.claude_logs/decisions/'
                      '2026-08-17-p50-map-modal-alignment-pretraining-proposal.md',
        'source_corpus': str(args.imagenet),
        'num_requested': len(files),
        'num_complete': complete,
        'image_size': args.size,
        'modals_selected': sel,
        'depth_backend_requested': args.depth_backend,
        'depth_backend_this_run': now_backend,
        'per_worker': stats,
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                     'torch': torch.__version__, 'pillow': Image.__version__},
        'args': vars(args),
    })
    if use_backend:
        meta['depth_backend'] = now_backend
    else:
        meta.setdefault('depth_backend', now_backend)   # 캐시 depth 원 출처 보존
    meta['modalities'] = {**meta.get('modalities', {}), **mod_meta}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[P50-gen] 완료 — {complete}/{len(files)} 세트, meta={meta_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
