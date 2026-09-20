"""
DELIVER 기준선(DGFusion / CAFuser) 재학습용 열화 커리큘럼 (2026-09-20).

DGFusion 의 DELIVER 학습 config 는 모달 드롭도 열화 증강도 꺼 둔 상태다
(`cafuser/config.py` 의 기본값 0.2 를 학습 config 가 `[0.,0.,0.,0.]` 로 덮어쓴다).
우리 모델은 열화 커리큘럼으로 학습하므로, 같은 조건에서 강건성을 비교하려면
기준선도 같은 커리큘럼으로 재학습해야 한다. 이 모듈이 그 커리큘럼의 열화 연산을 담는다.

핵심 설계.
- 모든 연산은 **정규화 이전(원본) 공간**에서 동작한다. 데이터셋 매퍼가 이미지를 읽은
  직후이자 기하 증강 앞에서 부른다.
- "지운다(erase)" 는 그 모달의 **채널별 평균값으로 채우는 것**을 뜻한다. 모델이 나중에
  `(x - mean) / std` 로 정규화하면 그 자리가 정확히 0 이 되기 때문이다
  (`dgfusion/dgfusion.py:347-348` 의 정규화). 평균값은 호출부가 넘겨 준다.
- 워커(데이터로더 서브프로세스)에서 도므로 **torch 를 쓰지 않고 numpy 와 cv2 만** 쓴다.

🔴 가우시안 노이즈(gaussian noise)와 salt-and-pepper 는 여기서 **구현하지 않는다.**
   강건성 평가 벤치(NM, noisy-modality 프로토콜)가 바로 그 두 열화를 시험 문제로 쓴다.
   학습에 같은 열화를 넣으면 시험 문제를 미리 보여 주는 셈(train/test leakage)이라
   강건성 수치가 부풀려진다. 그래서 학습 커리큘럼에서는 의도적으로 배제한다.

커리큘럼 severity 는 학습 진행도(현재 iteration / 전체 iteration)에 따라 상한이 올라간다.
그런데 severity 를 계산하려면 현재 iteration 이 필요한데, 데이터로더 워커는 학습 루프의
현재 step 을 직접 알 수 없다. `multiprocessing.Value` 를 fork 로 물려받는 방식으로 푼다
(아래 `SHARED_STEP` / `set_shared_step` / `current_severity` 참조).
"""

import logging
import os
import warnings

import cv2
import numpy as np

# 모달별 열화 후보 사전.
# - CAMERA(RGB): 패치 드롭·가우시안 블러·노출(감마). 광학 카메라에서 실제로 나타나는 열화.
# - DEPTH: 패치 드롭·홀(구멍)·블러. depth 센서는 반사·흡수로 값이 비는 홀이 특징적이다.
# - LIDAR: 패치 드롭·jitter(화소 변위). 포인트 투영의 위치 흔들림을 모사.
# - EVENT: 패치 드롭·저해상. 이벤트 카메라의 성긴 해상도를 모사.
MODAL_DEGRADATIONS = {
    "CAMERA": ("patch_drop", "gaussian_blur", "exposure"),
    "DEPTH": ("patch_drop", "depth_hole", "gaussian_blur"),
    "LIDAR": ("patch_drop", "lidar_jitter"),
    "EVENT": ("patch_drop", "event_lowres"),
}

# 매퍼가 넘기는 모달 이름을 위 사전 키로 정규화하기 위한 별칭.
_MODAL_ALIASES = {
    "RGB": "CAMERA",
    "IMG": "CAMERA",
    "IMAGE": "CAMERA",
    "CAMERA": "CAMERA",
    "DEPTH": "DEPTH",
    "D": "DEPTH",
    "LIDAR": "LIDAR",
    "L": "LIDAR",
    "EVENT": "EVENT",
    "E": "EVENT",
}


# ---------------------------------------------------------------------------
# 커리큘럼 step 공유 (데이터로더 워커 <- 학습 루프)
# ---------------------------------------------------------------------------
# 학습 쪽에서 `multiprocessing.Value('i', 0)` 을 만들어 `set_shared_step` 으로 여기에 심고,
# 매 iteration 그 값을 갱신한다. 워커는 fork(리눅스 기본 start method)로 이 전역을 그대로
# 물려받으므로, 별도 IPC 없이 워커가 현재 step 을 읽을 수 있다.
SHARED_STEP = None

# 공유 step 이 없을 때(예: 평가 경로, spawn 방식) 경고를 딱 한 번만 찍기 위한 플래그.
_WARNED_NO_STEP = False


def set_shared_step(value):
    """학습 루프가 만든 `multiprocessing.Value` 를 모듈 전역에 심는다.

    fork 로 워커가 이 전역을 물려받게 하려면, 데이터로더를 만들기 **전에** 불러야 한다.
    """
    global SHARED_STEP
    SHARED_STEP = value


def current_severity(cfg, max_iter):
    """현재 학습 진행도에 해당하는 severity 상한을 돌려준다.

    cfg 는 `DATASETS.DELIVER.DEGRADE` 노드(속성 `CURRICULUM`, `CURRICULUM_FRACTIONS` 보유).
    진행도 = 현재 step / 전체 iteration 이며, `CURRICULUM_FRACTIONS` 의 앞 구간부터 차례로
    대응하는 `CURRICULUM` 상한을 적용한다. 예: fractions [0.33,0.66,1.0], curriculum
    [0.3,0.6,1.0] 이면 진행도 0~33% 는 0.3, 33~66% 는 0.6, 그 이후는 1.0.

    공유 step 이 없으면(SHARED_STEP is None) 조용히 다른 값을 쓰지 않고 **경고를 한 번 찍은 뒤
    가장 높은 상한(커리큘럼 마지막 값 = 1.0)** 으로 동작한다. 값이 없다는 사실을 숨기지 않기
    위함이다.
    """
    curriculum = list(cfg.CURRICULUM)
    fractions = list(cfg.CURRICULUM_FRACTIONS)

    if SHARED_STEP is None:
        global _WARNED_NO_STEP
        if not _WARNED_NO_STEP:
            warnings.warn(
                "degradation.SHARED_STEP 가 설정되지 않았다 — 커리큘럼 진행도를 알 수 없어 "
                "severity 상한을 최대(1.0)로 고정한다. 학습이라면 train_net.py 의 step 훅이 "
                "먼저 set_shared_step 을 부르는지 확인하라.",
                RuntimeWarning,
            )
            _WARNED_NO_STEP = True
        return float(curriculum[-1])

    step = int(SHARED_STEP.value)
    frac = step / float(max(1, max_iter))
    for ceiling, boundary in zip(curriculum, fractions):
        if frac <= boundary:
            return float(ceiling)
    return float(curriculum[-1])


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------
def _value_range(dtype):
    """감마 보정처럼 [0,1] 정규화가 필요한 연산의 값 범위를 dtype 으로 추정한다.

    이 이미지들은 원본 공간에서 0~255 uint8 로 읽히므로 정수형이면 그 최대값을, 실수형이면
    255.0 을 쓴다(이미 0~255 스케일이라는 가정).
    """
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 255.0


def _as_channel_mean(mean, channels):
    """평균값을 길이 C 배열로 만든다(스칼라·짧은 리스트도 허용)."""
    arr = np.asarray(mean, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, channels)
    if arr.size != channels:
        raise ValueError(
            "채널별 평균 길이(%d)가 이미지 채널 수(%d)와 다르다" % (arr.size, channels))
    return arr


# ---------------------------------------------------------------------------
# 개별 열화 연산 (모두 HxWxC ndarray 를 받아 같은 dtype 의 새 배열을 돌려준다)
# ---------------------------------------------------------------------------
def patch_drop(img, mean, sev, rng, ratio=None, grid=8):
    """이미지를 grid×grid 로 나눠 비율 r 만큼의 패치를 채널별 평균으로 채운다.

    r ~ U(0, 0.75*sev). 채운 자리는 정규화 후 0 이 된다(모달 부분 결측 모사).
    smoke 테스트가 비율을 검증할 수 있도록, `ratio` 를 주면 무작위 추출 대신 그 값을 쓴다.
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    fill = _as_channel_mean(mean, channels)

    r = float(rng.uniform(0.0, 0.75 * sev)) if ratio is None else float(ratio)
    r = min(max(r, 0.0), 1.0)

    out = img.astype(np.float32, copy=True)
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)

    total = grid * grid
    n_drop = int(round(r * total))
    if n_drop <= 0:
        return img.copy()

    idx = rng.permutation(total)[:n_drop]
    for flat in idx:
        gy, gx = divmod(int(flat), grid)
        y0, y1 = ys[gy], ys[gy + 1]
        x0, x1 = xs[gx], xs[gx + 1]
        if img.ndim == 3:
            out[y0:y1, x0:x1, :] = fill
        else:
            out[y0:y1, x0:x1] = fill[0]
    return out.astype(img.dtype)


def gaussian_blur(img, mean, sev, rng, sigma=None):
    """가우시안 블러. sigma ~ U(0, 3*sev). cv2.GaussianBlur(ksize 는 sigma 에서 자동)."""
    s = float(rng.uniform(0.0, 3.0 * sev)) if sigma is None else float(sigma)
    if s <= 0.0:
        return img.copy()
    # ksize=(0,0) 이면 cv2 가 sigma 로부터 커널 크기를 정한다.
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=s, sigmaY=s)
    if blurred.ndim != img.ndim:  # 단일 채널이 (H,W) 로 눌리는 경우 복원
        blurred = blurred.reshape(img.shape)
    return blurred.astype(img.dtype)


def exposure(img, mean, sev, rng, gamma=None):
    """노출(감마) 보정. gamma 를 log 공간에서 균등하게 뽑되, [0.3, 3] 구간을 sev 로 1.0 쪽으로
    좁힌다. sev=0 이면 gamma=1.0(무변화), sev=1 이면 [0.3, 3] 전체.
    """
    lo, hi = np.log(0.3), np.log(3.0)
    if gamma is None:
        u = float(rng.uniform(sev * lo, sev * hi))  # sev 로 log 구간을 0(=gamma 1) 쪽으로 축소
        g = float(np.exp(u))
    else:
        g = float(gamma)

    vr = _value_range(img.dtype)
    norm = np.clip(img.astype(np.float32) / vr, 0.0, 1.0)
    out = np.power(norm, g) * vr
    out = np.clip(out, 0.0, vr)
    return out.astype(img.dtype)


def depth_hole(img, mean, sev, rng, max_holes=12, max_axis_frac=0.25):
    """무작위 타원/원 구멍 여러 개를 채널별 평균으로 채운다(depth 센서의 결측 홀 모사).

    개수·크기를 sev 에 비례시킨다. sev=0 이면 구멍이 없어 무변화.
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    fill = _as_channel_mean(mean, channels)

    n_holes = int(round(sev * max_holes))
    if n_holes <= 0:
        return img.copy()

    mask = np.zeros((h, w), dtype=np.uint8)
    max_axis = max(1, int(sev * max_axis_frac * min(h, w)))
    for _ in range(n_holes):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        ax = int(rng.integers(1, max_axis + 1))
        ay = int(rng.integers(1, max_axis + 1))
        angle = int(rng.integers(0, 180))
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, color=1, thickness=-1)

    out = img.astype(np.float32, copy=True)
    if img.ndim == 3:
        out[mask == 1] = fill
    else:
        out[mask == 1] = fill[0]
    return out.astype(img.dtype)


def lidar_jitter(img, mean, sev, rng, max_disp=5.0):
    """화소 위치를 작은 무작위 변위로 흔든다(remap). 변위 크기를 sev 에 비례시킨다.

    LiDAR 포인트가 이미지 평면에 투영될 때의 위치 흔들림을 모사한다. 경계는 반사로 채운다.
    """
    h, w = img.shape[:2]
    disp = float(sev) * max_disp
    if disp <= 0.0:
        return img.copy()

    grid_y, grid_x = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    dx = rng.uniform(-disp, disp, size=(h, w)).astype(np.float32)
    dy = rng.uniform(-disp, disp, size=(h, w)).astype(np.float32)
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    if out.ndim != img.ndim:
        out = out.reshape(img.shape)
    return out.astype(img.dtype)


def event_lowres(img, mean, sev, rng, max_reduction=0.9):
    """축소 후 확대(INTER_NEAREST)로 해상도를 떨어뜨린다. 축소 비율을 sev 에 비례시킨다.

    이벤트 카메라의 성긴 해상도를 모사한다. sev=0 이면 축소가 없어 무변화.
    """
    h, w = img.shape[:2]
    scale = 1.0 - float(sev) * max_reduction
    scale = min(max(scale, 1e-3), 1.0)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    if new_h == h and new_w == w:
        return img.copy()

    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if out.ndim != img.ndim:
        out = out.reshape(img.shape)
    return out.astype(img.dtype)


_DEGRADE_FUNCS = {
    "patch_drop": patch_drop,
    "gaussian_blur": gaussian_blur,
    "exposure": exposure,
    "depth_hole": depth_hole,
    "lidar_jitter": lidar_jitter,
    "event_lowres": event_lowres,
}


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def apply_degradation(img, modality, mean, sev, rng, cfg=None):
    """모달 하나에 열화 하나를 적용한다.

    모달의 후보(MODAL_DEGRADATIONS) 중 하나를 rng 로 골라 적용한다. 어떤 열화가 걸릴지·
    파라미터는 모두 rng 에서 나오므로, 같은 seed 로 만든 rng 는 같은 결과를 준다.
    sev 는 이번 샘플의 severity 상한이며, 각 함수가 내부에서 U(0, ...*sev) 로 세부값을 뽑는다.
    cfg 는 현재 필요 없지만 향후 파라미터화 여지를 위해 서명에 남긴다.
    """
    key = _MODAL_ALIASES.get(str(modality).upper())
    if key is None:
        return img  # 모르는 모달은 건드리지 않는다
    candidates = MODAL_DEGRADATIONS[key]
    choice = candidates[int(rng.integers(0, len(candidates)))]
    out = _DEGRADE_FUNCS[choice](img, mean, sev, rng)
    _record_applied(key, choice, sev, img, out)
    return out


# ---------------------------------------------------------------------------
# 적용 표식 로그 — "열화가 실제로 걸리고 있는가" 를 학습 로그에서 확인하기 위한 것이다.
# 학습을 띄운 뒤에는 못 고치므로 기동 전에 넣어 둔다(판정 세션 요청 2026-09-20).
# 데이터로더 워커마다 독립으로 세며, LOG_EVERY 번에 한 번만 찍어 로그를 더럽히지 않는다.
# 찍는 것: 모달 · 고른 연산자 · severity 상한 · 실제로 값이 바뀐 화소 비율.
# ---------------------------------------------------------------------------
LOG_EVERY = int(os.environ.get("DEGRADE_LOG_EVERY", "500"))
_APPLIED_COUNT = 0


def _record_applied(modality, op_name, sev, before, after):
    global _APPLIED_COUNT
    _APPLIED_COUNT += 1
    if LOG_EVERY <= 0 or (_APPLIED_COUNT % LOG_EVERY) != 1:
        return
    try:
        changed = float((before != after).mean())
    except Exception:
        changed = float("nan")
    logging.getLogger("dgfusion").info(
        "[degrade] pid=%d 적용 %d 회째 · 모달=%s · 연산자=%s · severity상한=%.3f · "
        "바뀐 화소 비율=%.4f",
        os.getpid(), _APPLIED_COUNT, modality, op_name, float(sev), changed)


def degrade_sample(images, means, cfg, rng, max_iter):
    """한 샘플의 여러 모달에 커리큘럼 열화를 적용한다(매퍼가 부르는 상위 진입점).

    - `cfg.ENABLED` 가 False 면 입력을 **그대로**(바이트 동일) 돌려준다.
    - 켜져 있으면 현재 진행도로 severity 상한을 구하고, 모달마다 독립적으로
      `cfg.PER_MODAL_PROB` 확률로 열화를 건다.

    images: {모달이름: HxWxC ndarray}, means: {모달이름: 채널별 평균}.
    반환: 같은 키를 가진 새 dict.
    """
    if not cfg.ENABLED:
        return images

    sev = current_severity(cfg, max_iter)
    prob = float(cfg.PER_MODAL_PROB)

    out = {}
    for modality, img in images.items():
        if img is not None and rng.random() < prob:
            mean = means[modality]
            out[modality] = apply_degradation(img, modality, mean, sev, rng, cfg)
        else:
            out[modality] = img
    return out
