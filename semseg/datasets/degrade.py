"""[P54-Q1] 합성 열화 주입기 — 품질 헤드 실현성 프로브(QAF)용.

제안서 `.claude_logs/decisions/2026-09-20-p54-quality-aware-fusion-proposal.md`
§2-1 을 그대로 구현한다. 핵심 규약:

- **주입은 정규화(Normalize) 후 배치 텐서에** 한다(`tools/missing_modality_eval.py`
  의 zero-fill 규약과 동일 — DataLoader 가 내놓는 배치는 이미 정규화된 상태다).
- 입력 = per-modal 배치 텐서 **리스트** (모달 M개, 각 (B,C,H,W)).
- 샘플당·모달당 **독립** 으로 확률 p(기본 0.5)로 열화한다. 열화 표본의 20%는
  완전 결측(0 채움, `missing_modality_eval.zero_fill` 정의 재사용).
- 반환 = (열화된 텐서 리스트, 라벨 dict). 라벨:
    presence  (B,M) ∈{0,1}   1=존재, 0=완전 결측
    severity  (B,M) ∈[0,1]   연산자 파라미터의 정규화값; 결측=1, clean=0
    mask      (B,M,H,W) 0/1  픽셀(패치 단위) 열화 여부, 입력 해상도
    mask16    (B,M,h,w) 0/1  위를 stride-16 격자로 max-pool 다운샘플
- **held-out 열화(가우시안 노이즈·salt-and-pepper)는 별도 함수**(`gaussian_noise`,
  `salt_pepper`)로 두고, 학습 경로(`Degrader(heldout=False)`)에서는 절대 호출하지
  않는다. `tools/smoke_quality_head.py` 가 monkeypatch 로 이를 강제한다.

모달 이름·순서는 하드코딩하지 않는다 — 호출자가 로더(`semseg/datasets/deliver.py`
의 `DATASET.MODALS`)에서 읽어 `modal_names` 로 넘긴다.
"""
from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor

# stride-16 = ViT/patch16 토큰 격자. 라벨 다운샘플·프로브 특징 해상도의 기준.
PATCH_STRIDE = 16

# 학습 경로에서 절대 쓰지 않는 held-out 연산자 이름(벤치 NM 열화, ImageNet-C 규칙).
HELDOUT_OPS = ("gaussian_noise", "salt_pepper")


# ===========================================================================
# 난수 헬퍼 — 전부 CPU 생성기로 뽑아 device 무관 재현성 확보(missing_modality_eval
# 와 같은 규약). 파라미터는 float, 마스크/노이즈는 CPU→device 이동.
# ===========================================================================
def _u(g: torch.Generator, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=g).item()


def _randint(g: torch.Generator, lo: int, hi: int) -> int:
    """[lo, hi] 정수(양끝 포함)."""
    return int(torch.randint(lo, hi + 1, (1,), generator=g).item())


def _choice(g: torch.Generator, seq):
    return seq[_randint(g, 0, len(seq) - 1)]


def _patchify(mask_hw: Tensor, stride: int = PATCH_STRIDE) -> Tensor:
    """(H,W) 0/1 → (h,w) 0/1, 블록 내 하나라도 1이면 1(max-pool)."""
    m = mask_hw[None, None].float()
    h = max(1, mask_hw.shape[-2] // stride)
    w = max(1, mask_hw.shape[-1] // stride)
    return F.adaptive_max_pool2d(m, (h, w))[0, 0]


# ===========================================================================
# 학습용 연산자 — 각 함수 (x(C,H,W), g) -> (x_out, severity∈[0,1], mask(H,W) 0/1)
# severity 는 연산자 파라미터를 [0,1] 로 정규화한 값이다(라벨).
# ===========================================================================
def patch_drop(x: Tensor, g: torch.Generator) -> Tuple[Tensor, float, Tensor]:
    """패치 격자를 비율 r~U(0,0.75) 만큼 0 으로. 패치 크기 16~64 px 무작위."""
    C, H, W = x.shape
    r = _u(g, 0.0, 0.75)
    ps = _randint(g, 16, 64)
    gh, gw = math.ceil(H / ps), math.ceil(W / ps)
    keep = (torch.rand(gh, gw, generator=g) >= r).float()
    drop = 1.0 - keep                                  # 1 = 드롭된 패치
    drop_full = F.interpolate(drop[None, None], size=(H, W), mode='nearest')[0, 0]
    out = x * (1.0 - drop_full).to(x.device)
    return out, r / 0.75, drop_full.to(x.device)


def gaussian_blur(x: Tensor, g: torch.Generator) -> Tuple[Tensor, float, Tensor]:
    """가우시안 블러 σ~U(0,3). 전역 열화(mask=전체 1)."""
    C, H, W = x.shape
    sigma = _u(g, 0.0, 3.0)
    if sigma < 1e-3:
        return x, 0.0, torch.zeros(H, W, device=x.device)
    k = max(3, int(2 * round(3 * sigma) + 1))          # 6σ 근방, 홀수
    out = TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])
    return out, sigma / 3.0, torch.ones(H, W, device=x.device)


def gamma_gain(x: Tensor, g: torch.Generator) -> Tuple[Tensor, float, Tensor]:
    """노출(이득) 변조 γ∈[0.3,3]. 정규화 공간이라 멱함수 대신 곱 이득으로 적용."""
    C, H, W = x.shape
    gamma = _u(g, 0.3, 3.0)
    out = x * gamma
    sev = abs(math.log(gamma)) / math.log(3.0)         # γ=1 → 0, 양끝 → ~1
    return out, min(sev, 1.0), torch.ones(H, W, device=x.device)


def color_shift(x: Tensor, g: torch.Generator) -> Tuple[Tensor, float, Tensor]:
    """RGB 채널별 상수 시프트 c~U(-1,1)(정규화 공간). RGB 전용."""
    C, H, W = x.shape
    smax = 1.0
    shift = (torch.rand(C, generator=g) * 2 - 1) * smax
    out = x + shift.view(C, 1, 1).to(x.device)
    sev = shift.abs().mean().item() / smax
    return out, min(sev, 1.0), torch.ones(H, W, device=x.device)


def depth_hole(x: Tensor, g: torch.Generator) -> Tuple[Tensor, float, Tensor]:
    """사각 홀 0 채움 + 홀 내부 스케일 노이즈. depth 전용."""
    C, H, W = x.shape
    frac = _u(g, 0.05, 0.5)                             # 홀 면적 비율
    hh = max(1, int(round(math.sqrt(frac) * H)))
    ww = max(1, int(round(math.sqrt(frac) * W)))
    top = _randint(g, 0, max(0, H - hh))
    left = _randint(g, 0, max(0, W - ww))
    mask = torch.zeros(H, W, device=x.device)
    mask[top:top + hh, left:left + ww] = 1.0
    out = x.clone()
    out[:, top:top + hh, left:left + ww] = 0.0
    # 홀 경계 밖에도 약한 스케일 노이즈(depth 센서 열화 모사)
    scale = 1.0 + (torch.rand(1, generator=g).item() - 0.5) * 0.5
    out = out * scale
    area = (hh * ww) / (H * W)
    return out, min(area / 0.5, 1.0), mask


def lidar_beamdrop(x: Tensor, g: torch.Generator) -> Tuple[Tensor, float, Tensor]:
    """행 단위 빔 드롭(0 채움) + 행 jitter(픽셀 이동). LiDAR 전용."""
    C, H, W = x.shape
    fr = _u(g, 0.0, 0.5)
    drop_rows = (torch.rand(H, generator=g) < fr)
    out = x.clone()
    mask = torch.zeros(H, W, device=x.device)
    out[:, drop_rows, :] = 0.0
    mask[drop_rows, :] = 1.0
    # 남은 행 일부를 좌우로 롤(jitter)
    shift = _randint(g, -8, 8)
    if shift != 0:
        out = torch.roll(out, shifts=shift, dims=2)
    return out, fr / 0.5, mask


def event_lowres(x: Tensor, g: torch.Generator) -> Tuple[Tensor, float, Tensor]:
    """저해상: 계수 f 로 다운샘플 후 nearest 업샘플. event 전용. 전역 열화."""
    C, H, W = x.shape
    f = _choice(g, (2, 4, 8))
    small = F.interpolate(x[None], scale_factor=1.0 / f, mode='bilinear',
                          align_corners=False, recompute_scale_factor=False)
    out = F.interpolate(small, size=(H, W), mode='nearest')[0]
    return out, (f - 1) / 7.0, torch.ones(H, W, device=x.device)


# per-modal 학습 연산자 목록(결측 제외). 모달 이름은 로더 기준(img/depth/lidar/event).
TRAIN_OPS: Dict[str, Tuple[str, ...]] = {
    'img':   ('patch_drop', 'gaussian_blur', 'gamma_gain', 'color_shift'),
    'depth': ('patch_drop', 'gaussian_blur', 'gamma_gain', 'depth_hole'),
    'lidar': ('patch_drop', 'gaussian_blur', 'lidar_beamdrop'),
    'event': ('patch_drop', 'event_lowres', 'gaussian_blur'),
}
# 모달 이름을 못 찾으면 이 일반 집합을 쓴다(하드코딩 회피, 안전한 기본).
_GENERIC_OPS = ('patch_drop', 'gaussian_blur', 'gamma_gain')


# ===========================================================================
# held-out 연산자(학습 금지) — 벤치 NM 열화. 별도 registry 로만 접근.
# ===========================================================================
def gaussian_noise(x: Tensor, g: torch.Generator,
                   modal: str = '') -> Tuple[Tensor, float, Tensor]:
    """가우시안 노이즈 σ∈{.1,.2,.5}. event 는 제외(§2-1 held-out 규칙)."""
    C, H, W = x.shape
    if modal == 'event':
        return x, 0.0, torch.zeros(H, W, device=x.device)
    sigma = _choice(g, (0.1, 0.2, 0.5))
    noise = torch.randn(x.shape, generator=g) * sigma
    out = x + noise.to(x.device)
    return out, sigma / 0.5, torch.ones(H, W, device=x.device)


def salt_pepper(x: Tensor, g: torch.Generator,
                modal: str = '') -> Tuple[Tensor, float, Tensor]:
    """salt-and-pepper 밀도 D∈{.05,.1,.2}. 정규화 공간의 이미지별 min/max 를 극값."""
    C, H, W = x.shape
    d = _choice(g, (0.05, 0.1, 0.2))
    u = torch.rand(1, H, W, generator=g).to(x.device)
    hit = (u < d).float()                              # (1,H,W) 채널 공통 위치
    salt = (torch.rand(1, H, W, generator=g).to(x.device) < 0.5).float()
    hi = x.amax(dim=(1, 2), keepdim=True)
    lo = x.amin(dim=(1, 2), keepdim=True)
    val = salt * hi + (1.0 - salt) * lo
    out = x * (1.0 - hit) + val * hit
    return out, d / 0.2, hit[0]


HELDOUT_OP_MODALS: Dict[str, Tuple[str, ...]] = {
    'img':   ('gaussian_noise', 'salt_pepper'),
    'depth': ('gaussian_noise', 'salt_pepper'),
    'lidar': ('gaussian_noise', 'salt_pepper'),
    'event': ('salt_pepper',),                          # event 가우시안 제외
}


# ===========================================================================
# [P56-A] 모달 충돌(conflict) 연산자 — 한 센서가 특정 영역에서 "확신에 차 틀린"
# 값을 내는 상황을 모사한다(반사·정반사 글레어·잘못된 내용). 학습(TRAIN_OPS)·
# held-out 어느 집합에도 넣지 않고, Degrader 의 conflict 경로에서만 직접 호출한다.
# 반환 규약은 학습 연산자와 동일: (x_out(C,H,W), severity∈[0,1], mask(H,W) 0/1).
# ===========================================================================
def _odd_le(k: int, dim: int) -> int:
    """홀수이고 dim 이하인 커널 크기(≥3). 작은 텐서에서 gaussian_blur 가 패딩으로
    죽지 않도록 커널을 이미지 변 길이 안으로 자른다."""
    k = int(k)
    if k % 2 == 0:
        k += 1
    m = dim if dim % 2 == 1 else dim - 1
    return max(3, min(k, m))


def region_conflict(x: Tensor, g: torch.Generator,
                    donor: Optional[Tensor] = None) -> Tuple[Tensor, float, Tensor]:
    """비정형 영역을 "틀렸지만 그럴듯한" 내용으로 덮는다.

    영역 마스크 = 저해상 랜덤장을 업샘플·블러한 뒤, 목표 면적비(U[0.10,0.40])에
    해당하는 분위수로 이진화한다. 경계는 ~8–16px 가우시안으로 페더링(soft alpha)해
    직사각형 경계가 남지 않게 한다. 내용 = donor(다른 표본의 같은 모달, 동일 shape)가
    있으면 그것, 없으면 x 를 좌우 반전 후 세로·가로로 변당 1/4 이상 롤한 것.
    출력 = alpha*content + (1−alpha)*x. mask = (alpha>0.5). severity = 면적비/0.40 클립.
    """
    C, H, W = x.shape
    dev = x.device
    area_frac = _u(g, 0.10, 0.40)
    # 저해상 랜덤장 → 업샘플 → 블러(부드러운 랜덤 필드). 전부 CPU 생성기에서 뽑는다.
    # 격자를 거칠게(변당 3~8칸) 잡아 영역이 반사·가림처럼 몇 개의 큰 덩어리가 되게 한다.
    # (H//16 격자는 16px 크기 얼룩 수십 개로 쪼개져 "한 영역이 틀린 상황"이 되지 않았다 —
    #  실제 DELIVER 시각화로 확인, 생각정리 검수 2026-09-29)
    cells = _randint(g, 3, 8)
    lo_h = max(2, min(cells, H))
    lo_w = max(2, min(cells, W))
    field = torch.rand(1, 1, lo_h, lo_w, generator=g)
    field = F.interpolate(field, size=(H, W), mode='bicubic', align_corners=False)
    kf = [_odd_le(19, H), _odd_le(19, W)]
    field = TF.gaussian_blur(field, kernel_size=kf, sigma=[3.0, 3.0])[0, 0]   # (H,W) CPU
    thr = torch.quantile(field.reshape(-1), 1.0 - area_frac)
    region = (field >= thr).float()
    # 경계 페더링(~8–16px) — 이진 영역을 블러해 soft alpha 를 만든다.
    feather = _randint(g, 8, 16)
    ka = [_odd_le(2 * feather + 1, H), _odd_le(2 * feather + 1, W)]
    alpha = TF.gaussian_blur(region[None, None], kernel_size=ka,
                             sigma=[max(feather / 2.0, 0.5)] * 2)[0, 0].clamp(0.0, 1.0)
    alpha = alpha.to(dev)
    if donor is not None:
        content = donor.to(dev)
    else:
        flipped = torch.flip(x, dims=[2])                       # 좌우 반전
        rh = _randint(g, max(1, H // 4), max(1, H - 1)) if H > 1 else 0
        rw = _randint(g, max(1, W // 4), max(1, W - 1)) if W > 1 else 0
        content = torch.roll(flipped, shifts=(rh, rw), dims=(1, 2))
    a = alpha.unsqueeze(0)                                       # (1,H,W) 브로드캐스트
    out = a * content + (1.0 - a) * x
    mask_hw = (alpha > 0.5).float()
    return out, min(area_frac / 0.40, 1.0), mask_hw


def specular_glare(x: Tensor, g: torch.Generator) -> Tuple[Tensor, float, Tensor]:
    """정반사 글레어(RGB 류). 1–4 개 부드러운 타원 블롭(총 면적 대략 2–15%) 안에서
    x 를 이 텐서 정규화 공간의 포화 백색(채널별 이미지 최대 + 작은 마진)으로 가우시안
    감쇠 혼합한다. 선택적으로 옅고 넓은 헤일로. mask = (혼합 가중>0.5). severity =
    면적비/0.15 클립. 정규화 상수를 하드코딩하지 않고 x 자체의 최대/최소로 백색을 정한다.
    """
    C, H, W = x.shape
    dev = x.device
    ys = torch.arange(H, dtype=torch.float32).view(H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, W)
    weight = torch.zeros(H, W)                                   # CPU 혼합 가중
    n_blobs = _randint(g, 1, 4)
    for _ in range(n_blobs):
        cy = _u(g, 0.0, float(max(H - 1, 0)))
        cx = _u(g, 0.0, float(max(W - 1, 0)))
        ay = max(_u(g, 0.03, 0.20) * H, 1.0)
        ax = max(_u(g, 0.03, 0.20) * W, 1.0)
        theta = _u(g, 0.0, math.pi)                             # 무작위 방향
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dy = ys - cy
        dx = xs - cx
        xr = dx * cos_t + dy * sin_t
        yr = -dx * sin_t + dy * cos_t
        d2 = (xr / ax) ** 2 + (yr / ay) ** 2
        weight = torch.maximum(weight, torch.exp(-0.5 * d2))    # 중심 1, 가우시안 감쇠
    if _u(g) < 0.5:                                             # 옅고 넓은 헤일로(선택)
        kh = [_odd_le(min(H, W), H), _odd_le(min(H, W), W)]
        halo = TF.gaussian_blur(weight[None, None], kernel_size=kh,
                                sigma=[max(min(H, W) / 6.0, 1.0)] * 2)[0, 0] * 0.3
        weight = torch.maximum(weight, halo)
    weight = weight.clamp(0.0, 1.0).to(dev)
    white = x.amax(dim=(1, 2), keepdim=True)                     # (C,1,1) 채널별 최대
    margin = 0.05 * (x.amax() - x.amin()).abs()
    white = white + margin
    w = weight.unsqueeze(0)                                      # (1,H,W)
    out = x * (1.0 - w) + white * w
    mask_hw = (weight > 0.5).float()
    area = float(mask_hw.mean())
    return out, min(area / 0.15, 1.0), mask_hw


# ===========================================================================
# 연산자 인덱스 공간 — 라벨 `op` (B,M) 정수 텐서가 참조한다.
#   0 = clean(열화 없음), 1 = 완전 결측, 나머지 = OPS 목록 순서(+2 오프셋).
# 학습 연산자와 held-out 연산자를 **같은 인덱스 공간**에 둔다(학습 경로에서
# held-out 은 여전히 호출하지 않지만, 분석·프로브가 동일 축으로 분해할 수 있게).
# ===========================================================================
OPS: Tuple[str, ...] = (
    'patch_drop', 'gaussian_blur', 'gamma_gain', 'color_shift',
    'depth_hole', 'lidar_beamdrop', 'event_lowres',     # 학습 연산자
    'gaussian_noise', 'salt_pepper',                    # held-out 연산자(같은 축)
    'region_conflict', 'specular_glare',                # [P56-A] 충돌 연산자(끝에 추가 — 기존 인덱스 불변)
)
OP_NAMES: Tuple[str, ...] = ('clean', 'missing') + OPS
_OP_INDEX: Dict[str, int] = {name: i for i, name in enumerate(OP_NAMES)}


def _dispatch(name: str, x: Tensor, g: torch.Generator, modal: str):
    """이름으로 모듈 전역 연산자를 찾아 호출한다 — monkeypatch(스모크 d)가 보이도록
    호출 시점에 getattr 로 해석한다. held-out 함수는 modal 인자를 받는다."""
    fn = getattr(sys.modules[__name__], name)
    if name in HELDOUT_OPS:
        return fn(x, g, modal)
    return fn(x, g)


# ===========================================================================
# Degrader — 클래스형 래퍼
# ===========================================================================
_DEFAULT_CFG = {
    'p_per_modal': 0.5,     # 샘플·모달당 열화 확률
    'missing_frac': 0.2,    # 열화 표본 중 완전 결측 비율
    # [P56-A] 모달 충돌 표본. conflict_p=0(기본)이면 아래 경로는 난수를 전혀 뽑지
    # 않아 기존 코드와 출력·생성기 상태가 bit-동일하다.
    'conflict_p': 0.0,                     # 표본이 conflict 표본일 확률
    'conflict_modals': ['img', 'depth'],   # 충돌을 넣을 후보 모달
    'conflict_glare_frac': 0.5,            # chosen=='img' 표본 중 specular_glare 비율
}


class Degrader:
    """합성 열화 주입기.

    Degrader(cfg, seed, heldout=False)
      cfg     : dict (p_per_modal, missing_frac). None 이면 기본값.
      seed    : int, 재현용.
      heldout : True 면 held-out 연산자(gaussian_noise/salt_pepper)만,
                False(학습) 면 TRAIN_OPS 만 쓴다. 학습 경로는 held-out 을
                절대 호출하지 않는다.
    """

    def __init__(self, cfg: Optional[dict] = None, seed: int = 0,
                 heldout: bool = False):
        self.cfg = dict(_DEFAULT_CFG)
        if cfg:
            self.cfg.update(cfg)
        self.heldout = bool(heldout)
        self.g = torch.Generator().manual_seed(int(seed))
        # [P56-A] 배치 1 학습에서 "다른 장면" donor 로 쓸 직전 호출의 clean 입력(모달별).
        # conflict_p==0 이면 채우지 않는다(기존 동작·메모리 불변).
        self._prev_clean: Optional[List[Tensor]] = None

    def _ops_for(self, modal: str) -> Tuple[str, ...]:
        if self.heldout:
            return HELDOUT_OP_MODALS.get(modal, HELDOUT_OPS)
        return TRAIN_OPS.get(modal, _GENERIC_OPS)

    def __call__(self, tensors: List[Tensor], modal_names: List[str]
                 ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        assert len(tensors) == len(modal_names), "모달 수와 이름 수 불일치"
        M = len(tensors)
        B, _, H, W = tensors[0].shape
        h, w = max(1, H // PATCH_STRIDE), max(1, W // PATCH_STRIDE)
        dev = tensors[0].device

        out = [t.clone() for t in tensors]
        presence = torch.ones(B, M, device=dev)
        severity = torch.zeros(B, M, device=dev)
        mask = torch.zeros(B, M, H, W, device=dev)
        op = torch.zeros(B, M, dtype=torch.long, device=dev)   # 0 = clean

        p = self.cfg['p_per_modal']
        mfrac = self.cfg['missing_frac']
        conflict_p = float(self.cfg.get('conflict_p', 0.0))
        conflict_modals = self.cfg.get('conflict_modals', ['img', 'depth'])
        glare_frac = float(self.cfg.get('conflict_glare_frac', 0.5))
        for b in range(B):
            # ── [P56-A] 충돌 표본 처리 (heldout 경로엔 없음) ──────────────────
            # conflict_p==0 이면 아래 _u 도 호출하지 않아 기존과 bit-동일하다.
            if conflict_p > 0.0 and not self.heldout and _u(self.g) < conflict_p:
                present = [nm for nm in conflict_modals if nm in modal_names]
                if present:
                    chosen = _choice(self.g, present)
                    mi = modal_names.index(chosen)
                    if chosen == 'img' and _u(self.g) < glare_frac:
                        x_out, sev, mk = specular_glare(out[mi][b], self.g)
                        op_idx = _OP_INDEX['specular_glare']
                    else:
                        if B > 1:
                            donor = tensors[mi][(b + 1) % B]
                        elif (self._prev_clean is not None
                              and self._prev_clean[mi].shape[1:] == tensors[mi].shape[1:]):
                            # 배치 1: 직전 호출의 다른 장면을 donor 로(자기 장면을 뒤집어
                            # 옮긴 내용은 도로→도로처럼 충돌이 약했다 — 시각화로 확인)
                            donor = self._prev_clean[mi][0]
                        else:
                            donor = None
                        x_out, sev, mk = region_conflict(out[mi][b], self.g, donor=donor)
                        op_idx = _OP_INDEX['region_conflict']
                    out[mi][b] = x_out
                    severity[b, mi] = float(sev)
                    mask[b, mi] = mk
                    op[b, mi] = op_idx
                    # 나머지 모달은 clean(정상 센서). presence 는 1 그대로.
                    continue
                # 후보 모달이 없으면 일반 per-modal 경로로 진행.
            for m in range(M):
                if _u(self.g) >= p:                    # 이 (샘플,모달)은 clean
                    continue
                if (not self.heldout) and (_u(self.g) < mfrac):
                    # 완전 결측(정규화 후 0 채움) — held-out 경로엔 결측 없음
                    out[m][b] = torch.zeros_like(out[m][b])
                    presence[b, m] = 0.0
                    severity[b, m] = 1.0
                    mask[b, m] = 1.0
                    op[b, m] = 1                       # 1 = 완전 결측
                    continue
                op_name = _choice(self.g, self._ops_for(modal_names[m]))
                x_out, sev, mk = _dispatch(op_name, out[m][b], self.g, modal_names[m])
                out[m][b] = x_out
                severity[b, m] = float(sev)
                mask[b, m] = mk
                op[b, m] = _OP_INDEX[op_name]

        if conflict_p > 0.0 and not self.heldout:
            self._prev_clean = [t.detach().clone() for t in tensors]
        mask16 = torch.stack(
            [torch.stack([_patchify(mask[b, m]) for m in range(M)]) for b in range(B)])
        return out, {
            'presence': presence,
            'severity': severity,
            'mask': mask,
            'mask16': mask16.to(dev),
            'op': op,
        }
