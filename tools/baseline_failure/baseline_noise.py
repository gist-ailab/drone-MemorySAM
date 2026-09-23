"""기준선(DGFusion/CAFuser) NM(노이즈 결측) 훅용 raw 공간 등가 함수 (2026-09-23).

판정 세션이 승인한 유도식을 그대로 구현한다. 핵심 결정은 **재구현하지 않고 우리
`tools/missing_modality_eval.py` 의 실제 함수(`gaussian_noise`·`sp_noise`)를 그대로
불러 쓰는 것**이다. 그 함수들은 정규화된 텐서에서 정의되는데, 기준선의 zero-modal
훅(`d2_zero_modality.patch`)은 **정규화 전(raw) 배치 텐서**를 다룬다. 그래서

    raw → 정규화((raw-mean)/std) → 우리 함수 적용 → 역정규화(×std+mean) → raw

로 감싼다. affine 변환은 순서·최댓값·최솟값을 보존하므로, S&P 의 amin/amax 극값도
그대로 등가다(채널마다 mean·std 가 달라도 마찬가지 — 정규화를 먼저 하고 나서
amin/amax 를 구하기 때문에 채널별 스케일 차이가 이미 반영돼 있다).

이 설계라서 "같은 수식인가" 검증은 재구현 오류 가능성이 없고, 남는 것은 이 래퍼의
affine 왕복이 손실 없는지뿐이다(`verify_baseline_noise.py` 가 그것과 우리 함수를
직접 호출한 결과를 1e-4 이내로 대조한다).

의존은 torch 와 우리 `tools/missing_modality_eval.py` 뿐이다(그 파일은 `tools.
baseline_failure.common` 을 통해 `semseg` 를 시도해 보고 실패하면 내장 클래스 사본으로
넘어가므로, `semseg` 가 없는 기준선 저장소에서도 그대로 동작한다 — 2026-09-19 이래
`cell_map.py`·`far_range_diag.py` 를 기준선 저장소에서 이미 이렇게 돌렸다).
"""
import torch

from tools.missing_modality_eval import gaussian_noise, sp_noise


def _norm(raw, mean, std):
    """raw: (C,H,W). mean, std: (C,) 또는 (C,1,1) 텐서. 반환: 정규화된 (C,H,W)."""
    mean_ = mean.to(dtype=raw.dtype, device=raw.device).reshape(-1, 1, 1)
    std_ = std.to(dtype=raw.dtype, device=raw.device).reshape(-1, 1, 1)
    return (raw - mean_) / std_, mean_, std_


def raw_sp_noise(raw, mean, std, density, generator):
    """raw (C,H,W) 에 S&P 를 걸어 raw 공간으로 돌려준다.

    density: 우리 도구의 nm_density 와 같은 정의(채널 공통 위치를 이 비율로 치환,
    절반 salt·절반 pepper). generator 는 CPU torch.Generator(재현용, 이어 쓴다).
    """
    norm, mean_, std_ = _norm(raw, mean, std)
    noised, mask = sp_noise(norm.unsqueeze(0), density, generator)
    return noised.squeeze(0) * std_ + mean_, mask.squeeze(0)


def raw_gaussian_noise(raw, mean, std, sigma, generator):
    """raw (C,H,W) 에 가법 Gaussian 을 걸어 raw 공간으로 돌려준다.

    sigma 는 우리 도구의 std 와 같은 정규화-공간 값(기본 0.2). 전 화소 적용.
    """
    norm, mean_, std_ = _norm(raw, mean, std)
    noised, mask = gaussian_noise(norm.unsqueeze(0), sigma, generator)
    return noised.squeeze(0) * std_ + mean_, mask.squeeze(0)
