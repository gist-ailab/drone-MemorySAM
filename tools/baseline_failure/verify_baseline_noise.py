#!/usr/bin/env python3
"""NM 훅 단위 검사 (판정 세션 지시 2026-09-23, 구현 전 필수 통과 조건).

같은 입력 텐서·같은 torch.Generator 시드로
  (raw → baseline_noise 훅 → 정규화)  대  (raw → 정규화 → 우리 함수 직접 호출)
를 비교해 최대 절대 오차 < 1e-4 를 S&P·Gaussian 각각 확인한다. RMM 도 같은 방식으로
`missing_modality_eval.rmm_mask`/`rmm_degrade` 와 `d2_zero_modality.patch` 의
BF_ZERO_RATIO 마스크 식이 같은 수식(rand>=ratio 유지)인지 별도로 확인한다(이쪽은
이미 같은 소스 코드를 공유하는 게 아니라 독립 재구현이었으므로 수식 자체를 대조한다).

torch·numpy 만 있으면 GPU·detectron2 없이 돈다.
"""
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure.baseline_noise import raw_gaussian_noise, raw_sp_noise  # noqa: E402
from tools.missing_modality_eval import gaussian_noise, rmm_mask, sp_noise  # noqa: E402

TOL = 1e-4


def check(name, ok, detail=""):
    print(f"{name:24s} {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def main():
    torch.manual_seed(0)
    C, H, W = 3, 17, 23   # 비정방·소수 크기로 broadcast 실수를 드러낸다
    raw = torch.rand(C, H, W) * 200.0 + 10.0     # DELIVER PIXEL_MEAN 대와 겹치는 범위
    # 실제 DELIVER DEPTH(HHA) 채널별 평균·표준편차(cafuser/config.py 실측, 채널마다 크게
    # 다르다 — 이 비대칭성이 등가식 검증의 핵심이다).
    mean = torch.tensor([110.34616, 66.50577, 117.07793])
    std = torch.tensor([44.36736295, 77.20659029, 105.55149919])

    all_pass = True

    # --- S&P ---
    for density in (0.05, 0.10, 0.20):
        gen_a = torch.Generator().manual_seed(42)
        gen_b = torch.Generator().manual_seed(42)

        raw_out, _ = raw_sp_noise(raw, mean, std, density, gen_a)
        norm_via_hook = (raw_out - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)

        norm_direct = (raw - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)
        norm_direct_out, _ = sp_noise(norm_direct.unsqueeze(0), density, gen_b)
        norm_direct_out = norm_direct_out.squeeze(0)

        diff = (norm_via_hook - norm_direct_out).abs().max().item()
        all_pass &= check(f"S&P d={density}", diff < TOL, f"max|Δ|={diff:.2e}")

    # --- Gaussian ---
    for sigma in (0.2,):
        gen_a = torch.Generator().manual_seed(7)
        gen_b = torch.Generator().manual_seed(7)

        raw_out, _ = raw_gaussian_noise(raw, mean, std, sigma, gen_a)
        norm_via_hook = (raw_out - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)

        norm_direct = (raw - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)
        norm_direct_out, _ = gaussian_noise(norm_direct.unsqueeze(0), sigma, gen_b)
        norm_direct_out = norm_direct_out.squeeze(0)

        diff = (norm_via_hook - norm_direct_out).abs().max().item()
        all_pass &= check(f"Gaussian σ={sigma}", diff < TOL, f"max|Δ|={diff:.2e}")

    # --- 밀도가 실제로 그 값 근처인지(대략 검사, 통계적 오차 허용 큼) ---
    density = 0.2
    gen = torch.Generator().manual_seed(1)
    _, mask = raw_sp_noise(raw, mean, std, density, gen)
    measured = mask.float().mean().item()
    all_pass &= check("S&P 밀도 근사", abs(measured - density) < 0.05,
                      f"목표 {density} 실측 {measured:.3f}")

    # --- 정규화 꺼짐(라운드트립) 검사: 노이즈 없이 raw->norm->raw 가 원본과 같은가 ---
    from tools.baseline_failure.baseline_noise import _norm
    norm_rt, mean_, std_ = _norm(raw, mean, std)
    raw_rt = norm_rt * std_ + mean_
    diff_rt = (raw_rt - raw).abs().max().item()
    all_pass &= check("정규화 왕복", diff_rt < TOL, f"max|Δ|={diff_rt:.2e}")

    # --- RMM 마스크 수식 대조: rmm_mask(우리 도구) vs BF_ZERO_RATIO 식(keep=rand>=ratio) ---
    ratio = 0.5
    shape = (C, H, W)
    gen_a = torch.Generator().manual_seed(3)
    gen_b = torch.Generator().manual_seed(3)
    keep_ours = rmm_mask(shape, ratio, gen_a, device="cpu")
    # d2_zero_modality.patch 의 BF_ZERO_RATIO 식(train_net.py:834 부근)을 그대로 옮긴다.
    keep_baseline = (torch.rand(shape, generator=gen_b) >= ratio).to(torch.float32)
    diff_rmm = (keep_ours - keep_baseline).abs().max().item()
    all_pass &= check("RMM 마스크 식", diff_rmm < TOL, f"max|Δ|={diff_rmm:.2e}")

    print()
    print("전체:", "PASS" if all_pass else "FAIL — 구현 전에 고쳐야 한다")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
