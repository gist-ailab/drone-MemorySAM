#!/usr/bin/env python3
"""tools/missing_modality_eval.py 프로토콜 계산 스모크(CPU, ~1분).

import·argparse 만 확인하지 않고, 프로토콜 계산을 실제 텐서로 assert 한다:
  (a) 조합 열거가 15개이고 전부-결측이 없다
  (b) Bernoulli 전확률 합이 1, 기대값 정규화가 옳다
  (c) zero-fill 이 정규화 후 텐서에만 적용되고 원본 배치는 불변이다
  (d) RMM 마스크 드롭 비율이 r±0.02
  (e) S&P 치환 픽셀 비율이 d±0.02
  (f) 더미 모델·합성 4모달 3장으로 CSV·summary 파일이 생성된다

실행:  python tools/smoke_missing_modality_eval.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.missing_modality_eval as mm  # noqa: E402
from tools.baseline_failure import common  # noqa: E402

M = 4
MODAL_NAMES = ["img", "depth", "event", "lidar"]
N_CLASSES = common.N_CLASSES     # 25 — write_outputs 의 class_names 길이와 정합
IGNORE = common.IGNORE_LABEL


def check_a_enumeration():
    subs = mm.enumerate_present_subsets(M)
    assert len(subs) == 15, f"조합 수 {len(subs)} != 15"
    assert all(len(s) >= 1 for s in subs), "전부-결측(present 0) 조합이 섞였다"
    assert tuple(range(M)) in subs, "전부-존재 조합이 없다"
    # 각 조합 유일
    assert len(set(subs)) == 15, "조합 중복"
    print("[a] 조합 열거 15개·전부-결측 없음 OK")


def check_b_bernoulli():
    # 전확률 합(전부-결측 포함, k=0..M) = 1
    for p in (0.2, 0.1, 0.05):
        total = sum(
            len(list(_combos(k))) * mm.bernoulli_weight(k, M, p)
            for k in range(M + 1))
        assert abs(total - 1.0) < 1e-9, f"p={p} 전확률 합 {total} != 1"
    # 모든 조합 miou 가 같으면 기대값 = 그 값(가중 정규화 검증)
    subs = mm.enumerate_present_subsets(M)
    records = [(M - len(s), 42.0) for s in subs]
    for p in (0.2, 0.1, 0.05):
        e = mm.bernoulli_expected(records, M, p)
        assert abs(e - 42.0) < 1e-9, f"정규화된 기대값 {e} != 42"
    # 열거 조합(전부-결측 제외) 원가중 합 = 1 − p^M
    for p in (0.2, 0.1, 0.05):
        s = sum(mm.bernoulli_weight(M - len(sub), M, p) for sub in subs)
        assert abs(s - (1.0 - p ** M)) < 1e-9, f"열거 가중합 {s} != 1−p^M"
    print("[b] Bernoulli 전확률 합=1·기대값 정규화 OK")


def _combos(k):
    import itertools
    return itertools.combinations(range(M), M - k) if k < M else [()]


def check_c_zero_fill():
    torch.manual_seed(0)
    base = [torch.randn(1, 3, 8, 8) for _ in range(M)]   # 정규화 후를 모사
    base_clone = [b.clone() for b in base]
    missing = (1, 3)                                     # depth, lidar 결측
    out = mm.zero_fill(base, missing)
    for i in missing:
        assert torch.count_nonzero(out[i]) == 0, f"결측 모달 {i} 가 0 이 아니다"
    for i in (0, 2):
        assert torch.equal(out[i], base[i]), f"존재 모달 {i} 가 바뀌었다"
    for i in range(M):
        assert torch.equal(base[i], base_clone[i]), f"원본 배치 {i} 가 변형됐다"
    print("[c] zero-fill 결측만 0·원본 배치 불변 OK")


def check_d_rmm_ratio():
    g = torch.Generator(); g.manual_seed(0)
    shape = (1, 3, 200, 200)
    for r in (0.25, 0.5, 0.75):
        keep = mm.rmm_mask(shape, r, g, torch.device("cpu"))
        drop = 1.0 - keep.mean().item()
        assert abs(drop - r) < 0.02, f"RMM r={r} 실측 드롭 {drop:.4f}"
    print("[d] RMM 드롭 비율 r±0.02 OK")


def check_e_sp_density():
    g = torch.Generator(); g.manual_seed(0)
    x = torch.randn(1, 3, 200, 200)
    for d in (0.05, 0.1, 0.2):
        _, mask = mm.sp_noise(x, d, g)
        ratio = mask.mean().item()
        assert abs(ratio - d) < 0.02, f"S&P d={d} 실측 치환 {ratio:.4f}"
    # 채널 공통: mask 는 (B,1,H,W)
    assert mask.shape == (1, 1, 200, 200), "S&P mask 가 채널 공통(B,1,H,W)이 아니다"
    print("[e] S&P 치환 비율 d±0.02·채널 공통 OK")


class DummyModel(torch.nn.Module):
    """입력 리스트를 받아 첫 모달의 채널 평균을 로짓으로 내는 최소 모델."""

    def forward(self, batched_input, multimask_output=True):
        x0 = batched_input[0]                            # (B,3,H,W)
        base = x0.mean(dim=1, keepdim=True)              # (B,1,H,W)
        bias = torch.linspace(0, 1, N_CLASSES).view(1, N_CLASSES, 1, 1)
        logits = base + bias                             # (B,C,H,W)
        return logits, None


def _synth_loader(n=3, H=16):
    batches = []
    rng = torch.Generator(); rng.manual_seed(1)
    for _ in range(n):
        images = [torch.randn(1, 3, H, H, generator=rng) for _ in range(M)]
        label = torch.randint(0, N_CLASSES, (H, H), generator=rng).long()
        label[0, 0] = IGNORE                             # ignore 픽셀 하나
        metas = [{"orig_h": H, "orig_w": H, "orig_label": label}]
        batches.append((images, torch.zeros(1), metas))
    return batches


def check_f_end_to_end():
    model = DummyModel().eval()
    loader = _synth_loader()
    # build_cases 는 gen_factory() 가 재현 시드가 설정된 Generator 를 반환하길 기대한다.
    cases = mm.build_cases("all", M, MODAL_NAMES, [0.5], [0.1], _mkgen)
    unpad = lambda p, h, w, model_size=None: p           # 이미 orig 크기
    argmax = lambda preds, n, adj: preds[:, :n].argmax(1)
    hists = mm.evaluate(model, loader, cases, N_CLASSES, IGNORE,
                        torch.device("cpu"), unpad, argmax)
    assert len(hists) == len(cases)
    # clean 조합 = emm k=0: 두 집계 모두 전부-존재를 clean 으로 참조
    with tempfile.TemporaryDirectory() as td:
        base, summary = mm.write_outputs(
            td, "val", cases, hists, M, MODAL_NAMES, list(common.CLASSES),
            [0.5], [0.1], "all", False, False, 0.2, "dummy.pth", "dummy.yaml")
        for fn in ("emm.csv", "rmm_r0.5.csv", "nm.csv", "summary.json", "summary.md"):
            assert (base / fn).exists(), f"산출 파일 누락: {fn}"
        assert "EMM" in summary and "RMM" in summary and "NM" in summary
        assert summary["EMM"]["n_combos"] == 15, "EMM 집계 조합 수 != 15"
        assert "E(p=0.2)" in summary["EMM"]
        # EMM csv 는 clean + 14 emm = 15 행 + 헤더
        rows = (base / "emm.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 1 + 15, f"emm.csv 행 수 {len(rows)} != 16"
    print("[f] end-to-end CSV·summary 생성·집계 OK")


def _mkgen():
    g = torch.Generator(); g.manual_seed(0)
    return g


def main():
    check_a_enumeration()
    check_b_bernoulli()
    check_c_zero_fill()
    check_d_rmm_ratio()
    check_e_sp_density()
    check_f_end_to_end()
    print("\n✅ 모든 스모크 통과")


if __name__ == "__main__":
    main()
