#!/usr/bin/env python3
"""probe_quality_blocks.py 단위 테스트 — CUDA/데이터셋/sam2/ckpt 불필요.

pytest 미설치 환경이므로 plain python 으로 실행하고, 모두 통과하면 'ALL PASS' 출력.
검증 대상:
  1. tokens_to_map: prefix(cls+register) 토큰 제거 후 (B,C,h,w) 변환
  2. map_captures_to_modalities: 순서 보존 + 개수 불일치 assert
  3. fit_logreg/logreg_auroc: 분리 가능 데이터(AUROC>0.95)·무작위(≈0.5)
  4. 도구 --dry_run 이 block_probe_summary.json 을 쓰는지
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.probe_quality_blocks import (          # noqa: E402
    tokens_to_map, map_captures_to_modalities, fit_logreg, logreg_auroc)

_n = 0


def check(cond, msg):
    global _n
    assert cond, f"FAIL: {msg}"
    _n += 1
    print(f"  ok: {msg}")


def test_tokens_to_map():
    B, C, h, w = 2, 8, 4, 5
    n_prefix = 5                                   # 1 cls + 4 register
    N = n_prefix + h * w
    # 마지막 h*w 토큰만 특정 값으로, prefix 는 다른 값으로 채워 제거 여부 확인
    t = torch.zeros(B, N, C)
    grid = torch.arange(h * w * C, dtype=torch.float).reshape(1, h * w, C).expand(B, -1, -1)
    t[:, n_prefix:] = grid
    t[:, :n_prefix] = -999.0                        # prefix 는 버려져야 함
    m = tokens_to_map(t, h, w)
    check(tuple(m.shape) == (B, C, h, w), f"tokens_to_map shape {tuple(m.shape)}")
    check(not torch.any(m == -999.0), "prefix(cls/register) 토큰이 제거됨")
    # (B,N,C)->(B,C,h,w) 값 정합: grid[b, i, c] == m[b, c, i//w, i%w]
    expect = grid.transpose(1, 2).reshape(B, C, h, w)
    check(torch.allclose(m, expect), "토큰→격자 재배열 값 일치")
    # NHWC(4D) 입력 경로
    t4 = torch.randn(B, h, w, C)
    m4 = tokens_to_map(t4, h, w)
    check(tuple(m4.shape) == (B, C, h, w), "4D(NHWC) 입력 permute 경로")
    check(torch.allclose(m4, t4.permute(0, 3, 1, 2)), "4D 값 permute 일치")


def test_map_captures():
    M = 4
    caps = [torch.full((2, 3), float(i)) for i in range(M)]   # 모달별 다른 값
    mapped = map_captures_to_modalities(caps, M)
    check(len(mapped) == M, "매핑 개수 == M")
    for i in range(M):
        check(bool(torch.all(mapped[i] == float(i))), f"모달 {i} 순서 보존")
    raised = False
    try:
        map_captures_to_modalities(caps[:3], M)               # 개수 부족
    except AssertionError:
        raised = True
    check(raised, "개수 불일치 시 AssertionError")


def test_logreg():
    torch.manual_seed(0)
    C, n = 6, 300
    mu = torch.zeros(C)
    mu[0] = 3.0
    # 분리 가능: 클래스1 은 +mu, 클래스0 은 -mu 근방
    Xtr = torch.cat([torch.randn(n, C) - mu, torch.randn(n, C) + mu])
    ytr = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    Xva = torch.cat([torch.randn(n, C) - mu, torch.randn(n, C) + mu])
    yva = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    au = logreg_auroc(Xtr, ytr, Xva, yva, steps=300)
    check(au > 0.95, f"분리 가능 데이터 AUROC={au:.3f} > 0.95")

    # 무작위 라벨 → AUROC ≈ 0.5
    Xtr_r = torch.randn(2 * n, C)
    ytr_r = (torch.rand(2 * n) > 0.5).long()
    Xva_r = torch.randn(2 * n, C)
    yva_r = (torch.rand(2 * n) > 0.5).long()
    au_r = logreg_auroc(Xtr_r, ytr_r, Xva_r, yva_r, steps=300)
    check(0.35 < au_r < 0.65, f"무작위 라벨 AUROC={au_r:.3f} ≈ 0.5")

    # 다항(5-way) 로짓 형태 확인
    logits = fit_logreg(Xtr, ytr, Xva, 5, steps=50)
    check(tuple(logits.shape) == (2 * n, 5), f"fit_logreg 5-way 로짓 shape {tuple(logits.shape)}")


def test_dry_run_writes_json():
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, str(_REPO / 'tools' / 'probe_quality_blocks.py'),
               '--dry_run', '--out', td, '--blocks', '2', '4', '6']
        r = subprocess.run(cmd, capture_output=True, text=True)
        check(r.returncode == 0, f"--dry_run 종료코드 0 (stderr: {r.stderr[-400:]})")
        js = Path(td) / 'block_probe_summary.json'
        check(js.exists(), "block_probe_summary.json 생성됨")
        data = json.loads(js.read_text())
        check(data.get('dry_run') is True, "summary.dry_run == True")
        check(data['sources'] == ['fusion_in', 'block2', 'block4', 'block6'],
              f"특징원 목록 {data['sources']}")
        check('train' in data and 'heldout' in data, "train/heldout 결과 포함")
        check((Path(td) / 'block_probe_table.md').exists(), "마크다운 표 생성됨")


def main():
    print("[test] tokens_to_map"); test_tokens_to_map()
    print("[test] map_captures_to_modalities"); test_map_captures()
    print("[test] logreg"); test_logreg()
    print("[test] dry_run json"); test_dry_run_writes_json()
    print(f"\nALL PASS ({_n} checks)")


if __name__ == '__main__':
    main()
