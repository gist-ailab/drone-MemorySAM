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
    tokens_to_map, map_captures_to_modalities, fit_logreg, logreg_auroc,
    frac_to_block, MixDepthHead, compute_real_eval, parse_condition_case)
from semseg.models.reliadino.quality_head import QualityHead   # noqa: E402

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


def test_real_eval_auroc():
    # 합성 per-image eta: motionblur/lidarjitter 는 clean 보다 높게 → AUROC≈1.
    modal_names = ['img', 'depth', 'event', 'lidar']
    mi_img = modal_names.index('img')
    mi_lidar = modal_names.index('lidar')
    conds, cases, rows = [], [], []
    rng = np.random.RandomState(0)
    for _ in range(6):                        # clean (sun/none)
        conds.append('sun'); cases.append('none')
        r = rng.uniform(0.0, 0.1, size=4); rows.append(r)
    for _ in range(6):                        # motionblur (cloud) → img eta 높음
        conds.append('cloud'); cases.append('motionblur')
        r = rng.uniform(0.0, 0.1, size=4); r[mi_img] = rng.uniform(0.8, 0.95); rows.append(r)
    for _ in range(6):                        # lidarjitter → lidar eta 높음
        conds.append('sun'); cases.append('lidarjitter')
        r = rng.uniform(0.0, 0.1, size=4); r[mi_lidar] = rng.uniform(0.8, 0.95); rows.append(r)
    tok = {'x': np.stack(rows)}
    scal = {'x': np.stack(rows)}
    res = compute_real_eval(tok, scal, conds, cases, ['x'], modal_names)['x']
    check(res['motionblur']['auroc'] > 0.95,
          f"motionblur(img) AUROC={res['motionblur']['auroc']:.3f} > 0.95")
    check(res['lidarjitter']['auroc'] > 0.95,
          f"lidarjitter(lidar) AUROC={res['lidarjitter']['auroc']:.3f} > 0.95")
    check(res['motionblur']['mean_eta'] > res['motionblur']['mean_eta_clean'],
          "motionblur mean_eta > clean")
    # eventlowres 표본 없음 → n_pos 0, AUROC nan
    check(res['eventlowres']['n_pos'] == 0, "eventlowres 표본 0")
    check(res['eventlowres']['auroc'] != res['eventlowres']['auroc'], "표본 없으면 nan")


def test_parse_paths():
    cases = [
        ('data/DELIVER/img/cloud/val/scene_0001_motionblur/000050_rgb_front.png',
         ('cloud', 'motionblur')),
        ('data/DELIVER/img/night/val/scene_x/000050_rgb_front.png', ('night', 'none')),
        ('data/DELIVER/img/sun/val/scn_lidarjitter/1_rgb.png', ('sun', 'lidarjitter')),
        ('data/DELIVER/img/fog/val/foo/0_rgb.png', ('fog', 'none')),
        ('data/DELIVER/img/rain/val/s_eventlowres/9_rgb.png', ('rain', 'eventlowres')),
    ]
    for path, exp in cases:
        got = parse_condition_case(path)
        check(got == exp, f"parse {path.split('/img/')[1]} → {got} (기대 {exp})")


def test_mixdepth():
    # frac→block: n_blocks 12/24 매핑
    exp24 = {0.1: 2, 0.25: 6, 0.5: 12, 0.75: 18, 1.0: 24}
    exp12 = {0.1: 1, 0.25: 3, 0.5: 6, 0.75: 9, 1.0: 12}
    for f, b in exp24.items():
        check(frac_to_block(f, 24) == b, f"frac {f}×24 → block {frac_to_block(f, 24)} (기대 {b})")
    for f, b in exp12.items():
        check(frac_to_block(f, 12) == b, f"frac {f}×12 → block {frac_to_block(f, 12)} (기대 {b})")
    check(frac_to_block(0.001, 12) == 1, "아주 작은 분수도 최소 블록 1")

    # softmax 가중 모듈 forward + weights 합=1
    torch.manual_seed(0)
    dim, M, F_, B, h, w = 8, 2, 3, 2, 4, 4
    head = MixDepthHead(dim, M, F_, hidden=4)
    w_init = head.weights()
    check(abs(float(w_init.sum()) - 1.0) < 1e-5, "softmax 가중 합=1")
    check(torch.allclose(w_init, torch.full((F_,), 1.0 / F_), atol=1e-6),
          "초기 가중 균등")
    flat = [torch.randn(B, dim, h, w) for _ in range(F_ * M)]
    out = head(flat)
    check(tuple(out['eta_scalar'].shape) == (B, M), f"eta_scalar shape {tuple(out['eta_scalar'].shape)}")
    check(tuple(out['eta_token'].shape) == (B, M, h, w), f"eta_token shape {tuple(out['eta_token'].shape)}")

    # 가중치까지 gradient 흐름
    head.zero_grad()
    out['eta_scalar'].sum().backward()
    check(head.logits.grad is not None, "logits.grad 존재")
    check(float(head.logits.grad.abs().sum()) > 0, "logits 로 gradient 흐름(비영)")


def _sd_equal(a, b):
    ka, kb = set(a), set(b)
    if ka != kb:
        return False
    return all(torch.allclose(a[k], b[k]) for k in ka)


def test_save_load_heads():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'h.pt'
        # QualityHead 왕복
        torch.manual_seed(1)
        h1 = QualityHead(8, 2, hidden=4)
        torch.save(h1.state_dict(), p)
        h2 = QualityHead(8, 2, hidden=4)
        check(not _sd_equal(h1.state_dict(), h2.state_dict()), "로드 전 두 헤드 상이")
        h2.load_state_dict(torch.load(p))
        check(_sd_equal(h1.state_dict(), h2.state_dict()), "QualityHead state_dict 왕복 일치")
        # MixDepthHead 왕복(가중 logits 포함)
        pm = Path(td) / 'm.pt'
        m1 = MixDepthHead(8, 2, 3, hidden=4)
        with torch.no_grad():
            m1.logits.copy_(torch.tensor([0.3, -0.7, 1.1]))
        torch.save(m1.state_dict(), pm)
        m2 = MixDepthHead(8, 2, 3, hidden=4)
        m2.load_state_dict(torch.load(pm))
        check(_sd_equal(m1.state_dict(), m2.state_dict()), "MixDepthHead state_dict 왕복 일치")
        check(torch.allclose(m1.weights(), m2.weights()), "mixdepth 가중 복원 일치")


def main():
    print("[test] tokens_to_map"); test_tokens_to_map()
    print("[test] map_captures_to_modalities"); test_map_captures()
    print("[test] logreg"); test_logreg()
    print("[test] real_eval auroc"); test_real_eval_auroc()
    print("[test] parse condition/case"); test_parse_paths()
    print("[test] mixdepth"); test_mixdepth()
    print("[test] save/load heads"); test_save_load_heads()
    print("[test] dry_run json"); test_dry_run_writes_json()
    print(f"\nALL PASS ({_n} checks)")


if __name__ == '__main__':
    main()
