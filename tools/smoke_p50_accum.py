#!/usr/bin/env python
"""[P50-MAP] gradient accumulation(--accum) 스모크 — meta/conventions.md §"코드 검수".

실행 (GPU·네트워크 불요, tiny ViT + 합성 8장):
    python tools/smoke_p50_accum.py

검사 항목
  ①  accum=1 == 기본(플래그 없음): 같은 seed·2 step 후 어댑터 텐서가 수치 동일
     (기본값 1 은 종전과 완전 동일 경로여야 한다)
  ②  eff-batch 동일 조건 [bs=4,accum=1] vs [bs=2,accum=2]: 한 에폭의 optimizer
     step 수가 같고(2==2), 산출 loss 가 유한하다
  ③  meta.json(=p50_meta) 에 accum·eff_batch 가 올바르게 기록된다
  ④  (핵심 등가) 마스킹을 고정한 뒤 [bs=4,accum=1] vs [bs=2,accum=2] 를 1 optimizer
     step 돌리면 학습 파라미터가 allclose — 마이크로배치를 accum 만큼 나눠 누적한
     grad 가 한 번에 큰 배치를 돌린 grad 와 수학적으로 같음을 확인한다
     (퍼뮤테이션 없이 같은 seed 로 데이터 순서를 고정해 비교)
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# CPU 멀티스레드 리덕션은 실행마다 합산 순서가 미세하게 달라진다 — accum 경로의
# 수치 동일성을 검증하려면 이 잡음을 제거해야 한다(단일 스레드 = 결정적 리덕션).
torch.set_num_threads(1)

import tools.p50_gen_pseudomodal as GEN                                    # noqa: E402
import tools.p50_pretrain_align as PRE                                     # noqa: E402
from tools.smoke_p50 import MODALS, SIZE, make_fake_imagenet, tiny_cfg     # noqa: E402

FAILS: list = []


def check(name: str, ok: bool, detail: str = ''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)
    return ok


def make_data(work: Path) -> Path:
    src = make_fake_imagenet(work / 'imagenet')
    out = work / 'pseudo'
    GEN.main(['--imagenet', str(src), '--out', str(out), '--n', '0',
              '--size', str(SIZE), '--batch', '4', '--gpu', '',
              '--depth-backend', 'synthetic', '--lidar-beams', '16',
              '--lidar-az-bins', '64', '--log-interval', '100'])
    return out


def run(work: Path, data: Path, tag: str, extra: list) -> tuple:
    """PRE.main 을 돌리고 (어댑터 state_dict, p50_meta) 를 돌려준다."""
    cfg_path = work / f'cfg_{tag}.yaml'
    cfg_path.write_text(yaml.safe_dump(tiny_cfg(), sort_keys=False))
    out = work / f'adapters_{tag}.pth'
    argv = ['--cfg', str(cfg_path), '--data', str(data), '--out', str(out),
            '--workers', '0', '--img-size', str(SIZE), '--num-classes', '25',
            '--amp', 'off', '--warmup-steps', '1', '--log-interval', '1'] + extra
    rc = PRE.main(argv)
    assert rc == 0, f"{tag} 종료코드 {rc}"
    ck = torch.load(out, map_location='cpu')
    return ck['model_state_dict'], ck['p50_meta']


def allclose_sd(a: dict, b: dict, atol: float, rtol: float) -> tuple:
    """두 어댑터 state_dict 의 모든 텐서가 allclose 인가 + 최대 오차."""
    if set(a) != set(b):
        return False, float('inf'), 'key 집합 불일치'
    worst = 0.0
    for k in a:
        d = float((a[k].float() - b[k].float()).abs().max())
        worst = max(worst, d)
        if not torch.allclose(a[k].float(), b[k].float(), atol=atol, rtol=rtol):
            return False, worst, k
    return True, worst, ''


# ═══════════════════════════════════════════════════════════════════════════
def test_default_identity(work: Path, data: Path):
    print("\n[①] accum=1 == 기본(플래그 없음) — 2 step 수치 동일")
    base = ['--epochs', '1', '--bs', '2', '--max-steps', '2']
    sd_default, m_default = run(work, data, 'def', base)              # accum 미지정
    sd_accum1, m_accum1 = run(work, data, 'a1', base + ['--accum', '1'])
    # accum=1 은 loss/1·매-마이크로 step 으로 종전과 동일 코드 경로다(별도 분기 없음).
    # 남는 ~1e-8 차이는 timm attention 등 CPU 커널의 실행-간 잡음(패치 무관)이라
    # bit 가 아니라 아주 좁은 tolerance 로 수치 동일성을 확인한다.
    ok, worst, where = allclose_sd(sd_default, sd_accum1, atol=1e-6, rtol=1e-5)
    check('①-1 어댑터 텐서 수치 동일 (accum=1 경로 무변경)', ok,
          f"worst={worst:.2e}" + (f" at {where}" if not ok else ''))
    check('①-2 meta.accum 기본값 1', m_default.get('accum') == 1,
          f"got={m_default.get('accum')}")
    check('①-3 두 run gstep 동일', m_default['gstep'] == m_accum1['gstep'] == 2,
          f"{m_default['gstep']} vs {m_accum1['gstep']}")


def test_effbatch_steps(work: Path, data: Path):
    print("\n[②③] eff-batch 동일 [bs=4,accum=1] vs [bs=2,accum=2] — step 수·meta")
    # 8 샘플 한 에폭: bs4 → 2 batch = 2 step, bs2·accum2 → 4 batch = 2 step
    _sd4, m4 = run(work, data, 'bs4', ['--epochs', '1', '--bs', '4', '--accum', '1'])
    _sd2, m2 = run(work, data, 'bs2', ['--epochs', '1', '--bs', '2', '--accum', '2'])
    check('②-1 optimizer step 수 동일 (2==2)', m4['gstep'] == m2['gstep'] == 2,
          f"bs4={m4['gstep']} bs2accum2={m2['gstep']}")
    fin = (torch.isfinite(torch.tensor(m4['train_loss']))
           and torch.isfinite(torch.tensor(m2['train_loss'])))
    check('②-2 두 run train_loss 유한', bool(fin),
          f"bs4={m4['train_loss']:.4f} bs2accum2={m2['train_loss']:.4f}")
    check('③-1 meta.accum 기록', m4.get('accum') == 1 and m2.get('accum') == 2,
          f"bs4={m4.get('accum')} bs2={m2.get('accum')}")
    # eff_batch = bs × world(1) × accum → 둘 다 4
    check('③-2 meta.eff_batch == bs×world×accum (둘 다 4)',
          m4.get('eff_batch') == 4 and m2.get('eff_batch') == 4,
          f"bs4={m4.get('eff_batch')} bs2={m2.get('eff_batch')}")


def test_accum_equivalence(work: Path, data: Path):
    print("\n[④] 핵심 등가 — 마스킹 고정, 1 step 후 파라미터 allclose")
    # masked_recon_loss 는 배치 전체의 masked 픽셀 수로 정규화한다. 따라서 모든
    # 샘플이 **동일한 마스크**(동일 masked 수)를 가질 때만 배치-크기와 무관하게
    # 샘플별 grad 기여가 정의된다 → accum 등가를 그 조건에서 검증한다.
    def fixed_masks(batch, num_modalities, num_tokens, mask_ratio=0.75,
                    alpha=1.0, device=None, generator=None):
        m, b, n = int(num_modalities), int(batch), int(num_tokens)
        k = max(1, int(round((1.0 - float(mask_ratio)) * n)))
        vis = torch.zeros(m, b, n, dtype=torch.bool, device=device or 'cpu')
        vis[..., :k] = True                       # 모든 (modal, sample) 에 동일 마스크
        return vis

    orig = PRE.sample_modal_token_masks
    PRE.sample_modal_token_masks = fixed_masks
    try:
        # limit 4·순차 aligned: 같은 seed → 같은 perm → 두 run 이 같은 4 샘플을 같은
        # 순서로 본다. bs4 는 1 batch, bs2·accum2 는 2 batch 를 1 step 으로 묶는다.
        common = ['--epochs', '1', '--limit', '4', '--max-steps', '1']
        sd4, _ = run(work, data, 'eqv4', common + ['--bs', '4', '--accum', '1'])
        sd2, _ = run(work, data, 'eqv2', common + ['--bs', '2', '--accum', '2'])
    finally:
        PRE.sample_modal_token_masks = orig

    # 4-way 합산과 (2+2)-way 합산은 수학적으로 같지만 부동소수 합산 순서가 달라
    # 1 step AdamW 후 ~1e-5 수준의 차이가 남는다 — 알고리즘적 등가이므로 그 폭을 허용.
    ok, worst, where = allclose_sd(sd4, sd2, atol=5e-5, rtol=1e-3)
    check('④-1 1 step 후 학습 파라미터 allclose', ok,
          f"worst={worst:.2e}" + (f" at {where}" if not ok else ''))
    # 실제로 학습이 일어났는지(전부 0 이 아닌 변화) 최소 검증 — 초기값 대비.
    base_model = PRE.build_reliadino(tiny_cfg(), 25)
    init = PRE.filter_adapter_state_dict(base_model.state_dict(),
                                         PRE.DEFAULT_ADAPTER_GROUPS)
    moved = any(not torch.allclose(sd4[k].float(), init[k].float())
                for k in sd4 if k in init)
    check('④-2 파라미터가 실제로 갱신됨(초기값과 다름)', moved)


def main() -> int:
    print("=" * 72)
    print("[P50-MAP] smoke — gradient accumulation (--accum)")
    print("=" * 72)
    work = Path(tempfile.mkdtemp(prefix='smoke_p50_accum_'))
    try:
        data = make_data(work)
        test_default_identity(work, data)
        test_effbatch_steps(work, data)
        test_accum_equivalence(work, data)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\n" + "=" * 72)
    if FAILS:
        print(f"❌ {len(FAILS)} FAIL: {FAILS}")
        return 1
    print("✅ ALL PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
