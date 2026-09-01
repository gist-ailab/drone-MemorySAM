#!/usr/bin/env python
"""[P50-EXT] 다중 루트 스모크 — tools/p50_pretrain_align.py 의 --data 콤마 구분 루트.

실행 (GPU·네트워크 불요 — 합성 미니 루트 2개, 각 4장):
    python tools/smoke_p50_multiroot.py

검사 항목 (스펙 §3 대응)
  ① 단일 루트 = 기존 동일 — 콤마 없는 문자열/1원소 리스트가 종전과 같은 표본을
     낳고, 루트 안 stem 순서·개수(4)가 index.txt 열거와 일치한다.
  ② 2루트 합집합 = 8샘플 — root_counts 가 루트별 4, 합 8, 표본 stem 이 두 루트의
     합집합과 정확히 일치한다.
  ③ stem 충돌 주입 시 명확한 에러 — 동일 stem 을 가진 루트 2개를 넘기면 조용히
     넘기지 않고 ValueError('충돌')로 세운다.
  ④ 2-step 학습 유한 — --data rootA,rootB 로 실제 2 step forward/backward 가
     유한하게 돌고, 산출 meta 에 roots/root_samples 가 기록된다.

기존 경로 회귀(smoke_p50.py / smoke_p50ext.py)는 별도 실행으로 확인한다.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.p50_gen_pseudomodal as GEN                                   # noqa: E402
import tools.p50_pretrain_align as PRE                                    # noqa: E402
import tools.smoke_p50 as BASE                                            # noqa: E402

SIZE = BASE.SIZE            # 128 (tiny ViT-16 → 8×8 토큰)
MODALS = BASE.MODALS       # ['img', 'depth', 'event', 'lidar']
N_PER_ROOT = 4             # 루트당 4장(스펙 §3)
FAILS: list = []


def check(name: str, ok: bool, detail: str = ''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)
    return ok


def make_mini_imagenet(root: Path, tag: str, n: int = N_PER_ROOT) -> Path:
    """클래스 디렉토리 1개(n_<tag>) × n장. stem = n_<tag>__<tag>_<i> 라서 tag 가
    다르면 루트 간 stem 이 disjoint 하고, tag 가 같으면 충돌한다(③ 주입용)."""
    rng = np.random.RandomState(abs(hash(tag)) % (2 ** 31))
    d = root / f"n_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        yy, xx = np.mgrid[0:180, 0:200].astype(np.float32)
        base = 0.5 + 0.5 * np.sin(xx / (6 + 3 * i)) * np.cos(yy / 11.0)
        img = np.stack([base, np.roll(base, 7, 1), 1 - base], -1)
        img = np.clip(img * 255 + rng.rand(180, 200, 3) * 12, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(d / f"{tag}_{i}.JPEG")
    return root


def gen(out: Path, src: Path) -> int:
    """합성 백엔드 미니 생성 런(기본 4모달, CPU)."""
    return GEN.main(['--imagenet', str(src), '--out', str(out), '--n', '0',
                     '--size', str(SIZE), '--batch', '4', '--gpu', '',
                     '--depth-backend', 'synthetic', '--lidar-beams', '16',
                     '--lidar-az-bins', '64', '--log-interval', '100'])


def index_stems(root: Path) -> list:
    return [s for s in (root / 'index.txt').read_text().splitlines() if s.strip()]


def build_corpora(work: Path):
    """루트 A/B(disjoint stem) + C(A 와 동일 stem, ③ 충돌 주입용)."""
    roots = {}
    for tag in ('A', 'B', 'C'):
        img_tag = tag if tag != 'C' else 'A'          # C 는 A 와 같은 tag → 같은 stem
        src = make_mini_imagenet(work / f"imagenet_{tag}", img_tag)
        out = work / f"corpus_{tag}"
        rc = gen(out, src)
        assert rc == 0, f"generator rc={rc} for {tag}"
        roots[tag] = out
    return roots['A'], roots['B'], roots['C']


# ═══════════════════════════════════════════════════════════════════════════
def test_single_root(rootA: Path):
    print("\n[1] 단일 루트 = 기존 동일 (콤마 없는 문자열 ≡ 1원소 리스트 ≡ index 열거)")
    idx = index_stems(rootA)
    check('1a index.txt 4 stem', len(idx) == N_PER_ROOT, f"stems={len(idx)}")

    ds_str = PRE.PseudoModalDataset(str(rootA), MODALS, SIZE, train=False)
    ds_list = PRE.PseudoModalDataset([rootA], MODALS, SIZE, train=False)
    check('1b 단일 루트 표본 수 == 4', len(ds_str) == N_PER_ROOT, f"len={len(ds_str)}")
    check('1c stem 순서 == index.txt 완비 열거', ds_str.stems == idx,
          f"{ds_str.stems[:2]} vs {idx[:2]}")
    check('1d 문자열/리스트 입력이 같은 표본을 낳는다',
          ds_str.stems == ds_list.stems and [str(r) for r in ds_str.roots]
          == [str(r) for r in ds_list.roots])
    check('1e root_counts = {rootA: 4}', ds_str.root_counts == {str(rootA): N_PER_ROOT},
          str(ds_str.root_counts))
    x = ds_str[0]
    check('1f 표본 텐서 shape (M,3,S,S) · 유한',
          tuple(x.shape) == (len(MODALS), 3, SIZE, SIZE) and bool(torch.isfinite(x).all()),
          str(tuple(x.shape)))


def test_union(rootA: Path, rootB: Path):
    print("\n[2] 2루트 합집합 = 8샘플")
    ds = PRE.PseudoModalDataset(f"{rootA},{rootB}", MODALS, SIZE, train=False)
    idxA, idxB = index_stems(rootA), index_stems(rootB)
    check('2a 합집합 표본 수 == 8', len(ds) == 2 * N_PER_ROOT, f"len={len(ds)}")
    check('2b root_counts 루트별 4', ds.root_counts == {str(rootA): N_PER_ROOT,
                                                        str(rootB): N_PER_ROOT},
          str(ds.root_counts))
    check('2c 표본 stem 집합 == 두 루트 합집합',
          set(ds.stems) == set(idxA) | set(idxB) and len(set(ds.stems)) == 2 * N_PER_ROOT,
          f"union={len(set(ds.stems))}")
    check('2d 두 루트 stem 이 실제 disjoint (전제)',
          set(idxA).isdisjoint(set(idxB)))
    check('2e roots 순서 보존 (A 먼저, B 다음)',
          [str(r) for r in ds.roots] == [str(rootA), str(rootB)])
    # 양쪽 루트의 표본을 실제로 읽어낼 수 있다(루트별 경로 결선 확인)
    x0, xlast = ds[0], ds[len(ds) - 1]
    check('2f 첫/끝 표본 모두 유한 텐서',
          bool(torch.isfinite(x0).all()) and bool(torch.isfinite(xlast).all())
          and tuple(x0.shape) == (len(MODALS), 3, SIZE, SIZE))


def test_collision(rootA: Path, rootC: Path):
    print("\n[3] stem 충돌 주입 → 명확한 에러")
    idxA, idxC = index_stems(rootA), index_stems(rootC)
    check('3a 전제 — A 와 C 가 동일 stem 을 가진다(충돌 조건 성립)',
          set(idxA) == set(idxC) and len(idxA) == N_PER_ROOT,
          f"A∩C={len(set(idxA) & set(idxC))}")
    try:
        PRE.PseudoModalDataset(f"{rootA},{rootC}", MODALS, SIZE, train=False)
        check('3b 충돌 루트 → ValueError', False, '예외가 안 났다')
    except ValueError as e:
        check('3b 충돌 루트 → ValueError(충돌 메시지)', '충돌' in str(e), str(e)[:80])


def test_pretrain_union(rootA: Path, rootB: Path, work: Path):
    print("\n[4] 2-step 학습 유한 (--data rootA,rootB)")
    import yaml
    cfg_path = work / 'tiny_multiroot.yaml'
    cfg_path.write_text(yaml.safe_dump(BASE.tiny_cfg(), sort_keys=False))
    out = work / 'p50_multiroot_adapters.pth'
    rc = PRE.main(['--cfg', str(cfg_path), '--data', f"{rootA},{rootB}",
                   '--out', str(out), '--epochs', '1', '--bs', '2', '--workers', '0',
                   '--max-steps', '2', '--img-size', str(SIZE), '--num-classes', '25',
                   '--amp', 'off', '--warmup-steps', '1', '--log-interval', '1'])
    # PRE.main 은 loss 비유한 시 RuntimeError 로 죽인다 → rc==0 이 유한성 증명
    check('4a 종료코드 0 (2-step forward/backward 유한)', rc == 0, f"rc={rc}")
    check('4b 어댑터 ckpt 저장', out.is_file())
    if out.is_file():
        meta = torch.load(out, map_location='cpu')['p50_meta']
        check('4c meta.roots = 두 루트',
              meta.get('roots') == [str(rootA), str(rootB)], str(meta.get('roots')))
        check('4d meta.root_samples 루트별 4 · 합 8',
              meta.get('root_samples') == {str(rootA): N_PER_ROOT, str(rootB): N_PER_ROOT}
              and meta.get('samples') == 2 * N_PER_ROOT,
              f"root_samples={meta.get('root_samples')} samples={meta.get('samples')}")


def main() -> int:
    print("=" * 72)
    print("[P50-EXT] smoke — p50_pretrain_align --data 다중 루트(합집합·충돌·2-step)")
    print("=" * 72)
    work = Path(tempfile.mkdtemp(prefix='smoke_p50_multiroot_'))
    try:
        rootA, rootB, rootC = build_corpora(work)
        test_single_root(rootA)
        test_union(rootA, rootB)
        test_collision(rootA, rootC)
        test_pretrain_union(rootA, rootB, work)
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
