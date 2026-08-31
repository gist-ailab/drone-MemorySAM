#!/usr/bin/env python
"""[P50-EXT] Phase1 스모크 — 편광(aolp/dolp)·NIR proxy 생성기 확장 + MCubeS 팔 사전학습.

실행 (GPU·네트워크 불요 — 합성 미니 RGB+가짜 depth 8장):
    python tools/smoke_p50ext.py

검사 항목 (지시문 §6 대응)
  ① 기존 4모달 경로 byte-동일 — --modals 기본값 ≡ 'depth,lidar,event' 명시 런
  ② aolp_sin/cos·dolp = uint8 PNG 단채널 — PNG 경유 역양자화 값역([-1,1]/[0,1])·
     nir PNG 저장/재로드 shape 일치 + 라운드트립: 생성 시 float 원값 vs PNG 재로드
     복원값의 |Δ|max ≤ 1/255 + ε (양자화 반 스텝)
  ③ normal 유도 수치 유한 (단위벡터)
  ④ pretrain --pretrain-modals rgb,aolp,dolp,nir — 2-step forward/backward 유한
  ⑤ 증분 모드 — depth 캐시 재사용, depth 재생성(백엔드 기동) 0회 + 기존 파일 무변경

기존 경로 회귀(smoke_p50.py 전체)는 별도 실행으로 확인한다.
"""
from __future__ import annotations

import hashlib
import json
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
N_IMG = BASE.N_IMG          # 8
LEGACY_SUBDIRS = ('rgb', 'depth', 'lidar', 'event')
EXT_SUBDIRS = ('aolp_sin', 'aolp_cos', 'dolp', 'nir')
FAILS: list = []


def check(name: str, ok: bool, detail: str = ''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)
    return ok


def gen(out: Path, src: Path, modals: str = None) -> int:
    """합성 백엔드 미니 생성 런(공통 인자 고정, CPU)."""
    argv = ['--imagenet', str(src), '--out', str(out), '--n', '0',
            '--size', str(SIZE), '--batch', '4', '--gpu', '',
            '--depth-backend', 'synthetic', '--lidar-beams', '16',
            '--lidar-az-bins', '64', '--log-interval', '100']
    if modals is not None:
        argv += ['--modals', modals]
    return GEN.main(argv)


def tree_hash(root: Path, subdirs) -> dict:
    """{상대경로: sha256} — 산출 내용 byte-동일 판정용(① 가드)."""
    h = {}
    for sub in subdirs:
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file():
                h[f"{sub}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return h


def tree_state(root: Path, subdirs) -> dict:
    """{상대경로: (mtime_ns, sha256)} — 파일 무변경(내용+타임스탬프) 판정용(⑤)."""
    st = {}
    for sub in subdirs:
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file():
                st[f"{sub}/{p.name}"] = (p.stat().st_mtime_ns,
                                         hashlib.sha256(p.read_bytes()).hexdigest())
    return st


# ═══════════════════════════════════════════════════════════════════════════
def test_byte_identical(work: Path, src: Path) -> Path:
    print("\n[1] 기존 4모달 경로 byte-동일 (--modals 기본값 가드)")
    a, b = work / 'g_default', work / 'g_explicit'
    rc1 = gen(a, src)                          # --modals 생략 → 기본값
    rc2 = gen(b, src, modals='depth,lidar,event')
    check('1a 두 런 종료코드 0', rc1 == 0 and rc2 == 0, f"rc={rc1},{rc2}")
    ha, hb = tree_hash(a, LEGACY_SUBDIRS), tree_hash(b, LEGACY_SUBDIRS)
    check('1b 산출 4모달 트리 sha256 전부 동일',
          ha == hb and len(ha) == 4 * N_IMG,
          f"files={len(ha)} vs {len(hb)}")
    meta = json.loads((a / 'meta.json').read_text())
    check('1c meta modals_selected = 기본 4모달',
          meta.get('modals_selected') == ['depth', 'lidar', 'event'],
          str(meta.get('modals_selected')))
    return a


def test_six_modal(work: Path, src: Path) -> Path:
    print("\n[2] 6모달 신규 생성 — aolp/dolp/nir 값역·원자 파일 포맷 + [3] normal 유한")
    out = work / 'g_six'
    rc = gen(out, src, modals='depth,lidar,event,aolp,dolp,nir')
    check('2a 6모달 런 종료코드 0', rc == 0, f"rc={rc}")
    stems = (out / 'index.txt').read_text().split()
    check('2b index.txt 전 stem complete', len(stems) == N_IMG, f"stems={len(stems)}")

    s0 = stems[0]

    def gray(name):
        """uint8 단채널 PNG 원자 파일 로드 — (PIL image, H×W uint8)."""
        im = Image.open(out / name / f"{s0}.png")
        return im, np.asarray(im)

    sin_im, sin_u8 = gray('aolp_sin')
    cos_im, cos_u8 = gray('aolp_cos')
    dolp_im, dolp_u8 = gray('dolp')
    check('2c aolp/dolp PNG = H×W uint8 단채널 (3채널 미리 합침 없음)',
          sin_im.mode == 'L' and cos_im.mode == 'L' and dolp_im.mode == 'L'
          and sin_u8.ndim == 2 and cos_u8.ndim == 2 and dolp_u8.ndim == 2
          and sin_u8.shape == (SIZE, SIZE) and cos_u8.shape == (SIZE, SIZE)
          and dolp_u8.shape == (SIZE, SIZE) and sin_u8.dtype == np.uint8,
          f"mode={sin_im.mode}/{cos_im.mode}/{dolp_im.mode} "
          f"shape={sin_u8.shape} dtype={sin_u8.dtype}")
    # 값역 검사도 PNG 경유 — 디스크 바이트에서 역양자화해 판정한다
    a_sin = PRE.dequant_u8(sin_u8, -1.0, 1.0)
    a_cos = PRE.dequant_u8(cos_u8, -1.0, 1.0)
    dolp = PRE.dequant_u8(dolp_u8, 0.0, 1.0)
    check('2d aolp(PNG 경유) 값역 [-1,1] + sin²+cos² ≈ 1 (2θ 인코딩, 양자화 허용치)',
          float(a_sin.min()) >= -1.0 and float(a_sin.max()) <= 1.0
          and float(a_cos.min()) >= -1.0 and float(a_cos.max()) <= 1.0
          and float(np.abs(a_sin ** 2 + a_cos ** 2 - 1.0).max()) <= 4.0 / 255.0 + 1e-6,
          f"sin∈[{a_sin.min():.3f},{a_sin.max():.3f}] "
          f"cos∈[{a_cos.min():.3f},{a_cos.max():.3f}]")
    check('2e dolp(PNG 경유) 값역 [0,1] + 분산 존재(단조 곡선 값 살아있음)',
          float(dolp.min()) >= 0.0 and float(dolp.max()) <= 1.0
          and float(dolp.std()) > 0.0,
          f"dolp∈[{dolp.min():.3f},{dolp.max():.3f}] std={dolp.std():.4f}")

    nir_im = Image.open(out / 'nir' / f"{s0}.png")
    nir = np.asarray(nir_im)
    check('2f nir PNG 저장/재로드 — 8bit 단채널 shape 일치',
          nir_im.mode == 'L' and nir.ndim == 2
          and nir.shape == (SIZE, SIZE) and nir.dtype == np.uint8,
          f"mode={nir_im.mode} shape={nir.shape} dtype={nir.dtype}")

    # nir 유도식 재현(결정론성) — luminance + excess-green
    rgb = np.asarray(Image.open(out / 'rgb' / f"{s0}.png")).astype(np.float32) / 255.0
    want = np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                   + 0.5 * np.maximum(0.0, rgb[..., 1] - 0.5 * (rgb[..., 0] + rgb[..., 2])),
                   0.0, 1.0)
    check('2g nir ≡ 유도식(luminance + 0.5·excess-green)',
          bool(np.array_equal(nir, np.clip(want * 255.0 + 0.5, 0, 255).astype(np.uint8))))

    meta = json.loads((out / 'meta.json').read_text())
    ext_meta = {k: json.dumps(v, ensure_ascii=False)
                for k, v in meta['modalities'].items() if k in ('aolp', 'dolp', 'nir')}
    check('2h meta.json — modals_selected + 신규 3모달 전부 proxy·양자화 1/255 명시',
          meta.get('modals_selected')
          == ['depth', 'lidar', 'event', 'aolp', 'dolp', 'nir']
          and len(ext_meta) == 3
          and all('proxy' in v.lower() for v in ext_meta.values())
          and all('1/255' in str(meta['modalities'][k].get('quantization', ''))
                  for k in ('aolp', 'dolp')),
          f"keys={sorted(ext_meta)} "
          f"aolp_quant={meta['modalities'].get('aolp', {}).get('quantization', '')}")

    # 라운드트립 — 생성 시 float 원값 vs PNG 재로드 복원값. 원값은 생성 경로
    # (rgb → synthetic 백엔드 D → 법선 → proxy 유도식)를 그대로 재현해 얻는다:
    # synthetic 은 rgb 에 대해 결정론적이고 rgb PNG 는 load_square uint8 를
    # 무손실 저장한 것이므로 재현 D 는 당시 float 원본 D 와 정확히 일치한다.
    rgb_u8 = np.asarray(Image.open(out / 'rgb' / f"{s0}.png"))
    t = torch.from_numpy(np.stack([rgb_u8]).astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    d_ref = GEN.SyntheticDepth(torch.device('cpu'))(t)[0].numpy()
    s_ref, c_ref = GEN.render_aolp_proxy(GEN.surface_normal(d_ref))
    dd_ref = GEN.render_dolp_proxy(GEN.surface_normal(d_ref))
    tol = 1.0 / 255.0 + 1e-6
    d_sin, d_cos, d_dolp = (float(np.abs(s_ref - a_sin).max()),
                            float(np.abs(c_ref - a_cos).max()),
                            float(np.abs(dd_ref - dolp).max()))
    check('2i aolp/dolp 라운드트립 — float 원값 vs PNG 복원값 |Δ|max ≤ 1/255+ε',
          max(d_sin, d_cos, d_dolp) <= tol,
          f"|Δ|max sin={d_sin:.2e} cos={d_cos:.2e} dolp={d_dolp:.2e} tol={tol:.2e}")

    # [3] normal 유도 — 생성 depth(HHA) 역변환 → 법선이 유한·단위벡터
    D = GEN.depth_from_cached_hha(out / 'depth' / f"{s0}.png")
    n = GEN.surface_normal(D)
    nrm = np.linalg.norm(n, axis=-1)
    check('3a 캐시 depth 역변환 D — 유한·[0,1]',
          D is not None and bool(np.isfinite(D).all())
          and float(D.min()) >= 0.0 and float(D.max()) <= 1.0,
          f"D∈[{D.min():.3f},{D.max():.3f}]")
    check('3b surface_normal — 전 원소 유한 + 단위벡터',
          bool(np.isfinite(n).all()) and float(np.abs(nrm - 1.0).max()) < 1e-4,
          f"||n||−1 max={float(np.abs(nrm - 1.0).max()):.2e}")
    n2 = GEN.surface_normal(D)
    s2, c2 = GEN.render_aolp_proxy(n2)
    d2 = GEN.render_dolp_proxy(n2)
    s3, c3 = GEN.render_aolp_proxy(GEN.surface_normal(D))
    d3 = GEN.render_dolp_proxy(GEN.surface_normal(D))
    check('3c aolp/dolp 재현 결정론(동일 입력 → 동일 출력)',
          bool(np.array_equal(s2, s3)) and bool(np.array_equal(c2, c3))
          and bool(np.array_equal(d2, d3)))
    return out


def test_pretrain_mcubes(data: Path, work: Path):
    print("\n[4] pretrain --pretrain-modals rgb,aolp,dolp,nir (2-step fwd/bwd)")
    import yaml
    cfg = BASE.tiny_cfg()
    cfg['DATASET']['MODALS'] = ['img', 'aolp', 'dolp', 'nir']
    cfg_path = work / 'tiny_ext.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = work / 'p50ext_adapters.pth'
    rc = PRE.main(['--cfg', str(cfg_path), '--data', str(data), '--out', str(out),
                   '--pretrain-modals', 'rgb,aolp,dolp,nir',
                   '--epochs', '1', '--bs', '2', '--workers', '0', '--max-steps', '2',
                   '--img-size', str(SIZE), '--num-classes', '25', '--amp', 'off',
                   '--warmup-steps', '1', '--log-interval', '1'])
    # PRE.main 은 loss 비유한 시 RuntimeError 로 죽인다 → rc==0 이 유한성 증명
    check('4a 종료코드 0 (2-step forward/backward 유한)', rc == 0, f"rc={rc}")
    check('4b 어댑터 ckpt 저장', out.is_file())

    # 로더 stack 관행 — mcubes.py 와 동일한지 데이터에서 직접 확인
    ds = PRE.PseudoModalDataset(str(data), ['img', 'aolp', 'dolp', 'nir'], SIZE,
                                train=False)
    x = ds[0]
    check('4c aolp 3ch = [sin,cos,sin] (ch0==ch2, ch0≠ch1)',
          bool(torch.allclose(x[1, 0], x[1, 2])) and not bool(torch.allclose(x[1, 0], x[1, 1])))
    check('4d dolp·nir 3채널 복제',
          bool(torch.allclose(x[2, 0], x[2, 1])) and bool(torch.allclose(x[2, 1], x[2, 2]))
          and bool(torch.allclose(x[3, 0], x[3, 1])) and bool(torch.allclose(x[3, 1], x[3, 2])))
    # 비정규화 통과 증명 — /255·z-score 를 거치지 않았다면 디스크 PNG 의 역양자화값과
    # 정확 일치해야 한다(train=False center crop 은 (0,0) = 원본 그대로, flip 도 없음).
    stem0 = ds.stems[0]
    s_disk = PRE.dequant_u8(np.asarray(Image.open(data / 'aolp_sin' / f"{stem0}.png")), -1.0, 1.0)
    c_disk = PRE.dequant_u8(np.asarray(Image.open(data / 'aolp_cos' / f"{stem0}.png")), -1.0, 1.0)
    d_disk = PRE.dequant_u8(np.asarray(Image.open(data / 'dolp' / f"{stem0}.png")), 0.0, 1.0)
    check('4e aolp/dolp 비정규화 통과 — 디스크 PNG 역양자화값과 정확 일치',
          bool(torch.equal(x[1, 0], torch.from_numpy(s_disk)))
          and bool(torch.equal(x[1, 1], torch.from_numpy(c_disk)))
          and bool(torch.equal(x[2, 0], torch.from_numpy(d_disk)))
          and float(x[1].abs().max()) <= 1.0 + 1e-6 and float(x[2].max()) <= 1.0 + 1e-6,
          f"sin eq={bool(torch.equal(x[1,0], torch.from_numpy(s_disk)))} "
          f"dolp eq={bool(torch.equal(x[2,0], torch.from_numpy(d_disk)))}")

    # 기본값 가드 — --pretrain-modals 미지정 시 cfg MODALS 그대로 로드(종전 동작)
    ds_legacy = PRE.PseudoModalDataset(str(data), BASE.MODALS, SIZE, train=False)
    check('4f 기본(미지정) 모달 = cfg 4모달 그대로 로드', len(ds_legacy) == N_IMG,
          f"samples={len(ds_legacy)}")


def test_incremental(src: Path, base_corpus: Path):
    print("\n[5] 증분 모드 — 기존 4모달 코퍼스에 aolp/dolp/nir 만 추가")
    before = tree_state(base_corpus, LEGACY_SUBDIRS)
    check('5a 전제 — 증분 전 기존 파일 32개(4모달×8)', len(before) == 4 * N_IMG,
          f"files={len(before)}")

    # depth 백엔드 기동(=depth 재생성) 0회 를 직접 계수한다
    calls = []
    orig_build = GEN.build_depth_backend

    def counting_build(*a, **k):
        calls.append(1)
        return orig_build(*a, **k)

    GEN.build_depth_backend = counting_build
    try:
        rc = gen(base_corpus, src, modals='aolp,dolp,nir')
    finally:
        GEN.build_depth_backend = orig_build
    check('5b 증분 런 종료코드 0 + depth 백엔드 기동 0회(재생성 0회)',
          rc == 0 and len(calls) == 0, f"rc={rc} backend_builds={len(calls)}")

    after = tree_state(base_corpus, LEGACY_SUBDIRS)
    check('5c 기존 4모달 파일 전부 무변경 (mtime+sha256)', before == after,
          f"changed={sorted(set(before) ^ set(after))[:3]}")

    counts = {sub: len(list((base_corpus / sub).glob('*.png'))) for sub in EXT_SUBDIRS}
    check('5d 신규 4디렉토리 × 8장 생성', all(v == N_IMG for v in counts.values()),
          str(counts))

    # 증분(캐시 D 유도) 결과도 PNG 경유 값역·유한성 준수
    s0 = (base_corpus / 'index.txt').read_text().split()[0]
    a_sin = PRE.dequant_u8(np.asarray(Image.open(base_corpus / 'aolp_sin' / f"{s0}.png")), -1.0, 1.0)
    a_cos = PRE.dequant_u8(np.asarray(Image.open(base_corpus / 'aolp_cos' / f"{s0}.png")), -1.0, 1.0)
    dolp = PRE.dequant_u8(np.asarray(Image.open(base_corpus / 'dolp' / f"{s0}.png")), 0.0, 1.0)
    check('5e 증분 aolp/dolp — 유한 + 값역([-1,1]/[0,1])',
          bool(np.isfinite(a_sin).all() and np.isfinite(a_cos).all()
               and np.isfinite(dolp).all())
          and float(np.abs(a_sin).max()) <= 1.0 and float(np.abs(a_cos).max()) <= 1.0
          and float(dolp.min()) >= 0.0 and float(dolp.max()) <= 1.0,
          f"|sin|max={float(np.abs(a_sin).max()):.3f} "
          f"dolp∈[{dolp.min():.3f},{dolp.max():.3f}]")

    meta = json.loads((base_corpus / 'meta.json').read_text())
    check('5f meta 병합 — depth 원 출처 보존 + 신규 모달 등재',
          'synthetic' in str(meta.get('depth_backend'))
          and meta.get('modals_selected') == ['aolp', 'dolp', 'nir']
          and set(meta['modalities']) >= {'rgb', 'depth', 'lidar', 'event',
                                          'aolp', 'dolp', 'nir'},
          f"depth_backend={meta.get('depth_backend')} "
          f"modals={meta.get('modals_selected')}")

    # 재실행 시 전량 skip (증분 산출물도 완결 세트로 인식)
    before2 = tree_state(base_corpus, EXT_SUBDIRS)
    rc2 = gen(base_corpus, src, modals='aolp,dolp,nir')
    after2 = tree_state(base_corpus, EXT_SUBDIRS)
    check('5g 증분 재런 = 전량 skip (파일 무변경)', rc2 == 0 and before2 == after2)


def main() -> int:
    print("=" * 72)
    print("[P50-EXT] smoke — 편광(aolp/dolp)/NIR proxy 생성 + 증분 + MCubeS 팔 사전학습")
    print("=" * 72)
    work = Path(tempfile.mkdtemp(prefix='smoke_p50ext_'))
    try:
        src = BASE.make_fake_imagenet(work / 'imagenet')
        base_corpus = test_byte_identical(work, src)
        six = test_six_modal(work, src)
        test_pretrain_mcubes(six, work)
        test_incremental(src, base_corpus)
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
