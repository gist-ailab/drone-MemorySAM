#!/usr/bin/env python
"""[P50-MAP] 합성 스모크 — meta/conventions.md §"코드 검수 파이프라인" 2단계.

실행 (GPU·네트워크 불요, tiny ViT + 합성 8장):
    python tools/smoke_p50.py

검사 항목
  A. 생성기       : 합성 미니 코퍼스 → rgb/depth/lidar/event PNG + meta.json
                    A1 4모달 전부 생성·재개(skip) 동작
                    A2 채널 왕복: PIL 저장본을 torchvision.io.read_image 로 읽었을 때
                       채널 순서가 보존되는가 (BGR 뒤집힘 사고 차단)
                    A3 lidar 가 **희소**한가 (dense depth 복사본이면 정렬 학습이 무의미)
                    A4 event 가 ±극성 2채널 + ch2==0 인가
                    A5 depth(HHA) 가 유한·uint8·3채널인가
  B. 마스킹       : MultiMAE 예산이 (1-ratio) 를 맞추고, 모달별로 **다르게** 갈리는가
  C. 사전학습     : 2 step 학습 → loss 유한·감소 가능, LoRA/융합/트렁크/FPN 에 grad 도달
  D. state_dict   : 산출에 recon 헤드·백본·head 가 **없고** 4그룹만 있는가
  E. 파인튠 로드  : 같은 cfg 로 지은 파인튠 모델에 로드 → unexpected == 0,
                    로드된 텐서가 실제로 값이 바뀌었는가(진짜 로드 확인)
  F. 무영향 가드  : PRETRAINED_ADAPTERS 키가 없으면 로더가 None + 파라미터 무변화
  G. 실패 가드    : 경로가 틀리면 조용히 넘기지 않고 FileNotFoundError
  H. DDP          : torchrun 2-proc (CPU/gloo)로 실제 기동 — find_unused_parameters
                    결선과 rank0 저장(module. 접두 제거)이 도는가
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

from semseg.models.reliadino import build_reliadino                       # noqa: E402
from semseg.models.reliadino.p50 import (filter_adapter_state_dict,       # noqa: E402
                                         load_pretrained_adapters,
                                         sample_modal_token_masks,
                                         token_mask_to_pixel_mask)
import tools.p50_gen_pseudomodal as GEN                                   # noqa: E402
import tools.p50_pretrain_align as PRE                                    # noqa: E402

MODALS = ['img', 'depth', 'event', 'lidar']
SIZE = 128            # tiny ViT-16 -> 8x8 토큰
N_IMG = 8
FAILS: list = []


def check(name: str, ok: bool, detail: str = ''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        FAILS.append(name)
    return ok


def tiny_cfg(pretrained_adapters: str = '', size: int = SIZE) -> dict:
    """본 config(configs/deliver/deliver_rgbdel_P46_c3only_p50map.yaml)의 tiny 판.
    모듈 토글은 대표 레시피와 같게 둔다 (router on / M2F on / P39 trunk on)."""
    m = {
        'NAME': 'ReliaDINO',
        'BACKBONE': 'P50MAP-tiny',
        'BACKBONE_TIMM': 'vit_tiny_patch16_224',
        'BACKBONE_FALLBACK': 'vit_tiny_patch16_224',
        'PRETRAINED_BACKBONE': False,
        'LORA_R': 4, 'LORA_ALPHA': None, 'FPN_DIM': 64,
        'FUSION': {'NUM_LAYERS': 1, 'NUM_HEADS': 4, 'MLP_RATIO': 2.0,
                   'AUX_HIDDEN': 32, 'AUX_CE_WEIGHT': 0.5, 'TRUNK': 'gated_mlp',
                   'ATTN_BIAS': {'ENABLE': False}},
        'CONSISTENCY': {'ENABLE': False},
        'GATE': {'ENABLE': False, 'VETO_FLOOR': {'ENABLE': False}},
        'CALIBRATION': {'ENABLE': False},
        'ROUTER': {'ENABLE': True, 'HIDDEN': 16},
        # ANCHORED 쿼리는 num_queries > num_classes(25) 를 요구한다 (m2f_head.py L102)
        'M2F': {'ENABLE': True, 'NUM_QUERIES': 30, 'NUM_LAYERS': 1, 'DIM': 64,
                'NUM_HEADS': 4, 'MLP_RATIO': 2.0, 'POINTS': 64, 'SRC': 'modal',
                'ANCHORED': True, 'POINT_QUOTA': 8},
        'P39': {'TRUNK_EXP': True, 'ARBITER': True, 'TRUNK_MODE': 'gated_mlp',
                'TRUNK_HIDDEN': 32,
                'VICREG': {'ENABLE': True, 'TOKENS': 64}},
        'P46': {'C3_PROTO': {'ENABLE': True, 'LAMBDA': 0.1}},
    }
    if pretrained_adapters:
        m['PRETRAINED_ADAPTERS'] = pretrained_adapters
    return {
        'MODEL': m,
        'DATASET': {'NAME': 'DELIVER', 'MODALS': MODALS, 'NUM_CLASSES': 25,
                    'IGNORE_LABEL': 255},
        'TRAIN': {'IMAGE_SIZE': [size, size], 'BATCH_SIZE': 1, 'DDP': False},
    }


# ═══════════════════════════════════════════════════════════════════════════
def make_fake_imagenet(root: Path) -> Path:
    """클래스 디렉토리 2개 × 4장. 구조적 패턴이라 depth/event 가 자명하게 검증된다."""
    rng = np.random.RandomState(0)
    for c in range(2):
        d = root / f"n0000000{c}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(N_IMG // 2):
            yy, xx = np.mgrid[0:180, 0:200].astype(np.float32)
            base = (0.5 + 0.5 * np.sin(xx / (6 + 3 * i)) * np.cos(yy / 11.0))
            img = np.stack([base, np.roll(base, 7, 1), 1 - base], -1)
            img = np.clip(img * 255 + rng.rand(180, 200, 3) * 12, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(d / f"n0000000{c}_{i}.JPEG")
    return root


def test_generator(work: Path) -> Path:
    print("\n[A] pseudo-모달 생성기")
    src = make_fake_imagenet(work / 'imagenet')
    out = work / 'pseudo'
    rc = GEN.main(['--imagenet', str(src), '--out', str(out), '--n', '0',
                   '--size', str(SIZE), '--batch', '4', '--gpu', '',
                   '--depth-backend', 'synthetic', '--lidar-beams', '16',
                   '--lidar-az-bins', '64', '--log-interval', '100'])
    check('A0 생성기 종료코드 0', rc == 0)

    counts = {m: len(list((out / m).glob('*.png'))) for m in GEN.MODAL_DIRS}
    check('A1 4모달 × 8장 생성', all(v == N_IMG for v in counts.values()), str(counts))
    meta = json.loads((out / 'meta.json').read_text())
    check('A1b meta.json 기록', meta['num_complete'] == N_IMG
          and 'synthetic' in meta['depth_backend'],
          f"backend={meta['depth_backend']} complete={meta['num_complete']}")
    stems = (out / 'index.txt').read_text().split()
    check('A1c index.txt', len(stems) == N_IMG)

    # 재개: 두 번째 호출은 전부 skip 이어야 한다 (mtime 으로 판별)
    before = {p: p.stat().st_mtime_ns for p in (out / 'rgb').glob('*.png')}
    GEN.main(['--imagenet', str(src), '--out', str(out), '--n', '0',
              '--size', str(SIZE), '--batch', '4', '--gpu', '',
              '--depth-backend', 'synthetic', '--lidar-beams', '16',
              '--lidar-az-bins', '64'])
    after = {p: p.stat().st_mtime_ns for p in (out / 'rgb').glob('*.png')}
    check('A1d 재개(이미 있는 세트 skip)', before == after)

    # A2 채널 왕복 — torchvision.io.read_image(=DELIVER 로더) 로 읽어 순서 확인
    from torchvision import io
    probe = np.zeros((8, 8, 3), np.uint8)
    probe[..., 0], probe[..., 1], probe[..., 2] = 10, 120, 240
    ppath = work / 'probe.png'
    Image.fromarray(probe).save(ppath)
    back = io.read_image(str(ppath))
    check('A2 채널 순서 왕복(R=10,G=120,B=240)',
          [int(back[c, 0, 0]) for c in range(3)] == [10, 120, 240],
          str([int(back[c, 0, 0]) for c in range(3)]))

    s0 = stems[0]
    lid = np.asarray(Image.open(out / 'lidar' / f"{s0}.png"))
    dens = float((lid[..., 0] > 0).mean())
    check('A3 lidar 희소(0<density<0.35)', 0.0 < dens < 0.35, f"density={dens:.3f}")
    check('A3b lidar 3채널 동일값 + 값범위 ≤ round(255*0.38)',
          bool((lid[..., 0] == lid[..., 1]).all() and (lid[..., 1] == lid[..., 2]).all()
               and lid.max() <= round(255 * GEN.LIDAR_VALUE_SCALE)),
          f"max={int(lid.max())}")

    ev = np.asarray(Image.open(out / 'event' / f"{s0}.png"))
    both = ((ev[..., 0] > 0) & (ev[..., 1] > 0)).mean()
    check('A4 event ch2==0 · 극성 배타적 · 양쪽 발화 존재',
          bool((ev[..., 2] == 0).all()) and both < 0.02
          and (ev[..., 0] > 0).any() and (ev[..., 1] > 0).any(),
          f"pos={float((ev[...,0]>0).mean()):.3f} neg={float((ev[...,1]>0).mean()):.3f} "
          f"overlap={float(both):.4f}")

    hha = np.asarray(Image.open(out / 'depth' / f"{s0}.png"))
    check('A5 depth=HHA 3채널 uint8 · 채널별 분산 존재',
          hha.shape == (SIZE, SIZE, 3) and hha.dtype == np.uint8
          and all(hha[..., c].std() > 0 for c in range(3)),
          f"std={[round(float(hha[...,c].std()),2) for c in range(3)]}")
    return out


def test_masking():
    print("\n[B] MultiMAE 마스킹")
    torch.manual_seed(0)
    b, m, n, ratio = 4, 4, 64, 0.75
    vis = sample_modal_token_masks(b, m, n, ratio, 1.0)
    check('B1 shape (M,B,N)', tuple(vis.shape) == (m, b, n), str(tuple(vis.shape)))
    tot = vis.reshape(m, b, n).sum(dim=(0, 2)).float()      # 샘플별 총 가시 토큰
    want = round((1 - ratio) * n * m)
    check('B2 총 가시 예산 == round((1-ratio)·N·M)',
          bool((tot == want).all()), f"got={tot.tolist()} want={want}")
    per = vis.float().mean(dim=2)                            # (M,B)
    spread = float(per.std(dim=0).mean())
    check('B3 모달별 가시율이 갈린다(Dirichlet)', spread > 0.02, f"std={spread:.3f}")
    check('B4 모달당 최소 1토큰 가시', bool((vis.sum(dim=2) >= 1).all()))
    pm = token_mask_to_pixel_mask(vis, 8, 8, (128, 128))
    check('B5 픽셀 마스크 shape/이진', tuple(pm.shape) == (m, b, 1, 128, 128)
          and bool(((pm == 0) | (pm == 1)).all()), str(tuple(pm.shape)))


def test_pretrain(data: Path, work: Path) -> Path:
    print("\n[C] 사전학습 2 step + [D] 산출 state_dict")
    cfg_path = work / 'tiny_pretrain.yaml'
    import yaml
    cfg_path.write_text(yaml.safe_dump(tiny_cfg(), sort_keys=False))
    out = work / 'p50_adapters.pth'
    rc = PRE.main(['--cfg', str(cfg_path), '--data', str(data), '--out', str(out),
                   '--epochs', '1', '--bs', '2', '--workers', '0',
                   '--max-steps', '2', '--img-size', str(SIZE),
                   '--num-classes', '25', '--amp', 'off', '--warmup-steps', '1',
                   '--log-interval', '1'])
    check('C1 사전학습 종료코드 0', rc == 0)
    check('C2 어댑터 ckpt 생성', out.is_file())

    ck = torch.load(out, map_location='cpu')
    sd = ck['model_state_dict']
    check('C3 p50_map 표식 + meta', ck.get('p50_map') is True
          and ck['p50_meta']['arm'] == 'multimae_cross_modal_masked_reconstruction')
    groups = {'lora': 0, 'fusion': 0, 'trunk': 0, 'fpn': 0}
    for k in sd:
        if k.endswith(('.a_q', '.b_q', '.a_v', '.b_v')):
            groups['lora'] += 1
        elif k.startswith('fusion.'):
            groups['fusion'] += 1
        elif k.startswith(('trunk_exp', 'trunk_xattn', 'trunk_gamma')):
            groups['trunk'] += 1
        elif k.startswith('fpn.'):
            groups['fpn'] += 1
    check('D1 4그룹 모두 존재', all(v > 0 for v in groups.values()), str(groups))
    bad = [k for k in sd if k.startswith(('encoder.backbone.blocks.0.attn.qkv.base',
                                          'head.', 'm2f.', 'recon.', 'p43'))
           or k.startswith('encoder.backbone.patch_embed')]
    check('D2 백본/head/M2F/recon 키 없음', not bad, str(bad[:5]))
    check('D3 그룹 합 == 전체 키 수', sum(groups.values()) == len(sd),
          f"{sum(groups.values())} vs {len(sd)}")
    return out


def test_ddp(data: Path, work: Path):
    """[H] DDP 결선 — torchrun 2-proc (CPU/gloo). find_unused_parameters 없이 죽는
    가지(router 등)가 있으므로 이 경로가 실제로 도는지 봐야 한다."""
    print("\n[H] DDP 2-proc 기동 (CPU/gloo)")
    import os as _os
    import subprocess
    import yaml
    cfg_path = work / 'tiny_ddp.yaml'
    cfg_path.write_text(yaml.safe_dump(tiny_cfg(), sort_keys=False))
    out = work / 'p50_ddp.pth'
    env = dict(_os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc_per_node=2', str(Path(__file__).resolve().parent / 'p50_pretrain_align.py'),
           '--cfg', str(cfg_path), '--data', str(data), '--out', str(out),
           '--epochs', '1', '--bs', '1', '--workers', '0', '--max-steps', '2',
           '--img-size', str(SIZE), '--num-classes', '25', '--amp', 'off',
           '--warmup-steps', '1', '--log-interval', '1']
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    ok = (r.returncode == 0 and out.is_file())
    if not ok:
        print(r.stdout[-2500:])
        print(r.stderr[-2500:])
    check('H1 torchrun 2-proc 종료코드 0 + ckpt 저장', ok, f"rc={r.returncode}")
    if out.is_file():
        sd = torch.load(out, map_location='cpu')['model_state_dict']
        check('H2 DDP 산출에 module. 접두 없음',
              not any(k.startswith('module.') for k in sd))
        check('H3 DDP 산출 키 수 == 단일GPU 산출 키 수(160)', len(sd) == 160, str(len(sd)))


def test_grad_reach(data: Path, work: Path):
    print("\n[C-2] gradient 도달 (LoRA / 융합 / 트렁크 / FPN)")
    cfg = tiny_cfg()
    base = build_reliadino(cfg, 25)
    PRE.freeze_outside_groups(base, ['lora', 'fusion', 'trunk', 'fpn'])
    net = PRE.P50AlignNet(base, 64, 0.75, 1.0)
    net.train()
    dset = PRE.PseudoModalDataset(str(data), MODALS, SIZE, train=True)
    x = torch.stack([dset[0], dset[1]], 0)
    loss, _pm, stats = net(x)
    check('C4 loss 유한 + >0', bool(torch.isfinite(loss)) and float(loss) > 0,
          f"loss={float(loss):.4f} vis={stats['visible_frac']:.3f}")
    loss.backward()

    def gmax(pred):
        ps = [p for n, p in base.named_parameters() if pred(n)]
        return max((float(p.grad.abs().sum()) if p.grad is not None else 0.0)
                   for p in ps)

    # LoRA 는 up-projection b_q/b_v 가 zero-init 이다 (encoder.MultiModalLoRAQKV).
    # 따라서 **init 시점에는** dq = (x·a_q)·b_q 가 b_q=0 이라 ∂L/∂a_q ≡ 0 이고
    # ∂L/∂b_q ≠ 0 이다 — LoRA 의 정상 동작이지 결선 불량이 아니다. 두 국면을 나눈다:
    #   C5a init : b_q/b_v 에 grad 도달 (= Q·V 두 슬라이스 모두 도통)
    #   C5b 1스텝 뒤(b≠0) : a_q/a_v 에도 grad 도달
    for name, pred in [('LoRA b_q(init)', lambda n: n.endswith('.b_q')),
                       ('LoRA b_v(init)', lambda n: n.endswith('.b_v')),
                       ('fusion', lambda n: n.startswith('fusion.layers')),
                       ('trunk', lambda n: n.startswith('trunk_exp')),
                       ('fpn', lambda n: n.startswith('fpn.'))]:
        g = gmax(pred)
        check(f'C5a {name} grad > 0', g > 0, f"|g|max={g:.3e}")
    gr = max(float(p.grad.abs().sum()) if p.grad is not None else 0.0
             for p in net.recon.parameters())
    check('C5a recon grad > 0', gr > 0, f"|g|max={gr:.3e}")
    check('C5a a_q grad == 0 (b_q zero-init, LoRA 정상)',
          gmax(lambda n: n.endswith('.a_q')) == 0.0)

    with torch.no_grad():                      # b 를 0 에서 떼어낸다 = 1스텝 뒤 상태
        for n, p in base.named_parameters():
            if n.endswith(('.b_q', '.b_v')):
                p.add_(0.05)
    net.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    loss2, _pm2, _ = net(x)
    loss2.backward()
    for name, suf in [('a_q', '.a_q'), ('a_v', '.a_v')]:
        g = gmax(lambda n, s=suf: n.endswith(s))
        check(f'C5b LoRA {name} grad > 0 (b≠0 이후)', g > 0, f"|g|max={g:.3e}")
    frozen = [n for n, p in base.named_parameters()
              if n.startswith('encoder.backbone') and 'qkv.a_' not in n
              and 'qkv.b_' not in n and p.requires_grad]
    check('C6 백본 frozen 유지', not frozen, str(frozen[:3]))


def test_finetune_load(adapters: Path, work: Path):
    print("\n[E] 파인튠 로드 / [F] 무영향 가드 / [G] 실패 가드")
    sd = torch.load(adapters, map_location='cpu')['model_state_dict']

    # E: 사전학습 어댑터를 새로 지은 파인튠 모델에 얹는다
    model = build_reliadino(tiny_cfg(), 25)
    key = next(k for k in sd if k.endswith('.b_v'))
    before = model.state_dict()[key].clone()
    info = load_pretrained_adapters(model, tiny_cfg(str(adapters))['MODEL'])
    check('E1 unexpected == 0', info['unexpected'] == 0, str(info))
    check('E2 로드 텐서 수 == 산출 키 수', info['loaded'] == len(sd))
    after = model.state_dict()[key]
    check('E3 값이 실제로 갈아끼워졌다',
          bool(torch.equal(after, sd[key])) and not torch.equal(after, before))
    missing_expected = [k for k in model.state_dict() if k not in sd]
    check('E4 missing == 모델 나머지 키(정상)',
          info['missing'] == len(missing_expected),
          f"missing={info['missing']} model_rest={len(missing_expected)}")

    # F: 키가 없으면 완전 무영향
    m2 = build_reliadino(tiny_cfg(), 25)
    h0 = hashlib.md5(b''.join(v.cpu().numpy().tobytes()
                              for v in m2.state_dict().values()
                              if v.dtype.is_floating_point)).hexdigest()
    ret = load_pretrained_adapters(m2, tiny_cfg()['MODEL'])
    h1 = hashlib.md5(b''.join(v.cpu().numpy().tobytes()
                              for v in m2.state_dict().values()
                              if v.dtype.is_floating_point)).hexdigest()
    check('F1 PRETRAINED_ADAPTERS 키 없음 → None', ret is None)
    check('F2 파라미터 완전 무변화 (md5 동일)', h0 == h1)
    ret2 = load_pretrained_adapters(m2, {'PRETRAINED_ADAPTERS': ''})
    check('F3 빈 문자열도 no-op', ret2 is None)

    # G: 틀린 경로는 죽어야 한다 (조용한 무사전학습 폴백 금지)
    try:
        load_pretrained_adapters(m2, {'PRETRAINED_ADAPTERS': str(work / 'nope.pth')})
        check('G1 없는 경로 → FileNotFoundError', False, '예외가 안 났다')
    except FileNotFoundError:
        check('G1 없는 경로 → FileNotFoundError', True)

    # G2: 아키텍처 불일치(모르는 키) → RuntimeError
    bogus = work / 'bogus.pth'
    torch.save({'model_state_dict': dict(sd, **{'fusion.__no_such_param__':
                                                torch.zeros(1)})}, bogus)
    try:
        load_pretrained_adapters(build_reliadino(tiny_cfg(), 25),
                                 {'PRETRAINED_ADAPTERS': str(bogus)})
        check('G2 unexpected 키 → RuntimeError', False, '예외가 안 났다')
    except RuntimeError:
        check('G2 unexpected 키 → RuntimeError', True)

    # G3: 클래스 수 불일치(fusion shape) → 조용한 부분로드 금지
    try:
        load_pretrained_adapters(build_reliadino(tiny_cfg(), 19),
                                 {'PRETRAINED_ADAPTERS': str(adapters)})
        check('G3 num_classes 불일치 → 예외', False, '예외가 안 났다')
    except RuntimeError:
        check('G3 num_classes 불일치 → 예외', True)


def test_filter_helper():
    print("\n[D-2] filter_adapter_state_dict (DDP 접두 제거)")
    fake = {'module.fusion.a': torch.zeros(1), 'module.head.cls.weight': torch.zeros(1),
            'module.encoder.backbone.blocks.0.attn.qkv.a_q': torch.zeros(1),
            'module.fpn.lateral.0.0.weight': torch.zeros(1),
            'module.trunk_gamma': torch.zeros(1)}
    out = filter_adapter_state_dict(fake)
    check('D4 module. 제거 + head 제외',
          set(out) == {'fusion.a', 'encoder.backbone.blocks.0.attn.qkv.a_q',
                       'fpn.lateral.0.0.weight', 'trunk_gamma'}, str(sorted(out)))
    out2 = filter_adapter_state_dict(fake, ['lora'])
    check('D5 그룹 선택 동작', set(out2) == {'encoder.backbone.blocks.0.attn.qkv.a_q'})
    try:
        filter_adapter_state_dict(fake, ['nope'])
        check('D6 미지 그룹 → ValueError', False)
    except ValueError:
        check('D6 미지 그룹 → ValueError', True)


def main() -> int:
    print("=" * 72)
    print("[P50-MAP] smoke — pseudo-modal 생성 + 정렬 사전학습 + 파인튠 로드")
    print("=" * 72)
    work = Path(tempfile.mkdtemp(prefix='smoke_p50_'))
    try:
        data = test_generator(work)
        test_masking()
        adapters = test_pretrain(data, work)
        test_ddp(data, work)
        test_grad_reach(data, work)
        test_filter_helper()
        test_finetune_load(adapters, work)
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
