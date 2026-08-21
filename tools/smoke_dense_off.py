#!/usr/bin/env python3
"""tools/smoke_dense_off.py — `p39_dense_off` 추론-전용 ablation 토글 스모크.

model.py의 P39-V5 arbiter 분기에 추가된 `p39_dense_off`(dense FPN 기여 제거 →
logits = q_scaled, 즉 **쿼리 단독**)가 다음을 만족하는지 랜덤 가중치·랜덤 입력
소형 모델로 검증한다:

  1. eval / flag=False  → 변경 **이전 코드**(--ref-rev의 model.py)와 |Δ|max == 0
  2. eval / flag=True   → 출력이 실제로 바뀐다 (|Δ|max > 0)
  3. train 모드 불변    → 동일 시드로 flag False/True forward 결과가 같고,
                          HEAD 버전과도 같다 (학습 경로가 플래그를 안 본다는 증거)
  4. 두 플래그 동시 True / arbiter 부재 모델 → ValueError (조용한 실패 금지)

1·3의 "이전 코드"는 `git show <ref-rev>:semseg/models/reliadino/model.py`를
패키지 안에 임시 모듈로 풀어서 import하고, 끝나면 지운다(finally). 가중치는 두
모델 사이에서 load_state_dict로 동기화하므로 timm 랜덤 init 차이는 무관하다.

Usage:  python tools/smoke_dense_off.py [--device cpu|cuda:N] [--ref-rev auto]
종료코드 0 = 4항 전부 통과.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PKG = REPO / 'semseg' / 'models' / 'reliadino'
REF_NAME = '_smoke_ref_model'
REF_PATH = PKG / f'{REF_NAME}.py'

# 작게: ViT-S/16 @256, 2 모달, 5 클래스. arbiter + m2f 켬(토글 대상 경로).
MODEL_KW = dict(
    num_classes=5,
    modalities=('img', 'lidar'),
    backbone='vit_small_patch16_224',
    backbone_fallback='',
    pretrained=False,
    img_size=256,
    lora_r=4,
    fpn_dim=64,
    fusion_layers=1,
    fusion_heads=4,
    aux_hidden=32,
    m2f_enable=True,
    m2f_num_queries=8,
    m2f_num_layers=1,
    m2f_dim=64,
    m2f_num_heads=4,
    p39_arbiter=True,
)
IMG_HW = 256


def _show(rev):
    return subprocess.run(
        ['git', 'show', f'{rev}:semseg/models/reliadino/model.py'],
        cwd=REPO, capture_output=True, text=True, check=True).stdout


def materialize_ref(rev):
    """기준(변경 전) model.py를 패키지 내 임시 모듈로 푼다 (상대 import 유지).

    rev='auto'면 model.py를 건드린 커밋을 최근순으로 훑어 p39_dense_off가
    아직 없는 첫 버전을 고른다(최대 20개) — 커밋 후에도 재실행 가능하도록."""
    if rev != 'auto':
        src = _show(rev)
        if 'p39_dense_off' in src:
            raise RuntimeError(
                f"'{rev}'의 model.py에 이미 p39_dense_off가 있다 — 기준은 **변경 전** "
                f"리비전이어야 한다 (--ref-rev <pre-change rev>).")
        REF_PATH.write_text(src)
        return rev
    revs = subprocess.run(
        ['git', 'log', '-20', '--format=%H', '--', 'semseg/models/reliadino/model.py'],
        cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    for r in revs:
        src = _show(r)
        if 'p39_dense_off' not in src:
            REF_PATH.write_text(src)
            return r[:9]
    raise RuntimeError('최근 20개 커밋에서 p39_dense_off 이전 model.py를 못 찾았다 '
                       '— --ref-rev로 직접 지정하라.')


def build(mod, device, seed=0):
    torch.manual_seed(seed)
    m = mod.ReliaDINO(**MODEL_KW).to(device)
    return m.eval()


def make_inputs(device, n_modal=2, seed=1234):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 3, IMG_HW, IMG_HW, generator=g).to(device)
            for _ in range(n_modal)]


def fwd(model, imgs, train=False, gt=None, seed=7):
    """동일 시드에서 1회 forward. train=True면 path-dropout RNG까지 고정."""
    torch.manual_seed(seed)
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        out = model(imgs, multimask_output=True) if gt is None else \
            model(imgs, True, gt)
    model.eval()
    return out[0].detach().float()


def maxabs(a, b):
    return float((a - b).abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ref-rev', default='auto',
                    help="기준(변경 전) model.py 리비전. 'auto'=자동 탐색")
    args = ap.parse_args()
    device = torch.device(args.device)
    results = []   # (id, name, pass, detail)

    ref_rev = materialize_ref(args.ref_rev)
    print(f'[smoke] reference model.py = {ref_rev}')
    try:
        import importlib
        ref_mod = importlib.import_module(f'semseg.models.reliadino.{REF_NAME}')
        new_mod = importlib.import_module('semseg.models.reliadino.model')

        ref = build(ref_mod, device)
        new = build(new_mod, device)
        new.load_state_dict(ref.state_dict())          # 가중치 완전 동기화
        assert not new.p39_dense_off and not new.p39_query_off, 'flags must default False'
        imgs = make_inputs(device, n_modal=len(MODEL_KW['modalities']))
        gt = torch.randint(0, MODEL_KW['num_classes'],
                           (1, IMG_HW, IMG_HW), device=device)

        # --- 1. eval, flag=False == HEAD 코드 ---------------------------------
        try:
            base_ref = fwd(ref, imgs)
            base_new = fwd(new, imgs)
            d = maxabs(base_ref, base_new)
            results.append(('1', 'eval flag=False == HEAD code', d == 0.0,
                            f'|Δ|max={d:.3e}'))
        except Exception:
            results.append(('1', 'eval flag=False == HEAD code', False,
                            traceback.format_exc(limit=3)))

        # --- 2. eval, flag=True 는 실제로 바뀐다 ------------------------------
        try:
            new.p39_dense_off = True
            off_new = fwd(new, imgs)
            new.p39_dense_off = False
            d = maxabs(base_new, off_new)
            results.append(('2', 'eval flag=True changes output', d > 0.0,
                            f'|Δ|max={d:.3e}'))
        except Exception:
            new.p39_dense_off = False
            results.append(('2', 'eval flag=True changes output', False,
                            traceback.format_exc(limit=3)))

        # --- 3. train 경로 불변 ------------------------------------------------
        try:
            t_ref = fwd(ref, imgs, train=True, gt=gt)
            t_off = fwd(new, imgs, train=True, gt=gt)
            new.p39_dense_off = True
            t_on = fwd(new, imgs, train=True, gt=gt)
            new.p39_dense_off = False
            d_flag = maxabs(t_off, t_on)      # flag False vs True (train)
            d_head = maxabs(t_ref, t_on)      # HEAD vs flag=True (train)
            results.append(('3', 'train path ignores flag', d_flag == 0.0 and d_head == 0.0,
                            f'|Δ|max(flagF vs flagT)={d_flag:.3e}, '
                            f'|Δ|max(HEAD vs flagT)={d_head:.3e}'))
        except Exception:
            new.p39_dense_off = False
            results.append(('3', 'train path ignores flag', False,
                            traceback.format_exc(limit=3)))

        # --- 4. 잘못된 조합은 에러 --------------------------------------------
        detail = []
        ok4 = True
        new.p39_query_off = True
        new.p39_dense_off = True
        try:
            fwd(new, imgs)
            ok4 = False
            detail.append('both-flags: NO ERROR (실패)')
        except ValueError as e:
            detail.append(f'both-flags: ValueError ✔ ({str(e)[:40]}…)')
        except Exception as e:
            ok4 = False
            detail.append(f'both-flags: wrong exc {type(e).__name__}: {e}')
        new.p39_query_off = False
        # arbiter 부재(legacy β) 경로에서도 무시 대신 에러인지
        saved = new.arb_lambda
        new.arb_lambda = None
        try:
            fwd(new, imgs)
            ok4 = False
            detail.append('legacy(no arbiter): NO ERROR (실패)')
        except ValueError as e:
            detail.append(f'legacy(no arbiter): ValueError ✔ ({str(e)[:40]}…)')
        except Exception as e:
            ok4 = False
            detail.append(f'legacy(no arbiter): wrong exc {type(e).__name__}: {e}')
        finally:
            new.arb_lambda = saved
            new.p39_dense_off = False
        results.append(('4', 'invalid combos raise', ok4, '; '.join(detail)))
    finally:
        REF_PATH.unlink(missing_ok=True)
        for p in (PKG / '__pycache__').glob(f'{REF_NAME}.*'):
            p.unlink(missing_ok=True)

    print('\n| # | check | result | detail |')
    print('|---|---|---|---|')
    for i, name, ok, det in results:
        print(f"| {i} | {name} | {'PASS' if ok else 'FAIL'} | {det} |")
    n_fail = sum(1 for *_, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
