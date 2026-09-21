#!/usr/bin/env python3
"""[P54-QAF] 학습 루프 종단 스모크 — `train_reliadino.main()` 을 작은 합성 데이터로 실제 실행.

왜 필요한가(2026-09-21): tools/smoke_qaf.py 는 모델 forward/손실만 돌려 학습 루프(`main`)를 한 번도
실행하지 않았다. 그 결과 (1) 두 패스 분기가 지운 변수를 루프 끝 `del` 이 다시 지워 UnboundLocalError
(2) F.binary_cross_entropy autocast 금지 (3) KD batchmean 픽셀 폭주 — 세 결함이 서버 기동 때마다
하나씩 드러났다. 이 스크립트는 **실제 Q3 yaml**(모델 구성 전부: TAPS·M2F·arbiter·router·VICReg·C3·QAF)
에서 크기 키만 tiny 로 줄이고, 합성 데이터셋으로 main 을 끝까지(에폭 2·평가·체크포인트·에폭 끝 로그)
돌린다. CUDA 가 필요하다(진짜 autocast). 없으면 SKIP.

  python tools/smoke_train_e2e.py [--cfg configs/…screen40_Q3.yaml] [--epochs 2] [--keep]

검사: 예외 없이 완주 · 손실 유한 · 로그에 `[QAF-T]`·`[QAF]` 표식 · ckpt 저장 · (교사 있음) 두 패스 KD 유한.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

K = 6
S = 64


class _SynthDS(torch.utils.data.Dataset):
    """DELIVER 와 같은 인터페이스(CLASSES·n_classes·ignore_label, (sample_list, label))."""
    CLASSES = [f'c{i}' for i in range(K)]
    n_classes = K
    ignore_label = 255

    def __init__(self, root, split, transform, modals, **kw):
        self.split, self.modals = split, list(modals)
        self.n = 32 if split == 'train' else 4

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(1000 + i)
        sample = [torch.randn(3, S, S, generator=g) for _ in self.modals]
        lbl = torch.randint(0, K, (S, S), generator=g)
        lbl[:4, :4] = 255
        return sample, lbl


def _tiny(cfg):
    m = cfg['MODEL']
    m['BACKBONE_TIMM'] = 'vit_tiny_patch16_224'
    m['BACKBONE_FALLBACK'] = 'vit_tiny_patch16_224'
    m['PRETRAINED_BACKBONE'] = False
    m['LORA_R'] = 2
    m['FPN_DIM'] = 64
    if isinstance(m.get('TAPS'), dict):
        m['TAPS']['LAYERS'] = [3, 6, 9, 12]
    fu = m.get('FUSION') or {}
    fu.update({'NUM_LAYERS': 1, 'NUM_HEADS': 4, 'MLP_RATIO': 1.0, 'AUX_HIDDEN': 32})
    m['FUSION'] = fu
    p39 = m.get('P39') or {}
    p39['TRUNK_HIDDEN'] = 64
    if isinstance(p39.get('VICREG'), dict):
        p39['VICREG']['TOKENS'] = 64
    m['P39'] = p39
    m2f = m.get('M2F') or {}
    m2f.update({'NUM_QUERIES': 8, 'NUM_LAYERS': 1, 'DIM': 32, 'NUM_HEADS': 4, 'MLP_RATIO': 1.0,
                'POINTS': 256})
    m['M2F'] = m2f
    if isinstance(m.get('ROUTER'), dict):
        m['ROUTER']['HIDDEN'] = 16
    if isinstance(m.get('QAF'), dict):
        m['QAF']['HEAD_HIDDEN'] = 16
    m['RESUME_ENABLE'] = False
    m['AUTO_RESUME'] = False
    m['RESUME_PATH'] = ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/jarvis-deliver_rgbdel_P46_c3only_seed20260821_screen40_Q3.yaml')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--no_teacher', action='store_true')
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("[SKIP] CUDA 없음 — 종단 스모크는 진짜 autocast 가 필요하다")
        return 0
    import train_reliadino as T
    from semseg.models.reliadino import build_reliadino

    with open(a.cfg, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    tmp = tempfile.mkdtemp(prefix='smoke_train_e2e_')
    cfg['DEVICE'] = 'cuda'
    cfg['SAVE_DIR'] = tmp
    _tiny(cfg)
    tr, ev = cfg['TRAIN'], cfg['EVAL']
    tr.update({'IMAGE_SIZE': [S, S], 'BATCH_SIZE': 4, 'EPOCHS': a.epochs, 'EVAL_START': 0,
               'EVAL_INTERVAL': 1, 'DDP': False, 'AMP': True, 'AMP_DTYPE': 'bfloat16',
               'SAVE_TOPK': 1, 'SAVE_TEST_CKPT': False})
    ev.update({'IMAGE_SIZE': [S, S], 'BATCH_SIZE': 2})
    cfg['DATASET'].update({'NAME': '_SynthDS', 'ROOT': ''})
    if isinstance(cfg.get('WANDB'), dict):
        cfg['WANDB']['MODE'] = 'disabled'
    qaf = tr.get('QAF') or {}
    # 교사: QAF off 모델의 state_dict 를 ckpt 로 저장해 TEACHER_CKPT 로 지정
    if qaf.get('ENABLE') and not a.no_teacher:
        tcfg = copy.deepcopy(cfg)
        tcfg['MODEL']['QAF'] = {'ENABLE': False}
        tm = build_reliadino(tcfg, K)
        tck = os.path.join(tmp, 'teacher.pth')
        torch.save({'model_state_dict': tm.state_dict()}, tck)
        qaf['TEACHER_CKPT'] = tck
        del tm
    # main 이 eval(dataset_cfg['NAME']) 로 T 의 전역에서 데이터셋을 찾는다
    T._SynthDS = _SynthDS
    save_dir = Path(tmp, 'run')
    os.makedirs(save_dir, exist_ok=True)
    log_path = save_dir / 'train.log'
    logger = logging.getLogger('smoke_e2e')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    T.fix_seeds(0)
    print(f"[e2e] cfg={a.cfg} tmp={tmp} epochs={a.epochs} teacher={'yes' if qaf.get('TEACHER_CKPT') else 'no'}")
    try:
        T.main(cfg, 0, save_dir, logger)
    except Exception as e:                      # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n[FAIL] 학습 루프가 예외로 종료: {type(e).__name__}: {e}")
        return 1
    log = log_path.read_text(encoding='utf-8', errors='ignore')
    ok = True
    for tag in ('[QAF]', '[QAF-T]'):
        hit = tag in log or tag in '\n'.join(str(x) for x in [])
        print(f"  [{'PASS' if hit else 'INFO'}] 로그 표식 {tag}: {hit}")
    ckpts = list(save_dir.glob('*.pth'))
    print(f"  [{'PASS' if ckpts else 'FAIL'}] 체크포인트 저장: {[c.name for c in ckpts][:3]}")
    ok = ok and bool(ckpts)
    if not a.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
