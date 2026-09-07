#!/usr/bin/env python3
"""tools/smoke_probe_feature_info.py — [E0] CPU 스모크(30초 이내).

검사 항목:
  1. LoRA `enabled` 토글 byte-동일성 — MultiModalLoRAQKV·SharedLoRAQKV 모두에서
     enabled=False 가 base(x) 와 정확히 같고(|Δ|max=0), enabled=True(delta 비영)
     가 base(x) 와 다르며, 기본값이 True 이고 state_dict 키가 토글로 안 바뀐다.
  2. 프로브 순수 로직 — majority_downsample / confusion_matrix /
     iou_recall_from_confusion / BalancedBank / LinearProbe(선형 분리 데이터에서
     높은 정확도) 정상 동작.

실행: python tools/smoke_probe_feature_info.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semseg.models.reliadino.encoder import MultiModalLoRAQKV, SharedLoRAQKV
from tools.probe_feature_info import (
    majority_downsample, confusion_matrix, iou_recall_from_confusion,
    BalancedBank, LinearProbe, resolve_taps)

torch.manual_seed(0)
FAIL = []


def check(cond, msg):
    print(('  ok ' if cond else '  FAIL ') + msg)
    if not cond:
        FAIL.append(msg)


def test_lora_toggle():
    print('[1] LoRA enabled 토글 byte-동일성')
    d = 8
    base = torch.nn.Linear(d, 3 * d, bias=True)
    x = torch.randn(2, 5, d)
    for cls, kwargs in [(MultiModalLoRAQKV, dict(num_modalities=3, r=4)),
                        (SharedLoRAQKV, dict(num_modalities=3, shared_r=4, residual_r=2))]:
        w = cls(base, **kwargs)
        check(w.enabled is True, f'{cls.__name__}: 기본 enabled=True')
        keys0 = set(w.state_dict().keys())

        # b_* 를 비영으로 채워 delta 가 실제로 0 이 아니게 한다.
        with torch.no_grad():
            for n, p in w.named_parameters():
                if n.startswith('b_'):
                    p.copy_(torch.randn_like(p))
        y_base = base(x)
        w.enabled = True
        y_on = w(x)
        w.enabled = False
        y_off = w(x)

        check(torch.equal(y_off, y_base),
              f'{cls.__name__}: enabled=False == base(x) (|Δ|max=0)')
        check(not torch.equal(y_on, y_base),
              f'{cls.__name__}: enabled=True != base(x) (delta 적용됨)')
        check(set(w.state_dict().keys()) == keys0,
              f'{cls.__name__}: 토글이 state_dict 키를 바꾸지 않음')

    # 기본 초기화(b_*=0)에서는 enabled True/False 가 둘 다 base 와 같아야 한다
    # (zero-init delta) — "기본값에서 forward byte-동일" 계약 확인.
    w2 = MultiModalLoRAQKV(base, num_modalities=3, r=4)
    check(torch.equal(w2(x), base(x)),
          'MultiModalLoRAQKV: zero-init(기본) forward == base(x)')


def test_pure_logic():
    print('[2] 프로브 순수 로직')
    # majority_downsample: 위쪽 절반=0, 아래=1 인 4x4 → 2x2 다수결.
    gt = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    maj = majority_downsample(gt, 2, 2, num_classes=2, ignore=255)
    check(maj.tolist() == [0, 0, 1, 1], f'majority_downsample 다수결 정답 (got {maj.tolist()})')
    # ignore 셀
    gti = torch.full((4, 4), 255)
    maji = majority_downsample(gti, 2, 2, 2, 255)
    check(bool((maji == 255).all()), 'majority_downsample 전-ignore → 전-ignore')

    # confusion + IoU/recall: 완벽 예측
    pred = torch.tensor([0, 0, 1, 1, 2, 2])
    tgt = torch.tensor([0, 0, 1, 1, 2, 2])
    h = confusion_matrix(pred, tgt, 3)
    iou, rec, miou = iou_recall_from_confusion(h)
    check(abs(miou - 100.0) < 1e-6, f'완벽 예측 mIoU=100 (got {miou})')
    check(all(abs(r - 100.0) < 1e-6 for r in rec), 'recall 전부 100')
    # 한 쌍 오분류: gt=0 픽셀 하나를 1 로
    pred2 = torch.tensor([0, 1, 1, 1, 2, 2])
    h2 = confusion_matrix(pred2, tgt, 3)
    _, rec2, miou2 = iou_recall_from_confusion(h2)
    check(miou2 < 100.0 and rec2[0] == 50.0, f'오분류 반영 (miou={miou2}, rec0={rec2[0]})')

    # BalancedBank cap
    bank = BalancedBank(num_classes=3, cap=10, ignore=255)
    feats = torch.randn(300, 4)
    labs = torch.randint(0, 3, (300,))
    bank.add(feats, labs)
    check(all(bank.count[c] <= 10 for c in range(3)), 'BalancedBank cap 준수')
    X, y = bank.build(torch.device('cpu'))
    check(X.shape[1] == 4 and X.dtype == torch.float32, 'BalancedBank build 형상/dtype')

    # LinearProbe: 선형 분리 3-클래스 → 높은 학습 정확도
    dev = torch.device('cpu')
    centers = torch.tensor([[3.0, 0], [-3, 0], [0, 3]])
    N = 600
    yy = torch.randint(0, 3, (N,))
    XX = centers[yy] + 0.3 * torch.randn(N, 2)
    pr = LinearProbe(2, 3, dev)
    pr.fit(XX, yy, epochs=40, lr=0.1, batch=256)
    acc = float((pr.predict(XX) == yy).float().mean())
    check(acc > 0.95, f'LinearProbe 선형 분리 정확도 {acc:.3f} > 0.95')

    # resolve_taps: 1-indexed → 0-indexed, 24블록에서 24→23
    check(resolve_taps([6, 12, 18, 24], 24) == [5, 11, 17, 23],
          f'resolve_taps 기본 매핑 (got {resolve_taps([6, 12, 18, 24], 24)})')


if __name__ == '__main__':
    test_lora_toggle()
    test_pure_logic()
    if FAIL:
        print(f'\nSMOKE FAILED ({len(FAIL)}건): ' + '; '.join(FAIL))
        sys.exit(1)
    print('\nSMOKE PASSED')
