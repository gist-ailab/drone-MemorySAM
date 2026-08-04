"""[C/D] Cross-modal FEATURE distillation: ReliaDINO(3-modal) -> YOLOv5m(RGB).

기존 GISTOLO(B)는 teacher 예측을 라벨에 더한 것뿐이라 멀티모달 '표현'은 사라졌다.
여기서는 teacher 의 융합 FPN 특징을 student neck 특징에 직접 정합한다.

  [C] feature KD          L_kd = 1 - cos( adapter(student_P4), teacher_s16 )
  [D] modality-aware      위 손실을 위치별로 (1 + beta * (1 - w_img)) 로 가중.
                          w_img = teacher 게이트의 RGB 비중. RGB 비중이 낮은 곳
                          = LiDAR/Thermal 이 캐리한 곳 = RGB student 가 스스로는
                          못 배우는 곳 -> 그곳의 증류를 더 세게 건다.

adapter 는 학습 전용이고 추론 경로에 없다 -> export 그래프 불변(i.MX 유지).
YOLOv5 소스는 수정하지 않고 런타임 몽키패치만 한다.

  YOLOV5_DIR=~/yolov5 python train_kd.py --feat-dir ~/dset/gistolo_teacher_feats \
      --kd-w 1.0 --modality-beta 2.0 -- --data ... --weights yolov5m.pt --epochs 50 ...
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatKD:
    """teacher 특징 캐시 + adapter + KD 손실. student 학습 루프에 훅으로 붙는다."""

    def __init__(self, feat_dir: str, kd_w: float, beta: float, dev, student_ch: int,
                 teacher_ch: int = 256):
        self.dir = feat_dir
        self.kd_w = kd_w
        self.beta = beta
        self.dev = dev
        # 학습 전용 1x1 adapter (추론 시 사용되지 않음)
        self.adapter = nn.Conv2d(student_ch, teacher_ch, 1).to(dev)
        self.cache: dict[str, tuple] = {}
        self.hit = self.miss = 0

    def params(self):
        return self.adapter.parameters()

    def _load(self, stem: str):
        if stem in self.cache:
            return self.cache[stem]
        p = os.path.join(self.dir, stem + '.npz')
        if not os.path.exists(p):
            self.cache[stem] = None
            return None
        z = np.load(p)
        feat = torch.from_numpy(z['feat'].astype(np.float32))
        gate = torch.from_numpy(z['gate']) if 'gate' in z else None
        self.cache[stem] = (feat, gate)
        return self.cache[stem]

    def loss(self, student_feat: torch.Tensor, paths) -> torch.Tensor:
        """student_feat: (B,C,h,w) neck P4. paths: 배치 이미지 경로 리스트."""
        feats, gates, idx = [], [], []
        for i, p in enumerate(paths):
            stem = os.path.splitext(os.path.basename(p))[0]
            got = self._load(stem)
            if got is None:
                self.miss += 1
                continue
            self.hit += 1
            feats.append(got[0])
            gates.append(got[1] if got[1] is not None else torch.zeros(3, *got[0].shape[1:]))
            idx.append(i)
        if not feats:
            return student_feat.sum() * 0.0

        t = torch.stack(feats).to(self.dev, non_blocking=True)          # (b,256,g,g)
        g = torch.stack(gates).to(self.dev, non_blocking=True)          # (b,3,g,g)
        s = self.adapter(student_feat[idx].float())
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode='bilinear', align_corners=False)

        # 위치별 코사인 거리
        d = 1.0 - F.cosine_similarity(s, t, dim=1)                      # (b,g,g)

        if self.beta > 0 and g.abs().sum() > 0:
            w_img = g[:, 0]                                             # RGB 게이트 비중
            weight = 1.0 + self.beta * (1.0 - w_img).clamp(0, 1)        # 비-RGB 의존도
            d = d * weight
            d = d / weight.mean().clamp(min=1e-6)                       # 스케일 유지
        return self.kd_w * d.mean()


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--feat-dir', required=True, help='extract_teacher_feats.py 출력')
    ap.add_argument('--kd-w', type=float, default=1.0, help='KD 손실 가중치 (C)')
    ap.add_argument('--modality-beta', type=float, default=2.0,
                    help='모달 인식 가중 강도 (D). 0 이면 순수 feature KD (C only)')
    ap.add_argument('--kd-layer', type=int, default=17,
                    help='YOLOv5m neck P4 모듈 인덱스 (기본 17 = stride-16 출력)')
    known, rest = ap.parse_known_args()

    y5 = os.environ.get('YOLOV5_DIR', os.path.expanduser('~/yolov5'))
    sys.path.insert(0, y5)
    os.chdir(y5)
    sys.argv = [sys.argv[0]] + rest

    import train as y5train
    from utils.loss import ComputeLoss

    state = {'kd': None, 'feat': None, 'paths': None}

    # 1) student neck 특징 포획 — forward hook (그래프 변경 아님)
    _orig_train = y5train.train

    def patched_train(hyp, opt, device, callbacks):
        import models.yolo as yolo_mod
        _orig_fwd = yolo_mod.DetectionModel._forward_once

        def fwd(self, x, profile=False, visualize=False):
            y, dt = [], []
            for m in self.model:
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)
                y.append(x if m.i in self.save else None)
                if m.i == known.kd_layer and isinstance(x, torch.Tensor):
                    state['feat'] = x                     # (B,C,h,w) neck P4
            return x
        yolo_mod.DetectionModel._forward_once = fwd

        # 2) 손실에 KD 항 추가
        _orig_call = ComputeLoss.__call__

        def call(self, p, targets):
            loss, items = _orig_call(self, p, targets)
            kd = state['kd']
            if kd is not None and state['feat'] is not None and state['paths'] is not None:
                l = kd.loss(state['feat'], state['paths'])
                loss = loss + l * state['feat'].shape[0]   # yolov5 loss 는 batch 곱 스케일
            return loss, items
        ComputeLoss.__call__ = call

        return _orig_train(hyp, opt, device, callbacks)

    y5train.train = patched_train

    # 3) 데이터로더가 주는 경로를 매 스텝 state 에 넣기 위해 create_dataloader 래핑
    import utils.dataloaders as dl
    _orig_loader = dl.create_dataloader

    def loader(*a, **k):
        out = _orig_loader(*a, **k)
        ld = out[0]
        _orig_iter = ld.__class__.__iter__

        def it(self):
            for batch in _orig_iter(self):
                state['paths'] = batch[2]                  # (im, labels, paths, shapes)
                yield batch
        ld.__class__.__iter__ = it
        return out
    dl.create_dataloader = loader

    opt = y5train.parse_opt(True)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ch = {17: 384}.get(known.kd_layer, 384)                # yolov5m P4 neck = 384ch
    state['kd'] = FeatKD(known.feat_dir, known.kd_w, known.modality_beta, dev, ch)
    print(f'[KD] feature distillation ON — layer {known.kd_layer} ({ch}ch -> 256ch adapter), '
          f'kd_w={known.kd_w}, modality_beta={known.modality_beta} '
          f'({"C+D" if known.modality_beta > 0 else "C only"})')

    # adapter 파라미터를 옵티마이저에 포함시키기 위해 smart_optimizer 래핑
    import utils.torch_utils as tu
    _orig_opt = tu.smart_optimizer

    def smart(model, name, lr, momentum, decay):
        o = _orig_opt(model, name, lr, momentum, decay)
        o.add_param_group({'params': list(state['kd'].params()), 'weight_decay': 0.0})
        print('[KD] adapter 파라미터를 옵티마이저에 추가 (학습 전용, 추론 경로 없음)')
        return o
    tu.smart_optimizer = smart

    y5train.main(opt)
    k = state['kd']
    print(f'[KD] teacher 특징 매칭 hit={k.hit} miss={k.miss}')


if __name__ == '__main__':
    main()
