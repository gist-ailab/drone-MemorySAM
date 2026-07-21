#!/usr/bin/env python3
"""
predict_muses_test.py — MUSES *test* (750 imgs) inference -> Codabench submission PNGs.

Derived from eval_muses_official.py (which was validated: val mIoU 80.86 official /
81.02 internal, letterbox round-trip bit-identical). The GEOMETRY IS COPIED VERBATIM
from that script — the only change is that instead of accumulating a confusion matrix
against GT (test GT is withheld by the benchmark), we write the argmax to a PNG.

Path (identical to the validated official-protocol val path):
  letterboxed 1024x1024 forward (fp32, no TTA)
    -> crop the letterbox padding out of the LOGITS
    -> bilinear upsample the cropped logits to native 1080x1920
    -> argmax
    -> write uint8 single-channel trainID PNG at 1920x1080

semseg/datasets/muses.py is NOT modified: MUSES.__init__ deliberately raises
FileNotFoundError for split=='test' to tell the trainer the GT is missing, and other
code depends on that. We subclass and mirror __getitem__ minus the GT read instead.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

if '--gpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[sys.argv.index('--gpu') + 1]
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402
import torchvision.transforms.functional as TF                      # noqa: E402
import yaml                                                         # noqa: E402
from torch.utils.data import DataLoader, Dataset                    # noqa: E402
from torchvision import io                                          # noqa: E402
from tqdm import tqdm                                               # noqa: E402

from semseg.augmentations_mm import get_val_augmentation            # noqa: E402
from semseg.datasets.muses import MUSES                             # noqa: E402
from semseg.models.reliadino.model import build_reliadino           # noqa: E402


# ------------------------------------------------- letterbox geometry (VERBATIM copy)
def letterbox_valid_box(orig_h: int, orig_w: int, side: int):
    """Region of the (side x side) network output that holds the real image."""
    S = max(orig_h, orig_w)
    top = (S - orig_h) // 2
    left = (S - orig_w) // 2
    s = side / S
    t0, t1 = int(round(top * s)), int(round((top + orig_h) * s))
    l0, l1 = int(round(left * s)), int(round((left + orig_w) * s))
    return t0, t1, l0, l1


def sanity_check_geometry(orig_h=1080, orig_w=1920, side=1024, ignore=255):
    """Round-trip proof that the inverse-letterbox indices are correct (VERBATIM)."""
    rng = np.random.RandomState(0)
    lbl = torch.from_numpy(rng.randint(0, 19, (1, orig_h, orig_w)).astype(np.uint8))

    padded = MUSES._pad_to_square(lbl, fill=ignore)
    S = max(orig_h, orig_w)
    assert padded.shape[1:] == (S, S), f"pad shape {padded.shape}"

    t0f, t1f, l0f, l1f = letterbox_valid_box(orig_h, orig_w, S)
    back = padded[:, t0f:t1f, l0f:l1f]
    assert back.shape == lbl.shape, f"roundtrip shape {back.shape} vs {lbl.shape}"
    assert torch.equal(back, lbl), "FULL-RES ROUNDTRIP MISMATCH"

    small = TF.resize(padded, (side, side), TF.InterpolationMode.NEAREST)
    t0, t1, l0, l1 = letterbox_valid_box(orig_h, orig_w, side)
    inside = small[:, t0:t1, l0:l1]
    assert (inside != ignore).all(), "crop box still contains ignore padding"
    n_ign_out = (small == ignore).sum().item() - (inside == ignore).sum().item()
    n_ign_tot = (small == ignore).sum().item()
    assert n_ign_out == n_ign_tot, "some padding leaked outside the crop box"
    return dict(box_1024=(t0, t1, l0, l1), box_full=(t0f, t1f, l0f, l1f))


# ------------------------------------------------- test dataset (no GT)
class MUSESTest(MUSES):
    """MUSES test split loader. Mirrors MUSES.__getitem__ EXACTLY except the GT read.

    MUSES.__init__ raises for split=='test' on purpose (the trainer relies on it), so
    we bypass it here rather than weakening the shared loader. A dummy ignore-filled
    mask keeps the shared transform pipeline on the identical code path as val; it is
    discarded before the sample is returned.
    """

    def __init__(self, root, transform=None, modals=('img', 'lidar', 'event'),
                 legacy_radar=False):
        Dataset.__init__(self)
        self.legacy_radar = bool(legacy_radar)
        self.root = root
        self.split = 'test'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = list(modals)
        self.return_meta = True
        self.files = sorted(glob.glob(os.path.join(root, 'frame_camera', 'test',
                                                   '*', '*', '*.png')))
        if not self.files:
            raise FileNotFoundError(f"No test images in {root}/frame_camera/test/*/*/*.png")
        print(f"Found {len(self.files)} test images.", flush=True)

    def _open_modal(self, m: str, p: str):
        """모달별 디코더 디스패치 — semseg/datasets/muses.py와 반드시 일치해야 한다.

        🔴 ISSUE-025 재발 방지: 예전에는 여기서
            `self._open_event(p) if m == 'event' else self._open_lidar(p)`
        한 줄로 처리해 **radar가 lidar 디코더로 흘렀다**. 학습 로더가 3d2bb9a로
        고쳐진 뒤에도 이 파일은 로직을 복제하고 있어 수정이 반영되지 않았고,
        그대로 두면 radar-fix로 학습한 모델을 **버그 디코더로 추론**하게 되어
        입력 분포가 어긋난다(= radar 픽스가 무익해 보이는 잘못된 결론).

        `legacy_radar=True`는 **버그 시절에 학습된 체크포인트를 재현할 때만** 쓴다.
        그 모델들은 오염된 radar를 전제로 학습됐으므로, 올바른 디코더를 쓰면
        오히려 성능이 떨어진다. 학습 시점과 추론을 일치시키는 것이 원칙이다.
        """
        if m == 'event':
            return self._open_event(p)
        if m == 'radar':
            return self._open_lidar(p) if self.legacy_radar else self._open_radar(p)
        return self._open_lidar(p)

    def __getitem__(self, index):
        rgb = str(self.files[index])

        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        H, W = sample['img'].shape[1:]
        for m in self.modals:
            if m == 'img':
                continue
            p = self._sibling(rgb, m)
            x = self._open_modal(m, p)
            if x.shape[1:] != (H, W):
                x = TF.resize(x, [H, W], TF.InterpolationMode.NEAREST)
            sample[m] = x

        # dummy mask: keeps the transform code path identical to val; never used.
        sample['mask'] = torch.full((1, H, W), self.ignore_label, dtype=torch.uint8)

        for k in sample:
            sample[k] = self._pad_to_square(
                sample[k], fill=self.ignore_label if k == 'mask' else 0)

        if self.transform:
            sample = self.transform(sample)
        del sample['mask']
        out = [sample[k] for k in self.modals]

        stem = Path(rgb).stem
        assert stem.endswith('_frame_camera'), f"unexpected stem {stem}"
        base = stem[:-len('_frame_camera')]          # REC0006_frame_042430
        cond = '/'.join(Path(rgb).parts[-3:-1])      # weather/tod
        return out, {'base': base, 'orig_h': int(H), 'orig_w': int(W),
                     'cond': cond, 'rgb_path': rgb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--out', required=True, help='dir for the PNGs')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--legacy-radar', action='store_true',
                    help='radar를 (버그가 있던) lidar 디코더로 읽는다. '
                         'ISSUE-025 수정 이전에 학습된 ckpt를 재현할 때만 사용.')
    args = ap.parse_args()

    pngdir = Path(args.out) / 'pred'
    pngdir.mkdir(parents=True, exist_ok=True)

    geo = sanity_check_geometry()
    print(f"[sanity] letterbox inverse verified: {geo}", flush=True)

    cfg = yaml.safe_load(open(args.cfg))
    dcfg, ecfg = cfg['DATASET'], cfg['EVAL']
    if args.dataset_root:
        dcfg['ROOT'] = args.dataset_root
    device = torch.device('cuda')

    valtransform = get_val_augmentation(ecfg['IMAGE_SIZE'], dataset_cfg=dcfg)
    if 'radar' in dcfg['MODALS']:
        mode = 'LEGACY(lidar 디코더 — 버그 재현)' if args.legacy_radar else 'FIXED(_open_radar)'
        print(f"[ISSUE-025] radar 디코딩 = {mode}. "
              f"학습 시점과 반드시 일치해야 한다.", flush=True)
    ds = MUSESTest(dcfg['ROOT'], valtransform, dcfg['MODALS'],
                   legacy_radar=args.legacy_radar)
    n_classes, class_names = ds.n_classes, ds.CLASSES
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = build_reliadino(cfg, n_classes)
    ck = torch.load(args.ckpt, map_location='cpu')
    state = ck.get('model_state_dict', ck)
    msg = model.load_state_dict(state, strict=False)
    assert not msg.missing_keys and not msg.unexpected_keys, \
        f"state_dict mismatch: missing={msg.missing_keys[:3]} unexpected={msg.unexpected_keys[:3]}"
    print(f"[ckpt] {Path(args.ckpt).name} epoch={ck.get('epoch', '?')} loaded clean", flush=True)
    model = model.to(device).eval()

    px_per_class = np.zeros(n_classes, dtype=np.int64)
    per_cond = {}
    degenerate = []
    written = []
    n_done = 0

    with torch.no_grad():
        for images, meta in tqdm(loader, desc='muses-test'):
            images = [x.to(device, non_blocking=True) for x in images]

            logits, _ = model(images, True)                  # (1,19,1024,1024) fp32

            H, W = int(meta['orig_h'][0]), int(meta['orig_w'][0])
            t0, t1, l0, l1 = letterbox_valid_box(H, W, logits.shape[-1])
            crop = logits[:, :, t0:t1, l0:l1]
            up = F.interpolate(crop, size=(H, W), mode='bilinear', align_corners=False)
            pred = up.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()
            assert pred.shape == (H, W), f"{pred.shape} != {(H, W)}"

            base = meta['base'][0]
            outp = pngdir / f"{base}.png"
            ok = cv2.imwrite(str(outp), pred)               # single-channel uint8
            assert ok, f"imwrite failed for {outp}"
            written.append(base)

            cnt = np.bincount(pred.reshape(-1), minlength=n_classes)
            px_per_class += cnt
            cond = meta['cond'][0]
            per_cond.setdefault(cond, 0)
            per_cond[cond] += 1
            nuniq = int((cnt > 0).sum())
            if nuniq <= 2:
                degenerate.append((base, nuniq,
                                   [class_names[i] for i in np.argsort(-cnt)[:3] if cnt[i] > 0]))

            n_done += 1
            if args.limit and n_done >= args.limit:
                break

    total = int(px_per_class.sum())
    summary = {
        'ckpt': str(args.ckpt),
        'epoch': ck.get('epoch', None),
        'n_images': n_done,
        'geometry': {k: list(v) for k, v in geo.items()},
        'per_condition_counts': per_cond,
        'class_pixel_share_pct': {c: round(100.0 * int(v) / total, 4)
                                  for c, v in zip(class_names, px_per_class)},
        'class_pixel_counts': {c: int(v) for c, v in zip(class_names, px_per_class)},
        'classes_never_predicted': [c for c, v in zip(class_names, px_per_class) if v == 0],
        'degenerate_images_le2_classes': degenerate,
    }
    with open(Path(args.out) / 'predict_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    with open(Path(args.out) / 'written_stems.txt', 'w') as f:
        f.write('\n'.join(sorted(written)) + '\n')

    print(f"\n===== wrote {n_done} PNGs to {pngdir} =====")
    print("--- per-condition image counts ---")
    for k in sorted(per_cond):
        print(f"  {k:<14} {per_cond[k]}")
    print("--- predicted class pixel share (%) ---")
    for c, v in zip(class_names, px_per_class):
        print(f"  {c:<15} {100.0 * v / total:7.4f}   ({int(v)})")
    print(f"--- classes never predicted: {summary['classes_never_predicted']}")
    print(f"--- images with <=2 distinct classes: {len(degenerate)}")
    for d in degenerate[:10]:
        print(f"    {d}")
    print(f"\n[saved] {Path(args.out)}/predict_summary.json")


if __name__ == '__main__':
    main()
