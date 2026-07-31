"""피아식별(IFF) 크롭 분류 데이터셋 빌더 — Allies vs Enemies.

인증 항목 (C) 피아식별 정확도는 "찾은 표적이 아군인가 적군인가"를 묻는
분류 지표라, 검출 mAP로는 검출실패와 오분류가 섞여 측정되지 않는다.
그래서 GT 박스를 크롭해 2-class 분류 데이터셋을 만든다.

  train = poongsan_v2 train 5 clips (전부 주간)  -> Allies 4753 / Enemies 3739
  test  = poongsan_v2 test  3 clips (야간 2 + 주간 1) -> Allies 1168 / Enemies 978

capture-holdout(클립 단위 분리)을 그대로 물려받으므로 프레임/인접프레임 누수가 없다.
🔴 train에는 야간 프레임이 0장이므로, 학습 시 저조도 증강이 필수다
(certification/../objdet/yolov5m-lowlight/night_aug.py 의 실측 캘리브레이션 재사용).

  python build_iff_crops.py --train-root ~/poongsan_v2_train3modal \
      --test-root ~/poongsan_v2 --out ~/dset/poongsan_iff_crops
"""
from __future__ import annotations

import argparse
import collections
import json
import os

CLASSES = ['Allies', 'Enemies']          # 2-class IFF


def build(split: str, root: str, ann_path: str, out_dir: str,
          min_size: float, pad: float, crop: int, flat: bool = False) -> dict:
    from PIL import Image

    ann = json.load(open(ann_path))
    name_by_id = {c['id']: c['name'] for c in ann['categories']}
    want = {cid for cid, n in name_by_id.items() if n in CLASSES}
    img_by_id = {im['id']: im for im in ann['images']}

    for c in CLASSES:
        os.makedirs(f'{out_dir}/{split}/{c}', exist_ok=True)

    stats = collections.Counter()
    for a in ann['annotations']:
        if a['category_id'] not in want:
            continue
        cls = name_by_id[a['category_id']]
        im = img_by_id[a['image_id']]
        x, y, w, h = a['bbox']
        if w <= 0 or h <= 0:
            stats[f'{cls}:bad_box'] += 1
            continue
        if (w * h) ** 0.5 < min_size:            # 너무 작은 박스는 정보가 없다
            stats[f'{cls}:too_small'] += 1
            continue

        rel = im['modalities']['rgb'] if 'modalities' in im else im['file_name']
        # flat 레이아웃(<clip>_<ts>.png 한 폴더)도 지원 — 3모달 완비본(7043)이 아니라
        # RGB 전체(12681)를 쓰려면 이쪽이 필요하다.
        src = (os.path.join(root, rel.replace('/rgb/', '_').replace('/', '_'))
               if flat else os.path.join(root, rel))
        if not os.path.exists(src):
            stats[f'{cls}:missing_img'] += 1
            continue

        # 패딩 붙여 크롭 (문맥이 조금 있어야 자세/장비가 보인다)
        px, py = w * pad, h * pad
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(im['width'], x + w + px), min(im['height'], y + h + py)
        try:
            img = Image.open(src).convert('RGB').crop((int(x1), int(y1), int(x2), int(y2)))
        except Exception:
            stats[f'{cls}:read_fail'] += 1
            continue
        img = img.resize((crop, crop), Image.BILINEAR)

        night = 'night' if im.get('low_light') else 'day'
        stem = os.path.splitext(os.path.basename(rel))[0]
        clip = rel.split('/')[0]
        img.save(f'{out_dir}/{split}/{cls}/{clip}_{stem}_{a["id"]}_{night}.jpg', quality=92)
        stats[cls] += 1
        stats[f'{cls}:{night}'] += 1
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-root', required=True, help='train 이미지 루트 (clip/rgb/*.png)')
    ap.add_argument('--test-root', required=True)
    ap.add_argument('--train-ann', default=None)
    ap.add_argument('--test-ann', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--crop', type=int, default=128, help='크롭 리사이즈 (박스 중앙값 ~160px)')
    ap.add_argument('--pad', type=float, default=0.15)
    ap.add_argument('--min-size', type=float, default=32.0, help='sqrt(w*h) 최소 px')
    ap.add_argument('--train-flat', action='store_true',
                    help='train-root 가 flat 레이아웃(<clip>_<ts>.png)일 때 — RGB 전체 12681장 사용')
    ap.add_argument('--test-flat', action='store_true')
    args = ap.parse_args()

    tr_ann = args.train_ann or f'{args.train_root}/_final_ann/instances_train_egofill.json'
    te_ann = args.test_ann or f'{args.test_root}/_final_ann/instances_test_common.json'

    print(f'[iff] crop={args.crop} pad={args.pad} min_size={args.min_size}')
    for split, root, ann, flat in (('train', args.train_root, tr_ann, args.train_flat),
                                   ('test', args.test_root, te_ann, args.test_flat)):
        s = build(split, root, ann, args.out, args.min_size, args.pad, args.crop, flat)
        kept = sum(s[c] for c in CLASSES)
        print(f'\n[{split}] kept {kept}')
        for c in CLASSES:
            print(f'  {c:8s} {s[c]:5d}   (night {s[f"{c}:night"]}, day {s[f"{c}:day"]})'
                  f'  dropped: small {s[f"{c}:too_small"]}, missing {s[f"{c}:missing_img"]}')
    print(f'\n[iff] -> {args.out}')


if __name__ == '__main__':
    main()
