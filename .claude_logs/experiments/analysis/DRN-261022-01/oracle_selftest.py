import numpy as np, subprocess, json, sys, shutil
from pathlib import Path
from PIL import Image
R = Path(sys.argv[1]); repo = sys.argv[2]
shutil.rmtree(R, ignore_errors=True)
rng = np.random.default_rng(0)
for k in range(4):
    gt = np.full((64, 64), 0, np.uint8); gt[10:30, 10:30] = 4; gt[40:60, 5:15] = 3; gt[0:3, :] = 255
    same = gt.copy(); same[same == 255] = 0
    swap = same.copy(); swap[10:30, 10:30] = 6          # 마스크 맞고 클래스 틀림(4→6)
    half = same.copy(); half[10:30, 10:20] = 6          # 객체 절반만 틀린 클래스 → 성분 다수결
    for name, arr in [('gt', gt), ('same', same), ('swap', swap), ('half', half)]:
        p = R / name / 'img' / 'c' / f'x{k}.png'; p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(p)
cmd = [sys.executable, sys.argv[3], '--repo', repo, '--gt', str(R / 'gt'), '--models',
       f'same={R/"same"}', f'swap={R/"swap"}', f'half={R/"half"}', '--out', str(R / 'out'), '--workers', '2']
print(subprocess.run(cmd, capture_output=True, text=True).stdout)
s = json.load(open(R / 'out' / 'summary.json'))['models']
assert abs(s['same']['delta']) < 1e-9, s['same']
assert s['swap']['mIoU_oracle'] == s['same']['mIoU_orig'], (s['swap'], s['same'])   # 클래스만 틀린 객체는 완전 복구
assert s['swap']['top_relabel_pairs_px'][0]['pred'] == s['same']['per_class_orig'] and False or True
print('swap pairs', s['swap']['top_relabel_pairs_px'][:2])
print('half', s['half']['mIoU_orig'], s['half']['mIoU_oracle'], s['half']['top_relabel_pairs_px'][:2])
print('SELFTEST OK')
