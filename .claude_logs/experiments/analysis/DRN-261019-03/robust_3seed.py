import json, glob, sys
import numpy as np
JA = sys.argv[1]   # jarvis robust_a2_20261019 local copy (s821, s902)
H3 = sys.argv[2]   # hpca100 a2s903 local copy
NAS = '/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/'
def load(p):
    f = glob.glob(p + '/*/missing_modality/summary.json')
    return json.load(open(f[0])) if f else None
A = {'821': JA + '/s821/', '902': JA + '/s902/', '903': H3 + '/'}
QT = {s: NAS + f'robust_q2_20260929/q2s{s}/q2s{s}_' for s in ['902', '903']}
QV = {s: NAS + f'robust_q2_val_20261008/q2s{s}_val_' for s in ['821', '902', '903']}
SUBS = {'lidar': 'img_depth_event', 'event': 'img_depth_lidar', 'depth': 'img_event_lidar', 'rgb': 'depth_event_lidar'}
rows = []
def row(name, a, q):
    a = [x for x in a if x is not None]; q = [x for x in q if x is not None]
    rows.append((name, a, q))
    am = np.mean(a) if a else float('nan'); qm = np.mean(q) if q else float('nan')
    print(f"{name:34s} A2 {'/'.join(f'{x:.2f}' for x in a):26s} mean {am:6.2f} | Q2 {'/'.join(f'{x:.2f}' for x in q):26s} mean {qm:6.2f} | d {am-qm:+.2f}")
g = lambda d, *k: (None if d is None else (d[k[0]] if len(k) == 1 else d[k[0]][k[1]]))
# test
row('test clean', [g(load(A[s] + 'test_emm'), 'clean_mIoU') for s in A], [57.73] + [g(load(QT[s] + 'emm'), 'clean_mIoU') for s in QT])
row('test EMM15 avg', [g(load(A[s] + 'test_emm'), 'EMM', 'avg') for s in A], [48.68] + [g(load(QT[s] + 'emm'), 'EMM', 'avg') for s in QT])
for lvl in ['0.05', '0.1', '0.2']:
    row(f'test NM S&P {lvl}', [g(load(A[s] + 'test_nm'), 'NM', lvl) for s in A], [g(load(QT[s] + 'nm'), 'NM', lvl) for s in QT])
for mod, sub in SUBS.items():
    for r in ['0.25', '0.5', '0.75']:
        a = [load(A[s] + 'test_rmm_' + sub) for s in A]; q = [load(QT[s] + 'rmm_' + sub) for s in QT]
        row(f'test RMM {mod} drop r{r} (clean-x)', [x['clean_mIoU'] - x['RMM'][r]['avg'] for x in a if x], [x['clean_mIoU'] - x['RMM'][r]['avg'] for x in q if x])
# val
row('val clean', [g(load(A[s] + 'val_emm'), 'clean_mIoU') for s in A], [g(load(QV[s] + 'emm'), 'clean_mIoU') for s in QV])
row('val EMM15 avg', [g(load(A[s] + 'val_emm'), 'EMM', 'avg') for s in A], [g(load(QV[s] + 'emm'), 'EMM', 'avg') for s in QV])
for r in ['0.25', '0.5', '0.75']:
    row(f'val RMM15 r{r} avg', [g(load(A[s] + 'val_rmm'), 'RMM', r)['avg'] if load(A[s] + 'val_rmm') else None for s in A],
        [g(load(QV[s] + 'rmm'), 'RMM', r)['avg'] if load(QV[s] + 'rmm') else None for s in QV])
# provenance
for s in A:
    d = load(A[s] + 'val_emm'); print('A2', s, 'val_emm batch', d['batch'], 'subset', d['subset_every'], d['model_path'].split('/')[-1])
