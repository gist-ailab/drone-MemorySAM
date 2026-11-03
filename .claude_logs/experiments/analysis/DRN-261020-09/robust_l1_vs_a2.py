import json, glob, sys
L = sys.argv[1]  # robust_l1_s821_20261029 local copy
A = sys.argv[2]  # robust_a2_20261019/s821 local copy
def load(p):
    f = glob.glob(p + '/*/missing_modality/summary.json')
    return json.load(open(f[0])) if f else None
rows = []
def add(name, a, l):
    rows.append((name, a, l)); print(f'{name:38s} A2 {a:7.2f} | L1 {l:7.2f} | Δ(L1−A2) {l-a:+.2f}')
e_a, e_l = load(A + '/test_emm'), load(L + '/test_emm')
add('test clean', e_a['clean_mIoU'], e_l['clean_mIoU'])
add('test EMM15 avg', e_a['EMM']['avg'], e_l['EMM']['avg'])
n_a, n_l = load(A + '/test_nm'), load(L + '/test_nm')
for k in ['0.05', '0.1', '0.2']:
    add(f'test NM S&P {k}', n_a['NM'][k], n_l['NM'][k])
for mod, sub in {'LiDAR': 'img_depth_event', 'Event': 'img_depth_lidar', 'Depth': 'img_event_lidar', 'RGB': 'depth_event_lidar'}.items():
    ra, rl = load(A + '/test_rmm_' + sub), load(L + '/test_rmm_' + sub)
    for r in ['0.25', '0.5', '0.75']:
        add(f'test RMM {mod} drop r{r} (clean−x)', ra['clean_mIoU'] - ra['RMM'][r]['avg'], rl['clean_mIoU'] - rl['RMM'][r]['avg'])
v_a, v_l = load(A + '/val_emm'), load(L + '/val_emm')
add('val clean', v_a['clean_mIoU'], v_l['clean_mIoU'])
add('val EMM15 avg', v_a['EMM']['avg'], v_l['EMM']['avg'])
vr_a, vr_l = load(A + '/val_rmm'), load(L + '/val_rmm')
for r in ['0.25', '0.5', '0.75']:
    add(f'val RMM15 r{r} avg', vr_a['RMM'][r]['avg'], vr_l['RMM'][r]['avg'])
print('batch A2', e_a['batch'], 'L1', e_l['batch'], '| ckpt L1', e_l['model_path'].split('/')[-1], '| A2', e_a['model_path'].split('/')[-1])
