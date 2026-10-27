import json, glob, sys
A = sys.argv[1]  # a2s903 root (local copy)
Q = '/drone_nas/drone/personal/jemo_maeng/src/Project/drone/drone-MemorySAM/analysis_logs/'
def load(p):
    f = glob.glob(p + '/*/missing_modality/summary.json')
    return json.load(open(f[0])) if f else None
pairs = {
    'test_emm': (A + '/test_emm', Q + 'robust_q2_20260929/q2s903/q2s903_emm'),
    'test_nm': (A + '/test_nm', Q + 'robust_q2_20260929/q2s903/q2s903_nm'),
    'val_emm': (A + '/val_emm', Q + 'robust_q2_val_20261008/q2s903_val_emm'),
    'val_rmm': (A + '/val_rmm', Q + 'robust_q2_val_20261008/q2s903_val_rmm'),
}
for sub in ['img_depth_event', 'img_depth_lidar', 'img_event_lidar', 'depth_event_lidar']:
    pairs['test_rmm_' + sub] = (A + '/test_rmm_' + sub, Q + 'robust_q2_20260929/q2s903/q2s903_rmm_' + sub)
for k, (a, q) in pairs.items():
    da, dq = load(a), load(q)
    if da is None or dq is None:
        print(k, 'MISSING', da is None, dq is None); continue
    proto = da['protocol'].upper()
    print(f"== {k}: clean A2 {da['clean_mIoU']:.2f} Q2 {dq['clean_mIoU']:.2f} | batch A2 {da['batch']} Q2 {dq['batch']} | subset_every {da['subset_every']}/{dq['subset_every']}")
    xa, xq = da[proto], dq[proto]
    if proto == 'EMM':
        for key in xa:
            if not isinstance(xa[key], dict):
                print(f'   {key}: A2 {xa[key]} Q2 {xq.get(key)}')
        print('   keys:', list(xa.keys())[:25])
    else:
        for key in xa:
            va, vq = xa[key], xq.get(key)
            if isinstance(va, dict):
                print(f"   {key}: A2 avg {va.get('avg')} Q2 avg {vq.get('avg') if vq else None} | n {va.get('n_combos')}")
            else:
                print(f'   {key}: A2 {va} Q2 {vq}')
