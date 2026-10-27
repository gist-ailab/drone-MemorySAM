import json, sys, numpy as np
D = sys.argv[1]
SEEDS = ['821', '902', '903']
S = {f'{p}s{s}': json.load(open(f'{D}/{p}s{s}.json')) for p in ['q2', 'a2'] for s in SEEDS}
CL = list(S['q2s821']['per_class_iou'].keys())
THIN = ['Pole', 'TrafficLight', 'Pedestrian', 'Static']
def m(k, ex=()): return np.nanmean([S[k]['per_class_iou'][c] for c in CL if c not in ex])
def d(c, s): return S['a2s' + s]['per_class_iou'][c] - S['q2s' + s]['per_class_iou'][c]
print('model mIoU25:', {k: round(v['mIoU'], 2) for k, v in S.items()})
rows = []
for s in SEEDS:
    d25 = m('a2s' + s) - m('q2s' + s); d24 = m('a2s' + s, ('RailTrack',)) - m('q2s' + s, ('RailTrack',))
    th = np.mean([d(c, s) for c in THIN])
    rows.append((d25, d24, th, d('TrafficLight', s)))
    print(f's{s}: A2 25={m("a2s"+s):.2f} Q2 25={m("q2s"+s):.2f} d25={d25:+.2f} | A2 24={m("a2s"+s,("RailTrack",)):.2f} Q2 24={m("q2s"+s,("RailTrack",)):.2f} d24={d24:+.2f} | thin4 {th:+.2f} | TL {d("TrafficLight", s):+.2f}')
r = np.array(rows)
print('3-seed mean: d25 %+.2f  d24 %+.2f  thin4 %+.2f  TL %+.2f' % tuple(r.mean(0)))
print('3-seed mean abs: A2 24 %.2f  Q2 24 %.2f' % (np.mean([m('a2s'+s,('RailTrack',)) for s in SEEDS]), np.mean([m('q2s'+s,('RailTrack',)) for s in SEEDS])))
out = ['class,' + ','.join(f'q2s{s},a2s{s},d{s}' for s in SEEDS) + ',d_mean3']
for c in CL:
    vals = []
    for s in SEEDS:
        vals += [f"{S['q2s'+s]['per_class_iou'][c]:.2f}", f"{S['a2s'+s]['per_class_iou'][c]:.2f}", f"{d(c, s):+.2f}"]
    out.append(c + ',' + ','.join(vals) + f",{np.mean([d(c, s) for s in SEEDS]):+.2f}")
open(f'{D}/a2_vs_q2_perclass_3seed.csv', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
