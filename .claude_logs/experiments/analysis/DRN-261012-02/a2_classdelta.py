import json,sys,numpy as np
D=sys.argv[1]
S={k:json.load(open(f'{D}/{k}.json')) for k in ['q2s821','q2s902','a2s821','a2s902']}
for k,v in S.items(): print(k, v['mIoU'], v['model_path'].split('/')[-1], v['num_images'])
CL=list(S['q2s821']['per_class_iou'].keys())
THIN=['Pole','TrafficLight','Pedestrian','Static']
def m(k,ex=()): return np.nanmean([S[k]['per_class_iou'][c] for c in CL if c not in ex])
def th(s): return np.mean([S['a2s'+s]['per_class_iou'][c]-S['q2s'+s]['per_class_iou'][c] for c in THIN])
print('n classes',len(CL))
for s in ['821','902']:
    print(s, 'A2 25=%.2f Q2 25=%.2f d25=%+.2f | d24=%+.2f | thin4=%+.2f'%(m('a2s'+s),m('q2s'+s),m('a2s'+s)-m('q2s'+s), m('a2s'+s,('RailTrack',))-m('q2s'+s,('RailTrack',)), th(s)))
d25=np.mean([m('a2s'+s)-m('q2s'+s) for s in ['821','902']]); d24=np.mean([m('a2s'+s,('RailTrack',))-m('q2s'+s,('RailTrack',)) for s in ['821','902']])
print('mean2: 25 A2 %.2f Q2 %.2f d %+.2f | 24 A2 %.2f Q2 %.2f d %+.2f | thin4 d %+.2f'%(np.mean([m('a2s821'),m('a2s902')]),np.mean([m('q2s821'),m('q2s902')]),d25,np.mean([m('a2s821',('RailTrack',)),m('a2s902',('RailTrack',))]),np.mean([m('q2s821',('RailTrack',)),m('q2s902',('RailTrack',))]),d24,(th('821')+th('902'))/2))
rows=['class,q2s821,a2s821,d821,q2s902,a2s902,d902,d_mean']
for c in CL:
    a=[S[f'{p}s{s}']['per_class_iou'][c] for s in ['821','902'] for p in ['q2','a2']]
    rows.append(f'{c},{a[0]:.2f},{a[1]:.2f},{a[1]-a[0]:+.2f},{a[2]:.2f},{a[3]:.2f},{a[3]-a[2]:+.2f},{(a[1]-a[0]+a[3]-a[2])/2:+.2f}')
open(f'{D}/a2_vs_q2_perclass.csv','w').write('\n'.join(rows)+'\n')
print('\n'.join(rows))
