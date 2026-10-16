import csv,sys,collections
import numpy as np
pr=sys.argv[1]; th=sys.argv[2]; OUT=sys.argv[3]
R=collections.defaultdict(dict)
for r in csv.DictReader(open(pr)):
    R[(r['model'],r['condition'])][r['class_name']]=r
CL=list(R[('Q2s821_1024','all')].keys())
THIN={'Pole','Pedestrian','TrafficLight','Static'}
def iou(m,c,cd='all'):
    x=R[(m,cd)][c]['iou']; return float(x)*100 if x else np.nan
def miou(m,cd='all'): return np.nanmean([iou(m,c,cd) for c in CL])
models=['Q2s821_1024','Q2s821_r1280','Q2s821_r1536','Q2s821_r2048','DGF','CAF']
print({m:round(miou(m),3) for m in models}, len(CL))
out=['class,group,'+','.join(f'iou_{m}' for m in models)+',d1536,d2048,contrib1536_to_miou,tp_d1536,fp_d1536,fn_d1536']
rows=[]
for c in CL:
    d15=iou('Q2s821_r1536',c)-iou('Q2s821_1024',c); d20=iou('Q2s821_r2048',c)-iou('Q2s821_1024',c)
    a=R[('Q2s821_1024','all')][c]; b=R[('Q2s821_r1536','all')][c]
    dt,dfp,dfn=[int(b[k])-int(a[k]) for k in ('tp_pixels','fp_pixels','fn_pixels')]
    rows.append((d15,c)); out.append(f"{c},{'thin4' if c in THIN else 'other'},"+','.join(f'{iou(m,c):.2f}' for m in models)+f",{d15:+.2f},{d20:+.2f},{d15/len(CL):+.3f},{dt},{dfp},{dfn}")
open(f'{OUT}/deliver_q2s821_perclass.csv','w').write('\n'.join(out)+'\n')
tot=miou('Q2s821_r1536')-miou('Q2s821_1024')
thin=sum(d for d,c in rows if c in THIN)/len(CL)
rows.sort(reverse=True)
print(f'd1536 total {tot:+.3f}  thin4 contrib {thin:+.3f} ({thin/tot:.0%})  other {tot-thin:+.3f}')
pos=sum(d for d,c in rows if d>0)/len(CL)
cum=0
for d,c in rows[:10]:
    cum+=d/len(CL); print(f'  {c:14s} {d:+6.2f} contrib {d/len(CL):+.3f} cum {cum/tot:.0%}')
print('bottom:',[(c,round(d,2)) for d,c in rows[-5:]])
for cd in ['cloud','fog','night','rain','sun']:
    print(cd, f"d1536={miou('Q2s821_r1536',cd)-miou('Q2s821_1024',cd):+.2f} d2048={miou('Q2s821_r2048',cd)-miou('Q2s821_1024',cd):+.2f}  thin4 d1536={sum(iou('Q2s821_r1536',c,cd)-iou('Q2s821_1024',c,cd) for c in THIN)/len(CL):+.2f}")
# thickness: recovered components (detected) per bin
T=collections.defaultdict(dict)
for r in csv.DictReader(open(th)): T[(r['model'],r['class_group'])][r['thickness_bin_patch']]=r
print('\nthickness bins: n_comp, det1024->1536, poor1024->1536, DGF det/poor, gained detected comps, gained pixel-recall-weighted comps')
for g in ['thin_4','other_21']:
    tot_gain=0; gains={}
    for b in ['<0.25','0.25-0.5','0.5-1','1-2','2-4','>=4']:
        a=T[('Q2s821_1024',g)][b]; c=T[('Q2s821_r1536',g)][b]; d=T[('DGF',g)][b]
        n=int(a['n_components']); gd=int(c['n_detected'])-int(a['n_detected'])
        good_a=int(a['n_detected'])-int(a['n_found_but_poor']); good_c=int(c['n_detected'])-int(c['n_found_but_poor'])
        gains[b]=good_c-good_a; tot_gain+=good_c-good_a
        print(f"{g:8s} {b:8s} n={n:8d} det {float(a['detection_rate']):.3f}->{float(c['detection_rate']):.3f} (DGF {float(d['detection_rate']):.3f}) poor {float(a['found_but_poor_rate']):.3f}->{float(c['found_but_poor_rate']):.3f} (DGF {float(d['found_but_poor_rate']):.3f}) +det={gd} +well_drawn={good_c-good_a}")
    print(g,'share of +well_drawn comps by bin', {b:f'{v/tot_gain:.0%}' for b,v in gains.items()})
