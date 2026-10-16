import json,os,sys
import numpy as np
D=sys.argv[1]; OUT=sys.argv[2]
os.makedirs(OUT,exist_ok=True)
S=['824','825','826']; R=['1024','1536','2048']
rep={(s,r):json.load(open(f'{D}/{s}_{r}.json')) for s in S for r in R}
CL=list(rep[('824','1024')]['official_native_1080x1920']['per_class'].keys())
CONDS=list(rep[('824','1024')]['per_condition_official'].keys())
THIN={'pole','traffic light','traffic sign','person','rider','bicycle','motorcycle'}
def v(x): return np.nan if x is None else float(x)
# overall per-class
lines=['class,group,'+','.join(f'iou{r}_mean' for r in R)+',d1536_mean,d2048_mean,d1536_seeds,d2048_seeds']
ov={}
for c in CL:
    a={r:np.array([v(rep[(s,r)]['official_native_1080x1920']['per_class'][c]) for s in S]) for r in R}
    d15=a['1536']-a['1024']; d20=a['2048']-a['1024']
    ov[c]=(d15,d20)
    lines.append(f"{c},{'thin' if c in THIN else 'area'},"+','.join(f'{np.nanmean(a[r]):.2f}' for r in R)+f",{np.nanmean(d15):+.2f},{np.nanmean(d20):+.2f},{'/'.join(f'{x:+.2f}' for x in d15)},{'/'.join(f'{x:+.2f}' for x in d20)}")
open(f'{OUT}/muses_overall_perclass.csv','w').write('\n'.join(lines)+'\n')
n=len(CL)
tot15=np.mean([np.nanmean(ov[c][0]) for c in CL]); tot20=np.mean([np.nanmean(ov[c][1]) for c in CL])
print('overall miou check', {r:np.mean([rep[(s,r)]['official_native_1080x1920']['mIoU_present_classes'] for s in S]) for r in R})
print(f'sum of class contrib d1536={tot15:+.3f} d2048={tot20:+.3f}')
for g in ['thin','area']:
    cs=[c for c in CL if (c in THIN)==(g=='thin')]
    print(g, len(cs), f"contrib1536={sum(np.nanmean(ov[c][0]) for c in cs)/n:+.3f} contrib2048={sum(np.nanmean(ov[c][1]) for c in cs)/n:+.3f}")
print('top 1536 gain:', sorted(((np.nanmean(ov[c][0]),c) for c in CL),reverse=True)[:8])
print('top 2048 loss:', sorted(((np.nanmean(ov[c][1]),c) for c in CL))[:8])
# per condition
rows=['condition,class,group,n_present,'+','.join(f'iou{r}_mean' for r in R)+',d1536_mean,d2048_mean,contrib2048_to_cond_miou,d2048_seeds']
cond_summary=['condition,n_images,'+','.join(f'miou{r}_mean' for r in R)+',d1536,d2048,d1536_seeds,d2048_seeds,top3_share_of_2048_drop,top3_classes']
night_contrib={}; c15={}
for cd in CONDS:
    pc={r:[rep[(s,r)]['per_condition_official'][cd] for s in S] for r in R}
    m={r:np.array([p['mIoU_present_classes'] for p in pc[r]]) for r in R}
    npres=pc['1024'][0]['n_present_classes']
    assert all(p['n_present_classes']==npres for r in R for p in pc[r]), cd
    contribs=[]
    for c in CL:
        a={r:np.array([v(p['per_class'].get(c)) for p in pc[r]]) for r in R}
        if np.all(np.isnan(a['1024'])): continue
        d15=a['1536']-a['1024']; d20=a['2048']-a['1024']
        if np.all(np.isnan(d20)): continue
        ct=np.nanmean(d20)/npres; ct15=np.nanmean(d15)/npres
        contribs.append((ct,c))
        night_contrib.setdefault(cd,{})[c]=ct
        c15.setdefault(cd,{})[c]=ct15
        rows.append(f"{cd},{c},{'thin' if c in THIN else 'area'},{npres},"+','.join(f'{np.nanmean(a[r]):.2f}' for r in R)+f",{np.nanmean(d15):+.2f},{np.nanmean(d20):+.2f},{ct:+.3f},{'/'.join(f'{x:+.2f}' for x in d20)}")
    D20=np.mean(m['2048']-m['1024']); contribs.sort()
    top3=contribs[:3]; share=sum(x for x,_ in top3)/D20 if D20<0 else float('nan')
    cond_summary.append(f"{cd},{pc['1024'][0]['n_images']},"+','.join(f'{np.mean(m[r]):.2f}' for r in R)+f",{np.mean(m['1536']-m['1024']):+.2f},{D20:+.2f},{'/'.join(f'{x:+.2f}' for x in m['1536']-m['1024'])},{'/'.join(f'{x:+.2f}' for x in m['2048']-m['1024'])},{share:.2f},{';'.join(f'{c}({x:+.2f})' for x,c in top3)}")
open(f'{OUT}/muses_condition_perclass.csv','w').write('\n'.join(rows)+'\n')
open(f'{OUT}/muses_condition_summary.csv','w').write('\n'.join(cond_summary)+'\n')
print('\n'.join(cond_summary))
# night aggregate: mean over 4 night conditions of class contributions (equal weight conditions)
for tod,src in [('night',night_contrib),('day',night_contrib),('night1536',c15),('day1536',c15)]:
    cds=[c for c in CONDS if c.split('/')[1]==tod.replace('1536','')]
    agg={}
    for cd in cds:
        for c,x in src[cd].items(): agg[c]=agg.get(c,0)+x/len(cds)
    tot=sum(agg.values())
    s=sorted(agg.items(),key=lambda t:t[1])
    thin=sum(x for c,x in agg.items() if c in THIN)
    print(tod,'mean cond d2048=',f'{tot:+.3f}','thin part',f'{thin:+.3f}','area part',f'{tot-thin:+.3f}','top3 share',f'{sum(x for _,x in s[:3])/tot:.2f}',[(c,round(x,3)) for c,x in s[:6]])
    open(f'{OUT}/muses_{tod}_contrib2048.csv','w').write('class,group,mean_contrib_to_cond_miou_2048\n'+'\n'.join(f"{c},{'thin' if c in THIN else 'area'},{x:+.4f}" for c,x in s)+'\n')
