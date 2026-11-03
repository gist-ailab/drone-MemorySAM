import csv, sys
base = sys.argv[1]
for split in ['val', 'test']:
    L = {r['present_modals']: float(r['mIoU']) for r in csv.DictReader(open(f'{base}/robust_l1/{split}_emm/{split}/missing_modality/emm.csv'))}
    A = {r['present_modals']: float(r['mIoU']) for r in csv.DictReader(open(f'{base}/robust_a2/s821/{split}_emm/{split}/missing_modality/emm.csv'))}
    print('==', split)
    for k in sorted(L, key=lambda k: (len(k.split('+')), k)):
        print(f'  {k:28s} A2 {A[k]:6.2f}  L1 {L[k]:6.2f}  Δ {L[k]-A[k]:+6.2f}')
