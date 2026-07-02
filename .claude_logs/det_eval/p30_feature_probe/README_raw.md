# P30-Det feature probe — raw 값 위치
경량 산출물(panel/summary/stats/script)은 이 폴더에 커밋됨.
**raw 텐서(mem/fused/reliability/router/det per image, 24개 npz, ~150MB)** 는 용량상 git 제외:
- 로컬 canonical: `/mnt/HDD2/src/logs/P29_vs_P30_v2_20260702/p30_feature_probe_full/raw/`
- hinton: `~/src/dm_eval/out_probe_p30_ep39_full/raw/`
- 재생성: `python probe_det_features.py --cfg <cfg> --det_checkpoint <ckpt> --out_dir <d> --n 24 --classes 8,6,7,9,4,3`
로드 예: `import numpy as np; d=np.load('<image_id>.npz'); d['mem_img']  # (256,64,64) fp16`
