#!/bin/bash
# CAFuser training setup on lecun, reusing conda env `dgfusion` from the July DGFusion eval.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion

cd /SSDb/jemo_maeng
if [ ! -d cafuser_train ]; then
  git clone https://github.com/timbroed/CAFuser.git cafuser_train
fi
cd cafuser_train

# OneFormer pinned to the commit INSTALL.md specifies (MSDeformAttn kernel already in the env)
if [ ! -d OneFormer ]; then
  git clone https://github.com/SHI-Labs/OneFormer
fi
(cd OneFormer && git checkout 4962ef6a96ffb76a76771bfa3e8b3587f209752b)

# pretrained backbone -> d2 format
mkdir -p pretrained
if [ ! -f pretrained/swin_tiny_patch4_window7_224_22k.pkl ]; then
  wget -q -P pretrained https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth
  python tools/convert-pretrained-model-to-d2.py pretrained/swin_tiny_patch4_window7_224_22k.pth pretrained/swin_tiny_patch4_window7_224_22k.pkl
  rm pretrained/swin_tiny_patch4_window7_224_22k.pth
fi

# dataset symlink -> LOCAL copy (never the sshfs mount)
mkdir -p datasets
rm -f datasets/deliver
ln -s /SSDb/jemo_maeng/dset/DELIVER datasets/deliver

# import smoke test (torch first, then the stack; dataset registration on import)
export PYTHONPATH=/SSDb/jemo_maeng/cafuser_train/OneFormer:/SSDb/jemo_maeng/cafuser_train
python -c "
import torch
import MultiScaleDeformableAttention
print('MSDeformAttn OK')
import oneformer, cafuser
from detectron2.data import DatasetCatalog
d = DatasetCatalog.get('deliver_semantic_train')
print('deliver train records:', len(d))
"
echo "SETUP_DONE"
