#!/bin/bash
# DGFusion(timbroed/DGFusion) 학습 환경 구축 — jarvis에서 2026-09-08 실전 검증된 통합본.
# 공식 INSTALL.md 절차에, 실제로 부딪힌 4가지 빌드 문제의 픽스를 전부 반영했다:
#   1) detectron2/mmcv: pip 빌드 격리 안에 torch/pkg_resources가 없어 실패 -> --no-build-isolation
#   2) CUDA 11.8 nvcc가 gcc 12를 거부 -> CC/CXX/CUDAHOSTCXX=gcc-11 강제
#   3) mmcv 1.6.2 setup.py가 구식 pkg_resources 요구 -> setuptools==59.5.0 선치
#   4) shi-labs.com natten wheel 인덱스 SSL 인증서 만료 -> --trusted-host로 해당 호스트만 우회
# 사용: TARGET_DIR과 CUDA_HOME을 서버에 맞게 바꿔 실행. conda env 이름 = dgfusion.
set -euo pipefail

TARGET_DIR=${TARGET_DIR:-/SSDb/jemo_maeng/dgfusion_train}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-11.8}
export PATH=$CUDA_HOME/bin:$PATH
export CC=${CC:-/usr/bin/gcc-11}
export CXX=${CXX:-/usr/bin/g++-11}
export CUDAHOSTCXX=$CXX
DELIVER_DIR=${DELIVER_DIR:-/SSDb/jemo_maeng/dset/DELIVER}

source $CONDA_ROOT/etc/profile.d/conda.sh

mkdir -p "$(dirname "$TARGET_DIR")"
[ -d "$TARGET_DIR" ] || git clone https://github.com/timbroed/DGFusion.git "$TARGET_DIR"
cd "$TARGET_DIR"

conda env list | grep -q "^dgfusion " || conda create -y -n dgfusion python=3.9
conda activate dgfusion

pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U opencv-python

python tools/setup_detectron2.py
pip install -e detectron2 --no-build-isolation

python tools/setup_oneformer.py
(cd OneFormer && git checkout 4962ef6a96ffb76a76771bfa3e8b3587f209752b)
python tools/setup_cafuser.py

pip install "git+https://github.com/timbroed/MUSES.git"
pip install "git+https://github.com/cocodataset/panopticapi.git"
pip install "git+https://github.com/mcordts/cityscapesScripts.git"

pip install setuptools==59.5.0
pip install mmcv==1.6.2 --no-build-isolation
pip install natten==0.17.1 -f https://shi-labs.com/natten/wheels/cu118/torch2.3.0/index.html --trusted-host shi-labs.com
grep -v -e '^-f https://shi-labs.com' -e '^natten' requirements.txt > /tmp/req_nonatten.txt
pip install -r /tmp/req_nonatten.txt
pip install wandb

# MSDeformAttn CUDA kernel
(cd OneFormer/oneformer/modeling/pixel_decoder/ops && sh make.sh)

# 학습 복원본 적용 (이 폴더의 산출물)
RESTORE_DIR=$(cd "$(dirname "$0")" && pwd)
cp "$RESTORE_DIR/train_net.py" train_net.py
(cd "$TARGET_DIR" && git apply "$RESTORE_DIR/dgfusion_training_restore.patch")

# 사전학습 백본
mkdir -p pretrained
if [ ! -f pretrained/swin_tiny_patch4_window7_224_22k.pkl ]; then
  wget -q -P pretrained https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth
  python tools/convert-pretrained-model-to-d2.py pretrained/swin_tiny_patch4_window7_224_22k.pth pretrained/swin_tiny_patch4_window7_224_22k.pkl
  rm pretrained/swin_tiny_patch4_window7_224_22k.pth
fi

# DELIVER 심링크 (반드시 로컬 디스크 사본 — sshfs 금지)
mkdir -p datasets
[ -e datasets/deliver ] || ln -s "$DELIVER_DIR" datasets/deliver

export PYTHONPATH=$TARGET_DIR/OneFormer:$TARGET_DIR
python - <<'EOF'
import torch, detectron2, natten
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
import MultiScaleDeformableAttention
print('MSDeformAttn OK')
import oneformer, cafuser, dgfusion
from detectron2.data import DatasetCatalog
print('deliver train records:', len(DatasetCatalog.get('deliver_semantic_train')))
EOF
echo "SETUP_DONE"
