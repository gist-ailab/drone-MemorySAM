#!/bin/bash
# 열화 킷을 적용한 뒤에도 (a) 발표 config 가 예전과 똑같이 동작하는지 본다.
# 기본값이 전부 꺼짐이어야 하고, 데이터로더가 만드는 배치가 바뀌면 안 된다.
set -uo pipefail
cd /SSDb/jemo_maeng/dgfusion_train
source /home/jemo_maeng/miniconda3/etc/profile.d/conda.sh
conda activate dgfusion
export PYTHONPATH=$PWD:$PWD/OneFormer DETECTRON2_DATASETS=$PWD/datasets WANDB_MODE=offline

python - <<'PY'
import sys
sys.argv = ["x"]
from train_net import setup


class A:
    config_file = "configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml"
    opts = []
    eval_only = True
    inference_only = False
    resume = False
    num_gpus = 1
    num_machines = 1
    machine_rank = 0
    dist_url = "auto"


cfg = setup(A)
d = cfg.DATASETS.DELIVER
print("[(a) 발표 config] RANDOM_DROP =", list(d.RANDOM_DROP))
print("[(a) 발표 config] DEGRADE.ENABLED =", d.DEGRADE.ENABLED)

# 매퍼가 실제로 열화를 끄는지: degrade_ctx 가 None 이어야 한다.
from dgfusion.data.dataset_mappers.deliver_semantic_dataset_mapper import (
    DELIVERSemanticDatasetMapper as M)
ret = M.from_config(cfg, is_train=True)
print("[(a) 발표 config] degrade_ctx =", ret.get("degrade_ctx"))

cfg2 = setup(type("B", (A,), {
    "config_file": "configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade.yaml"}))
ret2 = M.from_config(cfg2, is_train=True)
ctx = ret2.get("degrade_ctx")
print("[(b) 열화 config] degrade_ctx 있음 =", ctx is not None)
if ctx is not None:
    deg, means, max_iter = ctx
    print("   모달 평균 키 =", sorted(means))
    print("   MAX_ITER =", max_iter, "| ENABLED =", deg.ENABLED)

# 평가 경로에서는 학습이 아니므로 꺼져 있어야 한다.
ret3 = M.from_config(cfg2, is_train=False)
print("[(b) 열화 config, 평가 경로] degrade_ctx =", ret3.get("degrade_ctx"))
PY
