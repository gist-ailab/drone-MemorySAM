import sys
sys.argv = ["x"]
from train_net import setup
from cafuser.data.dataset_mappers.deliver_semantic_dataset_mapper import (
    DELIVERSemanticDatasetMapper as M)


class A:
    config_file = "configs/deliver/swin/cafuser_swin_tiny_bs8_200k_deliver_clde.yaml"
    opts = []
    eval_only = True
    inference_only = False
    resume = False
    num_gpus = 1
    num_machines = 1
    machine_rank = 0
    dist_url = "auto"


cfg_a = setup(A)
ret_a = M.from_config(cfg_a, is_train=True)
print("[(a) CAFuser 발표 config] RANDOM_DROP =", list(cfg_a.DATASETS.DELIVER.RANDOM_DROP))
print("[(a) CAFuser 발표 config] degrade_ctx =", ret_a.get("degrade_ctx"))

cfg_b = setup(type("B", (A,), {
    "config_file": "configs/deliver/swin/cafuser_swin_tiny_bs8_200k_deliver_clde_degrade.yaml"}))
ret_b = M.from_config(cfg_b, is_train=True)
ctx = ret_b.get("degrade_ctx")
print("[(b) CAFuser 열화 config] RANDOM_DROP =", list(cfg_b.DATASETS.DELIVER.RANDOM_DROP))
print("[(b) CAFuser 열화 config] degrade_ctx 있음 =", ctx is not None)
if ctx is not None:
    deg, means, max_iter = ctx
    print("   모달 평균 키 =", sorted(means), "| MAX_ITER =", max_iter,
          "| ENABLED =", deg.ENABLED)

ret_b_eval = M.from_config(cfg_b, is_train=False)
print("[(b) CAFuser 열화 config, 평가 경로] degrade_ctx =", ret_b_eval.get("degrade_ctx"))
