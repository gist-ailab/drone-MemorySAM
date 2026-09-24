"""열화가 실제로 걸리는지 학습과 같은 코드 경로로 증명한다.

돌고 있는 학습을 건드리지 않고, 같은 config 로 학습용 매퍼를 만들어 몇 장을 통과시킨다.
데이터로더 워커의 로그가 학습 로그 파일로 오지 않아, 표식을 직접 보기 위한 것이다.
모달마다 어떤 연산자가 걸렸고 화소가 얼마나 바뀌었는지를 찍는다.
"""
import os
import sys

sys.argv = ["x"]
os.environ.setdefault("DEGRADE_LOG_EVERY", "1")

import numpy as np  # noqa: E402
from train_net import setup  # noqa: E402


class A:
    config_file = ("configs/deliver/swin/"
                   "dgfusion_swin_tiny_bs8_200k_deliver_clde_degrade.yaml")
    opts = []
    eval_only = True
    inference_only = False
    resume = False
    num_gpus = 1
    num_machines = 1
    machine_rank = 0
    dist_url = "auto"


cfg = setup(A)
from dgfusion.data import degradation as D  # noqa: E402
from dgfusion.data.dataset_mappers.deliver_semantic_dataset_mapper import (  # noqa: E402
    DELIVERSemanticDatasetMapper as M)

print("\n[증명] DEGRADE.ENABLED =", cfg.DATASETS.DELIVER.DEGRADE.ENABLED,
      "| RANDOM_DROP =", list(cfg.DATASETS.DELIVER.RANDOM_DROP))

ret = M.from_config(cfg, is_train=True)
deg_cfg, means, max_iter = ret["degrade_ctx"]
print("[증명] 모달 평균 키 =", sorted(means), "| MAX_ITER =", max_iter)

# 학습 진행도별 severity 상한. 학습 훅이 심는 공유 step 을 흉내 낸다.
for frac in (0.0, 0.5, 0.99):
    step = int(max_iter * frac)

    class _V:
        value = step

    D.set_shared_step(_V())
    print(f"[증명] 진행도 {frac:.0%} (step {step:,d}) -> severity 상한 "
          f"{D.current_severity(deg_cfg, max_iter):.3f}")

# 실제 이미지 크기의 합성 배열로 모달별 적용을 확인한다(데이터셋 읽기 없이).
D.set_shared_step(type("V", (), {"value": int(max_iter * 0.99)})())
sev = D.current_severity(deg_cfg, max_iter)
rng = np.random.default_rng(0)
print(f"\n[증명] severity 상한 {sev:.3f} 에서 모달별 적용 표본")
for modality in ["CAMERA", "DEPTH", "LIDAR", "EVENT"]:
    img = rng.integers(0, 256, size=(1042, 1042, 3), dtype=np.uint8)
    mean = means[modality]
    for _ in range(3):
        out = D.apply_degradation(img, modality, mean, sev, rng)
        changed = float((img != out).mean())
        print(f"    {modality:7s} 바뀐 화소 비율 {changed:.4f}")

# 꺼짐 확인 — 발표 config 에서는 열화가 걸리지 않아야 한다.
cfg_a = setup(type("B", (A,), {
    "config_file": "configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml"}))
print("\n[증명] (a) 발표 config 의 degrade_ctx =",
      M.from_config(cfg_a, is_train=True).get("degrade_ctx"))
