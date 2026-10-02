"""스모크 ①: 열화 커리큘럼의 현재 step 이 데이터로더 워커에 실제로 보이는가 (GPU 불필요).

배경: DGFusion (b)·CAFuser (b) 학습 로그에 "degradation.SHARED_STEP 가 설정되지 않았다 — severity 상한을
최대(1.0)로 고정한다" 경고가 워커 수(랭크 4 x 워커 2 = 8)만큼 찍혔다. 워커가 공유 step 을 못 받으면 커리큘럼
(0.3 -> 0.6 -> 1.0)이 한 번도 작동하지 않고 학습 내내 severity 가 1.0 이었다는 뜻이다.

방법: 실제 학습과 같은 경로로 train 로더를 만든다(Trainer.build_train_loader, 워커 2). 학습 쪽 훅과 똑같이
multiprocessing.Value 를 degradation.set_shared_step 으로 심은 뒤 iter(loader) 로 워커를 띄워, 워커가 찍는
`[degrade] ... severity상한=` 표식을 읽는다. --spawn 이면 실제 4장 학습처럼 랭크를 torch.multiprocessing.spawn 으로
띄우고 그 안에서 로더를 만든다(랭크 프로세스 = spawn, 데이터로더 워커 = 그 안에서의 기본 방식).
출력: 설정한 step 과 워커가 보고한 severity 상한의 대응.
사용: DEGRADE_LOG_EVERY=1 python smoke_cafb_worker_step.py [--spawn] [--steps 50000 100000 150000]
"""
import argparse
import io
import os
import re
import sys

os.environ.setdefault("DEGRADE_LOG_EVERY", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.getcwd())


def expected_cap(cfg_deg, step, max_iter):
    frac = step / float(max_iter)
    for cap, upto in zip(cfg_deg.CURRICULUM, cfg_deg.CURRICULUM_FRACTIONS):
        if frac < upto:
            return cap
    return cfg_deg.CURRICULUM[-1]


def run(rank, steps, config_file, report_dir):
    import multiprocessing

    from train_net import Trainer, setup
    from dgfusion.data import degradation as dg

    class A:
        opts = ["DATALOADER.NUM_WORKERS", "2", "SOLVER.IMS_PER_BATCH", "2"]
        eval_only = False
        inference_only = False
        resume = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "auto"
    A.config_file = config_file
    cfg = setup(A)
    max_iter = cfg.SOLVER.MAX_ITER
    lines = []
    for step in steps:
        shared = multiprocessing.Value("i", int(step))
        if os.environ.get("DEGRADE_STEP_FILE") == "1":
            # 옵션 B: 학습 훅과 같은 방식으로 파일 공유만 쓴다(SHARED_STEP 은 심지 않는다 — spawn 워커 조건 재현).
            dg.enable_step_file("/tmp/smoke_cafb_ws/stepfile")
            dg.write_step_file(int(step))
            dg._STEP_FILE_CACHE[0] = 0.0
        else:
            dg.set_shared_step(shared)
        print(f"=== STEP {step} 기대 상한 {expected_cap(cfg.DATASETS.DELIVER.DEGRADE, step, max_iter)} ===", flush=True)
        loader = Trainer.build_train_loader(cfg)
        it = iter(loader)                      # 여기서 워커가 fork/spawn 된다
        for _ in range(6):
            next(it)
        # 같은 로더에서 값을 바꿨을 때 워커가 따라가는지(학습 중 매 iteration 갱신과 같은 동작)
        shared.value = int(step) + 1
        if os.environ.get("DEGRADE_STEP_FILE") == "1":
            dg.write_step_file(int(step) + 1)
        print(f"=== STEP {step + 1} (같은 로더에서 갱신) 기대 상한 {expected_cap(cfg.DATASETS.DELIVER.DEGRADE, step + 1, max_iter)} ===", flush=True)
        for _ in range(4):
            next(it)
        del it, loader
        lines.append(step)
    open(os.path.join(report_dir, f"rank{rank}.txt"), "w").write(" ".join(map(str, lines)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", default="configs/deliver/swin/cafuser_swin_tiny_bs8_200k_deliver_clde_degrade.yaml")
    ap.add_argument("--steps", type=int, nargs="+", default=[50000, 100000, 150000])
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--nprocs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs("/tmp/smoke_cafb_ws", exist_ok=True)
    if args.spawn:
        import torch.multiprocessing as tmp
        tmp.spawn(run, args=(args.steps, args.config_file, "/tmp/smoke_cafb_ws"), nprocs=args.nprocs, join=True)
    else:
        run(0, args.steps, args.config_file, "/tmp/smoke_cafb_ws")
    print("DONE")
