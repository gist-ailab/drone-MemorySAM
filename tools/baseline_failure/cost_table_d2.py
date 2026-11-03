#!/usr/bin/env python3
"""DRN-261027-02: DGFusion·CAFuser 논문 비용표를 기준선 저장소에서 측정한다.

이 파일을 기준선 저장소 루트에 복사해 그 환경에서 실행한다. 학습 파라미터는
Trainer.build_model 직후 requires_grad=True인 수다. FLOPs는 fp32 fvcore 기준
(1 MAC = 1 flop)이고 미지원 연산별 횟수를 남긴다. 지연은 고정한 첫 테스트 배치
한 장을 warmup 뒤 CUDA 이벤트로 매회 재고 ms/img 평균·중앙값·모표준편차를 낸다.
메모리는 측정 구간의 PyTorch 최대 할당·예약량이다.

예:
  python cost_table_d2.py --config-file configs/dgfusion.yaml \
    --weights output/model.pth --trainer-module test_net --trainer-class Tester \
    --name DGFusion --out /tmp/dgfusion_cost.json
  python cost_table_d2.py --config-file configs/cafuser.yaml \
    --weights output/model.pth --name CAFuser --out /tmp/cafuser_cost.json
"""

import argparse
import hashlib
import importlib
import json
import shlex
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path


def file_md5(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total_params=total, trainable_params=trainable,
                total_params_m=total / 1e6, trainable_params_m=trainable / 1e6)


def count_flops(model, inputs):
    result = dict(gflops=None, flops_convention='fvcore (1 MAC = 1 flop)',
                  unsupported_ops={}, flops_error=None)
    try:
        from fvcore.nn import FlopCountAnalysis

        with torch.no_grad():
            analysis = FlopCountAnalysis(model, (inputs,))
            result['gflops'] = analysis.total() / 1e9
            result['unsupported_ops'] = dict(analysis.unsupported_ops())
    except Exception as exc:
        result['flops_error'] = f'{type(exc).__name__}: {exc}'
    return result


def measure(model, inputs, device, dtype, warmup, iters):
    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if dtype == 'bf16' else nullcontext
    with torch.no_grad():
        for _ in range(warmup):
            with autocast():
                model(inputs)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        events = []
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast():
                model(inputs)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize(device)
        samples = [start.elapsed_time(end) for start, end in events]
        allocated = torch.cuda.max_memory_allocated(device) / 1e9
        reserved = torch.cuda.max_memory_reserved(device) / 1e9
    return dict(latency_ms_mean=statistics.mean(samples),
                latency_ms_median=statistics.median(samples),
                latency_ms_std=statistics.pstdev(samples),
                peak_memory_allocated_gb=allocated,
                peak_memory_reserved_gb=reserved)


def input_shapes(inputs):
    # detectron2 배치: list[dict]. 실제 텐서 필드의 shape를 모두 남긴다.
    return [{key: list(value.shape) for key, value in sample.items()
             if isinstance(value, torch.Tensor)} for sample in inputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--trainer-module', default='train_net')
    parser.add_argument('--trainer-class', default='Trainer')
    parser.add_argument('--eval-size', type=int, default=1024)
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--opts', nargs='*', default=[])
    parser.add_argument('--name', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--dtype', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.eval_size < 1:
        parser.error('--warmup >= 0, --iters >= 1, --eval-size >= 1 이어야 합니다')

    # 기준선 저장소의 의존성은 실행 시점에만 가져온다.
    global torch
    import torch
    from detectron2.checkpoint import DetectionCheckpointer

    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('실제 비용 측정에는 사용 가능한 CUDA GPU가 필요합니다')
    torch.cuda.set_device(device)
    trainer_mod = importlib.import_module(args.trainer_module)
    TrainerCls = getattr(trainer_mod, args.trainer_class)
    setup = getattr(trainer_mod, 'setup')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    class _A:
        config_file = args.config_file
        opts = ['MODEL.WEIGHTS', args.weights,
                'MODEL.DEVICE', str(device),
                'MODEL.IS_TRAIN', 'False',
                'DATASETS.TEST_SEMANTIC', f"('deliver_semantic_{args.split}',)",
                'DATASETS.DELIVER.EVAL_SIZE', str(args.eval_size),
                'OUTPUT_DIR', str(out.parent / '_d2log')] + list(args.opts)
        eval_only = True
        inference_only = False
        resume = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = 'auto'

    cfg = setup(_A)
    model = TrainerCls.build_model(cfg)
    params = count_params(model)
    checkpoint = Path(args.weights).resolve()
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        args.weights, resume=False)
    model.eval()

    ds_name = cfg.DATASETS.TEST_SEMANTIC[0]
    loader = TrainerCls.build_test_loader(cfg, ds_name)
    inputs = next(iter(loader))
    if len(inputs) != 1:
        raise ValueError(f'비교 조건은 batch_size=1입니다: loader batch={len(inputs)}')

    flops = count_flops(model, inputs)
    timing = measure(model, inputs, device, args.dtype, args.warmup, args.iters)
    result = dict(name=args.name, cfg_path=str(Path(args.config_file).resolve()),
                  checkpoint_path=str(checkpoint), checkpoint_md5=file_md5(checkpoint),
                  gpu_name=torch.cuda.get_device_name(device),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=str(device), dtype=args.dtype, effective_dtype=args.dtype,
                  split=args.split, input_shapes=input_shapes(inputs), batch_size=len(inputs),
                  warmup=args.warmup, iters=args.iters, command=shlex.join(sys.argv),
                  argv=sys.argv, dry_run=False, trainer_module=args.trainer_module,
                  trainer_class=args.trainer_class, eval_size=args.eval_size,
                  **params, **flops, **timing)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{args.name}: {result["latency_ms_mean"]:.3f} ms/img, '
          f'{result["gflops"]} GFLOPs -> {out}')


if __name__ == '__main__':
    main()
