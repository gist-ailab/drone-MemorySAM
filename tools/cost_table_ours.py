#!/usr/bin/env python3
"""DRN-261027-02: ReliaDINO(A2·Q2·E1) 논문 비용표를 같은 입력에서 측정한다.

학습 파라미터는 체크포인트를 load_model로 읽은 직후 requires_grad=True인 수다.
FLOPs는 fp32 fvcore 기준(1 MAC = 1 flop)이며 미지원 연산은 별도로 기록한다.
지연은 고정된 실제 이미지 한 장을 warmup 뒤 CUDA 이벤트로 매회 재고, 평균·
중앙값·모표준편차를 ms/img로 낸다. 메모리는 측정 구간의 PyTorch 최대 할당·예약량이다.

예:
  python tools/cost_table_ours.py --cfg configs/eval/deliver.yaml \
    --model_path /path/to/model.pth --name A2 --out /tmp/a2_cost.json
  python tools/cost_table_ours.py --dry_run --device cpu --out /tmp/cost_dry.json
"""

import argparse
import hashlib
import json
import shlex
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def count_flops(model, inputs, dry_run=False):
    result = dict(gflops=None, flops_convention='fvcore (1 MAC = 1 flop)',
                  unsupported_ops={}, flops_error=None)
    try:
        from fvcore.nn import FlopCountAnalysis

        # fvcore는 keyword 인자를 전달하지 못하므로 평가와 같은 forward를 감싼다.
        class EvalForward(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, images):
                if dry_run:
                    return self.wrapped(images[0])
                return self.wrapped(images, multimask_output=True)

        with torch.no_grad():
            analysis = FlopCountAnalysis(EvalForward(model), (inputs,))
            result['gflops'] = analysis.total() / 1e9
            result['unsupported_ops'] = dict(analysis.unsupported_ops())
    except Exception as exc:
        result['flops_error'] = f'{type(exc).__name__}: {exc}'
    return result


def measure(model, inputs, device, dtype, warmup, iters, forward):
    use_cuda = device.type == 'cuda'
    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if use_cuda and dtype == 'bf16' else nullcontext
    with torch.no_grad():
        for _ in range(warmup):
            with autocast():
                forward(model, inputs)
        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            events = []
            for _ in range(iters):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                with autocast():
                    forward(model, inputs)
                end.record()
                events.append((start, end))
            torch.cuda.synchronize(device)
            samples = [start.elapsed_time(end) for start, end in events]
            allocated = torch.cuda.max_memory_allocated(device) / 1e9
            reserved = torch.cuda.max_memory_reserved(device) / 1e9
        else:
            samples = []
            for _ in range(iters):
                start = time.perf_counter()
                forward(model, inputs)
                samples.append((time.perf_counter() - start) * 1000)
            allocated = reserved = None
    return dict(latency_ms_mean=statistics.mean(samples),
                latency_ms_median=statistics.median(samples),
                latency_ms_std=statistics.pstdev(samples),
                peak_memory_allocated_gb=allocated,
                peak_memory_reserved_gb=reserved)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cfg')
    parser.add_argument('--model_path')
    parser.add_argument('--name', default='ReliaDINO')
    parser.add_argument('--out', required=True)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--dtype', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--split', default='test')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1 or args.size < 1:
        parser.error('--warmup >= 0, --iters >= 1, --size >= 1 이어야 합니다')
    device = torch.device(args.device)
    if args.dry_run:
        if device.type != 'cpu':
            parser.error('--dry_run은 --device cpu로 실행합니다')
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).to(device)
        params = count_params(model)
        model.eval()
        images = [torch.randn(1, 3, 16, 16, device=device)]
        forward = lambda m, xs: m(xs[0])
        cfg_path = checkpoint_path = checkpoint_md5 = None
    else:
        if not args.cfg or not args.model_path:
            parser.error('--cfg와 --model_path가 필요합니다')
        if device.type != 'cuda' or not torch.cuda.is_available():
            parser.error('실제 비용 측정에는 사용 가능한 CUDA GPU가 필요합니다')
        torch.cuda.set_device(device)
        import yaml
        import val as valmod

        cfg_path = str(Path(args.cfg).resolve())
        checkpoint = Path(args.model_path).resolve()
        with Path(cfg_path).open(encoding='utf-8') as stream:
            cfg = yaml.safe_load(stream)
        model = valmod.load_model(cfg, checkpoint, device)
        params = count_params(model)
        checkpoint_path = str(checkpoint)
        checkpoint_md5 = file_md5(checkpoint)
        model.eval()
        dataset_cfg = cfg['DATASET']
        transform = valmod.get_val_augmentation(args.size, dataset_cfg=dataset_cfg)
        dataset, _ = valmod.create_dataset(dataset_cfg, args.split, transform, args.split,
                                           macvi=False, eval_day=False)
        loader = DataLoader(dataset, batch_size=1, collate_fn=valmod._collate_fn)
        images, _, _ = next(iter(loader))
        images = [x.to(device) for x in images]
        forward = lambda m, xs: m(xs, multimask_output=True)

    flops = count_flops(model, images, dry_run=args.dry_run)
    timing = measure(model, images, device, args.dtype, args.warmup, args.iters, forward)
    result = dict(name=args.name, cfg_path=cfg_path, checkpoint_path=checkpoint_path,
                  checkpoint_md5=checkpoint_md5, gpu_name=torch.cuda.get_device_name(device)
                  if device.type == 'cuda' else None,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=str(device), dtype=args.dtype,
                  effective_dtype=args.dtype if device.type == 'cuda' else 'fp32',
                  split=args.split, input_shapes=[list(x.shape) for x in images],
                  batch_size=images[0].shape[0], warmup=args.warmup, iters=args.iters,
                  command=shlex.join(sys.argv), argv=sys.argv, dry_run=args.dry_run,
                  trainer_module=None, trainer_class=None, eval_size=args.size,
                  **params, **flops, **timing)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{args.name}: {result["latency_ms_mean"]:.3f} ms/img, '
          f'{result["gflops"]} GFLOPs -> {out}')


if __name__ == '__main__':
    main()
