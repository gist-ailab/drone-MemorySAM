"""피아식별(IFF) 분류기 체크포인트 → 드론 탑재용 ONNX 내보내기 + 검증.

인증 대상 분류기(MobileNetV3-small, classifier→2-class)를 i.MX NPU(eIQ)에서
돌릴 수 있는 ONNX 로 확정한다. 내보낸 뒤 같은 스크립트 안에서 곧바로 검증한다
(구조 검사 + PyTorch와 onnxruntime 결과 대조). 검증이 실패하면 exit 1.

🔴 opset 은 12 로 고정해야 한다 — i.MX eIQ 변환기가 opset 12 까지만 안정적으로
받아들이기 때문. torch 2.9+ 의 기본 dynamo exporter 는 opset 요청을 무시하고 18 로
올려버리므로, 반드시 dynamo=False(레거시 torch.onnx.export 경로)로 내보낸다.

  python export_iff_onnx.py --ckpt weights/iff_mobilenetv3.pt --crop 128
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from iff_eval import build_model

GRN, RED, BOLD, RST = '\033[32m', '\033[31m', '\033[1m', '\033[0m'

# ImageNet 정규화 — iff_eval.build_transforms 의 test 변환과 동일해야 한다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='weights/iff_mobilenetv3.pt',
                    help='피아식별 분류기 체크포인트 (iff_eval.py 가 저장한 형식)')
    ap.add_argument('--out', default='outputs/onnx/iff_mobilenetv3_op12.onnx',
                    help='내보낼 ONNX 경로')
    ap.add_argument('--crop', type=int, default=None,
                    help='입력 한 변 픽셀 (기본은 체크포인트의 crop, 없으면 128)')
    ap.add_argument('--opset', type=int, default=12,
                    help='ONNX opset (i.MX eIQ 대상이라 12 고정)')
    args = ap.parse_args()

    dev = torch.device('cpu')          # 내보내기·검증은 CPU 에서 (재현성·환경 무관)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f'체크포인트가 없습니다: {args.ckpt}\n'
                         f'  먼저 학습하세요:  python iff_eval.py --mode train '
                         f'--data <크롭 디렉터리> --ckpt {args.ckpt}')

    # ---- 체크포인트에서 클래스·크롭을 읽어 모델을 iff_eval 과 동일하게 구성 ----
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    classes = ck.get('classes', ['Allies', 'Enemies'])
    crop = args.crop if args.crop is not None else int(ck.get('crop', 128))

    model = build_model(dev, n_cls=len(classes))
    model.load_state_dict(ck['state_dict'])
    model.eval()

    print(f'\n{BOLD}╔═══ 피아식별(IFF) ONNX 내보내기 ═══╗{RST}')
    print(f'  Checkpoint : {args.ckpt}  (학습 seed {ck.get("seed")}, {ck.get("epochs")} ep)')
    print(f'  Classes    : {classes}')
    print(f'  Input      : 1x3x{crop}x{crop} RGB, ImageNet 정규화, batch 1 고정')
    print(f'  Opset      : {args.opset}  (i.MX eIQ 대상 — dynamo=False 레거시 경로)')

    # ---- ONNX 내보내기 ----
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    dummy = torch.randn(1, 3, crop, crop, dtype=torch.float32)   # batch 1 고정
    torch.onnx.export(
        model, dummy, args.out,
        input_names=['input'], output_names=['logits'],
        opset_version=args.opset,
        do_constant_folding=True,
        # dynamo=False 필수 — torch 2.9+ 기본 dynamo exporter 는 opset 요청을
        # 무시하고 18 로 올려버려서 eIQ 변환이 깨진다. 레거시 경로만 opset 12 를
        # 실제로 지킨다.
        dynamo=False,
    )
    print(f'  내보내기 완료 -> {args.out}')

    # ==== 검증 ====
    ok = verify(args.out, model, crop, dummy)
    sys.exit(0 if ok else 1)


def verify(onnx_path: str, torch_model, crop: int, sample: torch.Tensor) -> bool:
    """내보낸 ONNX 를 구조·수치 양쪽으로 검증한다.

    구조: onnx.checker 통과 + opset/IR/입출력 shape/노드수/연산자 종류 출력.
    수치: 같은 랜덤 입력에 대해 PyTorch 와 onnxruntime 결과의 최대 절대차가
    atol=1e-4 이내인지 판정. 둘 다 통과해야 True.
    """
    import onnx
    import onnxruntime as ort

    print(f'\n{BOLD}╔═══ 검증 ═══╗{RST}')

    # ---- 구조 검사 ----
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    opsets = {op.domain or 'ai.onnx': op.version for op in m.opset_import}

    def _shape(vi):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField('dim_value') else (d.dim_param or '?'))
        return dims

    ins = [(vi.name, _shape(vi)) for vi in m.graph.input]
    outs = [(vi.name, _shape(vi)) for vi in m.graph.output]
    op_types = sorted({n.op_type for n in m.graph.node})

    print(f'  onnx.checker : {GRN}통과{RST}')
    print(f'  opset        : {opsets}')
    print(f'  IR version   : {m.ir_version}')
    print(f'  입력         : {ins}')
    print(f'  출력         : {outs}')
    print(f'  노드 수      : {len(m.graph.node)}')
    print(f'  연산자 종류  : {len(op_types)}종 — {", ".join(op_types)}')

    # ---- 수치 대조 (PyTorch vs onnxruntime) ----
    x = torch.randn(1, 3, crop, crop, dtype=torch.float32)
    with torch.no_grad():
        ref = torch_model(x).cpu().numpy()
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    got = sess.run(['logits'], {'input': x.numpy()})[0]

    max_abs = float(np.max(np.abs(ref - got)))
    close = np.allclose(ref, got, atol=1e-4)
    print(f'\n  PyTorch vs onnxruntime')
    print(f'    최대 절대차 : {max_abs:.3e}')
    tag = f'{GRN}일치{RST}' if close else f'{RED}불일치{RST}'
    print(f'    allclose(atol=1e-4) : {close}  {tag}')

    if close:
        print(f'\n  {GRN}►►  검증 통과 — 드론 탑재용 ONNX 확정  ◄◄{RST}')
    else:
        print(f'\n  {RED}►►  검증 실패 — 수치 오차 초과 (atol=1e-4)  ◄◄{RST}')
    print(f'{BOLD}╚════════════╝{RST}\n')
    return close


if __name__ == '__main__':
    main()
