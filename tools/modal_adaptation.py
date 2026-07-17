#!/usr/bin/env python3
"""
tools/modal_adaptation.py — [분석항목 1] VFM 인코더에서 non-RGB 모달리티가 adapter로
얼마나 적응했는가 — model-agnostic, P31/32/33/34+ 재사용.

Measures, PER MODALITY, how much the injected adapters (SoftMoE-LoRA / plain LoRA)
actually change the frozen VFM encoder's behavior, by running the SAME forward twice —
adapters ACTIVE vs adapters ZEROED (forward hook `output*0` on every adapter module,
valid because every adapter site is ADDITIVE: qkv += moe(x) / qkv += B(A(x))) — and
diffing the per-modal signals the model already exposes:

  A. encoder-feature shift : ||f_on − f_off|| / ||f_off||, cos(f_on, f_off)
                             (from `_last_per_modal_feats`, fpn[0] 32ch stride-4)
  B. output shift          : same on per-modal seg logits (`_last_per_modal_outputs`)
  C. per-modal accuracy Δ  : pixel-acc(on) − pixel-acc(off) of each modality's
                             STANDALONE prediction vs GT — "adapter가 이 모달의 인식을
                             실제로 개선했는가" (적응의 최종 지표)

Interpretation: RGB는 VFM 사전학습 도메인이라 shift가 작아도 정상. non-RGB(depth/
event/lidar)의 shift/Δacc가 RGB보다 작으면 = adapter가 그 모달에 적응하지 못한 것
(dead adaptation). 정적 보완 지표(per-site ||dW||, per-expert 사용률)는
tools/adapter_health.py (state_dict-only)가 담당 — 함께 읽을 것.

Usage:
  python tools/modal_adaptation.py --cfg <model.yaml> --model_path <ckpt> \
    --dataset-root <DELIVER> --conditions night,sun --max-imgs 40 --gpu 0 --out <prefix>

Output: <out>.json + <out>.md  (per-condition × per-modality table)
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import val as V

WR = 256


def rs_nn(t, hw):
    return F.interpolate(t[None, None].float(), size=(hw, hw), mode='nearest')[0, 0].long()


def find_adapter_modules(core):
    """Every additive adapter module on the encoder path, family-agnostic.
    Returns (modules, kind, mechanism). SAM2 SoftMoE family exposes moe_layers_q/v;
    plain-LoRA families inject linear_b_* (output added → hook o*0); ReliaDINO(P34)의
    MultiModalLoRAQKV는 delta가 내부에서 합쳐지므로 `scale=0` 토글로 끈다."""
    mods = []
    if hasattr(core, 'moe_layers_q'):
        mods = list(core.moe_layers_q) + list(getattr(core, 'moe_layers_v', []))
        if mods:
            return mods, 'softmoe_lora', 'hook'
    for name, m in core.named_modules():
        if name.rsplit('.', 1)[-1] in ('linear_b_q', 'linear_b_v', 'linear_b'):
            mods.append(m)
    if mods:
        return mods, 'plain_lora', 'hook'
    for name, m in core.named_modules():
        if type(m).__name__ == 'MultiModalLoRAQKV' and hasattr(m, 'scale'):
            mods.append(m)
    if mods:
        return mods, 'mm_lora', 'scale'
    return mods, 'none', 'none'


class AdapterSwitch:
    """Context manager: disable every adapter — additive 모듈은 output hook `o*0`,
    scale-carrying wrapper(MultiModalLoRAQKV)는 scale=0."""
    def __init__(self, modules, mechanism='hook'):
        self.modules = modules
        self.mechanism = mechanism
        self.handles = []
        self.saved = []

    def __enter__(self):
        if self.mechanism == 'scale':
            self.saved = [m.scale for m in self.modules]
            for m in self.modules:
                m.scale = 0.0
        else:
            self.handles = [m.register_forward_hook(lambda mod, i, o: o * 0)
                            for m in self.modules]
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.mechanism == 'scale':
            for m, s in zip(self.modules, self.saved):
                m.scale = s
            self.saved = []


def flat_stats(f_on, f_off):
    """Relative shift + cosine between two feature tensors (any shape)."""
    a, b = f_on.flatten().float(), f_off.flatten().float()
    rel = (a - b).norm().item() / max(b.norm().item(), 1e-8)
    cos = F.cosine_similarity(a[None], b[None]).item()
    return rel, cos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--conditions', default='night,sun')
    ap.add_argument('--max-imgs', type=int, default=40)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg['MODEL']['RESUME_ENABLE'] = False
    ds_cfg = cfg['DATASET']
    if args.dataset_root:
        ds_cfg['ROOT'] = args.dataset_root
    if isinstance(ds_cfg.get('PHYSAUG'), dict):
        ds_cfg['PHYSAUG']['ENABLE'] = False
    device = torch.device(cfg['DEVICE'])
    V.setup_cudnn()
    isz = cfg.get('TEST', {}).get('IMAGE_SIZE', cfg['EVAL']['IMAGE_SIZE'])
    transform = V.get_val_augmentation(isz, dataset_cfg=ds_cfg)
    model = V.load_model(cfg, Path(args.model_path), device)
    model.eval()
    core = model.module if hasattr(model, 'module') else model

    adapters, kind, mechanism = find_adapter_modules(core)
    modals = ds_cfg.get('MODALS', [])
    report = {'model': Path(args.model_path).stem, 'adapter_kind': kind,
              'n_adapter_modules': len(adapters), 'modals': modals, 'conditions': {}}
    if not adapters:
        report['error'] = 'no additive adapter modules found — nothing to toggle'
        Path(args.out + '.json').write_text(json.dumps(report, indent=1))
        print('[modal_adaptation] SKIP: no adapters found'); return

    for cond in [c.strip() for c in args.conditions.split(',') if c.strip()]:
        ds_cfg['CASE'] = cond
        dataset, _ = V.create_dataset(ds_cfg, 'test', transform, 'test', macvi=False, eval_day=False)
        ign = getattr(dataset, 'ignore_label', 255)
        n = min(args.max_imgs, len(dataset))
        M = None
        acc = None  # per modality: [feat_rel, feat_cos, out_rel, out_cos, acc_on, acc_off, px]
        fused_rel_sum = 0.0
        for idx in range(n):
            images, label, _ = dataset[idx]
            imgs = [im.unsqueeze(0).to(device) for im in images]
            gt = rs_nn(torch.as_tensor(np.asarray(label)).to(device), WR).cpu().numpy()
            valid = gt != ign
            with torch.no_grad():
                model(imgs, multimask_output=True)
            f_on = [t.clone() for t in core._last_per_modal_feats]
            _pmo = getattr(core, '_last_per_modal_outputs', None)
            o_on = [t.clone() for t in _pmo] if _pmo is not None else None
            with AdapterSwitch(adapters, mechanism), torch.no_grad():
                model(imgs, multimask_output=True)
            f_off = [t.clone() for t in core._last_per_modal_feats]
            _pmo = getattr(core, '_last_per_modal_outputs', None)
            o_off = [t.clone() for t in _pmo] if _pmo is not None else None
            if M is None:
                M = len(f_on)
                acc = np.zeros((M, 7))
            for i in range(M):
                fr, fc = flat_stats(f_on[i], f_off[i])
                if o_on is not None and o_off is not None:
                    orr, oc = flat_stats(o_on[i], o_off[i])
                    p_on = rs_nn(o_on[i][0].argmax(0).to(device), WR).cpu().numpy()
                    p_off = rs_nn(o_off[i][0].argmax(0).to(device), WR).cpu().numpy()
                    acc[i] += [fr, fc, orr, oc,
                               float(((p_on == gt) & valid).sum()),
                               float(((p_off == gt) & valid).sum()),
                               float(valid.sum())]
                else:
                    # per-modal 출력이 없는 family(예: ReliaDINO eval) — feat 지표만
                    acc[i] += [fr, fc, 0.0, 0.0, 0.0, 0.0, float(valid.sum())]
        res = {}
        for i in range(M):
            name = modals[i] if i < len(modals) else f'mod{i}'
            res[name] = {
                'feat_shift_rel': round(acc[i, 0] / n, 4),
                'feat_cos': round(acc[i, 1] / n, 4),
                'out_shift_rel': round(acc[i, 2] / n, 4),
                'out_cos': round(acc[i, 3] / n, 4),
                'acc_adapter_on': round(acc[i, 4] / max(acc[i, 6], 1), 4),
                'acc_adapter_off': round(acc[i, 5] / max(acc[i, 6], 1), 4),
                'acc_delta': round((acc[i, 4] - acc[i, 5]) / max(acc[i, 6], 1), 4),
            }
        report['conditions'][cond] = {'n': n, 'per_modality': res}
        print(f"[modal_adaptation] {cond}: " + '  '.join(
            f"{k}: Δfeat={v['feat_shift_rel']:.3f} Δacc={v['acc_delta']:+.4f}"
            for k, v in res.items()), flush=True)

    Path(args.out + '.json').write_text(json.dumps(report, indent=1))
    # markdown table
    lines = [f"# Modal adaptation report — `{report['model']}`",
             f"- adapter: **{kind}** ×{len(adapters)} modules (additive, zeroed via forward hook)",
             "- 읽는 법: non-RGB의 `feat_shift_rel`(적응 크기)·`acc_delta`(적응 효과)가 RGB 대비"
             " 현저히 작으면 그 모달은 adapter가 적응 못한 것 (dead adaptation).",
             ""]
    for cond, cd in report['conditions'].items():
        lines += [f"## {cond} (n={cd['n']})",
                  "| modality | feat_shift_rel | feat_cos | out_shift_rel | acc(on) | acc(off) | Δacc |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in cd['per_modality'].items():
            lines.append(f"| {k} | {v['feat_shift_rel']} | {v['feat_cos']} | {v['out_shift_rel']} "
                         f"| {v['acc_adapter_on']} | {v['acc_adapter_off']} | {v['acc_delta']:+.4f} |")
        lines.append("")
    Path(args.out + '.md').write_text('\n'.join(lines))
    print(f"[modal_adaptation] wrote {args.out}.json / .md")


if __name__ == '__main__':
    main()
