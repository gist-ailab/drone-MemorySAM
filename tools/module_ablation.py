#!/usr/bin/env python3
"""
tools/module_ablation.py — [분석항목 3] fusion/제안 모듈의 전후(A/B) 수치·피쳐 비교
— model-agnostic (존재하는 토글만 적용, graceful skip), P31/32/33/34+ 재사용.

Same checkpoint, same images — flips ONE named runtime switch at a time and reports,
vs the untouched baseline:

  - per-class IoU Δ + mIoU Δ  (WR-grid; 헤드라인 IoU는 eval_per_domain으로)
  - fused-feature change      : cos(m_feat_base, m_feat_toggled), rel-shift
  - fused-output change       : pixel-level pred agreement rate

TOGGLES (attr가 없으면 자동 skip — 어느 P 버전이든 안전):
  rbma_off        core.lambda_bias ← 0          (RBMA memory-attn bias 제거, P27+)
  router_off      core.learned_router_enable ← False   (P30+ learned router → 기본 융합)
  ctd_off         core.class_token_decoder_enable ← False (P30 CTD 출력 대체 제거)
  sdc_off         core.sdc_enable ← False       (P29+ SDC 조건 게이트 제거)
  temp_off        core.rbma_log_temp ← 0        (P31 보정 temperature 제거)
  cons_off        core.lambda_cons ← 0          (P31 consistency 2차 bias 제거)
  amf_uniform     core.amf_mode ← 'uniform'     (출력 융합 uniform으로)
  p44_validity_off  fusion.p44_validity_renorm ← False  (P44-V1 presence 재정규화 제거)
                  ⚠️ P44의 나머지(B-1 gradient·B-2 손실·B-3 마스킹·P45)는 학습 전용이라
                     eval 토글이 없다 — 그 효과는 학습된 가중치에 이미 들어 있다.

Usage:
  python tools/module_ablation.py --cfg <model.yaml> --model_path <ckpt> \
    --dataset-root <DELIVER> --conditions night --max-imgs 40 --gpu 0 \
    --toggles rbma_off,router_off,sdc_off --out <prefix>

Output: <out>.json + <out>.md
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


# name -> (getter, setter) pairs on `core`; setter returns restore-fn or None if N/A
def make_toggles(core):
    T = {}

    def attr_toggle(name, attr, value, is_param=False):
        if not hasattr(core, attr) or getattr(core, attr) is None:
            return
        def apply():
            old = getattr(core, attr)
            if is_param:
                saved = old.detach().clone()
                with torch.no_grad():
                    old.fill_(value)
                def restore():
                    with torch.no_grad():
                        old.copy_(saved)
                return restore
            else:
                setattr(core, attr, value)
                return lambda: setattr(core, attr, old)
        T[name] = apply

    attr_toggle('rbma_off', 'lambda_bias', 0.0, is_param=True)
    attr_toggle('router_off', 'learned_router_enable', False)
    attr_toggle('ctd_off', 'class_token_decoder_enable', False)
    attr_toggle('sdc_off', 'sdc_enable', False)
    attr_toggle('temp_off', 'rbma_log_temp', 0.0, is_param=True)
    attr_toggle('cons_off', 'lambda_cons', 0.0, is_param=True)
    attr_toggle('amf_uniform', 'amf_mode', 'uniform')
    # [P34 ReliaDINO] fusion 서브모듈 위 토글 (없으면 자동 skip)
    fus = getattr(core, 'fusion', None)
    if fus is not None:
        def fus_toggle(name, attr, value, is_param=False):
            if not hasattr(fus, attr) or getattr(fus, attr) is None:
                return
            def apply():
                old = getattr(fus, attr)
                if is_param:
                    saved = old.detach().clone()
                    with torch.no_grad():
                        old.fill_(value)
                    def restore():
                        with torch.no_grad():
                            old.copy_(saved)
                    return restore
                setattr(fus, attr, value)
                return lambda: setattr(fus, attr, old)
            T[name] = apply
        fus_toggle('p34_bias_off', 'lambda1', 0.0, is_param=True)      # B_cal attn-bias
        fus_toggle('p34_cons_off', 'lambda2', 0.0, is_param=True)      # B_cons 2차 항
        fus_toggle('p34_gate_off', 'gate_enable', False)               # reliability gate
        fus_toggle('p34_veto_off', 'veto_floor', False)                # veto floor
        fus_toggle('p34_calib_off', 'calibrate', False)                # temperature 보정
        fus_toggle('p36_router_off', 'router_alpha', 0.0, is_param=True)  # [P36] router residual 제거
        if getattr(fus, 'p44_validity_renorm', False):
            # [P44-V1] presence 재정규화 = P44에서 **추론 경로에 걸리는 유일한**
            # 항목이라 A/B 토글이 의미를 갖는다. B-1(gradient 통합)·B-2(peer 증류
            # 손실)·B-3(입력 마스킹)·P45(style 일관성)는 전부 **학습 전용**이라
            # eval 토글이 정의상 Δ=0 → 등록하지 않는다(07-21 감사: 죽은 토글 행이
            # "기여 없음" 오판을 만든다).
            fus_toggle('p44_validity_off', 'p44_validity_renorm', False)
        cefr = getattr(fus, 'cefr', None)
        if cefr is not None and hasattr(cefr, 'a'):
            def _cefr_apply():
                old = cefr.a
                saved = old.detach().clone()
                with torch.no_grad():
                    old.fill_(-20.0)          # σ(a)→0 = CEFR blend 차단
                def restore():
                    with torch.no_grad():
                        old.copy_(saved)
                return restore
            T['p37_cefr_off'] = _cefr_apply

    # [P38] MaskQueryLite semantic 잔차 차단 (logits += beta·sem_q → beta=0)
    # 감사 2026-07-21: P39 arbiter 활성 모델에서는 beta 경로가 미사용이라
    # 이 토글이 항상 Δ=0 → "m2f 기여 없음"으로 오판됨. beta 경로가 실제
    # 결선일 때(arb_lambda 부재)만 등록.
    m2f = getattr(core, 'm2f', None)
    if getattr(core, 'arb_lambda', None) is not None:
        m2f_beta_live = None      # arbiter가 β를 대체 — p38_m2f_off 무의미
    else:
        m2f_beta_live = m2f
    if m2f_beta_live is not None and hasattr(m2f_beta_live, 'beta'):
        m2f = m2f_beta_live
        def _m2f_apply():
            old = m2f.beta
            saved = old.detach().clone()
            with torch.no_grad():
                old.fill_(0.0)
            def restore():
                with torch.no_grad():
                    old.copy_(saved)
                return None
            return restore
        T['p38_m2f_off'] = _m2f_apply

    # [P39] Dual-Path Compete (det_module_ablation.py와 동일 계약)
    # 감사 2026-07-21: 플래그 attr는 전 세대 모델에 존재하므로 attr 존재만으로
    # 등록하면 P38 이하 ckpt에서 Δ=0 no-op 행이 생겨 "기여 없음"으로 오판됨.
    # 해당 모듈이 실재할 때만 등록.
    if getattr(core, 'arb_lambda', None) is not None:
        attr_toggle('p39_query_off', 'p39_query_off', True)
        # ⚠️ 해석 주의: p39_dense_off는 다른 토글과 달리 "모듈을 끈 효과"가 아니라
        # **쿼리 단독 성능**을 읽는다(dense FPN 기여 제거 → logits = q_scaled).
        # 표의 '+면 기여' 문구를 그대로 적용하지 마라 — Δ가 크게 +여도 정상이며,
        # p39_query_off(Δ≈0)와 짝지어 읽어야 진단이 확정된다.
        attr_toggle('p39_dense_off', 'p39_dense_off', True)
    # trunk_xattn = [A/B trunk] FUSION.TRUNK: xattn 팔. 같은 플래그가 두 구현을
    # 모두 끄므로(_apply_trunk_exp) 여기서 함께 등록해야 xattn ckpt에서도 잰다.
    if (getattr(core, 'trunk_exp', None) is not None
            or getattr(core, 'trunk_xattn', None) is not None):
        attr_toggle('p39_trunkexp_off', 'p39_trunkexp_off', True)
    if m2f is not None:
        def m2f_toggle(name, attr, value):
            if not hasattr(m2f, attr) or getattr(m2f, attr) is None:
                return
            def apply():
                old = getattr(m2f, attr)
                setattr(m2f, attr, value)
                return lambda: setattr(m2f, attr, old)
            T[name] = apply
        m2f_toggle('p39_modalsrc_off', 'use_modal_src', False)  # V2 -> fused-only
        m2f_toggle('p39_anchored_off', 'anchored', False)       # V3 -> free queries only

    # [P43] PanopticDual. LATERAL feeds the shared trunk -> a real semantic Δ.
    # The mask-cls head, by design, does NOT touch the semantic logits (it is an
    # independent primary loss), so `p43_m2f_off` can only move mIoU when the
    # run is configured to decode semantics from the queries. Registering it
    # unconditionally would print a guaranteed Δ=0 row and be misread as "dead
    # module" — the exact false positive this tool warns about — so it is
    # registered only when the head is actually on the eval semantic path.
    if getattr(core, 'p43_lateral', None) is not None:
        attr_toggle('p43_lateral_off', 'p43_lateral_off', True)
    if (getattr(core, 'p43', None) is not None
            and str(getattr(core, 'p43_sem_source', 'pixel')) != 'pixel'):
        attr_toggle('p43_m2f_off', 'p43_m2f_off', True)
    return T


def miou_of(cf):
    C = cf.shape[0]
    return float(np.nanmean([cf[c, c] / max(1, cf[c, :].sum() + cf[:, c].sum() - cf[c, c])
                             for c in range(C)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--conditions', default='night')
    ap.add_argument('--split', default='test', help="MUSES는 GT 있는 'val' 사용")
    ap.add_argument('--max-imgs', type=int, default=40)
    ap.add_argument('--toggles', default='rbma_off,router_off,ctd_off,sdc_off,temp_off,cons_off,amf_uniform')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--viz-num', type=int, default=0,
                    help='condition당 N장: toggle 전후 비교 패널 저장 (RGB/GT/pred_base/pred_off/불일치맵)')
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

    toggles = make_toggles(core)
    req = [t.strip() for t in args.toggles.split(',') if t.strip()]
    avail = [t for t in req if t in toggles]
    skipped = [t for t in req if t not in toggles]
    report = {'model': Path(args.model_path).stem, 'toggles_available': avail,
              'toggles_skipped_no_attr': skipped, 'conditions': {}}
    print(f"[module_ablation] available={avail} skipped={skipped}")

    for cond in [c.strip() for c in args.conditions.split(',') if c.strip()]:
        ds_cfg['CASE'] = cond
        dataset, _ = V.create_dataset(ds_cfg, args.split, transform, args.split, macvi=False, eval_day=False)
        C = len(dataset.CLASSES); CLASSES = list(dataset.CLASSES)
        ign = getattr(dataset, 'ignore_label', 255)
        n = min(args.max_imgs, len(dataset))
        cf = {k: np.zeros((C, C), np.int64) for k in ['base'] + avail}
        feat_cos = {k: 0.0 for k in avail}
        feat_rel = {k: 0.0 for k in avail}
        agree = {k: 0.0 for k in avail}
        for idx in range(n):
            _s = dataset[idx]
            images, label = _s[0], _s[1]
            imgs = [im.unsqueeze(0).to(device) for im in images]
            gt = rs_nn(torch.as_tensor(np.asarray(label)).to(device), WR).cpu().numpy()
            valid = gt != ign
            with torch.no_grad():
                out = model(imgs, multimask_output=True)
            base_pred = rs_nn(out[0][0].argmax(0), WR).cpu().numpy()
            base_feat = out[1][0].detach().float()
            np.add.at(cf['base'], (gt[valid], base_pred[valid]), 1)
            viz_preds = {}
            for k in avail:
                restore = toggles[k]()
                try:
                    with torch.no_grad():
                        o2 = model(imgs, multimask_output=True)
                finally:
                    if restore:
                        restore()
                p2 = rs_nn(o2[0][0].argmax(0), WR).cpu().numpy()
                f2 = o2[1][0].detach().float()
                np.add.at(cf[k], (gt[valid], p2[valid]), 1)
                a, b = base_feat.flatten(), f2.flatten()
                feat_cos[k] += F.cosine_similarity(a[None], b[None]).item()
                feat_rel[k] += (a - b).norm().item() / max(b.norm().item(), 1e-8)
                agree[k] += float((p2[valid] == base_pred[valid]).mean())
                if idx < args.viz_num:
                    viz_preds[k] = p2
            # [항목③ 시각화] toggle 전후 비교 패널 (모듈이 예측을 '어디서' 바꾸는지)
            if idx < args.viz_num and viz_preds:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                K = len(viz_preds)
                fig, axes = plt.subplots(K, 5, figsize=(19, 3.6 * K), squeeze=False)
                rgb = images[0].permute(1, 2, 0).cpu().numpy()
                rgb = (rgb - rgb.min()) / max(rgb.max() - rgb.min(), 1e-6)
                gt_show = np.where(gt == ign, -1, gt)
                for r, (k, p2) in enumerate(viz_preds.items()):
                    dis = ((p2 != base_pred) & valid)
                    for c, (im, ttl, kw) in enumerate([
                            (rgb, 'RGB', {}),
                            (gt_show, 'GT', dict(cmap='tab20', vmin=-1, vmax=C - 1)),
                            (base_pred, 'pred (module ON)', dict(cmap='tab20', vmin=-1, vmax=C - 1)),
                            (p2, f'pred ({k})', dict(cmap='tab20', vmin=-1, vmax=C - 1)),
                            (dis, f'disagree {dis.mean()*100:.1f}%', dict(cmap='Reds'))]):
                        axes[r, c].imshow(im, **kw)
                        axes[r, c].set_title(ttl, fontsize=9)
                        axes[r, c].axis('off')
                fig.suptitle(f'{cond} #{idx} — module A/B', fontsize=11)
                fig.tight_layout()
                vdir = Path(args.out + '_viz'); vdir.mkdir(parents=True, exist_ok=True)
                fig.savefig(vdir / f'{cond}_{idx:03d}.png', dpi=110)
                plt.close(fig)
        base_iou = np.array([cf['base'][c, c] / max(1, cf['base'][c, :].sum()
                             + cf['base'][:, c].sum() - cf['base'][c, c]) for c in range(C)])
        res = {'n': n, 'base_miou': round(miou_of(cf['base']) * 100, 2), 'toggles': {}}
        for k in avail:
            iou_k = np.array([cf[k][c, c] / max(1, cf[k][c, :].sum()
                              + cf[k][:, c].sum() - cf[k][c, c]) for c in range(C)])
            d = (base_iou - iou_k) * 100      # +면 모듈이 기여(끄면 하락)
            top = np.argsort(-np.abs(d))[:6]
            res['toggles'][k] = {
                'miou_delta_when_off': round((miou_of(cf['base']) - miou_of(cf[k])) * 100, 2),
                'fused_feat_cos': round(feat_cos[k] / n, 4),
                'fused_feat_shift_rel': round(feat_rel[k] / n, 4),
                'pred_agreement': round(agree[k] / n, 4),
                'top_class_deltas': {CLASSES[c]: round(float(d[c]), 2) for c in top},
            }
        report['conditions'][cond] = res
        print(f"[module_ablation] {cond}: base={res['base_miou']} " + '  '.join(
            f"{k}:Δ={v['miou_delta_when_off']:+.2f}" for k, v in res['toggles'].items()), flush=True)

    Path(args.out + '.json').write_text(json.dumps(report, indent=1))
    lines = [f"# Module ablation (A/B) — `{report['model']}`",
             "- `miou_delta_when_off` **+면 모듈이 기여** (끄니 하락). `pred_agreement`≈1 &"
             " `feat_cos`≈1이면 모듈이 사실상 no-op (ISSUE-022류 죽은 모듈 감지).", ""]
    for cond, cd in report['conditions'].items():
        lines += [f"## {cond} (n={cd['n']}, base mIoU {cd['base_miou']})",
                  "| toggle | ΔmIoU(off) | feat cos | feat shift | pred agree | top class Δ |",
                  "|---|---|---|---|---|---|"]
        for k, v in cd['toggles'].items():
            tc = ', '.join(f"{c}:{x:+.1f}" for c, x in v['top_class_deltas'].items())
            lines.append(f"| {k} | {v['miou_delta_when_off']:+.2f} | {v['fused_feat_cos']} "
                         f"| {v['fused_feat_shift_rel']} | {v['pred_agreement']} | {tc} |")
        lines.append("")
    Path(args.out + '.md').write_text('\n'.join(lines))
    print(f"[module_ablation] wrote {args.out}.json / .md")


if __name__ == '__main__':
    main()
