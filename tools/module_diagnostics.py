#!/usr/bin/env python3
"""
tools/module_diagnostics.py — STATISTICAL module diagnostics over the DELIVER test set
(per-condition), reusable for P26/P27/P28/P29/P30...

Goes beyond per-image panels: aggregates, per condition, the signals our modules emit
(per-modal predictions, RBMA reliability=1-H, UAMM fusion weights, MoE gates) and ties them
to errors, answering "where & WHY our modules fail":

  A. per-class IoU + top confusion target            (where classes drop, what they become)
  B. per-modal competence  (per-class recall of each modality's standalone prediction)
                                                      ("which modality CAN do a class")
  B2. per-modal standalone mIoU (full confusion matrix per modality's own argmax vs GT,
      not just recall) + trivial majority-class-baseline mIoU, for the same condition
                                                      ("is this modality's own readout
                                                       above the information floor, or
                                                       is it near a do-nothing baseline")
  C. reliability informativeness: AUROC(reliability -> per-modal correct), per modality
                                                      (does RBMA's signal actually predict correctness?)
  D. UAMM fusion allocation: mean weight per modality (overall + per class) + MIS-ALLOCATION rate
       (GT-c pixels where the max-UAMM modality is wrong but another modality is right)
                                                      (does fusion weight the competent modality? Mode C)
  E. drop-modality dMIoU per modality (subset)        (per-modality importance = dead modalities)
  F. MoE gate usage (if captured)                     (expert collapse, Mode D)

All per-pixel stats are computed on a common WRxWR grid (default 256) so different-resolution
signals (output-res preds/reliability vs FPN-res UAMM) align. Headline IoU should still come from
tools/eval_per_domain.py (GT-resolution); the IoU here is for relative structure.

Usage:
  python tools/module_diagnostics.py --cfg <model.yaml> --model_path <ckpt> \
    --dataset-root <DELIVER> --conditions cloud,fog,night,rain,sun \
    --max-imgs 120 --ablate-n 20 --gpu 0 --out <prefix>
"""
import argparse, os, sys, math, json
# --gpu MUST be honored before torch initializes CUDA. torch is imported at module
# load (below), so the os.environ assignment inside main() is too late — it always
# ran on physical GPU0 regardless of --gpu. Scan argv here and set the mask first.
# (Same early-scan fix as tools/feature_stats.py.)
if '--gpu' in sys.argv:
    _gi = sys.argv.index('--gpu')
    if _gi + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_gi + 1]
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import val as V

WR = 256  # common working resolution for per-pixel module stats

def rs_nn(t, hw):   # nearest resize a (H,W) long/uint tensor -> (hw,hw)
    return F.interpolate(t[None, None].float(), size=(hw, hw), mode='nearest')[0, 0].long()

def rs_bl(t, hw):   # bilinear resize a (H,W) float tensor
    return F.interpolate(t[None, None].float(), size=(hw, hw), mode='bilinear', align_corners=False)[0, 0]

def auroc_from_hist(neg, pos):
    # neg[b], pos[b] = counts of incorrect/correct in reliability-bin b (bins ascending)
    N, P = neg.sum(), pos.sum()
    if N == 0 or P == 0: return float('nan')
    below = 0.0; auc = 0.0
    for b in range(len(neg)):
        auc += pos[b] * (below + 0.5 * neg[b])
        below += neg[b]
    return float(auc / (N * P))

def miou_of_conf(cf):
    # cf: (C,C) confusion matrix (rows=gt, cols=pred) -> mean per-class IoU (0..1)
    C = cf.shape[0]
    return float(np.nanmean([cf[c, c] / max(1, cf[c, :].sum() + cf[:, c].sum() - cf[c, c]) for c in range(C)]))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True); ap.add_argument('--model_path', required=True)
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--conditions', default='cloud,fog,night,rain,sun')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--max-imgs', type=int, default=120, help='cap images/condition for full stats')
    ap.add_argument('--ablate-n', type=int, default=20, help='images/condition for drop-modality')
    ap.add_argument('--gpu', default='0'); ap.add_argument('--out', required=True)
    args = ap.parse_args()
    # CUDA_VISIBLE_DEVICES was already set from the early argv scan (top of file),
    # before torch imported — this is kept only as a harmless idempotent restate.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    import yaml
    with open(args.cfg) as f: cfg = yaml.safe_load(f)
    cfg['MODEL']['RESUME_ENABLE'] = False
    ds_cfg = cfg['DATASET']
    if args.dataset_root: ds_cfg['ROOT'] = args.dataset_root
    if isinstance(ds_cfg.get('PHYSAUG'), dict): ds_cfg['PHYSAUG']['ENABLE'] = False
    device = torch.device(cfg['DEVICE']); V.setup_cudnn()
    isz = cfg.get('TEST', {}).get('IMAGE_SIZE', cfg['EVAL']['IMAGE_SIZE'])
    transform = V.get_val_augmentation(isz, dataset_cfg=ds_cfg)
    model = V.load_model(cfg, Path(args.model_path), device); model.eval()
    core = model.module if hasattr(model, 'module') else model
    NB = 20  # reliability bins

    report = {'model': Path(args.model_path).stem, 'conditions': {}}
    for cond in [c.strip() for c in args.conditions.split(',') if c.strip()]:
        ds_cfg['CASE'] = cond
        dataset, _ = V.create_dataset(ds_cfg, args.split, transform, args.split, macvi=False, eval_day=False)
        C = len(dataset.CLASSES); CLASSES = list(dataset.CLASSES); ign = getattr(dataset, 'ignore_label', 255)
        n = min(args.max_imgs, len(dataset))
        conf = np.zeros((C, C), np.int64)                    # final pred vs gt
        modal_tp = None; gt_cnt = np.zeros(C, np.int64)      # B: per-modal per-class recall
        modal_conf = None                                     # B2: per-modal standalone confusion (mIoU)
        rel_hist = None                                       # C: [M][NB][2]
        uamm_sum = None; uamm_px = 0                          # D: overall mean weight
        uamm_cls = None; cls_px = np.zeros(C, np.int64)       # D: per-class mean weight
        misalloc = np.zeros(C, np.int64); misden = np.zeros(C, np.int64)
        Mn = None
        for idx in range(n):
            images, label, _ = dataset[idx]
            imgs = [im.unsqueeze(0).to(device) for im in images]
            with torch.no_grad():
                m_out, _ = model(imgs, multimask_output=True)
            pmo = getattr(core, '_last_per_modal_outputs', None)
            usp = getattr(core, '_last_uamm_spatial', None)
            M = len(imgs)
            if Mn is None:
                Mn = M; modal_tp = np.zeros((M, C), np.int64)
                modal_conf = [np.zeros((C, C), np.int64) for _ in range(M)]
                rel_hist = np.zeros((M, NB, 2), np.int64); uamm_sum = np.zeros(M); uamm_cls = np.zeros((M, C))
            gt = rs_nn(torch.as_tensor(np.asarray(label)).to(device), WR).cpu().numpy()
            fp = rs_nn(m_out[0].argmax(0), WR).cpu().numpy()
            valid = gt != ign
            # A confusion
            gg, ff = gt[valid], fp[valid]
            np.add.at(conf, (gg, ff), 1)
            for c in range(C): gt_cnt[c] += int((gg == c).sum())
            # per-modal pred + reliability
            pm_pred = []; pm_rel = []
            for i in range(M):
                lo = pmo[i][0].to(device).float()             # (C,H,W)
                pm_pred.append(rs_nn(lo.argmax(0), WR).cpu().numpy())
                p = F.softmax(lo, 0); H = -(p * (p + 1e-8).log()).sum(0)
                pm_rel.append(rs_bl((1.0 - H / math.log(C)).detach(), WR).cpu().numpy())
            # B competence + C calibration
            for i in range(M):
                corr = (pm_pred[i] == gt) & valid
                for c in range(C): modal_tp[i, c] += int((corr & (gt == c)).sum())
                np.add.at(modal_conf[i], (gg, pm_pred[i][valid]), 1)   # B2: standalone confusion
                rv = np.clip((pm_rel[i][valid] * NB).astype(int), 0, NB - 1)
                cc = (pm_pred[i][valid] == gg).astype(int)
                np.add.at(rel_hist[i], (rv, cc), 1)
            # D fusion allocation
            if usp is not None:
                w = np.stack([rs_bl(torch.as_tensor(usp[i][0, 0]).to(device), WR).cpu().numpy() for i in range(M)], 0)  # (M,WR,WR)
                wv = w.reshape(M, -1)[:, valid.reshape(-1)]
                for i in range(M): uamm_sum[i] += wv[i].sum()
                uamm_px += int(valid.sum())
                amax = w.argmax(0)                              # (WR,WR) winning modality
                for c in range(C):
                    mask = (gt == c) & valid
                    cls_px[c] += int(mask.sum())
                    for i in range(M): uamm_cls[i, c] += float(w[i][mask].sum())
                    # misallocation: winner wrong but some modality right
                    if mask.sum() == 0: continue
                    win_pred = np.choose(np.clip(amax, 0, M - 1), pm_pred)  # winner's pred per pixel
                    win_wrong = (win_pred != gt) & mask
                    any_right = np.zeros_like(mask)
                    for i in range(M): any_right |= (pm_pred[i] == gt)
                    misalloc[c] += int((win_wrong & any_right).sum()); misden[c] += int(mask.sum())
        # E: drop-modality dMIoU (subset) — per-modality importance
        na = min(args.ablate_n, n)
        miou_of = miou_of_conf
        cf_full = np.zeros((C, C), np.int64); cf_drop = [np.zeros((C, C), np.int64) for _ in range(Mn)]
        for idx in range(na):
            images, label, _ = dataset[idx]
            base = [im.unsqueeze(0).to(device) for im in images]
            gt = rs_nn(torch.as_tensor(np.asarray(label)).to(device), WR).cpu().numpy(); valid = gt != ign
            with torch.no_grad(): mo, _ = model(base, multimask_output=True)
            fp = rs_nn(mo[0].argmax(0), WR).cpu().numpy(); np.add.at(cf_full, (gt[valid], fp[valid]), 1)
            for i in range(Mn):
                dz = [(torch.zeros_like(base[j]) if j == i else base[j]) for j in range(Mn)]
                with torch.no_grad(): mo, _ = model(dz, multimask_output=True)
                fpi = rs_nn(mo[0].argmax(0), WR).cpu().numpy(); np.add.at(cf_drop[i], (gt[valid], fpi[valid]), 1)
        miou_full = miou_of(cf_full); dmiou = [round((miou_full - miou_of(cf_drop[i])) * 100, 2) for i in range(Mn)]

        # finalize condition
        iou = np.array([conf[c, c] / max(1, conf[c, :].sum() + conf[:, c].sum() - conf[c, c]) for c in range(C)])
        conf_off = conf.copy(); np.fill_diagonal(conf_off, 0)
        top_conf = {CLASSES[c]: (CLASSES[int(conf_off[c].argmax())] if conf_off[c].sum() else '-') for c in range(C)}
        comp = {CLASSES[c]: [round(float(modal_tp[i, c] / max(1, gt_cnt[c])), 3) for i in range(Mn)] for c in range(C)}
        # B2: per-modal standalone mIoU (full confusion, not just recall) + trivial majority-class baseline
        modal_miou = [round(miou_of_conf(modal_conf[i]) * 100, 2) for i in range(Mn)]
        modal_iou_per_class = {CLASSES[c]: [round(float(
            modal_conf[i][c, c] / max(1, modal_conf[i][c, :].sum() + modal_conf[i][:, c].sum() - modal_conf[i][c, c]) * 100), 2)
            for i in range(Mn)] for c in range(C)}
        majority_c = int(np.argmax(gt_cnt)); valid_total = int(gt_cnt.sum())
        trivial_miou = round(float(gt_cnt[majority_c]) / max(valid_total, 1) / C * 100, 2)
        auroc = [round(auroc_from_hist(rel_hist[i, :, 0], rel_hist[i, :, 1]), 3) for i in range(Mn)]
        umean = [round(float(uamm_sum[i] / max(1, uamm_px)), 3) for i in range(Mn)] if uamm_px else None
        ucls = {CLASSES[c]: [round(float(uamm_cls[i, c] / max(1, cls_px[c])), 3) for i in range(Mn)] for c in range(C)} if uamm_px else None
        mis = {CLASSES[c]: round(float(misalloc[c] / max(1, misden[c])), 3) for c in range(C)}
        moe = getattr(core, '_last_moe_gates', None)
        report['conditions'][cond] = {
            'n': n, 'M': Mn,
            'iou': {CLASSES[c]: round(float(iou[c] * 100), 2) for c in range(C)},
            'top_confusion': top_conf, 'modal_competence': comp,
            'modal_standalone_miou': modal_miou, 'modal_standalone_iou_per_class': modal_iou_per_class,
            'trivial_majority_class': CLASSES[majority_c], 'trivial_miou': trivial_miou,
            'reliability_auroc': auroc, 'uamm_mean': umean, 'uamm_per_class': ucls,
            'misallocation_rate': mis,
            'drop_modality_dmiou': dmiou, 'ablate_miou_full': round(miou_full * 100, 2),
            'moe_gates_shape': (list(np.shape(moe)) if moe is not None else None),
        }
        print(f"[diag] {cond}: mIoU~{np.nanmean(iou)*100:.1f}  modalMIoU(standalone)={modal_miou} "
              f"trivial={trivial_miou}({CLASSES[majority_c]})  relAUROC={auroc}  uammMean={umean} dropMIoU={dmiou}", flush=True)

    Path(args.out + '.json').write_text(json.dumps(report, indent=1))
    print(f"[diag] wrote {args.out}.json")

if __name__ == '__main__':
    main()
