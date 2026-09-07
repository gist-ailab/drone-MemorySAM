#!/usr/bin/env python3
"""tools/probe_feature_info.py — [E0] 특징 정보 프로브 (구조 사다리 S0).

정본 설계: .claude_logs/decisions/2026-09-07-daily-cycle-experiment-cards.md §1 "E0".

질문: "클래스·센서 정보가 DINOv3 **원 특징**(frozen, 어댑터 이전)에는 있는데
어댑터·헤드를 거치며 사라지는가?" 를 **학습 없이**(선형 헤드만) 잰다.

방법: 여러 특징 세트마다 토큰 단위 (N, D) 행렬을 뽑아, 다항 로지스틱 회귀(1층
선형, 표준화 후 AdamW 몇십 epoch, 클래스 균형)를 fit-split 토큰으로 학습하고
eval-split 에서 클래스별 recall·IoU 와 mIoU 를 잰다. 붕괴 클래스(DELIVER:
RailTrack·Wall·Water·Bridge·TrafficLight)를 표 상단에 강조한다.

특징 세트:
  1. raw_taps      : LoRA 를 **끈** frozen DINOv3 에서 --taps 블록 출력을 채널
                     concat. 센서별(raw_taps/img …) + 전 센서 concat(raw_taps/all).
  2. adapted_last  : 현행 모델의 어댑터(LoRA) 적용 마지막 블록 특징
                     (`_last_per_modal_feats`). 센서별 + 전 센서 concat.
  3. fused_prehead : 디코드 직전 fused 특징(`_last_fused_prehead`).
  4. (선택) adapted_taps : LoRA 켠 상태의 탭 concat(--include-adapted-taps).

오라클 센서 선택 프로브(--oracle-dir 주어질 때만): H16 오라클
(tools/oracle_spatial_modality.py)의 캐시에서 토큰별 "정답 센서 부분집합" 라벨을
재구성해, 같은 특징 세트 위에서 부분집합 분류 정확도와 우연(다수 클래스) 정확도를
잰다.

⚠️ 순수 추론 — 백본은 frozen, 프로브 선형층만 학습한다.

실행 예:
  python tools/probe_feature_info.py \
    --cfg configs/eval/yeon-deliver_rgbdel_P46_c3only_lam01_base_eval768.yaml \
    --model_path outputs/.../epochXX_..._checkpoint.pth \
    --taps 6,12,18,24 --fit-split train --fit-images 400 \
    --eval-splits val,test --eval-images 300 --tokens-per-class 4000 \
    --out /drone_nas/.../analysis_logs/probe_feature_info_deliver \
    [--oracle-dir /drone_nas/.../oracle_spatial_modality_deliver] [--gpu 0]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# CUDA_VISIBLE_DEVICES 는 torch import 전에 argv 에서 선반영(feature_stats 관례).
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
    elif _a.startswith('--gpu='):
        os.environ['CUDA_VISIBLE_DEVICES'] = _a.split('=', 1)[1]
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

DELIVER_COLLAPSE = ['RailTrack', 'Wall', 'Water', 'Bridge', 'TrafficLight']


# ───────────────────── 순수 로직 (스모크에서 torch만으로 재사용) ─────────────────────

def majority_downsample(gt, out_h, out_w, num_classes, ignore=255):
    """GT (H,W) long 텐서 → 토큰 격자 (out_h*out_w,) 다수결 라벨.

    각 GT 픽셀을 출력 셀 (y*out_w+x) 로 매핑하고, 셀마다 클래스 히스토그램의
    argmax 를 취한다. ignore(또는 num_classes 이상)는 집계에서 제외하고, 유효
    픽셀이 없는 셀은 ignore 로 둔다. 반환: (out_h*out_w,) long.
    """
    import torch
    H, W = gt.shape
    dev = gt.device
    ys = (torch.arange(H, device=dev) * out_h // H).clamp_(0, out_h - 1)
    xs = (torch.arange(W, device=dev) * out_w // W).clamp_(0, out_w - 1)
    cell = (ys[:, None] * out_w + xs[None, :]).reshape(-1)          # (H*W,)
    g = gt.reshape(-1).long()
    valid = (g != ignore) & (g >= 0) & (g < num_classes)
    out = torch.full((out_h * out_w,), ignore, dtype=torch.long, device=dev)
    if valid.any():
        idx = cell[valid] * num_classes + g[valid]
        counts = torch.bincount(idx, minlength=out_h * out_w * num_classes)
        counts = counts.view(out_h * out_w, num_classes)
        maj = counts.argmax(dim=1)
        has = counts.sum(dim=1) > 0
        out[has] = maj[has]
    return out


def confusion_matrix(pred, target, num_classes, ignore=255):
    """pred/target: (N,) long. 반환: (C,C) float64 numpy, hist[gt, pred]."""
    import torch
    keep = (target != ignore) & (target >= 0) & (target < num_classes) \
        & (pred >= 0) & (pred < num_classes)
    if not keep.any():
        return np.zeros((num_classes, num_classes), dtype=np.float64)
    g = target[keep].long()
    p = pred[keep].long()
    flat = g * num_classes + p
    bc = torch.bincount(flat, minlength=num_classes ** 2)[: num_classes ** 2]
    return bc.reshape(num_classes, num_classes).cpu().numpy().astype(np.float64)


def iou_recall_from_confusion(hist):
    """confusion (C,C) → (per_class_iou %, per_class_recall %, mIoU %).

    recall_c = TP_c / (Σ_p hist[c,p]) = 대각 / 행합(그 클래스 GT 픽셀 중 맞힌 비율).
    IoU_c   = TP_c / (행합 + 열합 − 대각).
    """
    hist = np.asarray(hist, dtype=np.float64)
    diag = np.diag(hist)
    row = hist.sum(1)   # GT 별 총 픽셀
    col = hist.sum(0)   # 예측 별 총 픽셀
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where((row + col - diag) > 0, diag / (row + col - diag), 0.0)
        rec = np.where(row > 0, diag / row, 0.0)
    iou = np.nan_to_num(iou, nan=0.0)
    rec = np.nan_to_num(rec, nan=0.0)
    # mIoU 는 GT 에 존재하는 클래스(row>0)에 대해서만 평균 — 데이터에 없는 클래스가
    # 0 으로 평균을 끌어내리지 않게(진단 지표의 표준 관례).
    present = row > 0
    miou = float(iou[present].mean()) if present.any() else 0.0
    return (iou * 100).tolist(), (rec * 100).tolist(), round(miou * 100, 4)


class LinearProbe:
    """표준화 + 1층 선형 다항 로지스틱 회귀. 백본과 무관한 순수 프로브."""

    def __init__(self, in_dim, num_classes, device):
        import torch
        self.torch = torch
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.device = device
        self.linear = torch.nn.Linear(in_dim, num_classes).to(device)
        self.mean = torch.zeros(in_dim, device=device)
        self.std = torch.ones(in_dim, device=device)

    def set_standardizer(self, X):
        self.mean = X.mean(0)
        self.std = X.std(0).clamp_min(1e-6)

    def _norm(self, X):
        return (X - self.mean) / self.std

    def fit(self, X, y, epochs=30, lr=1e-2, batch=8192, weight_decay=1e-4,
            class_balanced=True, log_prefix=''):
        """X: (N,D) float32, y: (N,) long — 둘 다 이미 device 위. 클래스 균형은
        CE class weight(역빈도)로 준다(bank 가 이미 대체로 균형이라 보정용)."""
        torch = self.torch
        self.set_standardizer(X)
        Xn = self._norm(X)
        opt = torch.optim.AdamW(self.linear.parameters(), lr=lr, weight_decay=weight_decay)
        cls_w = None
        if class_balanced:
            cnt = torch.bincount(y, minlength=self.num_classes).float()
            w = torch.where(cnt > 0, 1.0 / cnt, torch.zeros_like(cnt))
            cls_w = (w / w[w > 0].mean()).to(self.device)   # 평균 1 로 정규화
        lossf = torch.nn.CrossEntropyLoss(weight=cls_w)
        N = Xn.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(N, device=self.device)
            tot = 0.0
            for s in range(0, N, batch):
                bi = perm[s:s + batch]
                opt.zero_grad()
                loss = lossf(self.linear(Xn[bi]), y[bi])
                loss.backward()
                opt.step()
                tot += float(loss) * bi.numel()
            if log_prefix and (ep == 0 or ep == epochs - 1):
                print(f"    {log_prefix} epoch {ep} loss={tot / N:.4f}", flush=True)

    @property
    def torch_no_grad(self):
        return self.torch.no_grad

    def predict(self, X):
        """X: (N,D) float32 device → (N,) long 예측(표준화 내장)."""
        with self.torch.no_grad():
            return self.linear(self._norm(X)).argmax(dim=1)


# ───────────────────── balanced 토큰 뱅크 (fit 전용) ─────────────────────

class BalancedBank:
    """클래스별 최대 cap 토큰을 CPU float16 으로 모으는 균형 뱅크."""

    def __init__(self, num_classes, cap, ignore=255):
        self.num_classes = num_classes
        self.cap = cap
        self.ignore = ignore
        self.buf = {c: [] for c in range(num_classes)}
        self.count = {c: 0 for c in range(num_classes)}

    def full(self):
        return all(self.count[c] >= self.cap for c in range(self.num_classes))

    def add(self, feats, labels):
        """feats: (N,D) torch(any device), labels: (N,) torch long."""
        import torch
        for c in range(self.num_classes):
            if self.count[c] >= self.cap:
                continue
            m = labels == c
            n = int(m.sum())
            if n == 0:
                continue
            take = min(n, self.cap - self.count[c])
            fc = feats[m]
            if take < n:
                idx = torch.randperm(n, device=fc.device)[:take]
                fc = fc[idx]
            self.buf[c].append(fc.detach().to('cpu', torch.float16))
            self.count[c] += take

    def build(self, device):
        """→ (X float32 device, y long device). 빈 클래스는 건너뛴다."""
        import torch
        xs, ys = [], []
        for c in range(self.num_classes):
            if not self.buf[c]:
                continue
            xc = torch.cat(self.buf[c], dim=0)
            xs.append(xc)
            ys.append(torch.full((xc.shape[0],), c, dtype=torch.long))
        if not xs:
            return None, None
        X = torch.cat(xs, dim=0).to(device, torch.float32)
        y = torch.cat(ys, dim=0).to(device)
        return X, y


# ───────────────────── 특징 추출 (모델 forward) ─────────────────────

def resolve_taps(taps, n_blocks):
    """--taps 를 0-indexed 블록 인덱스로. 1-indexed 블록/층 번호로 해석하고
    [1, n_blocks] 로 clamp 한 뒤 -1 한다(기본 6,12,18,24 · 24블록 → 5,11,17,23)."""
    return sorted({min(max(int(t), 1), n_blocks) - 1 for t in taps})


class TapCapture:
    """지정 블록 출력을 hook 으로 잡아 (B,C,h,w) 맵으로 변환."""

    def __init__(self, encoder, block_indices):
        self.encoder = encoder
        self.block_indices = block_indices
        self.buf = {}
        self.handles = []

    def __enter__(self):
        for slot, li in enumerate(self.block_indices):
            def mk(s):
                def hook(_m, _i, out):
                    self.buf[s] = out[0] if isinstance(out, tuple) else out
                return hook
            self.handles.append(
                self.encoder.backbone.blocks[li].register_forward_hook(mk(slot)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def maps(self, h, w):
        return [self.encoder._to_map(self.buf[s], h, w)
                for s in range(len(self.block_indices))]


def extract_feature_sets(model, core, imgs, tap_indices, modal_names,
                         include_adapted_taps, device):
    """이미지 1장(각 텐서 (1,3,H,W) device) → {세트명: (C,h,w) float 텐서} dict.

    per-modal 세트는 raw_taps(LoRA off)·adapted_last(full forward 버퍼)에서,
    fused_prehead 는 full forward 버퍼에서, adapted_taps(선택)는 LoRA on 재-encode
    에서 얻는다. 반환 맵은 모두 CPU float32(뱅크 add 시 다시 옮긴다).
    """
    import torch
    M = len(modal_names)
    enc = core.encoder
    sets = {}

    # [OOM fix 2026-09-07] 추출 전체를 no_grad 로 감싼다 — RAW/adapted 인코딩이 그래프를
    # 들고 있어 이미지마다 누적돼 24GB 에서 OOM(bengio 실측). 프로브는 학습이 아니다.
    with torch.no_grad(), TapCapture(enc, tap_indices) as tc:
        # 1) RAW — LoRA off, per-modal encode. try/finally 로 반드시 원복.
        enc.set_lora_enabled(False)
        try:
            raw_taps = []   # per sensor: list over taps of (C,h,w)
            for i in range(M):
                tc.buf.clear()
                last = enc(imgs[i], i)                       # (1,C,h,w)
                h, w = last.shape[-2:]
                raw_taps.append([t[0].detach().float().cpu() for t in tc.maps(h, w)])
        finally:
            enc.set_lora_enabled(True)

        # 4) ADAPTED taps (선택) — LoRA on per-modal encode.
        adapted_taps = None
        if include_adapted_taps:
            adapted_taps = []
            for i in range(M):
                tc.buf.clear()
                last = enc(imgs[i], i)
                h, w = last.shape[-2:]
                adapted_taps.append([t[0].detach().float().cpu() for t in tc.maps(h, w)])

    # 2)/3) full forward (LoRA on) → per-modal feats + fused_prehead 버퍼.
    with torch.no_grad():
        model(imgs, multimask_output=True)
    per_modal = core._last_per_modal_feats           # list of (1,C,h,w)
    fused_prehead = getattr(core, '_last_fused_prehead', None)

    # raw_taps: 센서별 tap-concat + all-concat
    for i, nm in enumerate(modal_names):
        sets[f'raw_taps/{nm}'] = torch.cat(raw_taps[i], dim=0)   # (ntaps*C, h, w)
    sets['raw_taps/all'] = torch.cat([sets[f'raw_taps/{nm}'] for nm in modal_names], dim=0)

    # adapted_last: 센서별 + all-concat
    if per_modal is not None:
        for i, nm in enumerate(modal_names):
            sets[f'adapted_last/{nm}'] = per_modal[i][0].float().cpu()
        sets['adapted_last/all'] = torch.cat(
            [sets[f'adapted_last/{nm}'] for nm in modal_names], dim=0)

    if fused_prehead is not None:
        sets['fused_prehead'] = fused_prehead[0].float().cpu()

    if adapted_taps is not None:
        for i, nm in enumerate(modal_names):
            sets[f'adapted_taps/{nm}'] = torch.cat(adapted_taps[i], dim=0)
        sets['adapted_taps/all'] = torch.cat(
            [sets[f'adapted_taps/{nm}'] for nm in modal_names], dim=0)

    return sets


def tokens_and_labels(feat_map, gt, num_classes, ignore, label_cache):
    """(C,h,w) 특징맵 → (h*w, C) 토큰 + (h*w,) 다수결 라벨. 라벨은 (h,w)별 캐시."""
    C, h, w = feat_map.shape
    key = (h, w)
    if key not in label_cache:
        # 특징맵은 CPU 이므로 라벨도 CPU 로 맞춘다(뱅크 add·인덱싱 디바이스 정합).
        label_cache[key] = majority_downsample(gt, h, w, num_classes, ignore).cpu()
    toks = feat_map.reshape(C, h * w).transpose(0, 1).contiguous()   # (h*w, C)
    return toks, label_cache[key]


# ───────────────────── 데이터셋 이터레이터 ─────────────────────

def iter_split(cfg, split, max_images, device):
    """(imgs[list of (1,3,H,W)], gt (H,W) long, meta) 를 순차로 내놓는다.

    GT 는 native(meta['orig_label']) 우선, 없으면 변환 label. imgs 는 배치 1.
    """
    import torch
    import val as V
    ds_cfg, eval_cfg = cfg['DATASET'], cfg['EVAL']
    transform = V.get_val_augmentation(eval_cfg['IMAGE_SIZE'], dataset_cfg=ds_cfg)
    dataset, _ = V.create_dataset(ds_cfg, split, transform, split, macvi=False, eval_day=False)
    n = len(dataset) if max_images < 0 else min(max_images, len(dataset))
    for idx in range(n):
        images, label, meta = dataset[idx]
        imgs = [im.unsqueeze(0).to(device) for im in images]
        gt = meta.get('orig_label')
        if gt is None:
            gt = label
        gt = torch.as_tensor(np.asarray(gt)).to(device).long()
        yield imgs, gt, meta


# ───────────────────── 오라클 센서 선택 (선택) ─────────────────────

def load_oracle_star(oracle_dir, split, num_classes, ignore):
    """H16 오라클 캐시(pred_{split}_mask*.npy + gt_{split}.npy)에서 이미지별
    per-pixel star(채택 부분집합 인덱스, 정답 없으면 -1)를 재구성한다.

    ⚠️ 미확인 가정: 오라클 캐시의 이미지 순서가 이 도구의 데이터셋 순서와 같아야
    토큰이 정렬된다(둘 다 split 전수를 셔플 없이 순회하면 성립). 캐시가 없으면
    None 을 돌린다. 반환: (stars: list[np.ndarray (H,W)], legend: dict).
    """
    from tools.oracle_spatial_modality import (
        enumerate_subsets, subset_bitmask, oracle_synthesize)
    cache = Path(oracle_dir) / 'cache'
    gt_path = cache / f'gt_{split}.npy'
    if not gt_path.exists():
        return None, None
    legend_path = Path(oracle_dir) / 'subset_legend.json'
    legend = json.loads(legend_path.read_text()) if legend_path.exists() else {}
    gt_all = np.load(gt_path, mmap_mode='r')
    N = gt_all.shape[0]
    # 부분집합 개수·순서는 legend 의 마스크 수로 추정(모달 수 = log2 상한 아님) —
    # 대신 존재하는 pred_ 파일들로 마스크를 모은다.
    masks = sorted(int(p.stem.split('mask')[1]) for p in cache.glob(f'pred_{split}_mask*.npy'))
    if not masks:
        return None, legend
    pred_maps = [np.load(cache / f'pred_{split}_mask{bm}.npy', mmap_mode='r') for bm in masks]
    full_index = len(masks) - 1   # enumerate_subsets 규약상 full 이 마지막(비트마스크 최대)
    stars = []
    for i in range(N):
        gt = np.asarray(gt_all[i]).astype(np.int64)
        preds = np.stack([np.asarray(pm[i]).astype(np.int64) for pm in pred_maps], axis=0)
        _, star = oracle_synthesize(preds, gt, full_index)
        stars.append(star.astype(np.int64))
    return stars, {'masks': masks, 'legend': legend}


# ───────────────────── 프로브 실행 ─────────────────────

def run_class_probe(cfg, model, core, tap_indices, modal_names, num_classes,
                    class_names, ignore, args, device):
    """클래스 선형 프로브: fit-split 으로 학습, eval-splits 에서 IoU/recall."""
    import torch
    # 1) fit — balanced bank 축적
    print(f"[probe] fit split={args.fit_split} images≤{args.fit_images} "
          f"tokens/class={args.tokens_per_class}")
    banks = {}          # set 이름 → BalancedBank
    for imgs, gt, _meta in iter_split(cfg, args.fit_split, args.fit_images, device):
        sets = extract_feature_sets(model, core, imgs, tap_indices, modal_names,
                                    args.include_adapted_taps, device)
        label_cache = {}
        for name, fmap in sets.items():
            toks, labs = tokens_and_labels(fmap, gt, num_classes, ignore, label_cache)
            if name not in banks:
                banks[name] = BalancedBank(num_classes, args.tokens_per_class, ignore)
            banks[name].add(toks, labs)
        if all(b.full() for b in banks.values()):
            print("[probe] 모든 세트 뱅크 포화 — fit 축적 조기 종료")
            break

    # 2) 세트별 프로브 학습
    probes = {}
    for name, bank in banks.items():
        X, y = bank.build(device)
        if X is None:
            print(f"[probe] {name}: 뱅크 비어 있음 — 건너뜀")
            continue
        pr = LinearProbe(X.shape[1], num_classes, device)
        pr.fit(X, y, epochs=args.epochs, lr=args.lr, log_prefix=f'fit[{name}]')
        probes[name] = pr
        del X, y
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 3) eval-splits: confusion 누적(토큰 저장 없이 전수 IoU/recall)
    results = {}   # split → {set → {'miou','per_class'}}
    for split in args.eval_splits:
        print(f"[probe] eval split={split} images≤{args.eval_images}")
        hist = {name: np.zeros((num_classes, num_classes), np.float64) for name in probes}
        for imgs, gt, _meta in iter_split(cfg, split, args.eval_images, device):
            sets = extract_feature_sets(model, core, imgs, tap_indices, modal_names,
                                        args.include_adapted_taps, device)
            label_cache = {}
            for name, pr in probes.items():
                if name not in sets:
                    continue
                toks, labs = tokens_and_labels(sets[name], gt, num_classes, ignore, label_cache)
                pred = pr.predict(toks.to(device, torch.float32))
                hist[name] += confusion_matrix(pred, labs.to(device), num_classes, ignore)
        split_res = {}
        for name, h in hist.items():
            iou, rec, miou = iou_recall_from_confusion(h)
            split_res[name] = {'miou': miou, 'iou': iou, 'recall': rec}
        results[split] = split_res
    return results


def run_oracle_probe(cfg, model, core, tap_indices, modal_names, num_classes,
                     ignore, args, device):
    """오라클 센서 선택 프로브(선택). fit = 캐시 있는 첫 eval-split, eval = 나머지.

    라벨 = per-token 다수결 star(부분집합 인덱스). star<0(정답 부분집합 없음)은 제외.
    반환: {'fit_split','eval': {split → {set → {'acc','chance'}}}, 'note'}.
    """
    import torch
    # 캐시 있는 split 목록
    star_by_split = {}
    for split in args.eval_splits:
        stars, meta = load_oracle_star(args.oracle_dir, split, num_classes, ignore)
        if stars is not None:
            star_by_split[split] = stars
    if not star_by_split:
        return {'note': 'oracle 캐시(pred_*/gt_*.npy) 없음 — 오라클 프로브 생략'}
    fit_split = args.oracle_fit_split if args.oracle_fit_split in star_by_split \
        else next(iter(star_by_split))
    n_subsets = int(max(int(s.max()) for s in star_by_split[fit_split]) + 1)
    n_subsets = max(n_subsets, 2)

    def star_label(gt, stars_img, oh, ow):
        """star 맵(H,W)을 토큰 격자로 다수결 다운샘플. star<0 은 별도 클래스로
        두지 않고 ignore 처리한다(majority_downsample 재사용 위해 +1 시프트)."""
        st = torch.as_tensor(stars_img).to(device).long()
        shifted = st + 1                                    # -1→0(오답=클래스 0)
        maj = majority_downsample(shifted, oh, ow, n_subsets + 1, ignore)
        empty = maj == ignore                               # 유효 픽셀 없는 셀
        maj = maj - 1                                       # 되돌림; 0→-1(오답)
        maj[(maj < 0) | empty] = ignore                    # 오답·빈 셀 모두 ignore
        return maj.cpu()                                    # 특징맵(CPU)과 디바이스 정합

    # fit: balanced bank(부분집합 라벨 기준)
    banks = {}
    for k, (imgs, gt, _m) in enumerate(iter_split(cfg, fit_split, args.eval_images, device)):
        if k >= len(star_by_split[fit_split]):
            break
        sets = extract_feature_sets(model, core, imgs, tap_indices, modal_names,
                                    args.include_adapted_taps, device)
        label_cache = {}
        for name, fmap in sets.items():
            C, h, w = fmap.shape
            labs = star_label(gt, star_by_split[fit_split][k], h, w)
            toks = fmap.reshape(C, h * w).transpose(0, 1).contiguous()
            if name not in banks:
                banks[name] = BalancedBank(n_subsets, args.tokens_per_class, ignore)
            banks[name].add(toks, labs)

    probes = {}
    for name, bank in banks.items():
        X, y = bank.build(device)
        if X is None:
            continue
        pr = LinearProbe(X.shape[1], n_subsets, device)
        pr.fit(X, y, epochs=args.epochs, lr=args.lr, log_prefix=f'oracle-fit[{name}]')
        probes[name] = pr

    out = {'fit_split': fit_split, 'n_subsets': n_subsets, 'eval': {},
           'note': 'star=오라클 채택 부분집합; 미확인 가정=캐시/데이터셋 이미지 순서 정합'}
    for split, stars in star_by_split.items():
        correct = {name: 0 for name in probes}
        total = {name: 0 for name in probes}
        chance_hist = np.zeros(n_subsets, np.float64)
        for k, (imgs, gt, _m) in enumerate(iter_split(cfg, split, args.eval_images, device)):
            if k >= len(stars):
                break
            sets = extract_feature_sets(model, core, imgs, tap_indices, modal_names,
                                        args.include_adapted_taps, device)
            for name, pr in probes.items():
                if name not in sets:
                    continue
                C, h, w = sets[name].shape
                labs = star_label(gt, stars[k], h, w)
                keep = labs != ignore
                if not bool(keep.any()):
                    continue
                toks = sets[name].reshape(C, h * w).transpose(0, 1)[keep].to(device, torch.float32)
                pred = pr.predict(toks)
                lk = labs[keep].to(device)
                correct[name] += int((pred == lk).sum())
                total[name] += int(lk.numel())
                if name == next(iter(probes)):
                    chance_hist += torch.bincount(lk, minlength=n_subsets).cpu().numpy()
        chance = float(chance_hist.max() / chance_hist.sum()) if chance_hist.sum() else 0.0
        out['eval'][split] = {
            name: {'acc': round(correct[name] / total[name], 4) if total[name] else None,
                   'chance': round(chance, 4)}
            for name in probes}
    return out


# ───────────────────── 리포트 ─────────────────────

def write_reports(results, oracle_res, class_names, collapse_idx, out_dir, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {'meta': meta, 'class_probe': results, 'oracle_probe': oracle_res}
    (out_dir / 'probe_results.json').write_text(json.dumps(payload, indent=2))

    collapse_names = [class_names[i] for i in collapse_idx]
    lines = ['# [E0] Feature-info probe',
             f"- model: `{meta.get('model')}`  taps(0-idx): {meta.get('tap_indices')}",
             f"- fit: {meta.get('fit_split')} (≤{meta.get('fit_images')} imgs, "
             f"{meta.get('tokens_per_class')} tok/class)  eval: {meta.get('eval_splits')}",
             '- 붕괴 클래스(강조): ' + ', '.join(collapse_names),
             '']
    splits = list(results.keys())
    set_names = sorted({n for s in results.values() for n in s})
    # 표: 특징세트 × split → mIoU + 붕괴 클래스 평균 recall
    header = '| feature set |' + ''.join(
        f" {s} mIoU | {s} collapse-rec |" for s in splits)
    sep = '|---|' + '---|---|' * len(splits)
    lines += [header, sep]
    for name in set_names:
        row = f'| {name} |'
        for s in splits:
            r = results[s].get(name)
            if r is None:
                row += ' — | — |'
                continue
            crec = np.mean([r['recall'][i] for i in collapse_idx])
            row += f" {r['miou']:.2f} | {crec:.1f} |"
        lines += [row]
    lines += ['', '## 붕괴 클래스 per-class recall (%)', '']
    lines += ['| feature set | split | ' + ' | '.join(collapse_names) + ' |',
              '|---|---|' + '---|' * len(collapse_names)]
    for name in set_names:
        for s in splits:
            r = results[s].get(name)
            if r is None:
                continue
            cells = ' | '.join(f"{r['recall'][i]:.1f}" for i in collapse_idx)
            lines += [f'| {name} | {s} | {cells} |']
    if oracle_res and oracle_res.get('eval'):
        lines += ['', '## 오라클 센서 선택 프로브 (부분집합 분류)', '',
                  f"fit_split={oracle_res.get('fit_split')} "
                  f"n_subsets={oracle_res.get('n_subsets')}  "
                  f"note: {oracle_res.get('note')}", '',
                  '| feature set | split | acc | chance |', '|---|---|---|---|']
        for split, sr in oracle_res['eval'].items():
            for name, v in sr.items():
                lines += [f"| {name} | {split} | {v['acc']} | {v['chance']} |"]
    (out_dir / 'probe_table.md').write_text('\n'.join(lines) + '\n')
    print(f"[probe] wrote {out_dir}/probe_results.json + probe_table.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--taps', default='6,12,18,24',
                    help='1-indexed 블록/층 번호(기본 6,12,18,24). [1,n_blocks] clamp 후 -1.')
    ap.add_argument('--fit-split', default='train')
    ap.add_argument('--fit-images', type=int, default=400)
    ap.add_argument('--eval-splits', default='val,test')
    ap.add_argument('--eval-images', type=int, default=300)
    ap.add_argument('--tokens-per-class', type=int, default=4000)
    ap.add_argument('--epochs', type=int, default=30, help='프로브 학습 epoch(20~50 권장)')
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--include-adapted-taps', action='store_true',
                    help='(선택) LoRA 켠 상태 탭 concat 세트 추가')
    ap.add_argument('--oracle-dir', default=None,
                    help='(선택) H16 오라클(tools/oracle_spatial_modality.py) 산출 디렉토리')
    ap.add_argument('--oracle-fit-split', default='val',
                    help='오라클 프로브 fit 에 쓸 split(캐시 있어야 함, 기본 val)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    args.eval_splits = [s.strip() for s in args.eval_splits.split(',') if s.strip()]
    taps = [int(t) for t in args.taps.split(',') if t.strip()]

    import torch
    import yaml
    import val as V

    cfg = yaml.safe_load(open(args.cfg))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V.setup_cudnn()
    model = V.load_model(cfg, Path(args.model_path), device)
    model.eval()
    core = model.module if hasattr(model, 'module') else model
    assert hasattr(core, 'encoder'), "ReliaDINO 계열이 아니다 — encoder 없음"

    ds_cfg = cfg['DATASET']
    modal_names = list(ds_cfg['MODALS'])
    ignore = ds_cfg.get('IGNORE_LABEL', 255)
    n_blocks = len(core.encoder.backbone.blocks)
    tap_indices = resolve_taps(taps, n_blocks)
    print(f"[probe] backbone blocks={n_blocks}  taps(0-idx)={tap_indices}  modals={modal_names}")

    # 클래스 이름/개수 (프로브 로더에서)
    transform = V.get_val_augmentation(cfg['EVAL']['IMAGE_SIZE'], dataset_cfg=ds_cfg)
    probe_ds, _ = V.create_dataset(ds_cfg, 'val', transform, 'val', macvi=False, eval_day=False)
    num_classes, class_names = probe_ds.n_classes, list(probe_ds.CLASSES)
    collapse_idx = [class_names.index(c) for c in DELIVER_COLLAPSE if c in class_names]

    results = run_class_probe(cfg, model, core, tap_indices, modal_names,
                              num_classes, class_names, ignore, args, device)

    oracle_res = None
    if args.oracle_dir:
        oracle_res = run_oracle_probe(cfg, model, core, tap_indices, modal_names,
                                      num_classes, ignore, args, device)

    meta = {
        'model': Path(args.model_path).stem, 'cfg': args.cfg,
        'tap_indices': tap_indices, 'n_blocks': n_blocks, 'modals': modal_names,
        'fit_split': args.fit_split, 'fit_images': args.fit_images,
        'eval_splits': args.eval_splits, 'eval_images': args.eval_images,
        'tokens_per_class': args.tokens_per_class,
        'collapse_classes': [class_names[i] for i in collapse_idx],
    }
    write_reports(results, oracle_res, class_names, collapse_idx, args.out, meta)


if __name__ == '__main__':
    main()
