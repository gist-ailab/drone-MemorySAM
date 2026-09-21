#!/usr/bin/env python3
"""[P54-QAF G-signal] 품질 토큰 η̂ 치환 검정(제안서 §4, 2606.26473).

QAF 모델(MODEL.QAF.ENABLE=true)이 품질 토큰 η̂ 을 **실제로 쓰는지**를 잰다. 같은
열화 입력에서 세 조건의 mIoU 를 비교한다:
  (i)  normal   — 정상 η̂ (quality_head 예측 그대로)
  (ii) shuffled — η̂ 를 배치/샘플 간 셔플(각 마스크 모달의 η̂_scalar·η̂_token 을
                  다른 샘플의 것으로 치환). B≥2 는 배치 내 순환 치환(고정점 없음),
                  B==1 은 이전 배치의 η̂ 를 쓰는 롤링 셔플.
  (iii) zero    — η̂ = 0 고정(key 마스크 log(1)=0 = 무마스크, 가중 = 산술평균 = 항등).

기대: clean 에서는 (i)≈(ii)≈(iii)(η̂ 무영향 = 항등), 열화(depth RMM r=0.5·완전
결측)에서는 (i) > (ii) 와 (i) > (iii) 가 +1.0 이상(η̂ 가 열화 모달의 오염을 실제로
막고 있다는 증거).

핵심 규약 (미수정 파일 재사용 — 복제 금지):
- 추론·로더·채점: val.py(load_model·create_dataset·get_val_augmentation·_collate_fn·
  _argmax_pred·_unpad_resize_to_orig)와 tools/baseline_failure/common.py 를 import.
- 열화 주입(zero-fill·RMM 마스크): tools/missing_modality_eval.py 의 zero_fill·
  rmm_degrade 를 그대로 import 재사용(정규화 후 배치 텐서에 주입).
- η̂ 오버라이드: semseg/models/reliadino/fusion.py 의 self._qaf_eta_override 훅으로
  주입(normal 조건은 훅 None → QAF 정상 경로와 byte-동일).

예:
  python tools/qaf_permutation_test.py \
    --cfg configs/jarvis-deliver_rgbdel_P46_c3only_seed20260821_screen40_Q3.yaml \
    --ckpt <QAF val-best.pth> --split val --subset_every 20 \
    --protocols clean,rmm_depth_0.5,missing_depth \
    --out /drone_nas/.../analysis_logs/qaf_permutation_YYYYMMDD
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common                      # noqa: E402
from tools.missing_modality_eval import zero_fill, rmm_degrade  # noqa: E402

CONDITIONS = ("normal", "shuffled", "zero")


# ===========================================================================
# η̂ 오버라이드 함수 — fusion._qaf_eta_override 훅에 주입할 callable
# ===========================================================================
def zero_override(qaf_pred: dict) -> dict:
    """η̂ = 0 고정. key 마스크 log(1−0)=0(무마스크), 가중 (1−0)=1(산술평균) = 항등."""
    out = dict(qaf_pred)
    out["eta_scalar"] = torch.zeros_like(qaf_pred["eta_scalar"])
    out["eta_token"] = torch.zeros_like(qaf_pred["eta_token"])
    return out


class ShuffleOverride:
    """η̂ 를 샘플 간 치환하는 결정론적 오버라이드(마스크 모달만).

    B≥2: 배치 내 순환 치환 perm=(arange+k)%B, k∈[1,B-1] 을 torch.Generator(seed)
         로 모달마다 독립 추출 → 고정점 없음(각 샘플이 다른 샘플의 η̂ 를 받음).
    B==1: 배치 내 셔플 불가 → 직전 배치의 η̂ 를 쓴다(롤링 셔플). 첫 배치는 이전이
         없어 자기 η̂ 유지(치환 없음, summary 에 반영되지 않는 극소수).

    프로토콜(열화 조건)마다 인스턴스 1개를 두어 롤링 상태를 분리한다.
    """

    def __init__(self, mask_idx, seed: int):
        self.mask_idx = list(mask_idx)
        self.g = torch.Generator()
        self.g.manual_seed(int(seed))
        self._prev = None                       # B==1 롤링용 직전 η̂ (cpu)

    def __call__(self, qaf_pred: dict) -> dict:
        es = qaf_pred["eta_scalar"]             # (B,m)
        et = qaf_pred["eta_token"]              # (B,m,h,w)
        B = es.shape[0]
        new_es, new_et = es.clone(), et.clone()
        if B >= 2:
            for j in self.mask_idx:
                k = 1 + int(torch.randint(0, B - 1, (1,), generator=self.g).item())
                perm = (torch.arange(B) + k) % B
                new_es[:, j] = es[perm, j]
                new_et[:, j] = et[perm, j]
        else:
            if self._prev is not None:
                pes, pet = self._prev
                if pes.shape == es.shape and pet.shape == et.shape:
                    for j in self.mask_idx:
                        new_es[:, j] = pes[:, j].to(es.device, es.dtype)
                        new_et[:, j] = pet[:, j].to(et.device, et.dtype)
            self._prev = (es.detach().cpu().clone(), et.detach().cpu().clone())
        out = dict(qaf_pred)
        out["eta_scalar"] = new_es
        out["eta_token"] = new_et
        return out


# ===========================================================================
# 프로토콜(열화 조건) apply_fn 빌더 — 순수 함수(스모크가 직접 assert)
# ===========================================================================
def build_protocols(names, modal_names, gen_factory):
    """이름 리스트 → [(name, apply_fn), ...]. apply_fn: base_list -> injected_list.

    지원 이름:
      clean               — 무열화(원본 그대로).
      missing_<modal>     — <modal> 을 정규화 후 0 으로 채움(완전 결측, zero_fill).
      rmm_<modal>_<r>     — <modal> 을 비율 r 픽셀×채널 독립 드롭(rmm_degrade).
    <modal> 은 DATASET.MODALS 에 있어야 한다(하드코딩 금지 — 없으면 시끄럽게 실패).
    gen_factory(): RMM 재현용 새 torch.Generator(seed 설정 완료)를 돌려준다.
    """
    idx_of = {nm: i for i, nm in enumerate(modal_names)}
    protocols = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        if name == "clean":
            protocols.append((name, lambda base: list(base)))
        elif name.startswith("missing_"):
            modal = name[len("missing_"):]
            if modal not in idx_of:
                raise ValueError(f"[perm] missing_<modal> 의 modal={modal!r} 가 "
                                 f"MODALS({modal_names})에 없다.")
            mi = (idx_of[modal],)
            protocols.append((name, lambda base, m=mi: zero_fill(base, m)))
        elif name.startswith("rmm_"):
            body = name[len("rmm_"):]
            try:
                modal, r_str = body.rsplit("_", 1)
                r = float(r_str)
            except ValueError:
                raise ValueError(f"[perm] rmm_<modal>_<r> 형식이 아니다: {name!r}")
            if modal not in idx_of:
                raise ValueError(f"[perm] rmm 의 modal={modal!r} 가 "
                                 f"MODALS({modal_names})에 없다.")
            mi = (idx_of[modal],)
            gen = gen_factory()
            protocols.append(
                (name, lambda base, m=mi, rr=r, g=gen: rmm_degrade(base, m, rr, g)))
        else:
            raise ValueError(
                f"[perm] 알 수 없는 프로토콜 {name!r} "
                f"(clean | missing_<modal> | rmm_<modal>_<r> 만 지원).")
    return protocols


# ===========================================================================
# 평가 루프
# ===========================================================================
def evaluate(model, loader, protocols, mask_idx, n_classes, ignore, device,
             unpad_fn, argmax_fn, shuffle_seed, limit=0):
    """각 배치를 프로토콜×조건(normal/shuffled/zero)으로 forward 해 혼동행렬 누적.

    shuffled 는 프로토콜마다 ShuffleOverride 1개(롤링 상태 분리)를 배치 순서대로
    호출한다. 반환 = {(protocol, condition): hist(n,n)}.
    """
    hists = {(p, c): np.zeros((n_classes, n_classes), dtype=np.int64)
             for (p, _) in protocols for c in CONDITIONS}
    shufflers = {p: ShuffleOverride(mask_idx, shuffle_seed) for (p, _) in protocols}
    fusion = model.fusion
    n_seen = 0
    with torch.no_grad():
        for images, labels, metas in tqdm(loader, desc="qaf-perm"):
            base = [x.to(device) for x in images]
            for pname, apply_fn in tqdm(protocols, desc="protocols", leave=False):
                imgs = apply_fn(base)
                for cond in CONDITIONS:
                    if cond == "normal":
                        fusion._qaf_eta_override = None
                    elif cond == "shuffled":
                        fusion._qaf_eta_override = shufflers[pname]
                    else:
                        fusion._qaf_eta_override = zero_override
                    output = model(imgs, multimask_output=True)
                    fusion._qaf_eta_override = None
                    logits = output[0] if isinstance(output, (tuple, list)) else output
                    preds = logits.softmax(dim=1)
                    pred_labels = argmax_fn(preds, n_classes, None)
                    for b in range(pred_labels.shape[0]):
                        meta = metas[b]
                        orig_label = meta.get("orig_label")
                        if orig_label is None:
                            continue
                        pred_b = pred_labels[b]
                        pred_resized = unpad_fn(
                            pred_b, meta["orig_h"], meta["orig_w"],
                            model_size=int(pred_b.shape[0]))
                        pred_np = pred_resized.cpu().numpy().astype(np.int64)
                        gt_np = orig_label.cpu().numpy().astype(np.int64)
                        hists[(pname, cond)] += common.confusion_matrix(
                            pred_np, gt_np, n_classes, ignore)
            n_seen += len(metas)
            if limit and n_seen >= limit:
                break
    return hists


def _miou(hist):
    _, miou = common.global_miou_from_cm(hist)
    return float(miou)


def build_report(protocols, hists):
    """표 protocol × {normal, shuffled, zero} + Δ(normal−shuffled, normal−zero)."""
    rows = []
    for pname, _ in protocols:
        mn = _miou(hists[(pname, "normal")])
        ms = _miou(hists[(pname, "shuffled")])
        mz = _miou(hists[(pname, "zero")])
        rows.append({
            "protocol": pname,
            "normal": round(mn, 4),
            "shuffled": round(ms, 4),
            "zero": round(mz, 4),
            "delta_normal_minus_shuffled": round(mn - ms, 4),
            "delta_normal_minus_zero": round(mn - mz, 4),
        })
    return rows


def _print_table(rows):
    print("\n[qaf-perm] G-signal 치환 검정 — protocol × {normal, shuffled, zero} mIoU")
    hdr = f"{'protocol':<20} {'normal':>9} {'shuffled':>9} {'zero':>9} " \
          f"{'Δ(n−sh)':>9} {'Δ(n−z)':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['protocol']:<20} {r['normal']:>9.4f} {r['shuffled']:>9.4f} "
              f"{r['zero']:>9.4f} {r['delta_normal_minus_shuffled']:>9.4f} "
              f"{r['delta_normal_minus_zero']:>9.4f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True, help="학습 config(QAF on). val.py 와 동일 로드.")
    ap.add_argument("--ckpt", required=True, help="QAF val-best 체크포인트.")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--protocols", default="clean,rmm_depth_0.5,missing_depth",
                    help="쉼표 구분. clean | missing_<modal> | rmm_<modal>_<r>.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="셔플/RMM 재현 시드.")
    ap.add_argument("--device", default=None, help="미지정 시 cfg.DEVICE 또는 자동.")
    ap.add_argument("--subset_every", type=int, default=1,
                    help="k>1 이면 k장마다 1장(결정론적 부분집합, 스크린용).")
    ap.add_argument("--batch", type=int, default=1,
                    help="평가 배치 크기. 셔플 검정은 B≥2 를 권장(배치 내 치환).")
    ap.add_argument("--limit", type=int, default=0, help="디버그용 앞 N 장만(0=전체).")
    args = ap.parse_args()

    import val as valmod

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(
        args.device if args.device else
        cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    valmod.setup_cudnn()

    dataset_cfg = cfg["DATASET"]
    eval_cfg = cfg["EVAL"]
    test_cfg = cfg.get("TEST", {})
    mode = args.split
    image_size = (eval_cfg["IMAGE_SIZE"] if mode == "val"
                  else test_cfg.get("IMAGE_SIZE", eval_cfg["IMAGE_SIZE"]))
    transform = valmod.get_val_augmentation(image_size, dataset_cfg=dataset_cfg)

    dataset, has_gt = valmod.create_dataset(dataset_cfg, mode, transform, mode,
                                            macvi=False, eval_day=False)
    base_dataset = dataset
    if not has_gt:
        print("[qaf-perm] ⚠️ has_gt=False — GT 없는 split 은 채점 불가", flush=True)

    if args.subset_every > 1:
        from torch.utils.data import Subset
        idx = list(range(0, len(dataset), args.subset_every))
        dataset = Subset(dataset, idx)
        print(f"[qaf-perm] ⚠️ 부분집합 평가: {len(idx)} 장(every {args.subset_every}) "
              f"— 스크린용(벤치 비교 아님).", flush=True)
    loader = DataLoader(dataset, batch_size=max(1, args.batch), num_workers=4,
                        pin_memory=False, collate_fn=valmod._collate_fn)

    model = valmod.load_model(cfg, Path(args.ckpt), device)
    model.eval()

    fusion = getattr(model, "fusion", None)
    if fusion is None or not getattr(fusion, "qaf_enable", False):
        print("[qaf-perm] ❌ 모델에 QAF 가 켜져 있지 않다(MODEL.QAF.ENABLE=true 인 "
              "config·ckpt 필요). 치환 검정은 η̂ 가 있어야 성립한다.", flush=True)
        sys.exit(2)
    if not hasattr(fusion, "_qaf_eta_override"):
        print("[qaf-perm] ❌ fusion 에 _qaf_eta_override 훅이 없다(코드 버전 불일치).",
              flush=True)
        sys.exit(2)

    modal_names = list(dataset_cfg["MODALS"])
    mask_idx = list(fusion.qaf_mask_idx)
    n_classes = base_dataset.n_classes
    ignore = base_dataset.ignore_label
    print(f"[qaf-perm] modals={modal_names} · η̂-mask idx={mask_idx} "
          f"({[modal_names[i] for i in mask_idx]}) · B={max(1, args.batch)}")

    def gen_factory():
        g = torch.Generator()
        g.manual_seed(args.seed)
        return g

    names = args.protocols.split(",")
    protocols = build_protocols(names, modal_names, gen_factory)
    print(f"[qaf-perm] protocols={[p for p, _ in protocols]} × conditions={CONDITIONS} "
          f"→ 배치마다 {len(protocols) * len(CONDITIONS)}회 forward")

    hists = evaluate(model, loader, protocols, mask_idx, n_classes, ignore, device,
                     valmod._unpad_resize_to_orig, valmod._argmax_pred,
                     shuffle_seed=args.seed, limit=args.limit)

    rows = build_report(protocols, hists)
    _print_table(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "split": mode,
        "modals": modal_names,
        "eta_mask_idx": mask_idx,
        "eta_mask_modals": [modal_names[i] for i in mask_idx],
        "batch": max(1, args.batch),
        "subset_every": int(args.subset_every),
        "seed": int(args.seed),
        "limit": int(args.limit),
        "benchmark_comparable": bool(args.subset_every == 1 and not args.limit),
        "conditions": list(CONDITIONS),
        "expectation": ("clean: normal≈shuffled≈zero(항등); 열화: "
                        "Δ(normal−shuffled)>+1.0 & Δ(normal−zero)>+1.0"),
        "rows": rows,
    }
    (out_dir / "perm_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[qaf-perm] -> {out_dir / 'perm_report.json'}")


if __name__ == "__main__":
    main()
