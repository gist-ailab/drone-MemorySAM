"""스모크 ②: CAFuser 매퍼의 열화 패치 — 토글 off 불변·on 통계·DGFusion 매퍼와 동일성 (GPU 불필요).

사용(같은 스크립트를 두 코드 트리에서 돌려 서로 비교한다):
  # 패치 전 원본 트리(cafuser 패키지를 백업 파일로 복원한 overlay)에서 — 열화 키가 아직 없다
  python smoke_cafb_mapper.py off  <out.json> --config configs/deliver/swin/cafuser_swin_tiny_bs8_200k_deliver_clde.yaml
  # 패치된 트리에서 같은 (a) config, 그리고 (b) config 를 DEGRADE.ENABLED False 로
  python smoke_cafb_mapper.py off  <out.json> --config configs/deliver/swin/cafuser_swin_tiny_bs8_200k_deliver_clde.yaml
  python smoke_cafb_mapper.py off  <out2.json> --config .../cafuser_..._degrade.yaml --opts DATASETS.DELIVER.DEGRADE.ENABLED False DATASETS.DELIVER.RANDOM_DROP "[0.,0.,0.,0.]"
  # 패치된 트리에서 on 통계 + DGFusion 매퍼와 동일성
  python smoke_cafb_mapper.py on <out.json> --config .../cafuser_..._degrade.yaml
  python smoke_cafb_mapper.py cmp a.json b.json     # 두 off 결과가 바이트 동일한가
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DEGRADE_LOG_EVERY", "0")
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402


def h(x):
    if torch.is_tensor(x):
        return hashlib.sha1(x.detach().cpu().numpy().tobytes()).hexdigest()[:16]
    if isinstance(x, np.ndarray):
        return hashlib.sha1(x.tobytes()).hexdigest()[:16]
    return None


def sample_hashes(out):
    r = {}
    for k in sorted(out):
        v = out[k]
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            r[k] = h(v)
        elif hasattr(v, "gt_masks"):          # detectron2 Instances
            r[k + ".gt_classes"] = h(v.gt_classes)
            r[k + ".gt_masks"] = h(v.gt_masks if torch.is_tensor(v.gt_masks) else torch.as_tensor(v.gt_masks))
    return r


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def build_cfg(config_file, opts):
    from train_net import setup

    class A:
        eval_only = False
        inference_only = False
        resume = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "auto"
    A.config_file = config_file
    A.opts = list(opts)
    return setup(A)


def pick_dicts(cfg, n):
    from detectron2.data import DatasetCatalog
    name = cfg.DATASETS.TRAIN[0]
    ds = DatasetCatalog.get(name)
    idx = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    return [ds[i] for i in idx], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["off", "on", "cmp"])
    ap.add_argument("out")
    ap.add_argument("out2", nargs="?")
    ap.add_argument("--config", default="configs/deliver/swin/cafuser_swin_tiny_bs8_200k_deliver_clde.yaml")
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    if args.mode == "cmp":
        a, b = json.load(open(args.out)), json.load(open(args.out2))
        bad = [(i, k) for i in a for k in a[i] if a[i][k] != b.get(i, {}).get(k)]
        print(f"[cmp] 샘플 {len(a)}개 · 항목 {sum(len(v) for v in a.values())}개 · 불일치 {len(bad)}개", "→ 바이트 동일" if not bad and a.keys() == b.keys() else f"→ 불일치 {bad[:5]}")
        sys.exit(0 if not bad and a.keys() == b.keys() else 1)

    from cafuser.data.dataset_mappers.deliver_semantic_dataset_mapper import DELIVERSemanticDatasetMapper as CM
    cfg = build_cfg(args.config, args.opts)
    dicts, idx = pick_dicts(cfg, args.n)

    if args.mode == "off":
        m = CM(cfg, True)
        res = {}
        for i, d in zip(idx, dicts):
            seed_all(1000 + i)
            res[str(i)] = sample_hashes(m(copy.deepcopy(d)))
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"[off] degrade_ctx={'있음' if getattr(m, 'degrade_ctx', None) is not None else '없음/None'} · RANDOM_DROP={list(cfg.DATASETS.DELIVER.RANDOM_DROP)} · 샘플 {len(res)}개 해시 저장 → {args.out}")
        return

    # --- on: 통계 + DGFusion 매퍼와 동일성 ---
    import multiprocessing
    from dgfusion.data import degradation as dg
    from dgfusion.data.dataset_mappers.deliver_semantic_dataset_mapper import DELIVERSemanticDatasetMapper as DM
    cfg_off = build_cfg(args.config, ["DATASETS.DELIVER.DEGRADE.ENABLED", "False", "DATASETS.DELIVER.RANDOM_DROP", "[0.,0.,0.,0.]"])
    m_on = CM(cfg, True)
    m_off = CM(cfg_off, True)
    assert m_on.degrade_ctx is not None and m_off.degrade_ctx is None
    try:
        dm_on = DM(cfg, True)
    except Exception as e:                                    # DGFusion 매퍼가 필요로 하는 키가 없으면 건너뛴다
        dm_on = None
        print("[on] DGFusion 매퍼 생성 실패(동일성 비교 생략):", repr(e)[:160])
    mods = ["CAMERA", "LIDAR", "EVENT", "DEPTH"]
    report = {}
    for step in (50000, 100000, 150000):
        dg.set_shared_step(multiprocessing.Value("i", step))
        exp_cap = dg.current_severity(cfg.DATASETS.DELIVER.DEGRADE, cfg.SOLVER.MAX_ITER)
        chg = {k: [] for k in mods}
        ab = {k: [] for k in mods}
        same_dgf = []
        for i, d in zip(idx, dicts):
            seed_all(2000 + i)
            o_off = m_off(copy.deepcopy(d))
            seed_all(2000 + i)
            o_on = m_on(copy.deepcopy(d))
            for k in mods:
                if k in o_on and k in o_off and torch.is_tensor(o_on[k]):
                    diff = (o_on[k].float() - o_off[k].float()).abs()
                    chg[k].append(float((diff > 0).float().mean()))
                    ab[k].append(float(diff.mean()))
            if dm_on is not None:
                seed_all(2000 + i)
                o_dg = dm_on(copy.deepcopy(d))
                same_dgf.append(all(torch.equal(o_on[k], o_dg[k]) for k in mods if k in o_on and k in o_dg and torch.is_tensor(o_on[k])))
        report[str(step)] = {
            "severity상한": exp_cap,
            "모달별 바뀐 화소 비율(샘플 평균)": {k: round(float(np.mean(v)), 4) for k, v in chg.items() if v},
            "모달별 평균|Δ|(0~255 단위)": {k: round(float(np.mean(v)), 3) for k, v in ab.items() if v},
            "CAFuser 매퍼 == DGFusion 매퍼 (모달 텐서 동일 샘플 수)": f"{sum(same_dgf)}/{len(same_dgf)}" if same_dgf else "비교 안 함",
        }
        print(f"[on] step {step}: {json.dumps(report[str(step)], ensure_ascii=False)}")
    json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
