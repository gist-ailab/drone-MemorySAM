#!/usr/bin/env python3
"""기준선(DGFusion/CAFuser) 강건 벤치(EMM/RMM/NM) 단일 순회 평가기 (2026-09-23).

판정 세션 승인(가) — 케이스마다 detectron2 전체 재추론 프로세스를 따로 띄우면
64 케이스 × 체크포인트 2개 ≈ 15 GPU-시간이 든다(2026-09-23 D4 보고). 우리 자체
`tools/missing_modality_eval.py` 와 같은 구조로, **데이터셋을 한 번만 순회하며
이미지마다 케이스 전부를 forward** 해 시간을 줄인다.

🔴 케이스 목록·생성기는 재정의하지 않는다. `tools.missing_modality_eval.build_cases`
를 그대로 불러 쓰고(판정 세션 지시 #1), 시드도 그 모듈의 `--seed` 기본값(0)과 같은
규약(`torch.Generator().manual_seed(seed)`)을 쓴다. 노이즈 수식도 `baseline_noise.py`
를 통해 우리 실제 함수를 그대로 호출한다(재구현 없음, D4 2026-09-23 단위검사로 확인).

모달 raw 텐서 ↔ 정규화 왕복은 기존 `probe_dgfusion.py` 의 `modal_order_from_cfg`(A12)
를 재사용한다 — 모델 pixel_mean/pixel_std 버퍼에서 모달 순서·통계를 읽는 검증된 경로다.

Gaussian NM(보조, std 0.2 한 레벨)은 `nm_density` 기본값과 값이 겹칠 수 있어(0.2 가
둘 다에 있을 수 있음) case id·density 속성이 charlie S&P 케이스와 부딪히지 않도록
별도로 관리하고, `write_outputs`(우리 도구의 실제 출력 함수, 형식 그대로 재사용)를
거치지 않고 이 스크립트가 직접 summary 에 addendum 으로 붙인다.

산출물: 우리 도구와 같은 형식 — `<out>/<split>/missing_modality/` 아래
emm.csv · rmm_r*.csv · nm.csv · nm_gaussian.csv · summary.json · summary.md.

예:
  python tools/baseline_failure/robust_bench_eval.py \
    --config-file configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml \
    --weights output/.../model_0079999.pth --split test \
    --out /ailab_mat2/.../robust_bench/dgf_a
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402
from tools.baseline_failure.probe_dgfusion import (  # noqa: E402
    MODAL_KEYS, deliver_gt_paths, modal_order_from_cfg)
from tools.missing_modality_eval import (  # noqa: E402
    _case_miou, _write_combo_csv, build_cases, write_outputs)


def model_modal_stats(model, cfg, modal):
    """probe_dgfusion.model_modal_mean 과 같은 검증 규칙으로 (인덱스, mean, std) 를 뽑는다.

    그 함수는 mean 만 반환한다 — NM 훅에는 std 도 필요해 여기서 둘 다 뽑는 버전을
    따로 둔다(같은 검증: 버퍼 존재·길이·모달 순서 확인, 추측 보정 없음).
    """
    import torch
    if modal not in MODAL_KEYS:
        raise ValueError(f"modal 은 {MODAL_KEYS} 중 하나여야 한다: {modal!r}")
    modals = modal_order_from_cfg(cfg)
    if modal not in modals:
        raise RuntimeError(
            f"모달 {modal} 이 cfg 모달 순서({modals}) 에 없다(추측 금지).")
    i = modals.index(modal)
    pm = getattr(model, "pixel_mean", None)
    ps = getattr(model, "pixel_std", None)
    if not (torch.is_tensor(pm) and torch.is_tensor(ps)):
        raise RuntimeError(
            "model.pixel_mean/pixel_std 버퍼가 없다 — dgfusion.py:347-348 구조가 "
            "아니다(추측 금지).")
    pm_flat = pm.detach().float().cpu().reshape(-1)
    ps_flat = ps.detach().float().cpu().reshape(-1)
    n = len(modals)
    if pm_flat.numel() != 3 * n or ps_flat.numel() != 3 * n:
        raise RuntimeError(
            f"pixel_mean/std 길이가 3*모달수({3 * n}) 가 아니다 — 버퍼 구조가 다르다"
            "(추측 금지).")
    return i, pm_flat[3 * i:3 * i + 3], ps_flat[3 * i:3 * i + 3]


def build_all_cases(protocol, modals, rmm_ratios, nm_density, nm_gaussian_std, seed):
    """S&P 축 케이스(`build_cases` 그대로)와 Gaussian 케이스(보조 1개)를 만들어
    (main_cases, gaussian_case_or_none) 로 돌려준다.

    Gaussian 은 값이 nm_density 와 겹칠 수 있어(둘 다 기본 0.2 를 포함) id·density 를
    S&P 케이스와 겹치지 않게 바꾼다 — 안 그러면 hists 딕셔너리에서 두 케이스가
    같은 키로 뭉개진다(추측이 아니라 실제로 겹치는 값이라 반드시 처리해야 한다).
    """
    import torch

    def gen_factory():
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    M = len(modals)
    cases = build_cases(protocol, M, modals, rmm_ratios, nm_density, gen_factory,
                        nm_gaussian=False)

    gaussian_case = None
    if protocol in ("nm", "all"):
        g_cases = build_cases("nm", M, modals, rmm_ratios, [nm_gaussian_std],
                              gen_factory, nm_gaussian=True,
                              nm_gaussian_std=nm_gaussian_std)
        gaussian_case = next(c for c in g_cases if c.group == "nm")
        gaussian_case.id = f"nm_gaussian_std{nm_gaussian_std}"
        gaussian_case.density = None   # S&P nm_density 값과 절대 안 겹치게(스칼라 표시용)

    ids = [c.id for c in cases] + ([gaussian_case.id] if gaussian_case else [])
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"케이스 id 가 겹친다(버그) — {ids}")
    return cases, gaussian_case


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="all", choices=["emm", "rmm", "nm", "all"])
    ap.add_argument("--rmm_ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--nm_density", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--nm_gaussian_std", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="missing_modality_eval 과 같은 기본값")
    ap.add_argument("--limit", type=int, default=0, help="디버그용 앞 N 장만(0=전체)")
    ap.add_argument("--deliver-root", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="이 case id 들만 계산한다(등가검증·스모크용). 예: "
                         "--only clean 'emm|present=LIDAR+EVENT+DEPTH' nm0.1")
    ap.add_argument("--list-cases", action="store_true",
                    help="케이스 id 목록만 찍고 종료(모델 로딩 없이, 스모크용)")
    args = ap.parse_args()

    modals_fixed = list(MODAL_KEYS)   # id 목록만 볼 때는 cfg 없이도 케이스를 만들 수 있게
    if args.list_cases:
        cases, gaussian_case = build_all_cases(
            args.protocol, modals_fixed, args.rmm_ratios, args.nm_density,
            args.nm_gaussian_std, args.seed)
        all_cases = cases + ([gaussian_case] if gaussian_case else [])
        print(f"[robust-bench] 케이스 {len(all_cases)}개 (protocol={args.protocol})")
        for c in all_cases:
            print(f"  {c.id}  group={c.group} n_missing={c.n_missing} "
                  f"ratio={c.ratio} density={c.density}")
        return

    # 지연 import — 기준선 repo 에서만 사용 가능.
    import torch  # noqa: F401
    from detectron2.checkpoint import DetectionCheckpointer
    from train_net import Trainer, setup

    class _A:
        config_file = args.config_file
        opts = ["MODEL.WEIGHTS", args.weights, "MODEL.IS_TRAIN", "False"]
        eval_only = True
        inference_only = False
        resume = False
        num_gpus = 1
        num_machines = 1
        machine_rank = 0
        dist_url = "auto"

    cfg = setup(_A)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        args.weights, resume=False)
    model.eval()

    modals = modal_order_from_cfg(cfg)
    stats = {}
    for m in modals:
        i, mean, std = model_modal_stats(model, cfg, m)
        stats[m] = (mean, std)
        print(f"[robust-bench] 모달 {m} 인덱스 {i} 평균/표준편차 확보")

    cases, gaussian_case = build_all_cases(
        args.protocol, modals, args.rmm_ratios, args.nm_density,
        args.nm_gaussian_std, args.seed)
    all_cases = cases + ([gaussian_case] if gaussian_case else [])

    if args.only:
        wanted = set(args.only)
        all_cases = [c for c in all_cases if c.id in wanted]
        if not all_cases:
            raise SystemExit(f"--only 에 해당하는 case 가 없다: {args.only}")
        cases = [c for c in cases if c.id in wanted]
        gaussian_case = gaussian_case if (gaussian_case and gaussian_case.id in wanted) else None
    print(f"[robust-bench] case {len(all_cases)}개 -> 이미지마다 {len(all_cases)}회 forward")

    ds_name = (cfg.DATASETS.TEST_SEMANTIC[0] if hasattr(cfg.DATASETS, "TEST_SEMANTIC")
              else cfg.DATASETS.TEST[0])
    loader = Trainer.build_test_loader(cfg, ds_name)

    n_classes = common.N_CLASSES
    ignore = common.IGNORE_LABEL
    hists = {c.id: np.zeros((n_classes, n_classes), dtype=np.int64) for c in all_cases}

    n_seen = 0
    for bi, batch in enumerate(loader):
        if args.limit and bi >= args.limit:
            break
        d0 = batch[0]
        file_name = d0.get("file_name", "")
        _, sem_gt_path = deliver_gt_paths(file_name, args.deliver_root)
        gt = common.load_gt_deliver(sem_gt_path)

        # base_list: 우리 도구의 "정규화된 배치 리스트"와 같은 형태 — 모달 순서대로
        # (1,C,H,W) 정규화 텐서. 배치 차원 1 은 missing_modality_eval 의 BS1 규약과
        # 같다(정규화 텐서 shape 이 같아야 case.apply_fn·gen 의 rand 소비가 등가다).
        base_list = []
        for m in modals:
            v = d0.get(m)
            if v is None:
                raise RuntimeError(f"배치 dict 에 모달 {m} 키가 없다: {list(d0)}")
            mean, std = stats[m]
            mean_ = mean.to(v.dtype).view(-1, 1, 1)
            std_ = std.to(v.dtype).view(-1, 1, 1)
            base_list.append(((v - mean_) / std_).unsqueeze(0))

        with torch.no_grad():
            for case in all_cases:
                imgs = case.apply_fn(base_list)
                d_case = dict(d0)
                for j, m in enumerate(modals):
                    mean, std = stats[m]
                    mean_ = mean.to(imgs[j].dtype).view(-1, 1, 1)
                    std_ = std.to(imgs[j].dtype).view(-1, 1, 1)
                    raw_j = (imgs[j].squeeze(0) * std_ + mean_).to(v.dtype)
                    d_case[m] = raw_j
                    if m == "CAMERA":
                        d_case["image"] = raw_j
                out = model([d_case])
                sem = out[0].get("sem_seg")
                if sem is None:
                    raise RuntimeError("모델 출력에 sem_seg 가 없다.")
                sem = sem.float().cpu().numpy()
                pred = (sem.argmax(0).astype(np.uint8) if sem.ndim == 3
                       else sem.astype(np.uint8))
                if pred.shape != gt.shape:
                    pred = common.resize_nearest(pred, gt.shape[0], gt.shape[1])
                hists[case.id] += common.confusion_matrix(pred, gt, n_classes, ignore)
        n_seen += 1
        if n_seen % 200 == 0:
            print(f"[robust-bench] {n_seen} 장 처리", flush=True)

    print(f"[robust-bench] 총 {n_seen} 장 완료")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_out, summary = write_outputs(
        args.out, args.split, cases, hists, len(modals), modals, list(common.CLASSES),
        args.rmm_ratios, args.nm_density, args.protocol, False, False,
        args.nm_gaussian_std, args.weights, args.config_file)

    if gaussian_case is not None:
        miou, _ = _case_miou(hists[gaussian_case.id])
        summary["NM_gaussian"] = {"std": args.nm_gaussian_std, "mIoU": round(miou, 4)}
        _write_combo_csv(base_out / "nm_gaussian.csv", [gaussian_case], hists,
                         list(common.CLASSES))
        (base_out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[robust-bench] NM Gaussian(std={args.nm_gaussian_std}) mIoU = {miou:.4f}")

    print(f"[robust-bench] clean mIoU = {summary['clean_mIoU']}")
    print(f"[robust-bench] -> {base_out}")


if __name__ == "__main__":
    main()
