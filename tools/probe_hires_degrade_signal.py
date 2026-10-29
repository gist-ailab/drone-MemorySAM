#!/usr/bin/env python3
"""DRN-261028-01: 학습 없이 A2 고해상 특징의 실제 열화 식별 신호를 측정한다.

DELIVER test 추론 중 ``model.hires.spm.stem`` 출력의 첫 항목(s4 c1)과
``model.encoder`` 호출 직후 ``last_taps[0]``(블록 6)에 forward hook을 건다.
센서 순서는 DATASET.MODALS(img, depth, event, lidar)이며 패딩 토큰도 포함한다.
각 토큰의 norm=채널 L2, hf=채널 평균 (F-3×3 평균 풀링)^2,
cvar=채널 분산(population)을 계산한다. 이미지 점수는 각 통계의 토큰
평균과 표준편차(population)이며, 토큰 AUROC에는 이미지마다 256개를 샘플한다.

예: python tools/probe_hires_degrade_signal.py --cfg configs/eval/a2s821_eval1024_hpca100_probe.yaml
    --model_path /path/to/A2_checkpoint.pth --split test --out /tmp/hires_probe
스모크: python tools/probe_hires_degrade_signal.py --dry_run
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SIGNALS = ("hires_s4", "block6")
SENSORS = ("img", "depth", "event", "lidar")
STATS = ("norm", "hf", "cvar")
CASES = ("motionblur", "overexposure", "underexposure", "lidarjitter", "eventlowres")
PARTITIONS = CASES + ("night", "fog")


def token_statistics(feature):
    """(B,C,H,W) 특징을 (B,3,H*W) float32 토큰 통계로 변환한다."""
    if feature.ndim != 4 or feature.shape[1] < 1:
        raise ValueError(f"특징 형상은 (B,C,H,W)여야 한다: {tuple(feature.shape)}")
    x = feature.float()
    local = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                         count_include_pad=False)
    values = torch.stack((
        torch.linalg.vector_norm(x, ord=2, dim=1),
        (x - local).square().mean(dim=1),
        x.var(dim=1, unbiased=False),
    ), dim=1)
    return values.flatten(start_dim=2)


def rank_auroc(pos, neg):
    """평균 동점 순위의 Mann–Whitney AUROC. 양쪽 중 하나가 비면 NaN."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg, dtype=np.float64).reshape(-1)
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("AUROC 입력에 NaN 또는 무한대가 있다")
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = rankdata(np.concatenate((pos, neg)), method="average")
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def _tap_map(tap, encoder, encoder_input):
    """이미 맵인 탭 또는 prefix 토큰이 붙은 (B,N,C)/(B,C,N)을 맵으로 변환."""
    if tap.ndim == 4:
        return tap
    if tap.ndim != 3:
        raise ValueError(f"블록 6 탭 형상 오류: {tuple(tap.shape)}")
    patch = int(encoder.patch)
    h, w = encoder_input.shape[-2] // patch, encoder_input.shape[-1] // patch
    n = h * w
    if tap.shape[1] >= n and tap.shape[1] <= n + 32:
        return tap[:, -n:, :].transpose(1, 2).reshape(tap.shape[0], -1, h, w)
    if tap.shape[2] >= n and tap.shape[2] <= n + 32:
        return tap[:, :, -n:].reshape(tap.shape[0], -1, h, w)
    raise ValueError(f"블록 6 토큰 수가 입력 격자 {h}×{w}와 다르다: {tuple(tap.shape)}")


class FeatureCapture:
    """공유 스템과 encoder를 센서 호출 순서대로 수집하고 호출 수를 검증한다."""

    def __init__(self, model, sensors):
        if getattr(model, "hires", None) is None:
            raise RuntimeError("MODEL.HIRES_ADAPTER.ENABLE=true 인 모델이 필요하다 (model.hires=None)")
        spm = model.hires.spm
        if spm.mode != "shared_stem" or spm.stem is None:
            raise RuntimeError(f"MODEL.HIRES_ADAPTER.MODE=shared_stem 이 필요하다 (실제: {spm.mode})")
        if getattr(model, "cmlc_enable", False):
            raise RuntimeError("CMLC 결합 인코딩에서는 센서별 encoder hook 호출을 보장할 수 없다")
        if int(spm.num_modalities) != len(sensors):
            raise RuntimeError("SPM 센서 수와 DATASET.MODALS 수가 다르다")
        self.sensors = sensors
        self.captured = {signal: [] for signal in SIGNALS}
        self.handles = [spm.stem.register_forward_hook(self._stem_hook),
                        model.encoder.register_forward_hook(self._encoder_hook)]

    def reset(self):
        for values in self.captured.values():
            values.clear()

    def _append(self, signal, feature):
        if len(self.captured[signal]) >= len(self.sensors):
            raise RuntimeError(f"{signal} hook 이 센서 수보다 많이 호출되었다")
        stats = token_statistics(feature)
        if stats.shape[0] != 1:
            raise RuntimeError("프로브는 DataLoader batch_size=1을 요구한다")
        self.captured[signal].append(stats[0].detach().cpu().numpy().copy())

    def _stem_hook(self, _module, _inputs, output):
        if not isinstance(output, (tuple, list)) or not output:
            raise RuntimeError("SPM stem 출력이 c1 포함 튜플이 아니다")
        self._append("hires_s4", output[0])

    def _encoder_hook(self, module, inputs, _output):
        taps = module.last_taps
        if not taps:
            raise RuntimeError("encoder.last_taps 가 비었다: TAPS 수집/HIRES 설정 확인")
        # 다음 센서 호출 시 last_taps가 덮이므로 hook 안에서 복사해 소비한다.
        tap = taps[0].detach().clone()
        self._append("block6", _tap_map(tap, module, inputs[0]))

    def checked(self):
        for signal, values in self.captured.items():
            assert len(values) == len(self.sensors), (
                f"{signal} hook 호출 {len(values)}회; 센서 {len(self.sensors)}개 기대")
        return self.captured

    def close(self):
        for handle in self.handles:
            handle.remove()


def _partition_masks(cases, conditions, partition):
    cases = np.asarray(cases)
    conditions = np.asarray(conditions)
    if partition in ("night", "fog"):
        return ((cases == "none") & (conditions == partition),
                (cases == "none") & (conditions == "sun"))
    return cases == partition, cases == "none"


def _auroc_rows(scores, samples, cases, conditions, sensors):
    rows = []
    for partition in PARTITIONS:
        pos_mask, neg_mask = _partition_masks(cases, conditions, partition)
        for si, signal in enumerate(SIGNALS):
            for mi, sensor in enumerate(sensors):
                for level, array, summaries in (
                    ("image", scores[:, si, mi], ("mean", "std")),
                    ("token", samples[:, si, mi], ("-",)),
                ):
                    for ti, stat in enumerate(STATS):
                        for qi, summary in enumerate(summaries):
                            vals = array[:, ti, qi] if level == "image" else array[:, ti, :]
                            pos = vals[pos_mask].reshape(-1)
                            neg = vals[neg_mask].reshape(-1)
                            raw = rank_auroc(pos, neg)
                            rows.append(dict(signal=signal, sensor=sensor,
                                             partition=partition, level=level,
                                             stat=stat, summary=summary,
                                             n_pos=len(pos), n_neg=len(neg),
                                             auroc_raw=raw,
                                             auroc_two_sided=max(raw, 1 - raw)
                                             if np.isfinite(raw) else float("nan")))
    return rows


def _write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path, auroc_rows, sensors):
    lines = ["# 열화 신호 AUROC 요약", "",
             "이미지 단위의 6개 점수 중 양방향 AUROC 최대값(점수명). "
             "열화 5종은 전체 case=none과, night/fog는 case=none인 sun과 비교한다.", ""]
    for signal in SIGNALS:
        lines += [f"## {signal}", "",
                  "| 파티션 | " + " | ".join(sensors) + " |",
                  "|---|" + "---|" * len(sensors)]
        for partition in PARTITIONS:
            cells = []
            for sensor in sensors:
                candidates = [r for r in auroc_rows
                              if r["signal"] == signal and r["sensor"] == sensor
                              and r["partition"] == partition and r["level"] == "image"
                              and np.isfinite(r["auroc_two_sided"])]
                best = max(candidates, key=lambda r: r["auroc_two_sided"]) if candidates else None
                cells.append(f'{best["auroc_two_sided"]:.3f} ({best["stat"]}_{best["summary"]})'
                             if best else "NA")
            lines.append("| " + partition + " | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def dry_run():
    generator = torch.Generator().manual_seed(0)
    stats = token_statistics(torch.randn((2, 4, 5, 6), generator=generator))
    assert stats.shape == (2, 3, 30) and torch.isfinite(stats).all()
    assert torch.allclose(token_statistics(torch.zeros(1, 4, 3, 3)),
                          torch.zeros(1, 3, 9))
    separated = rank_auroc([2, 3, 4], [-2, -1, 0])
    same = rank_auroc(np.arange(100), np.arange(100))
    tied = rank_auroc([1, 1, 1], [1, 1, 1])
    reverse = rank_auroc([-2, -1], [1, 2])
    assert max(separated, 1 - separated) == 1.0
    assert abs(same - 0.5) < 1e-12 and tied == 0.5
    assert max(reverse, 1 - reverse) == 1.0
    print("[dry_run] 토큰 통계·AUROC·동점 처리 통과")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", help="DELIVER 평가 config")
    ap.add_argument("--model_path", help="학습된 ReliaDINO A2 checkpoint")
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--out", help="산출물 디렉터리")
    ap.add_argument("--limit", type=int, default=0, help="앞 N장만 분석; 0=전체")
    ap.add_argument("--tokens_per_image", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    if not all((args.cfg, args.model_path, args.out)):
        ap.error("--cfg, --model_path, --out 은 --dry_run 이 아닐 때 필요하다")
    if args.limit < 0 or args.tokens_per_image < 1:
        ap.error("--limit 은 0 이상, --tokens_per_image 는 1 이상이어야 한다")

    # val.py의 모델 생성은 무거우므로 dry_run 경로와 완전히 분리한다.
    import val as valmod
    from tools.baseline_failure import common

    with open(args.cfg, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg["DATASET"]
    if str(dataset_cfg.get("NAME", "")).upper() != "DELIVER":
        raise ValueError("이 프로브는 DATASET.NAME=DELIVER 전용이다")
    sensors = tuple(dataset_cfg["MODALS"])
    if sensors != SENSORS:
        raise ValueError(f"DATASET.MODALS 순서는 {SENSORS} 이어야 한다: {sensors}")
    model_cfg = cfg["MODEL"]
    if model_cfg.get("NAME") != "ReliaDINO" or not model_cfg.get("HIRES_ADAPTER", {}).get("ENABLE"):
        raise ValueError("ReliaDINO + MODEL.HIRES_ADAPTER.ENABLE=true config가 필요하다")
    tap_layers = model_cfg.get("TAPS", {}).get("LAYERS", [6, 12, 18, 24])
    if not tap_layers or int(tap_layers[0]) != 6:
        raise ValueError("encoder.last_taps[0]이 블록 6이 되려면 TAPS.LAYERS[0]=6이어야 한다")
    device = torch.device(cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    valmod.setup_cudnn()
    image_size = (cfg["EVAL"]["IMAGE_SIZE"] if args.split == "val" else
                  cfg.get("TEST", {}).get("IMAGE_SIZE", cfg["EVAL"]["IMAGE_SIZE"]))
    transform = valmod.get_val_augmentation(image_size, dataset_cfg=dataset_cfg)
    dataset, _ = valmod.create_dataset(dataset_cfg, args.split, transform, args.split,
                                       macvi=False, eval_day=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=False,
                        collate_fn=valmod._collate_fn)
    model = valmod.load_model(cfg, Path(args.model_path), device)
    model.eval()
    capture = FeatureCapture(model, sensors)
    rng = np.random.default_rng(args.seed)
    image_rows, score_list, sample_list = [], [], []
    seen_ids = set()
    try:
        with torch.no_grad():
            for images, _labels, metas in tqdm(loader, desc=f"hires probe [{args.split}]"):
                capture.reset()
                images = [x.to(device) for x in images]
                model(images, multimask_output=True)
                captured = capture.checked()
                meta = metas[0]
                rgb_path = meta.get("paths", {}).get("img")
                if rgb_path is None:
                    raise RuntimeError("meta['paths']['img']가 없다")
                image_id = common.image_id_from_rel(rgb_path)
                if image_id in seen_ids:
                    raise RuntimeError(f"image_id 중복: {image_id}")
                seen_ids.add(image_id)
                condition, case = common.parse_condition_case(rgb_path)
                if case not in (*CASES, "none"):
                    raise RuntimeError(f"알 수 없는 열화 case: {case}")
                scores = np.empty((len(SIGNALS), len(sensors), len(STATS), 2), dtype=np.float32)
                samples = np.empty((len(SIGNALS), len(sensors), len(STATS),
                                    args.tokens_per_image), dtype=np.float32)
                row = dict(image_id=image_id, condition=condition, case=case)
                for si, signal in enumerate(SIGNALS):
                    for mi, sensor in enumerate(sensors):
                        values = captured[signal][mi]
                        scores[si, mi, :, 0] = values.mean(axis=1)
                        scores[si, mi, :, 1] = values.std(axis=1, ddof=0)
                        # 격자가 256개보다 작을 때만 복원 추출해 직사각 NPZ를 유지한다.
                        indices = rng.choice(values.shape[1], size=args.tokens_per_image,
                                             replace=values.shape[1] < args.tokens_per_image)
                        samples[si, mi] = values[:, indices]
                        for ti, stat in enumerate(STATS):
                            for qi, summary in enumerate(("mean", "std")):
                                row[f"{signal}_{sensor}_{stat}_{summary}"] = float(scores[si, mi, ti, qi])
                if not np.isfinite(scores).all() or not np.isfinite(samples).all():
                    raise RuntimeError(f"비유한 특징 통계: {image_id}")
                image_rows.append(row)
                score_list.append(scores)
                sample_list.append(samples)
                if args.limit and len(image_rows) >= args.limit:
                    break
    finally:
        capture.close()

    if args.split == "test" and args.limit == 0:
        assert len(image_rows) == 1897, f"DELIVER test 장수: {len(image_rows)} != 1897"
    if not image_rows:
        raise RuntimeError("분석한 이미지가 없다")
    scores = np.stack(score_list)
    samples = np.stack(sample_list)
    cases = np.array([r["case"] for r in image_rows])
    conditions = np.array([r["condition"] for r in image_rows])
    auroc_rows = _auroc_rows(scores, samples, cases, conditions, sensors)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_columns = ["image_id", "condition", "case"] + [
        f"{signal}_{sensor}_{stat}_{summary}" for signal in SIGNALS
        for sensor in sensors for stat in STATS for summary in ("mean", "std")]
    _write_csv(out / "image_scores.csv", image_rows, image_columns)
    np.savez_compressed(out / "token_samples.npz", samples=samples,
                        image_ids=np.array([r["image_id"] for r in image_rows]),
                        conditions=conditions, cases=cases, signals=np.array(SIGNALS),
                        sensors=np.array(sensors), stats=np.array(STATS))
    _write_csv(out / "auroc.csv", auroc_rows,
               ["signal", "sensor", "partition", "level", "stat", "summary",
                "n_pos", "n_neg", "auroc_raw", "auroc_two_sided"])
    _write_summary(out / "auroc_summary.md", auroc_rows, sensors)
    partition_counts = {p: {"positive": int(_partition_masks(cases, conditions, p)[0].sum()),
                             "negative": int(_partition_masks(cases, conditions, p)[1].sum())}
                        for p in PARTITIONS}
    run = dict(arguments=vars(args), cfg=str(Path(args.cfg).resolve()),
               model_path=str(Path(args.model_path).resolve()),
               split=args.split, out=str(out.resolve()), limit=args.limit,
               tokens_per_image=args.tokens_per_image, seed=args.seed,
               num_images=len(image_rows), case_counts=dict(Counter(cases)),
               partition_counts=partition_counts, sensors=list(sensors),
               signals=list(SIGNALS))
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print(f"[probe] {len(image_rows)}장 분석 완료: {out}")


if __name__ == "__main__":
    main()
