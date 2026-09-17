#!/usr/bin/env python3
"""DELIVER 결측·열화 모달 강건성 프로토콜(EMM/RMM/NM) 평가기.

DELIVER 신작(벤치마크 arXiv 2503.18445, 코드 `Chenfei-Liao/Multi-Modal-Semantic-
Segmentation-Robustness-Benchmark`)은 거의 전부 결측 모달 수치를 병기한다. 우리
모델(ReliaDINO)은 이 프로토콜로 평가된 적이 없어, 이 도구가 우리 `val.py` 추론
경로를 **그대로 재사용**해 같은 프로토콜의 수치를 낸다.

핵심 규약 (val.py 미수정 — `tools/baseline_failure/dump_preds_ours.py` 와 동일한
import-재사용 방식):
- 추론은 val.py 와 동일하다(해상도 EVAL.IMAGE_SIZE, BS1, TTA 없음, native-GT 채점).
  실제로 `val.load_model`·`val.create_dataset`·`val.get_val_augmentation`·
  `val._collate_fn`·`val._argmax_pred`·`val._unpad_resize_to_orig` 를 import 해 쓴다.
- 채점은 `tools/baseline_failure/common.py` 의 혼동행렬·mIoU 규약(25클래스, 부재
  클래스 IoU=0 포함)을 쓴다 = val.py 의 `metrics.update(...)` native-GT 경로와 동일.
- **주입은 정규화(Normalize) 후 배치 텐서에** 한다(벤치마크와 동일: `images[i].zero_()`).
  get_val_augmentation 이 Normalize 를 포함하므로, DataLoader 가 내놓는 배치 텐서는
  이미 정규화된 상태다. 여기서 그 배치 텐서를 복제해 주입한다(원본 배치 불변).

프로토콜:
- EMM(Entire-Missing Modality): 모달 M개에서 결측 0~M−1개의 조합을 전부 열거
  (전부-존재 포함, 전부-결측 제외 → 4모달이면 15조합). 결측 모달은 정규화 후 0 으로
  채운다. 집계 = ① 15조합 단순평균 ② Bernoulli 기대값 E(p), p∈{0.2,0.1,0.05}.
- RMM(Random-Missing Modality): 같은 15조합에서 "결측"을 완전 0 대신 픽셀×채널 독립
  드롭(비율 r)으로 주입. 시드 고정 생성기로 재현 가능. 집계 EMM 과 동일 2종, r별.
- NM(Noise on Modality): 결측 없이 4모달 전부에 salt-and-pepper(밀도 d) 노이즈.
  Gaussian 은 벤치마크 릴리스에서 주석 처리돼 있으므로 `--nm_gaussian` 옵션으로만 둔다.

예:
  python tools/missing_modality_eval.py \
    --cfg configs/eval/bengio-deliver_rgbdel_P46_c3only_seed20260821_eval1024_E1.yaml \
    --model_path <val-best.pth> --split val --protocol all \
    --expected_clean_miou 68.0 \
    --out /drone_nas/.../analysis_logs/reliadino_missing_modality_20260917
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# 리포 루트를 path 에 추가해 val.py / semseg / tools 를 import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.baseline_failure import common  # noqa: E402

CLEAN_ID = "clean"


# ===========================================================================
# 프로토콜 계산 — 순수 함수(모델·데이터셋 무관, 스모크에서 직접 assert 한다)
# ===========================================================================
def enumerate_present_subsets(M):
    """결측 0~M−1개(=존재 1~M개) 조합을 present 인덱스 튜플로 전부 열거한다.

    전부-존재(결측 0)는 포함하고 전부-결측(존재 0)은 제외한다 → 2^M − 1 개
    (4모달이면 15). 반환 순서는 결측 수(n_missing) 오름차순, 그 안에서 인덱스 순.
    """
    all_idx = tuple(range(M))
    subsets = []
    for n_present in range(M, 0, -1):            # M(결측0) → 1(결측 M-1)
        for present in itertools.combinations(all_idx, n_present):
            subsets.append(present)
    return subsets


def missing_of(present, M):
    """present 인덱스 튜플 → 결측 인덱스 튜플."""
    ps = set(present)
    return tuple(i for i in range(M) if i not in ps)


def zero_fill(base_list, missing_idx):
    """정규화된 배치 리스트에서 결측 모달을 0 텐서로 교체한 **새 리스트**를 만든다.

    참조만 교체하므로 base_list 의 원본 텐서는 불변이다(벤치마크 `images[i].zero_()`
    를 in-place 아닌 비파괴 방식으로 재현 — 한 배치로 여러 조합을 돌리기 위함).
    """
    out = list(base_list)
    for i in missing_idx:
        out[i] = torch.zeros_like(base_list[i])
    return out


def rmm_mask(shape, r, generator, device):
    """픽셀×채널 독립 드롭 마스크(값 1=유지, 0=드롭). rand<r 인 원소를 드롭한다.

    CPU 생성기로 뽑아 device 로 옮겨 CPU/GPU 무관하게 재현 가능하게 한다.
    """
    keep = (torch.rand(shape, generator=generator) >= r).to(torch.float32)
    return keep.to(device)


def rmm_degrade(base_list, missing_idx, r, generator):
    """결측 모달을 완전 0 대신 비율 r 의 픽셀×채널 독립 드롭으로 열화한 새 리스트."""
    out = list(base_list)
    for i in missing_idx:
        keep = rmm_mask(base_list[i].shape, r, generator, base_list[i].device)
        out[i] = base_list[i] * keep
    return out


def sp_positions(B, H, W, d, generator, device):
    """채널 공통 salt-and-pepper 위치·극성. 반환 (mask, is_salt).

    mask: (B,1,H,W) float, 1 = 노이즈로 치환되는 위치(채널 공통).
    is_salt: (B,1,H,W) float, 1 = salt(max), 0 = pepper(min). mask==1 인 곳만 유효.
    """
    u = torch.rand((B, 1, H, W), generator=generator)
    mask = (u < d).to(torch.float32)
    salt = (torch.rand((B, 1, H, W), generator=generator) < 0.5).to(torch.float32)
    return mask.to(device), salt.to(device)


def sp_noise(x, d, generator):
    """정규화된 텐서 x(B,C,H,W)에 salt-and-pepper 를 적용한 (noised, mask) 반환.

    채널 공통 위치를 밀도 d 로 뽑아, 절반은 salt(그 이미지·모달의 최댓값), 절반은
    pepper(최솟값)로 치환한다. 정규화 공간의 이미지별 min/max 를 극값으로 쓴다
    (벤치마크의 [0,255] min/max 를 정규화 공간으로 옮긴 정의 — summary 에 기록).
    mask 는 (B,1,H,W) 로 실제 치환된 위치를 나타내며 스모크의 밀도 검증에 쓴다.
    """
    B, C, H, W = x.shape
    mask, salt = sp_positions(B, H, W, d, generator, x.device)
    lo = x.amin(dim=(1, 2, 3), keepdim=True)     # (B,1,1,1) 이미지·모달별 최솟값
    hi = x.amax(dim=(1, 2, 3), keepdim=True)
    noise_val = salt * hi + (1.0 - salt) * lo    # (B,1,H,W), 채널 공통
    noised = x * (1.0 - mask) + noise_val * mask
    return noised, mask


def gaussian_noise(x, std, generator):
    """정규화 공간 가법 Gaussian 노이즈(옵션). std 는 정규화 스케일 기준.

    반환 (noised, mask) — mask 는 전 픽셀 1(모든 픽셀에 노이즈)로 둔다(밀도 개념 없음).
    """
    noise = torch.empty_like(x).normal_(mean=0.0, std=std, generator=generator)
    mask = torch.ones((x.shape[0], 1, *x.shape[2:]), device=x.device)
    return x + noise, mask


def bernoulli_weight(n_missing, M, p):
    """조합(결측 n_missing개)의 Bernoulli 확률 p^k (1−p)^(M−k)."""
    return (p ** n_missing) * ((1.0 - p) ** (M - n_missing))


def bernoulli_expected(records, M, p):
    """records = [(n_missing, miou), ...] → Σ w·miou / Σ w.

    전부-결측 조합을 제외했으므로 열거된 조합의 총확률(1 − p^M 은 아니고 실제
    열거된 조합 합)로 정규화한다 — records 에 들어온 조합만으로 정규화한다.
    """
    num = 0.0
    den = 0.0
    for k, miou in records:
        w = bernoulli_weight(k, M, p)
        num += w * miou
        den += w
    return num / den if den > 0 else float("nan")


# ===========================================================================
# 케이스 구성
# ===========================================================================
class Case:
    """평가 단위 하나(고유 id + base 배치 리스트 → 주입된 리스트 변환)."""

    def __init__(self, cid, group, present_names, n_missing, apply_fn,
                 ratio=None, density=None):
        self.id = cid
        self.group = group                 # 'clean' | 'emm' | 'rmm' | 'nm'
        self.present_names = present_names
        self.n_missing = n_missing
        self.apply_fn = apply_fn           # base_list -> injected_list
        self.ratio = ratio                 # RMM r
        self.density = density             # NM d


def _present_names(present, modal_names):
    return "+".join(modal_names[i] for i in present)


def build_cases(protocol, M, modal_names, rmm_ratios, nm_density,
                gen_factory, nm_gaussian=False, nm_gaussian_std=0.2):
    """활성 프로토콜에 맞는 Case 목록을 만든다. clean 은 항상 1개 포함한다.

    gen_factory() 는 매 호출 새 torch.Generator(재현용 seed 설정 완료)를 돌려준다 —
    RMM/NM apply_fn 이 배치마다 이 생성기를 이어 써 run-to-run 재현된다.
    """
    all_present = tuple(range(M))
    cases = [Case(CLEAN_ID, "clean", _present_names(all_present, modal_names),
                  0, lambda base: list(base))]

    subsets = enumerate_present_subsets(M)
    non_clean = [s for s in subsets if len(s) < M]     # 전부-존재는 clean 이 담당

    if protocol in ("emm", "all"):
        for present in non_clean:
            miss = missing_of(present, M)
            cid = f"emm|present={_present_names(present, modal_names)}"
            cases.append(Case(
                cid, "emm", _present_names(present, modal_names), len(miss),
                (lambda base, m=miss: zero_fill(base, m))))

    if protocol in ("rmm", "all"):
        for r in rmm_ratios:
            gen = gen_factory()
            for present in non_clean:
                miss = missing_of(present, M)
                cid = f"rmm{r}|present={_present_names(present, modal_names)}"
                cases.append(Case(
                    cid, "rmm", _present_names(present, modal_names), len(miss),
                    (lambda base, m=miss, rr=r, g=gen: rmm_degrade(base, m, rr, g)),
                    ratio=r))

    if protocol in ("nm", "all"):
        for d in nm_density:
            gen = gen_factory()
            cid = f"nm{d}"
            if nm_gaussian:
                cases.append(Case(
                    cid, "nm", _present_names(all_present, modal_names), 0,
                    (lambda base, dd=d, g=gen, s=nm_gaussian_std:
                     [gaussian_noise(x, s, g)[0] for x in base]),
                    density=d))
            else:
                cases.append(Case(
                    cid, "nm", _present_names(all_present, modal_names), 0,
                    (lambda base, dd=d, g=gen: [sp_noise(x, dd, g)[0] for x in base]),
                    density=d))
    return cases


# ===========================================================================
# 평가 루프 (데이터셋·프레임워크 무관 — 스모크가 dummy 로더로 직접 호출)
# ===========================================================================
def evaluate(model, loader, cases, n_classes, ignore, device,
             unpad_fn, argmax_fn, limit=0):
    """한 번의 데이터 로딩 루프에서 모든 case 를 배치마다 순차 forward 한다.

    각 배치의 정규화된 텐서(base)를 case.apply_fn 으로 복제·주입해 forward → argmax →
    native 해상도 복원 → 혼동행렬 누적. 반환 = {case_id: hist(n,n)}.
    """
    hists = {c.id: np.zeros((n_classes, n_classes), dtype=np.int64) for c in cases}
    n_seen = 0
    with torch.no_grad():
        for images, labels, metas in tqdm(loader, desc="missing-modality"):
            base = [x.to(device) for x in images]
            for case in tqdm(cases, desc="cases", leave=False):
                imgs = case.apply_fn(base)
                output = model(imgs, multimask_output=True)
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
                    hists[case.id] += common.confusion_matrix(
                        pred_np, gt_np, n_classes, ignore)
            n_seen += len(metas)
            if limit and n_seen >= limit:
                break
    return hists


# ===========================================================================
# 출력 (CSV / summary)
# ===========================================================================
# 벤치마크 계열(arXiv 2503.18445), DELIVER val, MiT-B2급 참고 수치(상수).
BENCH_EMM_AVG = {
    "CMNeXt": 37.90, "GeminiFusion": 37.07, "StitchFusion": 41.98,
    "MAGIC": 44.97, "MAGIC++": 44.85, "EQUISeg": 48.22, "RobustSeg": 49.72,
}
BENCH_MAGICPP_E02 = 59.18                 # MAGIC++ E(p=0.2)
BENCH_RMM_MAGICPP = {0.25: 53.92, 0.5: 49.31, 0.75: 47.06}   # MAGIC++ RMM 평균


def _case_miou(hist):
    ious, miou = common.global_miou_from_cm(hist)
    return miou, ious


def _write_combo_csv(path, cases, hists, class_names):
    """조합별 mIoU + 25 클래스 IoU CSV. combo, present_modals, n_missing, mIoU, <25>."""
    import csv
    header = ["combo", "present_modals", "n_missing", "mIoU"] + list(class_names)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for c in cases:
            miou, ious = _case_miou(hists[c.id])
            w.writerow([c.id, c.present_names, c.n_missing,
                        f"{miou:.4f}"] + [f"{v:.4f}" for v in ious])


def aggregate_missing(cases_group, clean_case, hists, M, ps=(0.2, 0.1, 0.05)):
    """EMM/RMM 한 묶음(cases_group) + clean(k=0)의 단순평균·Bernoulli 기대값 집계.

    반환 {"avg": ..., "E(p=..)": ..., "n_combos": ...}.
    """
    records = []
    clean_miou, _ = _case_miou(hists[clean_case.id])
    records.append((0, clean_miou))       # k=0 (전부-존재) = clean
    for c in cases_group:
        miou, _ = _case_miou(hists[c.id])
        records.append((c.n_missing, miou))
    avg = float(np.mean([m for _, m in records]))
    out = {"avg": round(avg, 4), "n_combos": len(records)}
    for p in ps:
        out[f"E(p={p})"] = round(bernoulli_expected(records, M, p), 4)
    return out


def write_outputs(out_root, split, cases, hists, M, modal_names, class_names,
                  rmm_ratios, nm_density, protocol, presence_renorm, nm_gaussian,
                  nm_gaussian_std, model_path, cfg_path):
    """emm.csv / rmm_r*.csv / nm_d*.csv / summary.json / summary.md 를 쓴다."""
    base = Path(out_root) / split / "missing_modality"
    base.mkdir(parents=True, exist_ok=True)

    clean_case = next(c for c in cases if c.group == "clean")
    clean_miou, _ = _case_miou(hists[clean_case.id])

    summary = {
        "split": split,
        "model_path": str(model_path),
        "cfg": str(cfg_path),
        "modals": list(modal_names),
        "num_modalities": M,
        "protocol": protocol,
        "presence_renorm": bool(presence_renorm),
        "clean_mIoU": round(clean_miou, 4),
        "protocol_defs": {
            "EMM": "결측 모달을 정규화 후 0 으로 채움(zero-fill). 전부-결측 제외 → 2^M−1 조합.",
            "RMM": "결측 모달을 픽셀×채널 독립 rand<r 드롭(비율 r). 시드 고정 재현.",
            "NM": ("salt-and-pepper(채널 공통 위치, 밀도 d, 정규화 공간 이미지별 min/max)"
                   if not nm_gaussian else
                   f"gaussian additive noise(std={nm_gaussian_std}, 정규화 공간)"),
            "injection_point": "Normalize 이후 배치 텐서(벤치마크와 동일)",
            "scoring": "native-GT, 25클래스, 부재 클래스 IoU=0 포함(val.py metrics 규약)",
        },
    }

    # EMM
    if protocol in ("emm", "all"):
        emm_cases = [c for c in cases if c.group == "emm"]
        ordered = [clean_case] + emm_cases
        _write_combo_csv(base / "emm.csv", ordered, hists, class_names)
        summary["EMM"] = aggregate_missing(emm_cases, clean_case, hists, M)

    # RMM (r별)
    if protocol in ("rmm", "all"):
        summary["RMM"] = {}
        for r in rmm_ratios:
            r_cases = [c for c in cases if c.group == "rmm" and c.ratio == r]
            ordered = [clean_case] + r_cases
            _write_combo_csv(base / f"rmm_r{r}.csv", ordered, hists, class_names)
            summary["RMM"][str(r)] = aggregate_missing(r_cases, clean_case, hists, M)

    # NM (d별)
    if protocol in ("nm", "all"):
        summary["NM"] = {}
        nm_rows = []
        for d in nm_density:
            nm_case = next(c for c in cases if c.group == "nm" and c.density == d)
            miou, _ = _case_miou(hists[nm_case.id])
            summary["NM"][str(d)] = round(miou, 4)
            nm_rows.append(nm_case)
        _write_combo_csv(base / "nm.csv", nm_rows, hists, class_names)

    (base / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_md(base / "summary.md", summary, rmm_ratios, protocol)
    return base, summary


def _write_summary_md(path, summary, rmm_ratios, protocol):
    """벤치마크 비교표(우리 vs 계열 baseline). 수치는 배경의 상수 + 각주."""
    L = ["# 결측·열화 모달 강건성 — ReliaDINO vs 벤치마크 계열", ""]
    L.append(f"- split: `{summary['split']}` · modals: `{summary['modals']}` · "
             f"presence_renorm: `{summary['presence_renorm']}`")
    L.append(f"- ckpt: `{summary['model_path']}`")
    L.append(f"- clean mIoU (전부-존재): **{summary['clean_mIoU']}**")
    L.append("")
    L.append("> 각주: 비교 수치는 벤치마크 계열(arXiv 2503.18445), DELIVER **val**, "
             "MiT-B2급. 해상도·백본이 우리와 다를 수 있어 절대 비교가 아닌 위치 파악용이다.")
    L.append("")

    if "EMM" in summary:
        ours = summary["EMM"]["avg"]
        L.append("## EMM — 15조합 단순평균 mIoU")
        L.append("")
        L.append("| 모델 | EMM avg |")
        L.append("|------|---------|")
        L.append(f"| **ReliaDINO (ours)** | **{ours}** |")
        for name, v in BENCH_EMM_AVG.items():
            L.append(f"| {name} | {v} |")
        L.append("")
        L.append("### Bernoulli 기대값 E(p)")
        L.append("")
        L.append("| p | ReliaDINO E(p) | MAGIC++ 참고 |")
        L.append("|---|----------------|-------------|")
        for p in (0.2, 0.1, 0.05):
            ref = BENCH_MAGICPP_E02 if p == 0.2 else "—"
            L.append(f"| {p} | {summary['EMM'].get(f'E(p={p})')} | {ref} |")
        L.append("")

    if "RMM" in summary:
        L.append("## RMM — 드롭 비율 r별 15조합 평균 mIoU")
        L.append("")
        L.append("| r | ReliaDINO avg | MAGIC++ 참고 |")
        L.append("|---|---------------|-------------|")
        for r in rmm_ratios:
            ours = summary["RMM"].get(str(r), {}).get("avg", "—")
            ref = BENCH_RMM_MAGICPP.get(r, "—")
            L.append(f"| {r} | {ours} | {ref} |")
        L.append("")

    if "NM" in summary:
        L.append("## NM — 노이즈 밀도 d별 mIoU (4모달 전부 노이즈)")
        L.append("")
        L.append("| d | ReliaDINO mIoU |")
        L.append("|---|----------------|")
        for d, v in summary["NM"].items():
            L.append(f"| {d} | {v} |")
        L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", required=True, help="평가 config (val.py 와 동일)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="기본 val(벤치마크 계열과 비교). test 도 가능.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="all", choices=["emm", "rmm", "nm", "all"])
    ap.add_argument("--rmm_ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--nm_density", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--presence_renorm", action="store_true",
                    help="P44-V1 결정론적 presence 재정규화를 켠다(기본 off=벤치마크 그대로).")
    ap.add_argument("--nm_gaussian", action="store_true",
                    help="NM 을 salt-and-pepper 대신 가법 Gaussian 으로(기본 off).")
    ap.add_argument("--nm_gaussian_std", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="RMM/NM 재현 시드")
    ap.add_argument("--expected_clean_miou", type=float, default=None,
                    help="등록 clean mIoU. 주면 전부-존재 mIoU 가 ±--tol 밖일 때 exit≠0.")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="디버그용 앞 N 장만(0=전체)")
    ap.add_argument("--batch", type=int, default=1,
                    help="평가 배치 크기(기본 1). native-GT 복원은 이미지별이라 배치 무관. "
                         "메모리 허용 시 2~4 로 올려 15조합 순차 forward 비용을 줄인다.")
    ap.add_argument("--subset_every", type=int, default=1,
                    help="k>1 이면 데이터셋 순서상 k장마다 1장(결정론적 부분집합)만 평가. "
                         "스크린용; 벤치마크 비교 수치는 반드시 1(전체)로 낸다. summary 에 기록.")
    args = ap.parse_args()

    import val as valmod

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(cfg.get("DEVICE",
                                  "cuda" if torch.cuda.is_available() else "cpu"))
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
        print("[mm-eval] ⚠️ has_gt=False — GT 없는 split 은 채점 불가", flush=True)

    if args.subset_every > 1:
        # 결정론적 부분집합(k장마다 1장): 조건·장면 순서를 그대로 훑으므로 조건 분포가 대체로 보존된다.
        from torch.utils.data import Subset
        idx = list(range(0, len(dataset), args.subset_every))
        dataset = Subset(dataset, idx)
        print(f"[mm-eval] ⚠️ 부분집합 평가: {len(idx)} 장(every {args.subset_every}) — "
              f"벤치마크 비교용 수치가 아니다(스크린).", flush=True)
    loader = DataLoader(dataset, batch_size=max(1, args.batch), num_workers=4,
                        pin_memory=False, collate_fn=valmod._collate_fn)

    model = valmod.load_model(cfg, Path(args.model_path), device)
    model.eval()

    # presence 재정규화(P44-V1) — forward 는 self.p44_validity_renorm 플래그로만 켜진다.
    if args.presence_renorm:
        if hasattr(model, "p44_validity_renorm"):
            model.p44_validity_renorm = True
            print("[mm-eval] presence_renorm ON (P44-V1 결정론적 재정규화)")
        else:
            print("[mm-eval] ⚠️ 모델에 p44_validity_renorm 경로가 없어 presence_renorm "
                  "무시 — 벤치마크(off)로 진행")

    modal_names = list(dataset_cfg["MODALS"])
    M = len(modal_names)
    n_classes = base_dataset.n_classes
    ignore = base_dataset.ignore_label

    def gen_factory():
        g = torch.Generator()
        g.manual_seed(args.seed)
        return g

    cases = build_cases(args.protocol, M, modal_names, args.rmm_ratios,
                        args.nm_density, gen_factory,
                        nm_gaussian=args.nm_gaussian,
                        nm_gaussian_std=args.nm_gaussian_std)
    print(f"[mm-eval] cases={len(cases)} (protocol={args.protocol}, M={M}) "
          f"→ 배치마다 {len(cases)}회 forward")

    hists = evaluate(model, loader, cases, n_classes, ignore, device,
                     valmod._unpad_resize_to_orig, valmod._argmax_pred,
                     limit=args.limit)

    base, summary = write_outputs(
        args.out, mode, cases, hists, M, modal_names, list(common.CLASSES),
        args.rmm_ratios, args.nm_density, args.protocol, args.presence_renorm,
        args.nm_gaussian, args.nm_gaussian_std, args.model_path, args.cfg)
    if args.subset_every > 1 or args.batch != 1:
        # 부분집합·배치 정보를 summary 에 남긴다(부분집합 수치는 벤치마크 비교 불가 표시).
        summary["subset_every"] = int(args.subset_every)
        summary["batch"] = int(args.batch)
        summary["benchmark_comparable"] = bool(args.subset_every == 1 and not args.limit)
        (base / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[mm-eval] clean mIoU = {summary['clean_mIoU']}")
    if "EMM" in summary:
        print(f"[mm-eval] EMM avg = {summary['EMM']['avg']}  "
              f"E(0.2)={summary['EMM'].get('E(p=0.2)')}")
    if "RMM" in summary:
        for r in args.rmm_ratios:
            print(f"[mm-eval] RMM r={r} avg = {summary['RMM'][str(r)]['avg']}")
    if "NM" in summary:
        for d in args.nm_density:
            print(f"[mm-eval] NM d={d} mIoU = {summary['NM'][str(d)]}")
    print(f"[mm-eval] -> {base}")

    # 검증: 전부-존재(clean) mIoU 가 등록값과 ±tol 안이어야 한다(전처리·주입 위치 오류 감지).
    if args.expected_clean_miou is not None:
        delta = abs(summary["clean_mIoU"] - args.expected_clean_miou)
        status = "OK" if delta <= args.tol else "MISMATCH"
        print(f"[mm-eval] clean 등록수치 {args.expected_clean_miou} 대비 "
              f"Δ={delta:.3f} → {status}")
        if delta > args.tol:
            if args.limit:
                print("[mm-eval] ⚠️ --limit 부분 실행이라 exit 안 함(전수 재현 시 재검증)")
            else:
                print("[mm-eval] ⚠️ clean 재현 실패 — 전처리/주입 위치/ckpt 확인")
                sys.exit(2)


if __name__ == "__main__":
    main()
