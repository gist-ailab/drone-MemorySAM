#!/usr/bin/env python3
"""P55 step-1: oracle modality-selector headroom probe (무학습).

학습된 멀티모달 세그멘테이션 ckpt 하나에 대해, **완벽한** per-window 모달 선택기가
전 모달 융합(clean) 대비 얼마나 mIoU 를 더 얻을 수 있는지를 상한(oracle)으로 잰다.
이 상한(headroom)이 작으면, 학습형 modality gate 가 얻을 것이 없다는 반증 근거가 된다.

동작:
- tools/missing_modality_eval.py(mme) 의 EMM 케이스를 그대로 재사용한다. mme.evaluate 를
  evaluate_with_oracle 로 몽키패치하되 **시그니처·반환값(dict case_id->hist)은 동일**하므로
  mme.main() 의 정규 산출물(summary/csv)은 그대로 나온다.
- mm_eval_v2 와 동일하게 legal-v2 하네스로 몽키패치한다
  (val._unpad_resize_to_orig = legal_rescore_v2._unpad_resize_to_orig_v2).
- oracle 계산은 이미지별로 native 해상도 pred_np(=원 evaluate 와 동일)를 잠깐 모아,
  W×W 윈도우마다 GT 대비 정답 픽셀이 가장 많은 후보를 골라 조립한다. 이미지 하나 분량의
  후보 예측만 메모리에 두고 다음 이미지로 넘어가면 버린다.

후보 집합(candidate set):
  - emm  : clean + 모든 EMM 케이스(비어있지 않은 모달 부분집합, 나머지 0-fill).
  - drop1: clean + 정확히 한 모달만 결측인 EMM 케이스.
Oracle: win{W}(비겹침 W×W 윈도우별 최선 후보) + img(전체 이미지 = 단일 윈도우).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:          # mm_eval_v2 와 같게: `python tools/...` 직접 실행 대비
    sys.path.insert(0, str(_REPO))

# 몽키패치가 채워 넣는 모듈 전역 (mme.main 실행 중 evaluate_with_oracle 가 적재).
ORACLE_WINDOWS = [64, 256]
ORACLE_HISTS = {}          # (candset, oracle) -> 혼동행렬(n,n)
ORACLE_CHOICES = {}        # (candset, oracle) -> 후보별 선택 윈도우 수
ORACLE_CAND_NAMES = {}     # candset -> [present_names, ...] (후보 순서)
ORACLE_CLEAN_HIST = None   # clean 케이스 혼동행렬(headroom 기준선)


# ===========================================================================
# 순수 조립 로직 (torch 무관 — 스모크가 직접 assert)
# ===========================================================================
def oracle_assemble(preds, gt, win, ignore):
    """윈도우별 최선 후보를 골라 oracle 예측을 조립한다.

    preds: [np.ndarray(H,W,int), ...] 후보 예측(index 0 = clean).
    win:   윈도우 한 변(px). None 이면 전체 이미지 = 단일 윈도우.
    각 윈도우에서 gt!=ignore 인 픽셀 중 정답이 가장 많은 후보를 고른다. 동점이면 낮은
    index(=clean 우선, 그다음 후보 순서). 반환 (oracle_pred(H,W), choice_counts[len]).
    """
    H, W = gt.shape
    n = len(preds)
    out = np.empty((H, W), dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    valid = gt != ignore
    ys = [(0, H)] if win is None else [(y, min(y + win, H)) for y in range(0, H, win)]
    xs = [(0, W)] if win is None else [(x, min(x + win, W)) for x in range(0, W, win)]
    for y0, y1 in ys:
        for x0, x1 in xs:
            gtw = gt[y0:y1, x0:x1]
            vw = valid[y0:y1, x0:x1]
            best_i, best_correct = 0, -1
            for i, p in enumerate(preds):
                correct = int(np.count_nonzero((p[y0:y1, x0:x1] == gtw) & vw))
                if correct > best_correct:      # strict '>' → 동점 시 낮은 index 유지
                    best_correct, best_i = correct, i
            out[y0:y1, x0:x1] = preds[best_i][y0:y1, x0:x1]
            counts[best_i] += 1
    return out, counts


# ===========================================================================
# mme.evaluate 대체 — 시그니처·반환값 동일 + oracle 부산물 누적
# ===========================================================================
def evaluate_with_oracle(model, loader, cases, n_classes, ignore, device,
                         unpad_fn, argmax_fn, limit=0):
    import torch
    from tqdm import tqdm
    from tools.baseline_failure import common
    global ORACLE_CLEAN_HIST

    hists = {c.id: np.zeros((n_classes, n_classes), dtype=np.int64) for c in cases}

    clean_case = next(c for c in cases if c.group == "clean")
    emm_cases = [c for c in cases if c.group == "emm"]
    candsets = {}
    if emm_cases:
        candsets["emm"] = [clean_case] + emm_cases
        drop1 = [c for c in emm_cases if c.n_missing == 1]
        if drop1:
            candsets["drop1"] = [clean_case] + drop1
    oracles = [(f"win{w}", w) for w in ORACLE_WINDOWS] + [("img", None)]
    for cs_name, cs_cases in candsets.items():
        ORACLE_CAND_NAMES[cs_name] = [c.present_names for c in cs_cases]
        for orc_name, _ in oracles:
            ORACLE_HISTS[(cs_name, orc_name)] = np.zeros((n_classes, n_classes), np.int64)
            ORACLE_CHOICES[(cs_name, orc_name)] = np.zeros(len(cs_cases), np.int64)
    candidate_ids = {clean_case.id} | {c.id for c in emm_cases}

    n_seen = 0
    with torch.no_grad():
        for images, labels, metas in tqdm(loader, desc="oracle-headroom"):
            base = [x.to(device) for x in images]
            cand_preds, gts = {}, {}     # b -> {case_id: pred int16} / b -> gt int64
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
                    pred_resized = unpad_fn(pred_b, meta["orig_h"], meta["orig_w"],
                                            model_size=int(pred_b.shape[0]))
                    pred_np = pred_resized.cpu().numpy().astype(np.int64)
                    gt_np = orig_label.cpu().numpy().astype(np.int64)
                    hists[case.id] += common.confusion_matrix(pred_np, gt_np, n_classes, ignore)
                    if case.id in candidate_ids:
                        cand_preds.setdefault(b, {})[case.id] = pred_np.astype(np.int16)
                        gts[b] = gt_np
            # 이미지별 oracle 조립 (배치 끝나면 cand_preds 폐기 → 한 배치 분량만 상주)
            for b, per in cand_preds.items():
                gt_np = gts[b]
                for cs_name, cs_cases in candsets.items():
                    preds_list = [per[c.id].astype(np.int64) for c in cs_cases]
                    for orc_name, win in oracles:
                        opred, counts = oracle_assemble(preds_list, gt_np, win, ignore)
                        key = (cs_name, orc_name)
                        ORACLE_HISTS[key] += common.confusion_matrix(opred, gt_np, n_classes, ignore)
                        ORACLE_CHOICES[key] += counts
            n_seen += len(metas)
            if limit and n_seen >= limit:
                break
    ORACLE_CLEAN_HIST = hists[clean_case.id]
    return hists


# ===========================================================================
# oracle 요약 출력
# ===========================================================================
def _write_oracle_outputs(out_root, split, windows):
    from tools import missing_modality_eval as mme
    base = Path(out_root) / split / "oracle_headroom"
    base.mkdir(parents=True, exist_ok=True)
    clean_miou = round(mme._case_miou(ORACLE_CLEAN_HIST)[0], 4)
    orc_order = [f"win{w}" for w in windows] + ["img"]

    result = {"split": split, "windows": list(windows),
              "clean_mIoU": clean_miou, "candidate_sets": {}}
    for cs_name, names in ORACLE_CAND_NAMES.items():
        entry = {"candidates": names, "oracles": {}}
        for orc_name in orc_order:
            key = (cs_name, orc_name)
            if key not in ORACLE_HISTS:
                continue
            miou = round(mme._case_miou(ORACLE_HISTS[key])[0], 4)
            counts = ORACLE_CHOICES[key]
            total = int(counts.sum())
            frac = {names[i]: (round(float(counts[i]) / total, 4) if total else 0.0)
                    for i in range(len(names))}
            entry["oracles"][orc_name] = {
                "mIoU": miou, "clean_mIoU": clean_miou,
                "headroom": round(miou - clean_miou, 4), "choice_fraction": frac}
        result["candidate_sets"][cs_name] = entry
    (base / "oracle_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Oracle modality-selector headroom (P55 step-1)", "",
         f"- split: `{split}` · clean mIoU: **{clean_miou}**",
         "- headroom = oracle mIoU − clean mIoU. 작을수록 학습형 gate 여지가 적다.", ""]
    for cs_name, entry in result["candidate_sets"].items():
        L += [f"## candidate set `{cs_name}` ({len(entry['candidates'])} 후보)", "",
              "| oracle | mIoU | clean | headroom |", "|--------|------|-------|----------|"]
        for orc_name, o in entry["oracles"].items():
            L.append(f"| {orc_name} | {o['mIoU']} | {o['clean_mIoU']} | {o['headroom']} |")
        L.append("")
    (base / "oracle_summary.md").write_text("\n".join(L), encoding="utf-8")
    return base, result


def main():
    global ORACLE_WINDOWS
    # --windows 만 먼저 떼어내고 나머지 인자는 mme.main 에 그대로 넘긴다.
    strip = argparse.ArgumentParser(add_help=False)
    strip.add_argument("--windows", type=int, nargs="+", default=[64, 256])
    ns_w, remaining = strip.parse_known_args()
    ORACLE_WINDOWS = ns_w.windows
    sys.argv = [sys.argv[0]] + remaining
    info = argparse.ArgumentParser(add_help=False)   # 읽기만(argv 불변)
    info.add_argument("--out")
    info.add_argument("--split", default="val")
    ns_i, _ = info.parse_known_args()

    import val
    from tools import legal_rescore_v2
    from tools import missing_modality_eval as mme
    val._unpad_resize_to_orig = legal_rescore_v2._unpad_resize_to_orig_v2
    print(f"[oracle] legal-v2 하네스({legal_rescore_v2.RESAMPLE_MODE}) · "
          f"windows={ORACLE_WINDOWS}", flush=True)
    mme.evaluate = evaluate_with_oracle
    mme.main()

    base, result = _write_oracle_outputs(ns_i.out, ns_i.split, ORACLE_WINDOWS)
    for cs, e in result["candidate_sets"].items():
        for orc, o in e["oracles"].items():
            print(f"[oracle] {cs}/{orc}: mIoU={o['mIoU']} "
                  f"clean={o['clean_mIoU']} headroom={o['headroom']}")
    print(f"[oracle] -> {base}")


if __name__ == "__main__":
    main()
