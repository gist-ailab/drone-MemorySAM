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
  - null : clean + K 개의 null 케이스(전 모달 입력에 가법 Gaussian 만 더한 것). 모달 정보가
           전혀 없는 후보들이므로, 여기서 나오는 headroom 은 순수 선택 편향(GT 로 최댓값을
           고르는 데서 오는 부풀림)이다. 대조군.
  - null_n{n}: clean + 앞 n 개 null 케이스. emm/drop1 과 후보 '개수'를 맞춘 대조군
           (net_headroom_vs_null 계산용).
Oracle: win{W}(비겹침 W×W 윈도우별 최선 후보) + img(전체 이미지 = 단일 윈도우).

--null_control K (>0) 를 주면 K 개 null 케이스를 만들어 같은 forward pass 에서 함께 평가한다.
각 null 케이스는 전 모달에 N(0, --null_sigma) 를 더하며(정규화 공간), 케이스별 고정 시드
(seed=args.seed*1000+i)로 재현된다. summary 에 emm/drop1 의 headroom 에서 개수 맞춘 null
대조군 headroom 을 뺀 net_headroom_vs_null 을 함께 기록한다(순수 선택 편향 차감치).

--emm_max_missing N (>0) 을 주면 n_missing>N 인 EMM 케이스를 평가 전에 케이스 목록에서
뺀다(forward 자체를 안 하므로 시간 절약). clean·null 케이스는 유지. N=1·4모달이면
케이스 = clean 1 + EMM 4(단일 모달 결측, drop1).

--by_condition 를 주면 DELIVER 경로의 condition(cloud/fog/night/rain/sun)·sensor-failure
case(motionblur/overexposure/underexposure/lidarjitter/eventlowres/none)별로 케이스·
(candset, oracle) 혼동행렬과 선택 카운트를 그룹에도 누적해 oracle_summary.json 의
by_condition + oracle_by_condition.md 로 낸다. 이미지 하나는 cond:<조건>, case:<케이스>
두 그룹에 동시 가입한다. 그룹 mIoU 는 nan-aware(그룹 GT 에 없는 클래스 제외 평균) —
전역 요약의 부재 클래스 IoU=0 포함 규약과 다르다(단위 % 는 동일).
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
ORACLE_DISAGREE = {}       # candset -> [clean 과 다른 픽셀 수 합, 비교 픽셀 수 합] (후보 다양성; null 대조군 sigma 가 적절한지 판단용)
ORACLE_CAND_NAMES = {}     # candset -> [present_names, ...] (후보 순서)
ORACLE_CLEAN_HIST = None   # clean 케이스 혼동행렬(headroom 기준선)
ORACLE_CASES = None        # evaluate 에 들어온 케이스 목록(by_condition 요약용)
ORACLE_BY_COND_ON = False  # --by_condition 플래그(evaluate 의 그룹 누적 분기)
ORACLE_BY_COND = {}        # group -> new_group_state() (cond:<c> / case:<k> 그룹 누적)


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
# by-condition 그룹 누적·요약 (순수 로직 — torch 무관, 스모크가 직접 assert)
# ===========================================================================
def condition_group_keys(img_path):
    """DELIVER 이미지 경로 → 그룹 키 리스트 ['cond:<조건>', 'case:<센서케이스>'].

    파싱은 tools.baseline_failure.common.parse_condition_case(경로의 조건 폴더 +
    scene 접미사 매칭)를 그대로 쓴다. 비-DELIVER 경로는 ('unknown','none') 으로
    떨어져 그룹이 몰리므로 by_condition 은 DELIVER 평가에만 쓰는 게 맞다.
    """
    from tools.baseline_failure import common
    cond, case = common.parse_condition_case(img_path)
    return [f"cond:{cond}", f"case:{case}"]


def new_group_state():
    """그룹 하나분 누적 상태. 행렬은 n_classes² int64 라 그룹 수×케이스 수여도 가볍다."""
    return {"n_images": 0, "case_hists": {}, "oracle_hists": {}, "oracle_choices": {}}


def accum_case_hist(state, case_id, pred, gt, n_classes, ignore):
    """이미지 한 장의 케이스 혼동행렬을 그룹 상태에 누적(채점 규약 = common 재사용)."""
    from tools.baseline_failure import common
    h = state["case_hists"].get(case_id)
    if h is None:
        h = state["case_hists"][case_id] = np.zeros((n_classes, n_classes), np.int64)
    h += common.confusion_matrix(pred, gt, n_classes, ignore)


def accum_oracle_hist(state, cs_name, orc_name, n_cands, opred, counts, gt,
                      n_classes, ignore):
    """이미지 한 장의 (candset, oracle) 혼동행렬·선택 카운트를 그룹 상태에 누적."""
    from tools.baseline_failure import common
    key = (cs_name, orc_name)
    h = state["oracle_hists"].get(key)
    if h is None:
        h = state["oracle_hists"][key] = np.zeros((n_classes, n_classes), np.int64)
    h += common.confusion_matrix(opred, gt, n_classes, ignore)
    c = state["oracle_choices"].get(key)
    if c is None:
        c = state["oracle_choices"][key] = np.zeros(n_cands, np.int64)
    c += counts


def group_miou_from_cm(cm):
    """그룹용 nan-aware mIoU(%, 부재 클래스 제외 평균).

    mme._case_miou → common.global_miou_from_cm 는 부재 클래스(union=0) IoU 를 0 으로
    넣어 전 클래스 평균에 포함한다(전 데이터셋 규약). 그룹(예: cond:night)에서는 GT 에
    한 번도 없는 클래스가 많아 0 포함 평균이 왜곡되므로, common.per_image_iou 의 NaN
    규약(GT 에 없는 클래스 = 평균 제외)으로 nanmean 한다.
    """
    from tools.baseline_failure import common
    return common.nanmean(common.per_image_iou(cm)) * 100.0


def build_by_condition_summary(by_cond, cases, cand_names, orc_order):
    """그룹 누적 상태 → oracle_summary.json 의 by_condition dict (순수 로직).

    net_headroom_vs_null 은 전역과 같은 개수맞춤 규칙(emm/drop1 → null_n{후보수−1})을
    그룹 혼동행렬에 그대로 적용한다. per_case_mIoU 키 = 케이스 present_names.
    """
    case_by_id = {c.id: c for c in cases}
    clean_case = next((c for c in cases if c.group == "clean"), None)
    out = {}
    for gname in sorted(by_cond):
        st = by_cond[gname]
        if clean_case is None or clean_case.id not in st["case_hists"]:
            continue
        clean_miou = group_miou_from_cm(st["case_hists"][clean_case.id])
        per_case = {}
        for cid, h in st["case_hists"].items():
            c = case_by_id.get(cid)
            # per-case 표는 clean/emm/null 만 — rmm/nm 는 전역 summary·csv 담당.
            if c is None or c.group not in ("clean", "emm", "null"):
                continue
            m = group_miou_from_cm(h)
            per_case[c.present_names] = {"mIoU": round(m, 4),
                                         "delta_vs_clean": round(m - clean_miou, 4)}
        css = {}
        for cs_name, names in cand_names.items():
            entry = {"oracles": {}}
            for orc_name in orc_order:
                key = (cs_name, orc_name)
                if key not in st["oracle_hists"]:
                    continue
                m = group_miou_from_cm(st["oracle_hists"][key])
                counts = st["oracle_choices"][key]
                total = int(counts.sum())
                frac = {names[i]: (round(float(counts[i]) / total, 4) if total else 0.0)
                        for i in range(len(names))}
                entry["oracles"][orc_name] = {
                    "mIoU": round(m, 4), "clean_mIoU": round(clean_miou, 4),
                    "headroom": round(m - clean_miou, 4), "choice_fraction": frac}
            css[cs_name] = entry
        for cs_name in ("emm", "drop1"):
            src = css.get(cs_name)
            if src is None or cs_name not in cand_names:
                continue
            matched_name = f"null_n{len(cand_names[cs_name]) - 1}"
            ref = css.get(matched_name)
            if ref is None:
                continue
            for orc_name, o in src["oracles"].items():
                if orc_name in ref["oracles"]:
                    o["net_headroom_vs_null"] = round(
                        o["headroom"] - ref["oracles"][orc_name]["headroom"], 4)
            src["null_control"] = matched_name
        out[gname] = {"n_images": int(st["n_images"]),
                      "clean_mIoU": round(clean_miou, 4),
                      "per_case_mIoU": per_case, "candidate_sets": css}
    return out


# ===========================================================================
# 케이스 목록 후처리 — EMM 결측 개수 상한 (평가 전 필터)
# ===========================================================================
def filter_emm_max_missing(cases, max_missing):
    """n_missing > max_missing 인 EMM 케이스를 뺀 **새 리스트**.

    필터 대상은 EMM 만 — clean·null·rmm·nm 케이스는 그대로 둔다. max_missing 이
    0/음수면 필터 없음(--emm_max_missing 기본값 0 = 전체 EMM 열거).
    """
    if not max_missing or max_missing <= 0:
        return list(cases)
    return [c for c in cases
            if c.group != "emm" or c.n_missing <= max_missing]


# ===========================================================================
# null 대조군 케이스 (선택 편향 측정용) — build_cases 결과에 덧붙인다
# ===========================================================================
def _make_null_cases(K, sigma, base_seed):
    """전 모달에 가법 Gaussian 만 더하는 null 케이스 K 개를 만든다.

    각 케이스는 결측 없이(FULL 입력) 전 모달에 N(0, sigma) 노이즈를 더하며, 케이스별
    고정 시드(base_seed*1000 + i)로 재현된다. mme.gaussian_noise 를 재사용한다
    (반환은 (noised, mask) 튜플이므로 [0] 만 취한다). 생성기는 입력 텐서의 device 에
    올려 CPU/GPU 무관하게 동작하며, 배치를 가로질러 스트림을 이어 써 run-to-run 재현된다.
    """
    import torch
    from tools import missing_modality_eval as mme

    cases = []
    for i in range(K):
        seed_i = base_seed * 1000 + i

        def apply(base, _seed=seed_i, _state={}, _s=sigma):
            g = _state.get("gen")
            if g is None:
                g = torch.Generator(device=base[0].device)
                g.manual_seed(_seed)
                _state["gen"] = g
            return [mme.gaussian_noise(x, _s, g)[0] for x in base]

        cases.append(mme.Case(f"null{i}", "null", f"null{i}", 0, apply))
    return cases


# ===========================================================================
# mme.evaluate 대체 — 시그니처·반환값 동일 + oracle 부산물 누적
# ===========================================================================
def evaluate_with_oracle(model, loader, cases, n_classes, ignore, device,
                         unpad_fn, argmax_fn, limit=0):
    import torch
    from tqdm import tqdm
    from tools.baseline_failure import common
    global ORACLE_CLEAN_HIST, ORACLE_CASES
    ORACLE_CASES = cases

    hists = {c.id: np.zeros((n_classes, n_classes), dtype=np.int64) for c in cases}

    clean_case = next(c for c in cases if c.group == "clean")
    emm_cases = [c for c in cases if c.group == "emm"]
    null_cases = [c for c in cases if c.group == "null"]
    candsets = {}
    if emm_cases:
        candsets["emm"] = [clean_case] + emm_cases
        drop1 = [c for c in emm_cases if c.n_missing == 1]
        if drop1:
            candsets["drop1"] = [clean_case] + drop1
    if null_cases:
        # 전체 null 후보군(clean + K null) + emm/drop1 과 후보 개수를 맞춘 대조군.
        candsets["null"] = [clean_case] + null_cases
        matched = []
        if emm_cases:
            matched.append(len(emm_cases))            # = len(candsets["emm"]) - 1
        if emm_cases and drop1:
            matched.append(len(drop1))                # = len(candsets["drop1"]) - 1
        for n in matched:
            k = min(len(null_cases), n)
            if k < n:
                print(f"[oracle] ⚠️ null_control K={len(null_cases)} < 개수맞춤에 필요한 "
                      f"{n} — null_n{n} 은 {k} 개만으로 계산(개수 불일치)", flush=True)
            candsets[f"null_n{n}"] = [clean_case] + null_cases[:k]
    oracles = [(f"win{w}", w) for w in ORACLE_WINDOWS] + [("img", None)]
    for cs_name, cs_cases in candsets.items():
        ORACLE_CAND_NAMES[cs_name] = [c.present_names for c in cs_cases]
        for orc_name, _ in oracles:
            ORACLE_HISTS[(cs_name, orc_name)] = np.zeros((n_classes, n_classes), np.int64)
            ORACLE_CHOICES[(cs_name, orc_name)] = np.zeros(len(cs_cases), np.int64)
    candidate_ids = ({clean_case.id} | {c.id for c in emm_cases}
                     | {c.id for c in null_cases})

    n_seen = 0
    with torch.no_grad():
        for images, labels, metas in tqdm(loader, desc="oracle-headroom"):
            base = [x.to(device) for x in images]
            # by_condition: 이 배치의 이미지별 그룹 키. GT 없는 이미지는 그룹에도 안 넣는다.
            batch_groups = {}
            if ORACLE_BY_COND_ON:
                for b, meta in enumerate(metas):
                    if meta.get("orig_label") is None:
                        continue
                    img_path = (meta.get("paths") or {}).get("img")
                    if not img_path:
                        continue
                    keys = condition_group_keys(img_path)
                    batch_groups[b] = keys
                    for g in keys:
                        ORACLE_BY_COND.setdefault(
                            g, new_group_state())["n_images"] += 1
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
                        # by_condition: 케이스 혼동행렬을 이미지의 그룹에도 누적.
                        if ORACLE_BY_COND_ON:
                            for g in batch_groups.get(b, ()):
                                accum_case_hist(ORACLE_BY_COND[g], case.id,
                                                pred_np, gt_np, n_classes, ignore)
            # 이미지별 oracle 조립 (배치 끝나면 cand_preds 폐기 → 한 배치 분량만 상주)
            for b, per in cand_preds.items():
                gt_np = gts[b]
                gkeys = batch_groups.get(b, ()) if ORACLE_BY_COND_ON else ()
                for cs_name, cs_cases in candsets.items():
                    preds_list = [per[c.id].astype(np.int64) for c in cs_cases]
                    d = ORACLE_DISAGREE.setdefault(cs_name, [0, 0])
                    for pp in preds_list[1:]:
                        d[0] += int(np.count_nonzero(pp != preds_list[0]))
                        d[1] += int(pp.size)
                    for orc_name, win in oracles:
                        opred, counts = oracle_assemble(preds_list, gt_np, win, ignore)
                        key = (cs_name, orc_name)
                        ORACLE_HISTS[key] += common.confusion_matrix(opred, gt_np, n_classes, ignore)
                        ORACLE_CHOICES[key] += counts
                        # by_condition: oracle 혼동행렬·선택 카운트도 그룹에 누적.
                        for g in gkeys:
                            accum_oracle_hist(ORACLE_BY_COND[g], cs_name, orc_name,
                                              len(cs_cases), opred, counts,
                                              gt_np, n_classes, ignore)
            n_seen += len(metas)
            if limit and n_seen >= limit:
                break
    ORACLE_CLEAN_HIST = hists[clean_case.id]
    return hists


# ===========================================================================
# oracle 요약 출력
# ===========================================================================
def _short_cand_label(modals, name):
    """후보 present_names → 표시용 짧은 라벨. 전체에서 한 모달만 빠지면 '−<모달>'."""
    if not modals:
        return name
    present = name.split("+")
    if len(present) != len(modals) - 1:
        return name
    missing = [m for m in modals if m not in present]
    return f"−{missing[0]}" if len(missing) == 1 else name


def _write_by_condition_md(path, by_cond, cases, cand_names, orc_order):
    """oracle_by_condition.md — 그룹(조건/센서케이스)별 한눈표.

    열: n_images · clean mIoU · 단일 모달 결측(drop1) 케이스의 clean 대비 Δ(mIoU) ·
    headroom/net_vs_null(기본 candset=drop1, 없으면 emm·첫 candset / oracle=win64,
    없으면 첫 win*) · 가장 많이 선택된 non-clean 후보(선택률).
    """
    clean_case = next((c for c in cases if c.group == "clean"), None)
    modals = clean_case.present_names.split("+") if clean_case else []
    drop_cols = []                      # (열 라벨 '−<모달>', present_names, 모달 인덱스)
    for c in cases:
        if c.group != "emm" or c.n_missing != 1:
            continue
        missing = [m for m in modals if m not in c.present_names.split("+")]
        if len(missing) == 1:
            drop_cols.append((f"−{missing[0]}", c.present_names,
                              modals.index(missing[0])))
    drop_cols.sort(key=lambda t: t[2])   # 모달 순서(−img, −depth, ...)

    cs_pref = next((cs for cs in ("drop1", "emm") if cs in cand_names),
                   next(iter(cand_names), None))
    orc_pref = ("win64" if "win64" in orc_order
                else next((o for o in orc_order if o.startswith("win")),
                          orc_order[0] if orc_order else None))

    L = ["# Oracle headroom by condition / sensor-failure case", "",
         "- 그룹 키 = `cond:<cloud|fog|night|rain|sun>` · "
         "`case:<motionblur|overexposure|underexposure|lidarjitter|eventlowres|none>` "
         "(DELIVER 경로에서 파싱, 이미지 하나는 두 그룹에 동시 가입).",
         "- 그룹 mIoU 는 nan-aware(그룹 GT 에 없는 클래스 제외 평균) — 전역 요약의 "
         "부재 클래스 IoU=0 포함 규약과 다르다(단위 % 는 동일).",
         f"- headroom/net 열 = `{cs_pref}/{orc_pref}` (candset 은 drop1 우선, 없으면 emm).",
         ""]
    cols = (["group", "n_images", "clean mIoU"]
            + [f"Δ{lab}" for lab, _, _ in drop_cols]
            + ["headroom", "net_vs_null", "top non-clean (frac)"])
    L.append("| " + " | ".join(cols) + " |")
    L.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for gname in sorted(by_cond):
        e = by_cond[gname]
        row = [gname, str(e["n_images"]), str(e["clean_mIoU"])]
        for _, pname, _ in drop_cols:
            row.append(str(e["per_case_mIoU"].get(pname, {}).get(
                "delta_vs_clean", "—")))
        o = None
        if cs_pref:
            o = e["candidate_sets"].get(cs_pref, {}).get("oracles", {}).get(orc_pref)
        if o is None:
            row += ["—", "—", "—"]
        else:
            row.append(str(o["headroom"]))
            row.append(str(o.get("net_headroom_vs_null", "—")))
            frac = o["choice_fraction"]
            non_clean = [(frac.get(n, 0.0), n) for n in cand_names.get(cs_pref, [])[1:]]
            if non_clean and max(non_clean)[0] > 0:
                _, top = max(non_clean)
                row.append(f"{_short_cand_label(modals, top)} ({frac.get(top, 0.0)})")
            else:
                row.append("—")
        L.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(L), encoding="utf-8")


def _write_oracle_outputs(out_root, split, windows, by_cond=None, cases=None):
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
        dd = ORACLE_DISAGREE.get(cs_name)
        entry["disagree_vs_clean"] = round(dd[0] / dd[1], 5) if dd and dd[1] else None
        result["candidate_sets"][cs_name] = entry

    # net_headroom_vs_null = headroom(real) − headroom(개수맞춤 null 대조군).
    # emm/drop1 은 각각 후보 개수가 같은 null_n{cands-1} 을 대조군으로 쓴다.
    for cs_name in ("emm", "drop1"):
        src = result["candidate_sets"].get(cs_name)
        if src is None:
            continue
        matched_name = f"null_n{len(src['candidates']) - 1}"
        ref = result["candidate_sets"].get(matched_name)
        if ref is None:
            continue
        for orc_name, o in src["oracles"].items():
            if orc_name in ref["oracles"]:
                o["net_headroom_vs_null"] = round(
                    o["headroom"] - ref["oracles"][orc_name]["headroom"], 4)
        src["null_control"] = matched_name

    if by_cond and cases:
        result["by_condition"] = build_by_condition_summary(
            by_cond, cases, ORACLE_CAND_NAMES, orc_order)
        _write_by_condition_md(base / "oracle_by_condition.md",
                               result["by_condition"], cases,
                               ORACLE_CAND_NAMES, orc_order)
    elif by_cond is not None:
        print("[oracle] ⚠️ by_condition: 누적된 그룹이 없다(meta['paths']['img'] 부재?) "
              "— by_condition 생략", flush=True)

    (base / "oracle_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Oracle modality-selector headroom (P55 step-1)", "",
         f"- split: `{split}` · clean mIoU: **{clean_miou}**",
         "- headroom = oracle mIoU − clean mIoU. 작을수록 학습형 gate 여지가 적다.",
         "- net_headroom_vs_null = headroom − 개수맞춤 null 대조군 headroom "
         "(순수 선택 편향 차감치).", ""]
    for cs_name, entry in result["candidate_sets"].items():
        head = f"## candidate set `{cs_name}` ({len(entry['candidates'])} 후보)"
        if entry.get("null_control"):
            head += f" · null 대조군 = `{entry['null_control']}`"
        L += [head, "",
              "| oracle | mIoU | clean | headroom | net_vs_null |",
              "|--------|------|-------|----------|-------------|"]
        for orc_name, o in entry["oracles"].items():
            net = o.get("net_headroom_vs_null", "—")
            L.append(f"| {orc_name} | {o['mIoU']} | {o['clean_mIoU']} | "
                     f"{o['headroom']} | {net} |")
        L.append("")
    (base / "oracle_summary.md").write_text("\n".join(L), encoding="utf-8")
    return base, result


def main():
    global ORACLE_WINDOWS, ORACLE_BY_COND_ON
    # --windows / null 대조군 인자만 먼저 떼어내고 나머지는 mme.main 에 그대로 넘긴다.
    strip = argparse.ArgumentParser(add_help=False)
    strip.add_argument("--windows", type=int, nargs="+", default=[64, 256])
    strip.add_argument("--null_control", type=int, default=0,
                       help="null 대조군 케이스 수 K(>0 이면 선택 편향 대조군 활성).")
    strip.add_argument("--null_sigma", type=float, default=0.05,
                       help="null 케이스 가법 Gaussian std(정규화 공간).")
    strip.add_argument("--emm_max_missing", type=int, default=0,
                       help=">0 이면 n_missing>N 인 EMM 케이스를 평가 전에 제외"
                            "(forward 안 함). N=1·4모달 → clean 1 + EMM 4(drop1).")
    strip.add_argument("--by_condition", action="store_true",
                       help="DELIVER 경로의 condition·sensor-failure case 별 그룹 누적 → "
                            "oracle_summary.json by_condition + oracle_by_condition.md.")
    ns_w, remaining = strip.parse_known_args()
    ORACLE_WINDOWS = ns_w.windows
    ORACLE_BY_COND_ON = ns_w.by_condition
    sys.argv = [sys.argv[0]] + remaining
    info = argparse.ArgumentParser(add_help=False)   # 읽기만(argv 불변)
    info.add_argument("--out")
    info.add_argument("--split", default="val")
    info.add_argument("--seed", type=int, default=0)   # mme 시드(null 케이스 시드 기준)
    ns_i, _ = info.parse_known_args()

    import val
    from tools import legal_rescore_v2
    from tools import missing_modality_eval as mme
    val._unpad_resize_to_orig = legal_rescore_v2._unpad_resize_to_orig_v2
    print(f"[oracle] legal-v2 하네스({legal_rescore_v2.RESAMPLE_MODE}) · "
          f"windows={ORACLE_WINDOWS}", flush=True)

    K = ns_w.null_control
    EMM_MAX = ns_w.emm_max_missing
    if K > 0 or EMM_MAX > 0:
        # mm_eval_v2 와 같은 방식으로 build_cases 를 감싼다: EMM 필터 → null 덧붙임.
        orig_build_cases = mme.build_cases

        def build_cases_patched(*a, **k):
            cases = orig_build_cases(*a, **k)
            if EMM_MAX > 0:
                n_before = sum(1 for c in cases if c.group == "emm")
                cases = filter_emm_max_missing(cases, EMM_MAX)
                n_after = sum(1 for c in cases if c.group == "emm")
                print(f"[oracle] emm_max_missing={EMM_MAX}: EMM {n_before}→{n_after} "
                      f"(n_missing>{EMM_MAX} 제외, forward 안 함)", flush=True)
            if K > 0:
                cases.extend(_make_null_cases(K, ns_w.null_sigma, ns_i.seed))
            return cases

        mme.build_cases = build_cases_patched
    if K > 0:
        print(f"[oracle] null-control: K={K} · sigma={ns_w.null_sigma} · "
              f"seed={ns_i.seed}*1000+i", flush=True)
    if ns_w.by_condition:
        print("[oracle] by_condition: cond:<condition>/case:<case> 그룹 누적 ON",
              flush=True)

    mme.evaluate = evaluate_with_oracle
    mme.main()

    base, result = _write_oracle_outputs(
        ns_i.out, ns_i.split, ORACLE_WINDOWS,
        by_cond=(ORACLE_BY_COND if ns_w.by_condition else None),
        cases=ORACLE_CASES)
    for cs, e in result["candidate_sets"].items():
        for orc, o in e["oracles"].items():
            print(f"[oracle] {cs}/{orc}: mIoU={o['mIoU']} "
                  f"clean={o['clean_mIoU']} headroom={o['headroom']}")
    if "by_condition" in result:
        print(f"[oracle] by_condition: {len(result['by_condition'])} groups "
              f"-> {base / 'oracle_by_condition.md'}")
    print(f"[oracle] -> {base}")


if __name__ == "__main__":
    main()
