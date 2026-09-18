"""DELIVER 조건별 Δ — 이미지별 mIoU 자료를 조건(cloud/fog/night/rain/sun)과
센서 열화 케이스로 갈라 모달 제거 전후를 비교한다.

입력 두 갈래:
  - 기준선(probe_dgfusion.py) CSV: file_name, seg_miou_img 열
  - 우리 모델(modality_zero_ablation.py --per_image_csv): img_path, miou, condition 열

조건은 DELIVER 경로 `.../img/<조건>/test/<장면>/<파일>` 에서 읽는다. 센서 열화 케이스는
장면 폴더 이름에 붙어 있으면 함께 센다.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

CASES = ["motionblur", "overexposure", "underexposure", "lidarjitter", "eventlowres"]


def cond_of(path: str) -> str:
    parts = Path(path.replace("\\", "/")).parts
    if "img" in parts:
        i = parts.index("img")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def case_of(path: str) -> str:
    low = path.lower()
    for c in CASES:
        if c in low:
            return c
    return "clean"


def load(path: Path):
    """(조건, 케이스) -> 이미지 stem -> mIoU"""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {}
    key_path = "file_name" if "file_name" in rows[0] else "img_path"
    key_miou = "seg_miou_img" if "seg_miou_img" in rows[0] else "miou"
    out = {}
    for r in rows:
        p = r[key_path]
        v = r[key_miou]
        if v in (None, "", "nan"):
            continue
        out[p] = (cond_of(p), case_of(p), float(v))
    return out


def summarize(base_csv: Path, zero_csv: Path, label: str):
    base, zero = load(base_csv), load(zero_csv)
    common = set(base) & set(zero)
    if not common:
        print(f"[{label}] 겹치는 이미지가 없다 — 경로 키가 다른지 확인하라")
        return
    by_cond = defaultdict(list)
    by_case = defaultdict(list)
    for p in common:
        c, cs, b = base[p]
        _, _, z = zero[p]
        by_cond[c].append(z - b)
        by_case[cs].append(z - b)

    print(f"\n=== {label} (겹친 이미지 {len(common)}장) ===")
    print("조건별 Δ(이미지평균 mIoU, 제거 후 − 기준):")
    for c in sorted(by_cond, key=lambda k: sum(by_cond[k]) / len(by_cond[k])):
        v = by_cond[c]
        print(f"  {c:12s} n={len(v):4d}  Δ {sum(v)/len(v):+7.2f}")
    if len(by_case) > 1:
        print("센서 열화 케이스별 Δ:")
        for c in sorted(by_case, key=lambda k: sum(by_case[k]) / len(by_case[k])):
            v = by_case[c]
            print(f"  {c:14s} n={len(v):4d}  Δ {sum(v)/len(v):+7.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit("사용: condition_delta.py <기준 CSV> <제거 CSV> <라벨> [...반복]")
    args = sys.argv[1:]
    for i in range(0, len(args), 3):
        summarize(Path(args[i]), Path(args[i + 1]), args[i + 2])
