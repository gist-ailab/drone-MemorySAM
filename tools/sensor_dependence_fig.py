#!/usr/bin/env python3
"""센서 제거 시 조건별 mIoU 변화(센서 의존도) 히트맵 그림.

``tools/sensor_dependence_by_condition.py`` 가 낸 run 폴더(들)의 ``dependence.csv`` 를 읽어,
모델마다 히트맵 한 장씩 가로로 나란히 그린다(같은 색 척도).
- 행 = 센서 4 개(no_img→RGB, no_depth→Depth, no_event→Event, no_lidar→LiDAR).
- 열 = 조건 6 개(Sun, Cloud, Rain, Fog, Night, Low exp.).
- 칸 값 = −drop_vs_clean(센서를 뺐을 때 mIoU 변화, 0 = 의존 없음, 음수 = 떨어짐).
- 색 = 단색 순차(0 이 가장 밝게). carbon 디자인 팔레트(SEQUENTIAL·rcparams)를 쓴다.
기존 파일은 수정하지 않는다.
"""
import argparse
import csv
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np

_CARBON_PATH = "/home/jemo/cl-jemo/labdesign/build/py/carbon_palette.py"
_DEFAULT_SEQUENTIAL = ['#ffffff', '#e4eeff', '#cadcff', '#afcbff', '#94b9ff',
                       '#7aa8fe', '#5f96fe', '#4485fe', '#2a73fe', '#0f62fe']

# 설정(센서 제거) → 행 이름, 그룹(조건) → 열 이름. 단일 출처.
SETTING_TO_ROW = [("no_img", "RGB"), ("no_depth", "Depth"),
                  ("no_event", "Event"), ("no_lidar", "LiDAR")]
GROUP_TO_COL = [("sun", "Sun"), ("cloud", "Cloud"), ("rain", "Rain"),
                ("fog", "Fog"), ("night", "Night"), ("lowexp", "Low exp.")]
ROW_LABELS = [r for _, r in SETTING_TO_ROW]
COL_LABELS = [c for _, c in GROUP_TO_COL]
DESIGN_SOURCE = "carbon@unknown"


def load_carbon():
    """carbon 팔레트 모듈을 importlib 로 불러 (SEQUENTIAL, rcparams) 반환.

    파일이 없으면 기본 SEQUENTIAL + 빈 rcparams 로 대체하고 경고한다.
    """
    if not Path(_CARBON_PATH).exists():
        warnings.warn(f"[sensor-dep-fig] carbon 팔레트 없음({_CARBON_PATH}) — 기본값 사용")
        return list(_DEFAULT_SEQUENTIAL), (lambda theme="light": {})
    spec = importlib.util.spec_from_file_location("carbon_palette", _CARBON_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seq = list(getattr(mod, "SEQUENTIAL", _DEFAULT_SEQUENTIAL)) or list(_DEFAULT_SEQUENTIAL)
    rc = getattr(mod, "rcparams", lambda theme="light": {})
    return seq, rc


def read_dependence_csv(path):
    """dependence.csv → {(setting, group): value(=−drop_vs_clean)} dict.

    값이 비어 있으면(NaN·미집계) 건너뛴다.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            drop = row.get("drop_vs_clean", "")
            if drop is None or str(drop).strip() == "":
                continue
            try:
                out[(row["setting"], row["group"])] = -float(drop)
            except ValueError:
                continue
    return out


def build_matrix(dep):
    """{(setting, group): value} → (4, 6) 행렬. 없는 칸은 NaN."""
    mat = np.full((len(SETTING_TO_ROW), len(GROUP_TO_COL)), np.nan, dtype=np.float64)
    for i, (setting, _) in enumerate(SETTING_TO_ROW):
        for j, (group, _) in enumerate(GROUP_TO_COL):
            if (setting, group) in dep:
                mat[i, j] = dep[(setting, group)]
    return mat


def make_figure(run_names, matrices, out_dir, seq, rc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update(rc(theme="light"))
    # 0(의존 없음)이 가장 밝게 → 순차 팔레트를 뒤집어 vmax(=0 쪽)가 흰색이 되게 한다.
    cmap = LinearSegmentedColormap.from_list("carbon_seq_r", list(seq)[::-1])

    finite = np.concatenate([m[np.isfinite(m)] for m in matrices]) \
        if any(np.isfinite(m).any() for m in matrices) else np.array([0.0])
    vmin = float(np.min(finite)) if finite.size else -1.0
    vmax = float(np.max(finite)) if finite.size else 0.0
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    n = len(run_names)
    fig, axes = plt.subplots(1, n, figsize=(2.2 + 3.0 * n, 3.4), squeeze=False)
    axes = axes[0]
    im = None
    for ax, name, mat in zip(axes, run_names, matrices):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(COL_LABELS)))
        ax.set_xticklabels(COL_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(ROW_LABELS)))
        ax.set_yticklabels(ROW_LABELS)
        ax.set_title(name)
        span = vmax - vmin if vmax > vmin else 1.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "–", ha="center", va="center", color="#888888")
                    continue
                norm = (v - vmin) / span          # 1=가장 밝음, 0=가장 어두움
                color = "#161616" if norm > 0.45 else "#ffffff"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color=color, fontsize=9)
    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
        cbar.set_label("mIoU change when sensor zeroed (0 = no dependence)")
    fig.suptitle("Sensor dependence by condition (−drop vs clean)", y=1.02)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "dependence_map.png"
    pdf = out_dir / "dependence_map.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_combined_csv(path, run_names, matrices):
    """합친 표: model, sensor, <조건 6열>."""
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "sensor"] + COL_LABELS)
        for name, mat in zip(run_names, matrices):
            for i, row_label in enumerate(ROW_LABELS):
                cells = ["" if not np.isfinite(mat[i, j]) else f"{mat[i, j]:.4f}"
                         for j in range(mat.shape[1])]
                writer.writerow([name, row_label] + cells)


def _write_sidecar(path):
    Path(str(path) + ".json").write_text(
        json.dumps({"design_source": DESIGN_SOURCE}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def _parse_runs(run_specs):
    runs = []
    for spec in run_specs:
        if "=" not in spec:
            raise argparse.ArgumentTypeError(f"--runs 항목은 이름=폴더 형식이어야 합니다: {spec!r}")
        name, folder = spec.split("=", 1)
        runs.append((name, folder))
    return runs


def run(run_specs, out):
    runs = _parse_runs(run_specs)
    seq, rc = load_carbon()
    names, matrices = [], []
    for name, folder in runs:
        csv_path = Path(folder) / "dependence.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"dependence.csv 없음: {csv_path}")
        names.append(name)
        matrices.append(build_matrix(read_dependence_csv(csv_path)))

    out_dir = Path(out)
    png, pdf = make_figure(names, matrices, out_dir, seq, rc)
    combined = out_dir / "dependence_map.csv"
    write_combined_csv(combined, names, matrices)
    for p in (png, pdf, combined):
        _write_sidecar(p)
    return png, pdf, combined


def _selftest():
    import tempfile
    rows_a = [
        ("no_img", "sun", "60", "5.0"), ("no_img", "night", "40", "12.0"),
        ("no_depth", "sun", "60", "1.0"), ("no_depth", "fog", "30", "3.0"),
        ("no_event", "rain", "25", "0.5"), ("no_lidar", "lowexp", "10", "2.0"),
        ("clean", "all", "100", "0.0"),
    ]
    rows_b = [
        ("no_img", "sun", "60", "4.0"), ("no_img", "night", "40", "6.0"),
        ("no_depth", "cloud", "30", "2.0"), ("no_lidar", "fog", "30", "1.5"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for sub, rows in (("A", rows_a), ("B", rows_b)):
            d = tmp / sub
            d.mkdir()
            with (d / "dependence.csv").open("w", newline="", encoding="utf-8") as s:
                w = csv.writer(s)
                w.writerow(["setting", "group", "n_images", "mIoU", "drop_vs_clean"])
                for setting, group, n, drop in rows:
                    w.writerow([setting, group, n, "0.0", drop])
        out = tmp / "fig"
        png, pdf, combined = run([f"modelA={tmp/'A'}", f"modelB={tmp/'B'}"], out)
        for p in (png, pdf, combined):
            assert Path(p).exists(), f"출력 없음: {p}"
            assert Path(str(p) + ".json").exists(), f"사이드카 없음: {p}.json"
        # 값 확인: no_img/night drop 12 → 값 −12
        dep = read_dependence_csv(tmp / "A" / "dependence.csv")
        assert abs(dep[("no_img", "night")] - (-12.0)) < 1e-9
        mat = build_matrix(dep)
        assert abs(mat[0, 4] - (-12.0)) < 1e-9   # RGB 행, Night 열
    print("SELFTEST_OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="합성 csv 두 개로 그림 생성만 검증하고 종료")
    ap.add_argument("--runs", nargs="+", help="이름=폴더 (각 폴더에 dependence.csv)")
    ap.add_argument("--out", help="출력 폴더")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.runs or not args.out:
        ap.error("--runs 와 --out 이 필요합니다(--selftest 제외)")
    png, pdf, combined = run(args.runs, args.out)
    print(f"[sensor-dep-fig] -> {png}\n[sensor-dep-fig] -> {pdf}\n"
          f"[sensor-dep-fig] -> {combined}")


if __name__ == "__main__":
    main()
