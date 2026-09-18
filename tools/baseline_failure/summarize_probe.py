"""probe_dgfusion CSV 요약 — 기준과 DEPTH 입력 0 치환을 나란히 출력한다."""
import csv
import statistics as st
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "probe_out")
tags = sys.argv[2:] or ["base", "zero_depth"]

for tag in tags:
    p = OUT / f"probe_{tag}.csv"
    if not p.exists():
        print(f"{tag:12s} (파일 없음: {p})")
        continue
    rows = list(csv.DictReader(open(p)))

    def col(key):
        vals = []
        for r in rows:
            v = r.get(key)
            if v not in (None, "", "nan"):
                vals.append(float(v))
        return vals

    ar, d1, mi = col("depth_absrel"), col("depth_d1"), col("seg_miou_img")
    src = rows[0].get("depth_src") if rows else "?"
    print(f"{tag:12s} n={len(rows):5d} src={src:12s} "
          f"AbsRel={st.mean(ar):8.4f} (중앙값 {st.median(ar):7.4f})  "
          f"delta1={st.mean(d1):.4f}  이미지평균 mIoU={st.mean(mi):6.2f}")
