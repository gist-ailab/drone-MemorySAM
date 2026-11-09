"""Generate the two runnable variants of the public benchmark scripts.

upstream/       byte-identical copies of the benchmark scripts (commit 53c7a6e).
run_upstream/   upstream + parameter lines only (RMM ratio r, NM level) read from env;
                run inside an unmodified DELIVER checkout (commit b3a76c3), e.g. CMNeXt.
ours/           run_upstream + import lines (data/metrics from bench_upstream/, a
                byte-identical copy of DELIVER b3a76c3) + model construction via
                ours_model.build_ours_model. Evaluation logic is untouched.

Every replacement must match exactly once; otherwise this script fails.
"""
import difflib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ['val_mm_EMM.py', 'val_mm_RMM.py', 'val_mm_NM.py']

PARAM = {
    'val_mm_RMM.py': [("    r = 0.25\n",
                       "    r = float(os.environ.get('MMSSB_RMM_R', '0.25'))  # [mmssb] parameter\n")],
    'val_mm_NM.py': [("        level = 0 # 0,1,2\n",
                      "        level = int(os.environ.get('MMSSB_NM_LEVEL', '0')) # 0,1,2  [mmssb] parameter\n")],
}

BOOT = ("import os, sys  # [mmssb-ours] bootstrap: adapter dir, bench_upstream parent, our repo\n"
        "sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "
        "os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.environ['OURS_REPO']]\n")

IMPORTS = [
    ("from semseg.models import *\n",
     "from ours_model import build_ours_model  # [mmssb-ours] replaces model zoo import\n"),
    ("from semseg.datasets import *\n",
     "from bench_upstream.deliver import DELIVER  # [mmssb-ours] DELIVER b3a76c3, byte-identical\n"),
    ("from semseg.augmentations_mm import get_val_augmentation\n",
     "from bench_upstream.augmentations_mm import get_val_augmentation  # [mmssb-ours] DELIVER b3a76c3\n"),
    ("from semseg.metrics import Metrics\n",
     "from bench_upstream.metrics import Metrics  # [mmssb-ours] DELIVER b3a76c3\n"),
]


def replace_once(text, old, new, name):
    n = text.count(old)
    if n != 1:
        sys.exit(f'{name}: expected exactly one match for {old!r}, found {n}')
    return text.replace(old, new)


def replace_model_lines(text, name):
    lines = text.splitlines(keepends=True)
    idx = [i for i, l in enumerate(lines)
           if l.strip().startswith("model = eval(cfg['MODEL']['NAME'])")]
    if len(idx) != 1 or not lines[idx[0] + 1].strip().startswith('msg = model.load_state_dict('):
        sys.exit(f'{name}: model construction lines not found as expected')
    i = idx[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines[i:i + 2] = [f"{indent}model, msg = build_ours_model(cfg, str(model_path))  "
                      f"# [mmssb-ours] model construction\n"]
    return ''.join(lines)


def write_with_diff(src_text, dst_text, dst, label):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(dst_text)
    diff = difflib.unified_diff(src_text.splitlines(keepends=True), dst_text.splitlines(keepends=True),
                                fromfile=f'upstream/{dst.name.replace("ours_", "")}', tofile=f'{label}/{dst.name}')
    (dst.parent / (dst.name + '.diff')).write_text(''.join(diff))


def main():
    for name in NAMES:
        src = (HERE / 'upstream' / name).read_text()
        run_up = src
        for old, new in PARAM.get(name, []):
            run_up = replace_once(run_up, old, new, name)
        write_with_diff(src, run_up, HERE / 'run_upstream' / name, 'run_upstream')

        ours = BOOT + run_up
        for old, new in IMPORTS:
            ours = replace_once(ours, old, new, name)
        ours = replace_model_lines(ours, name)
        write_with_diff(src, ours, HERE / 'ours' / f'ours_{name}', 'ours')
        print(f'ok: {name}')


if __name__ == '__main__':
    main()
