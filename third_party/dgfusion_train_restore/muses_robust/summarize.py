"""Summarize scored detectron2 MUSES robustness settings."""
import csv
import json
import sys
from pathlib import Path


def main(root):
    rows = []
    for path in sorted(Path(root).glob('*/report.json')):
        report = json.loads(path.read_text())
        rows.append({'subset': path.parent.name,
                     'mIoU': report['official_native_1080x1920']['mIoU_present_classes'],
                     **{k: v['mIoU_present_classes'] for k, v in report['per_condition_official'].items()}})
    if not rows:
        raise SystemExit(f'No reports in {root}')
    cols = ['subset', 'mIoU'] + sorted(set().union(*(r.keys() for r in rows)) - {'subset', 'mIoU'})
    if len(rows) == 15:
        rows.append({'subset': 'mean15',
                     **{key: round(sum(float(row[key]) for row in rows) / 15, 4)
                        for key in cols if key != 'subset' and all(key in row for row in rows)}})
    with (Path(root) / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} settings -> {root}/summary.csv')


if __name__ == '__main__':
    main(sys.argv[1])
