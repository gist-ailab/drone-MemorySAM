#!/usr/bin/env python3
"""CPU checks for the default QAF plan and L9 pass selection."""
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_reliadino import _qaf_pass_plan  # noqa: E402


def main():
    for seed in (20260821, 20260902):
        path = ROOT / 'configs' / (f'yeon-deliver_rgbdel_P46_c3only_seed{seed}'
                                   '_screen40_A2hires.yaml')
        qaf = yaml.safe_load(path.read_text())['TRAIN']['QAF']
        assert 'SINGLE_PASS' not in qaf
        assert qaf['ENABLE'] is True and qaf['CLEAN_PASS'] is True
        assert qaf['DEGRADE_P'] == 0.5 and qaf['MISSING_FRAC'] == 0.2
        assert qaf['SEVERITY_SCHEDULE'] == [[0, 0.3], [10, 0.6], [20, 1.0]]
        explicit_false = {**qaf, 'SINGLE_PASS': False}
        assert {k: v for k, v in explicit_false.items() if k != 'SINGLE_PASS'} == qaf
        assert _qaf_pass_plan(qaf['ENABLE'], qaf.get('SINGLE_PASS', False)) == (False, True, True)
        assert _qaf_pass_plan(explicit_false['ENABLE'], explicit_false['SINGLE_PASS']) == (False, True, True)

    assert _qaf_pass_plan(False, False) == (False, False, False)
    assert _qaf_pass_plan(True, False) == (False, True, True)
    assert _qaf_pass_plan(True, True) == (True, False, False)
    try:
        _qaf_pass_plan(False, True)
    except ValueError:
        pass
    else:
        raise AssertionError('SINGLE_PASS without QAF.ENABLE must fail')
    print('SMOKE OK')


if __name__ == '__main__':
    main()
