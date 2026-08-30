#!/usr/bin/env python3
"""평가 하네스 해시 가드 (ISSUE-033 재발 방지, 2026-08-31 도입).

정본 채점기(legal 수치를 만드는 코드·config)는 이 매니페스트로 동결된다.
karpathy/autoresearch 의 "평가 하네스는 에이전트 불가침" 원칙의 구조화 —
우리는 ISSUE-033(측정 드라이버 이원화로 base outlier 허상)으로 같은 교훈을
규약으로만 갖고 있었고, 이 가드가 그것을 검사 가능하게 만든다.

사용:
    python tools/eval_harness_guard.py --check           # legal eval 전 필수. 불일치 시 exit 1
    python tools/eval_harness_guard.py --freeze          # 의도적 하네스 변경 후 재동결
                                                         # (변경 사유를 커밋 메시지에 명시할 것)

규칙 (meta/conventions.md · ISSUE-033):
  - legal 수치(논문·registry 인용)를 만드는 모든 eval 은 실행 전 --check 를 통과해야 한다.
  - --check FAIL 상태에서 만든 수치는 legal 로 인용 금지.
  - 하네스를 바꿔야 할 정당한 이유가 생기면: 변경 → --freeze → 커밋(사유 명시) →
    이전 수치와의 정합성 재검(대표 ckpt 1개 재채점)이 의무다.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / 'configs' / 'eval' / 'HARNESS_MANIFEST.sha256'

# 동결 대상 = "legal 수치를 산출하는 경로" 최소집합.
# 로더(deliver/muses/mcubes.py)는 학습에도 쓰여 전체 동결이 과하지만, GT 채점에
# 직접 관여하므로 포함한다 — 학습 목적 수정이 필요하면 --freeze 절차를 따른다.
PROTECTED = [
    'val.py',
    'tools/eval_muses_official.py',
    'semseg/metrics.py',
    'semseg/datasets/deliver.py',
    'semseg/datasets/muses.py',
    'semseg/datasets/mcubes.py',
    'configs/eval/yeon-deliver_rgbdel_P46_c3only_lam01_base_eval1024.yaml',
    'configs/eval/yeon-deliver_rgbdel_P46_c3only_lam01_base_eval768.yaml',
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def freeze() -> int:
    lines = []
    for rel in PROTECTED:
        p = REPO / rel
        if not p.exists():
            print(f'[guard] MISSING (freeze 불가): {rel}', file=sys.stderr)
            return 1
        lines.append(f'{sha256(p)}  {rel}')
    MANIFEST.write_text('\n'.join(lines) + '\n')
    print(f'[guard] FROZEN {len(lines)} files -> {MANIFEST.relative_to(REPO)}')
    return 0


def check() -> int:
    if not MANIFEST.exists():
        print('[guard] FAIL: 매니페스트 없음 — 먼저 --freeze 하라.', file=sys.stderr)
        return 1
    want = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            want[rel.strip()] = digest
    bad = []
    for rel, digest in want.items():
        p = REPO / rel
        if not p.exists():
            bad.append((rel, 'MISSING'))
        elif sha256(p) != digest:
            bad.append((rel, 'MODIFIED'))
    if bad:
        print('[guard] FAIL — 평가 하네스가 동결 시점과 다르다:', file=sys.stderr)
        for rel, why in bad:
            print(f'  {why:8s} {rel}', file=sys.stderr)
        print('[guard] 이 상태로 만든 수치는 legal 인용 금지. 의도적 변경이면 '
              '--freeze 후 커밋(사유 명시)+대표 ckpt 재채점.', file=sys.stderr)
        return 1
    print(f'[guard] OK — {len(want)} files match frozen manifest.')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check', action='store_true')
    g.add_argument('--freeze', action='store_true')
    a = ap.parse_args()
    return freeze() if a.freeze else check()


if __name__ == '__main__':
    sys.exit(main())
