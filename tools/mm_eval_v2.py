#!/usr/bin/env python3
"""missing_modality_eval.py 를 legal v2 하네스(nearest-exact 복원)로 실행하는 래퍼.

val.py·missing_modality_eval.py 는 수정하지 않고 두 가지만 몽키패치한다.
1) val._unpad_resize_to_orig -> tools/legal_rescore_v2 의 nearest-exact 판(헤드라인 채점과 같은 하네스).
2) 환경변수 MM_PRESENT_ONLY 가 있으면 build_cases 결과를 clean + present 가 그 값과 같은 케이스로 좁힌다.
   예: MM_PRESENT_ONLY=img+event+lidar  -> depth 한 모달만 결측·열화시킨 EMM/RMM 케이스만 돈다.
인자는 missing_modality_eval.py 와 동일하다.
"""
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import val  # noqa: E402
from tools import legal_rescore_v2  # noqa: E402
from tools import missing_modality_eval as mme  # noqa: E402


def main():
    val._unpad_resize_to_orig = legal_rescore_v2._unpad_resize_to_orig_v2
    print(f"[mm-eval-v2] _unpad_resize_to_orig -> {legal_rescore_v2.RESAMPLE_MODE}", flush=True)
    only = os.environ.get("MM_PRESENT_ONLY", "").strip()
    if only:
        orig = mme.build_cases

        def filtered(*a, **k):
            cases = orig(*a, **k)
            kept = [c for c in cases if c.group == "clean" or c.present_names == only]
            print(f"[mm-eval-v2] MM_PRESENT_ONLY={only}: {len(cases)} -> {len(kept)} cases", flush=True)
            return kept

        mme.build_cases = filtered
        orig_md = mme._write_summary_md

        def relabeled_md(path, *a, **k):
            # 필터 실행에서 summary.md 의 "15조합 평균" 틀 문구가 단일 부분집합 값을 전체 평균으로
            # 오독하게 만든다(2026-09-24). 문구를 부분집합 이름으로 바꾸고 머리에 경고 줄을 넣는다.
            orig_md(path, *a, **k)
            t = Path(path).read_text(encoding="utf-8")
            t = t.replace("15조합", f"필터 부분집합(present={only})")
            t = (f"> ⚠️ MM_PRESENT_ONLY={only}: 아래 값은 15조합 평균이 아니라 "
                 f"이 부분집합 하나(+clean)의 값이다.\n\n") + t
            Path(path).write_text(t, encoding="utf-8")

        mme._write_summary_md = relabeled_md
    mme.main()


if __name__ == "__main__":
    main()
