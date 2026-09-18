#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""headline.yaml → 벤치 표 생성기 (2026-09-18 신설 — docs-structure-audit R2·R3·권고 #11·#13).

- .claude_logs/experiments/headline.yaml(헤드라인 수치 단일 정본)을 읽어
  .claude_logs/status/current.md 의 <!-- headline:begin --> … <!-- headline:end -->
  블록을 생성물로 교체한다(멱등 — 몇 번 돌려도 같은 결과).
- 값은 yaml 에서 "읽기만" 한다. 여기서 차감·평균 등 계산을 하지 않는다.
- --check : 파일을 고치지 않고 생성물과 현재 블록 비교(다르면 exit 1).
- load_headline() 은 노션 빌더(.claude/skills/notion-experiment-log/
  paper_page_builder.py sec_summary)도 import 해 쓴다.

실행: /home/jemo/anaconda3/envs/MMSS_SAM/bin/python tools/gen_headline_tables.py [--check]
"""
import argparse
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADLINE_PATH = os.path.join(REPO, ".claude_logs", "experiments", "headline.yaml")
CURRENT_MD = os.path.join(REPO, ".claude_logs", "status", "current.md")

BEGIN = "<!-- headline:begin -->"
END = "<!-- headline:end -->"


def load_headline(path=HEADLINE_PATH):
    """headline.yaml 전체 dict 반환(노션 빌더·기타 도구의 단일 출처)."""
    import yaml  # MMSS_SAM env 의 PyYAML 사용(6.0.3 확인됨)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"headline.yaml 없음: {path} — develop 병합·체크아웃 상태 확인 필요"
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rel_link(path_from_repo_root, from_dir):
    """repo 상대 경로(.claude_logs/...)를 대상 파일 기준 상대 링크로 바꾼다."""
    if from_dir is None:
        return path_from_repo_root
    return os.path.relpath(os.path.join(REPO, path_from_repo_root), from_dir)


def _fmt_sota(bench, entry):
    """SOTA 열 — 1차(같은 모달 계열) / 2차(적은 모달). split=val 이면 *_val 변형."""
    val = entry.get("split") == "val"
    s1 = bench.get("sota_val" if val else "sota")
    s2 = bench.get("sota2_val" if val else "sota2")
    parts = []
    if s1:
        parts.append(f"{s1['name']} {s1['value']}")
    if s2:
        parts.append(f"2차 {s2['name']} {s2['value']}")
    return " / ".join(parts) if parts else "—"


def _fmt_ours(e):
    """우리 최고 열 — 값(±std·best·병기값)·프로토콜 약칭·근거·규칙·상태."""
    head = f"**{e['value']}**"
    if e.get("std") is not None:
        head += f" ±{e['std']}"
    if e.get("best") is not None:
        head += f" · best {e['best']}"
    if e.get("protocol_short"):
        head += f"({e['protocol_short']})"
    for i in ("", "2"):
        alt = e.get(f"value_alt{i}")
        if alt is not None:
            tag = e.get(f"protocol_short_alt{i}", "")
            head += f" / {alt}({tag})" if tag else f" / {alt}"
    tail = [x for x in (e.get("basis"), e.get("ckpt_rule"), e.get("status")) if x]
    return head + (" — " + " · ".join(tail) if tail else "")


def render_table(data, from_dir=None):
    """마커 블록 안에 들어갈 markdown 표(문자열)."""
    rows = [["벤치·분할", "SOTA(1차 같은 모달 / 2차 적은 모달)",
             "우리 최고(값·근거·규칙·상태)", "격차", "근거"]]
    for key, b in data["benchmarks"].items():
        ours = b.get("ours", [])
        for i, e in enumerate(ours):
            label = b.get("label", key) if i == 0 else "〃"
            if e.get("split"):
                label += f" · {e['split']}"
            gap = " · ".join(x for x in (e.get("gap_1st"), e.get("gap_2nd")) if x) or "—"
            srcs = []
            for s in e.get("source", []):
                link = _rel_link(s["path"], from_dir)
                srcs.append(f"[{s['label']}]({link})")
            rows.append([label, _fmt_sota(b, e), _fmt_ours(e), gap,
                         " · ".join(srcs) or "—"])
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "|" + "---|" * len(rows[0]))
    return "\n".join(lines)


def replace_block(text, block):
    """마커 사이를 교체(멱등). 마커가 없으면 예외로 종료."""
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    new = BEGIN + "\n" + block + "\n" + END
    if not pattern.search(text):
        raise SystemExit(f"마커 없음: {BEGIN} … {END} — 대상 파일에 마커를 먼저 둘 것")
    return pattern.sub(lambda _: new, text, count=1)


def main():
    ap = argparse.ArgumentParser(description="headline.yaml → current.md 벤치 표")
    ap.add_argument("--check", action="store_true",
                    help="파일을 고치지 않고 블록 일치만 검사(불일치 exit 1)")
    ap.add_argument("--stdout", action="store_true", help="표를 파일 대신 stdout 으로")
    args = ap.parse_args()

    data = load_headline()
    with open(CURRENT_MD, encoding="utf-8") as f:
        text = f.read()
    block = render_table(data, from_dir=os.path.dirname(CURRENT_MD))

    if args.stdout:
        print(block)
        return

    new_text = replace_block(text, block)
    if args.check:
        if new_text != text:
            print("불일치: current.md 헤드라인 블록이 headline.yaml 과 다름(생성기 재실행 필요)")
            sys.exit(1)
        print("일치: current.md 헤드라인 블록 = headline.yaml 생성물")
        return

    if new_text != text:
        with open(CURRENT_MD, "w", encoding="utf-8") as f:
            f.write(new_text)
        print("갱신: current.md 헤드라인 블록 교체 완료")
    else:
        print("변화 없음(멱등)")


if __name__ == "__main__":
    main()
