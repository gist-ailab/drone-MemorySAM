#!/usr/bin/env python3
"""configs/ 내 모든 YAML에 DATASET 서브루트(RGB/THERMAL/LIDAR) 및 EVAL.EVAL_DAY 일괄 적용."""
import re
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
SUBROOT_BLOCK = (
    "  RGB_SUBROOT   : null\n"
    "  THERMAL_SUBROOT : null\n"
    "  LIDAR_SUBROOT : null"
)

EVAL_DAY_LINE = "  EVAL_DAY      : false                                   # true면 test 시 RGB 서브루트를 zed_day로 사용\n"


def add_subroots(content: str) -> str:
    if "RGB_SUBROOT" in content or "DATASET:" not in content:
        return content
    # DATASET 블록 내 첫 MODALS 줄 바로 다음에 서브루트 삽입
    pattern = re.compile(r"^(\s+MODALS\s+:\s+\[.*\])\s*(?:#.*)?$", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return content
    # MODALS 줄 다음 줄 시작 위치에 서브루트 블록 삽입
    line_end = content.find("\n", match.end())
    if line_end == -1:
        insert_pos = len(content)
    else:
        insert_pos = line_end + 1
    return content[:insert_pos] + "\n" + SUBROOT_BLOCK + "\n" + content[insert_pos:]


def add_eval_day(content: str) -> str:
    if "EVAL_DAY" in content or "EVAL:" not in content:
        return content
    # EVAL: 블록 찾기
    eval_start = content.find("\nEVAL:")
    if eval_start == -1:
        return content
    # EVAL 블록 내 첫 BATCH_SIZE (다음 최상위 키 전까지)
    block_start = eval_start + 1
    next_section = content.find("\n\n", block_start)
    block_end = next_section if next_section != -1 else len(content)
    block = content[block_start:block_end]
    # "  BATCH_SIZE" 로 시작하는 줄 찾기 (EVAL 블록 내)
    batch_match = re.search(r"^  BATCH_SIZE\s+:\s*.*$", block, re.MULTILINE)
    if not batch_match:
        return content
    insert_in_block = batch_match.end()
    new_block = block[:insert_in_block] + "\n" + EVAL_DAY_LINE.rstrip() + "\n" + block[insert_in_block:]
    return content[:block_start] + new_block + content[block_end:]


def main():
    yaml_files = list(CONFIGS_DIR.rglob("*.yaml"))
    updated = []
    for path in sorted(yaml_files):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Skip (read error): {path} - {e}")
            continue
        orig = text
        text = add_subroots(text)
        text = add_eval_day(text)
        if text != orig:
            path.write_text(text, encoding="utf-8")
            updated.append(path)
    print(f"Updated {len(updated)} config(s):")
    for p in updated:
        print(f"  {p.relative_to(CONFIGS_DIR.parent)}")


if __name__ == "__main__":
    main()
