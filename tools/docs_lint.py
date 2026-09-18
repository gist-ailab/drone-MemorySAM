#!/usr/bin/env python3
"""문서 체계 린트 (.claude_logs 구조 규칙 자동 검사 — 2026-09-18 구조 감사 R4·권고 #14).

감사 보고서 `.claude_logs/meta/2026-09-18-docs-structure-audit.md` §⑥ R4 표의 9종 검사를
`tools/eval_harness_guard.py`(채점기 동결 검사)와 같은 방식 — "규칙을 스크립트로 강제" — 로
구현한다. 규칙 본체는 `meta/conventions.md` §2(2026-09-18 감사 조치 R1·R3·R4).

사용:
    python tools/docs_lint.py                          # 전체 9종. ERROR 있으면 exit 1
    python tools/docs_lint.py --strict                 # WARN 만 있어도 exit 1
    python tools/docs_lint.py --only links,tables      # 검사 일부만
    python tools/docs_lint.py --json /tmp/lint.json    # 결과를 JSON 으로도 저장

검사 9종 (R4 표 그대로):
    moc        폴더 *.md 중 해당 00_MOC.md(루트는 00_INDEX.md)에 링크·언급되지 않은 파일
    links      상대 md 링크(+폴더 링크)가 실재하지 않음 ([[wikilink]]·백틱·코드펜스 제외)
    size       1,200줄 초과 (archive/·research/vault/ 제외; monitor-log·arch-evolution·
               status/history-* 는 롤오버 대상 표시로 WARN 하향)
    freshness  본문 최대 YYYY-MM-DD(미래 날짜 무시) > frontmatter updated:
    tables     표 헤더+구분선 다음 줄이 표 행도 빈 줄도 아님 (current.md:39-56형 파손)
    owns       frontmatter owns: 주제를 두 문서 이상이 선언
    numbers    experiments/headline.yaml 수치가 정본 문서 밖에 등장 (WARN). 파일 없으면 SKIP
    naming     파일명이 kebab-case/YYYY-MM-DD-접두/00_MOC·00_INDEX/구번호 스텁 규약 위반 (WARN)
    stubs      .claude_logs 루트 구번호 스텁(^\d\d_*.md)이 6줄 초과 (= 누가 구경로에 썼다)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO / '.claude_logs'

CHECKS = ['moc', 'links', 'size', 'freshness', 'tables', 'owns', 'numbers', 'naming', 'stubs']

VAULT = 'research/vault/'          # NAS 볼트 동기화 사본(손편집 금지) — 규약상 검사 제외 축
SIZE_LIMIT = 1200                  # R1: .claude_logs md 줄수 상한
STUB_MAX_LINES = 6                 # 루트 구번호 리다이렉트 스텁의 규격 줄수
# size 검사에서 ERROR 대 WARN(롤오버 대상 표시)으로 등급 하향하는 파일
SIZE_WARN_ONLY = (
    'experiments/monitor-log.md', 'models/arch-evolution.md',
    # 2026-09-18 판정 세션: 분할 과제 착수 전까지 롤오버 대상 표시(WARN)로 둔다 — 감사 권고 #18·#19
    'experiments/log.md', 'issues/issues-and-fixes.md', 'models/figures-ascii.md',
)
# numbers 검사에서 값 등장이 허용되는 정본 경로(절대 혹은 접두사)
NUMBERS_CANONICAL_EXACT = {
    'experiments/headline.yaml', 'experiments/judgment-ledger.md',
    'experiments/registry.md', 'experiments/log.md',
}
NUMBERS_CANONICAL_PREFIX = (
    'status/', 'decisions/', 'issues/', 'archive/', 'research/', 'meta/',
)
# naming 규약이 명시적으로 허용하는 특수형 (conventions.md §2 — history-<YYYY>H<n> 반기 파일)
NAMING_EXTRA_OK = re.compile(r'^history-\d{4}H[12]\.md$')

DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')
LINK_RE = re.compile(r'\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+"[^"]*")?\s*\)')
INLINE_CODE_RE = re.compile(r'`[^`]*`')
SKIP_SCHEMES = ('http://', 'https://', 'mailto:', 'ftp://')


class Doc:
    """md 파일 1개의 파싱 결과(검사 공통 전처리 캐시)."""

    def __init__(self, path: Path, root: Path):
        self.path = path
        self.rel = path.relative_to(root).as_posix()
        self.lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        # frontmatter: 첫 줄이 ---면 다음 ---까지. fm_end = 본문 시작줄(0-based)
        self.fm_end = 0
        self.frontmatter: list[str] = []
        if self.lines and self.lines[0].strip() == '---':
            for i in range(1, len(self.lines)):
                if self.lines[i].strip() == '---':
                    self.frontmatter = self.lines[1:i]
                    self.fm_end = i + 1
                    break
        # 코드 펜스 마스크(본문 영역만): True = 펜스 안
        self.in_fence = [False] * len(self.lines)
        fence = False
        for i, line in enumerate(self.lines):
            if line.strip().startswith(('```', '~~~')):
                fence = not fence
                self.in_fence[i] = True  # 펜스 경계 줄 자체도 제외
                continue
            self.in_fence[i] = fence

    def body_lines(self):
        """(인덱스, 원본 줄) 본문·펜스 밖만 순회."""
        for i in range(self.fm_end, len(self.lines)):
            if not self.in_fence[i]:
                yield i, self.lines[i]

    def fm_value(self, key):
        m = re.search(rf'^{key}:\s*(.*)$', '\n'.join(self.frontmatter), re.M)
        return m.group(1).strip() if m else None

    def fm_list(self, key):
        """frontmatter 목록 값 — owns: [a, b] 인라인 / owns:\\n  - a 블록 양쪽."""
        head = self.fm_value(key)
        items = []
        if head is None:
            return items
        if head.startswith('['):
            items += [x.strip().strip('\'"') for x in head.strip('[]').split(',') if x.strip()]
        elif head:
            items += [head.strip('\'"')]
        # 블록 목록(- item) — head 가 비었거나 [..] 가 아닌 직후 행들
        started = False
        for line in self.frontmatter:
            if re.match(rf'^{key}:', line):
                started = True
                continue
            if started:
                m = re.match(r'^\s+-\s+(.+)$', line)
                if m:
                    items.append(m.group(1).strip().strip('\'"'))
                elif line.strip() and not re.match(r'^\s+-', line):
                    break
        return items


def display(path: Path) -> str:
    """보고용 경로 — 리포 기준 상대경로, 리포 밖이면 절대경로."""
    try:
        rel = path.relative_to(REPO).as_posix()
        return rel if not rel.startswith('..') else str(path)
    except ValueError:
        return str(path)


def strip_inline_code(line: str) -> str:
    return INLINE_CODE_RE.sub('', line)


def link_target(raw: str):
    """링크 대상 정규화 — 검사 대상이 아니면 None 반환."""
    t = raw.strip().strip('<>')
    if not t or t.startswith(SKIP_SCHEMES) or t.startswith('/') or t.startswith('#'):
        return None
    return t.split('#', 1)[0] or None


# ---------------------------------------------------------------- 검사 1: moc
def collect_mocs(root: Path):
    """{폴더 상대경로: MOC Doc} — 루트는 00_MOC.md 우선, 없으면 00_INDEX.md."""
    mocs = {}
    for p in sorted(root.rglob('00_MOC.md')):
        mocs[p.parent.relative_to(root).as_posix()] = Doc(p, root)
    top = root / '00_INDEX.md'
    if '.' not in mocs and top.exists():
        mocs['.'] = Doc(top, root)
    return mocs


def check_moc(root: Path, docs: dict[str, Doc], out: list):
    mocs = collect_mocs(root)

    def mentioned(name: str, moc: Doc) -> bool:
        # 매핑표·취소선 행처럼 링크가 아니라 텍스트로 언급된 경우도 등록으로 인정.
        pat = re.compile(r'(?<![\w.\-])' + re.escape(name) + r'(?![\w\-])')
        return any(pat.search(line) for _, line in moc.body_lines())

    def linked(target_doc: Doc, moc: Doc) -> bool:
        base = moc.path.parent
        for _, line in moc.body_lines():
            for m in LINK_RE.finditer(strip_inline_code(line)):
                t = link_target(m.group(1))
                if t and (base / t).resolve() == target_doc.path.resolve():
                    return True
        return False

    for rel, doc in docs.items():
        if rel.startswith(VAULT) or Path(rel).name in ('00_MOC.md', '00_INDEX.md'):
            continue
        # 자기 폴더~루트까지 조상 MOC 링크로 등록 인정(하위 폴더 파일의 상위 MOC 등록 포함)
        ancestors, cur = [], Path(rel).parent
        while True:
            ancestors.append(cur.as_posix() if cur != Path('.') else '.')
            if cur == Path('.'):
                break
            cur = cur.parent
        if not any(a in mocs and (linked(doc, mocs[a]) or mentioned(doc.path.name, mocs[a]))
                   for a in ancestors):
            where = next((a for a in ancestors if a in mocs), '.')
            out.append(('ERROR', 'moc', doc.path, 1,
                        f'{where} 의 MOC/INDEX 에 링크·언급 없음 — 폴더 00_MOC.md 에 등록하라'))


# -------------------------------------------------------------- 검사 2: links
def check_links(docs: dict[str, Doc], out: list):
    # 감사 방법론과 동일하게 "상대 md 링크"로 한정(.py/.yaml/.csv 등 코드 경로는 문서가
    # 아니라 리포 이력의 영역 — 2026-07 시절 삭제 파일을 역사 문서가 계속 지목한다).
    # 폴더 링크(끝 /)는 존재만 확인한다. [[wikilink]]·백틱·코드펰스는 애초에 추출 안 함.
    for rel, doc in docs.items():
        if rel.startswith(VAULT):
            continue
        base = doc.path.parent
        for i, line in doc.body_lines():
            for m in LINK_RE.finditer(strip_inline_code(line)):
                t = link_target(m.group(1))
                if t is None or not (t.endswith('.md') or t.endswith('/')):
                    continue
                if not (base / t).exists():
                    out.append(('ERROR', 'links', doc.path, i + 1,
                                f'링크 대상 없음: {t}'))


# --------------------------------------------------------------- 검사 3: size
def check_size(docs: dict[str, Doc], out: list):
    for rel, doc in docs.items():
        if rel.startswith(VAULT) or rel.startswith('archive/'):
            continue
        n = len(doc.lines)
        if n <= SIZE_LIMIT:
            continue
        if (rel in SIZE_WARN_ONLY or rel.startswith('status/history-')):
            out.append(('WARN', 'size', doc.path, 1,
                        f'{n}줄 — 1,200줄 상한 초과(롤오버/분할 대상)'))
        else:
            out.append(('ERROR', 'size', doc.path, 1,
                        f'{n}줄 — 1,200줄 상한 초과(R1: 롤오버/분할 필요)'))


# ---------------------------------------------------------- 검사 4: freshness
def check_freshness(docs: dict[str, Doc], out: list, today: date):
    for rel, doc in docs.items():
        updated = doc.fm_value('updated')
        if not updated:
            continue
        fm_date = DATE_RE.search(updated)
        if not fm_date:
            continue
        try:
            fm_d = date.fromisoformat(fm_date.group(0))
        except ValueError:
            continue
        body_max = None
        for _, line in doc.body_lines():
            for m in DATE_RE.finditer(line):
                try:
                    d = date.fromisoformat(m.group(0))
                except ValueError:
                    continue
                if d <= today and (body_max is None or d > body_max):
                    body_max = d
        if body_max and body_max > fm_d:
            out.append(('WARN', 'freshness', doc.path, 1,
                        f'본문 최대 날짜 {body_max} > frontmatter updated: {fm_d} — 갱신하라'))


# -------------------------------------------------------------- 검사 5: tables
def _is_delim(line: str) -> bool:
    s = re.sub(r'^[>\s]+', '', line.strip())  # 인용문 안 표시까지 허용
    return bool(s) and '|' in s and '-' in s and all(c in '|-: ' for c in s)


def check_tables(docs: dict[str, Doc], out: list):
    for rel, doc in docs.items():
        body = [i for i, _ in doc.body_lines()]
        for pos, i in enumerate(body):
            line = doc.lines[i]
            if not _is_delim(line):
                continue
            if pos == 0 or '|' not in doc.lines[body[pos - 1]]:  # 헤더 줄이 없으면 표 아님
                continue
            if pos + 1 >= len(body):
                continue  # 구분선으로 끝난 표 — 빈 줄 종료와 동일 취급
            nxt = doc.lines[body[pos + 1]].strip()
            if nxt and '|' not in nxt:
                out.append(('ERROR', 'tables', doc.path, body[pos + 1] + 1,
                            f'표 구분선 다음 줄이 표 행이 아님: "{nxt[:40]}" — 표가 중간에 끊김'))


# ---------------------------------------------------------------- 검사 6: owns
def check_owns(docs: dict[str, Doc], out: list):
    seen: dict[str, Doc] = {}
    for rel, doc in docs.items():
        for topic in doc.fm_list('owns'):
            if topic in seen:
                out.append(('ERROR', 'owns', doc.path, 1,
                            f'owns "{topic}" 중복 선언 — 이미 {display(seen[topic].path)} '
                            f'({seen[topic].rel}) 가 정본(R3: 문서 하나에 정본 하나)'))
            else:
                seen[topic] = doc


# ------------------------------------------------------------- 검사 7: numbers
def load_headline_values(root: Path):
    """headline.yaml 의 value·value_24cls 수치(소수 둘째 자리) — 없으면 (None, 사유)."""
    hp = root / 'experiments' / 'headline.yaml'
    if not hp.exists():
        return None, '파일 없음'
    text = hp.read_text(encoding='utf-8', errors='replace')
    values = set()
    try:
        import yaml  # 있으면 쓰고
        data = yaml.safe_load(text)
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ('value', 'value_24cls') and isinstance(v, (int, float)):
                        values.add(round(float(v), 2))
                    else:
                        stack.append(v)
            elif isinstance(node, list):
                stack.extend(node)
    except ImportError:  # 폴백: 정규식으로 value·value_24cls 키의 숫자만 추출(중첩 인라인 포함)
        for m in re.finditer(r'\bvalue(?:_24cls)?\s*:\s*([0-9]+(?:\.[0-9]+)?)', text):
            values.add(round(float(m.group(1)), 2))
    return values, None


NUM_RE = re.compile(r'(?<![\d.\w])\d+\.\d+')


def check_numbers(root: Path, docs: dict[str, Doc], out: list):
    values, why = load_headline_values(root)
    if values is None:
        out.append(('SKIP', 'numbers', root / 'experiments' / 'headline.yaml', 1,
                    f'{why} — 수치 확산 검사 생략'))
        return
    if not values:
        out.append(('SKIP', 'numbers', root / 'experiments' / 'headline.yaml', 1,
                    'value/value_24cls 수치 없음 — 검사 생략'))
        return
    watched = {f'{v:.2f}': v for v in values}
    for rel, doc in docs.items():
        if rel in NUMBERS_CANONICAL_EXACT or rel.startswith(NUMBERS_CANONICAL_PREFIX):
            continue
        for i, line in enumerate(doc.lines):
            for m in NUM_RE.finditer(line):
                if round(float(m.group(0)), 2) in values:
                    out.append(('WARN', 'numbers', doc.path, i + 1,
                                f'헤드라인 수치 {m.group(0)} 등장 — 정본(headline.yaml·current 등)'
                                f' 밖 복제 금지, 링크로 대체'))
                    break  # 줄당 1건


# -------------------------------------------------------------- 검사 8: naming
KEBAB = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*\.md$')
DATE_PREFIX = re.compile(r'^\d{4}-\d{2}-\d{2}-[a-z0-9]+(-[a-z0-9]+)*\.md$')
LEGACY_STUB = re.compile(r'^\d\d_.*\.md$')


def check_naming(docs: dict[str, Doc], out: list):
    for rel, doc in docs.items():
        name = Path(rel).name
        if rel.startswith(VAULT):
            continue
        if (KEBAB.match(name) or DATE_PREFIX.match(name) or LEGACY_STUB.match(name)
                or name in ('00_MOC.md', '00_INDEX.md') or NAMING_EXTRA_OK.match(name)):
            continue
        out.append(('WARN', 'naming', doc.path, 1,
                    f'파일명 "{name}" — kebab-case(또는 YYYY-MM-DD- 접두) 규약 위반'))


# --------------------------------------------------------------- 검사 9: stubs
def check_stubs(root: Path, out: list):
    for p in sorted(root.glob('[0-9][0-9]_*.md')):
        if p.name.startswith('00_'):  # 00_INDEX.md 는 스텁이 아니라 현행 인덱스
            continue
        n = len(p.read_text(encoding='utf-8', errors='replace').splitlines())
        if n > STUB_MAX_LINES:
            out.append(('ERROR', 'stubs', p, 1,
                        f'구번호 리다이렉트 스텁이 {n}줄(>{STUB_MAX_LINES}) — 구경로에 내용을'
                        f' 쓰지 마라(신규는 새 경로에)'))


# ----------------------------------------------------------------------- main
def run(root: Path, only: list[str]) -> list[tuple]:
    docs: dict[str, Doc] = {}
    for p in sorted(root.rglob('*.md')):
        docs[p.relative_to(root).as_posix()] = Doc(p, root)
    today = date.today()
    out: list[tuple] = []

    if 'moc' in only:
        check_moc(root, docs, out)
    if 'links' in only:
        check_links(docs, out)
    if 'size' in only:
        check_size(docs, out)
    if 'freshness' in only:
        check_freshness(docs, out, today)
    if 'tables' in only:
        check_tables(docs, out)
    if 'owns' in only:
        check_owns(docs, out)
    if 'numbers' in only:
        check_numbers(root, docs, out)
    if 'naming' in only:
        check_naming(docs, out)
    if 'stubs' in only:
        check_stubs(root, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='.claude_logs 문서 체계 린트(감사 R4 9종)')
    ap.add_argument('--root', default=str(DEFAULT_ROOT),
                    help=f'검사 루트(기본 {DEFAULT_ROOT})')
    ap.add_argument('--strict', action='store_true', help='WARN 도 exit 1 로 취급')
    ap.add_argument('--json', metavar='PATH', help='결과 JSON 저장 경로')
    ap.add_argument('--only', default=','.join(CHECKS),
                    help=f'실행할 검사 쉼표 목록(기본 전부): {",".join(CHECKS)}')
    a = ap.parse_args()

    root = Path(a.root).resolve()
    if not root.is_dir():
        print(f'[lint] FAIL: 검사 루트 없음 — {root}', file=sys.stderr)
        return 1
    only = [x.strip() for x in a.only.split(',') if x.strip()]
    bad = [x for x in only if x not in CHECKS]
    if bad:
        print(f'[lint] FAIL: 알 수 없는 검사명 {bad} — 가능: {", ".join(CHECKS)}', file=sys.stderr)
        return 1

    findings = run(root, only)
    for sev, check, path, line, msg in findings:
        print(f'[{sev}] {check} {display(path)}:{line} {msg}')

    n_err = sum(1 for f in findings if f[0] == 'ERROR')
    n_warn = sum(1 for f in findings if f[0] == 'WARN')
    n_skip = sum(1 for f in findings if f[0] == 'SKIP')
    print('\n== 요약 ==')
    print('| 검사 | ERROR | WARN | SKIP |')
    print('|---|---|---|---|')
    for c in CHECKS:
        if c not in only:
            continue
        fs = [f for f in findings if f[1] == c]
        print(f'| {c} | {sum(1 for f in fs if f[0] == "ERROR")} '
              f'| {sum(1 for f in fs if f[0] == "WARN")} '
              f'| {sum(1 for f in fs if f[0] == "SKIP")} |')
    exit_code = 1 if n_err else (1 if (a.strict and n_warn) else 0)
    print(f'합계 ERROR {n_err} · WARN {n_warn} · SKIP {n_skip} → exit {exit_code}'
          + (' (strict)' if a.strict else ''))
    print('[lint] OK — 문서 체계 검사 통과.' if exit_code == 0 else
          '[lint] FAIL — ERROR 를 0으로 만든 뒤 병합하라(규칙 = meta/conventions.md §2).')

    if a.json:
        payload = {
            'root': str(root),
            'checks': only,
            'summary': {'error': n_err, 'warn': n_warn, 'skip': n_skip, 'exit_code': exit_code},
            'findings': [
                {'severity': sev, 'check': check, 'path': display(path), 'line': line,
                 'message': msg}
                for sev, check, path, line, msg in findings
            ],
        }
        Path(a.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
                                encoding='utf-8')
        print(f'[lint] JSON → {a.json}')
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
