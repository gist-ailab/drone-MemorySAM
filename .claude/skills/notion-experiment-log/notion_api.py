"""
Notion 실험노트 기록 헬퍼 — notion-experiment-log 스킬 전용.

인증: ~/.claude.json 의 mcpServers.notion-owner.env.NOTION_TOKEN 을 읽는다.
      토큰은 이 저장소에 절대 쓰지 않는다(머신 로컬 값).

사용:
    import sys; sys.path.insert(0, ".claude/skills/notion-experiment-log")
    from notion_api import *

핵심 함수:
    call(method, path, body)          — 저수준 REST
    find_rows(db_id)                  — DB 행 목록 [(title, id)]
    find_page(db_id, "P39")           — 제목 부분일치로 페이지 찾기 (동적 갱신의 출발점)
    page_blocks(pid)                  — 자식 블록 전체
    section_range(pid, "결과")        — 그 heading 절의 블록 id 목록
    replace_section(pid, "결과", blks) — 절 통째 교체 (멱등: 재실행해도 중복 없음)
    append_after(pid, after_id, blks) — 특정 블록 뒤에 삽입
    edit_text(bid, "새 텍스트")        — 블록 텍스트 교체
    table_rows / set_row / add_rows / rebuild_table / find_table — 표 조작
    upload(path) / image_block(fid)   — 이미지 업로드
    mermaid_png(text, out) / put_diagram(...) — 아키텍처 도면
    audit(pid)                        — 금지 출처·금지 수사 검사
"""
import json, os, re, subprocess, tempfile, urllib.request, uuid, mimetypes

# ---------------------------------------------------------------- auth / core
def _token():
    p = os.path.expanduser("~/.claude.json")
    return json.load(open(p))["mcpServers"]["notion-owner"]["env"]["NOTION_TOKEN"]

TOKEN = _token()
BASE = "https://api.notion.com/v1"
VER = "2022-06-28"


def _req(method, url, data=None, headers=None):
    r = urllib.request.Request(url, method=method, data=data, headers=headers or {})
    try:
        return json.load(urllib.request.urlopen(r))
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_body": e.read().decode()[:600]}


def call(method, path, body=None):
    """저수준 Notion REST. 실패 시 {"_error":code, "_body":...} 반환(예외 아님)."""
    return _req(method, BASE + path,
                json.dumps(body).encode() if body else None,
                {"Authorization": f"Bearer {TOKEN}", "Notion-Version": VER,
                 "Content-Type": "application/json"})


def ok(r):
    return not (isinstance(r, dict) and "_error" in r)


# ------------------------------------------------------------- block builders
def rt(s, bold=False, code=False, color=None):
    o = {"type": "text", "text": {"content": s},
         "annotations": {"bold": bold, "code": code}}
    if color:
        o["annotations"]["color"] = color
    return o


def para(*r):  return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": list(r)}}
def h1(s):     return {"object": "block", "type": "heading_1", "heading_1": {"rich_text": [rt(s)]}}
def h2(s):     return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [rt(s)]}}
def h3(s):     return {"object": "block", "type": "heading_3", "heading_3": {"rich_text": [rt(s)]}}
def div():     return {"object": "block", "type": "divider", "divider": {}}
def bul(*r):   return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": list(r)}}
def quote(*r): return {"object": "block", "type": "quote", "quote": {"rich_text": list(r)}}


def callout(rich, emoji="💡", color="gray_background"):
    return {"object": "block", "type": "callout",
            "callout": {"rich_text": rich, "icon": {"type": "emoji", "emoji": emoji}, "color": color}}


def code(s, lang="yaml"):
    return {"object": "block", "type": "code", "code": {"rich_text": [rt(s)], "language": lang}}


def toggle(s, kids):
    return {"object": "block", "type": "toggle", "toggle": {"rich_text": [rt(s)], "children": kids}}


def _cells(row):
    return [[rt(c)] if isinstance(c, str) and c else ([] if not c else c) for c in row]


def table(rows, header=True):
    return {"object": "block", "type": "table",
            "table": {"table_width": len(rows[0]), "has_column_header": header, "has_row_header": False,
                      "children": [{"object": "block", "type": "table_row",
                                    "table_row": {"cells": _cells(r)}} for r in rows]}}


def link_page(page_id):
    return {"object": "block", "type": "link_to_page",
            "link_to_page": {"type": "page_id", "page_id": page_id}}


def image_block(file_upload_id, caption=""):
    o = {"object": "block", "type": "image",
         "image": {"type": "file_upload", "file_upload": {"id": file_upload_id}}}
    if caption:
        o["image"]["caption"] = [rt(caption)]
    return o


# ------------------------------------------------------------------- readers
def text_of(b):
    t = b.get(b.get("type"))
    if not isinstance(t, dict):
        return ""
    return "".join(x.get("plain_text", "") for x in (t.get("rich_text") or []) if isinstance(x, dict))


def page_blocks(pid):
    out, cur = [], None
    while True:
        r = call("GET", f"/blocks/{pid}/children?page_size=100" + (f"&start_cursor={cur}" if cur else ""))
        if not ok(r):
            return out
        out += r["results"]
        if not r.get("has_more"):
            return out
        cur = r["next_cursor"]


def find_rows(db_id):
    r = call("POST", f"/databases/{db_id}/query",
             {"page_size": 100, "sorts": [{"timestamp": "created_time", "direction": "descending"}]})
    if not ok(r):
        return []
    out = []
    for row in r["results"]:
        ti = ""
        for v in row["properties"].values():
            if v.get("type") == "title":
                ti = "".join(x["plain_text"] for x in v["title"])
        out.append((ti, row["id"]))
    return out


def find_page(db_id, needle):
    """제목 부분일치로 페이지 1개. 없으면 None, 여러 개면 가장 최근 것."""
    for ti, pid in find_rows(db_id):
        if needle.lower() in ti.lower():
            return pid
    return None


# ------------------------------------------------- section-level dynamic edit
_H = ("heading_1", "heading_2", "heading_3")


def section_range(pid, heading_text, level=None):
    """(heading_id, [본문 블록 id...]) — 본문은 다음 동급 이상 heading 직전까지."""
    blocks = page_blocks(pid)
    start = None
    for i, b in enumerate(blocks):
        if b["type"] in _H and heading_text in text_of(b):
            if level and b["type"] != level:
                continue
            start = i
            break
    if start is None:
        return None, []
    lv = _H.index(blocks[start]["type"])
    body = []
    for b in blocks[start + 1:]:
        if b["type"] in _H and _H.index(b["type"]) <= lv:
            break
        body.append(b["id"])
    return blocks[start]["id"], body


def replace_section(pid, heading_text, new_blocks, level=None):
    """절 본문 통째 교체. **멱등** — 같은 스크립트를 다시 돌려도 중복이 안 생긴다."""
    hid, body = section_range(pid, heading_text, level)
    if hid is None:
        return {"_error": "no-such-heading", "_body": heading_text}
    for bid in body:
        call("DELETE", f"/blocks/{bid}")
    if new_blocks:
        return call("PATCH", f"/blocks/{pid}/children", {"children": new_blocks[:100], "after": hid})
    return {"ok": True}


def append_after(pid, after_id, blocks):
    return call("PATCH", f"/blocks/{pid}/children", {"children": blocks[:100], "after": after_id})


def append_end(pid, blocks):
    return call("PATCH", f"/blocks/{pid}/children", {"children": blocks[:100]})


def edit_text(bid, s, **kw):
    b = call("GET", f"/blocks/{bid}")
    if not ok(b):
        return b
    t = b["type"]
    return call("PATCH", f"/blocks/{bid}", {t: {"rich_text": [rt(s, **kw)]}})


def delete(bid):
    return call("DELETE", f"/blocks/{bid}")


# --------------------------------------------------------------- table edits
def table_rows(tid):
    return call("GET", f"/blocks/{tid}/children?page_size=100").get("results", [])


def set_row(row_id, cells):
    return call("PATCH", f"/blocks/{row_id}", {"table_row": {"cells": _cells(cells)}})


def add_rows(tid, rows):
    return call("PATCH", f"/blocks/{tid}/children",
                {"children": [{"object": "block", "type": "table_row",
                               "table_row": {"cells": _cells(r)}} for r in rows]})


def rebuild_table(tid, new_rows):
    """기존 행 덮어쓰기 + 부족분 추가. 행이 줄면 남는 행은 그대로 둔다(수동 확인)."""
    rows = table_rows(tid)
    for i, r in enumerate(rows):
        if i < len(new_rows):
            set_row(r["id"], new_rows[i])
    if len(new_rows) > len(rows):
        add_rows(tid, new_rows[len(rows):])
    return len(rows), len(new_rows)


def find_table(pid, header_contains):
    """헤더 행에 특정 문자열이 있는 표의 id."""
    for b in page_blocks(pid):
        if b["type"] == "table" and b.get("has_children"):
            rows = table_rows(b["id"])
            if not rows:
                continue
            head = " | ".join("".join(y.get("plain_text", "") for y in c)
                              for c in rows[0]["table_row"]["cells"])
            if header_contains in head:
                return b["id"]
    return None


# ------------------------------------------------------------------- uploads
def upload(path, name=None):
    """로컬 파일 → Notion file_upload id. 실패 시 dict."""
    name = name or os.path.basename(path)
    ct = mimetypes.guess_type(name)[0] or "application/octet-stream"
    fu = _req("POST", f"{BASE}/file_uploads",
              json.dumps({"filename": name, "content_type": ct}).encode(),
              {"Authorization": f"Bearer {TOKEN}", "Notion-Version": VER,
               "Content-Type": "application/json"})
    if not ok(fu):
        return fu
    blob = open(path, "rb").read()
    bd = "----n" + uuid.uuid4().hex
    body = (f"--{bd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ct}\r\n\r\n").encode() + blob + f"\r\n--{bd}--\r\n".encode()
    sent = _req("POST", fu["upload_url"], body,
                {"Authorization": f"Bearer {TOKEN}", "Notion-Version": VER,
                 "Content-Type": f"multipart/form-data; boundary={bd}"})
    return fu["id"] if ok(sent) else sent


MERMAID_CFG = {
    "theme": "base",
    "themeVariables": {"fontFamily": "DejaVu Sans, sans-serif", "fontSize": "15px",
                       "primaryColor": "#eef2f7", "primaryTextColor": "#1f2933",
                       "primaryBorderColor": "#9aa8bb", "lineColor": "#7b8794",
                       "secondaryColor": "#ffffff", "tertiaryColor": "#f7f9fc"},
    "flowchart": {"htmlLabels": True, "curve": "basis", "nodeSpacing": 40,
                  "rankSpacing": 55, "padding": 10},
}

# 대비 고정 classDef — 노션 라이트/다크 어느 쪽에서도 읽히도록 color 를 명시한다.
# (color 를 빼면 다크모드에서 밝은 배경 + 밝은 글자로 안 보인다 — 실제 사고)
CLASSDEFS = """
  classDef default fill:#f6f8fb,stroke:#8e9cb0,stroke-width:1px,color:#1f2933
  classDef new fill:#fff1c9,stroke:#b8860b,stroke-width:2px,color:#3b2f00
  classDef off fill:#edf0f4,stroke:#a8b2c0,stroke-width:1px,stroke-dasharray:4 3,color:#5d6773
  classDef bug fill:#fbdedc,stroke:#b3423f,stroke-width:2px,color:#5a1a18
  classDef old fill:#e6ebf2,stroke:#8e9cb0,stroke-width:1px,color:#39424e
"""


def mermaid_png(mmd_text, out_png):
    """mermaid 소스 → PNG. node(nvm) + mmdc 필요. 실패 시 None."""
    d = tempfile.mkdtemp()
    src = os.path.join(d, "d.mmd")
    body = mmd_text if "classDef" in mmd_text else mmd_text.rstrip() + "\n" + CLASSDEFS
    open(src, "w").write(body)
    json.dump(MERMAID_CFG, open(os.path.join(d, "cfg.json"), "w"))
    open(os.path.join(d, "pp.json"), "w").write('{"args":["--no-sandbox","--disable-gpu"]}')
    env = dict(os.environ)
    nvm = os.path.expanduser("~/.nvm/versions/node")
    if os.path.isdir(nvm):
        vs = sorted(os.listdir(nvm))
        if vs:
            env["PATH"] = f"{nvm}/{vs[-1]}/bin:" + env.get("PATH", "")
    try:
        subprocess.run(["npx", "-y", "@mermaid-js/mermaid-cli@11", "-i", src, "-o", out_png,
                        "-c", os.path.join(d, "cfg.json"), "-p", os.path.join(d, "pp.json"),
                        "-b", "white", "-s", "3", "-q"],
                       capture_output=True, env=env, timeout=180)
    except Exception:
        return None
    return out_png if os.path.exists(out_png) else None


def put_diagram(pid, after_id, mmd_text, caption, tmp_png=None):
    """mermaid 를 렌더해 [그림 + 접힌 소스 토글]을 after_id 뒤에 넣는다."""
    png = tmp_png or os.path.join(tempfile.mkdtemp(), "arch.png")
    if not mermaid_png(mmd_text, png):
        return {"_error": "mermaid-render-failed"}
    fid = upload(png)
    if not isinstance(fid, str):
        return fid
    return append_after(pid, after_id, [
        image_block(fid, caption),
        toggle("mermaid 소스 (수정용)", [code(mmd_text, "mermaid")]),
    ])


# --------------------------------------------------------------------- audit
FORBIDDEN_SRC = re.compile(
    r"claude_logs|monitor-log|arch-evolution|registry\.md|status/current|history-2026|"
    r"issues-and-fixes|ral-paper-plan|plan\.md|experiments/analysis|failure-keys|"
    r"decisions/|09_benchmark|fact_related|fact_experiments|nas_jm/Research|_paper_submission")
FORBIDDEN_TONE = re.compile(
    r"구조적 성과|유일한 성과|의미가 있다|주목할|흥미롭게도|무려|확연히|시사한다|뼈아프")


def audit(pid, deep=True):
    """금지 출처·금지 수사 검사. {'src': [...], 'tone': [...]}. 기록 후 반드시 실행."""
    hits = {"src": [], "tone": []}

    def scan(bid):
        for b in page_blocks(bid):
            s = text_of(b)
            if FORBIDDEN_SRC.search(s):
                hits["src"].append((b["id"], s[:90]))
            if FORBIDDEN_TONE.search(s):
                hits["tone"].append((b["id"], s[:90]))
            if b["type"] == "table" and b.get("has_children"):
                for row in table_rows(b["id"]):
                    flat = " ".join("".join(y.get("plain_text", "") for y in c)
                                    for c in row["table_row"]["cells"])
                    if FORBIDDEN_SRC.search(flat):
                        hits["src"].append((row["id"], flat[:90]))
            elif deep and b["type"] in ("toggle", "callout") and b.get("has_children"):
                scan(b["id"])
    scan(pid)
    return hits
