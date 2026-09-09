"""실험노트 DB(실험당 1페이지) 일괄 생성·갱신기 (2026-09-08).

입력: JSON 배열 파일들(exp_*.json) — 각 원소가 실험 하나(스키마는 EXP_SCHEMA_KEYS 참조).
동작: 제목의 실험 식별자(key)로 기존 페이지를 찾고, 있으면 본문 전체 교체, 없으면 DB 행 생성.
       페이지 구조 = 요약 callout / [무효 callout] / 배경 / 제안 구조(도면·구성요소·근거·세팅) / 결과 / 판정·분석 / 출처.
사용: EXP_JSON_DIR=<dir> python exp_pages_builder.py [--only P46,P50] [--dry]
"""
import os, sys, re, glob, json, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notion_api import *  # noqa

DB = "8ec54838-faff-4534-bc05-590dcebcc21a"          # 실험노트
BM = "3a388e43-5b22-813d-829b-e55ae2ca3a77"          # 📊 벤치마크 (누적)
PAPER = "33d05310-a165-408a-b0b8-ec4427d1fe2c"       # My Paper
PROJ = "f1a55348-9c2f-443d-8187-5c8174c1cfbf"        # 🌐 Project Pages
JSON_DIR = os.environ.get("EXP_JSON_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_exp_json"))
EXP_SCHEMA_KEYS = ["key", "title", "period", "track", "status", "summary", "background", "architecture", "results", "analysis", "verdict", "todo", "sources"]

# 내부 문서명 → 중립 표현 (랩 공용 노션에 내부 로그 경로가 새지 않도록; audit() 금지 패턴과 짝)
_SANITIZE = [
    (re.compile(r"\.claude_logs/?"), "레포 로그 "),
    (re.compile(r"models/arch-evolution\.md|arch-evolution\.md|arch-evolution"), "레포 아키텍처 문서"),
    (re.compile(r"experiments/monitor-log\.md|monitor-log\.md|monitor-log"), "레포 모니터 로그"),
    (re.compile(r"issues/issues-and-fixes\.md|issues-and-fixes\.md|issues-and-fixes"), "레포 이슈 문서"),
    (re.compile(r"experiments/registry\.md|registry\.md"), "레포 실험 레지스트리"),
    (re.compile(r"status/history-2026H[12]\.md|history-2026H[12]\.md|status/history|history-2026"), "레포 이력 문서"),
    (re.compile(r"status/current\.md|status/current"), "레포 현재 상태 문서"),
    (re.compile(r"experiments/log\.md"), "레포 실험 로그"),
    (re.compile(r"experiments/plan\.md|plan\.md"), "레포 실험 계획 문서"),
    (re.compile(r"experiments/analysis/[\w\-\.]+"), "레포 분석 문서"),
    (re.compile(r"experiments/analysis"), "레포 분석 문서"),
    (re.compile(r"decisions/[\w\-\.]+"), "레포 제안 문서"),
    (re.compile(r"decisions/"), "레포 제안 문서 "),
    (re.compile(r"failure-keys|failure-analysis"), "실패 분석 문서"),
    (re.compile(r"ral-paper-plan|09_benchmark|fact_related|fact_experiments|nas_jm/Research|_paper_submission"), "레포 리서치 문서"),
]

def sanitize(x):
    if isinstance(x, str):
        for pat, rep in _SANITIZE:
            x = pat.sub(rep, x)
        return x
    if isinstance(x, list): return [sanitize(i) for i in x]
    if isinstance(x, dict): return {k: (v if k == "_internal_refs" else sanitize(v)) for k, v in x.items()}
    return x

def _s(x):
    if x is None: return "기록 없음"
    if isinstance(x, (list, tuple)): return " · ".join(_s(i) for i in x) if x else "기록 없음"
    if isinstance(x, dict): return "; ".join(f"{k}: {_s(v)}" for k, v in x.items())
    return str(x)

def _lst(x):
    if not x or x == "기록 없음": return []
    if isinstance(x, str): return [x]
    return [(_s(i) if not isinstance(i, str) else i) for i in x]

def _clip(s, n=1900):
    s = str(s); return s if len(s) <= n else s[: n - 1] + "…"

def _tbl(rows):
    rows = [r for r in rows if isinstance(r, (list, tuple)) and r]
    if len(rows) < 2: return None
    w = max(len(r) for r in rows)
    norm = [[_clip(str(c)) for c in list(r) + [""] * (w - len(r))] for r in rows]
    return table(norm)

def bul_list(items):
    return [bul(rt(_clip(i))) for i in _lst(items)]

def find_existing(key):
    # 제목 접두어 "[…] KEY" 위치에서만 매칭 (본문 언급 "E1(4탭)" 같은 것에 걸리지 않게), key 뒤는 영숫자·점·하이픈 금지
    pat = re.compile(rf"^\[[^\]]*\]\s*(?:{re.escape(key)})(?![\w.\-])")
    for ti, pid in find_rows(DB):
        if pat.search(ti):
            return pid, ti
    return None, None

def create_row(title):
    r = call("POST", "/pages", {
        "parent": {"database_id": DB},
        "properties": {
            "Experiments": {"title": [rt(title)]},
            "My Paper": {"relation": [{"id": PAPER}]},
            "🌐 Project Pages": {"relation": [{"id": PROJ}]},
        },
        "children": []})
    if not ok(r):
        print("create failed", r); return None
    return r["id"]

def set_title(pid, title):
    call("PATCH", f"/pages/{pid}", {"properties": {"Experiments": {"title": [rt(title)]}}})

def diagram_blocks(mmd, caption):
    """mermaid → [image, toggle(source)]; 렌더 실패 시 code 블록만."""
    if not mmd or mmd == "기록 없음" or not str(mmd).strip().startswith("flowchart"):
        return []
    png = os.path.join(tempfile.mkdtemp(), "arch.png")
    out = []
    if mermaid_png(mmd, png):
        fid = upload(png)
        if isinstance(fid, str):
            out.append(image_block(fid, caption))
    out.append(toggle("mermaid 소스 (수정용)", [code(_clip(mmd), "mermaid")]))
    return out

def build_blocks(e):
    S = e.get("summary", {}) or {}
    A = e.get("architecture", {}) or {}
    R = e.get("results", {}) or {}
    V = e.get("verdict", {}) or {}
    b = []
    b.append(callout([rt("변경: ", bold=True), rt(_clip(_s(S.get("change")))), rt("\n결과: ", bold=True), rt(_clip(_s(S.get("result")))),
                      rt("\n판정: ", bold=True), rt(_clip(_s(S.get("verdict")))),
                      rt(f"\n기간 {_s(e.get('period'))} · 트랙 {_s(e.get('track'))} · 상태 {_s(e.get('status'))}")], "🎯", "blue_background"))
    inv = _lst(R.get("invalid"))
    if inv:
        b.append(callout([rt("무효·강등 수치(논문 사용 금지, 격리 보관): ", bold=True), rt(_clip(" / ".join(inv)))], "🚩", "red_background"))
    # 1 배경
    b.append(h1("배경 — 직전까지의 문제")); b += bul_list(e.get("background")) or [para(rt("기록 없음"))]
    # 2 제안 구조
    b.append(h1("제안 구조"))
    b.append(para(rt(_clip(_s(A.get("one_liner"))), bold=True)))
    b += diagram_blocks(A.get("mermaid"), f"{e.get('key')} 구조 — 노랑 = 이 버전에서 추가·변경된 노드")
    comps = A.get("components") or []
    t = _tbl([["변경 ID / 모듈", "내용", "코드 경로"]] + [list(c) for c in comps if isinstance(c, (list, tuple))])
    if t: b.append(h3("구성 요소")); b.append(t)
    if _lst(A.get("rationale")): b.append(h3("근거 사슬 — 왜 이 설계인가")); b += bul_list(A.get("rationale"))
    if _lst(A.get("setting")): b.append(h3("구현·세팅")); b += bul_list(A.get("setting"))
    # 3 결과
    b.append(h1("결과"))
    b.append(para(rt("게이트: ", bold=True), rt(_clip(_s(R.get("gate"))))))
    t = _tbl(R.get("table") or [])
    if t: b.append(t)
    if _lst(R.get("per_condition")): b.append(h3("조건별")); b += bul_list(R.get("per_condition"))
    if _lst(R.get("per_class")): b.append(h3("클래스별")); b += bul_list(R.get("per_class"))
    b.append(para(rt("전 버전·선행연구 누적 비교는 벤치마크 페이지 참조 →")))
    b.append(link_page(BM))
    # 4 판정
    b.append(h1("판정"))
    b.append(bul(rt("작동했나: ", bold=True), rt(_clip(_s(V.get("worked"))))))
    b.append(bul(rt("원인·해석: ", bold=True), rt(_clip(_s(V.get("why"))))))
    b.append(bul(rt("다음으로 넘긴 것: ", bold=True), rt(_clip(_s(V.get("handoff"))))))
    if _lst(e.get("analysis")): b.append(h2("결과 분석 (표준분석·모듈 진단)")); b += bul_list(e.get("analysis"))
    if _lst(e.get("todo")):
        b.append(callout([rt("미기록·확인 필요: ", bold=True), rt(_clip(" / ".join(_lst(e.get("todo")))))], "⚠️", "yellow_background"))
    # 5 출처
    b.append(callout([rt("출처(코드·config·ckpt·서버·커밋만): ", bold=True), rt(_clip(" · ".join(_lst(e.get("sources")))))], "📎", "gray_background"))
    b.append(para(rt(f"이 페이지는 레포 정본 문서에서 생성됨(생성기 exp_pages_builder.py, {time.strftime('%Y-%m-%d')}). 수치 갱신 시 레포 먼저, 노션은 재생성.")))
    return b

def upsert(e, dry=False):
    e = sanitize(e)
    key, title = e["key"], e["title"]
    # 제목 접두어가 key와 다르면 JSON의 "match"(예: "DGFusion Swin-T")로 찾는다
    pid, old = find_existing(e.get("match") or key)
    blocks = build_blocks(e)
    if dry:
        print(("UPDATE " if pid else "CREATE ") + title, "| blocks", len(blocks)); return pid
    if pid is None:
        pid = create_row(title)
        if pid is None: return None
        print("created", title)
    else:
        for bl in page_blocks(pid):
            call("DELETE", f"/blocks/{bl['id']}")
        if old != title: set_title(pid, title)
        print("replaced", title, "(구 제목:", old, ")")
    for i in range(0, len(blocks), 100):
        r = call("PATCH", f"/blocks/{pid}/children", {"children": blocks[i:i+100]})
        if not ok(r): print("  append failed", r); break
        time.sleep(0.3)
    hits = audit(pid)
    if hits["src"] or hits["tone"]:
        print("  AUDIT HIT", hits)
    return pid

def main():
    only = None; dry = "--dry" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--only"): only = set(a.split("=", 1)[1].split(","))
    exps = []
    for f in sorted(glob.glob(os.path.join(JSON_DIR, "exp_*.json"))):
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception as ex:
            print("JSON load failed", f, ex); continue
        exps += data if isinstance(data, list) else [data]
    seen = set()
    for e in exps:
        if not isinstance(e, dict) or "key" not in e: continue
        if only and e["key"] not in only: continue
        if e["key"] in seen: continue
        seen.add(e["key"])
        try:
            upsert(e, dry)
        except Exception as ex:
            print("FAILED", e.get("key"), ex)
        time.sleep(0.3)
    print("done", len(seen))

if __name__ == "__main__":
    main()
