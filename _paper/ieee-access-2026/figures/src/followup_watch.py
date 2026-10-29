"""Watch notes/followups.yaml and tell the paper session when a follow-up is ready to apply.

A follow-up is ready when every dependency is finished:
  DRN-...   lab-plan item with a recorded verdict, or status done/withdrawn/superseded
            (.claude_logs/plan/plan.yaml)
  digits... lab-inbox message whose status is done (.claude_logs/inbox/**/<id>.md)
  FU-..     another follow-up whose status in followups.yaml is done
For each ready follow-up with status open, one lab-inbox request goes to specialist-paper with the
dependency outcomes and the matching branch text. Sent follow-ups are remembered in a state file
outside git (<repo>/.watchdog/paper_followups_state.json), keyed by follow-up id + outcomes, so a new
verdict on the same follow-up triggers a new message but repeated runs do not.

Usage (repo = the hub checkout that lab-plan and lab-inbox write to):
  python3 followup_watch.py --repo <repo>            # report + send
  python3 followup_watch.py --repo <repo> --dry-run  # report only
  python3 followup_watch.py --repo <repo> --ref origin/develop   # read followups.yaml from a git ref
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REL = "_paper/ieee-access-2026/notes/followups.yaml"


def load_registry(repo, ref):
    if ref:
        text = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{REL}"],
                              check=True, capture_output=True, text=True).stdout
    else:
        text = (repo / REL).read_text(encoding="utf-8")
    return yaml.safe_load(text)["items"]


def plan_outcomes(repo):
    data = yaml.safe_load((repo / ".claude_logs/plan/plan.yaml").read_text(encoding="utf-8"))
    out = {}
    for it in data.get("items", []):
        v = it.get("verdict") or {}
        out[it["id"]] = (it.get("status"), v.get("decision"), v.get("summary", ""))
    return out


def inbox_status(repo, msg_id):
    for p in (repo / ".claude_logs/inbox").rglob(f"{msg_id}.md"):
        m = re.search(r"^status:\s*(\S+)", p.read_text(encoding="utf-8"), re.M)
        return m.group(1) if m else None
    return None


def dep_outcome(repo, dep, plan, fu_status):
    """Return (finished, outcome label, detail)."""
    if dep.startswith("FU-"):
        st = fu_status.get(dep)
        return st == "done", st or "unknown", ""
    if re.match(r"^\d", dep):
        st = inbox_status(repo, dep)
        return st == "done", st or "not found", ""
    st, decision, summary = plan.get(dep, (None, None, ""))
    if st is None:
        return False, "not in plan", ""
    # measurement-only items may close as done without a verdict
    return decision is not None or st in ("done", "withdrawn", "superseded"), decision or st, summary


def message(fu, outcomes):
    lines = [f"## 원고 팔로업 {fu['id']}: {fu['title']}", "",
             "followup_watch.py 자동 통지. 의존 항목이 모두 끝났다. 아래 결과에 맞는 분기대로 원고를 고치고, "
             f"notes/followups.yaml 의 {fu['id']} status 를 done 으로 바꿔 같은 커밋에 넣는다.", "",
             "### 의존 항목 결과"]
    for dep, (_, label, summary) in outcomes.items():
        lines.append(f"- {dep}: {label}" + (f" — {summary[:300]}" if summary else ""))
    lines += ["", "### 원고 위치"] + [f"- {w}" for w in fu.get("where", [])]
    lines += ["", "### 지금 원고에 둔 서술", fu.get("now", ""), "", "### 결과별 수정"]
    for k, v in (fu.get("branches") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "판정이 여러 개면 method 의 판정 요약을 기준으로 분기를 고른다. 분기가 맞지 않으면 method 에 묻는다."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    items = load_registry(repo, args.ref)
    plan = plan_outcomes(repo)
    fu_status = {it["id"]: it.get("status", "open") for it in items}
    state_path = repo / ".watchdog/paper_followups_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    for fu in items:
        if fu.get("status", "open") == "done":
            print(f"{fu['id']} done")
            continue
        outcomes = {d: dep_outcome(repo, d, plan, fu_status) for d in fu.get("depends_on") or []}
        pending = [d for d, o in outcomes.items() if not o[0]]
        if pending:
            print(f"{fu['id']} waiting: " + ", ".join(f"{d}({outcomes[d][1]})" for d in pending))
            continue
        sig = hashlib.md5(json.dumps({d: o[1] for d, o in outcomes.items()}, sort_keys=True)
                          .encode()).hexdigest()[:8]
        if state.get(fu["id"]) == sig:
            print(f"{fu['id']} ready, already notified ({sig})")
            continue
        print(f"{fu['id']} READY → notify specialist-paper ({sig})")
        if args.dry_run:
            continue
        r = subprocess.run(["lab-inbox", "send", str(repo), "--to", "specialist-paper", "--type", "request",
                            "--ref", fu["id"], "--dedup-key", f"paper-followup-{fu['id']}-{sig}",
                            "--body", message(fu, outcomes)], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode == 0:
            state[fu["id"]] = sig
            state_path.parent.mkdir(exist_ok=True)
            state_path.write_text(json.dumps(state, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
