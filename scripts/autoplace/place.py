#!/usr/bin/env python3
"""place.py — 빈 GPU 자동 배치(계획 수립 및 선택적 기동).

scripts/autoplace/queue.tsv 의 대기열을 priority 오름차순으로 읽어,
probe_free_gpus.sh 로 조사한 각 서버의 빈 GPU 에 작업을 배정한다.
--launch 를 주면 배정된 작업을 ssh + tmux 로 실제 기동하고
state 파일(--state, 기본 scripts/autoplace/state/launched.tsv)에
"id\thost\tgpus\tISO시각\tsession" 한 줄을 append 한다.

사용법:
  python3 scripts/autoplace/place.py                    # 계획만 출력(아무것도 실행하지 않음)
  python3 scripts/autoplace/place.py --launch           # 실제 기동 + state 기록
  python3 scripts/autoplace/place.py --launch --limit 2 # 이번 회차 최대 2건

안전 규칙(위반 경로가 없도록 구현됨):
  - 빈 GPU 판정을 통과하지 못한 GPU 에는 절대 배치하지 않는다(판정은 probe 가 유일한 출처).
  - 같은 id 를 두 번 띄우지 않는다(state 파일에 이미 있는 id 는 건너뛴다).
  - servers.conf policy=off 서버는 probe 도 하지 않는다.
  - sudo, pip install, git 명령을 쓰지 않는다. 파일을 지우지 않는다(state 는 append 만).
"""
import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # scripts/autoplace -> 저장소 루트
DEFAULT_SERVERS_CONF = os.path.join(REPO_ROOT, "scripts", "servers.conf")
DEFAULT_QUEUE = os.path.join(SCRIPT_DIR, "queue.tsv")
DEFAULT_STATE = os.path.join(SCRIPT_DIR, "state", "launched.tsv")
DEFAULT_HOSTENV = os.path.join(SCRIPT_DIR, "hostenv.tsv")
PROBE_SCRIPT = os.path.join(SCRIPT_DIR, "probe_free_gpus.sh")


def die(msg):
    print(f"place: 오류 — {msg}", file=sys.stderr)
    sys.exit(1)


def parse_servers_conf(path):
    """servers.conf(파이프 구분 6열)를 읽어 {alias: 정보} 반환. policy=off 서버는 제외."""
    servers = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = [x.strip() for x in line.rstrip("\n").split("|")]
            alias = fields[0]
            if not alias:
                continue
            policy = fields[5] if len(fields) >= 6 else ""
            policy = policy.strip()
            if policy == "off":
                continue
            banned = []
            if policy.startswith("ban:"):
                banned = [b.strip() for b in policy[len("ban:"):].split(",") if b.strip()]
            elif policy not in ("", "-", "none"):
                print(f"place: 주의 — '{alias}' 의 알 수 없는 policy '{policy}' 는 무시함({path}:{lineno})",
                      file=sys.stderr)
            servers[alias] = {
                "repo": fields[1] if len(fields) > 1 else "",
                "env": fields[2] if len(fields) > 2 else "",
                "banned": set(banned),
            }
    return servers


def parse_hostenv(path):
    """hostenv.tsv(탭 4열)를 읽어 {host: (conda_sh, pylibs, port_base)} 반환.

    FILL_ME 가 하나라도 있는 host 는 결과에서 제외한다(배치 금지).
    파일 자체가 없으면 빈 dict(전 서버 배치 불가)로 동작한다.
    """
    envs = {}
    if not os.path.isfile(path):
        print(f"place: 주의 — hostenv 파일이 없음: {path} (모든 서버 배치 불가)", file=sys.stderr)
        return envs
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                print(f"place: 주의 — hostenv {lineno}번째 줄이 4열이 아님, 건너뜀: {stripped!r}", file=sys.stderr)
                continue
            host, conda_sh, pylibs, port_base = [x.strip() for x in fields]
            if any(v == "FILL_ME" or v == "" for v in (conda_sh, port_base)):
                continue
            if pylibs in ("FILL_ME", ""):
                pylibs = ""  # pylibs 는 선택값: 없으면 PYTHONPATH 접두어를 생략
            try:
                base = int(port_base)
            except ValueError:
                print(f"place: 주의 — hostenv '{host}' 의 master_port_base 가 정수 아님('{port_base}'), 제외",
                      file=sys.stderr)
                continue
            envs[host] = {"conda_sh": conda_sh, "pylibs": pylibs, "port_base": base}
    return envs


def parse_queue(path):
    """queue.tsv(탭 9열)를 읽어 작업 목록을 반환. priority 오름차순, 같으면 파일 순서."""
    jobs = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[0].strip() == "id":
                continue  # 헤더 줄
            if len(fields) != 9:
                die(f"queue {lineno}번째 줄이 9열이 아님({len(fields)}열): {stripped!r}")
            jid, priority, hosts, ngpu, repo, config, session, epochs, note = [x.strip() for x in fields]
            if not jid:
                die(f"queue {lineno}번째 줄의 id 가 비어 있음")
            if jid in seen:
                print(f"place: 주의 — queue 의 id 중복 '{jid}'({lineno}번째 줄), 첫 행만 사용", file=sys.stderr)
                continue
            seen.add(jid)
            try:
                prio = int(priority)
            except ValueError:
                die(f"queue '{jid}' 의 priority 가 정수가 아님: {priority!r}")
            if ngpu not in ("1", "2"):
                die(f"queue '{jid}' 의 ngpu 는 1 또는 2 여야 함: {ngpu!r}")
            jobs.append({
                "id": jid, "priority": prio, "hosts": hosts, "ngpu": int(ngpu),
                "repo": repo, "config": config, "session": session,
                "epochs": epochs, "note": note,
            })
    jobs.sort(key=lambda j: j["priority"])  # sort 는 안정적: 동일 priority 는 파일 순서 유지
    return jobs


def parse_state_ids(path):
    """state 파일에서 이미 기동된 id 집합을 반환. 파일이 없으면 빈 집합."""
    ids = set()
    if not os.path.isfile(path):
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ids.add(stripped.split("\t")[0].strip())
    return ids


def probe_free(server):
    """probe_free_gpus.sh 를 호출해 빈 GPU 인덱스 목록(문자열 리스트)을 반환. 실패 시 None."""
    proc = subprocess.run(["bash", PROBE_SCRIPT, server], capture_output=True, text=True)
    if proc.returncode != 0:
        reason = (proc.stderr.strip() or "(출력 없음)").splitlines()[0]
        print(f"place: 주의 — '{server}' 프로브 실패, 이번 회차에서 제외: {reason}", file=sys.stderr)
        return None
    return [tok for tok in proc.stdout.strip().split() if tok]


def build_remote_command(job, host_info, hostenv, gpus):
    """지정된 형식의 tmux 기동 명령(원격 셸 문자열)을 만든다."""
    q = shlex.quote
    env_name = host_info["env"]
    port = hostenv["port_base"] + int(gpus[0])
    gpus_str = ",".join(gpus)
    repo = job["repo"]
    log_path = f"{repo}/logs/{job['session']}_launch.log"
    pythonpath = ":".join(p for p in (hostenv["pylibs"], repo, f"{repo}/semseg/models/sam2") if p)
    inner = (
        f"cd {q(repo)} && source {q(hostenv['conda_sh'])} && conda activate {q(env_name)} && "
        f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && "
        f"export PYTHONPATH={pythonpath} && "
        f"export CUDA_VISIBLE_DEVICES={gpus_str} && "
        f"torchrun --standalone --nproc_per_node={job['ngpu']} --master_port={port} "
        f"train_reliadino.py --cfg {q(job['config'])} > {q(log_path)} 2>&1"
    )
    return f"tmux new-session -d -s {q(job['session'])} {q(inner)}", port, log_path


def print_table(rows, header):
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))


def main():
    ap = argparse.ArgumentParser(description="빈 GPU 자동 배치(계획/기동)")
    ap.add_argument("--queue", default=DEFAULT_QUEUE, help="대기열 tsv (기본: %(default)s)")
    ap.add_argument("--state", default=DEFAULT_STATE, help="기동 기록 tsv (기본: %(default)s)")
    ap.add_argument("--launch", action="store_true", help="실제로 ssh+tmux 기동 (없으면 계획만 출력)")
    ap.add_argument("--limit", type=int, default=4, help="이번 회차 최대 배치 수 (기본: %(default)s)")
    ap.add_argument("--servers-conf", default=DEFAULT_SERVERS_CONF, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if not os.path.isfile(args.servers_conf):
        die(f"servers.conf 없음: {args.servers_conf}")
    if not os.path.isfile(args.queue):
        die(f"대기열 파일 없음: {args.queue}")

    servers = parse_servers_conf(args.servers_conf)
    hostenv = parse_hostenv(DEFAULT_HOSTENV)
    jobs = parse_queue(args.queue)
    launched = parse_state_ids(args.state)

    # 1) 서버별 빈 GPU 조사 (policy=off 서버는 parse 단계에서 이미 제외됨)
    free = {}
    for alias in servers:
        gpus = probe_free(alias)
        if gpus is None:
            continue
        # hostenv 미등록/FILL_ME, conda_env 미상(FILL_ME) 서버에는 배치할 수 없다
        if alias not in hostenv:
            print(f"place: 주의 — '{alias}' 은(는) hostenv.tsv 에 사용 가능한 값이 없어 배치 후보에서 제외",
                  file=sys.stderr)
            continue
        if servers[alias]["env"] in ("", "FILL_ME"):
            print(f"place: 주의 — '{alias}' 은(는) servers.conf conda_env 미상(FILL_ME)이라 배치 후보에서 제외",
                  file=sys.stderr)
            continue
        free[alias] = [g for g in gpus if g not in servers[alias]["banned"]]

    # 2) 대기열을 priority 순으로 배정
    assignments = []   # (job, host, gpus, port)
    skipped = []       # (job, 사유)
    for job in jobs:
        if job["id"] in launched:
            skipped.append((job, "이미 기동됨(state)"))
            continue
        if len(assignments) >= args.limit:
            skipped.append((job, f"이번 회차 한도({args.limit}) 초과"))
            continue
        want = job["hosts"].split(",") if job["hosts"] != "any" else list(free.keys())
        placed = None
        for host in want:
            host = host.strip()
            if not host:
                continue
            if host not in servers:
                continue  # policy=off 이거나 registry 에 없는 서버
            if host in free and len(free[host]) >= job["ngpu"]:
                placed = host
                break
        if placed is None:
            skipped.append((job, f"빈 GPU 부족 또는 hosts 제약(hosts={job['hosts']})"))
            continue
        gpus = free[placed][: job["ngpu"]]
        free[placed] = free[placed][job["ngpu"]:]
        port = hostenv[placed]["port_base"] + int(gpus[0])
        assignments.append((job, placed, gpus, port))

    # 3) 결과 출력
    print(f"== [autoplace] 배치 계획 (모드: {'기동' if args.launch else '계획만'}, 한도 {args.limit}) ==")
    if assignments:
        print_table(
            [(j["id"], j["priority"], h, ",".join(g), p, j["session"]) for j, h, g, p in assignments],
            ["id", "prio", "host", "gpus", "port", "session"],
        )
    else:
        print("(배정 가능한 작업 없음)")
    if skipped:
        print("-- 건너뛴 항목 --")
        for j, why in skipped:
            print(f"  {j['id']}: {why}")

    if not args.launch:
        print("== 계획만 출력 (--launch 를 주면 실제 기동) ==")
        return 0

    # 4) 기동 + state 기록
    os.makedirs(os.path.dirname(args.state), exist_ok=True)
    failures = 0
    for job, host, gpus, port in assignments:
        remote, port, log_path = build_remote_command(job, servers[host], hostenv[host], gpus)
        print(f">> {host}: '{job['id']}' 기동 (session={job['session']} gpus={','.join(gpus)} port={port})")
        print(f"   로그: {log_path}")
        proc = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            failures += 1
            print(f"place: 기동 실패 '{job['id']}' @ {host} (ssh exit {proc.returncode})",
                  file=sys.stderr)
            if proc.stdout.strip():
                print(f"  stdout: {proc.stdout.strip()[:500]}", file=sys.stderr)
            if proc.stderr.strip():
                print(f"  stderr: {proc.stderr.strip()[:500]}", file=sys.stderr)
            continue  # state 에 기록하지 않는다 → 다음 회차에 재시도된다
        now = datetime.now().isoformat(timespec="seconds")
        with open(args.state, "a", encoding="utf-8") as f:
            f.write(f"{job['id']}\t{host}\t{','.join(gpus)}\t{now}\t{job['session']}\n")
        print(f"   기동 완료, state 기록: {args.state}")

    if failures:
        print(f"place: {failures}건 기동 실패 — 위 stderr 참고", file=sys.stderr)
        return 1
    print(f"== [autoplace] 기동 완료: {len(assignments) - failures}건 ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
