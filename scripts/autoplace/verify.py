#!/usr/bin/env python3
"""verify.py — 최근 자동 배치된 작업이 살아 있는지 검증한다.

state 파일(place.py 가 기록한 launched.tsv)에서 최근 N 분 안에 기록된 항목마다
그 서버의 <repo>/logs/<session>_launch.log 를 ssh 로 읽어 네 가지를 판정한다.
  (1) 파라미터 마커: 'total_trainable' 또는 'lora_params_total' 줄이 있는가
  (2) 치명 오류 없음: 'Traceback|OutOfMemory|Error' 가 없는가
  (3) 학습 전진: 'Epoch [' 진행 줄이 있고 반복 인덱스가 0 이 아닌가
  (4) GPU 점유: 배정 GPU 각각의 memory.used 가 3000MiB 이상인가

repo 경로는 queue.tsv 의 같은 id 행에서 읽고, 없으면 servers.conf 의
해당 서버 repo_path 로 대체한다.

사용법:
  python3 scripts/autoplace/verify.py
  python3 scripts/autoplace/verify.py --since-min 60
  python3 scripts/autoplace/verify.py --repo-log-name muses_p52_e1_s902

종료코드: FAIL 이 하나라도 있으면 1, 아니면 0.
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_STATE = os.path.join(SCRIPT_DIR, "state", "launched.tsv")
DEFAULT_QUEUE = os.path.join(SCRIPT_DIR, "queue.tsv")
DEFAULT_SERVERS_CONF = os.path.join(REPO_ROOT, "scripts", "servers.conf")

SSH_OPTS = ["-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]

ERROR_PATTERN = re.compile(r"Traceback|OutOfMemory|Error")
EPOCH_ITER_PATTERN = re.compile(r"Epoch[^\n]*?\[\s*(\d+)\s*/\s*\d+\s*\]")


def ssh_run(host, remote_cmd):
    """ssh 명령을 실행해 (rc, stdout, stderr) 를 반환한다."""
    proc = subprocess.run(["ssh", *SSH_OPTS, host, remote_cmd], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def load_queue_repos(queue_path):
    """queue.tsv 에서 {id: repo} 반환. 없는 id 는 제외."""
    repos = {}
    if not os.path.isfile(queue_path):
        return repos
    with open(queue_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 5 and fields[0].strip() != "id":
                repos[fields[0].strip()] = fields[4].strip()
    return repos


def load_server_repos(conf_path):
    """servers.conf 에서 {alias: repo_path} 반환."""
    repos = {}
    if not os.path.isfile(conf_path):
        return repos
    with open(conf_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = [x.strip() for x in line.rstrip("\n").split("|")]
            if fields[0]:
                repos[fields[0]] = fields[1] if len(fields) > 1 else ""
    return repos


def load_state(state_path, since_min, session_filter):
    """state 파일에서 최근 since_min 분 안에 기록된 항목을 반환.

    반환: [(id, host, gpus, 시각문자열, session)] — 시각 파싱 실패 행은 경고 후 제외.
    """
    entries = []
    if not os.path.isfile(state_path):
        return entries
    cutoff = datetime.now() - timedelta(minutes=since_min)
    with open(state_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split("\t")
            if len(fields) != 5:
                print(f"verify: 주의 — state {lineno}번째 줄이 5열이 아님, 건너뜀: {stripped!r}",
                      file=sys.stderr)
                continue
            jid, host, gpus, ts, session = [x.strip() for x in fields]
            try:
                when = datetime.fromisoformat(ts)
            except ValueError as exc:
                print(f"verify: 주의 — state '{jid}' 시각 파싱 실패({ts!r}: {exc}), 건너뜀", file=sys.stderr)
                continue
            if when < cutoff:
                continue
            if session_filter and session != session_filter:
                continue
            entries.append((jid, host, [g for g in gpus.split(",") if g], ts, session))
    return entries


def check_log(host, log_path):
    """로그 기반 3개 판정. 반환: [(항목명, PASS/FAIL, 근거), ...]"""
    q = shlex.quote
    remote = (
        f"f={q(log_path)}; "
        f"if [ -f \"$f\" ]; then "
        f"grep -m1 -E 'total_trainable|lora_params_total' \"$f\"; echo __SEP__; "
        f"s=$(grep -anE 'total_trainable|lora_params_total' \"$f\" | tail -1 | cut -d: -f1); [ -z \"$s\" ] && s=1; "
        f"tail -n +$s \"$f\" | grep -m1 -aE 'Traceback|OutOfMemory|Error'; echo __SEP__; "
        f"grep -E 'Epoch \\[' \"$f\" | tail -5; "
        f"else echo __NOFILE__; fi"
    )
    rc, out, err = ssh_run(host, remote)
    if rc != 0:
        reason = (err.strip() or out.strip() or f"ssh exit {rc}")[:200]
        return [
            ("파라미터 마커", "FAIL", f"ssh 실패: {reason}"),
            ("치명 오류 없음", "FAIL", f"ssh 실패: {reason}"),
            ("학습 전진(Epoch)", "FAIL", f"ssh 실패: {reason}"),
        ]
    if "__NOFILE__" in out:
        why = f"로그 파일 없음: {log_path}"
        return [
            ("파라미터 마커", "FAIL", why),
            ("치명 오류 없음", "FAIL", why),
            ("학습 전진(Epoch)", "FAIL", why),
        ]
    parts = out.split("__SEP__")
    marker_line = parts[0].strip() if len(parts) > 0 else ""
    error_line = parts[1].strip() if len(parts) > 1 else ""
    epoch_lines = [l for l in (parts[2].splitlines() if len(parts) > 2 else []) if l.strip()]
    stderr_note = f" (ssh stderr: {err.strip()[:120]})" if err.strip() else ""

    results = []
    if marker_line:
        results.append(("파라미터 마커", "PASS", marker_line[:120]))
    else:
        results.append(("파라미터 마커", "FAIL", "total_trainable/lora_params_total 줄을 아직 못 찾음" + stderr_note))

    if error_line:
        results.append(("치명 오류 없음", "FAIL", error_line[:160]))
    else:
        results.append(("치명 오류 없음", "PASS", "Traceback/OutOfMemory/Error 일치 없음"))

    max_iter = 0
    last_line = ""
    for line in epoch_lines:
        m = EPOCH_ITER_PATTERN.search(line)
        if m:
            max_iter = max(max_iter, int(m.group(1)))
            last_line = line.strip()
    if max_iter > 0:
        results.append(("학습 전진(Epoch)", "PASS", f"반복 인덱스 최대 {max_iter} | {last_line[:100]}"))
    else:
        seen = last_line or (epoch_lines[0].strip() if epoch_lines else "")
        results.append(("학습 전진(Epoch)", "FAIL",
                        f"반복 인덱스 0 이하 또는 Epoch 줄 없음 | {seen[:100]}" if seen
                        else "'Epoch [' 진행 줄이 로그에 없음"))
    return results


def check_gpu_memory(host, gpus):
    """배정 GPU 전부가 3000MiB 이상인지 판정. 반환: (PASS/FAIL, 근거)."""
    remote = (f"nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits "
              f"-i {','.join(gpus)}")
    rc, out, err = ssh_run(host, remote)
    if rc != 0:
        reason = (err.strip() or out.strip() or f"ssh exit {rc}")[:200]
        return ("FAIL", f"ssh/nvidia-smi 실패: {reason}")
    used = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                used[parts[0]] = int(parts[1])
            except ValueError:
                continue
    missing = [g for g in gpus if g not in used]
    if missing:
        return ("FAIL", f"쿼리에서 GPU 누락: {','.join(missing)} | 출력: {out.strip()[:160]}")
    low = [g for g in gpus if used[g] < 3000]
    detail = " ".join(f"GPU{g}={used[g]}MiB" for g in gpus)
    if low:
        return ("FAIL", f"3000MiB 미만: {','.join(low)} | {detail}")
    return ("PASS", detail)


def main():
    ap = argparse.ArgumentParser(description="최근 자동 배치 작업 검증")
    ap.add_argument("--state", default=DEFAULT_STATE, help="state tsv (기본: %(default)s)")
    ap.add_argument("--since-min", type=int, default=30, help="최근 N 분 이내 항목만 검사 (기본: %(default)s)")
    ap.add_argument("--repo-log-name", default=None,
                    help="지정하면 이 세션 이름을 가진 항목만 검사 (선택)")
    args = ap.parse_args()

    if not os.path.isfile(args.state):
        print(f"verify: state 파일 없음(검사할 항목 없음): {args.state}")
        return 0

    entries = load_state(args.state, args.since_min, args.repo_log_name)
    if not entries:
        print(f"verify: 최근 {args.since_min} 분 안에 기록된 항목 없음"
              + (f" (세션 필터: {args.repo_log_name})" if args.repo_log_name else ""))
        return 0

    queue_repos = load_queue_repos(DEFAULT_QUEUE)
    server_repos = load_server_repos(DEFAULT_SERVERS_CONF)

    n_fail = 0
    print(f"== [autoplace] 검증 (최근 {args.since_min}분, {len(entries)}건) ==")
    for jid, host, gpus, ts, session in entries:
        repo = queue_repos.get(jid) or server_repos.get(host, "")
        print(f"\n-- {jid} @ {host} gpus={','.join(gpus)} session={session} launched={ts}")
        if not repo:
            print("   FAIL  repo 경로 확인 불가      근거: queue.tsv/servers.conf 어디에도 없음")
            n_fail += 1
            continue
        log_path = f"{repo}/logs/{session}_launch.log"
        print(f"   로그: {log_path}")
        rows = check_log(host, log_path)
        verdict, reason = check_gpu_memory(host, gpus)
        rows.append(("GPU 점유(3000MiB+)", verdict, reason))
        for name, v, why in rows:
            mark = "PASS" if v == "PASS" else "FAIL"
            if v != "PASS":
                n_fail += 1
            print(f"   {mark}  {name:<22} 근거: {why}")

    print(f"\n== [autoplace] 검증 요약: FAIL {n_fail}건 ==")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
