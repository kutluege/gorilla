"""Autonomous campaign supervisor: keep a replicate campaign running across
tunnel outages, GPU-machine restarts, and server wedges, resuming from the
correct point without operator action.

Recovery policy (pre-registered, PREREGISTRATION_C1.md AMENDMENTS 1+2):

- WEDGE (runner stalled, plain requests still served): kill the runner
  tree; the interrupted arm has no scored entries -> ARM-level resume
  (`--start-arm`); completed sibling arms of the replicate are kept.
- OUTAGE (tunnel/machine down: tunnel_probe error, inference_errors gate,
  or tunnel down when the runner died): the server instance may have
  changed -> the interrupted replicate is deleted IN FULL and rerun
  (`--start-replicate`, no start-arm). Completed earlier replicates are
  kept (instance is a replicate-level covariate; contrasts are
  within-replicate).
- In all cases the interrupted/partial arm trees are deleted before
  relaunch (never resumed into), and every action is appended to a
  supervisor log next to the manifest.

The supervisor runs detached (Start-Process) and loops until the manifest
carries `run_end` for a fully completed campaign, or --max-relaunches is
exhausted. It adopts an already-running runner on startup (watch-only until
it exits or stalls).

    python bfcl_eval/scripts/campaign_supervisor.py \
        --arms-json gov_logs/hnav_rag_c1_arms.json \
        --result-root result_hnav_rag_c1 --score-root score_hnav_rag_c1 \
        --gov-log-root gov_logs/hnav_rag_c1 \
        --manifest result_hnav_rag_c1/manifest.jsonl --replicates 5
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
BASE_URL = "http://localhost:8000/v1"

STALL_S = 1500          # no result-tree/manifest growth for 25 min = stalled
POLL_S = 120
RELAUNCH_COOLDOWN_S = 60


def log(sup_log, msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(sup_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tunnel_up():
    try:
        with urllib.request.urlopen(f"{BASE_URL}/models", timeout=10) as r:
            return b"Qwen" in r.read()
    except Exception:
        return False


# ------------------------------------------------------------ pure helpers

def load_arm_labels(arms_json_path):
    arms = json.loads(Path(arms_json_path).read_text(encoding="utf-8"))
    return [a["label"] for a in arms]


def completed_cells(manifest_path, disk_root=None):
    """(rep, arm) cells with BOTH a clean generate (exit 0, zero inference
    errors when recorded) and a clean evaluate whose score files exist.

    disk_root: when given, score files recorded in the manifest are
    RE-VERIFIED on disk NOW -- manifest history alone lies after a recovery
    deletes a replicate's trees (the exact bug hit on 2026-08-09: the
    supervisor resumed past arms whose trees it had itself deleted)."""
    gen_ok, eval_ok = set(), {}
    if not Path(manifest_path).exists():
        return set()
    for line in open(manifest_path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") != "cmd_end" or r.get("exit_code") != 0:
            continue
        cell = (r.get("replicate"), r.get("arm"))
        if r.get("phase") == "generate":
            if r.get("inference_errors", 0) == 0:
                gen_ok.add(cell)
        elif r.get("phase") == "evaluate":
            files = r.get("score_files") or []
            if all(f.get("exists") for f in files):
                eval_ok[cell] = [f["path"] for f in files]
    done = set()
    for cell, paths in eval_ok.items():
        if cell not in gen_ok:
            continue
        if disk_root is not None and not all(
                (Path(disk_root) / p).exists() for p in paths):
            continue                      # trees deleted since -- not done
        done.add(cell)
    return done


def resume_point(replicates, labels, done_cells, mode):
    """(replicate, [arms to run]) for the next launch -- ONE replicate at a
    time, so holes anywhere in the sequence are filled surgically.

    mode 'arm' (wedge, window continuous): only the replicate's missing
    arms run; completed siblings are kept. mode 'replicate' (outage /
    unknown): every arm of the replicate reruns for instance consistency.
    Returns None when the campaign is complete."""
    for rep in range(1, replicates + 1):
        missing = [a for a in labels if (rep, a) not in done_cells]
        if not missing:
            continue
        return rep, (labels if mode == "replicate" else missing)
    return None


def last_stop_reason(manifest_path, killed_marker):
    """'wedge' | 'outage' | 'unknown' for the most recent stop."""
    if killed_marker.exists():
        return "wedge"
    last_error = None
    if Path(manifest_path).exists():
        for line in open(manifest_path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "error":
                last_error = r.get("stage")
    if last_error in ("tunnel_probe", "preflight_probe", "inference_errors"):
        return "outage"
    return "unknown"


# ------------------------------------------------------- process handling

def find_runner_pid(result_root):
    """PID of a live run_gov_replicates.py for this campaign, else None."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine "
             "-match 'run_gov_replicates' -and $_.CommandLine -notmatch "
             "'Get-CimInstance' -and $_.Name -match 'python' } | "
             "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    fallback = None
    for line in out.splitlines():
        if "|" not in line:
            continue
        pid_s, cmdline = line.split("|", 1)
        pid_s = pid_s.strip()
        if not pid_s.isdigit():
            continue
        if result_root in cmdline:
            return int(pid_s)
        fallback = int(pid_s)
    return fallback          # any runner counts on a single-campaign machine


def kill_tree(pid):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                   capture_output=True)


def newest_activity(result_root, manifest_path):
    """Most recent mtime across the result tree + manifest."""
    newest = 0.0
    root = Path(result_root)
    if root.exists():
        for p in root.rglob("*.json"):
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                pass
    if Path(manifest_path).exists():
        newest = max(newest, Path(manifest_path).stat().st_mtime)
    return newest


# ------------------------------------------------------------- tree hygiene

def clean_for_resume(cfg, rep, arms_to_clean):
    """Delete the given arms' trees for the replicate -- the relaunch must
    never resume into a partial tree."""
    doomed = []
    for a in arms_to_clean:
        doomed += [
            Path(cfg["result_root"]) / f"rep{rep:02d}" / a,
            Path(cfg["score_root"]) / f"rep{rep:02d}" / a,
            Path(cfg["gov_log_root"]) / f"rep{rep:02d}_{a}",
        ]
    removed = []
    for d in doomed:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))
    return removed


# ------------------------------------------------------------------- main

def launch_runner(cfg, rep, only_arms):
    """One replicate per launch: --replicates caps at `rep`, --only-arm
    restricts to the missing arms. The supervisor loop advances replicate
    by replicate, so holes anywhere heal."""
    args = [PY, str(Path(__file__).parent / "run_gov_replicates.py"),
            "--replicates", str(rep),
            "--start-replicate", str(rep),
            "--arms-json", f"@{cfg['arms_json']}",
            "--result-root", cfg["result_root"],
            "--score-root", cfg["score_root"],
            "--gov-log-root", cfg["gov_log_root"],
            "--manifest", cfg["manifest"]]
    for a in only_arms:
        args += ["--only-arm", a]
    logfile = open(cfg["runner_log"], "ab")
    import os
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env.setdefault("REMOTE_OPENAI_BASE_URL", BASE_URL)
    return subprocess.Popen(args, cwd=REPO_ROOT, stdout=logfile,
                            stderr=subprocess.STDOUT, env=env)


def campaign_complete(cfg, labels):
    done = completed_cells(REPO_ROOT / cfg["manifest"], disk_root=REPO_ROOT)
    want = {(rep, a) for rep in range(1, cfg["replicates"] + 1)
            for a in labels}
    return want <= done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-json", required=True)
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--score-root", required=True)
    ap.add_argument("--gov-log-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--max-relaunches", type=int, default=50)
    args = ap.parse_args()

    cfg = {k.replace("-", "_"): getattr(args, k.replace("-", "_"))
           for k in ("arms_json", "result_root", "score_root",
                     "gov_log_root", "manifest", "replicates")}
    cfg["runner_log"] = str(REPO_ROOT / "tmp" /
                            f"supervisor_runner_{Path(cfg['result_root']).name}.log")
    sup_log = REPO_ROOT / "tmp" / \
        f"supervisor_{Path(cfg['result_root']).name}.log"
    killed_marker = REPO_ROOT / "tmp" / \
        f"supervisor_killed_{Path(cfg['result_root']).name}.marker"

    labels = load_arm_labels(REPO_ROOT / cfg["arms_json"])
    log(sup_log, f"supervisor start: {cfg['result_root']} arms={labels} "
                 f"replicates={cfg['replicates']}")

    child = None
    relaunches = 0
    while True:
        if campaign_complete(cfg, labels):
            log(sup_log, "campaign COMPLETE; supervisor exiting")
            break

        # a runner we own, or one launched externally
        pid = child.pid if (child and child.poll() is None) \
            else find_runner_pid(cfg["result_root"])
        if pid:
            age = time.time() - newest_activity(REPO_ROOT / cfg["result_root"],
                                                REPO_ROOT / cfg["manifest"])
            if age > STALL_S and tunnel_up():
                log(sup_log, f"STALL {age:.0f}s with tunnel up -> "
                             f"kill wedged runner pid={pid}")
                killed_marker.write_text("wedge", encoding="utf-8")
                kill_tree(pid)
                child = None
            elif int(time.time()) % 3600 < POLL_S:
                log(sup_log, f"heartbeat: runner pid={pid} alive, "
                             f"last activity {age:.0f}s ago")
            time.sleep(POLL_S)
            continue

        # runner is down and campaign incomplete -> recover
        if relaunches >= args.max_relaunches:
            log(sup_log, "max relaunches reached; giving up")
            break
        reason = last_stop_reason(REPO_ROOT / cfg["manifest"], killed_marker)
        killed_marker.unlink(missing_ok=True)
        while not tunnel_up():
            log(sup_log, "tunnel DOWN; waiting 120s")
            time.sleep(120)
        mode = "arm" if reason == "wedge" else "replicate"
        done = completed_cells(REPO_ROOT / cfg["manifest"],
                               disk_root=REPO_ROOT)
        point = resume_point(cfg["replicates"], labels, done, mode)
        if point is None:
            continue                      # loop will see complete
        rep, run_arms = point
        removed = clean_for_resume(
            {k: REPO_ROOT / v if k.endswith("root") else v
             for k, v in cfg.items()}, rep, run_arms)
        log(sup_log, f"resume reason={reason} mode={mode} -> "
                     f"rep{rep:02d} arms={run_arms} "
                     f"(cleaned {len(removed)} trees)")
        child = launch_runner(cfg, rep, run_arms)
        relaunches += 1
        log(sup_log, f"runner launched pid={child.pid} "
                     f"(relaunch {relaunches}/{args.max_relaunches})")
        time.sleep(RELAUNCH_COOLDOWN_S)


if __name__ == "__main__":
    main()
