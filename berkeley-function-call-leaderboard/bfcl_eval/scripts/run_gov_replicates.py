"""
Governance replicate runner  --  Plan 2 Step 2 (v2 SS8.3).

Drives N replicates of the two-arm A/B against the tunneled vLLM endpoint,
STRICTLY SEQUENTIAL, arms alternated within each replicate
(B1, G1, B2, G2, ...), `--num-threads 1`, `--skip-server-setup`.

Per replicate k the layout is (paths relative to the BFCL root):
    <result-root>/rep<k>/<model_slug>/agentic/memory/<backend>/...
    <score-root>/rep<k>/<model_slug>/...
    <gov-log-root>/rep<k>/governance_log.jsonl      (governed arm only;
        the middleware log is append-mode and CWD-relative, so each
        replicate gets its own GOV_LOG_DIR)

Every command start/end is appended to a JSONL manifest carrying: git HEAD,
served-model id + vLLM version (probed from the endpoint), all GOV_* env
handed to the child, REMOTE_OPENAI_* wiring, seed (always null -- the bfcl
CLI has no seed flag; temperature is recorded instead), arm, replicate
index, timestamps, exit codes.

A failed command STOPS the runner (v2 SS8.3: never silently retried).
Restarting is an explicit operator action via --start-replicate.

Statistical protocol notes baked in here:
  - concurrency is forbidden (E10+MIG: vLLM concurrent batching breaks A/B
    determinism) -> --num-threads 1 is hard-coded, not configurable;
  - the student scenario stays IN the generation runs; its pre-registered
    exclusion is analysis-time (analyze_gov_replicates.py).

Usage:
  export REMOTE_OPENAI_BASE_URL="http://localhost:8000/v1"   # the tunnel
  python bfcl_eval/scripts/run_gov_replicates.py \
      --replicates 5 --gov-env GOV_SIM_HIGH=0.95 --gov-env GOV_DELTA=0.32
  # dry run (print the plan, touch nothing):
  python bfcl_eval/scripts/run_gov_replicates.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BFCL_ROOT = Path(__file__).resolve().parents[2]

BASELINE_ARM = "baseline"
GOVERNED_ARM = "governed"

DEFAULT_BASELINE_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FC"
DEFAULT_GOVERNED_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FC-GOV"
DEFAULT_TEST_CATEGORY = "memory_kv,memory_vector"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in test_gov_replicates.py)
# ---------------------------------------------------------------------------

def arm_sequence(n_replicates, start=1):
    """[(1, baseline), (1, governed), (2, baseline), ...] -- the binding order."""
    return [
        (rep, arm)
        for rep in range(start, n_replicates + 1)
        for arm in (BASELINE_ARM, GOVERNED_ARM)
    ]


def rep_dir(root, rep):
    return f"{root}/rep{rep:02d}"


def build_generate_cmd(cfg, model, rep):
    return [
        cfg["python"], "-m", "bfcl_eval", "generate",
        "--model", model,
        "--test-category", cfg["test_category"],
        "--skip-server-setup",
        "--num-threads", "1",
        "--temperature", str(cfg["temperature"]),
        "--result-dir", rep_dir(cfg["result_root"], rep),
        "--allow-overwrite",
    ]


def build_evaluate_cmd(cfg, model, rep):
    return [
        cfg["python"], "-m", "bfcl_eval", "evaluate",
        "--model", model,
        "--test-category", cfg["test_category"],
        "--result-dir", rep_dir(cfg["result_root"], rep),
        "--score-dir", rep_dir(cfg["score_root"], rep),
        "--partial-eval",
    ]


def build_arm_env(cfg, arm, rep, base_env=None):
    """Child env: tunnel wiring passes through; GOV_* is set explicitly.

    The baseline arm runs the plain -FC handler, but GOV_ENABLED=0 is still
    exported so a mis-registered handler cannot silently govern the baseline.
    The governed arm gets the calibrated cfg["gov_env"] plus a per-replicate
    GOV_LOG_DIR (the middleware log is append-mode; mixing replicates in one
    file would corrupt the analysis).
    """
    env = dict(base_env if base_env is not None else os.environ)
    stale = [k for k in env if k.startswith("GOV_")]
    for k in stale:
        del env[k]
    if arm == BASELINE_ARM:
        env["GOV_ENABLED"] = "0"
    else:
        env["GOV_ENABLED"] = "1"
        env.update(cfg["gov_env"])
        env["GOV_LOG_DIR"] = str(
            Path(cfg["gov_log_root"]) / f"rep{rep:02d}"
        )
    return env


def gov_env_of(env):
    return {k: v for k, v in sorted(env.items()) if k.startswith("GOV_")}


def probe_server(base_url):
    """Probe the tunneled endpoint: served model id + vLLM version.

    /v1/models is required (its data[0].id is the served-model identity the
    manifest must pin); /version is best-effort (older vLLMs lack it).
    Raises on a dead tunnel -- the runner must not start blind.
    """
    root = base_url.rstrip("/")
    with urllib.request.urlopen(f"{root}/models", timeout=15) as r:
        models = json.loads(r.read().decode("utf-8"))
    served = [m.get("id") for m in models.get("data", [])]
    version = None
    try:
        with urllib.request.urlopen(
            f"{root.removesuffix('/v1')}/version", timeout=10
        ) as r:
            version = json.loads(r.read().decode("utf-8")).get("version")
    except Exception:
        pass
    return {"served_models": served, "vllm_version": version}


def append_manifest(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def git_head(repo_root):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The sequential loop
# ---------------------------------------------------------------------------

def default_run_cmd(cmd, env):
    """Foreground child sharing stdout/stderr; returns the exit code."""
    return subprocess.run(cmd, cwd=BFCL_ROOT, env=env).returncode


def run_replicates(cfg, run_cmd=default_run_cmd, probe=probe_server,
                   manifest_writer=append_manifest, clock=time.time):
    """Execute the full B1,G1,B2,G2,... sequence. Returns the manifest path.

    Any nonzero exit code or probe failure appends an `error` record and
    raises SystemExit(1) -- a failed replicate stops the runner, it is never
    silently retried (v2 SS8.3).
    """
    manifest = cfg["manifest"]
    run_id = cfg.get("run_id") or datetime.now(timezone.utc).strftime(
        "govrep_%Y%m%dT%H%M%SZ")
    base_record = {"run_id": run_id}
    models = {BASELINE_ARM: cfg["baseline_model"],
              GOVERNED_ARM: cfg["governed_model"]}

    try:
        server = probe(cfg["base_url"])
    except Exception as exc:
        manifest_writer(manifest, {
            **base_record, "ts": utc_now(), "event": "error",
            "stage": "preflight_probe", "detail": str(exc),
            "base_url": cfg["base_url"],
        })
        print(f"[gov-rep] FATAL: endpoint probe failed ({exc}); is the tunnel up?")
        raise SystemExit(1)

    manifest_writer(manifest, {
        **base_record, "ts": utc_now(), "event": "run_start",
        "git_head": git_head(BFCL_ROOT),
        "base_url": cfg["base_url"],
        "remote_tokenizer_path": os.environ.get("REMOTE_OPENAI_TOKENIZER_PATH"),
        "served_models": server["served_models"],
        "vllm_version": server["vllm_version"],
        "seed": None,  # bfcl generate has no seed flag; temperature is the only knob
        "temperature": cfg["temperature"],
        "test_category": cfg["test_category"],
        "replicates": cfg["replicates"],
        "start_replicate": cfg.get("start_replicate", 1),
        "arm_order_per_replicate": [BASELINE_ARM, GOVERNED_ARM],
        "num_threads": 1,
        "skip_server_setup": True,
        "models": models,
        "gov_env_governed": dict(sorted(cfg["gov_env"].items())),
        "result_root": cfg["result_root"],
        "score_root": cfg["score_root"],
        "gov_log_root": cfg["gov_log_root"],
    })

    for rep, arm in arm_sequence(cfg["replicates"],
                                 cfg.get("start_replicate", 1)):
        env = build_arm_env(cfg, arm, rep)
        for phase, cmd in (
            ("generate", build_generate_cmd(cfg, models[arm], rep)),
            ("evaluate", build_evaluate_cmd(cfg, models[arm], rep)),
        ):
            # Generation needs the tunnel; re-probe so a dropped SSH session
            # stops the run at a clean boundary instead of mid-scenario.
            if phase == "generate":
                try:
                    probe(cfg["base_url"])
                except Exception as exc:
                    manifest_writer(manifest, {
                        **base_record, "ts": utc_now(), "event": "error",
                        "stage": "tunnel_probe", "replicate": rep, "arm": arm,
                        "detail": str(exc),
                    })
                    print(f"[gov-rep] FATAL: tunnel down before rep{rep:02d}/{arm}.")
                    raise SystemExit(1)

            manifest_writer(manifest, {
                **base_record, "ts": utc_now(), "event": "cmd_start",
                "replicate": rep, "arm": arm, "phase": phase,
                "model": models[arm], "cmd": cmd, "gov_env": gov_env_of(env),
            })
            t0 = clock()
            exit_code = run_cmd(cmd, env)
            manifest_writer(manifest, {
                **base_record, "ts": utc_now(), "event": "cmd_end",
                "replicate": rep, "arm": arm, "phase": phase,
                "exit_code": exit_code, "duration_s": round(clock() - t0, 3),
            })
            if exit_code != 0:
                manifest_writer(manifest, {
                    **base_record, "ts": utc_now(), "event": "error",
                    "stage": phase, "replicate": rep, "arm": arm,
                    "exit_code": exit_code,
                    "detail": "command failed; runner stopped (never retried)",
                })
                print(
                    f"[gov-rep] FATAL: rep{rep:02d}/{arm}/{phase} exited "
                    f"{exit_code}; stopping (resume explicitly with "
                    f"--start-replicate {rep})."
                )
                raise SystemExit(1)
        print(f"[gov-rep] rep{rep:02d}/{arm} complete.")

    manifest_writer(manifest, {
        **base_record, "ts": utc_now(), "event": "run_end",
    })
    print(f"[gov-rep] all {cfg['replicates']} replicates complete; "
          f"manifest: {manifest}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_gov_env(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--gov-env expects KEY=VALUE, got: {p}")
        k, v = p.split("=", 1)
        if not k.startswith("GOV_"):
            raise SystemExit(f"--gov-env keys must start with GOV_: {k}")
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--start-replicate", type=int, default=1,
                    help="Explicit resume point after a stopped run")
    ap.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    ap.add_argument("--governed-model", default=DEFAULT_GOVERNED_MODEL)
    ap.add_argument("--test-category", default=DEFAULT_TEST_CATEGORY)
    ap.add_argument("--temperature", type=float, default=0.001)
    ap.add_argument("--result-root", default="result_gov_replicates")
    ap.add_argument("--score-root", default="score_gov_replicates")
    ap.add_argument("--gov-log-root", default="gov_logs/replicates")
    ap.add_argument("--manifest", default=None,
                    help="Manifest JSONL (default <result-root>/manifest.jsonl)")
    ap.add_argument("--gov-env", action="append", default=[],
                    metavar="GOV_KEY=VALUE",
                    help="Calibrated governed-arm setting (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the command plan; execute nothing")
    args = ap.parse_args()

    if not os.environ.get("REMOTE_OPENAI_BASE_URL"):
        raise SystemExit(
            "REMOTE_OPENAI_BASE_URL is not set. Bring up the SSH tunnel and "
            "export REMOTE_OPENAI_BASE_URL=http://localhost:8000/v1 first "
            "(Plan 2 SS0.1)."
        )

    cfg = {
        "python": sys.executable,
        "replicates": args.replicates,
        "start_replicate": args.start_replicate,
        "baseline_model": args.baseline_model,
        "governed_model": args.governed_model,
        "test_category": args.test_category,
        "temperature": args.temperature,
        "result_root": args.result_root,
        "score_root": args.score_root,
        "gov_log_root": args.gov_log_root,
        "manifest": args.manifest or f"{args.result_root}/manifest.jsonl",
        "gov_env": parse_gov_env(args.gov_env),
        "base_url": os.environ["REMOTE_OPENAI_BASE_URL"],
    }
    # Manifest path is BFCL-root-relative like the result/score roots.
    if not Path(cfg["manifest"]).is_absolute():
        cfg["manifest"] = str(BFCL_ROOT / cfg["manifest"])

    if args.dry_run:
        print(f"[gov-rep] DRY RUN -- {args.replicates} replicates, order:")
        models = {BASELINE_ARM: cfg["baseline_model"],
                  GOVERNED_ARM: cfg["governed_model"]}
        for rep, arm in arm_sequence(args.replicates, args.start_replicate):
            print(f"  rep{rep:02d}/{arm}:")
            print("    " + " ".join(build_generate_cmd(cfg, models[arm], rep)))
            print("    " + " ".join(build_evaluate_cmd(cfg, models[arm], rep)))
            if arm == GOVERNED_ARM:
                env = build_arm_env(cfg, arm, rep)
                print(f"    gov_env: {gov_env_of(env)}")
        return

    run_replicates(cfg)


if __name__ == "__main__":
    main()
