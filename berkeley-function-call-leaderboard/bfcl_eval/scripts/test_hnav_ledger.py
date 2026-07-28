"""Tests for hnav_ledger.py. Run: python bfcl_eval/scripts/test_hnav_ledger.py"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import hnav_ledger as hl  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def tmp_ledger():
    return Path(tempfile.mkdtemp(prefix="ledger_")) / "experiment_ledger.jsonl"


def e(eid="alt7_parser", n=7, status="proposed", **kw):
    d = {"experiment_id": eid, "hypothesis_number": n, "name": "parser robustness",
         "status": status}
    d.update(kw)
    return d


def test_append_and_load():
    p = tmp_ledger()
    rec = hl.append_entry(e(), ledger_path=p)
    check("ts + git_head stamped", "ts" in rec and "git_head" in rec)
    rows = hl.load_ledger(p)
    check("one row persisted", len(rows) == 1 and rows[0]["status"] == "proposed")
    hl.append_entry(e(status="falsifier_running"), ledger_path=p)
    hl.append_entry(e(status="rejected", decision="dead: <1% parse failures"),
                    ledger_path=p)
    check("three rows in order",
          [r["status"] for r in hl.load_ledger(p)]
          == ["proposed", "falsifier_running", "rejected"])
    check("latest_status picks last",
          hl.latest_status(p)["alt7_parser"]["status"] == "rejected")


def test_validation():
    p = tmp_ledger()
    try:
        hl.append_entry({"experiment_id": "x", "name": "n", "status": "proposed"},
                        ledger_path=p)
        check("missing hypothesis_number rejected", False)
    except ValueError:
        check("missing hypothesis_number rejected", True)
    try:
        hl.append_entry(e(status="bogus"), ledger_path=p)
        check("bad status rejected", False)
    except ValueError:
        check("bad status rejected", True)
    try:
        hl.append_entry(e(n=11), ledger_path=p)
        check("hypothesis_number > 10 rejected", False)
    except ValueError:
        check("hypothesis_number > 10 rejected", True)


def test_monotonicity():
    p = tmp_ledger()
    hl.append_entry(e(status="proposed"), ledger_path=p)
    hl.append_entry(e(status="full_run"), ledger_path=p)
    try:
        hl.append_entry(e(status="proposed"), ledger_path=p)
        check("status regression rejected", False)
    except ValueError:
        check("status regression rejected", True)
    hl.append_entry(e(status="rejected"), ledger_path=p)   # reject always allowed
    check("reject-from-anywhere allowed", True)
    try:
        hl.append_entry(e(status="full_run"), ledger_path=p)
        check("entry after terminal rejected", False)
    except ValueError:
        check("entry after terminal rejected", True)
    check("assert_monotonic passes on valid ledger",
          hl.assert_monotonic(hl.load_ledger(p)))


def test_exhaustion_cap():
    p = tmp_ledger()
    for i in range(1, 11):
        hl.append_entry(e(eid=f"alt{i}", n=i), ledger_path=p)
    try:
        hl.append_entry(e(eid="alt_extra", n=5), ledger_path=p)
        check("11th distinct alternative refused", False)
    except ValueError:
        check("11th distinct alternative refused", True)
    check("10 distinct allowed", len(hl.latest_status(p)) == 10)


def main():
    for fn in [test_append_and_load, test_validation, test_monotonicity,
               test_exhaustion_cap]:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
