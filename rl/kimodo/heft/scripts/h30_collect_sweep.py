"""h30: run the collection over every buildable task and say which ones yield data.

The question this answers is not "does the robot succeed" -- `h19_task_screen`
and `tasks.md` already cover that -- but **which tasks produce EXPORTABLE
episodes**, which is a strictly smaller set: an episode has to pass the usable
gate (success, no fall, physics, naturalness) before a single frame reaches the
dataset. On `IThorBathroomApproachSink` those two numbers are 100% and 20%.

    conda activate kimodo_rl && cd /path/to/humanoid_benchmark
    unset all_proxy ALL_PROXY

    python -m rl.kimodo.heft.scripts.h30_collect_sweep --episodes 10
    python -m rl.kimodo.heft.scripts.h30_collect_sweep --tasks IThorBathroomApproachToilet
    python -m rl.kimodo.heft.scripts.h30_collect_sweep --report-only     # read it back

⚠️ **ONE SUBPROCESS PER TASK, and that is not tidiness.** A task that raises
during scene construction, wedges the simulator, or runs out of memory must cost
one task rather than the sweep -- there is no point in a four-hour run that dies
on the third of forty-five and reports nothing. It also returns the GPU between
tasks, so a leak in one task cannot accumulate across the run.

⚠️ **RESUMABLE, keyed on the dataset directory.** Re-running skips any task whose
dataset already exists, so an interrupted sweep continues instead of starting
over -- and so a task cannot be silently collected twice into two directories
that later get merged.

The report is rewritten after EVERY task, so it can be read while the sweep is
still going. That is deliberate: this run takes hours and the whole point is to
be able to look at it partway through.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

#: Stop before the disk does. A compiled scene is 120-210 MB and lives until its
#: task's export finishes, so a sweep needs headroom, not just a non-zero number.
#: Measured on this machine: 45 tasks x 10 episodes would want ~72 GB of them if
#: they were never cleaned up, which is why h7 removes them per task.
MIN_FREE_GB = 25.0


def buildable_tasks() -> list[str]:
    """The 48-task set minus the three that raise before a policy ever runs."""
    from rl.kimodo.heft import adapters48

    return sorted(t for t in adapters48.TASKS if t not in adapters48.CANNOT_BUILD)


def _free_gb(path: str) -> float:
    return shutil.disk_usage(os.path.abspath(path) if os.path.exists(path)
                             else ".").free / 2 ** 30


def _row_for(task: str, out_root: str, h7_out: str, tag: str,
             rc: int, secs: float, note: str = "") -> dict:
    """One task's line in the report, read back from h7's own summary.

    Read from the file rather than scraped from stdout: the summary is already
    the authoritative record and parsing the log would be a second, worse copy
    of it that drifts the moment a print statement changes.
    """
    row = {"task": task, "rc": rc, "secs": round(secs, 1), "note": note}
    sp = os.path.join(h7_out, f"{tag}.json")
    if os.path.exists(sp):
        try:
            s = json.load(open(sp))["summary"]
        except Exception as e:  # noqa: BLE001
            row["note"] = (note + f" unreadable summary: {e}").strip()
            return row
        for k in ("episodes", "success_rate", "usable_rate", "physics_ok_rate",
                  "natural_ok_rate", "fall_rate", "gate", "reject_reasons"):
            row[k] = s.get(k)
    ds = os.path.join(out_root, task)
    row["dataset"] = ds if os.path.isdir(ds) else None
    if row["dataset"]:
        try:
            info = json.load(open(os.path.join(ds, "meta", "info.json")))
            row["frames"] = info.get("total_frames")
            row["dataset_episodes"] = info.get("total_episodes")
        except Exception:  # noqa: BLE001
            pass
    return row


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'task':44s}{'ep':>4}{'succ':>7}{'USABLE':>8}{'frames':>8}  note")
    print("-" * 88)
    for r in sorted(rows, key=lambda x: -(x.get("usable_rate") or -1)):
        succ = r.get("success_rate")
        use = r.get("usable_rate")
        print(f"{r['task']:44s}{r.get('episodes') or 0:>4}"
              f"{'' if succ is None else f'{succ:.2f}':>7}"
              f"{'' if use is None else f'{use:.2f}':>8}"
              f"{r.get('frames') or '':>8}  {r.get('note','')}")
    ok = [r for r in rows if (r.get("frames") or 0) > 0]
    print("-" * 88)
    print(f"{len(ok)}/{len(rows)} tasks produced data; "
          f"{sum(r.get('frames') or 0 for r in ok)} frames total")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="default: every buildable task in the 48-set")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--ckpt",
                    default="rl/kimodo/_out/heft/ckpt/p41c_hands_kimodo_best.pt")
    ap.add_argument("--out", default="rl/kimodo/_out/heft/collect",
                    help="one LeRobot dataset per task underneath this")
    ap.add_argument("--h7-out", default="rl/kimodo/_out/heft/h7_sweep",
                    help="where h7's own summaries and state bundles go")
    ap.add_argument("--report", default=None, help="default: <out>/sweep.json")
    ap.add_argument("--gate", choices=("usable", "success"), default="usable")
    ap.add_argument("--no-natural", action="store_true",
                    help="forwarded to h7: drop the smoothness screen from the "
                         "export gate. Recorded in the report, because a sweep "
                         "run this way is not comparable with one that was not")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="seconds per task before it is killed and recorded as "
                         "a timeout. A hung simulator must not eat the sweep")
    ap.add_argument("--report-only", action="store_true",
                    help="print the existing report and exit; safe mid-sweep")
    args = ap.parse_args()

    report_path = args.report or os.path.join(args.out, "sweep.json")
    if args.report_only:
        if not os.path.exists(report_path):
            raise SystemExit(f"[h30] no report at {report_path}")
        _print_table(json.load(open(report_path))["tasks"])
        return 0

    tasks = args.tasks if args.tasks else buildable_tasks()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.h7_out, exist_ok=True)
    rows = []
    if os.path.exists(report_path):
        rows = json.load(open(report_path)).get("tasks", [])
    done = {r["task"] for r in rows if r.get("dataset") or r.get("rc") == 0}

    print(f"[h30] {len(tasks)} tasks x {args.episodes} episodes, gate={args.gate}"
          + (" (naturalness NOT gated)" if args.no_natural else ""))
    print(f"[h30] datasets -> {args.out}   report -> {report_path}")
    if done:
        print(f"[h30] resuming: {len(done)} already done, skipping them")
    t_all = time.time()

    for i, task in enumerate(tasks, 1):
        ds = os.path.join(args.out, task)
        if task in done or os.path.exists(ds):
            print(f"[h30] {i}/{len(tasks)} {task}: already collected, skipping")
            continue
        free = _free_gb(args.out)
        if free < MIN_FREE_GB:
            print(f"[h30] STOPPING: {free:.1f} GB free, below the {MIN_FREE_GB} GB "
                  f"bar. {len(rows)} tasks are already in {report_path}; free "
                  f"space and re-run to continue where this left off.")
            break
        tag = f"sweep_{task}"
        cmd = [sys.executable, "-m", "rl.kimodo.heft.scripts.h7_scene_rollout",
               "--env", task, "--task-set", "48", "--ckpt", args.ckpt,
               "--episodes", str(args.episodes), "--seed", str(args.seed),
               "--tag", tag, "--gate", args.gate,
               "--out", args.h7_out, "--save-data", ds]
        if args.no_natural:
            cmd.append("--no-natural")
        print(f"[h30] {i}/{len(tasks)} {task}  ({free:.0f} GB free)", flush=True)
        t0 = time.time()
        note, rc = "", -1
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.timeout)
            rc = r.returncode
            if rc:
                # The last line of stderr is almost always the SystemExit message
                # h7 raises for an unbuildable scene, which is the single most
                # useful thing to carry into the table.
                tail = [ln for ln in (r.stderr or "").strip().splitlines() if ln.strip()]
                note = (tail[-1] if tail else "failed")[:120]
        except subprocess.TimeoutExpired:
            note = f"TIMEOUT after {args.timeout:.0f}s"
        except Exception as e:  # noqa: BLE001
            note = f"{type(e).__name__}: {e}"[:120]
        rows = [x for x in rows if x["task"] != task]
        rows.append(_row_for(task, args.out, args.h7_out, tag, rc,
                             time.time() - t0, note))
        last = rows[-1]
        print(f"[h30]   -> succ {last.get('success_rate')}  "
              f"usable {last.get('usable_rate')}  frames {last.get('frames')}  "
              f"{last.get('note','')}", flush=True)
        # Rewritten every task on purpose: a sweep this long has to be readable
        # while it runs, and a report written only at the end is a report that
        # does not exist if anything interrupts it.
        with open(report_path, "w") as f:
            json.dump({"ckpt": args.ckpt, "episodes": args.episodes,
                       "gate": args.gate, "no_natural": bool(args.no_natural),
                       "seed": args.seed,
                       "elapsed_s": round(time.time() - t_all, 1),
                       "tasks": rows}, f, indent=2)

    _print_table(rows)
    print(f"\n[h30] report: {report_path}")
    print(f"[h30] datasets that have data are under {args.out}/<task>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
