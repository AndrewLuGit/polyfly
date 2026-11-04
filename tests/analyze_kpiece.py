"""Summarize a tests/bench_kpiece.py JSON file as compact tables.

    python3 tests/analyze_kpiece.py data/csvs/kpiece_results/screen.json

Reports solve rate, median/p90 planning time, median trajectory duration, iterations and
grid size per config, plus per-maze solve counts. Failed rows are excluded from the time and
duration medians (they have no trajectory); the solve rate is always over all rows.
"""

import argparse
import collections
import json

import numpy as np


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def p90(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(values, 90)) if values else float("nan")


def fmt(value, digits=2):
    return "   -  " if value is None or not np.isfinite(value) else ("%6.*f" % (digits, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()

    with open(args.path) as stream:
        records = json.load(stream)
    if args.scenarios:
        records = [r for r in records if r["scenario"] in set(args.scenarios)]

    grouped = collections.defaultdict(list)
    for row in records:
        grouped[row["config"]].append(row)
    scenarios = sorted({row["scenario"] for row in records})

    print("config              solved  succ%   med_plan  p90_plan  med_search  med_traj  med_iter  med_cells  med_att")
    print("-" * 112)
    for name, rows in sorted(grouped.items(), key=lambda kv: -sum(r["success"] for r in kv[1])):
        solved = [r for r in rows if r["success"]]
        success_rate = 100.0 * len(solved) / max(len(rows), 1)
        print(
            "%-18s %3d/%-3d %6.1f %s %s %s %s %s %s %s"
            % (
                name,
                len(solved),
                len(rows),
                success_rate,
                fmt(median([r["plan_time"] for r in solved])),
                fmt(p90([r["plan_time"] for r in solved])),
                fmt(median([r["search_time"] for r in solved])),
                fmt(median([r["path_duration"] for r in solved])),
                fmt(median([r["iterations"] for r in solved]), 0),
                fmt(median([r["cells"] for r in solved]), 0),
                fmt(median([r.get("junction_attempts") for r in solved]), 0),
            )
        )

    print("\nper-scenario: solved / median plan / median trajectory")
    for name, rows in sorted(grouped.items(), key=lambda kv: -sum(r["success"] for r in kv[1])):
        cells = []
        for scenario in scenarios:
            scenario_rows = [r for r in rows if r["scenario"] == scenario]
            solved = [r for r in scenario_rows if r["success"]]
            cells.append(
                "%s %d/%d %s/%s"
                % (
                    scenario.replace("maze_", "m"),
                    len(solved),
                    len(scenario_rows),
                    fmt(median([r["plan_time"] for r in solved])),
                    fmt(median([r["path_duration"] for r in solved])),
                )
            )
        print("  %-18s %s" % (name, " | ".join(cells)))

    # Junction / strategy breakdown, if present.
    if any(r.get("junction_stats") for r in records):
        print("\njunction stats (summed over runs)")
        for name, rows in sorted(grouped.items()):
            totals = collections.Counter()
            for row in rows:
                for key, value in (row.get("junction_stats") or {}).items():
                    if isinstance(value, (int, float)):
                        totals[key] += value
            active = {k: int(v) for k, v in totals.items() if v}
            if active:
                print("  %-18s %s" % (name, active))


if __name__ == "__main__":
    main()
