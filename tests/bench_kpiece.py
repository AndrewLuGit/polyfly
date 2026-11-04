"""Sequential sweep harness for bidirectional KPIECE (and an RRT-Connect baseline).

Runs one config at a time, in-process, so machine load is constant across configs and wall
clock times are comparable within the run. Screening should use an *iteration budget*
(`--iters N`, `--timeout 1e9`), which is load-independent because the search is deterministic
for a fixed (params, seed). Wall-clock A/B runs use `--timeout` with `--iters 0`.

Every run reloads the scenario YAML (no shared mutable params), so overrides cannot leak
between configs.

    POLYFLY_DIR=$(pwd) PYTHONPATH=src python3 tests/bench_kpiece.py \
        --configs tests/kpiece_configs/screen.json --mazes 1 2 3 --seeds 0 1 \
        --iters 2000 --out data/csvs/kpiece_results/screen.json

Add `--baseline-rrt` to interleave the tuned RRT-Connect on the same (maze, seed) pair and
report it in the same table. That is the comparison that tests whether coverage-driven
KPIECE exploration beats RRT-Connect on the hard mazes.
"""

import argparse
import copy
import json
import os
import sys
import time
import types

import numpy as np

REPO = os.environ.get("POLYFLY_DIR", os.getcwd())
sys.path.insert(0, os.path.join(REPO, "src"))

try:
    import torch  # noqa: F401  (planner.py imports it; unused by these paths)
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")

from poly_fly.optimal_planner.bidirectional_kpiece import solve_with_kpiece  # noqa: E402
from poly_fly.optimal_planner.kinodynamic_rrt_connect import solve_with_rrt  # noqa: E402
from poly_fly.utils.utils import MPC, dictToClass, yamlToDict  # noqa: E402


def scenario_path(name):
    if name.startswith("experiments/"):
        return os.path.join(REPO, "data/params", name + ("" if name.endswith(".yaml") else ".yaml"))
    return os.path.join(REPO, "data/params/experiments", name + ".yaml")


def load_params(name):
    return dictToClass(MPC, yamlToDict(scenario_path(name)))


def arc_length(path, samples=400):
    if path is None:
        return None
    times = np.linspace(0.0, path.duration(), samples)
    positions = path.samples_at(times, 0)
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def run_one(scenario, seed, config, iters, timeout):
    params = load_params(scenario)
    for key, value in config.get("overrides", {}).items():
        setattr(params, key, value)
    params.rrt_seed = seed
    params.rrt_timeout = float(timeout)
    if iters is not None:
        params.rrt_max_iterations = int(iters)

    planner = config.get("planner", "kpiece")
    started = time.monotonic()
    try:
        if planner == "rrt":
            result = solve_with_rrt(params)
            success = bool(result.success)
            path = result.path
            plan_time = float(result.duration)
            search_time = plan_time
            calibration_time = 0.0
            iterations = int(result.iterations)
            nodes = len(result.explored_nodes)
            pieces = None
            cells = None
            stats = None
            calibration = []
            junction_stats = {}
            strategy = "rrt"
            path_source = "final"
        else:
            result = solve_with_kpiece(params)
            success = bool(result.success)
            path = result.path
            plan_time = float(result.duration)
            search_time = float(result.search_duration)
            calibration_time = float(result.calibration_duration)
            iterations = int(result.iterations)
            nodes = int(result.nodes)
            pieces = int(result.pieces)
            cells = int(result.cells)
            stats = result.stats
            calibration = result.calibration
            junction_stats = result.junction_stats
            strategy = result.strategy
            path_source = result.path_source
    except Exception as exc:  # a planner crash is a failed row, not a lost sweep
        return {
            "scenario": scenario,
            "seed": seed,
            "config": config["name"],
            "planner": planner,
            "success": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "plan_time": time.monotonic() - started,
            "search_time": None,
            "calibration_time": None,
            "path_duration": None,
            "arc_length": None,
            "iterations": None,
            "nodes": None,
            "pieces": None,
            "cells": None,
            "strategy": None,
            "path_source": None,
            "stats": None,
            "calibration": [],
            "junction_stats": {},
            "junction_attempts": None,
        }

    return {
        "scenario": scenario,
        "seed": seed,
        "config": config["name"],
        "planner": planner,
        "success": success,
        "error": None,
        "plan_time": plan_time,
        "search_time": search_time,
        "calibration_time": calibration_time,
        "path_duration": float(path.duration()) if path is not None else None,
        "arc_length": arc_length(path),
        "iterations": iterations,
        "nodes": nodes,
        "pieces": pieces,
        "cells": cells,
        "strategy": strategy,
        "path_source": path_source,
        "stats": stats,
        "calibration": calibration,
        "junction_stats": junction_stats,
        "junction_attempts": (
            junction_stats.get("attempts") if junction_stats else None
        ),
    }


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else float("nan")


def _p90(values):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, 90)) if values else float("nan")


def summarize(records):
    import collections

    grouped = collections.defaultdict(list)
    for row in records:
        grouped[row["config"]].append(row)
    scenarios = sorted({row["scenario"] for row in records})
    seeds = sorted({row["seed"] for row in records})

    print(
        "%-18s %7s %9s %9s %9s %9s %9s %9s %8s  %s"
        % (
            "config",
            "solved",
            "med_time",
            "p90_time",
            "med_search",
            "med_path",
            "med_iter",
            "med_cells",
            "med_att",
            "per-scenario seed cells (time or X)",
        )
    )
    for name, rows in sorted(grouped.items(), key=lambda kv: -sum(r["success"] for r in kv[1])):
        solved = [r for r in rows if r["success"]]
        cells = []
        for scenario in scenarios:
            pair = []
            for seed in seeds:
                hits = [r for r in rows if r["scenario"] == scenario and r["seed"] == seed]
                if hits:
                    pair.append("%.2f" % hits[0]["plan_time"] if hits[0]["success"] else "X")
            cells.append("%s:%s" % (scenario.replace("maze_", "m"), ",".join(pair)))
        print(
            "%-18s %3d/%-3d %9.2f %9.2f %9.2f %9.2f %9.0f %9.0f %8.0f  %s"
            % (
                name,
                len(solved),
                len(rows),
                _median([r["plan_time"] for r in solved]),
                _p90([r["plan_time"] for r in solved]),
                _median([r["search_time"] for r in solved]),
                _median([r["path_duration"] for r in solved]),
                _median([r["iterations"] for r in solved]),
                _median([r["cells"] for r in solved]),
                _median([r.get("junction_attempts") for r in solved]),
                " ".join(cells),
            )
        )

    # Per-scenario medians for the most informative rows.
    print("\nper-scenario success / median plan / median trajectory")
    for name, rows in sorted(grouped.items()):
        parts = []
        for scenario in scenarios:
            scenario_rows = [r for r in rows if r["scenario"] == scenario]
            solved = [r for r in scenario_rows if r["success"]]
            parts.append(
                "%s %d/%d %.2f/%.2f"
                % (
                    scenario.replace("maze_", "m"),
                    len(solved),
                    len(scenario_rows),
                    _median([r["plan_time"] for r in solved]),
                    _median([r["path_duration"] for r in solved]),
                )
            )
        print("  %-18s %s" % (name, " | ".join(parts)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    parser.add_argument("--mazes", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--scenarios", type=str, nargs="*", default=None,
                        help="scenario names (default: maze_<N> for --mazes)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--iters", type=int, default=2000,
                        help="iteration budget; 0 disables it (wall-clock mode)")
    parser.add_argument("--timeout", type=float, default=1e9)
    parser.add_argument("--baseline-rrt", action="store_true",
                        help="interleave the tuned RRT-Connect on the same (scenario, seed)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.configs) as stream:
        configs = json.load(stream)
    if args.baseline_rrt:
        configs = list(configs) + [
            {"name": "rrt_baseline", "overrides": {}, "planner": "rrt"}
        ]

    scenarios = args.scenarios or ["maze_%d" % maze for maze in args.mazes]
    iters = args.iters if args.iters > 0 else None

    records = []
    started = time.monotonic()
    try:
        for scenario in scenarios:
            for seed in args.seeds:
                for config in configs:
                    row = run_one(scenario, seed, config, iters, args.timeout)
                    records.append(row)
                    if row["success"]:
                        detail = "ok %6.2fs path %6.2fs iter %6d cells %s" % (
                            row["plan_time"],
                            row["path_duration"],
                            row["iterations"] or 0,
                            row["cells"] if row["cells"] is not None else "-",
                        )
                    else:
                        detail = "FAIL %6.2fs %s" % (
                            row["plan_time"], row["error"] or "no path"
                        )
                    print(
                        "  %-10s seed %d %-18s %s" % (scenario, seed, config["name"], detail),
                        flush=True,
                    )
    finally:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as stream:
            json.dump(records, stream, indent=1)
        print("\n=== summary (%.1fs total) ===" % (time.monotonic() - started))
        summarize(records)


if __name__ == "__main__":
    main()
