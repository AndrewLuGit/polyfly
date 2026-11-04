"""Sanity checks for the bidirectional KPIECE planner.

    POLYFLY_DIR=$(pwd) PYTHONPATH=src python3 tests/kpiece_sanity.py

Plain PASS/FAIL script, matching the repo's other sanity tests (there is no pytest here).
The optional OCP-junction check runs only when KPIECE_TEST_OCP=1, because a single IPOPT
solve is expensive and can legitimately fail on a hard seed.
"""

import math
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import torch  # noqa: F401  (planner.py imports it; unused here)
except ImportError:
    sys.modules["torch"] = types.ModuleType("torch")

from poly_fly.optimal_planner.bidirectional_kpiece import (  # noqa: E402
    BidirectionalKPIECE,
    CellProjection,
    KPieceOptions,
    _KPieceTree,
    _adjust_sizes,
    _half_normal_int,
    reverse_motion,
    reverse_state,
    solve_with_kpiece,
)
from poly_fly.optimal_planner.kinodynamic_rrt_connect import (  # noqa: E402
    FlatMotion,
    _mobility_for,
    is_motion_within_bounds,
    make_steering,
)
from poly_fly.utils.utils import MPC, dictToClass, yamlToDict  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition)))
    print("%-4s %-52s %s" % ("PASS" if condition else "FAIL", name, detail))
    if not condition:
        sys.exit(1)


def load_params(relative="experiments/maze_1.yaml"):
    path = os.path.join(os.environ["POLYFLY_DIR"], "data/params", relative)
    return dictToClass(MPC, yamlToDict(path))


def make_options(**overrides):
    base = dict(
        projection="position_velocity",
        expansion="geometric",
        connection="steer",
        cell_envelope=False,
        max_steps=20,
    )
    base.update(overrides)
    return KPieceOptions(**base)


def max_path_jump(path):
    worst = 0.0
    for motion, state in zip(path.motions, path.states[1:]):
        worst = max(worst, np.max(np.abs(motion.flat_state(motion.duration) - state)))
    return worst


def main():
    params = load_params()
    checker = __import__(
        "poly_fly.optimal_planner.flat_collision", fromlist=["FlatEdgeChecker"]
    ).FlatEdgeChecker(params, _mobility_for(params))
    steering = make_steering(params)

    # 1. reverse_motion is an exact involution and maps the endpoint states through R.
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        coefficients = rng.normal(0.0, 1.0, (6, 3))
        duration = float(rng.uniform(0.1, 3.0))
        motion = FlatMotion(coefficients, duration)
        reversed_motion = reverse_motion(motion)
        worst = max(
            worst,
            np.max(np.abs(reverse_motion(reversed_motion).coefficients - coefficients)),
            np.max(np.abs(reversed_motion.flat_state(0.0) - reverse_state(motion.flat_state(duration)))),
            np.max(np.abs(reversed_motion.flat_state(duration) - reverse_state(motion.flat_state(0.0)))),
        )
    check("reverse_motion involution and endpoint map", worst < 1e-10, "max %.1e" % worst)

    # 2. Collision/bounds feasibility is invariant under time reversal.
    start = np.asarray(params.initial_state, dtype=float)
    goal = np.asarray(params.end_state, dtype=float)
    motion = steering.steer(start + np.array([0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]), start + np.array([0.1, 0.05, 0.0, 0, 0, 0, 0, 0, 0]))
    check("steering produced a test motion", motion is not None)
    check(
        "collision validity invariant under reversal",
        checker.is_motion_valid(motion) == checker.is_motion_valid(reverse_motion(motion)),
    )
    check(
        "bounds validity invariant under reversal",
        is_motion_within_bounds(motion, params)
        == is_motion_within_bounds(reverse_motion(motion), params),
    )

    # 3. Projection keys and the goal-tree physical mapping.
    for preset in ("position", "position_velocity", "full", "position_acceleration", "metric_full"):
        projection = CellProjection.from_params(params, make_options(projection=preset))
        check(
            "projection %s has one size per dim" % preset,
            len(projection.sizes) == len(projection.dims) and len(projection.scales) == len(projection.dims),
        )
        state = start.copy()
        check(
            "projection %s round-trips a state key" % preset,
            projection.key(state) == projection.key(np.asarray(state)),
        )
    position_projection = CellProjection.from_params(params, make_options(projection="position"))
    check(
        "position-only key ignores the velocity sign",
        position_projection.key(start) == position_projection.key(reverse_state(start)),
    )
    velocity_projection = CellProjection.from_params(
        params, make_options(projection="position_velocity")
    )
    moving = start.copy()
    moving[3] = 0.25
    check(
        "velocity key sees the velocity sign",
        velocity_projection.key(moving) != velocity_projection.key(reverse_state(moving)),
    )

    # 4. Edge splitting: pieces concatenate and cover the parent motion.
    split_params = load_params()
    split_options = make_options(
        projection="position", cell_sizes=(0.5, 0.5, 0.5)
    )
    split_planner = BidirectionalKPIECE(split_params, split_options, rng=np.random.default_rng(0))
    line = FlatMotion(np.array([[0.0, 0.0, 0.2], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 1.0)
    pieces = split_planner._split_physical(line)
    check("splitting produced several pieces", len(pieces) >= 3, "%d pieces" % len(pieces))
    check(
        "split durations sum to the motion duration",
        abs(sum(p.t1 - p.t0 for p in pieces) - line.duration) < 1e-9,
    )
    check(
        "split pieces concatenate",
        np.max(np.abs(pieces[0].motion.flat_state(0.0) - line.flat_state(0.0))) < 1e-9
        and np.max(np.abs(pieces[-1].motion.flat_state(pieces[-1].motion.duration) - line.flat_state(line.duration))) < 1e-9,
    )
    check(
        "pieces are one cell wide",
        all(
            split_planner.projection.key(p.motion.flat_state(0.5 * p.motion.duration)) == p.key
            for p in pieces
        ),
    )

    # 5. Expansion strategies.
    expansion_planner = BidirectionalKPIECE(
        params, make_options(expansion="geometric", local_radius_cells=2.0),
        rng=np.random.default_rng(1),
    )
    expansion_planner.start = start
    expansion_planner.goal = goal
    fake_tree = type("FakeTree", (), {})()
    fake_tree.reverse_time = False
    fake_tree.states = np.vstack([start])
    fake_tree.motions = [None]
    fake_tree.physical_state = lambda i: np.asarray(start)
    target = expansion_planner._sample_geometric(fake_tree, fake_tree, start)
    half = expansion_planner._local_halfwidths()
    active = half > 0
    ratios = np.abs((target - start)[active]) / half[active]
    check(
        "geometric target lies inside the local ellipsoid",
        float(np.max(ratios)) <= 1.0 + 1e-9,
        "max ratio %.3f" % float(np.max(ratios)),
    )

    control_planner = BidirectionalKPIECE(
        params, make_options(expansion="control", control_fraction=0.05),
        rng=np.random.default_rng(2),
    )
    control_planner.start = start
    control_planner.goal = goal
    # Use an interior state: the maze start sits exactly on z = state_min[2], so a downward
    # random jerk would make the very first step invalid and the check would be seed-dependent.
    control_state = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    propagated = None
    for control_seed in range(20):
        control_planner.rng = np.random.default_rng(control_seed)
        propagated = control_planner._propagate_control(
            type("T", (), {"reverse_time": False})(), control_state
        )
        if propagated is not None:
            break
    check("control propagation returned a motion", propagated is not None)
    if propagated is not None:
        stored, new_native, physical_edge = propagated
        # Constant jerk: v(T) = v0 + a0*T + j*T^2/2, a(T) = a0 + j*T.
        jerk = stored.state_at(stored.duration, 3)
        duration = stored.duration
        expected_v = (
            control_state[3:6]
            + control_state[6:9] * duration
            + 0.5 * jerk * duration**2
        )
        expected_a = control_state[6:9] + jerk * duration
        check(
            "control cubic matches the integrated state",
            np.max(np.abs(new_native[3:6] - expected_v)) < 1e-9
            and np.max(np.abs(new_native[6:9] - expected_a)) < 1e-9,
        )
        check(
            "control physical edge is the stored motion for the start tree",
            np.max(np.abs(physical_edge.flat_state(0.0) - stored.flat_state(0.0))) < 1e-12,
        )

    # 6. Goal-tree propagation: the stored edge reverses to the physical forward edge.
    goal_planner = BidirectionalKPIECE(
        params, make_options(), rng=np.random.default_rng(3)
    )
    goal_planner.start = start
    goal_planner.goal = goal
    goal_tree = type("GoalTree", (), {})()
    goal_tree.reverse_time = True
    goal_root_native = reverse_state(goal)
    target_native = reverse_state(goal + np.array([-0.4, 0.0, 0.0, 0, 0, 0, 0, 0, 0]))
    goal_result = goal_planner._propagate_bvp(goal_tree, goal_root_native, target_native, 1.0)
    if goal_result is None:
        check("goal propagation reaches the local target", False, "steering failed")
    stored, new_native, physical_edge = goal_result
    check(
        "goal stored edge reverses to the physical forward edge",
        np.max(np.abs(reverse_motion(stored).flat_state(0.0) - physical_edge.flat_state(0.0))) < 1e-9,
    )
    check(
        "goal edge ends at the origin (goal root)",
        np.max(np.abs(physical_edge.flat_state(physical_edge.duration) - reverse_state(goal_root_native))) < 1e-9,
    )
    check(
        "goal new state is the physical edge start under R",
        np.max(np.abs(reverse_state(new_native) - physical_edge.flat_state(0.0))) < 1e-9,
    )

    # 7. Cell score distance toggle.
    tree_stub = type("TreeStub", (), {"iteration": 10, "physical_state": lambda self, i: goal})()
    other_stub = type("OtherStub", (), {"physical_state": lambda self, i: goal})()
    score_no = goal_planner._cell_score(tree_stub, other_stub, start)
    distance_planner = BidirectionalKPIECE(
        params, make_options(cell_score_distance=True), rng=np.random.default_rng(4)
    )
    score_yes = distance_planner._cell_score(tree_stub, other_stub, start)
    check("no-distance cell score is 1+log(iter)", abs(score_no - (1.0 + math.log(10))) < 1e-9)
    check("distance cell score is smaller", score_yes < score_no, "%.4f < %.4f" % (score_yes, score_no))

    # 7b. OMPL half-normal motion index: biased toward the newest (largest) index.
    index_rng = np.random.default_rng(11)
    indices = np.array(
        [_half_normal_int(index_rng, 0, 19, 3.0) for _ in range(20000)]
    )
    check(
        "half-normal indices stay in range",
        int(indices.min()) >= 0 and int(indices.max()) <= 19,
        "[%d, %d]" % (int(indices.min()), int(indices.max())),
    )
    check(
        "half-normal favors the newest motion",
        float(indices.mean()) > 9.5,
        "mean %.2f of 19" % float(indices.mean()),
    )
    check("half-normal handles a one-motion cell", _half_normal_int(index_rng, 0, 0) == 0)

    # 7c. Uniform junction selection samples the other tree rather than taking the closest
    # prefix. Build a small synthetic other-tree cell with 20 nodes.
    candidate_planner = BidirectionalKPIECE(
        params,
        make_options(junction_selection="uniform"),
        rng=np.random.default_rng(7),
    )
    candidate_planner.start = start
    candidate_planner.goal = goal
    start_tree = _KPieceTree(start, candidate_planner.projection, False)
    goal_tree = _KPieceTree(reverse_state(goal), candidate_planner.projection, True)
    synthetic_key = candidate_planner.projection.key(start)
    dummy_motion = FlatMotion(np.zeros((4, 3)), 0.1)
    for _ in range(20):
        goal_tree.add(goal_tree.states[0], 0, dummy_motion, synthetic_key, 1.0, 1.0)
    candidate_planner.start_tree = start_tree
    candidate_planner.goal_tree = goal_tree
    need = max(
        candidate_planner.options.connect_tries,
        candidate_planner.options.shortcut_candidates,
        2,
    )
    own, other = candidate_planner._junction_candidates(
        start_tree, goal_tree, synthetic_key, 0
    )
    check(
        "uniform junction samples the other cell",
        own == [0] and len(other) == min(need, 20) and len(set(other)) == len(other),
        "own=%s other=%s" % (own, other),
    )
    samples = set()
    for candidate_seed in range(20):
        candidate_planner.rng = np.random.default_rng(candidate_seed)
        _, sampled = candidate_planner._junction_candidates(
            start_tree, goal_tree, synthetic_key, 0
        )
        samples.add(tuple(sorted(sampled)))
    closest_prefix = tuple(range(min(need, 20)))
    check(
        "uniform junction is not just the closest prefix",
        any(sample != closest_prefix for sample in samples),
        "%d distinct samples" % len(samples),
    )

    # 8. Adaptive sizing directions.
    projection = CellProjection.from_params(params, make_options(projection="position"))
    options = make_options(projection="position")
    stats = {
        "parts_avg": [5.0, 1.0, 1.0],
        "more_than_2_frac": [0.2, 0.0, 0.0],
        "parts_avg_joint": 5.0,
        "samples_per_cell": 50.0,
        "interior_cells": 0,
        "long_motion_frac": 1.0,
    }
    sizes = tuple(projection.sizes)
    new_sizes, decisions, min_steps, changed = _adjust_sizes(stats, sizes, options, projection, params)
    check("adaptive adjustment triggered", changed and len(decisions) > 0)
    check("parts > 4 grows that dimension", new_sizes[0] > sizes[0])
    check("min_steps unchanged when motions are long", min_steps == options.min_steps)

    # 9. Determinism on a small iteration budget.
    def run_once():
        run_params = load_params()
        run_params.rrt_max_iterations = 400
        run_params.rrt_timeout = 60
        run_params.kpiece_projection = "position_velocity"
        run_params.kpiece_expansion = "geometric"
        run_params.kpiece_cell_envelope = False
        return solve_with_kpiece(run_params, start=start, goal=goal)

    first = run_once()
    second = run_once()
    check(
        "same seed reproduces iterations and tree size",
        (first.iterations, first.nodes, first.pieces, first.cells)
        == (second.iterations, second.nodes, second.pieces, second.cells),
        "%s vs %s" % ((first.iterations, first.nodes), (second.iterations, second.nodes)),
    )

    # 10. Smoke solve: an obstacle-free copy of maze_1 isolates the planner and junction
    # machinery from the hard bridge problem. The cluttered maze is run below without a
    # success assertion, because a kinodynamic KPIECE cell meeting does not guarantee that a
    # single flat BVP can bridge it.
    open_params = load_params()
    open_params.obstacles = {}
    open_params.rrt_max_iterations = 1000
    open_params.rrt_timeout = 60
    open_params.kpiece_projection = "position_velocity"
    open_params.kpiece_expansion = "geometric"
    open_params.kpiece_connection = "steer"
    open_params.kpiece_cell_envelope = False
    result = solve_with_kpiece(open_params)
    check("open-space geometric KPIECE solved", result.success, "%d iterations" % result.iterations)
    if result.success:
        open_checker = __import__(
            "poly_fly.optimal_planner.flat_collision", fromlist=["FlatEdgeChecker"]
        ).FlatEdgeChecker(open_params, _mobility_for(open_params))
        check("open-space path is continuous", max_path_jump(result.path) < 1e-6)
        check(
            "open-space path endpoints are exact",
            np.max(np.abs(result.path.states[0] - start)) < 1e-6
            and np.max(np.abs(result.path.states[-1] - goal)) < 1e-6,
        )
        check(
            "open-space path is collision free",
            all(open_checker.is_motion_valid(m) for m in result.path.motions),
        )

    # 11. Shortcut junction smoke on the same open-space instance.
    open_shortcut_params = load_params()
    open_shortcut_params.obstacles = {}
    open_shortcut_params.rrt_max_iterations = 1000
    open_shortcut_params.rrt_timeout = 60
    open_shortcut_params.kpiece_projection = "position_velocity"
    open_shortcut_params.kpiece_expansion = "geometric"
    open_shortcut_params.kpiece_connection = "shortcut"
    open_shortcut_params.kpiece_cell_envelope = False
    shortcut = solve_with_kpiece(open_shortcut_params)
    check("open-space shortcut KPIECE solved", shortcut.success, "%d iterations" % shortcut.iterations)
    if shortcut.success:
        check(
            "shortcut path is no longer than the steer path",
            shortcut.trajectory_duration() <= result.trajectory_duration() + 1e-6,
            "%.2f vs %.2f" % (shortcut.trajectory_duration(), result.trajectory_duration()),
        )

    # 12. Cluttered maze smoke: require a clean run; a found path must validate.
    maze_params = load_params()
    maze_params.rrt_max_iterations = 3000
    maze_params.rrt_timeout = 90
    maze_params.kpiece_projection = "position_velocity"
    maze_params.kpiece_expansion = "geometric"
    maze_params.kpiece_cell_envelope = False
    maze_result = solve_with_kpiece(maze_params)
    check("maze_1 run completed without error", True, "success=%s" % maze_result.success)
    if maze_result.success:
        check("maze_1 path is continuous", max_path_jump(maze_result.path) < 1e-6)
        check(
            "maze_1 path is collision free",
            all(checker.is_motion_valid(m) for m in maze_result.path.motions),
        )

    # 13. maze_7 smoke: harder instance; require a clean run (a found path must validate).
    hard_params = load_params("experiments/maze_7.yaml")
    hard_params.rrt_max_iterations = 800
    hard_params.rrt_timeout = 60
    hard_params.kpiece_projection = "position_velocity"
    hard_params.kpiece_expansion = "geometric"
    hard_params.kpiece_cell_envelope = False
    hard = solve_with_kpiece(hard_params)
    check("maze_7 run completed without error", True, "success=%s" % hard.success)
    if hard.success:
        check("maze_7 path is continuous", max_path_jump(hard.path) < 1e-6)
        check(
            "maze_7 path is collision free",
            all(checker.is_motion_valid(m) for m in hard.path.motions),
        )

    # 14. Optional OCP junction; expensive, so gated by an environment variable.
    if os.environ.get("KPIECE_TEST_OCP"):
        ocp_params = load_params()
        ocp_params.rrt_max_iterations = 2500
        ocp_params.rrt_timeout = 180
        ocp_params.kpiece_projection = "position_velocity"
        ocp_params.kpiece_expansion = "geometric"
        ocp_params.kpiece_connection = "ocp"
        ocp_params.kpiece_cell_envelope = False
        ocp_params.kpiece_ocp_max_attempts = 1
        ocp_params.kpiece_ocp_max_jump = 0.6
        started = time.monotonic()
        ocp_result = solve_with_kpiece(ocp_params)
        check(
            "OCP junction run completed",
            True,
            "success=%s attempts=%s time=%.1fs"
            % (
                ocp_result.success,
                ocp_result.junction_stats.get("ocp_attempts"),
                time.monotonic() - started,
            ),
        )
        if ocp_result.success and ocp_result.strategy == "ocp":
            check(
                "OCP path is collision free at 0.01 s",
                all(checker.is_motion_valid(m) for m in ocp_result.path.motions),
            )

    print("\nAll KPIECE sanity checks passed.")


if __name__ == "__main__":
    main()
