"""Sanity checks for the kinodynamic RRT-Connect planner.

Run with the repo's dev environment:

    POLYFLY_DIR=$(pwd) PYTHONPATH=src python3 tests/kinodynamic_rrt_sanity.py

There is no pytest in this project's environment, so this is a plain script that prints
PASS/FAIL per check and exits non-zero on the first failure.
"""

import math
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import torch  # noqa: F401  (imported but unused by planner.py)
except ImportError:  # keep the planner importable in a torch-free environment
    sys.modules["torch"] = types.ModuleType("torch")

from poly_fly.optimal_planner.kinodynamic_rrt_connect import (  # noqa: E402
    ACC_NORM_EPS,
    ORDER,
    OUTPUT_DIM,
    FlatEdgeChecker,
    FlatLimits,
    FlatMotion,
    FixedDurationSteering,
    MinimumEffortSteering,
    _body_boxes_batch,
    _bound_violation,
    _boxes_hit_obstacles,
    _effort_form,
    _end_derivatives_inverse,
    _mobility_for,
    is_motion_within_bounds,
    make_steering,
    path_to_sol_values,
    poly_extrema,
    resample_path_for_initial_guess,
    solve_with_rrt,
)
from poly_fly.optimal_planner.planner import Planner  # noqa: E402
from poly_fly.utils.utils import MPC, dictToClass, yamlToDict  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition)))
    print("%-4s %-46s %s" % ("PASS" if condition else "FAIL", name, detail))
    if not condition:
        sys.exit(1)


def load_params(relative="experiments/maze_1.yaml"):
    path = os.path.join(os.environ["POLYFLY_DIR"], "data/params", relative)
    return dictToClass(MPC, yamlToDict(path))


def reference_obb_overlap(center_a, rotation_a, half_a, center_b, half_b):
    """Independent, normalized, loop-based separating-axis test (obstacle is axis-aligned)."""
    axes = [rotation_a[:, i] for i in range(3)] + [np.eye(3)[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            axes.append(np.cross(rotation_a[:, i], np.eye(3)[:, j]))
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-9:
            continue
        unit = axis / length
        radius_a = np.sum(half_a * np.abs(rotation_a.T @ unit))
        radius_b = np.sum(half_b * np.abs(unit))
        if abs((center_b - center_a) @ unit) > radius_a + radius_b + 1e-12:
            return False
    return True


def main():
    rng = np.random.default_rng(0)
    fixed = FixedDurationSteering(1.0)

    # 1. closed-form BVP reproduces both boundary states.
    worst = 0.0
    for _ in range(300):
        f = rng.normal(0, 1, (ORDER, OUTPUT_DIM))
        t = rng.normal(0, 1, (ORDER, OUTPUT_DIM))
        T = rng.uniform(0.05, 10.0)
        motion = FlatMotion(fixed.coefficients_over(f, t, T), T)
        for level in range(ORDER):
            worst = max(
                worst,
                np.abs(motion.state_at(0.0, level) - f[level]).max(),
                np.abs(motion.state_at(T, level) - t[level]).max(),
            )
    check("flat BVP boundary conditions", worst < 1e-10, "max residual %.1e" % worst)

    # 2. coefficients agree with a direct linear solve of the same system.
    worst = 0.0
    for _ in range(50):
        f = rng.normal(0, 1, (ORDER, OUTPUT_DIM))
        t = rng.normal(0, 1, (ORDER, OUTPUT_DIM))
        T = rng.uniform(0.1, 5.0)
        coeffs = fixed.coefficients_over(f, t, T)
        system = np.zeros((2 * ORDER, 2 * ORDER))
        for level in range(ORDER):
            for j in range(2 * ORDER):
                system[level, j] = math.factorial(level) if j == level else 0.0
                if j >= level:
                    system[ORDER + level, j] = (
                        math.factorial(j) / math.factorial(j - level) * T ** (j - level)
                    )
        rhs = np.concatenate([f, t])
        worst = max(worst, np.abs(system @ coeffs - rhs).max())
    check("steering matches a direct 6x6 solve", worst < 1e-9, "max residual %.1e" % worst)

    # 3. exact-integer matrix helpers.
    m_matrix = np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float)
    check(
        "M^-1 / effort form match reference values",
        np.abs(_end_derivatives_inverse(3) - np.linalg.inv(m_matrix)).max() < 1e-12
        and np.abs(
            _effort_form(3) - np.array([[720, -360, 60], [-360, 192, -36], [60, -36, 9]])
        ).max()
        == 0.0,
    )

    # 4. minimum-effort duration is the true minimiser of J(T).
    effort = MinimumEffortSteering(rho=30.0)
    worst_T = worst_J = 0.0
    for _ in range(120):
        f = rng.normal(0, 1, (ORDER, OUTPUT_DIM))
        t = rng.normal(0, 1, (ORDER, OUTPUT_DIM))
        optimum = effort.optimal_duration(f, t)
        q = effort.effort_numerator(f, t)
        durations = np.linspace(0.01, 30.0, 200000)
        costs = np.polyval(q[::-1], durations) / durations**5 + effort.rho * durations
        best = int(np.argmin(costs))
        worst_T = max(worst_T, abs(optimum[0] - durations[best]) / durations[best])
        worst_J = max(worst_J, (optimum[1] - costs[best]) / costs[best])
    check("min-effort duration vs 200k-point scan", worst_T < 1e-3, "rel err %.1e" % worst_T)
    check("min-effort cost vs scan minimum", worst_J < 1e-9, "rel err %.1e" % worst_J)

    # 5. closed-form cost equals a numerical jerk integral.
    worst = 0.0
    for _ in range(30):
        f = rng.normal(0, 1, (ORDER, OUTPUT_DIM))
        t = rng.normal(0, 1, (ORDER, OUTPUT_DIM))
        T = rng.uniform(0.3, 4.0)
        motion = FlatMotion(fixed.coefficients_over(f, t, T), T)
        times = np.linspace(0, T, 200001)
        numeric = np.trapezoid((motion.samples_at(times, ORDER) ** 2).sum(axis=1), times)
        worst = max(worst, abs(motion.cost(ORDER) - numeric) / max(numeric, 1e-9))
    check("closed-form effort integral", worst < 1e-6, "rel err %.1e" % worst)

    # 6. exact bounds extrema bracket a dense scan.
    worst_lo = worst_hi = 0.0
    for _ in range(300):
        coeffs = rng.normal(0, 1, rng.integers(1, 6))
        T = rng.uniform(0.1, 3.0)
        low, high = poly_extrema(coeffs, T)
        values = np.polyval(coeffs[::-1], np.linspace(0, T, 200001))
        worst_lo = max(worst_lo, values.min() - low)
        worst_hi = max(worst_hi, high - values.max())
    check("poly_extrema brackets a dense scan", worst_lo >= -1e-9 and worst_hi >= -1e-9)

    # 7. batched separating-axis test agrees with the independent reference.
    false_separated = false_overlap = 0
    for _ in range(20000):
        center_a = rng.normal(0, 0.35, 3)
        center_b = rng.normal(0, 0.35, 3)
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        rotation *= np.sign(np.linalg.det(rotation))
        half_a = rng.uniform(0.05, 0.25, 3)
        half_b = rng.uniform(0.05, 0.25, 3)
        got = bool(
            _boxes_hit_obstacles(
                center_a[None], rotation[None], half_a, center_b[None], half_b[None]
            )[0]
        )
        expected = reference_obb_overlap(center_a, rotation, half_a, center_b, half_b)
        false_separated += int(expected and not got)
        false_overlap += int(got and not expected)
    check(
        "OBB separating-axis test is exact",
        false_separated == 0 and false_overlap == 0,
        "false-sep %d, false-ovl %d" % (false_separated, false_overlap),
    )

    # 8. orientation-aware poses match the optimal planner's own conventions.
    params = load_params()
    mobility = _mobility_for(params)
    worst = 0.0
    for _ in range(100):
        accelerations = rng.uniform(-6, 6, (7, 3))
        positions = rng.normal(0, 2, (7, 3))
        boxes = _body_boxes_batch(params, positions, accelerations, True)
        for i in range(7):
            rotation_cable, p_hat = mobility.cable_rotation(accelerations[i])
            rotation_quad = mobility.quadrotor_rotation(accelerations[i])
            worst = max(
                worst,
                np.abs(boxes[0][1][i] - rotation_cable).max(),
                np.abs(boxes[2][1][i] - rotation_quad).max(),
                np.abs(
                    boxes[2][0][i]
                    - (
                        positions[i]
                        - (params.cable_length + params.robot_height / 2.0) * p_hat
                    )
                ).max(),
            )
    check("batched poses match Planner's conventions", worst < 1e-12, "max diff %.1e" % worst)

    # 9. the flatness singularity and NaN states are rejected, not silently accepted.
    checker = FlatEdgeChecker(params, mobility)
    free_fall = np.array([0, 0, 0, 0, 0, 0, 0, 0, -9.81])
    check(
        "singular / non-finite states rejected",
        not checker.is_state_valid(free_fall)
        and not checker.is_state_valid(np.full(9, np.nan))
        and float(np.linalg.norm(free_fall[6:9] + [0, 0, 9.81])) < ACC_NORM_EPS,
    )

    # 10. an edge through a wall is rejected; a clear one is accepted.
    into_wall = FlatMotion(
        fixed.coefficients_over(
            np.array([0.2, 0.0, 0.2, 0, 0, 0, 0, 0, 0], float).reshape(ORDER, OUTPUT_DIM),
            np.array([1.8, 0.0, 0.2, 0, 0, 0, 0, 0, 0], float).reshape(ORDER, OUTPUT_DIM),
            2.0,
        ),
        2.0,
    )
    check("edge through a wall is rejected", not checker.is_motion_valid(into_wall))

    # 11. end-to-end on a scenario that is reliably solvable, so this suite stays fast and
    # deterministic. (maze_1 is a deliberately tight maze: a straight edge through its
    # corridor is valid, but finding one is a narrow-passage search, so it takes far longer
    # and is exercised separately rather than gating this suite.)
    params = load_params()
    params.obstacles = {"wall1": dict(x=1.0, y=0.0, z=0.0, l=0.2, b=2.0, h=3.0)}
    params.initial_state = [0.0, 0.0, 0.0] + [0.0] * 6
    params.end_state = [2.5, 0.0, 0.0] + [0.0] * 6
    params.rrt_timeout = 60.0
    params.rrt_seed = 0
    started = time.time()
    result = solve_with_rrt(params)
    check(
        "RRT-Connect plans around a wall",
        result.success,
        "%.1fs, %d nodes" % (time.time() - started, len(result.explored_nodes)),
    )

    dense = FlatEdgeChecker(params, mobility)
    dense.step = checker.step / 10.0
    overturned = 0
    for motion in result.path.motions:
        if not dense.is_motion_valid(motion):
            overturned += 1
    check(
        "accepted edges survive a 10x denser check",
        overturned == 0,
        "%d overturned" % overturned,
    )

    # 12. the returned path is continuous, on-bounds, and reaches the goal exactly.
    path = result.path
    worst = 0.0
    for state, motion in zip(path.states[:-1], path.motions):
        worst = max(worst, np.abs(motion.flat_state(0.0) - state).max())
        if not is_motion_within_bounds(motion, params):
            check("path edges satisfy state/input bounds", False)
    goal_error = np.abs(path.states[-1] - np.asarray(params.end_state, float)).max()
    check(
        "path is continuous, bounded and reaches the goal",
        worst < 1e-9 and goal_error < 1e-9 and is_motion_within_bounds(path.motions[0], params),
        "continuity %.1e, goal %.1e" % (worst, goal_error),
    )

    # 13. seed sizes satisfy the assertions in Planner.initialize_variables.
    positions, velocities, inputs = resample_path_for_initial_guess(path, params)
    check(
        "warm-start seeds have the shapes initialize_variables needs",
        len(velocities) == len(positions) and len(inputs) == len(positions) - 1,
        "pos %d, vel %d, input %d" % (len(positions), len(velocities), len(inputs)),
    )

    # 14. CSV payload layout matches what save_result/interpolate expect.
    sol_values = path_to_sol_values(path, params)
    rows, cols = sol_values["x"].shape
    check(
        "sol_values layout is (N+1, 9) / (N, 3) / (N,)",
        cols == 9
        and sol_values["u"].shape == (rows - 1, 3)
        and sol_values["t"].shape == (rows - 1,),
        "x %s, u %s, t %s" % (sol_values["x"].shape, sol_values["u"].shape, sol_values["t"].shape),
    )

    # 15. planning is reproducible for a fixed seed.
    other = solve_with_rrt(params)
    check(
        "fixed seed reproduces the same path",
        other.success
        and abs(other.path.duration() - path.duration()) < 1e-12
        and len(other.path.motions) == len(path.motions),
    )

    # 16. the fixed-duration steerer is selectable and its edges are valid closed-form
    # solutions. It is checked on nearby states rather than by planning a whole maze: because
    # its duration does not scale with distance, a step-limited extension toward a distant
    # sample is a poor fit for it, so it is not a drop-in replacement for minimum-effort.
    fixed_steerer = FixedDurationSteering(duration=1.5)
    rng_steer = np.random.default_rng(5)
    worst = 0.0
    accepted = 0
    for _ in range(200):
        a = np.zeros(9)
        b = a + rng_steer.normal(0, 0.05, 9)
        b[0:3] *= 4.0  # short moves with modest derivatives
        motion = fixed_steerer.steer(a, b)
        if motion is None:
            continue
        if not checker.is_motion_valid(motion):
            continue
        accepted += 1
        worst = max(
            worst,
            np.abs(motion.flat_state(0.0) - a).max(),
            np.abs(motion.flat_state(motion.duration) - b).max(),
        )
    params.rrt_steering = "fixed"
    check(
        "fixed-duration steerer produces valid closed-form edges",
        accepted > 0 and worst < 1e-10 and isinstance(make_steering(params), FixedDurationSteering),
        "%d/200 accepted, boundary residual %.1e" % (accepted, worst),
    )
    params.rrt_steering = "minimum_effort"

    # 17. the closed forms must not lose a critical point on the degenerate polynomials the
    # solver is prone to: near-double roots (what a symmetric manoeuvre produces) and exactly
    # repeated ones. Under-bracketing is the failure that matters -- it would accept a
    # trajectory that leaves the limits -- so that is what this asserts.
    adversarial = []
    for _ in range(4000):
        r = rng.normal(0, 1, 3)
        adversarial += [
            np.poly([r[0], r[0], r[1], r[1], r[2]])[::-1],
            np.poly([r[0]] * 3 + [r[1], r[1]])[::-1],
            np.poly([r[0], r[0], r[0], r[1], r[2]])[::-1],
            np.poly([r[0], r[0], r[1], r[2], r[2]])[::-1],
        ]
    for _ in range(6000):
        adversarial.append(rng.normal(0, 1, rng.integers(1, 6)))
    under = 0
    worst_under = 0.0
    # Tolerance is 1e-7, not 1e-9: splitting a near-double root loses a few digits to
    # cancellation in the discriminant, which shows up here at ~2e-8 (measured worst over
    # 300k polynomials). That is nanometres in state units and far inside the eps the limits
    # check already tolerates. The failure this guards against is structural -- a critical
    # point missed outright -- and that lands at order 1, so 1e-7 separates the two cleanly.
    for coefficients in adversarial:
        duration_here = float(rng.uniform(0.05, 8.0))
        low, high = poly_extrema(coefficients, duration_here)
        values = np.polyval(np.asarray(coefficients)[::-1], np.linspace(0, duration_here, 20001))
        scale = max(1.0, float(np.abs(values).max()))
        gap = max(float(values.min()) - low, high - float(values.max()))
        if gap > 1e-7 * scale:
            under += 1
            worst_under = max(worst_under, gap / scale)
    check(
        "closed-form extrema never under-bracket",
        under == 0,
        "%d/%d under, worst %.1e" % (under, len(adversarial), worst_under),
    )

    # 18. the bound-aware steerer keeps its promise: every motion it hands back is admissible,
    # and it only moves away from the effort optimum when the optimum is not.
    limits = FlatLimits.from_params(params)
    plain = MinimumEffortSteering(rho=float(params.rrt_rho))
    aware = MinimumEffortSteering(rho=float(params.rrt_rho), limits=limits)
    rng_b = np.random.default_rng(11)
    inadmissible = 0
    deviated = rescued = 0
    for _ in range(600):
        a = np.zeros(9)
        b = np.zeros(9)
        a[0:3] = rng_b.uniform(-1.0, 4.5, 3)
        b[0:3] = rng_b.uniform(-1.0, 4.5, 3)
        a[2] = b[2] = rng_b.uniform(0.05, 0.45)
        a[3:6] = rng_b.uniform(-1.5, 1.5, 3)
        b[3:6] = rng_b.uniform(-1.5, 1.5, 3)
        a[6:9] = rng_b.uniform(-0.3, 0.3, 3)
        b[6:9] = rng_b.uniform(-0.3, 0.3, 3)
        optimum = plain.steer(a, b)
        optimum_ok = optimum is not None and is_motion_within_bounds(optimum, params)
        motion = aware.steer(a, b)
        if motion is None:
            continue
        if not is_motion_within_bounds(motion, params):
            inadmissible += 1
        if optimum_ok:
            # must not have wandered off the optimum when the optimum was already fine
            if abs(motion.duration - optimum.duration) > 1e-12:
                deviated += 1
        else:
            rescued += 1
    check(
        "bound-aware steerer only returns admissible motions",
        inadmissible == 0 and deviated == 0,
        "%d inadmissible, %d needless deviations, %d rescued of 600"
        % (inadmissible, deviated, rescued),
    )

    # 19. the early exit used on the hot path must not change the verdict.
    mismatched = 0
    for _ in range(400):
        a = rng_b.normal(0, 1, 9) * np.array([1.5, 1.5, 0.15, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3])
        b = rng_b.normal(0, 1, 9) * np.array([1.5, 1.5, 0.15, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3])
        motion = plain.steer(a, b)
        if motion is None:
            continue
        rows = motion.coefficients.tolist()
        full = _bound_violation(motion, limits, rows) <= 0.0
        quick = _bound_violation(motion, limits, rows, early_exit=True) <= 0.0
        if full != quick:
            mismatched += 1
    check("early-exit bounds check agrees with the full score", mismatched == 0)

    print("\nAll %d checks passed." % len(RESULTS))


if __name__ == "__main__":
    main()
