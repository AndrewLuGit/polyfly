"""Solve the swing-through-the-window scenario and check the answer independently.

    POLYFLY_DIR=$(pwd) PYTHONPATH=src python3 tests/window_ocp_check.py [--yaml experiments/window.yaml]

The OCP's collision constraints are only imposed at its own knots, and its geometry is a
little more permissive than the checker's (it models the payload as axis aligned and asks for
an L2 standoff, while ``FlatEdgeChecker`` rotates the payload with the cable and inflates the
obstacles by ``margin`` in each axis).  On a window this tight the difference matters: with
knots ~0.3 m apart a solution can satisfy every knot and still put the cable through the
lintel in between.  So this script rolls the solution out with the OCP's own dynamics (right
hand side, RK4, jerk held over each interval) at a much finer step and validates every sample
with the checker -- and it reports the largest margin the whole trajectory still clears, which
is the number that says how much room the swing actually had.

Exit status is non-zero if the solution misses the goal or if any dense sample collides.
"""

import argparse
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import torch  # noqa: F401  (imported but unused by planner.py)
except ImportError:  # keep the planner importable in a torch-free environment
    sys.modules["torch"] = types.ModuleType("torch")

from poly_fly.optimal_planner.kinodynamic_rrt_connect import (  # noqa: E402
    FlatEdgeChecker,
    _mobility_for,
)
from poly_fly.optimal_planner.planner import Planner, dictToClass, yamlToDict  # noqa: E402
from poly_fly.utils.utils import MPC  # noqa: E402

GRAVITY = 9.81
DENSE_STEP = 0.01  # seconds between the validator's samples


def rk4(state, jerk, dt):
    """One RK4 step of the OCP's flat dynamics: d/dt (p, v, a) = (v, a, jerk)."""

    def fdot(x):
        return np.concatenate([x[3:6], x[6:9], jerk])

    k1 = fdot(state)
    k2 = fdot(state + dt / 2 * k1)
    k3 = fdot(state + dt / 2 * k2)
    k4 = fdot(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(states, jerks, steps):
    """Dense trajectory between the OCP's knots, using the OCP's own integration."""
    dense = [states[0]]
    for i, dt in enumerate(steps):
        n = max(2, int(round(dt / DENSE_STEP)))
        sub = dt / n
        current = states[i].copy()
        for _ in range(n):
            current = rk4(current, jerks[i], sub)
            dense.append(current.copy())
    return np.array(dense)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="experiments/window.yaml")
    args = parser.parse_args()

    params = dictToClass(
        MPC, yamlToDict(os.path.join(os.environ["POLYFLY_DIR"], "data/params", args.yaml))
    )
    planner = Planner(params, plot=False)
    planner.setup(viz_cb=False)
    solution, _, success, iterations, opt_time = planner.optimize()
    print("\nSolver: success=%s iterations=%s opt_time=%.1fs" % (success, iterations, opt_time))
    if not success:
        print("FAIL: the solver did not converge")
        return 1

    states = solution["x"]
    jerks = solution["u"]
    steps = np.array(
        [float(planner.opti.value(planner.variables["t"][i])) for i in range(params.horizon)]
    )
    dense = rollout(states, jerks, steps)
    print("horizon %d knots, total time %.2f s (knot spacing %.3f s, min %.3f s)" %
          (params.horizon, steps.sum(), steps.mean(), steps.min()))
    print("dense rollout: %d samples at %.3f s | peak |v| %.2f m/s | peak |a| %.2f m/s^2 | "
          "peak |jerk| %.1f m/s^3" %
          (len(dense), DENSE_STEP, np.abs(dense[:, 3:6]).max(), np.abs(dense[:, 6:9]).max(),
           np.abs(jerks).max()))

    def cleared(margin):
        checker = FlatEdgeChecker(params, _mobility_for(params), margin=margin)
        return all(checker.is_state_valid(state) for state in dense)

    checker = FlatEdgeChecker(params, _mobility_for(params))
    invalid = [i for i, state in enumerate(dense) if not checker.is_state_valid(state)]
    low, high = 0.0, 0.15
    for _ in range(14):
        mid = 0.5 * (low + high)
        if cleared(mid):
            low = mid
        else:
            high = mid

    tilt = np.degrees(np.arctan2(dense[:, 6], dense[:, 8] + GRAVITY))
    wall = params.obstacles["sill"]
    inside = (dense[:, 0] > wall["x"] - wall["l"] / 2.0) & (dense[:, 0] < wall["x"] + wall["l"] / 2.0)
    goal = np.asarray(params.end_state[:3], dtype=float)
    goal_error = np.abs(dense[-1, :3] - goal).max()

    print("\nCheck against FlatEdgeChecker (margin %.2f):" % params.margin)
    print("  colliding samples : %d of %d%s" %
          (len(invalid), len(dense),
           ("  first at x=%.2f, tilt %.1f deg" % (dense[invalid[0], 0], tilt[invalid[0]]))
           if invalid else ""))
    print("  lean inside wall  : %s" %
          ("%.1f .. %.1f deg" % (tilt[inside].min(), tilt[inside].max()) if inside.any()
           else "never inside"))
    print("  largest margin the whole trajectory clears: %.3f m" % low)
    print("  goal error        : %.2e m (tolerance %.2f)" % (goal_error, 0.05))

    ok = not invalid and goal_error <= 0.05
    print("\n%s" % ("PASS: the planned swing clears the window, densely verified."
                    if ok else
                    "FAIL: the plan is not collision free between the OCP's knots -- tighten "
                    "max_dt (finer knots) or the geometry."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
