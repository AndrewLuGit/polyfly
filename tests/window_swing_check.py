"""Checks for the swing-through-the-window scenario (data/params/experiments/window.yaml).

The scenario is a wall with a window in it.  The payload hangs 0.617 m below the quadrotor,
so with the cable vertical the pair stands 0.717 m tall and cannot fit the opening; both
bodies are inside the wall at once for part of the crossing, so the pair has to be
"shortened" -- the cable swung by roughly 40 deg -- for as long as the quadrotor is in the
wall.  Swinging it along the flight direction (the payload lagging, then following) is the
cheap way to do that: the payload's speed changes by a couple of m/s and no sideways room is
needed.

This script rebuilds that swing from its acceleration programme and validates it against the
planner's own collision geometry (``FlatEdgeChecker``, the same orientation-aware
OBB-vs-box test the OCP's constraints mirror), then checks that the warm start stored in the
YAML is exactly that swing.

Run with the repo's dev environment:

    POLYFLY_DIR=$(pwd) PYTHONPATH=src python3 tests/window_swing_check.py

There is no pytest in this project's environment, so this is a plain script that prints
PASS/FAIL per check and exits non-zero on the first failure.
"""

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
from poly_fly.optimal_planner.planner import dictToClass, yamlToDict  # noqa: E402
from poly_fly.utils.utils import MPC  # noqa: E402

GRAVITY = 9.81
YAML = "experiments/window.yaml"
DT = 0.002

# The swing's acceleration programme, in the flight direction: ease up to A1, ramp to A2 and
# hold it while the cable threads the wall, release, cruise, then brake to rest.  The phases
# are triggered by the payload's position, not by time, so the geometry is what sets them.
A1, A2, SLOPE, BRAKE, SETTLE = 0.9, 11.0, 70.0, 12.0, 0.45
X_LEAN, X_RELEASE = 0.0, 1.0

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition)))
    print("%-4s %-52s %s" % ("PASS" if condition else "FAIL", name, detail))
    if not condition:
        sys.exit(1)


def load_params(relative=YAML):
    path = os.path.join(os.environ["POLYFLY_DIR"], "data/params", relative)
    return dictToClass(MPC, yamlToDict(path))


# --------------------------------------------------------------------------------------
# the swing
# --------------------------------------------------------------------------------------
def integrate_programme(phases, slope, x_start, dt=DT, t_max=20.0):
    """Ramp a_x toward each phase's target at `slope` until the payload passes that phase's
    x trigger (``None`` means "until the payload comes to rest").  Returns t, a, v, x."""
    ts, accs, vels, xs = [0.0], [0.0], [0.0], [x_start]
    x, v, a, t, idx = x_start, 0.0, 0.0, 0.0, 0
    while t < t_max:
        target, trigger = phases[min(idx, len(phases) - 1)]
        a_new = a + np.clip(target - a, -slope * dt, slope * dt)
        v_new = v + 0.5 * (a + a_new) * dt
        if trigger is None and v_new < 0.0:
            v_new = 0.0  # the payload stops; it does not roll back
        x_new = x + 0.5 * (v + v_new) * dt
        if idx < len(phases) - 1:
            if (trigger is not None and x_new >= trigger) or (trigger is None and v_new <= 0.0):
                idx += 1
        x, v, a, t = x_new, v_new, a_new, t + dt
        ts.append(t)
        accs.append(a)
        vels.append(v)
        xs.append(x)
        if idx >= len(phases) - 1 and abs(v) <= 1e-9 and abs(a) <= 1e-9:
            break  # settled on the goal
    return np.array(ts), np.array(accs), np.array(vels), np.array(xs)


def build_swing(x_start, x_goal, brake=BRAKE, dt=DT):
    """Solve where the brake has to start so the payload stops exactly on the goal (the
    deceleration ramps up at SLOPE, which adds a little distance over v^2/2a), then
    integrate."""
    def run(x_brake):
        phases = [
            (A1, X_LEAN),
            (A2, X_RELEASE),
            (0.0, x_brake),
            (-brake, None),
            (0.0, None),
        ]
        return integrate_programme(phases, SLOPE, x_start, dt)

    # the final position rises monotonically with where the brake starts, so bisect on it
    low, high = X_RELEASE, x_goal + 1.0
    for _ in range(50):
        x_brake = 0.5 * (low + high)
        t, a, v, x = run(x_brake)
        if x[-1] < x_goal:
            low = x_brake
        else:
            high = x_brake
        if high - low < 1e-9:
            break
    # settle: a short hover on the goal, so the seed ends quiescent (the OCP pins the
    # initial and final jerks to zero) and the knot count matches the horizon.
    settle = np.arange(dt, SETTLE + 1e-9, dt)
    t = np.concatenate([t, t[-1] + settle])
    a = np.concatenate([a, np.zeros_like(settle)])
    v = np.concatenate([v, np.zeros_like(settle)])
    x = np.concatenate([x, np.full_like(settle, x[-1])])
    return t, a, v, x


def swing_states(params, dt=DT):
    z = params.initial_state[2]
    t, a, v, x = build_swing(params.initial_state[0], params.end_state[0], dt=dt)
    pos = np.stack([x, np.zeros_like(x), np.full_like(x, z)], axis=1)
    vel = np.stack([v, np.zeros_like(x), np.zeros_like(x)], axis=1)
    acc = np.stack([a, np.zeros_like(x), np.zeros_like(x)], axis=1)
    return t, pos, vel, acc


def state(pos, vel, acc):
    s = np.zeros(9)
    s[:3], s[3:6], s[6:9] = pos, vel, acc
    return s


def tilt_deg(acc):
    return np.degrees(np.arctan2(acc[..., 0], acc[..., 2] + GRAVITY))


# --------------------------------------------------------------------------------------
def main():
    params = load_params()
    if "--write" in sys.argv:  # regenerate the stored warm start first, then check it
        write_yaml(params, *swing_states(params))
        params = load_params()
    checker = FlatEdgeChecker(params, _mobility_for(params))
    wall = params.obstacles["sill"]
    x_lo = wall["x"] - wall["l"] / 2.0 - params.margin
    x_hi = wall["x"] + wall["l"] / 2.0 + params.margin
    lintel = params.obstacles["lintel"]
    sill = params.obstacles["sill"]
    win_z = (sill["z"] + sill["h"] / 2.0, lintel["z"] - lintel["h"] / 2.0)
    jamb = params.obstacles["jamb_left"]
    win_y = abs(jamb["y"]) - jamb["b"] / 2.0  # inner face of the jamb = window half width
    print("window scenario: %s" % YAML)
    print("window y in [%.2f, %.2f], z in [%.2f, %.2f]  (%.2f x %.2f m, clear %.2f m tall)"
          % (-win_y, win_y, win_z[0], win_z[1], 2 * win_y, win_z[1] - win_z[0],
             win_z[1] - win_z[0] - 2 * params.margin))
    print("wall's inflated x span [%.2f, %.2f], margin %.2f\n" % (x_lo, x_hi, params.margin))

    # -- 1. a vertical cable cannot cross -------------------------------------------------
    ys = np.arange(params.state_min[1], params.state_max[1] + 1e-9, 0.01)
    zs = np.arange(params.state_min[2], params.state_max[2] + 1e-9, 0.01)
    stations = np.arange(x_lo + params.payload_radius, x_hi - params.payload_radius + 1e-9, 0.025)
    open_stations = []
    for x in stations:
        for y in ys:
            for z in zs:
                if checker.is_state_valid(state((x, y, z), np.zeros(3), np.zeros(3))):
                    open_stations.append((x, y, z))
                    break
            else:
                continue
            break
    check(
        "vertical cable: no station inside the wall is passable",
        not open_stations,
        "%d stations swept x payload box inside the wall; %d passable%s"
        % (len(stations), len(open_stations),
           (" e.g. x=%.2f y=%.2f z=%.2f" % open_stations[0]) if open_stations else ""),
    )

    # -- 2. the swing crosses, densely checked ---------------------------------------------
    t, pos, vel, acc = swing_states(params)
    bad = [i for i in range(len(pos)) if not checker.is_state_valid(state(pos[i], vel[i], acc[i]))]
    check(
        "the swing is collision free and on bounds",
        not bad,
        "%d of %d samples invalid (%s..%s s, peak speed %.2f m/s)"
        % (len(bad), len(pos), ("%.2f" % t[bad[0]]) if bad else "-", "%.2f" % t[-1],
           np.abs(vel[:, 0]).max()),
    )

    # -- 3. ... and it really is a swing ---------------------------------------------------
    inside = (pos[:, 0] > x_lo) & (pos[:, 0] < x_hi)
    peak = np.abs(tilt_deg(acc)[inside]).max() if inside.any() else 0.0
    check(
        "the cable is leaned by at least 35 deg inside the wall",
        peak >= 35.0,
        "peak lean %.1f deg over %.2f m of travel (0 deg is impassable)" % (peak, 0.4),
    )

    # -- 4. the same path with the cable held vertical must fail ---------------------------
    n_bad = sum(
        1
        for i in range(len(pos))
        if not checker.is_state_valid(state(pos[i], vel[i], np.zeros(3)))
    )
    check(
        "the same path with a vertical cable collides",
        n_bad > 0,
        "%d of %d samples collide" % (n_bad, len(pos)),
    )

    # -- 5. the YAML stores exactly this swing as its warm start ---------------------------
    positions = np.array(params.payload_pos_init, dtype=float)
    velocities = np.array(params.payload_vel_init, dtype=float)
    inputs = np.array(params.payload_input_init, dtype=float)
    horizon = int(params.horizon)
    check(
        "warm start has the shapes initialize_variables needs",
        len(positions) == horizon + 1 and len(velocities) == horizon + 1
        and len(inputs) == horizon,
        "horizon %d, pos %d, vel %d, input %d"
        % (horizon, len(positions), len(velocities), len(inputs)),
    )
    times = np.linspace(0.0, t[-1], horizon + 1)
    want_pos = np.stack([np.interp(times, t, pos[:, i]) for i in range(3)], axis=1)
    want_vel = np.stack([np.interp(times, t, vel[:, i]) for i in range(3)], axis=1)
    jerk = np.stack([np.gradient(acc[:, i], t) for i in range(3)], axis=1)
    want_in = np.stack([np.interp(times[:-1], t, jerk[:, i]) for i in range(3)], axis=1)
    check(
        "warm start matches the validated swing",
        np.abs(positions - want_pos).max() < 5e-4
        and np.abs(velocities - want_vel).max() < 5e-4
        and np.abs(inputs - want_in).max() < 1.0,
        "max offsets: pos %.1e m, vel %.1e m/s, jerk %.1e m/s^3"
        % (np.abs(positions - want_pos).max(), np.abs(velocities - want_vel).max(),
           np.abs(inputs - want_in).max()),
    )

    # -- 6. start and goal -----------------------------------------------------------------
    goal = np.asarray(params.end_state[:3], dtype=float)
    check(
        "swing starts and ends on the scenario's states",
        np.abs(pos[0] - np.asarray(params.initial_state[:3])).max() < 1e-9
        and np.abs(pos[-1] - goal).max() < 0.03
        and abs(vel[-1, 0]) < 1e-6,
        "start %s, end %s" % (np.round(pos[0], 2), np.round(pos[-1], 2)),
    )

    print("\n%d checks, all passed." % len(RESULTS))
    print("peak lean %.1f deg | peak speed %.2f m/s | peak jerk %.1f m/s^3 | crossing %.2f s"
          % (peak, np.abs(vel[:, 0]).max(), np.abs(jerk).max(), t[-1]))


def write_yaml(params, t, pos, vel, acc):
    """Re-emit the warm-start block of the scenario from the validated swing."""
    horizon = int(params.horizon)
    times = np.linspace(0.0, t[-1], horizon + 1)
    positions = np.stack([np.interp(times, t, pos[:, i]) for i in range(3)], axis=1)
    velocities = np.stack([np.interp(times, t, vel[:, i]) for i in range(3)], axis=1)
    jerk = np.stack([np.gradient(acc[:, i], t) for i in range(3)], axis=1)
    inputs = np.stack([np.interp(times[:-1], t, jerk[:, i]) for i in range(3)], axis=1)

    path = os.path.join(os.environ["POLYFLY_DIR"], "data/params", YAML)
    with open(path) as fh:
        head = fh.read().split("payload_pos_init:")[0].rstrip() + "\n"
    lines = [head]
    lines.append("# Warm start: a dynamically consistent swing, regenerated by")
    lines.append("# tests/window_swing_check.py --write (it re-derives and validates it first).")
    lines.append("payload_pos_init: [")
    lines += ["  [%.4f, %.4f, %.3f]," % tuple(row) for row in positions]
    lines.append("]")
    lines.append("payload_vel_init: [")
    lines += ["  [%.4f, %.4f, %.4f]," % tuple(row) for row in velocities]
    lines.append("]")
    lines.append("payload_input_init: [")
    lines += ["  [%.2f, %.2f, %.2f]," % tuple(row) for row in inputs]
    lines.append("]")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("rewrote the warm start in %s (%.2f s of motion over %d knots)"
          % (path, t[-1], len(positions)))


if __name__ == "__main__":
    main()
