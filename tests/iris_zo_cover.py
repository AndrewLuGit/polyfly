"""Grow and validate a convex cover of the payload's collision-free configuration space.

    POLYFLY_DIR=$(pwd) PYTHONPATH=src python3 tests/iris_zo_cover.py \
        --yaml experiments/maze_1.yaml

The configuration is ``q = [p, a]`` (payload position and acceleration); cable and
quadrotor poses follow from the flatness map, and collision checking uses
``FlatEdgeChecker.configs_collision_free``, i.e. the same payload/cable/quadrotor boxes as
the OCP and the RRT.

The script grids the configuration box implied by ``state_min``/``state_max``, draws one
IRIS-ZO region at **every collision-free seed**, and reports
  * each region's independently measured colliding-sample fraction (hit-and-run samples),
  * the fraction of uniformly sampled collision-free configurations covered by the union,
  * the fraction of the whole domain covered.

It also runs quick self-tests of the MVEE, the sampler, the 6-D checker's equivalence with
``is_state_valid``, and a 2-D toy IRIS-ZO with a known answer.  The process exits non-zero
if a check fails.

Grid defaults are tuned for maze_1: ``--pos-step 1.0`` and three acceleration levels per
axis (``--acc-step 0``), giving ~1314 free seeds and a ~6-8 minute run.  The 1.0 step is
deliberate: the S-corridor's central gap is at x=2, and the coarser 2.0 grid has x planes
{-1, 1, 3, 5}, so it never seeds that gap and leaves the corridor uncovered.  The IRIS
preset is ``epsilon=0.05``, ``delta=0.1``, two outer iterations,
coordinate hit-and-run with 20 mixing steps, and the absolute volume termination disabled
(``--termination-threshold 1e-6``) because valid 6-D regions can be smaller than Drake's
default threshold.  Use ``--drake-defaults`` for the paper's stricter settings, and
``--max-seeds``/``--time-budget`` as safety valves on larger environments.

Every region is also measured as it is grown (``--grow-validation-samples``, 100 mixing
steps).  If its colliding fraction exceeds ``--grow-validation-threshold`` (5%), it is
regrown with a stricter, better-mixed solve, up to ``--max-retries`` times.  This matters
in narrow passages, where too few mixing steps can make the statistical test accept a
thin region that still contains a collision sliver.  The final per-region validation is
run separately with ``--validate-samples`` and ``--validation-mixing-steps``.

``--plot`` (or ``--save-plot``) draws an x-y cross-section of the cover; four sliders
sweep the z position and the x, y and z acceleration components.  The solid rectangles
are obstacles at the current z, the dashed outlines their ``margin`` inflation used by the
collision checker, and the dots are the free seeds projected onto x-y.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from poly_fly.optimal_planner.flat_collision import FlatEdgeChecker  # noqa: E402
from poly_fly.optimal_planner.iris_zo import (  # noqa: E402
    Ellipsoid,
    EmptyPolytopeError,
    HPolyhedron,
    IrisZoError,
    IrisZoOptions,
    iris_zo,
    payload_domain_from_params,
    unit_ball_volume,
)
from poly_fly.utils.utils import MPC, dictToClass, yamlToDict  # noqa: E402

# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.results = []

    def check(self, name, condition, detail=""):
        ok = bool(condition)
        self.results.append((name, ok))
        print("%-4s %-58s %s" % ("PASS" if ok else "FAIL", name, detail))
        return ok

    def failed(self):
        return any(not ok for _, ok in self.results)


# --------------------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------------------
def axis_values(lower, upper, step):
    """Evenly spaced values including both endpoints; the step is an upper bound."""
    span = float(upper - lower)
    if span <= 0.0:
        raise ValueError("axis has non-positive width")
    if step <= 0.0:
        raise ValueError("grid steps must be positive")
    count = int(math.ceil(span / step - 1e-9)) + 1
    return np.linspace(lower, upper, max(count, 2))


def build_grid(lower, upper, pos_step, acc_step, z_levels):
    x = axis_values(lower[0], upper[0], pos_step)
    y = axis_values(lower[1], upper[1], pos_step)
    z = (
        np.asarray([float(v) for v in z_levels], dtype=float)
        if z_levels
        else axis_values(lower[2], upper[2], pos_step)
    )
    if acc_step > 0.0:
        ax = axis_values(lower[3], upper[3], acc_step)
        ay = axis_values(lower[4], upper[4], acc_step)
        az = axis_values(lower[5], upper[5], acc_step)
    else:
        ax = np.linspace(lower[3], upper[3], 3)
        ay = np.linspace(lower[4], upper[4], 3)
        az = np.linspace(lower[5], upper[5], 3)
    gx, gy, gz, gax, gay, gaz = np.meshgrid(x, y, z, ax, ay, az, indexing="ij")
    seeds = np.column_stack(
        [gx.ravel(), gy.ravel(), gz.ravel(), gax.ravel(), gay.ravel(), gaz.ravel()]
    )
    return seeds, (x, y, z, ax, ay, az)


def collision_free_chunked(checker, configs, chunk=20000):
    result = np.zeros(len(configs), dtype=bool)
    for start in range(0, len(configs), chunk):
        stop = min(start + chunk, len(configs))
        result[start:stop] = checker.configs_collision_free(configs[start:stop])
    return result


def collision_fraction(checker, region, rng, n_samples, mixing_steps):
    """Independent Monte Carlo estimate of a region's colliding volume fraction."""
    if n_samples <= 0:
        return float("nan")
    samples = region.polytope.uniform_sample(rng, n_samples, mixing_steps)
    return float(1.0 - checker.configs_collision_free(samples).mean())


# --------------------------------------------------------------------------------------
# self-tests
# --------------------------------------------------------------------------------------
def self_tests(checks, params, checker):
    # MVEE: unit box and an anisotropic box have the box-shaped ellipsoid.
    for half in (np.ones(6), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])):
        polytope = HPolyhedron.from_box(-half, half)
        ellipsoid = polytope.maximum_volume_inscribed_ellipsoid()
        expected_volume = unit_ball_volume(6) * float(np.prod(half))
        support = np.linalg.norm(
            polytope.A @ np.linalg.inv(ellipsoid.A), axis=1
        )
        violation = float(
            np.max(polytope.A @ ellipsoid.center + support - polytope.b)
        )
        checks.check(
            "MVEE box %s" % np.array2string(half, precision=0),
            abs(ellipsoid.volume - expected_volume) <= 1e-4 * expected_volume
            and violation <= 1e-7,
            "vol %.6f expected %.6f, max face violation %.2e"
            % (ellipsoid.volume, expected_volume, violation),
        )

    # The right-isosceles triangle has a known Steiner inellipse.
    A = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    b = np.array([0.0, 0.0, 1.0])
    ellipsoid = HPolyhedron(A, b).maximum_volume_inscribed_ellipsoid()
    expected_area = math.pi / (6.0 * math.sqrt(3.0))
    checks.check(
        "MVEE triangle (Steiner inellipse)",
        abs(ellipsoid.volume - expected_area) <= 1e-3 * expected_area,
        "area %.6f expected %.6f, centre %s"
        % (ellipsoid.volume, expected_area, np.array2string(ellipsoid.center, precision=5)),
    )

    # Hit-and-run: uniform in [-1,1]^6 has zero mean and variance 1/3.
    box = HPolyhedron.from_box(-np.ones(6), np.ones(6))
    samples = box.uniform_sample(np.random.default_rng(0), 20000, mixing_steps=30)
    violation = float(np.max(box.A @ samples.T - box.b[:, None]))
    means = np.abs(samples.mean(axis=0)).max()
    variances = np.abs(samples.var(axis=0) - 1.0 / 3.0).max()
    checks.check(
        "hit-and-run moments",
        violation <= 1e-6 and means <= 0.02 and variances <= 0.02,
        "max violation %.1e, |mean| %.4f, |var-1/3| %.4f"
        % (violation, means, variances),
    )

    # The 6-D batched checker must agree with the 9-D state checker.
    lo = np.asarray(params.state_min, dtype=float)
    hi = np.asarray(params.state_max, dtype=float)
    states = lo + (hi - lo) * np.random.default_rng(1).random((300, 9))
    configs = states[:, [0, 1, 2, 6, 7, 8]]
    state_free = np.array([checker.is_state_valid(s) for s in states])
    config_free = checker.configs_collision_free(configs)
    checks.check(
        "6-D checker == is_state_valid",
        np.array_equal(state_free, config_free),
        "%d/%d valid" % (int(state_free.sum()), len(states)),
    )
    singular = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -9.81]])
    checks.check(
        "free-fall configuration rejected",
        not bool(checker.configs_collision_free(singular)[0]),
    )
    nan_config = np.array([[np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]])
    checks.check(
        "NaN configuration rejected",
        not bool(checker.configs_collision_free(nan_config)[0]),
    )

    # 2-D toy: obstacle is the half-space x > 0.5.
    domain = HPolyhedron.from_box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    collision_free = lambda X: np.asarray(X)[:, 0] <= 0.5  # noqa: E731
    options = IrisZoOptions(
        num_particles=300,
        epsilon=0.05,
        delta=0.1,
        max_iterations=2,
        max_iterations_separating_planes=5,
        max_separating_planes_per_iteration=5,
        mixing_steps=30,
        relax_margin=True,
        random_seed=7,
    )
    region = iris_zo(
        collision_free,
        Ellipsoid.from_center_radius(np.array([-0.5, 0.0]), 0.05),
        domain,
        options,
    )
    validation = region.polytope.uniform_sample(
        np.random.default_rng(8), 5000, mixing_steps=30
    )
    colliding = float(1.0 - collision_free(validation).mean())
    polygon_area = _section_area(region.polytope)
    checks.check(
        "2-D toy IRIS-ZO",
        bool(region.contains(np.array([[-0.5, 0.0]]))[0])
        and colliding == 0.0
        and 2.9 <= polygon_area <= 3.05,
        "area %.4f (expected ~2.97), colliding %.3f"
        % (polygon_area, colliding),
    )


def _section_area(polytope):
    """Exact area of a 2-D polytope, by integrating the y interval over x."""
    xs = np.linspace(-1.0, 1.0, 4001)
    total = 0.0
    for x in xs:
        lower, upper, ok = -np.inf, np.inf, True
        for a, b in zip(polytope.A, polytope.b):
            rhs = b - a[0] * x
            if abs(a[1]) < 1e-12:
                if rhs < 0.0:
                    ok = False
                    break
            elif a[1] > 0.0:
                upper = min(upper, rhs / a[1])
            else:
                lower = max(lower, rhs / a[1])
        if ok and upper > lower:
            total += (upper - lower) * (xs[1] - xs[0])
    return total


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
def obstacle_rectangles(params, z):
    """Obstacle x-y footprints at height ``z``: actual and margin-inflated boxes."""
    actual, inflated = [], []
    margin = float(params.margin)
    for key in params.obstacles.keys():
        obs = params.obstacles[key]
        if abs(z - obs["z"]) > obs["h"] / 2.0:
            continue
        half_l, half_b = obs["l"] / 2.0, obs["b"] / 2.0
        actual.append((obs["x"] - half_l, obs["y"] - half_b, obs["l"], obs["b"]))
        inflated.append(
            (
                obs["x"] - half_l - margin,
                obs["y"] - half_b - margin,
                obs["l"] + 2.0 * margin,
                obs["b"] + 2.0 * margin,
            )
        )
    return actual, inflated


def section_polygon_2d(polytope, z, acceleration):
    """Ordered vertices of the x-y cross-section at fixed ``(z, a)``, or ``None``.

    A convex 2-D polygon's vertices are exactly the feasible pairwise intersections of its
    boundary lines, so no LP or convex-hull library is needed.  This keeps the slider
    redraw cheap even with >1000 regions.
    """
    a0 = np.asarray(acceleration, dtype=float)
    fixed = np.array([z, a0[0], a0[1], a0[2]])
    A_xy = polytope.A[:, :2]
    b_slice = polytope.b - polytope.A[:, 2:] @ fixed
    norms = np.linalg.norm(A_xy, axis=1)
    keep = norms > 1e-9
    if np.any(b_slice[~keep] < 0.0):
        return None
    A_xy, b_slice = A_xy[keep], b_slice[keep]
    m = len(b_slice)
    if m < 3:
        return None

    first, second = np.triu_indices(m, k=1)
    det = A_xy[first, 0] * A_xy[second, 1] - A_xy[first, 1] * A_xy[second, 0]
    valid = np.abs(det) > 1e-12
    if not valid.any():
        return None
    first, second, det = first[valid], second[valid], det[valid]
    x = (b_slice[first] * A_xy[second, 1] - b_slice[second] * A_xy[first, 1]) / det
    y = (A_xy[first, 0] * b_slice[second] - A_xy[second, 0] * b_slice[first]) / det
    points = np.column_stack([x, y])

    scale = 1.0 + float(np.max(np.abs(b_slice)))
    feasible = np.all(points @ A_xy.T <= b_slice + 1e-7 * scale, axis=1)
    points = points[feasible]
    if len(points) < 3:
        return None
    # Several constraints can meet at one corner; collapse the duplicates.
    _, unique = np.unique(np.round(points / (1e-9 * scale)), axis=0, return_index=True)
    points = points[np.sort(unique)]
    if len(points) < 3:
        return None
    # All remaining points are vertices; sort them around their mean for a closed polygon.
    center = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    points = points[order]
    area = 0.5 * abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )
    if area < 1e-10:
        return None
    return points


def plot_cover(
    params,
    regions,
    free_seeds,
    position_bounds,
    acceleration_bounds,
    initial_slice,
    max_regions,
    path,
    show,
):
    """Interactive x-y slice of the cover with sliders for z and the accelerations."""
    import matplotlib

    if path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.widgets import Slider

    lo, hi = position_bounds
    a_lo, a_hi = acceleration_bounds
    z0, ax0, ay0, az0 = initial_slice
    z0 = float(np.clip(z0, lo[2], hi[2]))
    ax0, ay0, az0 = np.clip([ax0, ay0, az0], a_lo, a_hi)

    fig, plot_ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.32)
    if max_regions <= 0 or len(regions) <= max_regions:
        drawn_regions = regions
    else:
        # Seed order is x-slowest, so the first N would all come from one x plane.
        stride = max(1, len(regions) // max_regions)
        drawn_regions = regions[::stride][:max_regions]
    region_color = "tab:blue"
    region_alpha = 0.08
    seed_xy = free_seeds[:, :2] if len(free_seeds) else np.empty((0, 2))

    def draw(z, ax_value, ay_value, az_value, gui_redraw):
        plot_ax.clear()
        actual, inflated = obstacle_rectangles(params, z)
        for x0, y0, width, height in inflated:
            plot_ax.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    height,
                    facecolor="none",
                    edgecolor="0.75",
                    linestyle="--",
                    linewidth=0.6,
                    zorder=1,
                )
            )
        for x0, y0, width, height in actual:
            plot_ax.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    height,
                    facecolor="0.55",
                    edgecolor="0.3",
                    alpha=0.65,
                    zorder=2,
                )
            )
        polygons = []
        for region in drawn_regions:
            vertices = section_polygon_2d(
                region.polytope, z, (ax_value, ay_value, az_value)
            )
            if vertices is not None:
                polygons.append(vertices)
        if polygons:
            # One collection instead of thousands of individual patches: this is what
            # keeps slider redraws fast for a full cover.
            plot_ax.add_collection(
                PolyCollection(
                    polygons,
                    facecolors=region_color,
                    edgecolors=region_color,
                    linewidths=0.2,
                    alpha=region_alpha,
                    zorder=3,
                )
            )
        handles = [
            Patch(facecolor=region_color, alpha=region_alpha, label="IRIS-ZO regions")
        ]
        if len(seed_xy):
            scatter = plot_ax.scatter(
                seed_xy[:, 0],
                seed_xy[:, 1],
                s=3,
                c="black",
                alpha=0.18,
                zorder=0,
                label="seeds (x-y projection)",
            )
            handles.append(scatter)
        plot_ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
        plot_ax.set_xlim(lo[0], hi[0])
        plot_ax.set_ylim(lo[1], hi[1])
        plot_ax.set_aspect("equal", adjustable="box")
        plot_ax.set_xlabel("x [m]")
        plot_ax.set_ylabel("y [m]")
        plot_ax.set_title(
            "IRIS-ZO cover slice: z = %.3f m, a = (%.2f, %.2f, %.2f) m/s$^2$\n"
            "showing %d of %d regions"
            % (z, ax_value, ay_value, az_value, len(drawn_regions), len(regions))
        )
        plot_ax.grid(True, color="0.9", linewidth=0.5)
        if gui_redraw:
            fig.canvas.draw_idle()

    slider_specs = (
        ("z [m]", lo[2], hi[2], z0),
        ("$a_x$ [m/s$^2$]", a_lo[0], a_hi[0], ax0),
        ("$a_y$ [m/s$^2$]", a_lo[1], a_hi[1], ay0),
        ("$a_z$ [m/s$^2$]", a_lo[2], a_hi[2], az0),
    )
    sliders = []
    for index, (label, vmin, vmax, value) in enumerate(slider_specs):
        slider_ax = fig.add_axes([0.18, 0.075 + 0.055 * (3 - index), 0.68, 0.032])
        sliders.append(Slider(slider_ax, label, vmin, vmax, valinit=value))

    def on_change(_):
        draw(sliders[0].val, sliders[1].val, sliders[2].val, sliders[3].val, True)

    for slider in sliders:
        slider.on_changed(on_change)

    draw(z0, ax0, ay0, az0, False)
    if path:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print("saved plot to %s" % path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yaml", default="experiments/maze_1.yaml")
    parser.add_argument("--pos-step", type=float, default=1.0)
    parser.add_argument(
        "--z-levels", default="", help="comma-separated z values; default uses --pos-step"
    )
    parser.add_argument(
        "--acc-step",
        type=float,
        default=0.0,
        help="acceleration grid step; 0 means three levels per axis",
    )
    parser.add_argument("--seed-radius", type=float, default=0.05)
    parser.add_argument("--max-seeds", type=int, default=0, help="0 = all free seeds")
    parser.add_argument("--time-budget", type=float, default=0.0, help="0 = unlimited")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--num-particles", type=int, default=300)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-inner", type=int, default=5)
    parser.add_argument("--max-planes", type=int, default=5)
    parser.add_argument("--mixing-steps", type=int, default=20)
    parser.add_argument("--bisection-steps", type=int, default=10)
    parser.add_argument(
        "--termination-threshold",
        type=float,
        default=0.0,
        help="absolute MVEE-volume change; 0 disables premature stop on tiny regions",
    )
    parser.add_argument("--relative-termination-threshold", type=float, default=1e-3)
    parser.add_argument("--configuration-space-margin", type=float, default=0.01)
    parser.add_argument(
        "--relax-margin", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mvee-iterations", type=int, default=5)
    parser.add_argument("--mvee-center-tol", type=float, default=1e-3)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--drake-defaults",
        action="store_true",
        help="use Drake's stricter statistical settings",
    )
    parser.add_argument("--validate-samples", type=int, default=500)
    parser.add_argument(
        "--validation-mixing-steps",
        type=int,
        default=100,
        help="mixing steps for the final per-region validation",
    )
    parser.add_argument(
        "--grow-validation-samples",
        type=int,
        default=300,
        help="samples used to gate each region as it is grown; 0 disables the gate",
    )
    parser.add_argument("--grow-validation-mixing", type=int, default=100)
    parser.add_argument(
        "--grow-validation-threshold",
        type=float,
        default=0.05,
        help="regrow a region when its measured colliding fraction exceeds this",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="stricter regrowth attempts for a region that fails the gate",
    )
    parser.add_argument("--coverage-samples", type=int, default=10000)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--self-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-z",
        type=float,
        default=None,
        help="initial z slice; defaults to the midpoint of the position bounds",
    )
    parser.add_argument(
        "--plot-acc",
        default="0,0,0",
        help="initial acceleration slice ax,ay,az; the sliders change it",
    )
    parser.add_argument(
        "--plot-max-regions",
        type=int,
        default=0,
        help="maximum number of regions drawn; 0 = all (a positive value speeds up the sliders)",
    )
    parser.add_argument("--save-plot", default="")
    parser.add_argument("--save", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    params = dictToClass(
        MPC, yamlToDict(os.path.join(os.environ["POLYFLY_DIR"], "data/params", args.yaml))
    )
    checker = FlatEdgeChecker(params)
    checks = Checks()

    if args.self_test:
        print("Self-tests")
        self_tests(checks, params, checker)
        print("")
        if args.self_test_only:
            return 0 if not checks.failed() else 1

    domain, lower, upper = payload_domain_from_params(params)
    checks.check(
        "domain has positive width",
        np.all(upper - lower > 0.0),
        "widths %s" % np.array2string(upper - lower, precision=3),
    )

    z_levels = [v for v in args.z_levels.split(",") if v.strip()] if args.z_levels else []
    seeds, axes = build_grid(lower, upper, args.pos_step, args.acc_step, z_levels)
    print(
        "grid: %d seeds (%d x %d x %d positions, %d x %d x %d accelerations) from %s"
        % (
            len(seeds),
            len(axes[0]),
            len(axes[1]),
            len(axes[2]),
            len(axes[3]),
            len(axes[4]),
            len(axes[5]),
            args.yaml,
        )
    )
    free = collision_free_chunked(checker, seeds)
    free_seeds = seeds[free]
    print("collision-free seeds: %d of %d" % (len(free_seeds), len(seeds)))
    if not len(free_seeds):
        print("FAIL: no collision-free seeds")
        return 1

    if args.drake_defaults:
        args.num_particles = 1000
        args.tau = 0.5
        args.delta = 0.05
        args.epsilon = 0.01
        args.max_iterations = 3
        args.max_inner = 20
        args.max_planes = 10
        args.mixing_steps = 50
        args.relax_margin = False

    options = IrisZoOptions(
        num_particles=args.num_particles,
        tau=args.tau,
        delta=args.delta,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
        max_iterations_separating_planes=args.max_inner,
        max_separating_planes_per_iteration=args.max_planes,
        verbose=args.verbose,
        configuration_space_margin=args.configuration_space_margin,
        relax_margin=args.relax_margin,
        random_seed=args.seed,
        mixing_steps=args.mixing_steps,
        bisection_steps=args.bisection_steps,
        termination_threshold=args.termination_threshold,
        relative_termination_threshold=args.relative_termination_threshold,
        max_test_samples=args.max_test_samples,
        mvee_iterations=args.mvee_iterations,
        mvee_center_tol=args.mvee_center_tol,
    )
    try:
        options.validate()
    except ValueError as exc:
        print("FAIL: invalid IRIS options: %s" % exc)
        return 1
    print(
        "IRIS preset: epsilon=%.3f delta=%.3f tau=%.2f particles=%d inner=%d planes=%d "
        "mixing=%d relax_margin=%s"
        % (
            options.epsilon,
            options.delta,
            options.tau,
            options.num_particles,
            options.max_iterations_separating_planes,
            options.max_separating_planes_per_iteration,
            options.mixing_steps,
            options.relax_margin,
        )
    )

    regions = []
    failures = []
    gate_fractions = []
    retried_regions = 0
    retry_attempts = 0
    # Running the sampler with too few mixing steps can let a thin, badly grown region pass
    # the statistical test; the gate below measures every region independently and regrows
    # the rare failures with a stricter, better-mixed solve.
    retry_options = {
        "mixing_steps": max(args.mixing_steps * 2, 50),
        "epsilon": min(args.epsilon, 0.025),
        "max_iterations": max(args.max_iterations, 3),
        "max_iterations_separating_planes": max(args.max_inner, 8),
        "max_separating_planes_per_iteration": max(args.max_planes, 8),
    }
    started = time.time()
    limit = len(free_seeds) if args.max_seeds <= 0 else min(args.max_seeds, len(free_seeds))
    for index in range(limit):
        if args.time_budget > 0.0 and time.time() - started > args.time_budget:
            print("time budget reached after %d seeds" % index)
            break
        seed = free_seeds[index]
        best_region = None
        best_fraction = float("inf")
        first_error = None
        retried = False
        for attempt in range(1 + max(int(args.max_retries), 0)):
            if attempt == 0:
                run_options = replace(options, random_seed=args.seed + index)
            else:
                retried = True
                retry_attempts += 1
                run_options = replace(
                    options,
                    random_seed=args.seed + index + 100000 * attempt,
                    **retry_options,
                )
            try:
                region = iris_zo(
                    checker.configs_collision_free,
                    Ellipsoid.from_center_radius(seed, args.seed_radius),
                    domain,
                    run_options,
                )
            except (IrisZoError, EmptyPolytopeError, ValueError) as exc:
                if attempt == 0:
                    first_error = str(exc)
                if args.verbose:
                    print(
                        "FAIL seed %d %s attempt %d: %s"
                        % (index, np.array2string(seed, precision=3), attempt, exc)
                    )
                continue
            if not bool(region.contains(seed[None, :])[0]):
                if attempt == 0:
                    first_error = "returned region does not contain its seed"
                continue

            fraction = collision_fraction(
                checker,
                region,
                np.random.default_rng(args.seed + 50000 + 1000 * index + attempt),
                args.grow_validation_samples,
                args.grow_validation_mixing,
            )
            if not np.isfinite(fraction):
                best_region, best_fraction = region, float("nan")
                break
            if fraction < best_fraction:
                best_fraction = fraction
                best_region = region
            if fraction <= args.grow_validation_threshold:
                break

        if best_region is None:
            failures.append((index, seed, first_error or "all attempts failed"))
            continue
        if retried:
            retried_regions += 1
        regions.append(best_region)
        gate_fractions.append(best_fraction)
        done = len(regions)
        if done % 25 == 0 or done == limit:
            elapsed = time.time() - started
            rate = elapsed / max(index + 1, 1)
            eta = rate * max(limit - index - 1, 0)
            print(
                "region %d/%d: seed %s, %d faces, ellipsoid volume %.3g, gate %.3f%s, "
                "%.1fs elapsed, %.0fs remaining"
                % (
                    done,
                    limit,
                    np.array2string(seed, precision=2),
                    len(best_region.polytope.b),
                    best_region.volume,
                    best_fraction,
                    " (retry)" if retried else "",
                    elapsed,
                    eta,
                )
            )
    elapsed = time.time() - started
    print(
        "\ngrown %d regions from %d free seeds in %.1fs (%.2fs/region); %d failures; "
        "%d regions retried (%d extra attempts)"
        % (
            len(regions),
            len(free_seeds),
            elapsed,
            elapsed / max(len(regions), 1),
            len(failures),
            retried_regions,
            retry_attempts,
        )
    )
    finite_gate = np.asarray(gate_fractions, dtype=float)
    finite_gate = finite_gate[np.isfinite(finite_gate)]
    if len(finite_gate):
        print(
            "growth gate: mean %.4f, worst %.4f (threshold %.3f)"
            % (finite_gate.mean(), finite_gate.max(), args.grow_validation_threshold)
        )
    for index, seed, reason in failures[:10]:
        print("  failure: seed %d %s -> %s" % (index, np.array2string(seed, precision=3), reason))

    # Independent Monte Carlo validation of every region.
    print("\nRegion validation (%d samples each)" % args.validate_samples)
    fractions = []
    rng = np.random.default_rng(args.seed + 100000)
    if args.validate_samples > 0:
        for region in regions:
            samples = region.polytope.uniform_sample(
                rng, args.validate_samples, mixing_steps=args.validation_mixing_steps
            )
            fraction = float(1.0 - checker.configs_collision_free(samples).mean())
            fractions.append(fraction)
    fractions = np.asarray(fractions)
    worst = float(fractions.max()) if len(fractions) else float("nan")
    if len(fractions):
        weights = np.asarray([r.volume for r in regions])
        weighted = float(np.average(fractions, weights=weights))
        worst_index = int(np.argmax(fractions))
        print(
            "  colliding fraction: mean %.4f, volume-weighted %.4f, worst %.4f, "
            "regions with >1%%: %d"
            % (fractions.mean(), weighted, worst, int((fractions > 0.01).sum()))
        )
        print(
            "  worst region: #%d seed %s (%.1f%% colliding)"
            % (
                worst_index,
                np.array2string(regions[worst_index].seed, precision=3),
                100.0 * worst,
            )
        )
    else:
        print("  no regions")

    # Coverage of the domain and of the collision-free domain.
    rng = np.random.default_rng(args.seed + 200000)
    samples = lower + (upper - lower) * rng.random((args.coverage_samples, 6))
    sample_free = collision_free_chunked(checker, samples)
    covered = np.zeros(len(samples), dtype=bool)
    for region in regions:
        covered |= region.contains(samples)
    coverage_domain = float(covered.mean())
    coverage_free = float(covered[sample_free].mean()) if sample_free.any() else 0.0
    print(
        "coverage: %.1f%% of the domain, %.1f%% of collision-free configurations "
        "(%d samples, %.1f%% free)"
        % (
            100.0 * coverage_domain,
            100.0 * coverage_free,
            len(samples),
            100.0 * sample_free.mean(),
        )
    )

    if args.save and not regions:
        print("no regions to save")
    if args.save and regions:
        offsets = np.cumsum([0] + [len(r.polytope.b) for r in regions])
        np.savez_compressed(
            args.save,
            A=np.vstack([r.polytope.A for r in regions]),
            b=np.concatenate([r.polytope.b for r in regions]),
            offsets=offsets,
            seeds=np.asarray([r.seed for r in regions]),
            centers=np.asarray([r.center for r in regions]),
            volumes=np.asarray([r.volume for r in regions]),
            iterations=np.asarray([r.iterations for r in regions]),
            terminations=np.asarray([r.termination for r in regions]),
            yaml=np.array(args.yaml),
            options=json.dumps(asdict(options)),
        )
        print("saved %d regions to %s" % (len(regions), args.save))

    if args.plot or args.save_plot:
        acceleration = np.array([float(v) for v in args.plot_acc.split(",")])
        if acceleration.size != 3:
            print("FAIL: --plot-acc must contain three comma-separated values")
            return 1
        z0 = args.plot_z if args.plot_z is not None else 0.5 * (lower[2] + upper[2])
        plot_cover(
            params,
            regions,
            free_seeds,
            (lower[:3], upper[:3]),
            (lower[3:], upper[3:]),
            (z0, acceleration[0], acceleration[1], acceleration[2]),
            args.plot_max_regions,
            args.save_plot,
            args.plot,
        )

    checks.check("at least one region", len(regions) > 0)
    checks.check(
        "every region contains its seed",
        all(bool(r.contains(r.seed[None, :])[0]) for r in regions),
    )
    checks.check(
        "region failure rate",
        len(failures) <= max(1, int(0.02 * limit)),
        "%d failures / %d seeds" % (len(failures), limit),
    )
    collision_bound = max(2.0 * options.epsilon, 0.05)
    checks.check(
        "worst region collision fraction",
        len(fractions) > 0 and worst <= collision_bound,
        "worst %.4f, bound %.3f" % (worst, collision_bound),
    )
    if args.min_coverage > 0.0:
        checks.check(
            "free-space coverage",
            coverage_free >= args.min_coverage,
            "%.1f%% >= %.1f%%"
            % (100.0 * coverage_free, 100.0 * args.min_coverage),
        )

    print("")
    return 0 if not checks.failed() else 1


if __name__ == "__main__":
    sys.exit(main())
