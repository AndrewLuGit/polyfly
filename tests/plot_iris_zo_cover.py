"""Plot a saved IRIS-ZO convex cover without rerunning the planner.

    POLYFLY_DIR=$(pwd) PYTHONPATH=src python3 tests/plot_iris_zo_cover.py cover.npz

The input is the ``.npz`` written by ``tests/iris_zo_cover.py --save``.  The plot is an
x-y cross-section; four sliders sweep the z position and the x, y and z acceleration
components.  Obstacles are solid rectangles (dashed: their ``margin`` inflation), regions
are translucent polygons, and the dots are the saved seeds projected onto x-y.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from iris_zo_cover import plot_cover  # noqa: E402
from poly_fly.optimal_planner.iris_zo import (  # noqa: E402
    HPolyhedron,
    IrisZoRegion,
    payload_domain_from_params,
)
from poly_fly.utils.utils import MPC, dictToClass, yamlToDict  # noqa: E402


def load_cover(path):
    """Rebuild ``IrisZoRegion`` objects from a saved cover."""
    data = np.load(path)
    for key in ("A", "b", "offsets", "seeds"):
        if key not in data.files:
            raise SystemExit("FAIL: %s is missing the '%s' array" % (path, key))

    offsets = np.asarray(data["offsets"], dtype=int)
    count = len(offsets) - 1
    if count < 1 or len(data["seeds"]) != count:
        raise SystemExit("FAIL: inconsistent region count in %s" % path)
    if offsets[0] != 0 or offsets[-1] != len(data["b"]):
        raise SystemExit("FAIL: offsets do not span the saved A/b arrays in %s" % path)

    seeds = np.asarray(data["seeds"], dtype=float)
    centers = (
        np.asarray(data["centers"], dtype=float) if "centers" in data.files else seeds
    )
    volumes = (
        np.asarray(data["volumes"], dtype=float)
        if "volumes" in data.files
        else np.full(count, np.nan)
    )
    iterations = (
        np.asarray(data["iterations"], dtype=int)
        if "iterations" in data.files
        else np.zeros(count, dtype=int)
    )
    terminations = (
        np.asarray(data["terminations"])
        if "terminations" in data.files
        else np.array(["unknown"] * count)
    )

    regions = []
    for index in range(count):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if stop <= start:
            raise SystemExit("FAIL: empty region %d in %s" % (index, path))
        polytope = HPolyhedron(data["A"][start:stop], data["b"][start:stop])
        regions.append(
            IrisZoRegion(
                polytope=polytope,
                seed=seeds[index],
                center=centers[index],
                volume=float(volumes[index]),
                iterations=int(iterations[index]),
                termination=str(terminations[index]),
            )
        )
    return data, regions


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("npz", help="cover .npz written by tests/iris_zo_cover.py --save")
    parser.add_argument(
        "--yaml", default="", help="override the yaml path stored in the npz"
    )
    parser.add_argument(
        "--plot-z",
        type=float,
        default=None,
        help="initial z slice; defaults to the midpoint of the position bounds",
    )
    parser.add_argument(
        "--plot-acc", default="0,0,0", help="initial acceleration slice ax,ay,az"
    )
    parser.add_argument(
        "--plot-max-regions",
        type=int,
        default=0,
        help="maximum number of regions drawn; 0 = all",
    )
    parser.add_argument(
        "--save-plot", default="", help="write the initial slice to this file"
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="open the interactive window (default: on unless --save-plot)",
    )
    args = parser.parse_args()

    data, regions = load_cover(args.npz)

    yaml_path = args.yaml
    if not yaml_path and "yaml" in data.files:
        yaml_path = str(data["yaml"])
    if not yaml_path:
        raise SystemExit("FAIL: the npz does not record its yaml; pass --yaml")
    try:
        base_dir = os.environ["POLYFLY_DIR"]
    except KeyError:
        raise SystemExit("FAIL: set POLYFLY_DIR to the repository root")

    params = dictToClass(MPC, yamlToDict(os.path.join(base_dir, "data/params", yaml_path)))
    _, lower, upper = payload_domain_from_params(params)

    acceleration = np.array([float(v) for v in args.plot_acc.split(",")], dtype=float)
    if acceleration.size != 3:
        raise SystemExit("FAIL: --plot-acc must contain three comma-separated values")
    z0 = args.plot_z if args.plot_z is not None else 0.5 * (lower[2] + upper[2])
    show = (not bool(args.save_plot)) if args.show is None else bool(args.show)

    faces = np.array([len(region.polytope.b) for region in regions])
    terminations = Counter(region.termination for region in regions)
    print("loaded %d regions from %s" % (len(regions), args.npz))
    print("  yaml: %s" % yaml_path)
    print("  faces: min %d, mean %.1f, max %d" % (faces.min(), faces.mean(), faces.max()))
    print("  terminations: %s" % dict(terminations))
    if "options" in data.files:
        try:
            options = json.loads(str(data["options"]))
        except (TypeError, ValueError):
            options = None
        if options:
            print(
                "  options: epsilon=%s delta=%s tau=%s particles=%s mixing=%s "
                "outer=%s inner=%s planes=%s"
                % (
                    options.get("epsilon"),
                    options.get("delta"),
                    options.get("tau"),
                    options.get("num_particles"),
                    options.get("mixing_steps"),
                    options.get("max_iterations"),
                    options.get("max_iterations_separating_planes"),
                    options.get("max_separating_planes_per_iteration"),
                )
            )
    print(
        "  drawing %s regions"
        % ("all" if args.plot_max_regions <= 0 else args.plot_max_regions)
    )

    plot_cover(
        params,
        regions,
        np.asarray(data["seeds"], dtype=float),
        (lower[:3], upper[:3]),
        (lower[3:], upper[3:]),
        (z0, acceleration[0], acceleration[1], acceleration[2]),
        args.plot_max_regions,
        args.save_plot,
        show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
