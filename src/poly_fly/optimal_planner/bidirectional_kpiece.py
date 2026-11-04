"""Bidirectional KPIECE for the cable-suspended payload.

KPIECE discretizes the state space into cells and uses the discretization to decide where to
expand: cells are scored by how often they have been selected, how much motion has passed
through them, and whether they are on the boundary of the explored grid. The search therefore
spends its samples on under-covered / frontier regions instead of drawing uniformly.

The planner is bidirectional. The start tree stores forward motions as usual. The goal tree is
rooted at ``R(goal) = [p, -v, a]`` and stores *time-reversed* motions, so reversing a stored
goal edge yields a forward-in-time motion toward the goal; no goal-side BVP is re-solved on
extraction. ``reverse_motion`` is the exact polynomial time reversal, and every tree edge has a
physical forward motion (the stored motion for the start tree, its reversal for the goal tree)
that is what collision checking, cell projection and the statistics use.

Three junction strategies are available when a piece of one tree lands in a cell occupied by
the other tree:

* ``steer``    -- one flat BVP between the nearest pair of endpoints (cheapest);
* ``shortcut`` -- search several cross pairs and run the greedy flat shortcut on the best
                  continuous candidate (middle);
* ``ocp``      -- concatenate the two chains with a jump and use that discontinuous trajectory
                  to seed the multiple-shooting OCP, then densely validate the result
                  (most expensive; the OCP's own collision constraints are permissive).

Two expansion strategies are implemented and are an experiment axis:

* ``geometric`` -- sample a local target around the selected state and steer to it;
* ``control``   -- sample a random jerk input and propagate it (OMPL's control KPIECE).

The KPIECE paper's cell-size calibration is implemented in ``_calibrate``: per-dimension
statistics (parts per motion, cells crossed per simulation step, samples per cell, interior
cells, motion length) decide whether each cell size should grow or shrink, and the planner is
restarted for a few rounds.

Standalone use::

    python -m poly_fly.optimal_planner.bidirectional_kpiece --yaml experiments/maze_1.yaml
"""

from __future__ import annotations

import argparse
import copy
import heapq
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from poly_fly.optimal_planner.flat_collision import ACC_NORM_EPS, ORDER, OUTPUT_DIM, FlatEdgeChecker
from poly_fly.optimal_planner.kinodynamic_rrt_connect import (
    N_INPUTS,
    N_STATES,
    FlatMotion,
    FlatPath,
    FlatSteering,
    _WeightedEuclidean,
    _goal_snap,
    _mobility_for,
    _shortcut,
    is_motion_within_bounds,
    make_steering,
    path_to_sol_values,
)

# Time reversal maps [p, v, a] to [p, -v, a]; the jerk input flips sign as well, but the input
# bounds are symmetric on every shipped configuration (asserted in the planner constructor).
_REVERSE_SIGNS = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

# Above this score the lazy heaps are rebuilt instead of accumulating stale entries.
_HEAP_REBUILD_RATIO = 4

# Dense OCP validation step and the OCP endpoint tolerance used by Planner's xdes_tol.
_OCP_DENSE_DT = 0.01
_OCP_GOAL_TOL = 0.05


def reverse_state(state: np.ndarray) -> np.ndarray:
    """``R([p, v, a]) = [p, -v, a]`` (a copy)."""
    return np.asarray(state, dtype=float) * _REVERSE_SIGNS


def reverse_motion(motion: FlatMotion) -> FlatMotion:
    """Exact time reversal: ``q(s) = p(T - s)``, coefficients ``c'_i`` as in the docstring.

    ``FlatMotion`` evaluates the position polynomial and its derivatives directly, so the
    returned object is simultaneously the physical motion traversed backwards and the
    reverse-time motion ``R(p(T - s))``. Reversing twice reproduces the original.
    """
    coefficients = np.asarray(motion.coefficients, dtype=float)
    rows, _ = coefficients.shape
    duration = float(motion.duration)
    reversed_coefficients = np.zeros_like(coefficients)
    for i in range(rows):
        sign = 1.0 if i % 2 == 0 else -1.0
        for k in range(i, rows):
            reversed_coefficients[i] += (
                sign * math.comb(k, i) * duration ** (k - i) * coefficients[k]
            )
    return FlatMotion(reversed_coefficients, duration)


def _assert_symmetric_limits(params) -> None:
    """Reversal flips velocity and jerk; the shipped bounds are symmetric, so check it."""
    low = np.asarray(params.state_min, dtype=float)
    high = np.asarray(params.state_max, dtype=float)
    for level in (1, 2):
        if not np.allclose(low[3 * level : 3 * level + 3], -high[3 * level : 3 * level + 3], atol=1e-9):
            raise ValueError(
                "bidirectional KPIECE requires symmetric velocity/acceleration bounds; "
                "states %d:%d are [%s] .. [%s]"
                % (3 * level, 3 * level + 3, low[3 * level : 3 * level + 3], high[3 * level : 3 * level + 3])
            )
    if not np.allclose(np.asarray(params.input_min), -np.asarray(params.input_max), atol=1e-9):
        raise ValueError("bidirectional KPIECE requires symmetric input bounds")


# --------------------------------------------------------------------------------------
# options and projection
# --------------------------------------------------------------------------------------
def _tuple_or_none(value) -> Optional[Tuple]:
    if value is None:
        return None
    return tuple(value)


@dataclass(frozen=True)
class KPieceOptions:
    projection: str = "position_velocity"
    projection_dims: Optional[Tuple[int, ...]] = None
    cells_per_dim: object = 8  # int or one count per projected dimension
    cell_sizes: Optional[Tuple[float, ...]] = None
    cell_envelope: bool = False
    expansion: str = "global"
    local_radius: Optional[float] = None
    local_radius_cells: float = 2.0
    local_dims: str = "projection"
    control_fraction: float = 1.0
    cell_score_distance: bool = False
    tree_selection: str = "alternate"
    motion_selection: str = "half_normal"  # half_normal (newest-biased) | uniform
    halfnormal_focus: float = 3.0
    junction_selection: str = "uniform"  # uniform (other tree) | closest
    border_fraction: float = 0.8
    good_score_factor: float = 0.9
    bad_score_factor: float = 0.45
    interesting_probability: float = 0.05
    min_steps: int = 1
    max_steps: int = 20
    connect_tries: int = 2
    connection: str = "steer"
    shortcut_candidates: int = 5
    ocp_max_attempts: int = 2
    ocp_max_jump: float = 1.0
    junction_attempts: int = 3
    adaptive: bool = False
    calibration_iters: int = 2000
    calibration_rounds: int = 3
    cell_factor: float = 2.0
    interior_rule_max_dim: int = 6

    @classmethod
    def from_params(cls, params) -> "KPieceOptions":
        def get(name, default):
            value = getattr(params, name, None)
            return default if value is None else value

        cells_per_dim = get("kpiece_cells_per_dim", 8)
        if isinstance(cells_per_dim, (list, tuple)):
            cells_per_dim = tuple(int(v) for v in cells_per_dim)
        else:
            cells_per_dim = int(cells_per_dim)

        return cls(
            projection=str(get("kpiece_projection", "position_velocity")),
            projection_dims=_tuple_or_none(get("kpiece_projection_dims", None)),
            cells_per_dim=cells_per_dim,
            cell_sizes=_tuple_or_none(get("kpiece_cell_sizes", None)),
            cell_envelope=bool(get("kpiece_cell_envelope", True)),
            expansion=str(get("kpiece_expansion", "geometric")).lower(),
            local_radius=get("kpiece_local_radius", None),
            local_radius_cells=float(get("kpiece_local_radius_cells", 2.0)),
            local_dims=str(get("kpiece_local_dims", "projection")).lower(),
            control_fraction=float(get("kpiece_control_fraction", 1.0)),
            cell_score_distance=bool(get("kpiece_cell_score_distance", False)),
            tree_selection=str(get("kpiece_tree_selection", "alternate")).lower(),
            motion_selection=str(get("kpiece_motion_selection", "half_normal")).lower(),
            halfnormal_focus=float(get("kpiece_halfnormal_focus", 3.0)),
            junction_selection=str(get("kpiece_junction_selection", "uniform")).lower(),
            border_fraction=float(get("kpiece_border_fraction", 0.8)),
            good_score_factor=float(get("kpiece_good_score_factor", 0.9)),
            bad_score_factor=float(get("kpiece_bad_score_factor", 0.45)),
            interesting_probability=float(get("kpiece_interesting_probability", 0.05)),
            min_steps=int(get("kpiece_min_steps", 1)),
            max_steps=int(get("kpiece_max_steps", 20)),
            connect_tries=int(get("kpiece_connect_tries", 2)),
            connection=str(get("kpiece_connection", "steer")).lower(),
            shortcut_candidates=int(get("kpiece_shortcut_candidates", 5)),
            ocp_max_attempts=int(get("kpiece_ocp_max_attempts", 2)),
            ocp_max_jump=float(get("kpiece_ocp_max_jump", 1.0)),
            junction_attempts=int(get("kpiece_junction_attempts", 3)),
            adaptive=bool(get("kpiece_adaptive", False)),
            calibration_iters=int(get("kpiece_calibration_iters", 2000)),
            calibration_rounds=int(get("kpiece_calibration_rounds", 3)),
            cell_factor=float(get("kpiece_cell_factor", 2.0)),
            interior_rule_max_dim=int(get("kpiece_interior_rule_max_dim", 6)),
        )


_PRESETS = {
    "position": ((0, 1, 2), (0, 0, 0)),
    "position_velocity": ((0, 1, 2, 3, 4, 5), (0, 0, 0, 1, 1, 1)),
    "full": ((0, 1, 2, 3, 4, 5, 6, 7, 8), (0, 0, 0, 1, 1, 1, 2, 2, 2)),
    "position_acceleration": ((0, 1, 2, 6, 7, 8), (0, 0, 0, 2, 2, 2)),
}
_METRIC_PRESETS = {"metric_full"}


@dataclass(frozen=True)
class CellProjection:
    """A linear projection of the flat state plus the cell size along each dimension."""

    name: str
    dims: Tuple[int, ...]
    levels: Tuple[int, ...]
    scales: Tuple[float, ...]
    sizes: Tuple[float, ...]
    offsets: Tuple[float, ...]

    @classmethod
    def from_params(cls, params, options: KPieceOptions) -> "CellProjection":
        if options.projection_dims is not None:
            dims = tuple(int(d) for d in options.projection_dims)
            levels = tuple(0 if d < 3 else 1 if d < 6 else 2 for d in dims)
        else:
            name = options.projection
            if name not in _PRESETS and name != "metric_full":
                raise ValueError(
                    "unknown kpiece_projection %r; expected one of %s or metric_full"
                    % (name, sorted(_PRESETS))
                )
            dims, levels = _PRESETS.get(name, _PRESETS["full"])
        name = options.projection if options.projection_dims is None else "custom"

        omega = float(getattr(params, "rrt_metric_omega", 2.0))
        metric_scaled = options.projection_dims is None and name in _METRIC_PRESETS
        level_weights = (1.0, 1.0 / omega, 1.0 / omega**2)
        scales = tuple(level_weights[level] if metric_scaled else 1.0 for level in levels)

        low = np.asarray(params.state_min, dtype=float)
        high = np.asarray(params.state_max, dtype=float)
        offsets = tuple(float(low[d] * scales[i]) for i, d in enumerate(dims))

        if options.cell_sizes is not None:
            if len(options.cell_sizes) != len(dims):
                raise ValueError(
                    "kpiece_cell_sizes must have one entry per projected dimension (%d), got %d"
                    % (len(dims), len(options.cell_sizes))
                )
            sizes = tuple(max(float(v), 1e-9) for v in options.cell_sizes)
        else:
            if isinstance(options.cells_per_dim, (list, tuple)):
                if len(options.cells_per_dim) != len(dims):
                    raise ValueError(
                        "kpiece_cells_per_dim must have one entry per projected dimension "
                        "(%d), got %d" % (len(dims), len(options.cells_per_dim))
                    )
                counts = [max(int(v), 1) for v in options.cells_per_dim]
            else:
                counts = [max(int(options.cells_per_dim), 1)] * len(dims)

            sizes_list = []
            for i, d in enumerate(dims):
                level = levels[i]
                scale = scales[i]
                if level > 0 and options.cell_envelope:
                    fraction = float(
                        getattr(params, "rrt_sample_vel_fraction", 0.02)
                        if level == 1
                        else getattr(params, "rrt_sample_acc_fraction", 0.005)
                    )
                    half = abs(float(high[d])) * fraction
                    span = min(2.0 * half, float(high[d] - low[d]))
                else:
                    span = float(high[d] - low[d])
                span = max(span, 1e-9)
                sizes_list.append(max(span * scale / counts[i], 1e-9))
            sizes = tuple(sizes_list)

        return cls(
            name=name,
            dims=dims,
            levels=levels,
            scales=scales,
            sizes=tuple(sizes),
            offsets=offsets,
        )

    def keys(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        coords = states[:, list(self.dims)] * np.asarray(self.scales)
        return np.floor(
            (coords - np.asarray(self.offsets)) / np.asarray(self.sizes)
        ).astype(np.int64)

    def key(self, state: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(v) for v in self.keys(state)[0])

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "dims": list(self.dims),
            "levels": list(self.levels),
            "scales": list(self.scales),
            "sizes": list(self.sizes),
        }


# --------------------------------------------------------------------------------------
# cell grid
# --------------------------------------------------------------------------------------
@dataclass
class _CellData:
    motions: List[int] = field(default_factory=list)
    coverage: float = 0.0
    selections: int = 1
    score: float = 1.0
    importance: float = 0.0
    version: int = 0
    neighbors: int = 0
    border: bool = True


def _half_normal_int(rng: np.random.Generator, r_min: int, r_max: int, focus: float = 3.0) -> int:
    """OMPL's ``RNG::halfNormalInt``: an integer in [r_min, r_max] biased toward r_max.

    OMPL uses this to select a motion from a KPIECE cell, and because motions are appended
    newest-last, the bias toward ``r_max`` favors recently added motions. The Gaussian is
    centred at ``r_max + 1 - r_min`` with standard deviation ``mean / focus``; the upper half
    is reflected below the mean and the result is floored and clamped.
    """
    if r_max <= r_min:
        return int(r_min)
    high = float(r_max + 1)
    mean = high - float(r_min)
    value = float(rng.normal(mean, mean / max(float(focus), 1e-9)))
    if value > mean:
        value = 2.0 * mean - value
    value = value + r_min if value >= 0.0 else float(r_min)
    return int(math.floor(min(value, high)))


class _CellGrid:
    """One tree's KPIECE grid, with OMPL's border/importance selection rule."""

    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self.cells: Dict[Tuple[int, ...], _CellData] = {}
        self.n_external = 0
        self.n_internal = 0
        self.n_pieces = 0
        self._external_heap: List[Tuple[float, int, Tuple[int, ...]]] = []
        self._internal_heap: List[Tuple[float, int, Tuple[int, ...]]] = []

    def __len__(self) -> int:
        return len(self.cells)

    def get(self, key: Tuple[int, ...]) -> Optional[_CellData]:
        return self.cells.get(key)

    def _neighbor_keys(self, key: Tuple[int, ...]):
        for axis in range(self.dimension):
            for delta in (-1, 1):
                neighbor = list(key)
                neighbor[axis] += delta
                yield tuple(neighbor)

    def neighbors(self, key: Tuple[int, ...]) -> int:
        return sum(1 for neighbor in self._neighbor_keys(key) if neighbor in self.cells)

    def _push(self, key: Tuple[int, ...]) -> None:
        cell = self.cells[key]
        cell.version += 1
        heap = self._internal_heap if not cell.border else self._external_heap
        heapq.heappush(heap, (-float(cell.importance), cell.version, key))
        if len(heap) > _HEAP_REBUILD_RATIO * max(1, len(self.cells)):
            self._rebuild_heaps()

    def _rebuild_heaps(self) -> None:
        self._external_heap = []
        self._internal_heap = []
        for key, cell in self.cells.items():
            cell.importance = self._importance(cell)
            cell.version += 1
            heap = self._internal_heap if not cell.border else self._external_heap
            heapq.heappush(heap, (-float(cell.importance), cell.version, key))

    def _importance(self, cell: _CellData) -> float:
        coverage = max(float(cell.coverage), 1e-9)
        selections = max(int(cell.selections), 1)
        return float(cell.score) / ((cell.neighbors + 1.0) * coverage * selections)

    def update(self, key: Tuple[int, ...]) -> None:
        cell = self.cells[key]
        cell.importance = self._importance(cell)
        self._push(key)

    def _pop(self, heap) -> Optional[Tuple[int, ...]]:
        while heap:
            _, version, key = heapq.heappop(heap)
            cell = self.cells.get(key)
            if cell is not None and cell.version == version:
                return key
        return None

    def frac_external(self) -> float:
        total = self.n_external + self.n_internal
        return float(self.n_external) / total if total else 0.0

    def select(
        self,
        border_fraction: float,
        rng: np.random.Generator,
        motion_selection: str = "half_normal",
        halfnormal_focus: float = 3.0,
    ):
        use_external = rng.random() < max(float(border_fraction), self.frac_external())
        key = self._pop(self._external_heap if use_external else self._internal_heap)
        if key is None:
            key = self._pop(self._internal_heap if use_external else self._external_heap)
        if key is None:
            # All heap entries are stale; rebuild once and retry.
            self._rebuild_heaps()
            key = self._pop(self._external_heap if use_external else self._internal_heap)
            if key is None:
                key = self._pop(self._internal_heap if use_external else self._external_heap)
        if key is None:
            return None
        cell = self.cells[key]
        cell.selections += 1
        self.update(key)
        if str(motion_selection).lower() == "half_normal":
            motion_index = _half_normal_int(
                rng, 0, len(cell.motions) - 1, float(halfnormal_focus)
            )
        else:
            motion_index = int(rng.integers(0, len(cell.motions)))
        node = int(cell.motions[motion_index])
        return key, node

    def add_piece(
        self, key: Tuple[int, ...], node: int, steps: float, score: float
    ) -> bool:
        """Insert a motion endpoint; returns True when the cell is new."""
        steps = max(float(steps), 1.0)  # OMPL counts at least one simulation step
        cell = self.cells.get(key)
        if cell is not None:
            cell.motions.append(int(node))
            cell.coverage += steps
            self.n_pieces += 1
            self.update(key)
            return False

        cell = _CellData(
            motions=[int(node)],
            coverage=steps,
            selections=1,
            score=float(score),
            neighbors=self.neighbors(key),
        )
        cell.border = cell.neighbors < 2 * self.dimension
        cell.importance = self._importance(cell)
        self.cells[key] = cell
        self.n_pieces += 1
        if cell.border:
            self.n_external += 1
        else:
            self.n_internal += 1
        self._push(key)

        for neighbor in self._neighbor_keys(key):
            other = self.cells.get(neighbor)
            if other is None:
                continue
            other.neighbors += 1
            if other.border and other.neighbors >= 2 * self.dimension:
                other.border = False
                self.n_external -= 1
                self.n_internal += 1
            self.update(neighbor)
        return True

    def decay(self, key: Tuple[int, ...], factor: float) -> None:
        cell = self.cells.get(key)
        if cell is None:
            return
        cell.score *= float(factor)
        if cell.score < 1e-12:
            self.rescale()
        else:
            self.update(key)

    def rescale(self) -> None:
        for cell in self.cells.values():
            cell.score += 1.0
            cell.importance = self._importance(cell)
        self._rebuild_heaps()

    def stats(self) -> dict:
        n_cells = len(self.cells)
        return {
            "cells": n_cells,
            "pieces": self.n_pieces,
            "interior_cells": self.n_internal,
            "external_cells": self.n_external,
            "frac_external": self.frac_external(),
            "samples_per_cell": float(self.n_pieces) / n_cells if n_cells else 0.0,
        }


# --------------------------------------------------------------------------------------
# tree
# --------------------------------------------------------------------------------------
class _KPieceTree:
    def __init__(
        self,
        root_state: np.ndarray,
        projection: CellProjection,
        reverse_time: bool,
        root_score: float = 1.0,
    ):
        self.states = np.asarray(root_state, dtype=float).reshape(1, N_STATES)
        self.parents: List[int] = [-1]
        self.motions: List[Optional[FlatMotion]] = [None]
        self.projection = projection
        self.reverse_time = bool(reverse_time)
        self.grid = _CellGrid(len(projection.dims))
        self.n_pieces = 0
        self.iteration = 1
        key = projection.key(self.physical_state(0))
        self.grid.add_piece(key, 0, 1.0, root_score)

    def __len__(self) -> int:
        return self.states.shape[0]

    def physical_state(self, index: int) -> np.ndarray:
        state = self.states[index]
        return reverse_state(state) if self.reverse_time else np.asarray(state, dtype=float)

    def physical_motion(self, index: int) -> Optional[FlatMotion]:
        motion = self.motions[index]
        if motion is None:
            return None
        return reverse_motion(motion) if self.reverse_time else motion

    def add(
        self,
        state: np.ndarray,
        parent: int,
        motion: FlatMotion,
        key: Tuple[int, ...],
        steps: float,
        score: float,
    ) -> int:
        self.states = np.vstack([self.states, np.asarray(state, dtype=float).reshape(1, N_STATES)])
        self.parents.append(int(parent))
        self.motions.append(motion)
        index = len(self) - 1
        self.grid.add_piece(key, index, steps, score)
        self.n_pieces += 1
        return index

    def chain_to_root(self, index: int):
        """Nodes from ``index`` up to the root (child-first), with their physical states."""
        nodes = []
        cursor = int(index)
        while cursor != -1:
            nodes.append(cursor)
            cursor = self.parents[cursor]
        return nodes


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------
class _StatsAccumulator:
    """Per-motion KPIECE statistics, sampled once per accepted physical edge."""

    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self.motions = 0
        self.long_motions = 0
        self.parts_sum = np.zeros(self.dimension, dtype=float)
        self.fast_cross_count = np.zeros(self.dimension, dtype=float)
        self.parts_joint_sum = 0.0
        self.duration_steps_sum = 0.0

    def record(self, motion: FlatMotion, projection: CellProjection, dt: float) -> dict:
        """Sample one physical forward edge and accumulate its discretization statistics."""
        duration = float(motion.duration)
        substep = max(min(0.01, dt / 5.0), duration / 511.0)
        if substep <= 0.0:
            substep = max(duration, 1e-9)
        times = np.arange(0.0, duration + 1e-12, substep)
        if times.size == 0 or times[-1] < duration - 1e-12:
            times = np.append(times, duration)
        samples = np.hstack(
            [motion.samples_at(times, level) for level in range(ORDER)]
        ).reshape(times.shape[0], N_STATES)
        keys = projection.keys(samples)  # (K, dim)

        changes = keys[1:] != keys[:-1]
        parts_per_dim = 1 + changes.sum(axis=0)
        parts_joint = 1 + int(changes.any(axis=1).sum())

        # Cells covered within one simulation time-step: group the substep samples by
        # floor(t/dt) and count cell runs inside each group. The first sample of a group is
        # included once; the boundary duplicate is harmless for the >2 test.
        step_ids = np.floor(times / max(dt, 1e-9)).astype(np.int64)
        fast = np.zeros(self.dimension, dtype=bool)
        boundaries = np.flatnonzero(np.diff(step_ids) != 0) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [times.shape[0]]))
        for start, end in zip(starts, ends):
            if end - start < 2:
                continue
            segment = keys[start:end]
            runs = 1 + (segment[1:] != segment[:-1]).sum(axis=0)
            fast |= runs > 2

        self.motions += 1
        self.duration_steps_sum += duration / max(dt, 1e-9)
        if duration >= 3.0 * dt:
            self.long_motions += 1
        self.parts_sum += parts_per_dim
        self.fast_cross_count += fast.astype(float)
        self.parts_joint_sum += parts_joint
        return {
            "parts": parts_per_dim.astype(float),
            "parts_joint": float(parts_joint),
            "fast": fast.astype(float),
            "duration_steps": duration / max(dt, 1e-9),
        }

    def as_dict(self, trees: Sequence[Optional[_KPieceTree]]) -> dict:
        n = max(self.motions, 1)
        valid = [tree for tree in trees if tree is not None]
        cells = sum(len(tree.grid) for tree in valid)
        pieces = sum(tree.grid.n_pieces for tree in valid)
        interior = sum(tree.grid.n_internal for tree in valid)
        external = sum(tree.grid.n_external for tree in valid)
        return {
            "motions": int(self.motions),
            "cells": int(cells),
            "pieces": int(pieces),
            "interior_cells": int(interior),
            "external_cells": int(external),
            "frac_external": float(external) / max(interior + external, 1),
            "parts_avg": (self.parts_sum / n).tolist(),
            "parts_avg_joint": float(self.parts_joint_sum / n),
            "more_than_2_frac": (self.fast_cross_count / n).tolist(),
            "long_motion_frac": float(self.long_motions / n),
            "duration_steps_avg": float(self.duration_steps_sum / n),
            "samples_per_cell": float(pieces) / max(cells, 1),
        }


# --------------------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------------------
@dataclass
class _Piece:
    motion: FlatMotion
    end_state: np.ndarray
    key: Tuple[int, ...]
    steps: float
    t0: float
    t1: float


@dataclass
class _KPIECEOutcome:
    path: FlatPath
    strategy: str
    ocp_solution: Optional[dict] = None


class BidirectionalKPIECE:
    def __init__(
        self,
        params,
        options: KPieceOptions,
        mobility=None,
        rng: Optional[np.random.Generator] = None,
    ):
        _assert_symmetric_limits(params)
        self.params = params
        self.options = options
        self.mobility = _mobility_for(params, mobility)
        self.checker = FlatEdgeChecker(params, self.mobility)
        self.steering = make_steering(params)
        self.bounds_guaranteed = getattr(self.steering, "limits", None) is not None
        self.metric = _WeightedEuclidean(float(getattr(params, "rrt_metric_omega", 2.0)))
        self.projection = CellProjection.from_params(params, options)
        self.rng = rng if rng is not None else np.random.default_rng(int(params.rrt_seed))
        self.max_steps = max(1, int(options.max_steps))
        self.min_steps = max(1, min(int(options.min_steps), self.max_steps))
        self.max_step_seconds = float(getattr(params, "rrt_max_step", 2.5))

        self.start: Optional[np.ndarray] = None
        self.goal: Optional[np.ndarray] = None
        self.start_tree: Optional[_KPieceTree] = None
        self.goal_tree: Optional[_KPieceTree] = None
        self.stats = _StatsAccumulator(len(self.projection.dims))
        self.explored_nodes: List[List[float]] = []
        self.explored_nodes_3d: List[List[float]] = []

        self.iterations = 0
        self.stop_on_success = True
        self.max_iterations = int(getattr(params, "rrt_max_iterations", 20000))
        self.timeout = float(getattr(params, "rrt_timeout", 60.0))
        self.calibration_mode = False
        self.best_calibration_path: Optional[FlatPath] = None

        self.junction_stats: Dict[str, float] = {
            "attempts": 0,
            "steer_attempts": 0,
            "steer_success": 0,
            "shortcut_attempts": 0,
            "shortcut_success": 0,
            "ocp_attempts": 0,
            "ocp_success": 0,
            "ocp_solve_failed": 0,
            "ocp_invalid": 0,
            "ocp_jump_too_large": 0,
            "ocp_skipped": 0,
        }
        self._ocp_attempts = 0
        self._junction_failures: Dict[Tuple[str, Tuple[int, ...]], int] = {}
        self._deadline: Optional[float] = None

    # -- recording ---------------------------------------------------------------------
    def _record(self, state_physical: np.ndarray) -> None:
        self.explored_nodes.append([float(state_physical[0]), float(state_physical[1])])
        self.explored_nodes_3d.append(
            [float(state_physical[0]), float(state_physical[1]), float(state_physical[2])]
        )

    def _local_halfwidths(self) -> np.ndarray:
        """Physical half-widths of the local expansion ellipsoid for every state dimension."""
        projection = self.projection
        multiplier = (
            float(self.options.local_radius)
            if self.options.local_radius is not None
            else float(self.options.local_radius_cells)
        )
        half = np.zeros(N_STATES)
        for i, dimension in enumerate(projection.dims):
            scale = projection.scales[i]
            half[dimension] = multiplier * projection.sizes[i] / max(scale, 1e-12)
        if self.options.local_dims == "full":
            low = np.asarray(self.params.state_min, dtype=float)
            high = np.asarray(self.params.state_max, dtype=float)
            vel_fraction = float(getattr(self.params, "rrt_sample_vel_fraction", 0.02))
            acc_fraction = float(getattr(self.params, "rrt_sample_acc_fraction", 0.005))
            for dimension in range(N_STATES):
                if half[dimension] > 0.0:
                    continue
                if dimension < 3:
                    half[dimension] = 0.25 * float(high[dimension] - low[dimension])
                elif dimension < 6:
                    half[dimension] = abs(float(high[dimension])) * vel_fraction
                else:
                    half[dimension] = abs(float(high[dimension])) * acc_fraction
        return half

    # -- sampling ----------------------------------------------------------------------
    def _opposite_root_native(self, tree: _KPieceTree, other: _KPieceTree) -> np.ndarray:
        root = other.physical_state(0)
        return reverse_state(root) if tree.reverse_time else root

    def _sample_global(self, tree: _KPieceTree, other: _KPieceTree) -> np.ndarray:
        """RRT-style global target, expressed in the tree's native frame."""
        params = self.params
        if self.rng.random() < float(getattr(params, "rrt_goal_bias", 0.05)):
            return self._opposite_root_native(tree, other)

        low = np.asarray(params.state_min, dtype=float)
        high = np.asarray(params.state_max, dtype=float)
        span = high[0:3] - low[0:3]
        inset = min(max(float(getattr(params, "rrt_sample_inset", 0.0)), 0.0), 0.49)
        position_low = low[0:3] + inset * span
        position_high = high[0:3] - inset * span

        state = np.zeros(N_STATES)
        if self.rng.random() < float(getattr(params, "rrt_bridge_bias", 0.2)):
            fraction = self.rng.random()
            position = self.start[0:3] + fraction * (self.goal[0:3] - self.start[0:3])
            position = position + self.rng.normal(
                0.0, float(getattr(params, "rrt_bridge_sigma", 0.2)), 3
            )
            state[0:3] = np.clip(position, position_low, position_high)
        else:
            state[0:3] = self.rng.uniform(position_low, position_high)
            velocity_span = np.abs(high[3:6]) * float(
                getattr(params, "rrt_sample_vel_fraction", 0.02)
            )
            acceleration_span = np.abs(high[6:9]) * float(
                getattr(params, "rrt_sample_acc_fraction", 0.005)
            )
            state[3:6] = self.rng.uniform(-velocity_span, velocity_span)
            state[6:9] = self.rng.uniform(-acceleration_span, acceleration_span)
        return reverse_state(state) if tree.reverse_time else state

    def _sample_geometric(self, tree: _KPieceTree, other: _KPieceTree, origin: np.ndarray):
        if self.rng.random() < float(getattr(self.params, "rrt_goal_bias", 0.05)):
            return self._opposite_root_native(tree, other)

        half = self._local_halfwidths()
        active = np.flatnonzero(half > 0.0)
        if active.size == 0:
            return origin.copy()
        direction = self.rng.normal(size=active.size)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.ones(active.size)
            norm = float(np.sqrt(active.size))
        radius = self.rng.random() ** (1.0 / max(active.size, 1))
        delta = np.zeros(N_STATES)
        delta[active] = direction / norm * radius * half[active]
        target = origin + delta
        low = np.asarray(self.params.state_min, dtype=float)
        high = np.asarray(self.params.state_max, dtype=float)
        return np.clip(target, low, high)

    def _sample_steps(self) -> int:
        steps = int(self.rng.integers(self.min_steps, self.max_steps + 1))
        cap = max(1, int(math.floor(self.max_step_seconds / max(self.params.dt, 1e-9))))
        return max(1, min(steps, cap))

    # -- propagation -------------------------------------------------------------------
    def _valid(self, motion: FlatMotion, bounds_guaranteed: bool = False) -> bool:
        if not self.checker.is_motion_valid(motion):
            return False
        return bounds_guaranteed or is_motion_within_bounds(motion, self.params)

    def _propagate_bvp(
        self, tree: _KPieceTree, origin_native: np.ndarray, target_native: np.ndarray, step: float
    ):
        """Time-limited BVP extension in the tree's native frame.

        Returns ``(stored_motion, new_native_state, physical_edge)`` or ``None``. The goal
        tree's physical forward edge is ``target_side -> origin``; its stored motion is the
        time reversal. The main path uses ``P = steer(target, origin)`` whose reversal ``Q`` is
        the reverse-time motion origin_rt -> target_rt; truncating Q at ``step`` is exactly the
        backward-time prefix the goal tree should store.
        """
        origin_native = np.asarray(origin_native, dtype=float)
        target_native = np.asarray(target_native, dtype=float)
        if self.metric.distance(origin_native, target_native) < self.metric.epsilon:
            return None

        if not tree.reverse_time:
            motion = self.steering.steer(origin_native, target_native)
            if motion is None:
                return None
            if motion.duration <= step:
                if not self._valid(motion, self.bounds_guaranteed):
                    return None
                return motion, np.asarray(target_native, dtype=float).copy(), motion

            waypoint = motion.flat_state(step)
            prefix = motion.truncated(step)
            if self._valid(prefix, self.bounds_guaranteed):
                return prefix, waypoint, prefix
            edge = self.steering.steer(origin_native, waypoint)
            if edge is not None and self._valid(edge, self.bounds_guaranteed):
                return edge, waypoint, edge
            blended = origin_native + (step / motion.duration) * (target_native - origin_native)
            edge = self.steering.steer(origin_native, blended)
            if edge is not None and self._valid(edge, self.bounds_guaranteed):
                return edge, blended, edge
            return None

        origin_phys = reverse_state(origin_native)
        target_phys = reverse_state(target_native)
        motion = self.steering.steer(target_phys, origin_phys)  # target -> origin
        if motion is None:
            return None
        reverse_time_motion = reverse_motion(motion)  # origin -> target, and reverse-time motion
        if reverse_time_motion.duration <= step:
            physical_edge = reverse_motion(reverse_time_motion)
            if not self._valid(physical_edge, self.bounds_guaranteed):
                return None
            return reverse_time_motion, target_native.copy(), physical_edge

        waypoint_native = reverse_time_motion.flat_state(step)
        prefix = reverse_time_motion.truncated(step)
        physical_edge = reverse_motion(prefix)
        if self._valid(physical_edge, self.bounds_guaranteed):
            return prefix, waypoint_native, physical_edge
        waypoint_phys = reverse_state(waypoint_native)
        edge = self.steering.steer(waypoint_phys, origin_phys)  # physical waypoint -> origin
        if edge is not None and self._valid(edge, self.bounds_guaranteed):
            return reverse_motion(edge), waypoint_native, edge
        blended_phys = origin_phys + (step / reverse_time_motion.duration) * (
            target_phys - origin_phys
        )
        edge = self.steering.steer(blended_phys, origin_phys)
        if edge is not None and self._valid(edge, self.bounds_guaranteed):
            return reverse_motion(edge), reverse_state(blended_phys), edge
        return None

    def _propagate_control(self, tree: _KPieceTree, origin_native: np.ndarray):
        """Sample a constant jerk and propagate it for a random number of simulation steps."""
        params = self.params
        dt = float(params.dt)
        steps = self._sample_steps()
        input_low = np.asarray(params.input_min, dtype=float)
        input_high = np.asarray(params.input_max, dtype=float)
        jerk = self.rng.uniform(input_low, input_high) * float(
            self.options.control_fraction
        )
        position, velocity, acceleration = (
            np.asarray(origin_native[0:3], dtype=float),
            np.asarray(origin_native[3:6], dtype=float),
            np.asarray(origin_native[6:9], dtype=float),
        )
        coefficients = np.vstack([position, velocity, 0.5 * acceleration, jerk / 6.0])
        full = FlatMotion(coefficients, steps * dt)

        last_valid = 0
        for k in range(1, steps + 1):
            t = k * dt
            state = full.flat_state(t)
            if not self.checker.is_state_valid(state, full, t):
                break
            last_valid = k
        if last_valid <= 0:
            return None
        stored = full.truncated(last_valid * dt)
        physical_edge = reverse_motion(stored) if tree.reverse_time else stored
        if not self._valid(physical_edge, bounds_guaranteed=False):
            for k in range(last_valid - 1, 0, -1):
                candidate = full.truncated(k * dt)
                edge = reverse_motion(candidate) if tree.reverse_time else candidate
                if self._valid(edge, bounds_guaranteed=False):
                    return candidate, candidate.flat_state(k * dt), edge
            return None
        return stored, stored.flat_state(stored.duration), physical_edge

    # -- splitting ---------------------------------------------------------------------
    def _split_physical(self, motion: FlatMotion) -> List[_Piece]:
        """Split a physical forward edge at joint-cell boundaries; pieces concatenate."""
        dt = max(float(self.params.dt), 1e-9)
        duration = float(motion.duration)
        substep = max(min(0.01, dt / 5.0), duration / 511.0)
        times = np.arange(0.0, duration + 1e-12, substep)
        if times.size == 0 or times[-1] < duration - 1e-12:
            times = np.append(times, duration)
        states = np.hstack(
            [motion.samples_at(times, level) for level in range(ORDER)]
        ).reshape(times.shape[0], N_STATES)
        keys = self.projection.keys(states)
        changes = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1))
        if changes.size == 0:
            return [
                _Piece(motion, motion.flat_state(duration), self.projection.key(states[0]),
                       max(1.0, duration / dt), 0.0, duration)
            ]

        boundaries = [0.0]
        for change in changes:
            left, right = float(times[change]), float(times[change + 1])
            left_key = self.projection.key(motion.flat_state(left))
            for _ in range(24):
                middle = 0.5 * (left + right)
                if self.projection.key(motion.flat_state(middle)) == left_key:
                    left = middle
                else:
                    right = middle
            boundary = 0.5 * (left + right)
            if boundary - boundaries[-1] > 1e-9 and duration - boundary > 1e-9:
                boundaries.append(boundary)
        boundaries.append(duration)

        pieces = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end - start <= 1e-12:
                continue
            piece = motion.slice(start, end)
            midpoint = 0.5 * (start + end)
            key = self.projection.key(motion.flat_state(midpoint))
            pieces.append(
                _Piece(
                    motion=piece,
                    end_state=piece.flat_state(piece.duration),
                    key=key,
                    steps=max(1.0, (end - start) / dt),
                    t0=start,
                    t1=end,
                )
            )
        return pieces

    # -- cell score --------------------------------------------------------------------
    def _cell_score(self, tree: _KPieceTree, other: _KPieceTree, endpoint_phys: np.ndarray) -> float:
        score = 1.0 + math.log(max(tree.iteration, 1))
        if self.options.cell_score_distance:
            distance = self.metric.distance(endpoint_phys, other.physical_state(0))
            score /= 1.0 + max(distance, 0.0)
        return float(score)

    # -- search ------------------------------------------------------------------------
    def _extend(self, tree: _KPieceTree, other: _KPieceTree) -> Optional[_KPIECEOutcome]:
        tree.iteration += 1
        selected = tree.grid.select(
            self.options.border_fraction,
            self.rng,
            self.options.motion_selection,
            self.options.halfnormal_focus,
        )
        if selected is None:
            return None
        selected_key, node = selected
        origin_native = tree.states[node]
        step = self._sample_steps() * float(self.params.dt)

        if self.options.expansion == "control":
            propagated = self._propagate_control(tree, origin_native)
        else:
            if self.options.expansion == "geometric":
                target = self._sample_geometric(tree, other, origin_native)
            else:
                target = self._sample_global(tree, other)
            propagated = self._propagate_bvp(tree, origin_native, target, step)

        if propagated is None:
            tree.grid.decay(selected_key, self.options.bad_score_factor)
            return None
        stored, new_native, physical_edge = propagated
        if not self._valid(physical_edge, bounds_guaranteed=self.bounds_guaranteed):
            tree.grid.decay(selected_key, self.options.bad_score_factor)
            return None

        self.stats.record(physical_edge, self.projection, float(self.params.dt))
        pieces = self._split_physical(physical_edge)
        average_pieces = 2.0 * float(tree.grid.n_pieces) / max(len(tree.grid), 1)
        interesting = False
        for piece in pieces:
            cell = tree.grid.get(piece.key)
            if (
                cell is None
                or len(cell.motions) <= average_pieces
                or piece.key in other.grid.cells
            ):
                interesting = True
                break

        tree.grid.decay(selected_key, self.options.good_score_factor)
        if not interesting and self.rng.random() >= self.options.interesting_probability:
            return None

        parent = node
        new_nodes: List[Tuple[int, Tuple[int, ...]]] = []
        if tree.reverse_time:
            for piece in reversed(pieces):
                reverse_piece = reverse_motion(piece.motion)
                endpoint_native = reverse_piece.flat_state(reverse_piece.duration)
                # The new goal-tree node's physical state is the *start* of the physical
                # piece; the reverse piece ends at R(piece.start_state).
                node_physical = reverse_state(endpoint_native)
                score = self._cell_score(tree, other, node_physical)
                parent = tree.add(
                    endpoint_native, parent, reverse_piece, piece.key, piece.steps, score
                )
                new_nodes.append((parent, piece.key))
                self._record(tree.physical_state(parent))
        else:
            for piece in pieces:
                score = self._cell_score(tree, other, piece.end_state)
                parent = tree.add(
                    piece.end_state, parent, piece.motion, piece.key, piece.steps, score
                )
                new_nodes.append((parent, piece.key))
                self._record(tree.physical_state(parent))

        if not new_nodes:
            return None

        shared = [(index, key) for index, key in new_nodes if key in other.grid.cells]
        shared.sort(
            key=lambda item: self.metric.distance(
                tree.physical_state(item[0]), other.physical_state(other.grid.cells[item[1]].motions[0])
            )
        )
        for index, key in shared[: max(1, int(self.options.junction_attempts))]:
            outcome = self._try_junction(tree, other, key, index)
            if outcome is not None:
                if self.stop_on_success:
                    return outcome
                if (
                    self.best_calibration_path is None
                    or outcome.path.duration() < self.best_calibration_path.duration()
                ):
                    self.best_calibration_path = outcome.path
        return None

    # -- junction chains ---------------------------------------------------------------
    def _start_chain(self, index: int):
        nodes = self.start_tree.chain_to_root(index)
        nodes.reverse()
        states = [self.start_tree.physical_state(i) for i in nodes]
        motions = [self.start_tree.motions[i] for i in nodes[1:]]
        return states, motions

    def _goal_chain(self, index: int):
        nodes = self.goal_tree.chain_to_root(index)
        states = [self.goal_tree.physical_state(i) for i in nodes]
        motions = [
            reverse_motion(self.goal_tree.motions[i]) for i in nodes[:-1]
        ]
        return states, motions

    def _assemble(
        self,
        start_index: int,
        goal_index: int,
        bridge_motions: Sequence[FlatMotion],
        mid_states: Sequence[np.ndarray] = (),
    ):
        start_states, start_motions = self._start_chain(start_index)
        goal_states, goal_motions = self._goal_chain(goal_index)
        if not bridge_motions and not mid_states:
            if np.max(np.abs(start_states[-1] - goal_states[0])) > 1e-7:
                return None
            states = start_states + goal_states[1:]
            motions = list(start_motions) + list(goal_motions)
        else:
            states = start_states + [np.asarray(s, dtype=float) for s in mid_states] + goal_states
            motions = list(start_motions) + list(bridge_motions) + list(goal_motions)
        return FlatPath(states=states, motions=motions)

    def _cross_chain_paths(
        self, start_index: int, goal_index: int, max_pairs: int = 30
    ):
        """Yield continuous paths whose junction edge spans the two full chains.

        This is the expensive option-2 search: rather than only trying the two meeting-cell
        endpoints, try a valid BVP from any state along the start chain to any state along the
        goal chain. A pair earlier than the junction can avoid a dynamically incompatible cell
        boundary. The chains themselves are already valid, so only the cross edge is new.
        """
        start_states, start_motions = self._start_chain(start_index)
        goal_states, goal_motions = self._goal_chain(goal_index)
        pairs = []
        for i, start_state in enumerate(start_states):
            for g, goal_state in enumerate(goal_states):
                pairs.append(
                    (self.metric.distance(start_state, goal_state), i, g)
                )
        pairs.sort(key=lambda item: item[0])
        for _, i, g in pairs[: max(1, int(max_pairs))]:
            bridge = self.steering.steer(start_states[i], goal_states[g])
            if bridge is None or not self._valid(bridge, self.bounds_guaranteed):
                continue
            states = start_states[: i + 1] + goal_states[g:]
            motions = list(start_motions[:i]) + [bridge] + list(goal_motions[g:])
            path = FlatPath(states=states, motions=motions)
            if self._validate_path(path):
                yield path

    def _bridge_sequences(self, start_phys: np.ndarray, goal_phys: np.ndarray):
        """Yield ``(bridge_motions, mid_states)`` connection options, cheapest first.

        The direct BVP is option 1. A two-segment bridge through an interpolated flat state
        is the extra search that makes the shortcut mode more than a single steer; each
        segment still has to be a valid bound-respecting BVP.
        """
        if self.metric.distance(start_phys, goal_phys) < 1e-9:
            yield [], []
            return
        direct = self.steering.steer(start_phys, goal_phys)
        if direct is not None and self._valid(direct, self.bounds_guaranteed):
            yield [direct], []
        for alpha in (0.5, 0.25, 0.75):
            middle = start_phys + alpha * (goal_phys - start_phys)
            first = self.steering.steer(start_phys, middle)
            if first is None or not self._valid(first, self.bounds_guaranteed):
                continue
            second = self.steering.steer(middle, goal_phys)
            if second is None or not self._valid(second, self.bounds_guaranteed):
                continue
            yield [first, second], [middle]

    def _validate_path(self, path: FlatPath) -> bool:
        if len(path.states) != len(path.motions) + 1:
            return False
        if np.max(np.abs(path.states[0] - self.start)) > 1e-6:
            return False
        if np.max(np.abs(path.states[-1] - self.goal)) > 1e-5:
            return False
        for index, motion in enumerate(path.motions):
            if motion is None:
                return False
            if not self.checker.is_motion_valid(motion):
                return False
            if not is_motion_within_bounds(motion, self.params):
                return False
            if np.max(np.abs(motion.flat_state(0.0) - path.states[index])) > 1e-6:
                return False
            if np.max(np.abs(motion.flat_state(motion.duration) - path.states[index + 1])) > 1e-6:
                return False
        return True

    def _try_junction(
        self, tree: _KPieceTree, other: _KPieceTree, key: Tuple[int, ...], index: int
    ) -> Optional[_KPIECEOutcome]:
        strategy = self.options.connection
        failure_key = (strategy, key)
        if self._junction_failures.get(failure_key, 0) >= 20:
            return None
        self.junction_stats["attempts"] += 1

        if strategy == "ocp":
            if tree is self.start_tree:
                start_index, goal_index = index, self._junction_other_node(
                    other, key, tree.physical_state(index)
                )
            else:
                start_index, goal_index = (
                    self._junction_other_node(other, key, tree.physical_state(index)),
                    index,
                )
            if start_index is None or goal_index is None:
                self._junction_failures[failure_key] = self._junction_failures.get(failure_key, 0) + 1
                return None
            jump = self.metric.distance(
                self.start_tree.physical_state(start_index),
                self.goal_tree.physical_state(goal_index),
            )
            if jump > float(self.options.ocp_max_jump):
                # A large jump usually means the chains are in different homotopy classes;
                # do not spend an expensive OCP attempt on it or consume the attempt budget.
                self.junction_stats["ocp_jump_too_large"] += 1
                self._junction_failures[failure_key] = self._junction_failures.get(failure_key, 0) + 1
                return None
            if self._ocp_attempts >= int(self.options.ocp_max_attempts):
                self.junction_stats["ocp_skipped"] += 1
                return None
            self.junction_stats["ocp_attempts"] += 1
            self._ocp_attempts += 1
            outcome = self._connect_ocp(start_index, goal_index)
            if outcome is None:
                self._junction_failures[failure_key] = self._junction_failures.get(failure_key, 0) + 1
            return outcome

        start_candidates, goal_candidates = self._junction_candidates(tree, other, key, index)

        if strategy == "shortcut":
            self.junction_stats["shortcut_attempts"] += 1
            limit = max(1, int(self.options.shortcut_candidates))
            if self.options.junction_selection == "closest":
                anchor = tree.physical_state(index)
                # Each candidate list belongs to a different tree; sort with its own accessor.
                start_candidates.sort(
                    key=lambda i: self.metric.distance(anchor, self.start_tree.physical_state(i))
                )
                goal_candidates.sort(
                    key=lambda i: self.metric.distance(anchor, self.goal_tree.physical_state(i))
                )
            start_candidates = start_candidates[:limit]
            goal_candidates = goal_candidates[:limit]
            candidates = []

            # Option 2 proper: try a cross-tree edge from any state of the start chain to any
            # state of the goal chain, not only the two meeting-cell endpoints. In uniform
            # mode the sampled other-tree nodes seed the search; in closest mode the nearest
            # pairs do. At most two pairs are expanded to bound the cost.
            cell_pairs = [
                (
                    self.metric.distance(
                        self.start_tree.physical_state(si),
                        self.goal_tree.physical_state(gi),
                    ),
                    si,
                    gi,
                )
                for si in start_candidates
                for gi in goal_candidates
            ]
            if self.options.junction_selection == "closest":
                cell_pairs.sort(key=lambda item: item[0])
            for _, si, gi in cell_pairs[:2]:
                for path in self._cross_chain_paths(si, gi, max_pairs=30):
                    candidates.append((path.duration(), path))

            for start_index in start_candidates:
                for goal_index in goal_candidates:
                    start_phys = self.start_tree.physical_state(start_index)
                    goal_phys = self.goal_tree.physical_state(goal_index)
                    for bridges, mids in self._bridge_sequences(start_phys, goal_phys):
                        path = self._assemble(start_index, goal_index, bridges, mids)
                        if path is not None and self._validate_path(path):
                            candidates.append((path.duration(), path))
            candidates.sort(key=lambda item: item[0])
            for _, path in candidates[:3]:
                deadline = self._deadline if self._deadline is not None else time.monotonic() + 5.0
                shortened = _shortcut(
                    path, self.checker, self.steering, self.params, deadline
                )
                if self._validate_path(shortened):
                    self.junction_stats["shortcut_success"] += 1
                    return _KPIECEOutcome(shortened, "shortcut")
            self._junction_failures[failure_key] = self._junction_failures.get(failure_key, 0) + 1
            return None

        self.junction_stats["steer_attempts"] += 1
        pairs = []
        for start_index in start_candidates:
            for goal_index in goal_candidates:
                start_phys = self.start_tree.physical_state(start_index)
                goal_phys = self.goal_tree.physical_state(goal_index)
                pairs.append((self.metric.distance(start_phys, goal_phys), start_index, goal_index))
        if self.options.junction_selection == "closest":
            pairs.sort(key=lambda item: item[0])
        for _, start_index, goal_index in pairs[: max(1, int(self.options.connect_tries))]:
            start_phys = self.start_tree.physical_state(start_index)
            goal_phys = self.goal_tree.physical_state(goal_index)
            if self.metric.distance(start_phys, goal_phys) < 1e-9:
                bridges, mids = [], []
            else:
                bridge = self.steering.steer(start_phys, goal_phys)
                if bridge is None or not self._valid(bridge, self.bounds_guaranteed):
                    continue
                bridges, mids = [bridge], []
            path = self._assemble(start_index, goal_index, bridges, mids)
            if path is not None and self._validate_path(path):
                self.junction_stats["steer_success"] += 1
                return _KPIECEOutcome(path, "steer")
        self._junction_failures[failure_key] = self._junction_failures.get(failure_key, 0) + 1
        return None

    def _closest_node(
        self, tree: _KPieceTree, key: Tuple[int, ...], physical: np.ndarray
    ) -> Optional[int]:
        cell = tree.grid.cells.get(key)
        if cell is None or not cell.motions:
            return None
        return min(cell.motions, key=lambda i: self.metric.distance(tree.physical_state(i), physical))

    def _junction_other_node(
        self, tree: _KPieceTree, key: Tuple[int, ...], physical: np.ndarray
    ) -> Optional[int]:
        """Pick the other tree's node for a junction: uniform by default, closest on option."""
        cell = tree.grid.cells.get(key)
        if cell is None or not cell.motions:
            return None
        if self.options.junction_selection == "closest":
            return self._closest_node(tree, key, physical)
        return int(cell.motions[int(self.rng.integers(0, len(cell.motions)))])

    def _junction_candidates(
        self, tree: _KPieceTree, other: _KPieceTree, key: Tuple[int, ...], index: int
    ):
        """Start/goal-node candidates for the steer and shortcut junction modes.

        ``junction_selection="uniform"`` samples the *other* tree's cell motions uniformly
        without replacement (the extended tree's new node is fixed); ``"closest"`` returns the
        full other list and lets the caller sort by distance.
        """
        other_nodes = list(other.grid.cells[key].motions)
        if self.options.junction_selection == "uniform":
            need = max(
                int(self.options.connect_tries),
                int(self.options.shortcut_candidates),
                2,
            )
            count = min(need, len(other_nodes))
            sampled = [
                int(value)
                for value in self.rng.choice(np.asarray(other_nodes), size=count, replace=False)
            ]
        else:
            sampled = other_nodes
        if tree is self.start_tree:
            return [index], sampled
        return sampled, [index]

    # -- OCP junction ------------------------------------------------------------------
    def _locate_chain(self, motions: Sequence[FlatMotion], chain_start: float, t: float):
        local = t - chain_start
        cursor = 0.0
        for index, motion in enumerate(motions):
            if local <= motion.duration + 1e-12 or index == len(motions) - 1:
                query = min(max(local - cursor, 0.0), motion.duration)
                return [
                    motion.state_at(query, level) for level in range(ORDER + 1)
                ]
            cursor += motion.duration
        return [np.zeros(OUTPUT_DIM) for _ in range(ORDER + 1)]

    def _connect_ocp(self, start_index: int, goal_index: int) -> Optional[_KPIECEOutcome]:
        try:
            return self._connect_ocp_impl(start_index, goal_index)
        except Exception as exc:  # OCP failures are normal search outcomes, never fatal
            if os.environ.get("KPIECE_DEBUG_OCP"):
                import traceback

                traceback.print_exc()
            self.junction_stats["ocp_solve_failed"] += 1
            self.junction_stats.setdefault("ocp_exception", 0)
            self.junction_stats["ocp_exception"] += 1
            return None

    def _connect_ocp_impl(self, start_index: int, goal_index: int) -> Optional[_KPIECEOutcome]:
        start_states, start_motions = self._start_chain(start_index)
        goal_states, goal_motions = self._goal_chain(goal_index)
        start_phys = self.start_tree.physical_state(start_index)
        goal_phys = self.goal_tree.physical_state(goal_index)
        jump = self.metric.distance(start_phys, goal_phys)
        if jump > float(self.options.ocp_max_jump):
            self.junction_stats["ocp_jump_too_large"] += 1
            return None
        duration_start = float(sum(m.duration for m in start_motions))
        duration_goal = float(sum(m.duration for m in goal_motions))
        total = duration_start + duration_goal
        if total <= 0.0:
            self.junction_stats["ocp_solve_failed"] += 1
            return None

        params = self.params
        horizon = int(params.horizon)
        if getattr(params, "min_time", False):
            max_duration = horizon * float(params.max_dt)
        else:
            max_duration = horizon * float(params.dt)
        # If the concatenated chains are longer than the OCP's time grid can represent, sample
        # them on a compressed timeline. The seed remains a start->goal path; the solver
        # repairs the dynamics and re-times it.
        time_scale = max(1.0, total / max(max_duration, 1e-9))
        sample_total = total / time_scale
        times = np.linspace(0.0, sample_total, horizon + 1)
        source_times = times * time_scale
        positions = np.zeros((horizon + 1, OUTPUT_DIM))
        velocities = np.zeros((horizon + 1, OUTPUT_DIM))
        accelerations = np.zeros((horizon + 1, OUTPUT_DIM))
        jerks = np.zeros((horizon + 1, OUTPUT_DIM))
        for i, t in enumerate(source_times):
            if t <= duration_start + 1e-12:
                levels = self._locate_chain(start_motions, 0.0, t)
            else:
                levels = self._locate_chain(goal_motions, duration_start, t)
            positions[i], velocities[i], accelerations[i], jerks[i] = levels

        params_ocp = copy.deepcopy(params)
        params_ocp.use_global_planner = False
        params_ocp.initial_state = self.start.tolist()
        params_ocp.end_state = self.goal.tolist()
        params_ocp.payload_pos_init = positions.tolist()
        params_ocp.payload_vel_init = velocities.tolist()
        params_ocp.payload_input_init = jerks[:-1].tolist()

        from poly_fly.optimal_planner.planner import Planner  # lazy: keeps this module standalone

        planner = Planner(params_ocp, plot=False)
        planner.setup(viz_cb=False)

        opti = planner.opti
        variables = planner.variables
        opti.set_initial(variables["x"][0:3, :], positions.T)
        opti.set_initial(variables["x"][3:6, :], velocities.T)
        opti.set_initial(variables["x"][6:9, :], accelerations.T)
        opti.set_initial(variables["u"], jerks[:-1].T)
        if getattr(params_ocp, "min_time", False):
            nominal = sample_total / max(horizon, 1)
            nominal = min(max(nominal, float(params_ocp.min_dt)), float(params_ocp.max_dt))
            opti.set_initial(variables["t"], [nominal] * horizon)

        solution, _, success, _, _ = planner.optimize()
        if not success:
            self.junction_stats["ocp_solve_failed"] += 1
            return None

        path = self._ocp_solution_to_path(solution, params_ocp, horizon)
        if path is None:
            if os.environ.get("KPIECE_DEBUG_OCP"):
                print("KPIECE OCP: solution -> path conversion failed", flush=True)
            self.junction_stats["ocp_invalid"] += 1
            return None
        if np.max(np.abs(path.states[0] - self.start)) > 1e-6:
            if os.environ.get("KPIECE_DEBUG_OCP"):
                print(
                    "KPIECE OCP: start mismatch %.3e"
                    % float(np.max(np.abs(path.states[0] - self.start))),
                    flush=True,
                )
            self.junction_stats["ocp_invalid"] += 1
            return None
        if np.max(np.abs(path.states[-1] - self.goal)) > _OCP_GOAL_TOL + 1e-6:
            if os.environ.get("KPIECE_DEBUG_OCP"):
                print(
                    "KPIECE OCP: goal error %.3e"
                    % float(np.max(np.abs(path.states[-1] - self.goal))),
                    flush=True,
                )
            self.junction_stats["ocp_invalid"] += 1
            return None
        if not self._validate_ocp_path(path):
            if os.environ.get("KPIECE_DEBUG_OCP"):
                print("KPIECE OCP: dense validation failed", flush=True)
                self._debug_ocp_dense(path)
            self.junction_stats["ocp_invalid"] += 1
            return None
        snapped = _goal_snap(path, self.goal, self.checker, self.steering, self.params)
        if snapped is not path and self._validate_ocp_path(snapped):
            path = snapped
        self.junction_stats["ocp_success"] += 1
        return _KPIECEOutcome(path, "ocp", ocp_solution=solution)

    def _ocp_solution_to_path(self, solution, params_ocp, horizon: int):
        try:
            states = np.asarray(solution["x"], dtype=float).reshape(horizon + 1, N_STATES)
            controls = np.asarray(solution["u"], dtype=float).reshape(horizon, N_INPUTS)
        except (KeyError, ValueError):
            return None
        if getattr(params_ocp, "min_time", False) and "t" in solution:
            intervals = np.asarray(solution["t"], dtype=float).reshape(-1)
            if intervals.shape[0] != horizon:
                return None
        else:
            intervals = np.full(horizon, float(params_ocp.dt))
        if np.any(intervals <= 0.0) or not np.isfinite(intervals).all():
            return None

        motions = []
        for i in range(horizon):
            coefficients = np.vstack(
                [states[i, 0:3], states[i, 3:6], 0.5 * states[i, 6:9], controls[i] / 6.0]
            )
            motion = FlatMotion(coefficients, float(intervals[i]))
            # RK4 with constant jerk is exact for a cubic; any residual is solver tolerance.
            if np.max(np.abs(motion.flat_state(motion.duration) - states[i + 1])) > 1e-3:
                return None
            motions.append(motion)
        return FlatPath(states=[row.copy() for row in states], motions=motions)

    def _dense_path_samples(self, path: FlatPath):
        samples = []
        jerks = []
        for motion in path.motions:
            count = max(2, int(math.ceil(motion.duration / _OCP_DENSE_DT)) + 1)
            local = np.linspace(0.0, motion.duration, count)
            samples.append(
                np.hstack([motion.samples_at(local, level) for level in range(ORDER)])
            )
            jerks.append(motion.samples_at(local, ORDER))
        dense = np.vstack(samples).reshape(-1, N_STATES)
        dense_jerks = np.vstack(jerks).reshape(-1, N_INPUTS)
        return dense, dense_jerks

    def _validate_ocp_path(self, path: FlatPath) -> bool:
        low = np.asarray(self.params.state_min, dtype=float)
        high = np.asarray(self.params.state_max, dtype=float)
        input_low = np.asarray(self.params.input_min, dtype=float)
        input_high = np.asarray(self.params.input_max, dtype=float)
        dense, dense_jerks = self._dense_path_samples(path)
        if not np.isfinite(dense).all() or not np.isfinite(dense_jerks).all():
            return False
        # IPOPT satisfies the constraints only to its tolerance (~1e-4); the dense samples
        # can therefore sit a few 1e-8 outside a bound. Accept that solver slack, but do not
        # clip the collision geometry: the obstacle check below uses the true poses.
        eps = 1e-6
        state_violation = max(float(np.max(low - dense)), float(np.max(dense - high)))
        jerk_violation = max(
            float(np.max(input_low - dense_jerks)),
            float(np.max(dense_jerks - input_high)),
        )
        if state_violation > eps or jerk_violation > eps:
            return False
        gravity = np.array([0.0, 0.0, 9.81])
        if np.any(np.linalg.norm(dense[:, 6:9] + gravity, axis=1) < ACC_NORM_EPS):
            return False
        # FlatEdgeChecker.configs_collision_free treats a bound violation as non-free with a
        # 1e-9 tolerance; clip only for that internal bound test, then check the true poses.
        configs = np.hstack([np.clip(dense[:, 0:3], low[0:3], high[0:3]),
                             np.clip(dense[:, 6:9], low[6:9], high[6:9])])
        if not self.checker.configs_collision_free(configs).all():
            return False
        return True

    def _debug_ocp_dense(self, path: FlatPath) -> None:
        low = np.asarray(self.params.state_min, dtype=float)
        high = np.asarray(self.params.state_max, dtype=float)
        input_low = np.asarray(self.params.input_min, dtype=float)
        input_high = np.asarray(self.params.input_max, dtype=float)
        dense, dense_jerks = self._dense_path_samples(path)
        state_violation = max(float(np.max(low - dense)), float(np.max(dense - high)))
        jerk_violation = max(
            float(np.max(input_low - dense_jerks)), float(np.max(dense_jerks - input_high))
        )
        configs = np.hstack([np.clip(dense[:, 0:3], low[0:3], high[0:3]),
                             np.clip(dense[:, 6:9], low[6:9], high[6:9])])
        free = self.checker.configs_collision_free(configs)
        print(
            "KPIECE OCP: dense %d samples, state violation %.3e, jerk violation %.3e, colliding %d"
            % (dense.shape[0], state_violation, jerk_violation, int(np.count_nonzero(~free))),
            flush=True,
        )

    # -- main loop ---------------------------------------------------------------------
    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[_KPIECEOutcome]:
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        if not self.checker.is_state_valid(self.start) or not self.checker.is_state_valid(self.goal):
            return None

        root_score = 1.0
        self.start_tree = _KPieceTree(self.start, self.projection, False, root_score)
        self.goal_tree = _KPieceTree(reverse_state(self.goal), self.projection, True, root_score)
        self._record(self.start)
        self._record(self.goal)
        self.explored_nodes_3d[1] = [float(self.goal[0]), float(self.goal[1]), float(self.goal[2])]

        # The roots can already share a cell (start and goal close together).
        start_key = self.projection.key(self.start)
        if start_key in self.goal_tree.grid.cells:
            outcome = self._try_junction(self.start_tree, self.goal_tree, start_key, 0)
            if outcome is not None:
                return outcome

        tree_a, tree_b = self.start_tree, self.goal_tree
        deadline = time.monotonic() + self.timeout
        self._deadline = deadline
        for _ in range(max(1, int(self.max_iterations))):
            if time.monotonic() > deadline:
                break
            self.iterations += 1

            if self.options.tree_selection == "alternate":
                swap = True
            elif self.options.tree_selection == "ratio":
                ratio = float(getattr(self.params, "rrt_tree_ratio", 0.5))
                smaller = min(len(tree_a), len(tree_b))
                larger = max(len(tree_a), len(tree_b), 1)
                swap = (larger - smaller) / larger < ratio
            else:  # "larger"
                swap = len(tree_a) < len(tree_b)
            if swap:
                tree_a, tree_b = tree_b, tree_a

            outcome = self._extend(tree_a, tree_b)
            if outcome is not None and self.stop_on_success:
                return outcome
        return None


# --------------------------------------------------------------------------------------
# adaptive cell-size calibration
# --------------------------------------------------------------------------------------
def _adjust_sizes(
    stats: dict,
    sizes: Tuple[float, ...],
    options: KPieceOptions,
    projection: CellProjection,
    params,
):
    """Paper-style per-dimension vote; returns ``(new_sizes, decisions, min_steps)``."""
    sizes = list(float(v) for v in sizes)
    dims = len(sizes)
    votes = np.zeros(dims)
    decisions = []

    parts = np.asarray(stats.get("parts_avg", np.ones(dims)), dtype=float)
    fast = np.asarray(stats.get("more_than_2_frac", np.zeros(dims)), dtype=float)
    for i in range(dims):
        if parts[i] > 4.0 or fast[i] > 0.10:
            votes[i] += 1.0
            decisions.append("dim %d: parts %.2f / fast %.2f -> grow" % (i, parts[i], fast[i]))
        elif parts[i] < 1.05:
            votes[i] -= 1.0
            decisions.append("dim %d: parts %.2f -> shrink" % (i, parts[i]))

    if dims <= options.interior_rule_max_dim:
        samples = float(stats.get("samples_per_cell", 0.0))
        if samples < 10.0:
            votes += 1.0
            decisions.append("samples/cell %.1f < 10 -> grow" % samples)
        elif samples > 200.0:
            votes -= 1.0
            decisions.append("samples/cell %.1f > 200 -> shrink" % samples)

    parts_joint = float(stats.get("parts_avg_joint", 1.0))
    if parts_joint > 4.0:
        votes += 1.0
        decisions.append("joint parts %.2f > 4 -> grow" % parts_joint)
    elif parts_joint < 1.05:
        votes -= 1.0
        decisions.append("joint parts %.2f < 1.05 -> shrink" % parts_joint)

    if (
        dims <= options.interior_rule_max_dim
        and int(stats.get("interior_cells", 0)) == 0
        and int(stats.get("cells", 0)) >= 100
    ):
        # The no-interior-cell signal is only meaningful once the grid is dense enough for a
        # cell to have a chance of all 2*dim neighbours; on a sparse grid it just fights the
        # samples-per-cell rule and makes the calibration oscillate.
        votes -= 1.0
        decisions.append("no interior cells -> shrink")

    low = np.asarray(params.state_min, dtype=float)
    high = np.asarray(params.state_max, dtype=float)
    factor = float(options.cell_factor)
    changed = False
    for i in range(dims):
        if votes[i] > 0:
            sizes[i] *= factor
            changed = True
        elif votes[i] < 0:
            sizes[i] /= factor
            changed = True
        dimension = projection.dims[i]
        span = abs(float(high[dimension] - low[dimension]) * projection.scales[i])
        if span > 0.0:
            sizes[i] = min(max(sizes[i], 1e-4 * span), 0.5 * span)
        sizes[i] = max(sizes[i], 1e-9)

    min_steps = int(options.min_steps)
    max_steps = int(options.max_steps)
    if float(stats.get("long_motion_frac", 1.0)) < 0.5:
        min_steps = min(max(min_steps + 1, 3), max_steps)
        decisions.append("long motions < 50%% -> min_steps=%d" % min_steps)
        changed = True

    return tuple(sizes), decisions, min_steps, changed


def _calibrate(params, start, goal, options: KPieceOptions, mobility, seed: int):
    """Run the paper's restart loop; returns ``(sizes, min_steps, rounds, best_path, time)``."""
    projection = CellProjection.from_params(params, options)
    sizes = tuple(projection.sizes)
    rounds = []
    best_path = None
    started = time.monotonic()
    min_steps = int(options.min_steps)
    seen_sizes = {tuple(round(float(v), 12) for v in sizes)}
    for round_index in range(max(1, int(options.calibration_rounds))):
        round_options = replace(
            options,
            adaptive=False,
            cell_sizes=sizes,
            min_steps=min_steps,
            connection="steer",
        )
        planner = BidirectionalKPIECE(
            params, round_options, mobility=mobility,
            rng=np.random.default_rng(seed),
        )
        planner.calibration_mode = True
        planner.stop_on_success = False
        planner.max_iterations = max(1, int(options.calibration_iters))
        planner.timeout = max(5.0, float(getattr(params, "rrt_timeout", 60.0)) / max(1, options.calibration_rounds))
        planner.plan(start, goal)
        stats = planner.stats.as_dict([planner.start_tree, planner.goal_tree])
        if planner.best_calibration_path is not None:
            if best_path is None or planner.best_calibration_path.duration() < best_path.duration():
                best_path = planner.best_calibration_path
        if int(stats.get("motions", 0)) < 20:
            # Too few motions for the averages to mean anything; keep the current sizes.
            rounds.append(
                {
                    "round": round_index,
                    "sizes_before": list(sizes),
                    "sizes_after": list(sizes),
                    "min_steps_before": min_steps,
                    "min_steps_after": min_steps,
                    "stats": stats,
                    "decisions": ["insufficient motions"],
                    "changed": False,
                }
            )
            break
        new_sizes, decisions, new_min_steps, changed = _adjust_sizes(
            stats, sizes, round_options, planner.projection, params
        )
        rounds.append(
            {
                "round": round_index,
                "sizes_before": list(sizes),
                "sizes_after": list(new_sizes),
                "min_steps_before": min_steps,
                "min_steps_after": new_min_steps,
                "stats": stats,
                "decisions": decisions,
                "changed": bool(changed),
            }
        )
        sizes = new_sizes
        min_steps = new_min_steps
        rounded = tuple(round(float(v), 12) for v in sizes)
        if not changed or rounded in seen_sizes:
            # No change, or the adjustment returned to a size already tried: the statistics
            # are noisy enough that another round would just oscillate.
            break
        seen_sizes.add(rounded)
    elapsed = time.monotonic() - started
    return sizes, min_steps, rounds, best_path, elapsed


# --------------------------------------------------------------------------------------
# result and entry points
# --------------------------------------------------------------------------------------
@dataclass
class KPIECEResult:
    path: Optional[FlatPath]
    explored_nodes: List[List[float]]
    explored_nodes_3d: List[List[float]]
    iterations: int
    nodes: int
    pieces: int
    cells: int
    duration: float
    search_duration: float
    calibration_duration: float
    success: bool
    seed: int
    stats: Optional[dict] = None
    calibration: List[dict] = field(default_factory=list)
    junction_stats: dict = field(default_factory=dict)
    path_source: str = "final"
    strategy: Optional[str] = None
    start_tree: Optional[_KPieceTree] = None
    goal_tree: Optional[_KPieceTree] = None

    def trajectory_duration(self) -> Optional[float]:
        return None if self.path is None else float(self.path.duration())


def solve_with_kpiece(
    params,
    start: Optional[Sequence[float]] = None,
    goal: Optional[Sequence[float]] = None,
    show_animation: bool = False,
    mobility=None,
) -> KPIECEResult:
    """Plan with bidirectional KPIECE. Never raises on search failure."""
    started = time.monotonic()
    options = KPieceOptions.from_params(params)
    start_state = np.asarray(params.initial_state if start is None else start, dtype=float)
    goal_state = np.asarray(params.end_state if goal is None else goal, dtype=float)

    calibration: List[dict] = []
    calibration_duration = 0.0
    calibration_path: Optional[FlatPath] = None
    if options.adaptive:
        sizes, min_steps, calibration, calibration_path, calibration_duration = _calibrate(
            params, start_state, goal_state, options, mobility, int(params.rrt_seed)
        )
        options = replace(options, cell_sizes=sizes, min_steps=min_steps, adaptive=False)

    planner = BidirectionalKPIECE(
        params, options, mobility=mobility,
        rng=np.random.default_rng(int(params.rrt_seed)),
    )
    search_started = time.monotonic()
    outcome = planner.plan(start_state, goal_state)
    search_duration = time.monotonic() - search_started

    path = outcome.path if outcome is not None else calibration_path
    path_source = "final" if outcome is not None else "calibration"
    strategy = outcome.strategy if outcome is not None else ("steer" if path is not None else None)
    if path is not None:
        deadline = search_started + float(getattr(params, "rrt_timeout", 60.0))
        if outcome is None or outcome.strategy != "ocp":
            path = _goal_snap(path, planner.goal, planner.checker, planner.steering, params)
        if outcome is not None and outcome.strategy == "shortcut":
            path = _shortcut(path, planner.checker, planner.steering, params, deadline)

    stats = planner.stats.as_dict([planner.start_tree, planner.goal_tree])
    nodes = sum(len(tree) for tree in (planner.start_tree, planner.goal_tree) if tree is not None)
    pieces = sum(
        tree.grid.n_pieces for tree in (planner.start_tree, planner.goal_tree) if tree is not None
    )
    cells = sum(
        len(tree.grid) for tree in (planner.start_tree, planner.goal_tree) if tree is not None
    )
    result = KPIECEResult(
        path=path,
        explored_nodes=planner.explored_nodes,
        explored_nodes_3d=planner.explored_nodes_3d,
        iterations=int(planner.iterations),
        nodes=int(nodes),
        pieces=int(pieces),
        cells=int(cells),
        duration=time.monotonic() - started,
        search_duration=search_duration,
        calibration_duration=calibration_duration,
        success=path is not None,
        seed=int(params.rrt_seed),
        stats=stats,
        calibration=calibration,
        junction_stats=dict(planner.junction_stats),
        path_source=path_source if path is not None else "none",
        strategy=strategy,
        start_tree=planner.start_tree,
        goal_tree=planner.goal_tree,
    )
    if show_animation:  # pragma: no cover - interactive only
        plot_kpiece(params, result)
    return result


# --------------------------------------------------------------------------------------
# plotting and CLI
# --------------------------------------------------------------------------------------
def _edge_polyline(motion: FlatMotion, samples: int = 30) -> np.ndarray:
    times = np.linspace(0.0, motion.duration, max(2, samples))
    return motion.samples_at(times, 0)


def plot_kpiece(params, result: KPIECEResult) -> None:  # pragma: no cover - interactive only
    import matplotlib.pyplot as plt

    from poly_fly.optimal_planner.kinodynamic_rrt_connect import _draw_obstacles

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _draw_obstacles(ax, params)
    for tree, color, label in (
        (result.start_tree, "tab:blue", "start tree"),
        (result.goal_tree, "tab:orange", "goal tree"),
    ):
        if tree is None:
            continue
        for index in range(1, len(tree)):
            motion = tree.physical_motion(index)
            if motion is None:
                continue
            points = _edge_polyline(motion)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, lw=0.5, alpha=0.45)
        nodes = np.vstack([tree.physical_state(i) for i in range(len(tree))])[:, :3]
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=2, color=color, label=label)
    ax.scatter(*np.asarray(params.initial_state)[:3], s=70, color="green", label="start")
    ax.scatter(*np.asarray(params.end_state)[:3], s=90, marker="*", color="red", label="goal")
    if result.path is not None:
        times = np.linspace(0.0, result.path.duration(), 400)
        points = result.path.samples_at(times, 0)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="black", lw=2.0, label="solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        "bidirectional KPIECE %s -- %d iterations, %d cells"
        % ("solved" if result.success else "failed", result.iterations, result.cells)
    )
    ax.legend(loc="upper left", fontsize="small")
    plt.show()


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Bidirectional KPIECE (no optimization)")
    parser.add_argument("--yaml", type=str, required=True, help="path relative to data/params")
    parser.add_argument("--plot", action="store_true", help="plot the search trees and the path")
    parser.add_argument("--save", action="store_true", help="write the trajectory CSV")
    args = parser.parse_args()

    from poly_fly.data_io.utils import PARAMS_DIR
    from poly_fly.optimal_planner.global_planner import OpenSetEmptyException
    from poly_fly.utils.utils import MPC, dictToClass, yamlToDict

    params = dictToClass(MPC, yamlToDict(os.path.join(PARAMS_DIR, args.yaml)))
    params.global_planner_type = "kpiece"
    result = solve_with_kpiece(params, show_animation=args.plot)
    if not result.success:
        raise OpenSetEmptyException(f"Bidirectional KPIECE found no path for {args.yaml}")

    print(
        "path segments: %d  duration: %.3f s  strategy: %s"
        % (len(result.path.motions), result.path.duration(), result.strategy)
    )
    print(
        "iterations: %d  nodes: %d  pieces: %d  cells: %d  wall clock: %.3f s"
        % (result.iterations, result.nodes, result.pieces, result.cells, result.duration)
    )
    if result.calibration:
        print("calibration rounds: %d" % len(result.calibration))
    if result.junction_stats:
        print("junction stats: %s" % result.junction_stats)

    if args.save:
        from poly_fly.optimal_planner.planner import save_result

        stem = os.path.join(params.rrt_csv_subdir, args.yaml)
        save_result(stem, params, path_to_sol_values(result.path, params))


if __name__ == "__main__":
    main()
