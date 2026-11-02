"""Orientation-aware collision geometry for the cable-suspended payload.

The optimal planner (``planner.py``), the kinodynamic RRT and IRIS-ZO all need the same
question answered: given the payload's flat state, where are the payload, cable and
quadrotor, and do those boxes hit an obstacle?  ``planner.py`` expresses the answer as dual
constraints in its OCP; this module provides the shared sampling-time geometry for the RRT
and IRIS-ZO.

Collision geometry depends only on the payload position ``p`` and acceleration ``a``:

* the cable direction is ``p_hat = -(a + g e_z) / ||a + g e_z||`` (unit vector from the
  quadrotor toward the payload),
* the payload box is centred at ``p``,
* the cable box is centred at ``p - (L/2) p_hat`` with its local z along ``p_hat``,
* the quadrotor box is centred at ``p - (L + h/2) p_hat`` and oriented by the flatness map.

Velocity plays no role.  This module therefore exposes both the 9-D flat-state checker used
by the RRT (``FlatEdgeChecker``) and a batched 6-D configuration check
``q = [p, a]`` used by IRIS-ZO.  Keeping the geometry here avoids duplicating it across the
RRT and the region-growing code.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

ORDER = 3  # flat output + 2 derivative levels -> 2k = 6 coefficients (quintic)
OUTPUT_DIM = 3  # flat output is the payload position
GRAVITY = 9.81

# Below this the flatness map divides by ~zero (a + g*e_z vanishes in free fall).
ACC_NORM_EPS = 1.0


def body_boxes_batch(
    params, positions: np.ndarray, accelerations: np.ndarray, use_robot_rotation: bool
):
    """Oriented boxes for payload, cable and quadrotor at ``N`` samples.

    Poses match the OCP's own collision geometry (``add_cable_to_obstacles_constraints`` /
    ``add_quadrotor_to_obstacles_constraints``), with every quantity computed for all
    samples at once: this runs inside the RRT's innermost loop, where per-sample numpy
    calls on 3-vectors dominate everything else.

    ``positions`` and ``accelerations`` are ``(N, 3)``; ``accelerations`` are the payload's
    raw accelerations (gravity is added here).  Returns three
    ``(centers, rotations, half_extents)`` triples, in that body order.
    """
    n = positions.shape[0]
    gravity = np.array([0.0, 0.0, GRAVITY])
    acc = accelerations + gravity
    norm = np.linalg.norm(acc, axis=1)
    p_hat = -acc / norm[:, None]

    # Vectorized port of Planner.get_cable_rotation: local z maps to the cable direction.
    x, y, z = p_hat[:, 0], p_hat[:, 1], p_hat[:, 2]
    denom = np.sqrt(z**2 + y**2 + 1e-5)
    rotation_cable = np.zeros((n, 3, 3))
    rotation_cable[:, 0, 0] = denom
    rotation_cable[:, 0, 2] = x
    rotation_cable[:, 1, 0] = -y * x / denom
    rotation_cable[:, 1, 1] = z / denom
    rotation_cable[:, 1, 2] = y
    rotation_cable[:, 2, 0] = -z * x / denom
    rotation_cable[:, 2, 1] = -y / denom
    rotation_cable[:, 2, 2] = z

    if use_robot_rotation:
        # Vectorized port of compute_quadrotor_rotation_matrix_no_jrk at yaw = 0, where the
        # heading reference b2d is the world +y axis.
        b3c = acc / norm[:, None]
        raw = np.stack([b3c[:, 2], np.zeros(n), -b3c[:, 0]], axis=1)  # cross([0,1,0], b3c)
        b1c = raw / np.linalg.norm(raw, axis=1)[:, None]
        b2c = np.stack(
            [
                b3c[:, 1] * b1c[:, 2] - b3c[:, 2] * b1c[:, 1],
                b3c[:, 2] * b1c[:, 0] - b3c[:, 0] * b1c[:, 2],
                b3c[:, 0] * b1c[:, 1] - b3c[:, 1] * b1c[:, 0],
            ],
            axis=1,
        )
        rotation_quad = np.stack([b1c, b2c, b3c], axis=2)
    else:
        rotation_quad = np.broadcast_to(np.eye(3), (n, 3, 3))

    half_payload = np.full(3, params.payload_radius)
    half_cable = np.array([params.cable_radius, params.cable_radius, params.cable_length / 2.0])
    half_quad = np.array([params.robot_radius, params.robot_radius, params.robot_height / 2.0])
    return [
        (positions, rotation_cable, half_payload),
        (positions - 0.5 * params.cable_length * p_hat, rotation_cable, half_cable),
        (
            positions - (params.cable_length + params.robot_height / 2.0) * p_hat,
            rotation_quad,
            half_quad,
        ),
    ]


def boxes_hit_obstacles(
    centers: np.ndarray,
    rotations: np.ndarray,
    half_extents: np.ndarray,
    obstacle_centers: np.ndarray,
    obstacle_half_extents: np.ndarray,
) -> np.ndarray:
    """Batched exact separating-axis test: oriented boxes vs axis-aligned obstacles.

    ``centers`` is ``(N, 3)``, ``rotations`` ``(N, 3, 3)``, ``half_extents`` ``(3,)``; the
    obstacles are ``(M, 3)``. Returns an ``(N,)`` boolean: does this sample's box hit *any*
    obstacle. Touching counts as a hit.
    """
    n = centers.shape[0]
    m = obstacle_centers.shape[0]
    if m == 0:
        return np.zeros(n, dtype=bool)

    # Candidate separating axes: the box's own axes, the obstacle's, and their 9 cross
    # products. Non-unit cross products are fine -- proj, ra and rb all scale together.
    # Built with plain slicing rather than np.cross: crossing a batch of 3-vectors with a
    # unit axis is just a permutation and a sign, and np.cross's overhead on tiny arrays
    # dominated this function.
    axes = np.empty((n, 15, 3))
    axes[:, 0, :] = rotations[:, :, 0]
    axes[:, 1, :] = rotations[:, :, 1]
    axes[:, 2, :] = rotations[:, :, 2]
    axes[:, 3, :] = np.array([1.0, 0.0, 0.0])
    axes[:, 4, :] = np.array([0.0, 1.0, 0.0])
    axes[:, 5, :] = np.array([0.0, 0.0, 1.0])
    for i in range(3):
        a0 = rotations[:, 0, i]
        a1 = rotations[:, 1, i]
        a2 = rotations[:, 2, i]
        base = 6 + 3 * i
        axes[:, base + 0, 0] = 0.0
        axes[:, base + 0, 1] = a2
        axes[:, base + 0, 2] = -a1
        axes[:, base + 1, 0] = -a2
        axes[:, base + 1, 1] = 0.0
        axes[:, base + 1, 2] = a0
        axes[:, base + 2, 0] = a1
        axes[:, base + 2, 1] = -a0
        axes[:, base + 2, 2] = 0.0

    # Projection radius of a box onto an axis is sum_k h_k |R[:, k] . axis|, i.e. the dot
    # product of each *column* of R (a local axis in world frame) with the candidate axis --
    # "nck" indexes component c of column k, not a row. The absolute value belongs on each
    # term: taking it of the sum lets terms cancel and under-reports the radius, which would
    # wrongly separate overlapping boxes.
    dots = np.einsum("nck,nac->nak", rotations, axes)  # (N, axis, k)
    radius_self = (np.abs(dots) * half_extents[None, None, :]).sum(axis=2)  # (N, axis)
    radius_obs = np.einsum("nac,mc->nma", np.abs(axes), obstacle_half_extents)  # (N, M, axis)

    delta = centers[:, None, :] - obstacle_centers[None, :, :]  # (N, M, 3)
    projected = np.abs(np.einsum("nmc,nac->nma", delta, axes))  # (N, M, axis)

    overlap = np.all(projected <= radius_self[:, None, :] + radius_obs, axis=2)  # (N, M)
    return overlap.any(axis=1)


class FlatEdgeChecker:
    """Validates flat states and polynomial edges against the obstacle set.

    Body poses mirror ``Planner.add_cable_to_obstacles_constraints`` and
    ``Planner.add_quadrotor_to_obstacles_constraints`` so the RRT's free space matches
    the optimal planner's constraint geometry:

    * payload   -- centred at the payload, local z along the cable
    * cable     -- centred at ``p - (L/2) p_hat``, local z along the cable, half-length L/2
    * quadrotor -- centred at ``p - (L + h/2) p_hat``, oriented by the flatness map

    The ``mobility`` argument is accepted for backwards compatibility and is unused; the
    geometry is computed directly from the params (see ``body_boxes_batch``).
    """

    def __init__(self, params, mobility: "object" = None, margin: Optional[float] = None):
        self.params = params
        self._mobility = mobility

        centers, halves = [], []
        for key in params.obstacles.keys():
            obs = params.obstacles[key]
            centers.append([obs["x"], obs["y"], obs["z"]])
            halves.append([obs["l"] / 2.0, obs["b"] / 2.0, obs["h"] / 2.0])
        self.obstacle_centers = np.asarray(centers, dtype=float).reshape(-1, 3)
        self.obstacle_half_extents = np.asarray(halves, dtype=float).reshape(-1, 3)

        margin = params.margin if margin is None else margin
        # Inflating the obstacles approximates the OCP's `margin` CBF slack; it is the
        # conservative direction (an L-infinity rather than L-2 ball).
        self.obstacle_half_extents = self.obstacle_half_extents + float(margin)

        smallest = (
            float(np.min(self.obstacle_half_extents)) if self.obstacle_half_extents.size else 0.05
        )
        self.step = min(float(params.rrt_collision_step), 0.25 * max(smallest, 1e-3))

    def sample_count(self, motion: "FlatMotion") -> int:
        span = motion.duration * motion.peak_speed()
        return max(2, int(math.ceil(span / self.step)))

    # -- validity ----------------------------------------------------------------------
    def is_state_valid(
        self, state: np.ndarray, motion: Optional["FlatMotion"] = None, t: float = 0.0
    ):
        """Single-state validity; ``motion``/``t`` additionally bound-check the jerk."""
        state = np.asarray(state, dtype=float)
        if not np.isfinite(state).all():
            return False  # NaN comparisons are False, so a NaN state would pass every test

        bounds = np.asarray([self.params.state_min, self.params.state_max], dtype=float)
        eps = 1e-9
        if np.any(state < bounds[0] - eps) or np.any(state > bounds[1] + eps):
            return False
        if motion is not None:
            jerk = motion.state_at(t, ORDER)
            if not np.isfinite(jerk).all():
                return False
            if np.any(jerk < np.asarray(self.params.input_min) - eps):
                return False
            if np.any(jerk > np.asarray(self.params.input_max) + eps):
                return False

        config = np.array([[state[0], state[1], state[2], state[6], state[7], state[8]]])
        return bool(self.configs_collision_free(config)[0])

    def configs_collision_free(self, configs: np.ndarray) -> np.ndarray:
        """Batched obstacle check for 6-D flat configurations ``q = [p, a]``.

        ``configs`` is ``(N, 6)``. Velocity does not move any body, so ``q`` carries all
        the information the collision geometry needs. Returns ``(N,)`` bool; out-of-bounds
        and flatness-singular configurations count as not collision free.
        """
        configs = np.asarray(configs, dtype=float)
        if configs.ndim == 1:
            configs = configs.reshape(1, -1)
        if configs.ndim != 2 or configs.shape[1] != 6:
            raise ValueError("configs must have shape (N, 6): [p, a]")

        ok = np.isfinite(configs).all(axis=1)

        bounds = np.asarray([self.params.state_min, self.params.state_max], dtype=float)
        index = [0, 1, 2, 6, 7, 8]
        lo, hi = bounds[0, index], bounds[1, index]
        eps = 1e-9
        ok &= np.all(configs >= lo - eps, axis=1)
        ok &= np.all(configs <= hi + eps, axis=1)

        gravity = np.array([0.0, 0.0, GRAVITY])
        acc = configs[:, 3:6] + gravity
        ok &= np.linalg.norm(acc, axis=1) >= ACC_NORM_EPS

        # Keep NaNs out of the pose math: rows already known invalid use hover.
        raw_acc = np.where(ok[:, None], configs[:, 3:6], np.zeros(3))
        boxes = body_boxes_batch(
            self.params,
            configs[:, 0:3],
            raw_acc,
            getattr(self.params, "use_robot_rotation", True),
        )
        checks = [True, self.params.rrt_check_cable, self.params.rrt_check_quadrotor]
        for (centers, rotations, half), enabled in zip(boxes, checks):
            if not enabled:
                continue
            finite = np.isfinite(centers).all(axis=1) & np.isfinite(rotations).all(axis=(1, 2))
            ok &= finite
            idx = np.flatnonzero(finite)
            if idx.size:
                hit = boxes_hit_obstacles(
                    centers[idx],
                    rotations[idx],
                    half,
                    self.obstacle_centers,
                    self.obstacle_half_extents,
                )
                ok[idx] &= ~hit
        return ok

    def is_motion_valid(self, motion: "FlatMotion", count: Optional[int] = None) -> bool:
        """Validity of a whole polynomial edge, evaluated at every sample at once.

        Unlike OMPL's coarse-to-fine stride walk, all samples are evaluated together. The
        batched SAT is cheap enough that vectorizing over the full sample set beats paying
        Python-level per-sample overhead for early rejection -- which mattered when the
        check was scalar, and dominated the planner's runtime.
        """
        n = self.sample_count(motion) if count is None else count
        times = np.linspace(0.0, motion.duration, n + 1)

        positions = motion.samples_at(times, 0)
        velocities = motion.samples_at(times, 1)
        accelerations = motion.samples_at(times, 2)
        jerks = motion.samples_at(times, ORDER)
        if not all(
            np.isfinite(arr).all() for arr in (positions, velocities, accelerations, jerks)
        ):
            return False

        if not self._samples_within_bounds(positions, velocities, accelerations, jerks):
            return False

        gravity = np.array([0.0, 0.0, GRAVITY])
        if np.any(np.linalg.norm(accelerations + gravity, axis=1) < ACC_NORM_EPS):
            return False  # flatness map is singular here

        boxes = body_boxes_batch(
            self.params,
            positions,
            accelerations,
            getattr(self.params, "use_robot_rotation", True),
        )
        checks = [True, self.params.rrt_check_cable, self.params.rrt_check_quadrotor]
        for (centers, rotations, half), enabled in zip(boxes, checks):
            if not enabled:
                continue
            if not (np.isfinite(centers).all() and np.isfinite(rotations).all()):
                return False
            if boxes_hit_obstacles(
                centers, rotations, half, self.obstacle_centers, self.obstacle_half_extents
            ).any():
                return False
        return True

    def _samples_within_bounds(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        jerks: np.ndarray,
    ) -> bool:
        eps = 1e-9
        lo = np.asarray(self.params.state_min, dtype=float)
        hi = np.asarray(self.params.state_max, dtype=float)
        for values, low, high in (
            (positions, lo[0:3], hi[0:3]),
            (velocities, lo[3:6], hi[3:6]),
            (accelerations, lo[6:9], hi[6:9]),
            (jerks, np.asarray(self.params.input_min), np.asarray(self.params.input_max)),
        ):
            if np.any(values < low - eps) or np.any(values > high + eps):
                return False
        return True
