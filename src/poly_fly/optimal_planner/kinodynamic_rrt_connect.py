"""Kinodynamic RRT-Connect for the cable-suspended payload.

Every edge of the tree is a dynamically feasible trajectory. Because the payload
model is a triple integrator (``Planner.fdot``: 9 states ``[p, v, a]``, 3 jerk
inputs) with payload position as the flat output, the two-point boundary value
problem between two flat states has a *closed form*: a quintic per axis.

The steering mathematics is a numpy port of OMPL's flat state space
(KavrakiLab/ompl-flask, branch ``flat``,
``src/ompl/base/spaces/flat/src/FlatMotion.cpp``), including its minimum-effort
steering rule ``J(T) = Q(T)/T^(2k-1) + rho*T`` with the optimal duration found
by rooting a degree-2k polynomial.

Collision checking reuses the orientation-aware polytopic geometry of the
optimal planner: the quadrotor, cable and payload are three distinct oriented
boxes placed by the flatness map, tested against the axis-aligned obstacle
boxes with an exact separating-axis test.

Running the planner on its own, with no downstream optimization::

    python -m poly_fly.optimal_planner.kinodynamic_rrt_connect --yaml experiments/maze_1.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial import polynomial as npoly

from poly_fly.optimal_planner.global_planner import OpenSetEmptyException

ORDER = 3  # flat output + 2 derivative levels -> 2k = 6 coefficients (quintic)
OUTPUT_DIM = 3  # flat output is the payload position
N_STATES = 9
N_INPUTS = 3
GRAVITY = 9.81

# Below this the flatness map divides by ~zero (a + g*e_z vanishes in free fall).
ACC_NORM_EPS = 1.0

TRAPPED, ADVANCED, REACHED = 0, 1, 2
# The sample fell outside the nearest node's dynamic domain, so nothing was attempted.
# Distinct from TRAPPED so the failure bookkeeping does not blame the node for a
# rejection that never reached it.
REJECTED = 3

# Bound-aware duration search: multiplicative offsets from the minimum-effort duration,
# ordered by how far they move it so the mildest admissible duration is found first. Ratio
# 1.3 keeps consecutive candidates well inside the narrow windows measured on the maze
# suite; going finer costs steering time on the ~85% of boundary value problems that turn
# out to have no admissible duration at any of them.
BOUND_AWARE_DURATION_FACTORS = (1.3, 0.769, 1.69, 0.592, 2.197, 0.455, 2.856, 0.35)

# Ceiling for the batch duration solve. Well past any edge the planner will ever consider
# (the search is bounded by `rrt_timeout` in seconds); it exists only to keep a diverging
# bracket from overflowing T^6.
_BATCH_MAX_DURATION = 1e3

# Bracketing scan for the batch duration solve: geometric offsets from the closed-form
# start, then bisection inside the sign change. The optimum sits inside a factor of two of
# that start on the maze distribution, but the ladder reaches 0.25x-8x so that a boundary
# value problem far from the three single-mechanism regimes still brackets. Eight
# bisections buy 1e-3 relative, which is far more than a ranking needs.
_BATCH_LADDER = 2.0 ** np.arange(-2, 4)
_BATCH_BISECTIONS = 8

# The effort cost Q(T) = integral_0^1 ||jerk(tau)||^2 dtau is a quadratic form
#     Q(T) = sum_{i,j} U[i,j] * <g_i(T), g_j(T)>,     U[i,j] = 1/(i+j+1),
# where g_i is the tau-polynomial for d_{i+3} over the three axes and <.,.> sums over
# axes only. Q is degree 2k-2 = 4 in T, and its coefficients are the (p, r) "power"
# products below.
#
# `_EFFORT_WEIGHT` carries the 1/(i+j+1) weight; the kronecker structure with the
# identity is what keeps the pairing *within* an axis. Without it the weight also
# couples, say, g_3's x-component to g_4's y-component, and Q stops being the integral
# of the squared norm -- measured against a numerical integral, that inflated it by up
# to 1.9x on random BVPs, which silently re-tuned every "minimum-effort" duration.
_EFFORT_ORDER = np.arange(ORDER)
_EFFORT_U = 1.0 / (_EFFORT_ORDER[:, None] + _EFFORT_ORDER[None, :] + 1.0)
_EFFORT_WEIGHT = np.kron(_EFFORT_U, np.eye(OUTPUT_DIM))
_EFFORT_SCALE = (3.0, 12.0, 30.0)  # (6, 24, 60) / 2


# --------------------------------------------------------------------------------------
# exact-integer helpers (ported from the reference; exact in float64 below 2**53)
# --------------------------------------------------------------------------------------
def _binomial(n: int, r: int) -> int:
    if r < 0 or r > n or n < 0:
        return 0
    return math.comb(n, r)


def _falling_factorial(n: int, r: int) -> int:
    """n! / (n - r)! as an exact integer."""
    if r < 0 or r > n:
        return 0
    out = 1
    for i in range(n - r + 1, n + 1):
        out *= i
    return out


def _end_derivatives_inverse(k: int) -> np.ndarray:
    """``M^-1`` where ``M_lj = (k+j)! / (k+j-l)!`` maps far-end shortfalls to coefficients."""
    inv = np.zeros((k, k))
    for j in range(k):
        for l in range(k):
            acc = 0
            for r in range(max(j, l), k):
                acc += _binomial(r, j) * _binomial(k - 1 + r - l, r - l)
            inv[j, l] = (-1) ** (j + l) * acc
    for l in range(k):
        inv[:, l] /= math.factorial(l)
    return inv


def _effort_form(k: int) -> np.ndarray:
    """``W = diag(i!) H^-1 diag(i!)`` with ``H`` the Hilbert matrix; exact integers."""
    form = np.zeros((k, k))
    for l in range(k):
        for m in range(k):
            i, j = k - 1 - l, k - 1 - m
            val = (
                (-1) ** (i + j)
                * math.factorial(i)
                * math.factorial(j)
                * (i + j + 1)
                * _binomial(k + i, k - j - 1)
                * _binomial(k + j, k - i - 1)
                * _binomial(i + j, i) ** 2
            )
            form[l, m] = val
    return form


# --------------------------------------------------------------------------------------
# closed-form flat steering
# --------------------------------------------------------------------------------------
def _derivative_coefficients(coefficients: np.ndarray, level: int) -> np.ndarray:
    """Exact coefficient shift: rows of the ``level``-th derivative of the polynomial."""
    rows = coefficients.shape[0]
    if level >= rows:
        return np.zeros((0, coefficients.shape[1]))
    out = np.zeros((rows - level, coefficients.shape[1]))
    for i in range(level, rows):
        out[i - level, :] = _falling_factorial(i, level) * coefficients[i, :]
    return out


class FlatMotion:
    """A polynomial motion in the flat output, stored in the *original* time variable.

    ``coefficients[i]`` is the coefficient of ``t**i`` (ascending power), one column per
    flat output axis. Callers rely on this: the reference stores the same way, and mixing
    it up with duration-scaled coefficients silently produces a wrong trajectory.
    """

    def __init__(self, coefficients: np.ndarray, duration: float):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim != 2:
            raise ValueError(f"coefficients must be 2-D, got shape {coefficients.shape}")
        if not np.isfinite(coefficients).all():
            raise ValueError("coefficients contain non-finite values")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"duration must be positive and finite, got {duration}")
        self.coefficients = coefficients
        self.duration = float(duration)
        self._peak_speed: Optional[float] = None

    def state_at(self, t: float, level: int = 0) -> np.ndarray:
        """``level``-th time derivative of the flat output at time ``t`` (default: position)."""
        rows = self.coefficients.shape[0]
        out = np.zeros(self.coefficients.shape[1])
        for i in range(rows - 1, level - 1, -1):
            out = out * t + _falling_factorial(i, level) * self.coefficients[i, :]
        return out

    def flat_state(self, t: float) -> np.ndarray:
        """The 9-vector ``[p, v, a]`` at time ``t``."""
        return np.concatenate([self.state_at(t, l) for l in range(ORDER)])

    def samples_at(self, times: np.ndarray, level: int = 0) -> np.ndarray:
        """Vectorized ``state_at`` -> ``(len(times), OUTPUT_DIM)``."""
        times = np.asarray(times, dtype=float)
        rows = self.coefficients.shape[0]
        out = np.zeros((times.shape[0], self.coefficients.shape[1]))
        weights = np.array(
            [_falling_factorial(i, level) for i in range(rows)]
        )  # row i weighted by i!/(i-level)!
        weighted = self.coefficients * weights[:, None]
        for i in range(rows - 1, level - 1, -1):
            out = out * times[:, None] + weighted[i, :]
        return out

    def truncated(self, duration: float) -> "FlatMotion":
        """The prefix of this motion over ``[0, duration]`` (still exactly a polynomial)."""
        if duration > self.duration + 1e-12:
            raise ValueError(f"cannot extend motion to {duration} > {self.duration}")
        return FlatMotion(self.coefficients, min(duration, self.duration))

    def slice(self, start: float, end: float) -> "FlatMotion":
        """The interior piece over ``[start, end]``, re-based so its own time 0 is ``start``.

        A Taylor shift of the coefficients, so the piece is the *same* trajectory seen from
        ``start`` rather than a fresh solution between the two states. Intermediate-state
        chains store one piece per hop, and this is what makes their concatenation
        reproduce the edge the steerer validated instead of re-solving it.
        """
        if not (0.0 <= start < end <= self.duration + 1e-12):
            raise ValueError(f"slice [{start}, {end}] is not inside [0, {self.duration}]")
        end = min(end, self.duration)
        rows = self.coefficients.shape[0]
        shifted = np.zeros_like(self.coefficients)
        for i in range(rows):
            for k in range(i, rows):
                shifted[i] += self.coefficients[k] * math.comb(k, i) * start ** (k - i)
        return FlatMotion(shifted, end - start)

    def derivative(self) -> "FlatMotion":
        coeffs = _derivative_coefficients(self.coefficients, 1)
        return FlatMotion(coeffs, self.duration)

    def cost(self, level: int = ORDER, rho: float = 0.0) -> float:
        """``integral_0^T ||p^(level)||^2 dt + rho*T`` in closed form."""
        T = self.duration
        rows = self.coefficients.shape[0]
        total = 0.0
        for i in range(level, rows):
            wi = _falling_factorial(i, level)
            for j in range(level, rows):
                wj = _falling_factorial(j, level)
                n = i + j - 2 * level + 1
                product = float(self.coefficients[i, :] @ self.coefficients[j, :])
                total += product * wi * wj * T**n / n
        return total + rho * T

    def peak_speed(self) -> float:
        """Max of ``||p'(t)||`` on ``[0, T]``; used only to size the collision sampling.

        Cached: it costs a companion-matrix root solve and is called once per edge.
        """
        if self._peak_speed is not None:
            return self._peak_speed
        self._peak_speed = self._compute_peak_speed()
        return self._peak_speed

    def _compute_peak_speed(self) -> float:
        vel = _derivative_coefficients(self.coefficients, 1)
        rows = vel.shape[0]
        if rows == 0:
            return 0.0

        # q(t) = ||p'(t)||^2: the antidiagonal sums of the coefficient Gram matrix.
        gram = vel @ vel.T
        flipped = gram[:, ::-1]
        q = np.array([flipped.trace(offset=(rows - 1) - j) for j in range(2 * rows - 1)])

        T = self.duration
        descending = q[::-1]
        best = max(float(np.polyval(descending, 0.0)), float(np.polyval(descending, T)))
        try:
            for root in np.roots(np.polyder(descending)):
                if abs(root.imag) <= 1e-9 and 0.0 < root.real < T:
                    best = max(best, float(np.polyval(descending, root.real)))
        except np.linalg.LinAlgError:  # pragma: no cover - companion matrix failure
            best = max(best, self._sampled_peak(q, T))
        return float(np.sqrt(max(best, 0.0)))

    def _sampled_peak(self, q: np.ndarray, T: float, n: int = 64) -> float:
        return float(np.max(npoly.polyval(np.linspace(0.0, T, n), q)))


class FlatSteering:
    """Solves the closed-form flat BVP ``from_ -> to`` over a given duration.

    ``limits`` turns the steering *bound aware*: every motion handed back is then guaranteed
    to respect the state and input limits on the whole interval, or ``steer`` returns
    ``None``. That is a promise the planner relies on -- it is what lets the step-limited
    extension stop re-checking bounds on a truncated prefix.
    """

    def __init__(
        self,
        order: int = ORDER,
        limits: Optional[FlatLimits] = None,
        min_duration: float = 1e-2,
        max_duration: float = 2.0,
    ):
        self.order = order
        self.limits = limits
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self._end_derivatives_inverse = _end_derivatives_inverse(order)
        self._effort_form = _effort_form(order)

    def _check_states(self, from_: np.ndarray, to: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from_ = np.asarray(from_, dtype=float)
        to = np.asarray(to, dtype=float)
        if from_.shape != (self.order, OUTPUT_DIM) or to.shape != (self.order, OUTPUT_DIM):
            raise ValueError(
                f"flat states must have shape ({self.order}, {OUTPUT_DIM}) row-per-derivative, "
                f"got {from_.shape} and {to.shape}"
            )
        if not (np.isfinite(from_).all() and np.isfinite(to).all()):
            raise ValueError("flat states contain non-finite values")
        return from_, to

    def coefficients_over(self, from_: np.ndarray, to: np.ndarray, duration: float) -> np.ndarray:
        """Quintic coefficients ``(2k, OUTPUT_DIM)`` in ``t``.

        Row ``i`` is the coefficient of ``t**i``; the motion matches ``from_`` at ``0`` and
        ``to`` at ``T``.

        Scalar arithmetic on plain lists, converted to an array once at the end. The
        operands are 3-vectors and numpy's per-call overhead was most of the ~120us this
        used to cost -- same trap as the bounds check, and this one is on the steering hot
        path, so the bound-aware duration search pays it several times per edge.
        """
        from_, to = self._check_states(from_, to)
        T = float(duration)
        if not math.isfinite(T) or T <= 0.0:
            raise ValueError(f"duration must be positive and finite, got {duration}")

        k = self.order
        near = from_.tolist()
        far = to.tolist()
        coefficients = [[0.0] * OUTPUT_DIM for _ in range(2 * k)]

        # Near end: C_i = y_0^(i) / i!
        for i in range(k):
            inverse_factorial = 1.0 / math.factorial(i)
            coefficients[i] = [inverse_factorial * value for value in near[i]]

        # Shortfall: coefficient of T^p in  T^l * y_f^(l) - sum_{i>=l} y_0^(i) * T^i / (i-l)!
        # The far-end term is a monomial of degree l, so it lands at p == l; entries with
        # p < l vanish. Horner in p folds in T^(p-l), scaled by T^l at the end.
        shortfall = []
        for l in range(k):
            value = [0.0] * OUTPUT_DIM
            for p in range(k - 1, l - 1, -1):
                inverse_factorial = 1.0 / math.factorial(p - l)
                if p == l:
                    value = [
                        value[axis] * T - inverse_factorial * near[p][axis] + far[l][axis]
                        for axis in range(OUTPUT_DIM)
                    ]
                else:
                    value = [
                        value[axis] * T - inverse_factorial * near[p][axis]
                        for axis in range(OUTPUT_DIM)
                    ]
            scale = T**l
            shortfall.append([scale * item for item in value])

        # e = M^-1 s ; C_{k+a} = e_a / T^(k+a)
        inverse = self._end_derivatives_inverse.tolist()
        for a in range(k):
            inverse_power = 1.0 / T ** (k + a)
            coefficients[k + a] = [
                inverse_power * sum(inverse[a][j] * shortfall[j][axis] for j in range(k))
                for axis in range(OUTPUT_DIM)
            ]

        return np.asarray(coefficients, dtype=float)

    def _duration(self, from_: np.ndarray, to: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    @staticmethod
    def _as_rows(state: np.ndarray) -> np.ndarray:
        """``[p, v, a]`` laid out as row-per-derivative ``(order, OUTPUT_DIM)``.

        The 9-vector flat state and the matrix form are the same memory, so this is a
        reshape -- but the distinction matters: everything below `steer` speaks the matrix
        form (matching the reference), while the planner speaks the 9-vector.
        """
        return np.asarray(state, dtype=float).reshape(ORDER, OUTPUT_DIM)

    def _candidate_durations(self, from_: np.ndarray, to: np.ndarray) -> Sequence[float]:
        """Durations to try, in preference order. The first admissible one wins."""
        duration = self._duration(from_, to)
        return () if duration is None else (duration,)

    def steer(self, from_state: np.ndarray, to_state: np.ndarray) -> Optional[FlatMotion]:
        """Steer between two 9-vector flat states; ``None`` when no motion is possible."""
        from_, to = self._as_rows(from_state), self._as_rows(to_state)
        if np.array_equal(from_, to):
            return None  # identical states: no motion exists between them
        for duration in self._candidate_durations(from_, to):
            motion = FlatMotion(self.coefficients_over(from_, to, duration), duration)
            if self.limits is None:
                return motion
            if (
                _bound_violation(
                    motion, self.limits, motion.coefficients.tolist(), early_exit=True
                )
                <= 0.0
            ):
                return motion
        return None


class FixedDurationSteering(FlatSteering):
    """Steers any state pair over the same fixed duration."""

    def __init__(self, duration: float, order: int = ORDER, limits: Optional[FlatLimits] = None):
        super().__init__(order, limits)
        if duration <= 0.0:
            raise ValueError("fixed duration must be positive")
        self.duration = float(duration)

    def _duration(self, from_: np.ndarray, to: np.ndarray) -> Optional[float]:
        return self.duration


def _effort_q_coefficients(froms: np.ndarray, tos: np.ndarray) -> np.ndarray:
    """Ascending coefficients of ``Q(T)`` for a batch of boundary value problems.

    ``froms`` and ``tos`` are both ``(N, 3, 3)`` row-per-derivative states. Every row is an
    independent problem, which is what lets the search rank by the duration of the edge it
    would actually fly (node -> target) rather than by the reverse -- the duration is not
    symmetric, and on a state moving toward the target the two can differ by a factor.
    Returns ``(N, 2k)``, with the coefficients above degree ``2k-2`` zero.
    """
    p0, v0, a0 = froms[:, 0], froms[:, 1], froms[:, 2]
    p1, v1, a1 = tos[:, 0], tos[:, 1], tos[:, 2]
    dp = p1 - p0

    # A[i][p] is the (N, 3) axis vector multiplying T^p in the numerator of d_{i+3},
    # carrying the 6/24/60 that turns d into g. Kept as separate contiguous arrays: the
    # same numbers inside one (N, 9, 3) block are reachable only through strided slices,
    # and on this workload a strided write costs more than the arithmetic it feeds.
    blocks = (
        (
            _EFFORT_SCALE[0] * 20.0 * dp,
            _EFFORT_SCALE[0] * -(8.0 * v1 + 12.0 * v0),
            _EFFORT_SCALE[0] * -(3.0 * a0 - a1),
        ),
        (
            _EFFORT_SCALE[1] * -30.0 * dp,
            _EFFORT_SCALE[1] * (14.0 * v1 + 16.0 * v0),
            _EFFORT_SCALE[1] * (3.0 * a0 - 2.0 * a1),
        ),
        (
            _EFFORT_SCALE[2] * 12.0 * dp,
            _EFFORT_SCALE[2] * -(6.0 * v1 + 6.0 * v0),
            _EFFORT_SCALE[2] * (a1 - a0),
        ),
    )
    # d[p] is (N, 9) over alpha = block * 3 + axis, matching `_EFFORT_WEIGHT`.
    d = [
        np.concatenate([blocks[0][p], blocks[1][p], blocks[2][p]], axis=1)
        for p in range(ORDER)
    ]
    # q_m = sum_{p + r = m} d[p] . (_EFFORT_WEIGHT @ d[r]), the axis dot falling out of
    # the sum over alpha. Only the upper triangle of (p, r) is walked: the form is
    # symmetric, so the mirrored pairs contribute an equal amount -- which on this
    # workload saves three of the nine reductions.
    weighted = [block @ _EFFORT_WEIGHT.T for block in d]
    parts: List[Optional[np.ndarray]] = [None] * (2 * ORDER - 1)
    for p in range(ORDER):
        for r in range(p, ORDER):
            term = (d[p] * weighted[r]).sum(axis=1)
            if r != p:
                term = term + term
            m = p + r
            parts[m] = term if parts[m] is None else parts[m] + term
    return np.stack(parts, axis=1)


def _batch_q_coefficients(from_rows: np.ndarray, tos: np.ndarray) -> np.ndarray:
    """``_effort_q_coefficients`` with one fixed origin and many targets.

    ``from_rows`` is a single flat state in row-per-derivative form, ``tos`` an ``(N, 9)``
    array of flat states.
    """
    tos = np.asarray(tos, dtype=float).reshape(-1, N_STATES)
    froms = np.broadcast_to(
        np.asarray(from_rows, dtype=float), (tos.shape[0], ORDER, OUTPUT_DIM)
    )
    return _effort_q_coefficients(froms, tos.reshape(-1, ORDER, OUTPUT_DIM))


def _duration_start_guess(froms: np.ndarray, tos: np.ndarray, rho: float, floor: float):
    """Closed-form optima of the three single-mechanism problems, as a bracket centre.

    The position term dominates whenever there is ground to cover; the other two exist so
    that the start does not collapse to zero on a pure velocity or acceleration change,
    where the position formula degenerates.
    """
    dp = tos[:, 0] - froms[:, 0]
    dv = tos[:, 1] - froms[:, 1]
    da = tos[:, 2] - froms[:, 2]
    return np.maximum.reduce(
        (
            (3600.0 * (dp * dp).sum(axis=1) / rho) ** (1.0 / 6.0),
            (576.0 * (dv * dv).sum(axis=1) / rho) ** (1.0 / 5.0),
            np.sqrt(6.0 * (da * da).sum(axis=1) / rho),
            np.full(froms.shape[0], floor),
        )
    )


def _horner_batch_2d(coefficients: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Horner for one polynomial per row against many ``t``: (N, deg+1) x (N, M) -> (N, M)."""
    total = coefficients[:, -1][:, None] * np.ones_like(t)
    for index in range(coefficients.shape[1] - 2, -1, -1):
        total = total * t + coefficients[:, index][:, None]
    return total


def _first_positive_root(slope: np.ndarray, guess: np.ndarray, lower: float) -> np.ndarray:
    """Smallest positive root of each row of ``slope``, by bracketing then bisection.

    Newton is deliberately not used. ``S`` is not monotone, and on a sizeable minority of
    boundary value problems the closed-form start lands where ``S`` is *decreasing*, from
    which Newton walks away from the root rather than toward it: measured over 400 random
    pairs it missed by more than 1% on 11.5% of them, and more iterations made it worse
    (the damping crawled to the floor) rather than better.

    The smallest root is the one wanted -- ``J`` diverges at both ends, and where a second
    local minimum exists at all it prices at over 100x the first on this state
    distribution -- so the scan looks for the *first* sign change from below zero, which
    bisection then narrows.
    """
    rows = np.arange(slope.shape[0])
    ladder = np.clip(guess[:, None] * _BATCH_LADDER[None, :], lower, _BATCH_MAX_DURATION)
    values = _horner_batch_2d(slope, ladder)
    above = values > 0.0
    index = np.argmax(above, axis=1)
    found = above[rows, index]
    high = np.where(found, ladder[rows, index], ladder[:, -1])
    low = np.where(
        found & (index > 0), ladder[rows, np.maximum(index - 1, 0)], np.full(len(rows), lower)
    )
    for _ in range(_BATCH_BISECTIONS):
        middle = 0.5 * (low + high)
        below = _horner_batch(slope, middle) <= 0.0
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    # A row whose ladder never turns positive keeps the top of the ladder: it is a
    # boundary value problem whose optimum lies beyond 16x the closed-form start, which in
    # practice means a nearly degenerate one. Erring large only makes it rank as far away.
    return np.where(found, 0.5 * (low + high), ladder[:, -1])


def _horner_batch(coefficients: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Horner evaluation of one polynomial per row: ``coefficients`` is (N, degree+1)."""
    total = coefficients[:, -1] * np.ones_like(t)
    for index in range(coefficients.shape[1] - 2, -1, -1):
        total = total * t + coefficients[:, index]
    return total


class MinimumEffortSteering(FlatSteering):
    """Picks the duration minimising ``J(T) = Q(T)/T^(2k-1) + rho*T``.

    ``Q(T) = integral_0^1 ||d^3 p / d tau^3||^2 d tau`` with ``tau = t/T`` is a polynomial
    in ``T`` of degree ``<= 2k-2``; ``S(T) = T^(2k) * dJ/dT`` is the degree-``2k`` polynomial
    ``S[m] = (m + 1 - 2k) * Q[m]`` for ``m < 2k-1`` and ``S[2k] = rho``. Pricing every
    positive real root of ``S`` is equivalent to the reference's "price every upward
    crossing": a downward crossing is a local maximum, and ``J -> inf`` at both ends.
    """

    def __init__(
        self,
        rho: float = 30.0,
        order: int = ORDER,
        limits: Optional[FlatLimits] = None,
        min_duration: float = 1e-2,
        max_duration: float = 2.0,
    ):
        super().__init__(order, limits, min_duration, max_duration)
        if rho <= 0.0:
            raise ValueError("rho must be positive, otherwise no finite duration is optimal")
        self.rho = float(rho)

    def effort_numerator(self, from_: np.ndarray, to: np.ndarray) -> np.ndarray:
        """Ascending coefficients of ``Q(T)``, length ``2k``.

        Per axis, the top-k coefficients of the tau-normalized polynomial as polynomials
        in T. For k == 3 (verified against a brute-force scan) these are
          d3 = ( 20dp - (8v1 + 12v0)T - (3a0 -  a1)T^2 ) / 2
          d4 = (-30dp + (14v1 + 16v0)T + (3a0 - 2a1)T^2 ) / 2
          d5 = ( 12dp -   6(v1 + v0)T + (a1 -  a0)T^2 ) / 2
        """
        from_, to = self._check_states(from_, to)
        return _batch_q_coefficients(from_, to.reshape(1, N_STATES))[0]

    def optimal_duration(self, from_: np.ndarray, to: np.ndarray) -> Optional[Tuple[float, float]]:
        """Returns ``(T*, J(T*))`` or ``None`` when no finite optimum exists."""
        from_, to = self._check_states(from_, to)
        if np.array_equal(from_, to):
            return None
        q = self.effort_numerator(from_, to)
        if np.max(np.abs(q)) < 1e-14:
            return None

        k = self.order
        slope = np.zeros(2 * k + 1)
        for m in range(2 * k - 1):
            slope[m] = (m + 1 - 2 * k) * q[m]
        slope[2 * k] = self.rho

        def cost(T: float) -> float:
            return float(npoly.polyval(T, q) / T ** (2 * k - 1) + self.rho * T)

        try:
            roots = npoly.polyroots(slope)
        except np.linalg.LinAlgError:  # pragma: no cover - companion matrix failure
            return None

        best: Optional[Tuple[float, float]] = None
        for root in roots:
            if abs(root.imag) > 1e-9 * (1.0 + abs(root.real)):
                continue
            T = float(root.real)
            if T <= 0.0 or not math.isfinite(T):
                continue
            c = cost(T)
            if not math.isfinite(c):
                continue
            if best is None or c < best[1]:
                best = (T, c)
        return best

    def optimal_duration_batch(self, from_: np.ndarray, tos: np.ndarray) -> np.ndarray:
        """Minimum-effort durations for many targets at once.

        ``S(T) = T*Q'(T) - (2k-1)*Q(T) + rho*T^(2k)`` is the numerator of ``J'``, so its
        smallest positive root is the optimal duration. The bracket comes from the
        closed-form optima of the three single-mechanism problems; see
        ``_first_positive_root`` for why the root is bracketed rather than Newton-solved.

        Deliberately *not* clipped to ``max_duration``: the effort optimum on this hardware
        sits around 2.7 s, above the configured 2.0 s, so clamping would saturate most of
        the ranking at the ceiling and destroy the ordering this exists to provide.

        This feeds a nearest-neighbour ranking, not a trajectory, so the exact root solver
        stays the one that shapes edges. The sanity suite measures the agreement.
        """
        tos = np.asarray(tos, dtype=float).reshape(-1, N_STATES)
        n = tos.shape[0]
        froms = np.broadcast_to(np.asarray(from_, dtype=float), (n, ORDER, OUTPUT_DIM))
        return self.durations_between(froms, tos.reshape(n, ORDER, OUTPUT_DIM))

    def durations_between(self, froms: np.ndarray, tos: np.ndarray) -> np.ndarray:
        """Optimal durations for a batch of independent boundary value problems.

        ``froms`` and ``tos`` are both ``(N, 3, 3)`` row-per-derivative states, so the
        origin may vary per row -- which is what the nearest-neighbour scan needs.
        """
        k = self.order
        n = froms.shape[0]
        q = _effort_q_coefficients(froms, tos)
        slope = np.zeros((n, 2 * k + 1))
        for m in range(2 * k - 1):
            slope[:, m] = (m + 1 - 2 * k) * q[:, m]
        slope[:, 2 * k] = self.rho
        guess = _duration_start_guess(froms, tos, self.rho, self.min_duration)
        return _first_positive_root(slope, guess, self.min_duration)

    def _duration(self, from_: np.ndarray, to: np.ndarray) -> Optional[float]:
        optimum = self.optimal_duration(from_, to)
        if optimum is None:
            return None
        return optimum[0]

    def _candidate_durations(self, from_: np.ndarray, to: np.ndarray) -> Sequence[float]:
        """The effort optimum, then geometrically outward while it stays inadmissible.

        ``J(T)`` prices effort against time and knows nothing about the limits: a
        minimum-effort quintic *starts* with non-zero jerk, and nothing in the cost objects
        to the trajectory leaving the box on the way. When the optimum turns out to be
        inadmissible the durations that are admissible form a narrow window around it
        (measured on the maze suite: a median of 0.18 decades wide), so the search walks
        outward in small multiplicative steps rather than jumping or bisecting -- both of
        which overshoot a window this narrow.
        """
        optimum = self.optimal_duration(from_, to)
        if optimum is None:
            return ()
        duration = optimum[0]
        if self.limits is None:
            return (duration,)

        # Deliberately not bounded by `max_duration`: on the maze suite the effort optimum
        # already sits above the configured 2.0 s, so clamping to it would silently disable
        # every upward step of the search -- which is the direction that does most of the
        # rescuing. `min_duration` is honoured because 1/T^5 in the coefficients blows up.
        candidates = [duration]
        for factor in BOUND_AWARE_DURATION_FACTORS:
            candidate = max(duration * factor, self.min_duration)
            if all(abs(candidate - seen) > 1e-12 * max(1.0, seen) for seen in candidates):
                candidates.append(candidate)
        return candidates


# --------------------------------------------------------------------------------------
# orientation-aware collision checking
# --------------------------------------------------------------------------------------
def _body_boxes_batch(
    params, positions: np.ndarray, accelerations: np.ndarray, use_robot_rotation: bool
):
    """Oriented boxes for payload, cable and quadrotor at ``N`` samples.

    Poses match the OCP's own collision geometry (``add_cable_to_obstacles_constraints`` /
    ``add_quadrotor_to_obstacles_constraints``), with every quantity computed for all
    samples at once: this runs inside the RRT's innermost loop, where per-sample numpy
    calls on 3-vectors dominate everything else.

    Returns three ``(centers, rotations, half_extents)`` triples, in that body order.
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
    half_quad = np.array(
        [params.robot_radius, params.robot_radius, params.robot_height / 2.0]
    )
    return [
        (positions, rotation_cable, half_payload),
        (positions - 0.5 * params.cable_length * p_hat, rotation_cable, half_cable),
        (
            positions - (params.cable_length + params.robot_height / 2.0) * p_hat,
            rotation_quad,
            half_quad,
        ),
    ]


def _boxes_hit_obstacles(
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
    """

    def __init__(self, params, mobility: "object", margin: Optional[float] = None):
        self.params = params
        self._mobility = mobility  # provides the cable/quadrotor rotation conventions

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

    def sample_count(self, motion: FlatMotion) -> int:
        span = motion.duration * motion.peak_speed()
        return max(2, int(math.ceil(span / self.step)))

    # -- validity ----------------------------------------------------------------------
    def is_state_valid(
        self, state: np.ndarray, motion: Optional[FlatMotion] = None, t: float = 0.0
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

        accelerations = state[6:9].reshape(1, 3)
        if float(np.linalg.norm(accelerations[0] + np.array([0.0, 0.0, GRAVITY]))) < ACC_NORM_EPS:
            return False  # flatness map is singular here
        boxes = _body_boxes_batch(
            self.params,
            state[:3].reshape(1, 3),
            accelerations,
            getattr(self.params, "use_robot_rotation", True),
        )
        checks = [True, self.params.rrt_check_cable, self.params.rrt_check_quadrotor]
        for (centers, rotations, half), enabled in zip(boxes, checks):
            if not enabled:
                continue
            if not (np.isfinite(centers).all() and np.isfinite(rotations).all()):
                return False
            if bool(
                _boxes_hit_obstacles(
                    centers, rotations, half, self.obstacle_centers, self.obstacle_half_extents
                )[0]
            ):
                return False
        return True

    def is_motion_valid(self, motion: FlatMotion, count: Optional[int] = None) -> bool:
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

        boxes = _body_boxes_batch(
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
            if _boxes_hit_obstacles(
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


def _horner(coefficients: Sequence[float], t: float) -> float:
    """Scalar Horner evaluation, ascending coefficients.

    Deliberately not ``np.polyval``: these polynomials hold six floats and numpy's per-call
    overhead is ~25us, which dwarfed the arithmetic in the planner's hottest loop.
    """
    total = 0.0
    for c in reversed(coefficients):
        total = total * t + c
    return total


def _cbrt(value: float) -> float:
    return math.copysign(abs(value) ** (1.0 / 3.0), value)


def _quadratic_roots(a: float, b: float, c: float, out: List[float]) -> List[float]:
    if a == 0.0:
        if b != 0.0:
            out.append(-c / b)
        return out
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        # A discriminant that is negative only by roundoff is a double root, and that case
        # is common here: a symmetric payload manoeuvre squares the quartic exactly, so the
        # pair arrives as -1e-17 rather than 0. Clamping is safe for an extrema test -- the
        # extra evaluation point can only widen the interval, never narrow it.
        if disc < -1e-12 * (b * b + abs(4.0 * a * c)):
            return out
        disc = 0.0
    root_disc = math.sqrt(disc)
    # Cancellation-free form: form the root of larger magnitude first and get the other
    # from the product c/a, rather than differencing two nearly equal numbers.
    q = -0.5 * (b + math.copysign(root_disc, b))
    if q == 0.0:
        out.append(0.0)
        return out
    out.append(q / a)
    out.append(c / q)
    return out


def _cubic_roots(a: float, b: float, c: float, d: float, out: List[float]) -> List[float]:
    if a == 0.0:
        return _quadratic_roots(b, c, d, out)
    p, q, r = b / a, c / a, d / a
    shift = p / 3.0
    big_p = q - p * shift
    big_q = 2.0 * shift**3 - p * q / 3.0 + r
    half = big_q / 2.0
    disc = half * half + (big_p / 3.0) ** 3
    if disc > 0.0:
        root_disc = math.sqrt(disc)
        out.append(_cbrt(-half + root_disc) + _cbrt(-half - root_disc) - shift)
        return out
    if disc == 0.0:
        u = _cbrt(-half)
        out.append(2.0 * u - shift)
        out.append(-u - shift)
        return out
    # Three distinct real roots. Cardano's formula loses most of its precision here, so use
    # the trigonometric form, which is well conditioned for this branch.
    magnitude = 2.0 * math.sqrt(-big_p / 3.0)
    cosine = max(-1.0, min(1.0, 3.0 * big_q / (big_p * magnitude)))
    theta = math.acos(cosine) / 3.0
    for k in range(3):
        out.append(magnitude * math.cos(theta - 2.0 * math.pi * k / 3.0) - shift)
    return out


def _quartic_roots(a: float, b: float, c: float, d: float, e: float, out: List[float]):
    """Ferrari, via the resolvent cubic. Yields real roots only."""
    if a == 0.0:
        return _cubic_roots(b, c, d, e, out)
    p, q, r, s = b / a, c / a, d / a, e / a
    shift = p / 4.0
    big_p = q - 6.0 * shift**2
    big_q = r - p * q / 2.0 + p**3 / 8.0
    big_r = s - p * r / 4.0 + p * p * q / 16.0 - 3.0 * p**4 / 256.0

    if big_q == 0.0:  # biquadratic: y^4 + P y^2 + R = 0
        for t in _quadratic_roots(1.0, big_p, big_r, []):
            if t >= 0.0:
                root = math.sqrt(t)
                out.append(root - shift)
                out.append(-root - shift)
        return out

    # Resolvent, from requiring (z - P)(z^2 - 4R) = Q^2 so the quartic splits into two
    # quadratics y^2 +/- w y + (z -/+ Q/w)/2 with w = sqrt(z - P).
    resolvent: List[float] = []
    _cubic_roots(1.0, -big_p, -4.0 * big_r, 4.0 * big_r * big_p - big_q * big_q, resolvent)
    z = max(resolvent)
    if z <= big_p:  # roundoff at the Q -> 0 boundary
        z = big_p + 1e-12 * (1.0 + abs(big_p))
    w = math.sqrt(z - big_p)
    offsets: List[float] = []
    _quadratic_roots(1.0, w, (z - big_q / w) / 2.0, offsets)
    _quadratic_roots(1.0, -w, (z + big_q / w) / 2.0, offsets)
    for root in offsets:
        out.append(root - shift)
    return out


def _numpy_real_roots(coefficients: Sequence[float]) -> List[float]:
    return [
        float(root.real)
        for root in np.roots(np.asarray(coefficients, dtype=float)[::-1])
        # Relative, so a double root -- which the companion matrix delivers as a conjugate
        # pair with a small imaginary part -- is not mistaken for a genuinely complex one.
        if abs(root.imag) <= 1e-9 * max(1.0, abs(root.real))
    ]


def _real_roots(coefficients: Sequence[float]) -> List[float]:
    """All real roots of an ascending-coefficient polynomial, closed form up to degree 4.

    The residual check is the safety net for the closed forms: a degenerate coefficient
    pattern that slips past the algebra produces a root that does not actually solve the
    polynomial, and the whole polynomial is then handed to ``np.roots`` instead of silently
    dropping a critical point (which would make an extrema test under-estimate and accept a
    trajectory that leaves the bounds).
    """
    degree = len(coefficients) - 1
    if degree < 1:
        return []
    roots: List[float] = []
    if degree == 1:
        roots.append(-coefficients[0] / coefficients[1])
    elif degree == 2:
        _quadratic_roots(coefficients[2], coefficients[1], coefficients[0], roots)
    elif degree == 3:
        _cubic_roots(coefficients[3], coefficients[2], coefficients[1], coefficients[0], roots)
    elif degree == 4:
        _quartic_roots(*reversed(coefficients), roots)
    else:
        return _numpy_real_roots(coefficients)

    for root in roots:
        magnitude = max(1.0, abs(root))
        # The natural scale of the evaluation at this root, so the test is relative to the
        # terms that actually cancel rather than to the largest coefficient alone.
        scale = sum(abs(c) * magnitude**i for i, c in enumerate(coefficients))
        if abs(_horner(coefficients, root)) > 1e-9 * scale:
            return _numpy_real_roots(coefficients)
    return roots


def _interior_critical_points(derivative: Sequence[float], duration: float) -> List[float]:
    """Real roots of an ascending-coefficient polynomial strictly inside ``(0, duration)``."""
    d = [float(value) for value in derivative]
    while len(d) > 1 and d[-1] == 0.0:
        d.pop()
    if len(d) <= 1:
        return []  # constant: no interior critical point
    return [root for root in _real_roots(d) if 0.0 < root < duration]


def poly_extrema(coefficients: Sequence[float], duration: float) -> Tuple[float, float]:
    """Exact min and max of ``sum coefficients[i] t^i`` on ``[0, duration]``."""
    c = [float(value) for value in np.asarray(coefficients, dtype=float).ravel()]
    while len(c) > 1 and c[-1] == 0.0:
        c.pop()
    if not c:
        return 0.0, 0.0
    if len(c) == 1:
        return c[0], c[0]

    values = [_horner(c, 0.0), _horner(c, duration)]
    derivative = [i * c[i] for i in range(1, len(c))]
    for root in _interior_critical_points(derivative, duration):
        values.append(_horner(c, root))
    return min(values), max(values)


@dataclass(frozen=True)
class FlatLimits:
    """The pointwise state and input limits a flat motion has to respect."""

    state_min: Tuple[float, ...]
    state_max: Tuple[float, ...]
    input_min: Tuple[float, ...]
    input_max: Tuple[float, ...]

    @classmethod
    def from_params(cls, params) -> "FlatLimits":
        return cls(
            state_min=tuple(float(v) for v in params.state_min),
            state_max=tuple(float(v) for v in params.state_max),
            input_min=tuple(float(v) for v in params.input_min),
            input_max=tuple(float(v) for v in params.input_max),
        )


def _bound_violation(
    motion: FlatMotion,
    limits: FlatLimits,
    coefficients: Optional[Sequence[Sequence[float]]] = None,
    early_exit: bool = False,
) -> float:
    """How far the motion exceeds its limits, as a fraction of each limit's own span.

    ``0.0`` means admissible. Reported as a continuous score rather than a flag so the
    magnitude says how much the duration would have to change, normalised by each limit's
    own span to make position, velocity, acceleration and jerk comparable.

    ``early_exit`` stops at the first constraint that is broken, which is what the duration
    search wants: it only needs the verdict, and over half of sampled motions break a
    position bound, so the remaining levels are never reached anyway.

    Derivative level ``l`` occupies state rows ``3l : 3l+3`` (position, velocity,
    acceleration); jerk is the control input and uses the input bounds instead.
    """
    eps = 1e-9
    rows = motion.coefficients if coefficients is None else coefficients
    n_rows = len(rows)
    worst = 0.0
    for level in range(ORDER):
        low_limit = limits.state_min[3 * level : 3 * level + 3]
        high_limit = limits.state_max[3 * level : 3 * level + 3]
        derivative_weights = [_falling_factorial(i, level) for i in range(level, n_rows)]
        for axis in range(OUTPUT_DIM):
            span = high_limit[axis] - low_limit[axis]
            column = [w * rows[level + j][axis] for j, w in enumerate(derivative_weights)]
            low, high = poly_extrema(column, motion.duration)
            # eps *widens* the admissible range: a state sitting exactly on a bound is
            # legitimate, and the start state commonly does (maze_1 launches at z = 0).
            worst = max(
                worst,
                (low_limit[axis] - eps - low) / span,
                (high - high_limit[axis] - eps) / span,
            )
            if early_exit and worst > 0.0:
                return worst
    jerk_weights = [_falling_factorial(i, ORDER) for i in range(ORDER, n_rows)]
    for axis in range(OUTPUT_DIM):
        span = limits.input_max[axis] - limits.input_min[axis]
        column = [w * rows[ORDER + j][axis] for j, w in enumerate(jerk_weights)]
        low, high = poly_extrema(column, motion.duration)
        worst = max(
            worst,
            (limits.input_min[axis] - eps - low) / span,
            (high - limits.input_max[axis] - eps) / span,
        )
        if early_exit and worst > 0.0:
            return worst
    return worst


def is_motion_within_bounds(motion: FlatMotion, params) -> bool:
    """Exact bounds test over the whole motion, by polynomial extrema rather than sampling."""
    return (
        _bound_violation(motion, FlatLimits.from_params(params), motion.coefficients.tolist())
        <= 0.0
    )


# --------------------------------------------------------------------------------------
# RRT-Connect
# --------------------------------------------------------------------------------------
class _WeightedEuclidean:
    """Position in metres, velocity and acceleration scaled by 1/omega.

    ``omega`` defaults to the payload pendulum's natural frequency ``sqrt(g/L)``, which
    prices a velocity error the way it shows up as a position error at that frequency.
    """

    epsilon = 1e-9

    def __init__(self, omega: float):
        level_weights = (1.0, 1.0 / omega, 1.0 / omega**2)
        self.weights = np.array([w for w in level_weights for _ in range(OUTPUT_DIM)])

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        delta = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) * self.weights
        return math.sqrt(float(delta @ delta))

    def nearest(self, states: np.ndarray, target: np.ndarray) -> int:
        delta = (states - target) * self.weights
        return int(np.argmin((delta * delta).sum(axis=1)))


class _OptimalDuration:
    """The minimum-effort duration of the edge from a node to the target.

    Steering toward a state that is far away in metres but already moving the right way
    comes out cheap, and that is precisely the information a fixed-weight Euclidean
    distance discards. Not a metric in the strict sense -- it is asymmetric and does not
    satisfy the triangle inequality -- but nearest-neighbour search only needs a
    consistent ordering.

    Every node in the tree needs its own root solve, so this costs one to two orders of
    magnitude more per query than the Euclidean form. That is the price of the ordering,
    and it is why the two are selectable rather than one replacing the other.
    """

    def __init__(self, steering: "MinimumEffortSteering"):
        self.steering = steering
        # The batch solve cannot report anything below its own floor, so the "these are the
        # same state" threshold has to sit just above it rather than at the Euclidean
        # scale -- otherwise a coincident pair reads as a small-but-real 10 ms edge and the
        # coincidence guards never fire.
        self.epsilon = 1.5 * float(steering.min_duration)

    @staticmethod
    def _rows(state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float).reshape(3, OUTPUT_DIM)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        target = np.asarray(b, dtype=float).reshape(1, N_STATES)
        return float(self.steering.optimal_duration_batch(self._rows(a), target)[0])

    def nearest(self, states: np.ndarray, target: np.ndarray) -> int:
        # node -> target, the direction the edge would actually be flown, matching
        # `distance`. Ranking by the reverse is cheaper -- it batches the tree as the list
        # of targets for one fixed origin -- but it is a different ordering, because the
        # duration is not symmetric: a node already moving toward the target is quick to
        # reach it and slow to be reached from it.
        count = states.shape[0]
        froms = states.reshape(count, ORDER, OUTPUT_DIM)
        targets = np.broadcast_to(
            np.asarray(target, dtype=float).reshape(ORDER, OUTPUT_DIM),
            (count, ORDER, OUTPUT_DIM),
        )
        durations = self.steering.durations_between(froms, targets)
        return int(np.argmin(durations))


class _Tree:
    def __init__(self, root: np.ndarray):
        self.states = np.asarray(root, dtype=float).reshape(1, N_STATES)
        self.parents: List[int] = [-1]
        self.motions: List[Optional[FlatMotion]] = [None]
        # Dynamic-domain radius per node, in the units of the active metric. Infinite means
        # "no restriction": the roots keep it forever, and so does every node that has not
        # yet had an extension refused.
        self.radii: List[float] = [math.inf]

    def __len__(self) -> int:
        return self.states.shape[0]

    @property
    def last(self) -> int:
        return self.states.shape[0] - 1

    def add(self, state: np.ndarray, parent: int, motion: FlatMotion) -> int:
        self.states = np.vstack([self.states, np.asarray(state, dtype=float).reshape(1, N_STATES)])
        self.parents.append(parent)
        self.motions.append(motion)
        self.radii.append(math.inf)
        return self.last

    def nearest(self, target: np.ndarray, metric) -> int:
        return metric.nearest(self.states, target)


@dataclass
class FlatPath:
    """A concatenated sequence of flat motions."""

    states: List[np.ndarray] = field(default_factory=list)  # M + 1 flat states
    motions: List[FlatMotion] = field(default_factory=list)  # M motions

    def duration(self) -> float:
        return float(sum(m.duration for m in self.motions))

    def _locate(self, t: float) -> Tuple[int, float]:
        if not self.motions:
            return 0, 0.0
        remaining = max(float(t), 0.0)
        for idx, motion in enumerate(self.motions):
            if remaining <= motion.duration or idx == len(self.motions) - 1:
                return idx, min(remaining, motion.duration)
            remaining -= motion.duration
        return len(self.motions) - 1, self.motions[-1].duration

    def state_at(self, t: float, level: int = 0) -> np.ndarray:
        idx, local = self._locate(t)
        return self.motions[idx].state_at(local, level)

    def samples_at(self, times: Sequence[float], level: int = 0) -> np.ndarray:
        return np.vstack([self.state_at(t, level) for t in times])


@dataclass
class RRTResult:
    path: Optional[FlatPath]
    explored_nodes: List[List[float]]
    explored_nodes_3d: List[List[float]]
    iterations: int
    duration: float
    success: bool
    seed: int
    # The search trees, handed over for `plot_trees`. None only when planning never started
    # because an endpoint state was invalid.
    start_tree: Optional[_Tree] = None
    goal_tree: Optional[_Tree] = None


class KinodynamicRRTConnect:
    """RRT-Connect over flat states, using closed-form polynomial steering."""

    def __init__(
        self,
        params,
        steering: FlatSteering,
        mobility: "object",
        rng: Optional[np.random.Generator] = None,
    ):
        self.params = params
        self.steering = steering
        self.mobility = mobility
        self.rng = rng if rng is not None else np.random.default_rng(params.rrt_seed)
        self.checker = FlatEdgeChecker(params, mobility)

        self.metric = make_metric(params, steering)
        # Extension step, in seconds of flight (see `_steer`). Intermediate states subdivide
        # each accepted extension at this spacing; 0 keeps one node per extension.
        self.max_step = float(params.rrt_max_step)
        self.intermediate_dt = float(getattr(params, "rrt_intermediate_dt", 0.0))
        self.retry_steps = max(1, int(params.rrt_retry_steps))
        # A bound-aware steerer has already rejected anything that leaves the limits, so
        # re-testing every accepted edge would just redo its work.
        self.bounds_guaranteed = getattr(steering, "limits", None) is not None
        self.metric_eps = self.metric.epsilon

        # Which tree takes the random sample each iteration. "larger" (the pre-existing
        # behaviour, and by measurement much the strongest of the three on this planner:
        # on the sanity suite's wall scenario it solves in 377 nodes where the reference's
        # ratio rule needs 1998 and strict alternation does not solve at all) grows the
        # bigger tree toward the sample and leaves the greedy CONNECT walk to the smaller
        # one. "ratio" is VAMP-RRTC's balance rule, "alternate" is a swap every iteration.
        self.tree_selection = str(getattr(params, "rrt_tree_selection", "larger")).lower()
        self.tree_ratio = float(getattr(params, "rrt_tree_ratio", 1.0))
        self.dynamic_domain = bool(getattr(params, "rrt_dynamic_domain", False))
        self.dd_alpha = float(getattr(params, "rrt_dd_alpha", 1e-4))
        # The domain radii are quoted as multiples of the step. That was unit-consistent when
        # the step was a metric distance; now that the step is a flight time, the radius is
        # only in the metric's units under the duration metric -- under the Euclidean one the
        # two sides of the comparison are metres against seconds, and the radius should be
        # set explicitly if the two are ever enabled together. VAMP's own defaults are
        # radius = 2x range and min_radius = 0.5x range.
        self.dd_radius = float(getattr(params, "rrt_dd_radius", 2.0)) * self.max_step
        self.dd_min_radius = float(getattr(params, "rrt_dd_min_radius", 0.5)) * self.max_step
        self.explored_nodes: List[List[float]] = []
        self.explored_nodes_3d: List[List[float]] = []
        # Live references to the trees the search grows, kept after `plan` returns so a
        # finished search can be plotted -- including a failed one, where the tree shape is
        # the whole point of looking.
        self.start_tree: Optional[_Tree] = None
        self.goal_tree: Optional[_Tree] = None

    # -- helpers -----------------------------------------------------------------------
    def _metric(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.metric.distance(a, b)

    def _sample(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        params = self.params
        if self.rng.random() < float(params.rrt_goal_bias):
            return goal.copy()

        lo = np.asarray(params.state_min, dtype=float)
        hi = np.asarray(params.state_max, dtype=float)
        # Keep headroom at each end of every position range. A quintic that starts or ends
        # with a non-zero derivative overshoots, and with `state_min[2] = 0` on a half-metre
        # z-slab that overshoot is the single largest source of rejected edges.
        span = hi[0:3] - lo[0:3]
        inset = min(max(float(params.rrt_sample_inset), 0.0), 0.49)
        position_lo = lo[0:3] + inset * span
        position_high = hi[0:3] - inset * span

        state = np.zeros(N_STATES)
        if self.rng.random() < float(params.rrt_bridge_bias):
            # Bridge sampling: draw near the straight line from start to goal. Narrow passages
            # are where uniform sampling starves -- threading one needs an edge aligned with
            # it almost exactly, which no box distribution produces in reasonable time.
            fraction = self.rng.random()
            position = start[0:3] + fraction * (goal[0:3] - start[0:3])
            position = position + self.rng.normal(0.0, float(params.rrt_bridge_sigma), 3)
            state[0:3] = np.clip(position, position_lo, position_high)
            return state  # from rest, so the edge stays near the line it was drawn on

        state[0:3] = self.rng.uniform(position_lo, position_high)
        velocity_span = np.abs(hi[3:6]) * float(params.rrt_sample_vel_fraction)
        acceleration_span = np.abs(hi[6:9]) * float(params.rrt_sample_acc_fraction)
        state[3:6] = self.rng.uniform(-velocity_span, velocity_span)
        state[6:9] = self.rng.uniform(-acceleration_span, acceleration_span)
        return state

    def _steer(self, near: np.ndarray, target: np.ndarray, step: float):
        """Steer toward ``target``, stopping after ``step`` seconds of the motion.

        The step is a flight time, not a metric distance: the duration a minimum-effort
        quintic takes is only a cube-root function of the distance it covers, so a metric
        step bought a varying amount of motion, and the stop state it produced carried the
        trajectory's velocity and acceleration -- which could leave it *farther* from the
        target than the node it grew from (measured on 299 of 301 walk extensions). In time,
        one step is a fixed slice of the motion itself. That is a cleaner step, not a cure
        for the walk: a step whose stop state is not the tree's new nearest still gets
        re-derived from the same parent, and the duplicates that produces are a property of
        the walk, not of the step's units.

        Returns a motion already checked against the obstacles and the state/input limits, so
        the caller only has to add it. For a step-limited move the stop state is read off the
        motion to ``target``; the truncated prefix is used when it is valid, and only
        otherwise is a fresh boundary value problem solved to that stop state. A prefix
        inherits the whole motion's initial jerk and is badly shaped for its own endpoints,
        so the re-solved edge is often the only one that fits -- but it costs a root solve,
        so it is the fallback rather than the default.
        """
        motion = self.steering.steer(near, target)
        if motion is None:
            return None, None, False

        if motion.duration <= step:
            if self._is_valid(motion):
                return motion, np.asarray(target, dtype=float).copy(), True
            return None, None, False

        waypoint = motion.flat_state(step)
        prefix = motion.truncated(step)
        if self._is_valid(prefix):
            return prefix, waypoint, False

        edge = self.steering.steer(near, waypoint)
        if edge is not None and self._is_valid(edge):
            return edge, waypoint, False

        # Last resort: a target interpolated in flat-state space by the same fraction of the
        # motion. The two attempts above inherit the shape of the motion to `target`, which is
        # badly matched when the steerer's duration does not scale with distance (as with
        # fixed-duration steering); blending the whole state gives that steerer a short,
        # gentle target instead.
        blended = near + (step / motion.duration) * (target - near)
        edge = self.steering.steer(near, blended)
        if edge is not None and self._is_valid(edge):
            return edge, blended, False
        return None, None, False

    def _is_valid(self, motion: FlatMotion) -> bool:
        if not self.checker.is_motion_valid(motion):
            return False
        # Skipping the bounds test is only sound because the limits are *pointwise*: a
        # truncated prefix of an admissible motion is itself admissible, so the promise the
        # bound-aware steerer made about the whole edge carries over to the prefix too.
        return self.bounds_guaranteed or is_motion_within_bounds(motion, self.params)

    def _record(self, state: np.ndarray) -> None:
        self.explored_nodes.append([float(state[0]), float(state[1])])
        self.explored_nodes_3d.append([float(state[0]), float(state[1]), float(state[2])])

    def _grow(self, tree: _Tree, target: np.ndarray, apply_domain: bool = True) -> int:
        """Extend the tree toward ``target``, shortening the step on rejection.

        A rejected edge usually means the quintic's interior excursion broke a state or input
        limit -- a long one overshoots, and a truncated one inherits the full motion's initial
        jerk. Halving the step and retrying converts most of those into shorter valid edges
        instead of dropping them, which matters most where the payload's z-slab is only
        ``state_max[2] - state_min[2]`` tall and any dip below ``state_min[2]`` is fatal.

        ``apply_domain`` gates VAMP's dynamic domain. Its radius bookkeeping rides on the
        main extension only; the CONNECT walk toward an already-reached node is exempt, as
        it is in the reference.
        """
        parent = tree.nearest(target, self.metric)
        near = tree.states[parent]
        distance = self._metric(near, target)
        if distance < self.metric_eps:
            return TRAPPED

        # Dynamic domain: a sample landing outside the nearest node's radius is dropped
        # before anything is steered. That is the whole saving -- the node is failing in
        # the same direction over and over, and re-solving the boundary value problem is
        # what costs. The radius shrinks on refusal and grows on success, so a node in a
        # dead end stops attracting samples while one that keeps working takes a wider
        # share of them.
        if self.dynamic_domain and apply_domain and tree.radii[parent] < distance:
            return REJECTED

        step = self.max_step
        for _ in range(self.retry_steps):
            motion, _, reached = self._steer(near, target, step)
            if motion is not None:
                if self.dynamic_domain and tree.radii[parent] != math.inf:
                    tree.radii[parent] *= 1.0 + self.dd_alpha
                self._extend(tree, parent, motion)
                return REACHED if reached else ADVANCED
            step *= 0.5

        if self.dynamic_domain and apply_domain:
            if tree.radii[parent] == math.inf:
                tree.radii[parent] = self.dd_radius
            else:
                tree.radii[parent] = max(
                    tree.radii[parent] * (1.0 - self.dd_alpha), self.dd_min_radius
                )
        return TRAPPED

    def _extend(self, tree: _Tree, parent: int, motion: FlatMotion) -> None:
        """Add ``motion`` to ``tree``, as a chain of nodes when intermediate states are on.

        This is OMPL's control-RRT ``intermediate_states`` option: the extension is walked
        state by state and every state spaced ``rrt_intermediate_dt`` apart joins the tree,
        so nearest-neighbour queries resolve at that spacing instead of at one node per
        extension -- which is what lets a greedy CONNECT walk keep advancing through a
        narrow region instead of stopping at the first node that does not help.

        Each node stores the *piece* of the motion arriving at it, so the chain concatenates
        back to exactly the edge the steerer validated; the parent of every node is the node
        before it, as in the reference.
        """
        spacing = self.intermediate_dt
        end = motion.duration
        start = 0.0
        node = parent
        if spacing > 0.0:
            for stop in np.arange(spacing, end, spacing):
                node = self._append(tree, node, motion, start, float(stop))
                start = float(stop)
        self._append(tree, node, motion, start, end)

    def _append(
        self, tree: _Tree, parent: int, motion: FlatMotion, start: float, end: float
    ) -> int:
        piece = motion if start == 0.0 and end >= motion.duration else motion.slice(start, end)
        state = piece.flat_state(piece.duration)
        tree.add(state, parent, piece)
        self._record(state)
        return tree.last

    # -- main loop ---------------------------------------------------------------------
    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[FlatPath]:
        if not self.checker.is_state_valid(start) or not self.checker.is_state_valid(goal):
            return None

        start_tree = _Tree(start)
        goal_tree = _Tree(goal)
        self.start_tree, self.goal_tree = start_tree, goal_tree
        self._record(start)
        self._record(goal)

        deadline = time.monotonic() + float(self.params.rrt_timeout)
        connect_limit = 100

        # Seeded the way the reference does it: `tree_a` starts as the *goal* tree so that
        # the first iteration's balance swap hands it to the start tree.
        tree_a, tree_b = goal_tree, start_tree
        tree_a_is_start = False

        for _ in range(int(self.params.rrt_max_iterations)):
            if time.monotonic() > deadline:
                break

            # Which tree gets the random sample, and which gets the greedy CONNECT walk.
            # See `tree_selection` for why "larger" is the default rather than the balance
            # rule from the reference.
            asize = len(tree_a)
            if self.tree_selection == "alternate":
                swap = True
            elif self.tree_selection == "ratio":
                swap = abs(asize - len(tree_b)) / asize < self.tree_ratio
            else:
                swap = asize < len(tree_b)
            if swap:
                tree_a, tree_b = tree_b, tree_a
                tree_a_is_start = not tree_a_is_start

            status = self._grow(tree_a, self._sample(start, goal))
            if status == TRAPPED or status == REJECTED:
                continue

            target = tree_a.states[tree_a.last]
            joined = None
            if self._metric(target, tree_b.states[0]) < self.metric_eps:
                # The extension landed exactly on the other tree's root, so the trees are
                # already joined. CONNECT would only see a zero-length target and report
                # TRAPPED, which used to discard a perfectly good direct connection.
                joined = 0
            else:
                for _ in range(connect_limit):
                    status = self._grow(tree_b, target, apply_domain=False)
                    if status == TRAPPED:
                        break
                    if status == REACHED:
                        joined = tree_b.last
                        break
            if joined is None:
                continue

            # `joined` indexes whichever tree CONNECT grew, which is the goal tree when the
            # start tree was the one just extended, and vice versa.
            if tree_a_is_start:
                path = self._extract(start_tree, tree_a.last, goal_tree, joined, self.steering)
            else:
                path = self._extract(start_tree, joined, goal_tree, tree_a.last, self.steering)
            if path is not None:
                return path
            # The junction was reached on the goal side but the edges cannot be re-solved
            # forward through it; keep growing both trees.

        return None

    def _extract(
        self,
        start_tree: _Tree,
        start_idx: int,
        goal_tree: _Tree,
        goal_idx: int,
        steering: FlatSteering,
    ) -> Optional[FlatPath]:
        """Walk the start tree back from ``start_idx``, then the goal tree from ``goal_idx``.

        Both trees store, at each node, the motion arriving there from its parent. Walking
        the start tree up from the junction and reversing both lists yields start-root ->
        junction with every edge already pointing forward.

        The goal tree is rooted at the *goal*, so its walk runs junction -> goal-root and
        every stored edge points the wrong way. A plain time-reversal is not usable: it
        would start at the junction with the negated velocity, breaking continuity with the
        incoming edge. Each hop is therefore re-solved as a fresh BVP in the forward
        direction, and the connection is rejected if any re-solved edge is invalid.
        """
        states, motions = [], []
        idx = start_idx
        while idx != -1:
            states.append(start_tree.states[idx])
            motions.append(start_tree.motions[idx])
            idx = start_tree.parents[idx]
        states.reverse()
        motions.reverse()

        walk = []
        idx = goal_idx
        while idx != -1:
            walk.append(goal_tree.states[idx])
            idx = goal_tree.parents[idx]

        # walk[0] is the junction, already the last element of `states`.
        for a, b in zip(walk[:-1], walk[1:]):
            motion = steering.steer(a, b)
            if motion is None:
                return None
            if not self.checker.is_motion_valid(motion):
                return None
            if not is_motion_within_bounds(motion, self.params):
                return None
            states.append(b)
            motions.append(motion)

        return FlatPath(states=states, motions=[m for m in motions if m is not None])


def _shortcut(
    path: FlatPath, checker: FlatEdgeChecker, steering: FlatSteering, params, deadline
):
    """Greedy shortcutting: replace a sub-chain with a single valid BVP edge."""
    if len(path.states) < 3:
        return path
    states = list(path.states)
    budget = int(params.rrt_shortcut_iters)
    i = 0
    while i < len(states) - 2 and budget > 0 and time.monotonic() < deadline:
        for j in range(len(states) - 1, i + 1, -1):
            budget -= 1
            if budget <= 0:
                break
            motion = steering.steer(states[i], states[j])
            if motion is None:
                continue
            if not checker.is_motion_valid(motion):
                continue
            if not is_motion_within_bounds(motion, params):
                continue
            states = states[: i + 1] + states[j:]
            break
        else:
            i += 1
            continue
        i += 1
    motions = []
    for a, b in zip(states[:-1], states[1:]):
        motion = steering.steer(a, b)
        if motion is None:
            return path  # shortcutting failed to reproduce the chain: keep the original
        motions.append(motion)
    return FlatPath(states=states, motions=motions)


def _goal_snap(
    path: FlatPath, goal: np.ndarray, checker: FlatEdgeChecker, steering: FlatSteering, params
):
    """Append a final edge to the exact goal state when one exists and is valid."""
    if not path.states:
        return path
    motion = steering.steer(path.states[-1], goal)
    if motion is None:
        return path
    if not checker.is_motion_valid(motion) or not is_motion_within_bounds(motion, params):
        return path
    return FlatPath(
        states=list(path.states) + [np.asarray(goal, dtype=float)],
        motions=list(path.motions) + [motion],
    )


class _DefaultMobility:
    """Flatness geometry from ``Planner``, resolved lazily.

    The lazy import is what keeps this module free of a circular dependency: ``planner``
    imports it at module level, so it must not import ``planner`` back at load time. Both
    helpers are passed through unchanged, so the RRT's collision geometry is by construction
    the same one the OCP's constraints use.
    """

    def __init__(self, params):
        self._params = params
        self._planner = None

    def _resolve(self):
        if self._planner is None:
            from poly_fly.optimal_planner.planner import Planner

            self._planner = Planner
        return self._planner

    def cable_rotation(self, acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._resolve().get_cable_rotation(acc)

    def quadrotor_rotation(self, acc: np.ndarray) -> np.ndarray:
        return self._resolve().compute_quadrotor_rotation_matrix_no_jrk(
            acc, self._params, use_robot_rotation=getattr(self._params, "use_robot_rotation", True)
        )


def _mobility_for(params, mobility=None):
    return mobility if mobility is not None else _DefaultMobility(params)


def resample_path_for_initial_guess(path: FlatPath, params):
    """Position, velocity and jerk samples sized for ``Planner.initialize_variables``.

    Returns ``(positions, velocities, inputs)`` with ``len(velocities) == len(positions)``
    and ``len(inputs) == len(positions) - 1``, which is what that method asserts.
    """
    horizon = int(params.horizon)
    times = np.linspace(0.0, path.duration(), horizon + 1)
    pos = path.samples_at(times, 0)
    vel = path.samples_at(times, 1)
    inp = path.samples_at(times[:-1], ORDER)  # one fewer sample, as the OCP expects
    return pos.tolist(), vel.tolist(), inp.tolist()


def path_to_sol_values(path: FlatPath, params, dt: Optional[float] = None) -> dict:
    """Dense uniform samples laid out as ``(N+1, 9)`` / ``(N, 3)`` / ``(N,)``.

    That is the shape ``Planner.save_result`` and ``interpolate`` expect.
    """
    dt = float(params.rrt_csv_dt) if dt is None else float(dt)
    total = path.duration()
    n = max(1, int(math.ceil(total / dt)))
    times = np.linspace(0.0, total, n + 1)
    x = np.hstack([path.samples_at(times, level) for level in range(ORDER)])
    u = path.samples_at(times[:-1], ORDER)
    t = np.diff(times)
    return {"x": x, "u": u, "t": t}


def make_metric(params, steering: FlatSteering) -> object:
    """Nearest-neighbour distance for the search, per ``rrt_metric``.

    ``euclidean`` weights velocity and acceleration errors against position. ``duration``
    ranks by the minimum-effort duration of the edge a node would actually fly, which
    prices "already moving the right way". Neither is tied to the step limit any more --
    the extension step is a flight time under both.
    """
    kind = str(getattr(params, "rrt_metric", "euclidean")).lower()
    if kind == "euclidean":
        return _WeightedEuclidean(float(getattr(params, "rrt_metric_omega", 2.0)))
    if kind == "duration":
        if not isinstance(steering, MinimumEffortSteering):
            raise ValueError(
                "rrt_metric 'duration' requires rrt_steering 'minimum_effort'; the fixed "
                "steerer has no duration to rank by"
            )
        return _OptimalDuration(steering)
    raise ValueError(f"unknown rrt_metric {kind!r}; expected 'euclidean' or 'duration'")


def make_steering(params) -> FlatSteering:
    kind = getattr(params, "rrt_steering", "minimum_effort")
    limits = None
    if bool(getattr(params, "rrt_bound_aware_duration", True)):
        limits = FlatLimits.from_params(params)
    if kind == "fixed":
        return FixedDurationSteering(duration=float(params.rrt_fixed_duration), limits=limits)
    if kind == "minimum_effort":
        return MinimumEffortSteering(
            rho=float(params.rrt_rho),
            limits=limits,
            min_duration=float(params.rrt_min_duration),
            max_duration=float(params.rrt_max_duration),
        )
    raise ValueError(f"unknown rrt_steering {kind!r}; expected 'minimum_effort' or 'fixed'")


def solve_with_rrt(
    params,
    start: Optional[Sequence[float]] = None,
    goal: Optional[Sequence[float]] = None,
    show_animation: bool = False,
    mobility=None,
) -> RRTResult:
    """Plan a dynamically feasible path between two flat states. Never raises on failure."""
    started = time.monotonic()
    rng = np.random.default_rng(int(params.rrt_seed))
    mobility = _mobility_for(params, mobility)
    steering = make_steering(params)

    start = np.asarray(params.initial_state if start is None else start, dtype=float)
    goal = np.asarray(params.end_state if goal is None else goal, dtype=float)

    planner = KinodynamicRRTConnect(params, steering, mobility, rng=rng)
    deadline = started + float(params.rrt_timeout)

    path = planner.plan(start, goal)
    if path is not None:
        path = _goal_snap(path, goal, planner.checker, steering, params)
        path = _shortcut(path, planner.checker, steering, params, deadline)

    result = RRTResult(
        path=path,
        explored_nodes=planner.explored_nodes,
        explored_nodes_3d=planner.explored_nodes_3d,
        iterations=len(planner.explored_nodes),
        duration=time.monotonic() - started,
        success=path is not None,
        seed=int(params.rrt_seed),
        start_tree=planner.start_tree,
        goal_tree=planner.goal_tree,
    )
    if show_animation:  # pragma: no cover - interactive only
        plot_trees(params, result)
    return result


# Tree edges are drawn as polylines; sampling each one at this rate in time is plenty to
# read the shape of a quintic without bloating the artist.
TREE_EDGE_SAMPLES_PER_SECOND = 20.0
TREE_EDGE_MAX_SAMPLES = 40


def _edge_polylines(tree: _Tree) -> np.ndarray:
    """NaN-separated 3-D payload positions along every edge of ``tree``.

    Positions only -- the derivatives are not part of this view. The whole tree becomes one
    plotting call rather than one artist per edge: a few thousand ``Line3D`` objects make
    matplotlib crawl, while NaN separators are free.
    """
    segments: List[np.ndarray] = []
    for index in range(1, len(tree)):
        motion = tree.motions[index]
        if motion is None:  # pragma: no cover - only the root has no incoming motion
            continue
        samples = max(
            2,
            min(
                TREE_EDGE_MAX_SAMPLES,
                int(math.ceil(motion.duration * TREE_EDGE_SAMPLES_PER_SECOND)),
            ),
        )
        segments.append(motion.samples_at(np.linspace(0.0, motion.duration, samples), 0))
        segments.append(np.full((1, OUTPUT_DIM), np.nan))
    return np.vstack(segments) if segments else np.empty((0, OUTPUT_DIM))


# The 12 edges of a box, as index pairs into the vertex list built in `_draw_obstacles`.
_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)


def _draw_obstacles(ax, params) -> None:
    """Obstacle boxes as wireframes.

    Wireframe rather than the filled faces ``poly_fly.utils.plot`` uses: six translucent
    faces per box, with several boxes stacked along the view ray, add up to an opaque one
    that swallows any tree behind it -- and mplot3d sorts a whole NaN-joined polyline by a
    single depth, so an entire tree can end up behind them. A tree you cannot see is no use
    in a debug view; the box edges still show where the walls are.
    """
    segments: List[np.ndarray] = []
    for key in params.obstacles.keys():
        obs = params.obstacles[key]
        x, y, z = obs["x"], obs["y"], obs["z"]
        l, b, h = obs["l"], obs["b"], obs["h"]
        vertices = np.array(
            [
                [x - l / 2, y - b / 2, z - h / 2],
                [x - l / 2, y + b / 2, z - h / 2],
                [x + l / 2, y + b / 2, z - h / 2],
                [x + l / 2, y - b / 2, z - h / 2],
                [x - l / 2, y - b / 2, z + h / 2],
                [x - l / 2, y + b / 2, z + h / 2],
                [x + l / 2, y + b / 2, z + h / 2],
                [x + l / 2, y - b / 2, z + h / 2],
            ]
        )
        for a, c in _BOX_EDGES:
            segments.append(vertices[[a, c]])
            segments.append(np.full((1, OUTPUT_DIM), np.nan))
    if not segments:
        return
    edges = np.vstack(segments)
    ax.plot(edges[:, 0], edges[:, 1], edges[:, 2], color="gray", lw=0.8, label="obstacles")


def plot_trees(params, result: RRTResult) -> None:  # pragma: no cover - interactive only
    """Debug view of the search in 3D: both trees, the obstacle set and the solution path.

    Every edge is drawn as the payload trajectory it actually is -- the stored motion
    sampled in time -- so a badly shaped quintic shows up as one, instead of being hidden
    by the straight chord that a node-only scatter would suggest. Start and goal trees get
    distinct colours; the solution path is overlaid in black. Also worth opening when the
    search *failed*: the trees are exactly what it managed to explore.
    """
    import matplotlib.pyplot as plt  # lazy: the repo forces TkAgg at import time

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _draw_obstacles(ax, params)

    for tree, color, label in (
        (result.start_tree, "tab:blue", "start tree"),
        (result.goal_tree, "tab:orange", "goal tree"),
    ):
        if tree is None or len(tree) < 2:
            continue
        edges = _edge_polylines(tree)
        ax.plot(edges[:, 0], edges[:, 1], edges[:, 2], color=color, lw=0.6, alpha=0.5, label=label)
        nodes = tree.states[1:, :3]
        ax.scatter(
            nodes[:, 0], nodes[:, 1], nodes[:, 2], s=2, color=color, label=label + " nodes"
        )

    if result.start_tree is not None:
        start = result.start_tree.states[0]
        ax.scatter(*start[:3], s=60, marker="o", color="green", label="start", depthshade=False)
    if result.goal_tree is not None:
        goal = result.goal_tree.states[0]
        ax.scatter(*goal[:3], s=90, marker="*", color="red", label="goal", depthshade=False)

    if result.success:
        times = np.linspace(0.0, result.path.duration(), 400)
        points = result.path.samples_at(times, 0)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color="black", lw=2.0, label="solution")

    state_min = np.asarray(params.state_min, dtype=float)[:3]
    state_max = np.asarray(params.state_max, dtype=float)[:3]
    ax.set_box_aspect(np.maximum(state_max - state_min, 1e-3))  # keep the scene un-squashed
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    start_nodes = 0 if result.start_tree is None else len(result.start_tree)
    goal_nodes = 0 if result.goal_tree is None else len(result.goal_tree)
    ax.set_title(
        "RRT-Connect %s -- %d start / %d goal nodes, %d iterations"
        % ("solved" if result.success else "failed", start_nodes, goal_nodes, result.iterations)
    )
    ax.legend(loc="upper left", fontsize="small")
    plt.show()


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Kinodynamic RRT-Connect (no optimization)")
    parser.add_argument("--yaml", type=str, required=True, help="path relative to data/params")
    parser.add_argument("--plot", action="store_true", help="plot the search trees and the path")
    parser.add_argument("--save", action="store_true", help="write the trajectory CSV")
    args = parser.parse_args()

    from poly_fly.data_io.utils import PARAMS_DIR
    from poly_fly.utils.utils import MPC, dictToClass, yamlToDict

    params = dictToClass(MPC, yamlToDict(os.path.join(PARAMS_DIR, args.yaml)))
    params.global_planner_type = "rrt_connect"
    result = solve_with_rrt(params, show_animation=args.plot)
    if not result.success:
        raise OpenSetEmptyException(f"RRT-Connect found no path for {args.yaml}")

    print(f"path segments: {len(result.path.motions)}  duration: {result.path.duration():.3f} s")
    print(f"explored nodes: {len(result.explored_nodes)}  wall clock: {result.duration:.3f} s")

    if args.save:
        from poly_fly.optimal_planner.planner import save_result

        stem = os.path.join(params.rrt_csv_subdir, args.yaml)
        save_result(stem, params, path_to_sol_values(result.path, params))


if __name__ == "__main__":
    main()
