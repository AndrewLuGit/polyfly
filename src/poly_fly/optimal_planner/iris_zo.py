"""IRIS-ZO in the 6-D flat configuration space of the cable-suspended payload.

Configuration space
-------------------
The payload's flat state is ``[p, v, a]`` (9-D), but collision geometry depends only on the
payload position ``p`` and acceleration ``a``: the cable and quadrotor poses are algebraic
functions of ``a`` and velocity never moves a body.  IRIS-ZO therefore grows regions in

    q = [p_x, p_y, p_z, a_x, a_y, a_z] in R^6.

Collision checking is passed in as a callable ``collision_free(configs) -> bool array`` so
this module has no dependency on the RRT or the OCP.  Use
``poly_fly.optimal_planner.flat_collision.FlatEdgeChecker.configs_collision_free`` for the
quadrotor/cable/payload geometry used by the rest of the repository.

Algorithm
---------
This is a Python port of Drake's ``planning/iris/iris_zo.cc`` (Werner et al., "Faster
Algorithms for Growing Collision-Free Convex Polytopes in Robot Configuration Space",
2024): sample particles uniformly from the current polytope with hit-and-run, bisect
colliding particles against the ellipsoid centre, add tangent halfspaces to the ellipsoid,
fit the maximum-volume inscribed ellipsoid (MVEE) of the resulting polytope, and repeat.

The one substitution forced by the available dependencies is the MVEE.  Drake solves an SDP;
here we solve the equivalent convex problem without an SDP solver:

1. shift the polytope by an interior origin,
2. build the polar point set ``{a_i / slack_i}`` (the polar of the max-volume inscribed
   ellipsoid is the min-volume enclosing ellipsoid of those points),
3. compute that enclosing ellipsoid with a barrier-Newton method on the dual,
4. polar-transform back.

The polar volume identity requires the origin to be the inscribed ellipsoid's centre, so
steps 1-4 are iterated until the centre stops moving (2-5 passes in practice); at the fixed
point the result is the exact MVEE.  A FIRI-style penalized L-BFGS formulation (GCOPTER
``firi.hpp``) and SLSQP on the Cholesky parametrization were both benchmarked and rejected:
in 6-D they were slower and returned suboptimal (FIRI: ~13% smaller on a random polytope;
SLSQP: far worse) ellipsoids.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import linprog

CollisionFreeFn = Callable[[np.ndarray], np.ndarray]

# Barrier-Newton schedule for the MVEE dual.
_MVEE_MU0 = 1.0
_MVEE_MU_MIN = 1e-9
_MVEE_MU_FACTOR = 0.1
_MVEE_INNER_ITERS = 100
_MVEE_TOL = 1e-9


class EmptyPolytopeError(RuntimeError):
    """The polytope is empty or lower dimensional."""


class IrisZoError(RuntimeError):
    """IRIS-ZO could not produce a valid region for this seed."""


def unit_ball_volume(dim: int) -> float:
    """Volume of the unit ball in ``dim`` dimensions."""
    return math.pi ** (dim / 2.0) / math.gamma(dim / 2.0 + 1.0)


@dataclass
class Ellipsoid:
    """``E = {x : ||A (x - center)|| <= 1}`` (Drake's Hyperellipsoid convention)."""

    center: np.ndarray
    A: np.ndarray

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float).reshape(-1)
        self.A = np.asarray(self.A, dtype=float)
        n = self.center.size
        if self.A.shape != (n, n):
            raise ValueError("Ellipsoid A must be square with one row per centre coordinate")
        if not (np.isfinite(self.center).all() and np.isfinite(self.A).all()):
            raise ValueError("Ellipsoid data must be finite")
        det = float(np.linalg.det(self.A))
        if not np.isfinite(det) or abs(det) < 1e-300:
            raise ValueError("Ellipsoid A is singular")

    @classmethod
    def from_center_radius(cls, center: np.ndarray, radius: float) -> "Ellipsoid":
        center = np.asarray(center, dtype=float).reshape(-1)
        radius = float(radius)
        if not radius > 0.0:
            raise ValueError("Ellipsoid radius must be positive")
        return cls(center, np.eye(center.size) / radius)

    @property
    def dimension(self) -> int:
        return self.center.size

    @property
    def volume(self) -> float:
        det = abs(float(np.linalg.det(self.A)))
        return unit_ball_volume(self.dimension) / det

    def mahalanobis_sq(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        y = (points - self.center) @ self.A.T
        return np.einsum("ij,ij->i", y, y)

    def contains(self, points: np.ndarray, tol: float = 1e-7) -> np.ndarray:
        return self.mahalanobis_sq(points) <= 1.0 + tol

    def ata_mul(self, v: np.ndarray) -> np.ndarray:
        return self.A.T @ (self.A @ np.asarray(v, dtype=float))


class HPolyhedron:
    """``P = {x : A x <= b}`` with normalized rows."""

    def __init__(self, A: np.ndarray, b: np.ndarray):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        if A.ndim != 2 or A.shape[0] != b.size:
            raise ValueError("A must be (m, n) and b must have length m")
        if not (np.isfinite(A).all() and np.isfinite(b).all()):
            raise ValueError("Polytope data must be finite")
        norms = np.linalg.norm(A, axis=1)
        keep = norms > 1e-12
        if not keep.any():
            raise ValueError("Polytope has no non-degenerate rows")
        if np.any(b[~keep] < 0.0):
            raise EmptyPolytopeError("Polytope contains an infeasible zero row")
        self.A = A[keep] / norms[keep, None]
        self.b = b[keep] / norms[keep]

    @classmethod
    def from_box(cls, lower: np.ndarray, upper: np.ndarray) -> "HPolyhedron":
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower.size != upper.size:
            raise ValueError("box bounds must have the same size")
        if np.any(upper - lower <= 0.0):
            raise ValueError("box must have positive width in every coordinate")
        n = lower.size
        A = np.vstack([np.eye(n), -np.eye(n)])
        return cls(A, np.concatenate([upper, -lower]))

    @property
    def dimension(self) -> int:
        return self.A.shape[1]

    # -- membership -------------------------------------------------------------------
    def contains_batch(self, points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return np.all(points @ self.A.T <= self.b[None, :] + tol, axis=1)

    def with_halfspace(self, a: np.ndarray, b: float) -> "HPolyhedron":
        a = np.asarray(a, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(a))
        if norm < 1e-12:
            return self
        return HPolyhedron(
            np.vstack([self.A, a / norm]),
            np.concatenate([self.b, [float(b) / norm]]),
        )

    # -- interior points --------------------------------------------------------------
    def chebyshev_center(self):
        """Return ``(center, radius)`` of the largest inscribed Euclidean ball."""
        n = self.dimension
        norms = np.linalg.norm(self.A, axis=1)
        objective = np.zeros(n + 1)
        objective[-1] = -1.0
        result = linprog(
            objective,
            A_ub=np.hstack([self.A, norms[:, None]]),
            b_ub=self.b,
            bounds=[(None, None)] * n + [(0.0, None)],
            method="highs",
        )
        if not result.success:
            raise EmptyPolytopeError("Polytope is empty")
        return result.x[:n], float(result.x[-1])

    def analytic_center(self, max_iter: int = 200) -> np.ndarray:
        """Unique maximizer of ``sum_i log(b_i - a_i^T x)``.

        The Chebyshev LP is degenerate (for a box it returns arbitrary points on a face of
        the optimal set), so its answer is a poor region/sampling centre.  The analytic
        centre is well-centered and is used for hit-and-run and for the MVEE polar trick.
        """
        center, radius = self.chebyshev_center()
        if radius <= 1e-12:
            raise EmptyPolytopeError("Polytope is lower dimensional")
        norms = np.linalg.norm(self.A, axis=1)
        # Start strictly inside the polytope shrunk by half the Chebyshev radius.
        b_shrunk = self.b - 0.5 * radius * norms
        x = center.copy()
        for _ in range(max_iter):
            slack = b_shrunk - self.A @ x
            if np.min(slack) <= 0.0:
                x = 0.5 * (x + center)
                continue
            gradient = -self.A.T @ (1.0 / slack)
            hessian = self.A.T @ ((1.0 / slack**2)[:, None] * self.A)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            decrement = float(gradient @ step)
            t = 1.0
            while t > 1e-14 and not np.all(b_shrunk - self.A @ (x + t * step) > 0.0):
                t *= 0.5
            if t <= 1e-14:
                break
            x = x + t * step
            if decrement < 1e-11:
                break
        return x

    # -- sampling ---------------------------------------------------------------------
    def uniform_sample(
        self,
        rng: np.random.Generator,
        n_samples: int,
        mixing_steps: int = 50,
        start: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Coordinate hit-and-run samples (one chain per sample).

        Each mixing step picks one random coordinate per chain and moves it uniformly
        along the exact slice of the polytope.  In this 6-D configuration space the free
        regions are often thin in a few directions; coordinate moves mix them several
        times faster than random-direction hit-and-run at the same number of steps.
        """
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if mixing_steps < 1:
            raise ValueError("mixing_steps must be positive")
        if start is None:
            start = self.analytic_center()
        x = np.tile(np.asarray(start, dtype=float), (n_samples, 1))
        A, b = self.A, self.b
        dim = self.dimension
        eps = 1e-12
        chain = np.arange(n_samples)
        for _ in range(int(mixing_steps)):
            coordinate = rng.integers(0, dim, size=n_samples)
            q = A[:, coordinate].T  # (N, m): row s is A[:, coordinate[s]]
            slack = b[None, :] - x @ A.T  # (N, m)
            pos = q > eps
            neg = q < -eps
            t_hi = np.where(pos, slack / np.where(pos, q, 1.0), np.inf).min(axis=1)
            t_lo = np.where(neg, slack / np.where(neg, q, 1.0), -np.inf).max(axis=1)
            if not (np.isfinite(t_lo).all() and np.isfinite(t_hi).all()):
                raise EmptyPolytopeError("Cannot sample an unbounded polytope")
            # Guard against roundoff-sized inversions of the interval.
            t_hi = np.maximum(t_hi, t_lo)
            theta = t_lo + (t_hi - t_lo) * rng.random(n_samples)
            x[chain, coordinate] += theta
        return x

    # -- MVEE -------------------------------------------------------------------------
    def maximum_volume_inscribed_ellipsoid(
        self, max_iterations: int = 5, center_tol: float = 1e-3
    ) -> Ellipsoid:
        """Maximum-volume ellipsoid inscribed in this polytope.

        Each pass shifts the polytope by an interior origin, polarizes it (the polar point
        set is ``{a_i / slack_i}``), computes the min-volume enclosing ellipsoid of those
        points with a barrier-Newton dual solve, and polarizes back.  The polar volume
        identity only holds when the origin is the final ellipsoid's centre, so the passes
        are iterated until the centre stops moving (or ``max_iterations`` is reached); at
        the fixed point this is the exact MVEE.  In practice 2-5 passes give a very accurate
        ellipsoid at a fraction of the cost of full convergence.

        Falls back to the (always inscribed) Chebyshev ball on numerical trouble.
        """
        center, radius = self.chebyshev_center()
        if radius <= 1e-12:
            raise EmptyPolytopeError("Polytope is lower dimensional")
        try:
            origin = self.analytic_center()
            ellipsoid = None
            for _ in range(max(1, int(max_iterations))):
                slack = self.b - self.A @ origin
                if np.min(slack) <= 0.0:
                    raise IrisZoError("polar origin is not strictly interior")
                mu, M = _mvee_points(self.A / slack[:, None])
                ellipsoid = _polar_back(mu, M, origin)
                step = float(np.linalg.norm(ellipsoid.center - origin))
                origin = ellipsoid.center
                if step <= float(center_tol):
                    break

            # Numerical safety: shrink the ellipsoid until every face is satisfied.
            # E = {c + L u}; support in direction a_i is ||L^T a_i|| = ||(A L)_i||.
            support = np.linalg.norm(self.A @ np.linalg.inv(ellipsoid.A), axis=1)
            room = self.b - self.A @ ellipsoid.center
            if np.any(room <= 0.0):
                raise IrisZoError("MVEE centre lies outside the polytope")
            ratios = np.where(support > 1e-12, room / np.maximum(support, 1e-12), np.inf)
            beta = float(np.min(ratios))
            if beta < 1.0:
                ellipsoid = Ellipsoid(ellipsoid.center, ellipsoid.A / max(beta, 1e-12))
            return ellipsoid
        except (IrisZoError, EmptyPolytopeError, np.linalg.LinAlgError, ValueError) as exc:
            warnings.warn(f"IRIS-ZO MVEE fallback to Chebyshev ball: {exc}")
            return Ellipsoid(center, np.eye(self.dimension) / radius)

    def chebyshev_ellipsoid(self) -> Ellipsoid:
        center, radius = self.chebyshev_center()
        if radius <= 1e-12:
            raise EmptyPolytopeError("Polytope is lower dimensional")
        return Ellipsoid(center, np.eye(self.dimension) / radius)


def _mvee_points(points: np.ndarray):
    """Min-volume enclosing ellipsoid of ``points`` via barrier-Newton on the dual.

    Returns ``(center, M)`` with ``E = {x : (x-center)^T M (x-center) <= 1}``.
    """
    points = np.asarray(points, dtype=float)
    m, n = points.shape
    if m < n + 1:
        raise IrisZoError("Not enough polar points to bound an ellipsoid")

    middle = points.mean(axis=0)
    scale = float(np.max(np.linalg.norm(points - middle, axis=1)))
    if scale <= 1e-12:
        raise IrisZoError("Polar points are degenerate")
    P = (points - middle) / scale

    # Augment with ones so the dual optimisation also finds the centre.
    Q = np.vstack([P.T, np.ones(m)])  # (n+1, m)
    u = np.full(m, 1.0 / m)
    mu = _MVEE_MU0

    while mu > _MVEE_MU_MIN:
        for _ in range(_MVEE_INNER_ITERS):
            T = Q @ (u[:, None] * Q.T)
            sign, logdet = np.linalg.slogdet(T)
            if sign <= 0:
                raise IrisZoError("MVEE dual matrix is not positive definite")
            T_inv = np.linalg.inv(T)
            Z = Q.T @ T_inv @ Q
            Z = 0.5 * (Z + Z.T)
            gradient = np.diag(Z) + mu / u
            hessian = -(Z * Z) - mu * np.diag(1.0 / u**2)

            kkt = np.zeros((m + 1, m + 1))
            kkt[:m, :m] = hessian
            kkt[:m, m] = 1.0
            kkt[m, :m] = 1.0
            try:
                step = np.linalg.solve(kkt, -np.r_[gradient, 0.0])[:m]
            except np.linalg.LinAlgError:
                raise IrisZoError("MVEE dual KKT system is singular")
            decrement = float(gradient @ step)
            if decrement / 2.0 < _MVEE_TOL:
                break

            t = 1.0
            current = logdet + mu * np.sum(np.log(u))
            while t > 1e-14:
                candidate = u + t * step
                if np.all(candidate > 0.0):
                    cand_sign, cand_logdet = np.linalg.slogdet(
                        Q @ (candidate[:, None] * Q.T)
                    )
                    cand_value = cand_logdet + mu * np.sum(np.log(candidate))
                    if cand_sign > 0 and cand_value >= current - 1e-12:
                        break
                t *= 0.5
            if t <= 1e-14:
                break
            u = u + t * step
            # Keep mu / u and 1 / u^2 finite for points that the barrier has driven to
            # (numerically) zero weight; the perturbation is far below the final mu.
            if np.min(u) < 1e-15:
                u = np.maximum(u, 1e-15)
                u /= u.sum()
        mu *= _MVEE_MU_FACTOR

    center = P.T @ u
    centered = P - center
    W = (centered * u[:, None]).T @ centered
    if not np.isfinite(W).all():
        raise IrisZoError("MVEE covariance is not finite")
    try:
        M = np.linalg.inv(W) / n
    except np.linalg.LinAlgError:
        raise IrisZoError("MVEE covariance is singular")
    return center * scale + middle, M / (scale * scale)


def _polar_back(mu: np.ndarray, M: np.ndarray, origin: np.ndarray) -> Ellipsoid:
    """Polar of ``{z : (z-mu)^T M (z-mu) <= 1}``, expressed about ``origin``."""
    Q = np.linalg.inv(M)
    H = Q - np.outer(mu, mu)
    H = 0.5 * (H + H.T)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    scale = max(1.0, float(eigenvalues.max()))
    if eigenvalues.min() < -1e-9 * scale:
        raise IrisZoError("MVEE polar matrix is not PSD")
    eigenvalues = np.maximum(eigenvalues, 1e-12 * scale)
    H = (eigenvectors * eigenvalues) @ eigenvectors.T
    H_inv = np.linalg.inv(H)
    shifted_center = -H_inv @ mu
    quadratic = 1.0 + float(mu @ H_inv @ mu)
    if quadratic <= 0.0 or not np.isfinite(quadratic):
        raise IrisZoError("MVEE polar reconstruction failed")
    sqrt_H = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    A_shape = sqrt_H / math.sqrt(quadratic)
    return Ellipsoid(origin + shifted_center, A_shape)


@dataclass
class IrisZoOptions:
    """IRIS-ZO options; defaults mirror Drake's CommonSampledIrisOptions."""

    num_particles: int = 1000
    tau: float = 0.5
    delta: float = 0.05
    epsilon: float = 0.01
    max_iterations: int = 3
    max_iterations_separating_planes: int = 20
    max_separating_planes_per_iteration: int = 10  # -1 = unlimited
    verbose: bool = False
    require_sample_point_is_contained: bool = True
    configuration_space_margin: float = 1e-2
    relax_margin: bool = False
    termination_threshold: float = 2e-2
    relative_termination_threshold: float = 1e-3
    remove_all_collisions_possible: bool = True
    random_seed: int = 1234
    mixing_steps: int = 50
    bisection_steps: int = 10
    # Local knobs (not in Drake).
    max_test_samples: int = 0  # 0 = uncapped; >0 weakens the delta guarantee
    mvee_method: str = "auto"  # "auto" | "chebyshev"
    mvee_iterations: int = 5  # polar fixed-point passes per MVEE
    mvee_center_tol: float = 1e-3  # stop once the MVEE centre moves less than this

    def validate(self) -> None:
        for name in ("tau", "delta", "epsilon"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.num_particles < 1:
            raise ValueError("num_particles must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.max_iterations_separating_planes < 1:
            raise ValueError("max_iterations_separating_planes must be positive")
        if self.max_separating_planes_per_iteration == 0 or (
            self.max_separating_planes_per_iteration < 0
            and self.max_separating_planes_per_iteration != -1
        ):
            raise ValueError("max_separating_planes_per_iteration must be -1 or positive")
        if self.mixing_steps < 1:
            raise ValueError("mixing_steps must be positive")
        if self.bisection_steps < 0:
            raise ValueError("bisection_steps must be nonnegative")
        if self.configuration_space_margin < 0.0:
            raise ValueError("configuration_space_margin must be nonnegative")
        if self.mvee_method not in ("auto", "chebyshev"):
            raise ValueError("mvee_method must be 'auto' or 'chebyshev'")
        if self.mvee_iterations < 1:
            raise ValueError("mvee_iterations must be positive")
        if self.mvee_center_tol < 0.0:
            raise ValueError("mvee_center_tol must be nonnegative")


@dataclass
class IrisZoRegion:
    """A grown polytope plus the statistics of the run that produced it."""

    polytope: HPolyhedron
    seed: np.ndarray
    center: np.ndarray
    volume: float
    iterations: int
    termination: str

    def contains(self, points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        return self.polytope.contains_batch(points, tol=tol)


def _bisect(
    collision_free: CollisionFreeFn,
    center: np.ndarray,
    points: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Bisect each colliding point toward the (collision-free) ellipsoid centre.

    Returns the last colliding point on each segment, matching Drake's IRIS-ZO.
    """
    if not bool(np.asarray(collision_free(center[None, :])).reshape(-1)[0]):
        return np.tile(center, (len(points), 1))
    lower = np.tile(np.asarray(center, dtype=float), (len(points), 1))
    upper = np.asarray(points, dtype=float).copy()
    for _ in range(int(steps)):
        middle = 0.5 * (lower + upper)
        free = np.asarray(collision_free(middle), dtype=bool).reshape(-1)
        lower = np.where(free[:, None], middle, lower)
        upper = np.where(free[:, None], upper, middle)
    return upper


def _tangent_plane(
    ellipsoid: Ellipsoid,
    point: np.ndarray,
    margin: float,
    relax_margin: bool,
):
    direction = ellipsoid.ata_mul(point - ellipsoid.center)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    a_face = direction / norm
    b_point = float(a_face @ point)
    b_face = b_point - float(margin)
    b_center = float(a_face @ ellipsoid.center)
    if b_center > b_face:
        if not relax_margin:
            raise IrisZoError(
                "configuration_space_margin cuts off the ellipsoid centre; "
                "set relax_margin=True or reduce the margin"
            )
        b_face = 0.5 * (b_point + b_center)
    return a_face, b_face


def iris_zo(
    collision_free: CollisionFreeFn,
    starting_ellipsoid: Ellipsoid,
    domain: HPolyhedron,
    options: Optional[IrisZoOptions] = None,
) -> IrisZoRegion:
    """Grow one probabilistically collision-free polytope around the ellipsoid centre."""
    opts = IrisZoOptions() if options is None else options
    opts.validate()

    ellipsoid = starting_ellipsoid
    seed = np.asarray(ellipsoid.center, dtype=float).copy()
    if domain.dimension != ellipsoid.dimension:
        raise ValueError("domain and starting ellipsoid dimensions differ")
    if not bool(domain.contains_batch(seed[None, :])[0]):
        raise IrisZoError("starting ellipsoid centre is outside the domain")
    if not bool(np.asarray(collision_free(seed[None, :])).reshape(-1)[0]):
        raise IrisZoError("starting ellipsoid centre is in collision")

    rng = np.random.default_rng(opts.random_seed)
    P_prev = domain
    P = domain
    previous_volume = 0.0
    last_volume = 0.0
    last_center = seed.copy()

    def region(polytope, volume, center, iteration, termination):
        return IrisZoRegion(
            polytope=polytope,
            seed=seed.copy(),
            center=np.asarray(center, dtype=float).copy(),
            volume=float(volume),
            iterations=int(iteration),
            termination=termination,
        )

    def finish(polytope, polytope_ellipsoid, iteration, termination):
        """Return ``polytope``, falling back to the previous one if it lost the seed.

        Drake checks ``require_sample_point_is_contained`` only after the volume and
        iteration termination tests.  That ordering lets a volume-terminated polytope cut
        off the seed; for a cover, every seed must stay in its own region, so the check is
        applied to every terminal polytope here.
        """
        if opts.require_sample_point_is_contained and not bool(
            polytope.contains_batch(seed[None, :])[0]
        ):
            if P_prev is domain:
                # Returning the domain would "contain the seed" but is not collision free.
                raise IrisZoError("the first grown polytope lost the seed")
            return region(
                P_prev, ellipsoid.volume, ellipsoid.center, iteration, "seed_not_contained"
            )
        return region(
            polytope,
            polytope_ellipsoid.volume,
            polytope_ellipsoid.center,
            iteration,
            termination,
        )

    for iteration in range(opts.max_iterations):
        outer_delta = (
            opts.delta
            if opts.max_iterations == 1
            else opts.delta * 6.0 / (math.pi**2 * (iteration + 1) ** 2)
        )
        P = domain
        test_passed = False
        for k in range(1, opts.max_iterations_separating_planes + 1):
            delta_k = outer_delta * 6.0 / (math.pi**2 * k * k)
            n_test = int(-2.0 * math.log(delta_k) / (opts.tau**2 * opts.epsilon) + 0.5)
            if opts.max_test_samples > 0:
                n_test = min(n_test, opts.max_test_samples)
            n_samples = max(n_test, opts.num_particles)

            samples = P.uniform_sample(rng, n_samples, opts.mixing_steps)
            free = np.asarray(collision_free(samples), dtype=bool).reshape(-1)
            if free.size != n_samples:
                raise IrisZoError("collision checker returned the wrong number of results")
            bad = ~free
            n_col_test = int(bad[:n_test].sum())
            test_passed = n_col_test <= (1.0 - opts.tau) * opts.epsilon * n_test

            if opts.verbose and not test_passed:
                warnings.warn(
                    f"IRIS-ZO outer {iteration} inner {k}: {n_col_test} collisions "
                    f"among {n_test} test samples"
                )
            if test_passed and not opts.remove_all_collisions_possible:
                break
            if not bad.any():
                break

            bad_points = samples[bad][: opts.num_particles]
            boundary = _bisect(
                collision_free, ellipsoid.center, bad_points, opts.bisection_steps
            )

            distances = ellipsoid.mahalanobis_sq(boundary)
            order = np.argsort(distances)
            redundant = np.zeros(len(order), dtype=bool)
            added = 0
            for index in order:
                if redundant[index]:
                    continue
                plane = _tangent_plane(
                    ellipsoid,
                    boundary[index],
                    opts.configuration_space_margin,
                    opts.relax_margin,
                )
                if plane is None:
                    continue
                a_face, b_face = plane
                P = P.with_halfspace(a_face, b_face)
                redundant[index] = True
                remaining = ~redundant
                redundant[remaining] |= (boundary[remaining] @ a_face - b_face) >= 0.0
                added += 1
                if (
                    opts.max_separating_planes_per_iteration > 0
                    and added >= opts.max_separating_planes_per_iteration
                ):
                    break

            if test_passed:
                break

        if opts.mvee_method == "chebyshev":
            ellipsoid_new = P.chebyshev_ellipsoid()
        else:
            ellipsoid_new = P.maximum_volume_inscribed_ellipsoid(
                max_iterations=opts.mvee_iterations,
                center_tol=opts.mvee_center_tol,
            )
        volume = ellipsoid_new.volume
        last_volume = volume
        last_center = ellipsoid_new.center.copy()

        delta_volume = volume - previous_volume
        if (
            opts.termination_threshold >= 0.0
            and delta_volume <= opts.termination_threshold
        ):
            return finish(P, ellipsoid_new, iteration + 1, "absolute_volume")
        if (
            opts.relative_termination_threshold >= 0.0
            and delta_volume / (previous_volume + 1e-6)
            <= opts.relative_termination_threshold
        ):
            return finish(P, ellipsoid_new, iteration + 1, "relative_volume")
        if iteration + 1 >= opts.max_iterations:
            return finish(P, ellipsoid_new, iteration + 1, "max_iterations")
        if not bool(np.asarray(collision_free(last_center[None, :])).reshape(-1)[0]):
            return finish(P, ellipsoid_new, iteration + 1, "center_in_collision")
        if opts.require_sample_point_is_contained and not bool(
            P.contains_batch(seed[None, :])[0]
        ):
            if P_prev is domain:
                raise IrisZoError("the first grown polytope lost the seed")
            return region(
                P_prev,
                ellipsoid.volume,
                ellipsoid.center,
                iteration + 1,
                "seed_not_contained",
            )

        previous_volume = volume
        P_prev = P
        ellipsoid = ellipsoid_new

    return region(P, last_volume, last_center, opts.max_iterations, "max_iterations")


def payload_domain_from_params(params):
    """Box domain over ``(p, a)`` from the planner's flat-state bounds.

    Returns ``(domain, lower, upper)`` where the latter two are the 6-D bounds.
    """
    state_min = np.asarray(params.state_min, dtype=float)
    state_max = np.asarray(params.state_max, dtype=float)
    index = [0, 1, 2, 6, 7, 8]
    lower = state_min[index]
    upper = state_max[index]
    return HPolyhedron.from_box(lower, upper), lower, upper
