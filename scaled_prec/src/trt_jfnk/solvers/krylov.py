"""SciPy CPU and pure-JAX Krylov backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres


@dataclass(frozen=True)
class KrylovOptions:
    backend: Literal["scipy", "jax"] = "scipy"
    method: Literal["gmres", "bicgstab"] = "gmres"
    rtol: float = 1.0e-8
    atol: float = 0.0
    max_iterations: int = 300
    restart: int = 50
    jax_solve_method: Literal["batched", "incremental"] = "batched"

    def __post_init__(self) -> None:
        if self.backend not in {"scipy", "jax"}:
            raise ValueError("Krylov backend must be scipy or jax")
        if self.method not in {"gmres", "bicgstab"}:
            raise ValueError("Krylov method must be gmres or bicgstab")
        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("Krylov tolerances must be nonnegative")
        if self.max_iterations < 1 or self.restart < 1:
            raise ValueError("Krylov iteration limits must be positive")


@dataclass(frozen=True)
class KrylovResult:
    solution: jax.Array
    converged: bool
    iterations: int
    residual_norm: float
    info: int | None


def _as_writable_numpy(value) -> np.ndarray:
    """Host copy that SciPy is free to mutate during orthogonalization."""

    return np.array(jax.device_get(value), copy=True, order="C")


def _solve_scipy(operator, rhs, preconditioner, options: KrylovOptions) -> KrylovResult:
    rhs_numpy = _as_writable_numpy(rhs)
    size = rhs_numpy.size

    def matvec(vector):
        return _as_writable_numpy(operator(jnp.asarray(vector, dtype=rhs.dtype)))

    linear_operator = LinearOperator((size, size), matvec=matvec, dtype=rhs_numpy.dtype)
    preconditioner_operator = None
    if preconditioner is not None:
        def psolve(vector):
            return _as_writable_numpy(preconditioner(jnp.asarray(vector, dtype=rhs.dtype)))

        preconditioner_operator = LinearOperator(
            (size, size), matvec=psolve, dtype=rhs_numpy.dtype
        )

    iteration_count = 0

    def count_iteration(_value):
        nonlocal iteration_count
        iteration_count += 1

    if options.method == "gmres":
        solution, info = gmres(
            linear_operator,
            rhs_numpy,
            M=preconditioner_operator,
            rtol=options.rtol,
            atol=options.atol,
            restart=options.restart,
            maxiter=options.max_iterations,
            callback=count_iteration,
            callback_type="legacy",
        )
    else:
        solution, info = bicgstab(
            linear_operator,
            rhs_numpy,
            M=preconditioner_operator,
            rtol=options.rtol,
            atol=options.atol,
            maxiter=options.max_iterations,
            callback=count_iteration,
        )

    solution_device = jnp.asarray(solution, dtype=rhs.dtype)
    residual_norm = float(jnp.linalg.norm(operator(solution_device) - rhs))
    threshold = max(options.atol, options.rtol * float(jnp.linalg.norm(rhs)))
    return KrylovResult(
        solution=solution_device,
        converged=bool(info == 0 and residual_norm <= max(threshold, 10.0 * np.finfo(rhs_numpy.dtype).eps)),
        iterations=iteration_count,
        residual_norm=residual_norm,
        info=int(info),
    )


def _solve_jax(operator, rhs, preconditioner, options: KrylovOptions) -> KrylovResult:
    import jax.scipy.sparse.linalg as jsparse

    preconditioner = (lambda vector: vector) if preconditioner is None else preconditioner
    if options.method == "gmres":
        # JAX counts maxiter in restart cycles; expose the API in total-vector
        # iterations to match the SciPy backend as closely as possible.
        restart_cycles = max(1, math.ceil(options.max_iterations / options.restart))
        solution, info = jsparse.gmres(
            operator,
            rhs,
            tol=options.rtol,
            atol=options.atol,
            restart=options.restart,
            maxiter=restart_cycles,
            M=preconditioner,
            solve_method=options.jax_solve_method,
        )
    else:
        solution, info = jsparse.bicgstab(
            operator,
            rhs,
            tol=options.rtol,
            atol=options.atol,
            maxiter=options.max_iterations,
            M=preconditioner,
        )

    residual_norm = float(jnp.linalg.norm(operator(solution) - rhs))
    threshold = max(options.atol, options.rtol * float(jnp.linalg.norm(rhs)))
    # Current JAX releases return info=None.  Verify the true residual instead.
    info_value = None if info is None else int(jax.device_get(info))
    converged = residual_norm <= max(threshold, 10.0 * np.finfo(np.dtype(rhs.dtype)).eps)
    return KrylovResult(solution, converged, -1, residual_norm, info_value)


def solve_krylov(
    operator: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    options: KrylovOptions,
    preconditioner: Callable[[jax.Array], jax.Array] | None = None,
) -> KrylovResult:
    """Solve a matrix-free linear system on the selected backend."""

    if options.backend == "scipy":
        return _solve_scipy(operator, rhs, preconditioner, options)
    return _solve_jax(operator, rhs, preconditioner, options)
