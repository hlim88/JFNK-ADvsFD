############## BiCGStab ################

import cupy
from cupyx.scipy.sparse.linalg._iterative import _make_system


def bicgstab(A, b, x0=None, *, rtol=1e-5, atol=0.0, maxiter=None, M=None,
             callback=None):
    """Uses BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

    Args:
        A (ndarray, spmatrix or LinearOperator): The real or complex matrix of
            the linear system with shape ``(n, n)``.
        b (cupy.ndarray): Right hand side of the linear system with shape
            ``(n,)`` or ``(n, 1)``.
        x0 (cupy.ndarray): Starting guess for the solution.
        rtol, atol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.
        M (ndarray, spmatrix or LinearOperator): Preconditioner for ``A``.
            The preconditioner should approximate the inverse of ``A``.
            ``M`` must be :class:`cupy.ndarray`,
            :class:`cupyx.scipy.sparse.spmatrix` or
            :class:`cupyx.scipy.sparse.linalg.LinearOperator`.
        callback (function): User-specified function to call after each
            iteration. It is called as ``callback(xk)``, where ``xk`` is the
            current solution vector.

    Returns:
        tuple:
            It returns ``x`` (cupy.ndarray) and ``info`` (int) where ``x`` is
            the converged solution and ``info`` provides convergence
            information.

    .. seealso:: :func:`scipy.sparse.linalg.bicgstab`
    """
    A, M, x, b = _make_system(A, M, x0, b)

    matvec = A.matvec
    psolve = M.matvec

    n = A.shape[0]
    if n == 0:
        return cupy.empty_like(b), 0

    b_norm = cupy.linalg.norm(b)
    if b_norm == 0:
        return b, 0

    atol = max(float(atol), rtol * float(b_norm))
    if maxiter is None:
        maxiter = 10 * n

    # Handle complex dot products appropriately like SciPy version
    dotprod = cupy.vdot if x.dtype.kind == 'c' else cupy.dot

    rhotol = cupy.finfo(x.dtype.char).eps ** 2
    omegatol = rhotol

    r = b - matvec(x)
    rtilde = r.copy()

    # Initialize vars to prevent linter warnings
    rho_prev, omega, alpha, p, v = 0.0, 0.0, 0.0, 0.0, 0.0

    iters = 0
    while True:
        r_norm = cupy.linalg.norm(r)
        if r_norm <= atol:  # reached abs tol
            break
        if iters >= maxiter:  # reached max iters
            break

        # pull from device to host
        rho = dotprod(rtilde, r).item()

        # Breakdown checks ensure no NaNs
        if abs(rho) < rhotol:  # rho tol breakdown on host
            return x, -10

        if iters > 0:
            if abs(omega) < omegatol:  # omega tol breakdown
                return x, -11

            beta = (rho / rho_prev) * (alpha / omega)
            p -= omega * v
            p *= beta
            p += r
        else:
            p = r.copy()

        phat = psolve(p)
        v = matvec(phat)
        rv = dotprod(rtilde, v).item()  # pull to host

        if rv == 0:
            return x, -11

        alpha = rho / rv
        r -= alpha * v

        # does a half-step check
        if cupy.linalg.norm(r) <= atol:
            x += alpha * phat
            break

        shat = psolve(r)
        t = matvec(shat)

        omega = dotprod(t, r).item() / dotprod(t, t).item()

        # host scalar times device array
        x += alpha * phat
        x += omega * shat
        r -= omega * t

        rho_prev = rho
        iters += 1

        if callback is not None:
            callback(x)

    info = 0
    # If the loop maxed out and it didn't converge,
    # return iter count as error code
    if iters >= maxiter and not (r_norm <= atol):
        info = iters

    return x, info

#########################################
