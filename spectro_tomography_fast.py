"""Accelerated sparse spectro-tomography; NumPy/SciPy or CuPy/CUDA.

Objective, separately for each slice:
    0.5 * ||sum_s diag(ref[:, s]) P x_s - b||_2**2
    + mu1 * sum_s (||Dx x_s||_1 + ||Dy x_s||_1).
Optional x >= 0. Differences are within each slice; there is no z-axis TV.

P is the ASTRA 'linear' projector exported once to CSR (or a supplied CSR).
The actual transpose is used, and F.T@F is never assembled. Iterative arrays
and sparse products stay on the selected device. CuPy's GMRES implementation
may itself use the CPU for its small Hessenberg least-squares problem.

Public API:
    build_astra_projector, Reconstructor, multistate_tomo_joint_TV,
    reconstruct_volume, run_trial (optional spectro_agent worker adapter).

CPU requirements: numpy, scipy, threadpoolctl; ASTRA to build the projector.
GPU additionally requires a compatible CuPy installation and NVIDIA device.
See README_FAST.md for usage, benchmarking, and numerical compatibility.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import inspect
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from tqdm import tqdm


def _positive_int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _shape(value):
    if len(value) != 2:
        raise ValueError("obj_size must contain (rows, columns)")
    return tuple(_positive_int(v, "obj_size entry", 2) for v in value)


def build_astra_projector(angle, obj_size, detector_count, rot_cen_offset=0.0,
                          detector_spacing=1.0):
    """Return a CPU CSR matrix in angle-major/detector-minor ordering.

    Pixel size is one, matching ASTRA create_vol_geom(rows, columns).
    Positive rot_cen_offset shifts the detector along its positive coordinate
    by that many detector pixels (ASTRA geom_postalignment convention).
    Nonzero offsets should be checked against the installed original solver.
    Save/reuse this CSR across trials/plans with identical angular geometry.
    """
    shape = _shape(obj_size)
    detector_count = _positive_int(detector_count, "detector_count")
    angle = np.asarray(angle, dtype=np.float32)
    if angle.ndim != 1 or not angle.size or not np.all(np.isfinite(angle)):
        raise ValueError("angle must be a finite, nonempty vector in radians")
    if not np.isfinite(rot_cen_offset) or not np.isfinite(detector_spacing) or detector_spacing <= 0:
        raise ValueError("Invalid detector spacing or rotation-center offset")
    try:
        import astra
    except ImportError as exc:
        raise ImportError("Install astra-toolbox, or supply an existing scipy CSR projector") from exc
    pg = astra.create_proj_geom("parallel", float(detector_spacing), detector_count, angle)
    if rot_cen_offset:
        pg = astra.geom_postalignment(pg, float(rot_cen_offset))
    vg = astra.create_vol_geom(*shape)
    pid = mid = None
    try:
        pid = astra.create_projector("linear", pg, vg)
        mid = astra.projector.matrix(pid)
        matrix = astra.matrix.get(mid).tocsr(copy=True)
    finally:
        if mid is not None:
            astra.matrix.delete(mid)
        if pid is not None:
            astra.projector.delete(pid)
    return matrix


def _gradient(x, xp, boundary):
    if boundary == "periodic":
        return xp.stack((xp.roll(x, -1, axis=-1)-x,
                         xp.roll(x, -1, axis=-2)-x), axis=1)
    g = xp.zeros((x.shape[0], 2, *x.shape[-2:]), dtype=x.dtype)
    g[:, 0, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    g[:, 1, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    return g


def _gradient_adjoint(p, xp, boundary):
    if boundary == "periodic":
        return (xp.roll(p[:, 0], 1, axis=-1)-p[:, 0]
                + xp.roll(p[:, 1], 1, axis=-2)-p[:, 1])
    out = xp.zeros((p.shape[0], *p.shape[-2:]), dtype=p.dtype)
    out[:, :, :-1] -= p[:, 0, :, :-1]
    out[:, :, 1:] += p[:, 0, :, :-1]
    out[:, :-1, :] -= p[:, 1, :-1, :]
    out[:, 1:, :] += p[:, 1, :-1, :]
    return out


def _laplacian(x, xp, boundary):
    if boundary == "periodic":
        return (4*x - xp.roll(x, 1, -1) - xp.roll(x, -1, -1)
                - xp.roll(x, 1, -2) - xp.roll(x, -1, -2))
    out = xp.zeros_like(x)
    dx, dy = x[:, :, 1:]-x[:, :, :-1], x[:, 1:, :]-x[:, :-1, :]
    out[:, :, :-1] -= dx
    out[:, :, 1:] += dx
    out[:, :-1, :] -= dy
    out[:, 1:, :] += dy
    return out


def _degree(shape, boundary):
    degree = np.full(shape, 4.0)
    if boundary == "neumann":
        degree[0] -= 1
        degree[-1] -= 1
        degree[:, 0] -= 1
        degree[:, -1] -= 1
    return degree


class Reconstructor:
    """Prepared geometry/codebook, reused for slices and random-start trials.

    backend='cpu' or 'cupy'; dtype='float32' for numerical comparisons, or
    'float32' for a potentially faster, smaller GPU working set. P and ref are
    fixed for this instance. If an acquisition plan changes, create a new
    instance reusing the same P when angles/detector geometry are unchanged.
    """

    def __init__(self, projector, ref, obj_size, *, backend="cpu", gpu_id=0,
                 dtype="float32", boundary="neumann"):
        self.shape = _shape(obj_size)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        if backend not in {"cpu", "gpu"}:
            raise ValueError("backend must be cpu or gpu; there is no silent fallback")
        if boundary not in {"neumann", "periodic"}:
            raise ValueError("boundary must be neumann or periodic")
        weights = np.asarray(ref, dtype=self.dtype)
        if weights.ndim != 2 or min(weights.shape) < 1 or not np.all(np.isfinite(weights)):
            raise ValueError("ref must be a finite (angles, states) matrix")
        if not sparse.issparse(projector):
            raise TypeError("projector must be a scipy sparse matrix")
        p = sparse.csr_matrix(projector, dtype=self.dtype, copy=True)
        p.sum_duplicates()
        p.sort_indices()
        if p.shape[1] != np.prod(self.shape) or p.shape[0] % len(weights):
            raise ValueError("Projector dimensions do not match obj_size/ref")
        if not p.shape[0] or not np.all(np.isfinite(p.data)):
            raise ValueError("Projector must have finite entries and nonempty rows")
        self.backend, self.boundary = backend, boundary
        self.gpu_id = _positive_int(gpu_id, "gpu_id", 0)
        self.states = weights.shape[1]
        self.sino_shape = (len(weights), p.shape[0] // len(weights))
        self.image_shape = (self.states, *self.shape)
        self.ref = weights.copy()
        self.projector_bytes = int(p.data.nbytes + p.indices.nbytes + p.indptr.nbytes)
        self.device = None
        self.xp, self.la = np, spla
        if backend == "gpu":
            try:
                import cupy as cp
                from cupyx.scipy import sparse as cpsparse
                from cupyx.scipy.sparse import linalg as cpla
                self.device = cp.cuda.Device(self.gpu_id)
                with self.device:
                    cp.zeros(1).sum().item()  # Fail clearly if the device/runtime cannot be used.
            except Exception as exc:
                raise RuntimeError(f"CuPy/CUDA device {self.gpu_id} is unavailable: {exc}") from exc
            self.xp, self.la = cp, cpla
        xp = self.xp
        # These CPU setup calculations happen once, outside iterative solves.
        w = np.repeat(weights, self.sino_shape[1], axis=0)
        degree = _degree(self.shape, boundary)
        diag_data = np.asarray(p.power(2).T @ (w*w)).T.reshape(self.image_shape)
        abs_p = abs(p)
        colsum = np.asarray(abs_p.T @ np.abs(w)).T.reshape(self.image_shape)
        rowsum = np.asarray(abs_p.sum(axis=1)).ravel() * np.abs(w).sum(axis=1)
        # ker(D) consists of a constant image per state. Check that F observes it.
        constant_response = (p @ np.ones(p.shape[1], dtype=self.dtype))[:, None] * w
        self.constant_rank = int(np.linalg.matrix_rank(constant_response))
        with self._context():
            self.p = p if backend == "cpu" else cpsparse.csr_matrix(p)
            self.pt = self.p.T.tocsr()
            self.w = xp.asarray(w)
            self.diag_data = xp.asarray(diag_data)
            self.degree = xp.asarray(degree, dtype=self.dtype)
            # Diagonal PDHG preconditioning (absolute row/column sums of [F; D]).
            # Zero data rows do not affect x; a unit denominator is harmless there.
            self.primal_denom = xp.asarray(colsum + degree[None])
            self.data_dual_denom = xp.asarray(np.where(rowsum > 0, rowsum, 1.0)).reshape(self.sino_shape)
        self._normal_cache = {}

    def _context(self):
        return self.device if self.device is not None else nullcontext()

    def _numpy(self, x):
        return self.xp.asnumpy(x) if self.backend == "gpu" else np.asarray(x)

    def _sync(self):
        if self.backend == "gpu":
            self.xp.cuda.get_current_stream().synchronize()

    def _forward(self, x):
        return self.xp.sum((self.p @ x.reshape(self.states, -1).T) * self.w,
                           axis=1).reshape(self.sino_shape)

    def _adjoint(self, y):
        return (self.pt @ (y.reshape(-1, 1)*self.w)).T.reshape(self.image_shape)

    def project(self, images):
        """Forward-project one state stack and return a NumPy sinogram."""
        a = np.asarray(images)
        if a.shape != self.image_shape or not np.all(np.isfinite(a)):
            raise ValueError(f"images must be finite with shape {self.image_shape}")
        with self._context():
            return self._numpy(self._forward(self.xp.asarray(a, dtype=self.dtype)))

    def _objective(self, x, b, tv, l1=0.0, *, use_tv=True, use_l1=False):
        residual = self._forward(x)-b
        value = 0.5*self.xp.sum(residual*residual)
        if use_tv:
            g = _gradient(x, self.xp, self.boundary)
            value += tv*self.xp.sum(self.xp.abs(g))
        if use_l1:
            value += l1*self.xp.sum(self.xp.abs(x))
        return float(value)

    def _normal(self, rho, positive, use_l1=False, use_tv=True):
        key = (rho, positive, use_l1, use_tv)
        if key not in self._normal_cache:
            xp = self.xp
            def matvec(v):
                x = v.reshape(self.image_shape)
                out = (self._adjoint(self._forward(x)) + rho*_laplacian(x, xp, self.boundary)
                       if use_tv else self._adjoint(self._forward(x)))
                if positive:
                    out += rho*x
                if use_l1:
                    out += rho*x
                return out.ravel()
            diag = self.diag_data + (rho*self.degree[None] if use_tv else 0.0)
            diag += (rho if positive else 0.0) + (rho if use_l1 else 0.0)
            inverse = 1/xp.maximum(diag, xp.finfo(self.dtype).tiny)
            size = int(np.prod(self.image_shape))
            a = self.la.LinearOperator((size, size), matvec=matvec, dtype=self.dtype)
            m = self.la.LinearOperator((size, size), matvec=lambda v: inverse.ravel()*v, dtype=self.dtype)
            self._normal_cache[key] = (a, m)
        return self._normal_cache[key]

    def _krylov(self, a, m, rhs, x, method, rtol, atol, maxiter, restart):
        # Solve a correction with x0=0, so warm starts also work with older
        # CuPy right-preconditioned GMRES implementations.
        xp = self.xp
        flat = x.ravel()
        residual = rhs.ravel() - a @ flat
        target = max(atol, rtol*float(xp.linalg.norm(rhs)))
        if float(xp.linalg.norm(residual)) <= target:
            return x, 0, float(xp.linalg.norm(residual)), target
        function = getattr(self.la, method)
        parameters = inspect.signature(function).parameters
        options = {"rtol" if "rtol" in parameters else "tol": 0.0,
                   "atol": target, "maxiter": maxiter, "M": m}
        if method == "gmres":
            options["restart"] = min(restart, maxiter)
            if self.backend == "cpu":
                # SciPy otherwise counts restart cycles, not inner iterations.
                options.update(callback=lambda _: None, callback_type="legacy")
        delta, info = function(a, residual, **options)
        out = flat + delta
        true_residual = float(xp.linalg.norm(rhs.ravel() - a @ out))
        if info < 0 or not bool(xp.all(xp.isfinite(out))):
            raise RuntimeError(f"{method} failed: info={info}, residual={true_residual}")
        # Keep actual residuals even if the library reports convergence differently.
        return out.reshape(self.image_shape), int(info), true_residual, target

    def reconstruct(self, sinogram, *, mu1=0.1, mu2=1.0, max_iter=100,
                    x0=None, seed=0, algorithm="admm", method="cg",
                    nonnegative=False, regularization="TV", mu3=None,
                    inner_rtol=1e-5, inner_atol=0.0,
                    inner_maxiter=100, gmres_restart=30, tol=0.0,
                    log_every=10, seq_save=False, pdhg_safety=0.95,
                    show_progress=False):
        """Return (images, diagnostics); images are NumPy (states, rows, cols).

        algorithm='admm': split TV, Jacobi-preconditioned CG or GMRES.
        algorithm='pdhg': diagonally preconditioned Chambolle-Pock, exact TV;
          no inner solver and no smoothing approximation. mu2 is unused.
        tol=0 means a fixed iteration count, useful for controlled experiments.
        tol>0 enables residual stopping checked at log_every and the last step.
        Nonnegativity uses an additional ADMM split, not clipped CG iterates.
        seq_save returns a host copy after every step in diagnostics['sequence'].
        show_progress displays outer-iteration progress; volume reconstruction
        disables it so only the slice-level progress is shown.
        """
        b = np.asarray(sinogram, dtype=self.dtype)
        if b.shape != self.sino_shape or not np.all(np.isfinite(b)):
            raise ValueError(f"sinogram must be finite with shape {self.sino_shape}")
        if algorithm not in {"admm", "pdhg"} or method not in {"cg", "gmres"}:
            raise ValueError("Use algorithm=admm/pdhg and method=cg/gmres")
        if regularization not in {"TV", "L1", "TV+L1"}:
            raise ValueError("regularization must be TV, L1, or TV+L1")
        use_tv = regularization in {"TV", "TV+L1"}
        use_l1 = regularization in {"L1", "TV+L1"}
        if mu3 is not None and (not np.isfinite(mu3) or mu3 < 0):
            raise ValueError("mu3 must be finite and nonnegative")
        if use_l1 and mu3 is None:
            raise ValueError("mu3 is required when regularization includes L1")
        l1_weight = float(mu3) if use_l1 else 0.0
        for name, value in (("mu1", mu1), ("inner_rtol", inner_rtol),
                            ("inner_atol", inner_atol), ("tol", tol)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(mu2) or mu2 <= 0:
            raise ValueError("mu2 must be finite and positive")
        if inner_rtol == inner_atol == 0 and algorithm == "admm":
            raise ValueError("At least one inner tolerance must be positive")
        if not 0 < pdhg_safety < 1:
            raise ValueError("pdhg_safety must lie strictly between zero and one")
        max_iter = _positive_int(max_iter, "max_iter")
        log_every = _positive_int(log_every, "log_every")
        inner_maxiter = _positive_int(inner_maxiter, "inner_maxiter")
        gmres_restart = _positive_int(gmres_restart, "gmres_restart")
        if algorithm == "admm" and method == "cg" and not nonnegative and self.constant_rank < self.states:
            raise ValueError("CG normal matrix is not guaranteed positive definite: constant states are unobservable; use PDHG or GMRES")
        initial = (np.random.default_rng(seed).random(self.image_shape) if x0 is None
                   else np.asarray(x0))
        if initial.shape != self.image_shape or not np.all(np.isfinite(initial)):
            raise ValueError(f"x0 must be finite with shape {self.image_shape}")
        with self._context():
            xp = self.xp
            x = xp.array(initial, dtype=self.dtype, copy=True)
            b = xp.asarray(b)
            if nonnegative:
                x = xp.maximum(x, 0)
            self._sync()
            start = time.perf_counter()
            diagnostics = {"algorithm": algorithm, "method": method if algorithm == "admm" else None,
                           "backend": self.backend, "gpu_id": self.gpu_id if self.backend == "gpu" else None,
                           "regularization": regularization, "mu3": l1_weight if use_l1 else None,
                           "dtype": self.dtype.name, "boundary": self.boundary,
                           "projector_bytes": self.projector_bytes, "history": [],
                           "inner_info_codes": [], "inner_residuals": [], "inner_targets": [],
                           "converged": False, "stopping_reason": "max_iter", "pid": os.getpid()}
            if seq_save:
                diagnostics["sequence"] = []
            if algorithm == "admm":
                out = self._admm(x, b, mu1, mu2, max_iter, method, nonnegative,
                                 use_tv, use_l1, l1_weight,
                                 inner_rtol, inner_atol, inner_maxiter, gmres_restart,
                                 tol, log_every, diagnostics, show_progress)
            else:
                out = self._pdhg(x, b, mu1, max_iter, nonnegative, use_tv,
                                 use_l1, l1_weight, pdhg_safety,
                                 tol, log_every, diagnostics, show_progress)
            if not bool(xp.all(xp.isfinite(out))):
                raise RuntimeError("Nonfinite reconstruction")
            diagnostics["final_objective"] = self._objective(
                out, b, mu1, l1_weight, use_tv=use_tv, use_l1=use_l1)
            diagnostics["data_residual_mse"] = float(xp.mean((self._forward(out)-b)**2))
            diagnostics["inner_nonconverged_steps"] = sum(
                info != 0 or residual > target*1.05
                for info, residual, target in zip(diagnostics["inner_info_codes"],
                                                 diagnostics["inner_residuals"], diagnostics["inner_targets"]))
            images = self._numpy(out).copy()
            self._sync()
            diagnostics["solve_seconds"] = time.perf_counter()-start
            return images, diagnostics

    def _admm(self, x, b, tv, rho, iterations, method, positive,
              use_tv, use_l1, l1_weight,
              rtol, atol, inner_maxiter, restart, tol, log_every, d, show_progress):
        xp = self.xp
        a, preconditioner = self._normal(rho, positive, use_l1, use_tv)
        rhs_data = self._adjoint(b)
        z = xp.zeros((self.states, 2, *self.shape), dtype=self.dtype) if use_tv else None
        u = xp.zeros_like(z) if use_tv else None
        v, up = (x.copy(), xp.zeros_like(x)) if positive else (None, None)
        s, us = (x.copy(), xp.zeros_like(x)) if use_l1 else (None, None)
        iteration_range = range(1, iterations+1)
        if show_progress:
            iteration_range = tqdm(iteration_range, total=iterations,
                                   desc="Reconstruction iterations")
        for iteration in iteration_range:
            rhs = (rhs_data + rho*_gradient_adjoint(z-u, xp, self.boundary)
                   if use_tv else rhs_data.copy())
            if positive:
                rhs += rho*(v-up)
            if use_l1:
                rhs += rho*(s-us)
            x, info, residual, target = self._krylov(a, preconditioner, rhs, x, method,
                                                   rtol, atol, inner_maxiter, restart)
            d["inner_info_codes"].append(info)
            d["inner_residuals"].append(residual)
            d["inner_targets"].append(target)
            if use_tv:
                gx = _gradient(x, xp, self.boundary)
                old_z = z
                temporary = gx+u
                z = xp.sign(temporary)*xp.maximum(xp.abs(temporary)-tv/rho, 0)
                u += gx-z
            if positive:
                old_v = v
                v = xp.maximum(x+up, 0)
                up += x-v
            if use_l1:
                old_s = s
                temporary = x+us
                s = xp.sign(temporary)*xp.maximum(xp.abs(temporary)-l1_weight/rho, 0)
                us += x-s
            out = v if positive else x
            if "sequence" in d:
                d["sequence"].append(self._numpy(out).copy())
            if iteration % log_every == 0 or iteration == iterations:
                if not use_l1:
                    primal2 = xp.sum((gx-z)**2)
                    dual = _gradient_adjoint(z-old_z, xp, self.boundary)
                    multiplier = _gradient_adjoint(u, xp, self.boundary)
                    constraint2, split2 = xp.sum(gx*gx), xp.sum(z*z)
                    count = z.size
                    if positive:
                        primal2 += xp.sum((x-v)**2)
                        dual += v-old_v
                        multiplier += up
                        constraint2 += xp.sum(x*x)
                        split2 += xp.sum(v*v)
                        count += x.size
                else:
                    primal2 = xp.asarray(0.0, dtype=x.dtype)
                    dual, multiplier = xp.zeros_like(x), xp.zeros_like(x)
                    constraint2 = split2 = xp.asarray(0.0, dtype=x.dtype)
                    count = 0
                    if use_tv:
                        primal2 += xp.sum((gx-z)**2)
                        dual += _gradient_adjoint(z-old_z, xp, self.boundary)
                        multiplier += _gradient_adjoint(u, xp, self.boundary)
                        constraint2 += xp.sum(gx*gx)
                        split2 += xp.sum(z*z)
                        count += z.size
                    if positive:
                        primal2 += xp.sum((x-v)**2)
                        dual += v-old_v
                        multiplier += up
                        constraint2 += xp.sum(x*x)
                        split2 += xp.sum(v*v)
                        count += x.size
                    if use_l1:
                        primal2 += xp.sum((x-s)**2)
                        dual += s-old_s
                        multiplier += us
                        constraint2 += xp.sum(x*x)
                        split2 += xp.sum(s*s)
                        count += x.size
                primal = float(xp.sqrt(primal2))
                dual = rho*float(xp.linalg.norm(dual))
                eps_primal = tol*(np.sqrt(count)+max(float(xp.sqrt(constraint2)), float(xp.sqrt(split2))))
                eps_dual = tol*(np.sqrt(x.size)+rho*float(xp.linalg.norm(multiplier)))
                objective = self._objective(out, b, tv, l1_weight,
                                            use_tv=use_tv, use_l1=use_l1)
                if not np.isfinite(objective):
                    raise RuntimeError("Nonfinite ADMM objective")
                d["history"].append({"iteration": iteration, "objective": objective,
                                     "primal_residual": primal, "dual_residual": dual})
                if tol > 0 and primal <= eps_primal and dual <= eps_dual and info == 0 and residual <= target*1.05:
                    d.update(converged=True, stopping_reason="admm_residuals")
                    break
        d["iterations"] = iteration
        return out

    def _pdhg(self, x, b, tv, iterations, positive, use_tv, use_l1,
              l1_weight, safety, tol, log_every, d, show_progress):
        xp = self.xp
        primal_denom = (self.primal_denom if use_tv else
                self.xp.maximum(self.primal_denom-self.degree[None], 1.0))
        tau = safety/primal_denom
        sigma_data = safety/self.data_dual_denom
        sigma_tv = safety/2.0
        p = xp.zeros((self.states, 2, *self.shape), dtype=self.dtype)
        q = xp.zeros_like(b)
        extrapolated = x.copy()
        scale_data = max(1.0, float(xp.linalg.norm(b)))
        scale_stationarity = max(1.0, float(xp.linalg.norm(self._adjoint(b))))
        iteration_range = range(1, iterations+1)
        if show_progress:
            iteration_range = tqdm(iteration_range, total=iterations,
                                   desc="Reconstruction iterations")
        for iteration in iteration_range:
            q = (q + sigma_data*(self._forward(extrapolated)-b))/(1+sigma_data)
            if use_tv:
                p = xp.clip(p + sigma_tv*_gradient(extrapolated, xp, self.boundary), -tv, tv)
            gradient = self._adjoint(q)
            if use_tv:
                gradient += _gradient_adjoint(p, xp, self.boundary)
            new = x - tau*gradient
            if use_l1 and positive:
                new = xp.maximum(new-tau*l1_weight, 0)
            elif use_l1:
                new = xp.sign(new)*xp.maximum(xp.abs(new)-tau*l1_weight, 0)
            elif positive:
                new = xp.maximum(new, 0)
            extrapolated, x = 2*new-x, new
            if "sequence" in d:
                d["sequence"].append(self._numpy(x).copy())
            if iteration % log_every == 0 or iteration == iterations:
                prediction = self._forward(x)
                gx = _gradient(x, xp, self.boundary)
                gradient = self._adjoint(q)+_gradient_adjoint(p, xp, self.boundary)
                projected = x-tau*gradient
                if use_l1 and positive:
                    projected = xp.maximum(projected-tau*l1_weight, 0)
                elif use_l1:
                    projected = xp.sign(projected)*xp.maximum(xp.abs(projected)-tau*l1_weight, 0)
                elif positive:
                    projected = xp.maximum(projected, 0)
                stationarity = float(xp.linalg.norm((x-projected)/tau))/scale_stationarity
                data_dual = float(xp.linalg.norm(q-(prediction-b)))/scale_data
                tv_dual = float(xp.linalg.norm((p-xp.clip(p+sigma_tv*gx, -tv, tv))/sigma_tv)) / max(1.0, float(xp.linalg.norm(gx)))
                objective = self._objective(x, b, tv, l1_weight,
                                            use_tv=use_tv, use_l1=use_l1)
                if not np.isfinite(objective):
                    raise RuntimeError("Nonfinite PDHG objective")
                d["history"].append({"iteration": iteration, "objective": objective,
                                     "stationarity": stationarity, "data_dual_residual": data_dual,
                                     "tv_dual_residual": tv_dual})
                if tol > 0 and max(stationarity, data_dual, tv_dual) <= tol:
                    d.update(converged=True, stopping_reason="pdhg_kkt_residuals")
                    break
        d["iterations"] = iteration
        return x


def multistate_tomo_joint_TV(sinogram, angle, rot_cen_offset, obj_size, ref,
                           mu1, mu2, max_iter, x0=None, nonnegative=False,
                           seq_save=False, method="cg", *, backend="cpu", gpu_id=0,
                           algorithm="admm", regularization="TV", mu3=None,
                           dtype="float32", boundary="neumann",
                           projector=None, reconstructor=None, rtol=1e-5, atol=0.0,
                           maxiter=100, restart=30, tol=0.0, log_every=1, seed=0,
                           return_diagnostics=False):
    """Driver-compatible entry point for the supplied Simulation.py.

    Returns (list_of_state_images, objective_history). With seq_save=True the
    first return is the per-iteration sequence, so rec[-1][state] still works.
    Supply a prepared reconstructor for repeated trials; that object's geometry,
    codebook, backend, precision and boundary take precedence over setup options.
    'pdhg' is accepted as a convenient method alias for algorithm='pdhg'.

    Parameters
    ----------
    sinogram : array_like, shape (angles, detectors)
        Measured projection data for one slice.
    angle : array_like, shape (angles,)
        Projection angles in radians.
    rot_cen_offset : float
        ASTRA detector alignment offset used when constructing the projector.
    obj_size : sequence of int
        Reconstructed image shape (rows, columns).
    ref : array_like, shape (angles, states)
        Spectral/state weights for each projection angle.
    mu1 : float
        Total-variation penalty weight; also used by the PDHG objective.
    mu2 : float
        ADMM penalty parameter. Unused by PDHG.
    mu3 : float, optional
        L1 penalty weight. Required when regularization includes 'L1'.
    max_iter : int
        Maximum number of outer ADMM or PDHG iterations.
    x0 : array_like, shape (states, rows, columns), optional
        Initial image estimate. Random initialization is used when omitted.
    nonnegative : bool, default False
        Constrain reconstructed images to be nonnegative.
    seq_save : bool, default False
        Return every iteration's image instead of only the final image.
    method : {'cg', 'gmres', 'pdhg'}, default 'cg'
        ADMM inner linear solver. 'pdhg' is an alias for algorithm='pdhg'.
    backend : {'cpu', 'gpu'}, default 'cpu'
        Array and sparse-operation backend.
    gpu_id : int, default 0
        CUDA device index when backend='gpu'.
    algorithm : {'admm', 'pdhg'}, default 'admm'
        Optimization algorithm.
    regularization : {'TV', 'L1', 'TV+L1'}, default 'TV'
        Penalize image gradients, image absolute values, or both.
    dtype : {'float32', 'float64'}, default 'float32'
        Floating-point precision used by the solver.
    boundary : {'neumann', 'periodic'}, default 'neumann'
        Boundary condition for total-variation differences.
    projector : scipy sparse matrix, optional
        System matrix. Built with ASTRA when omitted and no reconstructor is
        supplied.
    reconstructor : Reconstructor, optional
        Prepared solver instance reused instead of constructing a new one.
    rtol : float, default 1e-5
        Relative tolerance for each ADMM inner solve.
    atol : float, default 0.0
        Absolute tolerance for each ADMM inner solve.
    maxiter : int, default 100
        Maximum iterations for each ADMM inner solve.
    restart : int, default 30
        Restart length for GMRES inner solves.
    tol : float, default 0.0
        Outer stopping tolerance; zero runs exactly max_iter iterations.
    log_every : int, default 1
        Record diagnostics every this many outer iterations.
    seed : int, default 0
        Seed for random image initialization when x0 is omitted.
    return_diagnostics : bool, default False
        Also return the full solver diagnostics dictionary.

    Returns
    -------
    rec : list of ndarray
        Final image per state, or iteration-major image sequences if seq_save.
    errors : list of float
        Recorded objective values.
    diagnostics : dict, optional
        Full solver diagnostics, returned only when return_diagnostics=True.
    """
    sinogram = np.asarray(sinogram)
    if sinogram.ndim != 2:
        raise ValueError("For a volume use reconstruct_volume with (slices, angles, detectors)")
    angle = np.asarray(angle)
    if angle.shape != (sinogram.shape[0],):
        raise ValueError("angle length must match the sinogram")
    if method == "pdhg":
        algorithm, method = "pdhg", "cg"
    if reconstructor is None:
        if projector is None:
            projector = build_astra_projector(angle, obj_size, sinogram.shape[1], rot_cen_offset)
        reconstructor = Reconstructor(projector, ref, obj_size, backend=backend,
                                      gpu_id=gpu_id, dtype=dtype, boundary=boundary)
    elif reconstructor.shape != _shape(obj_size) or not np.array_equal(reconstructor.ref, np.asarray(ref, dtype=reconstructor.dtype)):
        raise ValueError("Prepared reconstructor does not match obj_size/ref")
    images, details = reconstructor.reconstruct(
        sinogram, mu1=mu1, mu2=mu2, max_iter=max_iter, x0=x0, seed=seed,
        algorithm=algorithm, method=method, nonnegative=nonnegative,
        regularization=regularization, mu3=mu3,
        inner_rtol=rtol, inner_atol=atol, inner_maxiter=maxiter,
        gmres_restart=restart, tol=tol, log_every=log_every, seq_save=seq_save,
        show_progress=True)
    rec = details.pop("sequence") if seq_save else list(images)
    errors = [h["objective"] for h in details["history"]]
    return (rec, errors, details) if return_diagnostics else (rec, errors)


_WORKER_ENGINE = _WORKER_SINOS = _WORKER_INITIAL = _WORKER_LIMIT = None
_WORKER_OPTIONS = _WORKER_SEEDS = None


def _initialize_worker(directory, matrix_shape, ref, obj_size, backend, gpu_id,
                       dtype, boundary, options, seeds, threads):
    global _WORKER_ENGINE, _WORKER_SINOS, _WORKER_INITIAL, _WORKER_LIMIT
    global _WORKER_OPTIONS, _WORKER_SEEDS
    from threadpoolctl import threadpool_limits
    _WORKER_LIMIT = threadpool_limits(limits=threads)
    path = Path(directory)
    data, indices, indptr = (np.load(path / (key+".npy"), mmap_mode="r", allow_pickle=False)
                             for key in ("data", "indices", "indptr"))
    matrix = sparse.csr_matrix((data, indices, indptr), shape=matrix_shape, copy=False)
    _WORKER_ENGINE = Reconstructor(matrix, ref, obj_size, backend=backend,
                                  gpu_id=gpu_id, dtype=dtype, boundary=boundary)
    _WORKER_SINOS = np.load(path / "sinograms.npy", mmap_mode="r", allow_pickle=False)
    _WORKER_INITIAL = (np.load(path / "initial.npy", mmap_mode="r", allow_pickle=False)
                       if (path / "initial.npy").exists() else None)
    _WORKER_OPTIONS, _WORKER_SEEDS = options, seeds


def _reconstruct_chunk(indices):
    result = []
    for i in indices:
        initial = None if _WORKER_INITIAL is None else _WORKER_INITIAL[:, i]
        rec, info = _WORKER_ENGINE.reconstruct(_WORKER_SINOS[i], x0=initial,
                                              seed=_WORKER_SEEDS[i], **_WORKER_OPTIONS)
        result.append((i, rec, info))
    return result


def reconstruct_volume(sinograms, angle, ref, obj_size, *, projector=None,
                       rot_cen_offset=0.0, backend="cpu", gpu_ids=(0,),
                       workers=None, dtype="float32", boundary="neumann",
                       x0=None, seed=0, threads_per_worker=1,
                       temp_dir=None, output_path=None, **solve_options):
    """Reconstruct independent slices concurrently, using spawned processes.

    Input: (slices, angles, detectors); output: (states, slices, rows, columns).
    x0, if given, has output shape. Slice seeds depend only on seed and slice
    index, never worker count/order. CPU workers default to 1. GPU workers
    default to len(gpu_ids); when workers exceed the selected GPU count, GPUs
    are assigned round-robin. Each worker retains its own prepared operator.
    Read-only input/CSR files are memory mapped, rather than pickled per slice.
    Each worker still owns its working matrix and transpose: allow memory for
    both matrices and solver vectors per worker. output_path optionally creates
    a .npy memmap. No cross-slice TV is applied.

    Put calls inside if __name__ == '__main__': when workers > 1. Run these calls
    from an importable .py script, not directly in a notebook cell.

    Parameters
    ----------
    sinograms : array_like, shape (slices, angles, detectors)
        Measured projection data for all slices.
    angle : array_like, shape (angles,)
        Projection angles in radians.
    ref : array_like, shape (angles, states)
        Spectral/state weights for each projection angle.
    obj_size : sequence of int
        Reconstructed image shape (rows, columns).
    projector : scipy sparse matrix, optional
        CSR-compatible system matrix with shape
        (angles * detectors, rows * columns). Built with ASTRA when omitted.
    rot_cen_offset : float, default 0.0
        ASTRA detector alignment offset; used only when building the projector.
    backend : {'cpu', 'gpu'}, default 'cpu'
        Array and sparse-operation backend.
    gpu_ids : sequence of int, default (0,)
        Distinct GPU IDs available to workers. Workers are assigned round-robin;
        the number of IDs is also the default GPU worker count.
    workers : int, optional
        Number of slice workers. Defaults to one on CPU or len(gpu_ids) on GPU,
        then is capped at the number of slices.
    dtype : {'float32', 'float64'}, default 'float32'
        Floating-point precision for reconstruction and output.
    boundary : {'neumann', 'periodic'}, default 'neumann'
        Boundary condition for the in-slice TV differences.
    x0 : array_like, shape (states, slices, rows, columns), optional
        Initial image estimate for every slice.
    seed : int, default 0
        Seed used to generate deterministic per-slice initialization seeds.
    threads_per_worker : int, default 1
        BLAS thread limit applied independently in each worker.
    temp_dir : path-like, optional
        Parent directory for temporary memory-mapped worker inputs.
    output_path : path-like, optional
        New .npy file for a memory-mapped output volume. Must not already exist.
    **solve_options
        Options forwarded to Reconstructor.reconstruct, including algorithm,
        method, mu1, mu2, regularization ('TV', 'L1', or 'TV+L1'), mu3 when L1
        is included, nonnegative, iteration limits, tolerances, and log_every.

    Returns
    -------
    output : ndarray or numpy.memmap, shape (states, slices, rows, columns)
        Reconstructed volume.
    metadata : dict
        Worker/backend details, per-slice diagnostics, timing, and slice seeds.
    """
    shape = _shape(obj_size)
    data = np.asarray(sinograms)
    ref, angle = np.asarray(ref), np.asarray(angle)
    if data.ndim != 3 or not data.shape[0] or not np.all(np.isfinite(data)):
        raise ValueError("sinograms must be finite with shape (slices, angles, detectors)")
    if ref.ndim != 2 or ref.shape[0] != data.shape[1] or angle.shape != (data.shape[1],):
        raise ValueError("Angles and reference rows must match sinograms")
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be cpu or gpu")
    if solve_options.get("seq_save", False):
        raise ValueError("Volume seq_save is disabled to avoid retaining every volume iterate")
    solve_options["show_progress"] = False
    gpu_ids = tuple(_positive_int(v, "gpu_id", 0) for v in gpu_ids)
    if backend == "gpu" and (not gpu_ids or len(set(gpu_ids)) != len(gpu_ids)):
        raise ValueError("Supply distinct gpu_ids for the GPUs to use")
    workers = _positive_int(workers if workers is not None else (len(gpu_ids) if backend == "gpu" else 1), "workers")
    workers = min(workers, len(data))
    worker_gpu_ids = [gpu_ids[i % len(gpu_ids)] for i in range(workers)] if backend == "gpu" else []
    threads_per_worker = _positive_int(threads_per_worker, "threads_per_worker")
    result_shape = (ref.shape[1], len(data), *shape)
    if x0 is not None:
        x0 = np.asarray(x0)
        if x0.shape != result_shape or not np.all(np.isfinite(x0)):
            raise ValueError(f"x0 must be finite with shape {result_shape}")
    if projector is None:
        projector = build_astra_projector(angle, shape, data.shape[2], rot_cen_offset)
    projector = sparse.csr_matrix(projector)
    if projector.shape != (data.shape[1]*data.shape[2], int(np.prod(shape))):
        raise ValueError("Projector dimensions do not match volume")
    if output_path is None:
        output = np.empty(result_shape, dtype=dtype)
    else:
        target = Path(output_path)
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(target, mode="w+", dtype=dtype, shape=result_shape)
    seeds = [int(s.generate_state(1, dtype=np.uint64)[0])
             for s in np.random.SeedSequence(seed).spawn(len(data))]
    details = [None]*len(data)
    start = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ImportError("Install threadpoolctl for volume thread control") from exc
    if workers == 1:
        with threadpool_limits(limits=threads_per_worker):
            engine = Reconstructor(projector, ref, shape, backend=backend,
                                   gpu_id=worker_gpu_ids[0] if backend == "gpu" else 0,
                                   dtype=dtype, boundary=boundary)
            for i in tqdm(range(len(data)), desc="Reconstructing slices"):
                output[:, i], details[i] = engine.reconstruct(
                    data[i], x0=None if x0 is None else x0[:, i], seed=seeds[i], **solve_options)
    else:
        # Explicit spawn avoids inheriting a CUDA context or BLAS thread state.
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="spectro-workers-", dir=temp_dir) as directory:
            root = Path(directory)
            for key in ("data", "indices", "indptr"):
                np.save(root / (key+".npy"), getattr(projector, key), allow_pickle=False)
            np.save(root / "sinograms.npy", data, allow_pickle=False)
            if x0 is not None:
                np.save(root / "initial.npy", x0, allow_pickle=False)
            common = (directory, projector.shape, ref, shape, backend)
            pools, futures = [], []
            progress = None
            try:
                # One executor per worker pins each process to its assigned device.
                for device in (worker_gpu_ids if backend == "gpu" else (0,)):
                    pool = ProcessPoolExecutor(
                        max_workers=1 if backend == "gpu" else workers,
                        mp_context=context, initializer=_initialize_worker,
                        initargs=(*common, device, dtype, boundary, solve_options, seeds, threads_per_worker))
                    pools.append(pool)
                # Small bounded chunks limit scheduling and result-buffer overhead.
                pending = {}
                next_slice = 0
                for lane in range(min(len(data), 2*workers)):
                    future = pools[lane % len(pools)].submit(_reconstruct_chunk, [next_slice])
                    pending[future] = lane % len(pools)
                    next_slice += 1
                progress = tqdm(total=len(data), desc="Reconstructing slices")
                while pending:
                    done = next(as_completed(pending))
                    lane = pending.pop(done)
                    for i, rec, info in done.result():
                        output[:, i], details[i] = rec, info
                        progress.update(1)
                    if next_slice < len(data):
                        pending[pools[lane].submit(_reconstruct_chunk, [next_slice])] = lane
                        next_slice += 1
            finally:
                if progress is not None:
                    progress.close()
                # Pending tasks are cancelled on error; running numerical tasks
                # finish before executors close (standard ProcessPool behavior).
                for pool in pools:
                    pool.shutdown(wait=True, cancel_futures=True)
    if isinstance(output, np.memmap):
        output.flush()
    metadata = {"workers": workers, "backend": backend,
                "gpu_ids": worker_gpu_ids,
                "slice_seeds": seeds, "slices": details,
                "volume_seconds": time.perf_counter()-start,
                "regularization": "independent 2D anisotropic TV per slice"}
    return output, metadata


def run_trial(config, truth, theta, codebook, init_seed, noise_seed):
    """Optional drop-in for spectro_agent.worker's run_trial import.

    Copy this file into spectro_agent/ first so its source hash is frozen.
    Set reconstruction.fast = {backend, algorithm, workers, gpu_ids, dtype,
    boundary, log_every, tol, gmres_restart} as needed. Existing solver,
    mu1/mu2, max_iter, inner tolerances, noise and initialization seeds are used.
    Requires backend='original': no synthetic/demo physics substitution.
    """
    if config["backend"] != "original":
        raise ValueError("The fast research adapter requires backend=original and ASTRA")
    truth = np.asarray(truth)
    if truth.ndim != 4 or not np.all(np.isfinite(truth)):
        raise ValueError("truth must be finite (states, slices, rows, columns)")
    r = config["reconstruction"]
    fast = dict(r.get("fast", {}))
    allowed = {"backend", "algorithm", "regularization", "mu3", "workers", "gpu_ids",
               "dtype", "boundary", "log_every", "tol", "gmres_restart",
               "threads_per_worker", "temp_dir"}
    if set(fast) - allowed:
        raise ValueError(f"Unknown reconstruction.fast settings: {sorted(set(fast)-allowed)}")
    if r["solver"] not in {"cg", "gmres", "pdhg"}:
        raise ValueError("Fast solver must be cg, gmres or pdhg")
    mode = config["noise"]["mode"]
    if mode not in {"none", "gaussian"}:
        raise ValueError("Noise mode must be none or gaussian")
    shape = truth.shape[-2:]
    projector = build_astra_projector(theta, shape, shape[1])
    # Fixed float64 data generation for CPU/GPU/precision comparisons.
    forward = Reconstructor(projector, codebook, shape, backend="cpu", dtype="float32")
    initial = np.random.default_rng(init_seed).random(truth.shape)
    rng = np.random.default_rng(noise_seed)
    sinograms = []
    for i in range(truth.shape[1]):
        sino = forward.project(truth[:, i])
        if mode == "gaussian":
            sino += rng.normal(0, config["noise"]["sigma"], sino.shape)
        sinograms.append(sino)
    fast.setdefault("algorithm", "pdhg" if r["solver"] == "pdhg" else "admm")
    rec, volume = reconstruct_volume(
        np.asarray(sinograms), theta, codebook, shape, projector=projector,
        x0=initial, seed=init_seed, mu1=r["mu1"], mu2=r["mu2"], max_iter=r["max_iter"],
        method="cg" if r["solver"] == "pdhg" else r["solver"],
        nonnegative=r["nonnegative"], inner_rtol=r["inner_rtol"],
        inner_maxiter=r["inner_maxiter"], **fast)
    each = np.mean((truth-np.asarray(rec, dtype=np.float32))**2, axis=(1, 2, 3))
    traces = [[h["objective"] for h in s["history"]] for s in volume["slices"]]
    changes = []
    for s in volume["slices"]:
        history = s["history"]
        last = history[-1]
        earlier = next((h for h in reversed(history[:-1]) if h["iteration"] <= last["iteration"]-10), None)
        changes.append(None if earlier is None else abs(last["objective"]-earlier["objective"])/max(abs(earlier["objective"]), 1e-15))
    metrics = {"mse": float(each.sum()), "mse_by_state": each.tolist(),
               "data_residual_mse": float(np.mean([s["data_residual_mse"] for s in volume["slices"]])),
               "objective_trace_by_slice": traces,
               "relative_objective_change_last_10_by_slice": changes,
               "inner_nonconverged_steps": sum(s["inner_nonconverged_steps"] for s in volume["slices"]),
               "inner_info_codes": [i for s in volume["slices"] for i in s["inner_info_codes"]],
               "acceleration": volume}
    return metrics, rec
