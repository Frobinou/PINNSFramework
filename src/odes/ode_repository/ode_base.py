from abc import ABC
import torch
from pydantic import BaseModel

from scipy.integrate import solve_ivp
import numpy as np


class BaseODE:
    def __init__(self, params: BaseModel):
        self.name = "Lotka Volterra"
        self.params: BaseModel = params

    def update_params(self, new_params: BaseModel):
        self.params = new_params

    # ---------- Core dynamics (factorisée) ----------
    def _dynamics(self, t, x, params):
        raise NotImplementedError

    # ---------- Torch version ----------
    def torch_ode(self, x: torch.Tensor) -> torch.Tensor:
        dx = self._dynamics(x.mT)
        return torch.stack(dx, dim=1)

    # ---------- Numpy / SciPy version ----------

    def _dynamics_numpy(self, t, x):
        return self._dynamics(x, t=t)

    # ---------- Validation ----------
    def simulate(
        self,
        t_span: tuple,
        x0: list,
        nb_points: int,
        method: str = "Radau",
        max_step: float = np.inf,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> object:
        """Numerically integrate the ODE system over a time interval.

        Uses ``scipy.integrate.solve_ivp``. Defaults to the ``Radau`` implicit
        solver which handles stiff systems robustly by adapting its internal
        step size to local stiffness. For non-stiff systems ``"RK45"`` or
        ``"DOP853"`` are faster alternatives.

        Args:
            t_span: Integration interval as ``(t0, tf)``.
            x0: Initial state vector of length ``n_vars``.
            nb_points: Number of evenly spaced output time points.
            method: Integration method passed to ``solve_ivp``.
                Recommended choices:

                - ``"Radau"``  — implicit, order 5, best for stiff systems
                (default).
                - ``"DOP853"`` — explicit, order 8, fast for smooth non-stiff
                systems.
                - ``"RK45"``   — explicit, order 5, general purpose.

                Defaults to ``"Radau"``.
            max_step: Maximum internal solver step size. With ``"Radau"`` the
                default ``np.inf`` is usually sufficient since the solver
                controls its own step size adaptively. For explicit methods on
                fast systems, consider setting this to
                ``(t_span[1] - t_span[0]) / nb_points``.
                Defaults to ``np.inf``.
            rtol: Relative tolerance for the solver. Defaults to ``1e-6``.
            atol: Absolute tolerance for the solver. Tighten this when state
                variables approach zero (e.g. near-extinction in predator-prey
                models). Defaults to ``1e-8``.

        Returns:
            ``scipy.integrate.OdeSolution`` result object. Access ``sol.t`` for
            the time vector and ``sol.y`` (shape ``(n_vars, nb_points)``) for
            the state trajectories.

        Raises:
            RuntimeWarning: If ``solve_ivp`` does not converge, a warning is
                issued with the solver message and the index of the last
                successful point. The partial solution is still returned so
                the caller can inspect how far integration progressed.

        Note:
            ``atol=1e-8`` is tighter than the scipy default (``1e-6``) to
            prevent near-zero state variables from accumulating floating-point
            drift, which is a common failure mode in predator-prey systems
            during low-population phases.
        """
        t_eval = np.linspace(*t_span, nb_points)

        sol = solve_ivp(
            fun=self._dynamics_numpy,
            t_span=t_span,
            y0=x0,
            t_eval=t_eval,
            method=method,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            import warnings
            warnings.warn(
                f"solve_ivp ({method}) did not converge: {sol.message}. "
                f"Returning partial solution up to t={sol.t[-1]:.4f} "
                f"({len(sol.t)}/{nb_points} points).",
                RuntimeWarning,
                stacklevel=2,
            )

        return sol
