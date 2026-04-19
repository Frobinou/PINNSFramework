from abc import ABC
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.odes.ode_repository.ode_base import BaseODE
from pathlib import Path


class ODEDataGenerator(ABC):
    """Abstract interface for ODE data generation.

    Designed to generalise Lotka-Volterra, CFAST, and other dynamical systems.
    Provides simulation, dataset generation, PINN-format conversion, and
    visualisation utilities.

    Attributes:
        params (dict): ODE parameters attached to every generated sample.
        ode (BaseODE): ODE instance used for numerical integration.
        target_cols (list[str]): Names of the state variable columns.
        input_cols (list[str]): Names of the input feature columns.
        t_span (tuple[float, float]): Integration interval ``[t0, tf]``.
        n_steps (int): Number of time steps in the numerical solution.
    """

    def __init__(
        self,
        params: dict,
        ode: BaseODE,
        target_cols: list[str],
        input_cols: list[str],
        t_span: tuple[float, float] = (0, 10),
        n_steps: int = 200,
    ):
        """Initialise the ODE data generator.

        Args:
            params: ODE parameters attached as constant columns to every
                generated sample (e.g. ``{"alpha": 1.0, "beta": 0.1}``).
            ode: ODE instance used for numerical integration.
            target_cols: Names of the state variable columns
                (e.g. ``["prey", "predator"]``).
            input_cols: Names of the input feature columns (e.g. ``["t"]``).
            t_span: Integration interval as ``(t0, tf)``. Defaults to ``(0, 10)``.
            n_steps: Number of evenly spaced time steps. Defaults to ``200``.
        """
        self.params = params
        self.ode: BaseODE = ode
        self.target_cols = target_cols
        self.input_cols = input_cols
        self.t_span = t_span
        self.n_steps = n_steps

    # ---------- Simulation ----------

    def simulate(self, x0) -> pd.DataFrame:
        """Run a single numerical simulation from an initial condition.

        Args:
            x0: Initial state vector of shape ``(n_vars,)``.

        Returns:
            DataFrame with one row per time step, containing the state
            variables, a ``t`` column, and one column per ODE parameter.
        """
        sol = self.ode.simulate(t_span=self.t_span, x0=x0, nb_points=self.n_steps)

        df = pd.DataFrame(sol.y.T, columns=self.target_cols)
        df["t"] = sol.t

        for k, v in self.params.items():
            df[k] = v

        return df

    # ---------- Dataset generation ----------
    def generate_dataset(
        self,
        n_sims: int = 100,
        x0_sampler=None,
        param_sampler=None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate a multi-simulation dataset by sampling initial conditions.

        If both ``x0_sampler`` and ``param_sampler`` are ``None``, a single
        simulation is run from ``np.ones(n_vars)`` using ``self.params``.
        In that case no ``run_id`` column is added and ``n_sims`` is ignored,
        since there is nothing to sample over.

        Each simulation is assigned a unique ``run_id`` so trajectories can
        be grouped later for plotting or PINN training.

        Args:
            n_sims: Number of independent simulations to run. Ignored when
                both ``x0_sampler`` and ``param_sampler`` are ``None``.
                Defaults to ``100``.
            x0_sampler: Callable returning a random initial condition array, or
                an indexable sequence of length ``n_sims``. If ``None`` and
                ``param_sampler`` is also ``None``, a single deterministic run
                is produced instead.
            param_sampler: Callable returning a fresh ``params`` dict for each
                simulation. If ``None`` and ``x0_sampler`` is also ``None``,
                a single deterministic run is produced instead.
            seed: NumPy random seed for reproducibility. Defaults to ``42``.

        Returns:
            DataFrame of all simulations. A ``run_id`` column is present only
            when more than one simulation is produced.
        """
        # Single-parameter-set shortcut: no sampling, no run_id overhead.
        if x0_sampler is None and param_sampler is None:
            n_vars = len(self.target_cols)
            return self.simulate(x0=np.ones(n_vars))

        np.random.seed(seed)
        result = pd.DataFrame()

        for i in range(n_sims):
            if param_sampler is not None:
                self.params = param_sampler()

            if x0_sampler is not None:
                x0 = x0_sampler() if callable(x0_sampler) else x0_sampler[i]
            else:
                x0 = np.ones(len(self.target_cols))

            df = self.simulate(x0)
            df["run_id"] = i
            result = pd.concat([result, df], ignore_index=True)

        return result

    def generate_and_save(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Generate a dataset and persist it as a Parquet file.

        Creates parent directories if they do not exist.

        Args:
            file_path: Destination ``.parquet`` file path.
            **kwargs: Forwarded to :meth:`generate_dataset`.

        Returns:
            The generated DataFrame (also written to disk).
        """
        if not file_path.exists():
            os.makedirs(file_path.parent, exist_ok=True)

        df = self.generate_dataset(**kwargs)
        df.to_parquet(file_path, index=False)
        return df

    # ---------- PINN-ready format ----------

    @staticmethod
    def to_pinn_format(
        df: pd.DataFrame,
        state_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a simulation DataFrame to PINN input/output arrays.

        The input array ``x`` contains the time column followed by any
        state columns present in the DataFrame, then all remaining columns
        (ODE parameters, extra features) excluding ``run_id``.

        Args:
            df: Simulation DataFrame as returned by :meth:`generate_dataset`.
            state_cols: Names of the state variable columns used as targets.

        Returns:
            Tuple ``(x, y)`` where:

            - ``x``: Input array of shape ``(N, n_features)`` as ``float32``.
            - ``y``: Target array of shape ``(N, n_state_vars)`` as ``float32``.
        """
        x = df[
            ["t"]
            + [k for k in df.columns if k in state_cols]
            + [k for k in df.columns if k not in state_cols + ["t", "run_id"]]
        ].values

        y = df[state_cols].values
        return x.astype(np.float32), y.astype(np.float32)

    # ---------- Visualisation ----------

    def plot_trajectories(
        self,
        df: pd.DataFrame,
        max_runs: int = 10,
        figsize: tuple[int, int] = (10, 4),
        alpha: float = 0.7,
    ) -> plt.Figure:
        """Plot time-domain trajectories for each state variable.

        Each run in ``df`` is drawn as a separate line. When ``df`` contains
        more than ``max_runs`` distinct trajectories, only the first
        ``max_runs`` are shown to keep the figure readable.

        One subplot is created per state variable (``self.target_cols``).

        Args:
            df: Simulation DataFrame containing a ``t`` column, one column
                per state variable, and a ``run_id`` column.
            max_runs: Maximum number of trajectories to overlay per subplot.
                Defaults to ``10``.
            figsize: Figure size as ``(width, height)`` in inches.
                Defaults to ``(10, 4)``.
            alpha: Line opacity. Defaults to ``0.7``.

        Returns:
            Matplotlib Figure — call ``plt.show()`` or ``fig.savefig()`` to render.

        Example:
            .. code-block:: python

                fig = generator.plot_trajectories(df, max_runs=5)
                fig.savefig("trajectories.png", dpi=150)
        """
        n_vars = len(self.target_cols)
        run_ids = df["run_id"].unique()[:max_runs]
        colors = cm.tab10(np.linspace(0, 1, len(run_ids)))

        fig, axes = plt.subplots(1, n_vars, figsize=figsize, sharey=False)
        # Ensure axes is always iterable even for a single variable.
        if n_vars == 1:
            axes = [axes]

        for ax, col in zip(axes, self.target_cols):
            for color, run_id in zip(colors, run_ids):
                run = df[df["run_id"] == run_id]
                ax.plot(run["t"], run[col], color=color, alpha=alpha,
                        linewidth=1.2, label=f"run {run_id}")

            ax.set_xlabel("t")
            ax.set_ylabel(col)
            ax.set_title(f"{col} over time")
            ax.grid(True, linewidth=0.4, alpha=0.5)

        # Single shared legend outside the subplots.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right",
                   fontsize=8, ncol=2, framealpha=0.7)
        fig.tight_layout()
        return fig

    def plot_phase_portrait(
        self,
        df: pd.DataFrame,
        x_var: str | None = None,
        y_var: str | None = None,
        max_runs: int = 10,
        figsize: tuple[int, int] = (6, 6),
        alpha: float = 0.7,
        show_direction: bool = True,
    ) -> plt.Figure:
        """Plot a phase-space portrait for a pair of state variables.

        Each run is drawn as a trajectory in the ``(x_var, y_var)`` plane.
        Start and end points are marked with ``o`` and ``x`` respectively so
        that the direction of time is immediately visible without arrows.
        When ``show_direction`` is ``True``, a mid-trajectory arrow is added
        on each run to make the flow direction explicit.

        By default ``x_var`` and ``y_var`` are the first two entries of
        ``self.target_cols``. Raises ``ValueError`` for systems with fewer
        than two state variables.

        Args:
            df: Simulation DataFrame as returned by :meth:`generate_dataset`.
            x_var: Column name for the horizontal axis. Defaults to
                ``self.target_cols[0]``.
            y_var: Column name for the vertical axis. Defaults to
                ``self.target_cols[1]``.
            max_runs: Maximum number of trajectories to overlay. Defaults to ``10``.
            figsize: Figure size as ``(width, height)`` in inches.
                Defaults to ``(6, 6)``.
            alpha: Line opacity. Defaults to ``0.7``.
            show_direction: If ``True``, draw a mid-trajectory arrow on each run
                to indicate the direction of time. Defaults to ``True``.

        Returns:
            Matplotlib Figure.

        Raises:
            ValueError: If ``self.target_cols`` has fewer than 2 entries and
                no explicit ``x_var``/``y_var`` are provided.

        Example:
            .. code-block:: python

                fig = generator.plot_phase_portrait(df, max_runs=8)
                fig.savefig("phase_portrait.png", dpi=150)
        """
        if len(self.target_cols) < 2 and (x_var is None or y_var is None):
            raise ValueError(
                "Phase portrait requires at least 2 state variables. "
                "Provide explicit x_var and y_var or extend target_cols."
            )

        x_var = x_var or self.target_cols[0]
        y_var = y_var or self.target_cols[1]

        run_ids = df["run_id"].unique()[:max_runs]
        colors = cm.tab10(np.linspace(0, 1, len(run_ids)))

        fig, ax = plt.subplots(figsize=figsize)

        for color, run_id in zip(colors, run_ids):
            run = df[df["run_id"] == run_id].reset_index(drop=True)
            x = run[x_var].to_numpy()
            y = run[y_var].to_numpy()

            ax.plot(x, y, color=color, alpha=alpha,
                    linewidth=1.2, label=f"run {run_id}")

            # Mark start (circle) and end (cross) so time direction is visible.
            ax.plot(x[0], y[0], "o", color=color, markersize=5)
            ax.plot(x[-1], y[-1], "x", color=color, markersize=6, markeredgewidth=1.5)

            if show_direction and len(x) > 2:
                # Draw an arrow at the midpoint of the trajectory.
                mid = len(x) // 2
                ax.annotate(
                    "",
                    xy=(x[mid + 1], y[mid + 1]),
                    xytext=(x[mid], y[mid]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=1.2,
                    ),
                )

        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(f"Phase portrait — {x_var} vs {y_var}")
        ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.7)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        return fig