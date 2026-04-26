from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes


class Animator:
    """
    Produces two simultaneous animations from a solver DataFrame:

      Left panel  — Value buildup: plots the target variable's value
                    growing frame by frame over time.
      Right panel — Object motion: animates a moving dot tracing the
                    physical path of the object over time.

    Receives its data from main.py which passes the solver result DataFrame.
    Plays both animations in a single matplotlib window.
    """

    def __init__(self) -> None:
        self._dataframe: Optional[pd.DataFrame] = None
        self._target: Optional[str] = None
        self._interval: int = 20

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def animate(
        self,
        dataframe: pd.DataFrame,
        target: str,
        known: dict[str, float],
        interval: int = 20,
    ) -> None:
        """
        Main entry point called by main.py.

        Args:
            dataframe: Time-series DataFrame from Solver {"t": [...], target: [...]}.
            target:    Name of the target variable (e.g. "v", "s", "KE").
            known:     Known values dict — used to reconstruct 2D path if possible.
            interval:  Milliseconds between animation frames.
        """
        self._validate_dataframe(dataframe, target)
        self._dataframe = dataframe
        self._target = target
        self._interval = interval

        t_data = dataframe["t"].to_numpy()
        y_data = dataframe[target].to_numpy()
        x_data = self._resolve_position_data(known, t_data, y_data)

        self._render(t_data, y_data, x_data)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dataframe(self, dataframe: pd.DataFrame, target: str) -> None:
        if dataframe is None or dataframe.empty:
            raise ValueError(
                "Animator received an empty DataFrame. "
                "Ensure Solver was called with time_steps > 0."
            )
        if "t" not in dataframe.columns:
            raise ValueError("DataFrame must contain a 't' (time) column.")
        if target not in dataframe.columns:
            raise ValueError(
                f"Target column '{target}' not found in DataFrame. "
                f"Available columns: {list(dataframe.columns)}"
            )

    # ------------------------------------------------------------------
    # Path resolution for the motion panel
    # ------------------------------------------------------------------

    def _resolve_position_data(
        self,
        known: dict[str, float],
        t_data: np.ndarray,
        y_data: np.ndarray,
    ) -> np.ndarray:
        """
        Attempts to build an x-axis position array for the motion panel.

        - If 's' (displacement) is the target, uses y_data directly as x.
        - If 'u' and 'a' are known, reconstructs s = u*t + 0.5*a*t²
        - Otherwise falls back to using time as the x-axis.
        """
        if self._target == "s":
            return y_data

        u = known.get("u", 0.0)
        a = known.get("a", 0.0)

        if "u" in known or "a" in known:
            return u * t_data + 0.5 * a * t_data ** 2

        return t_data

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        t_data: np.ndarray,
        y_data: np.ndarray,
        x_data: np.ndarray,
    ) -> None:
        assert self._target is not None
        fig = plt.figure(figsize=(13, 5))
        fig.patch.set_facecolor("#0f0f1a")
        fig.suptitle(
            f"Physics Modeler — Animation: {self._target} vs time",
            color="white",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

        gs = GridSpec(1, 2, figure=fig, wspace=0.35)
        ax_value  = fig.add_subplot(gs[0, 0])
        ax_motion = fig.add_subplot(gs[0, 1])

        self._style_axes(ax_value,  f"{self._target} over time", "time (s)", self._target)
        self._style_axes(ax_motion, "Object motion",             "position", self._target)

        # Static reference lines
        ax_value.plot(t_data, y_data, color="#2a2a4a", linewidth=1.2, zorder=1)
        ax_motion.plot(x_data, y_data, color="#2a2a4a", linewidth=1.2, zorder=1)

        # Animated elements — value buildup panel
        live_line_v,  = ax_value.plot([], [], color="#00d4ff", linewidth=2, zorder=2)
        live_dot_v,   = ax_value.plot([], [], "o", color="#ff6b6b", markersize=7, zorder=3)
        value_text    = ax_value.text(
            0.05, 0.92, "", transform=ax_value.transAxes,
            color="#ffffff", fontsize=9
        )

        # Animated elements — motion panel
        live_trail_m, = ax_motion.plot([], [], color="#a78bfa", linewidth=2, zorder=2)
        live_dot_m,   = ax_motion.plot([], [], "o", color="#ff6b6b", markersize=9, zorder=3)
        motion_text   = ax_motion.text(
            0.05, 0.92, "", transform=ax_motion.transAxes,
            color="#ffffff", fontsize=9
        )

        self._set_axis_limits(ax_value,  t_data, y_data)
        self._set_axis_limits(ax_motion, x_data, y_data)

        def update(frame: int):
            # Value buildup panel
            live_line_v.set_data(t_data[:frame], y_data[:frame])
            live_dot_v.set_data([t_data[frame - 1]], [y_data[frame - 1]])
            value_text.set_text(
                f"t = {t_data[frame - 1]:.2f}s  |  {self._target} = {y_data[frame - 1]:.3f}"
            )

            # Motion panel
            live_trail_m.set_data(x_data[:frame], y_data[:frame])
            live_dot_m.set_data([x_data[frame - 1]], [y_data[frame - 1]])
            motion_text.set_text(
                f"pos = {x_data[frame - 1]:.2f}  |  {self._target} = {y_data[frame - 1]:.3f}"
            )

            return live_line_v, live_dot_v, value_text, live_trail_m, live_dot_m, motion_text

        frame_count = len(t_data)
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=range(1, frame_count + 1),
            interval=self._interval,
            blit=True,
            repeat=False,
        )

        plt.tight_layout()
        plt.show()

        # Keep reference alive so garbage collector doesn't kill the animation
        _ = anim

    # ------------------------------------------------------------------
    # Styling helpers
    # ------------------------------------------------------------------

    def _style_axes(
        self, ax: Axes, title: str, xlabel: str, ylabel: str
    ) -> None:
        ax.set_facecolor("#12122a")
        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")
        ax.grid(True, color="#1e1e3a", linewidth=0.6, linestyle="--")

    def _set_axis_limits(
        self,
        ax: Axes,
        x_data: np.ndarray,
        y_data: np.ndarray,
        padding: float = 0.08,
    ) -> None:
        x_range = x_data.max() - x_data.min() or 1.0
        y_range = y_data.max() - y_data.min() or 1.0
        ax.set_xlim(x_data.min() - padding * x_range, x_data.max() + padding * x_range)
        ax.set_ylim(y_data.min() - padding * y_range, y_data.max() + padding * y_range)