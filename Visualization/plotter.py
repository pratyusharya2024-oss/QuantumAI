from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes

class Plotter:
    """
    Static graph renderer for the Physics Modeler pipeline.

    Receives a time-series DataFrame from the Solver (via main.py)
    and renders a clean, publication-quality graph in a matplotlib window.

    Renders two panels:
      Left  — Primary plot: target variable vs time.
      Right — Rate of change: numerical derivative of target vs time,
              showing how fast the quantity is changing.

    No user interaction happens here — main.py owns all prompts.
    """

    def __init__(self) -> None:
        self._dataframe: Optional[pd.DataFrame] = None
        self._target: Optional[str] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def plot(self, dataframe: pd.DataFrame, target: str) -> None:
        """
        Main entry point called by main.py.

        Args:
            dataframe: Time-series DataFrame from Solver {"t": [...], target: [...]}.
            target:    Name of the target variable column (e.g. "v", "s", "KE").
        """
        self._validate(dataframe, target)
        self._dataframe = dataframe
        self._target = target

        t_data = dataframe["t"].to_numpy()
        y_data = dataframe[target].to_numpy()
        dy_data = self._compute_rate_of_change(t_data, y_data)

        self._render(t_data, y_data, dy_data)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, dataframe: pd.DataFrame, target: str) -> None:
        if dataframe is None or dataframe.empty:
            raise ValueError(
                "Plotter received an empty DataFrame. "
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
    # Derived data
    # ------------------------------------------------------------------

    def _compute_rate_of_change(
        self, t_data: np.ndarray, y_data: np.ndarray
    ) -> np.ndarray:
        """Numerical derivative dy/dt using numpy gradient."""
        return np.gradient(y_data, t_data)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        t_data: np.ndarray,
        y_data: np.ndarray,
        dy_data: np.ndarray,
    ) -> None:
        assert self._target is not None
        fig = plt.figure(figsize=(13, 5))
        fig.patch.set_facecolor("#0f0f1a")
        fig.suptitle(
            f"Physics Modeler — {self._target} vs time",
            color="white",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

        gs = GridSpec(1, 2, figure=fig, wspace=0.35)
        ax_primary = fig.add_subplot(gs[0, 0])
        ax_rate    = fig.add_subplot(gs[0, 1])

        self._draw_primary(ax_primary, t_data, y_data)
        self._draw_rate(ax_rate, t_data, dy_data)

        self._style_axes(
            ax_primary,
            title=f"{self._target} over time",
            xlabel="time (s)",
            ylabel=self._target,
        )
        self._style_axes(
            ax_rate,
            title=f"rate of change of {self._target}",
            xlabel="time (s)",
            ylabel=f"d({self._target})/dt",
        )

        plt.tight_layout()
        plt.show()

    def _draw_primary(
        self, ax: Axes, t_data: np.ndarray, y_data: np.ndarray
    ) -> None:
        ax.plot(
            t_data, y_data,
            color="#00d4ff",
            linewidth=2.0,
            zorder=2,
            label=self._target,
        )

        peak_idx = int(np.argmax(np.abs(y_data)))
        ax.annotate(
            f"peak: {y_data[peak_idx]:.3f}",
            xy=(t_data[peak_idx], y_data[peak_idx]),
            xytext=(t_data[peak_idx], y_data[peak_idx] * 0.85),
            color="#ff6b6b",
            fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.2),
        )

        ax.fill_between(t_data, y_data, alpha=0.12, color="#00d4ff", zorder=1)
        ax.axhline(0, color="#444466", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8, labelcolor="white", facecolor="#12122a", edgecolor="#2a2a4a")

    def _draw_rate(
        self, ax: Axes, t_data: np.ndarray, dy_data: np.ndarray
    ) -> None:
        positive = dy_data >= 0
        ax.fill_between(
            t_data, dy_data,
            where=positive.tolist(),
            color="#a78bfa",
            alpha=0.6,
            label="increasing",
            zorder=2,
        )
        ax.fill_between(
            t_data, dy_data,
            where=(~positive).tolist(),
            color="#f87171",
            alpha=0.6,
            label="decreasing",
            zorder=2,
        )
        ax.plot(t_data, dy_data, color="#ffffff", linewidth=1.0, zorder=3)
        ax.axhline(0, color="#444466", linewidth=0.8, linestyle="--")
        ax.legend(fontsize=8, labelcolor="white", facecolor="#12122a", edgecolor="#2a2a4a")

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    def _style_axes(
        self,
        ax: Axes,
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> None:
        ax.set_facecolor("#12122a")
        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", color="#1e1e3a", linewidth=0.6, linestyle="--")
        ax.grid(True, which="minor", color="#16163a", linewidth=0.3, linestyle=":")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4a")