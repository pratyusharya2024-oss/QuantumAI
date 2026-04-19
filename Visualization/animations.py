from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import importlib
import importlib.util
import tempfile
import wave
from pathlib import Path


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
        self._paused: bool = False
        self._audio_player = None
        self._audio_playback = None
        self._music_on: bool = False
        self._music_file: Optional[Path] = None

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
        self._paused = False

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
        self._attach_controls(fig, anim)
        self._start_soft_music()

        plt.tight_layout()
        plt.show()
        self._stop_soft_music()

        # Keep reference alive so garbage collector doesn't kill the animation
        _ = anim

    def _attach_controls(self, fig: plt.Figure, anim: animation.FuncAnimation) -> None:
        """
        Adds UI controls directly in the animation window:
        - Pause / Play toggle button.
        - Music ON / OFF toggle button.
        """
        pause_ax = fig.add_axes([0.41, 0.01, 0.09, 0.05])
        music_ax = fig.add_axes([0.52, 0.01, 0.12, 0.05])

        pause_button = Button(pause_ax, "Pause", color="#23233f", hovercolor="#2b2b52")
        music_button = Button(music_ax, "Music: ON", color="#23233f", hovercolor="#2b2b52")

        pause_button.label.set_color("white")
        music_button.label.set_color("white")

        def toggle_pause(_event) -> None:
            self._paused = not self._paused
            if self._paused:
                anim.event_source.stop()
                pause_button.label.set_text("Play")
            else:
                anim.event_source.start()
                pause_button.label.set_text("Pause")
            fig.canvas.draw_idle()

        def toggle_music(_event) -> None:
            self._music_on = not self._music_on
            if self._music_on:
                self._start_soft_music()
                music_button.label.set_text("Music: ON")
            else:
                self._stop_soft_music()
                music_button.label.set_text("Music: OFF")
            fig.canvas.draw_idle()

        pause_button.on_clicked(toggle_pause)
        music_button.on_clicked(toggle_music)
        self._music_on = True

    def _start_soft_music(self) -> None:
        """
        Plays low-volume ambient tone if simpleaudio is available.
        Falls back silently when the package is unavailable.
        """
        if not self._music_on:
            return

        if self._audio_player is None:
            if importlib.util.find_spec("simpleaudio") is None:
                return
            self._audio_player = importlib.import_module("simpleaudio")

        if self._audio_playback is not None and self._audio_playback.is_playing():
            return

        if self._music_file is None:
            self._music_file = self._build_soft_track_file()
        wave_obj = self._audio_player.WaveObject.from_wave_file(str(self._music_file))
        self._audio_playback = wave_obj.play()

    def _stop_soft_music(self) -> None:
        if self._audio_playback is not None and self._audio_playback.is_playing():
            self._audio_playback.stop()
        self._audio_playback = None

    def _build_soft_track_file(self) -> Path:
        """
        Generates a short ambient two-tone WAV loop at low volume.
        """
        sample_rate = 44100
        duration = 8.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone_a = np.sin(2 * np.pi * 220 * t)
        tone_b = np.sin(2 * np.pi * 329.63 * t)
        envelope = np.clip(np.sin(np.pi * t / duration), 0, 1)
        signal = (0.08 * tone_a + 0.05 * tone_b) * envelope
        pcm = np.int16(signal * 32767)

        file_path = Path(tempfile.gettempdir()) / "physics_modeler_soft_music.wav"
        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return file_path

    # ------------------------------------------------------------------
    # Styling helpers
    # ------------------------------------------------------------------

    def _style_axes(
        self, ax: plt.Axes, title: str, xlabel: str, ylabel: str
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
        ax: plt.Axes,
        x_data: np.ndarray,
        y_data: np.ndarray,
        padding: float = 0.08,
    ) -> None:
        x_range = x_data.max() - x_data.min() or 1.0
        y_range = y_data.max() - y_data.min() or 1.0
        ax.set_xlim(x_data.min() - padding * x_range, x_data.max() + padding * x_range)
        ax.set_ylim(y_data.min() - padding * y_range, y_data.max() + padding * y_range)
