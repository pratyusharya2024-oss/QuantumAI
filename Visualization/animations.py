"""
animator.py
===========
Visualization Module: Animations with Pause/Play, Music, Formula Display
Covers: All 5 physics modules with themed animations
Each animation shows: Formula → Steps → Answer → Visual Animation
Music: Calm background tones via pygame
Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Button
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.patheffects as pe
import math
import threading

# ─────────────────────────────────────────────
# MUSIC ENGINE (pygame — calm background tones)
# ─────────────────────────────────────────────

def _start_music(theme="sine"):
    """
    Generates and plays calm background music using pygame.
    Falls back silently if pygame not installed.
    theme: 'sine' = calm sine wave tones
    """
    try:
        import pygame
        import numpy as np_music
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

        sample_rate = 44100
        duration    = 4.0
        t           = np_music.linspace(0, duration, int(sample_rate * duration))

        # Calm ambient chord — mix of low sine waves
        frequencies = [130.8, 164.8, 196.0, 261.6]  # C3 E3 G3 C4
        wave = sum(
            0.18 * np_music.sin(2 * np_music.pi * f * t) for f in frequencies
        )
        # Fade in and out
        fade = np_music.ones_like(wave)
        fade_len = int(sample_rate * 0.4)
        fade[:fade_len]  = np_music.linspace(0, 1, fade_len)
        fade[-fade_len:] = np_music.linspace(1, 0, fade_len)
        wave = (wave * fade * 32767).astype(np_music.int16)

        sound = pygame.sndarray.make_sound(wave)
        sound.play(loops=-1)
        return sound
    except Exception:
        return None


def _stop_music(sound):
    """Stops background music."""
    try:
        if sound:
            sound.stop()
    except Exception:
        pass


# ─────────────────────────────────────────────
# THEME PALETTE
# ─────────────────────────────────────────────

THEMES = {
    "sunset": {
        "bg": "#0d0a0e", "axes_bg": "#110f15",
        "title": "#ff9a5c", "text": "#f4c97a",
        "accent": "#ff6b35", "accent2": "#e05c5c",
        "colors": ["#ff6b35", "#ff9a5c", "#f4c97a", "#e05c5c", "#ffb347"],
        "button_face": "#2a1410", "button_text": "#ff9a5c",
        "glow": "#ff6b35",
    },
    "deepspace": {
        "bg": "#03030f", "axes_bg": "#05051a",
        "title": "#7eb8f7", "text": "#a0c4ff",
        "accent": "#4fc3f7", "accent2": "#7986cb",
        "colors": ["#4fc3f7", "#7986cb", "#b39ddb", "#4dd0e1", "#9575cd"],
        "button_face": "#05051a", "button_text": "#4fc3f7",
        "glow": "#4fc3f7",
    },
    "neon": {
        "bg": "#050508", "axes_bg": "#08080f",
        "title": "#00ff9f", "text": "#ff00ff",
        "accent": "#00ff9f", "accent2": "#ff00ff",
        "colors": ["#00ff9f", "#ff00ff", "#00cfff", "#ff6600", "#ffe600"],
        "button_face": "#08080f", "button_text": "#00ff9f",
        "glow": "#00ff9f",
    },
}

MODULE_THEME = {
    "kinematics":    "sunset",
    "gravitation":   "deepspace",
    "energetics":    "neon",
    "quantum":       "deepspace",
    "thermodynamics":"sunset",
}


# ─────────────────────────────────────────────
# BASE ANIMATOR CLASS
# ─────────────────────────────────────────────

class BaseAnimator:
    """
    Base class for all physics animators.
    Handles figure setup, formula panel, pause/play,
    animated button, and music.
    """

    def __init__(self, theme_name, title, formula, steps, answer):
        self.theme_name = theme_name
        self.t          = THEMES[theme_name]
        self.title_str  = title
        self.formula    = formula
        self.steps      = steps
        self.answer     = answer
        self.paused     = False
        self.anim       = None
        self.sound      = None
        self._btn_glow  = 0
        self._build_figure()

    def _build_figure(self):
        t = self.t
        self.fig = plt.figure(figsize=(13, 7), facecolor=t["bg"])

        # Layout: left = animation, right = formula panel
        self.ax_anim  = self.fig.add_axes([0.02, 0.18, 0.58, 0.75])
        self.ax_panel = self.fig.add_axes([0.63, 0.18, 0.35, 0.75])
        self.ax_btn   = self.fig.add_axes([0.38, 0.04, 0.12, 0.07])

        for ax in [self.ax_anim, self.ax_panel]:
            ax.set_facecolor(t["axes_bg"])
            for spine in ax.spines.values():
                spine.set_edgecolor(t["accent"])
                spine.set_linewidth(1.2)
            ax.tick_params(colors=t["text"])

        self.ax_panel.set_xticks([])
        self.ax_panel.set_yticks([])

        # Title
        self.fig.text(0.33, 0.96, self.title_str,
                      color=t["title"], fontsize=15,
                      fontweight="bold", ha="center",
                      path_effects=[pe.withStroke(
                          linewidth=3, foreground=t["bg"])])

        # Formula panel content
        self._draw_formula_panel()

        # Pause/Play button
        self.btn = Button(self.ax_btn, "⏸  Pause",
                          color=t["button_face"],
                          hovercolor=t["accent"])
        self.btn.label.set_color(t["button_text"])
        self.btn.label.set_fontsize(10)
        self.btn.label.set_fontweight("bold")
        self.btn.on_clicked(self._toggle_pause)

        # Animated glow border on button (updated each frame)
        self._glow_rect = self.ax_btn.add_patch(
            FancyBboxPatch((0, 0), 1, 1,
                           boxstyle="round,pad=0.05",
                           linewidth=2,
                           edgecolor=t["glow"],
                           facecolor="none",
                           transform=self.ax_btn.transAxes,
                           zorder=10, alpha=0.8))

    def _draw_formula_panel(self):
        """Draws formula, steps, and answer on the right panel."""
        t   = self.t
        ax  = self.ax_panel
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        y = 0.93
        ax.text(0.5, y, "📐 Formula",
                color=t["title"], fontsize=11,
                fontweight="bold", ha="center",
                transform=ax.transAxes)
        y -= 0.08
        ax.text(0.5, y, self.formula,
                color=t["accent"], fontsize=10,
                ha="center", style="italic",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=t["bg"],
                          edgecolor=t["accent"],
                          alpha=0.8))
        y -= 0.10
        ax.text(0.05, y, "📋 Steps",
                color=t["title"], fontsize=10,
                fontweight="bold",
                transform=ax.transAxes)
        y -= 0.07
        for step in self.steps:
            ax.text(0.05, y, f"▸  {step}",
                    color=t["text"], fontsize=8.5,
                    transform=ax.transAxes,
                    wrap=True)
            y -= 0.065

        y -= 0.04
        ax.axhline(y + 0.03, color=t["accent"],
                   linewidth=0.8, alpha=0.5,
                   xmin=0.03, xmax=0.97)
        y -= 0.03
        ax.text(0.05, y, "✅ Answer",
                color=t["title"], fontsize=10,
                fontweight="bold",
                transform=ax.transAxes)
        y -= 0.08
        ax.text(0.5, y, self.answer,
                color=t["accent2"], fontsize=10,
                fontweight="bold", ha="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=t["bg"],
                          edgecolor=t["accent2"],
                          alpha=0.9))

    def _toggle_pause(self, event):
        self.paused = not self.paused
        if self.paused:
            self.btn.label.set_text("▶  Play")
        else:
            self.btn.label.set_text("⏸  Pause")
        self.fig.canvas.draw_idle()

    def _pulse_button(self, frame):
        """Pulses button glow every frame."""
        alpha = 0.4 + 0.5 * abs(math.sin(frame * 0.08))
        self._glow_rect.set_alpha(alpha)

    def start(self):
        """Starts music and shows the animation."""
        self.sound = _start_music()
        plt.show()
        _stop_music(self.sound)


# ─────────────────────────────────────────────
# ── KINEMATICS: Projectile Animation (Sunset)
# ─────────────────────────────────────────────

class ProjectileAnimator(BaseAnimator):
    """
    Animates a projectile trajectory with:
    - Ball flying through arc
    - Velocity vector arrow
    - Trail effect
    - Formula + steps + answer panel
    """

    def __init__(self, v_initial, angle_deg, formula, steps, answer, g=9.81):
        super().__init__(
            theme_name = "sunset",
            title      = f"Projectile Motion  |  v={v_initial} m/s  θ={angle_deg}°",
            formula    = formula,
            steps      = steps,
            answer     = answer
        )
        self.v0    = v_initial
        self.angle = angle_deg
        self.g     = g
        self._build_trajectory()
        self._setup_animation()

    def _build_trajectory(self):
        angle_rad = math.radians(self.angle)
        T  = 2 * self.v0 * math.sin(angle_rad) / self.g
        self.t_arr = np.linspace(0, T, 300)
        self.x_arr = self.v0 * math.cos(angle_rad) * self.t_arr
        self.y_arr = (self.v0 * math.sin(angle_rad) * self.t_arr
                      - 0.5 * self.g * self.t_arr ** 2)
        self.vx    = self.v0 * math.cos(angle_rad)

    def _setup_animation(self):
        t  = self.t
        ax = self.ax_anim
        ax.set_xlim(-0.5, max(self.x_arr) * 1.1)
        ax.set_ylim(-0.5, max(self.y_arr) * 1.3)
        ax.set_xlabel("x (m)", color=t["text"])
        ax.set_ylabel("y (m)", color=t["text"])
        ax.axhline(0, color=t["accent"], linewidth=1, alpha=0.5)

        # Ghost trajectory
        ax.plot(self.x_arr, self.y_arr,
                color=t["accent"], linewidth=1,
                linestyle="--", alpha=0.2)

        self.trail_line, = ax.plot([], [], color=t["accent"],
                                   linewidth=2, alpha=0.6)
        self.ball = Circle((0, 0), radius=max(self.x_arr) * 0.015,
                            color=t["accent2"], zorder=5)
        ax.add_patch(self.ball)
        self.arrow = FancyArrowPatch((0, 0), (0, 0),
                                     arrowstyle="-|>",
                                     color=t["colors"][2],
                                     linewidth=2,
                                     mutation_scale=15)
        ax.add_patch(self.arrow)
        self.time_text = ax.text(0.05, 0.92, "", transform=ax.transAxes,
                                 color=t["text"], fontsize=9)

        def update(frame):
            if self.paused:
                return self.trail_line, self.ball, self.arrow, self.time_text
            i  = frame % len(self.t_arr)
            x  = self.x_arr[i]
            y  = self.y_arr[i]
            vy = (self.v0 * math.sin(math.radians(self.angle))
                  - self.g * self.t_arr[i])
            scale = max(self.x_arr) * 0.12
            self.trail_line.set_data(self.x_arr[:i], self.y_arr[:i])
            self.ball.set_center((x, y))
            self.arrow.set_positions((x, y),
                                     (x + self.vx * scale / self.v0,
                                      y + vy * scale / self.v0))
            self.time_text.set_text(
                f"t = {self.t_arr[i]:.2f}s  |  "
                f"x={x:.1f}m  y={y:.1f}m")
            self._pulse_button(frame)
            return self.trail_line, self.ball, self.arrow, self.time_text

        self.anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.t_arr),
            interval=30, blit=True, repeat=True)


# ─────────────────────────────────────────────
# ── KINEMATICS: 1D Motion Animation (Sunset)
# ─────────────────────────────────────────────

class Motion1DAnimator(BaseAnimator):
    """Animates 1D motion with moving block and velocity arrow."""

    def __init__(self, v_initial, acceleration, total_time,
                 formula, steps, answer):
        super().__init__(
            theme_name = "sunset",
            title      = f"1D Motion  |  u={v_initial} m/s  a={acceleration} m/s²",
            formula    = formula, steps=steps, answer=answer)
        self.u   = v_initial
        self.a   = acceleration
        t_arr    = np.linspace(0, total_time, 300)
        self.x   = self.u * t_arr + 0.5 * self.a * t_arr ** 2
        self.v   = self.u + self.a * t_arr
        self.t_arr = t_arr
        self._setup_animation()

    def _setup_animation(self):
        t  = self.t
        ax = self.ax_anim
        x_max = max(abs(self.x)) * 1.3 or 10
        ax.set_xlim(-x_max, x_max)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color=t["accent"], linewidth=1, alpha=0.4)
        ax.set_xlabel("Position (m)", color=t["text"])
        ax.set_yticks([])

        self.block = FancyBboxPatch((-0.5, -0.4), 1, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=t["accent"],
                                    edgecolor=t["accent2"],
                                    linewidth=2, zorder=5)
        ax.add_patch(self.block)
        self.vel_text = ax.text(0, 0.9, "", ha="center",
                                color=t["text"], fontsize=9,
                                transform=ax.transAxes)
        self.arrow = FancyArrowPatch((0, 0), (1, 0),
                                     arrowstyle="-|>",
                                     color=t["colors"][2],
                                     linewidth=2.5,
                                     mutation_scale=18)
        ax.add_patch(self.arrow)

        def update(frame):
            if self.paused:
                return self.block, self.vel_text, self.arrow
            i  = frame % len(self.t_arr)
            xp = self.x[i]
            vp = self.v[i]
            self.block.set_x(xp - 0.5)
            scale = min(abs(vp) * 0.4, x_max * 0.4)
            sign  = 1 if vp >= 0 else -1
            self.arrow.set_positions((xp, 0.2),
                                     (xp + sign * scale, 0.2))
            self.vel_text.set_text(
                f"t={self.t_arr[i]:.2f}s  x={xp:.2f}m  v={vp:.2f}m/s")
            self._pulse_button(frame)
            return self.block, self.vel_text, self.arrow

        self.anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.t_arr),
            interval=30, blit=True, repeat=True)


# ─────────────────────────────────────────────
# ── GRAVITATION: Orbital Animation (Deep Space)
# ─────────────────────────────────────────────

class OrbitalAnimator(BaseAnimator):
    """Animates a planet orbiting a star with glow and trail."""

    def __init__(self, central_mass, orbital_radius,
                 formula, steps, answer):
        G  = 6.674e-11
        T  = 2 * math.pi * math.sqrt(orbital_radius ** 3 / (G * central_mass))
        v  = math.sqrt(G * central_mass / orbital_radius)
        super().__init__(
            theme_name = "deepspace",
            title      = (f"Orbital Motion  |  "
                          f"r={orbital_radius:.2e}m  "
                          f"T={T:.2f}s"),
            formula    = formula, steps=steps, answer=answer)
        omega       = 2 * math.pi / T
        self.t_arr  = np.linspace(0, T, 360)
        self.x_arr  = orbital_radius * np.cos(omega * self.t_arr)
        self.y_arr  = orbital_radius * np.sin(omega * self.t_arr)
        self.r      = orbital_radius
        self._setup_animation()

    def _setup_animation(self):
        t  = self.t
        ax = self.ax_anim
        r  = self.r
        ax.set_xlim(-r * 1.4, r * 1.4)
        ax.set_ylim(-r * 1.4, r * 1.4)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", color=t["text"])
        ax.set_ylabel("y (m)", color=t["text"])

        # Ghost orbit
        theta = np.linspace(0, 2 * math.pi, 360)
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color=t["accent"], linewidth=1,
                linestyle="--", alpha=0.2)

        # Star glow
        for size, alpha in [(r * 0.18, 0.08), (r * 0.10, 0.15),
                            (r * 0.06, 0.5)]:
            ax.add_patch(Circle((0, 0), size,
                                color=t["colors"][2], alpha=alpha))
        ax.add_patch(Circle((0, 0), r * 0.04,
                            color="#ffffff", zorder=5))

        self.trail, = ax.plot([], [], color=t["accent"],
                              linewidth=1.5, alpha=0.5)
        self.planet = Circle((r, 0), r * 0.05,
                             color=t["accent2"], zorder=6)
        ax.add_patch(self.planet)
        self.info = ax.text(0.05, 0.95, "",
                            transform=ax.transAxes,
                            color=t["text"], fontsize=8.5)
        trail_len = 60

        def update(frame):
            if self.paused:
                return self.trail, self.planet, self.info
            i = frame % len(self.t_arr)
            x = self.x_arr[i]
            y = self.y_arr[i]
            start = max(0, i - trail_len)
            self.trail.set_data(self.x_arr[start:i],
                                self.y_arr[start:i])
            self.planet.set_center((x, y))
            self.info.set_text(
                f"t={self.t_arr[i]:.1f}s  "
                f"x={x:.2e}m  y={y:.2e}m")
            self._pulse_button(frame)
            return self.trail, self.planet, self.info

        self.anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.t_arr),
            interval=25, blit=True, repeat=True)


# ─────────────────────────────────────────────
# ── ENERGETICS: SHM Spring Animation (Neon)
# ─────────────────────────────────────────────

class SHMAnimator(BaseAnimator):
    """
    Animates SHM spring system with:
    - Bouncing block on spring
    - Live KE/PE bar chart
    """

    def __init__(self, spring_constant, amplitude, mass,
                 formula, steps, answer):
        omega = math.sqrt(spring_constant / mass)
        T     = 2 * math.pi / omega
        super().__init__(
            theme_name = "neon",
            title      = (f"SHM  |  k={spring_constant} N/m  "
                          f"A={amplitude} m  m={mass} kg"),
            formula    = formula, steps=steps, answer=answer)
        self.k     = spring_constant
        self.A     = amplitude
        self.m     = mass
        self.omega = omega
        self.t_arr = np.linspace(0, 2 * T, 300)
        self.x_arr = amplitude * np.cos(omega * self.t_arr)
        self.v_arr = -amplitude * omega * np.sin(omega * self.t_arr)
        self.E_tot = 0.5 * spring_constant * amplitude ** 2
        self._setup_animation()

    def _setup_animation(self):
        t  = self.t
        ax = self.ax_anim

        # Split animation area: top = spring, bottom = energy bars
        ax.set_xlim(-self.A * 2.5, self.A * 2.5)
        ax.set_ylim(-2.5, 3.5)
        ax.set_yticks([])
        ax.set_xlabel("Position (m)", color=t["text"])

        # Wall
        ax.axvline(-self.A * 2, color=t["text"],
                   linewidth=3, alpha=0.5)
        # Equilibrium
        ax.axvline(0, color=t["accent"],
                   linewidth=0.8, linestyle="--", alpha=0.3)

        self.spring_line, = ax.plot([], [], color=t["accent"],
                                    linewidth=2.5, zorder=3)
        self.block = FancyBboxPatch((0, 0.5), self.A * 0.4, 1.2,
                                    boxstyle="round,pad=0.05",
                                    facecolor=t["accent2"],
                                    edgecolor=t["text"],
                                    linewidth=1.5, zorder=5)
        ax.add_patch(self.block)

        # Energy bars background
        ax.add_patch(FancyBboxPatch((-self.A * 2.4, -2.3),
                                    self.A * 1.8, 1.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor=t["axes_bg"],
                                    edgecolor=t["accent"],
                                    linewidth=1, alpha=0.7))
        ax.text(-self.A * 2.35, -0.6, "KE",
                color=t["colors"][0], fontsize=8, fontweight="bold")
        ax.text(-self.A * 1.25, -0.6, "PE",
                color=t["colors"][1], fontsize=8, fontweight="bold")

        self.ke_bar = ax.barh(-1.5, 0, height=0.5,
                              left=-self.A * 2.3,
                              color=t["colors"][0], alpha=0.85)[0]
        self.pe_bar = ax.barh(-2.0, 0, height=0.5,
                              left=-self.A * 2.3,
                              color=t["colors"][1], alpha=0.85)[0]
        self.energy_text = ax.text(0.62, 0.08, "",
                                   transform=ax.transAxes,
                                   color=t["text"], fontsize=8)

        wall_x = -self.A * 2

        def _spring_coils(x_start, x_end, y=1.1, n_coils=8):
            """Generates spring coil points."""
            xs = np.linspace(x_start, x_end, n_coils * 10)
            ys = y + 0.25 * np.sin(
                np.linspace(0, n_coils * 2 * math.pi, len(xs)))
            return xs, ys

        def update(frame):
            if self.paused:
                return (self.spring_line, self.block,
                        self.ke_bar, self.pe_bar, self.energy_text)
            i  = frame % len(self.t_arr)
            x  = self.x_arr[i]
            v  = self.v_arr[i]
            ke = 0.5 * self.m * v ** 2
            pe = 0.5 * self.k * x ** 2

            sx, sy = _spring_coils(wall_x, x - self.A * 0.2)
            self.spring_line.set_data(sx, sy)
            self.block.set_x(x - self.A * 0.2)
            bar_scale = self.A * 1.6 / self.E_tot
            self.ke_bar.set_width(ke * bar_scale)
            self.pe_bar.set_width(pe * bar_scale)
            self.energy_text.set_text(
                f"KE={ke:.3f}J  PE={pe:.3f}J  Total={ke+pe:.3f}J")
            self._pulse_button(frame)
            return (self.spring_line, self.block,
                    self.ke_bar, self.pe_bar, self.energy_text)

        self.anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.t_arr),
            interval=30, blit=True, repeat=True)


# ─────────────────────────────────────────────
# ── QUANTUM: Radioactive Decay Animation (Deep Space)
# ─────────────────────────────────────────────

class RadioactiveDecayAnimator(BaseAnimator):
    """
    Animates radioactive decay:
    - Grid of dots (nuclei) disappearing over time
    - Decay curve drawing itself live
    """

    def __init__(self, N_initial, half_life, total_time,
                 formula, steps, answer):
        super().__init__(
            theme_name = "deepspace",
            title      = (f"Radioactive Decay  |  "
                          f"N₀={N_initial:.0e}  t½={half_life}s"),
            formula    = formula, steps=steps, answer=answer)
        self.N0        = N_initial
        self.half_life = half_life
        lam            = math.log(2) / half_life
        self.t_arr     = np.linspace(0, total_time, 300)
        self.N_arr     = N_initial * np.exp(-lam * self.t_arr)
        self.A_arr     = lam * self.N_arr
        self._setup_animation()

    def _setup_animation(self):
        t  = self.t
        ax = self.ax_anim
        ax.set_xlim(0, self.t_arr[-1])
        ax.set_ylim(0, self.N0 * 1.1)
        ax.set_xlabel("Time (s)", color=t["text"])
        ax.set_ylabel("Nuclei N(t)", color=t["text"])

        # Ghost full curve
        ax.plot(self.t_arr, self.N_arr,
                color=t["accent"], linewidth=1,
                linestyle="--", alpha=0.15)
        # Half-life markers
        for i in range(1, 5):
            hl = i * self.half_life
            if hl < self.t_arr[-1]:
                ax.axvline(hl, color=t["colors"][2],
                           linewidth=0.8, linestyle=":",
                           alpha=0.4)
                ax.text(hl, self.N0 * 0.95,
                        f"t½×{i}", color=t["colors"][2],
                        fontsize=7, ha="center")

        self.decay_line, = ax.plot([], [], color=t["accent"],
                                   linewidth=2.5, zorder=5)
        self.dot, = ax.plot([], [], "o",
                            color=t["accent2"],
                            markersize=8, zorder=6)
        self.info = ax.text(0.55, 0.88, "",
                            transform=ax.transAxes,
                            color=t["text"], fontsize=8.5)
        ax.fill_between(self.t_arr, self.N_arr,
                        alpha=0.05, color=t["accent"])

        def update(frame):
            if self.paused:
                return self.decay_line, self.dot, self.info
            i = frame % len(self.t_arr)
            self.decay_line.set_data(self.t_arr[:i+1],
                                     self.N_arr[:i+1])
            self.dot.set_data([self.t_arr[i]], [self.N_arr[i]])
            pct = self.N_arr[i] / self.N0 * 100
            self.info.set_text(
                f"t={self.t_arr[i]:.1f}s\n"
                f"N={self.N_arr[i]:.3e}\n"
                f"Remaining: {pct:.1f}%\n"
                f"Activity: {self.A_arr[i]:.3e} Bq")
            self._pulse_button(frame)
            return self.decay_line, self.dot, self.info

        self.anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.t_arr),
            interval=25, blit=True, repeat=True)


# ─────────────────────────────────────────────
# ── QUANTUM: Bohr Electron Transition (Deep Space)
# ─────────────────────────────────────────────

class BohrTransitionAnimator(BaseAnimator):
    """
    Animates electron jumping between Bohr orbits
    with photon flash on transition.
    """

    def __init__(self, n_initial, n_final, Z=1,
                 formula="E = -13.6 × Z²/n² eV",
                 steps=None, answer=""):
        h   = 6.626e-34
        c   = 3e8
        e   = 1.602e-19
        E_H = 13.6
        dE  = abs(-E_H * Z**2 / n_final**2 - (-E_H * Z**2 / n_initial**2))
        lam = h * c / (dE * e) * 1e9
        if steps is None:
            steps = [
                f"E_i = -13.6×{Z}²/{n_initial}² = {-E_H*Z**2/n_initial**2:.2f} eV",
                f"E_f = -13.6×{Z}²/{n_final}² = {-E_H*Z**2/n_final**2:.2f} eV",
                f"ΔE  = {dE:.4f} eV",
                f"λ   = hc/ΔE = {lam:.2f} nm",
            ]
        super().__init__(
            theme_name = "deepspace",
            title      = f"Bohr Model  n={n_initial} → n={n_final}  (Z={Z})",
            formula    = formula, steps=steps,
            answer     = answer or f"λ = {lam:.2f} nm  |  ΔE = {dE:.4f} eV")
        self.n_i   = n_initial
        self.n_f   = n_final
        self.Z     = Z
        self.a0    = 5.292e-11
        self.frame_count = 0
        self._setup_animation()

    def _setup_animation(self):
        t  = self.t
        ax = self.ax_anim
        n_max = max(self.n_i, self.n_f) + 1
        r_max = self.a0 * n_max ** 2 / self.Z * 1.3
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Nucleus
        ax.add_patch(Circle((0, 0), r_max * 0.04,
                            color=t["colors"][2], zorder=5,
                            alpha=0.9))
        ax.text(0, 0, "p⁺", color="#fff",
                ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=6)

        # Draw orbits
        self.orbit_radii = {}
        for n in range(1, n_max + 1):
            r = self.a0 * n ** 2 / self.Z
            self.orbit_radii[n] = r
            theta = np.linspace(0, 2 * math.pi, 300)
            alpha = 0.5 if n in (self.n_i, self.n_f) else 0.2
            lw    = 1.8 if n in (self.n_i, self.n_f) else 0.8
            col   = t["colors"][n % len(t["colors"])]
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    color=col, linewidth=lw, alpha=alpha)
            ax.text(r + r_max * 0.02, 0, f"n={n}",
                    color=col, fontsize=7, va="center")

        # Electron
        r_start = self.orbit_radii[self.n_i]
        self.electron = Circle((r_start, 0),
                               r_max * 0.025,
                               color=t["accent"], zorder=7)
        ax.add_patch(self.electron)

        # Photon flash
        self.photon = ax.scatter([], [], s=200,
                                 color=t["colors"][4],
                                 alpha=0, zorder=8,
                                 marker="*")
        self.info = ax.text(0.02, 0.95, "",
                            transform=ax.transAxes,
                            color=t["text"], fontsize=8)

        ORBIT_FRAMES  = 90
        TRANS_FRAMES  = 30
        TOTAL         = ORBIT_FRAMES * 2 + TRANS_FRAMES

        def update(frame):
            if self.paused:
                return self.electron, self.photon, self.info
            f = frame % TOTAL
            if f < ORBIT_FRAMES:
                # Orbiting at n_initial
                n   = self.n_i
                r   = self.orbit_radii[n]
                ang = 2 * math.pi * f / ORBIT_FRAMES
                self.electron.set_center((r * math.cos(ang),
                                          r * math.sin(ang)))
                self.photon.set_offsets([[0, 0]])
                self.photon.set_alpha(0)
                self.info.set_text(f"Electron at n={self.n_i}")
            elif f < ORBIT_FRAMES + TRANS_FRAMES:
                # Transition flash
                prog = (f - ORBIT_FRAMES) / TRANS_FRAMES
                r_i  = self.orbit_radii[self.n_i]
                r_f  = self.orbit_radii[self.n_f]
                r    = r_i + (r_f - r_i) * prog
                self.electron.set_center((r, 0))
                flash_alpha = math.sin(prog * math.pi)
                self.photon.set_offsets([[r * 1.3, r * 0.3]])
                self.photon.set_alpha(flash_alpha)
                self.info.set_text("⚡ Transition!")
            else:
                # Orbiting at n_final
                n   = self.n_f
                r   = self.orbit_radii[n]
                ang = 2 * math.pi * (f - ORBIT_FRAMES - TRANS_FRAMES) / ORBIT_FRAMES
                self.electron.set_center((r * math.cos(ang),
                                          r * math.sin(ang)))
                self.photon.set_alpha(0)
                self.info.set_text(f"Electron at n={self.n_f}")
            self._pulse_button(frame)
            return self.electron, self.photon, self.info

        self.anim = animation.FuncAnimation(
            self.fig, update, frames=TOTAL,
            interval=30, blit=True, repeat=True)


# ─────────────────────────────────────────────
# ── THERMODYNAMICS: Carnot Cycle Animation (Sunset)
# ─────────────────────────────────────────────

class CarnotAnimator(BaseAnimator):
    """Animates the Carnot cycle tracing itself on a PV diagram."""

    def __init__(self, T_hot, T_cold, V1, V2,
             formula, steps, answer, n=1.0):
        super().__init__(
            theme_name = "sunset",
            title      = f"Carnot Cycle  |  T_h={T_hot}K  T_c={T_cold}K",
            formula    = formula, steps=steps, answer=answer)
        R     = 8.314
        gamma = 1.4
        V_AB  = np.linspace(V1, V2, 80)
        P_AB  = n * R * T_hot / V_AB
        V3    = V2 * (T_hot / T_cold) ** (1 / (gamma - 1))
        V_BC  = np.linspace(V2, V3, 80)
        P_BC  = (n * R * T_hot / V2 ** gamma) * V_BC ** (-gamma)
        V4    = V1 * (T_hot / T_cold) ** (1 / (gamma - 1))
        V_CD  = np.linspace(V3, V4, 80)
        P_CD  = n * R * T_cold / V_CD
        V_DA  = np.linspace(V4, V1, 80)
        P_DA  = (n * R * T_cold / V4 ** gamma) * V_DA ** (-gamma)
        self.Vs = np.concatenate([V_AB, V_BC, V_CD, V_DA])
        self.Ps = np.concatenate([P_AB, P_BC, P_CD, P_DA])
        self.segment_colors = (
            [self.t["colors"][0]] * 80 +
            [self.t["colors"][1]] * 80 +
            [self.t["colors"][2]] * 80 +
            [self.t["colors"][3]] * 80
        )
        self._setup_animation()

    def _setup_animation(self):
        t  = self.t
        ax = self.ax_anim
        ax.set_xlim(min(self.Vs) * 0.9, max(self.Vs) * 1.1)
        ax.set_ylim(min(self.Ps) * 0.9, max(self.Ps) * 1.1)
        ax.set_xlabel("Volume (m³)", color=t["text"])
        ax.set_ylabel("Pressure (Pa)", color=t["text"])
        # Ghost full cycle
        ax.plot(self.Vs, self.Ps, color=t["accent"],
                linewidth=1, linestyle="--", alpha=0.15)
        labels = ["Isothermal Exp.", "Adiabatic Exp.",
                  "Isothermal Comp.", "Adiabatic Comp."]
        for i, (label, col) in enumerate(
                zip(labels, [t["colors"][j] for j in range(4)])):
            ax.plot([], [], color=col, linewidth=2, label=label)
        ax.legend(facecolor=t["bg"], edgecolor=t["accent"],
                  labelcolor=t["text"], fontsize=7,
                  loc="upper right")

        self.cycle_line, = ax.plot([], [], linewidth=2.5, zorder=5)
        self.dot, = ax.plot([], [], "o",
                            color=t["accent2"],
                            markersize=9, zorder=6)
        self.stage_text = ax.text(0.05, 0.95, "",
                                  transform=ax.transAxes,
                                  color=t["text"], fontsize=9)
        stages = ["Isothermal Expansion",
                  "Adiabatic Expansion",
                  "Isothermal Compression",
                  "Adiabatic Compression"]

        def update(frame):
            if self.paused:
                return self.cycle_line, self.dot, self.stage_text
            i     = frame % len(self.Vs)
            stage = i // 80
            self.cycle_line.set_data(self.Vs[:i+1], self.Ps[:i+1])
            self.cycle_line.set_color(self.segment_colors[i])
            self.dot.set_data([self.Vs[i]], [self.Ps[i]])
            self.stage_text.set_text(
                f"Stage: {stages[min(stage, 3)]}\n"
                f"V={self.Vs[i]:.4f} m³  P={self.Ps[i]:.1f} Pa")
            self._pulse_button(frame)
            return self.cycle_line, self.dot, self.stage_text

        self.anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.Vs),
            interval=25, blit=True, repeat=True)


# ─────────────────────────────────────────────
# QUICK LAUNCH HELPERS
# ─────────────────────────────────────────────

def animate_projectile(v_initial, angle_deg, formula, steps, answer):
    """Launch projectile animation."""
    a = ProjectileAnimator(v_initial, angle_deg, formula, steps, answer)
    a.start()


def animate_1d_motion(v_initial, acceleration, total_time,
                      formula, steps, answer):
    """Launch 1D motion animation."""
    a = Motion1DAnimator(v_initial, acceleration, total_time,
                         formula, steps, answer)
    a.start()


def animate_orbit(central_mass, orbital_radius, formula, steps, answer):
    """Launch orbital animation."""
    a = OrbitalAnimator(central_mass, orbital_radius,
                        formula, steps, answer)
    a.start()


def animate_shm(spring_constant, amplitude, mass, formula, steps, answer):
    """Launch SHM spring animation."""
    a = SHMAnimator(spring_constant, amplitude, mass,
                    formula, steps, answer)
    a.start()


def animate_decay(N_initial, half_life, total_time,
                  formula, steps, answer):
    """Launch radioactive decay animation."""
    a = RadioactiveDecayAnimator(N_initial, half_life, total_time,
                                 formula, steps, answer)
    a.start()


def animate_bohr(n_initial, n_final, Z=1):
    """Launch Bohr electron transition animation."""
    a = BohrTransitionAnimator(n_initial, n_final, Z)
    a.start()


def animate_carnot(T_hot, T_cold, V1, V2,
                   formula="η = 1 - T_c/T_h",
                   steps=None, answer="", n=1.0):
    """Launch Carnot cycle animation."""
    if steps is None:
        eta = 1 - T_cold / T_hot
        steps = [
            f"T_hot  = {T_hot} K",
            f"T_cold = {T_cold} K",
            f"η = 1 - {T_cold}/{T_hot}",
            f"η = {eta:.4f} = {eta*100:.2f}%",
        ]
        answer = answer or f"Efficiency η = {eta*100:.2f}%"
    a = CarnotAnimator(T_hot, T_cold, V1, V2, n, formula, steps, answer)
    a.start()