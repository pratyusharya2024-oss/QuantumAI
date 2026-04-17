"""
plotter.py
==========
Visualization Module: Static Graphs
Covers: All 5 physics modules with themed plots
Themes:
    - Kinematics    → Sunset
    - Gravitation   → Deep Space
    - Energetics    → Neon
    - Quantum       → Deep Space
    - Thermodynamics→ Sunset
Supports: Single / Double / Triple plots, same or mixed modules
Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.figure import Figure


# ─────────────────────────────────────────────
# THEMES
# ─────────────────────────────────────────────

THEMES = {

    "sunset": {
        "bg":           "#0d0a0e",
        "axes_bg":      "#110f15",
        "grid":         "#2a1f2e",
        "title":        "#ff9a5c",
        "label":        "#f4c97a",
        "tick":         "#c97b4b",
        "spine":        "#3d2b1f",
        "colors": [
            "#ff6b35",   # orange
            "#ff9a5c",   # peach
            "#f4c97a",   # yellow
            "#e05c5c",   # red
            "#ffb347",   # amber
            "#ff7f7f",   # salmon
        ],
        "legend_bg":    "#1a1218",
        "legend_edge":  "#3d2b1f",
    },

    "deepspace": {
        "bg":           "#03030f",
        "axes_bg":      "#05051a",
        "grid":         "#0d1133",
        "title":        "#7eb8f7",
        "label":        "#a0c4ff",
        "tick":         "#5a7fbf",
        "spine":        "#0d1f4a",
        "colors": [
            "#4fc3f7",   # sky blue
            "#7986cb",   # indigo
            "#b39ddb",   # lavender
            "#4dd0e1",   # cyan
            "#9575cd",   # purple
            "#80cbc4",   # teal
        ],
        "legend_bg":    "#05051a",
        "legend_edge":  "#0d1f4a",
    },

    "neon": {
        "bg":           "#050508",
        "axes_bg":      "#08080f",
        "grid":         "#121220",
        "title":        "#00ff9f",
        "label":        "#ff00ff",
        "tick":         "#00cfff",
        "spine":        "#1a0033",
        "colors": [
            "#00ff9f",   # neon green
            "#ff00ff",   # magenta
            "#00cfff",   # electric blue
            "#ff6600",   # neon orange
            "#ffe600",   # neon yellow
            "#ff0066",   # hot pink
        ],
        "legend_bg":    "#08080f",
        "legend_edge":  "#1a0033",
    },
}

# Module → theme mapping
MODULE_THEME = {
    "kinematics":      "sunset",
    "gravitation":     "deepspace",
    "energetics":      "neon",
    "quantum":         "deepspace",
    "thermodynamics":  "sunset",
}


# ─────────────────────────────────────────────
# CORE STYLE HELPERS
# ─────────────────────────────────────────────

def _apply_theme(fig, axes, theme_name):
    """Applies a theme to figure and all axes."""
    t = THEMES[theme_name]
    fig.patch.set_facecolor(t["bg"])
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(t["axes_bg"])
        ax.grid(True, color=t["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(colors=t["tick"], which="both", labelsize=9)
        ax.xaxis.label.set_color(t["label"])
        ax.yaxis.label.set_color(t["label"])
        ax.title.set_color(t["title"])
        for spine in ax.spines.values():
            spine.set_edgecolor(t["spine"])


def _style_legend(ax, theme_name):
    """Styles the legend for a given axis."""
    t = THEMES[theme_name]
    leg = ax.legend(
        facecolor=t["legend_bg"],
        edgecolor=t["legend_edge"],
        labelcolor=t["label"],
        fontsize=8,
        framealpha=0.9
    )
    return leg


def _make_figure(n_plots, theme_name, figsize=None):
    """Creates figure and axes layout for 1, 2, or 3 subplots."""
    t = THEMES[theme_name]
    if n_plots == 1:
        fs = figsize or (9, 5)
        fig, ax = plt.subplots(figsize=fs)
        axes = [ax]
    elif n_plots == 2:
        fs = figsize or (14, 5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fs)
        axes = [ax1, ax2]
    elif n_plots == 3:
        fs = figsize or (18, 5)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=fs)
        axes = [ax1, ax2, ax3]
    else:
        raise ValueError("n_plots must be 1, 2, or 3.")
    fig.patch.set_facecolor(t["bg"])  # type: ignore
    plt.tight_layout(pad=3.0)
    return fig, axes


# ─────────────────────────────────────────────
# ── KINEMATICS PLOTS  (Sunset)
# ─────────────────────────────────────────────

def plot_projectile_trajectory(df, ax=None, theme="sunset"):
    """Plots projectile x vs y trajectory."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["x (m)"], df["y (m)"],
            color=t["colors"][0], linewidth=2.2, label="Trajectory")
    ax.fill_between(df["x (m)"], df["y (m)"],
                    alpha=0.12, color=t["colors"][0])
    ax.set_title("Projectile Trajectory", fontsize=13, fontweight="bold")
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Height (m)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_velocity_time(df, ax=None, theme="sunset"):
    """Plots velocity components over time."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["time (s)"], df["vx (m/s)"],
            color=t["colors"][0], linewidth=2, label="vx (m/s)")
    ax.plot(df["time (s)"], df["vy (m/s)"],
            color=t["colors"][1], linewidth=2, label="vy (m/s)", linestyle="--")
    ax.axhline(0, color=t["tick"], linewidth=0.8, linestyle=":")
    ax.set_title("Velocity Components vs Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_position_time(df, ax=None, theme="sunset"):
    """Plots x and y position over time."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["time (s)"], df["x (m)"],
            color=t["colors"][2], linewidth=2, label="x (m)")
    ax.plot(df["time (s)"], df["y (m)"],
            color=t["colors"][0], linewidth=2, label="y (m)", linestyle="--")
    ax.set_title("Position vs Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_1d_motion(df, ax=None, theme="sunset"):
    """Plots 1D position and velocity over time."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["time (s)"], df["position (m)"],
            color=t["colors"][0], linewidth=2, label="Position (m)")
    ax2 = ax.twinx()
    ax2.plot(df["time (s)"], df["velocity (m/s)"],
             color=t["colors"][1], linewidth=2,
             linestyle="--", label="Velocity (m/s)")
    ax2.tick_params(colors=t["tick"])
    ax2.yaxis.label.set_color(t["label"])
    ax.set_title("1D Motion Simulation", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax2.set_ylabel("Velocity (m/s)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────
# ── GRAVITATION PLOTS  (Deep Space)
# ─────────────────────────────────────────────

def plot_orbital_path(df, ax=None, theme="deepspace"):
    """Plots circular orbital path (x vs y)."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["x (m)"], df["y (m)"],
            color=t["colors"][0], linewidth=2, label="Orbit")
    ax.plot(0, 0, "o", color=t["colors"][2], markersize=10, label="Central Body")
    ax.set_aspect("equal")
    ax.set_title("Orbital Path", fontsize=13, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_gravitational_field(df, ax=None, theme="deepspace"):
    """Plots gravitational field strength vs distance."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["distance (m)"], df["field strength (m/s²)"],
            color=t["colors"][1], linewidth=2.2, label="g(r)")
    ax.fill_between(df["distance (m)"], df["field strength (m/s²)"],
                    alpha=0.1, color=t["colors"][1])
    ax.set_title("Gravitational Field Strength vs Distance",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Field Strength (m/s²)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_orbital_velocity_time(df, ax=None, theme="deepspace"):
    """Plots vx and vy over time for an orbit."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["time (s)"], df["vx (m/s)"],
            color=t["colors"][0], linewidth=2, label="vx")
    ax.plot(df["time (s)"], df["vy (m/s)"],
            color=t["colors"][3], linewidth=2, linestyle="--", label="vy")
    ax.set_title("Orbital Velocity Components", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────
# ── ENERGETICS PLOTS  (Neon)
# ─────────────────────────────────────────────

def plot_shm_energy(df, ax=None, theme="neon"):
    """Plots KE, PE, Total Energy in SHM over time."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["time (s)"], df["KE (J)"],
            color=t["colors"][0], linewidth=2, label="KE")
    ax.plot(df["time (s)"], df["PE (J)"],
            color=t["colors"][1], linewidth=2, label="PE", linestyle="--")
    ax.plot(df["time (s)"], df["Total Energy (J)"],
            color=t["colors"][2], linewidth=1.5,
            linestyle=":", label="Total E")
    ax.set_title("SHM Energy vs Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_projectile_energy(df, ax=None, theme="neon"):
    """Plots KE, PE, Total Energy during projectile flight."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["time (s)"], df["KE (J)"],
            color=t["colors"][0], linewidth=2, label="KE")
    ax.plot(df["time (s)"], df["PE (J)"],
            color=t["colors"][3], linewidth=2, linestyle="--", label="PE")
    ax.plot(df["time (s)"], df["Total Energy (J)"],
            color=t["colors"][2], linewidth=1.5,
            linestyle=":", label="Total E")
    ax.set_title("Projectile Energy vs Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_shm_displacement(df, ax=None, theme="neon"):
    """Plots SHM displacement and velocity over time."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["time (s)"], df["displacement (m)"],
            color=t["colors"][1], linewidth=2, label="Displacement")
    ax2 = ax.twinx()
    ax2.plot(df["time (s)"], df["velocity (m/s)"],
             color=t["colors"][4], linewidth=2,
             linestyle="--", label="Velocity")
    ax2.tick_params(colors=t["tick"])
    ax.set_title("SHM Displacement & Velocity", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (m)")
    ax2.set_ylabel("Velocity (m/s)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────
# ── QUANTUM MECHANICS PLOTS  (Deep Space)
# ─────────────────────────────────────────────

def plot_radioactive_decay(df, ax=None, theme="deepspace"):
    """Plots nuclei remaining and activity vs time."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(df["time"], df["nuclei remaining"],
            color=t["colors"][0], linewidth=2.2, label="Nuclei N(t)")
    ax.fill_between(df["time"], df["nuclei remaining"],
                    alpha=0.1, color=t["colors"][0])
    ax2 = ax.twinx()
    ax2.plot(df["time"], df["activity (Bq)"],
             color=t["colors"][1], linewidth=2,
             linestyle="--", label="Activity A(t)")
    ax2.tick_params(colors=t["tick"])
    ax2.yaxis.label.set_color(t["label"])
    ax2.set_ylabel("Activity (Bq)")
    ax.set_title("Radioactive Decay", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Nuclei Remaining")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_hydrogen_energy_levels(df, ax=None, theme="deepspace"):
    """Plots hydrogen atom energy levels as horizontal lines."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    for _, row in df.iterrows():
        n   = int(row["n"])
        E   = row["energy (eV)"]
        col = t["colors"][n % len(t["colors"])]
        ax.hlines(E, xmin=0.2, xmax=0.8,
                  colors=col, linewidth=2.5)
        ax.text(0.82, E, f"n={n}  {E:.2f} eV",
                color=col, fontsize=8, va="center")
    ax.set_xlim(0, 1.2)
    ax.set_title("Hydrogen Energy Levels (Bohr Model)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Energy (eV)")
    ax.set_xticks([])
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_wavefunction_particle_box(n_levels, L, ax=None, theme="deepspace"):
    """
    Plots particle-in-a-box wavefunctions for given quantum levels.
    n_levels: list of n values e.g. [1, 2, 3]
    L: box length in meters
    """
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    x = np.linspace(0, L, 500)
    for i, n in enumerate(n_levels):
        psi = np.sqrt(2 / L) * np.sin(n * np.pi * x / L)
        col = t["colors"][i % len(t["colors"])]
        ax.plot(x, psi, color=col, linewidth=2, label=f"n={n}")
        ax.fill_between(x, psi, alpha=0.07, color=col)
    ax.axhline(0, color=t["tick"], linewidth=0.8)
    ax.set_title("Particle in a Box — Wavefunctions",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("ψ(x)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────
# ── THERMODYNAMICS PLOTS  (Sunset)
# ─────────────────────────────────────────────

def plot_pv_diagram(volumes, pressures, label="Process",
                    ax=None, theme="sunset"):
    """Plots a PV diagram for any thermodynamic process."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(volumes, pressures,
            color=t["colors"][0], linewidth=2.5, label=label)
    ax.fill_between(volumes, pressures, alpha=0.1, color=t["colors"][0])
    ax.set_title("PV Diagram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Volume (m³)")
    ax.set_ylabel("Pressure (Pa)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_carnot_cycle(T_hot, T_cold, V1, V2, n=1.0,
                      ax=None, theme="sunset"):
    """
    Plots a Carnot cycle on a PV diagram.
    T_hot, T_cold: reservoir temps in K
    V1, V2: min and max volumes for isothermal expansion
    """
    t   = THEMES[theme]
    R   = 8.314
    gamma = 1.4
    solo  = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]

    # Isothermal expansion (A→B) at T_hot
    V_AB = np.linspace(V1, V2, 200)
    P_AB = n * R * T_hot / V_AB

    # Adiabatic expansion (B→C)
    V3   = V2 * (T_hot / T_cold) ** (1 / (gamma - 1))
    V_BC = np.linspace(V2, V3, 200)
    P_BC = (n * R * T_hot / V2 ** gamma) * V_BC ** (-gamma)

    # Isothermal compression (C→D) at T_cold
    V4   = V1 * (T_hot / T_cold) ** (1 / (gamma - 1))
    V_CD = np.linspace(V3, V4, 200)
    P_CD = n * R * T_cold / V_CD

    # Adiabatic compression (D→A)
    V_DA = np.linspace(V4, V1, 200)
    P_DA = (n * R * T_cold / V4 ** gamma) * V_DA ** (-gamma)

    ax.plot(V_AB, P_AB, color=t["colors"][0], lw=2.2, label="Isothermal Exp.")
    ax.plot(V_BC, P_BC, color=t["colors"][1], lw=2.2, label="Adiabatic Exp.")
    ax.plot(V_CD, P_CD, color=t["colors"][2], lw=2.2, label="Isothermal Comp.")
    ax.plot(V_DA, P_DA, color=t["colors"][3], lw=2.2, label="Adiabatic Comp.")

    ax.set_title("Carnot Cycle — PV Diagram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Volume (m³)")
    ax.set_ylabel("Pressure (Pa)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_temperature_time(times, temperatures, label="Heating",
                          ax=None, theme="sunset"):
    """Plots temperature vs time for heating/cooling."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(times, temperatures,
            color=t["colors"][0], linewidth=2.5, label=label)
    ax.fill_between(times, temperatures,
                    alpha=0.12, color=t["colors"][1])
    ax.set_title("Temperature vs Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (K)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


def plot_entropy_temperature(temperatures, entropies,
                             ax=None, theme="sunset"):
    """Plots entropy change vs temperature (TS diagram)."""
    t = THEMES[theme]
    solo = ax is None
    if solo:
        fig, axes = _make_figure(1, theme)
        ax = axes[0]
    ax.plot(temperatures, entropies,
            color=t["colors"][2], linewidth=2.5, label="ΔS")
    ax.set_title("Entropy vs Temperature (TS Diagram)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Entropy (J/K)")
    _apply_theme(plt.gcf() if solo else ax.figure, [ax], theme)
    _style_legend(ax, theme)
    if solo:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────
# MULTI PLOT — Double & Triple
# ─────────────────────────────────────────────

def multi_plot(plot_configs, layout="horizontal"):
    """
    Renders 2 or 3 plots together — same or mixed modules.

    plot_configs: list of dicts, each with keys:
        - "func"   : the plot function to call
        - "args"   : positional args for that function (list)
        - "kwargs" : keyword args (dict), e.g. {"theme": "neon"}
        - "theme"  : theme name string (used for figure bg)

    layout: "horizontal" (side by side) only for now

    Example — double plot:
        multi_plot([
            {"func": plot_projectile_trajectory,
             "args": [proj_df], "kwargs": {}, "theme": "sunset"},
            {"func": plot_shm_energy,
             "args": [shm_df],  "kwargs": {}, "theme": "neon"},
        ])

    Example — triple plot:
        multi_plot([
            {"func": plot_projectile_trajectory,
             "args": [proj_df], "kwargs": {}, "theme": "sunset"},
            {"func": plot_shm_energy,
             "args": [shm_df],  "kwargs": {}, "theme": "neon"},
            {"func": plot_radioactive_decay,
             "args": [decay_df],"kwargs": {}, "theme": "deepspace"},
        ])
    """
    n = len(plot_configs)
    if n not in (2, 3):
        raise ValueError("multi_plot supports 2 or 3 plots only.")

    # Use first config's theme for overall figure background
    base_theme = plot_configs[0]["theme"]
    fig, axes  = _make_figure(n, base_theme)

    for i, cfg in enumerate(plot_configs):
        func   = cfg["func"]
        args   = cfg.get("args", [])
        kwargs = cfg.get("kwargs", {})
        theme  = cfg.get("theme", base_theme)
        # Inject ax and theme into the function
        func(*args, ax=axes[i], theme=theme, **kwargs)
        # Re-apply theme since ax was passed externally
        _apply_theme(fig, [axes[i]], theme)

    plt.tight_layout(pad=2.5)
    plt.show()