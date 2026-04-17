from __future__ import annotations
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import math
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

# modules kinematics.py
from sympy import Symbol, Eq, symbols
from Core.logic_engine import Logic_Engine
 
def register(engine: Logic_Engine) -> None:
    u, v, a, t, s, g, theta = symbols('u v a t s g theta')
    engine.register_formula('velocity',                Eq(v, u + a * t))
    engine.register_formula('displacement',            Eq(s, u * t + (a * t**2) / 2))
    engine.register_formula('final_velocity_squared',  Eq(v**2, u**2 + 2 * a * s))
    engine.register_formula('displacement_avg',        Eq(s, ((u + v) / 2) * t))

DATA_FOLDER = Path.cwd() / "data"
GRAVITY = 9.81

__all__ = [
    "calculate_speed",
    "calculate_distance",
    "calculate_time",
    "calculate_average_velocity",
    "calculate_acceleration",
    "velocity_from_acceleration",
    "displacement_uat",
    "velocity_squared",
    "displacement_avg_velocity",
    "final_velocity_no_time",
    "time_to_stop",
    "stopping_distance",
    "projectile_time_of_flight",
    "projectile_max_height",
    "projectile_range",
    "projectile_optimal_angle",
    "projectile_initial_components",
    "projectile_velocity_at_time",
    "projectile_position_at_time",
    "relative_velocity_1d",
    "relative_velocity_2d",
    "angular_velocity",
    "centripetal_acceleration",
    "centripetal_force",
    "period_of_revolution",
    "frequency_of_revolution",
    "simulate_1d_motion",
    "simulate_projectile",
    "save_simulation_to_csv",
    "projectile_summary",
    "motion_summary",
]


# ─────────────────────────────────────────────
# BASIC LEVEL — School / Introductory Formulas
# ─────────────────────────────────────────────


def calculate_speed(distance: float, time: float) -> float:
    """Speed = Distance / Time."""
    if time == 0:
        raise ValueError("Time cannot be zero.")
    return distance / time



def calculate_distance(speed: float, time: float) -> float:
    """Distance = Speed × Time."""
    return speed * time



def calculate_time(distance: float, speed: float) -> float:
    """Time = Distance / Speed."""
    if speed == 0:
        raise ValueError("Speed cannot be zero.")
    return distance / speed



def calculate_average_velocity(displacement: float, time: float) -> float:
    """Average velocity = Displacement / Time."""
    if time == 0:
        raise ValueError("Time cannot be zero.")
    return displacement / time



def calculate_acceleration(v_final: float, v_initial: float, time: float) -> float:
    """Acceleration = (v_final - v_initial) / time."""
    if time == 0:
        raise ValueError("Time cannot be zero.")
    return (v_final - v_initial) / time


# ─────────────────────────────────────────────
# GENERAL / HIGH SCHOOL — Equations of Motion
# ─────────────────────────────────────────────


def velocity_from_acceleration(v_initial: float, acceleration: float, time: float) -> float:
    """Final velocity: v = u + at."""
    return v_initial + acceleration * time



def displacement_uat(v_initial: float, time: float, acceleration: float) -> float:
    """Displacement: s = ut + 0.5 a t²."""
    return v_initial * time + 0.5 * acceleration * time ** 2



def velocity_squared(v_initial: float, acceleration: float, displacement: float) -> float:
    """Final velocity from v² = u² + 2as."""
    value = v_initial ** 2 + 2 * acceleration * displacement
    if value < 0:
        raise ValueError("Negative value under square root — check inputs.")
    return math.sqrt(value)



def displacement_avg_velocity(v_initial: float, v_final: float, time: float) -> float:
    """Displacement using average velocity: s = ((u + v) / 2) × t."""
    return ((v_initial + v_final) / 2) * time



def final_velocity_no_time(v_initial: float, acceleration: float, displacement: float) -> float:
    """Alias for v² = u² + 2as."""
    return velocity_squared(v_initial, acceleration, displacement)



def time_to_stop(v_initial: float, acceleration: float) -> float:
    """Time to reach zero velocity under constant acceleration."""
    if acceleration == 0:
        raise ValueError("Acceleration cannot be zero.")
    return -v_initial / acceleration



def stopping_distance(v_initial: float, acceleration: float) -> float:
    """Distance traveled while decelerating to zero velocity."""
    if acceleration >= 0:
        raise ValueError("Acceleration must be negative (deceleration).")
    return -(v_initial ** 2) / (2 * acceleration)


# ─────────────────────────────────────────────
# PROJECTILE MOTION
# ─────────────────────────────────────────────


def projectile_time_of_flight(v_initial: float, angle_deg: float, g: float = GRAVITY) -> float:
    """Total time of flight for ground-to-ground projectile motion."""
    angle_rad = math.radians(angle_deg)
    return (2 * v_initial * math.sin(angle_rad)) / g



def projectile_max_height(v_initial: float, angle_deg: float, g: float = GRAVITY) -> float:
    """Maximum height reached during projectile motion."""
    angle_rad = math.radians(angle_deg)
    return (v_initial * math.sin(angle_rad)) ** 2 / (2 * g)



def projectile_range(v_initial: float, angle_deg: float, g: float = GRAVITY) -> float:
    """Horizontal range of a projectile launched from and landing at the same height."""
    angle_rad = math.radians(angle_deg)
    return (v_initial ** 2 * math.sin(2 * angle_rad)) / g



def projectile_optimal_angle() -> float:
    """Optimal launch angle for maximum range on level ground."""
    return 45.0



def projectile_initial_components(v_initial: float, angle_deg: float) -> Tuple[float, float]:
    """Horizontal and vertical velocity components at launch."""
    angle_rad = math.radians(angle_deg)
    return v_initial * math.cos(angle_rad), v_initial * math.sin(angle_rad)



def projectile_velocity_at_time(v_initial: float, angle_deg: float, t: float, g: float = GRAVITY) -> Tuple[float, float]:
    """Velocity components of a projectile at time t."""
    vx, vy0 = projectile_initial_components(v_initial, angle_deg)
    return vx, vy0 - g * t



def projectile_position_at_time(v_initial: float, angle_deg: float, t: float, g: float = GRAVITY) -> Tuple[float, float]:
    """Position of a projectile at time t."""
    vx, vy0 = projectile_initial_components(v_initial, angle_deg)
    x = vx * t
    y = vy0 * t - 0.5 * g * t ** 2
    return x, y


# ─────────────────────────────────────────────
# RELATIVE MOTION
# ─────────────────────────────────────────────


def relative_velocity_1d(v_a: float, v_b: float) -> float:
    """Velocity of A relative to B in one dimension."""
    return v_a - v_b



def relative_velocity_2d(v_a: Sequence[float], v_b: Sequence[float]) -> Tuple[float, float]:
    """Relative velocity in two dimensions with magnitude and direction."""
    if len(v_a) != 2 or len(v_b) != 2:
        raise ValueError("Both v_a and v_b must be 2D vectors of length 2.")
    rel_vx = v_a[0] - v_b[0]
    rel_vy = v_a[1] - v_b[1]
    magnitude = math.hypot(rel_vx, rel_vy)
    angle = math.degrees(math.atan2(rel_vy, rel_vx))
    return magnitude, angle


# ─────────────────────────────────────────────
# CIRCULAR MOTION
# ─────────────────────────────────────────────


def angular_velocity(linear_velocity: float, radius: float) -> float:
    """Angular velocity: ω = v / r."""
    if radius == 0:
        raise ValueError("Radius cannot be zero.")
    return linear_velocity / radius



def centripetal_acceleration(linear_velocity: float, radius: float) -> float:
    """Centripetal acceleration: a_c = v² / r."""
    if radius == 0:
        raise ValueError("Radius cannot be zero.")
    return linear_velocity ** 2 / radius



def centripetal_force(mass: float, linear_velocity: float, radius: float) -> float:
    """Centripetal force required to keep an object in circular motion."""
    return mass * centripetal_acceleration(linear_velocity, radius)



def period_of_revolution(radius: float, linear_velocity: float) -> float:
    """Period of revolution for circular motion."""
    if linear_velocity == 0:
        raise ValueError("Velocity cannot be zero.")
    return (2 * math.pi * radius) / linear_velocity



def frequency_of_revolution(radius: float, linear_velocity: float) -> float:
    """Frequency of revolution for circular motion."""
    return 1 / period_of_revolution(radius, linear_velocity)


# ─────────────────────────────────────────────
# SIMULATION — Numerical (NumPy-based)
# ─────────────────────────────────────────────


def simulate_1d_motion(v_initial: float, acceleration: float, total_time: float, steps: int = 1000) -> pd.DataFrame:
    """Simulates 1D motion and returns a DataFrame with time, position, and velocity."""
    if total_time <= 0:
        raise ValueError("Total time must be positive.")
    if steps < 2:
        raise ValueError("Steps must be at least 2.")
    t = np.linspace(0, total_time, steps)
    velocity = v_initial + acceleration * t
    position = v_initial * t + 0.5 * acceleration * t ** 2
    return pd.DataFrame({
        "time (s)": t,
        "position (m)": position,
        "velocity (m/s)": velocity,
    })



def simulate_projectile(v_initial: float, angle_deg: float, steps: int = 500, g: float = GRAVITY) -> pd.DataFrame:
    """Simulates a projectile trajectory and returns a DataFrame for the flight."""
    if steps < 2:
        raise ValueError("Steps must be at least 2.")
    total_time = projectile_time_of_flight(v_initial, angle_deg, g)
    if total_time <= 0:
        return pd.DataFrame(columns=["time (s)", "x (m)", "y (m)", "vx (m/s)", "vy (m/s)"])
    t = np.linspace(0, total_time, steps)
    vx0, vy0 = projectile_initial_components(v_initial, angle_deg)
    x = vx0 * t
    y = vy0 * t - 0.5 * g * t ** 2
    vx = np.full_like(t, vx0)
    vy = vy0 - g * t
    mask = y >= 0
    return pd.DataFrame({
        "time (s)": t[mask],
        "x (m)": x[mask],
        "y (m)": y[mask],
        "vx (m/s)": vx[mask],
        "vy (m/s)": vy[mask],
    })



def save_simulation_to_csv(df: pd.DataFrame, filename: str = "kinematics_simulation.csv", directory: Path | None = None) -> Path:
    """Saves a simulation DataFrame to CSV in the specified data folder."""
    directory = Path(directory) if directory is not None else DATA_FOLDER
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / filename
    df.to_csv(file_path, index=False)
    return file_path


# ─────────────────────────────────────────────
# QUICK SUMMARY PRINT UTILITY
# ─────────────────────────────────────────────


def projectile_summary(v_initial: float, angle_deg: float, g: float = GRAVITY) -> None:
    """Prints a summary of projectile motion results."""
    print("=" * 40)
    print("   PROJECTILE MOTION SUMMARY")
    print("=" * 40)
    print(f"  Initial Velocity : {v_initial} m/s")
    print(f"  Launch Angle     : {angle_deg}°")
    print(f"  Gravity          : {g} m/s²")
    print("-" * 40)
    print(f"  Time of Flight   : {projectile_time_of_flight(v_initial, angle_deg, g):.4f} s")
    print(f"  Max Height       : {projectile_max_height(v_initial, angle_deg, g):.4f} m")
    print(f"  Range            : {projectile_range(v_initial, angle_deg, g):.4f} m")
    print("=" * 40)



def motion_summary(v_initial: float, acceleration: float, total_time: float) -> None:
    """Prints a summary of 1D motion results."""
    v_final = velocity_from_acceleration(v_initial, acceleration, total_time)
    disp = displacement_uat(v_initial, total_time, acceleration)
    print("=" * 40)
    print("   1D MOTION SUMMARY")
    print("=" * 40)
    print(f"  Initial Velocity : {v_initial} m/s")
    print(f"  Acceleration     : {acceleration} m/s²")
    print(f"  Time             : {total_time} s")
    print("-" * 40)
    print(f"  Final Velocity   : {v_final:.4f} m/s")
    print(f"  Displacement     : {disp:.4f} m")
    print("=" * 40)
