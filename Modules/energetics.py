import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
import pandas as pd
from typing import cast
from sympy import Eq
from sympy import Rational
from sympy import Symbol, Eq, symbols, sqrt
from Core.logic_engine import Logic_Engine

def register(engine: Logic_Engine) -> None:
    m, v, h, g, k, x, F, d, P, t, W = symbols('m v h g k x F d P t W')
    engine.register_formula('kinetic_energy', cast(Eq, Eq(W, m * v**2 / 2)))
    engine.register_formula('gravitational_pe', cast(Eq, Eq(W, m * g * h)))
    engine.register_formula('elastic_pe', cast(Eq, Eq(W, k * x**2 / 2)))
    engine.register_formula('work_done', cast(Eq, Eq(W, F * d)))
    engine.register_formula('power', cast(Eq,Eq(P, W / t)))
# ─────────────────────────────────────────────
# BASIC — Work & Energy
# ─────────────────────────────────────────────

def work_done(force, displacement, angle_deg=0):
    """
    Work done by a force.
    W = F * d * cos(θ)
    angle_deg: angle between force and displacement (default 0 = parallel)
    """
    return force * displacement * math.cos(math.radians(angle_deg))


def kinetic_energy(mass, velocity):
    """
    Kinetic Energy: KE = (1/2) * m * v²
    """
    return 0.5 * mass * velocity ** 2


def gravitational_pe(mass, height, g=9.81):
    """
    Gravitational Potential Energy: PE = m * g * h
    """
    return mass * g * height


def elastic_pe(spring_constant, extension):
    """
    Elastic Potential Energy: PE = (1/2) * k * x²
    """
    return 0.5 * spring_constant * extension ** 2


def work_energy_theorem(mass, v_initial, v_final):
    """
    Work-Energy Theorem: W_net = ΔKE = (1/2)m(v²_f - v²_i)
    """
    return 0.5 * mass * (v_final ** 2 - v_initial ** 2)


# ─────────────────────────────────────────────
# POWER
# ─────────────────────────────────────────────

def power_from_work(work, time):
    """
    Power: P = W / t
    """
    if time == 0:
        raise ValueError("Time cannot be zero.")
    return work / time


def power_from_force(force, velocity, angle_deg=0):
    """
    Instantaneous Power: P = F * v * cos(θ)
    """
    return force * velocity * math.cos(math.radians(angle_deg))


def power_from_energy_change(delta_energy, time):
    """
    P = ΔE / t — general power from any energy change
    """
    if time == 0:
        raise ValueError("Time cannot be zero.")
    return delta_energy / time


def time_from_power(work, power):
    """
    Time = Work / Power
    """
    if power == 0:
        raise ValueError("Power cannot be zero.")
    return work / power


def force_from_power(power, velocity, angle_deg=0):
    """
    F = P / (v * cos(θ))
    """
    denom = velocity * math.cos(math.radians(angle_deg))
    if denom == 0:
        raise ValueError("Velocity or angle makes denominator zero.")
    return power / denom


# ─────────────────────────────────────────────
# EFFICIENCY & ENERGY CONVERSION
# ─────────────────────────────────────────────

def efficiency(useful_output, total_input):
    """
    Efficiency = (Useful Output / Total Input) × 100%
    Returns percentage.
    """
    if total_input == 0:
        raise ValueError("Total input energy cannot be zero.")
    return (useful_output / total_input) * 100


def useful_output_energy(efficiency_percent, total_input):
    """
    Useful Output = (efficiency / 100) × Total Input
    """
    return (efficiency_percent / 100) * total_input


def wasted_energy(total_input, useful_output):
    """
    Wasted Energy = Total Input - Useful Output
    """
    return total_input - useful_output


def energy_conversion_chain(input_energy, *efficiencies):
    """
    Computes output energy through a chain of energy conversions.
    Each efficiency is a percentage (0–100).
    e.g., energy_conversion_chain(1000, 80, 70, 90)
    """
    energy = input_energy
    for eff in efficiencies:
        energy = energy * (eff / 100)
    return energy


# ─────────────────────────────────────────────
# CONSERVATION OF ENERGY
# ─────────────────────────────────────────────

def total_mechanical_energy(mass, velocity, height, g=9.81):
    """
    Total Mechanical Energy: E = KE + PE = (1/2)mv² + mgh
    """
    return kinetic_energy(mass, velocity) + gravitational_pe(mass, height, g)


def velocity_from_height_drop(height, g=9.81, v_initial=0):
    """
    Velocity after falling height h (energy conservation).
    v = sqrt(v_i² + 2gh)
    """
    return math.sqrt(v_initial ** 2 + 2 * g * height)


def max_height_from_velocity(velocity, g=9.81):
    """
    Maximum height reached given initial velocity (KE → PE).
    h = v² / (2g)
    """
    return velocity ** 2 / (2 * g)


def spring_launch_velocity(spring_constant, extension, mass):
    """
    Velocity of object launched by spring (PE_spring → KE).
    (1/2)kx² = (1/2)mv²  →  v = x * sqrt(k/m)
    """
    return extension * math.sqrt(spring_constant / mass)


def pendulum_max_velocity(length, angle_deg, g=9.81):
    """
    Maximum velocity of pendulum at the bottom.
    Uses energy conservation: v = sqrt(2gL(1 - cos(θ)))
    """
    angle_rad = math.radians(angle_deg)
    return math.sqrt(2 * g * length * (1 - math.cos(angle_rad)))


def pendulum_max_height(length, angle_deg):
    """
    Maximum height of pendulum bob.
    h = L(1 - cos(θ))
    """
    return length * (1 - math.cos(math.radians(angle_deg)))


# ─────────────────────────────────────────────
# COLLISIONS
# ─────────────────────────────────────────────

def momentum(mass, velocity):
    """Linear momentum: p = mv"""
    return mass * velocity


def elastic_collision_velocities(m1, u1, m2, u2):
    """
    1D Elastic Collision — both momentum and KE conserved.
    Returns (v1_final, v2_final)
    """
    v1 = ((m1 - m2) * u1 + 2 * m2 * u2) / (m1 + m2)
    v2 = ((m2 - m1) * u2 + 2 * m1 * u1) / (m1 + m2)
    return v1, v2


def inelastic_collision_velocity(m1, u1, m2, u2):
    """
    Perfectly Inelastic Collision — objects stick together.
    v = (m1*u1 + m2*u2) / (m1 + m2)
    """
    return (m1 * u1 + m2 * u2) / (m1 + m2)


def kinetic_energy_lost_inelastic(m1, u1, m2, u2):
    """
    KE lost in a perfectly inelastic collision.
    ΔKE = KE_before - KE_after
    """
    v = inelastic_collision_velocity(m1, u1, m2, u2)
    ke_before = kinetic_energy(m1, u1) + kinetic_energy(m2, u2)
    ke_after  = kinetic_energy(m1 + m2, v)
    return ke_before - ke_after


def coefficient_of_restitution(v1_after, v2_after, v1_before, v2_before):
    """
    e = (v2_after - v1_after) / (v1_before - v2_before)
    e = 1 → elastic, e = 0 → perfectly inelastic
    """
    denom = v1_before - v2_before
    if denom == 0:
        raise ValueError("Initial velocities cannot be equal.")
    return (v2_after - v1_after) / denom


def impulse(force, time):
    """Impulse: J = F × t"""
    return force * time


def impulse_from_momentum_change(mass, v_initial, v_final):
    """J = m(v_f - v_i) = Δp"""
    return mass * (v_final - v_initial)


# ─────────────────────────────────────────────
# SIMPLE HARMONIC MOTION (SHM) ENERGY
# ─────────────────────────────────────────────

def shm_total_energy(spring_constant, amplitude):
    """
    Total energy in SHM: E = (1/2) * k * A²
    """
    return 0.5 * spring_constant * amplitude ** 2


def shm_kinetic_energy(spring_constant, amplitude, displacement):
    """
    KE at displacement x in SHM.
    KE = (1/2) * k * (A² - x²)
    """
    if abs(displacement) > amplitude:
        raise ValueError("Displacement cannot exceed amplitude.")
    return 0.5 * spring_constant * (amplitude ** 2 - displacement ** 2)


def shm_potential_energy(spring_constant, displacement):
    """
    PE at displacement x in SHM.
    PE = (1/2) * k * x²
    """
    return 0.5 * spring_constant * displacement ** 2


def shm_velocity_at_displacement(angular_freq, amplitude, displacement):
    """
    Velocity in SHM at displacement x.
    v = ω * sqrt(A² - x²)
    """
    if abs(displacement) > amplitude:
        raise ValueError("Displacement cannot exceed amplitude.")
    return angular_freq * math.sqrt(amplitude ** 2 - displacement ** 2)


def shm_max_velocity(angular_freq, amplitude):
    """
    Maximum velocity in SHM (at equilibrium, x=0).
    v_max = ω * A
    """
    return angular_freq * amplitude


def shm_max_acceleration(angular_freq, amplitude):
    """
    Maximum acceleration in SHM (at extremes, x=A).
    a_max = ω² * A
    """
    return angular_freq ** 2 * amplitude


# ─────────────────────────────────────────────
# THERMAL / HEAT ENERGY (Basic)
# ─────────────────────────────────────────────

def heat_energy(mass, specific_heat_capacity, delta_temp):
    """
    Heat energy: Q = mcΔT
    """
    return mass * specific_heat_capacity * delta_temp


def latent_heat_energy(mass, specific_latent_heat):
    """
    Energy during phase change: Q = mL
    """
    return mass * specific_latent_heat


def final_temperature(initial_temp, heat_added, mass, specific_heat):
    """
    T_final = T_initial + Q / (mc)
    """
    return initial_temp + heat_added / (mass * specific_heat)


def heat_transfer_rate(thermal_conductivity, area, delta_temp, thickness):
    """
    Heat conduction rate (Fourier's Law): Q/t = k * A * ΔT / d
    Returns power in Watts.
    """
    if thickness == 0:
        raise ValueError("Thickness cannot be zero.")
    return thermal_conductivity * area * delta_temp / thickness


# ─────────────────────────────────────────────
# SIMULATION — Energy over Time (NumPy)
# ─────────────────────────────────────────────

def simulate_shm_energy(spring_constant, amplitude, mass, steps=1000):
    """
    Simulates KE, PE, and total energy over one full SHM cycle.
    Returns a Pandas DataFrame.
    """
    omega = math.sqrt(spring_constant / mass)
    T = 2 * math.pi / omega
    t = np.linspace(0, T, steps)

    x  = amplitude * np.cos(omega * t)           # displacement
    v  = -amplitude * omega * np.sin(omega * t)  # velocity

    ke    = 0.5 * mass * v ** 2
    pe    = 0.5 * spring_constant * x ** 2
    total = ke + pe

    df = pd.DataFrame({
        "time (s)"         : t,
        "displacement (m)" : x,
        "velocity (m/s)"   : v,
        "KE (J)"           : ke,
        "PE (J)"           : pe,
        "Total Energy (J)" : total
    })
    return df


def simulate_projectile_energy(mass, v_initial, angle_deg, steps=500, g=9.81):
    """
    Simulates KE, PE, and total mechanical energy during projectile flight.
    Returns a Pandas DataFrame.
    """
    angle_rad = math.radians(angle_deg)
    T = 2 * v_initial * math.sin(angle_rad) / g
    t = np.linspace(0, T, steps)

    vx = v_initial * math.cos(angle_rad)
    vy = v_initial * math.sin(angle_rad) - g * t
    v  = np.sqrt(vx ** 2 + vy ** 2)
    y  = v_initial * math.sin(angle_rad) * t - 0.5 * g * t ** 2
    y  = np.maximum(y, 0)

    ke    = 0.5 * mass * v ** 2
    pe    = mass * g * y
    total = ke + pe

    df = pd.DataFrame({
        "time (s)"         : t,
        "height (m)"       : y,
        "speed (m/s)"      : v,
        "KE (J)"           : ke,
        "PE (J)"           : pe,
        "Total Energy (J)" : total
    })
    return df


def save_simulation_to_csv(df, filename="energetics_simulation.csv"):
    """Saves simulation DataFrame to CSV in data/ folder."""
    path = f"data/{filename}"
    df.to_csv(path, index=False)
    print(f"Simulation saved to {path}")


# ─────────────────────────────────────────────
# SUMMARY PRINT UTILITIES
# ─────────────────────────────────────────────

def collision_summary(m1, u1, m2, u2):
    """Prints elastic and inelastic collision comparison."""
    v1e, v2e = elastic_collision_velocities(m1, u1, m2, u2)
    vi = inelastic_collision_velocity(m1, u1, m2, u2)
    ke_lost = kinetic_energy_lost_inelastic(m1, u1, m2, u2)

    print("=" * 48)
    print("           COLLISION SUMMARY")
    print("=" * 48)
    print(f"  Object 1: mass={m1} kg, velocity={u1} m/s")
    print(f"  Object 2: mass={m2} kg, velocity={u2} m/s")
    print("-" * 48)
    print("  ELASTIC COLLISION:")
    print(f"    v1_final = {v1e:.4f} m/s")
    print(f"    v2_final = {v2e:.4f} m/s")
    print("-" * 48)
    print("  PERFECTLY INELASTIC COLLISION:")
    print(f"    v_combined = {vi:.4f} m/s")
    print(f"    KE Lost    = {ke_lost:.4f} J")
    print("=" * 48)


def energy_summary(mass, velocity, height, g=9.81):
    """Prints KE, PE, and Total Mechanical Energy."""
    ke = kinetic_energy(mass, velocity)
    pe = gravitational_pe(mass, height, g)
    te = ke + pe
    print("=" * 40)
    print("      MECHANICAL ENERGY SUMMARY")
    print("=" * 40)
    print(f"  Mass     : {mass} kg")
    print(f"  Velocity : {velocity} m/s")
    print(f"  Height   : {height} m")
    print("-" * 40)
    print(f"  KE       : {ke:.4f} J")
    print(f"  PE       : {pe:.4f} J")
    print(f"  Total    : {te:.4f} J")
    print("=" * 40)