import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
import pandas as pd

from typing import cast
from sympy import Symbol, Eq, symbols, pi, sqrt
from Core.logic_engine import Logic_Engine
 
def register(engine: Logic_Engine) -> None:
    G, M, m, r, F, g, v, T = symbols('G M m r F g v T')
    engine.register_formula('gravitational_force', cast(Eq,Eq(F, G * M * m / r**2)))
    engine.register_formula('field_strength', cast(Eq,Eq(g, G * M / r**2)))
    engine.register_formula('orbital_velocity', cast(Eq,Eq(v, sqrt(G * M / r))))
    engine.register_formula('orbital_period', cast(Eq,Eq(T, 2 * pi * sqrt(r**3 / (G * M)))))

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

G = 6.674e-11        # Universal Gravitational Constant (N·m²/kg²)
M_EARTH = 5.972e24   # Mass of Earth (kg)
R_EARTH = 6.371e6    # Radius of Earth (m)
M_SUN   = 1.989e30   # Mass of Sun (kg)
R_SUN   = 6.963e8    # Radius of Sun (m)
M_MOON  = 7.342e22   # Mass of Moon (kg)
R_MOON  = 1.737e6    # Radius of Moon (m)
AU      = 1.496e11   # 1 Astronomical Unit (m)


# ─────────────────────────────────────────────
# BASIC — Newton's Law of Gravitation
# ─────────────────────────────────────────────

def gravitational_force(m1, m2, r):
    """
    Newton's Law of Gravitation: F = G * m1 * m2 / r²
    m1, m2: masses in kg
    r: distance between centers in m
    Returns force in Newtons
    """
    if r == 0:
        raise ValueError("Distance r cannot be zero.")
    return G * m1 * m2 / r ** 2


def gravitational_field_strength(mass, r):
    """
    Gravitational field strength at distance r from mass M.
    g = G * M / r²
    Returns g in m/s²
    """
    if r == 0:
        raise ValueError("Distance r cannot be zero.")
    return G * mass / r ** 2


def surface_gravity(mass, radius):
    """
    Surface gravitational acceleration for a body.
    g = G * M / R²
    """
    return gravitational_field_strength(mass, radius)


def weight_on_planet(mass_object, planet_mass, planet_radius):
    """
    Weight of an object on a planet's surface.
    W = m * g_surface
    """
    g = surface_gravity(planet_mass, planet_radius)
    return mass_object * g


def gravitational_potential_energy(m1, m2, r):
    """
    Gravitational Potential Energy: U = -G * m1 * m2 / r
    Negative because it's a bound system.
    """
    if r == 0:
        raise ValueError("Distance r cannot be zero.")
    return -G * m1 * m2 / r


def gravitational_potential(mass, r):
    """
    Gravitational potential at distance r: V = -G * M / r
    Returns V in J/kg
    """
    if r == 0:
        raise ValueError("Distance r cannot be zero.")
    return -G * mass / r


# ─────────────────────────────────────────────
# ESCAPE VELOCITY
# ─────────────────────────────────────────────

def escape_velocity(mass, radius):
    """
    Escape velocity from a body's surface.
    v_esc = sqrt(2 * G * M / R)
    """
    return math.sqrt(2 * G * mass / radius)


def escape_velocity_earth():
    """Escape velocity from Earth's surface (~11.2 km/s)"""
    return escape_velocity(M_EARTH, R_EARTH)


def escape_velocity_moon():
    """Escape velocity from Moon's surface (~2.38 km/s)"""
    return escape_velocity(M_MOON, R_MOON)


def escape_velocity_sun():
    """Escape velocity from Sun's surface (~617.5 km/s)"""
    return escape_velocity(M_SUN, R_SUN)


# ─────────────────────────────────────────────
# ORBITAL MECHANICS
# ─────────────────────────────────────────────

def orbital_velocity(central_mass, orbital_radius):
    """
    Orbital velocity for a circular orbit.
    v_orb = sqrt(G * M / r)
    """
    if orbital_radius == 0:
        raise ValueError("Orbital radius cannot be zero.")
    return math.sqrt(G * central_mass / orbital_radius)


def orbital_period(central_mass, orbital_radius):
    """
    Orbital period for a circular orbit.
    T = 2π * sqrt(r³ / G*M)
    """
    return 2 * math.pi * math.sqrt(orbital_radius ** 3 / (G * central_mass))


def orbital_radius_from_period(central_mass, period):
    """
    Orbital radius given the period.
    r = (G * M * T² / 4π²)^(1/3)
    """
    return ((G * central_mass * period ** 2) / (4 * math.pi ** 2)) ** (1 / 3)


def orbital_kinetic_energy(satellite_mass, central_mass, orbital_radius):
    """
    Kinetic energy of a satellite in circular orbit.
    KE = G * M * m / (2r)
    """
    return G * central_mass * satellite_mass / (2 * orbital_radius)


def orbital_total_energy(satellite_mass, central_mass, orbital_radius):
    """
    Total mechanical energy of orbiting satellite.
    E = -G * M * m / (2r)  (always negative for bound orbit)
    """
    return -G * central_mass * satellite_mass / (2 * orbital_radius)


def orbital_angular_momentum(satellite_mass, central_mass, orbital_radius):
    """
    Angular momentum of circular orbit.
    L = m * v * r
    """
    v = orbital_velocity(central_mass, orbital_radius)
    return satellite_mass * v * orbital_radius


# ─────────────────────────────────────────────
# KEPLER'S LAWS
# ─────────────────────────────────────────────

def kepler_third_law_ratio(T1, r1, T2=None, r2=None):
    """
    Kepler's Third Law: T² / r³ = constant
    - If T2 given, returns r2.
    - If r2 given, returns T2.
    - If neither, returns the ratio T1²/r1³.
    """
    ratio = T1 ** 2 / r1 ** 3
    if T2 is not None:
        return (T2 ** 2 / ratio) ** (1 / 3)   # returns r2
    elif r2 is not None:
        return math.sqrt(ratio * r2 ** 3)      # returns T2
    else:
        return ratio


def semi_major_axis_from_aphelion_perihelion(r_aph, r_per):
    """
    Semi-major axis of elliptical orbit.
    a = (r_aphelion + r_perihelion) / 2
    """
    return (r_aph + r_per) / 2


def orbital_eccentricity(r_aph, r_per):
    """
    Eccentricity of an elliptical orbit.
    e = (r_aph - r_per) / (r_aph + r_per)
    0 = circle, 0<e<1 = ellipse, 1 = parabola, >1 = hyperbola
    """
    return (r_aph - r_per) / (r_aph + r_per)


def velocity_at_aphelion(central_mass, r_aph, r_per):
    """
    Velocity at aphelion using conservation of energy & angular momentum.
    v_aph = sqrt(2GM * r_per / (r_aph * (r_aph + r_per)))
    """
    return math.sqrt(2 * G * central_mass * r_per / (r_aph * (r_aph + r_per)))


def velocity_at_perihelion(central_mass, r_aph, r_per):
    """
    Velocity at perihelion.
    v_per = sqrt(2GM * r_aph / (r_per * (r_aph + r_per)))
    """
    return math.sqrt(2 * G * central_mass * r_aph / (r_per * (r_aph + r_per)))


# ─────────────────────────────────────────────
# SATELLITES — GEO / LEO
# ─────────────────────────────────────────────

def geostationary_orbit_radius(central_mass=M_EARTH, period=86400):
    """
    Radius of geostationary orbit (T = 24 hrs = 86400 s by default).
    Returns radius from Earth's center in meters.
    """
    return orbital_radius_from_period(central_mass, period)


def geostationary_orbit_altitude(central_mass=M_EARTH, planet_radius=R_EARTH):
    """
    Altitude of geostationary orbit above Earth's surface (~35,786 km).
    """
    r_geo = geostationary_orbit_radius(central_mass)
    return r_geo - planet_radius


def satellite_altitude_from_period(period, central_mass=M_EARTH, planet_radius=R_EARTH):
    """
    Altitude above surface for a satellite with a given orbital period.
    """
    r = orbital_radius_from_period(central_mass, period)
    return r - planet_radius


# ─────────────────────────────────────────────
# TIDAL FORCES (Advanced)
# ─────────────────────────────────────────────

def tidal_force(m_primary, m_secondary, r, size):
    """
    Approximate tidal force experienced across an object of given size.
    F_tidal ≈ 2 * G * m_primary * m_secondary * size / r³
    """
    if r == 0:
        raise ValueError("Distance r cannot be zero.")
    return 2 * G * m_primary * m_secondary * size / r ** 3


def roche_limit(primary_radius, density_primary, density_secondary):
    """
    Roche Limit — distance within which tidal forces break up a satellite.
    d = R_primary * (2 * rho_primary / rho_secondary)^(1/3)
    """
    return primary_radius * (2 * density_primary / density_secondary) ** (1 / 3)


# ─────────────────────────────────────────────
# SIMULATION — Circular Orbit (NumPy)
# ─────────────────────────────────────────────

def simulate_circular_orbit(central_mass, orbital_radius, steps=1000):
    """
    Simulates one complete circular orbit using parametric equations.
    Returns DataFrame with x, y, vx, vy at each time step.
    """
    T = orbital_period(central_mass, orbital_radius)
    v = orbital_velocity(central_mass, orbital_radius)
    t = np.linspace(0, T, steps)
    omega = 2 * math.pi / T

    x  =  orbital_radius * np.cos(omega * t)
    y  =  orbital_radius * np.sin(omega * t)
    vx = -v * np.sin(omega * t)
    vy =  v * np.cos(omega * t)

    df = pd.DataFrame({
        "time (s)"  : t,
        "x (m)"     : x,
        "y (m)"     : y,
        "vx (m/s)"  : vx,
        "vy (m/s)"  : vy
    })
    return df


def simulate_gravitational_field(central_mass, r_min, r_max, steps=500):
    """
    Simulates gravitational field strength g vs distance from a body.
    Returns DataFrame with distance and field strength.
    """
    r = np.linspace(r_min, r_max, steps)
    g = G * central_mass / r ** 2

    df = pd.DataFrame({
        "distance (m)"       : r,
        "field strength (m/s²)": g
    })
    return df


def save_simulation_to_csv(df, filename="gravitation_simulation.csv"):
    """Saves simulation DataFrame to CSV in the data/ folder."""
    path = f"data/{filename}"
    df.to_csv(path, index=False)
    print(f"Simulation saved to {path}")


# ─────────────────────────────────────────────
# SUMMARY PRINT UTILITIES
# ─────────────────────────────────────────────

def orbital_summary(central_mass, orbital_radius, satellite_mass=1.0):
    """Prints a full orbital mechanics summary."""
    print("=" * 45)
    print("        ORBITAL MECHANICS SUMMARY")
    print("=" * 45)
    print(f"  Central Body Mass  : {central_mass:.3e} kg")
    print(f"  Orbital Radius     : {orbital_radius:.3e} m")
    print(f"  Satellite Mass     : {satellite_mass:.3e} kg")
    print("-" * 45)
    print(f"  Orbital Velocity   : {orbital_velocity(central_mass, orbital_radius):.4f} m/s")
    print(f"  Orbital Period     : {orbital_period(central_mass, orbital_radius):.4f} s")
    print(f"  Total Orb. Energy  : {orbital_total_energy(satellite_mass, central_mass, orbital_radius):.4e} J")
    print(f"  Escape Velocity    : {escape_velocity(central_mass, orbital_radius):.4f} m/s")
    print("=" * 45)


def body_gravity_summary(name, mass, radius):
    """Prints gravitational summary for any planetary body."""
    print("=" * 45)
    print(f"   GRAVITY SUMMARY — {name.upper()}")
    print("=" * 45)
    print(f"  Mass             : {mass:.3e} kg")
    print(f"  Radius           : {radius:.3e} m")
    print(f"  Surface Gravity  : {surface_gravity(mass, radius):.4f} m/s²")
    print(f"  Escape Velocity  : {escape_velocity(mass, radius) / 1000:.4f} km/s")
    print(f"  Geo. Orbit Alt.  : N/A (use geostationary_orbit_altitude)")
    print("=" * 45)