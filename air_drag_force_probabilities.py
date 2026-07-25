"""
Compute aerodynamic drag force for three atmospheric density scenarios.

Uses the same drag model as circular_orbit_decay_scratch.py:
    F_drag = 0.5 * rho * v^2 * Cd * A

Orbital speed is computed for a circular orbit at the default altitude 280 km.
"""

from __future__ import annotations

import math

# Earth constants (same as in project scripts)
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6378137.0  # m

# Requested spacecraft parameters
DEFAULT_MASS_KG = 2.0
DEFAULT_AREA_M2 = 0.00503
DEFAULT_CD = 2.2

# Circular-orbit altitude used to compute orbital speed
DEFAULT_H0_KM = 280.0

# Density bounds provided by user
DENSITY_SCENARIOS = {
    "P5": 1.427e-11,
    "P50": 2.292e-11,
    "P95": 3.494e-11,
}


def circular_orbit_speed_m_s(h_km: float) -> float:
    """Return circular orbital speed at altitude h_km."""
    r_m = R_EARTH + h_km * 1000.0
    return math.sqrt(MU_EARTH / r_m)


def drag_force_n(rho_kg_m3: float, v_m_s: float, cd: float, area_m2: float) -> float:
    """Return drag force in newtons."""
    return 0.5 * rho_kg_m3 * v_m_s * v_m_s * cd * area_m2


def main() -> None:
    v0_m_s = circular_orbit_speed_m_s(DEFAULT_H0_KM)

    print("=== Drag Force for Density Scenarios ===")
    print(f"Mass: {DEFAULT_MASS_KG:.3f} kg")
    print(f"Area: {DEFAULT_AREA_M2:.6f} m^2")
    print(f"Cd: {DEFAULT_CD:.3f}")
    print(f"Reference altitude: {DEFAULT_H0_KM:.1f} km")
    print(f"Reference orbital speed: {v0_m_s:.3f} m/s")
    print()
    print("Scenario | Density (kg/m^3) | Drag force (N) | Drag accel (m/s^2)")
    print("---------|------------------|----------------|-------------------")

    for label, rho in DENSITY_SCENARIOS.items():
        force_n = drag_force_n(rho, v0_m_s, DEFAULT_CD, DEFAULT_AREA_M2)
        accel_m_s2 = force_n / DEFAULT_MASS_KG
        print(f"{label:>8} | {rho:>16.3e} | {force_n:>14.6e} | {accel_m_s2:>17.6e}")


if __name__ == "__main__":
    main()
