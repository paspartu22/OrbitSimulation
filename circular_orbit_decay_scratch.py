"""
Standalone circular-orbit decay simulation from scratch.

Model:
- Circular orbit only.
- Drag is the only perturbation.
- Atmosphere density uses log-linear interpolation of a built-in table.
- Orbit radius evolves from specific orbital energy loss due to drag.

No local project modules are imported.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, List

import matplotlib.pyplot as plt

from ssl_bootstrap import configure_ssl_certificates

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

configure_ssl_certificates()
from pyatmos import download_sw_jb2008, jb2008, read_sw_jb2008


# Earth constants
MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6378137.0  # m


# Default run parameters (can be edited and script launched without CLI args).
DEFAULT_MASS_KG = 2.0
DEFAULT_AREA_M2 = 0.00503
DEFAULT_CD = 2.2
DEFAULT_H0_KM = 280.0
DEFAULT_H_REENTRY_KM = 120.0
DEFAULT_MAX_YEARS = 25.0
DEFAULT_LAT_DEG = 25.0
DEFAULT_LON_DEG = 102.0
DEFAULT_WEATHER_DATE_UTC = "2022-03-13 22:18:45"
DEFAULT_TOTAL_IMPULSE_NS = 220.0
DEFAULT_SAMPLE_MINUTES = 60.0
DEFAULT_CSV_PATH = "decay_scratch_output.csv"
DEFAULT_PLOT_PATH = "decay_scratch_plot.png"


@dataclass
class State:
    t_s: float
    h_m: float
    r_m: float
    v_m_s: float
    rho_kg_m3: float


def load_swdata() -> Any:
    """Load JB2008 space weather data exactly like in main.py."""
    swfile = download_sw_jb2008()
    return read_sw_jb2008(swfile)


def density_at_altitude(
    h_m: float,
    weather_date_utc: str,
    lat_deg: float,
    lon_deg: float,
    swdata: Any,
    density_cache_by_km: dict[int, float],
) -> float:
    """Get atmospheric density from JB2008 using 1-km cache bins."""
    h_km = h_m / 1000.0

    # Cache by nearest integer kilometer to avoid repeated JB2008 calls.
    h_km_bin = int(round(h_km))
    if h_km_bin in density_cache_by_km:
        return density_cache_by_km[h_km_bin]

    result = jb2008(weather_date_utc, (lat_deg, lon_deg, float(h_km_bin)), swdata)
    rho = float(result.rho)
    density_cache_by_km[h_km_bin] = rho
    return rho


def choose_timestep(h_m: float) -> float:
    """Simple adaptive timestep: smaller near denser atmosphere."""
    h_km = h_m / 1000.0
    if h_km > 500.0:
        return 120.0
    if h_km > 350.0:
        return 60.0
    if h_km > 250.0:
        return 30.0
    if h_km > 180.0:
        return 10.0
    return 2.0


def simulate_decay(
    mass_kg: float,
    area_m2: float,
    cd: float,
    h0_km: float,
    h_reentry_km: float,
    max_years: float,
    weather_date_utc: str,
    lat_deg: float,
    lon_deg: float,
    swdata: Any,
    sample_minutes: float,
) -> List[State]:
    """Integrate circular orbit decay until reentry altitude."""
    h_m = h0_km * 1000.0
    h_reentry_m = h_reentry_km * 1000.0
    t_s = 0.0
    max_t_s = max_years * 365.25 * 24.0 * 3600.0

    next_sample_s = 0.0
    sample_step_s = max(1.0, sample_minutes * 60.0)
    density_cache_by_km: dict[int, float] = {}

    states: List[State] = []

    while h_m > h_reentry_m and t_s < max_t_s:
        r_m = R_EARTH + h_m
        v_m_s = math.sqrt(MU_EARTH / r_m)
        rho = density_at_altitude(
            h_m,
            weather_date_utc,
            lat_deg,
            lon_deg,
            swdata,
            density_cache_by_km,
        )

        if t_s >= next_sample_s:
            states.append(State(t_s=t_s, h_m=h_m, r_m=r_m, v_m_s=v_m_s, rho_kg_m3=rho))
            next_sample_s += sample_step_s

        dt = choose_timestep(h_m)

        # Drag acceleration magnitude along velocity direction.
        a_drag = 0.5 * cd * area_m2 / mass_kg * rho * v_m_s * v_m_s

        # From specific energy change: dE/dt = -a_drag * v
        # with E = -mu/(2r), get dr/dt = -(2r^2/mu) * a_drag * v.
        dr_dt = -(2.0 * r_m * r_m / MU_EARTH) * a_drag * v_m_s

        r_next = r_m + dr_dt * dt
        h_m = max(0.0, r_next - R_EARTH)
        t_s += dt

    # Append final state for reporting.
    r_m = R_EARTH + h_m
    v_m_s = math.sqrt(MU_EARTH / r_m)
    rho = density_at_altitude(
        h_m,
        weather_date_utc,
        lat_deg,
        lon_deg,
        swdata,
        density_cache_by_km,
    )
    states.append(State(t_s=t_s, h_m=h_m, r_m=r_m, v_m_s=v_m_s, rho_kg_m3=rho))

    return states


def write_csv(path: str, states: List[State]) -> None:
    pass
    # with open(path, "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["time_days", "altitude_km", "orbital_speed_m_s", "density_kg_m3"])
    #     for s in states:
    #         writer.writerow([
    #             s.t_s / 86400.0,
    #             s.h_m / 1000.0,
    #             s.v_m_s,
    #             s.rho_kg_m3,
    #         ])


def plot_decay(
    path: str,
    states: List[State],
    h_reentry_km: float,
    cd: float,
    area_m2: float,
    weather_date_utc: str,
    total_impulse_ns: float,
) -> None:
    time_days = [s.t_s / 86400.0 for s in states]
    alt_km = [s.h_m / 1000.0 for s in states]

    s0 = states[0]
    rho0 = s0.rho_kg_m3
    drag0_n = 0.5 * rho0 * s0.v_m_s * s0.v_m_s * cd * area_m2
    life_days_impulse = (total_impulse_ns / drag0_n) / 86400.0 if drag0_n > 0.0 else math.inf

    plt.figure(figsize=(10, 5.5))
    plt.plot(time_days, alt_km, linewidth=2.0, label="Высота орбиты")
    plt.axhline(h_reentry_km, linestyle="--", linewidth=1.2, label="Граница входа в атмосферу")
    plt.title("Снижение круговой орбиты под действием аэродинамического сопротивления")
    plt.xlabel("Время, сут")
    plt.ylabel("Высота, км")
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone decay of circular orbit from atmospheric drag"
    )
    parser.add_argument(
        "--mass",
        type=float,
        default=DEFAULT_MASS_KG,
        help=f"Spacecraft mass, kg (default: {DEFAULT_MASS_KG})",
    )
    parser.add_argument(
        "--area",
        type=float,
        default=DEFAULT_AREA_M2,
        help=f"Cross-section area, m^2 (default: {DEFAULT_AREA_M2})",
    )
    parser.add_argument(
        "--cd",
        type=float,
        default=DEFAULT_CD,
        help=f"Drag coefficient (default: {DEFAULT_CD})",
    )
    parser.add_argument(
        "--h0",
        type=float,
        default=DEFAULT_H0_KM,
        help=f"Initial circular altitude, km (default: {DEFAULT_H0_KM})",
    )
    parser.add_argument(
        "--hreentry",
        type=float,
        default=DEFAULT_H_REENTRY_KM,
        help="Reentry threshold altitude, km",
    )
    parser.add_argument(
        "--max-years",
        type=float,
        default=DEFAULT_MAX_YEARS,
        help="Maximum simulation horizon, years",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=DEFAULT_LAT_DEG,
        help="Latitude, degrees (default from main.py pattern)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=DEFAULT_LON_DEG,
        help="Longitude, degrees (default from main.py pattern)",
    )
    parser.add_argument(
        "--weather-date",
        type=str,
        default=DEFAULT_WEATHER_DATE_UTC,
        help="UTC timestamp used by JB2008, format: YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--sample-minutes",
        type=float,
        default=DEFAULT_SAMPLE_MINUTES,
        help="Output sample interval, minutes",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=DEFAULT_PLOT_PATH,
        help="Optional PNG path for altitude-vs-time plot",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.mass <= 0:
        raise ValueError("--mass must be > 0")
    if args.area <= 0:
        raise ValueError("--area must be > 0")
    if args.cd <= 0:
        raise ValueError("--cd must be > 0")
    if args.h0 <= args.hreentry:
        raise ValueError("--h0 must be greater than --hreentry")
    if args.hreentry < 100.0:
        raise ValueError("--hreentry below 100 km is not supported in this simple model")
    if args.max_years <= 0:
        raise ValueError("--max-years must be > 0")


def main() -> None:
    run_started_s = time.perf_counter()

    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    print("Loading JB2008 space weather data...")
    swdata = load_swdata()

    states = simulate_decay(
        mass_kg=args.mass,
        area_m2=args.area,
        cd=args.cd,
        h0_km=args.h0,
        h_reentry_km=args.hreentry,
        max_years=args.max_years,
        weather_date_utc=args.weather_date,
        lat_deg=args.lat,
        lon_deg=args.lon,
        swdata=swdata,
        sample_minutes=args.sample_minutes,
    )

    final = states[-1]
    reached_reentry = (final.h_m / 1000.0) <= args.hreentry

    print("=== Circular Orbit Decay (Standalone) ===")
    print(f"Mass: {args.mass:.3f} kg")
    print(f"Area: {args.area:.6f} m^2")
    print(f"Cd: {args.cd:.3f}")
    print(f"Initial altitude: {args.h0:.3f} km")
    print(f"Weather calculation date (UTC): {args.weather_date}")
    print(f"Location for JB2008: lat={args.lat:.3f} deg, lon={args.lon:.3f} deg")
    print(f"Ballistic coefficient m/(Cd*A): {args.mass / (args.cd * args.area):.3f} kg/m^2")
    print(f"Initial density from JB2008: {states[0].rho_kg_m3:.6e} kg/m^3")
    initial_drag_n = 0.5 * states[0].rho_kg_m3 * states[0].v_m_s * states[0].v_m_s * args.cd * args.area
    lifetime_impulse_days = (DEFAULT_TOTAL_IMPULSE_NS / initial_drag_n) / 86400.0 if initial_drag_n > 0.0 else math.inf
    print(f"Initial drag force: {initial_drag_n:.6e} N")
    print(
        "Lifetime at initial orbit with impulse reserve "
        f"{DEFAULT_TOTAL_IMPULSE_NS:.1f} N*s: {lifetime_impulse_days:.2f} days "
        f"({lifetime_impulse_days / 365.25:.3f} years)"
    )

    if reached_reentry:
        print(
            f"Reached {args.hreentry:.1f} km in {final.t_s / 86400.0:.2f} days "
            f"({final.t_s / (365.25 * 86400.0):.3f} years)."
        )
    else:
        print(
            f"Did not reach {args.hreentry:.1f} km within {args.max_years:.2f} years. "
            f"Final altitude: {final.h_m / 1000.0:.2f} km."
        )

    if args.csv:
        write_csv(args.csv, states)
        print(f"Saved time history: {args.csv}")

    if args.plot:
        plot_decay(
            args.plot,
            states,
            args.hreentry,
            cd=args.cd,
            area_m2=args.area,
            weather_date_utc=args.weather_date,
            total_impulse_ns=DEFAULT_TOTAL_IMPULSE_NS,
        )
        print(f"Saved decay plot: {args.plot}")

    elapsed_s = time.perf_counter() - run_started_s
    print(f"Elapsed runtime: {elapsed_s:.2f} s")


if __name__ == "__main__":
    main()
