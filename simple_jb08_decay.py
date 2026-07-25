"""
Простой расчет спуска круговой орбиты по модели JB2008 (JB08).

Скрипт выводит:
- плотность атмосферы на начальной высоте,
- мгновенную скорость снижения dh/dt,
- линейную оценку времени до 110 км,
- интегральную оценку времени до 110 км.
"""

import argparse
import sys
import numpy as np


# Защита от UnicodeEncodeError в не-UTF-8 консолях Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from orbit_decay import OrbitDecayCalculator


def run_simple_decay(
    altitude_km: float,
    mass_kg: float,
    cd: float,
    area_m2: float,
    lat_deg: float,
    lon_deg: float,
    atmosphere_time: str,
    with_integration: bool,
) -> None:
    """Выполнить простой расчет спуска и вывести ключевые результаты."""
    calc = OrbitDecayCalculator(
        mass=mass_kg,
        cd=cd,
        area=area_m2,
        lat=lat_deg,
        lon=lon_deg,
        start_date=atmosphere_time,
        atmosphere_time=atmosphere_time,
    )

    details = calc.decay_rate_detailed(altitude_km)
    linear_days = details["time_to_reentry_days_linear"]

    integrated_days = np.inf
    if with_integration:
        if np.isfinite(linear_days) and linear_days > 0:
            integration_window_days = min(max(365.0, linear_days * 3.0), 3650.0)
        else:
            integration_window_days = 3650.0

        integrated_days = calc.estimate_reentry_time_integrated(
            h_initial=altitude_km,
            max_days=integration_window_days,
            num_steps=300,
        )

    print("=" * 72)
    print("ПРОСТОЙ РАСЧЕТ СПУСКА КРУГОВОЙ ОРБИТЫ (JB2008 / JB08)")
    print("=" * 72)
    print(f"Высота орбиты: {altitude_km:.1f} км")
    print(f"Масса КА: {mass_kg:.3f} кг")
    print(f"Cd: {cd:.3f}")
    print(f"Площадь: {area_m2:.6f} м^2")
    print(f"Координаты: lat={lat_deg:.2f} deg, lon={lon_deg:.2f} deg")
    print(f"Время атмосферы: {atmosphere_time}")
    print()
    print(f"Плотность: {details['density_kg_m3']:.3e} кг/м^3")
    print(f"Орбитальная скорость: {details['velocity_km_s']:.4f} км/с")
    print(f"Скорость спуска: {details['decay_rate_km_day']:.6f} км/сут")
    print(f"Линейная оценка до 110 км: {linear_days:.2f} сут ({linear_days / 365.25:.3f} лет)")

    if with_integration and np.isfinite(integrated_days):
        print(
            "Интегральная оценка до 110 км: "
            f"{integrated_days:.2f} сут ({integrated_days / 365.25:.3f} лет)"
        )
    elif with_integration:
        print("Интегральная оценка до 110 км: не достигнуто в окне расчета")
    else:
        print("Интегральная оценка: пропущена (добавьте --with-integration)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Простой расчет спуска круговой орбиты по модели JB2008 (JB08)."
    )
    parser.add_argument("--altitude", type=float, default=280.0, help="Начальная высота, км")
    parser.add_argument("--mass", type=float, default=2.0, help="Масса КА, кг")
    parser.add_argument("--cd", type=float, default=2.2, help="Коэффициент сопротивления")
    parser.add_argument("--area", type=float, default=0.00503, help="Площадь, м^2")
    parser.add_argument("--lat", type=float, default=0.0, help="Широта, град")
    parser.add_argument("--lon", type=float, default=0.0, help="Долгота, град")
    parser.add_argument(
        "--time",
        type=str,
        default="2020-01-01 00:00:00",
        help="Время атмосферы в формате YYYY-MM-DD HH:MM:SS",
    )
    parser.add_argument(
        "--with-integration",
        action="store_true",
        help="Добавить интегральную оценку времени спуска (дольше выполняется)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simple_decay(
        altitude_km=args.altitude,
        mass_kg=args.mass,
        cd=args.cd,
        area_m2=args.area,
        lat_deg=args.lat,
        lon_deg=args.lon,
        atmosphere_time=args.time,
        with_integration=args.with_integration,
    )