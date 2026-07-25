"""
Разлёт группировки: старт 300×300 → финиш 280×280.

Схема:
  Все КА стартуют на круговой 300 км.
  Половина (вперёд по фазе):
    сразу уходит на переходную эллиптическую 300×280,
    затем поочерёдно круговизируется на финальную 280×280.
  Половина (назад по фазе):
    остаётся на 300×300,
    затем по одному: 300×300 → 300×280 → 280×280.

Готовность КА = момент, когда одновременно:
  1) набрана целевая фаза,
  2) орбита уже круговая 280×280 (конец 2-го импульса).

Импульсы каждого КА:
  1) в апогее: круговая 300 → эллипс 300×280 (опускание перигея);
  2) в перигее: эллипс → круговая 280 (опускание апогея).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


R_EARTH_KM = 6371.0
MU_KM3_S2 = 3.986004418e5
G0 = 9.80665

N_SATS = 10
H_START_KM = 300.0
H_FINAL_KM = 280.0
H_APOGEE_KM = 300.0
H_PERIGEE_KM = 280.0

MASS_KG = 2.0
THRUST_N = 6e-3
TOTAL_IMPULSE_NS = 220.0
PROPELLANT_MASS_KG = 0.13

PLOT_PATH = Path(__file__).with_name("constellation_phasing.png")
PLOT_ALT_PATH = Path(__file__).with_name("constellation_phasing_altitudes.png")


@dataclass(frozen=True)
class CircularOrbit:
    h_km: float

    @property
    def a_km(self) -> float:
        return R_EARTH_KM + self.h_km

    @property
    def n_rad_s(self) -> float:
        return math.sqrt(MU_KM3_S2 / self.a_km**3)

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.n_rad_s

    @property
    def v_m_s(self) -> float:
        return math.sqrt(MU_KM3_S2 / self.a_km) * 1000.0


@dataclass(frozen=True)
class EllipticalOrbit:
    hp_km: float
    ha_km: float

    @property
    def a_km(self) -> float:
        return R_EARTH_KM + 0.5 * (self.hp_km + self.ha_km)

    @property
    def n_rad_s(self) -> float:
        return math.sqrt(MU_KM3_S2 / self.a_km**3)

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.n_rad_s

    def v_perigee_m_s(self) -> float:
        rp = R_EARTH_KM + self.hp_km
        return math.sqrt(MU_KM3_S2 * (2.0 / rp - 1.0 / self.a_km)) * 1000.0

    def v_apogee_m_s(self) -> float:
        ra = R_EARTH_KM + self.ha_km
        return math.sqrt(MU_KM3_S2 * (2.0 / ra - 1.0 / self.a_km)) * 1000.0


def burn_time_s(dv_m_s: float, mass_kg: float = MASS_KG, thrust_n: float = THRUST_N) -> float:
    return (mass_kg * dv_m_s) / thrust_n if thrust_n > 0 else math.inf


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "inf"
    days = int(seconds // 86400)
    rem = seconds - days * 86400
    hours = int(rem // 3600)
    rem -= hours * 3600
    minutes = int(rem // 60)
    secs = rem - minutes * 60
    return f"{days} сут {hours:02d} ч {minutes:02d} мин {secs:04.1f} с"


def compute_phasing() -> dict:
    n_fwd = N_SATS // 2
    n_bwd = N_SATS - n_fwd
    phase_step = 2.0 * math.pi / N_SATS

    circ300 = CircularOrbit(H_START_KM)
    circ280 = CircularOrbit(H_FINAL_KM)
    ellip = EllipticalOrbit(H_PERIGEE_KM, H_APOGEE_KM)

    dv1 = circ300.v_m_s - ellip.v_apogee_m_s()
    dv2 = ellip.v_perigee_m_s() - circ280.v_m_s
    dv_total = dv1 + dv2

    t_burn1 = burn_time_s(dv1)
    t_burn2 = burn_time_s(dv2)
    t_to_first_perigee = 0.5 * ellip.period_s

    n300 = circ300.n_rad_s
    nell = ellip.n_rad_s
    n280 = circ280.n_rad_s
    dn_280_ell = n280 - nell
    dn_300_280 = n300 - n280

    # Готовность = конец 2-го импульса (круговая 280×280), не «фаза набрана на эллипсе».
    dt_fwd = phase_step / dn_280_ell
    t_f_ell = t_burn1
    t_f_circ_start = [
        t_f_ell + t_to_first_perigee + i * dt_fwd
        for i in range(n_fwd)
    ]
    t_f_ready = [t + t_burn2 for t in t_f_circ_start]
    phi_f = [-i * phase_step for i in range(n_fwd)]

    def phase_of(a_times: dict, t: float) -> float:
        """
        n_ell на всём эллипсе, включая 2-й импульс.
        n_280 — только после фактического выхода на круговую 280×280.
        """
        t_ell = a_times["t_ell_s"]
        t_ready = a_times["t_ready_s"]
        if t <= t_ell:
            return n300 * t
        if t <= t_ready:
            return n300 * t_ell + nell * (t - t_ell)
        return n300 * t_ell + nell * (t_ready - t_ell) + n280 * (t - t_ready)

    f0_times = {"t_ell_s": t_f_ell, "t_ready_s": t_f_ready[0]}

    def relative_phase_b_vs_f0(t_leave_300: float) -> float:
        t_ready_b = t_leave_300 + t_to_first_perigee + t_burn2
        b_times = {"t_ell_s": t_leave_300, "t_ready_s": t_ready_b}
        return phase_of(b_times, t_ready_b) - phase_of(f0_times, t_ready_b)

    assignments: list[dict] = []

    for i in range(n_fwd):
        assignments.append(
            {
                "id": i,
                "group": "forward",
                "label": f"F{i}",
                "phi_target_deg": math.degrees(phi_f[i]),
                "t_ell_s": t_f_ell,
                "t_circ_start_s": t_f_circ_start[i],
                "t_ready_s": t_f_ready[i],
                "t_done_s": t_f_ready[i],
                "dv_m_s": dv_total,
            }
        )

    for j in range(n_bwd):
        target = -(n_fwd + j) * phase_step
        lo = t_f_ready[0]
        hi = lo + abs(target) / abs(dn_300_280) + 5 * ellip.period_s
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            err = relative_phase_b_vs_f0(mid) - target
            if err > 0:
                lo = mid
            else:
                hi = mid
        t_leave = 0.5 * (lo + hi)
        t_ready = t_leave + t_to_first_perigee + t_burn2
        assignments.append(
            {
                "id": n_fwd + j,
                "group": "backward",
                "label": f"B{j}",
                "phi_target_deg": math.degrees(target),
                "t_ell_s": t_leave,
                "t_circ_start_s": t_leave + t_to_first_perigee,
                "t_ready_s": t_ready,
                "t_done_s": t_ready,
                "dv_m_s": dv_total,
            }
        )

    t_mission = max(a["t_ready_s"] for a in assignments)

    phi_f0 = phase_of(assignments[0], t_mission)
    for a in assignments:
        rel = phase_of(a, t_mission) - phi_f0
        tgt = math.radians(a["phi_target_deg"])
        rel_aligned = tgt + 2.0 * math.pi * round((rel - tgt) / (2.0 * math.pi))
        a["phi_final_deg"] = math.degrees(rel_aligned)

    impulse = MASS_KG * dv_total
    propellant = PROPELLANT_MASS_KG * (impulse / TOTAL_IMPULSE_NS)

    for a in assignments:
        a["propellant_kg"] = propellant
        a["impulse_ns"] = impulse
        a["propellant_pct"] = 100.0 * propellant / PROPELLANT_MASS_KG

    return {
        "circ300": circ300,
        "circ280": circ280,
        "ellip": ellip,
        "n_fwd": n_fwd,
        "n_bwd": n_bwd,
        "phase_step_deg": math.degrees(phase_step),
        "dv1": dv1,
        "dv2": dv2,
        "dv_total": dv_total,
        "t_burn1": t_burn1,
        "t_burn2": t_burn2,
        "t_to_first_perigee": t_to_first_perigee,
        "dt_fwd": dt_fwd,
        "n300": n300,
        "nell": nell,
        "n280": n280,
        "assignments": assignments,
        "t_mission": t_mission,
        "impulse": impulse,
        "propellant": propellant,
        "propellant_pct": 100.0 * propellant / PROPELLANT_MASS_KG,
        "dv_budget": TOTAL_IMPULSE_NS / MASS_KG,
    }


def build_histories(result: dict, n_points: int = 1200) -> dict:
    t_mission = result["t_mission"]
    t = np.linspace(0.0, t_mission, n_points)
    t_b1 = result["t_burn1"]
    t_b2 = result["t_burn2"]
    n300 = result["n300"]
    nell = result["nell"]
    n280 = result["n280"]

    f0 = next(a for a in result["assignments"] if a["id"] == 0)

    def orbit_phase(a: dict, ti: float) -> float:
        t_ell = a["t_ell_s"]
        t_ready = a["t_ready_s"]
        if ti <= t_ell:
            return n300 * ti
        if ti <= t_ready:
            return n300 * t_ell + nell * (ti - t_ell)
        return n300 * t_ell + nell * (t_ready - t_ell) + n280 * (ti - t_ready)

    histories = {}
    for a in result["assignments"]:
        hp = np.full_like(t, H_START_KM)
        ha = np.full_like(t, H_START_KM)
        rel_raw = np.zeros_like(t)
        t_ell = a["t_ell_s"]
        t_cs = a["t_circ_start_s"]
        t_ready = a["t_ready_s"]

        for i, ti in enumerate(t):
            if ti < t_ell - t_b1:
                hp[i] = H_START_KM
                ha[i] = H_START_KM
            elif ti < t_ell:
                frac = (ti - (t_ell - t_b1)) / t_b1 if t_b1 > 0 else 1.0
                hp[i] = H_START_KM + frac * (H_PERIGEE_KM - H_START_KM)
                ha[i] = H_APOGEE_KM
            elif ti < t_cs:
                hp[i] = H_PERIGEE_KM
                ha[i] = H_APOGEE_KM
            elif ti < t_ready:
                frac = (ti - t_cs) / t_b2 if t_b2 > 0 else 1.0
                hp[i] = H_PERIGEE_KM
                ha[i] = H_APOGEE_KM + frac * (H_FINAL_KM - H_APOGEE_KM)
            else:
                hp[i] = H_FINAL_KM
                ha[i] = H_FINAL_KM

            rel_raw[i] = orbit_phase(a, ti) - orbit_phase(f0, ti)

        rel_unwrapped = np.unwrap(rel_raw)
        # Горизонталь фазы только после готовности = 280×280
        ready_mask = t >= t_ready
        if np.any(ready_mask):
            target = math.radians(a["phi_target_deg"])
            first_ready = int(np.argmax(ready_mask))
            last = float(rel_unwrapped[first_ready])
            target_cont = target + 2.0 * math.pi * round((last - target) / (2.0 * math.pi))
            rel_unwrapped[ready_mask] = target_cont

        histories[a["id"]] = {
            "t_s": t,
            "hp_km": hp,
            "ha_km": ha,
            "phi_deg": np.degrees(rel_unwrapped),
        }
    return histories


def _color(sat_id: int):
    return plt.cm.tab10(sat_id % 10)


def plot_phasing(result: dict, path: Path = PLOT_PATH) -> Path:
    assignments = sorted(result["assignments"], key=lambda a: a["id"])
    histories = build_histories(result)
    t_days = result["t_mission"] / 86400.0

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Разлёт группировки: 300×300 → 300×280 → 280×280",
        fontsize=14,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(2, 2, 1, projection="polar")
    th = np.linspace(0, 2 * np.pi, 360)
    ax1.plot(th, np.ones_like(th), color="0.75", lw=1)
    ax1.scatter([0], [1], s=180, c="0.4", zorder=3, label="старт (все)")
    for a in assignments:
        ang = math.radians(a["phi_final_deg"]) % (2 * math.pi)
        marker = "o" if a["group"] == "forward" else "s"
        ax1.scatter(
            [ang], [1.0], s=70, c=[_color(a["id"])],
            marker=marker, edgecolors="k", linewidths=0.4, zorder=4,
        )
        ax1.text(ang, 1.2, a["label"], ha="center", va="center", fontsize=8)
    ax1.set_title("Финиш на 280×280 (○ вперёд, □ назад)", pad=16)
    ax1.set_yticklabels([])
    ax1.set_ylim(0, 1.4)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=8)

    ax2 = fig.add_subplot(2, 2, 2)
    for a in assignments:
        h = histories[a["id"]]
        ls = "-" if a["group"] == "forward" else "--"
        ax2.plot(h["t_s"] / 86400, h["hp_km"], color=_color(a["id"]), ls=ls, lw=1.4,
                 label=f"{a['label']} hp")
        ax2.plot(h["t_s"] / 86400, h["ha_km"], color=_color(a["id"]), ls=ls, lw=1.0, alpha=0.55)
    ax2.axhline(H_START_KM, color="0.4", ls=":", lw=1)
    ax2.axhline(H_FINAL_KM, color="0.4", ls=":", lw=1)
    ax2.set_xlabel("Время, сут")
    ax2.set_ylabel("Высота, км")
    ax2.set_title("Перигей (ярк.) и апогей (бл.): уход → эллипс → 280")
    ax2.set_ylim(H_FINAL_KM - 5, H_START_KM + 5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=5, fontsize=6, loc="lower right")

    ax3 = fig.add_subplot(2, 2, 3)
    for a in assignments:
        h = histories[a["id"]]
        ls = "-" if a["group"] == "forward" else "--"
        ax3.plot(
            h["t_s"] / 86400, h["phi_deg"],
            color=_color(a["id"]), ls=ls, lw=1.8,
            label=f"{a['label']} → {a['phi_target_deg']:+.0f}°",
        )
        ax3.axhline(a["phi_target_deg"], color=_color(a["id"]), ls=":", alpha=0.25, lw=0.8)
        # Метка готовности = выход на 280×280
        ax3.axvline(a["t_ready_s"] / 86400, color=_color(a["id"]), ls=":", alpha=0.2, lw=0.7)
    ax3.axvline(t_days, color="k", ls="--", lw=1, label="все на 280×280")
    ax3.set_xlabel("Время, сут")
    ax3.set_ylabel("Фаза отн. F0, °")
    ax3.set_title("Накопление фазы (горизонт = готов: фаза + 280×280)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7, ncol=2, loc="best")

    ax4 = fig.add_subplot(2, 2, 4)
    ids = [a["id"] for a in assignments]
    ready_days = [a["t_ready_s"] / 86400 for a in assignments]
    prop_g = [a["propellant_kg"] * 1000 for a in assignments]
    prop_pct = [a["propellant_pct"] for a in assignments]
    colors = [_color(i) for i in ids]
    x = np.arange(len(ids))
    w = 0.38

    bars1 = ax4.bar(x - w / 2, ready_days, w, color=colors, edgecolor="k", lw=0.4)
    ax4.set_ylabel("Готовность (фаза + 280×280), сут")
    ax4.set_xticks(x)
    ax4.set_xticklabels([a["label"] for a in assignments])
    ax4.set_title("Готовность = целевая фаза и круговая 280×280")
    ax4.grid(True, axis="y", alpha=0.3)

    ax4b = ax4.twinx()
    bars2 = ax4b.bar(
        x + w / 2, prop_g, w, color=colors, alpha=0.45,
        edgecolor="k", lw=0.4, hatch="//",
    )
    ymax = max(prop_g) * 1.5
    ax4b.set_ylim(0, ymax)
    ax4b.set_ylabel("Расход РТ, г (% от запаса)")
    for xi, g, p in zip(x, prop_g, prop_pct):
        ax4b.text(xi + w / 2, g + ymax * 0.02, f"{g:.1f} г\n({p:.1f}%)",
                  ha="center", va="bottom", fontsize=7)

    ax4.legend([bars1, bars2], ["готовность 280×280", "РТ, г (%)"], loc="upper left", fontsize=8)

    summary = (
        f"Миссия: {format_duration(result['t_mission'])}  |  "
        f"Δv = {result['dv1']:.2f} + {result['dv2']:.2f} = {result['dv_total']:.2f} м/с  |  "
        f"РТ: {result['propellant']*1e3:.2f} г ({result['propellant_pct']:.1f}%) на КА"
    )
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_altitudes_per_sat(result: dict, path: Path = PLOT_ALT_PATH) -> Path:
    """Отдельный график апогей/перигей для каждого КА."""
    assignments = sorted(result["assignments"], key=lambda a: a["id"])
    histories = build_histories(result)
    n = len(assignments)
    ncols = 5
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 3.4 * nrows), sharex=True, sharey=True
    )
    axes_flat = np.atleast_1d(axes).ravel()
    fig.suptitle(
        "Высота апогея и перигея по каждому КА",
        fontsize=14,
        fontweight="bold",
    )

    for idx, a in enumerate(assignments):
        ax = axes_flat[idx]
        h = histories[a["id"]]
        t_d = h["t_s"] / 86400.0
        c = _color(a["id"])

        ax.fill_between(
            t_d, h["hp_km"], h["ha_km"], color=c, alpha=0.15, linewidth=0
        )
        ax.plot(t_d, h["ha_km"], color=c, lw=2.0, label="апогей")
        ax.plot(t_d, h["hp_km"], color=c, lw=2.0, ls="--", label="перигей")
        ax.axhline(H_START_KM, color="0.55", ls=":", lw=0.9)
        ax.axhline(H_FINAL_KM, color="0.55", ls=":", lw=0.9)
        ax.axvline(a["t_ell_s"] / 86400.0, color="0.35", ls="--", lw=0.9, alpha=0.7)
        ax.axvline(a["t_ready_s"] / 86400.0, color="0.1", ls="-.", lw=1.0, alpha=0.8)

        group_ru = "вперёд" if a["group"] == "forward" else "назад"
        ax.set_title(
            f"{a['label']} ({group_ru}), φ={a['phi_target_deg']:+.0f}°",
            fontsize=10,
        )
        ax.set_ylim(H_FINAL_KM - 5, H_START_KM + 5)
        ax.grid(True, alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("h, км")
        if idx >= n - ncols:
            ax.set_xlabel("Время, сут")
        ax.legend(fontsize=7, loc="lower left")

        # Подпись ключевых моментов
        ax.text(
            0.98, 0.95,
            f"эллипс: {a['t_ell_s']/86400:.2f} сут\n"
            f"280×280: {a['t_ready_s']/86400:.2f} сут",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="0.8"),
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.text(
        0.5, 0.005,
        "пунктир верт. — уход на эллипс; штрих-пунктир — готовность 280×280; "
        "заливка — диапазон апогей–перигей",
        ha="center", fontsize=9,
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def print_report(result: dict) -> None:
    print("=" * 72)
    print("РАЗЛЁТ: 300×300 → 300×280 → 280×280")
    print("=" * 72)
    print("\nОрбиты")
    print(f"  Старт:     круговая {H_START_KM:.0f} км,  T={result['circ300'].period_s/60:.3f} мин")
    print(f"  Переход:   эллипс {H_APOGEE_KM:.0f}×{H_PERIGEE_KM:.0f} км,  "
          f"T={result['ellip'].period_s/60:.3f} мин")
    print(f"  Финиш:     круговая {H_FINAL_KM:.0f} км,  T={result['circ280'].period_s/60:.3f} мин")
    print(f"  Шаг фазы:  {result['phase_step_deg']:.1f} °")

    print("\nИмпульсы (на каждый КА, одинаково)")
    print(f"  1) апогей  300→эллипс:  Δv = {result['dv1']:.4f} м/с,  "
          f"t = {format_duration(result['t_burn1'])}")
    print(f"  2) перигей эллипс→280:  Δv = {result['dv2']:.4f} м/с,  "
          f"t = {format_duration(result['t_burn2'])}")
    print(f"  Сумма Δv:               {result['dv_total']:.4f} м/с "
          f"(бюджет {result['dv_budget']:.1f} м/с)")
    print(f"  Импульс:                {result['impulse']:.2f} Н·с "
          f"({100*result['impulse']/TOTAL_IMPULSE_NS:.1f}% от {TOTAL_IMPULSE_NS:.0f})")
    print(f"  Рабочее тело:           {result['propellant']*1e3:.2f} г "
          f"({result['propellant_pct']:.1f}% от {PROPELLANT_MASS_KG*1e3:.0f} г)")

    print("\nСтратегия")
    print(f"  Вперёд (F): {result['n_fwd']} КА — сразу на эллипс, затем поочерёдно на 280")
    print(f"    интервал между стартами 2-го импульса: {format_duration(result['dt_fwd'])}")
    print(f"  Назад  (B): {result['n_bwd']} КА — ждут на 300, затем по одному эллипс→280")
    print("  Готовность = целевая фаза И круговая орбита 280×280 (конец 2-го импульса)")

    print("-" * 72)
    print(f"{'id':>3} {'метка':>5} {'группа':>8} {'φ цел.':>8} {'φ факт.':>8} "
          f"{'уход на элл.':>22} {'готов 280×280':>22}")
    print("-" * 72)
    for a in sorted(result["assignments"], key=lambda x: x["phi_final_deg"], reverse=True):
        print(
            f"{a['id']:3d} {a['label']:>5} {a['group']:>8} "
            f"{a['phi_target_deg']:+8.1f} {a['phi_final_deg']:+8.1f} "
            f"{format_duration(a['t_ell_s']):>22} {format_duration(a['t_ready_s']):>22}"
        )

    phis = sorted(a["phi_final_deg"] for a in result["assignments"])
    gaps = []
    for i in range(len(phis)):
        d = phis[(i + 1) % len(phis)] - phis[i]
        if i == len(phis) - 1:
            d += 360.0
        gaps.append(d)
    print("\nПроверка равномерности (зазоры между соседями), °:")
    print("  " + ", ".join(f"{g:.2f}" for g in gaps))

    print("\n" + "=" * 72)
    print(f"ВРЕМЯ МАНЁВРА (все на 280×280 с фазой): {format_duration(result['t_mission'])}")
    print(f"  ({result['t_mission']/86400:.3f} сут)")
    print("=" * 72)


def main() -> None:
    result = compute_phasing()
    print_report(result)
    path = plot_phasing(result)
    path_alt = plot_altitudes_per_sat(result)
    print(f"\nГрафики сохранены: {path}")
    print(f"Высоты по КА:      {path_alt}")


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
