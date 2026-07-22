"""
Расчет деградации орбиты под воздействием аэродинамического сопротивления атмосферы
Используется модель JB2008 с учетом космической погоды

Параметры КА:
- Высота орбиты: h = 280 км
- Масса КА: m = 2.0 кг
- Коэффициент аэродинамического сопротивления: Cd = 2.2
- Площадь аэродинамического сопротивления (модель): A = 0.00503 м²
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from datetime import datetime
from ssl_bootstrap import configure_ssl_certificates


configure_ssl_certificates()
from pyatmos import download_sw_jb2008, read_sw_jb2008, jb2008


class OrbitDecayCalculator:
    """Расчет деградации орбиты с использованием JB2008 атмосферной модели"""
    
    # Константы
    EARTH_RADIUS = 6371.0  # км
    GM_EARTH = 3.986004418e5  # км³/с² (гравитационный параметр Земли)
    REENTRY_ALTITUDE = 110.0  # км
    G0 = 9.80665  # м/с²

    # Характеристики ДУ
    THRUST_N = 6e-3
    TOTAL_IMPULSE_NS = 220.0
    PROPELLANT_MASS_KG = 0.13
    
    def __init__(self, mass: float = 2.0, cd: float = 2.2, area: float = 0.00503, lat: float = 0.0, lon: float = 0.0,
                 start_date: str = '2020-01-01 00:00:00',
                 atmosphere_time: str = '2020-01-01 00:00:00'):
        """
        Инициализация калькулятора деградации орбиты с JB2008
        
        Args:
            mass: Масса КА в кг
            cd: Коэффициент аэродинамического сопротивления (безразмерный)
            area: Площадь фронтального сечения в м²
            lat: Широта орбиты в градусах
            lon: Долгота орбиты в градусах
            start_date: Начальная дата в формате 'YYYY-MM-DD HH:MM:SS'
            atmosphere_time: Фиксированное время атмосферы для статического расчета
        """
        self.mass = mass
        self.cd = cd
        self.area = area
        self.lat = lat
        self.lon = lon
        self.start_date_str = start_date
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        self.atmosphere_time_str = atmosphere_time
        self.atmosphere_time = datetime.strptime(atmosphere_time, '%Y-%m-%d %H:%M:%S')

        self.mass_current = mass
        self.impulse_used_ns = 0.0
        self.propellant_used_kg = 0.0
        self.isp_s = self.TOTAL_IMPULSE_NS / (self.PROPELLANT_MASS_KG * self.G0)
        
        # Загружаем данные космической погоды
        print("Загрузка данных космической активности для JB2008...")
        try:
            configure_ssl_certificates()
            swfile = download_sw_jb2008()
            self.swdata = read_sw_jb2008(swfile)
            print("✓ Данные космической активности загружены")
        except Exception as e:
            print(f"✗ Ошибка при загрузке данных: {e}")
            print("Используем пустой набор данных (будут использованы значения по умолчанию)")
            self.swdata = None
        
    def get_density_at_altitude(self, h: float, t_days: float = 0) -> float:
        """
        Получить плотность атмосферы на высоте h в км (JB2008)
        
        Args:
            h: Высота в км
            t_days: Не используется в статическом режиме
            
        Returns:
            Плотность в кг/м³
        """
        # odeint может передавать высоту как ndarray([h]); JB2008 ожидает скаляр.
        h_scalar = float(np.atleast_1d(h)[0])

        # Статическая атмосфера: используем фиксированный момент времени.
        time_str = self.atmosphere_time.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            jb08_result = jb2008(time_str, (self.lat, self.lon, h_scalar), self.swdata)
            return jb08_result.rho
        except Exception as e:
            print(f"Ошибка JB2008 на {time_str}, высота {h_scalar}км: {type(e).__name__}")
            # Возвращаем минимальное значение плотности в случае ошибки
            return 1e-15
    
    def orbital_velocity(self, h: float) -> float:
        """
        Скорость на круговой орбите
        
        Args:
            h: Высота в км
            
        Returns:
            Скорость в км/с
        """
        r = self.EARTH_RADIUS + h
        return np.sqrt(self.GM_EARTH / r)
    
    def orbital_period(self, h: float) -> float:
        """
        Период обращения спутника
        
        Args:
            h: Высота в км
            
        Returns:
            Период в секундах
        """
        r = self.EARTH_RADIUS + h
        return 2 * np.pi * np.sqrt(r**3 / self.GM_EARTH)

    def apogee_velocity(self, rp_km: float, ra_km: float) -> float:
        """Скорость в апогее эллиптической орбиты (км/с)."""
        rp_r = self.EARTH_RADIUS + rp_km
        ra_r = self.EARTH_RADIUS + ra_km
        a = 0.5 * (rp_r + ra_r)
        return np.sqrt(self.GM_EARTH * (2.0 / ra_r - 1.0 / a))

    def perigee_from_apogee_state(self, ra_km: float, v_apogee_km_s: float) -> float:
        """Перигей (км) после импульса, заданного в апогее."""
        ra_r = self.EARTH_RADIUS + ra_km
        a_new = 1.0 / (2.0 / ra_r - (v_apogee_km_s ** 2) / self.GM_EARTH)
        rp_r = 2.0 * a_new - ra_r
        return rp_r - self.EARTH_RADIUS

    def delta_v_to_restore_perigee(self, rp_km: float, ra_km: float, rp_target_km: float) -> float:
        """Требуемый Δv в апогее для подъема перигея до rp_target (км/с)."""
        rp_target = min(rp_target_km, ra_km)
        if rp_km >= rp_target:
            return 0.0

        v_now = self.apogee_velocity(rp_km, ra_km)
        v_target = self.apogee_velocity(rp_target, ra_km)
        return max(0.0, v_target - v_now)

    def apply_apogee_burn(self, rp_km: float, ra_km: float, dv_km_s: float) -> tuple:
        """Применить импульс в апогее с учетом ограничений ДУ."""
        if dv_km_s <= 0.0:
            return rp_km, 0.0, 0.0, 0.0, 0.0

        impulse_left = self.TOTAL_IMPULSE_NS - self.impulse_used_ns
        if impulse_left <= 0.0 or self.mass_current <= 0.0:
            return rp_km, 0.0, 0.0, 0.0, 0.0

        dv_max_m_s = impulse_left / self.mass_current
        dv_apply_m_s = min(dv_km_s * 1000.0, dv_max_m_s)
        dv_apply_km_s = dv_apply_m_s / 1000.0

        impulse_used = self.mass_current * dv_apply_m_s
        burn_time_s = impulse_used / self.THRUST_N
        prop_used = self.PROPELLANT_MASS_KG * (impulse_used / self.TOTAL_IMPULSE_NS)

        self.impulse_used_ns += impulse_used
        self.propellant_used_kg += prop_used
        self.mass_current = max(self.mass - self.propellant_used_kg, self.mass - self.PROPELLANT_MASS_KG)

        v_old = self.apogee_velocity(rp_km, ra_km)
        v_new = v_old + dv_apply_km_s
        rp_new = self.perigee_from_apogee_state(ra_km, v_new)
        rp_new = min(max(rp_new, 0.0), ra_km)

        return rp_new, dv_apply_km_s, burn_time_s, impulse_used, prop_used
    
    def drag_force(self, h: float, t_days: float = 0) -> float:
        """
        Сила аэродинамического сопротивления
        
        F_drag = 0.5 * ρ * v² * Cd * A
        
        Args:
            h: Высота в км
            t_days: Время в днях от начальной даты
            
        Returns:
            Сила в Н
        """
        rho = self.get_density_at_altitude(h, t_days)  # кг/м³ (JB2008)
        v = self.orbital_velocity(h) * 1000  # м/с
        
        F_drag = 0.5 * rho * v**2 * self.cd * self.area
        return F_drag
    
    def deceleration(self, h: float, t_days: float = 0) -> float:
        """
        Замедление от аэродинамического сопротивления
        
        a = F_drag / m
        
        Args:
            h: Высота в км
            t_days: Время в днях от начальной даты
            
        Returns:
            Ускорение в м/с²
        """
        F = self.drag_force(h, t_days)
        return F / self.mass
    
    def decay_rate(self, h: float, t_days: float = 0) -> float:
        """
        Скорость понижения орбиты (dh/dt)
        
        Упрощенная формула:
        dh/dt = -(F_drag * v) / (m * g_orbital)
        где g_orbital = v²/r
        
        Args:
            h: Высота в км
            t_days: Время в днях от начальной даты
            
        Returns:
            Скорость понижения в км/день
        """
        v = self.orbital_velocity(h)  # км/с
        r = self.EARTH_RADIUS + h  # км
        g_orbital = v**2 / r  # км/с²
        
        F = self.drag_force(h, t_days)  # Н
        
        # dh/dt в км/с
        dh_dt = -(F * v * 1000) / (self.mass * g_orbital * 1000**2)
        
        # Преобразование в км/день
        dh_dt_per_day = dh_dt * 86400  # км/день
        
        return dh_dt_per_day
    
    def decay_rate_detailed(self, h: float, t_days: float = 0) -> dict:
        """
        Подробный расчет скорости понижения орбиты
        
        Args:
            h: Высота в км
            t_days: Время в днях от начальной даты
            
        Returns:
            Словарь с деталями расчета
        """
        rho = self.get_density_at_altitude(h, t_days)
        v = self.orbital_velocity(h)
        r = self.EARTH_RADIUS + h
        F = self.drag_force(h, t_days)
        dh_dt = self.decay_rate(h, t_days)
        
        time_linear = max(0.0, h - self.REENTRY_ALTITUDE) / (-dh_dt) if dh_dt < 0 else np.inf

        return {
            'altitude_km': h,
            'density_kg_m3': rho,
            'velocity_km_s': v,
            'orbital_period_s': self.orbital_period(h),
            'orbital_period_min': self.orbital_period(h) / 60,
            'drag_force_N': F,
            'deceleration_m_s2': self.deceleration(h, t_days),
            'decay_rate_km_day': dh_dt,
            # Линейная оценка (по локальному dh/dt на начальной высоте).
            'time_to_reentry_days_linear': time_linear,
            # Для обратной совместимости.
            'time_to_reentry_days': time_linear,
        }

    def estimate_reentry_time_integrated(self, h_initial: float, max_days: float,
                                         num_steps: int = 2000) -> float:
        """Оценить время до REENTRY_ALTITUDE по интегрированной траектории."""
        t, h = self.integrate_decay(h_initial, days=max_days, num_steps=num_steps)
        if len(h) > 0 and h[-1] <= self.REENTRY_ALTITUDE + 1e-6:
            return float(t[-1])
        return np.inf

    def simulate_station_keeping(self, h_initial: float, days: float = 3650.0,
                                 maneuver_interval_days: float = 1.0) -> dict:
        """Удержание орбиты: маневр в апогее раз в сутки для подъема перигея."""
        self.mass_current = self.mass
        self.impulse_used_ns = 0.0
        self.propellant_used_kg = 0.0

        t_days = [0.0]
        rp_hist = [h_initial]
        ra_hist = [h_initial]
        burn_days = []
        burn_dv_m_s = []
        burn_impulse_ns = []
        burn_time_s = []
        burn_propellant_kg = []
        impulse_left_hist = [self.TOTAL_IMPULSE_NS]

        target_rp = h_initial
        dt = maneuver_interval_days
        steps = int(np.ceil(days / dt))

        for step in range(1, steps + 1):
            t = step * dt
            rp = rp_hist[-1]
            ra = ra_hist[-1]

            if rp <= self.REENTRY_ALTITUDE:
                break

            rp_next = max(0.0, rp + self.decay_rate(rp) * dt)
            ra_next = max(rp_next, ra + self.decay_rate(ra) * dt)

            dv_need = self.delta_v_to_restore_perigee(rp_next, ra_next, target_rp)
            rp_after_burn, dv_applied_km_s, burn_time, impulse_used, prop_used = self.apply_apogee_burn(
                rp_next, ra_next, dv_need
            )

            if dv_applied_km_s > 0.0:
                burn_days.append(t)
                burn_dv_m_s.append(dv_applied_km_s * 1000.0)
                burn_impulse_ns.append(impulse_used)
                burn_time_s.append(burn_time)
                burn_propellant_kg.append(prop_used)

            t_days.append(t)
            rp_hist.append(rp_after_burn)
            ra_hist.append(ra_next)
            impulse_left_hist.append(max(0.0, self.TOTAL_IMPULSE_NS - self.impulse_used_ns))

            if rp_after_burn <= self.REENTRY_ALTITUDE:
                break

        return {
            't_days': np.array(t_days),
            'rp_km': np.array(rp_hist),
            'ra_km': np.array(ra_hist),
            'burn_days': np.array(burn_days),
            'burn_dv_m_s': np.array(burn_dv_m_s),
            'burn_impulse_ns': np.array(burn_impulse_ns),
            'burn_time_s': np.array(burn_time_s),
            'burn_propellant_kg': np.array(burn_propellant_kg),
            'impulse_left_ns': np.array(impulse_left_hist),
            'mass_final_kg': self.mass_current,
            'impulse_used_ns': self.impulse_used_ns,
            'propellant_used_kg': self.propellant_used_kg,
        }

    def plot_station_keeping(self, result: dict, figsize: tuple = (14, 8)):
        """Графики station-keeping с ежедневными импульсами в апогее."""
        t = result['t_days']
        rp = result['rp_km']
        ra = result['ra_km']
        burns_t = result['burn_days']
        burns_dv = result['burn_dv_m_s']
        impulse_left = result['impulse_left_ns']

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        ax = axes[0]
        ax.plot(t, rp, label='Перигей, км', color='tab:blue', linewidth=2.0)
        ax.plot(t, ra, label='Апогей, км', color='tab:orange', linewidth=2.0)
        ax.axhline(self.REENTRY_ALTITUDE, color='tab:red', linestyle='--',
                   label=f'Порог входа ({self.REENTRY_ALTITUDE:.0f} км)')
        ax.axhline(rp[0], color='tab:green', linestyle=':', label='Целевой перигей')
        ax.set_ylabel('Высота, км')
        ax.set_title('Удержание орбиты: маневр в апогее 1 раз в сутки')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        ax = axes[1]
        ax.plot(t, impulse_left, color='tab:purple', linewidth=2.0, label='Остаток импульса, Н·с')
        if len(burns_t) > 0:
            ax2 = ax.twinx()
            ax2.scatter(burns_t, burns_dv, s=12, color='tab:red', alpha=0.7, label='Δv маневра, м/с')
            ax2.set_ylabel('Δv маневра, м/с')
        ax.set_xlabel('Время, дни')
        ax.set_ylabel('Остаток импульса, Н·с')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes
    
    def integrate_decay(self, h_initial: float, days: float = 100,
                       num_steps: int = 1000) -> tuple:
        """
        Интегрировать уравнение деградации орбиты до высоты REENTRY_ALTITUDE
        Используется JB2008 модель с учетом времени
        
        Args:
            h_initial: Начальная высота в км
            days: Время интегрирования в днях
            num_steps: Количество шагов
            
        Returns:
            Кортеж (время в днях, высота в км). Симуляция заканчивается на h=REENTRY_ALTITUDE
        """
        error_count = [0]  # Счетчик ошибок
        
        def dh_dt_func(h, t):
            h_scalar = float(np.atleast_1d(h)[0])
            if h_scalar <= self.REENTRY_ALTITUDE:  # Спутник достиг границы входа
                return 0
            try:
                rate = self.decay_rate(h_scalar, t)
            except Exception as e:
                error_count[0] += 1
                if error_count[0] <= 5:
                    print(f"  Ошибка при h={h_scalar:.1f}, t={t:.1f}: {type(e).__name__}: {e}")
                return 0

            return rate
        
        t_array = np.linspace(0, days, num_steps)
        try:
            h_array = odeint(dh_dt_func, h_initial, t_array, rtol=1e-4, atol=1e-3)
        except Exception as e:
            print(f"Ошибка при интеграции ODE: {e}")
            return t_array, np.full_like(t_array, h_initial)
            
        h_array = h_array.flatten()
        
        if error_count[0] > 0:
            print(f"  Всего ошибок в ODE: {error_count[0]}")
        
        # Находим индекс, где высота впервые достигает или опускается ниже REENTRY_ALTITUDE.
        # Если пересечение не попало точно в сетку, принудительно завершаем
        # траекторию на REENTRY_ALTITUDE, когда остаемся в узкой окрестности порога.
        idx_100 = np.where(h_array <= self.REENTRY_ALTITUDE)[0]
        if len(idx_100) > 0:
            # Берем все значения до первого пересечения порога высоты.
            idx = idx_100[0]
            t_array = t_array[:idx+1]
            h_array = h_array[:idx+1]
            # Убедимся, что последняя точка ровно на пороге высоты входа.
            if idx < len(h_array) - 1 or h_array[-1] < self.REENTRY_ALTITUDE:
                h_array[-1] = self.REENTRY_ALTITUDE
        elif h_array[-1] <= self.REENTRY_ALTITUDE + 0.05:
            h_array[-1] = self.REENTRY_ALTITUDE
        
        return t_array, h_array
    
    def plot_decay_analysis(self, altitude: float = 280.0, figsize: tuple = (16, 5)):
        """
        Визуализировать анализ деградации орбиты
        
        Args:
            altitude: Начальная высота в км
            figsize: Размер фигуры
        """
        details = self.decay_rate_detailed(altitude)
        linear_days = details['time_to_reentry_days_linear']
        integration_window = min(max(linear_days * 3, 365.0), 10000.0) if np.isfinite(linear_days) else 1000.0
        integrated_days = self.estimate_reentry_time_integrated(altitude, max_days=integration_window)
        display_days = integrated_days if np.isfinite(integrated_days) else linear_days
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'Анализ деградации орбиты на высоте {altitude} км\n'
                     f'КА: m={self.mass} кг, Cd={self.cd}, A={self.area} м²',
                     fontsize=14, fontweight='bold')
        
        # 1. Информационная панель
        ax = axes[0]
        ax.axis('off')
        info_text = f"""
        ПАРАМЕТРЫ КА И ОРБИТЫ:
        
        Масса: m = {self.mass} кг
        Cd (коэффициент сопротивления) = {self.cd}
        Площадь фронтального сечения: A = {self.area} м²
        
        ПАРАМЕТРЫ НА ВЫСОТЕ {altitude} км:
        
        Плотность: ρ = {details['density_kg_m3']:.3e} кг/м³
        Скорость: v = {details['velocity_km_s']:.3f} км/с
        Период: T = {details['orbital_period_min']:.2f} мин
        
        Сила сопротивления: F = {details['drag_force_N']:.3e} Н
        Замедление: a = {details['deceleration_m_s2']:.3e} м/с²
        
        Скорость понижения: dh/dt = {details['decay_rate_km_day']:.3f} км/день
        Время до высоты {self.REENTRY_ALTITUDE:.0f} км (интегр.): {display_days:.1f} дней
        Линейная оценка: {linear_days:.1f} дней
        """
        ax.text(0.05, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Рассчитываем деградацию (используется для всех графиков)
        days_to_reentry = display_days if np.isfinite(display_days) else 1000
        t, h = self.integrate_decay(altitude, days=days_to_reentry, num_steps=500)
        
        # 2. Эволюция орбиты во времени
        ax = axes[1]
        ax.plot(t, h, 'b-', linewidth=2.5, label='Высота орбиты')
        ax.axhline(y=self.REENTRY_ALTITUDE, color='r', linestyle='--', linewidth=2,
               label=f'Граница входа ({self.REENTRY_ALTITUDE:.0f} км)')
        ax.fill_between(t, 0, self.REENTRY_ALTITUDE, alpha=0.2, color='red', label='Зона входа в атмосферу')
        ax.set_xlabel('Время (дни)', fontsize=11)
        ax.set_ylabel('Высота орбиты (км)', fontsize=11)
        ax.set_title('Эволюция орбиты', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, altitude + 50)
        
        # 3. Плотность атмосферы от времени
        ax = axes[2]
        densities = [self.get_density_at_altitude(hi) for hi in h]
        ax.semilogy(t, densities, 'g-', linewidth=2.5)
        ax.set_xlabel('Время (дни)', fontsize=11)
        ax.set_ylabel('Плотность (кг/м³)', fontsize=11)
        ax.set_title('Плотность атмосферы от времени', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        return fig, axes
    
    def print_analysis(self, altitude: float = 280.0):
        """Вывести подробный анализ"""
        print("\n" + "=" * 80)
        print("АНАЛИЗ ДЕГРАДАЦИИ ОРБИТЫ (JB2008 ATMOSPHERE MODEL)")
        print("=" * 80)
        
        print(f"\nПАРАМЕТРЫ КА:")
        print(f"  Масса: {self.mass} кг")
        print(f"  Коэффициент сопротивления Cd: {self.cd}")
        print(f"  Площадь фронтального сечения: {self.area} м²")
        print(f"  Характеристический масштаб (m/Cd/A): {self.mass/(self.cd*self.area):.2f} кг/м²")
        
        print(f"\nПАРАМЕТРЫ ОРБИТЫ:")
        print(f"  Широта: {self.lat}°, Долгота: {self.lon}°")
        print(f"  Начальная дата: {self.start_date_str}")
        print(f"  Фиксированное время атмосферы: {self.atmosphere_time_str}")
        
        print(f"\nПАРАМЕТРЫ НА ВЫСОТЕ {altitude} км (начальный момент времени):")
        details = self.decay_rate_detailed(altitude, t_days=0)
        linear_days = details['time_to_reentry_days_linear']
        integration_window = min(max(linear_days * 3, 365.0), 10000.0) if np.isfinite(linear_days) else 1000.0
        integrated_days = self.estimate_reentry_time_integrated(altitude, max_days=integration_window)
        
        print(f"\n  Атмосферные:")
        print(f"    Плотность: ρ = {details['density_kg_m3']:.3e} кг/м³")
        
        print(f"\n  Орбитальные:")
        print(f"    Скорость: v = {details['velocity_km_s']:.4f} км/с = {details['velocity_km_s']*1000:.1f} м/с")
        print(f"    Период: T = {details['orbital_period_min']:.3f} мин = {details['orbital_period_s']:.1f} с")
        
        print(f"\n  Динамика:")
        print(f"    Сила сопротивления: F_drag = {details['drag_force_N']:.3e} Н")
        print(f"    Замедление: a = {details['deceleration_m_s2']:.3e} м/с²")
        
        print(f"\n  РЕЗУЛЬТАТЫ:")
        print(f"    Скорость понижения орбиты: dh/dt = {details['decay_rate_km_day']:.6f} км/день")
        if np.isfinite(integrated_days):
            print(f"    Время до высоты {self.REENTRY_ALTITUDE:.0f} км (интегр.): {integrated_days:.2f} дней")
            print(f"                                             ({integrated_days/365.25:.3f} лет)")
        else:
            print(f"    Время до высоты {self.REENTRY_ALTITUDE:.0f} км (интегр.): не достигнуто")
        print(f"    Линейная оценка (локальная): {linear_days:.2f} дней")
        print(f"                                 ({linear_days/365.25:.3f} лет)")


def main():
    """Главная функция"""
    
    # Параметры КА
    mass = 2.0  # кг
    cd = 2.2
    area = 0.00503  # м²
    
    # Параметры орбиты и времени
    lat = 0.0  # широта в градусах
    lon = 0.0  # долгота в градусах
    start_date = '2020-01-01 00:00:00'  # начальная дата
    atmosphere_time = '2020-01-01 00:00:00'  # статическое состояние атмосферы
    
    # Создаем калькулятор
    calc = OrbitDecayCalculator(mass=mass, cd=cd, area=area, 
                                lat=lat, lon=lon, start_date=start_date,
                                atmosphere_time=atmosphere_time)
    
    # Анализ на высоте 280 км
    altitude = 280.0
    
    # Вывод анализа
    calc.print_analysis(altitude)
    
    # Визуализация
    print("\n" + "=" * 80)
    print("Создание графиков...")
    print("=" * 80)
    
    fig, _ = calc.plot_decay_analysis(altitude=altitude)
    plt.savefig('orbit_decay_analysis_280km.png', dpi=150, bbox_inches='tight')
    print("✓ График сохранен: orbit_decay_analysis_280km.png")

    print("\n" + "=" * 80)
    print("РЕЖИМ УДЕРЖАНИЯ ОРБИТЫ (МАНЕВР В АПОГЕЕ 1 РАЗ В СУТКИ)")
    print("=" * 80)

    keep = calc.simulate_station_keeping(altitude, days=3650, maneuver_interval_days=1.0)
    fig_keep, _ = calc.plot_station_keeping(keep)
    plt.savefig('orbit_station_keeping_280km.png', dpi=150, bbox_inches='tight')
    print("✓ График сохранен: orbit_station_keeping_280km.png")

    burn_count = len(keep['burn_days'])
    last_day = float(keep['t_days'][-1]) if len(keep['t_days']) > 0 else 0.0
    rp_last = float(keep['rp_km'][-1]) if len(keep['rp_km']) > 0 else 0.0
    impulse_left = max(0.0, calc.TOTAL_IMPULSE_NS - keep['impulse_used_ns'])

    print(f"  Выполнено маневров: {burn_count}")
    print(f"  Длительность моделирования: {last_day:.1f} дней")
    print(f"  Финальный перигей: {rp_last:.2f} км")
    print(f"  Израсходовано импульса: {keep['impulse_used_ns']:.2f} Н·с")
    print(f"  Остаток импульса: {impulse_left:.2f} Н·с")
    print(f"  Израсходовано рабочего тела: {keep['propellant_used_kg']:.4f} кг")
    
    # Анализ для разных космических погод
    print("\n" + "=" * 80)
    print("ПРИМЕЧАНИЕ:")
    print("=" * 80)
    print("\nJB2008 - это эмпирическая атмосферная модель, разработанная в Space Force,")
    print("которая учитывает изменения солнечной активности и геомагнитных возмущений.")
    print("Она дает более реалистичные результаты для предсказания деградации орбиты,")
    print("особенно при учете вариаций космической погоды во времени.")
    
    plt.close('all')
    print("\n✓ Анализ завершен!")


if __name__ == "__main__":
    main()
