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
from datetime import datetime, timedelta
from pyatmos import download_sw_jb2008, read_sw_jb2008, jb2008


class OrbitDecayCalculator:
    """Расчет деградации орбиты с использованием JB2008 атмосферной модели"""
    
    # Константы
    EARTH_RADIUS = 6371.0  # км
    GM_EARTH = 3.986004418e5  # км³/с² (гравитационный параметр Земли)
    
    def __init__(self, mass: float = 2.0, cd: float = 2.2, area: float = 0.00503,
                 lat: float = 0.0, lon: float = 0.0, start_date: str = '2026-01-01 00:00:00'):
        """
        Инициализация калькулятора деградации орбиты с JB2008
        
        Args:
            mass: Масса КА в кг
            cd: Коэффициент аэродинамического сопротивления (безразмерный)
            area: Площадь фронтального сечения в м²
            lat: Широта орбиты в градусах
            lon: Долгота орбиты в градусах
            start_date: Начальная дата в формате 'YYYY-MM-DD HH:MM:SS'
        """
        self.mass = mass
        self.cd = cd
        self.area = area
        self.lat = lat
        self.lon = lon
        self.start_date_str = start_date
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        
        # Загружаем данные космической погоды
        print("Загрузка данных космической активности для JB2008...")
        try:
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
            t_days: Время в днях от начальной даты (для симуляции деградации)
            
        Returns:
            Плотность в кг/м³
        """
        # Расчитываем текущую дату
        current_date = self.start_date + timedelta(days=t_days)
        time_str = current_date.strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            jb08_result = jb2008(time_str, (self.lat, self.lon, h), self.swdata)
            return jb08_result.rho
        except Exception as e:
            print(f"Ошибка JB2008 на {time_str}, высота {h}км: {type(e).__name__}")
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
        
        return {
            'altitude_km': h,
            'density_kg_m3': rho,
            'velocity_km_s': v,
            'orbital_period_s': self.orbital_period(h),
            'orbital_period_min': self.orbital_period(h) / 60,
            'drag_force_N': F,
            'deceleration_m_s2': self.deceleration(h, t_days),
            'decay_rate_km_day': dh_dt,
            'time_to_reentry_days': h / (-dh_dt) if dh_dt < 0 else np.inf
        }
    
    def integrate_decay(self, h_initial: float, days: float = 100,
                       num_steps: int = 1000) -> tuple:
        """
        Интегрировать уравнение деградации орбиты до высоты 100 км
        Используется JB2008 модель с учетом времени
        
        Args:
            h_initial: Начальная высота в км
            days: Время интегрирования в днях
            num_steps: Количество шагов
            
        Returns:
            Кортеж (время в днях, высота в км). Симуляция заканчивается на h=100 км
        """
        error_count = [0]  # Счетчик ошибок
        
        def dh_dt_func(h, t):
            if h <= 100:  # Спутник достиг границы атмосферы
                return 0
            # Ограничиваем минимальное время до входа (для численной стабильности)
            try:
                rate = self.decay_rate(h, t)
            except Exception as e:
                error_count[0] += 1
                if error_count[0] <= 5:
                    print(f"  Ошибка при h={h:.1f}, t={t:.1f}: {type(e).__name__}: {e}")
                return 0
                
            if h > 100 and rate < 0 and (h - 100) / (-rate) < 0.001:
                # Если остаток времени очень мал, возвращаем ноль
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
        
        # Находим индекс, где высота впервые достигает или опускается ниже 100 км
        idx_100 = np.where(h_array <= 100)[0]
        if len(idx_100) > 0:
            # Берем все значения до первого пересечения h=100
            idx = idx_100[0]
            t_array = t_array[:idx+1]
            h_array = h_array[:idx+1]
            # Убедимся, что последняя точка ровно на 100 км
            if idx < len(h_array) - 1 or h_array[-1] < 100:
                h_array[-1] = 100.0
        
        return t_array, h_array
    
    def plot_decay_analysis(self, altitude: float = 280.0, figsize: tuple = (16, 5)):
        """
        Визуализировать анализ деградации орбиты
        
        Args:
            altitude: Начальная высота в км
            figsize: Размер фигуры
        """
        details = self.decay_rate_detailed(altitude)
        
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
        Время до входа в атмосферу: {details['time_to_reentry_days']:.1f} дней
        """
        ax.text(0.05, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Рассчитываем деградацию (используется для всех графиков)
        days_to_reentry = min(details['time_to_reentry_days'], 1000)
        t, h = self.integrate_decay(altitude, days=days_to_reentry, num_steps=500)
        
        # 2. Эволюция орбиты во времени
        ax = axes[1]
        ax.plot(t, h, 'b-', linewidth=2.5, label='Высота орбиты')
        ax.axhline(y=100, color='r', linestyle='--', linewidth=2, label='Граница атмосферы (100 км)')
        ax.fill_between(t, 0, 100, alpha=0.2, color='red', label='Зона входа в атмосферу')
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
        
        print(f"\nПАРАМЕТРЫ НА ВЫСОТЕ {altitude} км (начальный момент времени):")
        details = self.decay_rate_detailed(altitude, t_days=0)
        
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
        print(f"    Время до входа в атмосферу: {details['time_to_reentry_days']:.2f} дней")
        print(f"                                 ({details['time_to_reentry_days']/365.25:.3f} лет)")


def main():
    """Главная функция"""
    
    # Параметры КА
    mass = 2.0  # кг
    cd = 2.2
    area = 0.00503  # м²
    
    # Параметры орбиты и времени
    lat = 0.0  # широта в градусах
    lon = 0.0  # долгота в градусах
    start_date = '2026-01-01 00:00:00'  # начальная дата
    
    # Создаем калькулятор
    calc = OrbitDecayCalculator(mass=mass, cd=cd, area=area, 
                                lat=lat, lon=lon, start_date=start_date)
    
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
