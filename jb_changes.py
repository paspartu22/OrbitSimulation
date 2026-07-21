from pyatmos import download_sw_jb2008, read_sw_jb2008, jb2008
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

print("\n" + "="*80)
print("КОЛЕБАНИЯ ПЛОТНОСТИ АТМОСФЕРЫ ОТ ВРЕМЕНИ (МОДЕЛЬ JB2008)")
print("="*80)

# Загружаем данные космической погоды
print("\nЗагрузка данных космической активности...")
try:
    swfile = download_sw_jb2008() 
    swdata = read_sw_jb2008(swfile)
    if swdata is None or len(swdata) == 0:
        raise ValueError("Downloaded space weather data is empty")
    print("✓ Данные космической активности загружены")
except Exception as e:
    print(f"✗ Ошибка при загрузке данных: {e}")
    print("Используем пустой набор данных (jb2008 будет использовать значения по умолчанию)")
    swdata = None

# Параметры орбиты
lat, lon, alt = 25, 102, 280  # широта, долгота в градусах, высота в км
start_date = datetime(2000, 1, 1)
end_date = datetime(2026, 6, 6)
period_days = (end_date - start_date).days
time_step = 5  # период в днях

# Создаем временной ряд с шагом 7 дней
times = []
current_date = start_date
while current_date <= end_date:
    times.append(current_date)
    current_date += timedelta(days=time_step)
num_points = len(times)
time_strings = [t.strftime('%Y-%m-%d %H:%M:%S') for t in times]

# Высоты, которые мы хотим отслеживать
altitudes_to_track = [280]
colors_alt = {280: 'b'}

print(f"\nРасчет плотности для разных высот за {period_days} дней...")
print(f"Диапазон высот: {altitudes_to_track} км")
print(f"Количество точек: {num_points}")

# Рассчитываем плотность для разных высот
densities_by_altitude = {alt: [] for alt in altitudes_to_track}
temperatures_by_altitude = {alt: [] for alt in altitudes_to_track}
f107_values = []
ap_values = []
days_array = []  # Используем только успешные точки времени
successful_indices = []
error_count = 0

for i, time_str in enumerate(time_strings):
    try:
        # Проверяем, есть ли данные для этой даты
        jb08_result = jb2008(time_str, (lat, lon, altitudes_to_track[0]), swdata)
        
        # Если успешно, добавляем данные для всех высот
        for alt_km in altitudes_to_track:
            jb08_result = jb2008(time_str, (lat, lon, alt_km), swdata)
            densities_by_altitude[alt_km].append(jb08_result.rho)
            temperatures_by_altitude[alt_km].append(jb08_result.T)
        
        # Извлекаем индексы космической активности
        if hasattr(jb08_result, 'F107') and hasattr(jb08_result, 'ap'):
            f107_values.append(jb08_result.F107)
            ap_values.append(jb08_result.ap)
        
        # Добавляем только успешные точки времени
        days_array.append((times[i] - start_date).days)
        successful_indices.append(i)
        
        if (i + 1) % 100 == 0:
            print(f"  Обработано: {i+1}/{num_points} точек (пропущено: {error_count})")
    except Exception as e:
        error_count += 1
        if error_count <= 10:  # Выводим только первые 10 ошибок
            print(f"  Ошибка на временном шаге {i}: {type(e).__name__}")
        elif error_count == 11:
            print(f"  ... (ошибки продолжаются, не все будут показаны)")

# Конвертируем в numpy array
days_array = np.array(days_array)
years_array = days_array / 365.25  # Преобразуем дни в годы

# Создаем графики
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Колебания плотности атмосферы по модели JB2008', 
             fontsize=14, fontweight='bold')

# График 1: Плотность для разных высот
ax = axes[0, 0]
for alt_km in altitudes_to_track:
    if len(densities_by_altitude[alt_km]) > 0:
        ax.semilogy(years_array, densities_by_altitude[alt_km], 
                   color=colors_alt[alt_km], linewidth=2.5, label=f'h = {alt_km} км')
ax.set_xlabel('Время (годы)', fontsize=11)
ax.set_ylabel('Плотность (кг/м³)', fontsize=11)
ax.set_title('Плотность атмосферы для разных высот', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)

# График 2: Температура для разных высот
ax = axes[0, 1]
for alt_km in altitudes_to_track:
    if len(temperatures_by_altitude[alt_km]) > 0:
        ax.plot(years_array, temperatures_by_altitude[alt_km], 
               color=colors_alt[alt_km], linewidth=2.5, label=f'h = {alt_km} км')
ax.set_xlabel('Время (годы)', fontsize=11)
ax.set_ylabel('Температура (K)', fontsize=11)
ax.set_title('Температура атмосферы для разных высот', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# График 3: F10.7 индекс
if f107_values:
    ax = axes[1, 0]
    ax.plot(years_array, f107_values, 'r-', linewidth=2.5)
    ax.set_xlabel('Время (годы)', fontsize=11)
    ax.set_ylabel('F10.7 (SFU)', fontsize=11)
    ax.set_title('Индекс солнечной активности F10.7', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# График 4: Ap индекс
if ap_values:
    ax = axes[1, 1]
    ax.plot(years_array, ap_values, 'b-', linewidth=2.5)
    ax.set_xlabel('Время (годы)', fontsize=11)
    ax.set_ylabel('Ap (nT)', fontsize=11)
    ax.set_title('Индекс геомагнитной активности Ap', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('density_jb2008_timeseries.png', dpi=150, bbox_inches='tight')
print("\n✓ График сохранен: density_jb2008_timeseries.png")
print(f"✓ Успешно обработано: {len(days_array)}/{num_points} точек ({100*len(days_array)//num_points}%)")
print(f"  Пропущено точек: {error_count}")
plt.show()

# Статистика
print("\n" + "="*80)
print("СТАТИСТИКА КОЛЕБАНИЙ ПЛОТНОСТИ")
print("="*80)

for alt_km in altitudes_to_track:
    densities = densities_by_altitude[alt_km]
    if densities:
        density_array = np.array(densities)
        print(f"\nВысота {alt_km} км:")
        print(f"  Min плотность:    {np.min(density_array):.3e} кг/м³")
        print(f"  Max плотность:    {np.max(density_array):.3e} кг/м³")
        print(f"  Среднее значение: {np.mean(density_array):.3e} кг/м³")
        print(f"  Стд. отклонение:  {np.std(density_array):.3e} кг/м³")
        if np.mean(density_array) > 0:
            print(f"  Вариация: {(np.max(density_array) - np.min(density_array))/np.mean(density_array)*100:.1f}%")

print("\n✓ Анализ завершен!")