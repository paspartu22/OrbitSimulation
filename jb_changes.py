import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import csv
from datetime import datetime, timedelta
from ssl_bootstrap import configure_ssl_certificates
configure_ssl_certificates()
from pyatmos import download_sw_jb2008, read_sw_jb2008, jb2008


def decimal_year_from_days(start_year: int, days: np.ndarray) -> np.ndarray:
    return start_year + np.asarray(days, dtype=float) / 365.25


def month_start(date_value: datetime) -> datetime:
    return datetime(date_value.year, date_value.month, 1)


def build_monthly_average_series(dates: list,
                                 series_by_name: dict) -> tuple:
    month_to_indices = {}
    for idx, date_value in enumerate(dates):
        key = month_start(date_value)
        month_to_indices.setdefault(key, []).append(idx)

    monthly_dates = sorted(month_to_indices.keys())
    monthly_series = {name: [] for name in series_by_name}

    for month_date in monthly_dates:
        month_indices = month_to_indices[month_date]
        for name, values in series_by_name.items():
            values_array = np.asarray(values, dtype=float)
            month_values = values_array[month_indices]
            if np.any(np.isfinite(month_values)):
                monthly_series[name].append(float(np.nanmean(month_values)))
            else:
                monthly_series[name].append(np.nan)

    return monthly_dates, monthly_series


def fit_harmonic_density_forecast(days: np.ndarray,
                                  densities: np.ndarray,
                                  forecast_days: np.ndarray,
                                  harmonic_count: int = 4,
                                  quantile_levels: tuple = (0.05, 0.5, 0.95)) -> dict:
    """Аппроксимация log(rho) линейным трендом и суммой синусоид."""
    x_days = np.asarray(days, dtype=float)
    x_years = x_days / 365.25
    y = np.log(np.clip(np.asarray(densities, dtype=float), 1e-20, None))

    if len(x_days) < 12:
        raise ValueError("Недостаточно точек для гармонического прогноза")

    # Делим ряд по времени: обучаемся на ранней части и выбираем количество гармоник
    # по качеству на последних точках, затем переобучаем на всем интервале.
    validation_points = max(6, min(24, len(x_days) // 5))
    train_count = len(x_days) - validation_points
    if train_count < 8:
        validation_points = max(4, len(x_days) // 4)
        train_count = len(x_days) - validation_points

    x_train_years = x_years[:train_count]
    y_train = y[:train_count]
    x_val_years = x_years[train_count:]
    y_val = y[train_count:]

    trend_coeffs = np.polyfit(x_train_years, y_train, deg=1)
    trend_train = np.polyval(trend_coeffs, x_train_years)
    detrended_train = y_train - trend_train

    sample_spacing_years = np.median(np.diff(x_days[:train_count])) / 365.25
    fft_freqs = np.fft.rfftfreq(len(detrended_train), d=sample_spacing_years)
    fft_amplitudes = np.abs(np.fft.rfft(detrended_train))
    valid_freq_mask = (fft_freqs > 0.0) & (fft_freqs >= 1.0 / 24.0) & (fft_freqs <= 1.0 / 0.25)
    valid_indices = np.where(valid_freq_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError("Не удалось выделить гармоники для прогноза")

    ranked_indices = valid_indices[np.argsort(fft_amplitudes[valid_indices])[::-1]]

    max_harmonics = max(1, min(int(harmonic_count), len(ranked_indices), 12))
    min_harmonics = 1 if max_harmonics < 3 else 3

    def build_design_matrix(t_years: np.ndarray, selected_freqs: np.ndarray) -> np.ndarray:
        columns = [np.ones_like(t_years), t_years]
        for freq in selected_freqs:
            omega_t = 2.0 * np.pi * freq * t_years
            columns.append(np.sin(omega_t))
            columns.append(np.cos(omega_t))
        return np.column_stack(columns)

    best_score = np.inf
    best_freqs = None
    for harmonic_num in range(min_harmonics, max_harmonics + 1):
        candidate_indices = np.sort(ranked_indices[:harmonic_num])
        candidate_freqs = fft_freqs[candidate_indices]

        x_train_design = build_design_matrix(x_train_years, candidate_freqs)
        coeffs_train, _, _, _ = np.linalg.lstsq(x_train_design, y_train, rcond=None)

        x_val_design = build_design_matrix(x_val_years, candidate_freqs)
        y_val_pred = x_val_design @ coeffs_train
        val_rmse = float(np.sqrt(np.mean((y_val_pred - y_val) ** 2)))

        if val_rmse < best_score:
            best_score = val_rmse
            best_freqs = candidate_freqs

    if best_freqs is None:
        raise ValueError("Не удалось подобрать набор гармоник")

    design_matrix = build_design_matrix(x_years, best_freqs)
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)

    forecast_years = np.asarray(forecast_days, dtype=float) / 365.25
    fitted_log = design_matrix @ coefficients
    forecast_log = build_design_matrix(forecast_years, best_freqs) @ coefficients

    residuals_log = y - fitted_log
    quantile_levels = np.asarray(quantile_levels, dtype=float)
    residual_quantiles = np.quantile(residuals_log, quantile_levels)

    fitted_quantiles = np.array([
        np.exp(fitted_log + residual_q) for residual_q in residual_quantiles
    ])
    forecast_quantiles = np.array([
        np.exp(forecast_log + residual_q) for residual_q in residual_quantiles
    ])

    return {
        'coefficients': coefficients,
        'frequencies_per_year': best_freqs,
        'periods_years': 1.0 / best_freqs,
        'fitted_density': np.exp(fitted_log),
        'forecast_density': np.exp(forecast_log),
        'rmse_log': float(np.sqrt(np.mean((fitted_log - y) ** 2))),
        'rmse_val_log': best_score,
        'harmonics_used': int(len(best_freqs)),
        'quantile_levels': quantile_levels,
        'fitted_quantiles': fitted_quantiles,
        'forecast_quantiles': forecast_quantiles,
    }

print("\n" + "="*80)
print("КОЛЕБАНИЯ ПЛОТНОСТИ АТМОСФЕРЫ ОТ ВРЕМЕНИ (МОДЕЛЬ JB2008)")
print("="*80)

# Загружаем данные космической погоды
print("\nЗагрузка данных космической активности...")
try:
    configure_ssl_certificates()
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
forecast_end_date = datetime(2030, 12, 31)
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
        
        # Извлекаем индексы космической активности и держим длины рядов синхронными
        if hasattr(jb08_result, 'F107'):
            f107_values.append(jb08_result.F107)
        else:
            f107_values.append(np.nan)

        if hasattr(jb08_result, 'ap'):
            ap_values.append(jb08_result.ap)
        else:
            ap_values.append(np.nan)
        
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
calendar_years_array = decimal_year_from_days(start_date.year, days_array)

successful_times = [times[i] for i in successful_indices]
forecast_times = []
current_date = (successful_times[-1] if successful_times else end_date) + timedelta(days=time_step)
while current_date <= forecast_end_date:
    forecast_times.append(current_date)
    current_date += timedelta(days=time_step)

monthly_input_series = {
    'density_280': densities_by_altitude[280],
    'temperature_280': temperatures_by_altitude[280],
    'f107': f107_values,
    'ap': ap_values,
}
monthly_dates, monthly_series = build_monthly_average_series(successful_times, monthly_input_series)
monthly_days_array = np.array([(t - start_date).days for t in monthly_dates], dtype=float)
monthly_calendar_years = decimal_year_from_days(start_date.year, monthly_days_array)

monthly_forecast_times = []
if monthly_dates:
    last_month = monthly_dates[-1]
else:
    last_month = month_start(end_date)

if last_month.month == 12:
    next_month = datetime(last_month.year + 1, 1, 1)
else:
    next_month = datetime(last_month.year, last_month.month + 1, 1)

while next_month <= forecast_end_date:
    monthly_forecast_times.append(next_month)
    if next_month.month == 12:
        next_month = datetime(next_month.year + 1, 1, 1)
    else:
        next_month = datetime(next_month.year, next_month.month + 1, 1)

forecast_days_array = np.array([(t - start_date).days for t in monthly_forecast_times], dtype=float)
forecast_calendar_years = decimal_year_from_days(start_date.year, forecast_days_array)

density_forecasts = {}
for alt_km in altitudes_to_track:
    density_series = np.array(monthly_series[f'density_{alt_km}'], dtype=float)
    density_mask = np.isfinite(density_series) & (density_series > 0)
    if np.count_nonzero(density_mask) > 8 and len(forecast_days_array) > 0:
        density_forecasts[alt_km] = fit_harmonic_density_forecast(
            days=monthly_days_array[density_mask],
            densities=density_series[density_mask],
            forecast_days=forecast_days_array,
            harmonic_count=10,
        )

solar_forecast = None
solar_years = np.array([])
f107_monthly_series = np.array(monthly_series['f107'], dtype=float)
f107_mask = np.isfinite(f107_monthly_series) & (f107_monthly_series > 0)
if np.count_nonzero(f107_mask) > 8 and len(forecast_days_array) > 0:
    f107_days = monthly_days_array[f107_mask]
    f107_series = f107_monthly_series[f107_mask]
    solar_years = decimal_year_from_days(start_date.year, f107_days)
    solar_forecast = fit_harmonic_density_forecast(
        days=f107_days,
        densities=f107_series,
        forecast_days=forecast_days_array,
        harmonic_count=10,
    )

# Экспортируем помесячные усредненные точки в CSV
csv_output_path = 'jb2008_monthly_points.csv'
csv_rows_count = len(monthly_dates)
with open(csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['month_start', 'calendar_year', 'density_280_kg_m3', 'temperature_280_K', 'f107_sfu', 'ap_nt'])
    for idx in range(csv_rows_count):
        date_value = monthly_dates[idx].strftime('%Y-%m-%d')
        year_value = monthly_calendar_years[idx]
        density_value = monthly_series['density_280'][idx]
        temperature_value = monthly_series['temperature_280'][idx]
        f107_value = monthly_series['f107'][idx]
        ap_value = monthly_series['ap'][idx]
        writer.writerow([date_value, f'{year_value:.6f}', density_value, temperature_value, f107_value, ap_value])

forecast_csv_output_path = 'jb2008_monthly_fft_forecast.csv'
with open(forecast_csv_output_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        'month_start',
        'calendar_year',
        'density_280_q05_kg_m3',
        'density_280_q50_kg_m3',
        'density_280_q95_kg_m3',
        'f107_fft_forecast_sfu',
    ])
    density_forecast_values = density_forecasts.get(280, {}).get('forecast_density', np.array([]))
    density_forecast_quantiles = density_forecasts.get(280, {}).get('forecast_quantiles', np.empty((0, 0)))
    solar_forecast_values = solar_forecast['forecast_density'] if solar_forecast is not None else np.array([])
    for idx, forecast_date in enumerate(monthly_forecast_times):
        if density_forecast_quantiles.shape[0] >= 3 and idx < density_forecast_quantiles.shape[1]:
            density_q05 = density_forecast_quantiles[0, idx]
            density_q50 = density_forecast_quantiles[1, idx]
            density_q95 = density_forecast_quantiles[2, idx]
        else:
            density_value = density_forecast_values[idx] if idx < len(density_forecast_values) else np.nan
            density_q05 = density_value
            density_q50 = density_value
            density_q95 = density_value
        solar_value = solar_forecast_values[idx] if idx < len(solar_forecast_values) else np.nan
        writer.writerow([
            forecast_date.strftime('%Y-%m-%d'),
            f'{forecast_calendar_years[idx]:.6f}',
            density_q05,
            density_q50,
            density_q95,
            solar_value,
        ])

target_year = 2028.0
density_quantiles_2028 = None
if 280 in density_forecasts:
    density_model = density_forecasts[280]
    q_fitted = np.asarray(density_model.get('fitted_quantiles', np.empty((0, 0))), dtype=float)
    q_forecast = np.asarray(density_model.get('forecast_quantiles', np.empty((0, 0))), dtype=float)
    if q_fitted.shape[0] >= 3:
        x_full = np.concatenate([monthly_calendar_years, forecast_calendar_years])
        q05_full = np.concatenate([q_fitted[0], q_forecast[0] if q_forecast.shape[0] >= 1 else np.array([])])
        q50_full = np.concatenate([q_fitted[1], q_forecast[1] if q_forecast.shape[0] >= 2 else np.array([])])
        q95_full = np.concatenate([q_fitted[2], q_forecast[2] if q_forecast.shape[0] >= 3 else np.array([])])
        if len(x_full) > 1 and x_full[0] <= target_year <= x_full[-1]:
            density_quantiles_2028 = {
                'q05': float(np.interp(target_year, x_full, q05_full)),
                'q50': float(np.interp(target_year, x_full, q50_full)),
                'q95': float(np.interp(target_year, x_full, q95_full)),
            }

# Создаем графики
fig, axes = plt.subplots(2, 2, figsize=(20, 10))
fig.suptitle('Колебания плотности атмосферы по модели JB2008', 
             fontsize=14, fontweight='bold')

# График 1: Плотность для разных высот
ax = axes[0, 0]
for alt_km in altitudes_to_track:
    if len(densities_by_altitude[alt_km]) > 0:
        monthly_density = np.array(monthly_series[f'density_{alt_km}'], dtype=float)
        density_mask = np.isfinite(monthly_density) & (monthly_density > 0)
        ax.semilogy(monthly_calendar_years[density_mask], monthly_density[density_mask],
                    color=colors_alt[alt_km], linewidth=2.0, label=f'JB2008, h = {alt_km} км')

        forecast_model = density_forecasts.get(alt_km)
        if forecast_model is not None:
            q_levels = np.asarray(forecast_model.get('quantile_levels', [0.05, 0.5, 0.95]), dtype=float)
            q_fitted = np.asarray(forecast_model.get('fitted_quantiles', np.empty((0, 0))), dtype=float)
            q_forecast = np.asarray(forecast_model.get('forecast_quantiles', np.empty((0, 0))), dtype=float)

            if q_fitted.shape[0] >= 3 and q_fitted.shape[1] == np.count_nonzero(density_mask):
                ax.semilogy(monthly_calendar_years[density_mask], q_fitted[0],
                            color='tab:green', linewidth=1.4, linestyle=':', label='FFT q0.05')
                ax.semilogy(monthly_calendar_years[density_mask], q_fitted[1],
                            color='tab:orange', linewidth=1.8, linestyle='-.', label='FFT q0.50')
                ax.semilogy(monthly_calendar_years[density_mask], q_fitted[2],
                            color='tab:red', linewidth=1.4, linestyle=':', label='FFT q0.95')

            if len(forecast_calendar_years) > 0 and q_forecast.shape[0] >= 3 and q_forecast.shape[1] == len(forecast_calendar_years):
                ax.semilogy(forecast_calendar_years, q_forecast[0],
                            color='tab:green', linewidth=1.4, linestyle='--')
                ax.semilogy(forecast_calendar_years, q_forecast[1],
                            color='tab:orange', linewidth=2.0, linestyle='--')
                ax.semilogy(forecast_calendar_years, q_forecast[2],
                            color='tab:red', linewidth=1.4, linestyle='--')

ax.set_xlabel('Календарный год', fontsize=11)
ax.set_ylabel('Плотность (кг/м³)', fontsize=11)
ax.set_title('Плотность атмосферы на 280 км', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

# График 2: Температура для разных высот
ax = axes[0, 1]
for alt_km in altitudes_to_track:
    if len(temperatures_by_altitude[alt_km]) > 0:
        monthly_temp = np.array(monthly_series[f'temperature_{alt_km}'], dtype=float)
        temp_mask = np.isfinite(monthly_temp)
        ax.plot(monthly_calendar_years[temp_mask], monthly_temp[temp_mask], 
               color=colors_alt[alt_km], linewidth=2.5, label=f'h = {alt_km} км')
ax.set_xlabel('Календарный год', fontsize=11)
ax.set_ylabel('Температура (K)', fontsize=11)
ax.set_title('Температура атмосферы для разных высот', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# График 3: скрыт (солнечная активность отключена)
axes[1, 0].axis('off')
axes[1, 0].set_title('Солнечная активность скрыта', fontsize=12, fontweight='bold')

# График 4: Ap индекс
if len(monthly_series['ap']) > 0:
    ax = axes[1, 1]
    ap_plot = np.array(monthly_series['ap'], dtype=float)
    ap_mask = np.isfinite(ap_plot)
    ax.plot(monthly_calendar_years[ap_mask], ap_plot[ap_mask], 'b-', linewidth=2.5)
    ax.set_xlabel('Календарный год', fontsize=11)
    ax.set_ylabel('Ap (nT)', fontsize=11)
    ax.set_title('Индекс геомагнитной активности Ap', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

for row_axes in axes:
    for axis_item in row_axes:
        axis_item.xaxis.set_major_locator(mticker.MultipleLocator(2))
        axis_item.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        axis_item.tick_params(axis='x', labelrotation=35)
        axis_item.axvline(x=target_year, color='0.25', linestyle='--', linewidth=1.2)

if density_quantiles_2028 is not None:
    axes[0, 0].text(
        0.02,
        0.03,
        (
            f'5%={density_quantiles_2028["q05"]:.3e}\n'
            f'50%={density_quantiles_2028["q50"]:.3e}\n'
            f'95%={density_quantiles_2028["q95"]:.3e} кг/м³'
        ),
        transform=axes[0, 0].transAxes,
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85, edgecolor='0.5')
    )

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('density_jb2008_timeseries.png', dpi=150, bbox_inches='tight')
print("\n✓ График сохранен: density_jb2008_timeseries.png")
print(f"✓ Точки выгружены в CSV: {csv_output_path}")
print(f"✓ FFT-прогноз выгружен в CSV: {forecast_csv_output_path}")
if density_quantiles_2028 is not None:
    print(
        f"✓ Значения: 5%={density_quantiles_2028['q05']:.3e}, "
        f"50%={density_quantiles_2028['q50']:.3e}, 95%={density_quantiles_2028['q95']:.3e} кг/м³"
    )
else:
    print("⚠ Не удалось вычислить значения 5%/50%/95% для целевого года")
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

        forecast_model = density_forecasts.get(alt_km)
        if forecast_model is not None and len(forecast_days_array) > 0:
            print("  FFT-прогноз по помесячным средним log(ρ): линейный тренд + сумма синусоид")
            print(f"  RMSE аппроксимации в log-масштабе: {forecast_model['rmse_log']:.4f}")
            print(f"  RMSE валидации в log-масштабе: {forecast_model['rmse_val_log']:.4f}")
            print(f"  Число гармоник (автоподбор): {forecast_model['harmonics_used']}")
            periods_str = ", ".join(f"{period:.2f}" for period in forecast_model['periods_years'])
            print(f"  Доминирующие периоды, лет: {periods_str}")
            print(f"  Прогноз на {forecast_end_date:%Y-%m-%d}: {forecast_model['forecast_density'][-1]:.3e} кг/м³")
            print(
                f"  Диапазон прогноза {end_date.year}-{forecast_end_date.year}: "
                f"{np.min(forecast_model['forecast_density']):.3e} .. {np.max(forecast_model['forecast_density']):.3e} кг/м³"
            )

print("\n✓ Анализ завершен!")