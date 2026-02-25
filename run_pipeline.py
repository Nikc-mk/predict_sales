"""
Главный скрипт прогнозирования продаж.

Пайплайн:
    1. Загрузка данных из CSV
    2. Преобразование в wide-формат (пивот)
    3. Создание календарных признаков
    4. Формирование обучающей выборки
    5. Обучение модели TabTransformer
    6. Прогноз на текущий месяц
    7. Валидация на тестовых данных
    8. Визуализация результатов

Запуск:
    python run_pipeline.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Импорт модулей проекта
from build_features import load_raw_data, pivot_daily, add_calendar_features
from dataset import TabularDataset
from train import train_model
from inference import predict_current_month


# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Путь к файлу с данными продаж
DATA_PATH = r"/home/nikolay/PycharmProjects/predict_sales/sales_data.csv"

# Список категорий для прогнозирования (фиксированный порядок!)
CATEGORIES = [
    "MA002","MA004","MA005","MA007","MA009","MA010","MA011","MA014","MA015",
    "MA017","MA019","MA020","MA021","MA022","MA025","MA026","MA029","MA030",
    "MA031","MA032","MA033","MA037","MA039","MA040","MA041","MA042","MA043",
    "MA044","MA045","MA046","MA047","MA048","MA049","MA050","MA051","MA052",
    "MA053","MA054","MA055","MA056","MA057","MA058","MA059","MA061","MA062",
    "MA063","MA064","MA065","MA066","MA067","MA068","MA069","MA070","MA071",
    "MA072","MA073","MA076","MA077","MA078"
]


def main():
    """
    Основная функция - выполняет полный пайплайн прогнозирования.
    
    Этапы:
        1. Загрузка и предобработка данных
        2. Создание признаков (features)
        3. Обучение модели
        4. Прогноз на текущий месяц
        5. Валидация и визуализация
    """
    
    # =========================================================================
    # ЭТАП 1: Загрузка и предобработка данных
    # =========================================================================
    
    print("="*50)
    print("ЭТАП 1: Загрузка данных")
    print("="*50)

    # Читаем CSV файл
    # Ожидаемый формат: date;category;sales
    raw = load_raw_data(DATA_PATH)
    
    # =========================================================================
    # ЭТАП 2: Преобразование данных и создание признаков
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 2: Подготовка признаков")
    print("="*50)
    
    # Преобразуем из long-формата в wide-формат
    # Строки = даты, Колонки = категории
    print("Создание pivot-таблицы...")
    wide = pivot_daily(raw, CATEGORIES)
    
    # Лаги вычисляются в dataset.py при построении выборки
    # (правильные значения для каждой категории)
    
    # Создаём календарные признаки
    # (день недели, выходные, рабочие дни и т.д.)
    print("Создание календарных признаков...")
    calendar = add_calendar_features(wide)
    
    # =========================================================================
    # ЭТАП 3: Формирование обучающей выборки
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 3: Создание обучающей выборки")
    print("="*50)
    
    # TabularDataset автоматически:
    # - Использует последние 2 года для обучения
    # - Использует 3-й год для расчёта признаков прошлого года
    # - Создаёт сэмплы: признаки + целевая переменная
    print("Создание обучающего датасета...")
    dataset_builder = TabularDataset(wide, calendar)
    samples = dataset_builder.build_samples()
    
    print(f"Размер датасета: {samples.shape}")
    print(f"  - Записей: {samples.shape[0]:,}")
    print(f"  - Признаков: {samples.shape[1]}")

    # Добавляем cat_id (числовой ID категории) для соответствия формату модели
    cat_encoder = LabelEncoder()
    samples["cat_id"] = cat_encoder.fit_transform(samples["category"])

    # Сохраняем обучающую выборку с cat_id
    samples.to_csv("training_samples.csv", index=False)
    print("Обучающая выборка сохранена в training_samples.csv")
    
    # =========================================================================
    # ЭТАП 4: Обучение модели
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 4: Обучение модели")
    print("="*50)
    
    # Обучаем TabTransformer на всех данных
    # Возвращаем модель и энкодер категорий
    model, encoder = train_model(samples)
    
    # =========================================================================
    # ЭТАП 5: Прогноз на текущий месяц
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 5: Прогноз на текущий месяц")
    print("="*50)
    
    # Загружаем scalers для нормализации признаков и target
    try:
        with open("scaler_X.pkl", "rb") as f:
            scaler_X = pickle.load(f)
        with open("scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)
    except FileNotFoundError:
        print("Внимание: scaler_X.pkl или scaler_y.pkl не найден, используем без нормализации")
        scaler_X = None
        scaler_y = None
    
    # Делаем прогноз для каждой категории
    # Модель предсказывает продажи на остаток месяца
    forecast = predict_current_month(
        model=model,
        data_upto_yesterday=wide.reset_index().rename(columns={'index': 'date'}),
        categories_list=CATEGORIES,
        calendar=calendar,
        scaler_X=scaler_X,
        scaler_y=scaler_y
    )
    
    print("\nРезультат прогноза:")
    print(forecast.head(10))
    
    # Сохраняем прогноз в CSV
    forecast.to_csv("forecast_output.csv", index=False)
    print(f"\nПрогноз сохранён в forecast_output.csv")
    
    # =========================================================================
    # ЭТАП 6: Валидация и визуализация
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 6: Валидация (3 последних месяца)")
    print("="*50)
    
    # Определяем 3 последних завершенных месяца
    # Последняя дата в данных
    last_date = wide.index.max()
    
    # Находим первый день текущего месяца
    current_month_start = last_date.replace(day=1)
    
    # Проверяем, завершен ли текущий месяц (последний день месяца есть в данных)
    # Если в данных есть данные за последний день месяца - месяц завершен
    current_month_days = current_month_start.days_in_month
    is_current_month_complete = last_date.day >= current_month_days
    
    if is_current_month_complete:
        # Текущий месяц завершен, берём 3 последних месяца включая текущий
        test_months = []
        for i in range(3):
            month = current_month_start - pd.DateOffset(months=i)
            test_months.append((month.year, month.month))
    else:
        # Текущий месяц НЕ завершен, берём 3 месяца ДО него
        test_months = []
        for i in range(1, 4):
            month = current_month_start - pd.DateOffset(months=i)
            test_months.append((month.year, month.month))
    
    print(f"Тестовые месяцы: {test_months}")
    
    # Обучаем модель на всех данных до первого тестового месяца
    first_test_month = pd.Timestamp(year=test_months[0][0], month=test_months[0][1], day=1)
    train_samples = samples[samples["date"] < first_test_month].copy()
    
    print(f"Обучающая выборка: {len(train_samples):,} записей")
    
    # Обучаем модель
    model_val, encoder_val = train_model(train_samples)

    # Теперь для каждого тестового месяца и для каждого дня
    # делаем прогноз и сравниваем с фактом
    
    # Получаем wide данные для расчёта фактических продаж
    wide_reset = wide.reset_index()
    wide_reset.columns = ['date'] + list(wide.columns)
    
    validation_results = []
    
    print("\nВалидация по дням...")
    
    for test_year, test_month in test_months:
        # Даты тестового месяца
        month_start = pd.Timestamp(year=test_year, month=test_month, day=1)
        if test_month == 12:
            month_end = pd.Timestamp(year=test_year+1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            month_end = pd.Timestamp(year=test_year, month=test_month+1, day=1) - pd.Timedelta(days=1)
        
        print(f"  Обработка месяца {test_year}-{test_month:02d}...")
        
        # Фактические продажи за весь месяц (по всем категориям)
        month_mask = (wide_reset['date'] >= month_start) & (wide_reset['date'] <= month_end)
        month_data = wide_reset[month_mask]
        
        # Факт за весь месяц (сумма всех продаж)
        fact_total = month_data[CATEGORIES].sum().sum()
        
        # Для каждого дня в месяце (с 1-го по предпоследний)
        for day in range(1, month_end.day):
            forecast_date = pd.Timestamp(year=test_year, month=test_month, day=day)
            
            # Факт с начала месяца по текущий день (включительно)
            fact_so_far_mask = (wide_reset['date'] >= month_start) & (wide_reset['date'] <= forecast_date)
            fact_so_far = wide_reset[fact_so_far_mask][CATEGORIES].sum().sum()
            
            # Для каждой категории делаем прогноз на остаток месяца
            predicted_remaining_total = 0
            
            for cat in CATEGORIES:
                # Признаки для модели на дату forecast_date
                # cumulative_sales: факт с начала месяца по текущий день
                cat_data = month_data[cat] if cat in month_data.columns else pd.Series([0])
                cum_mask = month_data['date'] <= forecast_date
                cumulative = month_data.loc[cum_mask, cat].sum() if cat in month_data.columns else 0
                
                # Лаги (суммы за периоды)
                last7 = wide.loc[:forecast_date].tail(7)[cat].sum() if cat in wide.columns else 0
                last14 = wide.loc[:forecast_date].tail(14)[cat].sum() if cat in wide.columns else 0
                last28 = wide.loc[:forecast_date].tail(28)[cat].sum() if cat in wide.columns else 0
                
                # lag1, lag7, lag14 - конкретные значения на день
                lag1_date = forecast_date - pd.Timedelta(days=1)
                lag7_date = forecast_date - pd.Timedelta(days=7)
                lag14_date = forecast_date - pd.Timedelta(days=14)
                
                lag1 = wide.loc[lag1_date, cat] if lag1_date in wide.index and cat in wide.columns else 0
                lag7 = wide.loc[lag7_date, cat] if lag7_date in wide.index and cat in wide.columns else 0
                lag14 = wide.loc[lag14_date, cat] if lag14_date in wide.index and cat in wide.columns else 0
                
                # Данные за прошлый год
                last_year_date = forecast_date - pd.DateOffset(years=1)
                last_year_mask = (wide.index.year == last_year_date.year) & \
                                 (wide.index.month == last_year_date.month) & \
                                 (wide.index.day <= last_year_date.day)
                sales_same_month_lastyear = wide.loc[last_year_mask, cat].sum() if cat in wide.columns else 0
                
                # Данные за предыдущий месяц
                prev_month_date = forecast_date - pd.DateOffset(months=1)
                prev_month_mask = (wide.index.year == prev_month_date.year) & \
                                  (wide.index.month == prev_month_date.month)
                sales_previous_month_total = wide.loc[prev_month_mask, cat].sum() if cat in wide.columns else 0
                
                # Календарные признаки
                cal_row = calendar.loc[forecast_date] if forecast_date in calendar.index else None
                if cal_row is not None:
                    days_left = cal_row["days_left"]
                    work_days_left = cal_row["work_days_left"]
                    day_of_week = cal_row["day_of_week"]
                    is_weekend = int(cal_row["is_weekend"])
                    year = cal_row["year"]
                    month_idx = cal_row["month_idx"]
                    month_sin = cal_row["month_sin"]
                    month_cos = cal_row["month_cos"]
                    day_sin = cal_row["day_sin"]
                    day_cos = cal_row["day_cos"]
                    weekday_sin = cal_row["weekday_sin"]
                    weekday_cos = cal_row["weekday_cos"]
                else:
                    days_in_month = forecast_date.days_in_month
                    days_left = days_in_month - day
                    work_days_left = days_left // 2
                    day_of_week = forecast_date.dayofweek
                    is_weekend = int(day_of_week >= 5)
                    year = forecast_date.year
                    month_idx = forecast_date.year * 12 + forecast_date.month
                    month_sin = np.sin(2 * np.pi * forecast_date.month / 12)
                    month_cos = np.cos(2 * np.pi * forecast_date.month / 12)
                    day_sin = np.sin(2 * np.pi * day / 31)
                    day_cos = np.cos(2 * np.pi * day / 31)
                    weekday_sin = np.sin(2 * np.pi * day_of_week / 7)
                    weekday_cos = np.cos(2 * np.pi * day_of_week / 7)
                
                # date_id: порядковый номер дня от начала данных
                first_date = wide_reset['date'].min()
                date_id = (forecast_date - first_date).days
                
                # Подготовка признаков для модели (в том же порядке, что при обучении!)
                x_num = np.array([[
                    date_id,
                    forecast_date.month,
                    day,
                    day_of_week,
                    is_weekend,
                    year,
                    month_idx,
                    month_sin,
                    month_cos,
                    day_sin,
                    day_cos,
                    weekday_sin,
                    weekday_cos,
                    days_left,
                    work_days_left,
                    cumulative,
                    last7,
                    last14,
                    last28,
                    lag1,
                    lag7,
                    lag14,
                    sales_same_month_lastyear,
                    sales_previous_month_total
                ]], dtype=np.float32)
                
                cat_id = CATEGORIES.index(cat)
                x_cat = np.array([cat_id], dtype=np.int64)
                
                # Предсказание
                pred = model_val.predict(x_num, x_cat)
                predicted_remaining_total += max(0, pred)
            
            # Общий прогноз = факт на сегодня + прогноз на остаток
            total_forecast = fact_so_far + predicted_remaining_total
            
            # Ошибка
            error_pct = (total_forecast - fact_total) / (fact_total + 1) * 100
            
            validation_results.append({
                'month': f"{test_year}-{test_month:02d}",
                'forecast_date': forecast_date,
                'day_of_month': day,
                'fact_total': fact_total,
                'fact_so_far': fact_so_far,
                'predicted_remaining': predicted_remaining_total,
                'total_forecast': total_forecast,
                'error_pct': error_pct
            })
    
    val_df = pd.DataFrame(validation_results)
    print(f"Всего записей валидации: {len(val_df)}")
    
    # === Визуализация ===
    
    print("\nСоздание графиков...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # График 1: Ошибка прогноза по дням (все месяцы)
    ax1 = axes[0, 0]
    for month in val_df['month'].unique():
        month_data = val_df[val_df['month'] == month]
        ax1.plot(month_data['day_of_month'], month_data['error_pct'], 
                 marker='o', markersize=3, label=month)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel("День месяца (когда сделан прогноз)")
    ax1.set_ylabel("Ошибка прогноза %")
    ax1.set_title("Зависимость ошибки прогноза от даты\n(положительная = перепрогноз)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Факт vs Прогноз по дням (для каждого месяца отдельно)
    ax2 = axes[0, 1]
    for month in val_df['month'].unique():
        month_data = val_df[val_df['month'] == month].sort_values('day_of_month')
        # fact_total - горизонтальная линия (константа для месяца)
        ax2.axhline(y=month_data['fact_total'].iloc[0], color='blue', alpha=0.3, linestyle='--')
        ax2.plot(month_data['day_of_month'], month_data['total_forecast'], 
                 marker='o', markersize=3, label=f'{month} (прогноз)')
    ax2.set_xlabel("День месяца (когда сделан прогноз)")
    ax2.set_ylabel("Продажи")
    ax2.set_title("Факт за месяц (горизонтальные линии) vs Прогноз на дату + остаток")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # График 3: Абсолютная ошибка по дням
    ax3 = axes[1, 0]
    val_df['abs_error_pct'] = val_df['error_pct'].abs()
    daily_abs_error = val_df.groupby('day_of_month')['abs_error_pct'].mean().reset_index()
    ax3.plot(daily_abs_error['day_of_month'], daily_abs_error['abs_error_pct'], 
             'g-o', markersize=4)
    ax3.set_xlabel("День месяца (когда сделан прогноз)")
    ax3.set_ylabel("Средняя абсолютная ошибка %")
    ax3.set_title("Точность прогноза по дням\n(чем меньше - тем точнее)")
    ax3.grid(True, alpha=0.3)
    
    # График 4: Сравнение по месяцам ( boxplot)
    ax4 = axes[1, 1]
    months = val_df['month'].unique()
    data_for_box = [val_df[val_df['month'] == m]['error_pct'].values for m in months]
    bp = ax4.boxplot(data_for_box, labels=months, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel("Месяц")
    ax4.set_ylabel("Ошибка прогноза %")
    ax4.set_title("Распределение ошибки по месяцам")
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("validation_plot.png", dpi=150)
    print("Графики сохранены в validation_plot.png")
    
    # Сохраняем данные валидации
    val_df.to_csv("validation_results.csv", index=False)
    print("Данные валидации сохранены в validation_results.csv")
    
    plt.show()

    print("\n" + "="*50)
    print("Пайплайн завершён!")
    print("="*50)


if __name__ == "__main__":
    main()
