"""
Модуль инференса (предсказания).

Назначение:
    Использует обученную модель для прогнозирования продаж на текущий месяц.
    
Логика работы:
    1. Для каждой категории собираем признаки на текущую дату
    2. Вычисляем:
       - cumulative_sales: накопленные продажи с начала месяца
       - лаги: продажи за последние 7, 14, 28 дней
       - данные за прошлый год и предыдущий месяц
    3. Предсказываем остаток месяца (продажи на оставшиеся дни)
    4. Суммируем: факт + прогноз = общий прогноз на месяц

Выходные данные:
    - fact_so_far: фактические продажи с начала месяца
    - predicted_remaining: предсказанные продажи на остаток месяца
    - total_forecast: общий прогноз на весь месяц
"""

import pandas as pd
import numpy as np
import torch
import pickle


# ---------- Вспомогательные функции ----------

def get_last_year_data(cat, day_of_month, wide_df):
    """
    Получает сумму продаж за аналогичный период прошлого года.
    
    Например, если сегодня 15-е число, получаем продажи с 1 по 15 прошлого года.
    
    Args:
        cat: категория товара
        day_of_month: текущий день месяца
        wide_df: DataFrame с продажами (широкий формат)
    
    Returns:
        Сумма продаж за период прошлого года
    """
    if cat not in wide_df.columns:
        return 0

    # Преобразуем даты в прошлый год
    # (корректируем день, т.к. в прошлом году может быть другое количество дней)
    last_year_dates = wide_df['date'].apply(
        lambda d: pd.Timestamp(
            year=d.year - 1,
            month=d.month,
            day=min(d.day, pd.Timestamp(year=d.year-1, month=d.month, day=1).days_in_month)
        )
    )

    # Фильтруем: только дни <= текущего дня
    mask = last_year_dates.dt.day <= day_of_month
    return wide_df.loc[mask, cat].sum()


def get_last_month_total(cat, wide_df):
    """
    Получает сумму продаж за полный предыдущий месяц.
    
    Args:
        cat: категория товара
        wide_df: DataFrame с продажами
    
    Returns:
        Сумма продаж за предыдущий месяц
    """
    if cat not in wide_df.columns:
        return 0

    # Определяем предыдущий месяц
    first_date = wide_df['date'].min()
    prev_month = first_date.month - 1 if first_date.month > 1 else 12
    prev_year = first_date.year if first_date.month > 1 else first_date.year - 1

    # Фильтруем данные за предыдущий месяц
    mask = (wide_df['date'].dt.year == prev_year) & (wide_df['date'].dt.month == prev_month)
    return wide_df.loc[mask, cat].sum()


# ---------- Основная функция прогнозирования ----------

def predict_current_month(model, data_upto_yesterday, categories_list, calendar, scaler_X=None, scaler_y=None):
    """
    Прогнозирует продажи на текущий месяц для всех категорий.
    
    Алгоритм:
        1. Для каждой категории вычисляем признаки на текущую дату
        2. Предсказываем остаток месяца (продажи на оставшиеся дни)
        3. Формируем результат: факт + прогноз
    
    Args:
        model: обученная модель TabTransformer
        data_upto_yesterday: DataFrame с продажами по вчерашний день
        categories_list: список категорий для прогнозирования
        calendar: DataFrame с календарными признаками
    
    Returns:
        DataFrame с прогнозами для каждой категории
        Колонки: category, fact_so_far, predicted_remaining, total_forecast
    """
    results = []

    # Сегодняшняя дата
    # TODO: в реальном приложении использовать дату, на которую делаем прогноз
    today = pd.Timestamp.now().normalize()

    # Добавляем колонку с датой, если её нет
    if 'date' not in data_upto_yesterday.columns:
        data_upto_yesterday = data_upto_yesterday.reset_index().rename(columns={'index': 'date'})

    day_of_month = today.day

    # Для каждой категории делаем прогноз
    for cat in categories_list:
        # === Вычисление признаков ===
        
        # Проверяем наличие категории в данных
        has_cat = cat in data_upto_yesterday.columns
        
        # cumulative_sales: накопленные продажи с начала месяца по сегодня
        cumulative_sales = (
            data_upto_yesterday.loc[data_upto_yesterday['date'].dt.day <= day_of_month, cat].sum()
            if has_cat else 0
        )
        
        # Лаги: продажи за последние 7, 14, 28 дней (суммы)
        sales_last_7_days = (
            data_upto_yesterday.loc[data_upto_yesterday['date'].dt.day > day_of_month - 7, cat].sum()
            if has_cat else 0
        )
        sales_last_14_days = (
            data_upto_yesterday.loc[data_upto_yesterday['date'].dt.day > day_of_month - 14, cat].sum()
            if has_cat else 0
        )
        sales_last_28_days = (
            data_upto_yesterday.loc[data_upto_yesterday['date'].dt.day > day_of_month - 28, cat].sum()
            if has_cat else 0
        )

        # Конкретные значения на день: lag1, lag7, lag14
        lag1 = data_upto_yesterday.loc[data_upto_yesterday['date'] == today - pd.Timedelta(days=1), cat].sum() if has_cat else 0
        lag7 = data_upto_yesterday.loc[data_upto_yesterday['date'] == today - pd.Timedelta(days=7), cat].sum() if has_cat else 0
        lag14 = data_upto_yesterday.loc[data_upto_yesterday['date'] == today - pd.Timedelta(days=14), cat].sum() if has_cat else 0

        # Данные за прошлый год и предыдущий месяц
        sales_same_month_lastyear_day_1_to_t = get_last_year_data(cat, day_of_month, data_upto_yesterday)
        sales_previous_month_total = get_last_month_total(cat, data_upto_yesterday)

        # Календарные признаки
        cal_today = calendar.loc[today] if today in calendar.index else None
        if cal_today is not None:
            days_left = cal_today["days_left"]
            work_days_left = cal_today["work_days_left"]
            day_of_week = cal_today["day_of_week"]
            is_weekend = int(cal_today["is_weekend"])
            year = cal_today["year"]
            month_idx = cal_today["month_idx"]
            month_sin = cal_today["month_sin"]
            month_cos = cal_today["month_cos"]
            day_sin = cal_today["day_sin"]
            day_cos = cal_today["day_cos"]
            weekday_sin = cal_today["weekday_sin"]
            weekday_cos = cal_today["weekday_cos"]
        else:
            days_left = 0
            work_days_left = 0
            day_of_week = today.dayofweek
            is_weekend = int(today.dayofweek >= 5)
            year = today.year
            month_idx = today.year * 12 + today.month
            month_sin = np.sin(2 * np.pi * today.month / 12)
            month_cos = np.cos(2 * np.pi * today.month / 12)
            day_sin = np.sin(2 * np.pi * day_of_month / 31)
            day_cos = np.cos(2 * np.pi * day_of_month / 31)
            weekday_sin = np.sin(2 * np.pi * day_of_week / 7)
            weekday_cos = np.cos(2 * np.pi * day_of_week / 7)

        # date_id: порядковый номер дня от начала данных
        first_date = data_upto_yesterday['date'].min()
        date_id = (today - first_date).days

        # Словарь признаков (в том же порядке, что при обучении!)
        features = {
            'date_id': date_id,
            'month': today.month,
            'day_of_month': day_of_month,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'year': year,
            'month_idx': month_idx,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'weekday_sin': weekday_sin,
            'weekday_cos': weekday_cos,
            'days_left': days_left,
            'work_days_left': work_days_left,
            'cumulative_sales': cumulative_sales,
            'sales_last_7': sales_last_7_days,
            'sales_last_14': sales_last_14_days,
            'sales_last_28': sales_last_28_days,
            'lag1': lag1,
            'lag7': lag7,
            'lag14': lag14,
            'sales_same_month_lastyear': sales_same_month_lastyear_day_1_to_t,
            'sales_previous_month_total': sales_previous_month_total
        }

        # === Подготовка данных для модели ===
        
        # Числовые признаки (в том же порядке, что при обучении!)
        x_num = np.array([[
            features['date_id'],
            features['month'],
            features['day_of_month'],
            features['day_of_week'],
            features['is_weekend'],
            features['year'],
            features['month_idx'],
            features['month_sin'],
            features['month_cos'],
            features['day_sin'],
            features['day_cos'],
            features['weekday_sin'],
            features['weekday_cos'],
            features['days_left'],
            features['work_days_left'],
            features['cumulative_sales'],
            features['sales_last_7'],
            features['sales_last_14'],
            features['sales_last_28'],
            features['lag1'],
            features['lag7'],
            features['lag14'],
            features['sales_same_month_lastyear'],
            features['sales_previous_month_total']
        ]], dtype=np.float32)

        # Нормализация признаков (если scaler передан)
        if scaler_X is not None:
            x_num = scaler_X.transform(x_num)

        # Категория: преобразуем название в ID
        cat_id = categories_list.index(cat)
        x_cat = np.array([cat_id], dtype=np.int64)

        # === Предсказание ===
        
        # Предсказываем остаток месяца (продажи на оставшиеся дни)
        # Модель возвращает нормализованное предсказание
        x_num_tensor = torch.tensor(x_num, dtype=torch.float32)
        x_cat_tensor = torch.tensor(x_cat, dtype=torch.int64)
        
        model.eval()
        with torch.no_grad():
            pred_scaled = model(x_num_tensor, x_cat_tensor).item()
        
        # Денормализуем предсказание
        if scaler_y is not None:
            predicted_remaining = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        else:
            predicted_remaining = pred_scaled

        # Общий прогноз = факт (уже продано) + прогноз (остаток)
        total_forecast = cumulative_sales + predicted_remaining

        # Сохраняем результат
        results.append({
            'category': cat,
            'fact_so_far': cumulative_sales,           # Факт с начала месяца
            'predicted_remaining': predicted_remaining, # Прогноз на остаток
            'total_forecast': total_forecast            # Общий прогноз
        })

    return pd.DataFrame(results)
