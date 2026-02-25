"""
Утилиты для построения признаков.

Используется в inference.py и validation.py для создания признаков
в том же формате, что и при обучении в dataset.py.
"""

import numpy as np
import pandas as pd


def build_features_dict(wide_df: pd.DataFrame, calendar: pd.DataFrame, 
                        forecast_date: pd.Timestamp, cat: str, 
                        categories_list: list) -> dict:
    """
    Строит словарь признаков для одной категории на заданную дату.
    
    Признаки соответствуют формату обучения в TabularDataset.
    
    Args:
        wide_df: DataFrame с продажами (wide-формат, индекс = дата)
        calendar: DataFrame с календарными признаками
        forecast_date: дата, на которую делаем прогноз
        cat: категория товара
        categories_list: список всех категорий (для ID)
    
    Returns:
        Словарь признаков в порядке FEATURE_ORDER
    """
    day = forecast_date.day
    month_start = forecast_date.replace(day=1)
    
    # === Кумулятивные продажи с начала месяца ===
    month_mask = (wide_df.index >= month_start) & (wide_df.index <= forecast_date)
    cumulative = wide_df.loc[month_mask, cat].sum() if cat in wide_df.columns else 0
    
    # === Лаги (суммы за периоды) ===
    last7 = wide_df.loc[:forecast_date].tail(7)[cat].sum() if cat in wide_df.columns else 0
    last14 = wide_df.loc[:forecast_date].tail(14)[cat].sum() if cat in wide_df.columns else 0
    last28 = wide_df.loc[:forecast_date].tail(28)[cat].sum() if cat in wide_df.columns else 0
    
    # Конкретные значения на день
    lag1_date = forecast_date - pd.Timedelta(days=1)
    lag7_date = forecast_date - pd.Timedelta(days=7)
    lag14_date = forecast_date - pd.Timedelta(days=14)
    
    lag1 = wide_df.loc[lag1_date, cat] if lag1_date in wide_df.index and cat in wide_df.columns else 0
    lag7 = wide_df.loc[lag7_date, cat] if lag7_date in wide_df.index and cat in wide_df.columns else 0
    lag14 = wide_df.loc[lag14_date, cat] if lag14_date in wide_df.index and cat in wide_df.columns else 0
    
    # === Данные за прошлый год ===
    last_year = forecast_date - pd.DateOffset(years=1)
    last_year_mask = (
        (wide_df.index.year == last_year.year) & 
        (wide_df.index.month == last_year.month) & 
        (wide_df.index.day <= last_year.day)
    )
    sales_same_month_lastyear = wide_df.loc[last_year_mask, cat].sum() if cat in wide_df.columns else 0
    
    # === Данные за предыдущий месяц ===
    prev_month = forecast_date - pd.DateOffset(months=1)
    prev_month_mask = (wide_df.index.year == prev_month.year) & (wide_df.index.month == prev_month.month)
    sales_previous_month_total = wide_df.loc[prev_month_mask, cat].sum() if cat in wide_df.columns else 0
    
    # === Календарные признаки ===
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
    
    # === date_id ===
    first_date = wide_df.index.min()
    date_id = (forecast_date - first_date).days
    
    return {
        'date_id': date_id,
        'month': forecast_date.month,
        'day_of_month': day,
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
        'cumulative_sales': cumulative,
        'sales_last_7': last7,
        'sales_last_14': last14,
        'sales_last_28': last28,
        'lag1': lag1,
        'lag7': lag7,
        'lag14': lag14,
        'sales_same_month_lastyear': sales_same_month_lastyear,
        'sales_previous_month_total': sales_previous_month_total
    }


def features_to_array(features: dict, feature_order: list) -> np.ndarray:
    """
    Преобразует словарь признаков в numpy массив.
    
    Args:
        features: словарь признаков
        feature_order: порядок признаков (из config.py)
    
    Returns:
        numpy array формы (1, num_features)
    """
    return np.array([[features[k] for k in feature_order]], dtype=np.float32)
