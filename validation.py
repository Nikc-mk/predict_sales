"""
Модуль валидации модели.

Функции:
    - Определение тестовых периодов
    - Валидация по дням (прогноз на каждый день)
    - Сбор метрик
"""

import pandas as pd
import numpy as np
import torch
import pickle

from config import CATEGORIES, FEATURE_ORDER
from feature_utils import build_features_dict, features_to_array


def get_test_months(wide_df: pd.DataFrame) -> list:
    """
    Определяет 3 последних завершенных месяца для тестирования.
    
    Args:
        wide_df: DataFrame с продажами
    
    Returns:
        Список кортежей (год, месяц)
    """
    last_date = wide_df.index.max()
    current_month_start = last_date.replace(day=1)
    
    # Проверяем, завершён ли текущий месяц
    current_month_days = current_month_start.days_in_month
    is_current_month_complete = last_date.day >= current_month_days
    
    test_months = []
    start_offset = 0 if is_current_month_complete else 1
    
    for i in range(start_offset, start_offset + 3):
        month = current_month_start - pd.DateOffset(months=i)
        test_months.append((month.year, month.month))
    
    # Сортируем от раннего к позднему (для корректного исключения из обучения)
    test_months.sort()
    
    return test_months


def validate_daily(model, wide_df: pd.DataFrame, calendar: pd.DataFrame,
                   test_months: list, scaler_X=None, scaler_y=None) -> pd.DataFrame:
    """
    Валидация: для каждого дня делаем прогноз и сравниваем с фактом.
    
    Args:
        model: обученная модель
        wide_df: DataFrame с продажами (wide-формат)
        calendar: DataFrame с календарными признаками
        test_months: список (год, месяц) для тестирования
    
    Returns:
        DataFrame с результатами валидации
    """
    wide_reset = wide_df.reset_index()
    if 'index' in wide_reset.columns:
        wide_reset = wide_reset.rename(columns={'index': 'date'})
    
    validation_results = []
    
    print("Валидация по дням...")
    
    for test_year, test_month in test_months:
        month_start = pd.Timestamp(year=test_year, month=test_month, day=1)
        if test_month == 12:
            month_end = pd.Timestamp(year=test_year + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            month_end = pd.Timestamp(year=test_year, month=test_month + 1, day=1) - pd.Timedelta(days=1)
        
        print(f"  Месяц {test_year}-{test_month:02d}...")
        
        # Фактические продажи за весь месяц
        month_mask = (wide_reset['date'] >= month_start) & (wide_reset['date'] <= month_end)
        month_data = wide_reset[month_mask]
        fact_total = month_data[CATEGORIES].sum().sum()
        
        # Для каждого дня (кроме последнего)
        for day in range(1, month_end.day):
            forecast_date = pd.Timestamp(year=test_year, month=test_month, day=day)
            
            # Факт с начала месяца по текущий день
            fact_so_far_mask = (wide_reset['date'] >= month_start) & (wide_reset['date'] <= forecast_date)
            fact_so_far = wide_reset[fact_so_far_mask][CATEGORIES].sum().sum()
            
            # Прогноз на остаток для каждой категории
            predicted_remaining_total = 0
            
            for cat in CATEGORIES:
                features = build_features_dict(wide_df, calendar, forecast_date, cat, CATEGORIES)
                x_num = features_to_array(features, FEATURE_ORDER)
                cat_id = CATEGORIES.index(cat)
                x_cat = np.array([cat_id], dtype=np.int64)
                
                # model.predict() ожидает НЕнормализованные данные
                # НО модель обучалась на нормализованных данных!
                # Поэтому нужно нормализовать вход и денормализовать выход
                
                # Нормализация входа (как при обучении!)
                if scaler_X is not None:
                    x_num_for_pred = scaler_X.transform(x_num)
                else:
                    x_num_for_pred = x_num
                
                # Предсказание
                pred_scaled = model.predict(x_num_for_pred, x_cat)
                
                # Денормализация
                if scaler_y is not None:
                    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
                else:
                    pred = pred_scaled
                
                predicted_remaining_total += max(0, pred)
            
            # Фактический остаток (продажи за оставшиеся дни)
            fact_remaining = fact_total - fact_so_far
            
            # Ошибка предсказания ОСТАТКА (это истинное качество модели)
            error_remaining_pct = (predicted_remaining_total - fact_remaining) / (fact_remaining + 1) * 100
            
            # Общий прогноз (для информации)
            total_forecast = fact_so_far + predicted_remaining_total
            
            validation_results.append({
                'month': f"{test_year}-{test_month:02d}",
                'forecast_date': forecast_date,
                'day_of_month': day,
                'days_left': month_end.day - day,  # Сколько дней осталось предсказать
                'fact_total': fact_total,
                'fact_so_far': fact_so_far,
                'fact_remaining': fact_remaining,
                'predicted_remaining': predicted_remaining_total,
                'total_forecast': total_forecast,
                'error_pct': error_remaining_pct  # Ошибка именно предсказания остатка!
            })
    
    return pd.DataFrame(validation_results)
