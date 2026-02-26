"""
Модуль инференса (предсказания).

Назначение:
    Использует обученную модель для прогнозирования продаж на текущий месяц.
    
Логика работы:
    1. Для каждой категории собираем признаки на текущую дату
    2. Вычисляем признаки через feature_utils
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

from feature_utils import build_features_dict, features_to_array
from config import FEATURE_ORDER


# ---------- Основная функция прогнозирования ----------

def predict_current_month(model, wide_df, categories_list, calendar, scaler_X=None, scaler_y=None):
    """
    Прогнозирует продажи на текущий месяц для всех категорий.
    
    Args:
        model: обученная модель TabTransformer
        wide_df: DataFrame с продажами (wide-формат, индекс = дата)
        categories_list: список категорий для прогнозирования
        calendar: DataFrame с календарными признаками
    
    Returns:
        DataFrame с прогнозами для каждой категории
    """
    results = []

    # Сегодняшняя дата
    today = pd.Timestamp.now().normalize()

    for cat in categories_list:
        # Используем общую функцию для построения признаков
        features = build_features_dict(wide_df, calendar, today, cat, categories_list)
        
        # Преобразуем в массив
        x_num = features_to_array(features, FEATURE_ORDER)
        
        # Нормализация
        if scaler_X is not None:
            x_num = scaler_X.transform(x_num)

        # Категория
        cat_id = categories_list.index(cat)
        x_cat = np.array([cat_id], dtype=np.int64)

        # Предсказание
        model.eval()
        with torch.no_grad():
            pred_scaled = model(
                torch.tensor(x_num, dtype=torch.float32),
                torch.tensor(x_cat, dtype=torch.int64)
            ).item()
        
        # Денормализация: expm1 для log1p (или обратно для StandardScaler)
        if scaler_y == "log1p":
            predicted_remaining = np.expm1(pred_scaled)
        elif scaler_y is not None:
            predicted_remaining = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        else:
            predicted_remaining = pred_scaled

        # Факт с начала месяца
        month_start = today.replace(day=1)
        month_mask = (wide_df.index >= month_start) & (wide_df.index <= today)
        cumulative_sales = wide_df.loc[month_mask, cat].sum() if cat in wide_df.columns else 0
        
        # Общий прогноз
        total_forecast = cumulative_sales + predicted_remaining

        results.append({
            'category': cat,
            'fact_so_far': cumulative_sales,
            'predicted_remaining': predicted_remaining,
            'total_forecast': total_forecast
        })

    return pd.DataFrame(results)
