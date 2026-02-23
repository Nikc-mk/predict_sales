"""
Модуль создания обучающей выборки (dataset builder).

Назначение:
    Преобразует сырые данные о продажах в формат, пригодный для обучения модели.
    Для каждого дня и категории создаётся набор признаков и целевая переменная.

Логика работы:
    1. Используются данные за 2 плавающих года для обучения
    2. Данные за 3-й год используются для расчёта признаков прошлого года
    3. Для каждого дня месяца (кроме последнего) создаётся сэмпл:
       - Признаки: накопленные продажи, лаги, календарные фичи
       - Целевая: продажи на оставшиеся дни месяца (то, что нужно предсказать)
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta


class TabularDataset:
    """
    Класс для создания табличной обучающей выборки.
    
    Пример использования:
        dataset = TabularDataset(wide_df, calendar)
        samples = dataset.build_samples()
    
    Признаки (features):
        - category: категория товара (категориальный признак)
        - month: месяц (1-12)
        - day_of_month: день месяца (1-31)
        - days_left: дней осталось до конца месяца
        - work_days_left: рабочих дней осталось
        - cumulative_sales: накопленные продажи с начала месяца
        - sales_last_7: продажи за последние 7 дней
        - sales_last_14: продажи за последние 14 дней
        - sales_last_28: продажи за последние 28 дней
        - sales_same_month_lastyear: продажи за тот же месяц прошлого года
        - sales_previous_month_total: продажи за предыдущий месяц
    
    Целевая переменная (target):
        - remaining: продажи на оставшиеся дни месяца
    """
    
    def __init__(self, wide_df: pd.DataFrame, calendar: pd.DataFrame, train_years: int = 2):
        """
        Инициализация датасета.
        
        Args:
            wide_df: DataFrame с продажами (широкий формат, все 3 года)
            calendar: DataFrame с календарными признаками
            train_years: количество лет для обучения (по умолчанию 2)
        """
        self.full_df = wide_df      # все данные (для лагов)
        self.cal = calendar         # календарные признаки
        self.categories = wide_df.columns  # список категорий

        # Определяем период обучения (последние train_years лет)
        max_date = wide_df.index.max()  # последняя дата в данных
        
        # Начало периода обучения - 2 года назад от конца данных
        self.train_start = max_date - pd.DateOffset(years=train_years) + pd.DateOffset(months=1, day=1)
        self.train_start = self.train_start.replace(day=1)  # начало месяца
        
        # Данные для обучения (2 плавающих года)
        self.df = wide_df[wide_df.index >= self.train_start]
        
        # Данные за 3-й год для расчёта признаков прошлого года
        self.history_df = wide_df[wide_df.index < self.train_start]

        print(f"Период обучения: {self.train_start.date()} - {max_date.date()}")
        print(f"История (для лагов): {self.history_df.index.min().date()} - {self.history_df.index.max().date()}")

    def _last_year_progress(self, cat, date) -> float:
        """
        Рассчитывает продажи за аналогичный период прошлого года.
        
        Например, если текущая дата 2025-02-15, то берём данные за 2024-02-01 по 2024-02-15.
        
        Args:
            cat: категория товара
            date: текущая дата
        
        Returns:
            Сумма продаж за аналогичный период прошлого года
        """
        # Дата в прошлом году
        last_year = date - pd.DateOffset(years=1)
        
        # Маска для поиска данных за прошлый год
        # (тот же месяц и дни <= текущего дня)
        mask = (
            (self.history_df.index.year == last_year.year)
            & (self.history_df.index.month == last_year.month)
            & (self.history_df.index.day <= date.day)
        )
        
        if mask.sum() > 0:
            return self.history_df.loc[mask, cat].sum()
        
        # Если нет данных за прошлый год - возвращаем 0
        return 0

    def _previous_month_total(self, cat, date) -> float:
        """
        Рассчитывает общие продажи за предыдущий месяц.
        
        Args:
            cat: категория товара
            date: текущая дата
        
        Returns:
            Сумма продаж за предыдущий месяц
        """
        # Предыдущий месяц
        prev = date - pd.DateOffset(months=1)
        
        mask = (
            (self.history_df.index.year == prev.year)
            & (self.history_df.index.month == prev.month)
        )
        
        if mask.sum() > 0:
            return self.history_df.loc[mask, cat].sum()
        
        return 0

    def _precompute_lags(self):
        """Предварительно вычисляет лаги для всех дат и категорий."""
        print("  Предварительный расчёт лагов...")
        
        # Для каждой категории создаём DataFrame с лагами
        lags_data = {}
        
        for cat in self.categories:
            if cat not in self.full_df.columns:
                continue
            
            series = self.full_df[cat]
            
            # Лаг 7: сумма за 7 дней (включая сегодня)
            lag7 = series.rolling(window=7, min_periods=1).sum().shift(1)
            # Лаг 14: сумма за 14 дней
            lag14 = series.rolling(window=14, min_periods=1).sum().shift(1)
            # Лаг 28: сумма за 28 дней
            lag28 = series.rolling(window=28, min_periods=1).sum().shift(1)
            
            lags_data[cat] = {
                'last7': lag7,
                'last14': lag14,
                'last28': lag28
            }
        
        self._lags_cache = lags_data
        print("  Лаги рассчитаны.")

    def _get_lag(self, cat, date, lag_type):
        """Быстрое получение лага из кэша."""
        if not hasattr(self, '_lags_cache'):
            self._precompute_lags()
        
        if cat not in self._lags_cache:
            return 0
        
        lag_series = self._lags_cache[cat].get(lag_type)
        if lag_series is None or date not in lag_series.index:
            return 0
        
        val = lag_series.loc[date]
        return 0 if pd.isna(val) else val

    def build_samples(self) -> pd.DataFrame:
        """
        Создаёт обучающую выборку (оптимизированная версия с кэшированием).
        
        Returns:
            DataFrame с признаками и целевой переменной
        """
        print("Создание обучающей выборки...")
        
        # Предварительно вычисляем лаги
        self._precompute_lags()
        
        # Копируем данные и добавляем вспомогательные колонки
        df = self.df.copy()
        
        # Создаём уникальный идентификатор месяца для groupby
        df['year_month'] = df.index.to_period('M')
        
        # Кумулятивная сумма продаж внутри каждого месяца
        df_cum = df.groupby('year_month')[list(self.categories)].cumsum()
        
        # Сумма продаж за месяц (для расчёта remaining)
        month_totals = df.groupby('year_month')[list(self.categories)].sum()
        
        # Преобразуем month_totals в словарь для быстрого доступа
        month_totals_dict = month_totals.to_dict('index')
        
        samples_list = []
        
        print("  Генерация сэмплов...")
        
        # Группируем по году-месяцу
        for period_idx, (period, period_data) in enumerate(df.groupby('year_month')):
            if period_idx % 10 == 0:
                print(f"    Период {period_idx+1}...")
            
            dates = period_data.index.tolist()
            
            for date in dates[:-1]:  # исключаем последний день
                day = date.day
                
                for cat in self.categories:
                    # Cumulative: кумулятивная сумма до текущего дня
                    cumulative = df_cum.loc[date, cat] if cat in df_cum.columns else 0
                    
                    # Target: остаток месяца = всего за месяц - cumulative
                    if period in month_totals_dict and cat in month_totals_dict[period]:
                        month_total = month_totals_dict[period][cat]
                        remaining = month_total - cumulative
                    else:
                        remaining = 0
                    
                    if remaining <= 0:
                        continue
                    
                    # Лаги из кэша
                    last7 = self._get_lag(cat, date, 'last7')
                    last14 = self._get_lag(cat, date, 'last14')
                    last28 = self._get_lag(cat, date, 'last28')
                    
                    samples_list.append({
                        "date": date,
                        "category": cat,
                        "month": date.month,
                        "day_of_month": day,
                        "days_left": self.cal.loc[date, "days_left"],
                        "work_days_left": self.cal.loc[date, "work_days_left"],
                        "cumulative_sales": cumulative,
                        "sales_last_7": last7,
                        "sales_last_14": last14,
                        "sales_last_28": last28,
                        "sales_same_month_lastyear": self._last_year_progress(cat, date),
                        "sales_previous_month_total": self._previous_month_total(cat, date),
                        "target": remaining,
                    })
            
        print(f"  Создано сэмплов: {len(samples_list):,}")
        return pd.DataFrame(samples_list)
