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

    def build_samples(self) -> pd.DataFrame:
        """
        Создаёт обучающую выборку.
        
        Для каждого дня месяца (кроме последнего) и каждой категории:
        1. Считаем накопленные продажи с начала месяца (cumulative)
        2. Определяем целевую - продажи на остаток месяца (remaining)
        3. Добавляем лаги и календарные признаки
        
        Returns:
            DataFrame с признаками и целевой переменной
        """
        samples = []

        # Итерируемся по каждой дате в периоде обучения
        for date in self.df.index:
            day = date.day
            
            # Маска для всех дней текущего месяца
            month_mask = (
                (self.df.index.year == date.year)
                & (self.df.index.month == date.month)
            )

            # Для каждой категории создаём сэмпл
            for cat in self.categories:
                # cumulative_sales: продажи с 1-го числа по текущий день
                cumulative = self.df.loc[month_mask & (self.df.index.day <= day), cat].sum()

                # remaining: продажи со следующего дня до конца месяца (целевая переменная)
                remaining = self.df.loc[month_mask & (self.df.index.day > day), cat].sum()

                # Пропускаем последний день месяца (нечего предсказывать)
                if remaining == 0 and day == self.df.loc[month_mask].index.day.max():
                    continue

                # Лаги: продажи за последние 7, 14, 28 дней (из всех доступных данных)
                last7 = self.full_df.loc[:date].tail(7)[cat].sum()
                last14 = self.full_df.loc[:date].tail(14)[cat].sum()
                last28 = self.full_df.loc[:date].tail(28)[cat].sum()

                # Формируем сэмпл
                sample = {
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
                    "target": remaining,  # то, что предсказываем
                }

                samples.append(sample)

        return pd.DataFrame(samples)
