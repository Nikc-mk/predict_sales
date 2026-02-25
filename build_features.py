"""
Модуль подготовки данных (feature engineering).

Основные функции:
1. Загрузка CSV файла с продажами
2. Преобразование данных из long-формата в wide-формат
3. Создание календарных признаков
4. Добавление временных лагов
"""

import pandas as pd
import numpy as np


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Загружает исходные данные из CSV файла.
    
    Ожидаемый формат CSV:
        date;category;sales
        2026-02-22;MA002;20420
    
    Args:
        path: Путь к CSV файлу
    
    Returns:
        DataFrame с колонками: date, category, sales
    """
    # Читаем CSV с разделителем ";"
    # utf-8-sig - для корректной обработки BOM-символа
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        header=0  # первая строка - заголовок
    )

    print("Колонки после чтения:", df.columns.tolist())

    # Переименовываем колонки в стандартные имена
    df.columns = ["date", "category", "sales"]

    # Парсим дату в формате ISO8601 (YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["date"])

    # Очистка числового столбца sales:
    # - убираем пробелы (разделитель тысяч)
    # - меняем запятую на точку (десятичный разделитель)
    df["sales"] = (
        df["sales"].astype(str)
        .str.replace(" ", "")
        .str.replace(",", ".")
        .astype(float)
    )

    return df[["date", "category", "sales"]]


def pivot_daily(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """
    Преобразует данные из long-формата (длинного) в wide-формат (широкий).
    
    Long-формат (каждая строка = одна категория за один день):
        date       category sales
        2026-02-22 MA002     100
        2026-02-22 MA004     200
    
    Wide-формат (каждая колонка = категория):
                    MA002  MA004  MA005
        2026-02-22  100    200    150
    
    Args:
        df: DataFrame в long-формате
        categories: Список категорий для включения в результат
    
    Returns:
        DataFrame в wide-формате с датой в качестве индекса
    """
    # Создаём сводную таблицу: даты по строкам, категории по колонкам
    wide = (
        df.pivot_table(
            index="date",           # строки - даты
            columns="category",    # колонки - категории
            values="sales",        # значения - продажи
            aggfunc="sum"          # сумма при дубликатах
        )
        .reindex(columns=categories)  # сохраняем порядок категорий
        .sort_index()                  # сортировка по дате
    )

    # Заполняем NaN нулями (дни без продаж)
    wide = wide.fillna(0)

    # Заполняем пропуски в датах (чтобы не было разрывов)
    full_range = pd.date_range(
        start=wide.index.min(),  # первая дата
        end=wide.index.max(),    # последняя дата
        freq="D"                 # ежедневно
    )
    wide = wide.reindex(full_range, fill_value=0)

    wide.index.name = "date"
    return wide


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт календарные признаки для каждого дня.
    
    Признаки:
        - day: день месяца (1-31)
        - month: месяц (1-12)
        - day_of_week: день недели (0=понедельник, 6=воскресенье)
        - weekday: alias для day_of_week (день недели)
        - year: год
        - month_idx: порядковый номер месяца (год * 12 + месяц) для тренда
        - time_idx: порядковый индекс дня для тренда
        - days_passed: дней прошло в месяце
        - days_left: дней осталось в месяце
        - is_weekend: выходной день (True/False)
        - work_day: рабочий день (1/0)
        - work_days_passed: рабочих дней прошло в месяце
        - work_days_left: рабочих дней осталось в месяце
        - month_sin, month_cos: циклическое кодирование месяца
        - day_sin, day_cos: циклическое кодирование дня
    
    Args:
        df: DataFrame с датами в индексе
    
    Returns:
        DataFrame с календарными признаками
    """
    cal = pd.DataFrame(index=df.index)

    # Базовые календарные признаки
    cal["day"] = cal.index.day              # день месяца
    cal["month"] = cal.index.month          # месяц
    cal["day_of_week"] = cal.index.dayofweek  # день недели (0-6)
    cal["weekday"] = cal["day_of_week"]     # alias для совместимости

    # Временной тренд
    cal["year"] = cal.index.year
    cal["month_idx"] = cal.index.year * 12 + cal.index.month  # порядковый номер месяца
    cal["time_idx"] = np.arange(len(cal))   # порядковый индекс дня

    # Сколько дней прошло/осталось в месяце
    cal["days_passed"] = cal["day"]
    month_days = cal.index.days_in_month    # количество дней в месяце
    cal["days_left"] = month_days - cal["day"]

    # Рабочие дни
    cal["is_weekend"] = cal["day_of_week"] >= 5  # 5=суббота, 6=воскресенье
    cal["work_day"] = (~cal["is_weekend"]).astype(int)  # 1 - рабочий, 0 - выходной

    # Кумулятивное количество рабочих дней в месяце
    cal["work_days_passed"] = cal.groupby(
        [cal.index.year, cal.index.month]
    )["work_day"].cumsum()

    # Всего рабочих дней в месяце
    total_workdays = cal.groupby(
        [cal.index.year, cal.index.month]
    )["work_day"].transform("sum")

    cal["work_days_left"] = total_workdays - cal["work_days_passed"]

    # Циклическое кодирование месяца (sin/cos)
    cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12)
    cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12)

    # Циклическое кодирование дня месяца (sin/cos)
    cal["day_sin"] = np.sin(2 * np.pi * cal["day"] / 31)
    cal["day_cos"] = np.cos(2 * np.pi * cal["day"] / 31)

    # Циклическое кодирование дня недели (sin/cos)
    cal["weekday_sin"] = np.sin(2 * np.pi * cal["day_of_week"] / 7)
    cal["weekday_cos"] = np.cos(2 * np.pi * cal["day_of_week"] / 7)

    return cal


def add_lags(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет временные лаги (предыдущие значения продаж).
    
    Лаги используются как признаки для модели:
    - lag1: продажи за вчера
    - lag7: продажи за прошлую неделю (этот же день недели)
    
    Args:
        wide_df: DataFrame в wide-формате
    
    Returns:
        DataFrame с добавленными колонками лагов
    """
    # Сдвигаем данные на 1 день назад (вчера)
    lag1 = wide_df.shift(1).add_suffix("_lag1")
    
    # Сдвигаем данные на 7 дней назад (прошлая неделя)
    lag7 = wide_df.shift(7).add_suffix("_lag7")

    # Объединяем исходный DataFrame с лагами
    return pd.concat([wide_df, lag1, lag7], axis=1)
