import pandas as pd
import numpy as np


def load_raw_data(path: str):
    # read CSV с заголовком
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        header=0  # берём первую строку как заголовок
    )

    print("Колонки после чтения:", df.columns.tolist())

    # присваиваем корректные имена
    df.columns = ["date", "category", "sales"]

    # преобразуем дату (ISO8601 - YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["date"])

    # очистка числового столбца
    df["sales"] = (
        df["sales"].astype(str)
        .str.replace(" ", "")
        .str.replace(",", ".")
        .astype(float)
    )

    return df[["date", "category", "sales"]]




def pivot_daily(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """
    long → wide
    """
    wide = (
        df.pivot_table(index="date", columns="category", values="sales", aggfunc="sum")
        .reindex(columns=categories)
        .sort_index()
    )

    wide = wide.fillna(0)

    # непрерывная дата
    full_range = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full_range, fill_value=0)

    wide.index.name = "date"
    return wide


# ---------- CALENDAR FEATURES ----------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    cal = pd.DataFrame(index=df.index)

    cal["day"] = cal.index.day
    cal["month"] = cal.index.month
    cal["day_of_week"] = cal.index.dayofweek

    cal["days_passed"] = cal["day"]

    month_days = cal.index.days_in_month
    cal["days_left"] = month_days - cal["day"]

    # рабочие дни
    cal["is_weekend"] = cal["day_of_week"] >= 5
    cal["work_day"] = (~cal["is_weekend"]).astype(int)

    cal["work_days_passed"] = cal.groupby(
        [cal.index.year, cal.index.month]
    )["work_day"].cumsum()

    total_workdays = cal.groupby(
        [cal.index.year, cal.index.month]
    )["work_day"].transform("sum")

    cal["work_days_left"] = total_workdays - cal["work_days_passed"]

    return cal


# ---------- LAGS ----------

def add_lags(wide_df: pd.DataFrame):
    lag1 = wide_df.shift(1).add_suffix("_lag1")
    lag7 = wide_df.shift(7).add_suffix("_lag7")

    return pd.concat([wide_df, lag1, lag7], axis=1)
