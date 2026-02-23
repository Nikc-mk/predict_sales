import pandas as pd
import numpy as np

# ---------- Функции для прошлогодних и прошлых месячных данных ----------

def get_last_year_data(cat, day_of_month, wide_df):
    """
    Сумма продаж с 1 по day_of_month прошлого года для категории cat
    wide_df: pivot-таблица с колонкой 'date' и категориями
    """
    if cat not in wide_df.columns:
        return 0

    # Преобразуем дату в прошлый год безопасно
    last_year_dates = wide_df['date'].apply(
        lambda d: pd.Timestamp(year=d.year - 1, month=d.month, day=min(d.day, pd.Timestamp(year=d.year-1, month=d.month, day=1).days_in_month))
    )

    mask = last_year_dates.dt.day <= day_of_month
    return wide_df.loc[mask, cat].sum()


def get_last_month_total(cat, wide_df):
    """
    Сумма продаж за предыдущий месяц для категории cat
    """
    if cat not in wide_df.columns:
        return 0

    first_date = wide_df['date'].min()
    prev_month = first_date.month - 1 if first_date.month > 1 else 12
    prev_year = first_date.year if first_date.month > 1 else first_date.year - 1

    mask = (wide_df['date'].dt.year == prev_year) & (wide_df['date'].dt.month == prev_month)
    return wide_df.loc[mask, cat].sum()


# ---------- Основная функция прогнозирования ----------

def predict_current_month(model, data_upto_yesterday, categories_list, calendar):
    """
    data_upto_yesterday: pivot-таблица (date x category) с колонками категорий
    categories_list: список категорий
    calendar: датафрейм с календарными признаками
    """
    results = []

    # Берем "сегодня" из календаря
    today = pd.Timestamp.now().normalize()

    # Добавим колонку с датой в data_upto_yesterday, если её нет
    if 'date' not in data_upto_yesterday.columns:
        data_upto_yesterday = data_upto_yesterday.reset_index().rename(columns={'index':'date'})

    day_of_month = today.day

    for cat in categories_list:
        cumulative_sales = data_upto_yesterday.loc[data_upto_yesterday['date'].dt.day <= day_of_month, cat].sum() if cat in data_upto_yesterday.columns else 0
        sales_last_7_days = data_upto_yesterday.loc[data_upto_yesterday['date'].dt.day > day_of_month - 7, cat].sum() if cat in data_upto_yesterday.columns else 0
        sales_last_14_days = data_upto_yesterday.loc[data_upto_yesterday['date'].dt.day > day_of_month - 14, cat].sum() if cat in data_upto_yesterday.columns else 0
        sales_last_28_days = data_upto_yesterday.loc[data_upto_yesterday['date'].dt.day > day_of_month - 28, cat].sum() if cat in data_upto_yesterday.columns else 0

        sales_same_month_lastyear_day_1_to_t = get_last_year_data(cat, day_of_month, data_upto_yesterday)
        sales_previous_month_total = get_last_month_total(cat, data_upto_yesterday)

        days_left = calendar.loc[today, "days_left"] if today in calendar.index else 0
        work_days_left = calendar.loc[today, "work_days_left"] if today in calendar.index else 0

        features = {
            'category_id': cat,
            'day_of_month': day_of_month,
            'days_left': days_left,
            'work_days_left': work_days_left,
            'cumulative_sales_day_1_to_t': cumulative_sales,
            'sales_last_7_days': sales_last_7_days,
            'sales_last_14_days': sales_last_14_days,
            'sales_last_28_days': sales_last_28_days,
            'sales_same_month_lastyear_day_1_to_t': sales_same_month_lastyear_day_1_to_t,
            'sales_previous_month_total': sales_previous_month_total
        }

        # Предсказание остатка
        predicted_remaining = model.predict(features)

        total_forecast = cumulative_sales + predicted_remaining

        results.append({
            'category': cat,
            'fact_so_far': cumulative_sales,
            'predicted_remaining': predicted_remaining,
            'total_forecast': total_forecast
        })

    return pd.DataFrame(results)
