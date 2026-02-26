"""
Конфигурация проекта.

Содержит константы и настройки, используемые во всех модулях.
"""

# Путь к файлу с данными продаж
DATA_PATH = r"/home/nikolay/PycharmProjects/predict_sales/sales_data.csv"

# Список категорий для прогнозирования (фиксированный порядок!)
CATEGORIES = [
    "MA002", "MA004", "MA005", "MA007", "MA009", "MA010", "MA011", "MA014", "MA015",
    "MA017", "MA019", "MA020", "MA021", "MA022", "MA025", "MA026", "MA029", "MA030",
    "MA031", "MA032", "MA033", "MA037", "MA039", "MA040", "MA041", "MA042", "MA043",
    "MA044", "MA045", "MA046", "MA047", "MA048", "MA049", "MA050", "MA051", "MA052",
    "MA053", "MA054", "MA055", "MA056", "MA057", "MA058", "MA059", "MA061", "MA062",
    "MA063", "MA064", "MA065", "MA066", "MA067", "MA068", "MA069", "MA070", "MA071",
    "MA072", "MA073", "MA076", "MA077", "MA078"
]

# Порядок признаков для модели (должен совпадать с обучением)
FEATURE_ORDER = [
    'date_id', 'month', 'day_of_month', 'day_of_week', 'is_weekend',
    'year', 'month_idx', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
    'weekday_sin', 'weekday_cos', 'days_left', 'work_days_left',
    'cumulative_sales', 'sales_last_7', 'sales_last_14', 'sales_last_28',
    'lag1', 'lag7', 'lag14', 'sales_same_month_lastyear', 'sales_previous_month_total'
]

# Режим обучения:
# True - обучать на всех данных до последней даты включительно (без валидации)
# False - оставлять 3 последних завершенных месяца для валидации
USE_ALL_DATA_FOR_TRAINING = False
