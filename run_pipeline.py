"""
Главный скрипт прогнозирования продаж.

Пайплайн:
    1. Загрузка данных из CSV
    2. Преобразование в wide-формат (пивот)
    3. Создание календарных признаков
    4. Формирование обучающей выборки
    5. Обучение модели TabTransformer
    6. Прогноз на текущий месяц
    7. Валидация на тестовых данных
    8. Визуализация результатов

Запуск:
    python run_pipeline.py
"""

import pickle
import pandas as pd

from config import DATA_PATH, CATEGORIES, USE_ALL_DATA_FOR_TRAINING
from build_features import load_raw_data, pivot_daily, add_calendar_features
from dataset import TabularDataset
from train import train_model
from inference import predict_current_month
from validation import get_test_months, validate_daily
from visualization import plot_validation_results


def main():
    """Основная функция - выполняет полный пайплайн прогнозирования."""
    
    # === ЭТАП 1: Загрузка данных ===
    print("=" * 50)
    print("ЭТАП 1: Загрузка данных")
    print("=" * 50)
    
    raw = load_raw_data(DATA_PATH)
    
    # === ЭТАП 2: Подготовка признаков ===
    print("\n" + "=" * 50)
    print("ЭТАП 2: Подготовка признаков")
    print("=" * 50)
    
    wide = pivot_daily(raw, CATEGORIES)
    calendar = add_calendar_features(wide)
    
    # === ЭТАП 3: Создание обучающей выборки ===
    print("\n" + "=" * 50)
    print("ЭТАП 3: Создание обучающей выборки")
    print("=" * 50)
    
    if USE_ALL_DATA_FOR_TRAINING:
        # Обучаем на всех данных до последней даты включительно
        test_months = None
        print("Режим: обучение на всех данных (без валидации)")
        dataset_builder = TabularDataset(wide, calendar, test_months=None)
    else:
        # Определяем тестовые месяцы ДО обучения, чтобы исключить их из обучающей выборки
        test_months = get_test_months(wide)
        print(f"Тестовые месяца (будут исключены из обучения): {test_months}")
        dataset_builder = TabularDataset(wide, calendar, test_months=test_months)
    
    samples = dataset_builder.build_samples()
    
    print(f"Размер датасета: {samples.shape}")
    samples.to_csv("training_samples.csv", index=False)
    
    # === ЭТАП 4: Обучение модели ===
    print("\n" + "=" * 50)
    print("ЭТАП 4: Обучение модели")
    print("=" * 50)

    model, _ = train_model(samples)

    # === ЭТАП 5: Прогноз на текущий месяц ===
    print("\n" + "=" * 50)
    print("ЭТАП 5: Прогноз на текущий месяц")
    print("=" * 50)
    
    try:
        with open("scaler_X.pkl", "rb") as f:
            scaler_X = pickle.load(f)
        with open("scaler_y.pkl", "rb") as f:
            scaler_y = pickle.load(f)
    except FileNotFoundError:
        scaler_X = None
        scaler_y = None
    
    forecast = predict_current_month(
        model=model,
        wide_df=wide,
        categories_list=CATEGORIES,
        calendar=calendar,
        scaler_X=scaler_X,
        scaler_y=scaler_y
    )
    
    print("\nРезультат прогноза:")
    print(forecast.head(10))
    forecast.to_csv("forecast_output.csv", index=False)
    
    # === ЭТАП 6: Валидация и визуализация ===
    if USE_ALL_DATA_FOR_TRAINING:
        print("\n" + "=" * 50)
        print("ЭТАП 6: Пропуск валидации (режим USE_ALL_DATA_FOR_TRAINING=True)")
        print("=" * 50)
        # Создаём пустой DataFrame для визуализации
        val_df = pd.DataFrame()
    else:
        print("\n" + "=" * 50)
        print("ЭТАП 6: Валидация (3 последних месяца)")
        print("=" * 50)
        
        # test_months уже определён в ЭТАПЕ 3
        print(f"Тестовые месяца: {test_months}")
        
        # Валидация
        val_df = validate_daily(model, wide, calendar, test_months, scaler_X, scaler_y)
        print(f"Записей валидации: {len(val_df)}")
        val_df.to_csv("validation_results.csv", index=False)
    
    # Визуализация (работает и с пустым DataFrame)
    plot_validation_results(val_df)
    
    print("\n" + "=" * 50)
    print("Пайплайн завершён!")
    print("=" * 50)


if __name__ == "__main__":
    main()
