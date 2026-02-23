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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

# Импорт модулей проекта
from build_features import load_raw_data, pivot_daily, add_calendar_features, add_lags
from dataset import TabularDataset
from train import train_model
from inference import predict_current_month


# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Путь к файлу с данными продаж
DATA_PATH = r"/home/nikolay/PycharmProjects/predict_sales/sales_data.csv"

# Список категорий для прогнозирования (фиксированный порядок!)
CATEGORIES = [
    "MA002","MA004","MA005","MA007","MA009","MA010","MA011","MA014","MA015",
    "MA017","MA019","MA020","MA021","MA022","MA025","MA026","MA029","MA030",
    "MA031","MA032","MA033","MA037","MA039","MA040","MA041","MA042","MA043",
    "MA044","MA045","MA046","MA047","MA048","MA049","MA050","MA051","MA052",
    "MA053","MA054","MA055","MA056","MA057","MA058","MA059","MA061","MA062",
    "MA063","MA064","MA065","MA066","MA067","MA068","MA069","MA070","MA071",
    "MA072","MA073","MA076","MA077","MA078"
]


def main():
    """
    Основная функция - выполняет полный пайплайн прогнозирования.
    
    Этапы:
        1. Загрузка и предобработка данных
        2. Создание признаков (features)
        3. Обучение модели
        4. Прогноз на текущий месяц
        5. Валидация и визуализация
    """
    
    # =========================================================================
    # ЭТАП 1: Загрузка и предобработка данных
    # =========================================================================
    
    print("="*50)
    print("ЭТАП 1: Загрузка данных")
    print("="*50)

    # Читаем CSV файл
    # Ожидаемый формат: date;category;sales
    raw = load_raw_data(DATA_PATH)
    
    # =========================================================================
    # ЭТАП 2: Преобразование данных и создание признаков
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 2: Подготовка признаков")
    print("="*50)
    
    # Преобразуем из long-формата в wide-формат
    # Строки = даты, Колонки = категории
    print("Создание pivot-таблицы...")
    wide = pivot_daily(raw, CATEGORIES)
    
    # Добавляем лаги (продажи за вчера и неделю назад)
    print("Добавление лагов...")
    wide = add_lags(wide)
    
    # Создаём календарные признаки
    # (день недели, выходные, рабочие дни и т.д.)
    print("Создание календарных признаков...")
    calendar = add_calendar_features(wide)
    
    # =========================================================================
    # ЭТАП 3: Формирование обучающей выборки
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 3: Создание обучающей выборки")
    print("="*50)
    
    # TabularDataset автоматически:
    # - Использует последние 2 года для обучения
    # - Использует 3-й год для расчёта признаков прошлого года
    # - Создаёт сэмплы: признаки + целевая переменная
    print("Создание обучающего датасета...")
    dataset_builder = TabularDataset(wide, calendar)
    samples = dataset_builder.build_samples()
    
    print(f"Размер датасета: {samples.shape}")
    print(f"  - Записей: {samples.shape[0]:,}")
    print(f"  - Признаков: {samples.shape[1]}")
    
    # =========================================================================
    # ЭТАП 4: Обучение модели
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 4: Обучение модели")
    print("="*50)
    
    # Обучаем TabTransformer на всех данных
    # Возвращаем модель и энкодер категорий
    model, encoder = train_model(samples)
    
    # =========================================================================
    # ЭТАП 5: Прогноз на текущий месяц
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 5: Прогноз на текущий месяц")
    print("="*50)
    
    # Делаем прогноз для каждой категории
    # Модель предсказывает продажи на остаток месяца
    forecast = predict_current_month(
        model=model,
        data_upto_yesterday=wide.reset_index().rename(columns={'index': 'date'}),
        categories_list=CATEGORIES,
        calendar=calendar
    )
    
    print("\nРезультат прогноза:")
    print(forecast.head(10))
    
    # Сохраняем прогноз в CSV
    forecast.to_csv("forecast_output.csv", index=False)
    print(f"\nПрогноз сохранён в forecast_output.csv")
    
    # =========================================================================
    # ЭТАП 6: Валидация и визуализация
    # =========================================================================
    
    print("\n" + "="*50)
    print("ЭТАП 6: Валидация на тестовых данных")
    print("="*50)
    
    # Разделяем данные на train/test
    # Тест = последний полный месяц
    test_date = samples["date"].max()
    test_month_start = test_date.replace(day=1)
    
    test_samples = samples[samples["date"] >= test_month_start].copy()
    train_samples = samples[samples["date"] < test_month_start].copy()

    print(f"Обучающая выборка: {len(train_samples):,} записей")
    print(f"Тестовая выборка: {len(test_samples):,} записей")
    
    # Обучаем модель только на train данных
    model_val, encoder_val = train_model(train_samples)

    # Добавляем cat_id для теста
    test_samples["cat_id"] = encoder_val.transform(test_samples["category"])

    # Предсказания на тесте
    X_num_test = test_samples.drop(columns=["category", "target", "cat_id", "date"]).values.astype("float32")
    X_cat_test = test_samples["cat_id"].values.astype("int64")

    model_val.eval()
    with torch.no_grad():
        X_num_tensor = torch.tensor(X_num_test, dtype=torch.float32)
        X_cat_tensor = torch.tensor(X_cat_test, dtype=torch.int64)
        predictions = model_val(X_num_tensor, X_cat_tensor).squeeze().numpy()

    test_samples["predicted"] = predictions

    # === Метрики качества ===
    
    # MAE (Mean Absolute Error) - средняя абсолютная ошибка
    mae = np.mean(np.abs(test_samples["target"] - test_samples["predicted"]))
    
    # RMSE (Root Mean Square Error) - корень из среднеквадратичной ошибки
    rmse = np.sqrt(np.mean((test_samples["target"] - test_samples["predicted"])**2))
    
    # MAPE (Mean Absolute Percentage Error) - средняя абсолютная ошибка в процентах
    mape = np.mean(
        np.abs((test_samples["target"] - test_samples["predicted"]) / (test_samples["target"] + 1))
    ) * 100
    
    print(f"\nМетрики качества на тесте:")
    print(f"  MAE:  {mae:,.0f} (средняя ошибка в рублях)")
    print(f"  RMSE: {rmse:,.0f} (штраф за большие ошибки)")
    print(f"  MAPE: {mape:.2f}% (ошибка в процентах)")
    
    # === Визуализация ===
    
    print("\nСоздание графиков...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # График 1: Scatter - факт vs предсказание
    # Идеально: все точки на диагонали
    ax1 = axes[0, 0]
    ax1.scatter(test_samples["target"], test_samples["predicted"], alpha=0.3, s=10)
    max_val = max(test_samples["target"].max(), test_samples["predicted"].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', label='Идеально')
    ax1.set_xlabel("Фактические продажи")
    ax1.set_ylabel("Предсказанные продажи")
    ax1.set_title("Факт vs Предсказание (scatter)")
    ax1.legend()

    # График 2: Сравнение по категориям (столбцы)
    ax2 = axes[0, 1]
    category_agg = test_samples.groupby("category").agg({
        "target": "sum",
        "predicted": "sum"
    }).reset_index()
    x = np.arange(len(category_agg))
    width = 0.35
    ax2.bar(x - width/2, category_agg["target"], width, label="Факт", alpha=0.7)
    ax2.bar(x + width/2, category_agg["predicted"], width, label="Прогноз", alpha=0.7)
    ax2.set_xlabel("Категория")
    ax2.set_ylabel("Продажи")
    ax2.set_title("Сравнение по категориям")
    ax2.legend()
    ax2.set_xticks(x[::5])
    ax2.set_xticklabels(category_agg["category"][::5], rotation=45, ha='right')

    # График 3: Динамика по дням (все категории суммарно)
    ax3 = axes[1, 0]
    daily_agg = test_samples.groupby("day_of_month").agg({
        "target": "sum",
        "predicted": "sum"
    }).reset_index()
    ax3.plot(daily_agg["day_of_month"], daily_agg["target"], 'b-o', label="Факт", markersize=4)
    ax3.plot(daily_agg["day_of_month"], daily_agg["predicted"], 'r--s', label="Прогноз", markersize=4)
    ax3.set_xlabel("День месяца")
    ax3.set_ylabel("Продажи")
    ax3.set_title("Динамика по дням (все категории)")
    ax3.legend()

    # График 4: Процент ошибки по категориям
    ax4 = axes[1, 1]
    category_error = category_agg.copy()
    category_error["error"] = category_error["predicted"] - category_agg["target"]
    category_error["error_pct"] = (category_error["error"] / (category_agg["target"] + 1)) * 100
    colors = ['green' if e >= 0 else 'red' for e in category_error["error"]]
    ax4.bar(range(len(category_error)), category_error["error_pct"], color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel("Категория (индекс)")
    ax4.set_ylabel("Ошибка %")
    ax4.set_title("Процент ошибки по категориям\n(зелёный = перепрогноз, красный = недопрогноз)")
    
    plt.tight_layout()
    plt.savefig("validation_plot.png", dpi=150)
    print("Графики сохранены в validation_plot.png")
    plt.show()

    print("\n" + "="*50)
    print("Пайплайн завершён!")
    print("="*50)


if __name__ == "__main__":
    main()
