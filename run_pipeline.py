import pandas as pd
from datetime import datetime

from build_features import load_raw_data, pivot_daily, add_calendar_features, add_lags
from dataset import TabularDataset
from train import train_model
from inference import predict_current_month

# путь к данным
DATA_PATH = r"C:\Users\matveyuk.n\VS_Project\Признаки_ПФМ\прогноз оборот\df_data.csv"

# список категорий (фиксированный)
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
    print("Загрузка данных...")
    raw = load_raw_data(DATA_PATH)  # read CSV с заголовком

    print("Pivot таблица...")
    wide = pivot_daily(raw, CATEGORIES)  # long → wide
    wide = add_lags(wide)                # добавляем лаги

    print("Календарные признаки...")
    calendar = add_calendar_features(wide)  # добавляем calendar features

    print("Создание обучающего датасета...")
    dataset_builder = TabularDataset(wide, calendar)
    samples = dataset_builder.build_samples()

    print("Размер датасета:", samples.shape)

    print("Обучение модели...")
    model, encoder = train_model(samples)

    print("Прогноз текущего месяца...")

    # прогноз на текущий день
    forecast = predict_current_month(
        model=model,
        data_upto_yesterday=wide.reset_index().rename(columns={'index':'date'}),  # wide с колонкой date
        categories_list=CATEGORIES,
        calendar=calendar
    )

    print("\nРезультат прогноза:")
    print(forecast.head())

    # сохраняем результат
    forecast.to_csv("forecast_output.csv", index=False)
    print("\nФайл forecast_output.csv сохранен.")


if __name__ == "__main__":
    main()
