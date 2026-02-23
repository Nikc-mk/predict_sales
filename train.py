"""
Модуль обучения модели.

Функции:
    - Подготовка данных для PyTorch
    - Обучение нейросети
    - Логирование процесса обучения

Алгоритм обучения:
    1. Кодирование категорий (LabelEncoder)
    2. Разделение на числовые и категориальные признаки
    3. Создание DataLoader с батчами
    4. Обучение на нескольких эпохах с использованием Huber Loss
    5. Возврат обученной модели и энкодера категорий
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

from model import TabTransformerModel


def train_model(df, epochs: int = 20, batch_size: int = 256, learning_rate: float = 1e-3):
    """
    Обучает модель TabTransformer на переданных данных.
    
    Args:
        df: DataFrame с признаками и целевой переменной
            Ожидаемые колонки: category, month, day_of_month, и др.
            target: целевая переменная (продажи на остаток месяца)
        epochs: количество эпох обучения (по умолчанию 5)
        batch_size: размер батча (по умолчанию 256)
        learning_rate: скорость обучения (по умолчанию 0.001)
    
    Returns:
        tuple: (обученная модель, энкодер категорий)
    """
    # === 1. Подготовка данных ===
    
    # Преобразуем категории в числовые ID
    # "MA002" -> 0, "MA004" -> 1, и т.д.
    cat_encoder = LabelEncoder()
    df["cat_id"] = cat_encoder.fit_transform(df["category"])

    # Целевая переменная
    y = df["target"].values.astype("float32")

    # Числовые признаки (все, кроме category, target, cat_id, date)
    X_num = df.drop(columns=["category", "target", "cat_id", "date"]).values.astype("float32")
    
    # Категориальные признаки (ID категорий)
    X_cat = df["cat_id"].values.astype("int64")

    print(f"Обучаем на {len(X_num)} сэмплах")
    print(f"Числовых признаков: {X_num.shape[1]}")
    print(f"Категорий: {len(cat_encoder.classes_)}")

    # === 2. Создание DataLoader ===
    
    # TensorDataset: объединяет numpy массивы в один датасет
    dataset = TensorDataset(
        torch.tensor(X_num),   # числовые признаки
        torch.tensor(X_cat),   # ID категорий
        torch.tensor(y),       # целевая переменная
    )

    # DataLoader: итерируется по батчам
    # shuffle=True перемешивает данные каждую эпоху
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === 3. Инициализация модели ===
    
    model = TabTransformerModel(
        num_categories=len(cat_encoder.classes_),  # количество категорий
        num_numeric=X_num.shape[1],                # количество числовых признаков
    )

    # Оптимизатор Adam: обновляет веса модели
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Функция потерь Huber: устойчива к выбросам
    # L1 loss для малых ошибок, L2 loss для больших
    loss_fn = torch.nn.HuberLoss()

    # === 4. Обучение ===
    
    print(f"\nОбучение модели ({epochs} эпох)...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Итерируемся по батчам
        for x_num, x_cat, target in loader:
            # Прямой проход: получаем предсказания
            pred = model(x_num, x_cat).squeeze()

            # Вычисляем ошибку
            loss = loss_fn(pred, target)

            # Обнуляем градиенты (чтобы не накапливались)
            optimizer.zero_grad()
            
            # Обратный проход: вычисляем градиенты
            loss.backward()
            
            # Обновляем веса
            optimizer.step()

            total_loss += loss.item()

        # Выводим среднюю ошибку за эпоху
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: {avg_loss:,.4f}")

    print("Обучение завершено!\n")

    return model, cat_encoder
