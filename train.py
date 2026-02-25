"""
Модуль обучения модели.

Функции:
    - Подготовка данных для PyTorch
    - Обучение нейросети
    - Логирование процесса обучения

Алгоритм обучения:
    1. Кодирование категорий (LabelEncoder)
    2. Нормализация числовых признаков и target
    3. Разделение на числовые и категориальные признаки
    4. Создание DataLoader с батчами
    5. Обучение на нескольких эпохах с использованием MSELoss
    6. Возврат обученной модели и энкодера категорий
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pickle

from model import TabTransformerModel


def train_model(df, epochs: int = 20, batch_size: int = 512, learning_rate: float = 1e-3):
    """
    Обучает модель TabTransformer на переданных данных.
    
    Args:
        df: DataFrame с признаками и целевой переменной
            Ожидаемые колонки: category, month, day_of_month, и др.
            target: целевая переменная (продажи на остаток месяца)
        epochs: количество эпох обучения (по умолчанию 20)
        batch_size: размер батча (по умолчанию 512)
        learning_rate: скорость обучения (по умолчанию 0.001)
    
    Returns:
        tuple: (обученная модель, энкодер категорий)
    """
    # === 1. Подготовка данных ===
    
    # Преобразуем категории в числовые ID
    cat_encoder = LabelEncoder()
    df["cat_id"] = cat_encoder.fit_transform(df["category"])

    # Целевая переменная
    y = df["target"].values.astype("float32")

    # Числовые признаки (все, кроме category, target, cat_id, date)
    X_num = df.drop(columns=["category", "target", "cat_id", "date"]).values.astype("float32")
    
    # Категориальные признаки (ID категорий)
    X_cat = df["cat_id"].values.astype("int64")

    # Нормализация числовых признаков
    scaler_X = StandardScaler()
    X_num = scaler_X.fit_transform(X_num)

    # Нормализация target (критически важно для обучения!)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).squeeze()

    # Сохраняем scalers для инференса
    with open("scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open("scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    print(f"Обучаем на {len(X_num)} сэмплах")
    print(f"Числовых признаков: {X_num.shape[1]}")
    print(f"Категорий: {len(cat_encoder.classes_)}")

    # === 2. Создание DataLoader ===
    
    dataset = TensorDataset(
        torch.tensor(X_num),
        torch.tensor(X_cat),
        torch.tensor(y),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === 3. Инициализация модели ===
    
    model = TabTransformerModel(
        num_categories=len(cat_encoder.classes_),
        num_numeric=X_num.shape[1],
        hidden_dim=128,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # === 4. Обучение ===
    
    print(f"\nОбучение модели ({epochs} эпох)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_num, x_cat, target in loader:
            pred = model(x_num, x_cat).squeeze()
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

    print("Обучение завершено!\n")

    # Сохраняем модель
    torch.save(model.state_dict(), "model.pt")
    print("Модель сохранена в model.pt\n")

    return model, cat_encoder
