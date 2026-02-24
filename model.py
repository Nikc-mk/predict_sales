"""
Модуль нейросетевой модели (TabTransformer).

Архитектура:
    Модель основана на Transformer Encoder и предназначена для работы с 
    табличными данными, содержащими числовые и категориальные признаки.

Структура:
    1. Embedding для категориального признака (категория товара)
    2. Объединение числовых признаков и эмбеддинга
    3. Transformer Encoder для изучения взаимосвязей
    4. Регрессор для предсказания продаж
"""

import torch
import torch.nn as nn


class TabTransformerModel(nn.Module):
    """
    Нейросетевая модель для прогнозирования продаж.
    
    Архитектура:
        - Embedding слой для категорий (преобразует ID категории в вектор)
        - Линейная проекция для объединения числовых признаков и эмбеддинга
        - Transformer Encoder ( self-attention механизм)
        - Регрессор (MLP для финального предсказания)
    
    Args:
        num_categories: количество уникальных категорий
        num_numeric: количество числовых признаков
        d_model: размерность модели (по умолчанию 64)
        n_heads: количество attention голов (по умолчанию 4)
        n_layers: количество слоёв энкодера (по умолчанию 7)
        emb_dim: размерность эмбеддинга категории (по умолчанию 32)
    """
    
    def __init__(
        self,
        num_categories,
        num_numeric,
        d_model=64,
        n_heads=4,
        n_layers=7,
        emb_dim=32,
    ):
        super().__init__()

        # Эмбеддинг для категорий: преобразует ID категории в вектор размерности emb_dim
        # Например: категория 0 -> вектор [0.1, -0.3, ...] размерности 32
        self.category_emb = nn.Embedding(num_categories, emb_dim)

        # Проекция: объединяем числовые признаки + эмбеддинг категории -> d_model
        self.input_proj = nn.Linear(num_numeric + emb_dim, d_model)

        # Transformer Encoder слой
        # Использует self-attention для захвата взаимосвязей между признаками
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=128,
            dropout=0.1,
        )

        # Несколько слоёв энкодера
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Регрессор: преобразует выход энкодера в предсказание продаж
        # MLP: d_model -> d_model/2 -> d_model/4 -> 1
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x_num, x_cat):
        """
        Прямой проход модели.
        
        Args:
            x_num: тензор числовых признаков, форма (batch_size, num_numeric)
            x_cat: тензор ID категорий, форма (batch_size,)
        
        Returns:
            Предсказанные продажи, форма (batch_size, 1)
        """
        # Получаем эмбеддинг категории
        # x_cat: (batch_size,) -> (batch_size, emb_dim)
        cat_emb = self.category_emb(x_cat)

        # Объединяем числовые признаки и эмбеддинг
        # x_num: (batch_size, num_numeric), cat_emb: (batch_size, emb_dim)
        # result: (batch_size, num_numeric + emb_dim)
        x = torch.cat([x_num, cat_emb], dim=1)

        # Проекция и добавление измерения для Transformer
        # (batch_size, d_model) -> (batch_size, 1, d_model)
        x = self.input_proj(x).unsqueeze(1)

        # Transformer Encoder
        x = self.encoder(x)

        # Берём первый токен (CLS-подобный подход)
        # (batch_size, 1, d_model) -> (batch_size, d_model)
        x = x[:, 0, :]

        # Финальное предсказание
        return self.regressor(x)

    def predict(self, x_num, x_cat):
        """
        Предсказание для инференса (одиночный сэмпл).
        
        Отличается от forward():
        - автоматически конвертирует numpy в тензоры
        - переключает модель в режим eval
        - возвращает число (float), а не тензор
        
        Args:
            x_num: числовые признаки (numpy array или тензор)
            x_cat: ID категории (numpy array или тензор)
        
        Returns:
            Предсказанное значение продаж (float)
        """
        self.eval()  # режим инференса (отключаем dropout и т.д.)
        
        with torch.no_grad():  # не вычисляем градиенты
            # Конвертация в тензоры, если переданы numpy arrays
            x_num = torch.tensor(x_num, dtype=torch.float32) if not isinstance(x_num, torch.Tensor) else x_num
            x_cat = torch.tensor(x_cat, dtype=torch.int64) if not isinstance(x_cat, torch.Tensor) else x_cat
            
            # Прямой проход и извлечение скалярного значения
            result = self.forward(x_num, x_cat).squeeze().item()
        
        return result
