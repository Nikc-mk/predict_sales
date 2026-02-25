"""
Модуль нейросетевой модели (упрощённый MLP).

Архитектура:
    Простая MLP модель для работы с табличными данными.

Структура:
    1. Embedding для категориального признака (категория товара)
    2. Объединение числовых признаков и эмбеддинга
    3. MLP регрессор
"""

import torch
import torch.nn as nn


class TabTransformerModel(nn.Module):
    """
    Упрощённая MLP модель для прогнозирования продаж.
    
    Архитектура:
        - Embedding слой для категорий
        - MLP для объединения числовых признаков и эмбеддинга
        - Регрессор для финального предсказания
    
    Args:
        num_categories: количество уникальных категорий
        num_numeric: количество числовых признаков
        hidden_dim: размерность скрытого слоя (по умолчанию 128)
        emb_dim: размерность эмбеддинга категории (по умолчанию 32)
    """
    
    def __init__(
        self,
        num_categories,
        num_numeric,
        hidden_dim=128,
        emb_dim=32,
    ):
        super().__init__()

        # Эмбеддинг для категорий
        self.category_emb = nn.Embedding(num_categories, emb_dim)

        # MLP: объединяем числовые признаки и эмбеддинг -> hidden -> output
        # num_numeric + emb_dim -> hidden_dim -> hidden_dim/2 -> 1
        self.mlp = nn.Sequential(
            nn.Linear(num_numeric + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
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
        cat_emb = self.category_emb(x_cat)

        # Объединяем числовые признаки и эмбеддинг
        x = torch.cat([x_num, cat_emb], dim=1)

        # MLP
        return self.mlp(x)

    def predict(self, x_num, x_cat):
        """
        Предсказание для инференса (одиночный сэмпл).
        
        Args:
            x_num: numpy array или tensor числовых признаков, форма (1, num_numeric)
            x_cat: numpy array или tensor ID категории, форма (1,) или int
        
        Returns:
            Предсказание (нормализованное)
        """
        self.eval()
        
        with torch.no_grad():
            # Преобразуем в тензоры если нужно
            x_num = torch.tensor(x_num, dtype=torch.float32) if not isinstance(x_num, torch.Tensor) else x_num
            x_cat = torch.tensor(x_cat, dtype=torch.int64) if not isinstance(x_cat, torch.Tensor) else x_cat
            
            # Убедимся, что x_num имеет форму (1, num_numeric)
            if x_num.dim() == 1:
                x_num = x_num.unsqueeze(0)
            
            # Убедимся, что x_cat имеет форму (1,)
            if x_cat.dim() == 0:
                x_cat = x_cat.unsqueeze(0)
            
            result = self.forward(x_num, x_cat).squeeze().item()
        
        return result
