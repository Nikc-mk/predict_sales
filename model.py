"""
Модуль нейросетевой модели на базе Transformer.

Архитектура:
    RealTabTransformer - трансформер для табличных данных.

Структура:
    1. Эмбеддинги для категориального признака (категория товара)
    2. Проекция числовых признаков в размерность трансформера
    3. Позиционное кодирование
    4. Слои Transformer Encoder с Self-Attention
    5. Выходной слой для регрессии
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Позиционное кодирование из оригинальной статьи Attention is All You Need"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RealTabTransformer(nn.Module):
    """
    Трансформер для табличных данных.
    
    Архитектура:
        - Embedding слой для категорий
        - Проекция числовых признаков в d_model
        - Позиционное кодирование
        - Transformer Encoder с Self-Attention
        - Регрессор для финального предсказания
    
    Args:
        num_categories: количество уникальных категорий
        num_numeric: количество числовых признаков
        d_model: размерность модели (по умолчанию 128)
        nhead: количество голов внимания (по умолчанию 8)
        num_layers: количество слоёв трансформера (по умолчанию 4)
        dim_feedforward: размерность feedforward слоя (по умолчанию 256)
        dropout: dropout (по умолчанию 0.1)
    """
    
    def __init__(
        self,
        num_categories,
        num_numeric,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # 1. Эмбеддинги для категорий
        self.category_emb = nn.Embedding(num_categories, d_model)
        
        # 2. Проекция числовых признаков - каждый признак проецируется в d_model
        # Используем Linear для каждого признака отдельно
        self.numeric_proj = nn.Linear(num_numeric, d_model)
        
        # 3. Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 4. Слои Transformer Encoder с Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # 5. Выходной слой
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x_num, x_cat):
        """
        Прямой проход модели.
        
        Args:
            x_num: тензор числовых признаков, форма (batch_size, num_numeric)
            x_cat: тензор ID категорий, форма (batch_size,)
        
        Returns:
            Предсказанные продажи, форма (batch_size, 1)
        """
        # Проецируем числовые признаки: (batch, num_numeric) -> (batch, d_model) -> (batch, num_numeric, d_model)
        # Сначала проецируем все признаки вместе
        numeric_projected = self.numeric_proj(x_num)  # (batch, d_model)
        # Затем расширяем, чтобы получить токен для каждого признака
        numeric_tokens = numeric_projected.unsqueeze(1).expand(-1, x_num.size(1), -1)  # (batch, num_numeric, d_model)
        
        # Категория тоже даёт один токен
        cat_token = self.category_emb(x_cat).unsqueeze(1)  # (batch, 1, d_model)
        
        # Объединяем: все числовые токены + категория
        combined = torch.cat([numeric_tokens, cat_token], dim=1)  # (batch, num_numeric+1, d_model)
        
        # Добавляем позиционное кодирование
        combined = self.pos_encoder(combined)
        
        # Пропускаем через Transformer Encoder
        transformer_output = self.transformer_encoder(combined)  # (batch, seq_len, d_model)
        
        # Усредняем выходы по seq_len
        output = transformer_output.mean(dim=1)  # (batch, d_model)
        
        # Предсказание
        return self.output_layer(output)

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


# Алиас для обратной совместимости
TabTransformerModel = RealTabTransformer
