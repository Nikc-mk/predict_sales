"""
Модуль визуализации результатов.

Функции:
    - Графики ошибок прогноза
    - Сравнение факт vs прогноз
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_validation_results(val_df: pd.DataFrame, output_path: str = "validation_plot.png"):
    """
    Строит 4 графика по результатам валидации.
    
    Оценивается качество ПРЕДСКАЗАНИЯ ОСТАТКА месяца.
    
    Графики:
        1. Ошибка предсказания остатка по дням (все месяцы)
        2. Факт остаток vs Прогноз остатка
        3. Абсолютная ошибка по горизонту предсказания
        4. Распределение ошибки по месяцам (boxplot)
    
    Args:
        val_df: DataFrame с результатами валидации
        output_path: путь для сохранения графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Фильтруем: только записи где есть что предсказывать (fact_remaining > 0)
    # и модель хоть что-то предсказала
    valid_df = val_df[(val_df['fact_remaining'] > 1000) & (val_df['predicted_remaining'] > 0)].copy()
    
    # === График 1: Ошибка по дням (только валидные записи) ===
    ax1 = axes[0, 0]
    for month in valid_df['month'].unique():
        month_data = valid_df[valid_df['month'] == month]
        ax1.plot(month_data['day_of_month'], month_data['error_pct'], 
                 marker='o', markersize=3, label=month)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel("День месяца (когда сделан прогноз)")
    ax1.set_ylabel("Ошибка предсказания остатка %")
    ax1.set_title("Ошибка предсказания ОСТАТКА месяца по дням\n(положительная = перепрогноз)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === График 2: Факт остаток vs Прогноз остатка (в абсолютных числах) ===
    ax2 = axes[0, 1]
    for month in val_df['month'].unique():
        month_data = val_df[val_df['month'] == month].sort_values('day_of_month')
        # Фактический остаток - горизонтальная линия
        ax2.axhline(y=month_data['fact_remaining'].iloc[0], color='blue', alpha=0.3, linestyle='--')
        # Предсказанный остаток - точки
        ax2.plot(month_data['day_of_month'], month_data['predicted_remaining'], 
                 marker='o', markersize=3, label=f'{month}')
    ax2.set_xlabel("День месяца (когда сделан прогноз)")
    ax2.set_ylabel("Продажи за остаток месяца")
    ax2.set_title("Фактический остаток (линии) vs Предсказанный остаток (точки)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # === График 3: Абсолютная ошибка в абсолютных числах ===
    ax3 = axes[1, 0]
    val_df = val_df.copy()
    val_df['abs_error'] = (val_df['predicted_remaining'] - val_df['fact_remaining']).abs()
    # Группируем по горизонту
    horizon_error = val_df.groupby('days_left').agg({
        'abs_error': ['mean', 'std'],
        'fact_remaining': 'mean'
    }).reset_index()
    horizon_error.columns = ['days_left', 'abs_error_mean', 'abs_error_std', 'fact_mean']
    
    ax3.bar(horizon_error['days_left'], horizon_error['fact_mean'], 
            alpha=0.3, label='Фактический остаток', color='blue')
    ax3.errorbar(horizon_error['days_left'], horizon_error['abs_error_mean'], 
                 yerr=horizon_error['abs_error_std'], fmt='r-o', markersize=4, 
                 capsize=3, label='Ошибка модели')
    ax3.set_xlabel("Горизонт предсказания (дней осталось)")
    ax3.set_ylabel("Продажи")
    ax3.set_title("Ошибка модели vs Фактический остаток по горизонту")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === График 4: Boxplot по месяцам (только валидные записи) ===
    ax4 = axes[1, 1]
    if len(valid_df) > 0:
        months = valid_df['month'].unique()
        data_for_box = [valid_df[valid_df['month'] == m]['error_pct'].values for m in months]
        bp = ax4.boxplot(data_for_box, labels=months, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
    else:
        ax4.text(0.5, 0.5, 'Нет валидных данных', ha='center', va='center', transform=ax4.transAxes)
    
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel("Месяц")
    ax4.set_ylabel("Ошибка предсказания остатка %")
    ax4.set_title("Распределение ошибки по месяцам")
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Графики сохранены в {output_path}")
    plt.show()
