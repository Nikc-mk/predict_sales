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
    # Если данных нет - просто сохраняем пустой график
    if val_df is None or len(val_df) == 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Нет данных для валидации', ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Графики сохранены в {output_path} (пустой график)")
        plt.show()
        return
    
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
    
    # === График 2: Ошибка прогноза общего объёма по формуле (fact_total - total_forecast) / abs(fact_total) ===
    ax2 = axes[0, 1]
    val_df = val_df.copy()
    val_df['total_error_pct'] = (val_df['fact_total'] - val_df['total_forecast']).abs() / val_df['fact_total'] * 100
    
    for month in val_df['month'].unique():
        month_data = val_df[val_df['month'] == month].sort_values('day_of_month')
        ax2.plot(month_data['day_of_month'], month_data['total_error_pct'], 
                 marker='o', markersize=3, label=month)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel("День месяца (когда сделан прогноз)")
    ax2.set_ylabel("Ошибка прогноза общего объёма %")
    ax2.set_title("MAPE факт прогноз по дням")
    ax2.legend()
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
    
    # === График 4: Bias по дням для каждого месяца ===
    ax4 = axes[1, 1]
    val_df = val_df.copy()
    # Bias = (forecast - fact) / abs(fact) * 100
    val_df['bias_pct'] = (val_df['total_forecast'] - val_df['fact_total']) / val_df['fact_total'].abs() * 100
    
    for month in val_df['month'].unique():
        month_data = val_df[val_df['month'] == month].sort_values('day_of_month')
        ax4.plot(month_data['day_of_month'], month_data['bias_pct'], 
                 marker='o', markersize=3, label=month)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel("День месяца")
    ax4.set_ylabel("Bias %")
    ax4.set_title("Bias по дням для каждого месяца")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Графики сохранены в {output_path}")
    plt.show()
