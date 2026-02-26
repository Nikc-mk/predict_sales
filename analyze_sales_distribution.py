"""
Модуль анализа распределения продаж.

Назначение:
    Анализ распределения продаж из sales_data.csv на нормальность.
    Помогает определить, какую нормализацию использовать для target.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

try:
    from config import DATA_PATH
except ImportError:
    DATA_PATH = "sales_data.csv"


def analyze_sales_distribution(csv_path: str = None):
    """
    Анализирует распределение продаж и тестирует на нормальность.
    
    Args:
        csv_path: путь к файлу. Если None - берется из config.DATA_PATH
    
    Выводит:
        - Статистики (среднее, медиана, std, skew, kurtosis)
        - Тесты на нормальность (Shapiro-Wilk, D'Agostino-Pearson)
        - Графики распределения
    """
    if csv_path is None:
        csv_path = DATA_PATH
    
    print("=" * 60)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ ПРОДАЖ")
    print("=" * 60)
    
    # Загрузка данных (явно используем ; как разделитель)
    df = pd.read_csv(csv_path, sep=';')
    
    print(f"\nЗагружено записей: {len(df)}")
    print(f"Колонки: {df.columns.tolist()}")
    
    # Определяем колонку с продажами (ищем в т.ч. по названию)
    sales_col = None
    for col in df.columns:
        if 'sale' in col.lower() or 'quantity' in col.lower() or 'amount' in col.lower():
            sales_col = col
            break
    
    if sales_col is None:
        # Пробуем найти числовые колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sales_col = numeric_cols[0]
            print(f"\nИспользуем первую числовую колонку: {sales_col}")
        else:
            print("ERROR: Не найдена колонка с продажами!")
            return
    
    sales = df[sales_col].dropna()
    sales = sales[sales > 0]  # Только положительные значения
    
    print(f"\nПродажи > 0: {len(sales)} записей")
    print(f"Всего уникальных значений: {sales.nunique()}")
    
    # === Базовые статистики ===
    print("\n" + "-" * 40)
    print("БАЗОВЫЕ СТАТИСТИКИ")
    print("-" * 40)
    print(f"Среднее:     {sales.mean():,.2f}")
    print(f"Медиана:     {sales.median():,.2f}")
    print(f"Std:         {sales.std():,.2f}")
    print(f"Min:         {sales.min():,.2f}")
    print(f"Max:         {sales.max():,.2f}")
    print(f"25%:         {sales.quantile(0.25):,.2f}")
    print(f"75%:         {sales.quantile(0.75):,.2f}")
    
    # Skewness и Kurtosis
    skew = sales.skew()
    kurt = sales.kurtosis()
    print(f"\nSkewness:    {skew:.4f}  (0 = симметрично)")
    print(f"Kurtosis:    {kurt:.4f}  (0 = нормальное)")
    
    # Интерпретация skewness
    print("\nИНТЕРПРЕТАЦИЯ:")
    if skew > 1:
        print("  → Сильно скошено вправо (heavy right tail)")
    elif skew > 0.5:
        print("  → Умеренно скошено вправо")
    elif skew < -1:
        print("  → Сильно скошено влево (heavy left tail)")
    elif skew < -0.5:
        print("  → Умеренно скошено влево")
    else:
        print("  → Приблизительно симметрично")
    
    # === Тесты на нормальность ===
    print("\n" + "-" * 40)
    print("ТЕСТЫ НА НОРМАЛЬНОСТЬ")
    print("-" * 40)
    
    # Тест на небольшой выборке (Shapiro-Wilk)
    sample_size = min(5000, len(sales))
    sample = sales.sample(n=sample_size, random_state=42)
    
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    print(f"\nShapiro-Wilk (n={sample_size}):")
    print(f"  Statistic: {shapiro_stat:.6f}")
    print(f"  p-value:   {shapiro_p:.6e}")
    print(f"  → {'НЕ нормальное' if shapiro_p < 0.05 else 'Нормальное'} (α=0.05)")
    
    # D'Agostino-Pearson test
    if len(sales) >= 20:
        dagostino_stat, dagostino_p = stats.normaltest(sales)
        print(f"\nD'Agostino-Pearson:")
        print(f"  Statistic: {dagostino_stat:.6f}")
        print(f"  p-value:   {dagostino_p:.6e}")
        print(f"  → {'НЕ нормальное' if dagostino_p < 0.05 else 'Нормальное'} (α=0.05)")
    
    # Anderson-Darling test
    anderson_result = stats.anderson(sales, dist='norm')
    print(f"\nAnderson-Darling:")
    print(f"  Statistic: {anderson_result.statistic:.6f}")
    for i, (cv, sl) in enumerate(zip(anderson_result.critical_values, anderson_result.significance_level)):
        result = "ОТКЛОНЯЕМ" if anderson_result.statistic > cv else "ПРИНИМАЕМ"
        print(f"  {sl}%: critical_value={cv:.3f} → {result} H0")
    
    # === Рекомендация по нормализации ===
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИЯ")
    print("=" * 60)
    
    if skew > 1 or kurt > 3:
        print("\n⚠️  РАСПРЕДЕЛЕНИЕ ДАЛЕКО ОТ НОРМАЛЬНОГО")
        print("\nРекомендуемые трансформации:")
        print("  1. log1p(y) - логарифм (стандарт для продаж)")
        print("  2. Box-Cox (требует y > 0)")
        print("  3. Yeo-Johnson (работает с y >= 0)")
        
        # Тестируем log1p
        log_sales = np.log1p(sales)
        print(f"\nПосле log1p:")
        print(f"  Skewness:  {log_sales.skew():.4f}")
        print(f"  Kurtosis:  {log_sales.kurtosis():.4f}")
        
        shapiro_log, p_log = stats.shapiro(log_sales.sample(n=min(5000, len(log_sales)), random_state=42))
        print(f"  Shapiro p-value: {p_log:.6e}")
        print(f"  → {'Нормальное' if p_log > 0.05 else 'Всё ещё не нормальное'}")
    else:
        print("\n✓ Распределение близко к нормальному")
        print("  StandardScaler может быть приемлем")
    
    # === Графики ===
    print("\n" + "-" * 40)
    print("СОЗДАНИЕ ГРАФИКОВ...")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Гистограмма
    ax1 = axes[0, 0]
    ax1.hist(sales, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_title("Распределение продаж")
    ax1.set_xlabel("Продажи")
    ax1.set_ylabel("Частота")
    
    # 2. Гистограмма после log1p
    ax2 = axes[0, 1]
    ax2.hist(np.log1p(sales), bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_title("Распределение log1p(продаж)")
    ax2.set_xlabel("log1p(Продажи)")
    ax2.set_ylabel("Частота")
    
    # 3. Q-Q plot для исходных данных
    ax3 = axes[1, 0]
    stats.probplot(sales, dist="norm", plot=ax3)
    ax3.set_title("Q-Q plot (исходные данные)")
    
    # 4. Q-Q plot после log1p
    ax4 = axes[1, 1]
    stats.probplot(np.log1p(sales), dist="norm", plot=ax4)
    ax4.set_title("Q-Q plot (log1p)")
    
    plt.tight_layout()
    plt.savefig("sales_distribution_analysis.png", dpi=150)
    print("Графики сохранены в sales_distribution_analysis.png")
    
    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)


if __name__ == "__main__":
    analyze_sales_distribution()  # автоматически берет DATA_PATH из config.py
