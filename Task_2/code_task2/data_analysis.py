import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_data():
    """Pobiera zbiór Heart Disease z UCI ML Repository."""
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features

    # Przekształcenie celu na problem binarny: 0 - zdrowy, 1 - chory (oryginalnie są wartości 0-4)
    y = (dataset.data.targets > 0).astype(int).values.ravel()
    return X, y


def analyze_data(X, y):
    """Wykonuje analizę zbioru danych (EDA)."""
    print("=== ANALIZA ZBIORU DANYCH ===")
    print(f"Liczebność zbioru: {X.shape[0]} wierszy, {X.shape[1]} cech.")

    print("\n1. Liczba brakujących danych w poszczególnych kolumnach:")
    missing_data = X.isnull().sum()
    print(missing_data[missing_data > 0])

    print("\n2. Rozkład zmiennej docelowej (0 = zdrowy, 1 = chory):")
    class_distribution = pd.Series(y).value_counts()
    print(class_distribution)
    print("=============================\n")