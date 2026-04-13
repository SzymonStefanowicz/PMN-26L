from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Importy własnych modułów
from PMiNAI.code_task2 import data_analysis, experiments
import preprocessing
import visualizations
import ssl
import os

# To obejście rozwiązuje problem braku certyfikatów na macOS
if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # 1. Pobranie i analiza danych
    X, y = data_analysis.load_data()
    data_analysis.analyze_data(X, y)

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Przygotowanie procesora danych
    preprocessor = preprocessing.build_preprocessor()

    # 3. Przeprowadzenie eksperymentu
    best_model = experiments.run_experiment(X_train, y_train, preprocessor)

    # 4. Ewaluacja na zbiorze testowym
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]

    print("\n=== RAPORT Z KLASYFIKACJI NA ZBIORZE TESTOWYM ===")
    print(classification_report(y_test, y_pred))

    # 5. Wizualizacje
    visualizations.plot_confusion_matrix(y_test, y_pred)
    visualizations.plot_roc_curve(y_test, y_score)

if __name__ == "__main__":
    main()