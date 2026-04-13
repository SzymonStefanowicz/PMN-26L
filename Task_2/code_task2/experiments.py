from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def run_experiment(X_train, y_train, preprocessor):
    """Przeprowadza eksperymenty z hiperparametrami dla regresji logistycznej."""
    # Definicja całego potoku uczenia
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
    ])

    # Parametry do eksperymentu
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'classifier__penalty': ['l1', 'l2']
    }

    print("=== TRENING I EKSPERYMENTY ===")
    print("Szukanie optymalnych parametrów za pomocą GridSearchCV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print(f"Najlepsze znalezione parametry: {grid_search.best_params_}")
    print(f"Najlepsza dokładność (Cross-Validation): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_