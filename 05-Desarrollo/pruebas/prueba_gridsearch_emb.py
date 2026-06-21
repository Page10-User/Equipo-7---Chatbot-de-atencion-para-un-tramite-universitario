"""
prueba_gridsearch_emb.py — GridSearchCV para SVM + Embeddings (versión memory-safe)
===================================================================================

El grid completo (4C × 6gamma × 5folds = 120 fits) explota por pagefile.
Solución: generar embeddings una sola vez, grid más chico, n_jobs=1.
"""

import sys
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import cargar_y_aplanar
from cleaner import preprocesar
from vectorizer import cargar_modelo_embeddings

RUTA_DATASET = Path(__file__).parent.parent.parent / "02-Dataset" / "02-Dataset.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

GRID_RBF_MEMSAFE = {
    "C": [0.1, 1.0, 10.0],
    "gamma": ["scale", "auto", 0.01, 0.001],
}


def ejecutar():
    print("=" * 60)
    print("  GridSearchCV: SVM RBF + Embeddings (memory-safe)")
    print("=" * 60)

    df = cargar_y_aplanar(str(RUTA_DATASET))
    textos = [preprocesar(t) for t in df["texto"]]
    etiquetas = df["intent"].to_numpy().copy()

    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=etiquetas
    )

    # Cargar modelo y generar embeddings UNA SOLA VEZ
    modelo = cargar_modelo_embeddings("distiluse-base-multilingual-cased-v2")
    print(f"\n  Generando embeddings (train, {len(X_train)} textos)...")
    X_train_emb = modelo.encode(X_train, show_progress_bar=True)
    print(f"  Generando embeddings (test, {len(X_test)} textos)...")
    X_test_emb = modelo.encode(X_test, show_progress_bar=True)

    # GridSearchCV con n_jobs=1 para no saturar memoria
    print(f"\n  Grid: {GRID_RBF_MEMSAFE}")
    print(f"  Total combinaciones: {len(GRID_RBF_MEMSAFE['C']) * len(GRID_RBF_MEMSAFE['gamma'])}")
    print(f"  Fits totales: {len(GRID_RBF_MEMSAFE['C']) * len(GRID_RBF_MEMSAFE['gamma']) * CV_FOLDS}")
    print(f"  n_jobs=1 (memory-safe)")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=SVC(random_state=RANDOM_STATE),
        param_grid=GRID_RBF_MEMSAFE,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        verbose=1,
        return_train_score=True,
    )

    t0 = time.time()
    grid.fit(X_train_emb, y_train)
    t1 = time.time()

    print(f"\n  Tiempo: {t1 - t0:.1f}s")
    print(f"\n  Mejores parámetros: {grid.best_params_}")
    print(f"  Best CV F1 macro: {grid.best_score_:.4f}")
    print(f"  CV F1 macro (train): {grid.cv_results_['mean_train_score'][grid.best_index_]:.4f}")

    # Evaluar en test set
    best = grid.best_estimator_
    y_pred = best.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    f1 = report["macro avg"]["f1-score"]

    print(f"\n  >>> Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  >>> Test F1 macro: {f1:.4f}")

    # Análisis de sobreajuste
    diff = grid.cv_results_["mean_train_score"][grid.best_index_] - grid.best_score_
    estado = "⚠️  POSIBLE SOBREAJUSTE" if diff > 0.10 else "✅ OK"
    print(f"\n  Diferencia train-val: {diff:.4f} {estado}")

    return {
        "best_params": grid.best_params_,
        "best_cv_f1": round(grid.best_score_, 4),
        "train_cv_f1": round(grid.cv_results_["mean_train_score"][grid.best_index_], 4),
        "test_acc": round(acc, 4),
        "test_f1": round(f1, 4),
        "tiempo": round(t1 - t0, 1),
    }


if __name__ == "__main__":
    r = ejecutar()
    print("\n" + "=" * 60)
    print("  RESULTADOS SVM RBF + Embeddings")
    print("=" * 60)
    for k, v in r.items():
        print(f"  {k}: {v}")
