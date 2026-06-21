"""
prueba_gridsearch.py — GridSearchCV + K-Fold Cross Validation (Fase 4)
======================================================================

Barre hiperparámetros para los 3 experimentos con validación cruzada K-Fold.
Reduce la varianza del test set y encuentra la configuración óptima.

Experimentos:
  1. NB + TF-IDF (1,1)     → alpha
  2. SVM Lineal + TF-IDF (1,1) → C
  3. SVM RBF + Embeddings  → C, gamma

Salida:
  - Mejores parámetros por experimento (CV score)
  - Re-evaluación en test set (para comparación directa con Fase 3)
  - Tabla comparativa final
"""

import sys
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import cargar_y_aplanar
from cleaner import preprocesar
from vectorizer import (
    crear_vectorizador_tfidf,
    vectorizar_tfidf,
    cargar_modelo_embeddings,
)

RUTA_DATASET = Path(__file__).parent.parent.parent / "02-Dataset" / "02-Dataset.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ============================================================
# Grids de hiperparámetros
# ============================================================

GRID_NB = {
    "alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
}

GRID_SVM_LINEAL = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
}

GRID_SVM_RBF = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", "auto", 0.1, 0.01, 0.001, 0.0001],
}


def ejecutar_gridsearch_tfidf(
    X_train, y_train, X_test, y_test, estimador, param_grid, nombre: str
) -> dict:
    """
    GridSearchCV con K-Fold para experimentos basados en TF-IDF.
    """
    print(f"\n{'='*60}")
    print(f"  GridSearchCV: {nombre}")
    print(f"{'='*60}")

    # Vectorizar (ya preprocesados)
    X_train_vec, vec = vectorizar_tfidf(X_train, entrenar=True)
    X_test_vec, _ = vectorizar_tfidf(X_test, vectorizador=vec, entrenar=False)

    print(f"\n  Buscando mejores hiperparámetros con CV={CV_FOLDS}...")
    print(f"  Grid: {param_grid}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=estimador,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    t0 = time.time()
    grid.fit(X_train_vec, y_train)
    t1 = time.time()

    print(f"\n  Tiempo: {t1 - t0:.1f}s")
    print(f"\n  Mejores parámetros: {grid.best_params_}")
    print(f"  Best CV F1 macro: {grid.best_score_:.4f}")
    print(f"  CV F1 macro (train): {grid.cv_results_['mean_train_score'][grid.best_index_]:.4f}")

    # Evaluar en test set
    best = grid.best_estimator_
    y_pred = best.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    f1 = report["macro avg"]["f1-score"]

    print(f"\n  >>> Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  >>> Test F1 macro: {f1:.4f}")

    resultados = {
        "nombre": nombre,
        "best_params": grid.best_params_,
        "best_cv_f1": round(grid.best_score_, 4),
        "train_cv_f1": round(grid.cv_results_["mean_train_score"][grid.best_index_], 4),
        "test_acc": round(acc, 4),
        "test_f1": round(f1, 4),
        "modelo": best,
        "vectorizador": vec,
        "tiempo": round(t1 - t0, 1),
    }
    return resultados


def ejecutar_gridsearch_embeddings(
    X_train, y_train, X_test, y_test, nombre: str
) -> dict:
    """
    GridSearchCV con K-Fold para SVM + Embeddings.
    """
    print(f"\n{'='*60}")
    print(f"  GridSearchCV: {nombre}")
    print(f"{'='*60}")

    # Cargar modelo de embeddings
    modelo = cargar_modelo_embeddings("distiluse-base-multilingual-cased-v2")

    print(f"\n  Generando embeddings (train)...")
    X_train_emb = modelo.encode(X_train, show_progress_bar=True)

    print(f"\n  Buscando mejores hiperparámetros con CV={CV_FOLDS}...")
    print(f"  Grid: {GRID_SVM_RBF}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=SVC(random_state=RANDOM_STATE),
        param_grid=GRID_SVM_RBF,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
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
    print(f"\n  Generando embeddings (test)...")
    X_test_emb = modelo.encode(X_test, show_progress_bar=True)
    best = grid.best_estimator_
    y_pred = best.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    f1 = report["macro avg"]["f1-score"]

    print(f"\n  >>> Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  >>> Test F1 macro: {f1:.4f}")

    resultados = {
        "nombre": nombre,
        "best_params": grid.best_params_,
        "best_cv_f1": round(grid.best_score_, 4),
        "train_cv_f1": round(grid.cv_results_["mean_train_score"][grid.best_index_], 4),
        "test_acc": round(acc, 4),
        "test_f1": round(f1, 4),
        "modelo": best,
        "vectorizador": None,
        "tiempo": round(t1 - t0, 1),
    }
    return resultados


def mostrar_tabla_final(resultados: list):
    """Muestra tabla comparativa de todos los experimentos."""
    filas = []
    for r in resultados:
        filas.append({
            "Experimento": r["nombre"],
            "Best Params": str(r["best_params"]),
            "CV F1 (val)": r["best_cv_f1"],
            "CV F1 (train)": r["train_cv_f1"],
            "Test Acc": f"{r['test_acc']:.2%}",
            "Test F1": r["test_f1"],
            "Tiempo (s)": r["tiempo"],
        })

    tabla = pd.DataFrame(filas)
    print("\n" + "=" * 80)
    print("  TABLA FINAL — GridSearchCV + K-Fold CV")
    print("=" * 80)
    print(f"\n{tabla.to_string(index=False)}")

    # Detectar sobreajuste
    print("\n  --- Análisis de Sobreajuste ---")
    for r in resultados:
        diff = r["train_cv_f1"] - r["best_cv_f1"]
        estado = "⚠️  POSIBLE SOBREAJUSTE" if diff > 0.10 else "✅ OK"
        print(f"  {r['nombre']:30s} | train={r['train_cv_f1']:.4f} val={r['best_cv_f1']:.4f} "
              f"diff={diff:.4f} {estado}")

    return tabla


if __name__ == "__main__":
    print("=" * 60)
    print("  FASE 4: GridSearchCV + K-Fold Cross Validation")
    print("=" * 60)

    # Cargar datos
    df = cargar_y_aplanar(str(RUTA_DATASET))
    textos = [preprocesar(t) for t in df["texto"]]
    etiquetas = df["intent"].to_numpy().copy()

    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=etiquetas
    )

    resultados = []

    # === EXP 1: NB + TF-IDF (1,1) ===
    r1 = ejecutar_gridsearch_tfidf(
        X_train, y_train, X_test, y_test,
        estimador=MultinomialNB(),
        param_grid=GRID_NB,
        nombre="NB + TF-IDF (1,1)",
    )
    resultados.append(r1)
    gc.collect()

    # === EXP 2: SVM Lineal + TF-IDF (1,1) ===
    r2 = ejecutar_gridsearch_tfidf(
        X_train, y_train, X_test, y_test,
        estimador=SVC(kernel="linear", random_state=RANDOM_STATE),
        param_grid=GRID_SVM_LINEAL,
        nombre="SVM Lineal + TF-IDF (1,1)",
    )
    resultados.append(r2)
    gc.collect()

    # === EXP 3: SVM RBF + Embeddings ===
    r3 = ejecutar_gridsearch_embeddings(
        X_train, y_train, X_test, y_test,
        nombre="SVM RBF + Embeddings",
    )
    resultados.append(r3)

    # === Tabla final ===
    mostrar_tabla_final(resultados)
