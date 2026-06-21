"""
prueba_stemming.py — Prueba de stemming SIN tocar el core.
Compara los 3 experimentos con y sin stemming y muestra diferencias.

Uso: python pruebas/prueba_stemming.py --stemming
     python pruebas/prueba_stemming.py (sin stemming, linea base)
     python pruebas/prueba_stemming.py --comparar (corre ambas y compara)
"""

import sys
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import cargar_y_aplanar
from cleaner import preprocesar as preprocesar_sin_stemming
from vectorizer import vectorizar_tfidf, cargar_modelo_embeddings, crear_vectorizador_tfidf

from util_stemming import preprocesar_con_stemming

RUTA_DATASET = Path(__file__).parent.parent.parent / "02-Dataset" / "02-Dataset.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ejecutar_experimentos(usar_stemming: bool) -> dict:
    """
    Ejecuta los 3 experimentos con o sin stemming.

    Returns
    -------
    dict: {nombre_experimento: {'accuracy': float, 'f1_macro': float}}
    """
    modo = "CON stemming" if usar_stemming else "SIN stemming"
    preprocesar = preprocesar_con_stemming if usar_stemming else preprocesar_sin_stemming

    print(f"\n{'='*60}")
    print(f"  EXPERIMENTOS {modo}")
    print(f"{'='*60}")

    # Cargar y preprocesar
    df = cargar_y_aplanar(str(RUTA_DATASET))
    textos = [preprocesar(t) for t in df["texto"]]
    etiquetas = df["intent"].to_numpy().copy()

    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=etiquetas
    )

    resultados = {}

    # === EXP 1: Naive Bayes + TF-IDF ===
    print(f"\n--- Experimento 1: Naive Bayes + TF-IDF ---")
    X_train_vec, vec = vectorizar_tfidf(X_train, entrenar=True)
    X_test_vec, _ = vectorizar_tfidf(X_test, vectorizador=vec, entrenar=False)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    f1 = report["macro avg"]["f1-score"]
    print(f"  Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")
    resultados["NB + TF-IDF"] = {"accuracy": acc, "f1_macro": f1}

    # === EXP 2: SVM + TF-IDF ===
    print(f"\n--- Experimento 2: SVM Lineal + TF-IDF ---")
    X_train_vec2, vec2 = vectorizar_tfidf(X_train, entrenar=True)
    X_test_vec2, _ = vectorizar_tfidf(X_test, vectorizador=vec2, entrenar=False)

    clf = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
    clf.fit(X_train_vec2, y_train)
    y_pred = clf.predict(X_test_vec2)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    f1 = report["macro avg"]["f1-score"]
    print(f"  Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")
    resultados["SVM + TF-IDF"] = {"accuracy": acc, "f1_macro": f1}

    # Liberar memoria antes de embeddings
    del X_train_vec, X_test_vec, vec, X_train_vec2, X_test_vec2, vec2, clf
    gc.collect()

    # === EXP 3: SVM + Embeddings ===
    print(f"\n--- Experimento 3: SVM + Embeddings ---")
    modelo = cargar_modelo_embeddings("distiluse-base-multilingual-cased-v2")
    X_train_emb = modelo.encode(X_train, show_progress_bar=True)
    X_test_emb = modelo.encode(X_test, show_progress_bar=True)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    clf.fit(X_train_emb, y_train)
    y_pred = clf.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    f1 = report["macro avg"]["f1-score"]
    print(f"  Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")
    resultados["SVM + Embeddings (RBF)"] = {"accuracy": acc, "f1_macro": f1}

    return resultados


def crear_tabla_comparativa(resultados_sin: dict, resultados_con: dict) -> pd.DataFrame:
    """Crea tabla comparativa: sin | con | diferencia."""
    filas = []
    for exp in resultados_sin:
        sin = resultados_sin[exp]
        con = resultados_con[exp]
        dif_acc = con["accuracy"] - sin["accuracy"]
        dif_f1 = con["f1_macro"] - sin["f1_macro"]
        filas.append({
            "Experimento": exp,
            "Accuracy SIN": f"{sin['accuracy']:.2%}",
            "Accuracy CON": f"{con['accuracy']:.2%}",
            "Diferencia Acc": f"{dif_acc:+.2%}",
            "F1 SIN": f"{sin['f1_macro']:.4f}",
            "F1 CON": f"{con['f1_macro']:.4f}",
            "Diferencia F1": f"{dif_f1:+.4f}",
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prueba de stemming")
    parser.add_argument(
        "--modo",
        choices=["sin", "con", "comparar"],
        default="comparar",
        help="'sin': solo sin stemming, 'con': solo con stemming, 'comparar': ambos (default)",
    )
    args = parser.parse_args()

    if args.modo in ("sin", "comparar"):
        resultados_sin = ejecutar_experimentos(usar_stemming=False)
    else:
        resultados_sin = None

    if args.modo in ("con", "comparar"):
        resultados_con = ejecutar_experimentos(usar_stemming=True)
    else:
        resultados_con = None

    if resultados_sin and resultados_con:
        print(f"\n{'='*60}")
        print(f"  TABLA COMPARATIVA: Stemming")
        print(f"{'='*60}")
        tabla = crear_tabla_comparativa(resultados_sin, resultados_con)
        print(f"\n{tabla.to_string(index=False)}")
    elif resultados_sin:
        print("\n[OK] Prueba SIN stemming completada.")
    else:
        print("\n[OK] Prueba CON stemming completada.")
