"""
prueba_ngrams.py — Prueba de ngram_range en TF-IDF para NB y SVM.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import cargar_y_aplanar
from cleaner import preprocesar
from vectorizer import crear_vectorizador_tfidf

RUTA_DATASET = Path(__file__).parent.parent.parent / "02-Dataset" / "02-Dataset.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NGRAM_RANGES = [(1, 1), (1, 2), (1, 3), (2, 2)]


def probar_ngrams():
    df = cargar_y_aplanar(str(RUTA_DATASET))
    textos = [preprocesar(t) for t in df["texto"]]
    etiquetas = df["intent"].to_numpy().copy()

    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=etiquetas
    )

    filas = []

    for ngram in NGRAM_RANGES:
        vec = crear_vectorizador_tfidf(max_features=500, min_df=2, ngram_range=ngram)

        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)
        n_term = X_train_vec.shape[1]
        dens = X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1])

        # NB
        nb = MultinomialNB(alpha=1.0)
        nb.fit(X_train_vec, y_train)
        y_pred_nb = nb.predict(X_test_vec)
        acc_nb = accuracy_score(y_test, y_pred_nb)
        report_nb = classification_report(y_test, y_pred_nb, zero_division=0, output_dict=True)
        f1_nb = report_nb["macro avg"]["f1-score"]

        # SVM
        svm = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_train_vec, y_train)
        y_pred_svm = svm.predict(X_test_vec)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        report_svm = classification_report(y_test, y_pred_svm, zero_division=0, output_dict=True)
        f1_svm = report_svm["macro avg"]["f1-score"]

        filas.append({
            "ngram_range": str(ngram),
            "terminos": n_term,
            "densidad": f"{dens:.4f}",
            "NB_acc": f"{acc_nb:.2%}",
            "NB_f1": round(f1_nb, 4),
            "SVM_acc": f"{acc_svm:.2%}",
            "SVM_f1": round(f1_svm, 4),
        })

        print(f"  ngram={str(ngram):7s} -> {n_term:3d} terminos (dens={dens:.4f}) | NB: {acc_nb:.2%} f1={f1_nb:.4f} | SVM: {acc_svm:.2%} f1={f1_svm:.4f}")

    return pd.DataFrame(filas)


if __name__ == "__main__":
    print("=" * 60)
    print("  PRUEBA: ngram_range en TF-IDF")
    print("=" * 60)
    tabla = probar_ngrams()
    print(f"\n{'='*60}")
    print(f"  TABLA COMPARATIVA: ngram_range")
    print(f"{'='*60}")
    print(f"\n{tabla.to_string(index=False)}")
