"""
prueba_max_features.py — Prueba de max_features en TF-IDF para SVM.
Compara NB y SVM con distintos tamanos de vocabulario.
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
from vectorizer import crear_vectorizador_tfidf, vectorizar_tfidf

RUTA_DATASET = Path(__file__).parent.parent.parent / "02-Dataset" / "02-Dataset.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2

MAX_FEATURES_VALS = [50, 100, 200, 500, 1000]
MIN_DF = 2


def probar_max_features():
    df = cargar_y_aplanar(str(RUTA_DATASET))
    textos = [preprocesar(t) for t in df["texto"]]
    etiquetas = df["intent"].to_numpy().copy()

    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=etiquetas
    )

    filas = []

    for mf in MAX_FEATURES_VALS:
        vec = crear_vectorizador_tfidf(max_features=mf, min_df=MIN_DF, ngram_range=(1, 2))

        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)
        n_term = X_train_vec.shape[1]

        # NB
        nb = MultinomialNB(alpha=1.0)
        nb.fit(X_train_vec, y_train)
        y_pred_nb = nb.predict(X_test_vec)
        acc_nb = accuracy_score(y_test, y_pred_nb)
        report_nb = classification_report(y_test, y_pred_nb, zero_division=0, output_dict=True)
        f1_nb = report_nb["macro avg"]["f1-score"]

        # SVM lineal
        svm = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_train_vec, y_train)
        y_pred_svm = svm.predict(X_test_vec)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        report_svm = classification_report(y_test, y_pred_svm, zero_division=0, output_dict=True)
        f1_svm = report_svm["macro avg"]["f1-score"]

        filas.append({
            "max_features": mf,
            "terminos_reales": n_term,
            "NB_acc": f"{acc_nb:.2%}",
            "NB_f1": round(f1_nb, 4),
            "SVM_acc": f"{acc_svm:.2%}",
            "SVM_f1": round(f1_svm, 4),
        })

        print(f"  max_features={mf:4d} -> {n_term:3d} terminos reales | NB: {acc_nb:.2%} f1={f1_nb:.4f} | SVM: {acc_svm:.2%} f1={f1_svm:.4f}")

    return pd.DataFrame(filas)


if __name__ == "__main__":
    print("=" * 60)
    print("  PRUEBA: max_features en TF-IDF")
    print("=" * 60)
    tabla = probar_max_features()
    print(f"\n{'='*60}")
    print(f"  TABLA COMPARATIVA: max_features")
    print(f"{'='*60}")
    print(f"\n{tabla.to_string(index=False)}")
