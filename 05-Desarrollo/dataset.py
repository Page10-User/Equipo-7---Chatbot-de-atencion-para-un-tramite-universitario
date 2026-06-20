"""
dataset.py — Carga y aplanado del dataset desde JSON.

Funciones:
    - cargar_dataset(ruta): carga el JSON completo como dict
    - aplanar_dataset(dataset): convierte el JSON a DataFrame (texto, intent)
    - cargar_y_aplanar(ruta): shortcut que hace ambos pasos

Formato de salida:
    DataFrame con columnas:
        - texto:   str (la pregunta del estudiante)
        - intent:  str (la intención asociada)
        - paso:    str (el paso del trámite: "1", "2", "general", "ninguno")
"""

import json
import pandas as pd
from pathlib import Path


def cargar_dataset(ruta: str) -> dict:
    """
    Carga el archivo JSON del dataset como diccionario.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo .json (absoluta o relativa al CWD).

    Returns
    -------
    dict
        Diccionario completo con metadatos e intenciones.
    """
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {ruta}")

    if ruta.suffix.lower() != ".json":
        raise ValueError(f"El archivo debe ser .json, se recibió: {ruta.suffix}")

    with open(ruta, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Validación mínima de estructura
    if "intents" not in dataset:
        raise KeyError(
            "El JSON no contiene la clave 'intents'. "
            "¿Es este el archivo de dataset correcto?"
        )

    version = dataset.get("version", "desconocida")
    cant_intents = len(dataset["intents"])
    print(f"[OK] Dataset cargado: versión {version}, {cant_intents} intenciones.")

    return dataset


def aplanar_dataset(dataset: dict) -> pd.DataFrame:
    """
    Convierte la estructura JSON en un DataFrame fila por pregunta.
    Cada pregunta individual se convierte en una fila con su intención.

    Parámetros
    ----------
    dataset : dict
        Diccionario del dataset (salida de cargar_dataset()).

    Returns
    -------
    pd.DataFrame
        Columnas: texto, intent, paso
    """
    filas = []

    for intent_obj in dataset["intents"]:
        intent = intent_obj["intent"]
        paso = intent_obj.get("paso", "ninguno")

        for pregunta in intent_obj.get("preguntas", []):
            filas.append({
                "texto": pregunta.strip(),
                "intent": intent,
                "paso": paso,
            })

    df = pd.DataFrame(filas)

    if df.empty:
        print("[WARN] El dataset aplanado está vacío. Revisá que haya preguntas en el JSON.")
    else:
        print(f"[OK] Dataset aplanado: {len(df)} preguntas, {df['intent'].nunique()} intenciones.")
        print(f"   Intenciones: {sorted(df['intent'].unique())}")

    return df


def cargar_y_aplanar(ruta: str) -> pd.DataFrame:
    """
    Shortcut: carga el JSON y lo aplana en un solo paso.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo .json.

    Returns
    -------
    pd.DataFrame
        Columnas: texto, intent, paso
    """
    dataset = cargar_dataset(ruta)
    df = aplanar_dataset(dataset)
    return df


# ============================================================
# Ejecución directa (para pruebas)
# ============================================================
if __name__ == "__main__":
    import sys

    # Ruta por defecto (desde la carpeta raíz del proyecto)
    ruta_json = Path(__file__).parent.parent / "02-Dataset" / "02-Dataset.json"

    if not ruta_json.exists():
        print(f"❌ No se encontró el dataset en: {ruta_json}")
        print("   Pasá la ruta como argumento: python dataset.py <ruta>")
        sys.exit(1)

    df = cargar_y_aplanar(str(ruta_json))
    print(f"\nPrimeras 5 filas:")
    print(df.head())
    print(f"\nDistribución por intención:")
    print(df["intent"].value_counts())
