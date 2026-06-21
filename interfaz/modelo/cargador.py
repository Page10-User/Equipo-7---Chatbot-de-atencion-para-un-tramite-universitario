"""
cargador.py — Puente entre la interfaz y los modelos guardados en 05-Desarrollo/.

Importa y envuelve las funciones de clasificar.py, cleaner.py y vectorizer.py
para que la interfaz Streamlit no tenga que lidiar con sys.path ni rutas relativas.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------
# 1. Agregar 05-Desarrollo/ al path para poder importar sus módulos
# ---------------------------------------------------------------
RUTA_DESARROLLO = Path(__file__).parent.parent.parent / "05-Desarrollo"
if str(RUTA_DESARROLLO) not in sys.path:
    sys.path.insert(0, str(RUTA_DESARROLLO))

# ---------------------------------------------------------------
# 2. Importar desde los módulos de 05-Desarrollo
# ---------------------------------------------------------------
# Importar cleaner primero para que quede cacheado en sys.modules
# (clasificar.py también lo importa, así evitamos búsqueda doble)
from cleaner import preprocesar

from clasificar import (
    cargar_metadata as _cargar_metadata,
    cargar_modelo as _cargar_modelo_desarrollo,
    clasificar as _clasificar_desarrollo,
)

# ---------------------------------------------------------------
# 3. Funciones públicas para la interfaz
# ---------------------------------------------------------------


def obtener_metadata() -> dict:
    """
    Carga y devuelve el metadata.json con info de todos los modelos.

    Returns
    -------
    dict
        Contenido completo de metadata.json
    """
    return _cargar_metadata()


def obtener_modelo(nombre_modelo: str):
    """
    Carga un modelo por su nombre corto ('svm', 'nb', 'svm_tfidf').

    Parameters
    ----------
    nombre_modelo : str
        Identificador del modelo (svm, nb, svm_tfidf)

    Returns
    -------
    tuple
        (clf, vectorizador, info)
        clf: clasificador (SVC, MultinomialNB, etc.)
        vectorizador: TfidfVectorizer o SentenceTransformer
        info: dict con metadatos del modelo
    """
    metadata = _cargar_metadata()
    return _cargar_modelo_desarrollo(nombre_modelo, metadata)


def clasificar(texto: str, clf, vectorizador, info: dict) -> dict:
    """
    Clasifica una pregunta y devuelve el intent detectado.

    Parameters
    ----------
    texto : str
        Pregunta del usuario.
    clf : estimator
        Clasificador cargado.
    vectorizador : TfidfVectorizer o SentenceTransformer
        Vectorizador correspondiente.
    info : dict
        Metadata del modelo.

    Returns
    -------
    dict
        { 'texto', 'intent', 'confianza', 'modelo' }
    """
    return _clasificar_desarrollo(texto, clf, vectorizador, info)
