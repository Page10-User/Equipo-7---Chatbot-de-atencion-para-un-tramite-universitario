"""
respuesta.py — Busca la respuesta textual para una intención detectada.

Carga el dataset JSON y mapea intent_id → respuesta.
"""

import json
from pathlib import Path
from typing import Optional


def cargar_dataset(ruta: str) -> list[dict]:
    """
    Carga la lista de intents desde el dataset JSON.

    Parameters
    ----------
    ruta : str
        Ruta absoluta al archivo 02-Dataset.json

    Returns
    -------
    list[dict]
        Lista de intents con campos: intent, paso, descripcion_interna,
        preguntas, respuesta
    """
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def buscar_respuesta(intent_id: str, intents: list[dict]) -> Optional[str]:
    """
    Busca la respuesta asociada a una intención.

    Parameters
    ----------
    intent_id : str
        Nombre del intent (ej: 'que_es_titulo_intermedio').
    intents : list[dict]
        Lista de intents cargada del dataset.

    Returns
    -------
    str or None
        Texto de respuesta, o None si no se encuentra el intent.
    """
    for intent in intents:
        if intent["intent"] == intent_id:
            return intent["respuesta"]
    return None
