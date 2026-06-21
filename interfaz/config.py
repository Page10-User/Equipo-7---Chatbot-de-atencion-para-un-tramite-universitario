"""
config.py — Configuración centralizada de la interfaz Streamlit.

Rutas, paleta de colores, opciones de modelos y constantes.
Todo lo que cambie de valor va acá, no en app.py.
"""

from pathlib import Path

# ============================================================
# Rutas del proyecto (relativas desde interfaz/)
# ============================================================
RUTA_INTERFAZ = Path(__file__).parent
RUTA_PROYECTO = RUTA_INTERFAZ.parent          # raíz del repo
RUTA_DESARROLLO = RUTA_PROYECTO / "05-Desarrollo"
RUTA_MODELOS = RUTA_DESARROLLO / "modelos"
RUTA_METADATA = RUTA_MODELOS / "metadata.json"
RUTA_DATASET = RUTA_PROYECTO / "02-Dataset" / "02-Dataset.json"

# ============================================================
# Nombres para mostrar vs IDs internos de modelos
# ============================================================
MODELOS_DISPONIBLES = {
    "SVM + Embeddings [GANADOR]": "svm",  # ganador
    "Naive Bayes + TF-IDF": "nb",
    "SVM Lineal + TF-IDF": "svm_tfidf",
}
MODELO_DEFECTO = "SVM + Embeddings [GANADOR]"

# ============================================================
# Paleta de colores (del diseño original del usuario)
# ============================================================
COLOR_BG = "#0B0F19"
COLOR_SIDEBAR_BG = "#0D1117"
COLOR_TEXT = "#FFFFFF"
COLOR_TEXT_SEC = "#9CA3AF"
COLOR_TEXT_MUTED = "#6B7280"
COLOR_ACCENT = "#60A5FA"
COLOR_ACCENT_BG = "rgba(59, 130, 246, 0.15)"
COLOR_BORDER = "rgba(255, 255, 255, 0.05)"
COLOR_INPUT_BG = "rgba(255, 255, 255, 0.02)"
COLOR_HOVER = "rgba(255, 255, 255, 0.02)"
COLOR_USER_BUBBLE = "rgba(59, 130, 246, 0.15)"
COLOR_ASSISTANT_BUBBLE = "rgba(255, 255, 255, 0.03)"

# ============================================================
# Textos estáticos
# ============================================================
APP_TITLE = "Asistente de Trámites"
APP_SUBTITLE = (
    "Soy tu asistente virtual. Puedo ayudarte con información "
    "sobre trámites, requisitos, plazos y procedimientos universitarios."
)
INPUT_PLACEHOLDER = "Escribe tu pregunta sobre el trámite..."
