"""
app.py — Entry point de la interfaz Streamlit.

Uso:
    streamlit run interfaz/app.py

Arquitectura:
    app.py  ←  modelo/cargador.py  ←  05-Desarrollo/clasificar.py
              modelo/respuesta.py  ←  02-Dataset/02-Dataset.json
              assets/estilo.css    (theme oscuro custom)
"""

import streamlit as st
import sys
from pathlib import Path

# Agregar interfaz/ al path para imports limpios
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    INPUT_PLACEHOLDER,
    MODELOS_DISPONIBLES,
    MODELO_DEFECTO,
    RUTA_DATASET,
    COLOR_TEXT,
    COLOR_TEXT_SEC,
    COLOR_TEXT_MUTED,
    COLOR_ACCENT,
)
from modelo.cargador import obtener_modelo, clasificar
from modelo.respuesta import cargar_dataset, buscar_respuesta

# ============================================================
# CONFIGURACIÓN INICIAL (DEBE ser lo primero de Streamlit)
# ============================================================
st.set_page_config(
    page_title=f"{APP_TITLE} - FACENA",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Inyectar CSS custom (sin comentarios /* */ para evitar que markdown los interprete)
css_path = Path(__file__).parent / "assets" / "estilo.css"
with open(css_path, "r", encoding="utf-8") as f:
    css_content = f.read()
st.html(f"<style>{css_content}</style>")

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# CACHE: recurso compartido (carga UNA vez)
# ============================================================
@st.cache_resource
def _cargar_dataset():
    return cargar_dataset(str(RUTA_DATASET))

@st.cache_resource
def _cargar_modelo(model_id: str):
    """Carga un modelo por ID y lo cachea en memoria."""
    return obtener_modelo(model_id)

# Cargar dataset al inicio
dataset_intents = _cargar_dataset()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"### 🎓 {APP_TITLE}")
    st.markdown("---")

    # Selector de modelo
    modelo_nombre = st.selectbox(
        "Modelo de clasificación",
        options=list(MODELOS_DISPONIBLES.keys()),
        index=list(MODELOS_DISPONIBLES.keys()).index(MODELO_DEFECTO),
        key="selector_modelo",
    )
    modelo_id = MODELOS_DISPONIBLES[modelo_nombre]

    # Badge de recomendado
    if modelo_nombre == MODELO_DEFECTO:
        st.markdown(
            f"<span style='color: {COLOR_ACCENT}; font-size: 0.8rem;'>"
            f"★ Modelo recomendado</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Botón limpiar chat
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# CARGA DE MODELO (con spinner la primera vez)
# ============================================================
try:
    with st.spinner(f"Cargando {modelo_nombre}..."):
        clf, vectorizador, info_modelo = _cargar_modelo(modelo_id)
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.info(
        "Ejecutá primero:  python 05-Desarrollo/modelo_final.py\n"
        "o verificá que los archivos .joblib estén en 05-Desarrollo/modelos/"
    )
    st.stop()

# ============================================================
# ENCABEZADO
# ============================================================
st.markdown(
    f"""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <h1 style="color: {COLOR_TEXT}; font-size: 1.6rem; margin-bottom: 0.3rem;">
            🎓 {APP_TITLE}
        </h1>
        <p style="color: {COLOR_TEXT_SEC}; font-size: 0.9rem; max-width: 540px;
                  margin: 0 auto;">
            {APP_SUBTITLE}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO — Bienvenida + preguntas sugeridas
# (solo se muestra si el chat está vacío)
# ============================================================
if not st.session_state.messages:

    # ---- Label ----
    st.markdown(
        f"<p style='color: {COLOR_TEXT_MUTED}; font-size: 0.75rem; "
        f"letter-spacing: 0.05em; text-align: center; margin-top: 1.5rem; "
        f"margin-bottom: 1rem;'>"
        f"PREGUNTAS SUGERIDAS</p>",
        unsafe_allow_html=True,
    )

    # ---- Helper: flujo para preguntas sugeridas ----
    def preguntar(texto: str):
        """Clasifica, busca respuesta y la agrega al historial."""
        resultado = clasificar(texto, clf, vectorizador, info_modelo)
        respuesta = buscar_respuesta(resultado["intent"], dataset_intents)
        if respuesta is None:
            respuesta = "Lo siento, no pude encontrar una respuesta para tu consulta."

        respuesta_completa = (
            f"{respuesta}\n\n---\n"
            f"*Intención: {resultado['intent']} | "
            f"Confianza: {resultado['confianza']:.1%}*"
        )

        st.session_state.messages.append({"role": "user", "content": texto})
        st.session_state.messages.append(
            {"role": "assistant", "content": respuesta_completa}
        )
        st.rerun()

    # ---- Grilla 2×2 de preguntas ----
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "📄 ¿Qué es el título intermedio?",
            type="secondary",
            use_container_width=True,
            key="sq_1",
        ):
            preguntar("¿Qué es el título intermedio?")

        if st.button(
            "📋 ¿Qué documentos necesito?",
            type="secondary",
            use_container_width=True,
            key="sq_3",
        ):
            preguntar(
                "¿Qué documentos necesito para el trámite del título intermedio?"
            )

    with col2:
        if st.button(
            "🚀 ¿Por dónde empiezo el trámite?",
            type="secondary",
            use_container_width=True,
            key="sq_2",
        ):
            preguntar("¿Por dónde empiezo el trámite del título intermedio?")

        if st.button(
            "⏱️ ¿Cuánto tarda el trámite?",
            type="secondary",
            use_container_width=True,
            key="sq_4",
        ):
            preguntar("¿Cuánto tarda el trámite del título intermedio?")

    # Hint para que escriba
    st.markdown(
        f"<p style='color: {COLOR_TEXT_MUTED}; font-size: 0.8rem; "
        f"text-align: center; margin-top: 2rem;'>"
        f"O escribí tu consulta abajo ⬇️</p>",
        unsafe_allow_html=True,
    )

# ============================================================
# CHAT — Historial de mensajes (si hay)
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# INPUT DE TEXTO (siempre visible al final)
# ============================================================
if prompt := st.chat_input(placeholder=INPUT_PLACEHOLDER, key="chat_input"):

    # 1. Mostrar pregunta del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Clasificar
    resultado = clasificar(prompt, clf, vectorizador, info_modelo)
    respuesta = buscar_respuesta(resultado["intent"], dataset_intents)
    if respuesta is None:
        respuesta = "Lo siento, no pude encontrar una respuesta para tu consulta."

    # 3. Armar respuesta
    respuesta_completa = (
        f"{respuesta}\n\n---\n"
        f"*Intención: {resultado['intent']} | "
        f"Confianza: {resultado['confianza']:.1%}*"
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": respuesta_completa}
    )

    st.rerun()
