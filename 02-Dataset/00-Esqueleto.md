Partiendo de la estructura propuesta → [[01-Estructura]]
Se elabora el siguiente esqueleto en formato .json que representará el dataset a utilizar para entrenar al agente.

``` JSON
{
  "tramite": "titulo_intermedio_pregrado",
  "version": "1.0",
  "fuente": "Tramitación de Diploma de Pregrado y Grado — FACENA UNNE",
  "descripcion": "Dataset de intenciones para el agente conversacional del trámite de Título Intermedio (Pregrado). Cada intención contiene una lista de variaciones de pregunta (mínimo 5, ideal 8) y una única respuesta basada en la documentación oficial.",
  "instrucciones": {
    "preguntas": "Escribir entre 5 y 8 variaciones naturales de cómo un estudiante preguntaría esto. Variar vocabulario, longitud y estructura. Incluir formas coloquiales.",
    "respuesta": "Redactar una respuesta clara, completa y basada en la documentación oficial. Usar lenguaje directo, como si le hablaras al estudiante."
  },
  "intents": [

    {
      "intent": "que_es_titulo_intermedio",
      "paso": "general",
      "descripcion_interna": "Preguntas sobre qué es el título intermedio, para qué sirve y quiénes pueden obtenerlo.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "inicio_tramite",
      "paso": "1",
      "descripcion_interna": "Preguntas sobre cómo y por dónde arrancar el trámite. Primer paso a dar.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "actualizar_siu",
      "paso": "1",
      "descripcion_interna": "Preguntas sobre cómo actualizar los datos personales en el SIU-Guaraní para que coincidan con la documentación.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "documentacion_requerida",
      "paso": "1",
      "descripcion_interna": "Preguntas sobre qué documentos hay que presentar en total para el trámite (vista general).",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "formato_dni",
      "paso": "1",
      "descripcion_interna": "Preguntas sobre cómo debe ser la copia del DNI: ambas caras, formato PDF, actualizado.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "libre_deuda_facena",
      "paso": "1",
      "descripcion_interna": "Preguntas sobre cómo obtener el certificado de libre deuda de la Biblioteca de FACENA.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "libre_deuda_central",
      "paso": "1",
      "descripcion_interna": "Preguntas sobre cómo obtener el certificado de libre deuda de la Biblioteca Central de la UNNE.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "foto_carnet",
      "paso": "1",
      "descripcion_interna": "Preguntas sobre los requisitos de la foto del carnet (tamaño, fondo, formato, dónde sacarla).",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "requisitos_extranjeros",
      "paso": "1",
      "descripcion_interna": "Preguntas sobre requisitos adicionales para estudiantes extranjeros o del MERCOSUR.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "continuidad_paso2",
      "paso": "2",
      "descripcion_interna": "Preguntas sobre cómo se avanza al Paso 2, qué correo llega, qué pasa si hay correcciones.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "generar_solicitud_siu",
      "paso": "2",
      "descripcion_interna": "Preguntas sobre cómo generar la nueva solicitud en el SIU: dónde ir, qué seleccionar, cómo guardarla.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "estado_solicitud",
      "paso": "2",
      "descripcion_interna": "Preguntas sobre cómo ver el estado de la solicitud, qué significa 'pendiente', dónde se ve en el SIU.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "pago_arancel",
      "paso": "2",
      "descripcion_interna": "Preguntas sobre cómo y dónde pagar el arancel del trámite.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "descarga_constancia",
      "paso": "2",
      "descripcion_interna": "Preguntas sobre cómo descargar la constancia del título en trámite una vez disponible.",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "tiempo_espera",
      "paso": "2",
      "descripcion_interna": "Preguntas sobre cuánto tarda el proceso, plazos estimados (ej: 48 horas hábiles para la constancia).",
      "preguntas": [],
      "respuesta": ""
    },

    {
      "intent": "fallback",
      "paso": "ninguno",
      "descripcion_interna": "Respuesta por defecto cuando el agente no reconoce la intención del usuario. No tiene preguntas de entrenamiento.",
      "preguntas": [],
      "respuesta": "Lo siento, no entendí tu consulta. Puedo ayudarte con información sobre el trámite del Título Intermedio (Pregrado) de FACENA-UNNE. Intentá reformular tu pregunta o consultá directamente con la secretaría de la facultad."
    }

  ]
}
```

El archivo .json está en el mismo directorio, llamado: [Dataset.json]