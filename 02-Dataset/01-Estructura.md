A continuación se presentará una estructura propuesta básica de COMO se organizará el dataset en formato JSON, respaldando esta elección con la idea de cuando necesitemos entrenar el clasificador, con 3 líneas de Python lo "aplastan" a un DataFrame de dos columnas `(texto, intent)` y listo, scikit-learn lo consume sin drama. Lo mejor de los dos mundos.

``` JSON
{
  "tramite": "titulo_intermedio_pregrado",
  "version": "1.0",
  "fuente": "Tramitación de Diploma de Pregrado y Grado — FACENA UNNE",
  "intents": [
    {
      "intent": "que_es_titulo_intermedio",
      "paso": "general",
      "preguntas": [],
      "respuesta": ""
    }
  ]
}
```

Simple, auto-documentado y fácil de extender con nuevos trámites en el futuro solo agregando más bloques.

---
