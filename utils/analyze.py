# -*- coding: utf-8 -*-
"""
Análisis de la entrevista con un LLM real de HuggingFace Inference.

Arquitectura:
  1. Se construye un prompt de sistema con el protocolo de imparcialidad.
  2. Se envía (perfil del puesto + transcripción) al modelo seleccionado.
  3. Se extrae el JSON de la respuesta con tolerancia a errores comunes.
  4. Se "sanitiza" el resultado: se garantizan las claves, se acotan los
     puntajes y SE RECALCULA el puntaje global ponderado con los pesos del
     perfil (nunca se confía en el número que el modelo declare).
  5. Si ningún modelo entrega JSON válido, se activa el modo de respaldo
     cualitativo (texto libre) en lugar de inventar datos.

Protocolo de imparcialidad: alineado con el marco del libro "Sesgos
inconscientes" de Pamela Fuller, Anne Chow y Mark Murphy (FranklinCovey):
Identificar el sesgo, Reformularlo, Cultivar la conexión y Elegir el coraje.
"""

import json
import re

from huggingface_hub import InferenceClient

# Modelos candidatos (el elegido va primero; si falla, se prueban los demás).
MODELOS_LLM = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
]

# Caracteres de transcripción que se envían al modelo (entrevistas largas).
LIMITE_TRANSCRIPCION = 12000

# Referencia metodológica del protocolo anti-sesgos.
MARCO_SESGOS = (
    "Pamela Fuller, Anne Chow y Mark Murphy — 'Sesgos inconscientes' "
    "(FranklinCovey): Identificar el sesgo, Reformularlo, Cultivar la conexión "
    "y Elegir el coraje."
)

RECOMENDACIONES_VALIDAS = [
    "Apoya con confianza",
    "Apoya con reservas",
    "Requiere más evidencia",
    "No apoya",
]

CONFIANZA_VALIDAS = ["Alta", "Media", "Baja"]

SYSTEM_PROMPT = f"""Eres un evaluador estructurado de entrevistas laborales que APOYA (no reemplaza) la decisión de un reclutador humano de Recursos Humanos.

Tu tarea: leer la transcripción de una entrevista y puntuar CADA competencia del perfil del puesto (0-100) usando ÚNICAMENTE evidencia verbal de la transcripción. No tienes acceso a la foto, al currículo ni a ninguna otra fuente: SOLO la transcripción.

PROTOCOLO DE IMPARCIALIDAD OBLIGATORIO (basado en el marco de {MARCO_SESGOS}):

1. IDENTIFICAR EL SESGO Y REFORMULARLO. Antes de puntuar, revisa tu propio razonamiento contra esta lista de sesgos y aplícale el control indicado:
   - Efecto halo / cuerno: puntúa cada competencia de forma INDEPENDIENTE, con su propia evidencia. Un dato muy bueno o muy malo no contamina las demás competencias.
   - Primacía y recencia: usa TODA la transcripción por igual. Los primeros y los últimos minutos no pesan más.
   - Contraste: compara SOLO contra el perfil del puesto. Nunca contra otros candidatos (no tienes acceso a ellos).
   - Afinidad / espejo: prohibido favorecer perfiles "parecidos a" el evaluador. Estereotipos de género, edad, origen, acento, religión o discapacidad: BLOQUEADOS como criterio.
   - Confirmación: registra SIEMPRE evidencia a favor Y en contra de cada competencia.
   - Atribución: considera el contexto narrado (despidos, reubicaciones, carencias) sin inventar culpas ni justificarlas automáticamente.
   - Anclaje: el primer dato (salario, universidad, empresa previa) NO fija el juicio del resto de la evaluación.
   - Disponibilidad: calibra con el conjunto completo de la evidencia, no con los ejemplos más vívidos o memorables.
   - Aprobación / ingratiation: simpatía, carisma y fluidez verbal NO son evidencia de competencia.
   - Pedigree: la marca de una universidad o empleador famoso NO suma por sí sola.
   - "Culture fit": PROHIBIDO como criterio. Solo se evalúan las competencias del perfil.
   - Pseudopredictores: no uses como evidencia cartas genéricas, años de graduación ni datos que no estén en la transcripción.
2. FALSAS EXPECTATIVAS: prohibido predecir desempeño futuro. Describe solo lo que la evidencia muestra, e indica en cada competencia su confianza de evidencia (Alta / Media / Baja).
3. CULTIVAR LA CONEXIÓN: evalúa considerando que el candidato respondió en una situación de presión; valora el contenido de las respuestas, no el estilo. Si la transcripción muestra que el ENTREVISTADOR hizo preguntas capciosas, interrumpió, buscó afinidad o indujo respuestas, repórtalo en "riesgos" con el prefijo "Sesgo del entrevistador (nombre del sesgo):".
4. ELEGIR EL CORAJE: si la evidencia es insuficiente, dilo. Puntúa bajo-intermedio con confianza "Baja" y explica qué evidencia faltaría. No inventes ni rellenes.
5. EVIDENCIA: cita o parafrasea el momento concreto de la entrevista que sustenta cada puntaje (qué dijo el candidato).

FORMATO DE SALIDA: responde ÚNICAMENTE con un JSON válido (puedes envolverlo en ```json). Estructura exacta:
{{
  "resumen_ejecutivo": "3-4 frases con la visión general del contenido de la entrevista",
  "recomendacion_apoyo": "una de: Apoya con confianza | Apoya con reservas | Requiere más evidencia | No apoya",
  "justificacion": "por qué se recomienda eso, citando el conjunto de evidencia",
  "competencias": [
    {{
      "nombre": "debe coincidir EXACTAMENTE con el nombre de la competencia del perfil",
      "puntaje": 0-100,
      "nivel": "etiqueta breve del puntaje",
      "confianza_evidencia": "Alta | Media | Baja",
      "evidencia": "cita o paráfrasis textual del candidato que sustenta el puntaje",
      "evidencia_contraria": "evidencia que juega en contra o matiza (si no hay, dilo)",
      "observacion": "matiz, contexto o qué verificar en una siguiente etapa"
    }}
  ],
  "fortalezas": ["3 a 5 fortalezas concretas basadas en la evidencia"],
  "riesgos": ["3 a 5 riesgos o puntos a verificar; incluye aquí los 'Sesgo del entrevistador (...)' si los detectaste"],
  "limitaciones": ["2 a 4 limitaciones de este análisis (ej.: solo contenido verbal, duración, calidad de audio)"]
}}
No agregues texto fuera del JSON."""

PROMPT_FALLBACK = (
    "Además, entrega una evaluación cualitativa breve por competencia en texto "
    "plano (sin JSON) explicando evidencia a favor y en contra."
)


def _extraer_json(texto: str) -> dict:
    """Extrae el primer objeto JSON válido de una respuesta del modelo.

    Tolera: cercos de código ```json ... ```, texto antes/después del JSON,
    y comas colgantes antes de } o ].
    """
    if not texto or not texto.strip():
        raise ValueError("Respuesta vacía del modelo.")

    limpio = texto.strip()

    # 1) Si hay cerco de código, extraer su contenido.
    cercos = re.findall(r"```(?:json)?\s*(.*?)```", limpio, flags=re.DOTALL)
    if cercos:
        limpio = cercos[0].strip()

    # 2) Recortar desde la primera llave hasta la última.
    inicio, fin = limpio.find("{"), limpio.rfind("}")
    if inicio == -1 or fin == -1 or fin <= inicio:
        raise ValueError("No se encontró un objeto JSON en la respuesta.")
    candidato = limpio[inicio : fin + 1]

    # 3) Intentar parsear; si falla, quitar comas colgantes y reintentar.
    try:
        return json.loads(candidato)
    except json.JSONDecodeError:
        sin_comas = re.sub(r",\s*([}\]])", r"\1", candidato)
        return json.loads(sin_comas)


def _como_texto(valor, max_len: int = 1200) -> str:
    """Convierte cualquier valor del modelo en texto plano seguro."""
    if valor is None:
        return ""
    if isinstance(valor, str):
        texto = valor.strip()
    elif isinstance(valor, list):
        texto = " · ".join(str(v).strip() for v in valor if str(v).strip())
    else:
        texto = str(valor).strip()
    return texto[:max_len]


def _como_lista(valor) -> list:
    """Convierte el campo en lista de strings (tolera string único o lista mixta)."""
    if valor is None:
        return []
    if isinstance(valor, str):
        partes = [p.strip(" -•\t") for p in re.split(r"[\n;]+", valor)]
        return [p for p in partes if p][:8]
    if isinstance(valor, list):
        salida = []
        for item in valor[:8]:
            texto = _como_texto(item, 400)
            if texto:
                salida.append(texto)
        return salida
    return [_como_texto(valor, 400)]


def _nivel_por_puntaje(puntaje: int) -> str:
    if puntaje >= 80:
        return "Fuerte"
    if puntaje >= 65:
        return "Adecuado"
    if puntaje >= 50:
        return "A verificar"
    return "Insuficiente"


def _recomendacion_por_puntaje(puntaje: int) -> str:
    if puntaje >= 75:
        return "Apoya con confianza"
    if puntaje >= 60:
        return "Apoya con reservas"
    if puntaje >= 45:
        return "Requiere más evidencia"
    return "No apoya"


def _sanitizar(datos: dict, perfil: dict) -> dict:
    """Garantiza estructura, acota valores y RECALCULA el puntaje ponderado."""
    datos = datos if isinstance(datos, dict) else {}

    # --- Competencias: una por cada competencia del perfil, con su peso real ---
    por_nombre = {}
    for comp in datos.get("competencias") or []:
        if isinstance(comp, dict):
            clave = str(comp.get("nombre") or "").strip().casefold()
            if clave:
                por_nombre[clave] = comp

    competencias_final = []
    pesos, puntajes = [], []
    for comp_perfil in perfil.get("competencias") or []:
        nombre = str(comp_perfil.get("nombre") or "").strip()
        if not nombre:
            continue
        try:
            peso = float(comp_perfil.get("peso") or 0)
        except (TypeError, ValueError):
            peso = 0.0
        comp_llm = por_nombre.get(nombre.casefold(), {})

        try:
            puntaje = int(round(float(comp_llm.get("puntaje", 0) or 0)))
        except (TypeError, ValueError):
            puntaje = 0
        puntaje = max(0, min(100, puntaje))

        confianza = str(comp_llm.get("confianza_evidencia") or "").strip().capitalize()
        if confianza not in CONFIANZA_VALIDAS:
            confianza = "Baja"

        competencias_final.append(
            {
                "nombre": nombre,
                "peso": peso,
                "puntaje": puntaje,
                "nivel": _como_texto(comp_llm.get("nivel"), 40) or _nivel_por_puntaje(puntaje),
                "confianza_evidencia": confianza,
                "evidencia": _como_texto(comp_llm.get("evidencia")) or "Sin evidencia reportada por el modelo.",
                "evidencia_contraria": _como_texto(comp_llm.get("evidencia_contraria")) or "No reportada.",
                "observacion": _como_texto(comp_llm.get("observacion")) or "Sin observaciones.",
            }
        )
        pesos.append(peso)
        puntajes.append(puntaje)

    # --- Puntaje global: SIEMPRE recalculado con los pesos del perfil ---
    if pesos and sum(pesos) > 0:
        puntaje_global = int(round(sum(p * j for p, j in zip(pesos, puntajes)) / sum(pesos)))
    elif puntajes:
        puntaje_global = int(round(sum(puntajes) / len(puntajes)))
    else:
        puntaje_global = 0
    puntaje_global = max(0, min(100, puntaje_global))

    recomendacion = _como_texto(datos.get("recomendacion_apoyo"), 60)
    if recomendacion not in RECOMENDACIONES_VALIDAS:
        recomendacion = _recomendacion_por_puntaje(puntaje_global)

    return {
        "modo_fallback": False,
        "modelo": str(datos.get("modelo") or ""),
        "puntaje_global": puntaje_global,
        "recomendacion_apoyo": recomendacion,
        "resumen_ejecutivo": _como_texto(datos.get("resumen_ejecutivo"), 900),
        "justificacion": _como_texto(datos.get("justificacion"), 1200),
        "competencias": competencias_final,
        "fortalezas": _como_lista(datos.get("fortalezas")),
        "riesgos": _como_lista(datos.get("riesgos")),
        "limitaciones": _como_lista(datos.get("limitaciones"))
        or [
            "Análisis basado únicamente en el contenido verbal de la transcripción.",
            "Los modelos de lenguaje pueden equivocarse: verifica la evidencia citada antes de decidir.",
        ],
    }


def _llamar_llm(token: str, modelo: str, system_prompt: str, user_prompt: str) -> str:
    """Llama al modelo vía la Inference API de HuggingFace y devuelve el texto."""
    try:
        cliente = InferenceClient(api_key=token)
    except TypeError:  # versiones antiguas de huggingface_hub
        cliente = InferenceClient(token=token)

    respuesta = cliente.chat_completion(
        model=modelo,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
        temperature=0.2,
    )
    contenido = respuesta.choices[0].message.content
    if not contenido or not str(contenido).strip():
        raise ValueError("El modelo devolvió una respuesta vacía.")
    return str(contenido)


def analizar_entrevista(token: str, modelo: str, perfil: dict, transcripcion: str) -> dict:
    """Ejecuta el análisis completo. Nunca lanza excepción salvo falta de token.

    Si ningún modelo entrega JSON válido, devuelve modo_fallback=True con el
    texto cualitativo del último modelo (o el detalle de los errores).
    """
    if not (token or "").strip():
        raise ValueError(
            "Falta el token de HuggingFace: pégalo en la barra lateral o "
            "configúralo como variable de entorno / secret (HF_TOKEN)."
        )

    transcripcion = (transcripcion or "").strip()
    if len(transcripcion) < 200:
        raise ValueError("La transcripción es demasiado corta para analizar (mínimo 200 caracteres).")

    recortada = transcripcion[:LIMITE_TRANSCRIPCION]
    aviso_recorte = (
        f"\n\n[NOTA: la transcripción superó {LIMITE_TRANSCRIPCION} caracteres y fue "
        "recortada. Analiza SOLO lo entregado y repórtalo en 'limitaciones'.]"
        if len(transcripcion) > LIMITE_TRANSCRIPCION
        else ""
    )

    competencias_txt = "\n".join(
        f"- {c.get('nombre', '')} (peso {c.get('peso', 0)}%): {c.get('descripcion', '')}"
        for c in perfil.get("competencias") or []
        if str(c.get("nombre") or "").strip()
    )
    user_prompt = f"""PERFIL DEL PUESTO
Cargo: {perfil.get('cargo', 'N/D')}
Área: {perfil.get('area', 'N/D')}
Descripción: {perfil.get('descripcion', 'N/D')}
Requisitos: {perfil.get('requisitos', 'N/D')}
Competencias a evaluar (con su peso para el promedio ponderado):
{competencias_txt or '(sin competencias definidas)'}

TRANSCRIPCIÓN DE LA ENTREVISTA (contenido verbal; el entrevistador habla también)
{recortada}{aviso_recorte}

Aplica el protocolo de imparcialidad completo y responde SOLO con el JSON pedido."""

    # El modelo elegido primero; los demás como respaldo automático.
    orden = [modelo] + [m for m in MODELOS_LLM if m != modelo]
    errores, ultimo_texto = [], ""

    for candidato in orden:
        try:
            texto = _llamar_llm(token.strip(), candidato, SYSTEM_PROMPT, user_prompt)
            ultimo_texto = texto
            datos = _extraer_json(texto)
            datos["modelo"] = candidato
            return _sanitizar(datos, perfil)
        except Exception as e:  # 403/503/429/JSON inválido → siguiente modelo
            errores.append(f"{candidato}: {e}")
            continue

    # --- Modo de respaldo cualitativo ---
    detalle = "\n".join(f"- {e}" for e in errores) or "- Sin detalle disponible."
    cuerpo = ultimo_texto.strip() if ultimo_texto and len(ultimo_texto.strip()) > 80 else (
        "El análisis estructurado no estuvo disponible en este momento (los "
        "modelos de la Inference API respondieron con error o sin JSON válido). "
        "Vuelve a intentarlo en unos minutos; si persiste, prueba otro modelo "
        "en la barra lateral.\n\nErrores registrados:\n" + detalle
    )
    return {
        "modo_fallback": True,
        "texto": cuerpo,
        "modelo": "",
        "puntaje_global": 0,
        "recomendacion_apoyo": "N/D",
        "resumen_ejecutivo": "",
        "justificacion": "",
        "competencias": [],
        "fortalezas": [],
        "riesgos": [],
        "limitaciones": ["Análisis no estructurado (modo de respaldo)."],
    }
