# -*- coding: utf-8 -*-
"""
Asistente IA de apoyo a la selección de personal — v2.

Aplicación Streamlit de 5 pasos:
  1. Perfil del puesto (plantillas + competencias ponderadas)
  2. Entrevista (transcripción local con Whisper, editable a mano)
  3. Foto del candidato/a (SOLO registro de expediente; NUNCA se envía a modelos)
  4. Análisis con un LLM real (HuggingFace Inference) bajo protocolo de imparcialidad
  5. Decisión humana + informe (PDF / Markdown / TXT) con firma del entrevistador/a R.H.

Protocolo de imparcialidad en 3 FASES, alineado con el libro "Sesgos
inconscientes" de Pamela Fuller, Anne Chow y Mark Murphy (FranklinCovey):
  FASE 1 · Identificar el sesgo y reformularlo   (checklist antes de analizar)
  FASE 2 · Cultivar la conexión                  (checklist antes de decidir)
  FASE 3 · Elegir el coraje                      (autorevisión de 15 sesgos
                                                  + declaración final obligatoria)

La decisión final siempre es de una persona (human-in-the-loop).
Diseño de IA por Ing. Hernán Quiroz.
"""

import os
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.analyze import (
    LIMITE_TRANSCRIPCION,
    MARCO_SESGOS,
    MODELOS_LLM,
    analizar_entrevista,
)
from utils.report import CREDITO_DISENO, generar_reporte_md
from utils.report_pdf import generar_reporte_pdf
from utils.templates import PLANTILLAS, obtener_plantilla
from utils.transcribe import MODELOS_WHISPER, cargar_modelo, transcribir_audio

st.set_page_config(
    page_title="Asistente IA · Selección de personal",
    page_icon="🧭",
    layout="wide",
)

# ------------------------------- Constantes ----------------------------------

IDIOMAS = {
    "Detección automática": None,
    "Español (es)": "es",
    "Inglés (en)": "en",
    "Portugués (pt)": "pt",
    "Francés (fr)": "fr",
}

# FASE 1 · Identificar el sesgo y reformularlo (antes de analizar)
CHECKS_FASE1 = [
    "Efecto halo/cuerno: evaluaré cada competencia por separado, sin dejar que "
    "una impresión general (muy buena o muy mala) contamine al resto.",
    "Primacía y recencia: daré el mismo peso a toda la entrevista, no solo al "
    "primer minuto ni a lo último que se habló.",
    "Afinidad/espejo: no premiaré al candidato/a por parecerse a mí (género, "
    "edad, origen, acento, escuela, gustos).",
    "Anclaje: los datos iniciales (sueldo actual, universidad, empresa famosa) "
    "no fijarán mi juicio sobre todo lo demás.",
    "Confirmación: anotaré evidencia a favor Y en contra de cada competencia, "
    "aunque incomode.",
    "'Culture fit' y pseudopredictores quedan fuera de la evaluación: solo "
    "cuentan las competencias del perfil.",
]

# FASE 2 · Cultivar la conexión (antes de decidir)
CHECKS_FASE2 = [
    "Escuché la situación del candidato/a desde su contexto (presión, nervios, "
    "recursos disponibles), valorando el contenido de sus respuestas y no el estilo.",
    "Di espacio a las respuestas largas y no induje las respuestas que yo quería "
    "oír (sin preguntas capciosas).",
    "Si la IA reportó 'Sesgo del entrevistador' en la sección de riesgos, lo "
    "reconozco y lo tendré en cuenta al decidir.",
    "Trataré a este candidato/a con el mismo respeto sea cual sea el resultado "
    "(conectar no es simpatizar).",
]

# FASE 3 · Elegir el coraje: autorevisión final de la decisión (15 sesgos)
CHECKS_FASE3 = [
    "Halo/cuerno: mi decisión no está dominada por UNA sola anécdota buena o mala.",
    "Primacía: no decidí por la primera impresión del inicio de la entrevista.",
    "Recencia: no decidí solo por lo último que se conversó.",
    "Contraste: comparé contra el perfil del puesto, no contra el candidato anterior.",
    "Afinidad: no favorecí similitudes conmigo (origen, escuela, intereses).",
    "Estereotipos: no asumí capacidades por género, edad, origen, acento, "
    "religión o discapacidad.",
    "Confirmación: revisé deliberadamente evidencia en contra de mi hipótesis inicial.",
    "Atribución: consideré el contexto de los fracasos narrados sin inventar "
    "culpas ni justificarlas automáticamente.",
    "Anclaje: el primer dato (sueldo, universidad, empresa) no ancló el resto "
    "de la evaluación.",
    "Disponibilidad: puse en el centro la evidencia completa, no el ejemplo más "
    "memorable.",
    "Aprobación: la simpatía, el carisma y la fluidez verbal no suman como "
    "competencia.",
    "Pedigree: la marca de una universidad o empleador famoso no suma por sí sola.",
    "'Culture fit': no usé el 'encaje cultural' como criterio de decisión.",
    "Pseudopredictores: no usé datos irrelevantes (cartas genéricas, años de "
    "graduación) como evidencia.",
    "Falsas expectativas: no predije desempeño futuro; describí solo la "
    "evidencia actual.",
]

GUIA_STAR = """
**Método STAR** — pide ejemplos REALES (no hipotéticos) y anota evidencia a favor y en contra:

| Paso | Pregunta tipo |
|---|---|
| **S** — Situación | "Cuéntame una situación concreta en la que tuvieras que…" |
| **T** — Tarea | "¿Cuál era tu responsabilidad exacta en ese caso?" |
| **A** — Acción | "¿Qué hiciste TÚ, paso a paso?" |
| **R** — Resultado | "¿Qué pasó al final? ¿Cómo lo supiste?" |

💡 Si el candidato/a responde en generalidades, devuelve siempre a un caso concreto:
"¿Puedes darme un ejemplo real de la última vez que te pasó?"
"""


# ------------------------------- Helpers --------------------------------------


def _token() -> str:
    """Resuelve el token de HuggingFace: barra lateral → variable de entorno → secrets."""
    tok = (st.session_state.get("hf_token") or "").strip()
    if not tok:
        tok = (os.environ.get("HF_TOKEN") or "").strip()
    if not tok:
        try:
            tok = str(st.secrets.get("HF_TOKEN") or "").strip()
        except Exception:
            tok = ""
    return tok


@st.cache_resource(show_spinner=False)
def _modelo_whisper(nombre: str):
    """Carga el modelo Whisper una sola vez y lo reutiliza entre sesiones."""
    return cargar_modelo(nombre)


def _perfil_actual() -> dict:
    if "perfil" not in st.session_state:
        st.session_state["perfil"] = obtener_plantilla("Genérico / Adaptable")
    return st.session_state["perfil"]


def _texto_transcripcion() -> str:
    return (st.session_state.get("transcripcion") or "").strip()


def _duracion_txt(info: dict) -> str:
    if not info:
        return "N/D"
    try:
        d = float(info.get("duracion"))
    except (TypeError, ValueError):
        return "N/D"
    if d >= 60:
        return f"{d / 60:.1f} min"
    return f"{d:.0f} s"


def _checklist(prefijo: str, items: list) -> bool:
    """Dibuja una lista de casillas persistente; True solo si TODAS están marcadas."""
    todas = True
    for i, texto in enumerate(items):
        if not st.checkbox(texto, key=f"{prefijo}_{i}"):
            todas = False
    return todas


def _slug(texto: str) -> str:
    base = (texto or "entrevista").strip().lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in base)
    return (slug.strip("-")[:40] or "entrevista")


def _renderizar_analisis(analisis: dict) -> None:
    """Muestra el resultado del análisis (estructurado o de respaldo)."""
    if analisis.get("modo_fallback"):
        st.warning(
            "El modelo no devolvió un JSON válido: se muestra el análisis "
            "cualitativo de respaldo, SIN puntajes. Puedes reintentar con otro "
            "modelo en la barra lateral."
        )
        st.markdown(analisis.get("texto", ""))
        return

    c1, c2 = st.columns(2)
    c1.metric("Puntaje global ponderado", f"{analisis.get('puntaje_global', 0)}/100")
    c2.metric("Recomendación de apoyo", analisis.get("recomendacion_apoyo", "N/D"))

    if analisis.get("resumen_ejecutivo"):
        st.markdown(f"**Resumen ejecutivo:** {analisis['resumen_ejecutivo']}")
    if analisis.get("justificacion"):
        st.markdown(f"**Justificación:** {analisis['justificacion']}")

    st.markdown("##### Evaluación por competencias")
    for c in analisis.get("competencias") or []:
        with st.container(border=True):
            fila = st.columns([3, 1])
            fila[0].markdown(f"**{c.get('nombre', 'Competencia')}** · peso {c.get('peso', 0)}%")
            fila[1].markdown(f"**{c.get('puntaje', 0)}/100** — {c.get('nivel', '')}")
            try:
                st.progress(min(max(int(c.get("puntaje", 0) or 0), 0), 100) / 100.0)
            except Exception:
                pass
            st.caption(f"Confianza de la evidencia: {c.get('confianza_evidencia', 'Baja')}")
            with st.expander("Evidencia a favor / en contra y observación"):
                st.markdown(f"- **A favor:** {c.get('evidencia', 'N/D')}")
                st.markdown(f"- **En contra o matiz:** {c.get('evidencia_contraria', 'N/D')}")
                st.markdown(f"- **Observación:** {c.get('observacion', 'N/D')}")

    colf, colr = st.columns(2)
    with colf:
        st.markdown("##### Fortalezas")
        for f in analisis.get("fortalezas") or ["Sin fortalezas reportadas."]:
            st.markdown(f"- {f}")
    with colr:
        st.markdown("##### Riesgos / puntos a verificar")
        for r in analisis.get("riesgos") or ["Sin riesgos reportados."]:
            st.markdown(f"- {r}")

    with st.expander("Limitaciones declaradas del análisis"):
        for l in analisis.get("limitaciones") or []:
            st.markdown(f"- {l}")
    st.caption(f"Modelo que produjo el análisis: {analisis.get('modelo') or 'N/D'}")


# ------------------------- Encabezado y barra lateral -------------------------

st.title("🧭 Asistente IA de apoyo a la selección de personal")
st.caption(
    f"{CREDITO_DISENO} · Transcripción local con Whisper · Análisis con LLM real "
    "(HuggingFace Inference) · La decisión final siempre es humana"
)

with st.sidebar:
    st.title("⚙️ Configuración")
    st.text_input(
        "Token de HuggingFace",
        type="password",
        key="hf_token",
        help="Se usa SOLO para llamar al LLM de análisis. En Streamlit Cloud "
        "puedes dejarlo vacío y configurarlo como secret (HF_TOKEN).",
    )
    modelo_whisper = st.selectbox(
        "Modelo Whisper (transcripción local)",
        MODELOS_WHISPER,
        index=1,
        help="En Streamlit Cloud (~1 GB de RAM) usa 'base'. En tu propio PC "
        "puedes usar 'small' o 'medium' para más precisión.",
    )
    modelo_llm = st.selectbox("Modelo LLM (análisis)", MODELOS_LLM, index=0)

    st.divider()
    st.markdown("### Protocolo de imparcialidad")
    st.markdown(
        "Alineado con el libro *Sesgos inconscientes* (Fuller, Chow y Murphy — "
        "FranklinCovey):\n\n"
        "1. **Identificar** el sesgo y reformularlo\n"
        "2. **Cultivar** la conexión\n"
        "3. **Elegir** el coraje"
    )
    st.caption(CREDITO_DISENO)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1️⃣ Perfil del puesto",
        "2️⃣ Entrevista",
        "3️⃣ Foto (expediente)",
        "4️⃣ Análisis IA",
        "5️⃣ Decisión e informe",
    ]
)

# ====================== PASO 1 · PERFIL DEL PUESTO ============================

with tab1:
    st.subheader("Paso 1 · Perfil del puesto")
    st.text_input(
        "Nombre de la empresa",
        key="empresa_nombre",
        placeholder="Ej.: Comercial Andina S.A.",
    )

    colp, colb = st.columns([2, 1])
    plantilla = colp.selectbox(
        "Partir de una plantilla (opcional)", ["(elegir)"] + list(PLANTILLAS.keys())
    )
    if colb.button("📂 Cargar plantilla", use_container_width=True):
        if plantilla != "(elegir)":
            st.session_state["perfil"] = obtener_plantilla(plantilla)
            for k in ("in_cargo", "in_area", "in_desc", "in_req", "in_comps"):
                st.session_state.pop(k, None)
            st.rerun()
        else:
            st.warning("Elige primero una plantilla de la lista.")

    perfil = _perfil_actual()

    colc, cola = st.columns(2)
    cargo = colc.text_input("Cargo", value=perfil.get("cargo", ""), key="in_cargo")
    area = cola.text_area("Área / departamento", value=perfil.get("area", ""), key="in_area", height=42)
    descripcion = st.text_area(
        "Descripción del puesto", value=perfil.get("descripcion", ""), key="in_desc", height=90
    )
    requisitos = st.text_area(
        "Requisitos", value=perfil.get("requisitos", ""), key="in_req", height=90
    )

    st.markdown("**Competencias del perfil** (el puntaje global es un promedio ponderado)")
    df_comp = st.data_editor(
        pd.DataFrame(perfil.get("competencias") or [], columns=["nombre", "descripcion", "peso"]),
        column_config={
            "nombre": st.column_config.TextColumn("Competencia", width="medium", required=True),
            "descripcion": st.column_config.TextColumn("Qué evalúa (evidencia verbal)", width="large"),
            "peso": st.column_config.NumberColumn("Peso %", min_value=0, max_value=100, step=5),
        },
        num_rows="dynamic",
        key="in_comps",
        use_container_width=True,
    )

    # Sincronización automática del perfil con lo que se ve en pantalla.
    _competencias = []
    for _, fila in df_comp.iterrows():
        nombre = str(fila.get("nombre") or "").strip()
        if not nombre:
            continue
        try:
            peso = float(fila.get("peso") or 0)
        except (TypeError, ValueError):
            peso = 0.0
        _competencias.append(
            {
                "nombre": nombre,
                "descripcion": str(fila.get("descripcion") or "").strip(),
                "peso": peso,
            }
        )
    st.session_state["perfil"] = {
        "cargo": (cargo or "").strip(),
        "area": (area or "").strip(),
        "descripcion": (descripcion or "").strip(),
        "requisitos": (requisitos or "").strip(),
        "competencias": _competencias,
    }
    _total = sum(c["peso"] for c in _competencias)
    st.caption(
        f"Competencias activas: {len(_competencias)} · Suma de pesos: {_total:.0f}% "
        "(si no suma 100, el promedio ponderado se ajusta proporcionalmente; "
        "el perfil se guarda automáticamente en la sesión)."
    )

    with st.expander("🌟 Guía STAR para entrevistar (método recomendado)"):
        st.markdown(GUIA_STAR)

# ====================== PASO 2 · ENTREVISTA ===================================

with tab2:
    st.subheader("Paso 2 · Entrevista")
    st.info(
        "El audio se procesa DENTRO de la propia aplicación con Whisper "
        "(nada del audio viaja a servicios externos de transcripción). "
        "En Streamlit Cloud el procesamiento ocurre en el contenedor de la app.",
        icon="🔒",
    )

    archivo = st.file_uploader(
        "Audio o video de la entrevista",
        type=["mp3", "wav", "m4a", "ogg", "flac", "aac", "mp4", "webm"],
    )
    col_id, col_mod = st.columns(2)
    idioma_txt = col_id.selectbox("Idioma de la entrevista", list(IDIOMAS.keys()))
    col_mod.caption(f"Modelo Whisper activo: **{modelo_whisper}** (cámbialo en la barra lateral)")

    if archivo and st.button("🎙️ Transcribir entrevista", type="primary", use_container_width=True):
        with st.status("Transcribiendo con Whisper (puede tardar varios minutos)…", expanded=True) as estado:
            st.write(f"Archivo: {archivo.name} · Modelo: {modelo_whisper}")
            ext = os.path.splitext(archivo.name)[1].lower() or ".mp3"
            ruta = os.path.join(
                tempfile.gettempdir(), f"entrevista_{datetime.now().strftime('%H%M%S')}{ext}"
            )
            try:
                with open(ruta, "wb") as f:
                    f.write(archivo.getvalue())
                modelo = _modelo_whisper(modelo_whisper)
                barra = st.progress(0.0, text="Transcribiendo…")

                def _cb(seg, info):
                    try:
                        if info and info.duration:
                            barra.progress(
                                min(seg.end / info.duration, 1.0),
                                text=f"Transcribiendo… {seg.end / 60:.0f} de {info.duration / 60:.0f} min",
                            )
                    except Exception:
                        pass  # la barra nunca rompe la transcripción

                texto, segmentos, info = transcribir_audio(ruta, modelo, IDIOMAS[idioma_txt], _cb)
                st.session_state["transcripcion"] = texto
                st.session_state["ta_transcripcion"] = texto  # refresca el cuadro editable
                st.session_state["segmentos"] = segmentos
                st.session_state["info_audio"] = {
                    "nombre": archivo.name,
                    "duracion": getattr(info, "duration", None),
                    "idioma": getattr(info, "language", None) or IDIOMAS[idioma_txt] or "auto",
                }
                estado.update(label="Transcripción lista ✅", state="complete")
            except Exception as e:
                estado.update(label="Error durante la transcripción", state="error")
                st.error(f"No se pudo transcribir: {e}")
            finally:
                try:
                    os.remove(ruta)
                except OSError:
                    pass

    if st.session_state.get("info_audio"):
        _ia = st.session_state["info_audio"]
        st.caption(
            f"Audio: {_ia.get('nombre', 'N/D')} · Duración: {_duracion_txt(_ia)} · "
            f"Idioma: {_ia.get('idioma') or '-'}"
        )

    transcripcion = st.text_area(
        "Transcripción (puedes corregirla a mano antes de analizar)",
        value=_texto_transcripcion(),
        key="ta_transcripcion",
        height=300,
    )
    st.session_state["transcripcion"] = (transcripcion or "").strip()

    _n = len(_texto_transcripcion())
    if _n:
        st.caption(f"Transcripción actual: {_n:,} caracteres".replace(",", "."))
    if _n > LIMITE_TRANSCRIPCION:
        st.caption(
            f"⚠️ El análisis usará los primeros {LIMITE_TRANSCRIPCION:,} caracteres "
            "(se declara en las limitaciones del informe).".replace(",", ".")
        )
    if st.session_state.get("segmentos"):
        with st.expander(
            f"Ver transcripción con marcas de tiempo ({len(st.session_state['segmentos'])} segmentos)"
        ):
            for _seg in st.session_state["segmentos"][:400]:
                st.markdown(
                    f"`[{int(_seg['inicio'] // 60):02d}:{int(_seg['inicio'] % 60):02d}]` {_seg['texto']}"
                )

# ====================== PASO 3 · FOTO (SOLO EXPEDIENTE) =======================

with tab3:
    st.subheader("Paso 3 · Foto del candidato/a (solo expediente)")
    st.error(
        "Privacidad: la foto NUNCA se envía a ningún modelo de IA ni influye en "
        "el puntaje. Queda únicamente como registro visual del expediente y "
        "puedes borrarla aquí mismo.",
        icon="🛡️",
    )
    fuente = st.radio(
        "¿Cómo quieres registrarla?",
        ["No registrar foto", "Tomar foto con la cámara", "Subir un archivo"],
        horizontal=True,
    )
    foto = None
    if fuente == "Tomar foto con la cámara":
        foto = st.camera_input("Cámara")
    elif fuente == "Subir un archivo":
        foto = st.file_uploader("Imagen (jpg / png)", type=["jpg", "jpeg", "png"], key="foto_archivo")
    if foto is not None:
        st.session_state["foto"] = {"bytes": foto.getvalue(), "nombre": foto.name}

    if st.session_state.get("foto"):
        cf1, cf2 = st.columns([1, 2])
        cf1.image(
            st.session_state["foto"]["bytes"],
            width=180,
            caption="Registro de expediente (no evaluativo)",
        )
        if cf2.button("🗑️ Borrar foto del expediente"):
            del st.session_state["foto"]
            st.rerun()
    else:
        st.caption("Sin foto registrada (opcional).")

# ====================== PASO 4 · ANÁLISIS CON IA (FASE 1) =====================

with tab4:
    st.subheader("Paso 4 · Análisis con IA (LLM real)")
    texto = _texto_transcripcion()
    perfil = st.session_state.get("perfil") or _perfil_actual()
    ncomps = len([c for c in perfil.get("competencias") or [] if c.get("nombre")])

    ma1, ma2, ma3 = st.columns(3)
    ma1.metric("Transcripción", f"{len(texto):,} car.".replace(",", "."))
    ma2.metric("Competencias del perfil", str(ncomps))
    ma3.metric("Token HuggingFace", "✅ listo" if _token() else "❌ falta")

    st.markdown("#### FASE 1 · Identificar el sesgo y reformularlo")
    st.caption(
        "Checklist obligatorio ANTES de analizar — marco del libro 'Sesgos "
        "inconscientes' (Fuller, Chow y Murphy — FranklinCovey)."
    )
    fase1_ok = _checklist("f1", CHECKS_FASE1)

    listo = bool(texto) and len(texto) >= 200 and ncomps > 0 and bool(_token()) and fase1_ok
    if st.button(
        "🔍 Analizar entrevista con la IA",
        type="primary",
        disabled=not listo,
        use_container_width=True,
    ):
        with st.status(
            "La IA está analizando la entrevista bajo el protocolo de imparcialidad…",
            expanded=True,
        ) as estado:
            st.write(f"LLM: {modelo_llm} · Whisper: {modelo_whisper}")
            try:
                analisis = analizar_entrevista(_token(), modelo_llm, perfil, texto)
                st.session_state["analisis"] = analisis
                estado.update(label="Análisis completado ✅", state="complete")
            except ValueError as e:
                estado.update(label="No se pudo analizar", state="error")
                st.error(str(e))
            except Exception as e:
                estado.update(label="No se pudo analizar", state="error")
                st.error(f"Error inesperado: {e}")

    if not _token():
        st.warning(
            "Falta el token de HuggingFace: pégalo en la barra lateral o "
            "configúralo como secret (HF_TOKEN) en los ajustes de la app."
        )
    if texto and len(texto) < 200:
        st.warning("La transcripción es demasiado corta: se necesitan al menos 200 caracteres.")
    if ncomps == 0:
        st.warning("Define al menos una competencia en el Paso 1.")

    if st.session_state.get("analisis"):
        st.divider()
        _renderizar_analisis(st.session_state["analisis"])

    with st.expander("¿Qué protocolo aplica la IA al analizar?"):
        st.markdown(MARCO_SESGOS)
        st.markdown(
            "- Evalúa cada competencia de forma independiente con evidencia textual "
            "a favor y en contra (anti halo y anti confirmación).\n"
            "- Pondera toda la transcripción por igual (anti primacía y recencia).\n"
            "- Compara solo contra el perfil del puesto (anti contraste y afinidad).\n"
            "- No predice desempeño futuro; declara la confianza de la evidencia.\n"
            "- Si detecta sesgos del ENTREVISTADOR en la conversación, los reporta en riesgos."
        )

# ============== PASO 5 · DECISIÓN HUMANA E INFORME (FASES 2 Y 3) ==============

with tab5:
    st.subheader("Paso 5 · Decisión humana e informe")

    analisis = st.session_state.get("analisis") or {}
    if not analisis:
        st.info("Completa el Paso 4: la IA debe analizar la entrevista antes de registrar la decisión.")

    st.markdown("#### FASE 2 · Cultivar la conexión")
    st.caption(
        "Del libro 'Sesgos inconscientes' (FranklinCovey): la empatía reduce el "
        "sesgo. Marca antes de decidir."
    )
    fase2_ok = _checklist("f2", CHECKS_FASE2)

    st.markdown("#### FASE 3 · Elegir el coraje")
    st.caption("Autorevisión final: el coraje de reconocer y corregir tu propio sesgo.")
    fase3_ok = _checklist("f3", CHECKS_FASE3)
    chk_debias = st.checkbox(
        "DECLARACIÓN FINAL: apliqué las 3 fases del protocolo (identificar y "
        "reformular el sesgo, cultivar la conexión y elegir el coraje) y baso mi "
        "decisión en la evidencia de la transcripción, no en impresiones.",
        key="chk_debias",
    )

    st.markdown("##### Decisión del reclutador/a (human-in-the-loop)")
    decision = st.radio(
        "Tu decisión",
        ["(Pendiente)", "Avanzar a siguiente etapa", "Contratar", "Descartar", "Requiere más evidencia"],
        index=0,
        horizontal=True,
    )
    justificacion = st.text_area(
        "Justificación humana (muy recomendable: queda en el informe)",
        key="in_justificacion",
        height=90,
    )
    colE, colV = st.columns(2)
    entrevistador = colE.text_input("Nombre del entrevistador/a o reclutador R.H.", key="in_entrevistador")
    entrevistado = colV.text_input("Nombre del entrevistado/a", key="in_entrevistado")

    info_audio = st.session_state.get("info_audio") or {}
    metadatos = {
        "empresa": (st.session_state.get("empresa_nombre") or "").strip(),
        "entrevistador": (entrevistador or "").strip(),
        "entrevistado": (entrevistado or "").strip(),
        "duracion_min": _duracion_txt(info_audio),
        "idioma": str(info_audio.get("idioma") or "-"),
        "modelo_whisper": modelo_whisper,
        "modelo_llm": analisis.get("modelo") or modelo_llm,
        "fecha": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "con_foto": bool(st.session_state.get("foto")),
    }
    decision_humana = {
        "decision": decision,
        "justificacion": (justificacion or "").strip(),
        "responsable": (entrevistador or "").strip(),
    }

    with st.expander("Ver las 3 cajas del encabezado del informe"):
        st.markdown(f"- **Nombre de la empresa:** {metadatos['empresa'] or '(en blanco para completar a mano)'}")
        st.markdown(f"- **Nombre del entrevistador/a R.H.:** {metadatos['entrevistador'] or '(en blanco para completar a mano)'}")
        st.markdown(f"- **Nombre del entrevistado/a:** {metadatos['entrevistado'] or '(en blanco para completar a mano)'}")
    if not justificacion.strip() and analisis:
        st.caption("💡 Escribe tu justificación humana: es la parte más valiosa del expediente.")

    st.divider()
    protocolo_ok = fase2_ok and fase3_ok and chk_debias
    if analisis and not protocolo_ok:
        st.info("🔒 Para desbloquear los informes, completa las FASES 2 y 3 con su declaración final.")
    elif not analisis:
        st.info("⏳ Genera primero el análisis en el Paso 4.")
    else:
        st.success("✅ Protocolo de imparcialidad completo: los informes están habilitados.")
        try:
            pdf_bytes = generar_reporte_pdf(perfil, metadatos, analisis, decision_humana)
            md_texto = generar_reporte_md(perfil, metadatos, analisis, decision_humana)
            nombre_archivo = f"informe_seleccion_{_slug(metadatos['entrevistado'] or perfil.get('cargo', ''))}"
            b1, b2, b3 = st.columns(3)
            b1.download_button(
                "📄 Descargar informe PDF",
                data=pdf_bytes,
                file_name=f"{nombre_archivo}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            b2.download_button(
                "📝 Descargar informe Markdown",
                data=md_texto.encode("utf-8"),
                file_name=f"{nombre_archivo}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            b3.download_button(
                "🗒️ Descargar informe TXT",
                data=md_texto.encode("utf-8"),
                file_name=f"{nombre_archivo}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"No se pudo generar el informe: {e}")

st.divider()
st.caption(
    f"{CREDITO_DISENO} · Informe de 9 secciones con firma del entrevistador/a R.H. · "
    "La IA apoya la decisión; no la toma."
)
