# -*- coding: utf-8 -*-
"""
Reporte en Markdown + textos fijos compartidos con el generador de PDF.

Estructura del informe (9 secciones):
  1. Identificación        4. Resultado global      7. Auditoría de imparcialidad
  2. Perfil del puesto     5. Evaluación por comp.  8. Decisión humana final
  3. Metodología           6. Fortalezas y riesgos  9. Validación y firma R.H.
"""

from datetime import datetime

# --- Textos fijos compartidos (también los usa utils/report_pdf.py) ----------

AVISO_ETICO = (
    "Aviso de uso responsable: este informe es un APOYO a la decisión; la "
    "decisión final siempre es de una persona."
)

CREDITO_DISENO = "Diseño de IA por Ing. Hernán Quiroz"

TEXTO_AUDITORIA = (
    "Este informe fue producido con un protocolo de imparcialidad alineado con "
    "el marco del libro 'Sesgos inconscientes' de Pamela Fuller, Anne Chow y "
    "Mark Murphy (FranklinCovey): Identificar el sesgo, Reformularlo, Cultivar "
    "la conexión y Elegir el coraje. La IA evaluó exclusivamente el contenido "
    "verbal de la transcripción; la fotografía del candidato/a (si existe) no "
    "se envió a ningún modelo ni influyó en el puntaje. Cada competencia fue "
    "puntuada de forma independiente con evidencia textual a favor y en contra "
    "(anti halo/cuerno y confirmación); toda la transcripción se ponderó por "
    "igual (anti primacía/recencia); la comparación fue únicamente contra el "
    "perfil del puesto (anti contraste y afinidad); y no se emitieron predicciones "
    "de desempeño futuro (anti falsas expectativas). Los modelos de lenguaje "
    "pueden equivocarse: el reclutador debe verificar la evidencia citada contra "
    "la transcripción antes de decidir."
)

LINEA_EN_BLANCO = "______________________________"

_REFERENCIA_LIBRO = (
    "Marco de referencia: Fuller, P., Chow, A. y Murphy, M., 'Sesgos "
    "inconscientes: cómo reformularlos, cultivar conexiones y crear equipos de "
    "alto rendimiento' (FranklinCovey)."
)


def _caja(valor: str) -> str:
    """Devuelve el valor o un marcador si la caja quedará en blanco."""
    valor = (valor or "").strip()
    return valor if valor else "*(en blanco para completar a mano)*"


def generar_reporte_md(perfil: dict, metadatos: dict, analisis: dict, decision_humana: dict) -> str:
    """Genera el reporte completo en Markdown (9 secciones)."""
    fecha = metadatos.get("fecha") or datetime.now().strftime("%d/%m/%Y %H:%M")
    lineas = []

    # --------------------------- Encabezado ---------------------------------
    lineas.append("# Informe de apoyo a la selección de personal")
    lineas.append(f"**Fecha del informe:** {fecha}")
    lineas.append(f"*{CREDITO_DISENO}*")
    lineas.append("")

    # 1. Identificación (tres cajas del encabezado)
    lineas.append("## 1. Identificación del proceso")
    lineas.append(f"- **Nombre de la empresa:** {_caja(metadatos.get('empresa', ''))}")
    lineas.append(f"- **Nombre del entrevistador/a o reclutador R.H.:** {_caja(metadatos.get('entrevistador', ''))}")
    lineas.append(f"- **Nombre del entrevistado/a:** {_caja(metadatos.get('entrevistado', ''))}")
    lineas.append(f"- **Duración de la entrevista:** {metadatos.get('duracion_min', 'N/D')}")
    lineas.append(f"- **Idioma detectado:** {metadatos.get('idioma', '-')}")
    lineas.append(f"- **Registro fotográfico adjunto al expediente:** {'Sí (no evaluativo)' if metadatos.get('con_foto') else 'No'}")
    lineas.append("")

    # 2. Perfil del puesto
    lineas.append("## 2. Perfil del puesto evaluado")
    lineas.append(f"- **Cargo:** {perfil.get('cargo') or 'N/D'}")
    lineas.append(f"- **Área:** {perfil.get('area') or 'N/D'}")
    if perfil.get("descripcion"):
        lineas.append(f"- **Descripción:** {perfil['descripcion']}")
    if perfil.get("requisitos"):
        lineas.append(f"- **Requisitos:** {perfil['requisitos']}")
    lineas.append("")
    lineas.append("**Competencias y pesos del promedio ponderado:**")
    lineas.append("")
    lineas.append("| Competencia | Peso | Qué evalúa |")
    lineas.append("|---|---|---|")
    for c in perfil.get("competencias") or []:
        nombre = str(c.get("nombre") or "").strip()
        if nombre:
            lineas.append(f"| {nombre} | {c.get('peso', 0)}% | {c.get('descripcion', '')} |")
    lineas.append("")

    # 3. Metodología
    lineas.append("## 3. Metodología")
    lineas.append(
        f"- Transcripción local de la entrevista con Whisper "
        f"({metadatos.get('modelo_whisper', 'N/D')}), ejecutada en la propia máquina y editable a mano."
    )
    lineas.append(
        f"- Análisis del contenido verbal con el LLM {metadatos.get('modelo_llm', 'N/D')} "
        "(HuggingFace Inference), con protocolo de imparcialidad obligatorio en el prompt."
    )
    lineas.append(
        "- El puntaje global se recalcula en la aplicación como promedio ponderado "
        "con los pesos del perfil (no se confía en el número declarado por el modelo)."
    )
    lineas.append("")

    # 4. Resultado global
    if analisis.get("modo_fallback"):
        lineas.append("## 4. Resultado del análisis (modo de respaldo)")
        lineas.append(
            "> El modelo no devolvió un JSON válido, por lo que NO se emitieron "
            "puntajes. A continuación se incluye el análisis cualitativo en texto libre."
        )
        lineas.append("")
        lineas.append(str(analisis.get("texto", "")))
        lineas.append("")
    else:
        lineas.append("## 4. Resultado global")
        lineas.append(
            f"- **Puntaje global ponderado: {analisis.get('puntaje_global', 0)}/100**"
        )
        lineas.append(f"- **Recomendación de apoyo:** {analisis.get('recomendacion_apoyo', 'N/D')}")
        if analisis.get("resumen_ejecutivo"):
            lineas.append(f"- **Resumen ejecutivo:** {analisis['resumen_ejecutivo']}")
        if analisis.get("justificacion"):
            lineas.append(f"- **Justificación:** {analisis['justificacion']}")
        lineas.append("")

        # 5. Evaluación por competencias
        lineas.append("## 5. Evaluación por competencias")
        for c in analisis.get("competencias") or []:
            lineas.append("")
            lineas.append(
                f"### {c.get('nombre', 'Competencia')} — {c.get('puntaje', 0)}/100 "
                f"({c.get('nivel', 'N/D')}) · peso {c.get('peso', 0)}%"
            )
            lineas.append(f"- **Confianza de la evidencia:** {c.get('confianza_evidencia', 'Baja')}")
            lineas.append(f"- **Evidencia a favor:** {c.get('evidencia', 'N/D')}")
            lineas.append(f"- **Evidencia contraria o matiz:** {c.get('evidencia_contraria', 'N/D')}")
            lineas.append(f"- **Observación:** {c.get('observacion', 'N/D')}")
        lineas.append("")

        # 6. Fortalezas y riesgos
        lineas.append("## 6. Fortalezas y riesgos")
        lineas.append("**Fortalezas detectadas:**")
        fortalezas = analisis.get("fortalezas") or []
        if fortalezas:
            lineas.extend(f"- {f}" for f in fortalezas)
        else:
            lineas.append("- Sin fortalezas reportadas.")
        lineas.append("")
        lineas.append("**Riesgos / puntos a verificar:**")
        riesgos = analisis.get("riesgos") or []
        if riesgos:
            lineas.extend(f"- {r}" for r in riesgos)
        else:
            lineas.append("- Sin riesgos reportados.")
        lineas.append("")

    # 7. Auditoría de imparcialidad y limitaciones
    lineas.append("## 7. Auditoría de imparcialidad y limitaciones")
    lineas.append(TEXTO_AUDITORIA)
    lineas.append("")
    lineas.append(f"*{_REFERENCIA_LIBRO}*")
    lineas.append("")
    lineas.append("**Limitaciones declaradas del análisis:**")
    limitaciones = analisis.get("limitaciones") or []
    if limitaciones:
        lineas.extend(f"- {l}" for l in limitaciones)
    else:
        lineas.append("- No se declararon limitaciones.")
    lineas.append("")

    # 8. Decisión humana final
    lineas.append("## 8. Decisión humana final")
    lineas.append(f"- **Decisión del reclutador/a:** {decision_humana.get('decision', '(Pendiente)')}")
    just = (decision_humana.get("justificacion") or "").strip()
    lineas.append(f"- **Justificación humana:** {just if just else '*(sin registrar)*'}")
    lineas.append(f"- **Responsable de la decisión:** {_caja(decision_humana.get('responsable', ''))}")
    lineas.append(
        "- **Nota:** la recomendación de la IA es un insumo de apoyo y NO sustituye "
        "la valoración del reclutador (human-in-the-loop)."
    )
    lineas.append("")

    # 9. Validación y firma
    lineas.append("## 9. Validación y firma del entrevistador/a de R.H.")
    lineas.append(
        "Declaro que he leído el presente informe, verifiqué la evidencia citada "
        "contra la transcripción de la entrevista, apliqué la autorevisión de "
        "sesgos y asumo la decisión registrada como propia."
    )
    lineas.append("")
    lineas.append(f"Lugar y fecha: {LINEA_EN_BLANCO}")
    lineas.append("")
    lineas.append(f"Nombre del entrevistador/a en letra imprenta: {LINEA_EN_BLANCO}")
    lineas.append("")
    lineas.append("**Firma del entrevistador/a (R.H.):**")
    lineas.append("")
    lineas.append(LINEA_EN_BLANCO)

    return "\n".join(lineas)
