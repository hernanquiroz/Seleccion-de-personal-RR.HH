# -*- coding: utf-8 -*-
"""
Informe formal en PDF (ReportLab) para el expediente de la vacante.

Características del documento:
  - Encabezado con título, subtítulo y crédito "Diseño de IA por Ing. Hernán
    Quiroz" en letra pequeña (8 pt, cursiva, alineado a la derecha).
  - Tres cajas de identificación: empresa, entrevistador/a R.H. y entrevistado/a.
    Los campos vacíos se imprimen en blanco para completar a mano.
  - 9 secciones (igual que el reporte Markdown), incluida la sección 9 de
    Validación y firma del entrevistador/a de R.H.
  - Pie de página con aviso de uso responsable y número de página.
"""

import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.report import AVISO_ETICO, CREDITO_DISENO, LINEA_EN_BLANCO, TEXTO_AUDITORIA

# Paleta (coherente con el tema de la aplicación)
_AZUL = colors.HexColor("#2563eb")
_TXT = colors.HexColor("#0f172a")
_GRIS = colors.HexColor("#64748b")
_FONDO_CAJA = colors.HexColor("#f1f5f9")
_BORDE = colors.HexColor("#cbd5e1")


def _estilos() -> dict:
    base = getSampleStyleSheet()
    f = {
        "normal": "Helvetica",
        "bold": "Helvetica-Bold",
        "italic": "Helvetica-Oblique",
        "bolditalic": "Helvetica-BoldOblique",
    }
    return {
        "titulo": ParagraphStyle(
            "titulo", parent=base["Title"], fontName=f["bold"], fontSize=16,
            leading=20, textColor=_AZUL, alignment=TA_CENTER, spaceAfter=2,
        ),
        "fecha": ParagraphStyle(
            "fecha", parent=base["Normal"], fontName=f["normal"], fontSize=8.5,
            leading=11, textColor=_GRIS, alignment=TA_CENTER,
        ),
        "subtitulo": ParagraphStyle(
            "subtitulo", parent=base["Normal"], fontName=f["normal"], fontSize=9,
            leading=12, textColor=_GRIS, alignment=TA_LEFT,
        ),
        "credito": ParagraphStyle(
            "credito", parent=base["Normal"], fontName=f["italic"], fontSize=8,
            leading=10.5, textColor=_GRIS, alignment=TA_RIGHT,
        ),
        "caja_titulo": ParagraphStyle(
            "caja_titulo", parent=base["Normal"], fontName=f["bold"], fontSize=7.5,
            leading=9.5, textColor=_GRIS, alignment=TA_LEFT,
        ),
        "caja_valor": ParagraphStyle(
            "caja_valor", parent=base["Normal"], fontName=f["normal"], fontSize=11,
            leading=14, textColor=_TXT, alignment=TA_LEFT,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName=f["bold"], fontSize=11.5,
            leading=14.5, textColor=_AZUL, spaceBefore=14, spaceAfter=5,
        ),
        "cuerpo": ParagraphStyle(
            "cuerpo", parent=base["Normal"], fontName=f["normal"], fontSize=9.5,
            leading=13.5, textColor=_TXT, alignment=TA_JUSTIFY, spaceAfter=4,
        ),
        "cuerpo_izq": ParagraphStyle(
            "cuerpo_izq", parent=base["Normal"], fontName=f["normal"], fontSize=9.5,
            leading=13.5, textColor=_TXT, alignment=TA_LEFT, spaceAfter=3,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontName=f["normal"], fontSize=8,
            leading=11, textColor=_GRIS, alignment=TA_JUSTIFY,
        ),
        "firma": ParagraphStyle(
            "firma", parent=base["Normal"], fontName=f["bold"], fontSize=9.5,
            leading=13, textColor=_TXT, alignment=TA_LEFT, spaceBefore=10,
        ),
        "linea": ParagraphStyle(
            "linea", parent=base["Normal"], fontName=f["normal"], fontSize=9.5,
            leading=13, textColor=_TXT, alignment=TA_LEFT,
        ),
    }


def _p(texto: str, estilo) -> Paragraph:
    """Crea un Paragraph escapando XML (&, <, >) para texto no controlado."""
    return Paragraph(escape(str(texto or "")), estilo)


def _pie(canvas, doc):
    """Pie de página: aviso de uso responsable + número de página."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_GRIS)
    canvas.drawString(2 * cm, 1.15 * cm, AVISO_ETICO)
    canvas.drawRightString(A4[0] - 2 * cm, 1.15 * cm, f"Página {doc.page}")
    canvas.restoreState()


def _cajas_identificacion(st: dict, metadatos: dict, avail: float):
    """Tabla con las tres cajas de identificación del expediente."""
    cajas = [
        ("NOMBRE DE LA EMPRESA", metadatos.get("empresa", "")),
        ("NOMBRE DEL ENTREVISTADOR/A O RECLUTADOR R.H.", metadatos.get("entrevistador", "")),
        ("NOMBRE DEL ENTREVISTADO/A", metadatos.get("entrevistado", "")),
    ]
    celdas, ancho = [], avail / 3.0
    for titulo, valor in cajas:
        valor = (valor or "").strip() or " "  # caja en blanco para completar a mano
        celdas.append([_p(titulo, st["caja_titulo"]), Spacer(1, 6), _p(valor, st["caja_valor"])])
    tabla = Table([celdas], colWidths=[ancho] * 3, hAlign="CENTER")
    tabla.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _FONDO_CAJA),
                ("BOX", (0, 0), (-1, -1), 0.8, _BORDE),
                ("INNERGRID", (0, 0), (-1, -1), 0.8, _BORDE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return tabla


def generar_reporte_pdf(perfil: dict, metadatos: dict, analisis: dict, decision_humana: dict) -> bytes:
    """Genera el informe completo y devuelve los bytes del PDF."""
    st = _estilos()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2.1 * cm,
        title="Informe de apoyo a la selección de personal",
        author="Ing. Hernán Quiroz",
        subject="Asistente IA de apoyo a la selección - informe de entrevista",
    )
    avail = A4[0] - 4 * cm
    historia = []

    # ----------------------------- Encabezado -------------------------------
    historia.append(_p("Informe de apoyo a la selección de personal", st["titulo"]))
    historia.append(_p(f"Fecha del informe: {metadatos.get('fecha', '')}", st["fecha"]))
    historia.append(Spacer(1, 8))

    # Subtítulo (izquierda, 60%) + crédito en letra pequeña (derecha, 40%)
    encabezado = Table(
        [[
            _p("Apoyo con IA · La decisión final siempre es humana", st["subtitulo"]),
            _p(CREDITO_DISENO, st["credito"]),
        ]],
        colWidths=[0.60 * avail, 0.40 * avail],
        hAlign="CENTER",
    )
    encabezado.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
    historia.append(encabezado)
    historia.append(Spacer(1, 5))
    historia.append(HRFlowable(width="100%", thickness=0.8, color=_BORDE))
    historia.append(Spacer(1, 10))

    # ------------------- Cajas de identificación (encabezado) ----------------
    historia.append(_cajas_identificacion(st, metadatos, avail))
    historia.append(Spacer(1, 6))

    # ----------------------------- 1. Identificación -------------------------
    historia.append(_p("1. Identificación del proceso", st["h2"]))
    for etiqueta, valor in [
        ("Nombre de la empresa:", metadatos.get("empresa", "")),
        ("Nombre del entrevistador/a o reclutador R.H.:", metadatos.get("entrevistador", "")),
        ("Nombre del entrevistado/a:", metadatos.get("entrevistado", "")),
        ("Duración de la entrevista:", metadatos.get("duracion_min", "N/D")),
        ("Idioma detectado:", metadatos.get("idioma", "-")),
        ("Registro fotográfico adjunto:", "Sí (no evaluativo)" if metadatos.get("con_foto") else "No"),
    ]:
        texto = f"<b>{escape(etiqueta)}</b> {escape(str(valor).strip() or '(en blanco para completar a mano)')}"
        historia.append(Paragraph(texto, st["cuerpo_izq"]))

    # ----------------------------- 2. Perfil del puesto ----------------------
    historia.append(_p("2. Perfil del puesto evaluado", st["h2"]))
    historia.append(_p(f"Cargo: {perfil.get('cargo') or 'N/D'} · Área: {perfil.get('area') or 'N/D'}", st["cuerpo_izq"]))
    if perfil.get("descripcion"):
        historia.append(_p(f"Descripción: {perfil['descripcion']}", st["cuerpo"]))
    if perfil.get("requisitos"):
        historia.append(_p(f"Requisitos: {perfil['requisitos']}", st["cuerpo"]))
    comps_perfil = [c for c in perfil.get("competencias") or [] if str(c.get("nombre") or "").strip()]
    if comps_perfil:
        filas = [[_p("Competencia", st["caja_titulo"]), _p("Peso", st["caja_titulo"]), _p("Qué evalúa", st["caja_titulo"])]]
        for c in comps_perfil:
            filas.append([_p(c.get("nombre", ""), st["cuerpo_izq"]), _p(f"{c.get('peso', 0)}%", st["cuerpo_izq"]), _p(c.get("descripcion", ""), st["cuerpo_izq"])])
        tabla_comp = Table(filas, colWidths=[0.26 * avail, 0.10 * avail, 0.64 * avail], repeatRows=1)
        tabla_comp.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _FONDO_CAJA),
                    ("GRID", (0, 0), (-1, -1), 0.5, _BORDE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        historia.append(tabla_comp)

    # ----------------------------- 3. Metodología ----------------------------
    historia.append(_p("3. Metodología", st["h2"]))
    historia.append(
        _p(
            "Transcripción local de la entrevista con Whisper "
            f"({metadatos.get('modelo_whisper', 'N/D')}), ejecutada en la propia máquina, "
            "y análisis del contenido verbal con el LLM "
            f"{metadatos.get('modelo_llm', 'N/D')} (HuggingFace Inference) bajo un "
            "protocolo de imparcialidad obligatorio. El puntaje global se recalcula "
            "en la aplicación como promedio ponderado con los pesos del perfil.",
            st["cuerpo"],
        )
    )

    # ----------------------------- 4-6. Resultados ---------------------------
    if analisis.get("modo_fallback"):
        historia.append(_p("4. Resultado del análisis (modo de respaldo)", st["h2"]))
        historia.append(
            _p(
                "El modelo no devolvió un JSON válido, por lo que NO se emitieron "
                "puntajes. Se incluye el análisis cualitativo en texto libre:",
                st["cuerpo"],
            )
        )
        historia.append(_p(analisis.get("texto", ""), st["cuerpo"]))
    else:
        historia.append(_p("4. Resultado global", st["h2"]))
        historia.append(
            Paragraph(
                f"<b>Puntaje global ponderado: {int(analisis.get('puntaje_global', 0))}/100</b> · "
                f"Recomendación de apoyo: <b>{escape(str(analisis.get('recomendacion_apoyo', 'N/D')))}</b>",
                st["cuerpo_izq"],
            )
        )
        if analisis.get("resumen_ejecutivo"):
            historia.append(_p(f"Resumen ejecutivo: {analisis['resumen_ejecutivo']}", st["cuerpo"]))
        if analisis.get("justificacion"):
            historia.append(_p(f"Justificación: {analisis['justificacion']}", st["cuerpo"]))

        historia.append(_p("5. Evaluación por competencias", st["h2"]))
        for c in analisis.get("competencias") or []:
            bloque = [
                Paragraph(
                    f"<b>{escape(str(c.get('nombre') or 'Competencia'))}</b> — "
                    f"{int(c.get('puntaje', 0) or 0)}/100 ({escape(str(c.get('nivel') or 'N/D'))}) · "
                    f"peso {c.get('peso', 0)}% · Confianza de la evidencia: "
                    f"{escape(str(c.get('confianza_evidencia') or 'Baja'))}",
                    st["cuerpo_izq"],
                ),
                _p(f"Evidencia a favor: {c.get('evidencia', 'N/D')}", st["cuerpo"]),
                _p(f"Evidencia contraria o matiz: {c.get('evidencia_contraria', 'N/D')}", st["cuerpo"]),
                _p(f"Observación: {c.get('observacion', 'N/D')}", st["cuerpo"]),
                Spacer(1, 6),
            ]
            historia.append(KeepTogether(bloque))

        historia.append(_p("6. Fortalezas y riesgos", st["h2"]))
        historia.append(Paragraph("<b>Fortalezas detectadas:</b>", st["cuerpo_izq"]))
        fortalezas = analisis.get("fortalezas") or []
        for f in fortalezas or ["Sin fortalezas reportadas."]:
            historia.append(Paragraph(f"• {escape(str(f))}", st["cuerpo_izq"]))
        historia.append(Paragraph("<b>Riesgos / puntos a verificar:</b>", st["cuerpo_izq"]))
        riesgos = analisis.get("riesgos") or []
        for r in riesgos or ["Sin riesgos reportados."]:
            historia.append(Paragraph(f"• {escape(str(r))}", st["cuerpo_izq"]))

    # ------------------- 7. Auditoría de imparcialidad -----------------------
    historia.append(_p("7. Auditoría de imparcialidad y limitaciones", st["h2"]))
    historia.append(_p(TEXTO_AUDITORIA, st["cuerpo"]))
    historia.append(Spacer(1, 4))
    historia.append(
        _p(
            "Marco de referencia: Fuller, P., Chow, A. y Murphy, M., 'Sesgos "
            "inconscientes: cómo reformularlos, cultivar conexiones y crear equipos "
            "de alto rendimiento' (FranklinCovey).",
            st["small"],
        )
    )
    historia.append(Spacer(1, 4))
    historia.append(Paragraph("<b>Limitaciones declaradas del análisis:</b>", st["cuerpo_izq"]))
    limitaciones = analisis.get("limitaciones") or ["No se declararon limitaciones."]
    for l in limitaciones:
        historia.append(Paragraph(f"• {escape(str(l))}", st["cuerpo_izq"]))

    # ----------------------------- 8. Decisión humana ------------------------
    historia.append(_p("8. Decisión humana final", st["h2"]))
    historia.append(
        _p(f"Decisión del reclutador/a: {decision_humana.get('decision', '(Pendiente)')}", st["cuerpo_izq"])
    )
    just = (decision_humana.get("justificacion") or "").strip()
    historia.append(_p(f"Justificación humana: {just if just else '(sin registrar)'}", st["cuerpo_izq"]))
    resp = (decision_humana.get("responsable") or "").strip() or "(en blanco para completar a mano)"
    historia.append(_p(f"Responsable de la decisión: {resp}", st["cuerpo_izq"]))
    historia.append(
        _p(
            "Nota: la recomendación de la IA es un insumo de apoyo y NO sustituye la "
            "valoración del reclutador (human-in-the-loop).",
            st["small"],
        )
    )

    # ------------------- 9. Validación y firma del R.H. ----------------------
    historia.append(_p("9. Validación y firma del entrevistador/a de R.H.", st["h2"]))
    bloque_firma = [
        _p(
            "Declaro que he leído el presente informe, verifiqué la evidencia citada "
            "contra la transcripción de la entrevista, apliqué la autorevisión de "
            "sesgos y asumo la decisión registrada como propia.",
            st["cuerpo"],
        ),
        Spacer(1, 14),
        Paragraph(f"Lugar y fecha: {LINEA_EN_BLANCO}", st["linea"]),
        Spacer(1, 12),
        Paragraph(f"Nombre del entrevistador/a en letra imprenta: {LINEA_EN_BLANCO}", st["linea"]),
        Spacer(1, 34),
        Paragraph("<b>Firma del entrevistador/a (R.H.):</b>", st["firma"]),
        Spacer(1, 46),
        Paragraph(LINEA_EN_BLANCO, st["linea"]),
    ]
    historia.append(KeepTogether(bloque_firma))

    doc.build(historia, onFirstPage=_pie, onLaterPages=_pie)
    return buf.getvalue()
