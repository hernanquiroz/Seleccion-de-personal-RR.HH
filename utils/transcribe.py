# -*- coding: utf-8 -*-
"""
Transcripción local de entrevistas con Whisper (faster-whisper, CPU int8).

El modelo se ejecuta EN LA MISMA MÁQUINA donde corre la app (nada del audio
sale a Internet). La primera vez que se usa un modelo, faster-whisper lo
descarga automáticamente (75 MB en 'base', ~465 MB en 'small').
"""

from typing import Callable, List, Optional, Tuple

# Opciones expuestas en la barra lateral (de más rápido a más preciso).
# En hosting gratuito con ~1 GB de RAM se recomienda 'base' o 'tiny'.
MODELOS_WHISPER = ["tiny", "base", "small", "medium"]

_modelos_cache = {}


def cargar_modelo(nombre: str):
    """Carga (y cachea) el modelo Whisper indicado.

    La primera vez descarga los pesos del modelo desde HuggingFace Hub.
    """
    nombre = (nombre or "small").strip().lower()
    if nombre in _modelos_cache:
        return _modelos_cache[nombre]

    # Import diferido: la app puede arrancar aunque el modelo aún no se use.
    from faster_whisper import WhisperModel

    modelo = WhisperModel(nombre, device="cpu", compute_type="int8")
    _modelos_cache[nombre] = modelo
    return modelo


def transcribir_audio(
    ruta: str,
    modelo,
    idioma: Optional[str] = None,
    on_progreso: Optional[Callable] = None,
) -> Tuple[str, List[dict], object]:
    """Transcribe un archivo de audio/video ya guardado en disco.

    Parámetros
    ----------
    ruta : ruta local del archivo (mp3, wav, m4a, mp4, etc.).
    modelo : modelo de faster-whisper obtenido con cargar_modelo().
    idioma : código ISO ('es') para forzar idioma, o None para detección automática.
    on_progreso : callback opcional (segmento, info) que se invoca por cada
        segmento transcrito, para dibujar la barra de progreso.

    Retorna
    -------
    (texto, segmentos, info)
        texto : transcripción completa unida en un solo string.
        segmentos : lista de dicts {"inicio", "fin", "texto"} con marcas de tiempo.
        info : objeto de faster-whisper con .duration y .language.
    """
    segmentos_iter, info = modelo.transcribe(
        ruta,
        language=idioma,
        vad_filter=True,       # filtra silencios y música de fondo
        beam_size=5,
    )

    partes: List[str] = []
    segmentos: List[dict] = []
    for seg in segmentos_iter:
        texto_seg = (seg.text or "").strip()
        if not texto_seg:
            continue
        segmentos.append(
            {"inicio": float(seg.start), "fin": float(seg.end), "texto": texto_seg}
        )
        partes.append(texto_seg)
        if on_progreso is not None:
            try:
                on_progreso(seg, info)
            except Exception:
                # La barra de progreso nunca debe romper la transcripción.
                pass

    texto = " ".join(partes).strip()
    return texto, segmentos, info
