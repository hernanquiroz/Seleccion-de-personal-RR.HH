# -*- coding: utf-8 -*-
"""
Plantillas de perfiles de puesto para la IA de apoyo a la selección de personal.

Cada plantilla define: cargo, área, descripción, requisitos y competencias
(nombre, qué evaluar y peso en %). Los pesos de cada plantilla suman 100.

Las competencias están redactadas para evaluarse ÚNICAMENTE con evidencia
verbal de la entrevista (método STAR: Situación, Tarea, Acción, Resultado),
siguiendo el enfoque del libro "Sesgos inconscientes" de Pamela Fuller,
Anne Chow y Mark Murphy (FranklinCovey): se evalúa conducta y resultados,
nunca afinidad, apariencia, simpatía ni "culture fit".
"""

from copy import deepcopy

PLANTILLAS = {
    "Genérico / Adaptable": {
        "cargo": "Analista / Administrativo",
        "area": "General",
        "descripcion": (
            "Puesto operativo-administrativo versátil: apoyo a procesos internos, "
            "registro y control de información, coordinación básica entre áreas y "
            "ejecución de tareas definidas con calidad y en plazo."
        ),
        "requisitos": (
            "Formación media o técnica (deseable superior en curso o terminada). "
            "Manejo de ofimática básica. Experiencia previa valorada pero no "
            "obligatoria. Disponibilidad horaria del puesto."
        ),
        "competencias": [
            {
                "nombre": "Comunicación efectiva",
                "descripcion": (
                    "Expresa ideas con claridad y orden; escucha y responde lo que "
                    "se le pregunta. Se evalúa SOLO con evidencia verbal (qué contó "
                    "y cómo lo estructuró), no por acento ni fluidez estética."
                ),
                "peso": 20,
            },
            {
                "nombre": "Trabajo en equipo y colaboración",
                "descripcion": (
                    "Cultiva conexiones y trabaja con otros hacia un resultado común "
                    "(cultivar la conexión, FranklinCovey). Evidencia: ejemplos "
                    "concretos de aporte a un equipo y de manejo de desacuerdos."
                ),
                "peso": 20,
            },
            {
                "nombre": "Resolución de problemas",
                "descripcion": (
                    "Identifica el problema, propone alternativas y actúa. Evidencia: "
                    "una situación STAR real con problema, acción propia y resultado."
                ),
                "peso": 20,
            },
            {
                "nombre": "Responsabilidad y orientación al resultado",
                "descripcion": (
                    "Cumple compromisos y termina lo que empieza. Evidencia: tareas o "
                    "metas que cerró, plazos respetados, errores asumidos y corregidos."
                ),
                "peso": 20,
            },
            {
                "nombre": "Adaptabilidad y aprendizaje",
                "descripcion": (
                    "Aprende de la experiencia y se ajusta a cambios. Evidencia: un "
                    "cambio real que vivió (proceso, herramienta o rol) y qué hizo."
                ),
                "peso": 20,
            },
        ],
    },
    "Ventas / Comercial": {
        "cargo": "Ejecutivo/a de ventas",
        "area": "Comercial",
        "descripcion": (
            "Responsable de atender, asesorar y convertir prospectos en clientes, "
            "cumplir cuotas de venta, hacer seguimiento a la cartera y reportar su "
            "gestión comercial de forma ordenada."
        ),
        "requisitos": (
            "Experiencia previa en ventas (deseable en el sector). Orientación a "
            "metas y trabajo por objetivos. Disponibilidad para desplazamientos "
            "según el puesto."
        ),
        "competencias": [
            {
                "nombre": "Orientación al cliente y cierre",
                "descripcion": (
                    "Detecta necesidades, propone soluciones y concreta la venta. "
                    "Evidencia: casos reales de ventas logradas (qué hizo paso a paso)."
                ),
                "peso": 25,
            },
            {
                "nombre": "Comunicación y persuasión ética",
                "descripcion": (
                    "Argumenta con claridad y honestidad, sin presionar ni engañar. "
                    "Evidencia: cómo presentó un producto y cómo manejó un 'no'."
                ),
                "peso": 20,
            },
            {
                "nombre": "Resiliencia comercial",
                "descripcion": (
                    "Tolera la frustración de rechazos y cuotas, y sostiene el ritmo. "
                    "Evidencia: periodo difícil concreto y qué hizo para recuperarse."
                ),
                "peso": 20,
            },
            {
                "nombre": "Organización y seguimiento",
                "descripcion": (
                    "Gestiona cartera y seguimientos sin perder clientes. Evidencia: "
                    "su método real de registro y control de prospectos."
                ),
                "peso": 20,
            },
            {
                "nombre": "Negociación y manejo de objeciones",
                "descripcion": (
                    "Responde objeciones y llega a acuerdos razonables. Evidencia: "
                    "una negociación STAR con resultado verificable."
                ),
                "peso": 15,
            },
        ],
    },
    "Atención al cliente": {
        "cargo": "Agente de servicio al cliente",
        "area": "Operaciones / Servicio",
        "descripcion": (
            "Atiende y resuelve consultas, solicitudes e inconvenientes de clientes "
            "en canal presencial, telefónico o digital, documentando cada caso y "
            "escalando lo que corresponda."
        ),
        "requisitos": (
            "Bachillerato o técnico. Experiencia en atención al público (deseable). "
            "Buena comunicación verbal. Disponibilidad de turnos si el puesto lo exige."
        ),
        "competencias": [
            {
                "nombre": "Orientación al servicio",
                "descripcion": (
                    "Prioriza resolver el problema del cliente dentro de las reglas. "
                    "Evidencia: casos concretos de servicio resuelto de inicio a fin."
                ),
                "peso": 25,
            },
            {
                "nombre": "Manejo de clientes difíciles",
                "descripcion": (
                    "Trata con respeto a personas molestas o exigentes y desescala el "
                    "conflicto. Evidencia: una queja real STAR y cómo terminó."
                ),
                "peso": 20,
            },
            {
                "nombre": "Comunicación clara y empática",
                "descripcion": (
                    "Se hace entender y valida la preocupación del otro sin perder el "
                    "foco. Se evalúa el contenido de lo que relata, no su acento ni "
                    "su simpatía (anti-sesgo: aprobación / ingratiation)."
                ),
                "peso": 20,
            },
            {
                "nombre": "Autocontrol emocional",
                "descripcion": (
                    "Mantiene la calma bajo presión. Evidencia: situación límite que "
                    "narró y qué hizo exactamente en el momento."
                ),
                "peso": 20,
            },
            {
                "nombre": "Registro y trazabilidad",
                "descripcion": (
                    "Documenta casos y respeta procesos. Evidencia: cómo llevaba el "
                    "control de sus atenciones en su trabajo previo."
                ),
                "peso": 15,
            },
        ],
    },
    "Analista / Administrativo": {
        "cargo": "Analista administrativo/a",
        "area": "Administración / Finanzas",
        "descripcion": (
            "Elabora y controla información operativa y financiera, apoya procesos "
            "de compras, facturación o inventarios, y produce reportes confiables "
            "para la toma de decisiones del área."
        ),
        "requisitos": (
            "Formación superior en administración, contabilidad, economía o afines. "
            "Manejo de Excel intermedio. Experiencia previa en funciones "
            "administrativas o de análisis (deseable)."
        ),
        "competencias": [
            {
                "nombre": "Análisis y atención al detalle",
                "descripcion": (
                    "Revisa información con rigor y detecta inconsistencias. "
                    "Evidencia: un caso donde encontró y corrigió un error real."
                ),
                "peso": 25,
            },
            {
                "nombre": "Organización y gestión del tiempo",
                "descripcion": (
                    "Administra varias tareas y plazos a la vez. Evidencia: cómo "
                    "organizó su carga en un periodo de alta demanda concreto."
                ),
                "peso": 20,
            },
            {
                "nombre": "Manejo de herramientas de análisis",
                "descripcion": (
                    "Usa herramientas (Excel, sistemas del rubro) para producir "
                    "resultados. Evidencia verbal: qué construyó con ellas y para "
                    "qué decisión sirvió. (Se evalúa la experiencia narrada, no "
                    "certificados: anti-sesgo de pseudopredictores.)"
                ),
                "peso": 20,
            },
            {
                "nombre": "Comunicación de reportes",
                "descripcion": (
                    "Presenta información compleja de forma entendible. Evidencia: "
                    "un reporte o presentación real que preparó y para quién."
                ),
                "peso": 20,
            },
            {
                "nombre": "Proactividad y mejora continua",
                "descripcion": (
                    "Propone y ejecuta mejoras sin que se lo ordenen. Evidencia: "
                    "una mejora concreta que impulsó y su efecto medible."
                ),
                "peso": 15,
            },
        ],
    },
    "Supervisión / Liderazgo": {
        "cargo": "Supervisor/a o jefe/a de área",
        "area": "Operaciones / Gestión",
        "descripcion": (
            "Conduce un equipo de trabajo: planifica la carga, acompaña y desarrolla "
            "a las personas, resuelve conflictos y responde por las metas del área "
            "ante la gerencia."
        ),
        "requisitos": (
            "Formación superior (deseable). Experiencia previa liderando equipos. "
            "Disponibilidad para reportar indicadores y participar en comités."
        ),
        "competencias": [
            {
                "nombre": "Liderazgo y delegación",
                "descripcion": (
                    "Asigna tareas según fortalezas y hace seguimiento sin micro-"
                    "gestionar. Evidencia: un equipo que lideró (tamaño, metas) y "
                    "cómo repartía el trabajo."
                ),
                "peso": 25,
            },
            {
                "nombre": "Conexión y desarrollo de personas",
                "descripcion": (
                    "Crea confianza psicológica, escucha y forma a su gente "
                    "('cultivar la conexión', FranklinCovey). Evidencia: alguien a "
                    "quien formó o retuvo y cómo lo hizo."
                ),
                "peso": 20,
            },
            {
                "nombre": "Toma de decisiones",
                "descripcion": (
                    "Decide con la información disponible y responde por el "
                    "resultado. Evidencia: una decisión difícil STAR, incluyendo "
                    "qué sacrificó."
                ),
                "peso": 20,
            },
            {
                "nombre": "Manejo de conflictos",
                "descripcion": (
                    "Interviene en disputas del equipo de forma justa y oportuna. "
                    "Evidencia: un conflicto real entre personas que medió y cómo "
                    "quedó. (Se cuida que su relato no culpe solo a una parte: "
                    "anti-sesgo de atribución.)"
                ),
                "peso": 20,
            },
            {
                "nombre": "Planificación y cumplimiento de metas",
                "descripcion": (
                    "Traduce objetivos en planes y los cumple. Evidencia: metas del "
                    "periodo anterior, resultado alcanzado y qué ajustó en el camino."
                ),
                "peso": 15,
            },
        ],
    },
}


def obtener_plantilla(nombre: str) -> dict:
    """Devuelve una copia editable de la plantilla solicitada.

    Si el nombre no existe, devuelve la plantilla 'Genérico / Adaptable'.
    """
    plantilla = PLANTILLAS.get(nombre) or PLANTILLAS["Genérico / Adaptable"]
    return deepcopy(plantilla)
