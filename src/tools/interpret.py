import re
from src.core.config import ruta_indices
from src.utils.text_normalizer import convertir_a_mayusculas
from src.core.engine import campos_detectados, sugerir_campos, buscar_campos_inteligente
from src.tools.name_search import buscar_nombre, buscar_nombre_componentes
from src.tools.attribute_search import buscar_atributo
from src.tools.address_search import buscar_direccion_combinada
from src.tools.phone_search import buscar_numero_telefono

def ejecutar_consulta_inteligente(prompt: str, analisis, llm_clasificador):
    """
    Estrategia inteligente para ejecutar consultas que prueba múltiples herramientas
    cuando es necesario y devuelve los mejores resultados.
    
    Args:
        prompt: La consulta original del usuario
        analisis: Resultado del analizador de intenciones
        llm_clasificador: Modelo para análisis avanzado
        
    Returns:
        str: Resultado de la búsqueda más relevante
    """
    tipo = analisis.get("tipo_busqueda")
    campo = analisis.get("campo", "")
    valor = analisis.get("valor", "")
    
    print(f"[INFO] Ejecutando consulta inteligente - Tipo: {tipo}, Campo: {campo}, Valor: {valor}")
    
    # VALIDAR QUE TENGAMOS UN VALOR PARA BUSCAR
    if not valor:
        print("[ERROR] Valor de búsqueda vacío. Usando texto completo de la consulta.")
        valor = prompt
    
    resultados = {}  # ALMACENAR RESULTADOS
    herramientas_probadas = set()  # REGISTRO DE HERRAMIENTAS YA EJECUTADAS
    
    # ESTRATEGIA 1: EJECUCIÓN DIRECTA SEGÚN TIPO DE CONSULTA
    if tipo == "nombre_componentes":
        print("[HERRAMIENTA] Ejecutando búsqueda por componentes de nombre")
        resultados["nombre_componentes"] = buscar_nombre_componentes(valor)
        herramientas_probadas.add("nombre_componentes")
    
    if tipo == "direccion":
        print("[HERRAMIENTA] Ejecutando búsqueda de dirección")
        resultados["direccion"] = buscar_direccion_combinada(valor)
        herramientas_probadas.add("direccion")
    
    elif tipo == "telefono":
        print("[HERRAMIENTA] Ejecutando búsqueda de teléfono")
        resultados["telefono"] = buscar_numero_telefono(valor)
        herramientas_probadas.add("telefono")
    
    elif tipo == "atributo" and campo:
        print(f"[HERRAMIENTA] Ejecutando búsqueda por atributo: {campo}={valor}")
        resultados["atributo"] = buscar_atributo(campo, valor, carpeta_indices=ruta_indices)
        herramientas_probadas.add("atributo")
    
    elif tipo == "nombre":
        print(f"[HERRAMIENTA] Ejecutando búsqueda por nombre: {valor}")
        resultados["nombre"] = buscar_nombre(valor)
        herramientas_probadas.add("nombre")
    
    # ESTRATEGIA 2: PROBAR MÚLTIPLES HERRAMIENTAS
    
    necesita_mas_busquedas = (
        not resultados or
        all(("No se encontraron coincidencias" in res or not res) for res in resultados.values()) or
        tipo == "desconocido"
    )
    
    if necesita_mas_busquedas:
        print("[INFO] Estrategia de múltiples herramientas activada")
        
        # DETERMINAR QUÉ HERRAMIENTAS PROBAR, EN ORDEN DE PRIORIDAD
        herramientas_pendientes = []
        
        # SI PARECE UN NÚMERO, PRIORIZAR BÚSQUEDA POR TELÉFONO
        if re.search(r'\d{7,}', valor) and "telefono" not in herramientas_probadas:
            herramientas_pendientes.append(("telefono", None))
        
        # BÚSQUEDA POR NOMBRE
        if "nombre" not in herramientas_probadas:
            herramientas_pendientes.append(("nombre", None))
        
        # SI HAY INDICIOS DE DIRECCIÓN, AGREGAR A LA LISTA
        if (re.search(r'\d+', valor) or 
            any(palabra in prompt.lower() for palabra in ["calle", "colonia", "avenida"])) and "direccion" not in herramientas_probadas:
            herramientas_pendientes.append(("direccion", None))
        
        # BÚSQUEDA POR CAMPOS INTELIGENTES
        if "atributo" not in herramientas_probadas:
            campos_disponibles = list(campos_detectados)
            campos_probables = sugerir_campos(valor, campos_disponibles)
            herramientas_pendientes.append(("atributo", campos_probables))
        
        for tipo_herramienta, params in herramientas_pendientes:
            if tipo_herramienta == "telefono":
                resultados["telefono"] = buscar_numero_telefono(valor)
            elif tipo_herramienta == "nombre":
                resultados["nombre"] = buscar_nombre(valor)
            elif tipo_herramienta == "direccion":
                resultados["direccion"] = buscar_direccion_combinada(valor)
            elif tipo_herramienta == "atributo" and params:
                resultados["atributo"] = buscar_campos_inteligente(valor, carpeta_indices=ruta_indices, campos_ordenados=params)
    
    # ESTRATEGIA 3: ANÁLISIS Y SELECCIÓN DEL MEJOR RESULTADO
    
    resultados_positivos = {
        k: v for k, v in resultados.items() 
        if v and "No se encontraron coincidencias" not in v
    }
    
    if not resultados_positivos:
        return f"No se encontraron coincidencias para '{valor}' en ninguna de las herramientas. Por favor, intenta con otra consulta más específica."
    
    if len(resultados_positivos) == 1:
        tipo_busqueda, respuesta = list(resultados_positivos.items())[0]
        return respuesta
        
    # EVALUAR LA CALIDAD DE UN RESULTADO
    def evaluar_calidad(texto_resultado):
        num_coincidencias = texto_resultado.count("Coincidencia")
        calidad_coincidencias = texto_resultado.count("exacta") * 2 + texto_resultado.count("parcial")
        lineas_datos = len(texto_resultado.split("\n"))
        
        return num_coincidencias * 10 + calidad_coincidencias * 5 + lineas_datos
    
    # SELECCIONAR EL MEJOR
    calidades = {k: evaluar_calidad(v) for k, v in resultados_positivos.items()}
    mejor_herramienta = max(calidades.items(), key=lambda x: x[1])[0]
    
    valores_calidad = sorted(calidades.values(), reverse=True)
    diferencia_significativa = len(valores_calidad) < 2 or valores_calidad[0] > valores_calidad[1] * 1.5
    
    if diferencia_significativa:
        return resultados_positivos[mejor_herramienta]
    else:
        tipos_ordenados = sorted(resultados_positivos.keys(), key=lambda k: calidades.get(k, 0), reverse=True)
        mejores_tipos = tipos_ordenados[:2]
        
        respuesta_combinada = "Se encontraron varios tipos de coincidencias:\n\n"
        for tipo in mejores_tipos:
            respuesta_combinada += f"--- RESULTADOS DE BÚSQUEDA POR {tipo.upper()} ---\n"
            respuesta_combinada += resultados_positivos[tipo]
            respuesta_combinada += "\n\n"
        
        return convertir_a_mayusculas(respuesta_combinada)
    
def preprocesar_consulta(prompt: str) -> str:
    """
    Pre-procesa la consulta del usuario para hacerla más estandarizada
    y facilitar su posterior análisis.
    
    Args:
        prompt: Consulta original del usuario
    
    Returns:
        str: Consulta pre-procesada
    """
    # NORMALIZAR ESPACIOS Y PUNTUACIÓN
    prompt = prompt.strip()
    prompt = re.sub(r'\s+', ' ', prompt)
    prompt = re.sub(r'([.,;:!?])(\w)', r'\1 \2', prompt)
    
    # NORMALIZAR CARACTERES ESPECIALES
    prompt = prompt.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    prompt = prompt.replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
    prompt = prompt.replace('ñ', 'n').replace('Ñ', 'N')
    
    abreviaturas = {
        'tel': 'telefono',
        'tel.': 'telefono',
        'teléf': 'telefono',
        'teléf.': 'telefono',
        'núm': 'numero',
        'núm.': 'numero',
        'num': 'numero',
        'num.': 'numero',
        'dir': 'direccion',
        'dir.': 'direccion',
        'direc': 'direccion',
        'direc.': 'direccion',
        'col': 'colonia',
        'col.': 'colonia',
        'av': 'avenida',
        'av.': 'avenida',
        'ave': 'avenida',
        'ave.': 'avenida',
        'c.p.': 'codigo postal',
        'cp': 'codigo postal',
        'cp.': 'codigo postal',
        'fracc': 'fraccionamiento',
        'fracc.': 'fraccionamiento',
    }
    
    palabras = prompt.split()
    for i, palabra in enumerate(palabras):
        palabra_lower = palabra.lower()
        if palabra_lower in abreviaturas:
            palabras[i] = abreviaturas[palabra_lower]
    
    prompt = ' '.join(palabras)
    
    # ELIMINAR PALABRAS VACÍAS AL INICIO DE LA CONSULTA
    palabras_inicio = ['por favor', 'podrias', 'puedes', 'quisiera', 'quiero', 'necesito', 'dame']
    for palabra in palabras_inicio:
        if prompt.lower().startswith(palabra):
            prompt = prompt[len(palabra):].strip()
    
    # CONVERTIR PREGUNTAS IMPLÍCITAS EN EXPLÍCITAS
    prompt_lower = prompt.lower()
    
    # CONVERTIR "EL TELÉFONO 1234567" A "QUIÉN TIENE EL TELÉFONO 1234567"
    if (prompt_lower.startswith('el telefono') or prompt_lower.startswith('telefono')) and re.search(r'\d{7,}', prompt_lower):
        prompt = 'quien tiene ' + prompt
    
    # CONVERTIR "LA DIRECCIÓN CALLE X" A "QUIÉN VIVE EN CALLE X"
    if prompt_lower.startswith('la direccion') or prompt_lower.startswith('direccion'):
        prompt = 'quien vive en ' + prompt.split('direccion')[1].strip()
    
    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', prompt) and len(prompt.split()) <= 4:
        prompt = 'busca informacion de ' + prompt
    
    return prompt

def obtener_prompt_clasificacion_con_ejemplos(consulta):
    """
    Genera un prompt para clasificación con ejemplos específicos para mejorar la precisión.
    
    Args:
        consulta: La consulta a clasificar
    
    Returns:
        str: Prompt mejorado con ejemplos
    """
    return f"""
    Eres un sistema experto que clasifica consultas para una base de datos de personas. Necesito que clasifiques la siguiente consulta:
    
    "{consulta}"
    
    Debes determinar:
    1. El tipo de búsqueda: "nombre", "telefono", "direccion", "atributo" o "nombre_componentes"
    2. El campo específico (si aplica)
    3. El valor a buscar
    
    EJEMPLOS DE CLASIFICACIÓN CORRECTA:
    
    Consulta: "¿Quién es Juan Pérez?"
    Clasificación: {{"tipo_busqueda": "nombre", "campo": "nombre_completo", "valor": "Juan Pérez"}}
    
    Consulta: "Dame información de María González"
    Clasificación: {{"tipo_busqueda": "nombre", "campo": "nombre_completo", "valor": "María González"}}
    
    Consulta: "¿De quién es el teléfono 5544332211?"
    Clasificación: {{"tipo_busqueda": "telefono", "campo": "telefono_completo", "valor": "5544332211"}}
    
    Consulta: "¿A quién pertenece este número: 9988776655?"
    Clasificación: {{"tipo_busqueda": "telefono", "campo": "telefono_completo", "valor": "9988776655"}}
    
    Consulta: "¿Quién vive en Calle Principal 123, Colonia Centro?"
    Clasificación: {{"tipo_busqueda": "direccion", "campo": "direccion", "valor": "Calle Principal 123, Colonia Centro"}}
    
    Consulta: "Busca la dirección Zoquipan 1260, Lagos del Country"
    Clasificación: {{"tipo_busqueda": "direccion", "campo": "direccion", "valor": "Zoquipan 1260, Lagos del Country"}}
    
    Consulta: "¿Quiénes son médicos?"
    Clasificación: {{"tipo_busqueda": "atributo", "campo": "ocupacion", "valor": "médico"}}
    
    Consulta: "Busca mujeres en la base de datos"
    Clasificación: {{"tipo_busqueda": "atributo", "campo": "sexo", "valor": "F"}}
    
    Consulta: "Encuentra personas que vivan en Zapopan"
    Clasificación: {{"tipo_busqueda": "atributo", "campo": "municipio", "valor": "Zapopan"}}
    
    Consulta: "Información de Zoquipan 1271"
    Clasificación: {{"tipo_busqueda": "direccion", "campo": "direccion", "valor": "Zoquipan 1271"}}
    
    Consulta: "La persona en el teléfono 1234567"
    Clasificación: {{"tipo_busqueda": "telefono", "campo": "telefono_completo", "valor": "1234567"}}
    
    Consulta: "Quiero información del domicilio Hidalgo 123"
    Clasificación: {{"tipo_busqueda": "direccion", "campo": "direccion", "valor": "Hidalgo 123"}}

    Consulta: "¿Cuántas personas de nombre Carla con apellidos que empiecen con M y V hay?"
    Clasificación: {{"tipo_busqueda": "nombre_componentes", "campo": "nombre_completo", "valor": "Carla M V"}}
    
    Consulta: "Quién se llama Carla con M y V"
    Clasificación: {{"tipo_busqueda": "nombre_componentes", "campo": "nombre_completo", "valor": "Carla M V"}}
    
    Consulta: "Encuentra personas con nombre Juan y apellidos que inicien con L y P"
    Clasificación: {{"tipo_busqueda": "nombre_componentes", "campo": "nombre_completo", "valor": "Juan L P"}}
    
    Consulta: "Busca María con iniciales A y T"
    Clasificación: {{"tipo_busqueda": "nombre_componentes", "campo": "nombre_completo", "valor": "María A T"}}
    
    REGLAS IMPORTANTES:
    - Si la consulta tiene un número telefónico (7+ dígitos) junto a palabras como "teléfono", "número", "contacto", SIEMPRE es tipo "telefono".
    - Si la consulta menciona "vive en", "domicilio", "calle", "colonia" o términos similares, SIEMPRE es tipo "direccion".
    - Si la consulta busca información general sobre un nombre propio, es tipo "nombre".
    - Si busca personas con características específicas (sexo, ocupación, municipio, etc.), es tipo "atributo".
    - Para VALOR, extrae SOLO la información relevante, sin palabras de pregunta ni verbos auxiliares.
    - Si la consulta busca un nombre completo junto con iniciales o partes de apellidos (como "Carla con M y V"), es tipo "nombre_componentes".
    - Si la consulta busca personas con un nombre específico y filtradas por iniciales o letras, es "nombre_componentes".
    - Si la consulta pregunta cuántas personas cumplen con criterios parciales de nombre, es "nombre_componentes".
    
    Responde con un objeto JSON que contenga exactamente "tipo_busqueda", "campo" y "valor".
    """