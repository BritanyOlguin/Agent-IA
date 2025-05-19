from src.utils.text_normalizer import normalizar_texto, convertir_a_mayusculas
from llama_index.core.retrievers import VectorIndexRetriever
from src.core.engine import indices, similitud, all_tools, extraer_componentes_nombre, evaluar_coincidencia_componentes
from llama_index.core.tools import FunctionTool

# --- 3) HERRAMIENTA 1: BUSCAR POR NOMBRE COMPLETO ---
def buscar_nombre(query: str) -> str:
    """
    Busca coincidencias de nombres completos o parciales y las retorna ordenadas por relevancia.
    Permite búsquedas por cualquier combinación de nombre/apellidos, en cualquier orden.
    Ahora también permite búsquedas por subcadena (ej: 'Val' encontrará Valeria, Valentino, etc.)
    """
    print(f"Ejecutando búsqueda de nombre: '{query}'")
    
    query = query.strip()
    query_norm = normalizar_texto(query)
    query_tokens = set(query_norm.split())
    
    # DETECTAR SI ES BÚSQUEDA PARCIAL
    es_busqueda_parcial = len(query_tokens) <= 5
    
    resultados_por_categoria = {
        "exactos": [],          # COINCIDENCIA EXACTA
        "completos": [],        # TODOS LOS TOKENS DE BÚSQUEDA ESTÁN PRESENTES
        "parciales_alta": [],   # COINCIDENCIA SIGNIFICATIVA
        "parciales_media": [],  # COINCIDENCIA PARCIAL BÁSICA
        "substring": [],        # COINCIDENCIAS POR SUBCADENA
        "posibles": []          # COINCIDENCIAS DE BAJA CONFIANZA PERO ÚTILES
    }
    
    # EVITAR DUPLICADOS
    registros_encontrados = set()
    
    # RECORRER TODOS LOS ÍNDICES
    for fuente, index in indices.items():
        try:
            # OBTENER CANDIDATOS INICIALES
            retriever = VectorIndexRetriever(index=index, similarity_top_k=8)
            nodes = retriever.retrieve(query)
            
            # PROCESAR CADA NODO ENCONTRADO
            for node in nodes:
                metadata = node.node.metadata
                
                # CREAR IDENTIFICADOR ÚNICO
                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                if not id_registro:
                    id_registro = node.node.node_id
                
                if id_registro in registros_encontrados:
                    continue
                
                # OBTENER NOMBRE COMPLETO DE METADATOS
                nombre_completo = (
                    metadata.get("nombre_completo", "") or 
                    metadata.get("nombre completo", "") or 
                    metadata.get("nombre", "")
                ).strip()
                
                if not nombre_completo:
                    continue
                
                # NORMALIZAR EL NOMBRE
                nombre_norm = normalizar_texto(nombre_completo)
                
                # DIVIDIR EL NOMBRE EN TOKENS INDIVIDUALES
                nombre_tokens = nombre_norm.split()
                nombre_tokens_set = set(nombre_tokens)
                                
                # DETECTAR SI HAY COINCIDENCIA EXACTA (MISMO NOMBRE)
                sim_texto = similitud(query_norm, nombre_norm)
                
                # EVALUAR SI TODOS LOS TOKENS DE LA CONSULTA ESTÁN EN EL NOMBRE
                tokens_coincidentes = query_tokens.intersection(nombre_tokens_set)
                ratio_consulta = len(tokens_coincidentes) / len(query_tokens) if query_tokens else 0
                
                # EVALUAR QUÉ PORCENTAJE DEL NOMBRE COINCIDE CON LA CONSULTA
                ratio_nombre = len(tokens_coincidentes) / len(nombre_tokens) if nombre_tokens else 0
                
                # VERIFICAR SI HAY COINCIDENCIA DE APELLIDOS
                apellidos_nombre = set(nombre_tokens[-min(2, len(nombre_tokens)):])
                apellidos_query = set()
                if len(query_tokens) >= 2:
                    apellidos_query = set(list(query_tokens)[-min(2, len(query_tokens)):])
                coincidencia_apellidos = len(apellidos_nombre.intersection(apellidos_query))
                
                # VERIFICAR COINCIDENCIA DE NOMBRE DE PILA (PRIMERA PALABRA)
                nombre_pila_coincide = False
                if nombre_tokens and query_tokens:
                    nombre_pila = nombre_tokens[0]
                    if nombre_pila in query_tokens:
                        nombre_pila_coincide = True
                
                # VERIFICAR SI HAY COINCIDENCIA POR SUBCADENA
                coincidencia_substring = False
                token_con_substring = None
                
                # BUSCAR LA SUBCADENA EN CADA TOKEN DEL NOMBRE
                for token in nombre_tokens:
                    if query_norm in token:
                        coincidencia_substring = True
                        token_con_substring = token
                        break
                
                resumen = [f"{k}: {v}" for k, v in metadata.items() 
                          if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                texto_resultado = f"Encontrado en {fuente}:\n" + "\n".join(resumen)
                                
                # COINCIDENCIA EXACTA O CASI EXACTA
                if sim_texto > 0.9 or (ratio_consulta > 0.9 and ratio_nombre > 0.9):
                    resultados_por_categoria["exactos"].append({
                        "texto": f"Coincidencia exacta: {texto_resultado}",
                        "score": sim_texto + 0.1,  # Bonificación para exactos
                        "fuente": fuente
                    })
                
                # TODOS LOS TOKENS DE LA CONSULTA ESTÁN EN EL NOMBRE
                elif ratio_consulta > 0.95:
                    resultados_por_categoria["completos"].append({
                        "texto": f"Coincidencia completa: {texto_resultado}",
                        "score": ratio_consulta * 0.9 + ratio_nombre * 0.1,
                        "fuente": fuente
                    })
                
                # COINCIDENCIA DE APELLIDOS SIGNIFICATIVA (AL MENOS UN APELLIDO COMPLETO)
                elif coincidencia_apellidos > 0 and ratio_consulta >= 0.5:
                    resultados_por_categoria["parciales_alta"].append({
                        "texto": f"Coincidencia parcial: {texto_resultado}",
                        "score": 0.7 + (coincidencia_apellidos * 0.15) + (ratio_consulta * 0.15),
                        "fuente": fuente
                    })
                
                # COINCIDENCIA DE NOMBRE DE PILA Y TOKENS SIGNIFICATIVOS
                elif nombre_pila_coincide and len(tokens_coincidentes) >= 1:
                    resultados_por_categoria["parciales_alta"].append({
                        "texto": f"Coincidencia parcial: {texto_resultado}",
                        "score": 0.65 + (ratio_consulta * 0.35),
                        "fuente": fuente
                    })
                
                # COINCIDENCIA POR SUBCADENA
                elif coincidencia_substring:
                    resultados_por_categoria["substring"].append({
                        "texto": f"Coincidencia por subcadena '{query}' en '{token_con_substring}': {texto_resultado}",
                        "score": 0.6,  # Puntuación media-alta
                        "fuente": fuente
                    })
                
                # COINCIDENCIA PARCIAL BÁSICA
                elif len(tokens_coincidentes) >= 1 and any(token in nombre_tokens_set for token in query_tokens):
                    resultados_por_categoria["parciales_media"].append({
                        "texto": f"Coincidencia parcial: {texto_resultado}",
                        "score": 0.4 + (ratio_consulta * 0.6),
                        "fuente": fuente
                    })
                
                # COINCIDENCIAS DE BAJA CONFIANZA PERO ÚTILES
                elif tokens_coincidentes and sim_texto > 0.3:
                    resultados_por_categoria["posibles"].append({
                        "texto": f"Posible coincidencia: {texto_resultado}",
                        "score": sim_texto,
                        "fuente": fuente
                    })
                
                registros_encontrados.add(id_registro)
            
            # BÚSQUEDA EXHAUSTIVA ESPECÍFICA PARA SUBCADENAS
            if es_busqueda_parcial:
                print(f"Realizando búsqueda exhaustiva para subcadena '{query_norm}'...")
                
                for node_id, doc in index.docstore.docs.items():
                    metadata = doc.metadata
                    if not metadata:
                        continue
                    
                    id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                    if not id_registro:
                        id_registro = node_id
                    
                    if id_registro in registros_encontrados:
                        continue
                    
                    nombre_completo = (
                        metadata.get("nombre_completo", "") or 
                        metadata.get("nombre completo", "") or 
                        metadata.get("nombre", "")
                    ).strip()
                    
                    if not nombre_completo:
                        continue
                    
                    nombre_norm = normalizar_texto(nombre_completo)
                    
                    # VERIFICAR SI LA SUBCADENA ESTÁ EN CUALQUIER PARTE DEL NOMBRE
                    if query_norm in nombre_norm:
                        # DIVIDIR EL NOMBRE PARA ENCONTRAR QUÉ TOKEN CONTIENE LA SUBCADENA
                        tokens = nombre_norm.split()
                        token_con_subcadena = None
                        
                        for token in tokens:
                            if query_norm in token:
                                token_con_subcadena = token
                                break
                        
                        resumen = [f"{k}: {v}" for k, v in metadata.items() 
                                  if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                        texto_resultado = f"Encontrado en {fuente}:\n" + "\n".join(resumen)
                        
                        resultados_por_categoria["substring"].append({
                            "texto": f"Coincidencia por subcadena '{query}' en '{token_con_subcadena}': {texto_resultado}",
                            "score": 0.5,
                            "fuente": fuente
                        })
                        
                        registros_encontrados.add(id_registro)
        
        except Exception as e:
            print(f"Error al buscar en índice {fuente}: {e}")
            continue
    
    todas_respuestas = []

    mostrar_solo_exactos = bool(resultados_por_categoria["exactos"])

    if mostrar_solo_exactos:
        resultados_ordenados = sorted(resultados_por_categoria["exactos"], key=lambda x: x["score"], reverse=True)
        todas_respuestas.append("🔍 COINCIDENCIAS EXACTAS:")
        for res in resultados_ordenados:
            todas_respuestas.append(res["texto"])

    if not mostrar_solo_exactos:
        if resultados_por_categoria["completos"]:
            resultados_ordenados = sorted(resultados_por_categoria["completos"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 COINCIDENCIAS COMPLETAS:")
            for res in resultados_ordenados:
                todas_respuestas.append(res["texto"])

        if resultados_por_categoria["parciales_alta"]:
            resultados_ordenados = sorted(resultados_por_categoria["parciales_alta"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 COINCIDENCIAS PARCIALES SIGNIFICATIVAS:")
            for res in resultados_ordenados:
                todas_respuestas.append(res["texto"])
                
        if resultados_por_categoria["substring"]:
            resultados_ordenados = sorted(resultados_por_categoria["substring"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 COINCIDENCIAS POR SUBCADENA:")
            for res in resultados_ordenados:
                todas_respuestas.append(res["texto"])

        if resultados_por_categoria["parciales_media"]:
            resultados_ordenados = sorted(resultados_por_categoria["parciales_media"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 COINCIDENCIAS PARCIALES:")
            for res in resultados_ordenados:
                todas_respuestas.append(res["texto"])

        if len(todas_respuestas) < 3 and resultados_por_categoria["posibles"]:
            resultados_ordenados = sorted(resultados_por_categoria["posibles"], key=lambda x: x["score"], reverse=True)
            todas_respuestas.append("\n🔍 POSIBLES COINCIDENCIAS (baja confianza):")
            for res in resultados_ordenados:
                todas_respuestas.append(res["texto"])

    if not todas_respuestas:
        return f"No se encontraron coincidencias para '{query}' en ninguna fuente."
    
    total_resultados = 0
    for categoria in resultados_por_categoria:
        total_resultados += len(resultados_por_categoria[categoria])
    
    contador_resultados = f"\n\nSE ENCONTRARON {total_resultados} RESULTADOS."

    return convertir_a_mayusculas("\n\n".join(todas_respuestas) + contador_resultados)


busqueda_global_tool = FunctionTool.from_defaults(
    fn=buscar_nombre,
    name="buscar_nombre",
    description=(
        "Usa esta herramienta para encontrar información completa de una persona en todas las bases, "
        "cuando el usuario da el nombre completo o parcial. Por ejemplo: 'Dame la información de Juan', "
        "'¿Qué sabes de Pérez?', 'Busca González', 'Encuentra a María Pérez'."
    )
)
all_tools.insert(0, busqueda_global_tool)

# --- 7) HERRAMIENTA 5: BUSCAR POR INICIALES DEL NOMBRE ---
def buscar_nombre_componentes(query: str) -> str:
    """
    Busca personas por componentes parciales de nombres.
    Permite buscar por nombre y filtrar por iniciales o partes de apellidos.
    
    Por ejemplo:
    - "Carla con apellidos M y V" 
    - "Carla M V"
    - "nombre Carla iniciales M V"
    
    Retorna coincidencias que contengan todos los componentes especificados.
    """
    print(f"Ejecutando búsqueda por componentes de nombre: '{query}'")
    
    # EXTRAER COMPONENTES DE LA CONSULTA
    componentes = extraer_componentes_nombre(query)
    if not componentes:
        return "No se pudieron identificar componentes de nombre en la consulta. Por favor especifica al menos un nombre o inicial."
    
    print(f"Componentes detectados: {componentes}")
    
    # BUSCAR CANDIDATOS INICIALES POR EL PRIMER COMPONENTE
    primer_componente = componentes[0]
    componentes_adicionales = componentes[1:] if len(componentes) > 1 else []
    
    resultados_por_categoria = {
        "coincidencias_completas": [], # COINCIDEN TODOS LOS COMPONENTES EXACTAMENTE
        "coincidencias_iniciales": [],  # ALGUNOS COMPONENTES COINCIDEN COMO INICIALES
        "coincidencias_parciales": []   # HAY COINCIDENCIAS PARCIALES CON TODOS LOS COMPONENTES
    }
    
    registros_encontrados = set()
    
    for fuente, index in indices.items():
        try:
            # BÚSQUEDA SEMÁNTICA PARA CANDIDATOS INICIALES
            retriever = VectorIndexRetriever(index=index, similarity_top_k=15)
            nodes = retriever.retrieve(primer_componente)
            
            # BÚSQUEDA EXHAUSTIVA
            todos_docs = index.docstore.docs.items()
            
            # COMBINAR RESULTADOS SEMÁNTICOS CON BÚSQUEDA COMPLETA
            nodos_procesados = set()
            
            # PROCESAR NODOS DE BÚSQUEDA SEMÁNTICA
            for node in nodes:
                metadata = node.node.metadata
                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                if not id_registro:
                    id_registro = node.node.node_id
                
                nodos_procesados.add(id_registro)
                
                if id_registro in registros_encontrados:
                    continue
                
                resultado = evaluar_coincidencia_componentes(metadata, componentes)
                if resultado:
                    categoria, score, texto = resultado
                    resultados_por_categoria[categoria].append({
                        "texto": texto,
                        "score": score,
                        "fuente": fuente
                    })
                    registros_encontrados.add(id_registro)
            
            if len(registros_encontrados) < 5:
                for node_id, doc in todos_docs:
                    metadata = doc.metadata
                    id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                    if not id_registro:
                        id_registro = node_id
                    
                    if id_registro in nodos_procesados or id_registro in registros_encontrados:
                        continue
                    
                    resultado = evaluar_coincidencia_componentes(metadata, componentes)
                    if resultado:
                        categoria, score, texto = resultado
                        resultados_por_categoria[categoria].append({
                            "texto": texto,
                            "score": score,
                            "fuente": fuente
                        })
                        registros_encontrados.add(id_registro)
            
        except Exception as e:
            print(f"Error al buscar en índice {fuente}: {e}")
            continue
    
    todas_respuestas = []
    total_coincidencias = sum(len(resultados_por_categoria[cat]) for cat in resultados_por_categoria)
    
    if total_coincidencias == 0:
        return f"No se encontraron personas que coincidan con todos los componentes: {', '.join(componentes)}"
    
    todas_respuestas.append(f"🔍 Se encontraron {total_coincidencias} personas que coinciden con los componentes: {', '.join(componentes)}")
    
    # COINCIDENCIAS COMPLETAS
    if resultados_por_categoria["coincidencias_completas"]:
        resultados_ordenados = sorted(resultados_por_categoria["coincidencias_completas"], 
                                      key=lambda x: x["score"], reverse=True)
        todas_respuestas.append("\n🔍 COINCIDENCIAS EXACTAS:")
        for res in resultados_ordenados:
            todas_respuestas.append(res["texto"])
    
    # COINCIDENCIAS POR INICIALES
    if resultados_por_categoria["coincidencias_iniciales"]:
        resultados_ordenados = sorted(resultados_por_categoria["coincidencias_iniciales"], 
                                      key=lambda x: x["score"], reverse=True)
        todas_respuestas.append("\n🔍 COINCIDENCIAS POR INICIALES:")
        for res in resultados_ordenados:
            todas_respuestas.append(res["texto"])
    
    # COINCIDENCIAS PARCIALES
    if resultados_por_categoria["coincidencias_parciales"]:
        resultados_ordenados = sorted(resultados_por_categoria["coincidencias_parciales"], 
                                      key=lambda x: x["score"], reverse=True)
        todas_respuestas.append("\n🔍 COINCIDENCIAS PARCIALES:")
        for res in resultados_ordenados:
            todas_respuestas.append(res["texto"])

    contador_resultados = f"\n\nSE ENCONTRARON {total_coincidencias} RESULTADOS."

    return convertir_a_mayusculas("\n\n".join(todas_respuestas) + contador_resultados)

buscar_nombre_componentes_tool = FunctionTool.from_defaults(
    fn=buscar_nombre_componentes,
    name="buscar_nombre_componentes",
    description=(
        "Usa esta herramienta cuando el usuario busca personas por componentes parciales de nombres. "
        "Es especialmente útil cuando se busca un nombre específico junto con iniciales o partes de apellidos. "
        "Por ejemplo: '¿Cuántas personas de nombre Carla con apellidos que empiecen con M y V hay?', "
        "'Quien se llama Carla con M y V', 'Nombre Juan iniciales L M', o 'María con P y G'."
    )
)

all_tools.insert(5, buscar_nombre_componentes_tool)