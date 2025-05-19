from src.utils.text_normalizer import normalizar_texto, convertir_a_mayusculas
from src.core.config import STOP_WORDS, CAMPOS_BUSQUEDA_EXACTA, CAMPOS_DIRECCION, TOLERANCIA_NUMERO_CERCANO, UMBRAL_PUNTAJE_MINIMO
import re
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from src.core.engine import indices, similitud, all_tools
from llama_index.core.tools import FunctionTool
from typing import Dict, Any

# --- 5) HERRAMIENTA 3: BUSCAR POR DIRECCION COMPLETA ---

def buscar_direccion_combinada(texto_direccion: str) -> str:
    """
    Busca coincidencias de dirección combinando búsqueda exacta por metadatos
    y búsqueda semántica con evaluación de componentes.

    Prioriza coincidencias exactas de "calle número", pero también busca
    coincidencias semánticas y evalúa relevancia basada en componentes
    y similitud numérica/textual en todos los índices.
    """
    print(f"\nBuscando dirección combinada: '{texto_direccion}'")

    # PREPROCESAMIENTO Y EXTRACCIÓN DE COMPONENTES
    texto_direccion_normalizado = normalizar_texto(texto_direccion)
    texto_direccion_normalizado = re.sub(r'([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])(\d+)', r'\1 \2', texto_direccion_normalizado)
    texto_direccion_normalizado = re.sub(r'(\d+)([a-zA-ZáéíóúüñÁÉÍÓÚÜÑ])', r'\1 \2', texto_direccion_normalizado)

    componentes_raw = re.split(r'[,\s]+', texto_direccion_normalizado)
    componentes = [c for c in componentes_raw if c and len(c) > 1]

    numeros_busqueda = [comp for comp in componentes if comp.isdigit()]
    calles_colonias_busqueda = [comp for comp in componentes if not comp.isdigit() and comp not in STOP_WORDS]
    componentes_clave = [comp for comp in componentes if comp not in STOP_WORDS]

    combinacion_principal = None
    combinacion_principal_norm = None
    if calles_colonias_busqueda and numeros_busqueda:
        combinacion_principal = f"{calles_colonias_busqueda[0]} {numeros_busqueda[0]}"
        combinacion_principal_norm = normalizar_texto(combinacion_principal)

    # ALMACENAMIENTO DE RESULTADOS
    todos_resultados_detalle: Dict[str, Dict[str, Any]] = {}

    # BÚSQUEDA ITERATIVA EN TODOS LOS ÍNDICES
    for fuente, index in indices.items():
        try:
            # BÚSQUEDA EXACTA POR METADATOS
            if combinacion_principal_norm:
                for campo_exacto in CAMPOS_BUSQUEDA_EXACTA:
                    campo_norm = normalizar_texto(campo_exacto)
                    try:
                        filters = MetadataFilters(filters=[
                            ExactMatchFilter(key=campo_norm, value=combinacion_principal_norm)
                        ])
                        retriever_exacto = VectorIndexRetriever(
                            index=index,
                            similarity_top_k=5,
                            filters=filters
                        )
                        nodes_exactos = retriever_exacto.retrieve(combinacion_principal)

                        if nodes_exactos:
                            for node in nodes_exactos:
                                metadata = node.node.metadata
                                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                                if not id_registro: id_registro = node.node.node_id

                                puntaje_actual = todos_resultados_detalle.get(id_registro, {}).get('puntaje', -1.0)
                                puntaje_nuevo = 1.0

                                if puntaje_nuevo > puntaje_actual:
                                    resumen = [f"{k}: {v}" for k, v in metadata.items()
                                               if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                                    todos_resultados_detalle[id_registro] = {
                                        'texto_base': f"Coincidencia exacta directa en {fuente}:\n" + "\n".join(resumen),
                                        'puntaje': puntaje_nuevo,
                                        'fuente': fuente,
                                        'id': id_registro,
                                        'metadata': metadata,
                                        'tipo': 'exacta_directa'
                                    }


                    except Exception as e_filter:
                        if 'Metadata key' not in str(e_filter):
                            print(f"[WARN] Error en búsqueda exacta por filtro en campo '{campo_exacto}' en {fuente}: {e_filter}")

            # BÚSQUEDA SEMÁNTICA Y EVALUACIÓN
            top_k_dinamico = min(10000, len(index.docstore.docs))
            retriever_semantico = VectorIndexRetriever(
                index=index,
                similarity_top_k=top_k_dinamico
            )
            consulta_semantica = " ".join(componentes_clave)
            if not consulta_semantica: consulta_semantica = texto_direccion_normalizado

            nodes_semanticos = retriever_semantico.retrieve(consulta_semantica)

            for node in nodes_semanticos:
                metadata = node.node.metadata
                id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                if not id_registro: id_registro = node.node.node_id

                puntaje_actual = todos_resultados_detalle.get(id_registro, {}).get('puntaje', -1.0)

                if puntaje_actual == 1.0:
                    continue

                texto_completo_registro = ""
                for k, v in metadata.items():
                    if k in CAMPOS_DIRECCION:
                        texto_completo_registro += f" {v}"
                texto_completo_registro_norm = normalizar_texto(texto_completo_registro)

                if not texto_completo_registro_norm:
                    continue

                tiene_numero_exacto = False
                tiene_numero_cercano = False
                numeros_en_registro = re.findall(r'\b\d+\b', texto_completo_registro_norm)

                if numeros_busqueda:
                    num_b_str = numeros_busqueda[0]
                    if num_b_str in numeros_en_registro:
                        tiene_numero_exacto = True
                    else:
                        try:
                            num_b_int = int(num_b_str)
                            for num_r_str in numeros_en_registro:
                                try:
                                    num_r_int = int(num_r_str)
                                    if abs(num_b_int - num_r_int) <= TOLERANCIA_NUMERO_CERCANO:
                                        tiene_numero_cercano = True
                                        break
                                except ValueError: continue
                        except ValueError: pass
                else:
                    tiene_numero_exacto = True
                    tiene_numero_cercano = True

                # COMPONENTES CLAVE
                componentes_clave_encontrados = sum(1 for comp in componentes_clave if comp in texto_completo_registro_norm)
                porcentaje_clave = componentes_clave_encontrados / len(componentes_clave) if componentes_clave else 1.0

                # TODOS LOS COMPONENTES
                componentes_encontrados = sum(1 for comp in componentes if comp in texto_completo_registro_norm)
                calificacion_componentes = componentes_encontrados / len(componentes) if componentes else 1.0

                # SIMILITUD TEXTUAL
                similitud_textual = similitud(texto_direccion_normalizado, texto_completo_registro_norm)

                # CALCULAR SCORE
                score_final = 0.0
                peso_num_exacto = 0.50
                peso_num_cercano = 0.20
                peso_clave = 0.35
                peso_componentes = 0.05
                peso_similitud = 0.10

                if tiene_numero_exacto:
                    score_final = (peso_num_exacto * 1.0) + (peso_clave * porcentaje_clave) + \
                                  (peso_componentes * calificacion_componentes) + (peso_similitud * similitud_textual)
                    # SI LA PRIMERA CALLE/COLONIA COINCIDE
                    if calles_colonias_busqueda and calles_colonias_busqueda[0] in texto_completo_registro_norm:
                        score_final = min(0.99, score_final * 1.1)
                elif tiene_numero_cercano:
                    score_final = (peso_num_cercano * 1.0) + (peso_clave * porcentaje_clave) + \
                                  (peso_componentes * calificacion_componentes) + (peso_similitud * similitud_textual)
                    score_final *= 0.85
                else:
                    if porcentaje_clave > 0.6:
                        score_final = (peso_clave * porcentaje_clave) + \
                                      (peso_componentes * calificacion_componentes) + (peso_similitud * similitud_textual)
                        score_final *= 0.65
                    else:
                        score_final = 0.0

                # FILTRADO ESTRICTO
                if numeros_busqueda and not (tiene_numero_exacto or tiene_numero_cercano):
                    continue
                if porcentaje_clave < 0.4 and not tiene_numero_exacto:
                    continue
                if score_final < (UMBRAL_PUNTAJE_MINIMO - 0.1):
                     continue

                if score_final > puntaje_actual:
                    tipo_resultado = "exacta_semantica" if tiene_numero_exacto and porcentaje_clave >= 0.8 else \
                                     "cercana_semantica" if tiene_numero_cercano else \
                                     "similar_semantica"

                    resumen = [f"{k}: {v}" for k, v in metadata.items()
                               if k not in ['fuente', 'archivo', 'fila_excel'] and v]

                    texto_display = f"Coincidencia ({tipo_resultado.replace('_', ' ')}) en {fuente} (Score: {score_final:.2f}):\n" + "\n".join(resumen)

                    todos_resultados_detalle[id_registro] = {
                        'texto_base': texto_display,
                        'puntaje': score_final,
                        'fuente': fuente,
                        'id': id_registro,
                        'metadata': metadata,
                        'tipo': tipo_resultado
                    }

        except Exception as e_index:
            print(f"[ERROR] Error procesando el índice {fuente}: {e_index}")
            continue

    # CONSOLIDACIÓN Y FORMATEO FINAL
    if not todos_resultados_detalle:
        return f"No se encontraron coincidencias relevantes para la dirección '{texto_direccion}'."

    resultados_ordenados = sorted(todos_resultados_detalle.values(), key=lambda x: x['puntaje'], reverse=True)

    resultados_filtrados = [res for res in resultados_ordenados if res['puntaje'] >= UMBRAL_PUNTAJE_MINIMO]

    if not resultados_filtrados and resultados_ordenados:
        resultados_finales = resultados_ordenados
        mensaje_intro = "No se encontraron coincidencias muy relevantes. Mostrando los más cercanos:\n\n"
    elif not resultados_filtrados and not resultados_ordenados:
         return f"No se encontraron coincidencias relevantes para la dirección '{texto_direccion}'."
    else:
        resultados_finales = resultados_filtrados
        tipos_encontrados = {res['tipo'] for res in resultados_finales}
        if 'exacta_directa' in tipos_encontrados or 'exacta_semantica' in tipos_encontrados:
             mensaje_intro = "Se encontraron las siguientes coincidencias:\n\n"
        elif 'cercana_semantica' in tipos_encontrados:
             mensaje_intro = "No se encontraron coincidencias exactas. Mostrando direcciones con números/componentes similares:\n\n"
        else:
             mensaje_intro = "No se encontraron coincidencias muy precisas. Mostrando los resultados más similares:\n\n"


    textos_resultados = []
    for res in resultados_finales:
        texto_limpio = re.sub(r'\s*\(Score: \d+\.\d+\)', '', res['texto_base']).strip() # LIMPIAR SCORE
        textos_resultados.append(texto_limpio)

    contador_resultados = f"\n\nSE ENCONTRARON {len(textos_resultados)} RESULTADOS."

    return convertir_a_mayusculas(mensaje_intro + "\n\n".join(textos_resultados) + contador_resultados)

buscar_direccion_tool = FunctionTool.from_defaults(
    fn=buscar_direccion_combinada,
    name="buscar_direccion_combinada",
    description=(
        "Usa esta herramienta cuando el usuario busca una dirección completa o parcial que contenga calle, número y posiblemente colonia o ciudad. "
        "Es especialmente útil para direcciones combinadas como 'ZOQUIPAN 1260, LAGOS DEL COUNTRY'. "
        "Por ejemplo: '¿De quién es esta dirección: ZOQUIPAN 1260, LAGOS DEL COUNTRY, ZAPOPAN?', "
        "'Busca Malva 101, San Luis de la Paz', 'Quién vive en casa #63, colinas del rey, Zapopan', 'información de zoquipan 1260'. "
        "Esta herramienta realiza búsquedas semánticas y exactas en componentes de dirección."
    )
)

all_tools.insert(3, buscar_direccion_tool)