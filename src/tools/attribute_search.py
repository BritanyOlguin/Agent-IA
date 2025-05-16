import os
from src.utils.text_normalizer import normalizar_texto, convertir_a_mayusculas
from src.core.engine import indices, all_tools, mapa_campos, campos_clave
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import FunctionTool
from src.core.config import ruta_indices

# --- 4) HERRAMIENTA 2: BUSCAR PERSONAS POR ATRIBUTO ---
def buscar_atributo(campo: str, valor: str, carpeta_indices: str) -> str:
    """
    Busca coincidencias por campo y valor en todos los índices.
    Versión mejorada con capacidad de búsqueda flexible y manejo de categorías.
    
    Características:
    - Búsqueda exacta con filtros de metadatos cuando campo es específico
    - Búsqueda en todos los campos cuando no se especifica campo o cuando no hay resultados
    - Manejo especial para categorías como sexo y ocupación
    - Normalización de valores para mejorar coincidencias
    """
    print(f"\nBuscando registros donde '{campo}' = '{valor}'\n")
    
    # NORMALIZAR CAMPO Y VALOR
    campo_normalizado = normalizar_texto(campo) if campo else ""
    valor_normalizado = normalizar_texto(valor)
    
    # CASOS ESPECIALES DE NORMALIZACIÓN
    if campo_normalizado in ["sexo", "genero"]:
        if valor_normalizado in ["hombre", "hombres", "masculino", "varon", "varones", "m"]:
            valor_normalizado = "m"
        elif valor_normalizado in ["mujer", "mujeres", "femenino", "f"]:
            valor_normalizado = "f"
    
    resultados = []
    registros_encontrados = set()
    
    busqueda_categorica = campo_normalizado in ["sexo", "genero", "ocupacion", "profesion"]
    
    campo_final = mapa_campos.get(campo_normalizado, campo_normalizado) if campo_normalizado else ""
    
    # FASE 1: BÚSQUEDA EXACTA POR FILTROS SI TENEMOS UN CAMPO ESPECÍFICO
    if campo_final and not busqueda_categorica:
        for nombre_dir in os.listdir(carpeta_indices):
            ruta_indice = os.path.join(carpeta_indices, nombre_dir)
            if not os.path.isdir(ruta_indice) or not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
                continue

            fuente = nombre_dir.replace("index_", "")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
                index = load_index_from_storage(storage_context)

                # BÚSQUEDA EXACTA CON FILTRO
                try:
                    filters = MetadataFilters(filters=[
                        ExactMatchFilter(key=campo_final, value=valor_normalizado)
                    ])
                    top_k_dinamico = min(10000, len(index.docstore.docs))
                    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k_dinamico, filters=filters)
                    nodes = retriever.retrieve(f"{campo_final} es {valor}")

                    if nodes:
                        for node in nodes:
                            metadata = node.node.metadata
                            id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                            if not id_registro:
                                id_registro = node.node.node_id
                                
                            if id_registro in registros_encontrados:
                                continue
                                
                            resumen = [f"{k}: {v}" for k, v in metadata.items() 
                                      if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                            resultados.append(f"Coincidencia exacta en {fuente}:\n" + "\n".join(resumen))
                            registros_encontrados.add(id_registro)
                except Exception as e_filter:
                    print(f"Error en filtro exacto para {fuente}: {e_filter}")
                    pass

            except Exception as e:
                print(f"Error al cargar índice {fuente}: {e}")
                continue
    
    # FASE 2: BUSCAR EN TODOS LOS DOCUMENTOS
    if busqueda_categorica or (not resultados and valor):
        print(f"Realizando búsqueda exhaustiva para '{valor_normalizado}'...")
        
        campos_a_buscar = []
        if campo_final:
            if campo_final in campos_clave:
                campos_a_buscar = [normalizar_texto(c) for c in campos_clave[campo_final]]
            else:
                campos_a_buscar = [campo_final]
        
        # RECORRER TODOS LOS ÍNDICES
        for fuente, index in indices.items():
            try:
                for node_id, doc in index.docstore.docs.items():
                    metadata = doc.metadata
                    if not metadata:
                        continue
                    
                    # CREAR ID ÚNICO
                    id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", ""))
                    if not id_registro:
                        id_registro = node_id
                    
                    if id_registro in registros_encontrados:
                        continue
                    
                    encontrado = False
                    coincidencia_campo = None
                    
                    # CAMPOS ESPECÍFICOS A BUSCAR
                    if campos_a_buscar:
                        for k, v in metadata.items():
                            k_norm = normalizar_texto(k)
                            
                            if k_norm in campos_a_buscar:
                                v_str = str(v).strip().lower()
                                v_norm = normalizar_texto(v_str)
                                
                                if v_norm == valor_normalizado or (
                                    len(valor_normalizado) > 4 and (
                                        valor_normalizado in v_norm or 
                                        v_norm in valor_normalizado
                                    )
                                ):
                                    encontrado = True
                                    coincidencia_campo = k
                                    break
                    
                    if not encontrado and len(valor_normalizado) >= 4:
                        for k, v in metadata.items():
                            if k in ['fuente', 'archivo', 'fila_excel']:
                                continue
                                
                            v_str = str(v).strip().lower()
                            v_norm = normalizar_texto(v_str)
                            
                            if v_norm == valor_normalizado or (
                                len(valor_normalizado) > 6 and 
                                valor_normalizado in v_norm
                            ):
                                encontrado = True
                                coincidencia_campo = k
                                break
                    
                    if encontrado:
                        tipo_coincidencia = "exacta" if coincidencia_campo else "en múltiples campos"
                        campo_texto = f" en campo '{coincidencia_campo}'" if coincidencia_campo else ""
                        
                        resumen = [f"{k}: {v}" for k, v in metadata.items() 
                                   if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                        resultados.append(f"Coincidencia {tipo_coincidencia}{campo_texto} en {fuente}:\n" + "\n".join(resumen))
                        registros_encontrados.add(id_registro)
                        
            except Exception as e:
                print(f"Error al buscar en índice {fuente}: {e}")
                continue
    
    # FORMATEAR RESPUESTA FINAL
    if resultados:
        total_registros = len(resultados)
        num_mostrados = total_registros
        
        if campo:
            mensaje_intro = f"Se encontraron {total_registros} registros para {campo}='{valor}'."
        else:
            mensaje_intro = f"Se encontraron {total_registros} registros que contienen '{valor}'."
        
        if total_registros > num_mostrados:
            mensaje_intro += f" Mostrando {num_mostrados} primeros resultados:"
        
        return convertir_a_mayusculas(mensaje_intro + "\n\n" + "\n\n".join(resultados[:num_mostrados]))
    else:
        if campo:
            return convertir_a_mayusculas(f"No se encontraron coincidencias para '{campo}: {valor}'.")
        else:
            return convertir_a_mayusculas(f"No se encontraron coincidencias para el valor '{valor}'.")

buscar_por_atributo_tool = FunctionTool.from_defaults(
    fn=lambda campo, valor: buscar_atributo(campo, valor, carpeta_indices=ruta_indices),
    name="buscar_atributo",
    description=(
        "Usa esta herramienta cuando el usuario busca por cualquier atributo específico o valor. "
        "Funciona tanto si se especifica el campo ('¿Quién tiene la clave IFE ABCDE?') "
        "como si solo se da un valor sin campo ('A quién pertenece ABCDE?'). "
        "También maneja categorías como sexo ('hombres', 'mujeres') y ocupación ('ingeniero', 'médico'). "
        "Por ejemplo: '¿Quién tiene el número 5544332211?', '¿A quién pertenece la CURP ABCD123?', "
        "'¿Qué personas viven en Querétaro?', 'Muestra a todas las mujeres', 'Busca ingenieros'."
    )
)
all_tools.insert(1, buscar_por_atributo_tool)