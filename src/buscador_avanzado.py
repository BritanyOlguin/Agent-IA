import os
import re
import torch
import numpy as np
from difflib import SequenceMatcher
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.retrievers import VectorIndexRetriever
from utils import normalizar_texto

class BuscadorAvanzado:
    def __init__(self, ruta_indices, llm, embed_model):
        self.ruta_indices = ruta_indices
        self.llm = llm
        self.embed_model = embed_model
        self.indices = {}
        self.campos_por_indice = {}
        self.todos_los_campos = set()
        
        # Cargar todos los índices y detectar campos disponibles
        self._cargar_indices()
        self._detectar_campos()
        
        # Definir mapeo de campos y sinónimos
        self._definir_mapeo_campos()
        
    def _cargar_indices(self):
        """Carga todos los índices disponibles en la ruta especificada"""
        print(f"\nCargando índices desde: {self.ruta_indices}")
        for nombre_dir in os.listdir(self.ruta_indices):
            ruta_indice = os.path.join(self.ruta_indices, nombre_dir)
            if not os.path.isdir(ruta_indice):
                continue
            if not os.path.exists(os.path.join(ruta_indice, "docstore.json")):
                continue
            
            fuente = nombre_dir.replace("index_", "")
            try:
                print(f"Cargando índice para: {fuente}")
                storage_context = StorageContext.from_defaults(persist_dir=ruta_indice)
                index = load_index_from_storage(storage_context)
                self.indices[fuente] = index
            except Exception as e:
                print(f"Error al cargar índice {ruta_indice}: {e}")
    
    def _detectar_campos(self):
        """Detecta todos los campos disponibles en los documentos indexados"""
        for fuente, index in self.indices.items():
            campos_fuente = set()
            for node_id, doc in index.docstore.docs.items():
                if hasattr(doc, 'metadata') and doc.metadata:
                    campos_fuente.update(doc.metadata.keys())
                break
            self.campos_por_indice[fuente] = list(campos_fuente)
            self.todos_los_campos.update(campos_fuente)
        print(f"Campos detectados: {', '.join(sorted(self.todos_los_campos))}")
    
    def _definir_mapeo_campos(self):
        """Define mapeos de campos y sinónimos para normalización"""
        # Mapeo de categorías de campos
        self.categorias_campos = {
            "nombre": ["nombre", "nombre completo", "nombrecompleto", "nombreafiliado"],
            "direccion": ["direccion", "dirección", "domicilio", "calle", "domicilio completo", "direccioncompleta"],
            "colonia": ["colonia", "fraccionamiento", "barrio", "asentamiento", "col"],
            "municipio": ["municipio", "delegacion", "delegación", "alcaldia", "alcaldía", "mpio", "ciudad"],
            "cp": ["cp", "código postal", "codigo postal", "codigopostal"],
            "estado": ["estado", "entidad", "entidadfederativa", "estado_provincia"],
            "telefono": ["telefono", "teléfono", "tel", "celular", "móvil", "movil", "contacto"],
            "correo": ["correo", "email", "e-mail", "mail", "correoelectronico"],
            "id": ["id", "clave", "folio", "identificador", "numero", "número", "no."],
            "fecha": ["fecha", "fechaafiliacion", "fecha afiliacion", "fechanacimiento", "fecha nacimiento"],
            "sexo": ["sexo", "genero", "género"],
        }
        
        # Crear mapeo inverso para normalización
        self.mapeo_campos = {}
        for categoria, aliases in self.categorias_campos.items():
            for alias in aliases:
                self.mapeo_campos[normalizar_texto(alias)] = categoria
        
        # Agregar campos detectados que no están en categorías predefinidas
        for campo in self.todos_los_campos:
            campo_norm = normalizar_texto(campo)
            if campo_norm not in self.mapeo_campos:
                self.mapeo_campos[campo_norm] = campo_norm
    
    def busqueda_estructurada(self, campo, valor, top_k=5):
        """
        Realiza una búsqueda estructurada especificando campo y valor.
        
        Args:
            campo (str): Campo específico a buscar
            valor (str): Valor a buscar en ese campo
            top_k (int): Cantidad máxima de resultados a devolver
            
        Returns:
            list: Lista de resultados encontrados
        """
        campo_norm = normalizar_texto(campo)
        valor_norm = normalizar_texto(valor)
        
        # Obtener la categoría normalizada del campo
        categoria = self.mapeo_campos.get(campo_norm, campo_norm)
        
        print(f"Buscando '{valor}' en campo '{campo}' (normalizado como '{categoria}')")
        resultados = []
        
        for fuente, index in self.indices.items():
            try:
                # Buscar campos que correspondan a la categoría buscada
                campos_a_buscar = self._obtener_campos_categoria(fuente, categoria)
                
                if not campos_a_buscar:
                    continue
                    
                for campo_actual in campos_a_buscar:
                    filters = MetadataFilters(filters=[
                        MetadataFilter.from_dict({
                            "key": campo_actual,
                            "value": valor_norm,
                            "operator": FilterOperator.EQ
                        })
                    ])
                    
                    retriever = VectorIndexRetriever(
                        index=index, 
                        similarity_top_k=top_k, 
                        filters=filters
                    )
                    
                    nodes = retriever.retrieve(f"{campo_actual} es {valor}")
                    
                    for node in nodes:
                        metadata = node.node.metadata
                        resultado = {
                            "fuente": fuente,
                            "score": node.score if hasattr(node, 'score') else 1.0,
                            "metadata": metadata
                        }
                        resultados.append(resultado)
                        
            except Exception as e:
                print(f"Error al buscar en {fuente}: {e}")
        
        # Ordenar resultados por score
        resultados.sort(key=lambda x: x["score"], reverse=True)
        
        return resultados[:top_k]
    
    def _obtener_campos_categoria(self, fuente, categoria):
        """Obtiene los campos disponibles en un índice que correspondan a una categoría"""
        campos_disponibles = self.campos_por_indice.get(fuente, [])
        
        # Si el nombre de categoría es exactamente un campo, devolver solo ese
        if categoria in campos_disponibles:
            return [categoria]
            
        # Buscar por mapeo inverso
        campos_categoria = []
        for campo in campos_disponibles:
            campo_norm = normalizar_texto(campo)
            if self.mapeo_campos.get(campo_norm) == categoria:
                campos_categoria.append(campo)
                
        return campos_categoria
    
    def analizar_direccion(self, texto_direccion):
        """
        Analiza un texto de dirección y extrae sus componentes
        
        Args:
            texto_direccion (str): Texto completo de la dirección
            
        Returns:
            dict: Componentes de la dirección
        """
        componentes = {}
        
        # Patrones regulares para detectar partes de dirección
        patrones = {
            "calle": r"(?:calle|c\.|av\.|avenida)\s+([a-zñáéíóú\s\d]+)(?:,|\s|$)",
            "numero": r"(?:núm\.|num\.|número|numero|#)\s*(\d+(?:\s*[a-z])?)",
            "colonia": r"(?:col\.|colonia)\s+([a-zñáéíóú\s\d]+)(?:,|\s|$)",
            "cp": r"(?:c\.p\.|cp|código postal|codigo postal)\s*(\d{4,5})",
            "municipio": r"(?:mpio\.|municipio|delegación|delegacion|alcaldía|alcaldia)\s+([a-zñáéíóú\s\d]+)(?:,|\s|$)",
            "estado": r"(?:edo\.|estado)\s+([a-zñáéíóú\s\d]+)(?:,|\s|$)"
        }
        
        # Extraer componentes con expresiones regulares
        for tipo, patron in patrones.items():
            match = re.search(patron, texto_direccion.lower())
            if match:
                componentes[tipo] = match.group(1).strip()
        
        # Analizar por comas para partes no identificadas
        partes = [p.strip() for p in texto_direccion.split(',') if p.strip()]
        
        # Heurísticas para clasificar partes no identificadas
        if len(partes) >= 1 and "calle" not in componentes:
            # Primera parte suele ser nombre de calle y número
            parte_calle = partes[0]
            # Separar número si está al final
            match_num = re.search(r"(\d+(?:\s*[a-z])?)$", parte_calle)
            if match_num:
                num = match_num.group(1)
                calle = parte_calle[:match_num.start()].strip()
                componentes["calle"] = calle
                componentes["numero"] = num
            else:
                componentes["calle"] = parte_calle
        
        if len(partes) >= 2 and "colonia" not in componentes:
            # Segunda parte suele ser colonia
            componentes["colonia"] = partes[1]
            
        if len(partes) >= 3 and "municipio" not in componentes and "cp" not in componentes:
            # Tercera parte puede ser municipio/CP
            parte3 = partes[2]
            if re.search(r"\d{4,5}", parte3):
                componentes["cp"] = re.search(r"\d{4,5}", parte3).group(0)
            else:
                componentes["municipio"] = parte3
                
        if len(partes) >= 4 and "estado" not in componentes:
            # Cuarta parte suele ser estado
            componentes["estado"] = partes[3]
        
        # Verificar si hay un CP en alguna parte del texto
        if "cp" not in componentes:
            match_cp = re.search(r"\b(\d{4,5})\b", texto_direccion)
            if match_cp:
                componentes["cp"] = match_cp.group(1)
        
        # Si es una dirección muy corta, considerar que es sólo calle
        if not componentes and len(texto_direccion.split()) <= 3:
            componentes["calle"] = texto_direccion
            
        return componentes
    
    def busqueda_lenguaje_natural(self, consulta, top_k=5):
        """
        Realiza una búsqueda a partir de una consulta en lenguaje natural.
        
        Args:
            consulta (str): Consulta en lenguaje natural
            top_k (int): Cantidad máxima de resultados a devolver
            
        Returns:
            tuple: (resultados, explicacion)
        """
        print(f"Analizando consulta: '{consulta}'")
        resultados = []
        explicacion = []
        
        # 1. Verificar si es una consulta de dirección
        if self._parece_direccion(consulta):
            explicacion.append("La consulta parece contener una dirección.")
            componentes = self.analizar_direccion(consulta)
            
            if componentes:
                explicacion.append(f"Componentes detectados: {componentes}")
                
                # Buscar por cada componente, priorizando los más específicos
                prioridad = ["calle", "numero", "colonia", "cp", "municipio", "estado"]
                
                for campo in prioridad:
                    if campo in componentes:
                        valor = componentes[campo]
                        
                        # Categorizar el campo para normalización
                        categoria_campo = "direccion" if campo in ["calle", "numero"] else campo
                        
                        # Buscar el valor en todos los índices
                        resultados_campo = self.busqueda_estructurada(
                            categoria_campo, valor, top_k=top_k
                        )
                        
                        if resultados_campo:
                            explicacion.append(f"Encontrados {len(resultados_campo)} resultados buscando '{valor}' como {campo}.")
                            resultados.extend(resultados_campo)
                            
                            # Si ya encontramos suficientes resultados, no seguir buscando
                            if len(resultados) >= top_k:
                                break
        
        # 2. Verificar si es una consulta por nombre
        elif self._parece_nombre(consulta):
            explicacion.append("La consulta parece contener un nombre de persona.")
            nombre_limpio = self._extraer_nombre(consulta)
            
            if nombre_limpio:
                explicacion.append(f"Nombre extraído: '{nombre_limpio}'")
                resultados_nombre = self.busqueda_estructurada("nombre", nombre_limpio, top_k=top_k)
                
                if resultados_nombre:
                    explicacion.append(f"Encontrados {len(resultados_nombre)} resultados buscando por nombre.")
                    resultados.extend(resultados_nombre)
        
        # 3. Si no hay resultados específicos, intentar una búsqueda semántica
        if not resultados:
            explicacion.append("Realizando búsqueda semántica general.")
            resultados = self._busqueda_semantica(consulta, top_k=top_k)
            if resultados:
                explicacion.append(f"Encontrados {len(resultados)} resultados con búsqueda semántica.")
        
        # 4. Eliminar duplicados y limitar a top_k
        resultados_unicos = []
        ids_vistos = set()
        
        for res in resultados:
            # Crear un identificador único basado en la metadata
            metadata = res["metadata"]
            id_string = str(metadata.get("nombre", "")) + str(metadata.get("direccion", ""))
            
            if id_string not in ids_vistos:
                ids_vistos.add(id_string)
                resultados_unicos.append(res)
                
                if len(resultados_unicos) >= top_k:
                    break
        
        return resultados_unicos, "\n".join(explicacion)
    
    def _parece_direccion(self, texto):
        """Verifica si un texto parece ser una dirección"""
        palabras_clave_direccion = [
            "calle", "avenida", "av", "boulevard", "blvd", "callejón", "callejon",
            "colonia", "col", "fraccionamiento", "fracc", "número", "numero", "no.", "#",
            "cp", "código postal", "codigo postal", "ciudad", "municipio", "estado"
        ]
        
        texto_lower = texto.lower()
        
        # Verificar si contiene palabras clave de dirección
        if any(palabra in texto_lower for palabra in palabras_clave_direccion):
            return True
            
        # Verificar si contiene un patrón de código postal
        if re.search(r"\b\d{5}\b", texto):
            return True
            
        # Verificar si tiene estructura de dirección (varias partes separadas por coma)
        partes = texto.split(",")
        if len(partes) >= 2:
            return True
            
        return False
    
    def _parece_nombre(self, texto):
        """Verifica si un texto parece ser un nombre de persona"""
        # Limpiar la consulta de preguntas comunes
        limpio = re.sub(r"quien es|quién es|información de|informacion de|datos de|buscar a", "", texto, flags=re.IGNORECASE).strip()
        
        # Contar palabras (nombres suelen tener 2-4 palabras)
        palabras = limpio.split()
        if 2 <= len(palabras) <= 4:
            # Verificar que no tenga números ni símbolos especiales
            if not re.search(r"[\d\$\%\#\@\!\?\¿\&]", limpio):
                return True
                
        return False
    
    def _extraer_nombre(self, texto):
        """Extrae el nombre de persona de una consulta"""
        # Patrones comunes para preguntas sobre personas
        patrones = [
            r"(?:quien|quién|quienes|quiénes) (?:es|son) (?:el|la|los|las)? ?(.+?)(?:\?|$)",
            r"(?:buscar|busca|encuentra|localiza) a (.+?)(?:\?|$)",
            r"(?:información|informacion|datos|info) (?:de|sobre|acerca de) (.+?)(?:\?|$)",
            r"(?:qué sabes|que sabes) (?:de|sobre|acerca de) (.+?)(?:\?|$)"
        ]
        
        for patron in patrones:
            match = re.search(patron, texto, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        # Si no coincide con patrones, devolver el texto limpio
        return re.sub(r"[^\w\s]", "", texto).strip()
    
    def _busqueda_semantica(self, consulta, top_k=5):
        """Realiza una búsqueda semántica basada en embeddings"""
        resultados = []
        
        for fuente, index in self.indices.items():
            try:
                # Utilizar el retriever sin filtros para búsqueda semántica
                retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=top_k
                )
                
                nodes = retriever.retrieve(consulta)
                
                for node in nodes:
                    metadata = node.node.metadata
                    resultado = {
                        "fuente": fuente,
                        "score": node.score if hasattr(node, 'score') else 0.0,
                        "metadata": metadata
                    }
                    resultados.append(resultado)
                    
            except Exception as e:
                print(f"Error en búsqueda semántica para {fuente}: {e}")
        
        # Ordenar por relevancia
        resultados.sort(key=lambda x: x["score"], reverse=True)
        return resultados[:top_k]
    
    def formatear_resultados(self, resultados):
        """Formatea los resultados para presentarlos al usuario"""
        if not resultados:
            return "No se encontraron resultados para la búsqueda."
            
        texto_resultados = []
        
        for i, res in enumerate(resultados, 1):
            fuente = res["fuente"]
            metadata = res["metadata"]
            
            # Formatear los campos relevantes
            campos_formateados = []
            # Priorizar campos importantes
            campos_prioritarios = ["nombre", "direccion", "colonia", "municipio", "estado", "cp", "telefono"]
            
            # Primero agregamos los campos prioritarios
            for campo in campos_prioritarios:
                for key, value in metadata.items():
                    if normalizar_texto(key) == campo and value:
                        campos_formateados.append(f"{key}: {value}")
                        break
            
            # Luego agregar el resto de campos con datos
            for key, value in metadata.items():
                if value and key not in ["fuente", "archivo", "fila_origen", "fila_excel"]:
                    # Verificar si ya agregamos este campo (normalizado)
                    normalizado = normalizar_texto(key)
                    if not any(normalizado == normalizar_texto(c.split(":")[0].strip()) for c in campos_formateados):
                        campos_formateados.append(f"{key}: {value}")
            
            result_text = f"📌 Resultado {i} (fuente: {fuente}):\n" + "\n".join(campos_formateados)
            texto_resultados.append(result_text)
            
        return "\n\n".join(texto_resultados)


# Función para manejar consultas naturales con el buscador avanzado
def procesar_consulta_natural(consulta, buscador):
    """
    Procesa una consulta en lenguaje natural y devuelve resultados formateados
    
    Args:
        consulta (str): La consulta del usuario
        buscador (BuscadorAvanzado): Instancia del buscador avanzado
        
    Returns:
        str: Texto con los resultados formateados
    """
    print(f"💬 Procesando consulta: {consulta}")
    
    if consulta.lower() == 'salir':
        return "👋 ¡Hasta luego!"
        
    try:
        # Realizar búsqueda en lenguaje natural
        resultados, explicacion = buscador.busqueda_lenguaje_natural(consulta, top_k=5)
        
        # Formatear y devolver resultados
        texto_resultados = buscador.formatear_resultados(resultados)
        
        if "No se encontraron resultados" in texto_resultados:
            return f"🔍 No se encontraron resultados para: '{consulta}'\n\nDetalles de la búsqueda:\n{explicacion}"
        else:
            return f"🔍 Resultados para: '{consulta}'\n\n{texto_resultados}"
            
    except Exception as e:
        return f"❌ Error al procesar la consulta: {str(e)}"


# Ejemplo de implementación en el script principal
def implementar_buscador_avanzado():
    # Esta función muestra cómo integrar el buscador avanzado en tu código principal
    
    # 1. Después de la configuración del modelo y antes del bucle principal
    buscador = BuscadorAvanzado(ruta_indices=ruta_indices, llm=llm, embed_model=embed_model)
    
    print("\n🤖 Agente listo con búsqueda avanzada. Escribe tu pregunta o 'salir' para terminar.")
    
    # 2. Modificar el bucle principal de chat
    while True:
        prompt = input("Pregunta: ")
        if prompt.lower() == 'salir':
            break
        if not prompt:
            continue
            
        try:
            # Usar el procesador de consultas naturales
            respuesta = procesar_consulta_natural(prompt, buscador)
            print(f"\n{respuesta}\n")
        except Exception as e:
            print(f"❌ Ocurrió un error durante la ejecución: {e}")
    
    # Limpiar memoria al salir
    del llm, embed_model, buscador
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("\n👋 ¡Hasta luego!")