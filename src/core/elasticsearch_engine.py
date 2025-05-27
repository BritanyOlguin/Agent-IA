"""
Motor de búsqueda usando Elasticsearch para consultas inteligentes y tolerantes a errores.
Reemplaza las búsquedas tradicionales con capacidades avanzadas.
"""

import os
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime
from unidecode import unidecode

class ElasticsearchEngine:
    """
    Motor de búsqueda avanzado usando Elasticsearch que permite:
    - Búsquedas tolerantes a errores ortográficos
    - Consultas en lenguaje natural
    - Búsquedas combinadas complejas
    - Análisis semántico
    - Velocidad extrema incluso con millones de registros
    """
    
    def __init__(self, host='localhost', port=9200):
        """Inicializa la conexión con Elasticsearch"""
        self.host = host
        self.port = port
        self.index_name = "agente_ciudadanos"
        
        # Configurar cliente de Elasticsearch
        self.es = Elasticsearch(
            [{'host': host, 'port': port, 'scheme': 'http'}],
            timeout=30,
            max_retries=10,
            retry_on_timeout=True,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Verificar conexión
        self._verify_connection()
        
        # Configurar índice si no existe
        self._setup_index()
    
    def _verify_connection(self):
        """Verifica que Elasticsearch esté ejecutándose"""
        try:
            info = self.es.info()
            print(f"✅ Conectado a Elasticsearch: {info['version']['number']}")
        except ConnectionError:
            raise Exception(
                "❌ No se puede conectar a Elasticsearch.\n"
                "Asegúrate de que esté ejecutándose en http://localhost:9200"
            )
    
    def _setup_index(self):
        """Configura el índice con mappings optimizados para búsquedas inteligentes"""
        
        # Configuración del índice con analizadores personalizados
        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "nombres_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "asciifolding",  # Remueve acentos
                                "nombres_synonym_filter",
                                "edge_ngram_filter"
                            ]
                        },
                        "direcciones_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "asciifolding",
                                "direcciones_synonym_filter"
                            ]
                        },
                        "telefono_analyzer": {
                            "type": "custom",
                            "tokenizer": "keyword",
                            "filter": ["lowercase"]
                        }
                    },
                    "filter": {
                        "nombres_synonym_filter": {
                            "type": "synonym",
                            "synonyms": [
                                "jose,pepe,chepe",
                                "francisco,paco,pancho",
                                "jesus,chuy,chucho",
                                "maria,mary",
                                "guadalupe,lupe",
                                "alejandro,alex",
                                "doctor,dr,medico",
                                "ingeniero,ing",
                                "profesor,maestro,profe"
                            ]
                        },
                        "direcciones_synonym_filter": {
                            "type": "synonym",
                            "synonyms": [
                                "avenida,av,ave",
                                "calle,c",
                                "boulevard,blvd",
                                "colonia,col",
                                "fraccionamiento,fracc",
                                "guadalajara,gdl",
                                "ciudad de mexico,df,cdmx"
                            ]
                        },
                        "edge_ngram_filter": {
                            "type": "edge_ngram",
                            "min_gram": 2,
                            "max_gram": 15
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "nombre_completo": {
                        "type": "text",
                        "analyzer": "nombres_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "suggest": {
                                "type": "completion",
                                "analyzer": "nombres_analyzer"
                            }
                        }
                    },
                    "telefono_completo": {
                        "type": "text",
                        "analyzer": "telefono_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "direccion": {
                        "type": "text",
                        "analyzer": "direcciones_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "municipio": {
                        "type": "text",
                        "analyzer": "direcciones_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "estado": {
                        "type": "text",
                        "analyzer": "direcciones_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "ocupacion": {
                        "type": "text",
                        "analyzer": "nombres_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "edad": {"type": "integer"},
                    "sexo": {"type": "keyword"},
                    "cp": {"type": "keyword"},
                    "fuente": {"type": "keyword"},
                    "fecha_indexado": {"type": "date"}
                }
            }
        }
        
        # Crear índice si no existe
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=index_settings)
            print(f"✅ Índice '{self.index_name}' creado con configuración optimizada")
        else:
            print(f"✅ Índice '{self.index_name}' ya existe")
    
    def index_data_from_files(self, data_folder: str):
        """
        Indexa todos los archivos de datos en Elasticsearch
        Reemplaza el proceso de indexación actual
        """
        print(f"🔄 Indexando datos desde: {data_folder}")
        
        if not os.path.exists(data_folder):
            print(f"❌ Carpeta no encontrada: {data_folder}")
            return
        
        total_indexed = 0
        
        for archivo_nombre in os.listdir(data_folder):
            ruta_archivo = os.path.join(data_folder, archivo_nombre)
            if not os.path.isfile(ruta_archivo):
                continue
            
            print(f"📁 Procesando: {archivo_nombre}")
            
            # Cargar datos del archivo
            df = self._cargar_dataframe(ruta_archivo)
            if df is None or df.empty:
                print(f"⚠️ No se pudo cargar: {archivo_nombre}")
                continue
            
            # Preparar documentos para indexar
            docs = self._prepare_documents(df, archivo_nombre)
            
            # Indexar en lotes para mejor rendimiento
            indexed_count = self._bulk_index(docs)
            total_indexed += indexed_count
            
            print(f"✅ Indexados {indexed_count} registros de {archivo_nombre}")
        
        print(f"🎯 Total indexado: {total_indexed} registros")
        return total_indexed
    
    def _cargar_dataframe(self, archivo: str) -> Optional[pd.DataFrame]:
        """Carga un DataFrame desde diferentes formatos"""
        ext = os.path.splitext(archivo)[1].lower()
        print(f"📄 Detectado archivo: {os.path.basename(archivo)} (tipo: {ext})")
        
        try:
            if ext in ['.xlsx', '.xls']:
                print("   Cargando archivo Excel...")
                df = pd.read_excel(archivo, dtype=str).fillna("")
                print(f"   ✅ Leídas {len(df)} filas, {len(df.columns)} columnas")
                return df
                
            elif ext == '.csv':
                print("   Cargando archivo CSV...")
                try:
                    df = pd.read_csv(archivo, dtype=str, encoding='utf-8').fillna("")
                    print(f"   ✅ Leídas {len(df)} filas, {len(df.columns)} columnas")
                    return df
                except UnicodeDecodeError:
                    print("   Probando con encoding latin-1...")
                    df = pd.read_csv(archivo, dtype=str, encoding='latin-1').fillna("")
                    print(f"   ✅ Leídas {len(df)} filas, {len(df.columns)} columnas")
                    return df
                    
            elif ext in ['.mdb', '.accdb']:
                print("   Cargando base de datos Access...")
                try:
                    import pyodbc
                    conn_str = (
                        r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
                        f"DBQ={archivo};"
                    )
                    conn = pyodbc.connect(conn_str)
                    cursor = conn.cursor()
                    
                    # Obtener todas las tablas
                    tables = [row.table_name for row in cursor.tables(tableType='TABLE') 
                            if not row.table_name.startswith('MSys')]
                    
                    if not tables:
                        print("   ❌ No se encontraron tablas en la base de datos")
                        return None
                    
                    print(f"   📋 Tablas encontradas: {tables}")
                    tabla_principal = tables[0]  # Usar la primera tabla
                    print(f"   📊 Usando tabla: {tabla_principal}")
                    
                    # Leer datos de la tabla
                    query = f"SELECT * FROM [{tabla_principal}]"
                    df = pd.read_sql(query, conn).fillna("")
                    conn.close()
                    
                    print(f"   ✅ Leídas {len(df)} filas, {len(df.columns)} columnas")
                    return df
                    
                except ImportError:
                    print("   ❌ pyodbc no está instalado. Ejecuta: pip install pyodbc")
                    return None
                except Exception as e:
                    print(f"   ❌ Error cargando Access: {e}")
                    return None
                    
            elif ext == '.txt':
                print("   Cargando archivo de texto...")
                try:
                    # Intentar detectar el delimitador
                    with open(archivo, 'r', encoding='utf-8') as f:
                        primera_linea = f.readline()
                    
                    # Detectar delimitador más probable
                    delimitadores = ['\t', '|', ';', ',']
                    mejor_delimitador = '\t'  # Por defecto tab
                    
                    for delim in delimitadores:
                        if primera_linea.count(delim) > 0:
                            mejor_delimitador = delim
                            break
                    
                    print(f"   🔍 Delimitador detectado: '{mejor_delimitador}'")
                    
                    df = pd.read_csv(archivo, dtype=str, encoding='utf-8', 
                                sep=mejor_delimitador, on_bad_lines='skip').fillna("")
                    print(f"   ✅ Leídas {len(df)} filas, {len(df.columns)} columnas")
                    return df
                    
                except UnicodeDecodeError:
                    print("   Probando con encoding latin-1...")
                    df = pd.read_csv(archivo, dtype=str, encoding='latin-1', 
                                sep=mejor_delimitador, on_bad_lines='skip').fillna("")
                    print(f"   ✅ Leídas {len(df)} filas, {len(df.columns)} columnas")
                    return df
                except Exception as e:
                    print(f"   ❌ Error procesando TXT: {e}")
                    return None
            else:
                print(f"   ⚠️ Formato no soportado: {ext}")
                return None
                
        except Exception as e:
            print(f"   ❌ Error cargando archivo {archivo}: {e}")
            return None
    
    def _prepare_documents(self, df: pd.DataFrame, fuente: str) -> List[Dict]:
        """Prepara documentos para indexar en Elasticsearch"""
        print(f"🔄 Preparando documentos de {fuente}...")
        print(f"📊 Columnas disponibles: {list(df.columns)}")
        
        docs = []
        
        # Mapeo mejorado de columnas (más flexible)
        column_mapping = {
            'nombre_completo': [
                'nombre_completo', 'nombre completo', 'nombre y apellidos', 'nombre_y_apellidos',
                'NOMBRE_COMPLETO', 'NOMBRE COMPLETO', 'nombre', 'NOMBRE', 'Name', 'NAME'
            ],
            'telefono_completo': [
                'telefono', 'teléfono', 'tel', 'celular', 'telefono_completo', 'TELEFONO',
                'TELÉFONO', 'TEL', 'CELULAR', 'phone', 'PHONE', 'numero_telefono'
            ],
            'direccion': [
                'direccion', 'dirección', 'domicilio', 'calle', 'DIRECCION', 'DIRECCIÓN',
                'DOMICILIO', 'CALLE', 'address', 'ADDRESS', 'domicilio_completo'
            ],
            'municipio': [
                'municipio', 'ciudad', 'MUNICIPIO', 'CIUDAD', 'city', 'CITY', 'localidad'
            ],
            'estado': [
                'estado', 'entidad', 'ESTADO', 'ENTIDAD', 'state', 'STATE', 'provincia'
            ],
            'ocupacion': [
                'ocupacion', 'ocupación', 'profesion', 'profesión', 'trabajo', 'empleo',
                'OCUPACION', 'OCUPACIÓN', 'PROFESION', 'PROFESIÓN', 'TRABAJO', 'job'
            ],
            'edad': [
                'edad', 'años', 'EDAD', 'AÑOS', 'age', 'AGE'
            ],
            'sexo': [
                'sexo', 'genero', 'género', 'SEXO', 'GENERO', 'GÉNERO', 'gender', 'sex'
            ],
            'cp': [
                'cp', 'codigo postal', 'código postal', 'CP', 'CODIGO POSTAL', 'zip'
            ]
        }
        
        # Detectar automáticamente las columnas que coinciden
        campo_detectado = {}
        columnas_usadas = set()
        
        for target_field, possible_columns in column_mapping.items():
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [p.lower() for p in possible_columns] and col not in columnas_usadas:
                    campo_detectado[target_field] = col
                    columnas_usadas.add(col)
                    print(f"   🎯 Detectado: {target_field} -> {col}")
                    break
        
        if not campo_detectado:
            print("   ⚠️ No se detectaron campos conocidos, usando nombres originales...")
            # Si no se detecta nada, usar las primeras columnas
            cols = list(df.columns)
            if len(cols) > 0: campo_detectado['nombre_completo'] = cols[0]
            if len(cols) > 1: campo_detectado['telefono_completo'] = cols[1]
            if len(cols) > 2: campo_detectado['direccion'] = cols[2]
        
        # Procesar cada fila
        docs_creados = 0
        for idx, row in df.iterrows():
            doc = {
                'fuente': fuente,
                'fecha_indexado': datetime.now().isoformat(),
                'id_original': str(idx)
            }
            
            tiene_datos = False
            
            # Mapear campos detectados
            for target_field, source_col in campo_detectado.items():
                if source_col in df.columns:
                    valor = row[source_col]
                    if pd.notna(valor) and str(valor).strip() and str(valor).strip().lower() != 'nan':
                        doc[target_field] = str(valor).strip()
                        tiene_datos = True
            
            # Agregar otros campos que no están en el mapeo
            for col in df.columns:
                if col not in columnas_usadas:
                    valor = row[col]
                    if pd.notna(valor) and str(valor).strip() and str(valor).strip().lower() != 'nan':
                        # Normalizar nombre del campo
                        field_name = col.lower().replace(' ', '_').replace('-', '_')
                        doc[field_name] = str(valor).strip()
                        tiene_datos = True
            
            # Solo agregar si tiene datos útiles
            if tiene_datos and len(doc) > 3:  # Más que solo metadatos
                docs.append(doc)
                docs_creados += 1
        
        print(f"   ✅ Preparados {docs_creados} documentos válidos de {len(df)} filas")
        
        # Mostrar ejemplo de documento
        if docs:
            print(f"   📋 Ejemplo de documento:")
            ejemplo = docs[0]
            for key, value in list(ejemplo.items())[:5]:  # Solo mostrar 5 campos
                print(f"      {key}: {value}")
        
        return docs
    
    def _bulk_index(self, docs: List[Dict], batch_size: int = 100) -> int:
        """Indexa documentos en lotes para mejor rendimiento con grandes volúmenes"""
        from elasticsearch.helpers import bulk
        import time
        
        print(f"📦 Indexando {len(docs)} documentos en lotes de {batch_size}...")
        
        total_success = 0
        total_failed = 0
        
        # Procesar en lotes
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            lote_num = (i // batch_size) + 1
            total_lotes = (len(docs) + batch_size - 1) // batch_size
            
            print(f"   📤 Procesando lote {lote_num}/{total_lotes} ({len(batch)} docs)...")
            
            def doc_generator():
                for doc in batch:
                    try:
                        # Limpiar datos problemáticos
                        clean_doc = {}
                        for key, value in doc.items():
                            if value and str(value).strip() and len(str(value).strip()) > 0:
                                # Limpiar valores muy largos o problemáticos
                                clean_value = str(value).strip()
                                if len(clean_value) > 1000:  # Limitar longitud
                                    clean_value = clean_value[:1000]
                                clean_doc[key] = clean_value
                        
                        if len(clean_doc) > 3:  # Solo indexar si tiene suficientes datos
                            yield {
                                "_index": self.index_name,
                                "_source": clean_doc
                            }
                    except Exception as e:
                        print(f"      ⚠️ Saltando documento problemático: {e}")
                        continue
            
            try:
                # Intentar indexar el lote
                success, failed = bulk(
                    self.es, 
                    doc_generator(), 
                    chunk_size=batch_size,
                    request_timeout=60,  # Aumentar timeout
                    max_retries=3,
                    initial_backoff=2,
                    max_backoff=600
                )
                
                total_success += success
                if failed:
                    total_failed += len(failed)
                    print(f"      ⚠️ {len(failed)} documentos fallaron en este lote")
                
                print(f"      ✅ Lote {lote_num}: {success} documentos indexados")
                
                # Pausa pequeña entre lotes para no sobrecargar
                if lote_num % 10 == 0:  # Cada 10 lotes
                    print(f"      ⏸️ Pausa breve... (progreso: {total_success} indexados)")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"      ❌ Error en lote {lote_num}: {e}")
                total_failed += len(batch)
                continue
        
        print(f"📊 Indexación completada:")
        print(f"   ✅ Exitosos: {total_success}")
        print(f"   ❌ Fallidos: {total_failed}")
        
        return total_success
    
    def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Búsqueda principal que maneja todo tipo de consultas
        """
        print(f"🔍 Buscando: '{query}'")
        
        # Construir consulta de Elasticsearch
        es_query = self._build_elasticsearch_query(query)
        
        try:
            # Ejecutar búsqueda
            response = self.es.search(
                index=self.index_name,
                body=es_query,
                size=max_results
            )
            
            # Procesar resultados
            results = self._process_search_results(response, query)
            return results
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return {"total": 0, "results": [], "error": str(e)}
    
    def _build_elasticsearch_query(self, query: str) -> Dict:
        """Construye una consulta compleja de Elasticsearch"""
        
        # Limpiar y analizar la query
        clean_query = self._clean_query(query)
        
        # Detectar tipo de búsqueda
        search_type = self._detect_search_type(clean_query)
        
        # Construir query según el tipo detectado
        if search_type == "phone":
            return self._build_phone_query(clean_query)
        elif search_type == "address":
            return self._build_address_query(clean_query)
        elif search_type == "name":
            return self._build_name_query(clean_query)
        else:
            return self._build_multi_field_query(clean_query)
    
    def _clean_query(self, query: str) -> str:
        """Limpia y normaliza la consulta"""
        # Remover acentos
        clean = unidecode(query.lower())
        # Remover caracteres especiales excepto espacios y números
        clean = re.sub(r'[^\w\s]', ' ', clean)
        # Normalizar espacios
        clean = ' '.join(clean.split())
        return clean.strip()
    
    def _detect_search_type(self, query: str) -> str:
        """Detecta el tipo de búsqueda basado en la consulta"""
        
        # Patrones para detectar números telefónicos
        if re.search(r'\b\d{7,}\b', query):
            return "phone"
        
        # Patrones para direcciones
        address_keywords = ['vive', 'direccion', 'domicilio', 'calle', 'avenida', 'colonia']
        if any(keyword in query for keyword in address_keywords):
            return "address"
        
        # Patrones para nombres
        name_keywords = ['quien', 'nombre', 'llama', 'persona']
        if any(keyword in query for keyword in name_keywords):
            return "name"
        
        return "general"
    
    def _build_phone_query(self, query: str) -> Dict:
        """Construye consulta especializada para teléfonos"""
        
        # Extraer números de la consulta
        numbers = re.findall(r'\d+', query)
        
        should_clauses = []
        for number in numbers:
            should_clauses.extend([
                {"match": {"telefono_completo": {"query": number, "fuzziness": "1"}}},
                {"wildcard": {"telefono_completo.keyword": f"*{number}*"}},
                {"prefix": {"telefono_completo.keyword": number}},
                {"suffix": {"telefono_completo.keyword": number}}
            ])
        
        return {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
        }
    
    def _build_address_query(self, query: str) -> Dict:
        """Construye consulta especializada para direcciones"""
        
        return {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["direccion^3", "municipio^2", "estado"],
                                "fuzziness": "AUTO",
                                "type": "best_fields"
                            }
                        },
                        {
                            "match_phrase_prefix": {
                                "direccion": query
                            }
                        }
                    ]
                }
            }
        }
    
    def _build_name_query(self, query: str) -> Dict:
        """Construye consulta especializada para nombres"""
        
        return {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "nombre_completo": {
                                    "query": query,
                                    "fuzziness": "AUTO",
                                    "boost": 3
                                }
                            }
                        },
                        {
                            "match_phrase_prefix": {
                                "nombre_completo": {
                                    "query": query,
                                    "boost": 2
                                }
                            }
                        },
                        {
                            "wildcard": {
                                "nombre_completo.keyword": f"*{query}*"
                            }
                        }
                    ]
                }
            }
        }
    
    def _build_multi_field_query(self, query: str) -> Dict:
        """Construye consulta que busca en múltiples campos"""
        
        return {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "nombre_completo^3",
                                    "telefono_completo^2",
                                    "direccion^2",
                                    "municipio",
                                    "ocupacion",
                                    "*"
                                ],
                                "fuzziness": "AUTO",
                                "type": "best_fields"
                            }
                        },
                        {
                            "query_string": {
                                "query": f"*{query}*",
                                "fields": ["*"],
                                "default_operator": "OR"
                            }
                        }
                    ]
                }
            }
        }
    
    def _process_search_results(self, response: Dict, original_query: str) -> Dict:
        """Procesa y formatea los resultados de Elasticsearch"""
        
        hits = response.get('hits', {})
        total = hits.get('total', {}).get('value', 0)
        results = []
        
        for hit in hits.get('hits', []):
            source = hit['_source']
            score = hit['_score']
            
            # Formatear resultado para mostrar
            formatted_result = self._format_result(source, score)
            results.append(formatted_result)
        
        return {
            "total": total,
            "results": results,
            "query": original_query,
            "took": response.get('took', 0)
        }
    
    def _format_result(self, source: Dict, score: float) -> Dict:
        """Formatea un resultado individual"""
        
        # Campos importantes para mostrar
        display_fields = [
            'nombre_completo', 'telefono_completo', 'direccion',
            'municipio', 'estado', 'ocupacion', 'edad', 'sexo'
        ]
        
        formatted = {
            'score': score,
            'fuente': source.get('fuente', 'Desconocida'),
            'data': {}
        }
        
        # Agregar campos disponibles
        for field in display_fields:
            if field in source and source[field]:
                formatted['data'][field] = source[field]
        
        # Agregar otros campos disponibles
        for key, value in source.items():
            if key not in display_fields and not key.startswith('fecha_') and value:
                formatted['data'][key] = value
        
        return formatted
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del índice"""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            count = self.es.count(index=self.index_name)
            
            return {
                "total_documents": count['count'],
                "index_size": stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                "status": "active"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def delete_index(self):
        """Elimina el índice completamente"""
        try:
            self.es.indices.delete(index=self.index_name)
            print(f"✅ Índice '{self.index_name}' eliminado")
        except NotFoundError:
            print(f"⚠️ Índice '{self.index_name}' no existe")
        except Exception as e:
            print(f"❌ Error eliminando índice: {e}")

# Función de utilidad para testing
def test_elasticsearch_engine():
    """Función de prueba"""
    try:
        engine = ElasticsearchEngine()
        
        # Obtener estadísticas
        stats = engine.get_stats()
        print(f"📊 Estadísticas: {stats}")
        
        # Prueba de búsqueda
        test_queries = [
            "Juan Pérez",
            "555123",
            "Guadalajara",
            "ingeniero"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Probando: {query}")
            results = engine.search(query, max_results=3)
            print(f"   Encontrados: {results['total']} resultados")
            
    except Exception as e:
        print(f"❌ Error en prueba: {e}")

if __name__ == "__main__":
    test_elasticsearch_engine()