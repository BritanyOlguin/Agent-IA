"""
Motor de búsqueda usando Elasticsearch para consultas inteligentes y tolerantes a errores.
Reemplaza las búsquedas tradicionales con capacidades avanzadas.
"""

import os
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime
from unidecode import unidecode
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

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

        # Traduccion de códigos INEGI a nombres de estados
        self.inegi_map = {
            '1': 'Aguascalientes', '2': 'Baja California', '3': 'Baja California Sur',
            '4': 'Campeche', '5': 'Coahuila', '6': 'Colima', '7': 'Chiapas',
            '8': 'Chihuahua', '9': 'Ciudad de México', '10': 'Durango',
            '11': 'Guanajuato', '12': 'Guerrero', '13': 'Hidalgo', '14': 'Jalisco',
            '15': 'México', '16': 'Michoacán', '17': 'Morelos', '18': 'Nayarit',
            '19': 'Nuevo León', '20': 'Oaxaca', '21': 'Puebla', '22': 'Querétaro',
            '23': 'Quintana Roo', '24': 'San Luis Potosí', '25': 'Sinaloa',
            '26': 'Sonora', '27': 'Tabasco', '28': 'Tamaulipas', '29': 'Tlaxcala',
            '30': 'Veracruz', '31': 'Yucatán', '32': 'Zacatecas'
        }
        
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

                # Inicializar modelos de NLP
        print("🧠 Cargando modelos de inteligencia artificial...")
        try:
            self.nlp = spacy.load("es_core_news_sm")
            print("✅ Modelo de español cargado")
        except OSError:
            print("⚠️ Modelo de español no encontrado, usando análisis básico")
            self.nlp = None
        
        # Inicializar modelo de embeddings semánticos
        try:
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ Modelo semántico cargado")
        except:
            print("⚠️ Modelo semántico no disponible")
            self.semantic_model = None
        
        # Diccionario de intenciones semánticas
        self.intention_patterns = {
            'buscar_propietario': [
                'de quien es', 'a quien pertenece', 'quien tiene', 'propietario de',
                'dueño de', 'titular de', 'quien es el dueño', 'a nombre de quien está',
                'quién posee', 'cuál es el poseedor de', 'saber el propietario',
                'identificar al dueño', 'quién lo registra', 'a quién le pertenece'
            ],
            'buscar_informacion': [
                'que datos', 'informacion de', 'dime sobre', 'busca a',
                'encuentra a', 'datos de', 'info de', 'dame los detalles de',
                'necesito saber de', 'obtén datos de', 'averigua sobre',
                'información acerca de', 'muéstrame de', 'qué sabes de',
                'reporte de', 'detalles de', 'quiero conocer de', 'qué información hay sobre'
            ],
            'verificar_existencia': [
                'existe', 'esta registrado', 'hay alguien', 'se encuentra',
                'está activo', 'es válido', 'está vigente', 'es real',
                'se puede encontrar', 'lo tienes en tu base', 'está en la base de datos',
                'ha sido dado de alta', 'existe registro de', 'confirmar si existe'
            ],
            'obtener_contacto': [
                'contacto de', 'cómo contacto a', 'número para', 'dirección de',
                'teléfono de', 'email de', 'comunicarme con', 'donde lo localizo'
            ],
            'solicitar_ubicacion': [
                'donde esta', 'ubicación de', 'dirección de', 'vive en', 'localidad de',
                'domicilio de', 'sitio de', 'en qué calle', 'en qué ciudad', 'en qué estado'
            ],
            'realizar_consulta_general': [
                'sobre', 'acerca de', 'qué hay de', 'información general de',
                'todo sobre', 'preguntas sobre'
            ]
        }
        
        # Diccionario de tipos de datos semánticos
        self.data_type_patterns = {
            'tarjeta': [
                'tarjeta', 'plastico', 'credito', 'debito', 'card', 'visa',
                'mastercard', 'bancaria', 'cuenta', 'saldo', 'financiera',
                'amex', 'diners', 'paypal', 'nip', 'cvv', 'expiracion',
                'vencimiento', 'numero de tarjeta', 'cuenta bancaria', 'caja de ahorro',
                'cheques', 'revolvente', 'departamental', 'puntos'
            ],
            'telefono': [
                'telefono', 'celular', 'movil', 'numero', 'tel', 'cell',
                'whatsapp', 'contacto', 'llamada', 'fijo', 'lada', 'linea',
                'telefonico', 'movistar', 'telcel', 'att', 'unefon', 'comunicacion',
                'telf', 'extension', 'ph', 'phone', 'telefónico'
            ],
            'documento': [
                'ife', 'credencial', 'electoral', 'identificacion', 'cedula',
                'documento', 'id', 'clave', 'pasaporte', 'licencia', 'curp',
                'rfc', 'nss', 'folio', 'identidad', 'oficial', 'dni', 'cédula de identidad',
                'licencia de conducir', 'identificación oficial', 'acta de nacimiento',
                'comprobante de domicilio', 'firma electrónica'
            ],
            'nombre_persona': [
                'nombre', 'persona', 'individuo', 'sujeto', 'ciudadano', 'hombre', 'mujer',
                'nombre completo', 'nombre y apellido', 'nombres', 'apellidos'
            ],
            'direccion_fisica': [
                'direccion', 'domicilio', 'calle', 'avenida', 'boulevard', 'colonia',
                'barrio', 'vecindario', 'codigo postal', 'cp', 'ciudad', 'municipio',
                'estado', 'pais', 'residencia', 'ubicacion', 'localidad', 'postal',
                'apartamento', 'num exterior', 'num interior', 'código postal', 'c.p.'
            ],
            'ocupacion_profesion': [
                'ocupacion', 'profesion', 'trabajo', 'empleo', 'profesional', 'puesto',
                'a que se dedica', 'oficio'
            ],
            'edad_genero': [
                'edad', 'años', 'genero', 'sexo', 'años de edad', 'fecha de nacimiento',
                'hombre', 'mujer', 'masculino', 'femenino'
            ],
            'producto_servicio': [
                'producto', 'servicio', 'articulo', 'bien', 'mercancia', 'suscripcion',
                'plan', 'paquete', 'modelo', 'tipo de producto', 'qué producto', 'qué servicio'
            ]
        }
    
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
                                "direcciones_synonym_filter",
                                "lugares_synonym_filter"
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
                        },
                        "lugares_synonym_filter": {
                            "type": "synonym",
                            "synonyms": [
                                "aguascalientes,ags",
                                "baja california,bc",
                                "baja california sur,bcs",
                                "campeche,camp",
                                "chiapas,chis",
                                "chihuahua,chih",
                                "ciudad de mexico,cdmx,df,distrito federal",
                                "coahuila,coah",
                                "colima,col",
                                "durango,dgo",
                                "estado de mexico,edomex,mex",
                                "guanajuato,gto",
                                "guerrero,gro",
                                "hidalgo,hgo",
                                "jalisco,jal,gdl",
                                "michoacan,mich",
                                "morelos,mor",
                                "nayarit,nay",
                                "nuevo leon,nl",
                                "oaxaca,oax",
                                "puebla,pue",
                                "queretaro,qro",
                                "quintana roo,q roo",
                                "san luis potosi,slp",
                                "sinaloa,sin",
                                "sonora,son",
                                "tabasco,tab",
                                "tamaulipas,tamps",
                                "tlaxcala,tlax",
                                "veracruz,ver",
                                "yucatan,yuc",
                                "zacatecas,zac"
                            ]
                        },
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

    def _analyze_query_semantically(self, query: str) -> Dict[str, Any]:
        """
        Analiza consultas complejas, extrayendo múltiples entidades y exclusiones.
        """
        query_lower = query.lower().strip()
        
        # El "expediente" del detective. Ahora guardará múltiples hallazgos.
        analysis = {
            'nombres': [],
            'ubicaciones': [],
            'profesiones': [],
            'palabras_clave': [],
            'exclusiones': {}  # <-- Aquí guardaremos lo que se debe excluir
        }
        
        # --- Lógica para detectar y extraer exclusiones ---
        exclusion_patterns = [
            r'menos ([\w\s]+)', r'excepto ([\w\s]+)',
            r'sin ([\w\s]+)', r'pero no ([\w\s]+)'
        ]
        
        query_para_analisis = query_lower

        for pattern in exclusion_patterns:
            match = re.search(pattern, query_para_analisis)
            if match:
                termino_excluido = match.group(1).strip()
                
                if 'general' not in analysis['exclusiones']:
                    analysis['exclusiones']['general'] = []
                analysis['exclusiones']['general'].append(termino_excluido)
                print(f"🕵️‍♂️ Detective encontró una exclusión: '{termino_excluido}'")
                
                # Limpiamos la consulta para no confundir al resto del análisis
                query_para_analisis = query_para_analisis.replace(match.group(0), '').strip()
        
        # --- Análisis con spaCy para extraer TODAS las entidades ---
        if self.nlp:
            doc = self.nlp(query_para_analisis)
            
            # Extraer todas las personas, ubicaciones, etc.
            for ent in doc.ents:
                if ent.label_ in ['PER', 'PERSON']:
                    analysis['nombres'].append(ent.text)
                elif ent.label_ in ['LOC', 'GPE']:
                    analysis['ubicaciones'].append(ent.text)
            
            # Extraer profesiones de la lista de patrones
            lista_profesiones = ['ingeniero', 'doctor', 'abogado', 'carpintero', 'medico', 'licenciado']
            for token in doc:
                if token.lemma_.lower() in lista_profesiones:
                    analysis['profesiones'].append(token.lemma_.lower())

        # Extraer palabras clave que no son entidades ya reconocidas
        entidades_encontradas = set(analysis['nombres'] + analysis['ubicaciones'] + analysis['profesiones'])
        analysis['palabras_clave'] = [
            word for word in query_para_analisis.split() 
            if len(word) > 2 and word not in entidades_encontradas
        ]
        
        return analysis
    
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
                'nombre_completo', 'nombre completo', 'nombre y apellidos', 'nombre_y_apellidos', 'NOMBRE_COMPLETO', 'NOMBRE COMPLETO', 'Nombre Completo', 'surname', 'full name', 'razon social', 'razón social', 'company name', 'business name', 'titular', 'beneficiario'
            ],
            'nombre': [
                'nombre', 'nombres', 'primer nombre', 'given name', 'first name', 'NOMBRE', 'Name', 'NAME'
            ],
            'paterno': [
                'paterno', 'apellido paterno', 'last name', 'surname', 'PATERNO'
            ],
            'materno': [
                'materno', 'apellido materno', 'MATERNO'
            ],
            'telefono_completo': [
                'telefono_completo', 'teléfono completo', 'celular', 'phone', 'móvil'
            ],
            'lada': [
                'lada', 'Lada', 'area code', 'código de área'
            ],
            'telefono': [
                'telefono', 'teléfono', 'tel', 'TELEFONO', 'Telefono'
            ],
            'direccion': [
                'direccion', 'dirección', 'domicilio', 'calle', 'DIRECCION', 'DIRECCIÓN', 'DOMICILIO',
                'CALLE', 'address', 'ADDRESS', 'domicilio_completo', 'DOMICILIO', 'COLONIA', 'SECTOR',
                'DOMICILIO: CALLE', 'NÃšMERO', 'COLONIA', 'Calle', 'Numero', 'Colonia',
                'avenida', 'av', 'ave', 'boulevard', 'blvd', 'código postal', 'cp', 'c.p.',
                'fraccionamiento', 'fracc', 'num_exterior', 'num_interior', 'localidad', 'barrio',
                'zip code', 'postal code', 'street', 'road', 'avenue', 'boulevard', 'suburb',
                'floor', 'apartment', 'suite', 'número'
            ],
            'municipio': [
                'municipio', 'ciudad', 'MUNICIPIO', 'CIUDAD', 'city', 'CITY', 'localidad', 'MUNICIPIO', 'Ciudad',
                'alcaldia', 'delegacion', 'demarcacion', 'urbanizacion', 'town', 'county'
            ],
            'estado': [
                'estado', 'entidad', 'ESTADO', 'ENTIDAD', 'state', 'STATE', 'provincia',
                'ESTADO DE ORIGEN', 'EDO REGISTRO', 'Edo Registro', 'Estado', 'ENTIDAD',
                'state name', 'region', 'provincie',
            ],
            'ocupacion': [
                'ocupacion', 'ocupación', 'profesion', 'profesión', 'trabajo', 'empleo',
                'OCUPACION', 'OCUPACIÓN', 'PROFESION', 'PROFESIÓN', 'TRABAJO', 'job', 'OCUPACION',
                'puesto', 'rol', 'cargo', 'actividad economica', 'economic activity',
                'profesion_oficio', 'giro', 'tipo_empleo'
            ],
            'edad': [
                'edad', 'años', 'EDAD', 'AÑOS', 'age', 'AGE',
                'fecha_nacimiento', 'fecha nacimiento', 'birthdate', 'dob', 'edad_años',
                'años cumplidos', 'antiguedad'
            ],
            'sexo': [
                'sexo', 'genero', 'género', 'SEXO', 'GENERO', 'GÉNERO', 'gender', 'sex', 'SEXO', 'Sexo',
                'masculino', 'femenino', 'male', 'female', 'genero_id', 'genero_persona'
            ],
            'cp': [
                'cp', 'codigo postal', 'código postal', 'CP', 'CODIGO POSTAL', 'zip', 'CODIGO POSTAL', 'CP',
                'zip code', 'postal code', 'c_p'
            ],
            'clave_ife': [
                'CLAVE IFE', 'ife', 'credencial', 'electoral', 'id electoral', 'folio_ife', 'folio ife',
                'identificacion', 'identificación', 'documento_identidad', 'clave_elector', 'no_credencial'
            ],
            'campo_14': [
                'Campo14', 'campo 14', 'field14', 'valor_extra'
            ],
            'info_producto': [
                'Producto', 'Tarjeta', 'tarjeta', 'producto', 'numero_tarjeta', 'no_tarjeta',
                'tipo tarjeta', 'tipo producto', 'servicio', 'service', 'product', 'item',
                'plan', 'modelo', 'num_contrato', 'contrato', 'id_producto', 'id_servicio',
                'cuenta_bancaria', 'cuenta', 'bank_account', 'numero_cuenta', 'num_cuenta',
                'plástico', 'credito', 'débito', 'credit card', 'debit card'
            ],
            'fecha_registro': [
                'fecha registro', 'fecha_registro', 'fecha de registro', 'fecha_alta', 'fecha alta',
                'date registered', 'registration date', 'signup date', 'fecha_ingreso', 'fecha_creacion'
            ],
            'email': [
                'email', 'correo', 'e-mail', 'mail', 'correo_electronico', 'email_address'
            ],
            'nacionalidad': [
                'nacionalidad', 'pais', 'país', 'country', 'nacionality', 'pais_nacimiento', 'lugar_nacimiento'
            ],
            'estado_civil': [
                'estado civil', 'estado_civil', 'marital status', 'civil_status',
                'soltero', 'casado', 'divorciado', 'viudo', 'union libre'
            ]
        }
        
        # Detectar automáticamente las columnas que coinciden
        campo_detectado = {}
        columnas_usadas = set()
        
        for target_field, possible_columns in column_mapping.items():
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [p.lower().strip() for p in possible_columns] and col not in columnas_usadas:
                    campo_detectado[target_field] = col
                    columnas_usadas.add(col)
                    print(f"   🎯 Detectado: {target_field} -> {col}")
                    break

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

            for col in df.columns:
                if col not in columnas_usadas:
                    valor = row[col]
                    if pd.notna(valor) and str(valor).strip() and str(valor).strip().lower() != 'nan':
                        field_name = col.lower().replace(' ', '_').replace('-', '_')
                        doc[field_name] = str(valor).strip()
                        tiene_datos = True

            # Traducción de estados
            nombres_posibles_estado = column_mapping.get('estado', [])
            campos_a_traducir = set(['estado'])
            for nombre in nombres_posibles_estado:
                campos_a_traducir.add(nombre.lower().replace(' ', '_').replace('-', '_'))
            
            for campo in list(campos_a_traducir):
                if campo in doc:
                    valor = doc[campo]
                    if str(valor).isdigit() and str(valor) in self.inegi_map:
                        doc[campo] = self.inegi_map[str(valor)]

            # Ensamblaje de teléfono
            if 'lada' in doc and 'telefono' in doc:
                doc['telefono_completo'] = str(doc['lada']) + str(doc['telefono'])
                
                doc.pop('lada', None)
                doc.pop('telefono', None)

            # Ensamblaje de nombres
            partes_nombre = []
            nombre_val = doc.get('nombre')
            paterno_val = doc.get('paterno')
            materno_val = doc.get('materno')

            if nombre_val: partes_nombre.append(str(nombre_val))
            if paterno_val: partes_nombre.append(str(paterno_val))
            if materno_val: partes_nombre.append(str(materno_val))

            if len(partes_nombre) >= 2:
                doc['nombre_completo'] = ' '.join(partes_nombre)
                doc.pop('nombre', None)
                doc.pop('paterno', None)
                doc.pop('materno', None)

            # Logica de normalización
            contenido_completo = [str(v) for k, v in doc.items() if k not in ['fuente', 'fecha_indexado', 'id_original'] and v]
            doc['contenido_completo'] = ' '.join(contenido_completo)

            if tiene_datos and len(doc) > 4:
                docs.append(doc)
        
        print(f"   ✅ Preparados {len(docs)} documentos válidos de {len(df)} filas")
        
        # Mostrar ejemplo de documento
        if docs:
            print(f"   📋 Ejemplo de documento procesado:")
            ejemplo_procesado = {k: v for k, v in docs[0].items() if k != 'contenido_completo'}
            for key, value in list(ejemplo_procesado.items())[:6]:
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
        Búsqueda principal que maneja todo tipo de consultas con IA
        """
        print(f"🔍 Buscando: '{query}'")
        
        analysis = self._analyze_query_semantically(query)
        
        print(f"🧠 Análisis IA: {analysis}")
        
        es_query = self._build_intelligent_query(analysis)
        print(f"✨ Usando búsqueda inteligente optimizada")
        
        try:
            response = self.es.search(
                index=self.index_name,
                body=es_query,
                size=max_results
            )
            
            results = self._process_search_results(response, query)
            
            results['analysis'] = analysis
            
            return results
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return {"total": 0, "results": [], "error": str(e)}
    
    def _build_elasticsearch_query(self, query: str) -> Dict:
        """Construye una consulta compleja de Elasticsearch (método tradicional mejorado)"""
        
        # Limpiar y analizar la query
        clean_query = self._clean_query(query)
        
        # Detectar tipo de búsqueda con patrones mejorados
        search_type = self._detect_search_type_improved(clean_query)
        
        # Construir query según el tipo detectado
        if search_type == "telefono":
            return self._build_phone_query_improved(clean_query)
        elif search_type == "tarjeta":
            return self._build_card_query_improved(clean_query)
        elif search_type == "direccion_fisica":
            return self._build_address_query(clean_query)
        elif search_type == "nombre_persona":
            return self._build_name_query(clean_query)
        else:
            return self._build_fallback_query(clean_query)
        
    def _build_intelligent_query(self, analysis: Dict) -> Dict:
        """
        Construye una consulta Elasticsearch basada en el análisis complejo,
        usando lógica booleana estricta (MUST, SHOULD, MUST_NOT).
        """
        must_clauses = []
        should_clauses = []
        must_not_clauses = []

        # --- Lógica para condiciones OBLIGATORIAS (MUST) ---
        # Si el detective encontró entidades, las hacemos obligatorias.
        if analysis.get('nombres'):
            for nombre in analysis['nombres']:
                must_clauses.append({"match": {"nombre_completo": {"query": nombre, "fuzziness": "AUTO"}}})
        
        if analysis.get('ubicaciones'):
            for ubicacion in analysis['ubicaciones']:
                must_clauses.append({"match": {"estado": {"query": ubicacion}}})
        
        if analysis.get('profesiones'):
            for profesion in analysis['profesiones']:
                must_clauses.append({"match": {"ocupacion": {"query": profesion}}})

        # --- Lógica para condiciones de EXCLUSIÓN (MUST_NOT) ---
        if analysis.get('exclusiones', {}).get('general'):
            for termino in analysis['exclusiones']['general']:
                must_not_clauses.append({"match": {"nombre_completo": termino}})
        
        # --- Lógica para condiciones DESEABLES (SHOULD) ---
        # Las palabras clave generales suman puntos, pero no son obligatorias.
        if analysis.get('palabras_clave'):
            for palabra in analysis['palabras_clave']:
                should_clauses.append({
                    "multi_match": {
                        "query": palabra,
                        "fields": ["contenido_completo"],
                        "fuzziness": "AUTO"
                    }
                })

        # Construir el cuerpo final de la consulta
        # Si no hay cláusulas MUST, al menos una SHOULD debe cumplirse.
        min_should_match = 1 if not must_clauses and should_clauses else 0
        
        query_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "must_not": must_not_clauses,
                    "should": should_clauses,
                    "minimum_should_match": min_should_match
                }
            }
        }
        print(f"📜 Estratega construyó un plan: {query_body}")
        return query_body
    
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
                {"wildcard": {"telefono_completo.keyword": f"*{number}"}}
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

    def _detect_search_type_improved(self, query: str) -> str:
        """Detección mejorada de tipos de búsqueda"""
        
        # Detectar números de tarjeta (15-16 dígitos)
        if re.search(r'\b\d{15,16}\b', query):
            return "card"
        
        # Detectar teléfonos (7-12 dígitos)
        if re.search(r'\b\d{7,12}\b', query):
            return "phone"
        
        # Detectar patrones de dirección
        address_patterns = [
            r'\b(calle|avenida|av|blvd|boulevard)\b',
            r'\b(colonia|col|sector|fracc)\b',
            r'\b(vive|direccion|domicilio)\b'
        ]
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in address_patterns):
            return "address"
        
        # Detectar patrones de nombre
        name_patterns = [
            r'\b(quien|nombre|persona|llama)\b',
            r'\b[A-ZÁÉÍÓÚ][a-záéíóú]+ [A-ZÁÉÍÓÚ][a-záéíóú]+\b'  # Nombres propios
        ]
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in name_patterns):
            return "name"
        
        return "general"

    def _build_phone_query_improved(self, query: str) -> Dict:
        """Consulta mejorada para teléfonos (SIN suffix que causaba error)"""
        
        numbers = re.findall(r'\d+', query)
        should_clauses = []
        
        for number in numbers:
            should_clauses.extend([
                {"match": {"telefono_completo": {"query": number, "fuzziness": "1"}}},
                {"wildcard": {"telefono_completo.keyword": f"*{number}*"}},
                {"prefix": {"telefono_completo.keyword": number}},
                # ELIMINADO: suffix (causaba el error)
                {"wildcard": {"telefono_completo.keyword": f"*{number}"}}  # Reemplaza suffix
            ])
        
        return {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
        }

    def _build_card_query_improved(self, query: str) -> Dict:
        """Consulta mejorada para tarjetas"""
        
        numbers = re.findall(r'\d+', query)
        should_clauses = []
        
        for number in numbers:
            should_clauses.extend([
                {"match": {"tarjeta": {"query": number, "fuzziness": "0"}}},
                {"wildcard": {"tarjeta.keyword": f"*{number}*"}},
                {"prefix": {"tarjeta.keyword": number}},
                {"term": {"tarjeta.keyword": number}}
            ])
        
        return {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
        }

    def _build_fallback_query(self, query: str) -> Dict:
        """Consulta de respaldo súper flexible"""
        
        return {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["*"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
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