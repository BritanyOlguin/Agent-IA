import os
import pandas as pd
import json
import random
from tqdm import tqdm
import sys
import re

# Añadir la ruta de src para importar normalizar_texto
sys.path.append(r"C:\Users\TEC-INT02\Documents\Agent-IA\src")
from normalizar_texto import normalizar_texto # Asumo que esta función es robusta

# Configuración de rutas
CARPETA_BD = r"C:\Users\TEC-INT02\Documents\Agent-IA\archivos"
CARPETA_SALIDA = r"C:\Users\TEC-INT02\Documents\Agent-IA\fine_tuning\datos"

# Formatos para las conversaciones
FORMATO_INSTRUCCION = """<instrucción>
Eres un asistente que consulta información personal en bases de datos. Responde la siguiente pregunta con precisión usando los datos disponibles.

{pregunta}
</instrucción>

<respuesta>
{respuesta}
</respuesta>"""

def cargar_dataframe(archivo):
    """Carga un dataframe desde diferentes formatos de archivo"""
    ext = os.path.splitext(archivo)[1].lower()
    print(f"📄 Cargando: {archivo}")
    conn = None # Inicializar conn para el bloque finally

    try:
        if ext in ['.xlsx', '.xls']:
            return pd.read_excel(archivo, dtype=str).fillna("")
        elif ext == '.csv':
            try:
                return pd.read_csv(archivo, dtype=str, encoding='utf-8').fillna("")
            except UnicodeDecodeError:
                return pd.read_csv(archivo, dtype=str, encoding='latin-1').fillna("")
        elif ext == '.mdb' or ext == '.accdb': # Añadido soporte para accdb
            try:
                import pyodbc
                # Asegurarse de que el driver correcto está especificado para mdb y accdb
                conn_str = f'DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={archivo};'
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                # Obtener solo tablas de usuario, no del sistema
                tables = [table_info.table_name for table_info in cursor.tables(tableType='TABLE') if not table_info.table_name.startswith('MSys')]
                if tables:
                    # Aquí podrías tener lógica para elegir una tabla o unirlas si hay varias.
                    # Por simplicidad, tomamos la primera.
                    print(f"    -> Tablas encontradas en '{os.path.basename(archivo)}': {tables}. Usando tabla: '{tables[0]}'")
                    df = pd.read_sql(f"SELECT * FROM [{tables[0]}]", conn)
                    return df.astype(str).fillna("")
                else:
                    print(f"⚠️ No se encontraron tablas de usuario en el archivo Access: {archivo}")
                    return None
            except ImportError:
                print(f"⚠️ La biblioteca 'pyodbc' no está instalada. No se puede leer {archivo}. Instala con: pip install pyodbc")
                return None
            except Exception as e:
                print(f"❌ Error al cargar {archivo} (Access): {e}")
                return None
            finally:
                if conn:
                    conn.close()
        elif ext == '.txt':
            try:
                # Mejorar la detección de separador para TXT o asumir uno común.
                # Si el formato es NOMBRE|FECHA_AFILIACION|ENTIDAD|, el separador es '|'.
                df = pd.read_csv(archivo, dtype=str, sep='|', header=0, encoding='utf-8').fillna("")
                # Podrías añadir un intento con 'latin-1' si falla utf-8
                return df
            except Exception as e:
                print(f"❌ Error al cargar {archivo} (TXT). Asegúrate del formato y separador. Error: {e}")
                return None
        else:
            print(f"⚠️ Formato no soportado: {ext}")
            return None
    except Exception as e:
        print(f"❌ Error general al cargar {archivo}: {e}")
        return None

# ---------------------------------------------------------------------------
# INICIO DE MODIFICACIÓN: ampliar mapeo_campos y lógica de detección
# ---------------------------------------------------------------------------
def detectar_y_normalizar_campos(df):
    """Detecta y normaliza campos comunes en diferentes formatos"""
    # Mapeo de nombres posibles para campos estándar. Sé lo más exhaustivo posible.
    mapeo_campos = {
        # Campos de Nombre
        'nombre_completo': ['nombre_completo', 'nombre completo', 'nombre y apellidos', 'nombrecompleto', 'nombre_y_apellido', 'full_name'],
        'nombre': ['nombre', 'nombres', 'name', 'first name', 'nombre de pila', 'nombre1', 'nomb'],
        'apellido_paterno': ['apellido_paterno', 'apellido_p', 'paterno', 'a_paterno', 'primer apellido', 'apellido1', 'ape_pat', 'apellidopaterno'],
        'apellido_materno': ['apellido_materno', 'apellido_m', 'materno', 'a_materno', 'segundo apellido', 'apellido2', 'ape_mat', 'apellidomaterno'],

        # Campos de Dirección
        'direccion_base': ['direccion', 'dirección', 'domicilio', 'domicilio particular', 'direccion completa', 'address'], # Campo que podría tener la dirección ya concatenada
        'calle': ['calle', 'street', 'nombre de la calle', 'nom_calle', 'calle_numero'], # A veces incluye número
        'numero_exterior': ['numero', 'num_ext', 'numero exterior', 'número exterior', 'no_exterior', 'numext', 'nroext', 'numero_calle'],
        'numero_interior': ['num_int', 'numero interior', 'número interior', 'no_interior', 'numint', 'nroint', 'interior', 'depto', 'departamento'],
        'colonia': ['colonia', 'col', 'barrio', 'fraccionamiento', 'fracc', 'poblacion', 'población'],
        'sector': ['sector', 'zona'],
        'municipio': ['municipio', 'ciudad', 'delegacion', 'delegación', 'alcaldia', 'alcaldía', 'city', 'localidad'],
        'estado': ['estado', 'entidad', 'provincia', 'state', 'entidad federativa', 'edo'],
        'cp': ['cp', 'c.p.', 'codigo_postal', 'código postal', 'zipcode', 'zip code', 'codigopostal', 'cod_postal'],
        'ciudad_especifica': ['ciudad', 'city'], # Para cuando 'ciudad' es un campo distinto a 'municipio' (ej. Ciudad de México vs. Alcaldía)

        # Campos de Contacto
        'telefono_directo': ['telefono', 'teléfono', 'tel', 'phone', 'numero de telefono', 'número de teléfono', 'fon', 'tel_numero'], # Teléfono sin lada
        'lada': ['lada', 'codigo de area', 'código de área', 'area_code', 'cod_area'],
        'telefono_completo': ['telefono_completo', 'tel_completo', 'telefono con lada', 'celular', 'movil', 'tel_celular', 'tel_movil'], # Podría ser el celular o un teléfono ya con lada

        # Campos de Identificación y Otros
        'tarjeta': ['tarjeta', 'numero de tarjeta', 'número de tarjeta', 'card number', 'nro_tarjeta', 'notarjeta'],
        'clave_ife': ['clave_ife', 'ife', 'clave de elector', 'credencial de elector', 'ine', 'clave_ine', 'id_elector'],
        'estado_de_origen': ['estado_de_origen', 'estado de origen', 'lugar de nacimiento estado', 'entidad de nacimiento', 'edo_origen', 'ent_origen'],
        'ocupacion': ['ocupacion', 'ocupación', 'profesion', 'profesión', 'puesto', 'job', 'actividad economica', 'actividad_economica'],
        'campo_14': ['campo_14', 'campo14', 'columna14', 'dato14', 'custom_field_14'], # Ejemplo de campo genérico
        # 'codigo_postal': ['codigo postal', 'código postal', 'cp', 'zipcode'], # Ya cubierto por 'cp'
        'estado_de_registro': ['estado_de_registro', 'estado de registro', 'entidad de registro', 'edo_registro', 'ent_registro'],
        'fecha_de_afiliacion': ['fecha_de_afiliacion', 'fecha de afiliacion', 'fecha_afiliacion', 'fec_afil', 'fecha alta', 'fec_alta', 'afiliacion_fecha'],
        'entidad_especifica': ['entidad', 'nombre de la entidad', 'nom_entidad'], # Podría ser una empresa, organización, etc. No confundir con 'estado'.
        'sexo': ['sexo', 'genero', 'género'],
        'curp': ['curp', 'clave unica de registro de poblacion'],
        'rfc': ['rfc', 'registro federal de contribuyentes'],
        'email': ['email', 'correo electronico', 'correo_electronico', 'e-mail'],
        'fecha_nacimiento': ['fecha_nacimiento', 'fecha de nacimiento', 'fec_nac', 'dob', 'birthdate'],
    }

    campos = {}
    # Normalizar los nombres de las columnas del DataFrame para una comparación más robusta
    df_columnas_normalizadas = {normalizar_texto(col).lower().replace(" ", "_"): col for col in df.columns}
    # print(f"    Columnas normalizadas del DF: {list(df_columnas_normalizadas.keys())}") # Para depuración

    for campo_std, posibles_nombres_sucios in mapeo_campos.items():
        for posible_nombre_sucio in posibles_nombres_sucios:
            # Normalizar también cada posible nombre del mapeo
            posible_nombre_limpio = normalizar_texto(posible_nombre_sucio).lower().replace(" ", "_")
            if posible_nombre_limpio in df_columnas_normalizadas:
                campos[campo_std] = df_columnas_normalizadas[posible_nombre_limpio]
                # print(f"    ✔️ Campo detectado: '{campo_std}' -> Columna original: '{campos[campo_std]}' (matcheado con '{posible_nombre_limpio}')") # Para depuración
                break # Tomar la primera coincidencia para este campo estándar
            # else: # Para depuración intensiva
                # print(f"    ❌ No match para '{posible_nombre_limpio}' en mapeo_campos['{campo_std}']")


    # --- CONSTRUCCIÓN DE CAMPOS COMPUESTOS ---

    # 1. Nombre Completo
    if 'nombre_completo' not in campos and any(c in campos for c in ['nombre', 'apellido_paterno', 'apellido_materno']):
        print("    ℹ️ Intentando construir 'nombre_completo' a partir de sus componentes.")
        nombres_completos_construidos = []
        for _, row in df.iterrows():
            parts = []
            if campos.get('nombre') and pd.notna(row[campos['nombre']]):
                parts.append(str(row[campos['nombre']]).strip())
            if campos.get('apellido_paterno') and pd.notna(row[campos['apellido_paterno']]):
                parts.append(str(row[campos['apellido_paterno']]).strip())
            if campos.get('apellido_materno') and pd.notna(row[campos['apellido_materno']]):
                parts.append(str(row[campos['apellido_materno']]).strip())
            
            nombre_construido = " ".join(filter(None, parts)) # filter(None, ...) elimina cadenas vacías
            nombres_completos_construidos.append(nombre_construido if nombre_construido else "")
        
        df['nombre_completo_construido'] = nombres_completos_construidos
        campos['nombre_completo'] = 'nombre_completo_construido'
        print(f"    ✔️ Campo 'nombre_completo' construido como 'nombre_completo_construido'.")

    # 2. Dirección Completa
    # Prioridad: 1. direccion_base, 2. Componentes detallados
    if 'direccion_completa' not in campos: # Si no hay un campo mapeado directamente a 'direccion_completa'
        print("    ℹ️ Intentando construir 'direccion_completa'.")
        direcciones_completas_construidas = []
        # Orden preferido de componentes para la dirección
        componentes_direccion_ordenados = [
            'calle', 'numero_exterior', 'numero_interior', 'colonia', 
            'sector', 'cp', 'municipio', 'ciudad_especifica', 'estado'
        ]
        for _, row in df.iterrows():
            dir_parts = []
            # Usar 'direccion_base' si existe y tiene contenido
            if campos.get('direccion_base') and pd.notna(row[campos['direccion_base']]) and str(row[campos['direccion_base']]).strip():
                dir_parts.append(str(row[campos['direccion_base']]).strip())
            else: # Construir desde componentes
                for comp_std in componentes_direccion_ordenados:
                    if campos.get(comp_std) and pd.notna(row[campos[comp_std]]):
                        valor_comp = str(row[campos[comp_std]]).strip()
                        if valor_comp: # Solo añadir si no está vacío
                            # Añadir prefijos comunes si no parecen estar ya (heurística simple)
                            prefijo = ""
                            if comp_std == 'colonia' and not any(p.lower().startswith(("col ", "col.", "fracc")) for p in dir_parts):
                                prefijo = "Col. "
                            elif comp_std == 'cp' and not any(p.lower().startswith(("cp ", "c.p")) for p in dir_parts):
                                prefijo = "C.P. "
                            elif comp_std == 'sector' and not any(p.lower().startswith("sector") for p in dir_parts):
                                prefijo = "Sector "
                            dir_parts.append(f"{prefijo}{valor_comp}")
            
            # Unir las partes, eliminando duplicados (si 'direccion_base' ya contenía algunos componentes)
            # y partes vacías.
            partes_unicas = []
            vistos = set()
            for parte in filter(None, dir_parts):
                # Normalizar para la comparación de duplicados, pero mantener el original
                parte_normalizada_simple = parte.lower().replace(",", "").strip()
                if parte_normalizada_simple not in vistos:
                    partes_unicas.append(parte)
                    vistos.add(parte_normalizada_simple)

            direccion_construida = ", ".join(partes_unicas)
            direcciones_completas_construidas.append(direccion_construida if direccion_construida else "")

        df['direccion_completa_construida'] = direcciones_completas_construidas
        campos['direccion_completa'] = 'direccion_completa_construida'
        print(f"    ✔️ Campo 'direccion_completa' construido como 'direccion_completa_construida'.")

    # 3. Teléfono Completo (LADA + Teléfono)
    # Prioridad: 1. telefono_completo (mapeado directamente), 2. lada + telefono_directo
    if 'telefono_completo' not in campos:
        if campos.get('lada') and campos.get('telefono_directo'):
            print("    ℹ️ Intentando construir 'telefono_completo' a partir de LADA y teléfono directo.")
            telefonos_completos_construidos = []
            for _, row in df.iterrows():
                lada_val = str(row[campos['lada']]).strip() if pd.notna(row[campos['lada']]) else ""
                tel_dir_val = str(row[campos['telefono_directo']]).strip() if pd.notna(row[campos['telefono_directo']]) else ""
                
                # Limpiar LADA y Teléfono de caracteres no numéricos (opcional, depende del formato esperado)
                lada_val_limpio = re.sub(r'\D', '', lada_val)
                tel_dir_val_limpio = re.sub(r'\D', '', tel_dir_val)

                if lada_val_limpio and tel_dir_val_limpio:
                    # Formato común: (LADA) TELEFONO o LADA-TELEFONO
                    # Aquí se opta por una simple concatenación, ajustar si se requiere formato específico.
                    telefonos_completos_construidos.append(f"{lada_val_limpio}{tel_dir_val_limpio}")
                elif tel_dir_val_limpio: # Si solo hay teléfono directo y no LADA
                    telefonos_completos_construidos.append(tel_dir_val_limpio)
                else:
                    telefonos_completos_construidos.append("")
            df['telefono_completo_construido'] = telefonos_completos_construidos
            campos['telefono_completo'] = 'telefono_completo_construido'
            print(f"    ✔️ Campo 'telefono_completo' construido como 'telefono_completo_construido'.")
        elif campos.get('telefono_directo'): # Si solo hay telefono_directo y no un 'telefono_completo' mapeado explícitamente
            print(f"    ℹ️ Usando '{campos.get('telefono_directo')}' como 'telefono_completo' (no se encontró LADA o campo 'telefono_completo' explícito).")
            campos['telefono_completo'] = campos.get('telefono_directo') # Usar el campo de teléfono directo como completo
    
    # Normalización de los *valores* en las columnas detectadas (USAR CON PRECAUCIÓN)
    # Esto se hace DESPUÉS de la detección y construcción de campos.
    # Es importante que `normalizar_texto` no elimine información vital (ej. números en CP o teléfono).
    # Considera aplicarlo solo a campos que son claramente texto libre.
    # for campo_std, col_original_df in campos.items():
    #     # Ejemplo: normalizar solo campos que se espera sean nombres o direcciones textuales
    #     if campo_std in ['nombre', 'apellido_paterno', 'apellido_materno', 'calle', 'colonia', 'ocupacion']:
    #         print(f"    -> Normalizando valores en columna '{col_original_df}' (mapeada a '{campo_std}')")
    #         try:
    #             # Asegurarse de que la columna existe antes de aplicar
    #             if col_original_df in df.columns:
    #                 df[col_original_df] = df[col_original_df].astype(str).apply(normalizar_texto)
    #             else:
    #                 print(f"    ⚠️ Advertencia: La columna '{col_original_df}' (para '{campo_std}') no se encontró en el DataFrame durante la normalización de valores.")
    #         except Exception as e:
    #             print(f"    ⚠️ Error al normalizar valores de la columna {col_original_df}: {e}")
    
    print(f"    Campos finales detectados y mapeados: {list(campos.keys())}")
    return df, campos
# ---------------------------------------------------------------------------
# FIN DE MODIFICACIÓN: ampliar mapeo_campos y lógica de detección
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# INICIO DE MODIFICACIÓN: ampliar plantillas en generar_ejemplos_qa
# ---------------------------------------------------------------------------
def generar_ejemplos_qa(df, campos, nombre_archivo, num_ejemplos=100):
    """
    Genera ejemplos de preguntas y respuestas basados en los datos y campos detectados.
    Incluye plantillas para una amplia variedad de consultas por atributos diferentes.
    """
    
    ejemplos = []
    
    # Se necesita al menos un 'nombre_completo' para la mayoría de las preguntas centradas en una persona.
    # Y para preguntas de "quién", se necesita algún otro identificador.
    if 'nombre_completo' not in campos:
        print(f"⚠️ No se detectó 'nombre_completo' en {nombre_archivo}. No se pueden generar ejemplos de QA centrados en personas. Saltando...")
        return []

    plantillas = []

    # --- PLANTILLAS DE INFORMACIÓN GENERAL POR NOMBRE ---
    plantillas.extend([
        {
            'tipo': 'info_general_por_nombre',
            'pregunta': '¿Quién es {nombre_completo}?',
            'respuesta': 'Información de {nombre_completo}:\n{datos_completos}',
            'requeridos': ['nombre_completo']
        },
        {
            'tipo': 'info_general_por_nombre',
            'pregunta': 'Dame todos los datos que tengas de {nombre_completo}.',
            'respuesta': 'Claro, aquí están los datos de {nombre_completo}:\n{datos_completos}',
            'requeridos': ['nombre_completo']
        },
        {
            'tipo': 'info_general_por_nombre',
            'pregunta': 'Busca información sobre {nombre_completo}',
            'respuesta': 'He encontrado esta información sobre {nombre_completo}:\n{datos_completos}',
            'requeridos': ['nombre_completo']
        },
        {
            'tipo': 'info_general_por_nombre',
            'pregunta': 'Información completa de {nombre_completo}',
            'respuesta': 'Datos completos de {nombre_completo}:\n{datos_completos}',
            'requeridos': ['nombre_completo']
        },
    ])

    # --- PLANTILLAS PARA ATRIBUTOS ESPECÍFICOS DE UNA PERSONA ---
    # (campo_std, frase_pregunta, frase_respuesta_atributo)
    atributos_persona = [
        # DIRECCIÓN Y UBICACIÓN
        ('direccion_completa', 'la dirección completa de {nombre_completo}', 'La dirección de {nombre_completo} es: {direccion_completa}.'),
        ('direccion_completa', 'el domicilio de {nombre_completo}', 'El domicilio de {nombre_completo} es: {direccion_completa}.'),
        ('direccion_completa', 'dónde vive {nombre_completo}', '{nombre_completo} vive en: {direccion_completa}.'),
        ('direccion_completa', 'la ubicación de {nombre_completo}', '{nombre_completo} se encuentra ubicado en: {direccion_completa}.'),
        
        ('domicilio', 'el domicilio registrado de {nombre_completo}', 'El domicilio registrado de {nombre_completo} es: {domicilio}.'),
        ('domicilio', 'dónde se encuentra {nombre_completo}', '{nombre_completo} se encuentra en: {domicilio}.'),
        
        ('calle', 'la calle donde vive {nombre_completo}', 'La calle donde vive {nombre_completo} es: {calle}.'),
        ('calle', 'en qué calle está {nombre_completo}', '{nombre_completo} está en la calle: {calle}.'),
        
        ('numero_exterior', 'el número exterior del domicilio de {nombre_completo}', 'El número exterior del domicilio de {nombre_completo} es: {numero_exterior}.'),
        ('numero_exterior', 'el número de casa de {nombre_completo}', 'El número de casa de {nombre_completo} es: {numero_exterior}.'),
        ('numero_exterior', 'en qué número vive {nombre_completo}', '{nombre_completo} vive en el número: {numero_exterior}.'),
        
        ('colonia', 'la colonia donde reside {nombre_completo}', '{nombre_completo} reside en la colonia: {colonia}.'),
        ('colonia', 'en qué colonia vive {nombre_completo}', '{nombre_completo} vive en la colonia: {colonia}.'),
        ('colonia', 'la colonia de {nombre_completo}', 'La colonia de {nombre_completo} es: {colonia}.'),
        
        ('sector', 'el sector donde vive {nombre_completo}', '{nombre_completo} vive en el sector: {sector}.'),
        ('sector', 'en qué sector está {nombre_completo}', '{nombre_completo} está en el sector: {sector}.'),
        
        ('municipio', 'el municipio o alcaldía de {nombre_completo}', 'El municipio/alcaldía de {nombre_completo} es: {municipio}.'),
        ('municipio', 'en qué municipio vive {nombre_completo}', '{nombre_completo} vive en el municipio: {municipio}.'),
        ('municipio', 'a qué municipio pertenece {nombre_completo}', '{nombre_completo} pertenece al municipio: {municipio}.'),
        
        ('ciudad_especifica', 'la ciudad de {nombre_completo}', 'La ciudad de {nombre_completo} es: {ciudad_especifica}.'),
        ('ciudad_especifica', 'en qué ciudad vive {nombre_completo}', '{nombre_completo} vive en la ciudad: {ciudad_especifica}.'),
        
        ('estado', 'el estado de residencia de {nombre_completo}', '{nombre_completo} reside en el estado de: {estado}.'),
        ('estado', 'en qué estado vive {nombre_completo}', '{nombre_completo} vive en el estado: {estado}.'),
        ('estado', 'a qué estado pertenece {nombre_completo}', '{nombre_completo} pertenece al estado: {estado}.'),
        
        ('cp', 'el código postal de {nombre_completo}', 'El código postal de {nombre_completo} es: {cp}.'),
        ('cp', 'el CP de {nombre_completo}', 'El CP de {nombre_completo} es: {cp}.'),
        ('cp', 'cuál es el código postal donde vive {nombre_completo}', 'El código postal donde vive {nombre_completo} es: {cp}.'),
        
        # CONTACTO
        ('telefono_completo', 'el número de teléfono de {nombre_completo}', 'El teléfono de {nombre_completo} es: {telefono_completo}.'),
        ('telefono_completo', 'el teléfono de contacto de {nombre_completo}', 'El teléfono de contacto de {nombre_completo} es: {telefono_completo}.'),
        ('telefono_completo', 'a qué número puedo llamar a {nombre_completo}', 'Puedes llamar a {nombre_completo} al número: {telefono_completo}.'),
        ('telefono_completo', 'cuál es el celular de {nombre_completo}', 'El celular de {nombre_completo} es: {telefono_completo}.'),
        
        ('lada', 'la lada de {nombre_completo}', 'La lada de {nombre_completo} es: {lada}.'),
        ('lada', 'el código de área de {nombre_completo}', 'El código de área de {nombre_completo} es: {lada}.'),
        
        # IDENTIFICACIÓN Y DATOS PERSONALES
        ('tarjeta', 'el número de tarjeta asociado a {nombre_completo}', 'El número de tarjeta de {nombre_completo} es: {tarjeta}.'),
        ('tarjeta', 'la tarjeta de {nombre_completo}', 'La tarjeta de {nombre_completo} es: {tarjeta}.'),
        ('tarjeta', 'qué tarjeta tiene {nombre_completo}', '{nombre_completo} tiene la tarjeta: {tarjeta}.'),
        
        ('clave_ife', 'la clave IFE o INE de {nombre_completo}', 'La clave IFE/INE de {nombre_completo} es: {clave_ife}.'),
        ('clave_ife', 'el número de credencial electoral de {nombre_completo}', 'El número de credencial electoral de {nombre_completo} es: {clave_ife}.'),
        ('clave_ife', 'la credencial de elector de {nombre_completo}', 'La credencial de elector de {nombre_completo} tiene la clave: {clave_ife}.'),
        
        ('estado_de_origen', 'el estado de origen de {nombre_completo}', 'El estado de origen de {nombre_completo} es: {estado_de_origen}.'),
        ('estado_de_origen', 'de qué estado es originario {nombre_completo}', '{nombre_completo} es originario del estado: {estado_de_origen}.'),
        ('estado_de_origen', 'de dónde es {nombre_completo}', '{nombre_completo} es de: {estado_de_origen}.'),
        
        ('ocupacion', 'la ocupación o profesión de {nombre_completo}', 'La ocupación de {nombre_completo} es: {ocupacion}.'),
        ('ocupacion', 'en qué trabaja {nombre_completo}', '{nombre_completo} trabaja como: {ocupacion}.'),
        ('ocupacion', 'a qué se dedica {nombre_completo}', '{nombre_completo} se dedica a: {ocupacion}.'),
        ('ocupacion', 'la profesión de {nombre_completo}', 'La profesión de {nombre_completo} es: {ocupacion}.'),
        
        ('sexo', 'el sexo de {nombre_completo}', 'El sexo de {nombre_completo} es: {sexo}.'),
        ('sexo', 'el género de {nombre_completo}', 'El género de {nombre_completo} es: {sexo}.'),
        
        ('curp', 'el CURP de {nombre_completo}', 'El CURP de {nombre_completo} es: {curp}.'),
        ('curp', 'la Clave Única de Registro de Población de {nombre_completo}', 'La CURP de {nombre_completo} es: {curp}.'),
        
        ('rfc', 'el RFC de {nombre_completo}', 'El RFC de {nombre_completo} es: {rfc}.'),
        ('rfc', 'el Registro Federal de Contribuyentes de {nombre_completo}', 'El RFC de {nombre_completo} es: {rfc}.'),
        
        ('email', 'el correo electrónico de {nombre_completo}', 'El email de {nombre_completo} es: {email}.'),
        ('email', 'el email de {nombre_completo}', 'El email de {nombre_completo} es: {email}.'),
        ('email', 'cómo contactar por correo a {nombre_completo}', 'Puedes contactar a {nombre_completo} al correo: {email}.'),
        
        ('fecha_nacimiento', 'la fecha de nacimiento de {nombre_completo}', 'La fecha de nacimiento de {nombre_completo} es: {fecha_nacimiento}.'),
        ('fecha_nacimiento', 'cuándo nació {nombre_completo}', '{nombre_completo} nació el: {fecha_nacimiento}.'),
        ('fecha_nacimiento', 'la fecha en que nació {nombre_completo}', '{nombre_completo} nació en fecha: {fecha_nacimiento}.'),
        
        # REGISTROS Y OTROS CAMPOS
        ('fecha_de_afiliacion', 'la fecha de afiliación de {nombre_completo}', 'La fecha de afiliación de {nombre_completo} es: {fecha_de_afiliacion}.'),
        ('fecha_de_afiliacion', 'cuándo se afilió {nombre_completo}', '{nombre_completo} se afilió el: {fecha_de_afiliacion}.'),
        ('fecha_de_afiliacion', 'desde cuándo está registrado {nombre_completo}', '{nombre_completo} está registrado desde: {fecha_de_afiliacion}.'),
        
        ('entidad_especifica', 'la entidad (empresa/organización) asociada a {nombre_completo}', 'La entidad asociada a {nombre_completo} es: {entidad_especifica}.'),
        ('entidad_especifica', 'a qué entidad pertenece {nombre_completo}', '{nombre_completo} pertenece a la entidad: {entidad_especifica}.'),
        ('entidad_especifica', 'con qué organización está relacionado {nombre_completo}', '{nombre_completo} está relacionado con la organización: {entidad_especifica}.'),
        
        ('estado_de_registro', 'el estado donde se registró {nombre_completo}', 'El estado de registro de {nombre_completo} es: {estado_de_registro}.'),
        ('estado_de_registro', 'dónde se registró {nombre_completo}', '{nombre_completo} se registró en: {estado_de_registro}.'),
        
        ('campo_14', 'el dato del campo 14 para {nombre_completo}', 'El valor del "campo 14" para {nombre_completo} es: {campo_14}.'),
        ('campo_14', 'qué tiene en el campo 14 {nombre_completo}', '{nombre_completo} tiene en el campo 14: {campo_14}.'),
    ]

    for campo_attr, frase_q, frase_a_attr in atributos_persona:
        if campo_attr in campos: # Solo añadir plantilla si el campo existe
            plantillas.extend([
                {
                    'tipo': f'atributo_persona_{campo_attr}',
                    'pregunta': f'¿Cuál es {frase_q}?',
                    'respuesta': f'{frase_a_attr}\nOtros datos disponibles:\n{{datos_adicionales_relevantes}}',
                    'requeridos': ['nombre_completo', campo_attr]
                },
                { # Variante de pregunta
                    'tipo': f'atributo_persona_{campo_attr}_v2',
                    'pregunta': f'Necesito saber {frase_q}.',
                    'respuesta': f'Entendido. {frase_a_attr}\nInformación adicional:\n{{datos_adicionales_relevantes}}',
                    'requeridos': ['nombre_completo', campo_attr]
                },
                { # Variante imperativa
                    'tipo': f'atributo_persona_{campo_attr}_v3',
                    'pregunta': f'Dime {frase_q}.',
                    'respuesta': f'{frase_a_attr}\nDetalles adicionales:\n{{datos_adicionales_relevantes}}',
                    'requeridos': ['nombre_completo', campo_attr]
                },
                { # Variante coloquial
                    'tipo': f'atributo_persona_{campo_attr}_v4',
                    'pregunta': f'Me gustaría saber {frase_q}',
                    'respuesta': f'{frase_a_attr}\nTambién tengo esta información:\n{{datos_adicionales_relevantes}}',
                    'requeridos': ['nombre_completo', campo_attr]
                }
            ])
    
    # --- PLANTILLAS DE BÚSQUEDA INVERSA (¿QUIÉN TIENE ESTE DATO?) ---
    # (campo_id_std, frase_pregunta_id, tipo_id_para_log)
    identificadores_unicos = [
        ('telefono_completo', 'el número de teléfono {telefono_completo}', 'telefono'),
        ('telefono_completo', 'este teléfono: {telefono_completo}', 'telefono'),
        ('telefono_completo', 'el celular {telefono_completo}', 'telefono'),
        ('telefono_completo', 'este número: {telefono_completo}', 'telefono'),
        
        ('clave_ife', 'la clave de elector {clave_ife}', 'ife_ine'),
        ('clave_ife', 'esta credencial electoral: {clave_ife}', 'ife_ine'),
        ('clave_ife', 'la credencial del INE {clave_ife}', 'ife_ine'),
        
        ('curp', 'el CURP {curp}', 'curp'),
        ('curp', 'esta CURP: {curp}', 'curp'),
        ('curp', 'la Clave Única de Registro {curp}', 'curp'),
        
        ('rfc', 'el RFC {rfc}', 'rfc'),
        ('rfc', 'este Registro Federal: {rfc}', 'rfc'),
        ('rfc', 'la clave fiscal {rfc}', 'rfc'),
        
        ('email', 'el correo electrónico {email}', 'email'),
        ('email', 'este email: {email}', 'email'),
        ('email', 'la dirección de correo {email}', 'email'),
        
        ('tarjeta', 'el número de tarjeta {tarjeta}', 'tarjeta'),
        ('tarjeta', 'esta tarjeta: {tarjeta}', 'tarjeta'),
        ('tarjeta', 'la tarjeta bancaria {tarjeta}', 'tarjeta'),
        ('tarjeta', 'la tarjeta número {tarjeta}', 'tarjeta'),
    ]
    
    for campo_id, frase_q_id, tipo_log_id in identificadores_unicos:
        if campo_id in campos: # Solo si el campo identificador existe
            plantillas.extend([
                {
                    'tipo': f'quien_tiene_{tipo_log_id}',
                    'pregunta': f'¿A quién pertenece {frase_q_id}?',
                    'respuesta': f'{frase_q_id.replace("{"+campo_id+"}", str(campos.get(campo_id))).capitalize()} pertenece a {{nombre_completo}}.\nDatos de esta persona:\n{{datos_completos_sin_id_principal}}',
                    'requeridos': [campo_id, 'nombre_completo'] # Necesitamos el ID y el nombre para la respuesta
                },
                {
                    'tipo': f'quien_tiene_{tipo_log_id}_v2',
                    'pregunta': f'Investiga de quién es {frase_q_id}.',
                    'respuesta': f'Investigando... {frase_q_id.replace("{"+campo_id+"}", str(campos.get(campo_id)))} corresponde a {{nombre_completo}}.\nInformación adicional:\n{{datos_completos_sin_id_principal}}',
                    'requeridos': [campo_id, 'nombre_completo']
                },
                {
                    'tipo': f'quien_tiene_{tipo_log_id}_v3',
                    'pregunta': f'¿De quién es {frase_q_id}?',
                    'respuesta': f'{frase_q_id.replace("{"+campo_id+"}", str(campos.get(campo_id))).capitalize()} pertenece a {{nombre_completo}}.\nOtros datos registrados:\n{{datos_completos_sin_id_principal}}',
                    'requeridos': [campo_id, 'nombre_completo']
                },
                {
                    'tipo': f'quien_tiene_{tipo_log_id}_v4',
                    'pregunta': f'Busca al dueño de {frase_q_id}',
                    'respuesta': f'El dueño de {frase_q_id.replace("{"+campo_id+"}", str(campos.get(campo_id)))} es {{nombre_completo}}.\nDetalles del registro:\n{{datos_completos_sin_id_principal}}',
                    'requeridos': [campo_id, 'nombre_completo']
                },
                {
                    'tipo': f'quien_tiene_{tipo_log_id}_v5',
                    'pregunta': f'Dime quién usa {frase_q_id}',
                    'respuesta': f'{frase_q_id.replace("{"+campo_id+"}", str(campos.get(campo_id))).capitalize()} es usado por {{nombre_completo}}.\nDatos registrados:\n{{datos_completos_sin_id_principal}}',
                    'requeridos': [campo_id, 'nombre_completo']
                },
            ])

    # --- PLANTILLAS DE BÚSQUEDA POR LOCALIZACIÓN (¿QUIÉN VIVE EN...?) ---
    if 'direccion_completa' in campos:
        plantillas.extend([
            {
                'tipo': 'quien_en_direccion',
                'pregunta': '¿Quién está registrado en la dirección {direccion_completa}?',
                'respuesta': 'En la dirección {direccion_completa} se encuentra: {nombre_completo}.\n{datos_adicionales_relevantes}',
                'requeridos': ['direccion_completa', 'nombre_completo']
            },
            {
                'tipo': 'quien_en_direccion_v2',
                'pregunta': '¿Quién vive en {direccion_completa}?',
                'respuesta': 'En {direccion_completa} vive: {nombre_completo}.\n{datos_adicionales_relevantes}',
                'requeridos': ['direccion_completa', 'nombre_completo']
            },
            {
                'tipo': 'quien_en_direccion_v3',
                'pregunta': 'Dame información de la dirección {direccion_completa}',
                'respuesta': 'En la dirección {direccion_completa} se encuentra: {nombre_completo}.\nInformación adicional:\n{datos_adicionales_relevantes}',
                'requeridos': ['direccion_completa', 'nombre_completo']
            },
            {
                'tipo': 'quien_en_direccion_v4',
                'pregunta': 'Buscar residentes en {direccion_completa}',
                'respuesta': 'Residentes en {direccion_completa}:\n- {nombre_completo}\nDetalles adicionales:\n{datos_adicionales_relevantes}',
                'requeridos': ['direccion_completa', 'nombre_completo']
            },
        ])
    
    if 'colonia' in campos and 'municipio' in campos:
        plantillas.extend([
            {
                'tipo': 'quienes_en_colonia_municipio',
                'pregunta': '¿Qué personas tienes registradas en la colonia {colonia} del municipio {municipio}?',
                'respuesta': 'En la colonia {colonia} del municipio {municipio}, tengo registrada a: {nombre_completo}.\n{datos_adicionales_relevantes}',
                'requeridos': ['colonia', 'municipio', 'nombre_completo']
            },
            {
                'tipo': 'quienes_en_colonia_municipio_v2',
                'pregunta': 'Dame los habitantes de la colonia {colonia} en {municipio}',
                'respuesta': 'Habitantes registrados en colonia {colonia}, {municipio}:\n- {nombre_completo}\n{datos_adicionales_relevantes}',
                'requeridos': ['colonia', 'municipio', 'nombre_completo']
            },
        ])
    
    # --- PLANTILLAS DE BÚSQUEDA POR MUNICIPIO (¿QUIÉN ES DE ZAPOPAN?) ---
    if 'municipio' in campos:
        plantillas.extend([
            {
                'tipo': 'busqueda_por_municipio',
                'pregunta': '¿Quién es de {municipio}?',
                'respuesta': 'He encontrado estas personas que son de {municipio}:\n- {nombre_completo}\n[Puede haber más resultados disponibles]',
                'requeridos': ['municipio', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_municipio_v2',
                'pregunta': '¿Quiénes viven en {municipio}?',
                'respuesta': 'Personas que viven en {municipio}:\n- {nombre_completo}\n[Existen más resultados disponibles]',
                'requeridos': ['municipio', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_municipio_v3',
                'pregunta': 'Dame todos los de {municipio}',
                'respuesta': 'Registros de personas en {municipio}:\n- {nombre_completo}\n[Hay más resultados]',
                'requeridos': ['municipio', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_municipio_v4',
                'pregunta': 'Habitantes de {municipio}',
                'respuesta': 'Habitantes registrados de {municipio}:\n- {nombre_completo}\n[Se encontraron más resultados]',
                'requeridos': ['municipio', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_municipio_v5',
                'pregunta': 'Busca personas de {municipio}',
                'respuesta': 'Personas de {municipio}:\n- {nombre_completo}\n[Existen más registros]',
                'requeridos': ['municipio', 'nombre_completo']
            },
        ])
    
    # --- PLANTILLAS DE BÚSQUEDA POR ESTADO ---
    if 'estado' in campos:
        plantillas.extend([
            {
                'tipo': 'busqueda_por_estado',
                'pregunta': '¿Quién es del estado de {estado}?',
                'respuesta': 'Personas del estado de {estado}:\n- {nombre_completo}\n[Hay más resultados disponibles]',
                'requeridos': ['estado', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_estado_v2',
                'pregunta': '¿Quiénes viven en el estado de {estado}?',
                'respuesta': 'Habitantes del estado de {estado}:\n- {nombre_completo}\n[Se encontraron más registros]',
                'requeridos': ['estado', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_estado_v3',
                'pregunta': 'Personas del estado {estado}',
                'respuesta': 'Personas registradas en el estado {estado}:\n- {nombre_completo}\n[Existen más resultados]',
                'requeridos': ['estado', 'nombre_completo']
            },
        ])
    
    # --- PLANTILLAS DE BÚSQUEDA POR OCUPACIÓN ---
    if 'ocupacion' in campos:
        plantillas.extend([
            {
                'tipo': 'busqueda_por_ocupacion',
                'pregunta': '¿Quiénes son {ocupacion}?',
                'respuesta': 'Personas con ocupación {ocupacion}:\n- {nombre_completo}\n[Hay más registros disponibles]',
                'requeridos': ['ocupacion', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_ocupacion_v2',
                'pregunta': 'Dame los {ocupacion} registrados',
                'respuesta': 'Registros de personas con ocupación {ocupacion}:\n- {nombre_completo}\n[Existen más resultados]',
                'requeridos': ['ocupacion', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_ocupacion_v3',
                'pregunta': 'Busca personas que sean {ocupacion}',
                'respuesta': 'Personas que son {ocupacion}:\n- {nombre_completo}\n[Se encontraron más coincidencias]',
                'requeridos': ['ocupacion', 'nombre_completo']
            },
        ])
    
    # --- PLANTILLAS DE BÚSQUEDA POR SEXO ---
    if 'sexo' in campos:
        plantillas.extend([
            {
                'tipo': 'busqueda_por_sexo',
                'pregunta': '¿Quiénes son {sexo}?',
                'respuesta': 'Personas de sexo {sexo}:\n- {nombre_completo}\n[Hay más resultados disponibles]',
                'requeridos': ['sexo', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_sexo_v2',
                'pregunta': 'Dame todas las personas de sexo {sexo}',
                'respuesta': 'Registros de personas de sexo {sexo}:\n- {nombre_completo}\n[Existen más resultados]',
                'requeridos': ['sexo', 'nombre_completo']
            },
        ])
        
        # Casos específicos para hombre/mujer basados en el valor del campo sexo
        if any(row[campos['sexo']].strip().upper() in ['M', 'MASCULINO', 'H', 'HOMBRE'] for _, row in df.iterrows()):
            plantillas.extend([
                {
                    'tipo': 'busqueda_por_sexo_hombre',
                    'pregunta': '¿Quiénes son hombres?',
                    'respuesta': 'Registros de hombres:\n- {nombre_completo}\n[Hay más resultados disponibles]',
                    'requeridos': ['sexo', 'nombre_completo'],
                    'condicion': lambda row, campos: row[campos['sexo']].strip().upper() in ['M', 'MASCULINO', 'H', 'HOMBRE']
                },
                {
                    'tipo': 'busqueda_por_sexo_hombre_v2',
                    'pregunta': 'Dame todos los hombres registrados',
                    'respuesta': 'Hombres registrados en la base de datos:\n- {nombre_completo}\n[Existen más resultados]',
                    'requeridos': ['sexo', 'nombre_completo'],
                    'condicion': lambda row, campos: row[campos['sexo']].strip().upper() in ['M', 'MASCULINO', 'H', 'HOMBRE']
                },
            ])

            if any(row[campos['sexo']].strip().upper() in ['F', 'FEMENINO', 'M', 'MUJER'] for _, row in df.iterrows()):
                    plantillas.extend([
                        {
                            'tipo': 'busqueda_por_sexo_mujer',
                            'pregunta': '¿Quiénes son mujeres?',
                            'respuesta': 'Registros de mujeres:\n- {nombre_completo}\n[Hay más resultados disponibles]',
                            'requeridos': ['sexo', 'nombre_completo'],
                            'condicion': lambda row, campos: row[campos['sexo']].strip().upper() in ['F', 'FEMENINO', 'M', 'MUJER']
                        },
                        {
                            'tipo': 'busqueda_por_sexo_mujer_v2',
                            'pregunta': 'Dame todas las mujeres registradas',
                            'respuesta': 'Mujeres registradas en la base de datos:\n- {nombre_completo}\n[Existen más resultados]',
                            'requeridos': ['sexo', 'nombre_completo'],
                            'condicion': lambda row, campos: row[campos['sexo']].strip().upper() in ['F', 'FEMENINO', 'M', 'MUJER']
                        },
                    ])
            
    # --- PLANTILLAS PARA BÚSQUEDA DE TARJETA ---
    if 'tarjeta' in campos:
        plantillas.extend([
            {
                'tipo': 'busqueda_por_tarjeta',
                'pregunta': '¿A quién pertenece el número de tarjeta {tarjeta}?',
                'respuesta': 'El número de tarjeta {tarjeta} pertenece a {nombre_completo}.\n{datos_adicionales_relevantes}',
                'requeridos': ['tarjeta', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_tarjeta_v2',
                'pregunta': '¿De quién es la tarjeta {tarjeta}?',
                'respuesta': 'La tarjeta {tarjeta} pertenece a {nombre_completo}.\n{datos_adicionales_relevantes}',
                'requeridos': ['tarjeta', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_tarjeta_v3',
                'pregunta': 'Busca dueño de la tarjeta {tarjeta}',
                'respuesta': 'El dueño de la tarjeta {tarjeta} es {nombre_completo}.\n{datos_adicionales_relevantes}',
                'requeridos': ['tarjeta', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_tarjeta_v4',
                'pregunta': 'Tarjeta {tarjeta} ¿a quién pertenece?',
                'respuesta': 'La tarjeta {tarjeta} está registrada a nombre de {nombre_completo}.\n{datos_adicionales_relevantes}',
                'requeridos': ['tarjeta', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_tarjeta_v5',
                'pregunta': 'Información sobre tarjeta {tarjeta}',
                'respuesta': 'Información de la tarjeta {tarjeta}:\nTitular: {nombre_completo}\n{datos_adicionales_relevantes}',
                'requeridos': ['tarjeta', 'nombre_completo']
            },
        ])
    
    # --- PLANTILLAS COMPONENTES DE NOMBRE ---
    # Para buscar personas por nombre + iniciales de apellidos
    plantillas.extend([
        {
            'tipo': 'nombre_componentes',
            'pregunta': '¿Cuántas personas de nombre {nombre_primero} con apellidos que empiecen con {inicial_paterno} y {inicial_materno} hay?',
            'respuesta': 'Encontré personas con nombre {nombre_primero} y apellidos que inician con {inicial_paterno} y {inicial_materno}:\n- {nombre_completo}\n[Puede haber más resultados]',
            'requeridos': ['nombre_completo'],
            'especial': 'componentes_nombre'
        },
        {
            'tipo': 'nombre_componentes_v2',
            'pregunta': 'Quién se llama {nombre_primero} con {inicial_paterno} y {inicial_materno}',
            'respuesta': 'Persona(s) de nombre {nombre_primero} con iniciales {inicial_paterno} y {inicial_materno}:\n- {nombre_completo}\n[Puede haber más registros]',
            'requeridos': ['nombre_completo'],
            'especial': 'componentes_nombre'
        },
        {
            'tipo': 'nombre_componentes_v3',
            'pregunta': 'Busca a {nombre_primero} con apellidos {inicial_paterno} y {inicial_materno}',
            'respuesta': 'Resultado de búsqueda para {nombre_primero} con apellidos que inician con {inicial_paterno} y {inicial_materno}:\n- {nombre_completo}',
            'requeridos': ['nombre_completo'],
            'especial': 'componentes_nombre'
        },
    ])
    
    # --- PLANTILLAS PARA BÚSQUEDA POR CÓDIGO POSTAL ---
    if 'cp' in campos:
        plantillas.extend([
            {
                'tipo': 'busqueda_por_cp',
                'pregunta': '¿Quiénes viven en el código postal {cp}?',
                'respuesta': 'Personas que viven en el código postal {cp}:\n- {nombre_completo}\n[Hay más resultados disponibles]',
                'requeridos': ['cp', 'nombre_completo']
            },
            {
                'tipo': 'busqueda_por_cp_v2',
                'pregunta': 'Dame todos los que tienen CP {cp}',
                'respuesta': 'Registros con código postal {cp}:\n- {nombre_completo}\n[Se encontraron más coincidencias]',
                'requeridos': ['cp', 'nombre_completo']
            },
        ])
    
    # --- FILTRAR PLANTILLAS VÁLIDAS ---
    
    # Filtrar plantillas: solo usar aquellas para las cuales TODOS sus campos 'requeridos' existen en `campos`
    plantillas_validas = []
    for p in plantillas:
        # Para plantillas especiales
        if 'especial' in p and p['especial'] == 'componentes_nombre':
            # Siempre incluir estas plantillas ya que se generarán los componentes del nombre
            # a partir del nombre_completo
            if 'nombre_completo' in campos:
                plantillas_validas.append(p)
            continue
            
        # Para plantillas con condición específica
        if 'condicion' in p:
            # Verificar si los campos requeridos existen
            todos_requeridos_presentes = True
            for req_campo_std in p.get('requeridos', []):
                if req_campo_std not in campos:
                    todos_requeridos_presentes = False
                    break
            
            if todos_requeridos_presentes:
                plantillas_validas.append(p)
            continue
        
        # Para plantillas normales
        todos_requeridos_presentes = True
        for req_campo_std in p.get('requeridos', []):
            if req_campo_std not in campos:
                todos_requeridos_presentes = False
                break
        if todos_requeridos_presentes:
            plantillas_validas.append(p)
    
    if not plantillas_validas:
        print(f"⚠️ No hay plantillas válidas para generar ejemplos con los campos detectados en {nombre_archivo}: {list(campos.keys())}. Saltando...")
        return []
    
    # --- GENERAR EJEMPLOS ---
    
    ejemplos = []
    filas_df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Mezclar filas
    
    ejemplos_generados_contador = 0
    max_ejemplos_por_archivo = num_ejemplos # Para no exceder el total global rápidamente

    pbar = tqdm(total=min(max_ejemplos_por_archivo, len(filas_df) * len(plantillas_validas)), 
                desc=f"Generando para {os.path.basename(nombre_archivo)}", unit=" ej")

    # Iterar sobre las filas y luego sobre las plantillas para cada fila
    for idx_fila, row in filas_df.iterrows():
        if ejemplos_generados_contador >= max_ejemplos_por_archivo:
            break
        
        random.shuffle(plantillas_validas) # Variar el orden de plantillas para cada fila

        for plantilla in plantillas_validas:
            if ejemplos_generados_contador >= max_ejemplos_por_archivo:
                break

            # Caso especial para plantillas de componentes de nombre
            if 'especial' in plantilla and plantilla['especial'] == 'componentes_nombre':
                nombre_completo = str(row[campos['nombre_completo']]).strip()
                if not nombre_completo:
                    continue
                
                partes = nombre_completo.split()
                if len(partes) < 3:  # Necesitamos al menos nombre y dos apellidos
                    continue
                
                nombre_primero = partes[0]
                inicial_paterno = partes[1][0] if len(partes[1]) > 0 else ""
                inicial_materno = partes[2][0] if len(partes) > 2 and len(partes[2]) > 0 else ""
                
                if not nombre_primero or not inicial_paterno or not inicial_materno:
                    continue
                
                # Preparar valores para formato
                valores_para_formato = {
                    'nombre_completo': nombre_completo,
                    'nombre_primero': nombre_primero,
                    'inicial_paterno': inicial_paterno,
                    'inicial_materno': inicial_materno
                }
                
                # Obtener datos adicionales para enriquecer la respuesta
                for campo_std_general, col_df_general in campos.items():
                    if campo_std_general not in valores_para_formato:
                        valor_celda_gral = str(row[col_df_general]).strip()
                        if valor_celda_gral and pd.notna(row[col_df_general]):
                            valores_para_formato[campo_std_general] = valor_celda_gral
                
                # Preparar datos adicionales relevantes
                lista_datos_adicionales = []
                campos_para_adicionales = ['direccion_completa', 'telefono_completo', 'ocupacion', 'email']
                for campo_std_adic in campos_para_adicionales:
                    if campo_std_adic in valores_para_formato:
                        nombre_campo_display = campo_std_adic.replace('_', ' ').capitalize()
                        lista_datos_adicionales.append(f"{nombre_campo_display}: {valores_para_formato[campo_std_adic]}")
                
                valores_para_formato['datos_adicionales_relevantes'] = "\n".join(lista_datos_adicionales) if lista_datos_adicionales else "No hay otros datos relevantes disponibles."
                
                # Formatear la pregunta y respuesta
                try:
                    pregunta_formateada = plantilla['pregunta'].format(**valores_para_formato)
                    respuesta_formateada = plantilla['respuesta'].format(**valores_para_formato)
                    
                    texto_final_formato_instruccion = FORMATO_INSTRUCCION.format(
                        pregunta=pregunta_formateada,
                        respuesta=respuesta_formateada
                    )
                    
                    ejemplos.append({
                        "text": texto_final_formato_instruccion,
                        "pregunta_original": pregunta_formateada,
                        "respuesta_original": respuesta_formateada,
                        "tipo_plantilla": plantilla['tipo'],
                        "fuente_archivo": os.path.basename(nombre_archivo),
                        "fila_indice_original": int(row.name)
                    })
                    
                    ejemplos_generados_contador += 1
                    pbar.update(1)
                    
                except Exception as e:
                    continue
                
                # Continuar con la siguiente plantilla
                continue

            # Caso para plantillas con condición específica
            if 'condicion' in plantilla:
                condicion_cumplida = plantilla['condicion'](row, campos)
                if not condicion_cumplida:
                    continue

            # Recolectar valores de la fila actual para los campos de la plantilla y campos generales
            valores_para_formato = {}
            todos_valores_requeridos_en_fila_ok = True

            # Primero, asegurar que los campos requeridos específicamente por la plantilla tengan valor en esta fila
            for campo_std_requerido in plantilla['requeridos']:
                col_df_requerido = campos[campo_std_requerido] # Sabemos que campo_std_requerido está en campos por el filtro previo
                valor_celda = str(row[col_df_requerido]).strip()
                if valor_celda and pd.notna(row[col_df_requerido]):
                    valores_para_formato[campo_std_requerido] = valor_celda
                else:
                    todos_valores_requeridos_en_fila_ok = False # Un campo requerido no tiene valor en esta fila
                    break 
            
            if not todos_valores_requeridos_en_fila_ok:
                continue # Pasar a la siguiente plantilla para esta fila

            # Si los requeridos están OK, obtener el resto de campos mapeados para datos_completos, etc.
            for campo_std_general, col_df_general in campos.items():
                if campo_std_general not in valores_para_formato: # Evitar sobrescribir los ya cargados
                    valor_celda_gral = str(row[col_df_general]).strip()
                    if valor_celda_gral and pd.notna(row[col_df_general]):
                            valores_para_formato[campo_std_general] = valor_celda_gral
            
            try:
                # Preparar 'datos_completos'
                lista_datos_completos = []
                campos_ordenados_para_respuesta = [ # Un orden sugerido para la presentación
                    'nombre_completo', 'fecha_nacimiento', 'curp', 'rfc', 'sexo', 'ocupacion', 
                    'direccion_completa', 'calle', 'numero_exterior', 'numero_interior', 'colonia', 
                    'municipio', 'ciudad_especifica', 'estado', 'cp', 
                    'telefono_completo', 'email', 'clave_ife', 
                    'fecha_de_afiliacion', 'entidad_especifica', 'estado_de_origen', 'estado_de_registro', 
                    'tarjeta', 'campo_14' # Otros al final
                ]
                for campo_std_resp in campos_ordenados_para_respuesta:
                    if campo_std_resp in valores_para_formato: # Si el campo tiene valor para esta fila
                        # No incluir campos construidos si su versión "original" mapeada es la misma (evitar redundancia)
                        if campo_std_resp.endswith("_construido") and \
                            campos.get(campo_std_resp.replace("_construido","")) == campos.get(campo_std_resp) :
                            continue

                        nombre_campo_display = campo_std_resp.replace('_', ' ').capitalize()
                        lista_datos_completos.append(f"{nombre_campo_display}: {valores_para_formato[campo_std_resp]}")
                
                valores_para_formato['datos_completos'] = "\n".join(lista_datos_completos) if lista_datos_completos else "No hay datos detallados disponibles."

                # Preparar 'datos_adicionales_relevantes' (subconjunto de datos_completos, excluyendo el foco principal de la pregunta si es posible)
                id_principal_pregunta = None
                # Identificar el campo principal usado en la pregunta (heurística: el primer placeholder que es un campo)
                placeholders_pregunta = re.findall(r'\{([^}]+)\}', plantilla['pregunta'])
                for ph in placeholders_pregunta:
                    if ph in campos.keys() and ph != 'nombre_completo': # No considerar nombre_completo como ID principal aquí
                        id_principal_pregunta = ph
                        break
                
                lista_datos_adicionales = []
                campos_para_adicionales = ['direccion_completa', 'telefono_completo', 'ocupacion', 'email'] # Ejemplo
                if 'nombre_completo' in valores_para_formato and id_principal_pregunta != 'nombre_completo': # Siempre añadir nombre si no es el ID
                    lista_datos_adicionales.append(f"Nombre Completo: {valores_para_formato['nombre_completo']}")

                for campo_std_adic in campos_para_adicionales:
                    if campo_std_adic in valores_para_formato and campo_std_adic != id_principal_pregunta:
                        nombre_campo_display = campo_std_adic.replace('_', ' ').capitalize()
                        lista_datos_adicionales.append(f"{nombre_campo_display}: {valores_para_formato[campo_std_adic]}")
                valores_para_formato['datos_adicionales_relevantes'] = "\n".join(lista_datos_adicionales) if lista_datos_adicionales else "No hay otros datos relevantes disponibles."


                # Preparar 'datos_completos_sin_id_principal' (todos los datos excepto el ID usado en la pregunta)
                lista_datos_sin_id = []
                for item_dc in lista_datos_completos:
                    ignorar_item = False
                    if id_principal_pregunta:
                        # Comprobar si el item_dc comienza con el nombre display del id_principal_pregunta
                        nombre_display_id = id_principal_pregunta.replace('_', ' ').capitalize()
                        if item_dc.startswith(nombre_display_id + ":"):
                            ignorar_item = True
                    if not ignorar_item:
                        lista_datos_sin_id.append(item_dc)
                valores_para_formato['datos_completos_sin_id_principal'] = "\n".join(lista_datos_sin_id) if lista_datos_sin_id else "No hay más datos disponibles."


                # Formatear la pregunta y respuesta finales
                pregunta_formateada = plantilla['pregunta'].format(**valores_para_formato)
                respuesta_formateada = plantilla['respuesta'].format(**valores_para_formato)

                # Validar que no queden placeholders sin reemplazar (importante)
                if '{' in pregunta_formateada or '{' in respuesta_formateada:
                    continue # Saltar este ejemplo

                texto_final_formato_instruccion = FORMATO_INSTRUCCION.format(
                    pregunta=pregunta_formateada,
                    respuesta=respuesta_formateada
                )

                ejemplos.append({
                    "text": texto_final_formato_instruccion,
                    "pregunta_original": pregunta_formateada,
                    "respuesta_original": respuesta_formateada,
                    "tipo_plantilla": plantilla['tipo'],
                    "fuente_archivo": os.path.basename(nombre_archivo),
                    "fila_indice_original": int(row.name) # Guardar el índice original de la fila del df
                })
                ejemplos_generados_contador += 1
                pbar.update(1)

            except KeyError as e:
                # Este error puede ocurrir si un placeholder en la plantilla no tiene un valor correspondiente en `valores_para_formato`
                continue 
            except Exception as e_gen:
                print(f"    ❌ Error inesperado al procesar plantilla '{plantilla['tipo']}', fila {idx_fila}, archivo {nombre_archivo}: {e_gen}")
                continue # Pasar a la siguiente plantilla o fila

    pbar.close()
    if ejemplos_generados_contador == 0 and len(filas_df) > 0 and len(plantillas_validas) > 0:
        print(f"    ℹ️ No se generaron ejemplos para {nombre_archivo} a pesar de tener filas y plantillas. Verificar si las filas tienen datos para los campos requeridos por las plantillas.")
    elif ejemplos_generados_contador > 0:
        print(f"    ✓ Generados {ejemplos_generados_contador} ejemplos para {os.path.basename(nombre_archivo)}")
    
    return ejemplos
# ---------------------------------------------------------------------------
# FIN DE MODIFICACIÓN: ampliar plantillas en generar_ejemplos_qa
# ---------------------------------------------------------------------------

def main():
    """Función principal para generar los datos de entrenamiento"""
    print("=== GENERADOR DE DATOS DE ENTRENAMIENTO MEJORADO ===")
    print(f"Buscando archivos en: {CARPETA_BD}")
    
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"Directorio de salida creado: {CARPETA_SALIDA}")

    todos_ejemplos = []

    ruta_feedback = r"C:\Users\TEC-INT02\Documents\Agent-IA\feedback\ejemplos_entrenamiento.json"

    if os.path.exists(ruta_feedback):
        try:
            with open(ruta_feedback, 'r', encoding='utf-8') as f:
                ejemplos_feedback = json.load(f)
                if ejemplos_feedback:
                    print(f"✓ Cargados {len(ejemplos_feedback)} ejemplos de retroalimentación")
                    
                    # Convertir ejemplos de feedback al formato de entrenamiento
                    ejemplos_convertidos = []
                    for ejemplo in ejemplos_feedback:
                        prompt = ejemplo.get('prompt', '')
                        tipo_busqueda = ejemplo.get('tipo_busqueda_correcta', '')
                        campo = ejemplo.get('campo_correcto', '')
                        valor = ejemplo.get('valor_correcto', '')
                        
                        if prompt and tipo_busqueda:
                            formato_instruccion = FORMATO_INSTRUCCION.format(
                                pregunta=prompt,
                                respuesta=f"Para la consulta '{prompt}', el tipo de búsqueda correcto es '{tipo_busqueda}', el campo es '{campo}' y el valor a buscar es '{valor}'."
                            )
                            
                            ejemplo_formateado = {
                                "text": formato_instruccion,
                                "pregunta_original": prompt,
                                "respuesta_original": f"Tipo: {tipo_busqueda}, Campo: {campo}, Valor: {valor}",
                                "tipo_plantilla": "feedback_usuario",
                                "fuente_archivo": "feedback_usuario",
                                "fila_indice_original": 0
                            }
                            ejemplos_convertidos.append(ejemplo_formateado)
                    
                    # Añadir los ejemplos convertidos a la lista general
                    if ejemplos_convertidos:
                        todos_ejemplos.extend(ejemplos_convertidos)
                        print(f"✓ Agregados {len(ejemplos_convertidos)} ejemplos de retroalimentación al conjunto de entrenamiento")
        except Exception as e:
            print(f"❌ Error al cargar ejemplos de retroalimentación: {e}")
    
    archivos_en_carpeta = [f for f in os.listdir(CARPETA_BD) if os.path.isfile(os.path.join(CARPETA_BD, f))]
    print(f"Archivos encontrados: {archivos_en_carpeta}")

    for nombre_fichero in archivos_en_carpeta:
        ruta_completa_fichero = os.path.join(CARPETA_BD, nombre_fichero)
        df_actual = cargar_dataframe(ruta_completa_fichero)
        
        if df_actual is None or df_actual.empty:
            print(f"⚠️ No se pudo cargar o el archivo está vacío: {nombre_fichero}. Saltando...")
            continue
        
        print(f"🔍 Detectando y normalizando campos para: {nombre_fichero} ({len(df_actual)} filas)")
        df_procesado, campos_detectados = detectar_y_normalizar_campos(df_actual.copy()) # Usar .copy() para evitar SettingWithCopyWarning
        
        if not campos_detectados:
            print(f"⚠️ No se detectaron campos utilizables en {nombre_fichero} según el mapeo. Saltando generación de QA.")
            continue
            
        # Ajustar el número de ejemplos por archivo según necesidad y tamaño del dataset
        # Podrías hacerlo proporcional al tamaño del df: num_ej_por_archivo = min(150, len(df_procesado) * 2)
        num_ej_por_archivo = 100 # O un valor fijo como antes
        print(f"🧠 Generando hasta {num_ej_por_archivo} ejemplos de QA para {nombre_fichero}...")
        ejemplos_de_este_archivo = generar_ejemplos_qa(df_procesado, campos_detectados, nombre_fichero, num_ejemplos=num_ej_por_archivo)
        todos_ejemplos.extend(ejemplos_de_este_archivo)
        print(f"Total ejemplos acumulados: {len(todos_ejemplos)}")
    
    num_total_ejemplos = len(todos_ejemplos)
    if num_total_ejemplos > 0:
        random.shuffle(todos_ejemplos)
        
        # Dividir en entrenamiento y validación
        # Puedes ajustar el porcentaje de división
        idx_division = int(num_total_ejemplos * 0.9)  # 90% entrenamiento, 10% validación
        ejemplos_entrenamiento = todos_ejemplos[:idx_division]
        ejemplos_validacion = todos_ejemplos[idx_division:]
        
        # Guardar en formato JSONL (JSON Lines), que es común para fine-tuning
        # Un objeto JSON por línea.
        ruta_entrenamiento_jsonl = os.path.join(CARPETA_SALIDA, "train_data.jsonl")
        with open(ruta_entrenamiento_jsonl, 'w', encoding='utf-8') as f_train:
            for ejemplo in ejemplos_entrenamiento:
                f_train.write(json.dumps(ejemplo, ensure_ascii=False) + '\n')
        
        ruta_validacion_jsonl = os.path.join(CARPETA_SALIDA, "val_data.jsonl")
        with open(ruta_validacion_jsonl, 'w', encoding='utf-8') as f_val:
            for ejemplo in ejemplos_validacion:
                f_val.write(json.dumps(ejemplo, ensure_ascii=False) + '\n')
        
        # Guardar una muestra para inspección manual (en formato de texto más legible)
        num_muestras_inspeccion = min(20, len(ejemplos_entrenamiento)) # Aumentado a 20 para mejor revisión
        ejemplos_para_muestra = random.sample(ejemplos_entrenamiento, num_muestras_inspeccion)
        ruta_archivo_muestra = os.path.join(CARPETA_SALIDA, "ejemplos_muestra_inspeccion.txt")
        with open(ruta_archivo_muestra, 'w', encoding='utf-8') as f_muestra:
            for i, ej_muestra in enumerate(ejemplos_para_muestra):
                f_muestra.write(f"--- EJEMPLO DE MUESTRA {i+1} ---\n")
                f_muestra.write(f"Fuente: {ej_muestra.get('fuente_archivo', 'N/A')}, Fila Original Idx: {ej_muestra.get('fila_indice_original', 'N/A')}\n")
                f_muestra.write(f"Tipo Plantilla: {ej_muestra.get('tipo_plantilla', 'N/A')}\n\n")
                # Escribir el texto completo que va al modelo (con instrucción y respuesta)
                f_muestra.write(ej_muestra.get("text", "FORMATO TEXT NO ENCONTRADO"))
                # Opcional: escribir pregunta y respuesta originales por separado si quieres verlas claramente
                # f_muestra.write("\n--- PREGUNTA ORIGINAL ---\n")
                # f_muestra.write(ej_muestra.get("pregunta_original", ""))
                # f_muestra.write("\n--- RESPUESTA ORIGINAL ---\n")
                # f_muestra.write(ej_muestra.get("respuesta_original", ""))
                f_muestra.write("\n\n" + "="*60 + "\n\n")
        
        print(f"\n✅ ¡Proceso completado! Datos generados:")
        print(f"  Total de ejemplos generados: {num_total_ejemplos}")
        print(f"  Ejemplos para entrenamiento: {len(ejemplos_entrenamiento)} (guardados en {ruta_entrenamiento_jsonl})")
        print(f"  Ejemplos para validación: {len(ejemplos_validacion)} (guardados en {ruta_validacion_jsonl})")
        print(f"  Muestra para inspección manual ({num_muestras_inspeccion} ejemplos): {ruta_archivo_muestra}")
    else:
        print("❌ No se pudieron generar ejemplos. Revisa tus archivos de datos, el mapeo de campos y los logs de errores.")

if __name__ == "__main__":
    main()