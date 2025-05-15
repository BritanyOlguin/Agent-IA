import os
import pickle
import json
import pandas as pd
from datetime import datetime

# Configuración de rutas
RUTA_FEEDBACK = r"C:\Users\TEC-INT02\Documents\Agent-IA\feedback"
ARCHIVO_FEEDBACK = os.path.join(RUTA_FEEDBACK, "feedback_registro.pkl")
CARPETA_DATOS = r"C:\Users\TEC-INT02\Documents\Agent-IA\fine_tuning\datos"

def diagnosticar_feedback():
    """
    Analiza el archivo de retroalimentación para detectar problemas
    y proporciona un informe detallado del estado actual.
    """
    print("\n=== DIAGNÓSTICO DE DATOS DE RETROALIMENTACIÓN ===\n")
    
    # Comprobar si existe el directorio
    if not os.path.exists(RUTA_FEEDBACK):
        print(f"❌ No existe el directorio de feedback: {RUTA_FEEDBACK}")
        print("   Creando directorio...")
        os.makedirs(RUTA_FEEDBACK, exist_ok=True)
    
    # Comprobar si existe el archivo de feedback
    if not os.path.exists(ARCHIVO_FEEDBACK):
        print(f"❌ No existe el archivo de feedback: {ARCHIVO_FEEDBACK}")
        print("   No hay datos de retroalimentación para analizar.")
        return False
    
    try:
        # Cargar el archivo de feedback
        with open(ARCHIVO_FEEDBACK, 'rb') as f:
            registro = pickle.load(f)
        
        if not registro:
            print("❌ El archivo de feedback existe pero está vacío.")
            return False
        
        print(f"✓ Cargado registro con {len(registro)} entradas")
        
        # Analizar distribución de feedback
        feedback_positivo = [r for r in registro if r.get('es_correcto', True)]
        feedback_negativo = [r for r in registro if not r.get('es_correcto', True)]
        
        print(f"\n📊 Distribución del feedback:")
        print(f"  - Total de entradas: {len(registro)}")
        print(f"  - Feedback positivo: {len(feedback_positivo)} ({len(feedback_positivo)/len(registro)*100:.1f}%)")
        print(f"  - Feedback negativo: {len(feedback_negativo)} ({len(feedback_negativo)/len(registro)*100:.1f}%)")
        
        # Analizar entradas de feedback negativo con sugerencias
        con_sugerencias = [r for r in feedback_negativo if r.get('sugerencia')]
        sin_sugerencias = [r for r in feedback_negativo if not r.get('sugerencia')]
        
        print(f"\n📋 Análisis de feedback negativo:")
        print(f"  - Con sugerencias: {len(con_sugerencias)} ({len(con_sugerencias)/len(feedback_negativo)*100 if feedback_negativo else 0:.1f}%)")
        print(f"  - Sin sugerencias: {len(sin_sugerencias)} ({len(sin_sugerencias)/len(feedback_negativo)*100 if feedback_negativo else 0:.1f}%)")
        
        # Analizar tipos de sugerencias
        tipos_sugerencias = {}
        for entrada in con_sugerencias:
            sugerencia = entrada.get('sugerencia', {})
            tipo = sugerencia.get('tipo_busqueda', 'desconocido')
            tipos_sugerencias[tipo] = tipos_sugerencias.get(tipo, 0) + 1
        
        if tipos_sugerencias:
            print("\n🔍 Tipos de sugerencias:")
            for tipo, count in tipos_sugerencias.items():
                print(f"  - {tipo}: {count} ({count/len(con_sugerencias)*100:.1f}%)")
        
        # Verificar formato de las sugerencias
        formatos_incorrectos = []
        for i, entrada in enumerate(con_sugerencias):
            sugerencia = entrada.get('sugerencia', {})
            if not isinstance(sugerencia, dict):
                formatos_incorrectos.append((i, f"Tipo incorrecto: {type(sugerencia)}"))
                continue
            
            if not sugerencia.get('tipo_busqueda'):
                formatos_incorrectos.append((i, "Falta 'tipo_busqueda'"))
            
            if 'valor' not in sugerencia:
                formatos_incorrectos.append((i, "Falta 'valor'"))
        
        if formatos_incorrectos:
            print("\n⚠️ Problemas de formato en sugerencias:")
            for idx, problema in formatos_incorrectos[:5]:  # Mostrar solo los primeros 5
                print(f"  - Entrada #{idx}: {problema}")
            
            if len(formatos_incorrectos) > 5:
                print(f"    ... y {len(formatos_incorrectos) - 5} problemas más")
        
        # Verificar si hay datos de entrenamiento generados
        ruta_train = os.path.join(CARPETA_DATOS, "train_data.jsonl")
        
        if os.path.exists(ruta_train):
            print("\n📁 Datos de entrenamiento:")
            try:
                ejemplos = []
                with open(ruta_train, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            ejemplos.append(json.loads(line))
                        except:
                            pass
                
                print(f"  - Ejemplos totales: {len(ejemplos)}")
                
                # Contar ejemplos de feedback
                ejemplos_feedback = [e for e in ejemplos if e.get('fuente_archivo') == 'feedback_usuario']
                print(f"  - Ejemplos de feedback: {len(ejemplos_feedback)} ({len(ejemplos_feedback)/len(ejemplos)*100 if ejemplos else 0:.1f}%)")
                
                # Verificar última modificación
                ultima_mod = datetime.fromtimestamp(os.path.getmtime(ruta_train))
                edad_datos = (datetime.now() - ultima_mod).days
                
                print(f"  - Última modificación: {ultima_mod.strftime('%Y-%m-%d %H:%M')}")
                print(f"  - Edad de los datos: {edad_datos} días")
                
                if edad_datos > 0:
                    print(f"  ⚠️ Los datos de entrenamiento no incluyen feedback reciente (últimos {edad_datos} días)")
                    
                    # Comprobar feedback más reciente
                    fechas_feedback = [datetime.strptime(r.get('timestamp', '2000-01-01'), "%Y-%m-%d %H:%M:%S") 
                                        for r in registro if r.get('timestamp')]
                    if fechas_feedback:
                        fecha_mas_reciente = max(fechas_feedback)
                        dias_desde_feedback = (datetime.now() - fecha_mas_reciente).days
                        
                        if dias_desde_feedback < edad_datos:
                            print(f"    📌 Hay feedback más reciente ({dias_desde_feedback} días) que no está incluido en los datos de entrenamiento")
                            print(f"    💡 Ejecute el script de generación de datos para incluir el feedback más reciente")
                
            except Exception as e:
                print(f"  ❌ Error al analizar datos de entrenamiento: {e}")
        else:
            print("\n⚠️ No se encontraron datos de entrenamiento generados.")
            print("   Ejecute el script de generación de datos para incluir su feedback en el entrenamiento.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al analizar feedback: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    diagnosticar_feedback()
    
    print("\n¿Qué desea hacer ahora?")
    print("1. Generar ejemplos de entrenamiento desde feedback")
    print("2. Iniciar entrenamiento con ejemplos existentes")
    print("3. Salir")
    
    opcion = input("Opción (1/2/3): ")
    
    if opcion == "1":
        print("\nGenerando ejemplos de entrenamiento desde feedback...")
        # Importar y ejecutar el script de generación de ejemplos
        try:
            import sys
            sys.path.append(RUTA_FEEDBACK)
            from feedback_generar import generar_ejemplos_entrenamiento
            generar_ejemplos_entrenamiento()
        except Exception as e:
            print(f"❌ Error al generar ejemplos: {e}")
            
    elif opcion == "2":
        print("\nIniciando entrenamiento con ejemplos existentes...")
        # Ejecutar el script de entrenamiento
        try:
            import subprocess
            subprocess.run(["python", r"C:\Users\TEC-INT02\Documents\Agent-IA\fine_tuning\scripts\entrenar_modelo.py"])
        except Exception as e:
            print(f"❌ Error al iniciar entrenamiento: {e}")