# Mejora del script feedback_generar.py
import os
import pickle
import json
import pandas as pd
from datetime import datetime

# Configuración de rutas
RUTA_FEEDBACK = r"C:\Users\TEC-INT02\Documents\Agent-IA\feedback"
ARCHIVO_FEEDBACK = os.path.join(RUTA_FEEDBACK, "feedback_registro.pkl")
ARCHIVO_EJEMPLOS = os.path.join(RUTA_FEEDBACK, "ejemplos_entrenamiento.json")
ARCHIVO_CSV = os.path.join(RUTA_FEEDBACK, "ejemplos_entrenamiento.csv")

# Crear directorio si no existe
os.makedirs(RUTA_FEEDBACK, exist_ok=True)

def generar_ejemplos_entrenamiento():
    """
    Genera ejemplos de entrenamiento a partir del registro de feedback
    y los guarda en formato JSON y CSV para análisis y entrenamiento.
    """
    print("\n=== GENERANDO EJEMPLOS DE ENTRENAMIENTO DESDE FEEDBACK ===")
    
    # Verificar si existe el archivo de feedback
    if not os.path.exists(ARCHIVO_FEEDBACK):
        print(f"❌ No se encontró archivo de feedback en: {ARCHIVO_FEEDBACK}")
        return False
    
    try:
        # Cargar registro de feedback
        with open(ARCHIVO_FEEDBACK, 'rb') as f:
            registro = pickle.load(f)
        
        print(f"✓ Cargado registro con {len(registro)} entradas")
        
        # Filtrar entradas con feedback negativo y sugerencias
        entradas_incorrectas = [r for r in registro if not r.get('es_correcto', True)]
        print(f"✓ Encontradas {len(entradas_incorrectas)} entradas con feedback negativo")
        
        entradas_con_sugerencia = [r for r in entradas_incorrectas if r.get('sugerencia')]
        print(f"✓ Encontradas {len(entradas_con_sugerencia)} entradas con sugerencias de corrección")
        
        # Verificar si hay suficientes ejemplos para entrenar
        if len(entradas_con_sugerencia) < 5:
            print(f"⚠️ Solo se encontraron {len(entradas_con_sugerencia)} ejemplos con sugerencias.")
            print("   Se recomienda tener al menos 10-20 ejemplos para un entrenamiento efectivo.")
            respuesta = input("¿Desea continuar de todos modos? (s/n): ")
            if respuesta.lower() != 's':
                print("Operación cancelada por el usuario.")
                return False
        
        # Formato para los ejemplos
        ejemplos = []
        for entrada in entradas_con_sugerencia:
            # Asegurar que la sugerencia tenga el formato esperado (diccionario)
            sugerencia = entrada.get('sugerencia', {})
            if isinstance(sugerencia, str):
                print(f"⚠️ Formato de sugerencia incorrecto: {sugerencia}")
                continue
                
            if not isinstance(sugerencia, dict):
                print(f"⚠️ Tipo de sugerencia incorrecto: {type(sugerencia)}")
                continue
                
            ejemplo = {
                "prompt": entrada['prompt'],
                "tipo_busqueda_correcta": sugerencia.get('tipo_busqueda', ''),
                "campo_correcto": sugerencia.get('campo', ''),
                "valor_correcto": sugerencia.get('valor', ''),
                "timestamp": entrada.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            }
            
            # Verificar que los campos requeridos tengan valores
            if not ejemplo["prompt"] or not ejemplo["tipo_busqueda_correcta"]:
                print(f"⚠️ Ejemplo incompleto: {ejemplo}")
                continue
                
            ejemplos.append(ejemplo)
        
        # Guardar ejemplos para entrenamiento
        if ejemplos:
            # Guardar en formato JSON
            with open(ARCHIVO_EJEMPLOS, 'w', encoding='utf-8') as f:
                json.dump(ejemplos, f, ensure_ascii=False, indent=2)
            
            # Guardar en formato CSV para visualización fácil
            df = pd.DataFrame(ejemplos)
            df.to_csv(ARCHIVO_CSV, index=False, encoding='utf-8')
            
            print(f"✅ Generados {len(ejemplos)} ejemplos para entrenamiento")
            print(f"   - JSON: {ARCHIVO_EJEMPLOS}")
            print(f"   - CSV: {ARCHIVO_CSV}")
            
            return True
        else:
            print("❌ No se pudieron generar ejemplos válidos.")
            return False
            
    except Exception as e:
        print(f"❌ Error al procesar archivo de feedback: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    resultado = generar_ejemplos_entrenamiento()
    
    if resultado:
        print("\n¿Qué deseas hacer ahora?")
        print("1. Iniciar entrenamiento con estos ejemplos")
        print("2. Solo generar ejemplos y salir")
        
        opcion = input("Opción (1/2): ")
        
        if opcion == "1":
            print("\nIniciando proceso de entrenamiento...")
            # Esta parte se implementará en la siguiente fase
            print("Funcionalidad en desarrollo. Por favor ejecuta el script de entrenamiento manualmente.")
        else:
            print("\nEjemplos generados. Ejecuta el script de entrenamiento cuando lo desees.")