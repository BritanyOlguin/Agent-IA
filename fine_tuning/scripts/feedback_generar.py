# Guarda este script como generar_ejemplos_desde_feedback.py
import os
import pickle
import json

# Rutas de archivos
RUTA_FEEDBACK = r"C:\Users\TEC-INT02\Documents\Agent-IA\feedback"
ARCHIVO_FEEDBACK = os.path.join(RUTA_FEEDBACK, "feedback_registro.pkl")
ARCHIVO_EJEMPLOS = os.path.join(RUTA_FEEDBACK, "ejemplos_entrenamiento.json")

# Crear directorio si no existe
os.makedirs(RUTA_FEEDBACK, exist_ok=True)

# Cargar registro de feedback
if os.path.exists(ARCHIVO_FEEDBACK):
    try:
        with open(ARCHIVO_FEEDBACK, 'rb') as f:
            registro = pickle.load(f)
        print(f"✓ Cargado registro con {len(registro)} entradas")
        
        # Filtrar solo entradas con feedback negativo (incorrectas)
        entradas_incorrectas = [r for r in registro if not r.get('es_correcto', True)]
        print(f"✓ Encontradas {len(entradas_incorrectas)} entradas con feedback negativo")
        
        # Filtrar entradas con sugerencias
        entradas_con_sugerencia = [r for r in entradas_incorrectas if r.get('sugerencia')]
        print(f"✓ Encontradas {len(entradas_con_sugerencia)} entradas con sugerencias")
        
        # Formato para los ejemplos
        ejemplos = []
        for entrada in entradas_con_sugerencia:
            ejemplo = {
                "prompt": entrada['prompt'],
                "tipo_busqueda_correcta": entrada.get('sugerencia', {}).get('tipo_busqueda'),
                "campo_correcto": entrada.get('sugerencia', {}).get('campo'),
                "valor_correcto": entrada.get('sugerencia', {}).get('valor')
            }
            ejemplos.append(ejemplo)
        
        # Guardar ejemplos para entrenamiento
        with open(ARCHIVO_EJEMPLOS, 'w', encoding='utf-8') as f:
            json.dump(ejemplos, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Generados {len(ejemplos)} ejemplos para entrenamiento en {ARCHIVO_EJEMPLOS}")
        
    except Exception as e:
        print(f"❌ Error al procesar archivo de feedback: {e}")
else:
    print(f"❌ No se encontró el archivo de feedback en {ARCHIVO_FEEDBACK}")