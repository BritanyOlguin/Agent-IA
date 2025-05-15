import os
import sys
import subprocess
import argparse
from datetime import datetime

# Configuración de rutas
RUTA_BASE = r"C:\Users\TEC-INT02\Documents\Agent-IA"
RUTA_FEEDBACK = os.path.join(RUTA_BASE, "feedback")
RUTA_DATOS = os.path.join(RUTA_BASE, "fine_tuning", "datos")
RUTA_SCRIPTS = os.path.join(RUTA_BASE, "fine_tuning", "scripts")
RUTA_MODELOS = os.path.join(RUTA_BASE, "fine_tuning", "modelos")

def verificar_rutas():
    """Verifica que todas las rutas necesarias existan"""
    rutas = [RUTA_BASE, RUTA_FEEDBACK, RUTA_DATOS, RUTA_MODELOS]
    for ruta in rutas:
        if not os.path.exists(ruta):
            print(f"❌ Ruta no encontrada: {ruta}")
            os.makedirs(ruta, exist_ok=True)
            print(f"   Directorio creado: {ruta}")

def ejecutar_script(ruta_script, mensaje=None):
    """Ejecuta un script de Python y muestra su salida"""
    if mensaje:
        print(f"\n=== {mensaje} ===")
    
    print(f"Ejecutando: {os.path.basename(ruta_script)}")
    
    try:
        resultado = subprocess.run([sys.executable, ruta_script], 
                                   check=True, 
                                   capture_output=True, 
                                   text=True)
        
        print(resultado.stdout)
        
        if resultado.stderr:
            print("ADVERTENCIAS/ERRORES:")
            print(resultado.stderr)
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar {os.path.basename(ruta_script)}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Flujo completo de entrenamiento con feedback")
    parser.add_argument("--diagnostico", action="store_true", help="Solo ejecutar diagnóstico de feedback")
    parser.add_argument("--generar", action="store_true", help="Solo generar ejemplos de entrenamiento")
    parser.add_argument("--entrenar", action="store_true", help="Solo ejecutar entrenamiento")
    parser.add_argument("--epochs", type=int, default=5, help="Número de epochs para entrenamiento")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate para entrenamiento")
    parser.add_argument("--batch", type=int, default=2, help="Batch size para entrenamiento")
    args = parser.parse_args()
    
    print("\n🤖 FLUJO DE ENTRENAMIENTO CON FEEDBACK")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar rutas
    verificar_rutas()
    
    # Crear scripts si no existen
    script_diagnostico = os.path.join(RUTA_FEEDBACK, "diagnostico_feedback.py")
    script_generar = os.path.join(RUTA_FEEDBACK, "feedback_generar.py")
    script_datos = os.path.join(RUTA_BASE, "generar_datos.py")
    script_entrenar = os.path.join(RUTA_BASE, "entrenar_modelo.py")
    
    # Paso 1: Diagnóstico (opcional)
    if args.diagnostico or not (args.generar or args.entrenar):
        # Si no existe el script de diagnóstico, crearlo
        if not os.path.exists(script_diagnostico):
            print(f"⚠️ Script de diagnóstico no encontrado. Creando...")
            # Aquí insertar el código para crear el script de diagnóstico
            # (omitido por brevedad)
        
        ejecutar_script(script_diagnostico, "DIAGNÓSTICO DE FEEDBACK")
        
        if args.diagnostico:
            return
    
    # Paso 2: Generar ejemplos (opcional)
    if args.generar or not args.entrenar:
        # Si no existe el script de generación, crearlo
        if not os.path.exists(script_generar):
            print(f"⚠️ Script de generación de ejemplos no encontrado. Creando...")
            # Aquí insertar el código para crear el script de generación
            # (omitido por brevedad)
        
        ejecutar_script(script_generar, "GENERANDO EJEMPLOS DE FEEDBACK")
        
        # Generar datos de entrenamiento
        ejecutar_script(script_datos, "GENERANDO DATOS DE ENTRENAMIENTO")
        
        if args.generar:
            return
    
    # Paso 3: Entrenar modelo
    if args.entrenar or not (args.diagnostico or args.generar):
        # Construir comando para entrenamiento con parámetros
        comando_entrenar = [
            sys.executable, 
            script_entrenar,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch),
            "--lr", str(args.lr)
        ]
        
        print("\n=== ENTRENANDO MODELO ===")
        print(f"Ejecutando: {' '.join(comando_entrenar)}")
        
        try:
            resultado = subprocess.run(comando_entrenar, check=True)
            print("✅ Entrenamiento completado con éxito")
        except subprocess.CalledProcessError:
            print("❌ Error durante el entrenamiento")

if __name__ == "__main__":
    main()