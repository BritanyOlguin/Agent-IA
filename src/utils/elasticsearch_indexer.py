"""
Script para indexar todos los datos en Elasticsearch.
Reemplaza el proceso de indexación tradicional.
"""

import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.elasticsearch_engine import ElasticsearchEngine

def main():
    """Función principal para indexar datos"""
    print("="*60)
    print("🚀 INDEXADOR DE ELASTICSEARCH")
    print("="*60)
    
    # Configurar rutas
    carpeta_datos = r"C:\Users\TEC-INT02\Documents\Agent-IA\data\source"
    
    if not os.path.exists(carpeta_datos):
        print(f"❌ Carpeta de datos no encontrada: {carpeta_datos}")
        print("   Verifica la ruta en el script.")
        return
    
    try:
        # Inicializar motor de Elasticsearch
        print("🔄 Conectando a Elasticsearch...")
        engine = ElasticsearchEngine()
        
        # Preguntar si eliminar índice existente
        stats = engine.get_stats()
        if stats.get('total_documents', 0) > 0:
            print(f"⚠️ Ya existen {stats['total_documents']} documentos indexados.")
            respuesta = input("¿Quieres eliminar el índice existente y reindexar todo? (s/n): ")
            
            if respuesta.lower().startswith('s'):
                print("🗑️ Eliminando índice existente...")
                engine.delete_index()
                # Recrear motor para configurar índice nuevo
                engine = ElasticsearchEngine()
            else:
                print("🔄 Agregando a índice existente...")
        
        # Indexar datos
        print(f"📁 Indexando archivos desde: {carpeta_datos}")
        total_indexado = engine.index_data_from_files(carpeta_datos)
        
        # Mostrar estadísticas finales
        print("\n" + "="*60)
        print("📊 INDEXACIÓN COMPLETADA")
        print("="*60)
        
        stats_finales = engine.get_stats()
        print(f"✅ Total de documentos: {stats_finales.get('total_documents', 0)}")
        print(f"💾 Tamaño del índice: {stats_finales.get('index_size', 0) / 1024 / 1024:.2f} MB")
        print(f"🎯 Documentos nuevos indexados: {total_indexado}")
        
        # Prueba rápida
        print("\n🧪 Ejecutando prueba rápida...")
        resultado_prueba = engine.search("test", max_results=1)
        if resultado_prueba['total'] > 0:
            print("✅ Elasticsearch funcionando correctamente")
        else:
            print("⚠️ No se encontraron resultados en la prueba")
        
        print("\n🎉 ¡Indexación completada exitosamente!")
        print("   Ahora puedes usar el agente con búsquedas de Elasticsearch.")
        
    except Exception as e:
        print(f"❌ Error durante la indexación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()