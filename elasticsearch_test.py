"""
Script para verificar que Elasticsearch esté funcionando correctamente.
Ejecuta este script antes de usar el agente.
"""

import requests
import sys
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

def verificar_elasticsearch():
    """Verifica que Elasticsearch esté ejecutándose"""
    print("🔍 Verificando Elasticsearch...")
    
    try:
        # Verificar si Elasticsearch responde
        response = requests.get('http://localhost:9200', timeout=5, verify=False)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Elasticsearch está ejecutándose")
            print(f"   Versión: {info.get('version', {}).get('number', 'Desconocida')}")
            return True
        else:
            print(f"❌ Elasticsearch respondió con código: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar a Elasticsearch en http://localhost:9200")
        print("   Asegúrate de que esté ejecutándose.")
        return False
    except Exception as e:
        print(f"❌ Error verificando Elasticsearch: {e}")
        return False

def verificar_modulo():
    """Verifica que el módulo de Elasticsearch funcione"""
    print("\n🔍 Verificando módulo de Python...")
    
    try:
        from src.core.elasticsearch_engine import ElasticsearchEngine
        
        engine = ElasticsearchEngine()
        stats = engine.get_stats()
        
        print(f"✅ Módulo funcionando correctamente")
        print(f"   Documentos indexados: {stats.get('total_documents', 0)}")
        print(f"   Tamaño del índice: {stats.get('index_size', 0) / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en módulo: {e}")
        return False

def prueba_busqueda():
    """Prueba una búsqueda simple"""
    print("\n🧪 Ejecutando prueba de búsqueda...")
    
    try:
        from src.core.elasticsearch_engine import ElasticsearchEngine
        
        engine = ElasticsearchEngine()
        
        # Búsqueda de prueba
        resultado = engine.search("test", max_results=1)
        
        if resultado['total'] > 0:
            print(f"✅ Prueba exitosa: {resultado['total']} resultados encontrados")
        else:
            print("⚠️ No se encontraron resultados (normal si no hay datos indexados)")
        
        print(f"   Tiempo de búsqueda: {resultado.get('took', 0)}ms")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def main():
    """Función principal de verificación"""
    print("="*60)
    print("🔧 VERIFICACIÓN DE ELASTICSEARCH")
    print("="*60)
    
    # Paso 1: Verificar servidor
    if not verificar_elasticsearch():
        print("\n💡 SOLUCIÓN:")
        print("   1. Descarga Elasticsearch de: https://www.elastic.co/downloads/elasticsearch")
        print("   2. Extrae el archivo ZIP")
        print("   3. Ejecuta: elasticsearch-x.x.x\\bin\\elasticsearch.bat")
        print("   4. Espera a que aparezca: 'started'")
        print("   5. Vuelve a ejecutar este script")
        return False
    
    # Paso 2: Verificar módulo
    if not verificar_modulo():
        print("\n💡 SOLUCIÓN:")
        print("   1. Ejecuta: pip install elasticsearch elasticsearch-dsl")
        print("   2. Verifica que los archivos estén en su lugar")
        print("   3. Vuelve a ejecutar este script")
        return False
    
    # Paso 3: Prueba de búsqueda
    if not prueba_busqueda():
        print("\n💡 SOLUCIÓN:")
        print("   1. Ejecuta el indexador: python src/utils/elasticsearch_indexer.py")
        print("   2. Vuelve a ejecutar este script")
        return False
    
    print("\n" + "="*60)
    print("🎉 ¡ELASTICSEARCH ESTÁ LISTO!")
    print("="*60)
    print("✅ Todas las verificaciones pasaron")
    print("🚀 Puedes ejecutar el agente con: python src/Agente.py")
    
    return True

if __name__ == "__main__":
    main()