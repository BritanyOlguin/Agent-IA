"""
Agente con sistema de paginación interactiva para navegar grandes volúmenes de resultados.
Permite ver resultados de 100 en 100 con navegación siguiente/anterior.
"""

import sys
from pathlib import Path
import math

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from src.core.elasticsearch_engine import ElasticsearchEngine

class AgentePaginacion:
    """Agente con sistema de paginación para grandes volúmenes de datos"""
    
    def __init__(self):
        self.engine = None
        self.total_docs = 0
        self.resultados_por_pagina = 50
        
        # Estado de navegación
        self.ultima_consulta = ""
        self.pagina_actual = 1
        self.total_resultados = 0
        self.total_paginas = 0
        self.todos_resultados = []  # Cache de todos los resultados
        
        try:
            print("🔄 Inicializando Elasticsearch...")
            self.engine = ElasticsearchEngine()
            
            # Verificar datos
            stats = self.engine.get_stats()
            self.total_docs = stats.get('total_documents', 0)
            
            if self.total_docs == 0:
                print("⚠️ No hay datos indexados. Ejecuta: python src/utils/elasticsearch_indexer.py")
            else:
                print(f"✅ Conectado a Elasticsearch con {self.total_docs:,} registros")
                
        except Exception as e:
            print(f"❌ Error conectando a Elasticsearch: {e}")
    
    def buscar_nueva_consulta(self, consulta: str) -> bool:
        """Realiza una nueva búsqueda y prepara la paginación"""
        if not self.engine:
            print("❌ Elasticsearch no está disponible")
            return False
        
        try:
            print(f"🔍 Buscando: '{consulta}'")
            print("⏳ Obteniendo todos los resultados... (esto puede tomar unos segundos)")
            
            # Obtener TODOS los resultados de una vez (Elasticsearch maneja esto eficientemente)
            resultados = self.engine.search(consulta, max_results=10000)  # Máximo práctico
            
            if resultados['total'] == 0:
                print(f"❌ No se encontraron resultados para: '{consulta}'")
                return False
            
            # Guardar estado
            self.ultima_consulta = consulta
            self.todos_resultados = resultados['results']
            self.total_resultados = len(self.todos_resultados)
            self.total_paginas = math.ceil(self.total_resultados / self.resultados_por_pagina)
            self.pagina_actual = 1
            
            print(f"✅ Búsqueda completada en {resultados['took']}ms")
            print(f"📊 Total encontrados: {self.total_resultados:,} resultados")
            print(f"📄 Se mostrarán en {self.total_paginas} páginas de {self.resultados_por_pagina} resultados cada una")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {str(e)}")
            return False
    
    def mostrar_pagina_actual(self):
        """Muestra la página actual de resultados"""
        if not self.todos_resultados:
            print("❌ No hay resultados para mostrar")
            return
        
        # Calcular índices de la página actual
        inicio = (self.pagina_actual - 1) * self.resultados_por_pagina
        fin = min(inicio + self.resultados_por_pagina, self.total_resultados)
        
        resultados_pagina = self.todos_resultados[inicio:fin]
        
        # Mostrar header de la página
        print("\n" + "="*80)
        print(f"📄 PÁGINA {self.pagina_actual} DE {self.total_paginas}")
        print(f"🔍 Consulta: '{self.ultima_consulta}'")
        print(f"📊 Mostrando resultados {inicio + 1:,} al {fin:,} de {self.total_resultados:,} totales")
        print("="*80)
        
        # Mostrar resultados de esta página
        for i, resultado in enumerate(resultados_pagina, start=inicio + 1):
            print(f"\n📋 RESULTADO #{i:,} (Relevancia: {resultado['score']:.2f})")
            print(f"📁 Fuente: {resultado['fuente']}")
            
            # Mostrar datos del registro
            for campo, valor in resultado['data'].items():
                if campo not in ['fuente', 'fecha_indexado', 'id_original']:
                    campo_mostrar = campo.replace('_', ' ').title()
                    print(f"   {campo_mostrar}: {valor}")
        
        # Mostrar footer con navegación
        print("\n" + "="*80)
        print(f"📄 PÁGINA {self.pagina_actual} DE {self.total_paginas} | Resultados {inicio + 1:,}-{fin:,} de {self.total_resultados:,}")
        
        # Mostrar opciones de navegación
        opciones = []
        if self.pagina_actual > 1:
            opciones.append("'anterior' (página anterior)")
        if self.pagina_actual < self.total_paginas:
            opciones.append("'siguiente' (página siguiente)")
        opciones.extend([
            "'pagina X' (ir a página específica)",
            "'pregunta' (nueva búsqueda)",
            "'salir' (terminar)"
        ])
        
        print("🧭 NAVEGACIÓN:")
        for opcion in opciones:
            print(f"   • {opcion}")
        print("="*80)
    
    def navegar_anterior(self) -> bool:
        """Va a la página anterior"""
        if self.pagina_actual > 1:
            self.pagina_actual -= 1
            print(f"⬅️ Navegando a página {self.pagina_actual}")
            return True
        else:
            print("❌ Ya estás en la primera página")
            return False
    
    def navegar_siguiente(self) -> bool:
        """Va a la página siguiente"""
        if self.pagina_actual < self.total_paginas:
            self.pagina_actual += 1
            print(f"➡️ Navegando a página {self.pagina_actual}")
            return True
        else:
            print("❌ Ya estás en la última página")
            return False
    
    def ir_a_pagina(self, numero_pagina: int) -> bool:
        """Va a una página específica"""
        if 1 <= numero_pagina <= self.total_paginas:
            self.pagina_actual = numero_pagina
            print(f"🎯 Navegando a página {self.pagina_actual}")
            return True
        else:
            print(f"❌ Página inválida. Debe estar entre 1 y {self.total_paginas}")
            return False
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas del sistema y búsqueda actual"""
        if self.engine:
            stats = self.engine.get_stats()
            print(f"\n📊 ESTADÍSTICAS DEL SISTEMA:")
            print(f"   📄 Total documentos en base: {stats.get('total_documents', 0):,}")
            print(f"   💾 Tamaño del índice: {stats.get('index_size', 0) / 1024 / 1024:.2f} MB")
            print(f"   ⚡ Estado: {stats.get('status', 'desconocido')}")
            
            if self.ultima_consulta:
                print(f"\n🔍 ESTADÍSTICAS DE BÚSQUEDA ACTUAL:")
                print(f"   🎯 Consulta: '{self.ultima_consulta}'")
                print(f"   📊 Resultados encontrados: {self.total_resultados:,}")
                print(f"   📄 Total de páginas: {self.total_paginas}")
                print(f"   📍 Página actual: {self.pagina_actual}")
                print(f"   📋 Resultados por página: {self.resultados_por_pagina}")

def main():
    """Función principal"""
    print("="*80)
    print("🚀 AGENTE DE BÚSQUEDA CON PAGINACIÓN INTERACTIVA")
    print("="*80)
    
    # Crear agente
    agente = AgentePaginacion()
    
    if not agente.engine:
        print("❌ No se pudo inicializar Elasticsearch")
        return
    
    print(f"📊 Base de datos: {agente.total_docs:,} registros disponibles")
    print(f"📄 Resultados por página: {agente.resultados_por_pagina}")
    
    print("\n" + "-"*80)
    print("💬 COMANDOS DISPONIBLES:")
    print("   🔍 [consulta] - Nueva búsqueda")
    print("   ➡️ siguiente - Página siguiente")
    print("   ⬅️ anterior - Página anterior") 
    print("   🎯 pagina [número] - Ir a página específica")
    print("   💡 ayuda - Mostrar ejemplos")
    print("   📊 estadisticas - Ver estadísticas")
    print("   🚪 salir - Terminar programa")
    print("-"*80)
    
    while True:
        try:
            comando = input("\n🎮 Comando: ").strip().lower()
            
            if not comando:
                continue
            
            # Comandos de salida
            if comando in ['salir', 'exit', 'quit']:
                print("\n👋 ¡Hasta luego!")
                break
            
            # Estadísticas
            elif comando in ['estadisticas', 'stats']:
                agente.mostrar_estadisticas()
            
            # Navegación - siguiente
            elif comando in ['siguiente', 'next', 'sig']:
                if agente.navegar_siguiente():
                    agente.mostrar_pagina_actual()
                    
            # Navegación - anterior
            elif comando in ['anterior', 'prev', 'ant']:
                if agente.navegar_anterior():
                    agente.mostrar_pagina_actual()
            
            # Ir a página específica
            elif comando.startswith('pagina '):
                try:
                    numero = int(comando.split('pagina ')[1])
                    if agente.ir_a_pagina(numero):
                        agente.mostrar_pagina_actual()
                except (ValueError, IndexError):
                    print("❌ Formato incorrecto. Usa: pagina [número]")
            
            # Nueva búsqueda
            else:
                # Tratar como nueva consulta
                if agente.buscar_nueva_consulta(comando):
                    agente.mostrar_pagina_actual()
            
        except KeyboardInterrupt:
            print("\n\n👋 Saliendo...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()