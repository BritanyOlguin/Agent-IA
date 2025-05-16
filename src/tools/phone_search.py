from src.utils.text_normalizer import normalizar_texto, convertir_a_mayusculas
from src.core.engine import indices, all_tools
from llama_index.core.tools import FunctionTool

# --- 6) HERRAMIENTA 4: BUSCAR POR NUMERO TELEFONICO ---
def buscar_numero_telefono(valor: str) -> str:
    """
    Búsqueda tolerante para teléfonos, lada o combinaciones incompletas.
    Evita duplicados entre campos como 'telefono', 'lada' y 'telefono_completo'.
    """
    campos_telefono = ["telefono_completo", "telefono", "lada"]
    valor_norm = normalizar_texto(valor)
    resultados = {}
    
    for fuente, index in indices.items():
        for campo in campos_telefono:
            try:
                for node_id, doc in index.docstore.docs.items():
                    metadata = doc.metadata
                    if not metadata or campo not in metadata:
                        continue

                    valor_campo = str(metadata.get(campo, "")).strip()
                    if not valor_campo:
                        continue

                    valor_campo_norm = normalizar_texto(valor_campo)

                    if valor_norm in valor_campo_norm or valor_campo_norm.endswith(valor_norm) or valor_campo_norm.startswith(valor_norm):
                        score = 1.0 if valor_norm == valor_campo_norm else \
                                0.9 if valor_campo_norm.endswith(valor_norm) else \
                                0.8 if valor_campo_norm.startswith(valor_norm) else \
                                0.5
                        
                        id_registro = str(metadata.get("id", "")) + str(metadata.get("fila_origen", "")) + str(metadata.get("nombre_completo", "")).strip().lower()
                        if not id_registro:
                            id_registro = doc.node.node_id
                        
                        # SOLO GUARDAMOS EL MEJOR RESULTADO POR REGISTRO
                        if id_registro not in resultados or resultados[id_registro]['score'] < score:
                            resumen = [f"{k}: {v}" for k, v in metadata.items() if k not in ['fuente', 'archivo', 'fila_excel'] and v]
                            resultados[id_registro] = {
                                'score': score,
                                'fuente': fuente,
                                'campo': campo,
                                'texto': f"Coincidencia en {fuente} (campo '{campo}'):\n" + "\n".join(resumen)
                            }
            except Exception as e:
                print(f"[WARN] Error revisando {fuente} campo {campo}: {e}")
                continue

    if not resultados:
        return f"No se encontraron coincidencias relevantes para el número '{valor}'."

    resultados_ordenados = sorted(resultados.values(), key=lambda x: -x['score'])
    return convertir_a_mayusculas("Se encontraron las siguientes coincidencias para número telefónico:\n\n" + "\n\n".join([r['texto'] for r in resultados_ordenados]))

buscar_telefono_tool = FunctionTool.from_defaults(
    fn=buscar_numero_telefono,
    name="buscar_numero_telefono",
    description=(
        "Usa esta herramienta cuando el campo detectado sea 'telefono_completo' y el usuario consulta por un número telefónico completo. "
        "Ejemplo: '¿Quién tiene el número 5544332211?', pero NO para lada o partes de teléfonos."
    )
)

all_tools.insert(4, buscar_telefono_tool)