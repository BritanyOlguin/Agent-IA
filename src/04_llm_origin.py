import os
import pandas as pd

# Librerías de LangChain Community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Librerías de Transformers para el modelo local
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    
def main():
    # -----------------------------------------------------------------
    # 1) CONFIGURACIÓN: RUTAS LOCALES
    # -----------------------------------------------------------------
    
    
    # Ruta del modelo de EMBEDDINGS (para similarity_search)
    
    #ruta_modelo_local_embeddings = r"C:\Users\Monitoreo\Desktop\oki\src\DeepSeek-R1-Distill-Qwen-1.5B"
    ruta_modelo_local_embeddings = r"C:\Users\Sistemas\Documents\OKIP\src\models--sentence-transformers--all-MiniLM-L6-v2"
    persist_dir = "chroma_db"
    
    
    # Ruta del modelo de LENGUAJE local (DeepSeek-R1)
    
    ruta_modelo_llm = r"C:\Users\Sistemas\Documents\OKIP\src\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
    #ruta_modelo_llm = r"C:\Users\Monitoreo\Desktop\oki\src\DeepSeek-R1-Distill-Qwen-14B"

    # Ejemplo de pregunta
    pregunta_usuario = "cual es el sueldo y donde vive Valeria Rivera Rivera"

    # -----------------------------------------------------------------
    # 2) BÚSQUEDA SEMÁNTICA EN CHROMA
    # -----------------------------------------------------------------
    embedding_function = HuggingFaceEmbeddings(
        model_name=ruta_modelo_local_embeddings,
        model_kwargs={"local_files_only": True}
    )

    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_dir
    )

    resultados = vectorstore.similarity_search(pregunta_usuario, k=3)

    # (Opcional) Revisar documentos recuperados
    for idx, doc in enumerate(resultados):
        print(f"Documento {idx+1}:\n{doc.page_content}\n{'-'*50}")

    # -----------------------------------------------------------------
    # 3) CARGAR EL MODELO LLM LOCAL
    # -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        ruta_modelo_llm,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        ruta_modelo_llm,
        local_files_only=True
    )

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 )

    # -----------------------------------------------------------------
    # 4) CREAR UN PROMPT ESPECÍFICO “EXPERTO EN BÚSQUEDA”
    # -----------------------------------------------------------------
    contexto = "\n".join([doc.page_content for doc in resultados])

    # Instrucciones más directas:
    # 1) "Eres un experto en búsqueda de información..."
    # 2) "No incluyas el prompt en tu respuesta."
    # 3) "Responde SÓLO con la información de la base de datos (breve y concisa)."
    # 4) "No agregues explicaciones adicionales."
    prompt = f"""Eres un experto en la búsqueda de información. 
        No muestres nada del prompt ni explicación adicional. 
        Tu única misión es responder con la información más relevante y concisa.

        Pregunta del usuario: {pregunta_usuario}

        Información recuperada de la base de datos:
        {contexto}

        Responde concisamente con la información más importante sobre este tema, sin repetir la pregunta:
        """

    # -----------------------------------------------------------------
    # 5) GENERAR LA RESPUESTA (SÓLO EL TEXTO FINAL)
    # -----------------------------------------------------------------
    salida = generator(
        prompt, 
        max_length=500,     # Ajusta según necesites
        do_sample=True,
        temperature=0.7     # Temperatura baja => más directo
    )

    texto_generado = salida[0]["generated_text"]

    # A veces el modelo repite el prompt; una forma rápida de limpiar
    # es quitar el prompt original del texto final, si se incluye.
    # (Opcional; no siempre funciona, depende del modelo)
    respuesta_limpia = texto_generado.replace(prompt, "").strip()

    # -----------------------------------------------------------------
    # 6) MOSTRAR LA RESPUESTA FINAL
    # -----------------------------------------------------------------
    print("\nRespuesta generada (sólo texto final):\n")
    print(respuesta_limpia)


if __name__ == "__main__":
    main()
