from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# 2) Crea la función de embeddings con la ruta local
embedding_function = HuggingFaceEmbeddings(
    #model_name=r"C:\Users\Monitoreo\Desktop\oki\src\DeepSeek-R1-Distill-Qwen-1.5B",
    model_name=r"C:\Users\Sistemas\Documents\OKIP\src\models--sentence-transformers--all-MiniLM-L6-v2",
    #model_name=r"C:\Users\Monitoreo\Desktop\oki\src\DeepSeek-R1-Distill-Qwen-14B",

    model_kwargs={"local_files_only": True}
)

# 3) Carga la base vectorial Chroma (ya creada/persistida)
vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="chroma_db"
)

# 4) Haz la consulta
pregunta_usuario = "Información sobre Valeria Rivera Rivera"
resultados = vectorstore.similarity_search(pregunta_usuario, k=3)

for idx, doc in enumerate(resultados):
    print(f"Resultado {idx+1}:\n{doc.page_content}\n")
