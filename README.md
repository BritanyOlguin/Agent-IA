# Obtención de datos (Back)

## 📌 Descripción
Este proyecto tiene la finalidad de la busqueda rapida de información de ciudadanos de SMA, mediante el filtrado de los datos de una base de datos.

## 🚀 Instalación

1. Crea entorno virtual
```sh
py -m venv venv
```
2. Activa entorno virtual
```sh
.\venv\Scripts\activate
```
3. Instalar las dependencias
```sh
pip install -r requirements.txt
pip install pyodbc
pip install dbfread

```
3.1 Si sale este error utiliza...

ERROR: Ignored the following versions that require a different python version: 0.5.12 Requires-Python >=3.7,<3.12; 0.5.13 Requires-Python >=3.7,<3.12; 0.5.14 Requires-Python >=3.7,<3.13; 1.10.0 Requires-Python <3.12,>=3.8; 1.10.0rc1 Requires-Python <3.12,>=3.8; 1.10.0rc2 Requires-Python <3.12,>=3.8; 1.10.1 Requires-Python <3.12,>=3.8; 1.11.0 Requires-Python <3.13,>=3.9; 1.11.0rc1 Requires-Python <3.13,>=3.9; 1.11.0rc2 Requires-Python <3.13,>=3.9; 1.11.1 Requires-Python <3.13,>=3.9; 1.11.2 Requires-Python <3.13,>=3.9; 1.11.3 Requires-Python <3.13,>=3.9; 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11; 1.26.0 Requires-Python <3.13,>=3.9; 1.26.1 Requires-Python <3.13,>=3.9; 1.6.2 Requires-Python >=3.7,<3.10; 1.6.3 Requires-Python >=3.7,<3.10; 1.7.0 Requires-Python >=3.7,<3.10; 1.7.1 Requires-Python >=3.7,<3.10; 1.7.2 Requires-Python >=3.7,<3.11; 1.7.3 Requires-Python >=3.7,<3.11; 1.8.0 Requires-Python >=3.8,<3.11; 1.8.0rc1 Requires-Python >=3.8,<3.11; 1.8.0rc2 Requires-Python >=3.8,<3.11; 1.8.0rc3 Requires-Python >=3.8,<3.11; 1.8.0rc4 Requires-Python >=3.8,<3.11; 1.8.1 Requires-Python >=3.8,<3.11; 1.9.0 Requires-Python >=3.8,<3.12; 1.9.0rc1 Requires-Python >=3.8,<3.12; 1.9.0rc2 Requires-Python >=3.8,<3.12; 1.9.0rc3 Requires-Python >=3.8,<3.12; 1.9.1 Requires-Python >=3.8,<3.12
ERROR: Could not find a version that satisfies the requirement torch==2.6.0+cu118 (from versions: 2.6.0)

```sh
python.exe -m pip install --upgrade pip
```

4. Mover los archivos a src
5. Instalar
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
5. Ejecutar el archivo

venv\src\utils\model_download.py

Cambiar la ruta si es necesario

6. Descomentar el primer modelo, ejecutar archivo, descomentar el segundo y comentar el primero y asi sucesivamente.

7. Instalar
```sh
pip install langchain-community

pip install openpyxl

pip install sentence-transformers
```

8. Borrar todo excepto los archivos js y ponerlos en la carpeta raiz del modelo all mini LM

9. Cambiar rutas

11. Instalar LlamaIndex
```sh
pip install llama-index sentence-transformers transformers pandas openpyxl
pip install llama-index pandas openpyxl sentence-transformers
pip install llama-index-embeddings-huggingface
pip install llama-index==0.12.26
pip install llama-index-cli llama-index-agent-openai llama-index-llms-openai llama-index-readers-file
pip install llama-index-llms-huggingface
pip install -U bitsandbytes
```

12. Instalar e5-large-v2
```sh
pip install transformers sentence-transformers huggingface-hub
```

13. Correr este comando, tienes que estar loggeado en hugginface
```sh
huggingface-cli login
huggingface-cli download intfloat/e5-large-v2 --local-dir "models\base\models--intfloat--e5-large-v2" --local-dir-use-symlinks False
```

ARCHIVO LLAMA_BD
1. Instalar
```sh
pip install llama-index fastapi uvicorn pandas openpyxl "llama-cpp-python[server]" python-dotenv python-multipart Jinja2
pip install llama-index-llms-llama-cpp
pip install llama-index-experimental
```

2. Configuraciones para el GPU del equipo
```sh
pip uninstall llama-cpp-python
Remove-Variable CMAKE_ARGS -ErrorAction SilentlyContinue
Remove-Variable FORCE_CMAKE -ErrorAction SilentlyContinue
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
pip install llama-cpp-python --no-cache-dir
```

3. Implementaciones para fine_tuning
```sh
pip install transformers==4.36.2 datasets==2.14.6 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.1 trl==0.7.4 scipy==1.11.4
pip install datasets
pip install peft
pip install trl
pip install --upgrade accelerate
```

## Instalar Elasticsearch
1. Correr comandos en terminal
```sh
pip install elasticsearch
pip install elasticsearch-dsl
pip install unidecode
```

2. Descargar e Instalar Elasticsearch
* Ve a: https://www.elastic.co/downloads/elasticsearch
* Descarga la versión para Windows (ZIP)
* Extrae en una carpeta como C:\elasticsearch-8.x.x
* Ejecuta: C:\elasticsearch-8.x.x\bin\elasticsearch.bat
* Verifica abriendo https://localhost:9200 en tu navegador

¡IMPORTANTE GUARDAR CONTRASEÑA Y MANTENER ABIERTO LA PANTALLA BASH!

## Deshabilitar Seguridad en Elasticsearch
1. Buscar el archivo de configuración:
config\elasticsearch.yml
2. Abrir el archivo elasticsearch.yml con un editor de texto, agregar estas líneas al final del archivo
```sh
# ======================== Elasticsearch Configuration =========================
# Configuración simple para desarrollo - SIN seguridad

# Cluster
cluster.name: agente-ia-cluster
node.name: node-1

# Network
network.host: localhost
http.host: 0.0.0.0
http.port: 9200

# Discovery
discovery.type: single-node

# Security - DESHABILITADA para desarrollo
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
xpack.security.http.ssl.enabled: false
xpack.security.transport.ssl.enabled: false

# Logging
cluster.routing.allocation.disk.threshold_enabled: false
```
3. Guardar el archivo

## Iniciar proyecto
1. Correr testeo de conexion con elastics
```sh
python elasticsearch_test.py
```

2. Correr indexado de datos
```sh
python src/utils/elasticsearch_indexer.py
```

3. Correr el agente
```sh
python -m src.Agente
```

## ENTRENAMIENTO DE DATOS

1. Ejecutar el agente_entrenamiento.py para retroalimentar al agente

```sh
python -m src.Agente_entrenamiento
```

2. Ejecutar el analyzer.py

```sh
python -m src.feedback.analyzer
```

3. Generar los ejemplos ejecutando el collector.py

```sh
python -m src.feedback.collector
```

4. Ejecuta data_generator.py

```sh
python -m src.training.data_generator
```

5. Entrena el modelo con el siguiente comando

```sh
python .\src\training\model_trainer.py --epochs 5 --batch_size 2 --lr 2e-4
```

6. Actualiza la ruta del modelo en Agente.py
```sh
ruta_tus_adaptadores_lora = r"C:\Users\TEC-INT02\Documents\Agent-IA\models\fine_tuned\llama3-8b-agente-consulta-YYYYMMDD_HHMM"
```
7. Prueba la inteligencia de tu nuevo agente
```sh
python C:\Users\TEC-INT02\Documents\Agent-IA\src\Agente.py
python -m src.Agente
```