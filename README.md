# Taller Hugging Face: fine-tuning de un LLM pequeño con LoRA

Notebook de una prueba de concepto: cargar un modelo pequeño, probarlo, cuantizarlo, adaptarlo con LoRA y volver a probarlo. Todo con librerías de Hugging Face (`transformers`, `datasets`, `peft`, `bitsandbytes`).

Con el notebook puedes:

- Probar el modelo base con preguntas de cafeterías (antes).
- Cuantizar el modelo a 4 bits (solo con GPU NVIDIA).
- Cargar un adaptador LoRA ya entrenado desde el Hub, o entrenar uno nuevo.
- Medir si el modelo ha aprendido los precios (antes y después).
- Chatear con el modelo en una demo interactiva.

## Contenido

- `LoRA_cafeteria_uam.ipynb`: el notebook.

## Qué es LoRA

Un modelo grande tiene miles de millones de pesos. Ajustarlo entero es caro. LoRA (_Low-Rank Adaptation_) congela todos los pesos originales y añade a algunas capas dos matrices pequeñas, `A` y `B`:

`W' = W + (alpha / r) · B · A`

Solo se entrenan `A` y `B`, que suelen ser menos del 1 % de los parámetros. El adaptador resultante pesa unos pocos MB, frente a los GB del modelo base. Si además el modelo base está cuantizado a 4 bits, la técnica se llama **QLoRA**.

## Modelos y datasets publicados

**Cafeterías UAM (Qwen2.5-0.5B)**

- Modelo: [Bellido/qwen2.5-0.5b-cafeterias-uam-lora](https://huggingface.co/Bellido/qwen2.5-0.5b-cafeterias-uam-lora)
- Dataset: [Bellido/cafeterias_uam_2026](https://huggingface.co/datasets/Bellido/cafeterias_uam_2026), 1000 preguntas y respuestas sobre productos y precios de las cafeterías de la UAM.
- Modelo pequeño (0,5B) entrenado para responder precios. Es el que usa este notebook por defecto.

**Transporte a Cantoblanco (Qwen2.5-1.5B)**

- Modelo: [Bellido/qwen2.5-1.5b-cantoblanco-transporte-en-lora](https://huggingface.co/Bellido/qwen2.5-1.5b-cantoblanco-transporte-en-lora)
- Dataset: [Bellido/cantoblanco-transporte-en](https://huggingface.co/datasets/Bellido/cantoblanco-transporte-en), preguntas y respuestas sobre cómo llegar al campus de Cantoblanco en Cercanías y autobús.
- Modelo algo más grande (1,5B), entrenado con datos de transporte. Su validación ha sido manual, sin métricas automáticas.

## Cómo ejecutarlo

**Google Colab:** abre el notebook en Colab, activa una GPU (T4) en _Entorno de ejecución_ y ejecuta las celdas en orden. Las dependencias se instalan en la primera celda.

**Local:** instala las dependencias (`pip install -U transformers torch datasets peft accelerate pandas bitsandbytes`) y ejecuta el notebook con Jupyter.

## Configuración

Las variables principales están en la celda de configuración:

| Variable       | Qué hace                                                                               |
| -------------- | -------------------------------------------------------------------------------------- |
| `MODEL_ID`     | Modelo base de Hugging Face (por defecto `Qwen/Qwen2.5-0.5B-Instruct`).                |
| `DATASET_ID`   | Dataset de preguntas y respuestas.                                                     |
| `ADAPTER_REPO` | Repo del adaptador LoRA en el Hub que se carga cuando no entrenas.                     |
| `ENTRENAR`     | `False`: carga el adaptador del Hub, sin entrenar. `True`: entrena un adaptador nuevo. |
| `EPOCHS`       | Número de épocas, solo si `ENTRENAR = True`.                                           |

Para usar el segundo modelo de transporte, cambia `MODEL_ID`, `DATASET_ID` y `ADAPTER_REPO` por los valores de la sección de modelos.

## Flujo del notebook

1. Detectar el hardware (GPU NVIDIA, Mac o CPU).
2. Cargar el dataset desde el Hub y separarlo en train (95 %) y eval (5 %).
3. Probar el modelo base con preguntas de ejemplo.
4. Cuantizar a 4 bits y comprobar que sigue respondiendo.
5. Aplicar LoRA: cargar el adaptador del Hub, o crear uno nuevo si `ENTRENAR = True`.
6. Entrenar (solo si `ENTRENAR = True`).
7. Comparar antes y después, y medir la precisión de precios.
8. Guardar el adaptador en local (solo si se ha entrenado).
9. Demo interactiva: escribe preguntas y mira las respuestas.

## Métrica de precios

Para el dataset de cafeterías, la métrica cuenta la fracción de respuestas que contienen todos los precios de la respuesta correcta (formato `1,05`). Se calcula sobre 20 preguntas de train y 20 de eval.
