# Código del proyecto

Scripts de Python desarrollados para la detección de retinopatía diabética mediante transfer learning.

## Descripción de los scripts

- **00_analisis_dataset.py** — Análisis exploratorio del dataset: distribución de clases, frecuencia de signos clínicos y tamaños originales de las imágenes.
- **01_preprocesamiento.py** — Carga, redimensionado a 224×224 píxeles y normalización de las imágenes.
- **02_dataset_splits.py** — División del dataset en entrenamiento (70%), validación (15%) y prueba (15%), y configuración del data augmentation.
- **03_modelos.py** — Construcción de las tres arquitecturas (ResNet50, DenseNet121, InceptionV3) con transfer learning.
- **04_entrenamiento.py** — Entrenamiento en dos fases y comparación de modelos.
- **05_evaluacion.py** — Evaluación final con análisis de umbrales y métricas por signo clínico.
- **requirements.txt** — Librerías necesarias para ejecutar el código.

## Ejecución

Los scripts deben ejecutarse en orden del 00 al 05. Se recomienda usar un entorno virtual con las dependencias del requirements.txt.
