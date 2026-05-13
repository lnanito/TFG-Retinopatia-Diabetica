# TFG - Detección de retinopatía diabética con IA

Trabajo de Fin de Grado centrado en la detección de retinopatía diabética y sus principales signos clínicos mediante modelos de deep learning aplicados a imágenes de fondo de ojo.

**Autora:** Elena Sanjuan Muñoz  
**Universidad:** Universidad Europea de Valencia  
**Curso:** 2025-2026

---

## Descripción

Este trabajo compara tres arquitecturas de redes neuronales convolucionales (ResNet50, DenseNet121 e InceptionV3) mediante transfer learning para detectar la presencia de retinopatía diabética y sus principales signos clínicos en imágenes de fondo de ojo.

A diferencia de otros enfoques, el modelo no solo clasifica si una imagen tiene o no retinopatía, sino que identifica de forma independiente seis signos clínicos: microaneurismas, hemorragias, exudados, neovasos, hemovitreo/opacidad de medios y láser.

---

## Resultados principales

- **Modelo ganador:** InceptionV3
- **Accuracy:** 89%
- **AUC:** 0.91
- **F1-score:** 0.66 (umbral óptimo 0.2)
- **Sensibilidad del 100%** en microaneurismas y hemorragias

---

## Estructura del repositorio

- `src/` — scripts de Python del proyecto
- `memoria/` — memoria del TFG en PDF

---

## Dataset

Las imágenes utilizadas fueron proporcionadas por la Fundación Oftalmológica del Mediterráneo (FOM) y no se incluyen en este repositorio por razones de privacidad y confidencialidad clínica.

---

## Requisitos

Ver `src/requirements.txt` para las dependencias necesarias.
