# Evaluación 3 · TLY1102 — Pipeline Integrado de Deep Learning

Clasificación de letras estáticas del alfabeto dactilológico ASL (*Sign Language MNIST*, variante de 10 clases:
A, B, C, D, E, I, L, M, N, O) comparando una **MLP** y una **CNN** optimizadas y auditadas.

**Integrantes:** Jeremy Concha · Bryan Jara · Eduardo Paredes
**Docente:** Jazna Meza · Duoc UC · Noviembre 2026

## Contenido

| Ruta | Descripción |
|---|---|
| `notebooks/TLY1102_E3_Pipeline.ipynb` | Notebook principal: pipeline completo, experimentos, auditoría y conclusiones |
| `presentacion/TLY1102_E3_Presentacion.pptx` | Presentación para la defensa |
| `data/raw/` | Dataset original (`sign_mnist_train.csv`, `sign_mnist_test.csv`) |
| `outputs/` | Registro de experimentos e historiales de entrenamiento |
| `models/` | Modelos finales entrenados (MLP y CNN, 3 semillas) |

## Pipeline

Carga de datos → análisis exploratorio → preprocesamiento (partición por grupos de casi-duplicados, normalización,
data augmentation) → construcción del modelo → entrenamiento → regularización → evaluación → auditoría (Grad-CAM,
LIME, robustez, sesgos) → optimización de hiperparámetros.

## Resultados principales

| | MLP | CNN |
|---|---|---|
| F1 macro, validación agrupada | 0,818 | 0,996 |
| Accuracy, test oficial | 0,873 | 0,9998 |
| F1 con desplazamiento de 4 px | 0,24 | 0,93 |
| Parámetros | 407.050 | 225.706 |

La CNN es la arquitectura recomendada. El análisis del notebook (sección C.2) muestra que el test oficial comparte
fotos originales con el conjunto de entrenamiento, por lo que su resultado no mide la generalización a personas nuevas.

## Ejecución en Google Colab

```python
!git clone https://github.com/JeremyDuoc/TAML-E3 /content/TLY1102-E3
```

Luego abrir `/content/TLY1102-E3/notebooks/TLY1102_E3_Pipeline.ipynb` y ejecutar todas las celdas. El notebook
reutiliza los experimentos de `outputs/` y los modelos de `models/`, por lo que se ejecuta en pocos minutos. Para
reentrenar todo desde cero basta con eliminar `outputs/experimentos.csv`, `outputs/historiales/` y `models/`.

## Ejecución local

```bash
pip install -r requirements.txt
jupyter notebook notebooks/TLY1102_E3_Pipeline.ipynb
```
