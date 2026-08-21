# Documento Fuente Original

> **Nota:** Este documento es la fuente original del proyecto (texto generado con Gemini).
> **Solo lectura** — no modificar. El documento vivo de seguimiento es [`MEMORIA.md`](../MEMORIA.md).
> Fecha de incorporación: 21/08/2026.

---

Título del Proyecto
DefectNet-Vision: Sistema de Inspección Visual Automática de Defectos en Superficies Industriales mediante Segmentación Semántica

## 1. El Problema Industrial y Valor de Negocio

En las líneas de producción industrial, una micro-fisura o un arañazo en un componente metálico o electrónico puede causar el fallo total del producto final.

- **Problema:** La inspección manual es lenta, costosa y propensa a la fatiga humana.
- **Solución:** Un sistema de Deep Learning basado en U-Net que inspecciona imágenes de piezas en tiempo real, segmenta la ubicación exacta y la forma del defecto a nivel de píxel, y emite un diagnóstico instantáneo: PIEZA ACEPTADA (PASS) o PIEZA DEFECTUOSA (FAIL).

## 2. Dataset Recomendado

Para este tipo de proyectos existen dos estándares de la industria que puedes utilizar:

- **DAGM 2007 (Recomendado para empezar):** Dataset sintético/realista muy popular de superficies industriales con textura donde se generan pequeños defectos visuales.
- **Kolektor Surface-Defect Dataset (KolektorSDD):** Un dataset real de la industria electrónica con imágenes de micro-fisuras en superficies plásticas/metálicas.
- **MVTec AD (MVTec Anomaly Detection):** El estándar internacional más avanzado de inspección de superficies industriales.

> **Recomendación:** Utiliza KolektorSDD o DAGM ya que incluyen máscaras de segmentación binarias muy bien definidas.

## 3. Arquitectura y Stack Técnico

- **Lenguaje & DL Framework:** Python 3.10+, PyTorch, PyTorch Lightning.
- **Librerías de Visión / Segmentación:** segmentation_models_pytorch (para arquitecturas U-Net con backbones preentrenados), OpenCV, PIL.
- **Augmentations:** Albumentations (crucial para simular variaciones de iluminación de la cámara en fábrica).
- **Métricas y Pérdida:** Combo Loss (Focal Loss + Dice Loss), mIoU (Mean Intersection over Union), F1-Score, Latencia de Inferencia (ms).
- **Tracking y Logs:** TensorBoard o Weights & Biases (W&B).
- **Despliegue & App:** Streamlit (para el dashboard interactivo) y Docker.

## 4. Estructura del Repositorio en GitHub

Una estructura limpia de código es fundamental para impresionar a reclutadores y líderes técnicos:

```
defectnet-vision/
│
├── data/                       # (Ignorado en .gitignore) Imágenes de entrenamiento/test
├── src/
│   ├── dataset.py              # Custom PyTorch Dataset + Pipelines de Albumentations
│   ├── model.py                # Definición de U-Net / FPN y Loss Functions
│   ├── train.py                # Loop de entrenamiento y validación
│   ├── evaluate.py             # Evaluación de mIoU, Precision/Recall y latencia
│   └── utils.py                # Funciones de post-procesado morfológico
│
├── app/
│   └── app.py                  # Interfaz Streamlit para control de calidad
│
├── notebooks/                  # EDA y experimentos iniciales (limpios)
├── Dockerfile                  # Contenedorización del servicio
├── requirements.txt
└── README.md                   # Presentación visual del proyecto
```

## 5. Fases de Implementación Paso a Paso

### Fase 1: Preparación del Dataset y Data Augmentation Avanzado

- **Pérdida por Desequilibrio:** En defectos de superficies, el 98% de los píxeles son "superficie sana" y solo el 2% son "defecto".
- **Pipelines de Augmentation con Albumentations:**
  - Rotaciones aleatorias (RandomRotate90, Flip).
  - Variaciones de iluminación/contraste (RandomBrightnessContrast, RandomGamma) para simular sombras en la línea de montaje.
  - Modificación de ruido (GaussNoise) para simular grano de cámaras industriales.

### Fase 2: Modelado y Loss Functions Especializadas

No debes usar solo CrossEntropyLoss o BCEWithLogitsLoss debido al desequilibrio de píxeles.

- **Focal Loss:** Reduce la importancia de los píxeles fáciles (superficie sana) y enfoca el gradiente en los píxeles difíciles (bordes del defecto).
- **Dice Loss:** Maximiza directamente la superposición entre la máscara real y la predicha.
- **Loss Combinada:**

$$\text{Loss}_{\text{Total}} = \alpha \cdot \text{FocalLoss} + \beta \cdot \text{DiceLoss}$$

- **Modelos a Probar:**
  - U-Net con encoder ResNet18 o EfficientNet-B0 (ligeros para alta velocidad de inferencia).
  - FPN (Feature Pyramid Network) como alternativa rápida.

### Fase 3: Post-procesado Morfológico e Inferencia Industrial

Una vez que el modelo devuelve la probabilidad por píxel:

1. Aplica un umbral de decisión (thresholding).
2. Usa operaciones morfológicas de OpenCV (cv2.morphologyEx con apertura/cierre) para eliminar pequeños píxeles de ruido aislados y rellenar lagunas en el defecto.
3. **Métrica de Negocio (Lógica de Decisión):**
   - Si el área total del defecto predicho supera un umbral $X$ de píxeles → FAIL (Defectuosa).
   - Si es inferior → PASS (Conforme).
4. **Medición de latencia:** Calcular cuántos milisegundos toma procesar una imagen y traducirlo a FPS (Frames Per Second).

### Fase 4: Dashboard Interactivo para Control de Calidad (Streamlit)

Crea una interfaz intuitiva donde un operario o ingeniero de calidad pueda:

1. Cargar una imagen de una pieza industrial.
2. Ver la imagen original, la máscara binaria predicha y el Overlay (superposición del defecto en rojo transparente).
3. Recibir el veredicto automático: Cartel grande en verde PASS o en rojo DEFECT DETECTED.
4. Mostrar métricas en tiempo real: Tiempo de inferencia (ms), área ocupada por el defecto (%) y nivel de confianza.
