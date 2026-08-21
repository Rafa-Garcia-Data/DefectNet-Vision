# DefectNet-Vision — Memoria del Proyecto

> **Sistema de inspección visual automática de defectos en superficies industriales mediante segmentación semántica**

| Campo | Valor |
|---|---|
| **Estado** | 🟡 Fase 0 — Planificación |
| **Fecha de inicio** | 21/08/2026 |
| **Última actualización** | 21/08/2026 |
| **Tipo de proyecto** | Portafolio Data Scientist (GitHub) |
| **Fuente original** | [`docs/proyecto_original.md`](docs/proyecto_original.md) |

---

## Modo de trabajo

Este proyecto se desarrolla de forma **colaborativa y recíproca** entre el autor y el asistente IA:

- Se explica cada concepto antes de escribir código (nada de código a ciegas).
- El autor es Data Scientist con base en Machine Learning, pero principiante en redes neuronales y visión artificial → las explicaciones parten de ese nivel.
- Las dudas técnicas se resuelven en el momento; los conceptos clave que surjan se pueden añadir a esta memoria.
- Cada sesión de trabajo queda registrada en la sección [Registro de sesiones](#10-registro-de-sesiones-changelog).

---

## 1. Descripción y valor de negocio

**Problema:** En líneas de producción industrial, una micro-fisura o arañazo en un componente metálico o electrónico puede causar el fallo total del producto final. La inspección manual es lenta, costosa y propensa a fatiga humana.

**Solución:** Sistema de Deep Learning basado en **U-Net** que inspecciona imágenes de piezas industriales, segmenta la ubicación y forma exacta del defecto **a nivel de píxel**, y emite un diagnóstico instantáneo:

- ✅ **PASS** — Pieza aceptada
- ❌ **FAIL** — Pieza defectuosa

**Valor de negocio:** automatización del control de calidad, reducción de costes de inspección manual, detección consistente y trazable.

---

## 2. Objetivos

### Objetivo general
Construir un pipeline completo de segmentación semántica de defectos industriales (datos → modelo → inferencia → dashboard), documentado como proyecto de portafolio profesional.

### Objetivos específicos
1. Entrenar un modelo U-Net (y comparar con FPN) sobre el dataset DAGM 2007.
2. Manejar el desequilibrio extremo de píxeles (~98% superficie sana / ~2% defecto) con Combo Loss.
3. Implementar post-procesado morfológico y lógica de decisión PASS/FAIL por área de defecto.
4. Medir métricas técnicas (mIoU, F1) y de negocio (latencia ms → FPS).
5. Desplegar un dashboard interactivo en Streamlit para control de calidad.
6. Contenerizar con Docker y presentar con un README visual atractivo.

---

## 3. Dataset

### Decisión: **DAGM 2007** ✅

Dataset sintético/realista de superficies industriales con textura, estándar en inspección óptica industrial.

| Característica | Detalle |
|---|---|
| Categorías | 10 clases de texturas distintas |
| Imágenes | ~1.000 train + ~1.000 test por categoría (la mitad aprox. contienen defectos) — *cifras a confirmar al descargar* |
| Anotación | Máscaras de segmentación **binarias** por píxel para las imágenes defectuosas |
| Reto principal | Desequilibrio extremo de píxeles: el defecto ocupa una fracción mínima de la imagen |

**Enlaces de descarga** (verificar al iniciar Fase 1):
- Oficial (Univ. Heidelberg): https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection
- Espejo en Kaggle: buscar "DAGM 2007"

### Alternativas futuras (no bloqueantes)
- **KolektorSDD**: dataset real de industria electrónica (micro-fisuras), ~400 imágenes.
- **MVTec AD**: estándar internacional más avanzado en anomaly detection.

---

## 4. Stack técnico

| Área | Herramienta |
|---|---|
| Lenguaje | Python 3.10+ |
| Framework DL | PyTorch + PyTorch Lightning |
| Segmentación | segmentation_models_pytorch (U-Net/FPN con backbones preentrenados) |
| Visión | OpenCV, PIL |
| Augmentations | Albumentations |
| Loss / Métricas | Focal Loss + Dice Loss (combo), mIoU, F1-Score, latencia de inferencia |
| Tracking | TensorBoard o Weights & Biases (W&B) — *por decidir* |
| App | Streamlit |
| Despliegue | Docker |

---

## 5. Arquitectura del modelo

### Modelos a probar
1. **U-Net** con encoder preentrenado ligero: ResNet18 o EfficientNet-B0 (prioridad: velocidad de inferencia).
2. **FPN** (Feature Pyramid Network) como alternativa rápida.

### Función de pérdida (crítica por el desequilibrio)
No usar solo CrossEntropy/BCE. Combinar:

```
Loss_Total = α · FocalLoss + β · DiceLoss
```

- **Focal Loss**: reduce el peso de píxeles fáciles (superficie sana) y enfoca el gradiente en los difíciles (bordes del defecto).
- **Dice Loss**: maximiza directamente la superposición entre máscara real y predicha.
- Valores de α y β: hiperparámetros a ajustar en Fase 2.

### Post-procesado (Fase 3)
1. Umbral de decisión (thresholding) sobre la probabilidad por píxel.
2. Operaciones morfológicas OpenCV (`cv2.morphologyEx`: apertura/cierre) para eliminar ruido aislado y rellenar lagunas.
3. Lógica de negocio: si área del defecto > umbral X píxeles → **FAIL**; si no → **PASS**.

---

## 6. Estructura objetivo del repositorio

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
├── docs/
│   └── proyecto_original.md    # Fuente original del proyecto (solo lectura)
├── Dockerfile                  # Contenedorización del servicio
├── requirements.txt
├── MEMORIA.md                  # Este documento (seguimiento)
└── README.md                   # Presentación visual del proyecto
```

---

## 7. Roadmap por fases

### Fase 0 — Planificación ✅
- [x] Definir proyecto, alcance y valor de negocio
- [x] Elegir dataset inicial (DAGM 2007)
- [x] Crear memoria de seguimiento (`MEMORIA.md`)
- [ ] Inicializar repo Git y estructura de carpetas

### Fase 1 — Preparación del dataset y Data Augmentation avanzado
- [ ] Descargar DAGM 2007 y organizar en `data/`
- [ ] EDA en `notebooks/`: explorar clases, tamaño, distribución de defectos
- [ ] Cuantificar el desequilibrio de píxeles (% fondo vs % defecto)
- [ ] Custom PyTorch Dataset (`src/dataset.py`)
- [ ] Pipelines de Albumentations:
  - [ ] Rotaciones aleatorias (`RandomRotate90`, `Flip`)
  - [ ] Iluminación/contraste (`RandomBrightnessContrast`, `RandomGamma`) — simula sombras de línea de montaje
  - [ ] Ruido (`GaussNoise`) — simula grano de cámaras industriales

### Fase 2 — Modelado y Loss Functions especializadas
- [ ] U-Net con encoder ResNet18 / EfficientNet-B0 (SMP)
- [ ] FPN como modelo alternativo
- [ ] Implementar Combo Loss (α·Focal + β·Dice) en `src/model.py`
- [ ] Configurar tracking (TensorBoard o W&B)
- [ ] Loop de entrenamiento/validación (`src/train.py`, PyTorch Lightning)
- [ ] Experimentos: comparar backbones y valores α/β
- [ ] Evaluar mIoU y F1-Score en validación

### Fase 3 — Post-procesado morfológico e inferencia industrial
- [ ] Thresholding sobre probabilidades por píxel
- [ ] Morfología OpenCV (apertura/cierre) en `src/utils.py`
- [ ] Lógica de decisión PASS/FAIL por área de defecto
- [ ] Evaluación completa (`src/evaluate.py`): mIoU, Precision/Recall
- [ ] Medición de latencia (ms/imagen) y traducción a FPS

### Fase 4 — Dashboard interactivo y despliegue
- [ ] `app/app.py` en Streamlit:
  - [ ] Carga de imagen de pieza industrial
  - [ ] Visualización: original + máscara binaria + overlay (defecto en rojo transparente)
  - [ ] Veredicto grande: PASS (verde) / DEFECT DETECTED (rojo)
  - [ ] Métricas en tiempo real: latencia ms, % área defecto, confianza
- [ ] Dockerfile y prueba de contenedor
- [ ] README.md visual para portafolio (gifs, métricas, arquitectura)

---

## 8. Métricas objetivo

> ⚠️ Valores preliminares; se ajustarán con los resultados del EDA y los primeros experimentos (Fase 2).

| Métrica | Objetivo preliminar |
|---|---|
| mIoU (clase defecto) | ≥ 0.70 |
| F1-Score (clase defecto) | ≥ 0.75 |
| Latencia de inferencia | < 100 ms/imagen (GPU) |
| Throughput | ≥ 10 FPS |
| Veredicto PASS/FAIL | Precisión de clasificación ≥ 95% |

---

## 9. Registro de decisiones técnicas

| # | Fecha | Decisión | Justificación |
|---|---|---|---|
| 1 | 21/08/2026 | Dataset inicial: **DAGM 2007** | 10 clases de textura con más variedad que KolektorSDD; máscaras binarias bien definidas; ideal para demostrar manejo de desequilibrio de píxeles. KolektorSDD/MVTec quedan como extensión futura. |
| 2 | 21/08/2026 | Idioma del proyecto: español | Documentación interna en español; el README público del portafolio podrá valorarse en inglés más adelante. |

---

## 10. Registro de sesiones (changelog)

| Fecha | Sesión | Trabajo realizado | Decisiones | Siguiente paso |
|---|---|---|---|---|
| 21/08/2026 | 1 | Creación de la memoria del proyecto a partir del documento fuente. Definición de alcance, stack, roadmap y métricas. | Dataset inicial: DAGM 2007. Modo de trabajo colaborativo con explicaciones. | Inicializar Git + estructura de carpetas; empezar Fase 1 (descarga DAGM y EDA). |
