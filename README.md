# Segmentación de Cuerpos de Agua con Sentinel-2

Proyecto para segmentación semántica de cuerpos de agua usando imágenes Sentinel-2 y una U-Net con backbone `resnet34` en `segmentation_models_pytorch`.

## Resumen

- Modelo: `U-Net`
- Backbone: `resnet34`
- Inicialización: `encoder_weights=None` (entrenamiento desde cero, sin pesos preentrenados)
- Entrada: 7 canales `Blue`, `Green`, `Red`, `NIR`, `SWIR1`, `SWIR2` y `MNDWI`
- Tamaño de parche: `256x256`
- Loss: `0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss`
- Dispositivo: `CUDA` si está disponible, si no `CPU`

## Estructura

- `notebook/Sent2_WaterBodies.ipynb`: entrenamiento, validación, test e inferencia exploratoria
- `inference.py`: pipeline de inferencia para GeoTIFF completos
- `experiments/`: historial de corridas, métricas y gráficas
- `weights/`: carpeta opcional para pesos locales fuera del tracking principal

## Instalación

```bash
pip install -r requirements.txt
```

## Datos

La carpeta `data/` no se incluye en el repositorio porque el volumen del dataset excede lo recomendable para GitHub.

El proyecto espera escenas y máscaras organizadas localmente. La descarga del dataset la haces de forma manual desde la fuente correspondiente. Una referencia usada durante el desarrollo fue:

- Zenodo Record 5205674: https://zenodo.org/records/5205674
- Dataset de test externo (`part5`): https://zenodo.org/records/11278238

### Estructura esperada

```text
data/
  dset-s2/
    tra_scene/
    tra_truth/
    val_scene/
    val_truth/
  part5/
    71/
    73/
    75/
    ...
```

El notebook también soporta múltiples carpetas de dataset dentro de `data/`, siempre que mantengan la misma estructura interna de `tra_scene`, `tra_truth`, `val_scene` y `val_truth`.

### Bandas esperadas

La inferencia espera que el GeoTIFF de entrada tenga estas 6 bandas crudas:

1. Blue
2. Green
3. Red
4. NIR
5. SWIR1
6. SWIR2

El índice espectral se calcula sobre la marcha como:

```text
MNDWI = (Green - SWIR1) / (Green + SWIR1 + 1e-6)
```

La normalización usada es:

```text
reflectancia = valor / 10000.0
```

## Entrenamiento

Abre el notebook y ejecuta las celdas en orden:

```bash
jupyter notebook notebook/Sent2_WaterBodies.ipynb
```

Cada entrenamiento crea automáticamente una carpeta nueva dentro de `experiments/`, por ejemplo:

```text
experiments/
  run_YYYYMMDD_HHMMSS/
    best_model.pth
    history.csv
    config.json
    val_metrics.csv
    learning_curves.png
    confusion_matrix.png
    precision_recall_curve.png
```

Eso evita sobrescribir corridas anteriores y deja trazabilidad de hiperparámetros, métricas y artefactos.

## Test

El notebook incluye una evaluación sobre escenas externas en `data/part5/` y guarda:

- métricas por escena
- métricas globales
- matriz de confusión
- curva precisión-recall
- una visualización diagnóstica de ejemplo

### Diferencia entre validation y test

Es importante resaltar que las escenas de `test` en `part5` son mucho más grandes que las escenas usadas normalmente en `validation`.

- Una escena típica de `validation` en este proyecto está alrededor de `700x700` a `1000x900` píxeles.
- Las escenas de `part5` están en `10980x10980` píxeles.

## Inferencia

La inferencia puede hacerse desde el notebook o directamente con:

```bash
python inference.py --input "data/dset-s2/val_scene/S2A_L2A_20190318_N0211_R061_6Bands_S1.tif" --model "experiments/run_20260620_001244/best_model.pth"
```

También acepta una carpeta como `data/part5/82` y resuelve automáticamente la imagen `*_img.tif` correspondiente.

El script:

- lee las bandas esperadas
- normaliza dividiendo por `10000.0`
- calcula `MNDWI`
- parte la escena completa en ventanas deslizantes de `256x256`
- reconstruye el mapa completo de probabilidad
- exporta máscara binaria GeoTIFF y mapa de probabilidad GeoTIFF
- guarda un `.json` con el formato de entrada y salida
- genera un `preview.png` con RGB, predicción y, si existe, máscara real y mapa de errores

## Notas

- No se usan pesos preentrenados.
- El criterio de selección del mejor modelo durante entrenamiento es `val_iou`.
- `inference.py` soporta checkpoints antiguos de 5 canales y checkpoints nuevos de 7 canales.
- El repositorio no incluye el dataset crudo para mantener liviano el versionado.

## Bibliografía

La carpeta local `Bibliografia/` reúne los artículos y documentos de apoyo consultados durante el desarrollo. En este repositorio esa carpeta está excluida de Git por tamaño, pero localmente se trabajó con archivos como:

- `Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf`
- `1-s2.0-S0022169426005068-main.pdf`
- `1-s2.0-S0022169426008097-main.pdf`
- `1-s2.0-S0022169426009388-main.pdf`
- `1-s2.0-S0034425723000032-main.pdf`
- `1-s2.0-S0034425725002810-main.pdf`
- `1-s2.0-S0924271625002692-main.pdf`
- `1-s2.0-S0952197623010886-main.pdf`
- `1-s2.0-S0957417421009386-main.pdf`
- `1-s2.0-S1569843225003887-main.pdf`
- `1-s2.0-S1569843226000300-main.pdf`
- `1-s2.0-S2352711026002463-main.pdf`
