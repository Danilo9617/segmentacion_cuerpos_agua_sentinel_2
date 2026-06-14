# Segmentacion de Cuerpos de Agua con Sentinel-2

Proyecto para segmentacion semantica de cuerpos de agua usando imagenes Sentinel-2 y una U-Net con backbone `resnet34` en `segmentation_models_pytorch`.

## Resumen

- Modelo: `U-Net`
- Backbone: `resnet34`
- Inicializacion: `encoder_weights=None` (No se utilizaron encoders preentenados - entrenado desde cero)
- Entrada: 5 canales `Blue`, `Green`, `Red`, `NIR` y `NDWI`
- Tamano de parche: `256x256`
- Loss: `0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss`
- Dispositivo: `CUDA` si esta disponible, si no `CPU`

## Estructura

- `notebook/Sent2_WaterBodies.ipynb`: entrenamiento, evaluacion e inferencia
- `weights/best_model_water.pth`: mejor checkpoint guardado
- `train_history_water.csv`: historial de entrenamiento por epoca
- `Challenge_Tecnico_Segmentacion_Agua_Sentinel2 1.pdf`: enunciado del challenge

## Instalacion

```bash
pip install -r requirements.txt
```

## Sobre los datos

La carpeta data/ está excluida de este repositorio debido a que el volumen total del dataset excede los límites recomendados para el control de versiones en GitHub y para optimizar el tiempo de clonación del proyecto.

Para replicar este entorno localmente y ejecutar el pipeline de entrenamiento, sigue estos pasos:

- Descarga de datos: Obtén el dataset original que contiene tanto las imágenes de Sentinel-2 como sus respectivas máscaras de anotación desde su repositorio oficial en Zenodo: Zenodo Record 5205674.

- Estructura local: Descomprime los archivos en la raíz del proyecto asegurándote de mantener la siguiente estructura jerárquica para que el DataLoader mapee y procese los archivos automáticamente:

## Dataset esperado

El notebook espera la siguiente estructura:

```text
data/dset-s2/
  tra_scene/
  tra_truth/
  val_scene/
  val_truth/
```

## Descarga y organizacion del dataset

El entrenamiento se apoyo en datos Sentinel-2 etiquetados para segmentacion de agua, almacenados localmente en la carpeta `data/`.

El flujo esperado es:

1. Descargar las escenas y mascaras desde la fuente correspondiente.
2. Organizar los archivos localmente dentro de `data/dset-s2/`.
3. Mantener separados los subconjuntos `tra_scene`, `tra_truth`, `val_scene` y `val_truth`.
4. Ejecutar el notebook una vez que esa estructura exista en disco.

Este repositorio no automatiza la descarga del dataset ni publica los archivos pesados dentro de GitHub. La intencion es mantener el repositorio liviano y centrado en la parte reproducible del pipeline de modelado.

Las mascaras se emparejan a partir del nombre del archivo. La inferencia espera que las primeras 4 bandas del `.tif` correspondan a:

1. Blue
2. Green
3. Red
4. NIR

El quinto canal se calcula en el momento como:

```text
NDWI = (Green - NIR) / (Green + NIR + 1e-6)
```

## Entrenamiento

Abre el notebook y ejecuta las celdas en orden:

```bash
jupyter notebook notebook/Sent2_WaterBodies.ipynb
```

El entrenamiento guarda:

- mejor modelo en `weights/best_model_water.pth`
- historial en `train_history_water.csv`

## Inferencia

El notebook ya incluye funciones para:

- cargar el checkpoint
- procesar imagenes `.tif`
- reconstruir la mascara completa por ventanas deslizantes
- exportar una mascara georreferenciada en formato GeoTIFF

## Notas

- No se usan pesos preentrenados.
- La normalizacion divide por `10000.0`, asumiendo reflectancia escalada tipica de Sentinel-2.
- El repositorio no incluye el dataset crudo en Git para no inflar el versionado.
- La carpeta `data/` debe reconstruirse localmente antes de entrenar o inferir.
