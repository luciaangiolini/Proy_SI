# Clasificación de Tipos Celulares con YOLOv8 para detección de células cancerígenas
## Facultad de Ingeniería UCC
Integrantes:
- **Angiolini, Lucía**
- **Pereyra, Lara**
- **Rolón, Luana**
- **Rueda, Mateo**
- **Saad Moisés, Selene**

En este proyecto desarrollamos un sistema de clasificación automática de imágenes de tipos celulares mediante técnicas de aprendizaje profundo. Utilizamos el framework YOLO (You Only Look Once) en su modalidad de clasificación de imágenes (YOLO-World o YOLOv8 classify) a través de la biblioteca Ultralytics, para identificar y categorizar células a partir de imágenes microscópicas.

## Descripción del caso

El conjunto de datos utilizado está compuesto por imágenes de 5 tipos celulares distintos:

- **BA**: Basófilos
- **ERB**: Eritroblastos
- **MO**: Monocitos
- **MYO**: Mielocitos
- **NGS**: Neutrófilos en banda

Estas imágenes representan células tanto normales como potencialmente indicadoras de patologías, como cáncer hematológico.
Los **mielocitos** en condiciones normales deben estar en médula, si pasan a sangre (es el caso de estas muestras), es un indicio de leucemia.
Las imágenes fueron preprocesadas y organizadas en carpetas por clase, respetando una estructura típica para clasificación.

## Modelo

El modelo se entrenó utilizando YOLOv8 en su configuración para clasificación. Aunque YOLO es ampliamente conocido por sus capacidades de detección de objetos en tiempo real, también ofrece una variante para tareas de clasificación pura, que es la que empleamos en este caso.

**Características del modelo entrenado:**

- Arquitectura: YOLOv8n-cls (`nano`, optimizada para velocidad)
- Número de clases: 5
- Tamaño de entrada: 64x64 píxeles

El modelo fue validado y exportado exitosamente, y puede utilizarse de forma local para realizar inferencias sobre imágenes nuevas.

## Inferencia y visualización

El modelo fue entrenado y validado en Google Colab, y los pesos resultantes fueron exportados para su uso local. En pruebas sobre imágenes del conjunto de test, el modelo fue capaz de clasificar correctamente los tipos celulares, con alta probabilidad asignada a la clase correcta.

### Ejemplo de visualización

- La imagen se abre con OpenCV.
- Se superpone una leyenda blanca con texto negro que indica:
  - La clase predicha (por ejemplo: `BA`)
  - La probabilidad asociada (truncada a dos decimales sin redondeo, por ejemplo: `0.99`)
  
Esto permite validar visualmente la predicción y resulta útil para análisis cualitativos o presentación a usuarios no técnicos.

## Requisitos

- Python 3.8+
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenCV (`opencv-python`)
- Numpy

```bash
pip install ultralytics opencv-python numpy
