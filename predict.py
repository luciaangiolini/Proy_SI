#Librerías
from ultralytics import YOLO
import numpy as np
import cv2
import math

#Llamar al modelo creado 
model = YOLO('./runs/classify/train3/weights/best.pt')  # load a custom model

#Imagen de test set a predecir IR MODIFICANDO!!
img_path = './data/cancer_dataset/test/IMAGEN_A_PREDECIR'
results = model(img_path)  # predict on an image

#Obtener los resultados de la predicción: probabilidades y clases
names_dict = results[0].names
probs = results[0].probs.data.tolist()
max_index = np.argmax(probs)
predicted_class = names_dict[max_index]
confidence = probs[max_index]
confidence_truncated = math.floor(confidence * 100) / 100

#Vamos a mostrar la imagen con su clase predicha
image = cv2.imread(img_path)

#Estilo con OpenCV para adecuar la imagen 
image = cv2.resize(image, (360, 363))
#Texto a incluir en la imagen
label_text = f'{predicted_class}: {confidence_truncated:.2f}'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
thickness = 1
text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
text_x, text_y = 10, 30

cv2.rectangle(image, 
              (text_x - 5, text_y - text_size[1] - 5), 
              (text_x + text_size[0] + 5, text_y + 5), 
              (255, 255, 255), 
              -1)

cv2.putText(image, label_text, 
            (text_x, text_y), 
            font, 
            font_scale, 
            (0, 0, 0), 
            thickness, 
            cv2.LINE_AA)

#Mostrar imagen y dejarla en pantalla hasta que se presione una tecla
cv2.imshow('Resultado', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Imprimimos los valores para corroborar
print(names_dict)
print(probs)
print(predicted_class)