# Para capturar el fotograma
import cv2

# Para procesar el arreglo de la imagen
import numpy as np

# Importar el módulo tensorflow y cargar el modelo
import tensorflow as tf

# Cargar el modelo
model = tf.keras.models.load_model('keras_model.h5')

# Adjuntando el índice de la cámara como 0 con la aplicación del software
camera = cv2.VideoCapture(0)

# Bucle infinito
while True:

    # Leyendo/Solicitando un fotograma de la cámara
    status, frame = camera.read()

    # Si somos capaces de leer exitosamente el fotograma
    if status:

        # Voltear la imagen
        frame = cv2.flip(frame, 1)

        # Cambiar el tamaño del fotograma
        frame_resized = cv2.resize(frame, (224, 224))

        # Expandir las dimensiones para que coincidan con el formato de entrada del modelo
        frame_expanded = np.expand_dims(frame_resized, axis=0)

        # Normalizar antes de alimentar al modelo
        frame_normalized = frame_expanded / 255.0

        # Obtener predicciones del modelo
        predictions = model.predict(frame_normalized)

        # Imprimir las predicciones
        print("Predicciones:", predictions)

        # Esperando 1ms
        code = cv2.waitKey(1)

        # Mostrando los fotogramas capturados
        cv2.imshow('Alimentar', frame)

        # Si se preciona la barra espaciadora, romper el bucle
        if code == 32:
            break

# Liberar la cámara de la aplicación del software
camera.release()

# Cerrar la ventana abierta
cv2.destroyAllWindows()
