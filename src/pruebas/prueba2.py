import airsim
import cv2
import numpy as np
from ultralytics import YOLO
import os

model = YOLO("./models/last.pt")

# Cargar modelo YOLO
model = YOLO('yolo11n.pt')

# Conectar al simulador
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Despegar
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -10, 5).join()

# Hover (mantener posición)
client.hoverAsync().join

# Obtener una imagen
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
])

# Convertir a formato útil
response = responses[0]
img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
img_rgb = img1d.reshape(response.height, response.width, 3)

# Para YOLO, guardar o procesar
cv2.imwrite("captura.png", img_rgb)

