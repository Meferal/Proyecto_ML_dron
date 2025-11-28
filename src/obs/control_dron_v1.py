import zmq
import json
import airsim
import time
import os
from pathlib import Path

# Obtener la ruta absoluta del script actual
script_path = Path("test.ipynb").resolve()
src = script_path.parent.parent   # src

# ==========================
# ZeroMQ Subscriber
# ==========================
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("[DRON] Suscrito a YOLO en tcp://localhost:5555")

# ==========================
# Configurar AirSim
# ==========================
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
client.moveToZAsync(-3, 1).join()

print("[DRON] Listo para recibir detecciones…")

# ==========================
# Loop de control
# ==========================
TARGET_CLASS = 1   # Ambulancia, para entorno City
# TARGET_CLASS = 6   # Animales, para entorno SubUrbanNeighborhood

while True:
    msg = socket.recv_string()
    data = json.loads(msg)

    detections = data["detections"]

    if len(detections) == 0:
        # Sin detecciones → navegación autónoma simple
        client.moveByVelocityAsync(1, 0, 0, 0.3)
        continue

    # Tomamos la detección más grande
    det = max(detections, key=lambda x: x["confidence"])

    cls = det["class"]
    x1, y1, x2, y2 = det["bbox"]

    cx = (x1 + x2) / 2

    # Control básico
    if cls == TARGET_CLASS:
        print("[DRON] Siguiendo objetivo…")

        # Control proporcional muy simple
        if cx < 400:     # izquierda
            client.moveByVelocityAsync(0, -1, 0, 0.05)
        elif cx > 880:   # derecha
            client.moveByVelocityAsync(0, 1, 0, 0.05)
        else:
            client.moveByVelocityAsync(1, 0, 0, 0.05)

    else:
        # Sin objetivo → navegación autónoma básica
        client.moveByVelocityAsync(1, 0, 0, 0.05)

    time.sleep(0.01)
