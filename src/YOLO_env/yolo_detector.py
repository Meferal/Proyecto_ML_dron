import zmq
import cv2
import numpy as np
import base64
import json
import time
from ultralytics import YOLO
import os
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
# Estamos en: src/YOLO_env/yolo_detector.py
# Queremos ir a: models/best_AirSim.pt
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent.parent  # Subimos: YOLO_env -> src -> PROYECTO
model_path = project_root / "models" / "best_AirSim.pt"

# --- ZMQ SETUP ---
context = zmq.Context()

# RECIBIR Imágenes (Puerto 5556)
socket_sub = context.socket(zmq.SUB)
socket_sub.connect("tcp://localhost:5556")
socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")
# CONFLATE: Mantener solo el último mensaje en el buffer (Evita lag si YOLO es lento)
try:
    socket_sub.setsockopt(zmq.CONFLATE, 1)
except AttributeError:
    print("[WARN] zmq.CONFLATE no disponible en esta versión.")

# PUBLICAR Detecciones (Puerto 5555)
socket_pub = context.socket(zmq.PUB)
socket_pub.bind("tcp://*:5555")

print(f"[YOLO] Iniciando servicio...")
print(f" -> Escuchando en :5556")
print(f" -> Publicando en :5555")

# --- CARGA MODELO ---
if model_path.exists():
    print(f"[YOLO] Cargando modelo: {model_path}")
    model = YOLO(str(model_path))
else:
    print(f"[ALERTA] No se encontró {model_path}. Usando 'yolo11n.pt' para pruebas.")
    # model = YOLO("yolo11n.pt")

# --- BUCLE DE INFERENCIA ---
while True:
    try:
        # 1. Esperar imagen (Bloqueante)
        msg = socket_sub.recv_json()

        # 2. Decodificar JPG Base64 -> Imagen OpenCV
        jpg_original = base64.b64decode(msg['image'])
        np_arr = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # 3. Inferencia YOLO11
        results = model(frame, verbose=False)

        # 4. Formatear resultados
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(), # [x1, y1, x2, y2]
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0])
                })

        # 5. Enviar respuesta
        response = {
            "timestamp": time.time(),
            "detections": detections
        }
        socket_pub.send_json(response)

        # Log ligero
        if detections:
            print(f"[DETECT] {len(detections)} objeto(s) detectado(s)")

    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(0.1)
