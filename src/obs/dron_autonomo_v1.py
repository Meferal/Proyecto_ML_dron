print("1. Iniciando script...")
import zmq
import airsim
import numpy as np
import cv2
import time
import base64
import json
from controller import DroneController

# --- CONFIGURACIÓN ---
#  TARGET_CLASS = 1    # Ambulancia en City
TARGET_CLASS = 6    # Animal en SubUrbanNeighborhood
CAMERA_NAME = "0"  # Cámara frontal

print("2. Configurando ZMQ...")
# --- ZMQ SETUP ---
context = zmq.Context()

# PUBLICAR Imagen (Puerto 5556)
socket_pub_img = context.socket(zmq.PUB)
socket_pub_img.bind("tcp://*:5556")

# SUSCRIBIRSE a Detecciones (Puerto 5555)
socket_sub_det = context.socket(zmq.SUB)
socket_sub_det.connect("tcp://localhost:5555")
socket_sub_det.setsockopt_string(zmq.SUBSCRIBE, "")

print("3. Conectando a AirSim...")
# --- AIRSIM SETUP ---
client = airsim.MultirotorClient()

print("4. Confirmando conexión (Esperando al simulador)...")
client.confirmConnection()

print("5. Habilitando API control...")
client.enableApiControl(True)
client.armDisarm(True)

print("6. [DRON] Despegando...")
client.takeoffAsync().join()
client.moveToZAsync(-2.0, 1).join() # Altura de vuelo 2m

controller = DroneController()
print("[DRON] Autonomía iniciada. Enviando vídeo a YOLO...")


def get_data_and_send():
    # Pedimos Imagen visual (Scene) y Profundidad (DepthPlanar)
    responses = client.simGetImages([
        airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False), # RGB no comprimido
        airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPlanar, True)    # Depth float
    ])

    if len(responses) < 2:
        return None

    # --- 1. PROCESAR IMAGEN PARA YOLO ---
    img_resp = responses[0]
    img1d = np.frombuffer(img_resp.image_data_uint8, dtype=np.uint8)

    # Calcular canales dinámicamente basado en el tamaño del array
    # Size esperado = W * H * Canales
    pixels = img_resp.width * img_resp.height

    if img1d.size == pixels * 3:
        # Caso 3 canales
        img_bgr = img1d.reshape(img_resp.height, img_resp.width, 3)

    elif img1d.size == pixels * 4:
        # Caso 4 canales (BGRA - Estándar en algunas versiones)
        img_bgra = img1d.reshape(img_resp.height, img_resp.width, 4)
        img_bgr = img_bgra[:, :, :3]  # Quitamos el canal Alpha

    else:
        print(f"[ERROR] Tamaño de imagen inesperado: {img1d.size} bytes para {img_resp.width}x{img_resp.height}")
        return None

    # COMPRESIÓN JPG: Reduce drásticamente el lag de red
    # quality=80 es un buen balance entre velocidad y precisión para YOLO
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Enviar por ZMQ
    socket_pub_img.send_json({"image": jpg_as_text, "timestamp": time.time()})

    # --- 2. PROCESAR DEPTH PARA CONTROL ---
    depth_resp = responses[1]
    depth = airsim.list_to_2d_float_array(
        depth_resp.image_data_float,
        depth_resp.width,
        depth_resp.height
    )
    return depth


# --- BUCLE PRINCIPAL ---
last_time = time.time()

# Variables para recordar la última velocidad
last_vx, last_vy = 0, 0

while True:

    # 0. "Heartbeat": Recordar al dron que seguimos vivos ANTES de procesar nada pesado
    # Enviamos la última velocidad conocida para reiniciar el contador del Watchdog
    client.moveByVelocityBodyFrameAsync(
        last_vx, last_vy, 0,
        duration=1,
        yaw_mode=airsim.YawMode(True, 0)
    )

    dt = time.time() - last_time
    last_time = time.time()
    duration = max(dt * 2, 0.1, 1)

    # 1. Enviar imagen y obtener profundidad local
    depth = get_data_and_send()

    if depth is None:
        continue

    # 2. Leer Detecciones (Non-blocking)
    # Leemos TODAS las pendientes y nos quedamos solo con la última para evitar latencia acumulada
    detections = []
    try:
        while True:
            msg = socket_sub_det.recv_json(flags=zmq.NOBLOCK)
            detections = msg["detections"]
    except zmq.Again:
        pass # No hay datos nuevos, usamos la lista vacía o lógica de 'target perdido'

    # 3. Lógica de Control
    target_box = None
    if detections:
        # Buscar la mejor detección de la clase objetivo
        valid_dets = [d for d in detections if d['class'] == TARGET_CLASS]
        if valid_dets:
            target_box = max(valid_dets, key=lambda x: x['confidence'])

    if target_box:
        # SEGUIMIENTO
        bbox = target_box['bbox']
        vx, vy, dist = controller.follow_target(bbox, depth, duration)

        # YawMode=0 mantiene el frente estable, útil para seguimiento visual
        client.moveByVelocityBodyFrameAsync(vx, vy, 0, duration, airsim.YawMode(False, 0))
        print(f"[FOLLOW] Conf: {target_box['confidence']:.2f} | Dist: {dist:.2f}m")

    else:
        # BÚSQUEDA / EVITACIÓN
        vx, vy = controller.avoid_obstacles(depth)
        client.moveByVelocityBodyFrameAsync(vx, vy, 0, duration, airsim.YawMode(False, 0))
        print(f"[SEARCH] Explorando... vx={vx:.1f}")

    # Pausa mínima para estabilidad del hilo
    # time.sleep(0.05)
    pass
