import zmq
import json
import airsim
import time
import numpy as np

# ---------------- CONFIG ----------------
TARGET_CLASS = 1           # clase objetivo a seguir, Ambulancia para City
# TARGET_CLASS = 6           # clase objetivo a seguir, Animales para SubUrbanNeighborhood
SAFE_DIST = 4.0            # mínima distancia a obstáculos
FOLLOW_DIST = 5.0          # distancia deseada al target
MAX_SPEED = 5.0            # velocidad máxima del dron
DEPTH_WEIGHT = 1.5         # factor de evasión lateral
CAMERA_NAME = "0"          # cámara frontal

# ---------------- ZeroMQ Subscriber ----------------
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

# ---------------- AirSim ----------------
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(-3, 1).join()

print("[DRON] Control avanzado iniciado...")


# ---------------- FUNCIONES ----------------
def get_depth():
    depth_response = client.simGetImages([
        airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPlanar, True, False)
    ])[0]
    depth = airsim.list_to_2d_float_array(
        depth_response.image_data_float,
        depth_response.width,
        depth_response.height
    )
    return depth


def avoid_obstacles(depth):
    h, w = depth.shape
    left = depth[:, :w//3].mean()
    center = depth[:, w//3:2*w//3].mean()
    right = depth[:, 2*w//3:].mean()

    vx, vy = MAX_SPEED, 0

    if center < SAFE_DIST:
        vx = 0  # frena
    if left < SAFE_DIST:
        vy += DEPTH_WEIGHT
    if right < SAFE_DIST:
        vy -= DEPTH_WEIGHT

    return vx, vy


def follow_target(box, depth):
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    depth_crop = depth[y1:y2, x1:x2]
    distance = float(depth_crop.mean())

    # Control de velocidad proporcional a la diferencia de distancia
    vx = np.clip(distance - FOLLOW_DIST, 0, MAX_SPEED)

    # Control lateral según offset horizontal
    frame_center = depth.shape[1] / 2
    offset = cx - frame_center
    vy = np.clip(-offset / frame_center * MAX_SPEED, -MAX_SPEED, MAX_SPEED)

    return vx, vy, distance


# ---------------- MAIN LOOP ----------------
while True:
    # --- recibir detecciones ---
    msg = socket.recv_string()
    data = json.loads(msg)
    detections = data["detections"]

    # --- obtener depth ---
    depth = get_depth()

    # --- determinar modo ---
    target_box = None
    if len(detections) > 0:
        # coger la detección de mayor confianza
        target_box = max(detections, key=lambda x: x["confidence"])
        if target_box["class"] != TARGET_CLASS:
            target_box = None

    # --- seguir target ---
    if target_box is not None:
        vx, vy, distance = follow_target(target_box["bbox"], depth)

        # mover y orientar la cámara hacia adelante
        yaw_rate = 0
        client.moveByVelocityBodyFrameAsync(vx, vy, 0, 0.1, 
            airsim.YawMode(is_rate=False, yaw_or_rate=yaw_rate))

        print(f"🔵 Siguiendo target: distancia={distance:.2f}, vx={vx:.2f}, vy={vy:.2f}")

    else:
        # --- evitar obstáculos y explorar ---
        vx, vy = avoid_obstacles(depth)
        client.moveByVelocityBodyFrameAsync(vx, vy, 0, 0.1, 
            airsim.YawMode(is_rate=False, yaw_or_rate=0))
        print(f"🟡 Explorando: vx={vx:.2f}, vy={vy:.2f}")

    time.sleep(0.05)
