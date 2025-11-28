import sys
import asyncio
import time
import base64
import json
import numpy as np
import cv2
import zmq

# --- 1. PARCHE COMPATIBILIDAD WINDOWS ---
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import airsim
from controller import DroneController

# --- CONFIGURACIÓN ---
TARGET_CLASS = 1
FLIGHT_ALTITUDE = -2.5
CAMERA_NAME = "0"

# Factor de suavizado (0.0 a 1.0). 
# 0.2 = Muy suave (lento en reaccionar)
# 0.8 = Muy reactivo (puede vibrar)
# 0.4 = Balanceado
ALPHA = 0.4


def body_to_world(vx_body, vy_body, yaw_rad):
    vx_world = (vx_body * np.cos(yaw_rad)) - (vy_body * np.sin(yaw_rad))
    vy_world = (vx_body * np.sin(yaw_rad)) + (vy_body * np.cos(yaw_rad))
    return vx_world, vy_world


def main():
    # --- INICIALIZACIÓN ---
    print("[INIT] Configurando ZeroMQ...")
    context = zmq.Context()
    socket_pub_img = context.socket(zmq.PUB)
    socket_pub_img.bind("tcp://*:5556")

    socket_sub_det = context.socket(zmq.SUB)
    socket_sub_det.connect("tcp://localhost:5555")
    socket_sub_det.setsockopt_string(zmq.SUBSCRIBE, "")

    print("[INIT] Conectando a AirSim...")
    client = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=5)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("[DRON] Despegando...")
    client.takeoffAsync().join()
    client.moveToZAsync(FLIGHT_ALTITUDE, 1).join()

    controller = DroneController()
    print("[DRON] Vuelo fluido iniciado. CTRL+C para salir.")

    last_time = time.time()

    # Variables para el suavizado de movimiento (Memoria del frame anterior)
    smooth_vx = 0.0
    smooth_vy = 0.0

    try:
        while True:
            # Calculamos dt real
            now = time.time()
            dt = now - last_time
            last_time = now

            # 1. Obtener Imagen y Depth
            responses = client.simGetImages([
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False),
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPlanar, True)
            ])

            if len(responses) < 2:
                continue

            # --- Procesar Imagen YOLO ---
            img_resp = responses[0]
            img1d = np.frombuffer(img_resp.image_data_uint8, dtype=np.uint8)
            pixels = img_resp.width * img_resp.height

            if img1d.size == pixels * 3:
                img_bgr = img1d.reshape(img_resp.height, img_resp.width, 3)
            elif img1d.size == pixels * 4:
                img_bgr = img1d.reshape(img_resp.height, img_resp.width, 4)[:, :, :3]
            else:
                continue

            # Enviar a YOLO (Comprimido)
            _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            socket_pub_img.send_json({"image": jpg_as_text, "timestamp": time.time()})

            # --- Procesar Depth ---
            depth_resp = responses[1]
            depth = airsim.list_to_2d_float_array(depth_resp.image_data_float, depth_resp.width, depth_resp.height)

            # 2. Recibir Detecciones
            detections = []
            try:
                while True:
                    msg = socket_sub_det.recv_json(flags=zmq.NOBLOCK)
                    detections = msg["detections"]
            except zmq.Again:
                pass

            # 3. Lógica de Control (PID)
            target_vx, target_vy = 0, 0
            yaw_rate = 0

            target_box = None
            if detections:
                valid_dets = [d for d in detections if d['class'] == TARGET_CLASS]
                if valid_dets:
                    target_box = max(valid_dets, key=lambda x: x['confidence'])

            if target_box:
                # SEGUIMIENTO
                bbox = target_box['bbox']
                target_vx, target_vy, dist = controller.follow_target(bbox, depth, dt)
                print(f"[FOLLOW] Dist: {dist:.2f}m | PID VX: {target_vx:.2f} | PID VY: {target_vy:.2f}")
            else:
                # BÚSQUEDA
                target_vx, target_vy = controller.avoid_obstacles(depth)
                yaw_rate = 15
                print(f"[SEARCH] Explorando... VX: {target_vx:.2f} | VY: {target_vy:.2f}")

            # --- 4. SUAVIZADO DE MOVIMIENTO (Low Pass Filter) ---
            # Esto evita tirones bruscos si el PID cambia de golpe
            smooth_vx = (ALPHA * target_vx) + ((1 - ALPHA) * smooth_vx)
            smooth_vy = (ALPHA * target_vy) + ((1 - ALPHA) * smooth_vy)

            # --- 5. APLICAR MOVIMIENTO (SIN JOIN) ---
            state = client.getMultirotorState()
            yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]

            vx_world, vy_world = body_to_world(smooth_vx, smooth_vy, yaw)

            # IMPORTANTE: 
            # 1. duration=1.0: Damos una orden larga para asegurar continuidad.
            # 2. NO usamos .join(): El bucle se repite inmediatamente y sobrescribe la orden
            #    antes de que termine el segundo, logrando un movimiento continuo.
            client.moveByVelocityZAsync(
                vx_world,
                vy_world,
                FLIGHT_ALTITUDE,
                duration=1.0,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
            )

            # Pequeña pausa técnica para no saturar la CPU, 
            # pero mucho menor que la duración del movimiento.
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[SALIDA] ¡Ctrl+C detectado! Aterrizando...")

    except Exception as e:
        print(f"\n[ERROR] {e}")

    finally:
        print("[SALIDA] Limpiando recursos...")
        try:
            # Frenar antes de salir
            client.moveByVelocityAsync(0, 0, 0, 1).join()
            client.enableApiControl(False)
        except:
            pass

        socket_pub_img.close()
        socket_sub_det.close()
        context.term()
        print("[SALIDA] Listo.")


if __name__ == "__main__":
    main()
