# airsim_arc_capture.py
# Requisitos: pip install airsim opencv-python numpy
import os
import math
import time
import numpy as np
import cv2
import airsim

# --- Parámetros de misión ---
ALTITUD = 5.0                # metros sobre el suelo
ARC_LENGTH = 20.0            # metros (longitud del arco que recorre)
ARC_ANGLE_DEG = 90.0         # ángulo total del arco (deg). Usamos 90° para un giro a la derecha
N_POINTS = 20                # puntos a muestrear a lo largo del arco
VELOCITY = 2.0               # m/s (velocidad del movimiento)
OUT_DIR = "airsim_caps"      # carpeta donde se guardan las imágenes
CAMERA_NAME = "0"            # nombre de la cámara (por defecto "0")
IMAGE_QUALITY = 90           # para PNG/JPEG (si aplica)

# --- Preparar carpeta de salida ---
os.makedirs(OUT_DIR, exist_ok=True)

# --- Conexión con AirSim ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# --- Despegue ---
print("Takeoff...")
client.takeoffAsync().join()

# Subir a ALTITUD (AirSim usa NED -> z negativa para altura)
target_z = -ALTITUD
print(f"Subiendo a {ALTITUD} m...")
client.moveToZAsync(target_z, 1.5).join()
time.sleep(1.0)

# --- Calcular puntos del arco ---
# Suponemos que el punto inicial está en (0,0) y el dron mira hacia +X.
# Construimos un arco hacia la derecha (clockwise). Para que la longitud del arco sea ARC_LENGTH:
theta = math.radians(ARC_ANGLE_DEG)            # ángulo en radianes
radius = ARC_LENGTH / theta                    # r = L / theta
print(f"Construyendo arco: longitud {ARC_LENGTH} m, ángulo {ARC_ANGLE_DEG}°, radio ≈ {radius:.3f} m")

# Centro de la circunferencia (a la derecha en -Y)
cx, cy = 0.0, -radius

# Angulo inicial desde el centro hacia el dron (drone en (0,0)) -> atan2(r, 0) = pi/2
angle0 = math.atan2(0.0 - cy, 0.0 - cx)  # debería ser +pi/2
# Generamos ángulos descendentes (clockwise) desde angle0 hasta angle0 - theta
angles = [angle0 - (i * (theta / max(1, N_POINTS - 1))) for i in range(N_POINTS)]

# Generamos waypoints (x,y,z)
waypoints = []
for ang in angles:
    x = cx + radius * math.cos(ang)
    y = cy + radius * math.sin(ang)
    z = target_z
    waypoints.append((x, y, z, ang))


# --- Función para capturar y guardar RGB + Depth ---
def save_images(step_idx):
    responses = client.simGetImages([
        airsim.ImageRequest(int(CAMERA_NAME), airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(int(CAMERA_NAME), airsim.ImageType.DepthPerspective, True, False)
    ])

    # Scene (RGB)
    scene_response = responses[0]
    if scene_response.width == 0:
        print(f"[WARN] Imagen RGB vacía en paso {step_idx}")
    else:
        img1d = np.frombuffer(scene_response.image_data_uint8, dtype=np.uint8) 
        img_rgb = img1d.reshape(scene_response.height, scene_response.width, 3)
        # AirSim devuelve en formato BGR? Normalmente es RGB. OpenCV usa BGR -> convertimos antes de guardar si queremos.
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        fname_rgb = os.path.join(OUT_DIR, f"rgb_{step_idx:03d}.png")
        cv2.imwrite(fname_rgb, img_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        print(f"Guardado RGB -> {fname_rgb}")

    # Depth
    depth_response = responses[1]
    # depth_response.image_data_float contiene un array de floats en fila si compress=False (fue True en request? aquí True->compressed?).
    # En la request hemos pedido compress=True para DepthPerspective (note: si tu AirSim devuelve .image_data_uint8 cuando comprimido)
    # vamos a manejar ambas posibilidades:
    if hasattr(depth_response, 'image_data_float') and depth_response.image_data_float:
        # depth floats lineales
        depth_img = np.array(depth_response.image_data_float, dtype=np.float32)
        depth_img = depth_img.reshape(depth_response.height, depth_response.width)
        # Normalizar para guardar como 16-bit PNG (mejor conservar precisión)
        dmin, dmax = np.nanmin(depth_img), np.nanmax(depth_img)
        if math.isfinite(dmin) and math.isfinite(dmax) and dmax > dmin:
            depth_norm = (depth_img - dmin) / (dmax - dmin)
        else:
            depth_norm = np.nan_to_num(depth_img)
        depth_16 = (depth_norm * 65535.0).astype(np.uint16)
        fname_depth = os.path.join(OUT_DIR, f"depth_{step_idx:03d}.png")
        cv2.imwrite(fname_depth, depth_16)
        print(f"Guardado Depth (float->16bit) -> {fname_depth}")
    else:
        # Si Depth viene comprimido en image_data_uint8 (png), lo guardamos directamente:
        if depth_response.image_data_uint8:
            img1d = np.frombuffer(depth_response.image_data_uint8, dtype=np.uint8)
            try:
                depth_png = cv2.imdecode(img1d, cv2.IMREAD_UNCHANGED)
                fname_depth = os.path.join(OUT_DIR, f"depth_{step_idx:03d}.png")
                cv2.imwrite(fname_depth, depth_png)
                print(f"Guardado Depth (comprimido) -> {fname_depth}")
            except Exception as e:
                print(f"[ERROR] al decodificar depth en paso {step_idx}: {e}")
        else:
            print(f"[WARN] No hay datos depth en paso {step_idx}")


# --- Misión: recorrer los waypoints ---
print("Iniciando recorrido del arco y captura de imágenes...")
for i, (x, y, z, ang) in enumerate(waypoints):
    # Calculamos velocidad y yaw para mirar en la dirección tangente
    # Tangente (derivada) para ángulo ang: v = r * (-sin(ang), cos(ang)) ; para clockwise (ang disminuye) la dirección de movimiento será proporcional a (-sin, cos)
    vx = -math.sin(ang)
    vy = math.cos(ang)
    # normalizar
    vnorm = math.hypot(vx, vy)
    if vnorm > 0:
        vx /= vnorm
        vy /= vnorm
    # Convertir a yaw en grados: atan2(vy, vx)
    yaw = math.degrees(math.atan2(vy, vx))
    # Mover al punto
    print(f"Waypoint {i+1}/{len(waypoints)} -> x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.1f}")
    # Usamos moveToPositionAsync para alcanzar la posición; join para sincronicidad
    client.moveToPositionAsync(x, y, z, VELOCITY, drivetrain=airsim.DrivetrainType.ForwardOnly,
                               yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw)).join()
    # Pequeña espera para estabilizar
    time.sleep(0.5)
    # Capturar y guardar imágenes
    save_images(i)

# --- Fin de misión: aterrizaje ---
print("Recorrido completado. Aterrizando...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("Misión finalizada. Imágenes guardadas en:", os.path.abspath(OUT_DIR))
