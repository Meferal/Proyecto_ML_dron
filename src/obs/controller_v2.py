import numpy as np
from pid import PID

# Distancias objetivo (en metros)
SAFE_DIST = 4.0
FOLLOW_DIST = 5.0
MAX_SPEED = 10.0


class DroneController:
    def __init__(self):
        # PID DISTANCIA (Avanzar/Retroceder)
        # Permite velocidad negativa para retroceder si se acerca mucho
        self.pid_distance = PID(Kp=0.8, Ki=0.01, Kd=0.5,
                                output_limits=(-MAX_SPEED, MAX_SPEED))

        # PID CENTRADO (Izquierda/Derecha)
        self.pid_center = PID(Kp=0.02, Ki=0.0001, Kd=0.005,
                              output_limits=(-2, 2))

    # -------------------------
    # EVITAR OBSTÁCULOS
    # -------------------------
    def avoid_obstacles(self, depth):
        h, w = depth.shape
        # Dividimos la imagen en 3 zonas
        left = np.min(depth[:, :w//3])
        center = np.min(depth[:, w//3:2*w//3])
        right = np.min(depth[:, 2*w//3:])

        vx = 2.0    # Velocidad base de exploración
        vy = 0.0

        if center < SAFE_DIST:
            vx = 0  # Freno si hay algo delante

        # Lógica simple de evasión
        if left < SAFE_DIST:
            vy += 1
        if right < SAFE_DIST:
            vy -= 1

        return vx, vy

    # -------------------------
    # SEGUIMIENTO DE OBJETIVO
    # -------------------------
    def follow_target(self, box, depth, dt):
        x1, y1, x2, y2 = map(int, box)

        # Corrección de límites por si YOLO se sale de la imagen
        h, w = depth.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        cx = (x1 + x2) / 2

        # --- CÁLCULO DE DISTANCIA ROBUSTO ---
        if x2 > x1 and y2 > y1:
            depth_crop = depth[y1:y2, x1:x2]
            if depth_crop.size > 0:
                # Usamos percentil 10 para ignorar el fondo/cielo lejano
                distance = float(np.percentile(depth_crop, 10))
            else:
                distance = FOLLOW_DIST # Valor neutro si falla el crop
        else:
            distance = FOLLOW_DIST

        # --- PID distancia ---
        error_dist = distance - FOLLOW_DIST
        vx = self.pid_distance.update(error_dist, dt)

        # --- PID centrado horizontal ---
        img_center = w / 2
        offset = cx - img_center
        # El error es negativo porque si el objeto está a la derecha, queremos ir a la derecha
        vy = self.pid_center.update(-offset, dt)

        return vx, vy, distance
