import numpy as np
from pid import PID

# Distancias de seguridad
FOLLOW_DIST = 3.0
MAX_SPEED = 10.0
SAFE_DIST = 3.0  # Si hay algo a menos de 3m en el centro, iniciamos evasión


class DroneController:
    def __init__(self):
        # PID DISTANCIA (Controla Velocidad Frontal - VX)
        self.pid_distance = PID(Kp=0.8, Ki=0.01, Kd=0.5,
                                output_limits=(-MAX_SPEED, MAX_SPEED))

        # PID CENTRADO (Controla Giro - YAW RATE)
        self.pid_center = PID(Kp=0.15, Ki=0.001, Kd=0.05,
                              output_limits=(-30, 30))

    # ---------------------------------------------------------
    # EVITACIÓN DE OBSTÁCULOS (Lógica de los 3 Sectores)
    # ---------------------------------------------------------
    def avoid_obstacles(self, depth):
        h, w = depth.shape

        # 1. DIVIDIR LA VISIÓN EN 3 SECTORES
        # Recortamos franjas verticales de la imagen de profundidad
        # [  IZQUIERDA  |   CENTRO   |   DERECHA  ]
        slice_w = w // 3

        strip_left   = depth[:, :slice_w]
        strip_center = depth[:, slice_w:2*slice_w]
        strip_right  = depth[:, 2*slice_w:]

        # 2. CALCULAR "PUNTUACIÓN" DE CADA SECTOR
        # Usamos la media para saber qué tan "abierto" está el camino.
        # Mayor valor = Más espacio libre (lejos).
        # Nota: Si el array está vacío (error de cámara), asumimos 0.
        score_left   = np.mean(strip_left)   if strip_left.size > 0 else 0
        score_center = np.mean(strip_center) if strip_center.size > 0 else 0
        score_right  = np.mean(strip_right)  if strip_right.size > 0 else 0

        # Para seguridad inmediata, miramos el punto más CERCANO del centro
        # (Para no chocarnos con una farola fina que la media no detecte)
        min_dist_center = np.min(strip_center) if strip_center.size > 0 else 0

        # 3. TOMAR DECISIÓN
        vx = 4.0        # Velocidad de crucero por defecto
        yaw_rate = 0.0  # Ir recto por defecto

        # CASO A: BLOQUEO FRONTAL
        # Si la media del centro es baja O hay un objeto muy cerca (< 2m)
        if score_center < SAFE_DIST or min_dist_center < 2.0:
            vx = 0.5 # Frenar drásticamente (pero no parar del todo para no estancarse)

            if score_center < (SAFE_DIST or min_dist_center < 2.0) and score_right < SAFE_DIST and score_left < SAFE_DIST:
                vx = 0  # Frenar completamente para evitar colision
                yaw_rate = 20  # Girar a la derecha dando media vuelta

            # Decidir ruta: ¿Izquierda o Derecha?
            if score_right > score_left:
                print(f"[AVOID] Obstáculo. Girando DERECHA (R:{score_right:.1f} > L:{score_left:.1f})")
                yaw_rate = 20  # Girar a la derecha
            else:
                print(f"[AVOID] Obstáculo. Girando IZQUIERDA (L:{score_left:.1f} > R:{score_right:.1f})")
                yaw_rate = -20 # Girar a la izquierda

        # CASO B: CAMINO LIBRE PERO NO PERFECTO
        # Si el centro está bien, pero un lado está MUY abierto, podemos corregir suavemente
        else:
            # Pequeña corrección para mantenerse centrado en pasillos
            # Si la derecha es mucho más abierta que la izquierda, girar un poco a la derecha
            if score_right > score_left + 2.0:
                yaw_rate = 5
            elif score_left > score_right + 2.0:
                yaw_rate = -5

        return vx, yaw_rate

    # -------------------------
    # SEGUIMIENTO
    # -------------------------
    def follow_target(self, box, depth, dt):
        x1, y1, x2, y2 = map(int, box)
        h, w = depth.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        cx = (x1 + x2) / 2

        distance = FOLLOW_DIST
        if x2 > x1 and y2 > y1:
            crop = depth[y1:y2, x1:x2]
            if crop.size > 0:
                distance = float(np.percentile(crop, 10))

        error_dist = distance - FOLLOW_DIST
        vx = self.pid_distance.update(error_dist, dt)

        img_center = w / 2
        offset = cx - img_center
        yaw_rate = self.pid_center.update(offset, dt)

        return vx, yaw_rate, distance
