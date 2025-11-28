import numpy as np
from pid import PID

SAFE_DIST = 2.0
FOLLOW_DIST = 4.0
MAX_SPEED = 5.0


class DroneController:

    def __init__(self):
        # PID para distancia al objetivo
        self.pid_distance = PID(Kp=1.0, Ki=0.01, Kd=0.3,
                                output_limits=(0, MAX_SPEED))

        # PID para centrado horizontal
        self.pid_center = PID(Kp=0.001, Ki=0.0001, Kd=0.005,
                              output_limits=(-2, 2))

    # -------------------------
    # EVITACIÓN DE OBSTÁCULOS
    # -------------------------
    def avoid_obstacles(self, depth):
        h, w = depth.shape
        left = depth[:, :w//3].mean()
        center = depth[:, w//3:2*w//3].mean()
        right = depth[:, 2*w//3:].mean()

        vx = MAX_SPEED
        vy = 0

        if center < SAFE_DIST:
            vx = 0

        if left < SAFE_DIST:
            vy += 2

        if right < SAFE_DIST:
            vy -= 2

        return vx, vy

    # -------------------------
    # SEGUIMIENTO DE OBJETIVO
    # -------------------------
    def follow_target(self, box, depth, dt):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2

        # profundidad media del bbox
        depth_crop = depth[y1:y2, x1:x2]
        distance = float(depth_crop.mean())

        # --- PID distancia ---
        error_dist = distance - FOLLOW_DIST
        vx = self.pid_distance.update(error_dist, dt)

        # --- PID centrado horizontal ---
        img_center = depth.shape[1] / 2
        offset = cx - img_center
        vy = self.pid_center.update(-offset, dt)

        return vx, vy, distance
