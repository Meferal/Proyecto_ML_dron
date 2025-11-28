class PID:
    def __init__(self, Kp, Ki, Kd, output_limits=(-1, 1)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limits = output_limits

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        # Proporcional
        P = self.Kp * error

        # Integral
        self.integral += error * dt
        I = self.Ki * self.integral

        # Derivativa
        D = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error

        # Salida limitada
        output = P + I + D
        min_out, max_out = self.output_limits
        return max(min_out, min(output, max_out))
