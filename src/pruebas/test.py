import airsim
import time


def check_movement():
    print("--- INICIANDO DIAGNÓSTICO ---")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("1. Conexión confirmada.")

    # Forzar control de API
    client.enableApiControl(True)
    print("2. API Control habilitado.")

    # Armar motores
    client.armDisarm(True)
    print("3. Motores armados.")

    # Despegue con timeout (para que no se congele si falla)
    print("4. Intentando despegar (espera 5 seg)...")
    f = client.takeoffAsync(timeout_sec=5)
    f.join()
    print("5. Despegue finalizado (o timeout cumplido).")

    # Moverse un poco hacia arriba para asegurar altura
    client.moveByVelocityZAsync(0, 0, -5, 2).join() # Subir a 5m de altura
    print("6. Altura ajustada.")

    print("7. Intentando movimiento frontal forzado...")
    # Moverse adelante a 2 m/s durante 3 segundos
    # args: (vx, vy, z, duration) -> z es negativo para altura fija
    client.moveByVelocityZAsync(2, 0, -5, 3).join()

    print("--- PRUEBA FINALIZADA ---")
    print("Si el dron no se movió, revisa la consola de Unreal Engine por errores en rojo.")

    client.hoverAsync().join()
    client.enableApiControl(False)


if __name__ == "__main__":
    check_movement()
