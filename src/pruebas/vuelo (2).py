import airsim
import time
import math

def volar_ruta_circular():
    """
    Drone despega, sube a 30 metros, realiza ruta circular y aterriza
    """
    
    # Conectar al simulador
    print("Conectando a AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("Conexión establecida")
    
    try:
        # Despegar
        print("Despegando...")
        client.takeoffAsync().join()
        time.sleep(2)
        
        # Subir a 30 metros (Z negativo = arriba)
        print("Elevando a 30 metros...")
        client.moveToPositionAsync(0, 0, -30, 5).join()
        print("Altura alcanzada: 30 metros")
        time.sleep(1)
        
        # Parámetros del círculo
        radio = 20  # Radio del círculo en metros
        num_puntos = 36  # Número de puntos en el círculo (cada 10 grados)
        velocidad = 5  # Velocidad en m/s
        altura = -30  # Mantener 30 metros de altura
        
        print(f"Iniciando ruta circular (radio: {radio}m, {num_puntos} puntos)...")
        
        # Generar waypoints del círculo
        waypoints = []
        for i in range(num_puntos + 1):  # +1 para cerrar el círculo
            angulo = 2 * math.pi * i / num_puntos
            x = radio * math.cos(angulo)
            y = radio * math.sin(angulo)
            waypoints.append(airsim.Vector3r(x, y, altura))
        
        # Volar la ruta circular
        print("Ejecutando ruta circular...")
        client.moveOnPathAsync(
            waypoints,
            velocity=velocidad,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0),
            lookahead=-1,
            adaptive_lookahead=1
        ).join()
        
        print("Ruta circular completada")
        time.sleep(1)
        
        # Volver al centro antes de aterrizar
        print("Volviendo al punto de origen...")
        client.moveToPositionAsync(0, 0, -30, 5).join()
        time.sleep(1)
        
        # Aterrizar
        print("Aterrizando...")
        client.landAsync().join()
        time.sleep(2)
        
        print("Aterrizaje completado")
        
    except Exception as e:
        print(f"Error durante el vuelo: {e}")
        print("Intentando aterrizar de emergencia...")
        client.landAsync().join()
    
    finally:
        # Desactivar control API
        client.armDisarm(False)
        client.enableApiControl(False)
        print("Misión finalizada")


def volar_ruta_circular_con_orientacion():
    """
    Versión alternativa: el drone mira hacia el centro del círculo mientras vuela
    """
    
    print("Conectando a AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    print("Conexión establecida")
    
    try:
        # Despegar y elevar
        print("Despegando...")
        client.takeoffAsync().join()
        time.sleep(2)
        
        print("Elevando a 30 metros...")
        client.moveToPositionAsync(0, 0, -30, 5).join()
        time.sleep(1)
        
        # Parámetros
        radio = 20
        num_puntos = 36
        velocidad = 5
        altura = -30
        
        print(f"Iniciando ruta circular con orientación hacia el centro...")
        
        # Volar punto por punto con orientación
        for i in range(num_puntos + 1):
            angulo = 2 * math.pi * i / num_puntos
            x = radio * math.cos(angulo)
            y = radio * math.sin(angulo)
            
            # Calcular yaw para mirar hacia el centro
            yaw = angulo + math.pi  # +180 grados para mirar al centro
            
            # Mover con orientación específica
            client.moveToPositionAsync(
                x, y, altura, velocidad,
                yaw_mode=airsim.YawMode(False, math.degrees(yaw))
            ).join()
        
        print("Ruta circular completada")
        time.sleep(1)
        
        # Volver y aterrizar
        print("Volviendo al punto de origen...")
        client.moveToPositionAsync(0, 0, -30, 5).join()
        time.sleep(1)
        
        print("Aterrizando...")
        client.landAsync().join()
        time.sleep(2)
        
        print("Aterrizaje completado")
        
    except Exception as e:
        print(f"Error durante el vuelo: {e}")
        print("Intentando aterrizar de emergencia...")
        client.landAsync().join()
    
    finally:
        client.armDisarm(False)
        client.enableApiControl(False)
        print("Misión finalizada")


if __name__ == "__main__":
    print("=" * 50)
    print("SIMULACIÓN DE VUELO CIRCULAR - AirSim")
    print("=" * 50)
    print("\nOpciones:")
    print("1. Ruta circular simple (orientación hacia adelante)")
    print("2. Ruta circular con orientación hacia el centro")
    
    opcion = input("\nSelecciona una opción (1 o 2): ").strip()
    
    if opcion == "1":
        volar_ruta_circular()
    elif opcion == "2":
        volar_ruta_circular_con_orientacion()
    else:
        print("Opción no válida. Ejecutando opción 1 por defecto...")
        volar_ruta_circular()