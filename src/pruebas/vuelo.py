import airsim
import numpy as np
import cv2
import time

# --- CONFIGURACIÓN ---
VELOCIDAD_AVANCE = 4.0      # Metros por segundo
VELOCIDAD_GIRO = 3.0        # Velocidad lateral al esquivar
DISTANCIA_SEGURIDAD = 5.0   # Distancia mínima al frente para activar evasión (metros)
TECHO_MAX_DEPTH = 20.0      # Clampear la profundidad máxima para evitar valores infinitos


def main():
    print("Conectando a AirSim...")
    # 1. Conexión con el simulador AirSim
    client = airsim.MultirotorClient()
    
    print("Confirmando conexión (Esperando al simulador)...")
    client.confirmConnection()
    
    print("Habilitando API control...")
    client.enableApiControl(True)

    print("Armando dron...")
    client.armDisarm(True)

    print("Despegando...")
    client.takeoffAsync().join()

    # Subimos un poco más para evitar el suelo como obstáculo inmediato
    print("Elevando...")
    client.moveByVelocityZAsync(0, 0, -3, 2).join()

    print("Iniciando navegación autónoma basada en Profundidad...")

    try:
        while True:
            # 2. Obtener imágenes
            # Solicitamos DepthPlanar (distancia en metros, float)
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
            ])

            if not responses:
                print("No se recibieron imágenes.")
                continue

            response = responses[0]

            # 3. Procesar la imagen de profundidad
            img1d = np.array(response.image_data_float, dtype=np.float32)

            # Reformatear a 2D (Alto x Ancho)
            img2d = img1d.reshape(response.height, response.width)

            # Limitar la profundidad máxima (el cielo puede ser infinito)
            img2d[img2d > TECHO_MAX_DEPTH] = TECHO_MAX_DEPTH

            # 4. Lógica de visión: Dividir en 3 franjas verticales
            h, w = img2d.shape
            w_tercio = w // 3

            # Extraer las secciones
            left_view = img2d[:, :w_tercio]
            center_view = img2d[:, w_tercio:2*w_tercio]
            right_view = img2d[:, 2*w_tercio:]

            # Calcular la distancia promedio en cada sección
            avg_left = np.mean(left_view)
            avg_center = np.mean(center_view)
            avg_right = np.mean(right_view)

            # 5. Visualización para depuración (Opcional, abre ventana OpenCV)
            # Normalizamos a 0-255 para verla en pantalla
            img_show = cv2.normalize(img2d, None, 0, 255, cv2.NORM_MINMAX)
            img_show = img_show.astype(np.uint8)
            cv2.imshow("Depth View", img_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 6. Toma de decisiones y Control
            # Usamos moveByVelocityBodyFrameAsync (X=frente, Y=derecha, Z=abajo)

            info = ""

            if avg_center < DISTANCIA_SEGURIDAD:
                # ¡Obstáculo enfrente!
                info = f"OBSTÁCULO ({avg_center:.1f}m) -> "

                if avg_left > avg_right:
                    # Más espacio a la izquierda
                    info += "Esquivando IZQUIERDA"
                    # X pequeña para no detenerse del todo, Y negativo es izquierda
                    client.moveByVelocityBodyFrameAsync(1.0, -VELOCIDAD_GIRO, 0, 0.1)
                else:
                    # Más espacio a la derecha
                    info += "Esquivando DERECHA"
                    # Y positivo es derecha
                    client.moveByVelocityBodyFrameAsync(1.0, VELOCIDAD_GIRO, 0, 0.1)
            else:
                # Camino despejado
                info = "AVANZANDO"
                client.moveByVelocityBodyFrameAsync(VELOCIDAD_AVANCE, 0, 0, 0.1)

            print(f"L:{avg_left:.1f} | C:{avg_center:.1f} | R:{avg_right:.1f} || Acción: {info}")

    except KeyboardInterrupt:
        print("Deteniendo...")
        client.reset()
        client.enableApiControl(False)

if __name__ == "__main__":
    main()
    