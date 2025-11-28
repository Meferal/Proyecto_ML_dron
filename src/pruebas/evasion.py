import airsim
import numpy as np
import cv2
import time

# --- PARÁMETROS DE NAVEGACIÓN ---
VELOCIDAD_AVANCE = 3.0       # Metros/segundo hacia adelante
VELOCIDAD_LATERAL = 2.0      # Velocidad al esquivar
UMBRAL_PELIGRO = 4.0         # Si algo está a menos de X metros, esquivar
MAX_DIST = 15.0              # Distancia máxima a considerar (para normalizar imagen)

def connect_drone():
    print("1. Conectando al dron...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("2. Dron conectado y armado.")
    return client

def main():
    client = connect_drone()

    print("3. Despegando...")
    client.takeoffAsync().join()
    
    # Subir un poco más para no chocar con el suelo
    client.moveByVelocityZAsync(0, 0, -2, 2).join()
    
    print(">>> INICIANDO PILOTO AUTOMÁTICO (Ctrl+C para salir) <<<")

    try:
        while True:
            # A. Obtener imagen de profundidad (DepthPlanar = Distancia real en metros)
            # ImageType 1 es DepthPlanar
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)
            ])

            if not responses:
                continue

            # B. Procesar la imagen (De lista 1D a Matriz 2D)
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float32)
            img2d = img1d.reshape(response.height, response.width)

            # C. Visualización (Normalizar para que OpenCV pueda mostrarlo)
            # Todo lo que esté más lejos de MAX_DIST lo pintamos como "lejos"
            img_vis = np.clip(img2d, 0, MAX_DIST)
            img_vis = img_vis / MAX_DIST * 255.0
            img_vis = img_vis.astype(np.uint8)
            
            # Aplicar mapa de color para verlo mejor (Azul=Cerca, Rojo=Lejos)
            img_color = cv2.applyColorMap(img_vis, cv2.COLORMAP_JET)

            # D. Lógica de Decisión
            # Dividimos la imagen en 3 zonas: Izquierda, Centro, Derecha
            w = response.width
            w3 = w // 3
            
            # Recortamos las zonas
            zone_left = img2d[:, 0:w3]
            zone_center = img2d[:, w3:2*w3]
            zone_right = img2d[:, 2*w3:]

            # Calculamos la distancia promedio en cada zona
            dist_left = np.mean(zone_left)
            dist_center = np.mean(zone_center)
            dist_right = np.mean(zone_right)

            # Decisión
            vy = 0 # Velocidad lateral (Y)
            vx = VELOCIDAD_AVANCE # Velocidad frontal (X)
            msg = "AVANZANDO"

            if dist_center < UMBRAL_PELIGRO:
                # Obstáculo detectado en el frente
                vx = 0.5 # Frenamos un poco
                
                if dist_left > dist_right:
                    msg = "<<< ESQUIVANDO IZQUIERDA"
                    vy = -VELOCIDAD_LATERAL # Y negativo es izquierda
                else:
                    msg = "ESQUIVANDO DERECHA >>>"
                    vy = VELOCIDAD_LATERAL  # Y positivo es derecha
            
            # E. Enviar comando de velocidad al dron
            # duration es pequeño para permitir actualizaciones rápidas
            client.moveByVelocityBodyFrameAsync(vx, vy, 0, 0.1)

            # Dibujar info en pantalla
            cv2.putText(img_color, f"L:{dist_left:.1f} C:{dist_center:.1f} R:{dist_right:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.putText(img_color, msg, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if "ESQUIVANDO" in msg else (0,255,0), 2)
            
            cv2.imshow("Vista Profundidad Dron", img_color)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Deteniendo...")
    
    # Reset final
    client.reset()
    client.enableApiControl(False)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()