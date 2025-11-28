import airsim
import time
import numpy as np

# Conexión al simulador
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Despegar (asíncrono)
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -30, 5).join()

# Aterrizar
# client.landAsync().join()

# Hover (mantener posición)
# client.hoverAsync().join()


"""
# Loop de detección
try:
    while True:
        """
        # Capturar imagen
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])
        
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        
        # Detección con YOLO
        results = model(img_rgb)
        
        # Procesar resultados
        for r in results:
            boxes = r.boxes
            for box in boxes:
                print(f"Objeto detectado: {model.names[int(box.cls)]}")
        
        time.sleep(0.1)
        """
        # Movimiento
        client.moveByVelocityAsync(vx=5, vy=0, vz=0, duration=3).join()
        
except KeyboardInterrupt:
    print("Deteniendo...")
"""
 
# Aterrizar
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
