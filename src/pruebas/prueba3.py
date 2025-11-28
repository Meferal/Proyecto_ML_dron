import airsim
import time

# Conectar al simulador
client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
client.moveToZAsync(-5, 2).join()

# Mover hacia adelante
client.moveByVelocityAsync(2, 0, 0, 3).join()

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
