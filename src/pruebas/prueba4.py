import airsim

client = airsim.MultirotorClient()
client.confirmConnection()

print("Conexión correcta.")