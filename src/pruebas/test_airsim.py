import airsim

print("Intentando conectar...")
client = airsim.MultirotorClient()
client.confirmConnection()
print("¡Conectado! El simulador responde.")
print(f"Estado del dron: {client.getMultirotorState().kinematics_estimated.position}")
