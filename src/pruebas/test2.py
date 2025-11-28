import airsim
import time
import sys

print("--- INICIANDO DIAGNÓSTICO ---")

try:
    # 1. Intentamos conectar
    print("1. Creando cliente (esperando socket)...")
    client = airsim.MultirotorClient()
    
    # 2. Confirmar conexión CON TIMEOUT
    # Si en 2 segundos no responde, saltará un error en lugar de congelarse
    print("2. Enviando ping al simulador (Haz clic en la ventana del simulador AHORA)...")
    if client.ping():
        print("   -> Ping recibido. El simulador está vivo.")
    else:
        print("   -> Ping fallido. El simulador está pausado o no escucha.")

    print("3. Confirmando conexión API...")
    client.confirmConnection()
    print("   -> ¡CONEXIÓN ESTABLECIDA!")

    # Si llegamos aquí, todo funciona
    print("4. Obteniendo estado del dron...")
    state = client.getMultirotorState()
    print(f"   -> Posición actual: {state.kinematics_estimated.position}")

except Exception as e:
    print(f"\n[ERROR CRÍTICO]: {e}")
    print("Posibles causas:")
    print(" - La ventana del simulador está minimizada o en pausa (Dale foco).")
    print(" - El settings.json en Documentos/AirSim no es correcto.")

print("--- FIN DIAGNÓSTICO ---")