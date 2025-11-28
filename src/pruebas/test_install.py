import torch
import cv2
import streamlit
from ultralytics import YOLO
import numpy as np
import pandas as pd

print("✓ PyTorch:", torch.__version__)
print("✓ OpenCV:", cv2.__version__)
print("✓ Streamlit:", streamlit.__version__)
print("✓ NumPy:", np.__version__)
print("✓ Pandas:", pd.__version__)
print("✓ Ultralytics instalado correctamente")

# Test rápido de YOLO
print("\nProbando YOLO11...")
model = YOLO('yolo11n.pt')  # Descargará el modelo si no existe
print("✓ YOLO11 funcionando correctamente")