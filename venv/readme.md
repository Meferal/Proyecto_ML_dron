# 📦 Configuración de Entornos Virtuales con Conda

Este proyecto utiliza **dos entornos virtuales independientes** para separar las dependencias del simulador **AirSim** y del sistema de visión basado en **YOLO**.  
A continuación se explica cómo crearlos, activarlos y utilizarlos correctamente.

---

## 🚁 Entorno `airsim_env`

Este entorno incluye todas las dependencias necesarias para ejecutar el simulador **AirSim**, comunicarse con él mediante la API Python y capturar imágenes RGB/Depth desde drones virtuales.

### 📘 Objetivo
- Controlar el dron desde Python  
- Acceder a las cámaras  
- Obtener medidas del entorno  
- Ejecutar scripts de navegación  

### 📥 Requisito adicional
Es necesario descargar la carpeta **`airsim`** del repositorio oficial de AirSim y copiarla dentro del directorio de trabajo:

```
AirSim-1.8.1-windows/PythonClient/airsim
```

### ▶️ Creación del entorno

```bash
conda create -n airsim_env python=3.8
conda activate airsim_env

pip install numpy opencv-python
pip install msgpack-rpc-python
pip install git+https://github.com/microsoft/AirSim.git#subdirectory=PythonClient
```

---

## 🧠 Entorno `drone_yolo`

Este entorno está diseñado para ejecutar modelos de detección basados en **Ultralytics YOLO**, incluyendo inferencia, entrenamiento y anotación.

### 📘 Objetivo
- Procesar imágenes capturadas por el dron  
- Detectar objetos con YOLO  
- Entrenar y optimizar modelos personalizados  

### ✔️ Creación del entorno

```bash
conda create -n drone_yolo python=3.11.9
conda activate drone_yolo

pip install ultralytics
pip install opencv-python
```

Como librerías básicas de trabajo.

---

## 📄 Archivos `requirements.txt` y `environment.yml`

El proyecto incluye archivos de configuración para reproducir ambos entornos:

### Para AirSim:
- requirements_airsim.txt
- environment_airsim.yml

### Para YOLO:
- requirements_drone_yolo.txt
- environment_drone_yolo.yml

### Instalación desde `requirements.txt`
```bash
pip install -r requirements_airsim.txt
pip install -r requirements_drone_yolo.txt
```

### Instalación desde `environment.yml` (recomendado)
```bash
conda env create -f environment_airsim.yml
conda env create -f environment_drone_yolo.yml
```

---

## 🏁 Resumen de entornos

| Entorno        | Propósito                                 | Python | Dependencias principales         |
|----------------|--------------------------------------------|--------|----------------------------------|
| **airsim_env** | Control del dron y comunicación con AirSim | 3.8    | AirSim API, msgpack, OpenCV      |
| **drone_yolo** | Detección y visión con YOLO                | 3.11.9 | Ultralytics YOLO, utilidades ML  |

---
