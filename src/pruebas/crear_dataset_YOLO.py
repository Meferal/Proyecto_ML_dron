"""
Script para convertir dataset de segmentación semántica a formato YOLO
Soporta:
- Detección con Bounding Boxes (YOLO Detection)
- Segmentación de Instancias (YOLO Segmentation)
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
from PIL import Image
import json

# ==========================================
# CONFIGURACIÓN
# ==========================================

# Rutas de tu dataset
BASE_PATH = 'datasets/semantic_drone_dataset'  # Cambiar según tu ruta

ORIGINAL_IMAGES = os.path.join(BASE_PATH, '/original_images')
RGB_MASKS = os.path.join(BASE_PATH, '/RGB_color_image_masks')
SEMANTIC_MASKS = os.path.join(BASE_PATH, 'label_images_semantic')
CSV_FILE = os.path.join(BASE_PATH, 'class_dict_seg.csv')  # Cambiar nombre si es diferente

# Ruta de salida (formato YOLO)
OUTPUT_PATH = os.path.join(BASE_PATH, 'semantic_drone_yolo_dataset')

# Modo de conversión
MODO = 'detection'  # 'detection' o 'segmentation'

# Parámetros
MIN_AREA = 100  # Área mínima del objeto (píxeles) para ser considerado
TRAIN_SPLIT = 0.8  # 80% train, 20% val

print("="*60)
print("🔄 CONVERSIÓN DE DATASET A FORMATO YOLO")
print("="*60)
print(f"\n📁 Dataset origen: {BASE_PATH}")
print(f"📁 Dataset destino: {OUTPUT_PATH}")
print(f"🎯 Modo: {MODO.upper()}")
print(f"📊 Train/Val split: {TRAIN_SPLIT*100:.0f}% / {(1-TRAIN_SPLIT)*100:.0f}%")

# ==========================================
# PASO 1: Leer Mapeo de Colores
# ==========================================

print("\n" + "="*60)
print("📋 PASO 1: Leer mapeo de colores RGB")
print("="*60)

# Leer CSV
df_colors = pd.read_csv(CSV_FILE)

print(f"\n✅ CSV cargado: {len(df_colors)} clases encontradas")
print("\n📝 Clases:")

# Crear diccionario de mapeo RGB -> clase_id y nombre
color_to_class = {}
class_names = []

for idx, row in df_colors.iterrows():
    name = row['name']
    r = int(row['r'])
    g = int(row['g'])
    b = int(row['b'])
    
    # RGB como tupla
    rgb_key = (r, g, b)
    color_to_class[rgb_key] = idx
    class_names.append(name)
    
    print(f"   {idx}: {name} - RGB({r}, {g}, {b})")

num_classes = len(class_names)
print(f"\n🎯 Total de clases: {num_classes}")

# ==========================================
# PASO 2: Crear Estructura de Carpetas YOLO
# ==========================================

print("\n" + "="*60)
print("📁 PASO 2: Crear estructura de carpetas")
print("="*60)

# Crear carpetas
folders = [
    'images/train',
    'images/val',
    'labels/train',
    'labels/val'
]

for folder in folders:
    folder_path = os.path.join(OUTPUT_PATH, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"   ✅ {folder}")

# ==========================================
# PASO 3: Funciones de Conversión
# ==========================================

def rgb_to_class_mask(rgb_mask, color_to_class):
    """
    Convierte máscara RGB a máscara de clases (cada píxel = ID de clase)
    """
    height, width = rgb_mask.shape[:2]
    class_mask = np.zeros((height, width), dtype=np.uint8)
    
    for rgb_color, class_id in color_to_class.items():
        # Crear máscara para este color
        mask = np.all(rgb_mask == rgb_color, axis=-1)
        class_mask[mask] = class_id
    
    return class_mask

def mask_to_bboxes(class_mask, class_id, min_area=100):
    """
    Extrae bounding boxes de una máscara de clase específica
    Retorna: lista de [class_id, x_center, y_center, width, height] normalizados
    """
    # Crear máscara binaria para esta clase
    binary_mask = (class_mask == class_id).astype(np.uint8)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    height, width = class_mask.shape
    
    for contour in contours:
        # Filtrar contornos pequeños
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Obtener bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Normalizar coordenadas (YOLO format)
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height
        
        bboxes.append([class_id, x_center, y_center, w_norm, h_norm])
    
    return bboxes

def mask_to_polygons(class_mask, class_id, min_area=100):
    """
    Extrae polígonos de segmentación de una máscara de clase específica
    Retorna: lista de [class_id, [x1, y1, x2, y2, ...]] normalizados
    """
    # Crear máscara binaria para esta clase
    binary_mask = (class_mask == class_id).astype(np.uint8)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    height, width = class_mask.shape
    
    for contour in contours:
        # Filtrar contornos pequeños
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Simplificar contorno
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convertir a lista de coordenadas normalizadas
        polygon = []
        for point in approx:
            x, y = point[0]
            x_norm = x / width
            y_norm = y / height
            polygon.extend([x_norm, y_norm])
        
        # YOLO segmentation necesita al menos 3 puntos
        if len(polygon) >= 6:  # 3 puntos = 6 coordenadas
            polygons.append([class_id, polygon])
    
    return polygons

def save_yolo_labels(labels, output_file, mode='detection'):
    """
    Guarda etiquetas en formato YOLO
    """
    with open(output_file, 'w') as f:
        for label in labels:
            if mode == 'detection':
                # Formato: class_id x_center y_center width height
                class_id, x_c, y_c, w, h = label
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
            elif mode == 'segmentation':
                # Formato: class_id x1 y1 x2 y2 x3 y3 ...
                class_id, polygon = label
                polygon_str = ' '.join([f"{coord:.6f}" for coord in polygon])
                f.write(f"{class_id} {polygon_str}\n")

# ==========================================
# PASO 4: Procesar Imágenes
# ==========================================

print("\n" + "="*60)
print("🔄 PASO 4: Procesando imágenes")
print("="*60)

# Listar todas las imágenes originales
original_files = sorted([f for f in os.listdir(ORIGINAL_IMAGES) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))])

print(f"\n📊 Total de imágenes: {len(original_files)}")

# Determinar split train/val
num_train = int(len(original_files) * TRAIN_SPLIT)
train_files = original_files[:num_train]
val_files = original_files[num_train:]

print(f"   Train: {len(train_files)}")
print(f"   Val: {len(val_files)}")

def process_image(img_name, split='train'):
    """
    Procesa una imagen y genera sus etiquetas YOLO
    """
    # Rutas
    img_path = os.path.join(ORIGINAL_IMAGES, img_name)
    
    # Buscar máscara RGB correspondiente
    mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
    rgb_mask_path = os.path.join(RGB_MASKS, mask_name)
    
    if not os.path.exists(rgb_mask_path):
        print(f"   ⚠️ Máscara no encontrada para: {img_name}")
        return False
    
    # Leer imagen y máscara
    img = cv2.imread(img_path)
    rgb_mask = cv2.imread(rgb_mask_path)
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
    
    # Convertir RGB mask a class mask
    class_mask = rgb_to_class_mask(rgb_mask, color_to_class)
    
    # Extraer etiquetas según el modo
    all_labels = []
    
    for class_id in range(num_classes):
        if MODO == 'detection':
            labels = mask_to_bboxes(class_mask, class_id, min_area=MIN_AREA)
        elif MODO == 'segmentation':
            labels = mask_to_polygons(class_mask, class_id, min_area=MIN_AREA)
        
        all_labels.extend(labels)
    
    # Si no hay objetos, saltar
    if len(all_labels) == 0:
        print(f"   ⚠️ Sin objetos: {img_name}")
        return False
    
    # Guardar imagen en carpeta correspondiente
    output_img_path = os.path.join(OUTPUT_PATH, 'images', split, img_name)
    cv2.imwrite(output_img_path, img)
    
    # Guardar etiquetas
    label_name = img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
    output_label_path = os.path.join(OUTPUT_PATH, 'labels', split, label_name)
    save_yolo_labels(all_labels, output_label_path, mode=MODO)
    
    return True

# Procesar train
print("\n🔄 Procesando imágenes de entrenamiento...")
train_success = 0
for i, img_name in enumerate(train_files):
    if process_image(img_name, split='train'):
        train_success += 1
    
    if (i + 1) % 50 == 0:
        print(f"   Procesadas: {i+1}/{len(train_files)}")

print(f"   ✅ Train: {train_success}/{len(train_files)} exitosas")

# Procesar val
print("\n🔄 Procesando imágenes de validación...")
val_success = 0
for i, img_name in enumerate(val_files):
    if process_image(img_name, split='val'):
        val_success += 1
    
    if (i + 1) % 50 == 0:
        print(f"   Procesadas: {i+1}/{len(val_files)}")

print(f"   ✅ Val: {val_success}/{len(val_files)} exitosas")

# ==========================================
# PASO 5: Crear archivo YAML
# ==========================================

print("\n" + "="*60)
print("📄 PASO 5: Crear archivo YAML de configuración")
print("="*60)

yaml_content = f"""# Dataset Configuration for YOLO
path: {OUTPUT_PATH}
train: images/train
val: images/val

# Classes
nc: {num_classes}
names: {class_names}
"""

yaml_path = os.path.join(OUTPUT_PATH, 'dataset.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n✅ YAML creado: {yaml_path}")
print("\n📋 Contenido:")
print(yaml_content)

# ==========================================
# PASO 6: Estadísticas del Dataset
# ==========================================

print("\n" + "="*60)
print("📊 PASO 6: Estadísticas del dataset")
print("="*60)

def count_objects_per_class(labels_dir):
    """
    Cuenta objetos por clase en un directorio de etiquetas
    """
    class_counts = {i: 0 for i in range(num_classes)}
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    
    return class_counts

train_labels_dir = os.path.join(OUTPUT_PATH, 'labels/train')
val_labels_dir = os.path.join(OUTPUT_PATH, 'labels/val')

train_counts = count_objects_per_class(train_labels_dir)
val_counts = count_objects_per_class(val_labels_dir)

print("\n📊 Distribución de objetos por clase:")
print(f"\n{'Clase':<20} {'Train':<10} {'Val':<10} {'Total':<10}")
print("-" * 50)

for i, class_name in enumerate(class_names):
    train_count = train_counts[i]
    val_count = val_counts[i]
    total = train_count + val_count
    print(f"{class_name:<20} {train_count:<10} {val_count:<10} {total:<10}")

print("-" * 50)
total_train = sum(train_counts.values())
total_val = sum(val_counts.values())
print(f"{'TOTAL':<20} {total_train:<10} {total_val:<10} {total_train + total_val:<10}")

# ==========================================
# PASO 7: Visualizar Ejemplos
# ==========================================

print("\n" + "="*60)
print("🖼️ PASO 7: Generar visualizaciones de ejemplo")
print("="*60)

def visualize_yolo_labels(img_path, label_path, class_names, mode='detection'):
    """
    Visualiza una imagen con sus etiquetas YOLO
    """
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # Leer etiquetas
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        class_name = class_names[class_id]
        
        # Color aleatorio por clase
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        if mode == 'detection':
            # Bounding box
            x_c, y_c, w, h = map(float, parts[1:5])
            
            # Desnormalizar
            x_c *= width
            y_c *= height
            w *= width
            h *= height
            
            # Calcular esquinas
            x1 = int(x_c - w/2)
            y1 = int(y_c - h/2)
            x2 = int(x_c + w/2)
            y2 = int(y_c + h/2)
            
            # Dibujar
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_name, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        elif mode == 'segmentation':
            # Polígono
            coords = list(map(float, parts[1:]))
            points = []
            
            for i in range(0, len(coords), 2):
                x = int(coords[i] * width)
                y = int(coords[i+1] * height)
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            
            # Dibujar polígono
            cv2.polylines(img, [points], True, color, 2)
            
            # Centroide para etiqueta
            M = cv2.moments(points)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(img, class_name, (cx, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

# Visualizar 3 ejemplos de train
train_imgs = sorted(os.listdir(os.path.join(OUTPUT_PATH, 'images/train')))[:3]

print("\n📸 Generando visualizaciones...")

for img_name in train_imgs:
    img_path = os.path.join(OUTPUT_PATH, 'images/train', img_name)
    label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(OUTPUT_PATH, 'labels/train', label_name)
    
    if os.path.exists(label_path):
        vis_img = visualize_yolo_labels(img_path, label_path, class_names, mode=MODO)
        
        # Guardar visualización
        vis_path = os.path.join(OUTPUT_PATH, f'vis_{img_name}')
        cv2.imwrite(vis_path, vis_img)
        print(f"   ✅ {vis_path}")

# ==========================================
# RESUMEN FINAL
# ==========================================

print("\n" + "="*60)
print("✅ CONVERSIÓN COMPLETADA")
print("="*60)

print(f"""
📊 Resumen:
   • Imágenes train: {train_success}
   • Imágenes val: {val_success}
   • Clases: {num_classes}
   • Modo: {MODO.upper()}
   • Objetos train: {total_train}
   • Objetos val: {total_val}

📁 Dataset YOLO generado en:
   {OUTPUT_PATH}

📄 Archivo de configuración:
   {yaml_path}

🚀 Próximo paso: Entrenar modelo
   Usa este YAML en tu entrenamiento:
   data='{yaml_path}'
""")