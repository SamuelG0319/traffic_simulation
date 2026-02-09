import os
from ultralytics import YOLO

# ============================
# CONFIGURACIÓN
# ============================

# Ruta al modelo (preentrenado)
MODEL_PATH = r"C:\Users\lasso\Desktop\tesis\yolov8s.pt"

# Ruta al data.yaml exportado desde Roboflow (VERSIÓN 3)
DATA_YAML = r"C:\Users\lasso\Desktop\vehicle_detection.v3-dataset_v3.yolov8\data.yaml"

# Carpeta donde YOLO guardará resultados
PROJECT_DIR = r"C:\Users\lasso\Desktop\tesis\outputs\metrics"

# Nombre del experimento
RUN_NAME = "yolov8s_eval"

# Umbrales
CONF_THRESH = 0.001   # bajo para evaluación (estándar)
IOU_THRESH = 0.5      # IoU de matching

# ============================
# EVALUACIÓN
# ============================

def run_evaluation():
    print("=== EVALUACIÓN YOLOv8 ===")
    print("Modelo:", MODEL_PATH)
    print("Dataset:", DATA_YAML)

    model = YOLO(MODEL_PATH)

    metrics = model.val(
        task="detect",
        data=DATA_YAML,        # fuerza uso de tus 4 clases
        split="test",          # usa SOLO test set
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=640,
        device="0",          # cámbialo a "0" si usas GPU
        workers=0,             # CRÍTICO en Windows
        project=PROJECT_DIR,
        name=RUN_NAME,
        save_json=True,
        save_conf=True,
        plots=True
    )

    print("\n=== RESULTADOS CLAVE ===")
    print(f"Precision (P): {metrics.box.mp:.4f}")
    print(f"Recall (R):    {metrics.box.mr:.4f}")
    print(f"mAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")

    print("\nResultados guardados en:")
    print(os.path.join(PROJECT_DIR, RUN_NAME))


# ============================
# ENTRY POINT (WINDOWS SAFE)
# ============================

if __name__ == "__main__":
    run_evaluation()