from ultralytics import YOLO
import json
import csv
import os


def main():
    MODEL_PATH = r"C:\Users\cisg1\Desktop\tesis\outputs\train\yolov8s_custom\weights\best.pt"
    DATA_YAML  = r"C:\Users\cisg1\Desktop\tesis\dataset\data.yaml"
    OUTPUT_DIR = r"C:\Users\cisg1\Desktop\tesis\outputs\metrics"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)

    results = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        device=0,
        plots=True,
        project=OUTPUT_DIR,
        name="yolov8s_eval",
        save=False
    )

    # ================= MÉTRICAS GLOBALES =================
    metrics_global = {
        "precision_mean": float(results.box.mp),
        "recall_mean": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map)
    }

    # Guardar JSON global
    with open(os.path.join(OUTPUT_DIR, "metrics_global.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_global, f, indent=2)

    # Guardar CSV global
    with open(os.path.join(OUTPUT_DIR, "metrics_global.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(metrics_global.keys())
        writer.writerow(metrics_global.values())

    # ================= MÉTRICAS POR CLASE =================
    class_names = results.names
    rows = []

    for i, name in class_names.items():
        p, r, ap50, ap = results.box.class_result(i)
        rows.append([name, p, r, ap50, ap])

    with open(os.path.join(OUTPUT_DIR, "metrics_by_class.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "AP50", "AP50_95"])
        writer.writerows(rows)

    print("Métricas guardadas correctamente.")


if __name__ == "__main__":
    main()
