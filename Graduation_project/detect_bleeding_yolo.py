import os
import json
import csv
import numpy as np
from ultralytics import YOLO


def train_yolov11(
    weights="yolov11n.pt",
    data_yaml="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=0.001,
    lrf=0.01,
    device=0,
    project="runs/train_bleeding",
    save_period=1
):
    """Train YOLOv11 model with full logging."""
    os.makedirs(project, exist_ok=True)

    print(f"🚀 Starting YOLOv11 training on dataset: {data_yaml}")
    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        device=device,
        save=True,
        save_period=save_period,
        val=True,
        verbose=True,
        project=project,
        exist_ok=True
    )

    exp_dir = results.save_dir
    print(f"\n📁 Full experiment directory: {exp_dir}")

    # === Lưu metrics ===
    if hasattr(results, "metrics") and results.metrics:
        metrics_file = os.path.join(exp_dir, "metrics_summary.json")
        with open(metrics_file, "w") as f:
            json.dump(results.metrics, f, indent=4)
        print(f"✅ Metrics saved to: {metrics_file}")

    # === Lưu confusion matrix nếu có ===
    if hasattr(model.validator, "confusion_matrix"):
        cm = model.validator.confusion_matrix.matrix
        np.save(os.path.join(exp_dir, "confusion_matrix.npy"), cm)
        print(f"✅ Confusion matrix saved to: {exp_dir}/confusion_matrix.npy")

    # === Lưu precision / recall per class nếu có ===
    try:
        names = model.names
        stats = model.validator.stats
        precs, recs = stats[0], stats[1]
        csv_path = os.path.join(exp_dir, "precision_recall_per_class.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name", "precision", "recall"])
            for i, name in names.items():
                writer.writerow([i, name, precs[i] if i < len(precs) else None, recs[i] if i < len(recs) else None])
        print(f"✅ Per-class metrics saved to: {csv_path}")
    except Exception as e:
        print(f"⚠️ Could not extract per-class metrics: {e}")

    print("🎯 Training completed successfully!")


if __name__ == "__main__":
    train_yolov11(
        weights="yolov11n.pt",
        data_yaml="data.yaml",
        epochs=200,
        imgsz=640,
        batch=64,
        lr0=0.001,
        lrf=0.01,
        device=0,
        project="runs/train_bleeding",
        save_period=1
    )
