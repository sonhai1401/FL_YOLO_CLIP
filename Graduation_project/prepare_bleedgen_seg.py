import os
import shutil

# === Đường dẫn gốc ===
ROOT = "WCEBleedGen"  # đổi thành đường dẫn thật của bạn
OUT_DIR = "data_prepared"
os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)

# === Hàm xử lý từng class ===
def process_class(class_name):
    img_dir = os.path.join(ROOT, class_name, "images")
    ann_dir = os.path.join(ROOT, class_name, "annotations")

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))])
    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith((".png", ".jpg"))])

    if len(img_files) != len(ann_files):
        print(f"⚠️ Warning: {class_name} có {len(img_files)} ảnh nhưng {len(ann_files)} annotation!")

    for i, (img, ann) in enumerate(zip(img_files, ann_files)):
        new_name = f"{class_name}_{i+1:04d}.png"

        shutil.copy(os.path.join(img_dir, img), os.path.join(OUT_DIR, "images", new_name))
        shutil.copy(os.path.join(ann_dir, ann), os.path.join(OUT_DIR, "masks", new_name))

    print(f"✅ Đã xử lý xong class '{class_name}' với {len(img_files)} ảnh.")

# === Chạy cho 2 class ===
for cls in ["bleeding", "non-bleeding"]:
    process_class(cls)

print("🎯 Hoàn tất chuẩn hóa toàn bộ dataset!")
