import os
import shutil
from sklearn.model_selection import train_test_split

# ==== ĐƯỜNG DẪN GỐC ====
ROOT = "data/WCEBleedGen"
OUTPUT = "data/yolo_wce_detect"

BLEED_IMG_DIR = os.path.join(ROOT, "bleeding", "Images")
BLEED_LABEL_DIR = os.path.join(ROOT, "bleeding", "Bounding boxes")

NONBLEED_IMG_DIR = os.path.join(ROOT, "non-bleeding", "images")

# ==== TẠO THƯ MỤC ====
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(OUTPUT, sub), exist_ok=True)

# ==== LẤY DANH SÁCH ẢNH ====
bleed_images = [f for f in os.listdir(BLEED_IMG_DIR) if f.endswith(('.jpg', '.png'))]
nonbleed_images = [f for f in os.listdir(NONBLEED_IMG_DIR) if f.endswith(('.jpg', '.png'))]

# ==== CHIA TẬP ====
train_bleed, val_bleed = train_test_split(bleed_images, test_size=0.2, random_state=42)
train_nonbleed, val_nonbleed = train_test_split(nonbleed_images, test_size=0.2, random_state=42)

def copy_pair(img_name, split, has_label=True):
    # copy ảnh
    src_img = os.path.join(BLEED_IMG_DIR if has_label else NONBLEED_IMG_DIR, img_name)
    dst_img = os.path.join(OUTPUT, f"images/{split}/{img_name}")
    shutil.copy(src_img, dst_img)

    # copy label nếu có
    if has_label:
        base = os.path.splitext(img_name)[0]
        possible_labels = [f for f in os.listdir(BLEED_LABEL_DIR) if base in f]
        if possible_labels:
            src_label = os.path.join(BLEED_LABEL_DIR, possible_labels[0])
            dst_label = os.path.join(OUTPUT, f"labels/{split}/{base}.txt")
            shutil.copy(src_label, dst_label)
    else:
        # sinh file label trống cho non-bleeding
        open(os.path.join(OUTPUT, f"labels/{split}/{os.path.splitext(img_name)[0]}.txt"), "w").close()

# ==== XỬ LÝ BLEEDING ====
for img_name in train_bleed:
    copy_pair(img_name, "train", has_label=True)
for img_name in val_bleed:
    copy_pair(img_name, "val", has_label=True)

# ==== XỬ LÝ NON-BLEEDING ====
for img_name in train_nonbleed:
    copy_pair(img_name, "train", has_label=False)
for img_name in val_nonbleed:
    copy_pair(img_name, "val", has_label=False)

print("[DONE] Dataset ready at:", OUTPUT)
