import os
import shutil
import random

root = "data/WCEBleedGen"
out_dir = "data/yolo_dataset"
os.makedirs(out_dir, exist_ok=True)

img_out = os.path.join(out_dir, "images")
lbl_out = os.path.join(out_dir, "labels")

for split in ["train", "val"]:
    os.makedirs(os.path.join(img_out, split), exist_ok=True)
    os.makedirs(os.path.join(lbl_out, split), exist_ok=True)

def copy_and_rename_images(src_img_dir, src_label_dir, prefix, split_ratio=0.8):
    imgs = [f for f in os.listdir(src_img_dir) if f.endswith((".png", ".jpg"))]
    random.shuffle(imgs)
    split_idx = int(len(imgs) * split_ratio)
    
    for i, img_name in enumerate(imgs):
        img_path = os.path.join(src_img_dir, img_name)
        base = os.path.splitext(img_name)[0]
        label_name = base + ".txt"
        lbl_path = os.path.join(src_label_dir, label_name)

        # Tạo tên mới để tránh trùng
        new_name = f"{prefix}_{i:04d}.png"
        new_lbl = f"{prefix}_{i:04d}.txt"

        split = "train" if i < split_idx else "val"
        shutil.copy(img_path, os.path.join(img_out, split, new_name))

        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(lbl_out, split, new_lbl))
        else:
            # nếu là non-bleeding mà không có bbox
            open(os.path.join(lbl_out, split, new_lbl), "w").close()

# bleeding có YOLO bbox
copy_and_rename_images(
    src_img_dir=os.path.join(root, "bleeding", "Images"),
    src_label_dir=os.path.join(root, "bleeding", "Bounding boxes"),
    prefix="bleeding"
)

# non-bleeding có mask, chưa có bbox
copy_and_rename_images(
    src_img_dir=os.path.join(root, "non-bleeding", "Images"),
    src_label_dir=os.path.join(root, "non-bleeding", "annotation"),
    prefix="nonbleeding"
)
