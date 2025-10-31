import os
import cv2
import numpy as np
from tqdm import tqdm

def mask_to_yolo_polygon(mask_path, class_id=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = (mask > 127).astype(np.uint8)

    # Tìm contour (đường biên vùng mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    yolo_polygons = []

    for cnt in contours:
        if len(cnt) < 3:
            continue
        cnt = cnt.squeeze()
        polygon = []
        for x, y in cnt:
            polygon.append(x / w)
            polygon.append(y / h)
        line = f"{class_id} " + " ".join([f"{p:.6f}" for p in polygon])
        yolo_polygons.append(line)

    return yolo_polygons


def convert_masks_to_yolo(data_dir="data_prepared", out_dir="yolo_seg"):
    subsets = ["train", "val"]
    os.makedirs(out_dir, exist_ok=True)

    for subset in subsets:
        img_dir = os.path.join(data_dir, "images", subset)
        mask_dir = os.path.join(data_dir, "masks", subset)
        label_dir = os.path.join(out_dir, "labels", subset)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "images", subset), exist_ok=True)

        print(f"\n📂 Converting {subset} set...")
        for fname in tqdm(os.listdir(img_dir)):
            if not fname.endswith(".png"):
                continue
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(mask_dir, fname)
            label_path = os.path.join(label_dir, fname.replace(".png", ".txt"))

            if not os.path.exists(mask_path):
                continue

            # class_id = 0 nếu bleeding trong tên, 1 nếu non-bleeding
            class_id = 0 if "bleeding" in fname else 1
            polygons = mask_to_yolo_polygon(mask_path, class_id)
            if polygons:
                with open(label_path, "w") as f:
                    f.write("\n".join(polygons))

            # Copy ảnh sang thư mục mới để giữ đồng bộ
            os.system(f'copy "{img_path}" "{os.path.join(out_dir, "images", subset)}" >nul 2>&1')

    print("\n✅ All segmentation labels generated successfully!")
    print(f"📁 Output: {out_dir}/labels/train/*.txt & val/*.txt")

if __name__ == "__main__":
    convert_masks_to_yolo()
