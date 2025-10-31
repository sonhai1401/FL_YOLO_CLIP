import os
import shutil
import random
from tqdm import tqdm

def prepare_dataset(base_dir="WCEBleedGen", out_dir="data_prepared", split_ratio=0.8):
    classes = ["bleeding", "non-bleeding"]
    image_ext = ".png"  # nếu là .jpg hoặc .jpeg thì sửa ở đây

    # Tạo thư mục đích
    for subset in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "masks", subset), exist_ok=True)

    for cls in classes:
        print(f"\n📦 Processing class: {cls}")
        img_dir = os.path.join(base_dir, cls, "images")
        ann_dir = os.path.join(base_dir, cls, "annotations")

        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(image_ext)])
        ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(image_ext)])

        n = min(len(img_files), len(ann_files))
        pairs = list(zip(img_files[:n], ann_files[:n]))
        random.shuffle(pairs)

        split_idx = int(split_ratio * n)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        for subset, data_pairs in [("train", train_pairs), ("val", val_pairs)]:
            for idx, (img_file, ann_file) in enumerate(tqdm(data_pairs, desc=f"{cls} -> {subset}")):
                new_name = f"{cls}_{idx:04d}{image_ext}"

                src_img = os.path.join(img_dir, img_file)
                src_ann = os.path.join(ann_dir, ann_file)

                dst_img = os.path.join(out_dir, "images", subset, new_name)
                dst_mask = os.path.join(out_dir, "masks", subset, new_name)

                shutil.copy(src_img, dst_img)
                shutil.copy(src_ann, dst_mask)

    print("\n✅ Dataset prepared successfully!")
    print(f"📁 Output directory: {out_dir}")
    print("Structure:")
    print("data_prepared/")
    print(" ├── images/train/")
    print(" ├── images/val/")
    print(" ├── masks/train/")
    print(" └── masks/val/")

if __name__ == "__main__":
    prepare_dataset(base_dir="WCEBleedGen", out_dir="data_prepared", split_ratio=0.8)
