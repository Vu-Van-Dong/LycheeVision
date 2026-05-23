import os
import random
import shutil
from pathlib import Path

# Duong dan thu muc
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "dataset" / "raw"
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "val"
TEST_DIR = BASE_DIR / "dataset" / "test"

# Ty le chia du lieu
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Dinh dang anh hop le
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def clear_folder(folder_path):
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


def copy_images(image_list, target_folder):
    target_folder.mkdir(parents=True, exist_ok=True)

    for image_path in image_list:
        shutil.copy2(image_path, target_folder / image_path.name)


def main():
    print("Bat dau chia dataset...")
    print(f"Thu muc raw: {RAW_DIR}")

    if not RAW_DIR.exists():
        print("Khong tim thay thu muc dataset/raw")
        return

    clear_folder(TRAIN_DIR)
    clear_folder(VAL_DIR)
    clear_folder(TEST_DIR)

    class_folders = [folder for folder in RAW_DIR.iterdir() if folder.is_dir()]

    if len(class_folders) == 0:
        print("Khong co thu muc lop nao trong dataset/raw")
        return

    for class_folder in class_folders:
        class_name = class_folder.name

        images = [
            file for file in class_folder.rglob("*")
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        copy_images(train_images, TRAIN_DIR / class_name)
        copy_images(val_images, VAL_DIR / class_name)
        copy_images(test_images, TEST_DIR / class_name)

        print(
            f"{class_name}: total={total}, "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )

    print("Hoan thanh chia dataset!")


if __name__ == "__main__":
    main()