  
    
import os                   #làm việc với hệ thống file
import random               #làm việc với dữ liệu ngẫu nhiên
import shutil               #làm việc với hệ thống file(sửa chữa, xóa, copy)
from pathlib import Path    #làm việc với đường dẫn file

# ======================
# THƯ VIỆN BỔ SUNG CHO DATA CLEANING
# ======================
from PIL import Image        #Xử lý ảnh cơ bản trong Python
import cv2
import numpy as np          #Xử lý mảng số

# ======================
# ĐƯỜNG DẪN THƯ MỤC DATASET
# ======================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "dataset" / "raw"
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "val"
TEST_DIR = BASE_DIR / "dataset" / "test"

# ======================
# TỶ LỆ CHIA DỮ LIỆU
# ======================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ======================
# ĐỊNH DẠNG ẢNH HỢP LỆ
# ======================
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

# ======================
# NGƯỠNG ẢNH MỜ (BLUR DETECTION)
# ======================
BLUR_THRESHOLD = 100.0


# =========================================================
# 1. XÓA THƯ MỤC CŨ VÀ TẠO MỚI
# =========================================================
def clear_folder(folder_path):
    """
    Xóa toàn bộ dữ liệu trong thư mục và tạo lại thư mục rỗng
    """
    if folder_path.exists():
        shutil.rmtree(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. COPY ẢNH VÀO THƯ MỤC ĐÍCH
# =========================================================
def copy_images(image_list, target_folder):
    """
    Sao chép danh sách ảnh sang thư mục đích
    """
    target_folder.mkdir(parents=True, exist_ok=True)

    for image_path in image_list:
        shutil.copy2(image_path, target_folder / image_path.name)


# =========================================================
# 3. KIỂM TRA ẢNH LỖI (CORRUPTED IMAGE)
# =========================================================
def is_image_valid(image_path):
    """
    Kiểm tra ảnh có bị lỗi hay không (không mở được file)
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False


# =========================================================
# 4. KIỂM TRA ẢNH MỜ (BLUR DETECTION)
# =========================================================
def is_blurry(image_path, threshold=BLUR_THRESHOLD):
    """
    Phát hiện ảnh mờ bằng Laplacian variance
    """
    try:
        img = cv2.imread(str(image_path))   # Đọc ảnh bằng OpenCV

        if img is None:
            return True

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Chuyển ảnh sang ảnh xám
        variance = cv2.Laplacian(gray, cv2.CV_64F).var() # Tính độ biến thiên Laplacian

        return variance < threshold    # Nếu độ biến thiên nhỏ hơn ngưỡng  => mờ
    except:
        return True


# =========================================================
# 5. HÀM CHÍNH - XỬ LÝ DATASET
# =========================================================
def main():
    print(" Bắt đầu xử lý dataset...")

    # kiểm tra dataset gốc
    if not RAW_DIR.exists():
        print(" Không tìm thấy dataset/raw")
        return

    # reset dataset output
    clear_folder(TRAIN_DIR)
    clear_folder(VAL_DIR)
    clear_folder(TEST_DIR)

    # lấy danh sách class
    class_folders = [f for f in RAW_DIR.iterdir() if f.is_dir()]

    if len(class_folders) == 0:
        print(" Không có class nào trong dataset/raw")
        return

    # =====================================================
    # LẶP QUA TỪNG CLASS
    # =====================================================
    for class_folder in class_folders:
        class_name = class_folder.name

        images = []

        # =================================================
        # CLEAN DATA: lọc ảnh lỗi + ảnh mờ + sai định dạng
        # =================================================
        for file in class_folder.rglob("*"):

            # chỉ lấy file ảnh hợp lệ
            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # kiểm tra ảnh lỗi
            if not is_image_valid(file):
                continue

            # kiểm tra ảnh mờ
            if is_blurry(file):
                continue

            images.append(file)

        # xáo trộn dữ liệu
        random.shuffle(images)

        # tính số lượng
        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        # chia dataset
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        # copy vào folder tương ứng
        copy_images(train_images, TRAIN_DIR / class_name)
        copy_images(val_images, VAL_DIR / class_name)
        copy_images(test_images, TEST_DIR / class_name)

        # log kết quả
        print(
            f" {class_name}: total={total}, "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )

    print(" Hoàn thành xử lý dataset!")

# =========================================================
# RUN FILE
# =========================================================
if __name__ == "__main__":
    main()
    
    
    