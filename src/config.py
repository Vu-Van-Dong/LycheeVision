import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODEL_DIR, "lychee_resnet50.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# CÁC SIÊU THAM SỐ

IMG_SIZE = 224                  # Kích thước ảnh đầu vào (224 × 224 pixel)
BATCH_SIZE = 32         #16     # Số lượng ảnh được xử lý trong mỗi batch khi huấn luyện
EPOCHS = 10                     # Số lần mô hình học toàn bộ tập dữ liệu
LEARNING_RATE = 0.0001          # Tốc độ học (Learning Rate) của bộ tối ưu Adam # Giá trị nhỏ giúp mô hình hội tụ ổn định hơn.
