import os
import json
import matplotlib.pyplot as plt
# Mô hình ResNet50 đã được huấn luyện trước trên ImageNet
from tensorflow.keras.applications import ResNet50     
# Hàm tiền xử lý dữ liệu theo chuẩn của ResNet50                                         
from tensorflow.keras.applications.resnet50 import preprocess_input   
# Các Callback hỗ trợ quá trình huấn luyện
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau    
# Các lớp xây dựng phần đầu ra của mô hình
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D     
# Đối tượng mô hình
from tensorflow.keras.models import Model     
# Bộ tối ưu Adam
from tensorflow.keras.optimizers import Adam        
# Công cụ đọc ảnh và tăng cường dữ liệu
from tensorflow.keras.preprocessing.image import ImageDataGenerator                             


# IMPORT CẤU HÌNH TỪ FILE config.py
from config import (
    TRAIN_DIR,
    VAL_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    MODEL_PATH,
    CLASS_INDICES_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
)
# HÀM XÂY DỰNG MÔ HÌNH RESNET50

def build_model(num_classes):                        # Khởi tạo mô hình ResNet50 sử dụng trọng số ImageNet
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Transfer Learning: dong bang cac lop cua ResNet50
    base_model.trainable = False

    x = base_model.output
    # GlobalAveragePooling2D giúp giảm số lượng tham số và tránh overfitting, chuyển các đặc trưng 2D thành vector 1D   
    x = GlobalAveragePooling2D()(x)                         
    # Dropout giúp giảm hiện tượng Overfitting
    x = Dense(256, activation="relu")(x)                                   
    x = Dropout(0.6)(x)         #0.5  
    output = Dense(num_classes, activation="softmax")(x)    
    # Lớp đầu ra sử dụng Softmax

    model = Model(inputs=base_model.input, outputs=output)      
    # Xây dựng mô hình hoàn chỉnh

   # Biên dịch mô hình
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

    # Tạo thư mục lưu kết quả nếu chưa tồn tại
def save_training_charts(history):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
  # Vẽ biểu đồ Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_chart.png"), dpi=300)
    plt.close()

   # Vẽ biểu đồ Loss
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_chart.png"), dpi=300)
    plt.close()


def main():
    # Tạo thư mục lưu model và kết quả
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,      # Chuẩn hóa dữ liệu theo ResNet50
        rotation_range=40,                            # Xoay ảnh ngẫu nhiên ±40°
        width_shift_range=0.2,                  # Dịch ngang ảnh ngẫu nhiên ±20% chiều rộng
        height_shift_range=0.2,                 # Dịch dọc ảnh ngẫu nhiên ±20% chiều cao
        zoom_range=0.2,                         # Phóng to/thu nhỏ ảnh ngẫu nhiên ±20%
        horizontal_flip=True,                   # Lật ảnh ngẫu nhiên theo chiều ngang
        brightness_range=[0.8, 1.2],            # Thay đổi độ sáng ngẫu nhiên
        fill_mode="nearest",                    # Điền các pixel trống bằng giá trị của pixel gần nhất
    )
    # Chỉ chuẩn hóa dữ liệu cho tập validation, không tăng cường dữ liệu
    val_datagen = ImageDataGenerator(           
        preprocessing_function=preprocess_input
    )
    # ĐỌC DỮ LIỆU TRAIN
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),               
        # Resize toàn bộ ảnh về 224×224
        batch_size=BATCH_SIZE,
        class_mode="categorical",                              
        # Phân loại nhiều lớp
        shuffle=True,
    )

    # ĐỌC DỮ LIỆU VALIDATION

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    # Lấy số lượng lớp
    num_classes = train_generator.num_classes

    print("So lop:", num_classes)
    print("Danh sach lop:", train_generator.class_indices)

    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=4)
    # Xây dựng mô hình
    model = build_model(num_classes)                
    model.summary()                                
    # Hiển thị cấu trúc mô hình
    # CALLBACKS
    callbacks = [
        ModelCheckpoint(                         
    # Lưu model có độ chính xác Validation cao nhất
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(                              # Dừng huấn luyện nếu mô hình không còn cải thiện
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(                          # Giảm Learning Rate khi mô hình học chậm
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    # HUẤN LUYỆN MÔ HÌNH
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    save_training_charts(history)                                  
    # Lưu biểu đồ Accuracy và Loss

    print("Da huan luyen xong.")
    print("Model duoc luu tai:", MODEL_PATH)
    print("Class indices duoc luu tai:", CLASS_INDICES_PATH)


if __name__ == "__main__":
    main()
    