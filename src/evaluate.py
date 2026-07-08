import json
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import (
    TEST_DIR,
    MODEL_PATH,
    CLASS_INDICES_PATH,
    OUTPUT_DIR,
    IMG_SIZE,
    BATCH_SIZE,
)


def get_target_size():
    # Neu IMG_SIZE = 224 thi doi thanh (224, 224)
    # Neu IMG_SIZE da la (224, 224) thi giu nguyen
    if isinstance(IMG_SIZE, int):
        return (IMG_SIZE, IMG_SIZE)
    return IMG_SIZE


def load_class_names():
    # Doc file class_indices.json da luu khi train
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    # Sap xep ten lop theo dung thu tu index: 0, 1, 2, ...
    index_to_class = {v: k for k, v in class_indices.items()}
    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    return class_names


def create_test_generator():
    # Preprocess dung voi ResNet50
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=get_target_size(),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return test_generator


def save_classification_report(test_loss, test_acc, report):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("===== TEST RESULT =====\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("===== CLASSIFICATION REPORT =====\n")
        f.write(report)

    print("Da luu classification report:", report_path)


def save_confusion_matrix(y_true, y_pred, class_names):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d"
    )

    plt.title("Confusion Matrix - LycheeVision ResNet50")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print("Da luu confusion matrix:", cm_path)


def main():
    print("Bat dau danh gia model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(f"Khong tim thay class_indices.json: {CLASS_INDICES_PATH}")

    print("Dang load model...")
    model = load_model(MODEL_PATH)

    print("Dang load class names...")
    class_names = load_class_names()
    print("Danh sach lop:", class_names)

    print("Dang load tap test...")
    test_generator = create_test_generator()

    print("Dang danh gia model tren tap test...")
    test_loss, test_acc = model.evaluate(test_generator)

    print("\n===== KET QUA TEST =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print("\nDang du doan tap test...")
    y_true = test_generator.classes
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n===== CLASSIFICATION REPORT =====")
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)

    save_classification_report(test_loss, test_acc, report)
    save_confusion_matrix(y_true, y_pred, class_names)

    print("\nHoan thanh danh gia model!")


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    