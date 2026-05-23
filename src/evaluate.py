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


def load_class_names():
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    index_to_class = {index: name for name, index in class_indices.items()}
    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    return class_names


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Chua tim thay model. Hay chay python src/train.py truoc.")

    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError("Chua tim thay class_indices.json. Hay train model truoc.")

    model = load_model(MODEL_PATH)
    class_names = load_class_names()

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    test_loss, test_acc = model.evaluate(test_generator)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    y_true = test_generator.classes
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )

    print(report)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Test Loss: " + str(test_loss) + "\n")
        f.write("Test Accuracy: " + str(test_acc) + "\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print("Da luu classification report:", report_path)
    print("Da luu confusion matrix:", cm_path)


if __name__ == "__main__":
    main()