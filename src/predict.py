import argparse
import json
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from config import MODEL_PATH, CLASS_INDICES_PATH, IMG_SIZE


def load_class_names():
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    index_to_class = {index: name for name, index in class_indices.items()}
    class_names = [index_to_class[i] for i in range(len(index_to_class))]

    return class_names


def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Chua co model. Hay chay python src/train.py truoc.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Khong tim thay anh: {image_path}")

    model = load_model(MODEL_PATH)
    class_names = load_class_names()

    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)[0]

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    class_name = class_names[class_index]

    return class_name, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Duong dan anh can du doan")
    args = parser.parse_args()

    class_name, confidence = predict_image(args.image)

    print("Ket qua du doan:", class_name)
    print("Do tin cay:", round(confidence * 100, 2), "%")


if __name__ == "__main__":
    main()